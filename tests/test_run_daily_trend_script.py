from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.exchanges.base import Environment, OrderResult


@pytest.fixture(autouse=True)
def _patch_secret_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import run_daily_trend

    monkeypatch.setattr(run_daily_trend, "_create_secret_manager", lambda args: SimpleNamespace())


def _fake_pipeline(env: Environment) -> SimpleNamespace:
    environment = SimpleNamespace(environment=env, risk_profile="balanced")
    bootstrap = SimpleNamespace(environment=environment, alert_router=SimpleNamespace())
    controller = SimpleNamespace()
    return SimpleNamespace(bootstrap=bootstrap, controller=controller)


def test_run_once_executes_single_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import run_daily_trend

    pipeline = _fake_pipeline(Environment.PAPER)
    captured_args: dict[str, Any] = {}

    def fake_build_pipeline(**kwargs: Any) -> SimpleNamespace:
        captured_args.update(kwargs)
        return pipeline

    monkeypatch.setattr(run_daily_trend, "build_daily_trend_pipeline", fake_build_pipeline)

    trading_controller_obj = SimpleNamespace()

    def fake_create_trading_controller(pipeline_arg: Any, alert_router: Any, **kwargs: Any) -> Any:
        assert pipeline_arg is pipeline
        assert alert_router is pipeline.bootstrap.alert_router
        captured_args["controller_args"] = kwargs
        return trading_controller_obj

    monkeypatch.setattr(run_daily_trend, "create_trading_controller", fake_create_trading_controller)

    calls: list[str] = []

    class DummyRunner:
        def __init__(
            self,
            *,
            controller: Any,
            trading_controller: Any,
            history_bars: int,
            clock=None,
        ) -> None:
            assert controller is pipeline.controller
            assert trading_controller is trading_controller_obj
            captured_args["history_bars"] = history_bars
            captured_args["clock"] = clock
            calls.append("init")

        def run_once(self) -> Iterable[OrderResult]:
            calls.append("run_once")
            return []

    monkeypatch.setattr(run_daily_trend, "DailyTrendRealtimeRunner", DummyRunner)

    exit_code = run_daily_trend.main(["--config", "config/core.yaml", "--run-once"])

    assert exit_code == 0
    assert calls == ["init", "run_once"]
    assert captured_args["environment_name"] == "binance_paper"
    assert captured_args["history_bars"] == 180
    assert captured_args["controller_args"]["health_check_interval"] == 3600.0


def test_refuses_live_environment_without_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import run_daily_trend

    pipeline = _fake_pipeline(Environment.LIVE)
    monkeypatch.setattr(run_daily_trend, "build_daily_trend_pipeline", lambda **_: pipeline)

    create_called = False

    def _fail_create(*_args: Any, **_kwargs: Any) -> None:
        nonlocal create_called
        create_called = True
        raise AssertionError("create_trading_controller powinien nie być wywołany")

    monkeypatch.setattr(run_daily_trend, "create_trading_controller", _fail_create)

    exit_code = run_daily_trend.main(["--config", "config/core.yaml", "--run-once"])

    assert exit_code == 3
    assert create_called is False


def test_dry_run_returns_success(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import run_daily_trend

    pipeline = _fake_pipeline(Environment.PAPER)
    monkeypatch.setattr(run_daily_trend, "build_daily_trend_pipeline", lambda **_: pipeline)

    exit_code = run_daily_trend.main(["--config", "config/core.yaml", "--dry-run"])

    assert exit_code == 0


def test_paper_smoke_uses_date_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from scripts import run_daily_trend

    dispatch_calls: list[Any] = []
    sync_calls: list[dict[str, Any]] = []
    collected_calls: list[dict[str, int]] = []

    start_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end_dt = datetime(2024, 2, 15, 23, 59, 59, 999000, tzinfo=timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    class DummyController:
        symbols = ("BTCUSDT",)
        interval = "1d"
        tick_seconds = 86400.0

        def collect_signals(self, *, start: int, end: int) -> list[Any]:
            collected_calls.append({"start": start, "end": end})
            return []

    class DummyBackfill:
        def synchronize(self, **kwargs: Any) -> None:
            sync_calls.append(kwargs)

    class DummyExecutionService:
        def ledger(self) -> list[dict[str, Any]]:
            return []

    history_bars_cap = 180
    tick_ms = int(DummyController.tick_seconds * 1000)
    window_duration_ms = max(0, end_ms - start_ms)
    approx_bars = max(1, int(window_duration_ms / tick_ms) + 1)
    expected_history = max(1, min(history_bars_cap, approx_bars))
    runner_start_ms = max(0, end_ms - expected_history * tick_ms)
    sync_start_ms = min(start_ms, runner_start_ms)
    required_bars = max(expected_history, max(1, int((end_ms - sync_start_ms) / tick_ms) + 1))

    class DummyStorage:
        def metadata(self) -> dict[str, str]:
            return {
                f"row_count::BTCUSDT::1d": str(required_bars + 5),
                f"last_timestamp::BTCUSDT::1d": str(end_ms + tick_ms),
            }

        def read(self, key: str) -> dict[str, Any]:
            assert key == "BTCUSDT::1d"
            return {
                "rows": [
                    [float(sync_start_ms - tick_ms), 0.0, 0.0, 0.0, 0.0, 0.0],
                    [float(end_ms + tick_ms), 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            }

    class DummyAlertRouter:
        def dispatch(self, message: Any) -> None:
            dispatch_calls.append(message)

        def health_snapshot(self) -> dict[str, Any]:
            return {"telegram": {"status": "ok"}}

    environment_cfg = SimpleNamespace(environment=Environment.PAPER, risk_profile="balanced")

    pipeline = SimpleNamespace(
        controller=DummyController(),
        backfill_service=DummyBackfill(),
        execution_service=DummyExecutionService(),
        bootstrap=SimpleNamespace(environment=environment_cfg, alert_router=DummyAlertRouter()),
        data_source=SimpleNamespace(storage=DummyStorage()),
    )

    captured_args: dict[str, Any] = {}

    def fake_build_pipeline(**kwargs: Any) -> SimpleNamespace:
        captured_args.update(kwargs)
        return pipeline

    monkeypatch.setattr(run_daily_trend, "build_daily_trend_pipeline", fake_build_pipeline)

    trading_controller = SimpleNamespace(
        maybe_report_health=lambda: None,
        process_signals=lambda signals: [],
    )

    monkeypatch.setattr(
        run_daily_trend,
        "create_trading_controller",
        lambda pipeline_arg, alert_router, **kwargs: trading_controller,
    )

    captured_runner: dict[str, Any] = {}

    class DummyRunner:
        def __init__(
            self,
            *,
            controller: Any,
            trading_controller: Any,
            history_bars: int,
            clock=None,
        ) -> None:
            captured_runner.update(
                {
                    "controller": controller,
                    "trading_controller": trading_controller,
                    "history_bars": history_bars,
                    "clock": clock,
                }
            )

        def run_once(self) -> list[OrderResult]:
            now = captured_runner["clock"]()
            captured_runner["now"] = now
            controller = captured_runner["controller"]
            tick_ms_local = int(getattr(controller, "tick_seconds", 86400.0) * 1000)
            history = int(captured_runner["history_bars"])
            end_ms_local = int(now.timestamp() * 1000)
            start_ms_local = max(0, end_ms_local - history * tick_ms_local)
            controller.collect_signals(start=start_ms_local, end=end_ms_local)
            return []

    monkeypatch.setattr(run_daily_trend, "DailyTrendRealtimeRunner", DummyRunner)

    report_dir = tmp_path / "smoke"

    def fake_mkdtemp(*_args: Any, **_kwargs: Any) -> str:
        report_dir.mkdir(parents=True, exist_ok=True)
        return str(report_dir)

    monkeypatch.setattr(tempfile, "mkdtemp", fake_mkdtemp)

    def fake_export_smoke_report(
        *,
        report_dir: Path,
        results: Iterable[Any],
        ledger: Iterable[Mapping[str, Any]],
        window: Mapping[str, str],
        environment: str,
        alert_snapshot: Mapping[str, Mapping[str, str]],
    ) -> Path:
        ledger_path = report_dir / "ledger.jsonl"
        ledger_path.write_text("", encoding="utf-8")
        summary = {
            "environment": environment,
            "window": dict(window),
            "orders": [
                {
                    "order_id": "OID-1",
                    "status": "filled",
                    "filled_quantity": "0.10",
                    "avg_price": "45000",
                }
            ],
            "ledger_entries": len(list(ledger)),
            "alert_snapshot": alert_snapshot,
        }
        summary_path = report_dir / "summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        return summary_path

    monkeypatch.setattr(run_daily_trend, "_export_smoke_report", fake_export_smoke_report)

    caplog.set_level("INFO")

    exit_code = run_daily_trend.main(
        [
            "--config",
            "config/core.yaml",
            "--environment",
            "binance_paper",
            "--paper-smoke",
            "--archive-smoke",
            "--date-window",
            "2024-01-01:2024-02-15",
        ]
    )

    assert exit_code == 0
    assert "adapter_factories" in captured_args
    assert "binance_spot" in captured_args["adapter_factories"]

    end_dt = datetime.fromisoformat("2024-02-15T23:59:59.999000+00:00")
    start_dt = datetime.fromisoformat("2024-01-01T00:00:00+00:00")
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    tick_ms = int(pipeline.controller.tick_seconds * 1000)
    window_duration_ms = max(0, end_ms - start_ms)
    approx_bars = max(1, int(window_duration_ms / tick_ms) + 1)
    expected_history = max(1, min(180, approx_bars))
    expected_runner_start = max(0, end_ms - expected_history * tick_ms)

    assert captured_runner["history_bars"] == expected_history
    assert captured_runner["now"] == end_dt

    assert sync_calls
    assert sync_calls[0]["start"] == min(start_ms, expected_runner_start)
    assert sync_calls[0]["end"] == end_ms

    assert collected_calls
    assert collected_calls[0]["start"] == expected_runner_start
    assert collected_calls[0]["end"] >= end_ms

    assert dispatch_calls, "Kanał alertów powinien otrzymać powiadomienie smoke"
    summary_bytes = (report_dir / "summary.json").read_bytes()
    expected_hash = hashlib.sha256(summary_bytes).hexdigest()
    alert_context = getattr(dispatch_calls[0], "context")
    assert alert_context["summary_sha256"] == expected_hash
    assert alert_context["summary_text_path"] == str(report_dir / "summary.txt")
    assert alert_context["readme_path"] == str(report_dir / "README.txt")

    summary_txt = (report_dir / "summary.txt").read_text(encoding="utf-8")
    assert "Zakres dat" in summary_txt
    assert "SHA-256 summary.json" in summary_txt

    readme_txt = (report_dir / "README.txt").read_text(encoding="utf-8")
    assert "Daily Trend – smoke test" in readme_txt

    archive_path = report_dir.with_suffix(".zip")
    assert archive_path.exists()
    assert alert_context["archive_path"] == str(archive_path)
    with zipfile.ZipFile(archive_path, "r") as archive:
        names = set(archive.namelist())
    assert {"summary.json", "summary.txt", "ledger.jsonl", "README.txt"}.issubset(names)

    log_messages = [record.message for record in caplog.records if "Podsumowanie smoke testu" in record.message]
    assert log_messages
    joined_log = "\n".join(log_messages)
    assert "Środowisko: binance_paper" in joined_log
    assert "Alerty:" in joined_log


def test_paper_smoke_requires_seeded_cache(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from scripts import run_daily_trend

    caplog.set_level("ERROR")

    class DummyController:
        symbols = ("BTCUSDT",)
        interval = "1d"
        tick_seconds = 86400.0

        def collect_signals(self, *, start: int, end: int) -> list[Any]:  # pragma: no cover - nie powinno zostać wywołane
            raise AssertionError("collect_signals nie powinno być wywołane przy braku cache")

    class DummyBackfill:
        def __init__(self) -> None:
            self.called = False

        def synchronize(self, **kwargs: Any) -> None:  # pragma: no cover - nie powinno zostać wywołane
            self.called = True
            raise AssertionError("backfill nie powinien być wywołany przy braku cache")

    class DummyExecutionService:
        def ledger(self) -> list[dict[str, Any]]:  # pragma: no cover - nie powinno zostać wywołane
            raise AssertionError("ledger nie powinien być odczytany")

    class EmptyStorage:
        def metadata(self) -> dict[str, str]:
            return {}

        def read(self, key: str) -> dict[str, Any]:
            raise KeyError(key)

    class DummyAlertRouter:
        def dispatch(self, message: Any) -> None:  # pragma: no cover - nie powinno zostać wywołane
            raise AssertionError("dispatch nie powinien być wywołany")

        def health_snapshot(self) -> dict[str, Any]:  # pragma: no cover - nie powinno zostać wywołane
            return {}

    environment_cfg = SimpleNamespace(environment=Environment.PAPER, risk_profile="balanced")
    pipeline = SimpleNamespace(
        controller=DummyController(),
        backfill_service=DummyBackfill(),
        execution_service=DummyExecutionService(),
        bootstrap=SimpleNamespace(environment=environment_cfg, alert_router=DummyAlertRouter()),
        data_source=SimpleNamespace(storage=EmptyStorage()),
    )

    def fake_build_pipeline(**kwargs: Any) -> SimpleNamespace:
        assert kwargs["environment_name"] == "binance_paper"
        return pipeline

    monkeypatch.setattr(run_daily_trend, "build_daily_trend_pipeline", fake_build_pipeline)
    monkeypatch.setattr(
        run_daily_trend,
        "create_trading_controller",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("create_trading_controller nie powinien być wywołany")),
    )
    monkeypatch.setattr(
        run_daily_trend,
        "DailyTrendRealtimeRunner",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("runner nie powinien być uruchomiony")),
    )

    exit_code = run_daily_trend.main(
        [
            "--config",
            "config/core.yaml",
            "--environment",
            "binance_paper",
            "--paper-smoke",
            "--date-window",
            "2024-01-01:2024-02-15",
        ]
    )

    assert exit_code == 1
    assert any("Cache offline" in record.message for record in caplog.records)


def test_render_smoke_summary_formats_alerts() -> None:
    from scripts import run_daily_trend

    summary = {
        "environment": "binance_paper",
        "window": {"start": "2024-01-01T00:00:00+00:00", "end": "2024-02-01T23:59:59+00:00"},
        "orders": [
            {"order_id": "O1", "status": "filled", "filled_quantity": "0.1", "avg_price": "42000"},
            {"order_id": "O2", "status": "cancelled", "filled_quantity": "0.0", "avg_price": None},
        ],
        "ledger_entries": 3,
        "alert_snapshot": {
            "telegram": {"status": "ok"},
            "email": {"status": "warn", "detail": "DNS failure"},
        },
    }

    rendered = run_daily_trend._render_smoke_summary(summary=summary, summary_sha256="deadbeef")

    assert "Środowisko: binance_paper" in rendered
    assert "Zakres dat: 2024-01-01T00:00:00+00:00 → 2024-02-01T23:59:59+00:00" in rendered
    assert "Liczba zleceń: 2" in rendered
    assert "Liczba wpisów w ledgerze: 3" in rendered
    assert "telegram: OK" in rendered
    assert "email: WARN (DNS failure)" in rendered
    assert rendered.endswith("SHA-256 summary.json: deadbeef")
