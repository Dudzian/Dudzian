from __future__ import annotations

import hashlib
import json
import sys
import sqlite3
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.config.models import (
    CoreReportingConfig,
    InstrumentBackfillWindow,
    InstrumentConfig,
    InstrumentUniverseConfig,
    SmokeArchiveLocalConfig,
    SmokeArchiveUploadConfig,
)
from bot_core.exchanges.base import Environment, OrderResult


@pytest.fixture(autouse=True)
def _patch_secret_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import run_daily_trend

    secret_manager = SimpleNamespace(
        load_secret_value=lambda key, **_: "",  # pragma: no cover - tylko stub
    )
    monkeypatch.setattr(run_daily_trend, "_create_secret_manager", lambda args: secret_manager)


def _fake_pipeline(env: Environment) -> SimpleNamespace:
    environment = SimpleNamespace(environment=env, risk_profile="balanced")
    bootstrap = SimpleNamespace(
        environment=environment,
        alert_router=SimpleNamespace(),
        core_config=SimpleNamespace(reporting=None),
    )
    controller = SimpleNamespace()
    return SimpleNamespace(
        bootstrap=bootstrap,
        controller=controller,
        strategy_name="core_daily_trend",
        controller_name="daily_trend_core",
    )


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
        raise AssertionError("create_trading_controller should not be invoked")

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

    archive_store = tmp_path / "archives"
    reporting_cfg = CoreReportingConfig(
        smoke_archive_upload=SmokeArchiveUploadConfig(
            backend="local",
            credential_secret=None,
            local=SmokeArchiveLocalConfig(
                directory=str(archive_store),
                filename_pattern="{environment}_{timestamp}_{hash}.zip",
                fsync=False,
            ),
        ),
    )

    pipeline = SimpleNamespace(
        controller=DummyController(),
        backfill_service=DummyBackfill(),
        execution_service=DummyExecutionService(),
        bootstrap=SimpleNamespace(
            environment=environment_cfg,
            alert_router=DummyAlertRouter(),
            core_config=reporting_cfg,
        ),
        data_source=SimpleNamespace(storage=DummyStorage()),
        strategy_name="core_daily_trend",
        controller_name="daily_trend_core",
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
        risk_state: Mapping[str, object] | None,
        data_checks: Mapping[str, object] | None = None,
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
            "risk_state": dict(risk_state or {}),
        }
        if data_checks is not None:
            summary["data_checks"] = json.loads(json.dumps(data_checks))
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
    summary_payload = json.loads(summary_bytes.decode("utf-8"))
    expected_hash = hashlib.sha256(summary_bytes).hexdigest()
    alert_context = getattr(dispatch_calls[0], "context")
    assert alert_context["summary_sha256"] == expected_hash
    assert alert_context["summary_text_path"] == str(report_dir / "summary.txt")
    assert alert_context["readme_path"] == str(report_dir / "README.txt")

    data_checks = summary_payload.get("data_checks")
    assert data_checks, "summary.json powinno zawierać sekcję data_checks"
    cache_info = data_checks.get("cache", {})
    assert "BTCUSDT" in cache_info
    cache_entry = cache_info["BTCUSDT"]
    assert int(cache_entry["required_bars"]) >= expected_history
    assert int(cache_entry["row_count"]) >= expected_history

    summary_txt = (report_dir / "summary.txt").read_text(encoding="utf-8")
    assert "Zakres dat" in summary_txt
    assert "SHA-256 summary.json" in summary_txt
    assert "Cache offline:" in summary_txt

    readme_txt = (report_dir / "README.txt").read_text(encoding="utf-8")
    assert "Daily Trend – smoke test" in readme_txt

    archive_path = report_dir.with_suffix(".zip")
    assert archive_path.exists()
    assert alert_context["archive_path"] == str(archive_path)
    assert alert_context["archive_upload_backend"] == "local"
    upload_location = alert_context["archive_upload_location"]
    assert upload_location.endswith(".zip")
    uploaded_files = list(archive_store.glob("*.zip"))
    assert uploaded_files, "Archiwum powinno zostać skopiowane do magazynu lokalnego"
    assert uploaded_files[0].name in upload_location
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
        strategy_name="core_daily_trend",
        controller_name="daily_trend_core",
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


def test_paper_smoke_manifest_gap_blocks_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from scripts import run_daily_trend

    caplog.set_level("ERROR")

    end_dt = datetime(2024, 2, 15, 23, 59, 59, 999000, tzinfo=timezone.utc)
    end_ms = int(end_dt.timestamp() * 1000)

    class DummyController:
        symbols = ("BTCUSDT",)
        interval = "1d"
        tick_seconds = 86400.0

        def collect_signals(self, *, start: int, end: int) -> list[Any]:  # pragma: no cover
            raise AssertionError("collect_signals nie powinno być wywołane przy błędzie manifestu")

    class DummyBackfill:
        def synchronize(self, **kwargs: Any) -> None:  # pragma: no cover
            raise AssertionError("Backfill nie powinien zostać uruchomiony przy błędzie manifestu")

    class DummyExecutionService:
        def ledger(self) -> list[dict[str, Any]]:  # pragma: no cover
            raise AssertionError("Ledger nie powinien być odczytany")

    class DummyStorage:
        def metadata(self) -> dict[str, str]:
            return {}

        def read(self, key: str) -> dict[str, Any]:  # pragma: no cover
            raise AssertionError("Odczyt danych nie powinien następować przy błędzie manifestu")

    class DummyAlertRouter:
        def dispatch(self, message: Any) -> None:  # pragma: no cover
            raise AssertionError("Alert nie powinien zostać wysłany")

        def health_snapshot(self) -> dict[str, Any]:  # pragma: no cover
            return {}

    instrument = InstrumentConfig(
        name="BTC",
        base_asset="BTC",
        quote_asset="USDT",
        categories=("majors",),
        exchange_symbols={"binance_spot": "BTCUSDT"},
        backfill_windows=(InstrumentBackfillWindow(interval="1d", lookback_days=90),),
    )
    universe = InstrumentUniverseConfig(
        name="paper_universe",
        description="test",
        instruments=(instrument,),
    )

    core_config = SimpleNamespace(instrument_universes={"paper_universe": universe})

    manifest_path = tmp_path / "ohlcv_manifest.sqlite"
    connection = sqlite3.connect(manifest_path)
    try:
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        insufficient_rows = 5
        last_ts = end_ms - int(DummyController.tick_seconds * 1000) * 2
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            ("row_count::BTCUSDT::1d", str(insufficient_rows)),
        )
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            ("last_timestamp::BTCUSDT::1d", str(last_ts)),
        )
        connection.commit()
    finally:
        connection.close()

    environment_cfg = SimpleNamespace(
        environment=Environment.PAPER,
        risk_profile="balanced",
        instrument_universe="paper_universe",
        data_cache_path=str(tmp_path),
        exchange="binance_spot",
    )

    pipeline = SimpleNamespace(
        controller=DummyController(),
        backfill_service=DummyBackfill(),
        execution_service=DummyExecutionService(),
        bootstrap=SimpleNamespace(
            environment=environment_cfg,
            alert_router=DummyAlertRouter(),
            core_config=core_config,
        ),
        data_source=SimpleNamespace(storage=DummyStorage()),
        strategy_name="core_daily_trend",
        controller_name="daily_trend_core",
    )

    def fake_build_pipeline(**kwargs: Any) -> SimpleNamespace:
        assert kwargs["environment_name"] == "binance_paper"
        return pipeline

    monkeypatch.setattr(run_daily_trend, "build_daily_trend_pipeline", fake_build_pipeline)
    monkeypatch.setattr(
        run_daily_trend,
        "create_trading_controller",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("controller nie powinien być tworzony")),
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
    assert any("Manifest danych OHLCV" in record.message for record in caplog.records)


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


def test_export_smoke_report_includes_metrics(tmp_path: Path) -> None:
    from scripts import run_daily_trend

    report_dir = tmp_path / "report"
    window = {
        "start": "2024-01-01T00:00:00+00:00",
        "end": "2024-01-08T23:59:59+00:00",
    }

    results = [
        OrderResult(
            order_id="OID-1",
            status="filled",
            filled_quantity=0.1,
            avg_price=40000.0,
            raw_response={},
        ),
        OrderResult(
            order_id="OID-2",
            status="filled",
            filled_quantity=0.1,
            avg_price=42000.0,
            raw_response={},
        ),
    ]

    ledger = [
        {
            "timestamp": 1704067200.0,
            "order_id": "OID-1",
            "symbol": "BTCUSDT",
            "side": "buy",
            "quantity": 0.1,
            "price": 40000.0,
            "fee": 1.2,
            "fee_asset": "USDT",
            "status": "filled",
            "leverage": 1.0,
            "position_value": 4000.0,
        },
        {
            "timestamp": 1704672000.0,
            "order_id": "OID-2",
            "symbol": "BTCUSDT",
            "side": "sell",
            "quantity": 0.1,
            "price": 42000.0,
            "fee": 1.1,
            "fee_asset": "USDT",
            "status": "filled",
            "leverage": 1.0,
            "position_value": 0.0,
        },
    ]

    risk_state = {
        "profile": "balanced",
        "active_positions": 1,
        "gross_notional": 4_200.0,
        "daily_loss_pct": 0.01,
        "drawdown_pct": 0.02,
        "force_liquidation": False,
        "limits": {
            "max_positions": 5,
            "max_leverage": 3.0,
            "daily_loss_limit": 0.5,
            "drawdown_limit": 0.1,
            "max_position_pct": 0.25,
            "target_volatility": 0.12,
            "stop_loss_atr_multiple": 2.5,
        },
    }

    summary_path = run_daily_trend._export_smoke_report(
        report_dir=report_dir,
        results=results,
        ledger=ledger,
        window=window,
        environment="binance_paper",
        alert_snapshot={"telegram": {"status": "ok"}},
        risk_state=risk_state,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary["metrics"]

    assert metrics["side_counts"]["buy"] == 1
    assert metrics["side_counts"]["sell"] == 1
    assert metrics["notional"]["buy"] == pytest.approx(4000.0)
    assert metrics["notional"]["sell"] == pytest.approx(4200.0)
    assert metrics["notional"]["total"] == pytest.approx(8200.0)
    assert metrics["total_fees"] == pytest.approx(2.3)
    assert metrics["last_position_value"] == pytest.approx(0.0)
    assert summary["risk_state"]["profile"] == "balanced"
    assert summary["risk_state"]["active_positions"] == 1
    assert summary["risk_state"]["gross_notional"] == pytest.approx(4_200.0)
    assert summary["risk_state"]["limits"]["max_positions"] == 5

    ledger_path = report_dir / "ledger.jsonl"
    assert ledger_path.exists()
    lines = [line for line in ledger_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 2


def test_export_smoke_report_serializes_data_checks(tmp_path: Path) -> None:
    from scripts import run_daily_trend

    report_dir = tmp_path / "report_data"
    window = {"start": "2024-03-01T00:00:00+00:00", "end": "2024-03-02T23:59:59+00:00"}
    data_checks = {
        "interval": "1d",
        "cache": {
            "BTCUSDT": {
                "coverage_bars": 3,
                "required_bars": 3,
                "row_count": 4,
            }
        },
    }

    summary_path = run_daily_trend._export_smoke_report(
        report_dir=report_dir,
        results=[],
        ledger=[],
        window=window,
        environment="binance_paper",
        alert_snapshot={},
        risk_state=None,
        data_checks=data_checks,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["data_checks"]["cache"]["BTCUSDT"]["row_count"] == 4
    assert summary["data_checks"]["interval"] == "1d"


def test_render_smoke_summary_with_metrics() -> None:
    from scripts import run_daily_trend

    summary = {
        "environment": "binance_paper",
        "window": {"start": "2024-01-01T00:00:00+00:00", "end": "2024-01-08T23:59:59+00:00"},
        "orders": [],
        "ledger_entries": 3,
        "alert_snapshot": {},
        "metrics": {
            "side_counts": {"buy": 2, "sell": 1},
            "notional": {"buy": 1234.5, "sell": 987.6, "total": 2222.1},
            "total_fees": 0.987654,
            "last_position_value": 4321.0,
        },
    }

    text = run_daily_trend._render_smoke_summary(summary=summary, summary_sha256="abc123")

    assert "Zlecenia BUY/SELL: 2/1" in text
    assert "Wolumen BUY: 1 234.50 | SELL: 987.60 | Razem: 2 222.10" in text
    assert "Łączne opłaty: 0.9877" in text
    assert "Ostatnia wartość pozycji: 4 321.00" in text
    assert "SHA-256 summary.json: abc123" in text


def test_render_smoke_summary_includes_risk_state() -> None:
    from scripts import run_daily_trend

    summary = {
        "environment": "binance_paper",
        "window": {"start": "2024-01-01", "end": "2024-01-31"},
        "orders": [],
        "ledger_entries": 0,
        "alert_snapshot": {"telegram": {"status": "ok"}},
        "metrics": {},
        "risk_state": {
            "profile": "balanced",
            "active_positions": 2,
            "gross_notional": 12_500.0,
            "daily_loss_pct": 0.0125,
            "drawdown_pct": 0.034,
            "force_liquidation": False,
            "limits": {
                "max_positions": 5,
                "max_position_pct": 0.25,
                "max_leverage": 3.0,
                "daily_loss_limit": 0.05,
                "drawdown_limit": 0.15,
                "target_volatility": 0.12,
                "stop_loss_atr_multiple": 2.5,
            },
        },
    }

    text = run_daily_trend._render_smoke_summary(summary=summary, summary_sha256="deadbeef")

    assert "Profil ryzyka: balanced" in text
    assert "Aktywne pozycje: 2 | Ekspozycja brutto: 12 500.00" in text
    assert "Dzienna strata: 1.25% | Obsunięcie: 3.40%" in text
    assert "Force liquidation: NIE" in text
    assert "Limity: max pozycje 5" in text


def test_render_smoke_summary_includes_data_checks() -> None:
    from scripts import run_daily_trend

    summary = {
        "environment": "binance_paper",
        "window": {"start": "2024-01-01", "end": "2024-01-31"},
        "orders": [],
        "ledger_entries": 0,
        "alert_snapshot": {},
        "data_checks": {
            "manifest": {
                "status": "ok",
                "entries": [
                    {
                        "symbol": "BTCUSDT",
                        "interval": "1d",
                        "issues": [],
                        "row_count": 31,
                        "required_rows": 31,
                        "gap_minutes": 0,
                        "last_timestamp_ms": 1706659199999,
                        "last_timestamp_iso": "2024-01-31T23:59:59+00:00",
                    }
                ],
            },
            "cache": {
                "BTCUSDT": {
                    "coverage_bars": 31,
                    "required_bars": 31,
                    "row_count": 35,
                }
            },
        },
    }

    text = run_daily_trend._render_smoke_summary(summary=summary, summary_sha256="feedbeef")

    assert "Manifest OHLCV:" in text
    assert "Cache offline:" in text
    assert text.endswith("SHA-256 summary.json: feedbeef")
