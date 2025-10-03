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
from typing import Any, Sequence

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
from bot_core.data.ohlcv.coverage_check import CoverageStatus
from bot_core.data.ohlcv.manifest_report import ManifestEntry


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
        risk_profile_name="balanced",
    )


def test_collect_storage_health_warns_on_low_space(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from scripts import run_daily_trend

    report_dir = tmp_path / "report"
    report_dir.mkdir()
    caplog.set_level("WARNING")

    megabyte = 1024 * 1024

    class FakeUsage:
        total = 200 * megabyte
        used = 190 * megabyte
        free = 10 * megabyte

    monkeypatch.setattr(run_daily_trend.shutil, "disk_usage", lambda _: FakeUsage)

    info = run_daily_trend._collect_storage_health(report_dir, min_free_mb=32)

    assert info["status"] == "low"
    assert pytest.approx(info["threshold_mb"], rel=1e-6) == 32
    assert any("Wolne miejsce w katalogu raportu" in record.message for record in caplog.records)


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
    assert captured_args["risk_profile_name"] is None


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


def test_risk_profile_override_passed_to_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import run_daily_trend

    pipeline = _fake_pipeline(Environment.PAPER)
    captured: dict[str, Any] = {}

    def fake_build_pipeline(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return pipeline

    monkeypatch.setattr(run_daily_trend, "build_daily_trend_pipeline", fake_build_pipeline)
    monkeypatch.setattr(run_daily_trend, "create_trading_controller", lambda *args, **kwargs: SimpleNamespace())

    class DummyRunner:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        def run_once(self) -> Iterable[OrderResult]:
            return []

    monkeypatch.setattr(run_daily_trend, "DailyTrendRealtimeRunner", DummyRunner)

    exit_code = run_daily_trend.main(
        [
            "--config",
            "config/core.yaml",
            "--run-once",
            "--risk-profile",
            "aggressive",
        ]
    )

    assert exit_code == 0
    assert captured["risk_profile_name"] == "aggressive"


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

    instrument = InstrumentConfig(
        name="BTC",
        base_asset="BTC",
        quote_asset="USDT",
        categories=("majors",),
        exchange_symbols={"binance_spot": "BTCUSDT"},
        backfill_windows=(InstrumentBackfillWindow(interval="1d", lookback_days=60),),
    )
    universe = InstrumentUniverseConfig(
        name="paper_universe",
        description="test",
        instruments=(instrument,),
    )

    manifest_path = tmp_path / "ohlcv_manifest.sqlite"
    manifest_path.touch()

    environment_cfg = SimpleNamespace(
        environment=Environment.PAPER,
        risk_profile="balanced",
        instrument_universe="paper_universe",
        data_cache_path=str(tmp_path),
        exchange="binance_spot",
    )

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
        risk_profile_name="balanced",
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

    operator_notes: list[str | None] = []

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
        storage_info: Mapping[str, object] | None = None,
        note: str | None = None,
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
        operator_notes.append(note)
        if note:
            summary["note"] = note
        if data_checks is not None:
            summary["data_checks"] = json.loads(json.dumps(data_checks))
        if storage_info is not None:
            summary["storage"] = json.loads(json.dumps(storage_info))
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
            "--smoke-note",
            "Testowa notatka",
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
    intervals_payload = cache_entry.get("intervals", {}) if isinstance(cache_entry, Mapping) else {}
    assert intervals_payload, "raport cache powinien zawierać metryki per interwał"
    interval_payload = intervals_payload.get("1d") or next(iter(intervals_payload.values()))
    assert int(interval_payload["required_bars"]) >= expected_history
    assert int(interval_payload["row_count"]) >= expected_history

    summary_txt = (report_dir / "summary.txt").read_text(encoding="utf-8")
    assert "Zakres dat" in summary_txt
    assert "SHA-256 summary.json" in summary_txt
    assert "Cache offline:" in summary_txt
    assert "Magazyn raportu:" in summary_txt
    assert "Notatka operatora: Testowa notatka" in summary_txt

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
    assert operator_notes[-1] == "Testowa notatka"
    assert alert_context["operator_note"] == "Testowa notatka"
    with zipfile.ZipFile(archive_path, "r") as archive:
        names = set(archive.namelist())
    assert {"summary.json", "summary.txt", "ledger.jsonl", "README.txt"}.issubset(names)

    log_messages = [record.message for record in caplog.records if "Podsumowanie smoke testu" in record.message]
    assert log_messages
    joined_log = "\n".join(log_messages)
    assert "Środowisko: binance_paper" in joined_log
    assert "Alerty:" in joined_log


def test_paper_smoke_custom_output_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from scripts import run_daily_trend

    base_output = tmp_path / "reports"
    report_dirs: list[Path] = []
    dispatch_calls: list[Any] = []
    collected_calls: list[dict[str, int]] = []
    sync_calls: list[dict[str, int]] = []
    start_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end_dt = datetime(2024, 1, 10, 23, 59, 59, 999000, tzinfo=timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    tick_ms = int(86400 * 1000)

    class DummyController:
        symbols = ("BTCUSDT",)
        interval = "1d"
        tick_seconds = 86400.0

        def collect_signals(self, *, start: int, end: int) -> list[object]:
            collected_calls.append({"start": start, "end": end})
            return []

    class DummyBackfill:
        def synchronize(self, *, symbols: Iterable[str], interval: str, start: int, end: int) -> None:
            sync_calls.append(
                {
                    "symbols": tuple(symbols),
                    "interval": interval,
                    "start": start,
                    "end": end,
                }
            )

    class DummyExecutionService:
        def ledger(self) -> Iterable[Mapping[str, object]]:
            return [
                {
                    "symbol": "BTCUSDT",
                    "side": "buy",
                    "quantity": 0.1,
                    "price": 40_000.0,
                    "fee": 0.1,
                }
            ]

    class DummyStorage:
        def metadata(self) -> dict[str, str]:
            return {
                "row_count::BTCUSDT::1d": "64",
                "last_timestamp::BTCUSDT::1d": str(end_ms + tick_ms),
            }

        def read(self, key: str) -> Mapping[str, object]:
            assert key == "BTCUSDT::1d"
            return {
                "rows": [
                    [float(start_ms - tick_ms), 0.0, 0.0, 0.0, 0.0, 0.0],
                    [float(end_ms + tick_ms), 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            }

    class DummyAlertRouter:
        def dispatch(self, message: Any) -> None:
            dispatch_calls.append(message)

        def health_snapshot(self) -> Mapping[str, Mapping[str, str]]:
            return {"telegram": {"status": "ok"}}

    pipeline = SimpleNamespace(
        controller=DummyController(),
        backfill_service=DummyBackfill(),
        execution_service=DummyExecutionService(),
        bootstrap=SimpleNamespace(
            environment=SimpleNamespace(environment=Environment.PAPER, risk_profile="balanced"),
            alert_router=DummyAlertRouter(),
            core_config=SimpleNamespace(reporting=None),
        ),
        data_source=SimpleNamespace(storage=DummyStorage()),
        strategy_name="core_daily_trend",
        controller_name="daily_trend_core",
        risk_profile_name="balanced",
    )

    monkeypatch.setattr(run_daily_trend, "build_daily_trend_pipeline", lambda **_: pipeline)

    trading_controller = SimpleNamespace(
        maybe_report_health=lambda: None,
        process_signals=lambda signals: [],
    )
    monkeypatch.setattr(
        run_daily_trend,
        "create_trading_controller",
        lambda pipeline_arg, alert_router, **kwargs: trading_controller,
    )

    class DummyRunner:
        def __init__(self, *, controller: Any, trading_controller: Any, history_bars: int, clock=None) -> None:
            self._controller = controller
            self._clock = clock or (lambda: datetime.now(timezone.utc))
            self._history_bars = history_bars

        def run_once(self) -> list[OrderResult]:
            now = self._clock()
            tick_ms_local = int(getattr(self._controller, "tick_seconds", 86400.0) * 1000)
            end_local = int(now.timestamp() * 1000)
            start_local = max(0, end_local - self._history_bars * tick_ms_local)
            self._controller.collect_signals(start=start_local, end=end_local)
            return []

    monkeypatch.setattr(run_daily_trend, "DailyTrendRealtimeRunner", DummyRunner)

    def fake_export_smoke_report(
        *,
        report_dir: Path,
        results: Iterable[OrderResult],
        ledger: Iterable[Mapping[str, object]],
        window: Mapping[str, str],
        environment: str,
        alert_snapshot: Mapping[str, Mapping[str, str]],
        risk_state: Mapping[str, object] | None,
        data_checks: Mapping[str, object] | None = None,
        storage_info: Mapping[str, object] | None = None,
        note: str | None = None,
    ) -> Path:
        report_dirs.append(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "ledger.jsonl").write_text("", encoding="utf-8")
        summary_payload = {
            "environment": environment,
            "window": dict(window),
            "orders": [],
            "ledger_entries": len(list(ledger)),
            "alert_snapshot": alert_snapshot,
        }
        if risk_state:
            summary_payload["risk_state"] = dict(risk_state)
        if data_checks:
            summary_payload["data_checks"] = json.loads(json.dumps(data_checks))
        if storage_info:
            summary_payload["storage"] = json.loads(json.dumps(storage_info))
        if note:
            summary_payload["note"] = note
        summary_path = report_dir / "summary.json"
        summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")
        return summary_path

    monkeypatch.setattr(run_daily_trend, "_export_smoke_report", fake_export_smoke_report)

    exit_code = run_daily_trend.main(
        [
            "--config",
            "config/core.yaml",
            "--environment",
            "binance_paper",
            "--paper-smoke",
            "--date-window",
            "2024-01-01:2024-01-10",
            "--smoke-output",
            str(base_output),
        ]
    )

    assert exit_code == 0
    assert base_output.exists()
    assert report_dirs, "powinien powstać katalog z raportem smoke"
    report_dir = report_dirs[0]
    assert report_dir.parent == base_output
    assert report_dir.name.startswith("daily_trend_smoke_")

    summary_path = report_dir / "summary.json"
    summary_txt = report_dir / "summary.txt"
    assert summary_path.exists()
    assert summary_txt.exists()

    assert dispatch_calls
    context = getattr(dispatch_calls[0], "context")
    assert dispatch_calls[0].severity == "info"
    assert context["report_dir"] == str(report_dir)
    assert context["summary_text_path"] == str(summary_txt)
    assert context.get("storage_status") == "ok"

    assert collected_calls, "kontroler powinien zebrać sygnały"
    assert sync_calls, "powinna zajść synchronizacja danych"


def test_paper_smoke_low_storage_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from scripts import run_daily_trend

    base_output = tmp_path / "reports"
    dispatch_calls: list[Any] = []
    collected_calls: list[dict[str, int]] = []
    sync_calls: list[dict[str, int]] = []
    report_dirs: list[Path] = []

    start_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end_dt = datetime(2024, 1, 5, 23, 59, 59, 999000, tzinfo=timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    tick_ms = int(86400 * 1000)

    class DummyController:
        symbols = ("BTCUSDT",)
        interval = "1d"
        tick_seconds = 86400.0

        def collect_signals(self, *, start: int, end: int) -> list[Any]:
            collected_calls.append({"start": start, "end": end})
            return []

    class DummyBackfill:
        def synchronize(self, *, symbols: Iterable[str], interval: str, start: int, end: int) -> None:
            sync_calls.append({"symbols": tuple(symbols), "interval": interval, "start": start, "end": end})

    class DummyExecutionService:
        def ledger(self) -> list[Mapping[str, object]]:
            return []

    required_bars = max(1, int((end_ms - start_ms) / tick_ms) + 1)

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
                    [float(start_ms - tick_ms), 0.0, 0.0, 0.0, 0.0, 0.0],
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
        bootstrap=SimpleNamespace(
            environment=environment_cfg,
            alert_router=DummyAlertRouter(),
            core_config=SimpleNamespace(reporting=None),
        ),
        data_source=SimpleNamespace(storage=DummyStorage()),
        strategy_name="core_daily_trend",
        controller_name="daily_trend_core",
        risk_profile_name="balanced",
    )

    monkeypatch.setattr(run_daily_trend, "build_daily_trend_pipeline", lambda **_: pipeline)

    trading_controller = SimpleNamespace(
        maybe_report_health=lambda: None,
        process_signals=lambda signals: [],
    )

    monkeypatch.setattr(
        run_daily_trend,
        "create_trading_controller",
        lambda pipeline_arg, alert_router, **kwargs: trading_controller,
    )

    class DummyRunner:
        def __init__(self, *, controller: Any, trading_controller: Any, history_bars: int, clock=None) -> None:
            self._controller = controller
            self._clock = clock or (lambda: datetime.now(timezone.utc))
            self._history_bars = history_bars

        def run_once(self) -> list[OrderResult]:
            now = self._clock()
            tick_ms_local = int(getattr(self._controller, "tick_seconds", 86400.0) * 1000)
            end_local = int(now.timestamp() * 1000)
            start_local = max(0, end_local - self._history_bars * tick_ms_local)
            self._controller.collect_signals(start=start_local, end=end_local)
            return []

    monkeypatch.setattr(run_daily_trend, "DailyTrendRealtimeRunner", DummyRunner)

    def fake_export_smoke_report(
        *,
        report_dir: Path,
        results: Iterable[OrderResult],
        ledger: Iterable[Mapping[str, object]],
        window: Mapping[str, str],
        environment: str,
        alert_snapshot: Mapping[str, Mapping[str, str]],
        risk_state: Mapping[str, object] | None,
        data_checks: Mapping[str, object] | None = None,
        storage_info: Mapping[str, object] | None = None,
        note: str | None = None,
    ) -> Path:
        report_dirs.append(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "ledger.jsonl").write_text("", encoding="utf-8")
        payload = {
            "environment": environment,
            "window": dict(window),
            "orders": [],
            "ledger_entries": len(list(ledger)),
            "alert_snapshot": {channel: dict(data) for channel, data in alert_snapshot.items()},
        }
        if risk_state:
            payload["risk_state"] = dict(risk_state)
        if data_checks:
            payload["data_checks"] = json.loads(json.dumps(data_checks))
        if storage_info:
            payload["storage"] = json.loads(json.dumps(storage_info))
        if note:
            payload["note"] = note
        summary_path = report_dir / "summary.json"
        summary_path.write_text(json.dumps(payload), encoding="utf-8")
        return summary_path

    monkeypatch.setattr(run_daily_trend, "_export_smoke_report", fake_export_smoke_report)

    def fake_storage_health(directory: Path, *, min_free_mb: float | None) -> Mapping[str, object]:
        return {
            "directory": str(directory),
            "status": "low",
            "free_mb": 8.0,
            "total_mb": 256.0,
            "threshold_mb": float(min_free_mb or 0.0),
        }

    monkeypatch.setattr(run_daily_trend, "_collect_storage_health", fake_storage_health)

    caplog.set_level("ERROR")

    exit_code = run_daily_trend.main(
        [
            "--config",
            "config/core.yaml",
            "--environment",
            "binance_paper",
            "--paper-smoke",
            "--date-window",
            "2024-01-01:2024-01-05",
            "--smoke-output",
            str(base_output),
            "--smoke-min-free-mb",
            "64",
            "--smoke-fail-on-low-space",
        ]
    )

    assert exit_code == 4
    assert report_dirs, "powinien powstać katalog raportu"
    report_dir = report_dirs[0]
    summary_path = report_dir / "summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    storage = payload.get("storage", {})
    assert storage.get("status") == "low"
    summary_txt = report_dir / "summary.txt"
    assert summary_txt.exists()
    summary_text = summary_txt.read_text(encoding="utf-8")
    assert "Magazyn raportu:" in summary_text

    assert dispatch_calls, "powinien zostać wysłany alert"
    alert = dispatch_calls[0]
    assert alert.severity == "warning"
    assert alert.context.get("storage_status") == "low"

    error_messages = [record.message for record in caplog.records if "wolne miejsce" in record.message]
    assert error_messages, "powinien pojawić się log błędu o wolnym miejscu"


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


def test_paper_smoke_manifest_gap_allowed_with_threshold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts import run_daily_trend

    report_dirs: list[Path] = []
    dispatch_calls: list[Any] = []

    gap_minutes = 120.0
    threshold_minutes = 180.0

    instrument = InstrumentConfig(
        name="BTC",
        base_asset="BTC",
        quote_asset="USDT",
        categories=("majors",),
        exchange_symbols={"binance_spot": "BTCUSDT"},
        backfill_windows=(InstrumentBackfillWindow(interval="1d", lookback_days=60),),
    )
    universe = InstrumentUniverseConfig(
        name="paper_universe",
        description="test",
        instruments=(instrument,),
    )

    manifest_path = tmp_path / "ohlcv_manifest.sqlite"
    manifest_path.touch()

    environment_cfg = SimpleNamespace(
        environment=Environment.PAPER,
        risk_profile="balanced",
        instrument_universe="paper_universe",
        data_cache_path=str(tmp_path),
        exchange="binance_spot",
    )

    class DummyController:
        symbols = ("BTCUSDT",)
        interval = "1d"
        tick_seconds = 86400.0

        def collect_signals(self, *, start: int, end: int) -> list[Any]:
            return []

    class DummyBackfill:
        def synchronize(self, **kwargs: Any) -> None:
            return None

    class DummyExecutionService:
        def ledger(self) -> list[dict[str, Any]]:
            return []

    class DummyAlertRouter:
        def dispatch(self, message: Any) -> None:
            dispatch_calls.append(message)

        def health_snapshot(self) -> Mapping[str, Mapping[str, str]]:
            return {"telegram": {"status": "ok"}}

    pipeline = SimpleNamespace(
        controller=DummyController(),
        backfill_service=DummyBackfill(),
        execution_service=DummyExecutionService(),
        bootstrap=SimpleNamespace(
            environment=environment_cfg,
            alert_router=DummyAlertRouter(),
            core_config=SimpleNamespace(reporting=None, instrument_universes={"paper_universe": universe}),
        ),
        data_source=None,
        strategy_name="core_daily_trend",
        controller_name="daily_trend_core",
        risk_profile_name="balanced",
    )

    monkeypatch.setattr(run_daily_trend, "build_daily_trend_pipeline", lambda **_: pipeline)

    trading_controller = SimpleNamespace(
        maybe_report_health=lambda: None,
        process_signals=lambda signals: [],
    )
    monkeypatch.setattr(
        run_daily_trend,
        "create_trading_controller",
        lambda pipeline_arg, alert_router, **kwargs: trading_controller,
    )

    class DummyRunner:
        def __init__(self, *, controller: Any, trading_controller: Any, history_bars: int, clock=None) -> None:
            self._controller = controller

        def run_once(self) -> list[OrderResult]:
            return []

    monkeypatch.setattr(run_daily_trend, "DailyTrendRealtimeRunner", DummyRunner)

    monkeypatch.setattr(
        run_daily_trend,
        "_collect_storage_health",
        lambda directory, *, min_free_mb=None: {
            "directory": str(directory),
            "status": "ok",
            "free_mb": 1024.0,
            "total_mb": 2048.0,
            "threshold_mb": float(min_free_mb or 0.0),
        },
    )

    def fake_export_smoke_report(
        *,
        report_dir: Path,
        results: Iterable[OrderResult],
        ledger: Iterable[Mapping[str, object]],
        window: Mapping[str, str],
        environment: str,
        alert_snapshot: Mapping[str, Mapping[str, str]],
        risk_state: Mapping[str, object] | None,
        data_checks: Mapping[str, object] | None = None,
        storage_info: Mapping[str, object] | None = None,
        note: str | None = None,
    ) -> Path:
        report_dirs.append(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "environment": environment,
            "window": dict(window),
            "orders": [],
            "ledger_entries": len(list(ledger)),
            "alert_snapshot": alert_snapshot,
        }
        if risk_state:
            payload["risk_state"] = dict(risk_state)
        if data_checks:
            payload["data_checks"] = json.loads(json.dumps(data_checks))
        if storage_info:
            payload["storage"] = json.loads(json.dumps(storage_info))
        if note:
            payload["note"] = note
        summary_path = report_dir / "summary.json"
        summary_path.write_text(json.dumps(payload), encoding="utf-8")
        (report_dir / "summary.txt").write_text("summary", encoding="utf-8")
        return summary_path

    monkeypatch.setattr(run_daily_trend, "_export_smoke_report", fake_export_smoke_report)

    def fake_evaluate_coverage(*, as_of: datetime, intervals: Sequence[str], **_: Any) -> list[CoverageStatus]:
        as_of_ms = int(as_of.timestamp() * 1000)
        last_timestamp = as_of_ms - int(gap_minutes * 60_000)
        last_dt = datetime.fromtimestamp(last_timestamp / 1000, tz=timezone.utc)
        statuses: list[CoverageStatus] = []
        for interval in intervals:
            entry = ManifestEntry(
                symbol="BTCUSDT",
                interval=interval,
                row_count=500,
                last_timestamp_ms=last_timestamp,
                last_timestamp_iso=last_dt.isoformat(),
                gap_minutes=gap_minutes,
                threshold_minutes=2880,
                status="ok",
            )
            statuses.append(
                CoverageStatus(
                    symbol="BTCUSDT",
                    interval=interval,
                    manifest_entry=entry,
                    required_rows=500,
                    issues=(),
                )
            )
        return statuses

    monkeypatch.setattr(run_daily_trend, "evaluate_coverage", fake_evaluate_coverage)

    base_output = tmp_path / "reports"
    exit_code = run_daily_trend.main(
        [
            "--config",
            "config/core.yaml",
            "--environment",
            "binance_paper",
            "--paper-smoke",
            "--date-window",
            "2024-01-01:2024-01-10",
            "--smoke-output",
            str(base_output),
            "--smoke-max-gap-minutes",
            str(threshold_minutes),
        ]
    )

    assert exit_code == 0
    assert report_dirs
    summary_path = report_dirs[0] / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest_info = payload.get("data_checks", {}).get("manifest", {})
    summary = manifest_info.get("summary", {})
    assert pytest.approx(summary.get("max_gap_threshold_minutes", 0.0), rel=1e-6) == threshold_minutes
    thresholds = summary.get("thresholds", {})
    assert pytest.approx(thresholds.get("max_gap_minutes", 0.0), rel=1e-6) == threshold_minutes
    assert "min_ok_ratio" not in thresholds
    worst_gap = summary.get("worst_gap", {})
    assert pytest.approx(worst_gap.get("gap_minutes", 0.0), rel=1e-6) == gap_minutes
    assert summary.get("status") == "ok"
    assert summary.get("ok_ratio") == pytest.approx(1.0)
    assert dispatch_calls


def test_paper_smoke_manifest_uses_environment_threshold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts import run_daily_trend

    dispatch_calls: list[Any] = []
    report_dirs: list[Path] = []

    gap_minutes = 75.0
    threshold_minutes = 120.0

    instrument = InstrumentConfig(
        name="BTC",
        base_asset="BTC",
        quote_asset="USDT",
        categories=("majors",),
        exchange_symbols={"binance_spot": "BTCUSDT"},
        backfill_windows=(InstrumentBackfillWindow(interval="1d", lookback_days=60),),
    )
    universe = InstrumentUniverseConfig(
        name="paper_universe",
        description="test",
        instruments=(instrument,),
    )

    manifest_path = tmp_path / "ohlcv_manifest.sqlite"
    manifest_path.touch()

    environment_cfg = SimpleNamespace(
        environment=Environment.PAPER,
        risk_profile="balanced",
        instrument_universe="paper_universe",
        data_cache_path=str(tmp_path),
        exchange="binance_spot",
        data_quality=SimpleNamespace(max_gap_minutes=threshold_minutes, min_ok_ratio=0.8),
    )

    class DummyController:
        symbols = ("BTCUSDT",)
        interval = "1d"
        tick_seconds = 86400.0

        def collect_signals(self, *, start: int, end: int) -> list[Any]:
            return []

    class DummyBackfill:
        def synchronize(self, **kwargs: Any) -> None:
            return None

    class DummyExecutionService:
        def ledger(self) -> list[dict[str, Any]]:
            return []

    class DummyAlertRouter:
        def dispatch(self, message: Any) -> None:
            dispatch_calls.append(message)

        def health_snapshot(self) -> Mapping[str, Mapping[str, str]]:
            return {"telegram": {"status": "ok"}}

    pipeline = SimpleNamespace(
        controller=DummyController(),
        backfill_service=DummyBackfill(),
        execution_service=DummyExecutionService(),
        bootstrap=SimpleNamespace(
            environment=environment_cfg,
            alert_router=DummyAlertRouter(),
            core_config=SimpleNamespace(reporting=None, instrument_universes={"paper_universe": universe}),
        ),
        data_source=None,
        strategy_name="core_daily_trend",
        controller_name="daily_trend_core",
        risk_profile_name="balanced",
    )

    monkeypatch.setattr(run_daily_trend, "build_daily_trend_pipeline", lambda **_: pipeline)

    trading_controller = SimpleNamespace(
        maybe_report_health=lambda: None,
        process_signals=lambda signals: [],
    )
    monkeypatch.setattr(
        run_daily_trend,
        "create_trading_controller",
        lambda pipeline_arg, alert_router, **kwargs: trading_controller,
    )

    class DummyRunner:
        def __init__(self, *, controller: Any, trading_controller: Any, history_bars: int, clock=None) -> None:
            self._controller = controller

        def run_once(self) -> list[OrderResult]:
            return []

    monkeypatch.setattr(run_daily_trend, "DailyTrendRealtimeRunner", DummyRunner)

    monkeypatch.setattr(
        run_daily_trend,
        "_collect_storage_health",
        lambda directory, *, min_free_mb=None: {
            "directory": str(directory),
            "status": "ok",
            "free_mb": 2048.0,
            "total_mb": 4096.0,
            "threshold_mb": float(min_free_mb or 0.0),
        },
    )

    def fake_export_smoke_report(**kwargs: Any) -> Path:
        report_dir = kwargs["report_dir"]
        report_dirs.append(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "environment": kwargs["environment"],
            "window": dict(kwargs["window"]),
            "orders": [],
            "ledger_entries": 0,
            "alert_snapshot": kwargs["alert_snapshot"],
            "data_checks": json.loads(json.dumps(kwargs.get("data_checks", {}))),
        }
        note = kwargs.get("note")
        if note:
            summary["note"] = note
        summary_path = report_dir / "summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        (report_dir / "summary.txt").write_text("summary", encoding="utf-8")
        return summary_path

    monkeypatch.setattr(run_daily_trend, "_export_smoke_report", fake_export_smoke_report)

    def fake_evaluate_coverage(*, as_of: datetime, intervals: Sequence[str], **_: Any) -> list[CoverageStatus]:
        as_of_ms = int(as_of.timestamp() * 1000)
        last_timestamp = as_of_ms - int(gap_minutes * 60_000)
        last_dt = datetime.fromtimestamp(last_timestamp / 1000, tz=timezone.utc)
        statuses: list[CoverageStatus] = []
        for interval in intervals:
            entry = ManifestEntry(
                symbol="BTCUSDT",
                interval=interval,
                row_count=500,
                last_timestamp_ms=last_timestamp,
                last_timestamp_iso=last_dt.isoformat(),
                gap_minutes=gap_minutes,
                threshold_minutes=None,
                status="ok",
            )
            statuses.append(
                CoverageStatus(
                    symbol="BTCUSDT",
                    interval=interval,
                    manifest_entry=entry,
                    required_rows=500,
                    issues=(),
                )
            )
        return statuses

    monkeypatch.setattr(run_daily_trend, "evaluate_coverage", fake_evaluate_coverage)

    base_output = tmp_path / "reports"
    exit_code = run_daily_trend.main(
        [
            "--config",
            "config/core.yaml",
            "--environment",
            "binance_paper",
            "--paper-smoke",
            "--date-window",
            "2024-01-01:2024-01-10",
            "--smoke-output",
            str(base_output),
        ]
    )

    assert exit_code == 0
    assert report_dirs
    summary_path = report_dirs[0] / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest_info = payload.get("data_checks", {}).get("manifest", {})
    summary = manifest_info.get("summary", {})
    assert pytest.approx(summary.get("max_gap_threshold_minutes", 0.0), rel=1e-6) == threshold_minutes
    assert pytest.approx(summary.get("min_ok_ratio_threshold", 0.0), rel=1e-6) == 0.8
    thresholds = summary.get("thresholds", {})
    assert pytest.approx(thresholds.get("max_gap_minutes", 0.0), rel=1e-6) == threshold_minutes
    assert pytest.approx(thresholds.get("min_ok_ratio", 0.0), rel=1e-6) == 0.8
    worst_gap = summary.get("worst_gap", {})
    assert pytest.approx(worst_gap.get("gap_minutes", 0.0), rel=1e-6) == gap_minutes
    assert summary.get("status") == "ok"
    assert dispatch_calls


def test_paper_smoke_manifest_min_ok_ratio_blocks_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts import run_daily_trend

    report_dirs: list[Path] = []

    instrument = InstrumentConfig(
        name="BTC",
        base_asset="BTC",
        quote_asset="USDT",
        categories=("majors",),
        exchange_symbols={"binance_spot": "BTCUSDT"},
        backfill_windows=(InstrumentBackfillWindow(interval="1d", lookback_days=30),),
    )
    universe = InstrumentUniverseConfig(
        name="paper_universe",
        description="test",
        instruments=(instrument,),
    )

    manifest_path = tmp_path / "ohlcv_manifest.sqlite"
    manifest_path.touch()

    environment_cfg = SimpleNamespace(
        environment=Environment.PAPER,
        risk_profile="balanced",
        instrument_universe="paper_universe",
        data_cache_path=str(tmp_path),
        exchange="binance_spot",
    )

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

    class DummyAlertRouter:
        def dispatch(self, message: Any) -> None:
            return None

        def health_snapshot(self) -> Mapping[str, Mapping[str, str]]:
            return {"telegram": {"status": "ok"}}

    pipeline = SimpleNamespace(
        controller=DummyController(),
        backfill_service=DummyBackfill(),
        execution_service=DummyExecutionService(),
        bootstrap=SimpleNamespace(
            environment=environment_cfg,
            alert_router=DummyAlertRouter(),
            core_config=SimpleNamespace(reporting=None, instrument_universes={"paper_universe": universe}),
        ),
        data_source=None,
        strategy_name="core_daily_trend",
        controller_name="daily_trend_core",
        risk_profile_name="balanced",
    )

    monkeypatch.setattr(run_daily_trend, "build_daily_trend_pipeline", lambda **_: pipeline)
    monkeypatch.setattr(
        run_daily_trend,
        "create_trading_controller",
        lambda pipeline_arg, alert_router, **kwargs: SimpleNamespace(maybe_report_health=lambda: None, process_signals=lambda _: []),
    )
    monkeypatch.setattr(
        run_daily_trend,
        "DailyTrendRealtimeRunner",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("runner nie powinien być uruchomiony")),
    )

    def fake_evaluate_coverage(*, as_of: datetime, intervals: Sequence[str], **_: Any) -> list[CoverageStatus]:
        statuses: list[CoverageStatus] = []
        for index, interval in enumerate(intervals):
            entry = ManifestEntry(
                symbol="BTCUSDT",
                interval=interval,
                row_count=100,
                last_timestamp_ms=int(as_of.timestamp() * 1000),
                last_timestamp_iso=as_of.isoformat(),
                gap_minutes=0.0,
                threshold_minutes=1440,
                status="ok",
            )
            issues: tuple[str, ...] = ("insufficient_rows:10<120",) if index == 0 else ()
            statuses.append(
                CoverageStatus(
                    symbol="BTCUSDT",
                    interval=interval,
                    manifest_entry=entry,
                    required_rows=120,
                    issues=issues,
                )
            )
        return statuses

    monkeypatch.setattr(run_daily_trend, "evaluate_coverage", fake_evaluate_coverage)
    monkeypatch.setattr(run_daily_trend, "_collect_storage_health", lambda directory, *, min_free_mb=None: {})

    exit_code = run_daily_trend.main(
        [
            "--config",
            "config/core.yaml",
            "--environment",
            "binance_paper",
            "--paper-smoke",
            "--date-window",
            "2024-01-01:2024-01-10",
            "--smoke-min-ok-ratio",
            "0.9",
        ]
    )

    assert exit_code == 1

def test_paper_smoke_manifest_skips_unrequired_intervals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts import run_daily_trend

    end_dt = datetime(2024, 2, 15, 23, 59, 59, 999000, tzinfo=timezone.utc)
    end_ms = int(end_dt.timestamp() * 1000)

    class DummyController:
        symbols = ("BTCUSDT",)
        interval = "1d"
        tick_seconds = 86400.0

        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def collect_signals(self, *, start: int, end: int) -> list[Any]:
            self.calls.append((start, end))
            return []

    class DummyBackfill:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def synchronize(self, **kwargs: Any) -> None:
            self.calls.append(kwargs)

    class DummyExecutionService:
        def ledger(self) -> list[dict[str, Any]]:
            return []

    class DummyStorage:
        def __init__(self, rows: list[list[float]]) -> None:
            self._rows = rows

        def metadata(self) -> dict[str, str]:
            return {
                "row_count::BTCUSDT::1d": str(len(self._rows)),
                "last_timestamp::BTCUSDT::1d": str(int(self._rows[-1][0])),
            }

        def read(self, key: str) -> dict[str, Any]:
            assert key == "BTCUSDT::1d"
            return {"rows": self._rows}

    class DummyAlertRouter:
        def __init__(self) -> None:
            self.messages: list[Any] = []

        def dispatch(self, message: Any) -> None:
            self.messages.append(message)

        def health_snapshot(self) -> dict[str, Any]:
            return {}

    instrument = InstrumentConfig(
        name="BTC",
        base_asset="BTC",
        quote_asset="USDT",
        categories=("majors",),
        exchange_symbols={"binance_spot": "BTCUSDT"},
        backfill_windows=(
            InstrumentBackfillWindow(interval="1d", lookback_days=90),
            InstrumentBackfillWindow(interval="1h", lookback_days=30),
        ),
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
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            ("row_count::BTCUSDT::1d", "90"),
        )
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            ("last_timestamp::BTCUSDT::1d", str(end_ms)),
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

    rows = [
        [float(1703980800000 + index * 86_400_000), 0.0, 0.0, 0.0, 0.0, 0.0]
        for index in range(150)
    ]

    storage = DummyStorage(rows)
    controller = DummyController()
    backfill = DummyBackfill()

    pipeline = SimpleNamespace(
        controller=controller,
        backfill_service=backfill,
        execution_service=DummyExecutionService(),
        bootstrap=SimpleNamespace(
            environment=environment_cfg,
            alert_router=DummyAlertRouter(),
            core_config=core_config,
        ),
        data_source=SimpleNamespace(storage=storage),
        strategy_name="core_daily_trend",
        controller_name="daily_trend_core",
        strategy=SimpleNamespace(required_intervals=lambda: ("1d",)),
    )

    monkeypatch.setattr(run_daily_trend, "build_daily_trend_pipeline", lambda **_: pipeline)

    class DummyTradingController:
        def __init__(self) -> None:
            self.health_calls = 0
            self.process_calls: list[Sequence[Any]] = []

        def maybe_report_health(self) -> None:
            self.health_calls += 1

        def process_signals(self, signals: Sequence[Any]) -> list[Any]:
            self.process_calls.append(tuple(signals))
            return []

    trading_controller = DummyTradingController()

    class DummyRunner:
        instances: list["DummyRunner"] = []

        def __init__(self, controller, trading_controller, history_bars, clock):
            self.controller = controller
            self.trading_controller = trading_controller
            self.history_bars = history_bars
            self.clock = clock
            DummyRunner.instances.append(self)

        def run_once(self) -> list[Any]:
            return []

    monkeypatch.setattr(run_daily_trend, "create_trading_controller", lambda *args, **kwargs: trading_controller)
    monkeypatch.setattr(run_daily_trend, "DailyTrendRealtimeRunner", DummyRunner)

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

    assert exit_code == 0
    assert backfill.calls, "backfill powinien zostać wywołany"
    assert DummyRunner.instances, "runner powinien zostać utworzony"
    assert pipeline.bootstrap.alert_router.messages, "powinien zostać wysłany alert smoke"


def test_collect_required_intervals_prefers_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import run_daily_trend

    instrument = InstrumentConfig(
        name="BTC",
        base_asset="BTC",
        quote_asset="USDT",
        categories=("majors",),
        exchange_symbols={"binance_spot": "BTCUSDT"},
        backfill_windows=(
            InstrumentBackfillWindow(interval="1d", lookback_days=365),
            InstrumentBackfillWindow(interval="1h", lookback_days=30),
            InstrumentBackfillWindow(interval="15m", lookback_days=7),
        ),
    )
    universe = InstrumentUniverseConfig(
        name="paper_universe",
        description="test",
        instruments=(instrument,),
    )

    core_config = SimpleNamespace(instrument_universes={"paper_universe": universe})
    environment_cfg = SimpleNamespace(
        instrument_universe="paper_universe",
        exchange="binance_spot",
    )

    class DummyStrategy:
        def required_intervals(self) -> tuple[str, ...]:
            return ("1d", "1h")

    pipeline = SimpleNamespace(
        controller=SimpleNamespace(interval="1d"),
        strategy=DummyStrategy(),
        bootstrap=SimpleNamespace(environment=environment_cfg, core_config=core_config),
    )

    result = run_daily_trend._collect_required_intervals(pipeline, symbols=("BTCUSDT",))

    assert result == ("1d", "1h")


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


def test_render_smoke_summary_includes_operator_note() -> None:
    from scripts import run_daily_trend

    summary = {
        "environment": "paper",
        "window": {"start": "2024-01-01T00:00:00+00:00", "end": "2024-01-31T23:59:59+00:00"},
        "orders": [],
        "ledger_entries": 0,
        "alert_snapshot": {},
        "note": "Notatka\nDruga linia",
    }

    rendered = run_daily_trend._render_smoke_summary(summary=summary, summary_sha256="1234")

    assert "Notatka operatora: Notatka Druga linia" in rendered


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
    assert metrics["realized_pnl_total"] == pytest.approx(200.0)

    per_symbol = metrics["per_symbol"]
    assert set(per_symbol.keys()) == {"BTCUSDT"}
    btc_metrics = per_symbol["BTCUSDT"]
    assert btc_metrics["orders"] == 2
    assert btc_metrics["buy_orders"] == 1
    assert btc_metrics["sell_orders"] == 1
    assert btc_metrics["buy_notional"] == pytest.approx(4000.0)
    assert btc_metrics["sell_notional"] == pytest.approx(4200.0)
    assert btc_metrics["total_notional"] == pytest.approx(8200.0)
    assert btc_metrics["fees"] == pytest.approx(2.3)
    assert btc_metrics["net_quantity"] == pytest.approx(0.0)
    assert btc_metrics["realized_pnl"] == pytest.approx(200.0)
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
        "intervals": ["1d"],
        "cache": {
            "BTCUSDT": {
                "intervals": {
                    "1d": {
                        "coverage_bars": 3,
                        "required_bars": 3,
                        "row_count": 4,
                    }
                }
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
    cache_payload = summary["data_checks"]["cache"]["BTCUSDT"]["intervals"]["1d"]
    assert cache_payload["row_count"] == 4
    assert summary["data_checks"]["intervals"] == ["1d"]


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
            "realized_pnl_total": 150.5,
            "per_symbol": {
                "BTCUSDT": {
                    "orders": 3,
                    "total_notional": 2222.1,
                    "fees": 0.987654,
                    "realized_pnl": 200.0,
                },
                "ETHUSDT": {
                    "orders": 1,
                    "total_notional": 500.0,
                    "fees": 0.1234,
                    "net_quantity": 0.25,
                    "last_position_value": 125.0,
                    "realized_pnl": -49.5,
                },
            },
        },
    }

    text = run_daily_trend._render_smoke_summary(summary=summary, summary_sha256="abc123")

    assert "Zlecenia BUY/SELL: 2/1" in text
    assert "Wolumen BUY: 1 234.50 | SELL: 987.60 | Razem: 2 222.10" in text
    assert "Łączne opłaty: 0.9877" in text
    assert "Realizowany PnL (brutto): 150.50" in text
    assert "Ostatnia wartość pozycji: 4 321.00" in text
    assert "Instrumenty: BTCUSDT: zlecenia 3, wolumen 2 222.10, opłaty 0.9877, PnL 200.00" in text
    assert (
        "ETHUSDT: zlecenia 1, wolumen 500.00, opłaty 0.1234, netto +0.2500, wartość 125.00, PnL -49.50"
        in text
    )
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
            "positions": {
                "BTCUSDT": {"side": "long", "notional": 7_500.0},
                "ETHUSDT": {"side": "short", "notional": 2_500.0},
            },
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
    assert "Pozycje: BTCUSDT: LONG 7 500.00; ETHUSDT: SHORT 2 500.00" in text
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
                "intervals": ["1d"],
                "required_rows": {"1d": 31},
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
                "summary": {
                    "total": 1,
                    "ok": 1,
                    "error": 0,
                    "manifest_status_counts": {"ok": 1},
                    "status": "ok",
                    "ok_ratio": 1.0,
                    "by_interval": {
                        "1d": {
                            "total": 1,
                            "ok": 1,
                            "error": 0,
                            "manifest_status_counts": {"ok": 1},
                            "ok_ratio": 1.0,
                            "status": "ok",
                            "worst_gap": {
                                "symbol": "BTCUSDT",
                                "interval": "1d",
                                "gap_minutes": 0.0,
                            },
                        }
                    },
                    "by_symbol": {
                        "BTCUSDT": {
                            "total": 1,
                            "ok": 1,
                            "error": 0,
                            "manifest_status_counts": {"ok": 1},
                            "ok_ratio": 1.0,
                            "status": "ok",
                            "worst_gap": {
                                "symbol": "BTCUSDT",
                                "interval": "1d",
                                "gap_minutes": 0.0,
                            },
                        }
                    },
                },
            },
            "cache": {
                "BTCUSDT": {
                    "intervals": {
                        "1d": {
                            "coverage_bars": 31,
                            "required_bars": 31,
                            "row_count": 35,
                        }
                    }
                }
            },
        },
    }

    text = run_daily_trend._render_smoke_summary(summary=summary, summary_sha256="feedbeef")

    assert "Manifest OHLCV:" in text
    assert "Manifest – status agregowany:" in text
    assert "Manifest – agregaty:" in text
    assert "Manifest – interwały:" in text
    assert "Cache offline:" in text
    assert text.endswith("SHA-256 summary.json: feedbeef")
