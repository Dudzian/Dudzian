from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

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


def test_paper_smoke_uses_date_window(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from scripts import run_daily_trend

    dispatch_calls: list[Any] = []
    sync_calls: list[dict[str, Any]] = []
    collected_calls: list[dict[str, int]] = []

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

    class DummyAlertRouter:
        def dispatch(self, message: Any) -> None:
            dispatch_calls.append(message)

        def health_snapshot(self) -> dict[str, Any]:
            return {}

    environment_cfg = SimpleNamespace(environment=Environment.PAPER, risk_profile="balanced")

    pipeline = SimpleNamespace(
        controller=DummyController(),
        backfill_service=DummyBackfill(),
        execution_service=DummyExecutionService(),
        bootstrap=SimpleNamespace(environment=environment_cfg, alert_router=DummyAlertRouter()),
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
            tick_ms = int(getattr(controller, "tick_seconds", 86400.0) * 1000)
            history = int(captured_runner["history_bars"])
            end_ms = int(now.timestamp() * 1000)
            start_ms = max(0, end_ms - history * tick_ms)
            controller.collect_signals(start=start_ms, end=end_ms)
            return []

    monkeypatch.setattr(run_daily_trend, "DailyTrendRealtimeRunner", DummyRunner)

    def fake_export_smoke_report(**kwargs: Any) -> Path:
        summary_path = tmp_path / "summary.json"
        summary_path.write_text("{}", encoding="utf-8")
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
