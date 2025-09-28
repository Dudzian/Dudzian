from __future__ import annotations

import sys
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
        def __init__(self, *, controller: Any, trading_controller: Any, history_bars: int) -> None:
            assert controller is pipeline.controller
            assert trading_controller is trading_controller_obj
            captured_args["history_bars"] = history_bars
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
