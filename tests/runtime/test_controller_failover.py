from __future__ import annotations

from datetime import timedelta

from bot_core.ai.health import ModelHealthMonitor
from bot_core.runtime import TradingController
from bot_core.strategies import StrategySignal

from tests.test_trading_controller import (
    CollectingDecisionJournal,
    DummyExecutionService,
    DummyRiskEngine,
    _account_snapshot,
    _router_with_channel,
)


def _signal(*, mode: str, side: str = "BUY") -> StrategySignal:
    return StrategySignal(
        symbol="BTC/USDT",
        side=side,
        confidence=0.8,
        metadata={
            "quantity": "1",
            "price": "100",
            "order_type": "market",
            "mode": mode,
        },
    )


def _controller(monitor: ModelHealthMonitor) -> tuple[TradingController, DummyExecutionService, CollectingDecisionJournal]:
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, _channel, _audit = _router_with_channel()
    journal = CollectingDecisionJournal()
    controller = TradingController(
        risk_engine=risk_engine,
        execution_service=execution,
        alert_router=router,
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-portfolio",
        environment="paper",
        risk_profile="balanced",
        health_check_interval=timedelta(hours=1),
        decision_journal=journal,
        ai_health_monitor=monitor,
        signal_mode_priorities={"ai": 100, "rules": 10},
        ai_signal_modes=("ai",),
        rules_signal_modes=("rules",),
    )
    return controller, execution, journal


def test_controller_prefers_ai_signals_when_healthy() -> None:
    monitor = ModelHealthMonitor()
    controller, execution, journal = _controller(monitor)

    controller.process_signals([_signal(mode="rules"), _signal(mode="ai")])

    assert [request.metadata.get("mode") for request in execution.requests] == ["ai", "rules"]
    assert all(event.event_type != "ai_failover" for event in journal.events)


def test_controller_skips_ai_signals_during_failover() -> None:
    monitor = ModelHealthMonitor()
    monitor.record_backend_failure(reason="backend_offline")
    controller, execution, journal = _controller(monitor)

    controller.process_signals([_signal(mode="rules"), _signal(mode="ai")])

    assert [request.metadata.get("mode") for request in execution.requests] == ["rules"]
    assert any(event.event_type == "ai_failover" and event.status == "activated" for event in journal.events)
    assert any(
        event.event_type == "signal_skipped" and event.metadata.get("reason") == "ai_failover_active"
        for event in journal.events
    )


def test_controller_restores_ai_after_failover() -> None:
    monitor = ModelHealthMonitor()
    controller, execution, journal = _controller(monitor)

    monitor.record_backend_failure(reason="temporary_outage")
    controller.process_signals([_signal(mode="rules"), _signal(mode="ai")])
    execution.requests.clear()
    journal.events.clear()

    monitor.resolve_backend_recovery()
    controller.process_signals([_signal(mode="rules"), _signal(mode="ai")])

    modes = [request.metadata.get("mode") for request in execution.requests]
    assert modes.count("ai") == 1
    assert modes.count("rules") == 1
    assert any(event.event_type == "ai_failover" and event.status == "cleared" for event in journal.events)
