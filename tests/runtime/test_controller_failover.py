from __future__ import annotations

import logging
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


def _controller(
    monitor: ModelHealthMonitor,
) -> tuple[TradingController, DummyExecutionService, CollectingDecisionJournal]:
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
    assert any(
        event.event_type == "ai_failover" and event.status == "activated"
        for event in journal.events
    )
    assert any(
        event.event_type == "signal_skipped"
        and event.metadata.get("reason") == "ai_failover_active"
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
    assert any(
        event.event_type == "ai_failover" and event.status == "cleared" for event in journal.events
    )


def test_process_signals_survives_ai_monitor_snapshot_exception(caplog) -> None:
    class _ExplodingMonitor(ModelHealthMonitor):
        def snapshot(self):  # type: ignore[override]
            raise RuntimeError("snapshot boom")

    monitor = _ExplodingMonitor()
    controller, execution, journal = _controller(monitor)

    with caplog.at_level(logging.ERROR):
        controller.process_signals([_signal(mode="rules"), _signal(mode="ai")])

    assert [request.metadata.get("mode") for request in execution.requests] == ["rules"]
    assert any(
        event.event_type == "ai_failover"
        and event.status == "activated"
        and event.metadata.get("reason") == "ai_health_snapshot_error"
        for event in journal.events
    )
    assert "snapshot" in caplog.text.lower()


def test_controller_does_not_duplicate_failover_activation_on_repeated_snapshot_error(
    caplog,
) -> None:
    class _ExplodingMonitor(ModelHealthMonitor):
        def snapshot(self):  # type: ignore[override]
            raise RuntimeError("snapshot boom")

    monitor = _ExplodingMonitor()
    controller, execution, journal = _controller(monitor)

    with caplog.at_level(logging.ERROR):
        controller.process_signals([_signal(mode="rules"), _signal(mode="ai")])
        controller.process_signals([_signal(mode="rules"), _signal(mode="ai")])

    assert [request.metadata.get("mode") for request in execution.requests] == ["rules", "rules"]
    activated_events = [
        event
        for event in journal.events
        if event.event_type == "ai_failover" and event.status == "activated"
    ]
    assert len(activated_events) == 1
    assert activated_events[0].metadata.get("reason") == "ai_health_snapshot_error"
    assert caplog.text.lower().count("snapshot failed") >= 2


def test_controller_recovers_after_snapshot_error_is_resolved() -> None:
    class _FlakyMonitor(ModelHealthMonitor):
        def __init__(self) -> None:
            super().__init__()
            self.fail_snapshot = True

        def snapshot(self):  # type: ignore[override]
            if self.fail_snapshot:
                raise RuntimeError("snapshot boom")
            return super().snapshot()

    monitor = _FlakyMonitor()
    controller, execution, journal = _controller(monitor)

    controller.process_signals([_signal(mode="rules"), _signal(mode="ai")])
    first_cycle_modes = [request.metadata.get("mode") for request in execution.requests]
    assert first_cycle_modes == ["rules"]
    assert any(
        event.event_type == "ai_failover" and event.status == "activated"
        for event in journal.events
    )

    execution.requests.clear()
    monitor.fail_snapshot = False
    controller.process_signals([_signal(mode="rules"), _signal(mode="ai")])

    second_cycle_modes = [request.metadata.get("mode") for request in execution.requests]
    assert second_cycle_modes == ["ai", "rules"]
    assert any(
        event.event_type == "ai_failover" and event.status == "cleared"
        for event in journal.events
    )
