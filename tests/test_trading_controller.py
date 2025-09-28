from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Sequence

import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.alerts import AlertMessage, DefaultAlertRouter, InMemoryAlertAuditLog
from bot_core.alerts.base import AlertChannel
from bot_core.execution import ExecutionService
from bot_core.observability import MetricsRegistry

from bot_core.exchanges.base import AccountSnapshot, OrderRequest, OrderResult
from bot_core.risk import RiskCheckResult, RiskEngine, RiskProfile
from bot_core.runtime import TradingController
from bot_core.runtime.journal import TradingDecisionEvent
from bot_core.strategies import StrategySignal


class CollectingChannel(AlertChannel):
    name = "collector"

    def __init__(self) -> None:
        self.messages: list[AlertMessage] = []

    def send(self, message: AlertMessage) -> None:
        self.messages.append(message)

    def health_check(self) -> Mapping[str, str]:
        return {"status": "ok", "latency_ms": "5"}


class CollectingDecisionJournal:
    def __init__(self) -> None:
        self.events: list[TradingDecisionEvent] = []

    def record(self, event: TradingDecisionEvent) -> None:
        self.events.append(event)

    def export(self) -> Sequence[Mapping[str, str]]:
        return tuple(event.as_dict() for event in self.events)


class DummyRiskEngine(RiskEngine):
    def __init__(self) -> None:
        self._result = RiskCheckResult(allowed=True)
        self._liquidate = False
        self.last_checks: list[tuple[OrderRequest, AccountSnapshot, str]] = []
        self._result_queue: list[RiskCheckResult] = []

    def register_profile(self, profile: RiskProfile) -> None:  # pragma: no cover - nieużywane
        return None

    def apply_pre_trade_checks(
        self,
        request: OrderRequest,
        *,
        account: AccountSnapshot,
        profile_name: str,
    ) -> RiskCheckResult:
        self.last_checks.append((request, account, profile_name))
        if self._result_queue:
            self._result = self._result_queue.pop(0)
        return self._result

    def on_fill(
        self,
        *,
        profile_name: str,
        symbol: str,
        side: str,
        position_value: float,
        pnl: float,
        timestamp: datetime | None = None,
    ) -> None:  # pragma: no cover - testy nie wymagają implementacji
        return None

    def should_liquidate(self, *, profile_name: str) -> bool:
        return self._liquidate

    def set_result(self, result: RiskCheckResult, *, liquidate: bool = False) -> None:
        self._result = result
        self._liquidate = liquidate
        self._result_queue.clear()

    def set_result_sequence(self, results: Sequence[RiskCheckResult], *, liquidate: bool = False) -> None:
        self._result_queue = list(results)
        if results:
            self._result = results[-1]
        self._liquidate = liquidate


class DummyExecutionService(ExecutionService):
    def __init__(self) -> None:
        self.requests: list[OrderRequest] = []

    def execute(self, request: OrderRequest, context) -> OrderResult:  # type: ignore[override]
        self.requests.append(request)
        return OrderResult(
            order_id="order-1",
            status="filled",
            filled_quantity=request.quantity,
            avg_price=request.price,
            raw_response={"context": context.metadata},
        )

    def cancel(self, order_id: str, context) -> None:  # type: ignore[override]
        return None

    def flush(self) -> None:
        return None


def _account_snapshot() -> AccountSnapshot:
    return AccountSnapshot(
        balances={},
        total_equity=100_000.0,
        available_margin=90_000.0,
        maintenance_margin=10_000.0,
    )


def _router_with_channel() -> tuple[DefaultAlertRouter, CollectingChannel, InMemoryAlertAuditLog]:
    audit = InMemoryAlertAuditLog()
    router = DefaultAlertRouter(audit_log=audit)
    channel = CollectingChannel()
    router.register(channel)
    return router, channel, audit


def _signal(side: str = "BUY", *, quantity: float = 1.0, price: float = 100.0) -> StrategySignal:
    return StrategySignal(
        symbol="BTC/USDT",
        side=side,
        confidence=0.75,
        metadata={"quantity": str(quantity), "price": str(price), "order_type": "market"},
    )


def test_controller_emits_alert_on_buy_signal() -> None:
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, channel, audit = _router_with_channel()
    journal = CollectingDecisionJournal()
    controller = TradingController(
        risk_engine=risk_engine,
        execution_service=execution,
        alert_router=router,
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        health_check_interval=timedelta(hours=1),
        decision_journal=journal,
    )

    results = controller.process_signals([_signal("BUY")])

    assert len(results) == 1
    assert execution.requests[0].side == "BUY"
    assert [message.category for message in channel.messages] == ["strategy", "execution"]
    assert channel.messages[1].severity == "info"
    exported = tuple(audit.export())
    assert len(exported) >= 2
    assert any(event["event"] == "order_executed" for event in journal.export())


def test_controller_alerts_on_risk_rejection_and_limit() -> None:
    risk_engine = DummyRiskEngine()
    risk_engine.set_result(
        RiskCheckResult(allowed=False, reason="Przekroczono dzienny limit straty."),
        liquidate=True,
    )
    execution = DummyExecutionService()
    router, channel, audit = _router_with_channel()
    journal = CollectingDecisionJournal()
    controller = TradingController(
        risk_engine=risk_engine,
        execution_service=execution,
        alert_router=router,
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        health_check_interval=timedelta(hours=1),
        decision_journal=journal,
    )

    results = controller.process_signals([_signal("SELL")])

    assert results == []
    severities = [message.severity for message in channel.messages]
    assert severities == ["info", "warning", "critical"]
    titles = [message.title for message in channel.messages]
    assert "Profil w trybie awaryjnym" in titles[2]
    exported = tuple(audit.export())
    assert len(exported) == 3
    assert exported[2]["severity"] == "critical"
    events = [event["event"] for event in journal.export()]
    assert "risk_rejected" in events


def test_controller_runs_health_report() -> None:
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, channel, _ = _router_with_channel()

    def clock() -> datetime:
        return datetime(2024, 1, 1, tzinfo=timezone.utc)

    controller = TradingController(
        risk_engine=risk_engine,
        execution_service=execution,
        alert_router=router,
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        health_check_interval=timedelta(seconds=0),
        clock=clock,
        decision_journal=CollectingDecisionJournal(),
    )

    controller.maybe_report_health(force=True)

    assert len(channel.messages) == 1


def test_controller_scales_quantity_when_risk_suggests_limit() -> None:
    risk_engine = DummyRiskEngine()
    disallowed = RiskCheckResult(
        allowed=False,
        reason="Limit ekspozycji przekroczony",
        adjustments={"max_quantity": 0.25},
    )
    allowed = RiskCheckResult(allowed=True)
    risk_engine.set_result_sequence([disallowed, allowed])

    execution = DummyExecutionService()
    router, channel, audit = _router_with_channel()
    journal = CollectingDecisionJournal()
    controller = TradingController(
        risk_engine=risk_engine,
        execution_service=execution,
        alert_router=router,
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        health_check_interval=timedelta(hours=1),
        decision_journal=journal,
    )

    results = controller.process_signals([_signal("BUY", quantity=1.0, price=100.0)])

    assert len(results) == 1
    assert execution.requests[0].quantity == pytest.approx(0.25)
    assert risk_engine.last_checks[0][0].quantity == pytest.approx(1.0)
    assert risk_engine.last_checks[1][0].quantity == pytest.approx(0.25)
    exported = tuple(audit.export())
    assert any(entry["channel"] == "collector" and entry["category"] == "strategy" for entry in exported)
    assert channel.messages[-1].category == "execution"
    events = [event["event"] for event in journal.export()]
    assert "risk_adjusted" in events


def test_decision_journal_event_order() -> None:
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, _, _ = _router_with_channel()
    journal = CollectingDecisionJournal()

    controller = TradingController(
        risk_engine=risk_engine,
        execution_service=execution,
        alert_router=router,
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        health_check_interval=timedelta(hours=1),
        decision_journal=journal,
    )

    controller.process_signals([_signal("BUY")])

    events = [event["event"] for event in journal.export()]
    assert events[:3] == ["signal_received", "risk_check_passed", "order_submitted"]
    assert "order_executed" in events


def test_controller_updates_metrics_counters_and_gauge() -> None:
    registry = MetricsRegistry()
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, _, _ = _router_with_channel()

    controller = TradingController(
        risk_engine=risk_engine,
        execution_service=execution,
        alert_router=router,
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        health_check_interval=timedelta(seconds=0),
        metrics_registry=registry,
    )

    controller.process_signals([_signal("BUY")])
    controller.maybe_report_health(force=True)

    base_labels = {"environment": "paper", "portfolio": "paper-1", "risk_profile": "balanced"}
    symbol_labels = {**base_labels, "symbol": "BTC/USDT"}

    signals_counter = registry.counter(
        "trading_signals_total",
        "Liczba sygnałów przetworzonych w TradingController (status=received/accepted/rejected).",
    )
    assert signals_counter.value(labels={**symbol_labels, "status": "received"}) == 1.0
    assert signals_counter.value(labels={**symbol_labels, "status": "accepted"}) == 1.0
    assert signals_counter.value(labels={**symbol_labels, "status": "rejected"}) == 0.0

    orders_counter = registry.counter(
        "trading_orders_total",
        "Liczba zleceń obsłużonych przez TradingController (result=submitted/executed/failed).",
    )
    assert orders_counter.value(labels={**symbol_labels, "result": "submitted", "side": "BUY"}) == 1.0
    assert orders_counter.value(labels={**symbol_labels, "result": "executed", "side": "BUY"}) == 1.0
    assert orders_counter.value(labels={**symbol_labels, "result": "failed", "side": "BUY"}) == 0.0

    health_counter = registry.counter(
        "trading_health_reports_total",
        "Liczba wysłanych raportów health-check przez TradingController.",
    )
    assert health_counter.value(labels=base_labels) == 1.0

    liquidation_gauge = registry.gauge(
        "trading_liquidation_state",
        "Stan trybu awaryjnego profilu ryzyka (1=liquidation, 0=normal).",
    )
    assert liquidation_gauge.value(labels=base_labels) == 0.0


def test_liquidation_metric_reflects_force_state() -> None:
    registry = MetricsRegistry()
    risk_engine = DummyRiskEngine()
    risk_engine.set_result(
        RiskCheckResult(allowed=False, reason="Przekroczono dzienny limit straty."),
        liquidate=True,
    )
    execution = DummyExecutionService()
    router, channel, _ = _router_with_channel()

    controller = TradingController(
        risk_engine=risk_engine,
        execution_service=execution,
        alert_router=router,
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        health_check_interval=timedelta(hours=1),
        metrics_registry=registry,
    )

    controller.process_signals([_signal("SELL")])

    base_labels = {"environment": "paper", "portfolio": "paper-1", "risk_profile": "balanced"}

    liquidation_gauge = registry.gauge(
        "trading_liquidation_state",
        "Stan trybu awaryjnego profilu ryzyka (1=liquidation, 0=normal).",
    )
    assert liquidation_gauge.value(labels=base_labels) == 1.0
    assert any(msg.severity == "critical" for msg in channel.messages)


class FailingExecutionService(ExecutionService):
    def execute(self, request: OrderRequest, context) -> OrderResult:  # type: ignore[override]
        raise RuntimeError("API niedostępne")

    def cancel(self, order_id: str, context) -> None:  # type: ignore[override]
        return None

    def flush(self) -> None:
        return None


def test_controller_emits_alert_on_execution_error() -> None:
    risk_engine = DummyRiskEngine()
    execution = FailingExecutionService()
    router, channel, audit = _router_with_channel()
    controller = TradingController(
        risk_engine=risk_engine,
        execution_service=execution,
        alert_router=router,
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        health_check_interval=timedelta(hours=1),
    )

    with pytest.raises(RuntimeError):
        controller.process_signals([_signal("SELL")])

    categories = [message.category for message in channel.messages]
    assert categories == ["strategy", "execution"]
    assert channel.messages[1].severity == "critical"
    exported = tuple(audit.export())
    assert len(exported) == 2
    assert exported[1]["severity"] == "critical"
