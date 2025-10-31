from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence


import pytest


from bot_core.alerts import DefaultAlertRouter, InMemoryAlertAuditLog
from bot_core.execution import ExecutionService
from bot_core.observability import MetricsRegistry
from bot_core.ui.api import build_explainability_feed

from bot_core.exchanges.base import AccountSnapshot, OrderRequest, OrderResult
from bot_core.risk import RiskCheckResult, RiskEngine, RiskProfile
from bot_core.runtime import TradingController
from bot_core.runtime.journal import TradingDecisionEvent
from bot_core.strategies import SignalLeg, StrategySignal

from tests._alert_channel_helpers import CollectingChannel


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

    def snapshot_state(self, profile_name: str) -> Mapping[str, object]:
        return {
            "profile": profile_name,
            "total_equity": 100_000.0,
            "available_margin": 50_000.0,
        }

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
    channel = CollectingChannel(health_overrides={"latency_ms": "5"})
    router.register(channel)
    return router, channel, audit


def _signal(
    side: str = "BUY",
    *,
    quantity: float = 1.0,
    price: float = 100.0,
    confidence: float | None = None,
) -> StrategySignal:
    return StrategySignal(
        symbol="BTC/USDT",
        side=side,
        confidence=0.75 if confidence is None else confidence,
        metadata={"quantity": str(quantity), "price": str(price), "order_type": "market"},
    )


class StubTCOReporter:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def record_execution(self, **payload: object) -> None:  # type: ignore[override]
        self.calls.append(dict(payload))


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


def test_controller_handles_multi_leg_signal() -> None:
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

    signal = StrategySignal(
        symbol="BTC/USDT",
        side="arbitrage_entry",
        confidence=0.85,
        intent="multi_leg",
        metadata={"order_type": "market"},
        legs=(
            SignalLeg(
                symbol="BTC/USDT",
                side="BUY",
                quantity=1.5,
                metadata={"price": 101.0},
            ),
            SignalLeg(
                symbol="BTC/USDT",
                side="SELL",
                quantity=1.5,
                metadata={"price": 102.0},
            ),
        ),
    )

    results = controller.process_signals([signal])

    assert len(results) == 2
    assert [request.side for request in execution.requests] == ["BUY", "SELL"]
    assert all(message.category == "execution" for message in channel.messages if message.category == "execution")
    assert any(event["event"] == "order_executed" for event in journal.export())


def test_controller_skips_neutral_signal() -> None:
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

    signal = StrategySignal(
        symbol="BTC/USDT",
        side="rebalance_delta",
        confidence=0.4,
        intent="neutral",
        metadata={"target_ratio": 0.25},
    )

    results = controller.process_signals([signal])

    assert results == []
    assert execution.requests == []
    assert all(message.category != "execution" for message in channel.messages)
    exported = tuple(audit.export())
    assert exported == ()


def test_controller_reverses_position_before_opening_new_one() -> None:
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

    signal = StrategySignal(
        symbol="BTC/USDT",
        side="SELL",
        confidence=0.9,
        metadata={
            "quantity": "2",
            "price": "101",
            "order_type": "market",
            "current_position_qty": "1.5",
            "current_position_side": "LONG",
            "reverse_position": "true",
        },
    )

    results = controller.process_signals([signal])

    assert len(results) == 1
    assert len(execution.requests) == 2
    close_request, open_request = execution.requests
    assert close_request.side == "SELL"
    assert close_request.quantity == pytest.approx(1.5)
    assert close_request.metadata["action"] == "close"
    assert open_request.side == "SELL"
    assert open_request.quantity == pytest.approx(2.0)

    journal_events = journal.export()
    assert any(event["event"] == "order_close_for_reversal" for event in journal_events)


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


def test_controller_records_tco_event_with_reporter() -> None:
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, _, _ = _router_with_channel()
    journal = CollectingDecisionJournal()
    reporter = StubTCOReporter()

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
        strategy_name="core_trend",
        exchange_name="binance_spot",
        tco_reporter=reporter,
        tco_metadata={"source": "unit-test"},
    )

    controller.process_signals([_signal("BUY")])

    assert reporter.calls, "Reporter should be invoked"
    record = reporter.calls[0]
    assert record["strategy"] == "core_trend"
    assert record["exchange"] == "binance_spot"
    metadata = record.get("metadata", {})
    assert isinstance(metadata, Mapping)
    assert metadata.get("source") == "unit-test"
    assert metadata.get("controller") == "TradingController"


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


def test_controller_records_explainability_metadata() -> None:
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

    signal = StrategySignal(
        symbol="BTC/USDT",
        side="BUY",
        confidence=0.8,
        metadata={
            "quantity": "1",
            "price": "100",
            "order_type": "market",
            "decision_engine": {
                "explainability": {
                    "model": "stub",
                    "method": "perturbation",
                    "feature_importance": {"momentum": 0.5, "volume": -0.2},
                }
            },
        },
    )

    controller.process_signals([signal])

    assert execution.requests
    metadata = execution.requests[0].metadata
    assert "ai_explainability_json" in metadata
    events = journal.export()
    order_event = next(event for event in events if event["event"] == "order_submitted")
    assert "signal_ai_explainability_method" in order_event
    feed = build_explainability_feed(journal, limit=1)
    assert feed
    entry = feed[0]
    assert entry.model == "stub"
    assert entry.top_features


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


def test_controller_filters_signal_when_probability_below_threshold() -> None:
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, channel, audit = _router_with_channel()
    journal = CollectingDecisionJournal()

    class _StubOrchestrator:
        def __init__(self) -> None:
            self.invocations: list = []

    orchestrator = _StubOrchestrator()
    controller = TradingController(
        risk_engine=risk_engine,
        execution_service=execution,
        alert_router=router,
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        decision_orchestrator=orchestrator,
        decision_min_probability=0.7,
        decision_default_notional=2_500.0,
        decision_journal=journal,
    )

    signal = _signal("BUY", confidence=0.3)
    signal.metadata = {
        "quantity": "1.0",
        "price": "100.0",
        "order_type": "market",
        "expected_probability": 0.4,
        "expected_return_bps": 6.0,
    }

    results = controller.process_signals([signal])

    assert results == []
    assert orchestrator.invocations == []
    assert len(channel.messages) == 1
    decision_events = [event for event in journal.events if event.event_type == "decision_evaluation"]
    assert any(event.status == "filtered" for event in decision_events)
    exported = tuple(audit.export())
    assert len(exported) == 1
    assert exported[0]["category"] == "strategy"


def test_controller_skips_risk_when_orchestrator_rejects_signal() -> None:
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, channel, audit = _router_with_channel()
    journal = CollectingDecisionJournal()

    class _RejectingOrchestrator:
        def __init__(self) -> None:
            self.invocations: list = []

        def evaluate_candidate(self, candidate, _snapshot):
            self.invocations.append(candidate)
            return SimpleNamespace(
                candidate=candidate,
                accepted=False,
                reasons=("net_edge_below",),
                cost_bps=25.0,
                net_edge_bps=-2.0,
                model_name="gbm_v1",
            )

    orchestrator = _RejectingOrchestrator()
    controller = TradingController(
        risk_engine=risk_engine,
        execution_service=execution,
        alert_router=router,
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        decision_orchestrator=orchestrator,
        decision_min_probability=0.4,
        decision_journal=journal,
    )

    signal = _signal("BUY", confidence=0.9)
    signal.metadata = {
        "quantity": "1.0",
        "price": "100.0",
        "order_type": "market",
        "expected_probability": 0.9,
        "expected_return_bps": 12.0,
    }

    results = controller.process_signals([signal])

    assert results == []
    assert len(orchestrator.invocations) == 1
    assert risk_engine.last_checks == []
    decision_events = [event for event in journal.events if event.event_type == "decision_evaluation"]
    assert any(event.status == "rejected" for event in decision_events)
    assert len(channel.messages) == 1
    exported = tuple(audit.export())
    assert all(entry.get("category") != "execution" for entry in exported)


def test_controller_attaches_decision_metadata_for_execution() -> None:
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, _, _ = _router_with_channel()
    journal = CollectingDecisionJournal()

    class _AcceptingOrchestrator:
        def __init__(self) -> None:
            self.invocations: list = []

        def evaluate_candidate(self, candidate, _snapshot):
            self.invocations.append(candidate)
            return SimpleNamespace(
                candidate=candidate,
                accepted=True,
                reasons=(),
                cost_bps=12.0,
                net_edge_bps=8.0,
                model_name="gbm_v2",
                model_expected_return_bps=14.0,
                model_success_probability=0.72,
            )

    orchestrator = _AcceptingOrchestrator()
    controller = TradingController(
        risk_engine=risk_engine,
        execution_service=execution,
        alert_router=router,
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        decision_orchestrator=orchestrator,
        decision_min_probability=0.4,
        decision_journal=journal,
    )

    signal = _signal("BUY", confidence=0.8)
    signal.metadata = {
        "quantity": "1.0",
        "price": "100.0",
        "order_type": "market",
        "expected_probability": 0.8,
        "expected_return_bps": 15.0,
    }

    results = controller.process_signals([signal])

    assert len(results) == 1
    assert execution.requests, "Zlecenie powinno zostać złożone"
    metadata = execution.requests[0].metadata
    assert metadata is not None and "decision_engine" in metadata
    decision_meta = metadata["decision_engine"]
    assert decision_meta["accepted"] is True
    assert decision_meta["model"] == "gbm_v2"
    decision_events = [event for event in journal.events if event.event_type == "decision_evaluation"]
    assert any(event.status == "accepted" for event in decision_events)
