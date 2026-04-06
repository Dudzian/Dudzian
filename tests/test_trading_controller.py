from __future__ import annotations

import logging
import tempfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence


import pytest


from bot_core.alerts import DefaultAlertRouter, InMemoryAlertAuditLog
from bot_core.execution import ExecutionService
from bot_core.observability import MetricsRegistry
from bot_core.ui.api import build_explainability_feed

from bot_core.ai.opportunity_lifecycle import (
    OpportunityLifecycleService,
    OpportunityPerformanceSnapshotConfig,
)
from bot_core.ai.trading_opportunity_shadow import (
    OpportunityOutcomeLabel,
    OpportunityShadowContext,
    OpportunityShadowRecord,
    OpportunityShadowRepository,
    OpportunityThresholdConfig,
)
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

    def set_result_sequence(
        self, results: Sequence[RiskCheckResult], *, liquidate: bool = False
    ) -> None:
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


class StatusExecutionService(ExecutionService):
    _USE_REQUEST_FILLED_QUANTITY = object()

    def __init__(
        self,
        *,
        status: str,
        filled_quantity: float | None | object = _USE_REQUEST_FILLED_QUANTITY,
        avg_price: float | None = 100.0,
    ) -> None:
        self.status = status
        self._filled_quantity = filled_quantity
        self._avg_price = avg_price
        self.requests: list[OrderRequest] = []

    def execute(self, request: OrderRequest, context) -> OrderResult:  # type: ignore[override]
        self.requests.append(request)
        return OrderResult(
            order_id="order-status-1",
            status=self.status,
            filled_quantity=request.quantity
            if self._filled_quantity is self._USE_REQUEST_FILLED_QUANTITY
            else self._filled_quantity,
            avg_price=self._avg_price,
            raw_response={"context": context.metadata},
        )

    def cancel(self, order_id: str, context) -> None:  # type: ignore[override]
        return None

    def flush(self) -> None:
        return None


class SequencedExecutionService(ExecutionService):
    def __init__(self, responses: Sequence[dict[str, object]]) -> None:
        self._responses = list(responses)
        self.requests: list[OrderRequest] = []

    def execute(self, request: OrderRequest, context) -> OrderResult:  # type: ignore[override]
        self.requests.append(request)
        if self._responses:
            payload = dict(self._responses.pop(0))
        else:
            payload = {}
        return OrderResult(
            order_id=str(payload.get("order_id", "order-seq")),
            status=str(payload.get("status", "filled")),
            filled_quantity=payload.get("filled_quantity", request.quantity),
            avg_price=payload.get("avg_price", request.price),
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


def _opportunity_autonomy_signal(
    mode: str,
    *,
    side: str = "BUY",
    assisted_approval: bool | None = None,
    include_mode: bool = True,
    include_decision_payload: bool = False,
    decision_effective_mode: str | None = None,
    decision_primary_reason: str | None = None,
    performance_guard_effective_mode: str | None = None,
    performance_guard_primary_reason: str | None = None,
    performance_guard_hard_breach: bool | None = None,
    performance_guard_blocked: bool | None = None,
    model_version: str | None = None,
    decision_source: str | None = None,
    decision_payload_model_version: str | None = None,
    decision_payload_decision_source: str | None = None,
) -> StrategySignal:
    signal = _signal(side=side)
    metadata: dict[str, object] = dict(signal.metadata)
    metadata["opportunity_shadow_record_key"] = "shadow-key-1"
    if include_mode:
        metadata["opportunity_autonomy_mode"] = mode
    if include_decision_payload:
        metadata["opportunity_autonomy_decision"] = {}
    if decision_effective_mode is not None:
        decision_payload = metadata.get("opportunity_autonomy_decision")
        if not isinstance(decision_payload, dict):
            decision_payload = {}
        decision_payload["effective_mode"] = decision_effective_mode
        metadata["opportunity_autonomy_decision"] = decision_payload
    if decision_primary_reason is not None:
        decision_payload = metadata.get("opportunity_autonomy_decision")
        if not isinstance(decision_payload, dict):
            decision_payload = {}
        decision_payload["primary_reason"] = decision_primary_reason
        metadata["opportunity_autonomy_decision"] = decision_payload
    if decision_payload_model_version is not None:
        decision_payload = metadata.get("opportunity_autonomy_decision")
        if not isinstance(decision_payload, dict):
            decision_payload = {}
        decision_payload["model_version"] = decision_payload_model_version
        metadata["opportunity_autonomy_decision"] = decision_payload
    if decision_payload_decision_source is not None:
        decision_payload = metadata.get("opportunity_autonomy_decision")
        if not isinstance(decision_payload, dict):
            decision_payload = {}
        decision_payload["decision_source"] = decision_payload_decision_source
        metadata["opportunity_autonomy_decision"] = decision_payload
    if (
        performance_guard_effective_mode is not None
        or performance_guard_primary_reason is not None
        or performance_guard_hard_breach is not None
        or performance_guard_blocked is not None
    ):
        decision_payload = metadata.get("opportunity_autonomy_decision")
        if not isinstance(decision_payload, dict):
            decision_payload = {}
        performance_guard_payload = decision_payload.get("performance_guard")
        if not isinstance(performance_guard_payload, dict):
            performance_guard_payload = {}
        if performance_guard_effective_mode is not None:
            performance_guard_payload["effective_mode"] = performance_guard_effective_mode
        if performance_guard_primary_reason is not None:
            performance_guard_payload["primary_reason"] = performance_guard_primary_reason
        if performance_guard_hard_breach is not None:
            performance_guard_payload["hard_breach"] = performance_guard_hard_breach
            performance_guard_payload["performance_guard_applied"] = True
        if performance_guard_blocked is not None:
            performance_guard_payload["blocked"] = performance_guard_blocked
            performance_guard_payload["performance_guard_applied"] = True
        decision_payload["performance_guard"] = performance_guard_payload
        metadata["opportunity_autonomy_decision"] = decision_payload
    metadata["opportunity_autonomy_primary_reason"] = f"reason:{mode}"
    if assisted_approval is not None:
        metadata["autonomy_assisted_approval"] = assisted_approval
    if model_version is not None:
        metadata["opportunity_model_version"] = model_version
    if decision_source is not None:
        metadata["opportunity_decision_source"] = decision_source
    signal.metadata = metadata
    return signal


def _last_event(journal: CollectingDecisionJournal, event_type: str) -> Mapping[str, str]:
    events = [event for event in journal.export() if event.get("event") == event_type]
    assert events, f"Missing event {event_type}"
    return events[-1]


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
    assert execution.requests[0].client_order_id
    assert channel.messages[1].context["client_order_id"] == execution.requests[0].client_order_id
    exported = tuple(audit.export())
    assert len(exported) >= 2
    assert any(event["event"] == "order_executed" for event in journal.export())


def _build_autonomy_controller(
    *,
    environment: str,
    opportunity_shadow_repository: OpportunityShadowRepository | None = None,
    order_metadata_defaults: Mapping[str, object] | None = None,
    performance_guard_recent_final_window_size: int | None = None,
    performance_guard_max_scan_labels: int | None = None,
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
        portfolio_id=f"{environment}-1",
        environment=environment,
        risk_profile="balanced",
        order_metadata_defaults=order_metadata_defaults,
        decision_journal=journal,
        opportunity_shadow_repository=opportunity_shadow_repository,
        performance_guard_recent_final_window_size=performance_guard_recent_final_window_size,
        performance_guard_max_scan_labels=performance_guard_max_scan_labels,
    )
    return controller, execution, journal


def _autonomy_shadow_repository_with_final_outcomes(
    realized_return_bps: Sequence[float],
    *,
    environment: str,
    portfolio_id: str,
) -> OpportunityShadowRepository:
    repo_dir = Path(tempfile.mkdtemp(prefix="autonomy-shadow-"))
    repository = OpportunityShadowRepository(repo_dir)
    labels = [
        OpportunityOutcomeLabel(
            symbol="BTCUSDT",
            decision_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index),
            correlation_key=f"perf-{index}",
            horizon_minutes=15,
            realized_return_bps=value,
            max_favorable_excursion_bps=max(value, 0.0),
            max_adverse_excursion_bps=min(value, 0.0),
            provenance={"environment": environment, "portfolio_id": portfolio_id},
            label_quality="final",
        )
        for index, value in enumerate(realized_return_bps)
    ]
    if labels:
        repository.append_outcome_labels(labels)
    return repository


def _autonomy_shadow_repository_with_mixed_scope_outcomes(
    rows: Sequence[tuple[float, str, str]],
) -> OpportunityShadowRepository:
    repo_dir = Path(tempfile.mkdtemp(prefix="autonomy-shadow-scope-"))
    repository = OpportunityShadowRepository(repo_dir)
    labels = [
        OpportunityOutcomeLabel(
            symbol="BTCUSDT",
            decision_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index),
            correlation_key=f"scope-{index}",
            horizon_minutes=15,
            realized_return_bps=value,
            max_favorable_excursion_bps=max(value, 0.0),
            max_adverse_excursion_bps=min(value, 0.0),
            provenance={"environment": environment, "portfolio_id": portfolio_id},
            label_quality="final",
        )
        for index, (value, environment, portfolio_id) in enumerate(rows)
    ]
    if labels:
        repository.append_outcome_labels(labels)
    return repository


def _autonomy_shadow_repository_with_mixed_lineage_outcomes(
    rows: Sequence[tuple[float, str, str, str, str]],
) -> OpportunityShadowRepository:
    repo_dir = Path(tempfile.mkdtemp(prefix="autonomy-shadow-lineage-"))
    repository = OpportunityShadowRepository(repo_dir)
    labels = [
        OpportunityOutcomeLabel(
            symbol="BTCUSDT",
            decision_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index),
            correlation_key=f"lineage-{index}",
            horizon_minutes=15,
            realized_return_bps=value,
            max_favorable_excursion_bps=max(value, 0.0),
            max_adverse_excursion_bps=min(value, 0.0),
            provenance={
                "environment": environment,
                "portfolio_id": portfolio_id,
                "model_version": model_version,
                "decision_source": decision_source,
            },
            label_quality="final",
        )
        for index, (value, environment, portfolio_id, model_version, decision_source) in enumerate(
            rows
        )
    ]
    if labels:
        repository.append_outcome_labels(labels)
    return repository


def test_opportunity_autonomy_denied_blocks_execution() -> None:
    controller, execution, journal = _build_autonomy_controller(
        environment="paper",
        opportunity_shadow_repository=_autonomy_shadow_repository_with_final_outcomes(
            [4.0, 3.0], environment="paper", portfolio_id="paper-1"
        ),
    )
    result = controller.process_signals([_opportunity_autonomy_signal("denied")])
    assert result == []
    assert execution.requests == []
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["status"] == "blocked"
    assert event["autonomy_mode"] == "denied"
    assert event["blocking_reason"] == "autonomy_mode_denied"


def test_opportunity_autonomy_shadow_only_blocks_paper_and_live() -> None:
    for environment in ("paper", "live"):
        controller, execution, journal = _build_autonomy_controller(
            environment=environment,
            opportunity_shadow_repository=_autonomy_shadow_repository_with_final_outcomes(
                [4.0, 3.0],
                environment=environment,
                portfolio_id=f"{environment}-1",
            ),
        )
        result = controller.process_signals([_opportunity_autonomy_signal("shadow_only")])
        assert result == []
        assert execution.requests == []
        event = _last_event(journal, "opportunity_autonomy_enforcement")
        assert event["status"] == "blocked"
        assert event["autonomy_mode"] == "shadow_only"
        assert event["blocking_reason"] == "autonomy_mode_shadow_only_blocks_order_execution"


def test_opportunity_autonomy_paper_autonomous_allows_paper_but_blocks_live() -> None:
    paper_controller, paper_execution, paper_journal = _build_autonomy_controller(
        environment="paper",
        opportunity_shadow_repository=_autonomy_shadow_repository_with_final_outcomes(
            [4.0, 3.0], environment="paper", portfolio_id="paper-1"
        ),
    )
    paper_result = paper_controller.process_signals(
        [_opportunity_autonomy_signal("paper_autonomous")]
    )
    assert len(paper_result) == 1
    assert len(paper_execution.requests) == 1
    paper_event = _last_event(paper_journal, "opportunity_autonomy_enforcement")
    assert paper_event["status"] == "allowed"
    assert paper_event["execution_permission"] == "allowed"

    live_controller, live_execution, live_journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=_autonomy_shadow_repository_with_final_outcomes(
            [4.0, 3.0], environment="live", portfolio_id="live-1"
        ),
    )
    live_result = live_controller.process_signals(
        [_opportunity_autonomy_signal("paper_autonomous")]
    )
    assert live_result == []
    assert live_execution.requests == []
    live_event = _last_event(live_journal, "opportunity_autonomy_enforcement")
    assert live_event["status"] == "blocked"
    assert live_event["blocking_reason"] == "paper_autonomy_blocks_live_environment"


def test_opportunity_autonomy_live_assisted_blocks_live_without_approval() -> None:
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=_autonomy_shadow_repository_with_final_outcomes(
            [16.0, 14.0, 12.0, 10.0, 8.0, 6.0],
            environment="live",
            portfolio_id="live-1",
        ),
    )
    result = controller.process_signals(
        [_opportunity_autonomy_signal("live_assisted", assisted_approval=False)]
    )
    assert result == []
    assert execution.requests == []
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["status"] == "blocked"
    assert event["blocking_reason"] == "live_assisted_requires_explicit_approval"
    assert event["assisted_override_used"] == "false"


def test_opportunity_autonomy_live_assisted_allows_live_with_approval() -> None:
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=_autonomy_shadow_repository_with_final_outcomes(
            [16.0, 14.0, 12.0, 10.0, 8.0, 6.0],
            environment="live",
            portfolio_id="live-1",
        ),
    )
    result = controller.process_signals(
        [_opportunity_autonomy_signal("live_assisted", assisted_approval=True)]
    )
    assert len(result) == 1
    assert len(execution.requests) == 1
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["status"] == "allowed"
    assert event["assisted_override_used"] == "true"


def test_opportunity_autonomy_live_autonomous_allows_live_execution() -> None:
    controller, execution, _journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=_autonomy_shadow_repository_with_final_outcomes(
            [9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
            environment="live",
            portfolio_id="live-1",
        ),
    )
    result = controller.process_signals([_opportunity_autonomy_signal("live_autonomous")])
    assert len(result) == 1
    assert len(execution.requests) == 1


def test_opportunity_autonomy_enforcement_uses_effective_mode_from_decision_payload() -> None:
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=_autonomy_shadow_repository_with_final_outcomes(
            [9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
            environment="live",
            portfolio_id="live-1",
        ),
    )
    result = controller.process_signals(
        [
            _opportunity_autonomy_signal(
                "live_autonomous",
                include_decision_payload=True,
                decision_effective_mode="paper_autonomous",
            )
        ]
    )
    assert result == []
    assert execution.requests == []
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["status"] == "blocked"
    assert event["autonomy_mode"] == "paper_autonomous"
    assert event["blocking_reason"] == "paper_autonomy_blocks_live_environment"


def test_opportunity_autonomy_enforcement_prefers_payload_primary_reason_for_effective_mode() -> (
    None
):
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=_autonomy_shadow_repository_with_final_outcomes(
            [9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
            environment="live",
            portfolio_id="live-1",
        ),
    )
    result = controller.process_signals(
        [
            _opportunity_autonomy_signal(
                "live_autonomous",
                include_decision_payload=True,
                decision_effective_mode="paper_autonomous",
                decision_primary_reason="downgraded_to_paper_due_to_quality",
            )
        ]
    )
    assert result == []
    assert execution.requests == []
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["autonomy_mode"] == "paper_autonomous"
    assert event["autonomy_primary_reason"] == "downgraded_to_paper_due_to_quality"
    assert event["performance_guard_primary_reason"] == "performance_guard_no_breach"
    assert event["performance_guard_source"] == "local_snapshot_source_of_truth"


def test_opportunity_autonomy_enforcement_prefers_more_conservative_performance_guard_mode() -> (
    None
):
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=_autonomy_shadow_repository_with_final_outcomes(
            [9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
            environment="live",
            portfolio_id="live-1",
        ),
    )
    result = controller.process_signals(
        [
            _opportunity_autonomy_signal(
                "live_autonomous",
                include_decision_payload=True,
                decision_effective_mode="live_assisted",
                decision_primary_reason="downgrade_governance",
                performance_guard_effective_mode="paper_autonomous",
                performance_guard_primary_reason="performance_soft_breach",
                performance_guard_hard_breach=False,
            )
        ]
    )
    assert result == []
    assert execution.requests == []
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["autonomy_mode"] == "live_assisted"
    assert event["autonomy_primary_reason"] == "downgrade_governance"
    assert event["performance_guard_source"] == "local_snapshot_source_of_truth"
    assert event["performance_guard_primary_reason"] == "performance_guard_no_breach"
    assert event["performance_guard_applied"] == "false"
    assert event["performance_guard_effective_mode"] == "live_assisted"
    assert event["blocking_reason"] == "live_assisted_requires_explicit_approval"


def test_opportunity_autonomy_enforcement_blocks_when_performance_guard_sets_blocked_true() -> None:
    controller, execution, journal = _build_autonomy_controller(
        environment="paper",
        opportunity_shadow_repository=_autonomy_shadow_repository_with_final_outcomes(
            [-12.0, -11.0, -10.0, -9.0, -8.0],
            environment="paper",
            portfolio_id="paper-1",
        ),
    )
    result = controller.process_signals(
        [
            _opportunity_autonomy_signal(
                "live_autonomous",
                include_decision_payload=True,
                decision_effective_mode="live_assisted",
                performance_guard_effective_mode="live_autonomous",
                performance_guard_primary_reason="manual_payload_should_not_override_local_guard",
                performance_guard_hard_breach=False,
                performance_guard_blocked=False,
                assisted_approval=True,
            )
        ]
    )
    assert result == []
    assert execution.requests == []
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["status"] == "blocked"
    assert event["blocking_reason"] == "performance_guard_local_kill_switch"
    assert event["performance_guard_applied"] == "true"
    assert event["performance_guard_blocked"] == "true"
    assert event["performance_guard_block_enforced"] == "true"
    assert event["performance_guard_hard_breach"] == "true"
    assert event["performance_guard_primary_reason"] == "hard_performance_breach_detected"
    assert event["execution_permission"] == "blocked"


def test_opportunity_shadow_key_without_autonomy_contract_does_not_trigger_enforcement() -> None:
    controller, execution, journal = _build_autonomy_controller(
        environment="paper",
        opportunity_shadow_repository=_autonomy_shadow_repository_with_final_outcomes(
            [4.0, 3.0], environment="paper", portfolio_id="paper-1"
        ),
    )
    result = controller.process_signals(
        [_opportunity_autonomy_signal("live_autonomous", include_mode=False)]
    )
    assert len(result) == 1
    assert len(execution.requests) == 1
    autonomy_events = [
        event
        for event in journal.export()
        if event.get("event") == "opportunity_autonomy_enforcement"
    ]
    assert autonomy_events == []


def test_opportunity_autonomy_permission_failure_fails_closed() -> None:
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=_autonomy_shadow_repository_with_final_outcomes(
            [4.0, 3.0], environment="live", portfolio_id="live-1"
        ),
    )
    result = controller.process_signals(
        [
            _opportunity_autonomy_signal(
                "live_autonomous",
                include_mode=False,
                include_decision_payload=True,
            )
        ]
    )
    assert result == []
    assert execution.requests == []
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["status"] == "blocked"
    assert event["blocking_reason"] == "autonomy_permission_evaluation_failed"
    assert event["autonomy_mode"] == "unavailable"


def test_opportunity_autonomy_enforcement_metadata_shape_is_stable() -> None:
    controller, _execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=_autonomy_shadow_repository_with_final_outcomes(
            [16.0, 14.0, 12.0, 10.0, 8.0, 6.0],
            environment="live",
            portfolio_id="live-1",
        ),
    )
    controller.process_signals(
        [_opportunity_autonomy_signal("live_assisted", assisted_approval=False)]
    )
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["autonomy_mode"] == "live_assisted"
    assert event["autonomous_execution_allowed"] == "false"
    assert event["autonomy_primary_reason"] == "reason:live_assisted"
    assert event["performance_guard_primary_reason"] == "performance_guard_no_breach"
    assert event["blocking_reason"] == "live_assisted_requires_explicit_approval"
    assert event["environment"] == "live"
    assert event["assisted_override_used"] == "false"


def test_opportunity_autonomy_runtime_local_snapshot_good_outcomes_allow_execution() -> None:
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=_autonomy_shadow_repository_with_final_outcomes(
            [9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
            environment="live",
            portfolio_id="live-1",
        ),
    )

    result = controller.process_signals([_opportunity_autonomy_signal("live_autonomous")])

    assert len(result) == 1
    assert len(execution.requests) == 1
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["status"] == "allowed"
    assert event["performance_guard_source"] == "local_snapshot_source_of_truth"
    assert event["performance_guard_effective_mode"] == "live_autonomous"
    assert event["performance_guard_primary_reason"] == "performance_guard_no_breach"


def test_opportunity_autonomy_runtime_local_snapshot_soft_breach_downgrades_and_blocks_live() -> (
    None
):
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=_autonomy_shadow_repository_with_final_outcomes(
            [1.0, -2.0], environment="live", portfolio_id="live-1"
        ),
    )

    result = controller.process_signals(
        [_opportunity_autonomy_signal("live_autonomous", assisted_approval=False)]
    )

    assert result == []
    assert execution.requests == []
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["status"] == "blocked"
    assert event["autonomy_mode"] == "live_assisted"
    assert event["autonomy_primary_reason"] == "insufficient_recent_final_outcomes_for_live"
    assert event["performance_guard_source"] == "local_snapshot_source_of_truth"
    assert (
        event["performance_guard_primary_reason"] == "insufficient_recent_final_outcomes_for_live"
    )
    assert event["blocking_reason"] == "live_assisted_requires_explicit_approval"


def test_opportunity_autonomy_runtime_local_snapshot_hard_breach_blocks_without_payload() -> None:
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=_autonomy_shadow_repository_with_final_outcomes(
            [-30.0, -20.0, -15.0, -11.0, -10.0],
            environment="live",
            portfolio_id="live-1",
        ),
    )

    result = controller.process_signals([_opportunity_autonomy_signal("live_autonomous")])

    assert result == []
    assert execution.requests == []
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["status"] == "blocked"
    assert event["autonomy_primary_reason"] == "hard_performance_breach_detected"
    assert event["performance_guard_hard_breach"] == "true"
    assert event["performance_guard_blocked"] == "true"
    assert event["blocking_reason"] == "performance_guard_local_kill_switch"


def test_opportunity_autonomy_runtime_missing_repository_fails_closed_with_audit_reason() -> None:
    controller, execution, journal = _build_autonomy_controller(environment="live")

    result = controller.process_signals([_opportunity_autonomy_signal("live_autonomous")])

    assert result == []
    assert execution.requests == []
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["status"] == "blocked"
    assert event["blocking_reason"] == "performance_guard_snapshot_source_unavailable"
    assert event["performance_guard_source"] == "missing_repository_fail_closed"


def test_opportunity_autonomy_runtime_no_final_outcomes_uses_conservative_behavior() -> None:
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=_autonomy_shadow_repository_with_final_outcomes(
            [], environment="live", portfolio_id="live-1"
        ),
    )

    result = controller.process_signals(
        [_opportunity_autonomy_signal("live_autonomous", assisted_approval=False)]
    )

    assert result == []
    assert execution.requests == []
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["status"] == "blocked"
    assert event["autonomy_mode"] == "live_assisted"
    assert (
        event["performance_guard_primary_reason"] == "insufficient_recent_final_outcomes_for_live"
    )


def test_opportunity_autonomy_runtime_local_guard_has_priority_over_payload_guard() -> None:
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=_autonomy_shadow_repository_with_final_outcomes(
            [-30.0, -20.0, -15.0, -11.0, -10.0],
            environment="live",
            portfolio_id="live-1",
        ),
    )

    result = controller.process_signals(
        [
            _opportunity_autonomy_signal(
                "live_autonomous",
                performance_guard_effective_mode="live_autonomous",
                performance_guard_primary_reason="payload_allows_live",
                performance_guard_hard_breach=False,
                performance_guard_blocked=False,
            )
        ]
    )

    assert result == []
    assert execution.requests == []
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["status"] == "blocked"
    assert event["performance_guard_source"] == "local_snapshot_source_of_truth"
    assert event["performance_guard_primary_reason"] == "hard_performance_breach_detected"
    assert event["performance_guard_blocked"] == "true"


def test_opportunity_autonomy_runtime_no_breach_ignores_payload_performance_guard_downgrade() -> (
    None
):
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=_autonomy_shadow_repository_with_final_outcomes(
            [9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
            environment="live",
            portfolio_id="live-1",
        ),
    )

    result = controller.process_signals(
        [
            _opportunity_autonomy_signal(
                "live_autonomous",
                include_decision_payload=True,
                performance_guard_effective_mode="paper_autonomous",
                performance_guard_primary_reason="payload_downgrade_should_be_ignored_when_local_guard_works",
                performance_guard_hard_breach=False,
                performance_guard_blocked=False,
            )
        ]
    )

    assert len(result) == 1
    assert len(execution.requests) == 1
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["status"] == "allowed"
    assert event["autonomy_mode"] == "live_autonomous"
    assert event["performance_guard_source"] == "local_snapshot_source_of_truth"
    assert event["performance_guard_primary_reason"] == "performance_guard_no_breach"
    assert event["performance_guard_effective_mode"] == "live_autonomous"


def test_opportunity_autonomy_runtime_missing_repository_allows_payload_guard_only_as_fallback_metadata() -> (
    None
):
    controller, execution, journal = _build_autonomy_controller(environment="paper")

    result = controller.process_signals(
        [
            _opportunity_autonomy_signal(
                "live_autonomous",
                include_decision_payload=True,
                performance_guard_effective_mode="shadow_only",
                performance_guard_primary_reason="payload_block_fallback",
                performance_guard_hard_breach=True,
                performance_guard_blocked=True,
            )
        ]
    )

    assert result == []
    assert execution.requests == []
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["status"] == "blocked"
    assert event["blocking_reason"] == "performance_guard_snapshot_source_unavailable"
    assert event["performance_guard_source"] == "missing_repository_fail_closed"
    assert event["performance_guard_block_enforced"] == "true"
    assert event["fallback_autonomy_mode"] == "shadow_only"
    assert event["fallback_autonomy_primary_reason"] == "payload_block_fallback"


def test_opportunity_autonomy_runtime_scoped_snapshot_ignores_other_environment_labels() -> None:
    repository = _autonomy_shadow_repository_with_mixed_scope_outcomes(
        [
            (8.0, "live", "live-1"),
            (7.0, "live", "live-1"),
            (6.0, "live", "live-1"),
            (5.0, "live", "live-1"),
            (4.0, "live", "live-1"),
            (3.0, "live", "live-1"),
            (-40.0, "paper", "paper-1"),
            (-30.0, "paper", "paper-1"),
            (-20.0, "paper", "paper-1"),
        ]
    )
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=repository,
    )

    result = controller.process_signals([_opportunity_autonomy_signal("live_autonomous")])

    assert len(result) == 1
    assert len(execution.requests) == 1
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["performance_guard_source"] == "local_snapshot_source_of_truth"
    assert event["performance_guard_scope_environment"] == "live"
    assert event["performance_guard_scope_portfolio"] == "live-1"
    assert event["performance_guard_scoped_label_count"] == "6"
    assert event["performance_guard_excluded_label_count"] == "3"


def test_opportunity_autonomy_runtime_scoped_snapshot_ignores_other_portfolio_labels() -> None:
    repository = _autonomy_shadow_repository_with_mixed_scope_outcomes(
        [
            (8.0, "live", "live-1"),
            (7.0, "live", "live-1"),
            (6.0, "live", "live-1"),
            (5.0, "live", "live-1"),
            (4.0, "live", "live-1"),
            (3.0, "live", "live-1"),
            (-35.0, "live", "live-2"),
            (-25.0, "live", "live-2"),
            (-15.0, "live", "live-2"),
        ]
    )
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=repository,
    )

    result = controller.process_signals([_opportunity_autonomy_signal("live_autonomous")])

    assert len(result) == 1
    assert len(execution.requests) == 1
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["performance_guard_scope_environment"] == "live"
    assert event["performance_guard_scope_portfolio"] == "live-1"
    assert event["performance_guard_scoped_label_count"] == "6"
    assert event["performance_guard_excluded_label_count"] == "3"
    assert event["performance_guard_effective_mode"] == "live_autonomous"


def test_opportunity_autonomy_runtime_scoped_snapshot_insufficient_local_evidence_is_conservative() -> (
    None
):
    repository = _autonomy_shadow_repository_with_mixed_scope_outcomes(
        [
            (5.0, "live", "live-1"),
            (-20.0, "paper", "paper-1"),
            (-18.0, "paper", "paper-1"),
            (-16.0, "paper", "paper-1"),
        ]
    )
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=repository,
    )

    result = controller.process_signals([_opportunity_autonomy_signal("live_autonomous")])

    assert result == []
    assert execution.requests == []
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["autonomy_mode"] == "live_assisted"
    assert event["autonomy_primary_reason"] == "insufficient_recent_final_outcomes_for_live"
    assert event["performance_guard_scoped_label_count"] == "1"
    assert event["performance_guard_excluded_label_count"] == "3"


def test_opportunity_autonomy_runtime_scoped_snapshot_excludes_partial_scope_and_reports_missing_scope_provenance() -> None:
    repo_dir = Path(tempfile.mkdtemp(prefix="autonomy-shadow-partial-scope-"))
    repository = OpportunityShadowRepository(repo_dir)
    repository.append_outcome_labels(
        [
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                correlation_key="scope-full-1",
                horizon_minutes=15,
                realized_return_bps=5.0,
                max_favorable_excursion_bps=5.0,
                max_adverse_excursion_bps=0.0,
                provenance={"environment": "live", "portfolio_id": "live-1"},
                label_quality="final",
            ),
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
                correlation_key="scope-full-2",
                horizon_minutes=15,
                realized_return_bps=4.0,
                max_favorable_excursion_bps=4.0,
                max_adverse_excursion_bps=0.0,
                provenance={"environment": "live", "portfolio_id": "live-1"},
                label_quality="final",
            ),
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc),
                correlation_key="scope-env-only",
                horizon_minutes=15,
                realized_return_bps=3.0,
                max_favorable_excursion_bps=3.0,
                max_adverse_excursion_bps=0.0,
                provenance={"environment": "live"},
                label_quality="final",
            ),
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, 0, 3, tzinfo=timezone.utc),
                correlation_key="scope-portfolio-only",
                horizon_minutes=15,
                realized_return_bps=2.0,
                max_favorable_excursion_bps=2.0,
                max_adverse_excursion_bps=0.0,
                provenance={"portfolio_id": "live-1"},
                label_quality="final",
            ),
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, 0, 4, tzinfo=timezone.utc),
                correlation_key="scope-wrong",
                horizon_minutes=15,
                realized_return_bps=-20.0,
                max_favorable_excursion_bps=0.0,
                max_adverse_excursion_bps=-20.0,
                provenance={"environment": "paper", "portfolio_id": "paper-1"},
                label_quality="final",
            ),
        ]
    )
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=repository,
    )

    result = controller.process_signals([_opportunity_autonomy_signal("live_autonomous")])

    assert result == []
    assert execution.requests == []
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["autonomy_mode"] == "live_assisted"
    assert event["autonomy_primary_reason"] == "insufficient_recent_final_outcomes_for_live"
    assert event["performance_guard_scoped_label_count"] == "2"
    assert event["performance_guard_excluded_label_count"] == "3"
    assert event["performance_guard_missing_scope_provenance_count"] == "2"
    assert event["performance_guard_scope_environment"] == "live"
    assert event["performance_guard_scope_portfolio"] == "live-1"


def test_opportunity_autonomy_runtime_scoped_snapshot_ignores_other_model_version() -> None:
    repository = _autonomy_shadow_repository_with_mixed_lineage_outcomes(
        [
            (8.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (7.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (6.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (5.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (4.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (3.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (-35.0, "live", "live-1", "B", "opportunity_ai_shadow"),
            (-25.0, "live", "live-1", "B", "opportunity_ai_shadow"),
            (-15.0, "live", "live-1", "B", "opportunity_ai_shadow"),
        ]
    )
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=repository,
    )

    result = controller.process_signals(
        [
            _opportunity_autonomy_signal(
                "live_autonomous",
                model_version="A",
                decision_source="opportunity_ai_shadow",
            )
        ]
    )

    assert len(result) == 1
    assert len(execution.requests) == 1
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["performance_guard_scope_model_version"] == "A"
    assert event["performance_guard_scope_decision_source"] == "opportunity_ai_shadow"
    assert event["performance_guard_scoped_label_count"] == "6"
    assert event["performance_guard_excluded_label_count"] == "3"


def test_opportunity_autonomy_runtime_lineage_payload_precedes_stale_request_metadata() -> None:
    repository = _autonomy_shadow_repository_with_mixed_lineage_outcomes(
        [
            (8.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (7.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (6.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (5.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (4.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (3.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (-35.0, "live", "live-1", "B", "opportunity_ai_shadow"),
            (-25.0, "live", "live-1", "B", "opportunity_ai_shadow"),
            (-15.0, "live", "live-1", "B", "opportunity_ai_shadow"),
        ]
    )
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=repository,
        order_metadata_defaults={
            "opportunity_model_version": "B",
            "opportunity_decision_source": "opportunity_ai_shadow",
        },
    )

    result = controller.process_signals(
        [
            _opportunity_autonomy_signal(
                "live_autonomous",
                include_decision_payload=True,
                decision_payload_model_version="A",
                decision_payload_decision_source="opportunity_ai_shadow",
            )
        ]
    )

    assert len(result) == 1
    assert len(execution.requests) == 1
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["performance_guard_scope_model_version"] == "A"
    assert event["performance_guard_scope_decision_source"] == "opportunity_ai_shadow"
    assert event["performance_guard_effective_mode"] == "live_autonomous"


def test_opportunity_autonomy_runtime_lineage_payload_precedes_stale_signal_metadata() -> None:
    repository = _autonomy_shadow_repository_with_mixed_lineage_outcomes(
        [
            (8.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (7.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (6.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (5.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (4.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (3.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (-35.0, "live", "live-1", "B", "opportunity_ai_shadow"),
            (-25.0, "live", "live-1", "B", "opportunity_ai_shadow"),
            (-15.0, "live", "live-1", "B", "opportunity_ai_shadow"),
        ]
    )
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=repository,
    )

    result = controller.process_signals(
        [
            _opportunity_autonomy_signal(
                "live_autonomous",
                model_version="B",
                decision_source="opportunity_ai_shadow",
                include_decision_payload=True,
                decision_payload_model_version="A",
                decision_payload_decision_source="opportunity_ai_shadow",
            )
        ]
    )

    assert len(result) == 1
    assert len(execution.requests) == 1
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["performance_guard_scope_model_version"] == "A"
    assert event["performance_guard_scope_decision_source"] == "opportunity_ai_shadow"
    assert event["performance_guard_effective_mode"] == "live_autonomous"


def test_opportunity_autonomy_runtime_lineage_request_payload_precedes_signal_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _autonomy_shadow_repository_with_mixed_lineage_outcomes(
        [
            (8.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (7.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (6.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (5.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (4.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (3.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (-35.0, "live", "live-1", "B", "opportunity_ai_shadow"),
            (-25.0, "live", "live-1", "B", "opportunity_ai_shadow"),
            (-15.0, "live", "live-1", "B", "opportunity_ai_shadow"),
        ]
    )
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=repository,
    )
    base_build_order_request = TradingController._build_order_request

    def _build_order_request_with_request_payload(
        self: TradingController,
        signal: StrategySignal,
        *,
        extra_metadata: Mapping[str, object] | None = None,
    ) -> OrderRequest:
        request = base_build_order_request(self, signal, extra_metadata=extra_metadata)
        request_metadata = dict(request.metadata or {})
        request_metadata["opportunity_autonomy_decision"] = {
            "effective_mode": "live_autonomous",
            "primary_reason": "request_payload_lineage_A",
            "model_version": "A",
            "decision_source": "opportunity_ai_shadow",
        }
        return replace(request, metadata=request_metadata)

    monkeypatch.setattr(
        TradingController,
        "_build_order_request",
        _build_order_request_with_request_payload,
    )

    result = controller.process_signals(
        [
            _opportunity_autonomy_signal(
                "live_autonomous",
                include_decision_payload=True,
                decision_payload_model_version="B",
                decision_payload_decision_source="opportunity_ai_shadow",
            )
        ]
    )

    assert len(result) == 1
    assert len(execution.requests) == 1
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["performance_guard_scope_model_version"] == "A"
    assert event["performance_guard_scope_decision_source"] == "opportunity_ai_shadow"
    assert event["performance_guard_effective_mode"] == "live_autonomous"


def test_opportunity_autonomy_runtime_lineage_uses_signal_payload_when_request_payload_has_no_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _autonomy_shadow_repository_with_mixed_lineage_outcomes(
        [
            (8.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (7.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (6.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (5.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (4.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (3.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (-35.0, "live", "live-1", "B", "opportunity_ai_shadow"),
            (-25.0, "live", "live-1", "B", "opportunity_ai_shadow"),
            (-15.0, "live", "live-1", "B", "opportunity_ai_shadow"),
        ]
    )
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=repository,
    )
    base_build_order_request = TradingController._build_order_request

    def _build_order_request_with_request_payload_without_lineage(
        self: TradingController,
        signal: StrategySignal,
        *,
        extra_metadata: Mapping[str, object] | None = None,
    ) -> OrderRequest:
        request = base_build_order_request(self, signal, extra_metadata=extra_metadata)
        request_metadata = dict(request.metadata or {})
        request_metadata["opportunity_autonomy_decision"] = {
            "effective_mode": "live_autonomous",
            "primary_reason": "request_payload_without_lineage",
        }
        return replace(request, metadata=request_metadata)

    monkeypatch.setattr(
        TradingController,
        "_build_order_request",
        _build_order_request_with_request_payload_without_lineage,
    )

    result = controller.process_signals(
        [
            _opportunity_autonomy_signal(
                "live_autonomous",
                include_decision_payload=True,
                decision_payload_model_version="A",
                decision_payload_decision_source="opportunity_ai_shadow",
            )
        ]
    )

    assert len(result) == 1
    assert len(execution.requests) == 1
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["performance_guard_scope_model_version"] == "A"
    assert event["performance_guard_scope_decision_source"] == "opportunity_ai_shadow"
    assert event["performance_guard_effective_mode"] == "live_autonomous"


def test_opportunity_autonomy_runtime_lineage_fallback_missing_repository_prefers_request_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
    )
    base_build_order_request = TradingController._build_order_request

    def _build_order_request_with_request_payload(
        self: TradingController,
        signal: StrategySignal,
        *,
        extra_metadata: Mapping[str, object] | None = None,
    ) -> OrderRequest:
        request = base_build_order_request(self, signal, extra_metadata=extra_metadata)
        request_metadata = dict(request.metadata or {})
        request_metadata["opportunity_autonomy_decision"] = {
            "effective_mode": "live_autonomous",
            "primary_reason": "request_payload_lineage_A",
            "model_version": "A",
            "decision_source": "opportunity_ai_shadow",
        }
        return replace(request, metadata=request_metadata)

    monkeypatch.setattr(
        TradingController,
        "_build_order_request",
        _build_order_request_with_request_payload,
    )

    result = controller.process_signals(
        [
            _opportunity_autonomy_signal(
                "live_autonomous",
                include_decision_payload=True,
                decision_payload_model_version="B",
                decision_payload_decision_source="opportunity_ai_shadow",
            )
        ]
    )

    assert len(result) == 0
    assert len(execution.requests) == 0
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["performance_guard_scope_model_version"] == "A"
    assert event["performance_guard_scope_decision_source"] == "opportunity_ai_shadow"
    assert event["performance_guard_source"] == "missing_repository_fail_closed"
    assert event["autonomy_primary_reason"] == "performance_guard_snapshot_source_unavailable"
    assert event["fallback_autonomy_mode"] == "live_autonomous"


def test_opportunity_autonomy_malformed_request_payload_does_not_mask_signal_payload_effective_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _autonomy_shadow_repository_with_final_outcomes(
        [8.0, 7.0, 6.0, 5.0, 4.0, 3.0],
        environment="live",
        portfolio_id="live-1",
    )
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=repository,
    )
    base_build_order_request = TradingController._build_order_request

    def _build_order_request_with_malformed_request_payload(
        self: TradingController,
        signal: StrategySignal,
        *,
        extra_metadata: Mapping[str, object] | None = None,
    ) -> OrderRequest:
        request = base_build_order_request(self, signal, extra_metadata=extra_metadata)
        request_metadata = dict(request.metadata or {})
        request_metadata["opportunity_autonomy_decision"] = "broken"
        return replace(request, metadata=request_metadata)

    monkeypatch.setattr(
        TradingController,
        "_build_order_request",
        _build_order_request_with_malformed_request_payload,
    )

    result = controller.process_signals(
        [
            _opportunity_autonomy_signal(
                "live_autonomous",
                include_decision_payload=True,
                decision_effective_mode="paper_autonomous",
            )
        ]
    )

    assert len(result) == 0
    assert len(execution.requests) == 0
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["autonomy_mode"] == "paper_autonomous"
    assert event["autonomous_execution_allowed"] == "false"


def test_opportunity_autonomy_empty_request_mapping_does_not_fall_through_to_signal_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _autonomy_shadow_repository_with_final_outcomes(
        [8.0, 7.0, 6.0, 5.0, 4.0, 3.0],
        environment="live",
        portfolio_id="live-1",
    )
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=repository,
    )
    base_build_order_request = TradingController._build_order_request

    def _build_order_request_with_empty_request_payload(
        self: TradingController,
        signal: StrategySignal,
        *,
        extra_metadata: Mapping[str, object] | None = None,
    ) -> OrderRequest:
        request = base_build_order_request(self, signal, extra_metadata=extra_metadata)
        request_metadata = dict(request.metadata or {})
        request_metadata["opportunity_autonomy_decision"] = {}
        return replace(request, metadata=request_metadata)

    monkeypatch.setattr(
        TradingController,
        "_build_order_request",
        _build_order_request_with_empty_request_payload,
    )

    result = controller.process_signals(
        [
            _opportunity_autonomy_signal(
                "live_autonomous",
                include_decision_payload=True,
                decision_effective_mode="paper_autonomous",
            )
        ]
    )

    assert len(result) == 1
    assert len(execution.requests) == 1
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["autonomy_mode"] == "live_autonomous"
    assert event["autonomous_execution_allowed"] == "true"


def test_opportunity_autonomy_empty_request_mapping_still_has_precedence_over_signal_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _autonomy_shadow_repository_with_final_outcomes(
        [8.0, 7.0, 6.0, 5.0, 4.0, 3.0],
        environment="live",
        portfolio_id="live-1",
    )
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=repository,
    )
    base_build_order_request = TradingController._build_order_request

    def _build_order_request_with_empty_request_payload(
        self: TradingController,
        signal: StrategySignal,
        *,
        extra_metadata: Mapping[str, object] | None = None,
    ) -> OrderRequest:
        request = base_build_order_request(self, signal, extra_metadata=extra_metadata)
        request_metadata = dict(request.metadata or {})
        request_metadata["opportunity_autonomy_decision"] = {}
        return replace(request, metadata=request_metadata)

    monkeypatch.setattr(
        TradingController,
        "_build_order_request",
        _build_order_request_with_empty_request_payload,
    )

    result = controller.process_signals(
        [
            _opportunity_autonomy_signal(
                "live_autonomous",
                include_decision_payload=True,
                decision_effective_mode="paper_autonomous",
                decision_primary_reason="signal_payload_reason_that_should_not_win",
            )
        ]
    )

    assert len(result) == 1
    assert len(execution.requests) == 1
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["autonomy_mode"] == "live_autonomous"
    assert event["autonomy_primary_reason"] == "reason:live_autonomous"


def test_opportunity_autonomy_malformed_request_payload_does_not_mask_signal_payload_primary_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _autonomy_shadow_repository_with_final_outcomes(
        [8.0, 7.0, 6.0, 5.0, 4.0, 3.0],
        environment="live",
        portfolio_id="live-1",
    )
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=repository,
    )
    base_build_order_request = TradingController._build_order_request

    def _build_order_request_with_malformed_request_payload(
        self: TradingController,
        signal: StrategySignal,
        *,
        extra_metadata: Mapping[str, object] | None = None,
    ) -> OrderRequest:
        request = base_build_order_request(self, signal, extra_metadata=extra_metadata)
        request_metadata = dict(request.metadata or {})
        request_metadata["opportunity_autonomy_decision"] = ["broken"]
        return replace(request, metadata=request_metadata)

    monkeypatch.setattr(
        TradingController,
        "_build_order_request",
        _build_order_request_with_malformed_request_payload,
    )

    result = controller.process_signals(
        [
            _opportunity_autonomy_signal(
                "live_autonomous",
                include_decision_payload=True,
                decision_effective_mode="live_autonomous",
                decision_primary_reason="signal_payload_primary_reason",
            )
        ]
    )

    assert len(result) == 1
    assert len(execution.requests) == 1
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["autonomy_primary_reason"] == "signal_payload_primary_reason"


def test_opportunity_autonomy_malformed_request_payload_does_not_mask_signal_payload_performance_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, execution, journal = _build_autonomy_controller(environment="live")
    base_build_order_request = TradingController._build_order_request

    def _build_order_request_with_malformed_request_payload(
        self: TradingController,
        signal: StrategySignal,
        *,
        extra_metadata: Mapping[str, object] | None = None,
    ) -> OrderRequest:
        request = base_build_order_request(self, signal, extra_metadata=extra_metadata)
        request_metadata = dict(request.metadata or {})
        request_metadata["opportunity_autonomy_decision"] = 123
        return replace(request, metadata=request_metadata)

    monkeypatch.setattr(
        TradingController,
        "_build_order_request",
        _build_order_request_with_malformed_request_payload,
    )

    result = controller.process_signals(
        [
            _opportunity_autonomy_signal(
                "live_autonomous",
                include_decision_payload=True,
                decision_effective_mode="live_autonomous",
                performance_guard_effective_mode="assisted",
                performance_guard_primary_reason="signal_payload_performance_guard_reason",
                performance_guard_blocked=True,
            )
        ]
    )

    assert len(result) == 0
    assert len(execution.requests) == 0
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["performance_guard_source"] == "missing_repository_fail_closed"
    assert event["performance_guard_effective_mode"] == "assisted"
    assert event["performance_guard_primary_reason"] == "signal_payload_performance_guard_reason"
    assert event["performance_guard_blocked"] == "true"


def test_opportunity_autonomy_valid_request_payload_still_precedes_valid_signal_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _autonomy_shadow_repository_with_final_outcomes(
        [8.0, 7.0, 6.0, 5.0, 4.0, 3.0],
        environment="live",
        portfolio_id="live-1",
    )
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=repository,
    )
    base_build_order_request = TradingController._build_order_request

    def _build_order_request_with_valid_request_payload(
        self: TradingController,
        signal: StrategySignal,
        *,
        extra_metadata: Mapping[str, object] | None = None,
    ) -> OrderRequest:
        request = base_build_order_request(self, signal, extra_metadata=extra_metadata)
        request_metadata = dict(request.metadata or {})
        request_metadata["opportunity_autonomy_decision"] = {
            "effective_mode": "live_autonomous",
            "primary_reason": "request_payload_reason",
        }
        return replace(request, metadata=request_metadata)

    monkeypatch.setattr(
        TradingController,
        "_build_order_request",
        _build_order_request_with_valid_request_payload,
    )

    result = controller.process_signals(
        [
            _opportunity_autonomy_signal(
                "live_autonomous",
                include_decision_payload=True,
                decision_effective_mode="paper_autonomous",
                decision_primary_reason="signal_payload_reason",
            )
        ]
    )

    assert len(result) == 1
    assert len(execution.requests) == 1
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["performance_guard_effective_mode"] == "live_autonomous"
    assert event["autonomy_primary_reason"] == "request_payload_reason"


def test_opportunity_autonomy_runtime_lineage_falls_back_to_signal_metadata_when_payload_lacks_lineage() -> (
    None
):
    repository = _autonomy_shadow_repository_with_mixed_lineage_outcomes(
        [
            (8.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (7.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (6.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (5.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (4.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (3.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (-35.0, "live", "live-1", "B", "opportunity_ai_shadow"),
            (-25.0, "live", "live-1", "B", "opportunity_ai_shadow"),
            (-15.0, "live", "live-1", "B", "opportunity_ai_shadow"),
        ]
    )
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=repository,
    )

    result = controller.process_signals(
        [
            _opportunity_autonomy_signal(
                "live_autonomous",
                model_version="A",
                decision_source="opportunity_ai_shadow",
                include_decision_payload=True,
            )
        ]
    )

    assert len(result) == 1
    assert len(execution.requests) == 1
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["performance_guard_scope_model_version"] == "A"
    assert event["performance_guard_scope_decision_source"] == "opportunity_ai_shadow"
    assert event["performance_guard_effective_mode"] == "live_autonomous"


def test_opportunity_autonomy_runtime_scoped_snapshot_ignores_other_decision_source() -> None:
    repository = _autonomy_shadow_repository_with_mixed_lineage_outcomes(
        [
            (8.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (7.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (6.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (5.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (4.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (3.0, "live", "live-1", "A", "opportunity_ai_shadow"),
            (-35.0, "live", "live-1", "A", "other_source"),
            (-25.0, "live", "live-1", "A", "other_source"),
            (-15.0, "live", "live-1", "A", "other_source"),
        ]
    )
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=repository,
    )

    result = controller.process_signals(
        [
            _opportunity_autonomy_signal(
                "live_autonomous",
                model_version="A",
                decision_source="opportunity_ai_shadow",
            )
        ]
    )

    assert len(result) == 1
    assert len(execution.requests) == 1
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["performance_guard_scope_model_version"] == "A"
    assert event["performance_guard_scope_decision_source"] == "opportunity_ai_shadow"
    assert event["performance_guard_scoped_label_count"] == "6"
    assert event["performance_guard_excluded_label_count"] == "3"


def test_opportunity_autonomy_runtime_scoped_snapshot_missing_lineage_provenance_is_conservative() -> (
    None
):
    repository = _autonomy_shadow_repository_with_mixed_scope_outcomes(
        [
            (5.0, "live", "live-1"),
            (4.0, "live", "live-1"),
            (3.0, "live", "live-1"),
            (2.0, "live", "live-1"),
        ]
    )
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=repository,
    )

    result = controller.process_signals(
        [
            _opportunity_autonomy_signal(
                "live_autonomous",
                model_version="A",
                decision_source="opportunity_ai_shadow",
            )
        ]
    )

    assert result == []
    assert execution.requests == []
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["autonomy_mode"] == "live_assisted"
    assert event["performance_guard_scoped_label_count"] == "0"
    assert event["performance_guard_excluded_label_count"] == "4"
    assert event["performance_guard_missing_lineage_provenance_count"] == "4"


def test_opportunity_autonomy_runtime_scoped_snapshot_excludes_partial_lineage_and_reports_missing_lineage_provenance() -> (
    None
):
    repo_dir = Path(tempfile.mkdtemp(prefix="autonomy-shadow-partial-lineage-"))
    repository = OpportunityShadowRepository(repo_dir)
    repository.append_outcome_labels(
        [
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                correlation_key="lineage-full-1",
                horizon_minutes=15,
                realized_return_bps=5.0,
                max_favorable_excursion_bps=5.0,
                max_adverse_excursion_bps=0.0,
                provenance={
                    "environment": "live",
                    "portfolio_id": "live-1",
                    "model_version": "A",
                    "decision_source": "opportunity_ai_shadow",
                },
                label_quality="final",
            ),
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
                correlation_key="lineage-full-2",
                horizon_minutes=15,
                realized_return_bps=4.0,
                max_favorable_excursion_bps=4.0,
                max_adverse_excursion_bps=0.0,
                provenance={
                    "environment": "live",
                    "portfolio_id": "live-1",
                    "model_version": "A",
                    "decision_source": "opportunity_ai_shadow",
                },
                label_quality="final",
            ),
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc),
                correlation_key="lineage-model-only",
                horizon_minutes=15,
                realized_return_bps=3.0,
                max_favorable_excursion_bps=3.0,
                max_adverse_excursion_bps=0.0,
                provenance={
                    "environment": "live",
                    "portfolio_id": "live-1",
                    "model_version": "A",
                },
                label_quality="final",
            ),
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, 0, 3, tzinfo=timezone.utc),
                correlation_key="lineage-source-only",
                horizon_minutes=15,
                realized_return_bps=2.0,
                max_favorable_excursion_bps=2.0,
                max_adverse_excursion_bps=0.0,
                provenance={
                    "environment": "live",
                    "portfolio_id": "live-1",
                    "decision_source": "opportunity_ai_shadow",
                },
                label_quality="final",
            ),
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, 0, 4, tzinfo=timezone.utc),
                correlation_key="lineage-wrong",
                horizon_minutes=15,
                realized_return_bps=-20.0,
                max_favorable_excursion_bps=0.0,
                max_adverse_excursion_bps=-20.0,
                provenance={
                    "environment": "live",
                    "portfolio_id": "live-1",
                    "model_version": "B",
                    "decision_source": "other_source",
                },
                label_quality="final",
            ),
        ]
    )
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=repository,
    )

    result = controller.process_signals(
        [
            _opportunity_autonomy_signal(
                "live_autonomous",
                model_version="A",
                decision_source="opportunity_ai_shadow",
            )
        ]
    )

    assert result == []
    assert execution.requests == []
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["autonomy_mode"] == "live_assisted"
    assert event["autonomy_primary_reason"] == "insufficient_recent_final_outcomes_for_live"
    assert event["performance_guard_scope_model_version"] == "A"
    assert event["performance_guard_scope_decision_source"] == "opportunity_ai_shadow"
    assert event["performance_guard_scoped_label_count"] == "2"
    assert event["performance_guard_excluded_label_count"] == "3"
    assert event["performance_guard_missing_lineage_provenance_count"] == "2"


def test_opportunity_autonomy_runtime_scoped_snapshot_excludes_combined_partial_scope_and_lineage_and_reports_both_missing_counters() -> (
    None
):
    repo_dir = Path(tempfile.mkdtemp(prefix="autonomy-shadow-combined-partial-"))
    repository = OpportunityShadowRepository(repo_dir)
    repository.append_outcome_labels(
        [
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                correlation_key="combined-full-1",
                horizon_minutes=15,
                realized_return_bps=5.0,
                max_favorable_excursion_bps=5.0,
                max_adverse_excursion_bps=0.0,
                provenance={
                    "environment": "live",
                    "portfolio_id": "live-1",
                    "model_version": "A",
                    "decision_source": "opportunity_ai_shadow",
                },
                label_quality="final",
            ),
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
                correlation_key="combined-full-2",
                horizon_minutes=15,
                realized_return_bps=4.0,
                max_favorable_excursion_bps=4.0,
                max_adverse_excursion_bps=0.0,
                provenance={
                    "environment": "live",
                    "portfolio_id": "live-1",
                    "model_version": "A",
                    "decision_source": "opportunity_ai_shadow",
                },
                label_quality="final",
            ),
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc),
                correlation_key="combined-partial-both",
                horizon_minutes=15,
                realized_return_bps=3.0,
                max_favorable_excursion_bps=3.0,
                max_adverse_excursion_bps=0.0,
                provenance={"environment": "live", "model_version": "A"},
                label_quality="final",
            ),
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, 0, 3, tzinfo=timezone.utc),
                correlation_key="combined-partial-scope-only",
                horizon_minutes=15,
                realized_return_bps=2.0,
                max_favorable_excursion_bps=2.0,
                max_adverse_excursion_bps=0.0,
                provenance={
                    "environment": "live",
                    "model_version": "A",
                    "decision_source": "opportunity_ai_shadow",
                },
                label_quality="final",
            ),
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, 0, 4, tzinfo=timezone.utc),
                correlation_key="combined-partial-lineage-only",
                horizon_minutes=15,
                realized_return_bps=1.0,
                max_favorable_excursion_bps=1.0,
                max_adverse_excursion_bps=0.0,
                provenance={
                    "environment": "live",
                    "portfolio_id": "live-1",
                    "model_version": "A",
                },
                label_quality="final",
            ),
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
                correlation_key="combined-wrong-full",
                horizon_minutes=15,
                realized_return_bps=-20.0,
                max_favorable_excursion_bps=0.0,
                max_adverse_excursion_bps=-20.0,
                provenance={
                    "environment": "paper",
                    "portfolio_id": "paper-1",
                    "model_version": "B",
                    "decision_source": "other_source",
                },
                label_quality="final",
            ),
        ]
    )
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=repository,
    )

    result = controller.process_signals(
        [
            _opportunity_autonomy_signal(
                "live_autonomous",
                model_version="A",
                decision_source="opportunity_ai_shadow",
            )
        ]
    )

    assert result == []
    assert execution.requests == []
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["autonomy_mode"] == "live_assisted"
    assert event["autonomy_primary_reason"] == "insufficient_recent_final_outcomes_for_live"
    assert event["performance_guard_scope_model_version"] == "A"
    assert event["performance_guard_scope_decision_source"] == "opportunity_ai_shadow"
    assert event["performance_guard_scoped_label_count"] == "2"
    assert event["performance_guard_excluded_label_count"] == "4"
    assert event["performance_guard_missing_scope_provenance_count"] == "2"
    assert event["performance_guard_missing_lineage_provenance_count"] == "2"


def test_opportunity_autonomy_runtime_scoped_snapshot_mixed_dataset_keeps_builder_diagnostics_parity_and_conservative_mode() -> (
    None
):
    repo_dir = Path(tempfile.mkdtemp(prefix="autonomy-shadow-mixed-guard-parity-"))
    repository = OpportunityShadowRepository(repo_dir)
    repository.append_outcome_labels(
        [
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                correlation_key="parity-full-1",
                horizon_minutes=15,
                realized_return_bps=5.0,
                max_favorable_excursion_bps=5.0,
                max_adverse_excursion_bps=0.0,
                provenance={
                    "environment": "live",
                    "portfolio_id": "live-1",
                    "model_version": "A",
                    "decision_source": "opportunity_ai_shadow",
                },
                label_quality="final",
            ),
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
                correlation_key="parity-full-2",
                horizon_minutes=15,
                realized_return_bps=4.0,
                max_favorable_excursion_bps=4.0,
                max_adverse_excursion_bps=0.0,
                provenance={
                    "environment": "live",
                    "portfolio_id": "live-1",
                    "model_version": "A",
                    "decision_source": "opportunity_ai_shadow",
                },
                label_quality="final",
            ),
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc),
                correlation_key="parity-combined-partial",
                horizon_minutes=15,
                realized_return_bps=3.0,
                max_favorable_excursion_bps=3.0,
                max_adverse_excursion_bps=0.0,
                provenance={"environment": "live", "model_version": "A"},
                label_quality="final",
            ),
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, 0, 3, tzinfo=timezone.utc),
                correlation_key="parity-scope-partial",
                horizon_minutes=15,
                realized_return_bps=2.0,
                max_favorable_excursion_bps=2.0,
                max_adverse_excursion_bps=0.0,
                provenance={
                    "environment": "live",
                    "model_version": "A",
                    "decision_source": "opportunity_ai_shadow",
                },
                label_quality="final",
            ),
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, 0, 4, tzinfo=timezone.utc),
                correlation_key="parity-lineage-partial",
                horizon_minutes=15,
                realized_return_bps=1.0,
                max_favorable_excursion_bps=1.0,
                max_adverse_excursion_bps=0.0,
                provenance={
                    "environment": "live",
                    "portfolio_id": "live-1",
                    "model_version": "A",
                },
                label_quality="final",
            ),
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
                correlation_key="parity-wrong-full",
                horizon_minutes=15,
                realized_return_bps=-20.0,
                max_favorable_excursion_bps=0.0,
                max_adverse_excursion_bps=-20.0,
                provenance={
                    "environment": "paper",
                    "portfolio_id": "paper-1",
                    "model_version": "B",
                    "decision_source": "other_source",
                },
                label_quality="final",
            ),
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, 0, 6, tzinfo=timezone.utc),
                correlation_key="parity-partial-only",
                horizon_minutes=15,
                realized_return_bps=0.5,
                max_favorable_excursion_bps=0.5,
                max_adverse_excursion_bps=0.0,
                provenance={
                    "environment": "live",
                    "portfolio_id": "live-1",
                    "model_version": "A",
                    "decision_source": "opportunity_ai_shadow",
                },
                label_quality="partial_exit",
            ),
        ]
    )
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=repository,
    )

    result = controller.process_signals(
        [
            _opportunity_autonomy_signal(
                "live_autonomous",
                model_version="A",
                decision_source="opportunity_ai_shadow",
            )
        ]
    )

    assert result == []
    assert execution.requests == []
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["autonomy_mode"] == "live_assisted"
    assert event["autonomy_primary_reason"] == "insufficient_recent_final_outcomes_for_live"
    assert event["performance_guard_scope_model_version"] == "A"
    assert event["performance_guard_scope_decision_source"] == "opportunity_ai_shadow"
    assert event["performance_guard_scoped_label_count"] == "3"
    assert event["performance_guard_excluded_label_count"] == "4"
    assert event["performance_guard_missing_scope_provenance_count"] == "2"
    assert event["performance_guard_missing_lineage_provenance_count"] == "2"
    assert event["performance_guard_effective_mode"] == "live_assisted"


def test_opportunity_autonomy_runtime_scoped_snapshot_parity_with_scan_and_window_limits() -> None:
    repo_dir = Path(tempfile.mkdtemp(prefix="autonomy-shadow-mixed-scan-window-"))
    repository = OpportunityShadowRepository(repo_dir)
    repository.append_outcome_labels(
        [
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                correlation_key="trimmed-old-full-1",
                horizon_minutes=15,
                realized_return_bps=10.0,
                max_favorable_excursion_bps=10.0,
                max_adverse_excursion_bps=0.0,
                provenance={
                    "environment": "live",
                    "portfolio_id": "live-1",
                    "model_version": "A",
                    "decision_source": "opportunity_ai_shadow",
                },
                label_quality="final",
            ),
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
                correlation_key="trimmed-old-full-2",
                horizon_minutes=15,
                realized_return_bps=9.0,
                max_favorable_excursion_bps=9.0,
                max_adverse_excursion_bps=0.0,
                provenance={
                    "environment": "live",
                    "portfolio_id": "live-1",
                    "model_version": "A",
                    "decision_source": "opportunity_ai_shadow",
                },
                label_quality="final",
            ),
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc),
                correlation_key="limited-combined-partial",
                horizon_minutes=15,
                realized_return_bps=3.0,
                max_favorable_excursion_bps=3.0,
                max_adverse_excursion_bps=0.0,
                provenance={"environment": "live", "model_version": "A"},
                label_quality="final",
            ),
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, 0, 3, tzinfo=timezone.utc),
                correlation_key="limited-scope-partial",
                horizon_minutes=15,
                realized_return_bps=2.0,
                max_favorable_excursion_bps=2.0,
                max_adverse_excursion_bps=0.0,
                provenance={
                    "environment": "live",
                    "model_version": "A",
                    "decision_source": "opportunity_ai_shadow",
                },
                label_quality="final",
            ),
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, 0, 4, tzinfo=timezone.utc),
                correlation_key="limited-lineage-partial",
                horizon_minutes=15,
                realized_return_bps=1.0,
                max_favorable_excursion_bps=1.0,
                max_adverse_excursion_bps=0.0,
                provenance={
                    "environment": "live",
                    "portfolio_id": "live-1",
                    "model_version": "A",
                },
                label_quality="final",
            ),
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
                correlation_key="limited-wrong-full",
                horizon_minutes=15,
                realized_return_bps=-20.0,
                max_favorable_excursion_bps=0.0,
                max_adverse_excursion_bps=-20.0,
                provenance={
                    "environment": "paper",
                    "portfolio_id": "paper-1",
                    "model_version": "B",
                    "decision_source": "other_source",
                },
                label_quality="final",
            ),
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, 0, 6, tzinfo=timezone.utc),
                correlation_key="limited-partial-only",
                horizon_minutes=15,
                realized_return_bps=0.5,
                max_favorable_excursion_bps=0.5,
                max_adverse_excursion_bps=0.0,
                provenance={
                    "environment": "live",
                    "portfolio_id": "live-1",
                    "model_version": "A",
                    "decision_source": "opportunity_ai_shadow",
                },
                label_quality="partial_exit",
            ),
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=datetime(2026, 1, 1, 0, 7, tzinfo=timezone.utc),
                correlation_key="limited-full-in-scope",
                horizon_minutes=15,
                realized_return_bps=6.0,
                max_favorable_excursion_bps=6.0,
                max_adverse_excursion_bps=0.0,
                provenance={
                    "environment": "live",
                    "portfolio_id": "live-1",
                    "model_version": "A",
                    "decision_source": "opportunity_ai_shadow",
                },
                label_quality="final",
            ),
        ]
    )

    lifecycle = OpportunityLifecycleService()
    _snapshot, expected_diagnostics = lifecycle.load_recent_performance_snapshot_with_scope_diagnostics(
        shadow_repository=repository,
        snapshot_config=OpportunityPerformanceSnapshotConfig(
            recent_final_window_size=2,
            max_scan_labels=6,
            scope_environment="live",
            scope_portfolio="live-1",
            scope_model_version="A",
            scope_decision_source="opportunity_ai_shadow",
            require_scope_provenance=True,
            require_lineage_provenance=True,
        ),
    )
    controller, execution, journal = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=repository,
        performance_guard_recent_final_window_size=2,
        performance_guard_max_scan_labels=6,
    )

    result = controller.process_signals(
        [
            _opportunity_autonomy_signal(
                "live_autonomous",
                model_version="A",
                decision_source="opportunity_ai_shadow",
            )
        ]
    )

    assert result == []
    assert execution.requests == []
    event = _last_event(journal, "opportunity_autonomy_enforcement")
    assert event["autonomy_mode"] == "live_assisted"
    assert event["autonomy_primary_reason"] == "insufficient_recent_final_outcomes_for_live"
    assert event["performance_guard_scoped_label_count"] == str(expected_diagnostics.scoped_label_count)
    assert event["performance_guard_excluded_label_count"] == str(
        expected_diagnostics.excluded_label_count
    )
    assert event["performance_guard_missing_scope_provenance_count"] == str(
        expected_diagnostics.missing_scope_provenance_count
    )
    assert event["performance_guard_missing_lineage_provenance_count"] == str(
        expected_diagnostics.missing_lineage_provenance_count
    )
    assert event["performance_guard_effective_mode"] == "live_assisted"


def test_opportunity_autonomy_runtime_performance_guard_defaults_remain_unchanged() -> None:
    repository = _autonomy_shadow_repository_with_final_outcomes(
        [8.0, 7.0, 6.0, 5.0, 4.0], environment="live", portfolio_id="live-1"
    )
    controller_default, execution_default, journal_default = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=repository,
    )
    controller_explicit, execution_explicit, journal_explicit = _build_autonomy_controller(
        environment="live",
        opportunity_shadow_repository=repository,
        performance_guard_recent_final_window_size=20,
        performance_guard_max_scan_labels=256,
    )

    result_default = controller_default.process_signals([_opportunity_autonomy_signal("live_autonomous")])
    result_explicit = controller_explicit.process_signals(
        [_opportunity_autonomy_signal("live_autonomous")]
    )

    assert result_default == result_explicit
    assert len(execution_default.requests) == len(execution_explicit.requests)
    event_default = _last_event(journal_default, "opportunity_autonomy_enforcement")
    event_explicit = _last_event(journal_explicit, "opportunity_autonomy_enforcement")
    assert event_default["autonomy_mode"] == event_explicit["autonomy_mode"]
    assert (
        event_default["performance_guard_effective_mode"]
        == event_explicit["performance_guard_effective_mode"]
    )
    assert (
        event_default["performance_guard_primary_reason"]
        == event_explicit["performance_guard_primary_reason"]
    )
    assert (
        event_default["performance_guard_snapshot_window"]
        == event_explicit["performance_guard_snapshot_window"]
    )
    assert (
        event_default["performance_guard_scoped_label_count"]
        == event_explicit["performance_guard_scoped_label_count"]
    )
    assert (
        event_default["performance_guard_excluded_label_count"]
        == event_explicit["performance_guard_excluded_label_count"]
    )
    assert (
        event_default["performance_guard_missing_scope_provenance_count"]
        == event_explicit["performance_guard_missing_scope_provenance_count"]
    )
    assert (
        event_default["performance_guard_missing_lineage_provenance_count"]
        == event_explicit["performance_guard_missing_lineage_provenance_count"]
    )


@pytest.mark.parametrize("invalid_value", [0, -1])
def test_trading_controller_rejects_non_positive_recent_final_window_size(
    invalid_value: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="performance_guard_recent_final_window_size musi być dodatnią liczbą całkowitą \\(> 0\\)",
    ):
        _build_autonomy_controller(
            environment="live",
            performance_guard_recent_final_window_size=invalid_value,
        )


@pytest.mark.parametrize("invalid_value", [0, -1])
def test_trading_controller_rejects_non_positive_max_scan_labels(
    invalid_value: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="performance_guard_max_scan_labels musi być dodatnią liczbą całkowitą \\(> 0\\)",
    ):
        _build_autonomy_controller(
            environment="live",
            performance_guard_max_scan_labels=invalid_value,
        )


@pytest.mark.parametrize("invalid_value", [True, 1.5, "2"])
def test_trading_controller_rejects_non_int_recent_final_window_size(
    invalid_value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="performance_guard_recent_final_window_size musi być dodatnią liczbą całkowitą \\(> 0\\)",
    ):
        _build_autonomy_controller(
            environment="live",
            performance_guard_recent_final_window_size=invalid_value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("invalid_value", [True, 1.5, "2"])
def test_trading_controller_rejects_non_int_max_scan_labels(
    invalid_value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="performance_guard_max_scan_labels musi być dodatnią liczbą całkowitą \\(> 0\\)",
    ):
        _build_autonomy_controller(
            environment="live",
            performance_guard_max_scan_labels=invalid_value,  # type: ignore[arg-type]
        )


def _shadow_record_for_key(
    *,
    correlation_key: str,
    decision_timestamp: datetime,
) -> OpportunityShadowRecord:
    return OpportunityShadowRecord(
        record_key=correlation_key,
        symbol="BTC/USDT",
        decision_timestamp=decision_timestamp,
        model_version="opportunity-v1",
        decision_source="opportunity_ai_shadow",
        expected_edge_bps=5.0,
        success_probability=0.7,
        confidence=0.3,
        proposed_direction="long",
        accepted=True,
        rejection_reason=None,
        rank=1,
        provenance={"probability_method": "test"},
        threshold_config=OpportunityThresholdConfig(),
        snapshot={},
        context=OpportunityShadowContext(environment="paper"),
    )


def _attach_proxy_label_with_lineage_inputs(
    *,
    tmp_path: Path,
    request_payload: object | None,
    signal_payload: object | None,
    request_metadata_lineage: Mapping[str, object] | None = None,
    signal_metadata_lineage: Mapping[str, object] | None = None,
    restored_tracker_lineage: Mapping[str, object] | None = None,
    request_side: str = "BUY",
) -> OpportunityOutcomeLabel:
    decision_timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    correlation_key = OpportunityShadowRecord.build_record_key(
        symbol="BTC/USDT",
        decision_timestamp=decision_timestamp,
        model_version="shadow-v0",
        rank=1,
    )
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    shadow_repo.append_shadow_records(
        [
            OpportunityShadowRecord(
                record_key=correlation_key,
                symbol="BTC/USDT",
                decision_timestamp=decision_timestamp,
                model_version="shadow-v0",
                decision_source="shadow-source",
                expected_edge_bps=5.0,
                success_probability=0.7,
                confidence=0.3,
                proposed_direction="long",
                accepted=True,
                rejection_reason=None,
                rank=1,
                provenance={"probability_method": "test"},
                threshold_config=OpportunityThresholdConfig(),
                snapshot={},
                context=OpportunityShadowContext(environment="paper"),
            )
        ]
    )
    if restored_tracker_lineage is not None:
        tracker_side = "BUY" if str(request_side).upper() == "SELL" else "SELL"
        tracker_provenance = {
            "source": "restored_tracker",
            "environment": "paper",
            "portfolio": "paper-1",
            **dict(restored_tracker_lineage),
        }
        shadow_repo.upsert_open_outcome(
            shadow_repo.OpenOutcomeState(
                correlation_key=correlation_key,
                symbol="BTC/USDT",
                side=tracker_side,
                entry_price=100.0,
                decision_timestamp=decision_timestamp,
                entry_quantity=1.0,
                closed_quantity=0.0,
                provenance=tracker_provenance,
            )
        )

    controller = TradingController(
        risk_engine=DummyRiskEngine(),
        execution_service=DummyExecutionService(),
        alert_router=_router_with_channel()[0],
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        decision_journal=CollectingDecisionJournal(),
        opportunity_shadow_repository=shadow_repo,
    )
    signal_metadata: dict[str, object] = {
        "opportunity_shadow_record_key": correlation_key,
        "opportunity_decision_timestamp": decision_timestamp.isoformat(),
    }
    if signal_payload is not None:
        signal_metadata["opportunity_autonomy_decision"] = signal_payload
    if signal_metadata_lineage:
        signal_metadata.update(dict(signal_metadata_lineage))
    signal = StrategySignal(
        symbol="BTC/USDT",
        side=str(request_side).upper(),
        confidence=0.8,
        metadata=signal_metadata,
    )
    request_metadata: dict[str, object] = {
        "opportunity_shadow_record_key": correlation_key,
        "opportunity_decision_timestamp": decision_timestamp.isoformat(),
    }
    if request_payload is not None:
        request_metadata["opportunity_autonomy_decision"] = request_payload
    if request_metadata_lineage:
        request_metadata.update(dict(request_metadata_lineage))
    request = OrderRequest(
        symbol="BTC/USDT",
        side=str(request_side).upper(),
        quantity=1.0,
        order_type="market",
        price=100.0,
        metadata=request_metadata,
    )
    result = OrderResult(
        order_id="order-1",
        status="filled",
        filled_quantity=1.0,
        avg_price=100.0,
        raw_response={"context": "unit_test"},
    )

    controller._try_attach_opportunity_outcome_label(  # pyright: ignore[reportPrivateUsage]
        signal=signal,
        request=request,
        result=result,
        normalized_status="filled",
        metadata={"filled_quantity": "1.00000000", "avg_price": "100.00000000"},
    )
    labels = shadow_repo.load_outcome_labels()
    assert len(labels) == 1
    return labels[0]


def test_controller_attaches_runtime_outcome_label_for_opportunity_shadow_record(
    tmp_path: Path,
) -> None:
    decision_timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    correlation_key = OpportunityShadowRecord.build_record_key(
        symbol="BTC/USDT",
        decision_timestamp=decision_timestamp,
        model_version="opportunity-v1",
        rank=1,
    )
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    shadow_repo.append_shadow_records(
        [
            _shadow_record_for_key(
                correlation_key=correlation_key, decision_timestamp=decision_timestamp
            )
        ]
    )
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, _channel, _audit = _router_with_channel()
    journal = CollectingDecisionJournal()
    controller = TradingController(
        risk_engine=risk_engine,
        execution_service=execution,
        alert_router=router,
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        decision_journal=journal,
        opportunity_shadow_repository=shadow_repo,
    )
    signal = _signal("BUY")
    signal.metadata = {
        **dict(signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
        "opportunity_decision_timestamp": decision_timestamp.isoformat(),
    }

    controller.process_signals([signal])

    labels = shadow_repo.load_outcome_labels()
    assert len(labels) == 1
    assert labels[0].correlation_key == correlation_key
    assert labels[0].label_quality == "execution_proxy_pending_exit"
    assert labels[0].provenance.get("environment") == "paper"
    assert labels[0].provenance.get("portfolio") == "paper-1"
    assert labels[0].provenance.get("model_version") == "opportunity-v1"
    assert labels[0].provenance.get("decision_source") == "opportunity_ai_shadow"
    attach_events = [
        event for event in journal.export() if event["event"] == "opportunity_outcome_attach"
    ]
    assert attach_events
    assert attach_events[-1]["status"] == "proxy_attached"


def test_controller_attach_lineage_request_payload_precedes_signal_payload(tmp_path: Path) -> None:
    label = _attach_proxy_label_with_lineage_inputs(
        tmp_path=tmp_path,
        request_payload={"model_version": "request-v1", "decision_source": "request-source"},
        signal_payload={"model_version": "signal-v1", "decision_source": "signal-source"},
    )

    assert label.provenance.get("model_version") == "request-v1"
    assert label.provenance.get("decision_source") == "request-source"


def test_controller_attach_lineage_request_payload_precedes_restored_tracker(
    tmp_path: Path,
) -> None:
    label = _attach_proxy_label_with_lineage_inputs(
        tmp_path=tmp_path,
        request_payload={"model_version": "request-v1", "decision_source": "request-source"},
        signal_payload={"model_version": "signal-v1", "decision_source": "signal-source"},
        restored_tracker_lineage={
            "model_version": "tracker-v1",
            "decision_source": "tracker-source",
        },
        request_side="SELL",
    )

    assert label.label_quality == "final"
    assert label.provenance.get("model_version") == "request-v1"
    assert label.provenance.get("decision_source") == "request-source"


def test_controller_attach_lineage_request_payload_without_lineage_uses_signal_payload(
    tmp_path: Path,
) -> None:
    label = _attach_proxy_label_with_lineage_inputs(
        tmp_path=tmp_path,
        request_payload={"mode": "live_autonomous"},
        signal_payload={"model_version": "signal-v1", "decision_source": "signal-source"},
    )

    assert label.provenance.get("model_version") == "signal-v1"
    assert label.provenance.get("decision_source") == "signal-source"


def test_controller_attach_lineage_signal_payload_precedes_restored_tracker_when_request_payload_missing_lineage(
    tmp_path: Path,
) -> None:
    label = _attach_proxy_label_with_lineage_inputs(
        tmp_path=tmp_path,
        request_payload={"mode": "live_autonomous"},
        signal_payload={"model_version": "signal-v1", "decision_source": "signal-source"},
        restored_tracker_lineage={
            "model_version": "tracker-v1",
            "decision_source": "tracker-source",
        },
        request_side="SELL",
    )

    assert label.label_quality == "final"
    assert label.provenance.get("model_version") == "signal-v1"
    assert label.provenance.get("decision_source") == "signal-source"


def test_controller_attach_lineage_malformed_request_payload_does_not_mask_signal_payload(
    tmp_path: Path,
) -> None:
    label = _attach_proxy_label_with_lineage_inputs(
        tmp_path=tmp_path,
        request_payload="not-a-mapping",
        signal_payload={"model_version": "signal-v1", "decision_source": "signal-source"},
    )

    assert label.provenance.get("model_version") == "signal-v1"
    assert label.provenance.get("decision_source") == "signal-source"


def test_controller_attach_lineage_empty_request_mapping_allows_signal_payload_lineage(
    tmp_path: Path,
) -> None:
    label = _attach_proxy_label_with_lineage_inputs(
        tmp_path=tmp_path,
        request_payload={},
        signal_payload={"model_version": "signal-v1", "decision_source": "signal-source"},
    )

    assert label.provenance.get("model_version") == "signal-v1"
    assert label.provenance.get("decision_source") == "signal-source"


def test_controller_attach_lineage_metadata_fallback_applies_after_payloads(tmp_path: Path) -> None:
    label = _attach_proxy_label_with_lineage_inputs(
        tmp_path=tmp_path,
        request_payload=None,
        signal_payload=None,
        request_metadata_lineage={
            "opportunity_model_version": "request-meta-v1",
            "opportunity_decision_source": "request-meta-source",
        },
        signal_metadata_lineage={
            "opportunity_model_version": "signal-meta-v1",
            "opportunity_decision_source": "signal-meta-source",
        },
    )

    assert label.provenance.get("model_version") == "request-meta-v1"
    assert label.provenance.get("decision_source") == "request-meta-source"


def test_controller_attach_lineage_restored_tracker_precedes_request_metadata(
    tmp_path: Path,
) -> None:
    label = _attach_proxy_label_with_lineage_inputs(
        tmp_path=tmp_path,
        request_payload=None,
        signal_payload=None,
        request_metadata_lineage={
            "opportunity_model_version": "request-meta-v1",
            "opportunity_decision_source": "request-meta-source",
        },
        restored_tracker_lineage={
            "model_version": "tracker-v1",
            "decision_source": "tracker-source",
        },
        request_side="SELL",
    )

    assert label.label_quality == "final"
    assert label.provenance.get("model_version") == "tracker-v1"
    assert label.provenance.get("decision_source") == "tracker-source"


def test_controller_attach_lineage_restored_tracker_precedes_signal_metadata(
    tmp_path: Path,
) -> None:
    label = _attach_proxy_label_with_lineage_inputs(
        tmp_path=tmp_path,
        request_payload=None,
        signal_payload=None,
        signal_metadata_lineage={
            "opportunity_model_version": "signal-meta-v1",
            "opportunity_decision_source": "signal-meta-source",
        },
        restored_tracker_lineage={
            "model_version": "tracker-v1",
            "decision_source": "tracker-source",
        },
        request_side="SELL",
    )

    assert label.label_quality == "final"
    assert label.provenance.get("model_version") == "tracker-v1"
    assert label.provenance.get("decision_source") == "tracker-source"


def test_controller_outcome_attach_duplicate_is_idempotent_noop(tmp_path: Path) -> None:
    decision_timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    correlation_key = OpportunityShadowRecord.build_record_key(
        symbol="BTC/USDT",
        decision_timestamp=decision_timestamp,
        model_version="opportunity-v1",
        rank=1,
    )
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    shadow_repo.append_shadow_records(
        [
            _shadow_record_for_key(
                correlation_key=correlation_key, decision_timestamp=decision_timestamp
            )
        ]
    )
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, _channel, _audit = _router_with_channel()
    journal = CollectingDecisionJournal()
    controller = TradingController(
        risk_engine=risk_engine,
        execution_service=execution,
        alert_router=router,
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        decision_journal=journal,
        opportunity_shadow_repository=shadow_repo,
    )
    signal = _signal("BUY")
    signal.metadata = {
        **dict(signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
        "opportunity_decision_timestamp": decision_timestamp.isoformat(),
    }

    controller.process_signals([signal])
    controller.process_signals([signal])

    labels = shadow_repo.load_outcome_labels()
    assert len(labels) == 1
    attach_events = [
        event for event in journal.export() if event["event"] == "opportunity_outcome_attach"
    ]
    assert attach_events[-1]["status"] == "duplicate_noop"


def test_controller_outcome_attach_rejects_missing_shadow_record(tmp_path: Path) -> None:
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, _channel, _audit = _router_with_channel()
    journal = CollectingDecisionJournal()
    controller = TradingController(
        risk_engine=risk_engine,
        execution_service=execution,
        alert_router=router,
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        decision_journal=journal,
        opportunity_shadow_repository=shadow_repo,
    )
    signal = _signal("BUY")
    signal.metadata = {**dict(signal.metadata), "opportunity_shadow_record_key": "missing-key"}

    controller.process_signals([signal])

    assert shadow_repo.load_outcome_labels() == []
    attach_events = [
        event for event in journal.export() if event["event"] == "opportunity_outcome_attach"
    ]
    assert attach_events
    assert attach_events[-1]["status"] == "missing_shadow_record"


def test_controller_outcome_attach_upgrades_proxy_to_final_on_exit(tmp_path: Path) -> None:
    decision_timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    correlation_key = OpportunityShadowRecord.build_record_key(
        symbol="BTC/USDT",
        decision_timestamp=decision_timestamp,
        model_version="opportunity-v1",
        rank=1,
    )
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    shadow_repo.append_shadow_records(
        [
            _shadow_record_for_key(
                correlation_key=correlation_key, decision_timestamp=decision_timestamp
            )
        ]
    )
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, _channel, _audit = _router_with_channel()
    journal = CollectingDecisionJournal()
    controller = TradingController(
        risk_engine=risk_engine,
        execution_service=execution,
        alert_router=router,
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        decision_journal=journal,
        opportunity_shadow_repository=shadow_repo,
    )
    open_signal = _signal("BUY", price=100.0)
    open_signal.metadata = {
        **dict(open_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
        "opportunity_decision_timestamp": decision_timestamp.isoformat(),
    }
    close_signal = _signal("SELL", price=110.0)

    controller.process_signals([open_signal])
    controller.process_signals([close_signal])

    labels = shadow_repo.load_outcome_labels()
    assert len(labels) == 1
    assert labels[0].label_quality == "final"
    assert labels[0].realized_return_bps == pytest.approx(1000.0, rel=1e-6)
    attach_events = [
        event for event in journal.export() if event["event"] == "opportunity_outcome_attach"
    ]
    assert any(event["status"] == "final_upgraded" for event in attach_events)


def test_controller_restores_open_outcome_tracker_after_restart(tmp_path: Path) -> None:
    decision_timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    correlation_key = OpportunityShadowRecord.build_record_key(
        symbol="BTC/USDT",
        decision_timestamp=decision_timestamp,
        model_version="opportunity-v1",
        rank=1,
    )
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    shadow_repo.append_shadow_records(
        [
            _shadow_record_for_key(
                correlation_key=correlation_key, decision_timestamp=decision_timestamp
            )
        ]
    )
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, _channel, _audit = _router_with_channel()
    controller_open = TradingController(
        risk_engine=risk_engine,
        execution_service=execution,
        alert_router=router,
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        decision_journal=CollectingDecisionJournal(),
        opportunity_shadow_repository=shadow_repo,
    )
    open_signal = _signal("BUY", price=100.0)
    open_signal.metadata = {
        **dict(open_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
        "opportunity_decision_timestamp": decision_timestamp.isoformat(),
    }
    controller_open.process_signals([open_signal])
    restored_rows = shadow_repo.load_open_outcomes()
    assert restored_rows
    assert restored_rows[0].provenance.get("environment") == "paper"
    assert restored_rows[0].provenance.get("portfolio") == "paper-1"
    assert restored_rows[0].provenance.get("model_version") == "opportunity-v1"
    assert restored_rows[0].provenance.get("decision_source") == "opportunity_ai_shadow"

    controller_close = TradingController(
        risk_engine=DummyRiskEngine(),
        execution_service=DummyExecutionService(),
        alert_router=router,
        account_snapshot_provider=_account_snapshot,
        portfolio_id="wrong-portfolio-after-restart",
        environment="live",
        risk_profile="balanced",
        decision_journal=CollectingDecisionJournal(),
        opportunity_shadow_repository=shadow_repo,
    )
    close_signal = _signal("SELL", price=110.0)
    close_signal.metadata = {
        **dict(close_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
        "opportunity_model_version": "wrong-after-restart",
        "opportunity_decision_source": "wrong-after-restart",
    }
    controller_close.process_signals([close_signal])
    labels = shadow_repo.load_outcome_labels()
    assert len(labels) == 1
    assert labels[0].label_quality == "final"
    assert labels[0].provenance.get("close_correlation_resolution") == "resolved_by_correlation_key"
    assert labels[0].provenance.get("environment") == "paper"
    assert labels[0].provenance.get("portfolio") == "paper-1"
    assert labels[0].provenance.get("model_version") == "opportunity-v1"
    assert labels[0].provenance.get("decision_source") == "opportunity_ai_shadow"
    assert shadow_repo.load_open_outcomes() == []


def test_controller_partial_close_after_restart_uses_restored_tracker_lineage(
    tmp_path: Path,
) -> None:
    decision_timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    correlation_key = OpportunityShadowRecord.build_record_key(
        symbol="BTC/USDT",
        decision_timestamp=decision_timestamp,
        model_version="opportunity-v1",
        rank=1,
    )
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    shadow_repo.append_shadow_records(
        [
            _shadow_record_for_key(
                correlation_key=correlation_key, decision_timestamp=decision_timestamp
            )
        ]
    )
    controller_open = TradingController(
        risk_engine=DummyRiskEngine(),
        execution_service=DummyExecutionService(),
        alert_router=_router_with_channel()[0],
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        decision_journal=CollectingDecisionJournal(),
        opportunity_shadow_repository=shadow_repo,
    )
    open_signal = _signal("BUY", price=100.0)
    open_signal.metadata = {
        **dict(open_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
        "opportunity_decision_timestamp": decision_timestamp.isoformat(),
    }
    controller_open.process_signals([open_signal])
    assert shadow_repo.load_open_outcomes()

    controller_close = TradingController(
        risk_engine=DummyRiskEngine(),
        execution_service=StatusExecutionService(
            status="partially_filled",
            filled_quantity=0.4,
            avg_price=110.0,
        ),
        alert_router=_router_with_channel()[0],
        account_snapshot_provider=_account_snapshot,
        portfolio_id="wrong-portfolio-after-restart",
        environment="live",
        risk_profile="balanced",
        decision_journal=CollectingDecisionJournal(),
        opportunity_shadow_repository=shadow_repo,
    )
    close_signal = _signal("SELL", price=110.0)
    close_signal.metadata = {
        **dict(close_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
    }
    controller_close.process_signals([close_signal])

    labels = shadow_repo.load_outcome_labels()
    assert len(labels) == 1
    assert labels[0].label_quality == "partial_exit_unconfirmed"
    assert labels[0].provenance.get("environment") == "paper"
    assert labels[0].provenance.get("portfolio") == "paper-1"
    assert labels[0].provenance.get("model_version") == "opportunity-v1"
    assert labels[0].provenance.get("decision_source") == "opportunity_ai_shadow"


def test_controller_partial_close_after_restart_with_legacy_scope_gap_is_auditable(
    tmp_path: Path,
) -> None:
    decision_timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    correlation_key = OpportunityShadowRecord.build_record_key(
        symbol="BTC/USDT",
        decision_timestamp=decision_timestamp,
        model_version="opportunity-v1",
        rank=1,
    )
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    shadow_repo.append_shadow_records(
        [
            _shadow_record_for_key(
                correlation_key=correlation_key, decision_timestamp=decision_timestamp
            )
        ]
    )
    controller_open = TradingController(
        risk_engine=DummyRiskEngine(),
        execution_service=DummyExecutionService(),
        alert_router=_router_with_channel()[0],
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        decision_journal=CollectingDecisionJournal(),
        opportunity_shadow_repository=shadow_repo,
    )
    open_signal = _signal("BUY", price=100.0)
    open_signal.metadata = {
        **dict(open_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
        "opportunity_decision_timestamp": decision_timestamp.isoformat(),
    }
    controller_open.process_signals([open_signal])
    open_rows = shadow_repo.load_open_outcomes()
    assert len(open_rows) == 1
    shadow_repo.upsert_open_outcome(
        shadow_repo.OpenOutcomeState(
            correlation_key=correlation_key,
            symbol=open_rows[0].symbol,
            side=open_rows[0].side,
            entry_price=open_rows[0].entry_price,
            decision_timestamp=open_rows[0].decision_timestamp,
            entry_quantity=open_rows[0].entry_quantity,
            closed_quantity=open_rows[0].closed_quantity,
            provenance={
                "source": "legacy_without_scope",
                "model_version": "legacy-v1",
                "decision_source": "legacy-source",
            },
        )
    )

    controller_close = TradingController(
        risk_engine=DummyRiskEngine(),
        execution_service=StatusExecutionService(
            status="partially_filled",
            filled_quantity=0.4,
            avg_price=110.0,
        ),
        alert_router=_router_with_channel()[0],
        account_snapshot_provider=_account_snapshot,
        portfolio_id="wrong-portfolio-after-restart",
        environment="live",
        risk_profile="balanced",
        decision_journal=CollectingDecisionJournal(),
        opportunity_shadow_repository=shadow_repo,
    )
    close_signal = _signal("SELL", price=110.0)
    close_signal.metadata = {
        **dict(close_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
    }
    controller_close.process_signals([close_signal])

    labels = shadow_repo.load_outcome_labels()
    assert len(labels) == 1
    assert labels[0].label_quality == "partial_exit_unconfirmed"
    assert "environment" not in labels[0].provenance
    assert "portfolio" not in labels[0].provenance
    assert labels[0].provenance.get("scope_continuity") == "restored_tracker_scope_missing"
    assert labels[0].provenance.get("model_version") == "legacy-v1"
    assert labels[0].provenance.get("decision_source") == "legacy-source"

    open_rows_after_partial = shadow_repo.load_open_outcomes()
    assert len(open_rows_after_partial) == 1
    assert "environment" not in open_rows_after_partial[0].provenance
    assert "portfolio" not in open_rows_after_partial[0].provenance
    assert (
        open_rows_after_partial[0].provenance.get("scope_continuity")
        == "missing_from_restored_open_outcome"
    )


def test_controller_final_close_after_restart_with_legacy_scope_gap_is_auditable(
    tmp_path: Path,
) -> None:
    decision_timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    correlation_key = OpportunityShadowRecord.build_record_key(
        symbol="BTC/USDT",
        decision_timestamp=decision_timestamp,
        model_version="opportunity-v1",
        rank=1,
    )
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    shadow_repo.append_shadow_records(
        [
            _shadow_record_for_key(
                correlation_key=correlation_key, decision_timestamp=decision_timestamp
            )
        ]
    )
    controller_open = TradingController(
        risk_engine=DummyRiskEngine(),
        execution_service=DummyExecutionService(),
        alert_router=_router_with_channel()[0],
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        decision_journal=CollectingDecisionJournal(),
        opportunity_shadow_repository=shadow_repo,
    )
    open_signal = _signal("BUY", price=100.0)
    open_signal.metadata = {
        **dict(open_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
        "opportunity_decision_timestamp": decision_timestamp.isoformat(),
    }
    controller_open.process_signals([open_signal])
    open_rows = shadow_repo.load_open_outcomes()
    assert len(open_rows) == 1
    shadow_repo.upsert_open_outcome(
        shadow_repo.OpenOutcomeState(
            correlation_key=correlation_key,
            symbol=open_rows[0].symbol,
            side=open_rows[0].side,
            entry_price=open_rows[0].entry_price,
            decision_timestamp=open_rows[0].decision_timestamp,
            entry_quantity=open_rows[0].entry_quantity,
            closed_quantity=open_rows[0].closed_quantity,
            provenance={
                "source": "legacy_without_scope",
                "model_version": "legacy-v1",
                "decision_source": "legacy-source",
            },
        )
    )

    controller_close = TradingController(
        risk_engine=DummyRiskEngine(),
        execution_service=DummyExecutionService(),
        alert_router=_router_with_channel()[0],
        account_snapshot_provider=_account_snapshot,
        portfolio_id="wrong-portfolio-after-restart",
        environment="live",
        risk_profile="balanced",
        decision_journal=CollectingDecisionJournal(),
        opportunity_shadow_repository=shadow_repo,
    )
    close_signal = _signal("SELL", price=110.0)
    close_signal.metadata = {
        **dict(close_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
    }
    controller_close.process_signals([close_signal])

    labels = shadow_repo.load_outcome_labels()
    assert len(labels) == 1
    assert labels[0].label_quality == "final"
    assert "environment" not in labels[0].provenance
    assert "portfolio" not in labels[0].provenance
    assert labels[0].provenance.get("scope_continuity") == "restored_tracker_scope_missing"
    assert labels[0].provenance.get("model_version") == "legacy-v1"
    assert labels[0].provenance.get("decision_source") == "legacy-source"
    assert shadow_repo.load_open_outcomes() == []


def _legacy_partial_scope_provenance(
    *, scope_variant: str, model_version: str = "legacy-v1", decision_source: str = "legacy-source"
) -> dict[str, object]:
    if scope_variant == "environment_only":
        return {
            "source": "legacy_partial_scope",
            "environment": "paper",
            "model_version": model_version,
            "decision_source": decision_source,
        }
    if scope_variant == "portfolio_only":
        return {
            "source": "legacy_partial_scope",
            "portfolio": "paper-1",
            "model_version": model_version,
            "decision_source": decision_source,
        }
    raise ValueError(f"Unsupported scope_variant={scope_variant!r}")


@pytest.mark.parametrize(
    ("scope_variant", "execution_status", "filled_quantity", "avg_price", "expected_label_quality"),
    [
        ("environment_only", "partially_filled", 0.4, 110.0, "partial_exit_unconfirmed"),
        ("environment_only", "filled", 1.0, 110.0, "final"),
        ("portfolio_only", "partially_filled", 0.4, 110.0, "partial_exit_unconfirmed"),
        ("portfolio_only", "filled", 1.0, 110.0, "final"),
    ],
)
def test_controller_restored_partial_scope_close_paths_preserve_known_scope_without_fallback(
    tmp_path: Path,
    scope_variant: str,
    execution_status: str,
    filled_quantity: float,
    avg_price: float,
    expected_label_quality: str,
) -> None:
    decision_timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    correlation_key = OpportunityShadowRecord.build_record_key(
        symbol="BTC/USDT",
        decision_timestamp=decision_timestamp,
        model_version="opportunity-v1",
        rank=1,
    )
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    shadow_repo.append_shadow_records(
        [
            _shadow_record_for_key(
                correlation_key=correlation_key, decision_timestamp=decision_timestamp
            )
        ]
    )
    shadow_repo.upsert_open_outcome(
        shadow_repo.OpenOutcomeState(
            correlation_key=correlation_key,
            symbol="BTC/USDT",
            side="BUY",
            entry_price=100.0,
            decision_timestamp=decision_timestamp,
            entry_quantity=1.0,
            closed_quantity=0.0,
            provenance=_legacy_partial_scope_provenance(scope_variant=scope_variant),
        )
    )
    controller = TradingController(
        risk_engine=DummyRiskEngine(),
        execution_service=StatusExecutionService(
            status=execution_status,
            filled_quantity=filled_quantity,
            avg_price=avg_price,
        ),
        alert_router=_router_with_channel()[0],
        account_snapshot_provider=_account_snapshot,
        portfolio_id="wrong-portfolio-after-restart",
        environment="live",
        risk_profile="balanced",
        decision_journal=CollectingDecisionJournal(),
        opportunity_shadow_repository=shadow_repo,
    )
    close_signal = _signal("SELL", price=avg_price)
    close_signal.metadata = {
        **dict(close_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
    }
    controller.process_signals([close_signal])

    labels = shadow_repo.load_outcome_labels()
    assert len(labels) == 1
    assert labels[0].label_quality == expected_label_quality
    if scope_variant == "environment_only":
        assert labels[0].provenance.get("environment") == "paper"
        assert "portfolio" not in labels[0].provenance
    else:
        assert labels[0].provenance.get("portfolio") == "paper-1"
        assert "environment" not in labels[0].provenance
    assert labels[0].provenance.get("scope_continuity") == "restored_tracker_scope_missing"
    assert labels[0].provenance.get("model_version") == "legacy-v1"
    assert labels[0].provenance.get("decision_source") == "legacy-source"


@pytest.mark.parametrize("scope_variant", ["environment_only", "portfolio_only"])
def test_controller_restored_partial_scope_proxy_attach_preserves_known_scope_without_fallback(
    tmp_path: Path,
    scope_variant: str,
) -> None:
    decision_timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    correlation_key = OpportunityShadowRecord.build_record_key(
        symbol="BTC/USDT",
        decision_timestamp=decision_timestamp,
        model_version="opportunity-v1",
        rank=1,
    )
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    shadow_repo.append_shadow_records(
        [
            _shadow_record_for_key(
                correlation_key=correlation_key, decision_timestamp=decision_timestamp
            )
        ]
    )
    shadow_repo.upsert_open_outcome(
        shadow_repo.OpenOutcomeState(
            correlation_key=correlation_key,
            symbol="BTC/USDT",
            side="BUY",
            entry_price=100.0,
            decision_timestamp=decision_timestamp,
            entry_quantity=1.0,
            closed_quantity=0.0,
            provenance=_legacy_partial_scope_provenance(scope_variant=scope_variant),
        )
    )
    controller = TradingController(
        risk_engine=DummyRiskEngine(),
        execution_service=StatusExecutionService(
            status="rejected",
            filled_quantity=0.0,
            avg_price=None,
        ),
        alert_router=_router_with_channel()[0],
        account_snapshot_provider=_account_snapshot,
        portfolio_id="wrong-portfolio-after-restart",
        environment="live",
        risk_profile="balanced",
        decision_journal=CollectingDecisionJournal(),
        opportunity_shadow_repository=shadow_repo,
    )
    signal = _signal("BUY", price=99.0)
    signal.metadata = {
        **dict(signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
        "opportunity_decision_timestamp": decision_timestamp.isoformat(),
    }
    controller.process_signals([signal])

    labels = shadow_repo.load_outcome_labels()
    assert len(labels) == 1
    assert labels[0].label_quality == "execution_proxy_pending_exit"
    if scope_variant == "environment_only":
        assert labels[0].provenance.get("environment") == "paper"
        assert "portfolio" not in labels[0].provenance
    else:
        assert labels[0].provenance.get("portfolio") == "paper-1"
        assert "environment" not in labels[0].provenance
    assert labels[0].provenance.get("scope_continuity") == "restored_tracker_scope_missing"
    assert labels[0].provenance.get("model_version") == "legacy-v1"
    assert labels[0].provenance.get("decision_source") == "legacy-source"


@pytest.mark.parametrize("scope_variant", ["environment_only", "portfolio_only"])
def test_controller_restored_partial_scope_multihop_partial_restart_final_preserves_known_scope(
    tmp_path: Path,
    scope_variant: str,
) -> None:
    decision_timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    correlation_key = OpportunityShadowRecord.build_record_key(
        symbol="BTC/USDT",
        decision_timestamp=decision_timestamp,
        model_version="opportunity-v1",
        rank=1,
    )
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    shadow_repo.append_shadow_records(
        [
            _shadow_record_for_key(
                correlation_key=correlation_key, decision_timestamp=decision_timestamp
            )
        ]
    )
    shadow_repo.upsert_open_outcome(
        shadow_repo.OpenOutcomeState(
            correlation_key=correlation_key,
            symbol="BTC/USDT",
            side="BUY",
            entry_price=100.0,
            decision_timestamp=decision_timestamp,
            entry_quantity=1.0,
            closed_quantity=0.0,
            provenance=_legacy_partial_scope_provenance(scope_variant=scope_variant),
        )
    )

    controller_partial = TradingController(
        risk_engine=DummyRiskEngine(),
        execution_service=StatusExecutionService(
            status="partially_filled",
            filled_quantity=0.4,
            avg_price=110.0,
        ),
        alert_router=_router_with_channel()[0],
        account_snapshot_provider=_account_snapshot,
        portfolio_id="wrong-portfolio-hop-1",
        environment="live",
        risk_profile="balanced",
        decision_journal=CollectingDecisionJournal(),
        opportunity_shadow_repository=shadow_repo,
    )
    partial_signal = _signal("SELL", price=110.0)
    partial_signal.metadata = {
        **dict(partial_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
    }
    controller_partial.process_signals([partial_signal])

    partial_labels = shadow_repo.load_outcome_labels()
    assert len(partial_labels) == 1
    assert partial_labels[0].label_quality == "partial_exit_unconfirmed"
    if scope_variant == "environment_only":
        assert partial_labels[0].provenance.get("environment") == "paper"
        assert "portfolio" not in partial_labels[0].provenance
    else:
        assert partial_labels[0].provenance.get("portfolio") == "paper-1"
        assert "environment" not in partial_labels[0].provenance
    assert partial_labels[0].provenance.get("scope_continuity") == "restored_tracker_scope_missing"

    open_rows_after_partial = shadow_repo.load_open_outcomes()
    assert len(open_rows_after_partial) == 1
    if scope_variant == "environment_only":
        assert open_rows_after_partial[0].provenance.get("environment") == "paper"
        assert "portfolio" not in open_rows_after_partial[0].provenance
    else:
        assert open_rows_after_partial[0].provenance.get("portfolio") == "paper-1"
        assert "environment" not in open_rows_after_partial[0].provenance
    assert (
        open_rows_after_partial[0].provenance.get("scope_continuity")
        == "missing_from_restored_open_outcome"
    )

    controller_final = TradingController(
        risk_engine=DummyRiskEngine(),
        execution_service=StatusExecutionService(
            status="filled",
            filled_quantity=0.6,
            avg_price=112.0,
        ),
        alert_router=_router_with_channel()[0],
        account_snapshot_provider=_account_snapshot,
        portfolio_id="wrong-portfolio-hop-2",
        environment="staging",
        risk_profile="balanced",
        decision_journal=CollectingDecisionJournal(),
        opportunity_shadow_repository=shadow_repo,
    )
    final_signal = _signal("SELL", price=112.0)
    final_signal.metadata = {
        **dict(final_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
    }
    controller_final.process_signals([final_signal])

    labels = shadow_repo.load_outcome_labels()
    assert len(labels) == 1
    assert labels[0].label_quality == "final"
    if scope_variant == "environment_only":
        assert labels[0].provenance.get("environment") == "paper"
        assert "portfolio" not in labels[0].provenance
    else:
        assert labels[0].provenance.get("portfolio") == "paper-1"
        assert "environment" not in labels[0].provenance
    assert labels[0].provenance.get("scope_continuity") == "restored_tracker_scope_missing"
    assert labels[0].provenance.get("model_version") == "legacy-v1"
    assert labels[0].provenance.get("decision_source") == "legacy-source"
    assert shadow_repo.load_open_outcomes() == []


@pytest.mark.parametrize(
    ("scope_variant", "close_status", "close_filled_quantity", "close_avg_price", "expected_label_quality"),
    [
        ("environment_only", "partially_filled", 0.4, 109.0, "partial_exit_unconfirmed"),
        ("environment_only", "filled", 1.0, 112.0, "final"),
        ("portfolio_only", "partially_filled", 0.4, 109.0, "partial_exit_unconfirmed"),
        ("portfolio_only", "filled", 1.0, 112.0, "final"),
    ],
)
def test_controller_restored_partial_scope_multihop_proxy_restart_close_preserves_contract(
    tmp_path: Path,
    scope_variant: str,
    close_status: str,
    close_filled_quantity: float,
    close_avg_price: float,
    expected_label_quality: str,
) -> None:
    decision_timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    correlation_key = OpportunityShadowRecord.build_record_key(
        symbol="BTC/USDT",
        decision_timestamp=decision_timestamp,
        model_version="opportunity-v1",
        rank=1,
    )
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    shadow_repo.append_shadow_records(
        [
            _shadow_record_for_key(
                correlation_key=correlation_key, decision_timestamp=decision_timestamp
            )
        ]
    )
    shadow_repo.upsert_open_outcome(
        shadow_repo.OpenOutcomeState(
            correlation_key=correlation_key,
            symbol="BTC/USDT",
            side="BUY",
            entry_price=100.0,
            decision_timestamp=decision_timestamp,
            entry_quantity=1.0,
            closed_quantity=0.0,
            provenance=_legacy_partial_scope_provenance(scope_variant=scope_variant),
        )
    )

    controller_proxy = TradingController(
        risk_engine=DummyRiskEngine(),
        execution_service=StatusExecutionService(
            status="rejected",
            filled_quantity=0.0,
            avg_price=None,
        ),
        alert_router=_router_with_channel()[0],
        account_snapshot_provider=_account_snapshot,
        portfolio_id="wrong-portfolio-proxy-hop",
        environment="live",
        risk_profile="balanced",
        decision_journal=CollectingDecisionJournal(),
        opportunity_shadow_repository=shadow_repo,
    )
    proxy_signal = _signal("BUY", price=99.0)
    proxy_signal.metadata = {
        **dict(proxy_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
        "opportunity_decision_timestamp": decision_timestamp.isoformat(),
    }
    controller_proxy.process_signals([proxy_signal])

    proxy_labels = shadow_repo.load_outcome_labels()
    assert len(proxy_labels) == 1
    assert proxy_labels[0].label_quality == "execution_proxy_pending_exit"
    if scope_variant == "environment_only":
        assert proxy_labels[0].provenance.get("environment") == "paper"
        assert "portfolio" not in proxy_labels[0].provenance
    else:
        assert proxy_labels[0].provenance.get("portfolio") == "paper-1"
        assert "environment" not in proxy_labels[0].provenance
    assert proxy_labels[0].provenance.get("scope_continuity") == "restored_tracker_scope_missing"
    assert proxy_labels[0].provenance.get("model_version") == "legacy-v1"
    assert proxy_labels[0].provenance.get("decision_source") == "legacy-source"

    open_rows_after_proxy = shadow_repo.load_open_outcomes()
    assert len(open_rows_after_proxy) == 1
    if scope_variant == "environment_only":
        assert open_rows_after_proxy[0].provenance.get("environment") == "paper"
        assert "portfolio" not in open_rows_after_proxy[0].provenance
    else:
        assert open_rows_after_proxy[0].provenance.get("portfolio") == "paper-1"
        assert "environment" not in open_rows_after_proxy[0].provenance

    controller_close = TradingController(
        risk_engine=DummyRiskEngine(),
        execution_service=StatusExecutionService(
            status=close_status,
            filled_quantity=close_filled_quantity,
            avg_price=close_avg_price,
        ),
        alert_router=_router_with_channel()[0],
        account_snapshot_provider=_account_snapshot,
        portfolio_id="wrong-portfolio-close-hop",
        environment="staging",
        risk_profile="balanced",
        decision_journal=CollectingDecisionJournal(),
        opportunity_shadow_repository=shadow_repo,
    )
    close_signal = _signal("SELL", price=close_avg_price)
    close_signal.metadata = {
        **dict(close_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
    }
    controller_close.process_signals([close_signal])

    labels = shadow_repo.load_outcome_labels()
    assert len(labels) == 1
    assert labels[0].label_quality == expected_label_quality
    if scope_variant == "environment_only":
        assert labels[0].provenance.get("environment") == "paper"
        assert "portfolio" not in labels[0].provenance
    else:
        assert labels[0].provenance.get("portfolio") == "paper-1"
        assert "environment" not in labels[0].provenance
    assert labels[0].provenance.get("scope_continuity") == "restored_tracker_scope_missing"
    assert labels[0].provenance.get("model_version") == "legacy-v1"
    assert labels[0].provenance.get("decision_source") == "legacy-source"

    open_rows_after_close = shadow_repo.load_open_outcomes()
    if expected_label_quality == "final":
        assert open_rows_after_close == []
    else:
        assert len(open_rows_after_close) == 1
        if scope_variant == "environment_only":
            assert open_rows_after_close[0].provenance.get("environment") == "paper"
            assert "portfolio" not in open_rows_after_close[0].provenance
        else:
            assert open_rows_after_close[0].provenance.get("portfolio") == "paper-1"
            assert "environment" not in open_rows_after_close[0].provenance
        assert (
            open_rows_after_close[0].provenance.get("scope_continuity")
            == "missing_from_restored_open_outcome"
        )


def test_controller_proxy_attach_after_restart_with_legacy_scope_gap_is_auditable(
    tmp_path: Path,
) -> None:
    decision_timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    correlation_key = OpportunityShadowRecord.build_record_key(
        symbol="BTC/USDT",
        decision_timestamp=decision_timestamp,
        model_version="opportunity-v1",
        rank=1,
    )
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    shadow_repo.append_shadow_records(
        [
            _shadow_record_for_key(
                correlation_key=correlation_key, decision_timestamp=decision_timestamp
            )
        ]
    )
    shadow_repo.upsert_open_outcome(
        shadow_repo.OpenOutcomeState(
            correlation_key=correlation_key,
            symbol="BTC/USDT",
            side="BUY",
            entry_price=100.0,
            decision_timestamp=decision_timestamp,
            entry_quantity=1.0,
            closed_quantity=0.0,
            provenance={
                "source": "legacy_without_scope",
                "model_version": "legacy-v1",
                "decision_source": "legacy-source",
            },
        )
    )

    controller = TradingController(
        risk_engine=DummyRiskEngine(),
        execution_service=StatusExecutionService(
            status="rejected",
            filled_quantity=0.0,
            avg_price=None,
        ),
        alert_router=_router_with_channel()[0],
        account_snapshot_provider=_account_snapshot,
        portfolio_id="wrong-portfolio-after-restart",
        environment="live",
        risk_profile="balanced",
        decision_journal=CollectingDecisionJournal(),
        opportunity_shadow_repository=shadow_repo,
    )
    signal = _signal("BUY", price=99.0)
    signal.metadata = {
        **dict(signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
        "opportunity_decision_timestamp": decision_timestamp.isoformat(),
    }
    controller.process_signals([signal])

    labels = shadow_repo.load_outcome_labels()
    assert len(labels) == 1
    assert labels[0].label_quality == "execution_proxy_pending_exit"
    assert "environment" not in labels[0].provenance
    assert "portfolio" not in labels[0].provenance
    assert labels[0].provenance.get("scope_continuity") == "restored_tracker_scope_missing"
    assert labels[0].provenance.get("model_version") == "legacy-v1"
    assert labels[0].provenance.get("decision_source") == "legacy-source"


def test_controller_multihop_restart_partial_then_final_persists_recovered_lineage(
    tmp_path: Path,
) -> None:
    decision_timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    correlation_key = OpportunityShadowRecord.build_record_key(
        symbol="BTC/USDT",
        decision_timestamp=decision_timestamp,
        model_version="shadow-v0",
        rank=1,
    )
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    shadow_repo.append_shadow_records(
        [
            OpportunityShadowRecord(
                record_key=correlation_key,
                symbol="BTC/USDT",
                decision_timestamp=decision_timestamp,
                model_version="shadow-v0",
                decision_source="shadow_source",
                expected_edge_bps=5.0,
                success_probability=0.7,
                confidence=0.3,
                proposed_direction="long",
                accepted=True,
                rejection_reason=None,
                rank=1,
                provenance={"probability_method": "test"},
                threshold_config=OpportunityThresholdConfig(),
                snapshot={},
                context=OpportunityShadowContext(environment="paper"),
            )
        ]
    )
    controller_open = TradingController(
        risk_engine=DummyRiskEngine(),
        execution_service=DummyExecutionService(),
        alert_router=_router_with_channel()[0],
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        decision_journal=CollectingDecisionJournal(),
        opportunity_shadow_repository=shadow_repo,
    )
    open_signal = _signal("BUY", price=100.0)
    open_signal.metadata = {
        **dict(open_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
        "opportunity_decision_timestamp": decision_timestamp.isoformat(),
    }
    controller_open.process_signals([open_signal])

    open_rows = shadow_repo.load_open_outcomes()
    assert len(open_rows) == 1
    shadow_repo.upsert_open_outcome(
        shadow_repo.OpenOutcomeState(
            correlation_key=correlation_key,
            symbol=open_rows[0].symbol,
            side=open_rows[0].side,
            entry_price=open_rows[0].entry_price,
            decision_timestamp=open_rows[0].decision_timestamp,
            entry_quantity=open_rows[0].entry_quantity,
            closed_quantity=open_rows[0].closed_quantity,
            provenance={
                "source": "legacy_without_lineage",
                "environment": "paper",
                "portfolio": "paper-1",
            },
        )
    )

    controller_partial = TradingController(
        risk_engine=DummyRiskEngine(),
        execution_service=StatusExecutionService(
            status="partially_filled",
            filled_quantity=0.4,
            avg_price=108.0,
        ),
        alert_router=_router_with_channel()[0],
        account_snapshot_provider=_account_snapshot,
        portfolio_id="wrong-portfolio-partial",
        environment="live",
        risk_profile="balanced",
        decision_journal=CollectingDecisionJournal(),
        opportunity_shadow_repository=shadow_repo,
    )
    partial_signal = _signal("SELL", price=108.0)
    partial_signal.metadata = {
        **dict(partial_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
        "opportunity_model_version": "recovered-v1",
        "opportunity_decision_source": "recovered-source",
    }
    controller_partial.process_signals([partial_signal])

    open_rows_after_partial = shadow_repo.load_open_outcomes()
    assert len(open_rows_after_partial) == 1
    assert open_rows_after_partial[0].provenance.get("environment") == "paper"
    assert open_rows_after_partial[0].provenance.get("portfolio") == "paper-1"
    assert open_rows_after_partial[0].provenance.get("model_version") == "recovered-v1"
    assert open_rows_after_partial[0].provenance.get("decision_source") == "recovered-source"
    partial_labels = shadow_repo.load_outcome_labels()
    assert len(partial_labels) == 1
    assert partial_labels[0].label_quality == "partial_exit_unconfirmed"
    assert partial_labels[0].provenance.get("environment") == "paper"
    assert partial_labels[0].provenance.get("portfolio") == "paper-1"
    assert partial_labels[0].provenance.get("model_version") == "recovered-v1"
    assert partial_labels[0].provenance.get("decision_source") == "recovered-source"

    controller_final = TradingController(
        risk_engine=DummyRiskEngine(),
        execution_service=StatusExecutionService(
            status="filled",
            filled_quantity=0.6,
            avg_price=112.0,
        ),
        alert_router=_router_with_channel()[0],
        account_snapshot_provider=_account_snapshot,
        portfolio_id="wrong-portfolio-final",
        environment="live",
        risk_profile="balanced",
        decision_journal=CollectingDecisionJournal(),
        opportunity_shadow_repository=shadow_repo,
    )
    final_signal = _signal("SELL", price=112.0)
    final_signal.metadata = {
        **dict(final_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
        "opportunity_model_version": "",
        "opportunity_decision_source": "",
    }
    controller_final.process_signals([final_signal])

    labels = shadow_repo.load_outcome_labels()
    assert len(labels) == 1
    assert labels[0].label_quality == "final"
    assert labels[0].provenance.get("environment") == "paper"
    assert labels[0].provenance.get("portfolio") == "paper-1"
    assert labels[0].provenance.get("model_version") == "recovered-v1"
    assert labels[0].provenance.get("decision_source") == "recovered-source"
    assert labels[0].provenance.get("model_version") != "shadow-v0"
    assert labels[0].provenance.get("decision_source") != "shadow_source"
    assert shadow_repo.load_open_outcomes() == []


def test_controller_close_without_correlation_key_is_rejected_when_ambiguous(
    tmp_path: Path,
) -> None:
    decision_timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    keyed_timestamps: list[tuple[str, datetime]] = []
    for rank in (1, 2):
        keyed_ts = decision_timestamp + timedelta(minutes=rank)
        key = OpportunityShadowRecord.build_record_key(
            symbol="BTC/USDT",
            decision_timestamp=keyed_ts,
            model_version="opportunity-v1",
            rank=rank,
        )
        keyed_timestamps.append((key, keyed_ts))
        shadow_repo.append_shadow_records(
            [_shadow_record_for_key(correlation_key=key, decision_timestamp=keyed_ts)]
        )
    controller = TradingController(
        risk_engine=DummyRiskEngine(),
        execution_service=DummyExecutionService(),
        alert_router=_router_with_channel()[0],
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        decision_journal=CollectingDecisionJournal(),
        opportunity_shadow_repository=shadow_repo,
    )
    for key, keyed_ts in keyed_timestamps:
        signal = _signal("BUY", price=100.0)
        signal.metadata = {
            **dict(signal.metadata),
            "opportunity_shadow_record_key": key,
            "opportunity_decision_timestamp": keyed_ts.isoformat(),
        }
        controller.process_signals([signal])

    close_signal = _signal("SELL", price=110.0)
    controller.process_signals([close_signal])

    labels = shadow_repo.load_outcome_labels()
    assert len(labels) == 2
    assert all(label.label_quality == "execution_proxy_pending_exit" for label in labels)
    assert len(shadow_repo.load_open_outcomes()) == 2


def test_controller_close_with_correlation_key_missing_tracker_is_unresolved_without_new_open_tracker(
    tmp_path: Path,
) -> None:
    decision_timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    correlation_key = OpportunityShadowRecord.build_record_key(
        symbol="BTC/USDT",
        decision_timestamp=decision_timestamp,
        model_version="opportunity-v1",
        rank=1,
    )
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    shadow_repo.append_shadow_records(
        [
            _shadow_record_for_key(
                correlation_key=correlation_key, decision_timestamp=decision_timestamp
            )
        ]
    )
    journal = CollectingDecisionJournal()
    controller = TradingController(
        risk_engine=DummyRiskEngine(),
        execution_service=DummyExecutionService(),
        alert_router=_router_with_channel()[0],
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        decision_journal=journal,
        opportunity_shadow_repository=shadow_repo,
    )
    close_signal = _signal("SELL", price=110.0)
    close_signal.metadata = {
        **dict(close_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
    }
    controller.process_signals([close_signal])

    assert shadow_repo.load_open_outcomes() == []
    assert shadow_repo.load_outcome_labels() == []
    attach_events = [
        event for event in journal.export() if event["event"] == "opportunity_outcome_attach"
    ]
    assert attach_events[-1]["status"] == "close_correlation_unresolved"
    assert attach_events[-1]["close_correlation_resolution"] == "missing"


def test_controller_close_with_correlation_key_side_mismatch_is_unresolved_without_new_open_tracker(
    tmp_path: Path,
) -> None:
    decision_timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    correlation_key = OpportunityShadowRecord.build_record_key(
        symbol="BTC/USDT",
        decision_timestamp=decision_timestamp,
        model_version="opportunity-v1",
        rank=1,
    )
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    shadow_repo.append_shadow_records(
        [
            _shadow_record_for_key(
                correlation_key=correlation_key, decision_timestamp=decision_timestamp
            )
        ]
    )
    journal = CollectingDecisionJournal()
    controller = TradingController(
        risk_engine=DummyRiskEngine(),
        execution_service=DummyExecutionService(),
        alert_router=_router_with_channel()[0],
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        decision_journal=journal,
        opportunity_shadow_repository=shadow_repo,
    )
    open_signal = _signal("BUY", price=100.0)
    open_signal.metadata = {
        **dict(open_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
        "opportunity_decision_timestamp": decision_timestamp.isoformat(),
    }
    controller.process_signals([open_signal])

    side_mismatch_close = _signal("BUY", price=109.0)
    side_mismatch_close.metadata = {
        **dict(side_mismatch_close.metadata),
        "opportunity_shadow_record_key": correlation_key,
    }
    controller.process_signals([side_mismatch_close])

    labels = shadow_repo.load_outcome_labels()
    assert len(labels) == 1
    assert labels[0].label_quality == "execution_proxy_pending_exit"
    open_outcomes = shadow_repo.load_open_outcomes()
    assert len(open_outcomes) == 1
    attach_events = [
        event for event in journal.export() if event["event"] == "opportunity_outcome_attach"
    ]
    assert attach_events[-1]["status"] == "close_correlation_unresolved"
    assert attach_events[-1]["close_correlation_resolution"] == "side_mismatch"


def test_controller_close_with_correlation_key_symbol_mismatch_is_unresolved_without_new_open_tracker(
    tmp_path: Path,
) -> None:
    decision_timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    correlation_key = OpportunityShadowRecord.build_record_key(
        symbol="BTC/USDT",
        decision_timestamp=decision_timestamp,
        model_version="opportunity-v1",
        rank=1,
    )
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    shadow_repo.append_shadow_records(
        [
            _shadow_record_for_key(
                correlation_key=correlation_key, decision_timestamp=decision_timestamp
            )
        ]
    )
    shadow_repo.upsert_open_outcome(
        shadow_repo.OpenOutcomeState(
            correlation_key=correlation_key,
            symbol="ETH/USDT",
            side="BUY",
            entry_price=100.0,
            entry_quantity=1.0,
            decision_timestamp=decision_timestamp,
            provenance={"source": "test"},
        )
    )
    journal = CollectingDecisionJournal()
    controller = TradingController(
        risk_engine=DummyRiskEngine(),
        execution_service=DummyExecutionService(),
        alert_router=_router_with_channel()[0],
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        decision_journal=journal,
        opportunity_shadow_repository=shadow_repo,
    )
    close_signal = _signal("SELL", price=110.0)
    close_signal.metadata = {
        **dict(close_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
    }
    controller.process_signals([close_signal])

    assert len(shadow_repo.load_open_outcomes()) == 1
    assert shadow_repo.load_outcome_labels() == []
    attach_events = [
        event for event in journal.export() if event["event"] == "opportunity_outcome_attach"
    ]
    assert attach_events[-1]["status"] == "close_correlation_unresolved"
    assert attach_events[-1]["close_correlation_resolution"] == "symbol_mismatch"


def test_controller_close_with_correlation_key_and_timestamp_hint_missing_tracker_still_unresolved(
    tmp_path: Path,
) -> None:
    decision_timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    correlation_key = OpportunityShadowRecord.build_record_key(
        symbol="BTC/USDT",
        decision_timestamp=decision_timestamp,
        model_version="opportunity-v1",
        rank=1,
    )
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    shadow_repo.append_shadow_records(
        [
            _shadow_record_for_key(
                correlation_key=correlation_key, decision_timestamp=decision_timestamp
            )
        ]
    )
    journal = CollectingDecisionJournal()
    controller = TradingController(
        risk_engine=DummyRiskEngine(),
        execution_service=DummyExecutionService(),
        alert_router=_router_with_channel()[0],
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        decision_journal=journal,
        opportunity_shadow_repository=shadow_repo,
    )
    close_signal = _signal("SELL", price=110.0)
    close_signal.metadata = {
        **dict(close_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
        "opportunity_decision_timestamp": decision_timestamp.isoformat(),
    }
    controller.process_signals([close_signal])
    assert shadow_repo.load_open_outcomes() == []
    assert shadow_repo.load_outcome_labels() == []
    attach_events = [
        event for event in journal.export() if event["event"] == "opportunity_outcome_attach"
    ]
    assert attach_events[-1]["status"] == "close_correlation_unresolved"
    assert attach_events[-1]["close_correlation_resolution"] == "missing"


def test_controller_close_with_correlation_key_and_timestamp_hint_side_mismatch_still_unresolved(
    tmp_path: Path,
) -> None:
    decision_timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    correlation_key = OpportunityShadowRecord.build_record_key(
        symbol="BTC/USDT",
        decision_timestamp=decision_timestamp,
        model_version="opportunity-v1",
        rank=1,
    )
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    shadow_repo.append_shadow_records(
        [
            _shadow_record_for_key(
                correlation_key=correlation_key, decision_timestamp=decision_timestamp
            )
        ]
    )
    journal = CollectingDecisionJournal()
    controller = TradingController(
        risk_engine=DummyRiskEngine(),
        execution_service=DummyExecutionService(),
        alert_router=_router_with_channel()[0],
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        decision_journal=journal,
        opportunity_shadow_repository=shadow_repo,
    )
    open_signal = _signal("BUY", price=100.0)
    open_signal.metadata = {
        **dict(open_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
        "opportunity_decision_timestamp": decision_timestamp.isoformat(),
    }
    controller.process_signals([open_signal])
    close_signal = _signal("BUY", price=109.0)
    close_signal.metadata = {
        **dict(close_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
        "opportunity_decision_timestamp": decision_timestamp.isoformat(),
    }
    controller.process_signals([close_signal])
    labels = shadow_repo.load_outcome_labels()
    assert len(labels) == 1
    assert labels[0].label_quality == "execution_proxy_pending_exit"
    attach_events = [
        event for event in journal.export() if event["event"] == "opportunity_outcome_attach"
    ]
    assert attach_events[-1]["status"] == "close_correlation_unresolved"
    assert attach_events[-1]["close_correlation_resolution"] == "side_mismatch"


def test_controller_close_with_correlation_key_and_timestamp_hint_symbol_mismatch_still_unresolved(
    tmp_path: Path,
) -> None:
    decision_timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    correlation_key = OpportunityShadowRecord.build_record_key(
        symbol="BTC/USDT",
        decision_timestamp=decision_timestamp,
        model_version="opportunity-v1",
        rank=1,
    )
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    shadow_repo.append_shadow_records(
        [
            _shadow_record_for_key(
                correlation_key=correlation_key, decision_timestamp=decision_timestamp
            )
        ]
    )
    shadow_repo.upsert_open_outcome(
        shadow_repo.OpenOutcomeState(
            correlation_key=correlation_key,
            symbol="ETH/USDT",
            side="BUY",
            entry_price=100.0,
            entry_quantity=1.0,
            decision_timestamp=decision_timestamp,
            provenance={"source": "test"},
        )
    )
    journal = CollectingDecisionJournal()
    controller = TradingController(
        risk_engine=DummyRiskEngine(),
        execution_service=DummyExecutionService(),
        alert_router=_router_with_channel()[0],
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        decision_journal=journal,
        opportunity_shadow_repository=shadow_repo,
    )
    close_signal = _signal("SELL", price=110.0)
    close_signal.metadata = {
        **dict(close_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
        "opportunity_decision_timestamp": decision_timestamp.isoformat(),
    }
    controller.process_signals([close_signal])
    assert len(shadow_repo.load_open_outcomes()) == 1
    assert shadow_repo.load_outcome_labels() == []
    attach_events = [
        event for event in journal.export() if event["event"] == "opportunity_outcome_attach"
    ]
    assert attach_events[-1]["status"] == "close_correlation_unresolved"
    assert attach_events[-1]["close_correlation_resolution"] == "symbol_mismatch"


def test_controller_partial_close_does_not_create_final_or_drop_tracker(tmp_path: Path) -> None:
    decision_timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    correlation_key = OpportunityShadowRecord.build_record_key(
        symbol="BTC/USDT",
        decision_timestamp=decision_timestamp,
        model_version="opportunity-v1",
        rank=1,
    )
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    shadow_repo.append_shadow_records(
        [
            _shadow_record_for_key(
                correlation_key=correlation_key, decision_timestamp=decision_timestamp
            )
        ]
    )
    execution = SequencedExecutionService(
        [
            {"status": "filled", "filled_quantity": 1.0, "avg_price": 100.0},
            {"status": "partially_filled", "filled_quantity": 0.4, "avg_price": 110.0},
        ]
    )
    journal = CollectingDecisionJournal()
    controller = TradingController(
        risk_engine=DummyRiskEngine(),
        execution_service=execution,
        alert_router=_router_with_channel()[0],
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        decision_journal=journal,
        opportunity_shadow_repository=shadow_repo,
    )
    open_signal = _signal("BUY", price=100.0)
    open_signal.metadata = {
        **dict(open_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
        "opportunity_decision_timestamp": decision_timestamp.isoformat(),
    }
    close_signal = _signal("SELL", price=110.0)
    close_signal.metadata = {
        **dict(close_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
    }
    controller.process_signals([open_signal])
    controller.process_signals([close_signal])

    labels = shadow_repo.load_outcome_labels()
    assert len(labels) == 1
    assert labels[0].label_quality == "partial_exit_unconfirmed"
    assert labels[0].provenance.get("close_confirmation") == "insufficient_evidence_for_final_close"
    assert labels[0].provenance.get("model_version") == "opportunity-v1"
    assert labels[0].provenance.get("decision_source") == "opportunity_ai_shadow"
    open_outcomes = shadow_repo.load_open_outcomes()
    assert len(open_outcomes) == 1
    assert open_outcomes[0].closed_quantity == pytest.approx(0.4, rel=1e-6)
    attach_events = [
        event for event in journal.export() if event["event"] == "opportunity_outcome_attach"
    ]
    assert attach_events[-1]["status"] == "quality_upgraded"


def test_controller_final_close_after_partial_upgrades_and_removes_tracker(tmp_path: Path) -> None:
    decision_timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    correlation_key = OpportunityShadowRecord.build_record_key(
        symbol="BTC/USDT",
        decision_timestamp=decision_timestamp,
        model_version="opportunity-v1",
        rank=1,
    )
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    shadow_repo.append_shadow_records(
        [
            _shadow_record_for_key(
                correlation_key=correlation_key, decision_timestamp=decision_timestamp
            )
        ]
    )
    execution = SequencedExecutionService(
        [
            {"status": "filled", "filled_quantity": 1.0, "avg_price": 100.0},
            {"status": "partially_filled", "filled_quantity": 0.4, "avg_price": 108.0},
            {"status": "filled", "filled_quantity": 0.6, "avg_price": 112.0},
        ]
    )
    controller = TradingController(
        risk_engine=DummyRiskEngine(),
        execution_service=execution,
        alert_router=_router_with_channel()[0],
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        decision_journal=CollectingDecisionJournal(),
        opportunity_shadow_repository=shadow_repo,
    )
    open_signal = _signal("BUY", price=100.0)
    open_signal.metadata = {
        **dict(open_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
        "opportunity_decision_timestamp": decision_timestamp.isoformat(),
    }
    partial_close_signal = _signal("SELL", price=108.0)
    partial_close_signal.metadata = {
        **dict(partial_close_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
    }
    final_close_signal = _signal("SELL", price=112.0)
    final_close_signal.metadata = {
        **dict(final_close_signal.metadata),
        "opportunity_shadow_record_key": correlation_key,
    }
    controller.process_signals([open_signal])
    controller.process_signals([partial_close_signal])
    controller.process_signals([final_close_signal])

    labels = shadow_repo.load_outcome_labels()
    assert len(labels) == 1
    assert labels[0].label_quality == "final"
    assert labels[0].provenance.get("close_confirmation") == "quantity_and_filled_status"
    assert labels[0].provenance.get("model_version") == "opportunity-v1"
    assert labels[0].provenance.get("decision_source") == "opportunity_ai_shadow"
    assert shadow_repo.load_open_outcomes() == []


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
    assert all(
        message.category == "execution"
        for message in channel.messages
        if message.category == "execution"
    )
    assert any(event["event"] == "order_executed" for event in journal.export())


def test_controller_multi_leg_uses_per_leg_quantity_when_parent_quantity_is_set() -> None:
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, _channel, _audit = _router_with_channel()
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
        confidence=0.91,
        intent="multi_leg",
        quantity=10.0,
        metadata={"order_type": "market"},
        legs=(
            SignalLeg(
                symbol="BTC/USDT",
                side="BUY",
                quantity=1.25,
                metadata={"price": 101.0},
            ),
            SignalLeg(
                symbol="ETH/USDT",
                side="SELL",
                quantity=2.75,
                metadata={"price": 102.0},
            ),
        ),
    )

    results = controller.process_signals([signal])

    assert len(results) == 2
    assert [request.side for request in execution.requests] == ["BUY", "SELL"]
    assert [request.symbol for request in execution.requests] == ["BTC/USDT", "ETH/USDT"]
    assert [request.quantity for request in execution.requests] == pytest.approx([1.25, 2.75])


def test_controller_multi_leg_runtime_provenance_keys_override_leg_metadata() -> None:
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, _channel, _audit = _router_with_channel()
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
        confidence=0.9,
        intent="multi_leg",
        metadata={"order_type": "market"},
        legs=(
            SignalLeg(
                symbol="BTC/USDT",
                side="BUY",
                quantity=1.0,
                exchange="BINANCE",
                metadata={
                    "price": 101.0,
                    "leg_index": 999,
                    "leg_count": 1,
                    "signal_intent": "single",
                    "exchange": "SPOOFED",
                },
            ),
            SignalLeg(
                symbol="ETH/USDT",
                side="SELL",
                quantity=2.0,
                exchange="KRAKEN",
                metadata={
                    "price": 102.0,
                    "leg_index": 888,
                    "leg_count": 1,
                    "signal_intent": "single",
                    "exchange": "SPOOFED",
                },
            ),
        ),
    )

    results = controller.process_signals([signal])

    assert len(results) == 2
    assert len(execution.requests) == 2

    first_request, second_request = execution.requests
    assert first_request.metadata["leg_index"] == 0
    assert first_request.metadata["leg_count"] == 2
    assert first_request.metadata["signal_intent"] == "multi_leg"
    assert first_request.metadata["exchange"] == "BINANCE"

    assert second_request.metadata["leg_index"] == 1
    assert second_request.metadata["leg_count"] == 2
    assert second_request.metadata["signal_intent"] == "multi_leg"
    assert second_request.metadata["exchange"] == "KRAKEN"

    submitted_events = [event for event in journal.export() if event["event"] == "order_submitted"]
    assert len(submitted_events) == 2
    assert submitted_events[0]["order_leg_index"] == "0"
    assert submitted_events[0]["order_leg_count"] == "2"
    assert submitted_events[0]["order_signal_intent"] == "multi_leg"
    assert submitted_events[0]["order_exchange"] == "BINANCE"
    assert submitted_events[1]["order_leg_index"] == "1"
    assert submitted_events[1]["order_leg_count"] == "2"
    assert submitted_events[1]["order_signal_intent"] == "multi_leg"
    assert submitted_events[1]["order_exchange"] == "KRAKEN"


def test_controller_multi_leg_isolates_nested_metadata_between_requests() -> None:
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, _channel, _audit = _router_with_channel()
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
        confidence=0.92,
        intent="multi_leg",
        metadata={
            "order_type": "market",
            "routing": {"policy": "parent", "tags": ["alpha", "beta"]},
        },
        legs=(
            SignalLeg(
                symbol="BTC/USDT",
                side="BUY",
                quantity=1.2,
                metadata={"price": 101.0, "risk": {"bucket": "first", "limits": [1, 2]}},
            ),
            SignalLeg(
                symbol="ETH/USDT",
                side="SELL",
                quantity=2.4,
                metadata={"price": 102.0, "risk": {"bucket": "second", "limits": [3, 4]}},
            ),
        ),
    )

    results = controller.process_signals([signal])

    assert len(results) == 2
    assert len(execution.requests) == 2
    assert [request.side for request in execution.requests] == ["BUY", "SELL"]
    assert [request.quantity for request in execution.requests] == pytest.approx([1.2, 2.4])

    first_request, second_request = execution.requests
    assert first_request.metadata["leg_index"] == 0
    assert first_request.metadata["leg_count"] == 2
    assert first_request.metadata["signal_intent"] == "multi_leg"
    assert second_request.metadata["leg_index"] == 1
    assert second_request.metadata["leg_count"] == 2
    assert second_request.metadata["signal_intent"] == "multi_leg"

    assert first_request.metadata["routing"] == {"policy": "parent", "tags": ["alpha", "beta"]}
    assert second_request.metadata["routing"] == {"policy": "parent", "tags": ["alpha", "beta"]}
    assert first_request.metadata["risk"] == {"bucket": "first", "limits": [1, 2]}
    assert second_request.metadata["risk"] == {"bucket": "second", "limits": [3, 4]}

    first_request.metadata["routing"]["policy"] = "mutated-first"
    first_request.metadata["routing"]["tags"].append("mutated")
    first_request.metadata["risk"]["limits"].append(999)

    assert second_request.metadata["routing"] == {"policy": "parent", "tags": ["alpha", "beta"]}
    assert second_request.metadata["risk"] == {"bucket": "second", "limits": [3, 4]}


def test_controller_multi_leg_enforces_unique_client_order_id_from_parent_anchor() -> None:
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, _channel, _audit = _router_with_channel()
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
        confidence=0.93,
        intent="multi_leg",
        metadata={
            "order_type": "market",
            "client_order_id": "parent-anchor-123",
        },
        legs=(
            SignalLeg(
                symbol="BTC/USDT",
                side="BUY",
                quantity=1.0,
                metadata={"price": 101.0},
            ),
            SignalLeg(
                symbol="BTC/USDT",
                side="SELL",
                quantity=1.0,
                metadata={"price": 102.0},
            ),
        ),
    )

    results = controller.process_signals([signal])

    assert len(results) == 2
    assert len(execution.requests) == 2

    first_request, second_request = execution.requests
    assert first_request.client_order_id == "parent-anchor-123-L1"
    assert second_request.client_order_id == "parent-anchor-123-L2"
    assert first_request.client_order_id != second_request.client_order_id

    assert first_request.metadata["parent_client_order_id"] == "parent-anchor-123"
    assert second_request.metadata["parent_client_order_id"] == "parent-anchor-123"
    assert first_request.metadata["client_order_id"] == first_request.client_order_id
    assert second_request.metadata["client_order_id"] == second_request.client_order_id

    submitted_events = [event for event in journal.export() if event["event"] == "order_submitted"]
    assert len(submitted_events) == 2
    assert [event["client_order_id"] for event in submitted_events] == [
        "parent-anchor-123-L1",
        "parent-anchor-123-L2",
    ]
    assert [event["order_client_order_id"] for event in submitted_events] == [
        "parent-anchor-123-L1",
        "parent-anchor-123-L2",
    ]
    assert [event["order_parent_client_order_id"] for event in submitted_events] == [
        "parent-anchor-123",
        "parent-anchor-123",
    ]


def test_controller_multi_leg_enforces_unique_client_order_id_for_duplicate_leg_ids_without_parent_anchor() -> (
    None
):
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, _channel, _audit = _router_with_channel()
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
        confidence=0.93,
        intent="multi_leg",
        metadata={"order_type": "market"},
        legs=(
            SignalLeg(
                symbol="BTC/USDT",
                side="BUY",
                quantity=1.0,
                metadata={"price": 101.0, "client_order_id": "dup-leg-id"},
            ),
            SignalLeg(
                symbol="BTC/USDT",
                side="SELL",
                quantity=1.0,
                metadata={"price": 102.0, "client_order_id": "dup-leg-id"},
            ),
        ),
    )

    results = controller.process_signals([signal])

    assert len(results) == 2
    assert len(execution.requests) == 2

    first_request, second_request = execution.requests
    assert first_request.client_order_id == "dup-leg-id"
    assert second_request.client_order_id == "dup-leg-id-L2"
    assert first_request.client_order_id != second_request.client_order_id

    submitted_events = [event for event in journal.export() if event["event"] == "order_submitted"]
    assert len(submitted_events) == 2
    assert [event["client_order_id"] for event in submitted_events] == [
        "dup-leg-id",
        "dup-leg-id-L2",
    ]
    assert [event["order_client_order_id"] for event in submitted_events] == [
        "dup-leg-id",
        "dup-leg-id-L2",
    ]


def test_controller_non_filled_result_not_recorded_as_order_executed() -> None:
    risk_engine = DummyRiskEngine()
    execution = StatusExecutionService(status="rejected", filled_quantity=0.0)
    router, channel, _audit = _router_with_channel()
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
    events = [event["event"] for event in journal.export()]
    assert "order_executed" not in events
    assert "order_execution_result" in events
    execution_message = next(
        message for message in channel.messages if message.category == "execution"
    )
    assert execution_message.severity == "warning"
    assert "zrealizowane" not in execution_message.title.lower()
    assert execution_message.context["status"] == "rejected"
    assert execution.requests[0].client_order_id
    assert execution_message.context["client_order_id"] == execution.requests[0].client_order_id


def test_controller_non_filled_result_preserves_zero_fill_and_missing_avg_price() -> None:
    risk_engine = DummyRiskEngine()
    execution = StatusExecutionService(status="rejected", filled_quantity=0.0, avg_price=None)
    router, _channel, _audit = _router_with_channel()
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

    controller.process_signals([_signal("BUY", quantity=2.0, price=123.0)])

    non_filled_event = next(
        event for event in journal.export() if event["event"] == "order_execution_result"
    )
    assert non_filled_event["status"] == "rejected"
    assert non_filled_event["filled_quantity"] == "0.00000000"
    assert non_filled_event["avg_price"] == "null"
    assert non_filled_event["filled_quantity"] != "2.00000000"
    assert non_filled_event["avg_price"] != "123.00000000"


def test_controller_partial_result_uses_distinct_journal_event() -> None:
    risk_engine = DummyRiskEngine()
    execution = StatusExecutionService(status="partially_filled", filled_quantity=0.25)
    router, channel, _audit = _router_with_channel()
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

    results = controller.process_signals([_signal("BUY", quantity=1.0)])

    assert len(results) == 1
    events = [event["event"] for event in journal.export()]
    assert "order_partially_executed" in events
    assert "order_executed" not in events
    execution_message = next(
        message for message in channel.messages if message.category == "execution"
    )
    assert execution_message.severity == "info"
    assert "częściowo" in execution_message.title.lower()


def test_controller_partial_result_without_execution_fill_data_does_not_fallback_to_request() -> (
    None
):
    risk_engine = DummyRiskEngine()
    execution = StatusExecutionService(
        status="partially_filled", filled_quantity=None, avg_price=None
    )
    router, _channel, _audit = _router_with_channel()
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

    controller.process_signals([_signal("BUY", quantity=3.0, price=321.0)])

    partial_event = next(
        event for event in journal.export() if event["event"] == "order_partially_executed"
    )
    assert partial_event["status"] == "partially_filled"
    assert partial_event["filled_quantity"] == "null"
    assert partial_event["avg_price"] == "null"
    assert partial_event["filled_quantity"] != "3.00000000"
    assert partial_event["avg_price"] != "321.00000000"


def test_controller_partial_result_without_filled_quantity_argument_falls_back_to_request() -> None:
    risk_engine = DummyRiskEngine()
    execution = StatusExecutionService(status="partially_filled", avg_price=None)
    router, _channel, _audit = _router_with_channel()
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

    controller.process_signals([_signal("BUY", quantity=3.0, price=321.0)])

    partial_event = next(
        event for event in journal.export() if event["event"] == "order_partially_executed"
    )
    assert partial_event["status"] == "partially_filled"
    assert partial_event["filled_quantity"] == "3.00000000"
    assert partial_event["avg_price"] == "null"


def test_controller_skips_neutral_signal() -> None:
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, channel, audit = _router_with_channel()
    journal = CollectingDecisionJournal()
    registry = MetricsRegistry()
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
        metrics_registry=registry,
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
    signals_counter = registry.counter(
        "trading_signals_total",
        "Liczba sygnałów przetworzonych w TradingController (status=received/accepted/rejected/adjusted/neutral/skipped).",
    )
    signal_labels = {
        "environment": "paper",
        "portfolio": "paper-1",
        "risk_profile": "balanced",
        "symbol": "BTC/USDT",
    }
    assert signals_counter.value(labels={**signal_labels, "status": "received"}) == 1.0
    assert signals_counter.value(labels={**signal_labels, "status": "neutral"}) == 1.0

    orders_counter = registry.counter(
        "trading_orders_total",
        "Liczba zleceń obsłużonych przez TradingController (result=submitted/executed/failed).",
    )
    order_labels = {**signal_labels, "side": "SELL"}
    assert orders_counter.value(labels={**order_labels, "result": "submitted"}) == 0.0
    assert orders_counter.value(labels={**order_labels, "result": "executed"}) == 0.0
    assert orders_counter.value(labels={**order_labels, "result": "not_filled"}) == 0.0


def test_controller_skips_unsupported_side_with_terminal_metric_and_without_side_effects() -> None:
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, channel, audit = _router_with_channel()
    journal = CollectingDecisionJournal()
    registry = MetricsRegistry()
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
        metrics_registry=registry,
    )

    signal = StrategySignal(
        symbol="BTC/USDT",
        side="ROTATE",
        confidence=0.41,
        metadata={"order_type": "market"},
    )

    results = controller.process_signals([signal])

    assert results == []
    assert execution.requests == []
    assert all(message.category != "execution" for message in channel.messages)
    assert tuple(audit.export()) == ()
    skipped_event = next(event for event in journal.export() if event["event"] == "signal_skipped")
    assert skipped_event["status"] == "skipped"
    assert skipped_event["reason"] == "unsupported_side"

    signals_counter = registry.counter(
        "trading_signals_total",
        "Liczba sygnałów przetworzonych w TradingController (status=received/accepted/rejected/adjusted/neutral/skipped).",
    )
    signal_labels = {
        "environment": "paper",
        "portfolio": "paper-1",
        "risk_profile": "balanced",
        "symbol": "BTC/USDT",
    }
    assert signals_counter.value(labels={**signal_labels, "status": "received"}) == 1.0
    assert signals_counter.value(labels={**signal_labels, "status": "skipped"}) == 1.0

    orders_counter = registry.counter(
        "trading_orders_total",
        "Liczba zleceń obsłużonych przez TradingController (result=submitted/executed/failed).",
    )
    for side in ("BUY", "SELL"):
        order_labels = {**signal_labels, "side": side}
        assert orders_counter.value(labels={**order_labels, "result": "submitted"}) == 0.0
        assert orders_counter.value(labels={**order_labels, "result": "executed"}) == 0.0
        assert orders_counter.value(labels={**order_labels, "result": "not_filled"}) == 0.0


def test_controller_skips_no_valid_legs_with_terminal_metric_and_without_side_effects() -> None:
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, channel, audit = _router_with_channel()
    journal = CollectingDecisionJournal()
    registry = MetricsRegistry()
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
        metrics_registry=registry,
    )

    signal = StrategySignal(
        symbol="BTC/USDT",
        side="arbitrage_entry",
        confidence=0.63,
        intent="multi_leg",
        metadata={"order_type": "market"},
        legs=(
            SignalLeg(
                symbol="BTC/USDT",
                side="HOLD",
                quantity=1.0,
                metadata={"price": 100.0},
            ),
            SignalLeg(
                symbol="BTC/USDT",
                side="BUY",
                quantity=0.0,
                metadata={"price": 101.0},
            ),
        ),
    )

    results = controller.process_signals([signal])

    assert results == []
    assert execution.requests == []
    assert all(message.category != "execution" for message in channel.messages)
    assert tuple(audit.export()) == ()
    skipped_event = next(event for event in journal.export() if event["event"] == "signal_skipped")
    assert skipped_event["status"] == "skipped"
    assert skipped_event["reason"] == "no_valid_legs"

    signals_counter = registry.counter(
        "trading_signals_total",
        "Liczba sygnałów przetworzonych w TradingController (status=received/accepted/rejected/adjusted/neutral/skipped).",
    )
    signal_labels = {
        "environment": "paper",
        "portfolio": "paper-1",
        "risk_profile": "balanced",
        "symbol": "BTC/USDT",
    }
    assert signals_counter.value(labels={**signal_labels, "status": "received"}) == 1.0
    assert signals_counter.value(labels={**signal_labels, "status": "skipped"}) == 1.0

    orders_counter = registry.counter(
        "trading_orders_total",
        "Liczba zleceń obsłużonych przez TradingController (result=submitted/executed/failed).",
    )
    for side in ("BUY", "SELL"):
        order_labels = {**signal_labels, "side": side}
        assert orders_counter.value(labels={**order_labels, "result": "submitted"}) == 0.0
        assert orders_counter.value(labels={**order_labels, "result": "executed"}) == 0.0
        assert orders_counter.value(labels={**order_labels, "result": "not_filled"}) == 0.0


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
    close_event = next(
        event for event in journal_events if event["event"] == "order_close_for_reversal"
    )
    # client_order_id is a generated identifier, while metadata bools are JSON-serialized.
    assert close_event.get("client_order_id")
    assert close_event.get("order_generated_client_order_id") == "true"
    close_risk_event = next(
        event for event in journal_events if event["event"] == "reversal_close_risk_check"
    )
    assert close_risk_event.get("status") == "allowed"
    assert close_risk_event.get("order_is_reducing") == "true"
    assert close_risk_event.get("order_reducing_only") == "true"


def test_controller_reversal_is_disabled_by_default() -> None:
    registry = MetricsRegistry()
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
        metrics_registry=registry,
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
        },
    )

    results = controller.process_signals([signal])

    assert len(results) == 1
    assert len(execution.requests) == 1
    assert execution.requests[0].metadata.get("action") != "close"
    events = [event["event"] for event in journal.export()]
    assert "order_close_for_reversal" not in events

    labels = {
        "environment": "paper",
        "portfolio": "paper-1",
        "risk_profile": "balanced",
        "symbol": "BTC/USDT",
        "reason": "disabled",
    }
    skipped_counter = registry.get("trading_reversal_skipped_total")
    assert skipped_counter.value(labels=labels) == 1.0


def test_controller_logs_untrusted_position_when_reversal_requested() -> None:
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
        side="SELL",
        confidence=0.9,
        metadata={
            "quantity": "2",
            "price": "101",
            "order_type": "market",
            "current_position_qty": "bad-value",
            "current_position_side": "LONG",
            "reverse_position": "true",
        },
    )

    results = controller.process_signals([signal])

    assert len(results) == 1
    assert len(execution.requests) == 1
    events = journal.export()
    skipped = next(
        event for event in events if event["event"] == "reversal_skipped_untrusted_position"
    )
    assert skipped["status"] == "skipped"


def test_controller_reversal_close_can_be_denied_by_risk() -> None:
    registry = MetricsRegistry()
    risk_engine = DummyRiskEngine()
    risk_engine.set_result_sequence(
        [RiskCheckResult(allowed=True), RiskCheckResult(allowed=False, reason="close blocked")]
    )
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
        metrics_registry=registry,
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
    assert len(execution.requests) == 1
    events = journal.export()
    close_risk_event = next(
        event for event in events if event["event"] == "reversal_close_risk_check"
    )
    assert close_risk_event["status"] == "rejected"
    denied_event = next(event for event in events if event["event"] == "reversal_denied_by_risk")
    assert denied_event["status"] == "rejected"

    labels = {
        "environment": "paper",
        "portfolio": "paper-1",
        "risk_profile": "balanced",
        "symbol": "BTC/USDT",
        "side": "SELL",
    }
    denied_counter = registry.get("trading_reversal_denied_by_risk_total")
    assert denied_counter.value(labels=labels) == 1.0


@pytest.mark.parametrize("close_status", ["rejected", "partially_filled", "canceled"])
def test_controller_does_not_open_reversal_when_close_order_is_not_filled(
    close_status: str,
) -> None:
    class CloseRejectedExecutionService(ExecutionService):
        def __init__(self, *, close_status: str) -> None:
            self._close_status = close_status
            self._is_close_filled = close_status in {"filled", "executed", "complete", "completed"}
            self.requests: list[OrderRequest] = []

        def execute(self, request: OrderRequest, context) -> OrderResult:  # type: ignore[override]
            self.requests.append(request)
            status = self._close_status if request.metadata.get("action") == "close" else "filled"
            return OrderResult(
                order_id=(
                    f"close-{status}-1" if request.metadata.get("action") == "close" else "open-1"
                ),
                status=status,
                filled_quantity=0.0 if not self._is_close_filled else request.quantity,
                avg_price=None if not self._is_close_filled else request.price,
                raw_response={"context": context.metadata},
            )

        def cancel(self, order_id: str, context) -> None:  # type: ignore[override]
            return None

        def flush(self) -> None:
            return None

    risk_engine = DummyRiskEngine()
    execution = CloseRejectedExecutionService(close_status=close_status)
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

    assert results == []
    assert len(execution.requests) == 1
    assert execution.requests[0].metadata.get("action") == "close"
    close_events = [
        event for event in journal.export() if event["event"] == "order_close_for_reversal"
    ]
    assert close_events
    assert close_events[-1]["status"] == close_status
    assert "order_submitted" in [event["event"] for event in journal.export()]
    assert "order_executed" not in [event["event"] for event in journal.export()]


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


def test_controller_skips_tco_for_partial_when_execution_data_missing() -> None:
    risk_engine = DummyRiskEngine()
    execution = StatusExecutionService(
        status="partially_filled", filled_quantity=None, avg_price=None
    )
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
    )

    controller.process_signals([_signal("BUY", quantity=3.0, price=321.0)])

    events = [event["event"] for event in journal.export()]
    assert "order_partially_executed" in events
    assert reporter.calls == []


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
    assert any(
        entry["channel"] == "collector" and entry["category"] == "strategy" for entry in exported
    )
    assert channel.messages[-1].category == "execution"
    events = [event["event"] for event in journal.export()]
    assert "risk_adjusted" in events


def test_controller_syncs_metadata_quantity_after_risk_adjustment() -> None:
    risk_engine = DummyRiskEngine()
    disallowed = RiskCheckResult(
        allowed=False,
        reason="Limit ekspozycji przekroczony",
        adjustments={"max_quantity": 0.25},
    )
    allowed = RiskCheckResult(allowed=True)
    risk_engine.set_result_sequence([disallowed, allowed])

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

    controller.process_signals([_signal("BUY", quantity=1.0, price=100.0)])

    assert execution.requests
    adjusted_request = execution.requests[0]
    assert adjusted_request.quantity == pytest.approx(0.25)
    assert float(adjusted_request.metadata["quantity"]) == pytest.approx(adjusted_request.quantity)

    exported = list(journal.export())
    risk_adjusted = next(event for event in exported if event["event"] == "risk_adjusted")
    submitted = next(event for event in exported if event["event"] == "order_submitted")
    for event in (risk_adjusted, submitted):
        assert float(event["quantity"]) == pytest.approx(0.25)
        assert float(event["order_quantity"]) == pytest.approx(0.25)


def test_controller_uses_recheck_reason_when_adjusted_order_is_rejected() -> None:
    risk_engine = DummyRiskEngine()
    first_reject = RiskCheckResult(
        allowed=False,
        reason="A",
        adjustments={"max_quantity": 0.25},
    )
    second_reject = RiskCheckResult(allowed=False, reason="B")
    risk_engine.set_result_sequence([first_reject, second_reject])

    execution = DummyExecutionService()
    router, channel, _ = _router_with_channel()
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

    assert results == []
    assert execution.requests == []
    risk_alert = next(message for message in channel.messages if message.category == "risk")
    assert risk_alert.body == "B"
    exported = list(journal.export())
    rejected = next(event for event in exported if event["event"] == "risk_rejected")
    assert rejected["reason"] == "B"


def test_controller_risk_adjust_preserves_existing_client_order_id() -> None:
    risk_engine = DummyRiskEngine()
    disallowed = RiskCheckResult(
        allowed=False,
        reason="Limit ekspozycji przekroczony",
        adjustments={"max_quantity": 0.25},
    )
    allowed = RiskCheckResult(allowed=True)
    risk_engine.set_result_sequence([disallowed, allowed])

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

    existing_client_order_id = "existing-client-id-123"
    signal = StrategySignal(
        symbol="BTC/USDT",
        side="BUY",
        confidence=0.75,
        metadata={
            "quantity": "1.0",
            "price": "100.0",
            "order_type": "market",
            "client_order_id": existing_client_order_id,
        },
    )

    controller.process_signals([signal])

    assert len(execution.requests) == 1
    adjusted_request = execution.requests[0]
    assert adjusted_request.quantity == pytest.approx(0.25)
    assert adjusted_request.client_order_id == existing_client_order_id
    assert adjusted_request.metadata is not None
    assert adjusted_request.metadata.get("generated_client_order_id") is not True

    exported = list(journal.export())
    risk_adjusted = next(event for event in exported if event["event"] == "risk_adjusted")
    submitted = next(event for event in exported if event["event"] == "order_submitted")
    for event in (risk_adjusted, submitted):
        assert event["client_order_id"] == existing_client_order_id
        assert event.get("order_client_order_id") == existing_client_order_id
        assert "order_generated_client_order_id" not in event


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


def test_record_decision_event_preserves_structured_metadata_for_serializer() -> None:
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

    controller._record_decision_event(  # pyright: ignore[reportPrivateUsage]
        "decision_evaluation",
        status="accepted",
        metadata={
            "structured": {"z": 2, "a": 1},
            "items": ["x", {"k": 1}],
            "flag": True,
            "missing": None,
        },
    )

    exported = journal.export()
    assert len(exported) == 1
    event = exported[0]
    assert event["structured"] == '{"a":1,"z":2}'
    assert event["items"] == '["x",{"k":1}]'
    assert event["flag"] == "true"
    assert event["missing"] == "null"


def test_record_decision_event_preserves_structured_signal_and_order_metadata() -> None:
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
            "structured": {"z": 2, "a": 1},
            "items": ["x", {"k": 1}],
        },
    )
    request = OrderRequest(
        symbol="BTC/USDT",
        side="BUY",
        quantity=1.0,
        order_type="market",
        price=100.0,
        metadata={
            "structured": {"beta": 2, "alpha": 1},
            "items": ["a", {"b": 2}],
            "flag": True,
            "missing": None,
        },
    )

    controller._record_decision_event(  # pyright: ignore[reportPrivateUsage]
        "order_submitted",
        signal=signal,
        request=request,
        status="submitted",
    )

    exported = journal.export()
    assert len(exported) == 1
    event = exported[0]
    assert event["signal_structured"] == '{"a":1,"z":2}'
    assert event["signal_items"] == '["x",{"k":1}]'
    assert event["order_structured"] == '{"alpha":1,"beta":2}'
    assert event["order_items"] == '["a",{"b":2}]'
    assert event["order_flag"] == "true"
    assert event["order_missing"] == "null"


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
        "Liczba sygnałów przetworzonych w TradingController (status=received/accepted/rejected/adjusted/neutral/skipped).",
    )
    assert signals_counter.value(labels={**symbol_labels, "status": "received"}) == 1.0
    assert signals_counter.value(labels={**symbol_labels, "status": "accepted"}) == 1.0
    assert signals_counter.value(labels={**symbol_labels, "status": "rejected"}) == 0.0

    orders_counter = registry.counter(
        "trading_orders_total",
        "Liczba zleceń obsłużonych przez TradingController (result=submitted/executed/failed).",
    )
    assert (
        orders_counter.value(labels={**symbol_labels, "result": "submitted", "side": "BUY"}) == 1.0
    )
    assert (
        orders_counter.value(labels={**symbol_labels, "result": "executed", "side": "BUY"}) == 1.0
    )
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


def test_liquidation_state_transition_events_are_emitted_once_per_state_change() -> None:
    registry = MetricsRegistry()
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
        metrics_registry=registry,
        decision_journal=journal,
    )

    base_labels = {"environment": "paper", "portfolio": "paper-1", "risk_profile": "balanced"}
    liquidation_gauge = registry.gauge(
        "trading_liquidation_state",
        "Stan trybu awaryjnego profilu ryzyka (1=liquidation, 0=normal).",
    )

    controller.process_signals([_signal("BUY")])
    transition_events = [e for e in journal.events if e.event_type == "liquidation_state_changed"]
    assert transition_events == []
    assert liquidation_gauge.value(labels=base_labels) == 0.0

    risk_engine.set_result(
        RiskCheckResult(allowed=False, reason="Przekroczono dzienny limit straty."),
        liquidate=True,
    )
    controller.process_signals([_signal("SELL")])
    transition_events = [e for e in journal.events if e.event_type == "liquidation_state_changed"]
    assert len(transition_events) == 1
    assert transition_events[0].status == "entered"
    assert transition_events[0].metadata["in_liquidation"] is True
    assert liquidation_gauge.value(labels=base_labels) == 1.0

    controller.process_signals([_signal("SELL")])
    transition_events = [e for e in journal.events if e.event_type == "liquidation_state_changed"]
    assert len(transition_events) == 1

    risk_engine.set_result(RiskCheckResult(allowed=True), liquidate=False)
    controller.process_signals([_signal("BUY")])
    transition_events = [e for e in journal.events if e.event_type == "liquidation_state_changed"]
    assert len(transition_events) == 2
    assert transition_events[1].status == "exited"
    assert transition_events[1].metadata["in_liquidation"] is False
    assert liquidation_gauge.value(labels=base_labels) == 0.0


def test_liquidation_critical_alert_is_idempotent_while_state_stays_active() -> None:
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
    )

    controller.process_signals([_signal("SELL")])
    controller.process_signals([_signal("SELL")])

    critical_risk_alerts = [
        msg
        for msg in channel.messages
        if msg.category == "risk"
        and msg.severity == "critical"
        and msg.title == "Profil w trybie awaryjnym"
    ]
    warning_risk_alerts = [
        msg for msg in channel.messages if msg.category == "risk" and msg.severity == "warning"
    ]
    info_strategy_alerts = [
        msg for msg in channel.messages if msg.category == "strategy" and msg.severity == "info"
    ]

    assert len(critical_risk_alerts) == 1
    assert len(warning_risk_alerts) == 2
    assert len(info_strategy_alerts) == 2
    assert len(channel.messages) == 5


class FailingExecutionService(ExecutionService):
    def __init__(self) -> None:
        self.requests: list[OrderRequest] = []

    def execute(self, request: OrderRequest, context) -> OrderResult:  # type: ignore[override]
        self.requests.append(request)
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
    execution_alert = channel.messages[1]
    assert len(execution.requests) == 1
    failed_request = execution.requests[0]
    assert failed_request.client_order_id
    assert execution_alert.context["client_order_id"] == failed_request.client_order_id
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
    decision_events = [
        event for event in journal.events if event.event_type == "decision_evaluation"
    ]
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

        def evaluate_candidate(self, candidate, _context):
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
    decision_events = [
        event for event in journal.events if event.event_type == "decision_evaluation"
    ]
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

        def evaluate_candidate(self, candidate, _context):
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
    decision_events = [
        event for event in journal.events if event.event_type == "decision_evaluation"
    ]
    assert any(event.status == "accepted" for event in decision_events)


def test_controller_extra_metadata_does_not_override_request_precedence_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, _channel, _audit = _router_with_channel()
    journal = CollectingDecisionJournal()

    class _AcceptingOrchestrator:
        def evaluate_candidate(self, candidate, _context):
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

    controller = TradingController(
        risk_engine=risk_engine,
        execution_service=execution,
        alert_router=router,
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        decision_orchestrator=_AcceptingOrchestrator(),
        decision_min_probability=0.4,
        decision_journal=journal,
    )

    def _colliding_decision_metadata(_self, _evaluation) -> Mapping[str, object]:
        return {
            "order_type": "limit",
            "time_in_force": "IOC",
            "client_order_id": "extra-cid-override",
            "exchange": "EXTRA-EXCHANGE",
            "decision_engine": {"accepted": True, "model": "gbm_v2"},
        }

    monkeypatch.setattr(
        TradingController,
        "_serialize_decision_evaluation",
        _colliding_decision_metadata,
    )

    signal = _signal("BUY", quantity=1.0, price=100.0, confidence=0.8)
    signal.metadata = {
        "quantity": "1.0",
        "price": "100.0",
        "order_type": "market",
        "time_in_force": "GTC",
        "client_order_id": "signal-cid-123",
        "exchange": "BINANCE",
        "expected_probability": 0.8,
        "expected_return_bps": 15.0,
    }

    results = controller.process_signals([signal])

    assert len(results) == 1
    assert len(execution.requests) == 1
    request = execution.requests[0]
    assert request.order_type == "MARKET"
    assert request.time_in_force == "GTC"
    assert request.client_order_id == "signal-cid-123"
    assert request.metadata["exchange"] == "BINANCE"
    assert request.metadata["decision_engine"]["model"] == "gbm_v2"

    submitted_event = next(
        event for event in journal.export() if event["event"] == "order_submitted"
    )
    assert submitted_event.get("order_type") == "MARKET"
    assert submitted_event.get("time_in_force") == "GTC"
    assert submitted_event.get("client_order_id") == "signal-cid-123"
    assert submitted_event.get("order_exchange") == "BINANCE"


def test_controller_generates_client_order_id_when_missing() -> None:
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, _channel, _audit = _router_with_channel()
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

    assert len(execution.requests) == 1
    request = execution.requests[0]
    assert request.client_order_id
    assert request.metadata is not None
    assert request.metadata.get("generated_client_order_id") is True

    order_submitted_event = next(
        event for event in journal.export() if event["event"] == "order_submitted"
    )
    assert order_submitted_event.get("client_order_id") == request.client_order_id
    assert order_submitted_event.get("order_generated_client_order_id") == "true"


def test_controller_normalizes_trimmed_client_order_id_and_time_in_force() -> None:
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, _channel, _audit = _router_with_channel()
    journal = CollectingDecisionJournal()
    controller = TradingController(
        risk_engine=risk_engine,
        execution_service=execution,
        alert_router=router,
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        decision_journal=journal,
    )

    signal = _signal("BUY", quantity=1.0, price=100.0)
    signal.metadata = {
        "quantity": "1.0",
        "price": "100.0",
        "order_type": "market",
        "client_order_id": " abc ",
        "time_in_force": " GTC ",
    }

    controller.process_signals([signal])

    assert len(execution.requests) == 1
    request = execution.requests[0]
    assert request.client_order_id == "abc"
    assert request.time_in_force == "GTC"

    submitted_event = next(
        event for event in journal.export() if event["event"] == "order_submitted"
    )
    assert submitted_event.get("client_order_id") == "abc"
    assert submitted_event.get("time_in_force") == "GTC"


def test_controller_treats_whitespace_string_fields_as_missing() -> None:
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, _channel, _audit = _router_with_channel()
    journal = CollectingDecisionJournal()
    controller = TradingController(
        risk_engine=risk_engine,
        execution_service=execution,
        alert_router=router,
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        decision_journal=journal,
    )

    signal = _signal("BUY", quantity=1.0, price=100.0)
    signal.metadata = {
        "quantity": "1.0",
        "price": "100.0",
        "order_type": "market",
        "client_order_id": "   ",
        "time_in_force": "   ",
    }

    controller.process_signals([signal])

    assert len(execution.requests) == 1
    request = execution.requests[0]
    assert request.client_order_id
    assert request.client_order_id.startswith("tc-")
    assert request.time_in_force is None
    assert request.metadata is not None
    assert request.metadata.get("generated_client_order_id") is True

    submitted_event = next(
        event for event in journal.export() if event["event"] == "order_submitted"
    )
    assert submitted_event.get("client_order_id") == request.client_order_id
    assert submitted_event.get("order_generated_client_order_id") == "true"
    assert "time_in_force" not in submitted_event


def test_controller_normalizes_optional_numeric_fields_and_journal_payload() -> None:
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, _channel, _audit = _router_with_channel()
    journal = CollectingDecisionJournal()
    controller = TradingController(
        risk_engine=risk_engine,
        execution_service=execution,
        alert_router=router,
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        decision_journal=journal,
    )

    signal = _signal("BUY", quantity=1.0, price=100.0)
    signal.metadata = {
        "quantity": "1.0",
        "price": "   ",
        "stop_price": "",
        "atr": "   ",
        "order_type": "market",
    }

    controller.process_signals([signal])

    assert len(execution.requests) == 1
    request = execution.requests[0]
    assert request.price is None
    assert request.stop_price is None
    assert request.atr is None

    assert request.metadata is not None
    assert "price" not in request.metadata
    assert "stop_price" not in request.metadata
    assert "atr" not in request.metadata

    submitted_event = next(
        event for event in journal.export() if event["event"] == "order_submitted"
    )
    assert "price" not in submitted_event
    assert "order_price" not in submitted_event
    assert "order_stop_price" not in submitted_event
    assert "order_atr" not in submitted_event


def test_controller_parses_optional_numeric_fields_to_float() -> None:
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, _channel, _audit = _router_with_channel()
    journal = CollectingDecisionJournal()
    controller = TradingController(
        risk_engine=risk_engine,
        execution_service=execution,
        alert_router=router,
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        decision_journal=journal,
    )

    signal = _signal("BUY", quantity=1.0, price=100.0)
    signal.metadata = {
        "quantity": "1.0",
        "price": "101.25",
        "stop_price": 99.5,
        "atr": "1.75",
        "order_type": "limit",
    }

    controller.process_signals([signal])

    assert len(execution.requests) == 1
    request = execution.requests[0]
    assert request.price == pytest.approx(101.25)
    assert request.stop_price == pytest.approx(99.5)
    assert request.atr == pytest.approx(1.75)

    submitted_event = next(
        event for event in journal.export() if event["event"] == "order_submitted"
    )
    assert submitted_event.get("price") == "101.25"
    assert submitted_event.get("order_price") == "101.25"
    assert submitted_event.get("order_stop_price") == "99.5"
    assert submitted_event.get("order_atr") == "1.75"


def test_controller_rejects_invalid_optional_numeric_field_value() -> None:
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, _channel, _audit = _router_with_channel()
    controller = TradingController(
        risk_engine=risk_engine,
        execution_service=execution,
        alert_router=router,
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
    )

    signal = _signal("BUY", quantity=1.0, price=100.0)
    signal.metadata = {
        "quantity": "1.0",
        "price": "not-a-number",
        "order_type": "market",
    }

    with pytest.raises(ValueError, match="price w metadanych musi być liczbą zmiennoprzecinkową"):
        controller.process_signals([signal])


@pytest.mark.parametrize(
    ("field_name", "expected_message"),
    [
        ("stop_price", "stop_price w metadanych musi być liczbą zmiennoprzecinkową"),
        ("atr", "atr w metadanych musi być liczbą zmiennoprzecinkową"),
    ],
)
def test_controller_rejects_invalid_stop_price_and_atr_values(
    field_name: str,
    expected_message: str,
) -> None:
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, _channel, _audit = _router_with_channel()
    controller = TradingController(
        risk_engine=risk_engine,
        execution_service=execution,
        alert_router=router,
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
    )

    signal = _signal("BUY", quantity=1.0, price=100.0)
    signal.metadata = {
        "quantity": "1.0",
        "price": "100.0",
        "order_type": "market",
        field_name: "invalid-value",
    }

    with pytest.raises(ValueError, match=expected_message):
        controller.process_signals([signal])


def test_process_signals_survives_health_dispatch_exception(caplog) -> None:
    risk_engine = DummyRiskEngine()
    execution = DummyExecutionService()
    router, _channel, _audit = _router_with_channel()
    initial_now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    current_now = initial_now

    def clock() -> datetime:
        return current_now

    controller = TradingController(
        risk_engine=risk_engine,
        execution_service=execution,
        alert_router=router,
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        health_check_interval=timedelta(seconds=1),
        clock=clock,
    )
    current_now = initial_now + timedelta(seconds=2)
    original_dispatch = router.dispatch

    def flaky_dispatch(message):
        if getattr(message, "category", "") == "health":
            raise RuntimeError("health dispatch boom")
        return original_dispatch(message)

    router.dispatch = flaky_dispatch  # type: ignore[assignment]

    with caplog.at_level(logging.ERROR):
        results = controller.process_signals([_signal("BUY")])

    assert len(results) == 1
    assert len(execution.requests) == 1
    assert "health-check" in caplog.text.lower()
