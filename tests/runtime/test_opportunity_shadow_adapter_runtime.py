from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from bot_core.ai.opportunity_shadow_adapter import OpportunityRuntimeShadowAdapter
from bot_core.ai.repository import FilesystemModelRepository
from bot_core.ai.trading_engine import OpportunitySnapshot, TradingOpportunityAI
from bot_core.ai.trading_opportunity_shadow import OpportunityShadowRepository
from bot_core.runtime.journal import TradingDecisionEvent
from bot_core.runtime.opportunity_runtime_controls import OpportunityRuntimeControls
from bot_core.runtime.pipeline import DecisionAwareSignalSink, InMemoryStrategySignalSink
from bot_core.strategies.base import StrategySignal


class _CollectingJournal:
    def __init__(self) -> None:
        self.events: list[TradingDecisionEvent] = []

    def record(self, event: TradingDecisionEvent) -> None:
        self.events.append(event)


class _AcceptingOrchestrator:
    def evaluate_candidate(self, candidate, _context):
        return SimpleNamespace(accepted=True, reasons=(), risk_flags=(), stress_failures=(), latency_ms=None)


class _RejectingOrchestrator:
    def evaluate_candidate(self, candidate, _context):
        return SimpleNamespace(accepted=False, reasons=(), risk_flags=(), stress_failures=(), latency_ms=None)


class _RiskEngine:
    def snapshot_state(self, _profile: str):
        return {}


class _PolicyAdapter:
    def __init__(
        self,
        result: OpportunityRuntimeShadowAdapter.PolicyProbeResult,
    ) -> None:
        self.result = result
        self.mode = result.mode

    def emit_shadow_proposal(self, **_kwargs):
        return self.result


def _train_model(repo_path: Path) -> TradingOpportunityAI:
    repo = FilesystemModelRepository(repo_path)
    engine = TradingOpportunityAI(repository=repo)
    samples = [
        OpportunitySnapshot(
            symbol="BTCUSDT",
            signal_strength=0.3 + idx * 0.1,
            momentum_5m=0.1 + idx * 0.01,
            volatility_30m=0.2 + idx * 0.01,
            spread_bps=2.0,
            fee_bps=1.0,
            slippage_bps=0.5,
            liquidity_score=0.7,
            risk_penalty_bps=0.2,
            realized_return_bps=8.0 + idx,
            as_of=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        )
        for idx in range(12)
    ]
    engine.fit(samples)
    engine.save_model(version="shadow-v1", activate=True)
    return engine


def _candidate_metadata() -> dict[str, object]:
    return {
        "signal_strength": 0.8,
        "momentum_5m": 0.2,
        "volatility_30m": 0.3,
        "spread_bps": 2.0,
        "fee_bps": 1.0,
        "slippage_bps": 0.5,
        "liquidity_score": 0.85,
        "risk_penalty_bps": 0.2,
        "as_of": "2024-01-01T12:00:00Z",
    }


def test_shadow_adapter_emits_proposal(tmp_path: Path) -> None:
    journal = _CollectingJournal()
    engine = _train_model(tmp_path / "repo")
    adapter = OpportunityRuntimeShadowAdapter(journal=journal, engine=engine)

    adapter.emit_shadow_proposal(
        candidate=SimpleNamespace(symbol="BTCUSDT", action="enter", metadata=_candidate_metadata()),
        signal=SimpleNamespace(side="BUY"),
        evaluation=SimpleNamespace(accepted=True),
        timestamp=datetime(2024, 1, 1, 12, 5, tzinfo=timezone.utc),
        strategy_name="trend-d1",
        schedule_name="trend-d1",
        risk_profile="balanced",
        environment="paper",
        portfolio="paper-main",
    )

    assert journal.events
    event = journal.events[-1]
    assert event.event_type == "opportunity_shadow_proposal"
    assert event.metadata["decision_source"] == "opportunity_ai_shadow"
    assert event.metadata["mode"] == "shadow"
    assert event.metadata["shadow"] == "true"
    assert str(event.metadata["model_version"])


def test_shadow_adapter_gracefully_degrades_without_model(tmp_path: Path) -> None:
    journal = _CollectingJournal()
    empty_repo = FilesystemModelRepository(tmp_path / "empty-repo")
    adapter = OpportunityRuntimeShadowAdapter(
        journal=journal,
        engine=TradingOpportunityAI(repository=empty_repo),
    )

    adapter.emit_shadow_proposal(
        candidate=SimpleNamespace(symbol="BTCUSDT", action="enter", metadata=_candidate_metadata()),
        signal=SimpleNamespace(side="BUY"),
        evaluation=SimpleNamespace(accepted=True),
        timestamp=datetime(2024, 1, 1, 12, 5, tzinfo=timezone.utc),
        strategy_name="trend-d1",
        schedule_name="trend-d1",
        risk_profile="balanced",
        environment="paper",
        portfolio="paper-main",
    )

    assert journal.events
    event = journal.events[-1]
    assert event.status == "degraded"
    assert event.metadata["degraded"] == "true"


def test_shadow_adapter_persists_shadow_record_when_repository_is_configured(tmp_path: Path) -> None:
    journal = _CollectingJournal()
    shadow_repository = OpportunityShadowRepository(tmp_path / "shadow")
    engine = _train_model(tmp_path / "repo")
    adapter = OpportunityRuntimeShadowAdapter(
        journal=journal,
        engine=engine,
        shadow_repository=shadow_repository,
    )

    adapter.emit_shadow_proposal(
        candidate=SimpleNamespace(symbol="BTCUSDT", action="enter", metadata=_candidate_metadata()),
        signal=SimpleNamespace(side="BUY"),
        evaluation=SimpleNamespace(accepted=True),
        timestamp=datetime(2024, 1, 1, 12, 5, tzinfo=timezone.utc),
        strategy_name="trend-d1",
        schedule_name="trend-d1",
        risk_profile="balanced",
        environment="paper",
        portfolio="paper-main",
    )

    records = shadow_repository.load_shadow_records()
    assert len(records) == 1
    assert records[0].model_version
    assert journal.events[-1].metadata["shadow_persistence_status"] == "persisted"
    assert journal.events[-1].metadata["shadow_record_key"] == records[0].record_key


def test_shadow_adapter_reports_persistence_failure_without_runtime_break(tmp_path: Path) -> None:
    class _BrokenShadowRepo:
        def append_shadow_records(self, _records):
            raise OSError("disk full")

    journal = _CollectingJournal()
    engine = _train_model(tmp_path / "repo")
    adapter = OpportunityRuntimeShadowAdapter(
        journal=journal,
        engine=engine,
        shadow_repository=_BrokenShadowRepo(),  # type: ignore[arg-type]
    )

    result = adapter.emit_shadow_proposal(
        candidate=SimpleNamespace(symbol="BTCUSDT", action="enter", metadata=_candidate_metadata()),
        signal=SimpleNamespace(side="BUY"),
        evaluation=SimpleNamespace(accepted=True),
        timestamp=datetime(2024, 1, 1, 12, 5, tzinfo=timezone.utc),
        strategy_name="trend-d1",
        schedule_name="trend-d1",
        risk_profile="balanced",
        environment="paper",
        portfolio="paper-main",
    )

    assert result.status == "proposal"
    assert result.shadow_persistence_status == "error"
    assert result.shadow_persistence_error == "disk full"
    degraded = [event for event in journal.events if event.status == "degraded"]
    assert degraded


def test_shadow_adapter_recovers_after_model_becomes_available(tmp_path: Path) -> None:
    journal = _CollectingJournal()
    repo_path = tmp_path / "recoverable-repo"
    repo = FilesystemModelRepository(repo_path)
    adapter = OpportunityRuntimeShadowAdapter(
        journal=journal,
        engine=TradingOpportunityAI(repository=repo),
        model_retry_cooldown_seconds=5.0,
        degraded_event_cooldown_seconds=30.0,
    )
    start = datetime(2024, 1, 1, 12, 5, tzinfo=timezone.utc)

    adapter.emit_shadow_proposal(
        candidate=SimpleNamespace(symbol="BTCUSDT", action="enter", metadata=_candidate_metadata()),
        signal=SimpleNamespace(side="BUY"),
        evaluation=SimpleNamespace(accepted=True),
        timestamp=start,
        strategy_name="trend-d1",
        schedule_name="trend-d1",
        risk_profile="balanced",
        environment="paper",
        portfolio="paper-main",
    )

    _train_model(repo_path)

    adapter.emit_shadow_proposal(
        candidate=SimpleNamespace(symbol="BTCUSDT", action="enter", metadata=_candidate_metadata()),
        signal=SimpleNamespace(side="BUY"),
        evaluation=SimpleNamespace(accepted=True),
        timestamp=start + timedelta(seconds=6),
        strategy_name="trend-d1",
        schedule_name="trend-d1",
        risk_profile="balanced",
        environment="paper",
        portfolio="paper-main",
    )

    degraded_events = [event for event in journal.events if event.status == "degraded"]
    proposal_events = [event for event in journal.events if event.status == "proposal"]
    assert len(degraded_events) == 1
    assert len(proposal_events) == 1


def test_shadow_adapter_throttles_degraded_event_spam(tmp_path: Path) -> None:
    journal = _CollectingJournal()
    adapter = OpportunityRuntimeShadowAdapter(
        journal=journal,
        engine=TradingOpportunityAI(repository=FilesystemModelRepository(tmp_path / "empty-repo-2")),
        model_retry_cooldown_seconds=0.0,
        degraded_event_cooldown_seconds=120.0,
    )
    base = datetime(2024, 1, 1, 12, 5, tzinfo=timezone.utc)

    for offset_seconds in (0, 1, 2, 3):
        adapter.emit_shadow_proposal(
            candidate=SimpleNamespace(symbol="BTCUSDT", action="enter", metadata=_candidate_metadata()),
            signal=SimpleNamespace(side="BUY"),
            evaluation=SimpleNamespace(accepted=True),
            timestamp=base + timedelta(seconds=offset_seconds),
            strategy_name="trend-d1",
            schedule_name="trend-d1",
            risk_profile="balanced",
            environment="paper",
            portfolio="paper-main",
        )

    degraded_events = [event for event in journal.events if event.status == "degraded"]
    assert len(degraded_events) == 1


def test_decision_sink_execution_path_unchanged_with_shadow_adapter(tmp_path: Path) -> None:
    journal = _CollectingJournal()
    shadow_repository = OpportunityShadowRepository(tmp_path / "shadow-runtime")
    adapter = OpportunityRuntimeShadowAdapter(
        journal=journal,
        engine=_train_model(tmp_path / "repo2"),
        shadow_repository=shadow_repository,
    )
    sink = DecisionAwareSignalSink(
        base_sink=InMemoryStrategySignalSink(),
        orchestrator=_AcceptingOrchestrator(),
        risk_engine=_RiskEngine(),
        default_notional=1000.0,
        environment="paper",
        exchange="BINANCE",
        min_probability=0.6,
        journal=journal,
        opportunity_shadow_adapter=adapter,
    )
    signal = StrategySignal(
        symbol="BTCUSDT",
        side="BUY",
        confidence=0.9,
        metadata={
            "expected_probability": 0.9,
            "expected_return_bps": 12.0,
            **_candidate_metadata(),
        },
    )

    sink.submit(
        strategy_name="trend-d1",
        schedule_name="trend-d1",
        risk_profile="balanced",
        timestamp=datetime(2024, 1, 1, 12, 5, tzinfo=timezone.utc),
        signals=(signal,),
    )

    exported = sink.export()
    assert len(exported) == 1
    assert len(exported[0][1]) == 1
    accepted_signal = exported[0][1][0]
    assert accepted_signal.metadata.get("opportunity_shadow_record_key")
    assert any(event.event_type == "decision_evaluation" for event in journal.events)
    assert any(event.event_type == "opportunity_shadow_proposal" for event in journal.events)


def test_decision_sink_without_shadow_adapter_still_works() -> None:
    sink = DecisionAwareSignalSink(
        base_sink=InMemoryStrategySignalSink(),
        orchestrator=_AcceptingOrchestrator(),
        risk_engine=_RiskEngine(),
        default_notional=1000.0,
        environment="paper",
        exchange="BINANCE",
        min_probability=0.6,
        journal=None,
        opportunity_shadow_adapter=None,
    )
    signal = StrategySignal(
        symbol="BTCUSDT",
        side="BUY",
        confidence=0.9,
        metadata={"expected_probability": 0.9, "expected_return_bps": 12.0},
    )

    sink.submit(
        strategy_name="trend-d1",
        schedule_name="trend-d1",
        risk_profile="balanced",
        timestamp=datetime(2024, 1, 1, 12, 5, tzinfo=timezone.utc),
        signals=(signal,),
    )

    exported = sink.export()
    assert len(exported) == 1
    assert len(exported[0][1]) == 1


def test_assist_mode_can_veto_base_acceptance() -> None:
    journal = _CollectingJournal()
    adapter = _PolicyAdapter(
        OpportunityRuntimeShadowAdapter.PolicyProbeResult(
            status="proposal",
            decision_available=True,
            accepted=False,
            model_version="assist-v1",
            decision_source="opportunity_ai_shadow",
            rejection_reason="probability_below_threshold",
            mode="assist",
        )
    )
    sink = DecisionAwareSignalSink(
        base_sink=InMemoryStrategySignalSink(),
        orchestrator=_AcceptingOrchestrator(),
        risk_engine=_RiskEngine(),
        default_notional=1000.0,
        environment="paper",
        exchange="BINANCE",
        min_probability=0.6,
        journal=journal,
        opportunity_shadow_adapter=adapter,
        opportunity_policy_mode="assist",
    )
    signal = StrategySignal(
        symbol="BTCUSDT",
        side="BUY",
        confidence=0.9,
        metadata={"expected_probability": 0.9, "expected_return_bps": 12.0},
    )

    sink.submit(
        strategy_name="trend-d1",
        schedule_name="trend-d1",
        risk_profile="balanced",
        timestamp=datetime(2024, 1, 1, 12, 5, tzinfo=timezone.utc),
        signals=(signal,),
    )
    assert sink.export() == ()
    event = journal.events[-1]
    assert event.metadata["opportunity_policy_mode"] == "assist"
    assert event.metadata["base_decision_accepted"] == "true"
    assert event.metadata["ai_decision_accepted"] == "false"
    assert event.metadata["final_decision_accepted"] == "false"
    assert event.metadata["ai_influenced_outcome"] == "true"
    assert event.metadata["decision_authority"] == "opportunity_ai_assist_policy"


def test_live_mode_degraded_keeps_safe_base_path() -> None:
    journal = _CollectingJournal()
    adapter = _PolicyAdapter(
        OpportunityRuntimeShadowAdapter.PolicyProbeResult(
            status="degraded",
            decision_available=False,
            accepted=None,
            degraded_reason="model_unavailable",
            mode="live",
        )
    )
    sink = DecisionAwareSignalSink(
        base_sink=InMemoryStrategySignalSink(),
        orchestrator=_AcceptingOrchestrator(),
        risk_engine=_RiskEngine(),
        default_notional=1000.0,
        environment="paper",
        exchange="BINANCE",
        min_probability=0.6,
        journal=journal,
        opportunity_shadow_adapter=adapter,
        opportunity_policy_mode="live",
    )
    signal = StrategySignal(
        symbol="BTCUSDT",
        side="BUY",
        confidence=0.9,
        metadata={"expected_probability": 0.9, "expected_return_bps": 12.0},
    )

    sink.submit(
        strategy_name="trend-d1",
        schedule_name="trend-d1",
        risk_profile="balanced",
        timestamp=datetime(2024, 1, 1, 12, 5, tzinfo=timezone.utc),
        signals=(signal,),
    )

    assert sink.export() == ()
    event = journal.events[-1]
    assert event.metadata["opportunity_policy_mode"] == "live"
    assert event.metadata["ai_decision_status"] == "degraded"
    assert event.metadata["ai_decision_available"] == "false"
    assert event.metadata["final_decision_accepted"] == "false"
    assert event.metadata["decision_authority"] == "opportunity_ai_live_fail_closed"
    assert event.metadata["live_gate_failed_closed"] == "true"
    assert event.metadata["opportunity_degraded_reason"] == "model_unavailable"


def test_live_mode_accepts_when_base_and_ai_accept() -> None:
    journal = _CollectingJournal()
    adapter = _PolicyAdapter(
        OpportunityRuntimeShadowAdapter.PolicyProbeResult(
            status="proposal",
            decision_available=True,
            accepted=True,
            model_version="live-v1",
            decision_source="opportunity_ai_shadow",
            mode="live",
        )
    )
    sink = DecisionAwareSignalSink(
        base_sink=InMemoryStrategySignalSink(),
        orchestrator=_AcceptingOrchestrator(),
        risk_engine=_RiskEngine(),
        default_notional=1000.0,
        environment="paper",
        exchange="BINANCE",
        min_probability=0.6,
        journal=journal,
        opportunity_shadow_adapter=adapter,
        opportunity_policy_mode="live",
    )
    signal = StrategySignal(
        symbol="BTCUSDT",
        side="BUY",
        confidence=0.9,
        metadata={"expected_probability": 0.9, "expected_return_bps": 12.0},
    )
    sink.submit(
        strategy_name="trend-d1",
        schedule_name="trend-d1",
        risk_profile="balanced",
        timestamp=datetime(2024, 1, 1, 12, 5, tzinfo=timezone.utc),
        signals=(signal,),
    )

    exported = sink.export()
    assert len(exported) == 1
    assert len(exported[0][1]) == 1
    event = journal.events[-1]
    assert event.metadata["final_decision_accepted"] == "true"
    assert event.metadata["ai_required_for_execution"] == "true"
    assert event.metadata["decision_authority"] == "shared_live_policy"


def test_live_mode_base_reject_cannot_be_overridden_by_ai() -> None:
    journal = _CollectingJournal()
    adapter = _PolicyAdapter(
        OpportunityRuntimeShadowAdapter.PolicyProbeResult(
            status="proposal",
            decision_available=True,
            accepted=True,
            model_version="live-v1",
            decision_source="opportunity_ai_shadow",
            mode="live",
        )
    )
    sink = DecisionAwareSignalSink(
        base_sink=InMemoryStrategySignalSink(),
        orchestrator=_RejectingOrchestrator(),
        risk_engine=_RiskEngine(),
        default_notional=1000.0,
        environment="paper",
        exchange="BINANCE",
        min_probability=0.6,
        journal=journal,
        opportunity_shadow_adapter=adapter,
        opportunity_policy_mode="live",
    )
    signal = StrategySignal(
        symbol="BTCUSDT",
        side="BUY",
        confidence=0.9,
        metadata={"expected_probability": 0.9, "expected_return_bps": 12.0},
    )
    sink.submit(
        strategy_name="trend-d1",
        schedule_name="trend-d1",
        risk_profile="balanced",
        timestamp=datetime(2024, 1, 1, 12, 5, tzinfo=timezone.utc),
        signals=(signal,),
    )

    assert sink.export() == ()
    event = journal.events[-1]
    assert event.metadata["base_decision_accepted"] == "false"
    assert event.metadata["ai_decision_accepted"] == "true"
    assert event.metadata["final_decision_accepted"] == "false"


def test_manual_kill_switch_disables_ai_and_restores_base_decision() -> None:
    journal = _CollectingJournal()
    adapter = _PolicyAdapter(
        OpportunityRuntimeShadowAdapter.PolicyProbeResult(
            status="proposal",
            decision_available=True,
            accepted=False,
            model_version="live-v1",
            decision_source="opportunity_ai_shadow",
            mode="live",
        )
    )
    sink = DecisionAwareSignalSink(
        base_sink=InMemoryStrategySignalSink(),
        orchestrator=_AcceptingOrchestrator(),
        risk_engine=_RiskEngine(),
        default_notional=1000.0,
        environment="paper",
        exchange="BINANCE",
        min_probability=0.6,
        journal=journal,
        opportunity_shadow_adapter=adapter,
        opportunity_policy_mode="live",
        opportunity_ai_kill_switch_override=True,
    )
    signal = StrategySignal(
        symbol="BTCUSDT",
        side="BUY",
        confidence=0.9,
        metadata={"expected_probability": 0.9, "expected_return_bps": 12.0},
    )
    sink.submit(
        strategy_name="trend-d1",
        schedule_name="trend-d1",
        risk_profile="balanced",
        timestamp=datetime(2024, 1, 1, 12, 5, tzinfo=timezone.utc),
        signals=(signal,),
    )

    exported = sink.export()
    assert len(exported) == 1
    event = journal.events[-1]
    assert event.metadata["final_decision_accepted"] == "true"
    assert event.metadata["opportunity_ai_enabled"] == "false"
    assert event.metadata["opportunity_ai_manual_kill_switch_active"] == "true"
    assert event.metadata["ai_required_for_execution"] == "false"
    assert event.metadata["opportunity_ai_disabled_reason"] == (
        "manual_kill_switch:DUDZIAN_OPPORTUNITY_AI_KILL_SWITCH"
    )


def test_runtime_enable_override_false_disables_ai_path() -> None:
    journal = _CollectingJournal()
    adapter = _PolicyAdapter(
        OpportunityRuntimeShadowAdapter.PolicyProbeResult(
            status="proposal",
            decision_available=True,
            accepted=False,
            model_version="assist-v1",
            decision_source="opportunity_ai_shadow",
            mode="assist",
        )
    )
    sink = DecisionAwareSignalSink(
        base_sink=InMemoryStrategySignalSink(),
        orchestrator=_AcceptingOrchestrator(),
        risk_engine=_RiskEngine(),
        default_notional=1000.0,
        environment="paper",
        exchange="BINANCE",
        min_probability=0.6,
        journal=journal,
        opportunity_shadow_adapter=adapter,
        opportunity_policy_mode="assist",
        opportunity_ai_enabled_override=False,
    )
    signal = StrategySignal(
        symbol="BTCUSDT",
        side="BUY",
        confidence=0.9,
        metadata={"expected_probability": 0.9, "expected_return_bps": 12.0},
    )
    sink.submit(
        strategy_name="trend-d1",
        schedule_name="trend-d1",
        risk_profile="balanced",
        timestamp=datetime(2024, 1, 1, 12, 5, tzinfo=timezone.utc),
        signals=(signal,),
    )
    exported = sink.export()
    assert len(exported) == 1
    event = journal.events[-1]
    assert event.metadata["opportunity_ai_enabled"] == "false"
    assert event.metadata["opportunity_ai_disabled_reason"] == (
        "runtime_override:DUDZIAN_OPPORTUNITY_AI_ENABLED"
    )


def test_runtime_control_plane_hot_updates_policy_without_restart() -> None:
    journal = _CollectingJournal()
    controls = OpportunityRuntimeControls(
        opportunity_ai_enabled=True,
        manual_kill_switch=False,
        policy_mode="assist",
    )
    adapter = _PolicyAdapter(
        OpportunityRuntimeShadowAdapter.PolicyProbeResult(
            status="proposal",
            decision_available=True,
            accepted=False,
            model_version="assist-v2",
            decision_source="opportunity_ai_shadow",
            mode="assist",
        )
    )
    sink = DecisionAwareSignalSink(
        base_sink=InMemoryStrategySignalSink(),
        orchestrator=_AcceptingOrchestrator(),
        risk_engine=_RiskEngine(),
        default_notional=1000.0,
        environment="paper",
        exchange="BINANCE",
        min_probability=0.6,
        journal=journal,
        opportunity_shadow_adapter=adapter,
        opportunity_policy_mode="shadow",
        opportunity_runtime_controls=controls,
    )
    signal = StrategySignal(
        symbol="BTCUSDT",
        side="BUY",
        confidence=0.9,
        metadata={"expected_probability": 0.9, "expected_return_bps": 12.0},
    )
    sink.submit(
        strategy_name="trend-d1",
        schedule_name="trend-d1",
        risk_profile="balanced",
        timestamp=datetime(2024, 1, 1, 12, 5, tzinfo=timezone.utc),
        signals=(signal,),
    )
    first_event = journal.events[-1]
    assert first_event.metadata["opportunity_policy_mode"] == "assist"
    assert first_event.metadata["final_decision_accepted"] == "false"

    controls.update(manual_kill_switch=True, policy_mode="live")
    sink.submit(
        strategy_name="trend-d1",
        schedule_name="trend-d1",
        risk_profile="balanced",
        timestamp=datetime(2024, 1, 1, 12, 6, tzinfo=timezone.utc),
        signals=(signal,),
    )
    second_event = journal.events[-1]
    assert second_event.metadata["opportunity_policy_mode"] == "live"
    assert second_event.metadata["opportunity_ai_enabled"] == "false"
    assert second_event.metadata["opportunity_ai_manual_kill_switch_active"] == "true"
    assert second_event.metadata["opportunity_ai_disabled_reason"] == "manual_kill_switch:runtime_control_plane"
