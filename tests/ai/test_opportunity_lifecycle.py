from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from bot_core.ai.opportunity_lifecycle import (
    OpportunityActivationReadiness,
    OpportunityAutonomyGateConfig,
    OpportunityAutonomyMode,
    OpportunityLifecycleService,
    OpportunityPersistedPromotionReadinessReport,
)
from bot_core.ai.opportunity_evaluation import (
    OpportunityDriftReport,
    OpportunityPromotionGateConfig,
    OpportunityPromotionGateResult,
    OpportunityPromotionReport,
    OpportunityTemporalComparison,
    OpportunityTemporalEvaluation,
)
from bot_core.ai.repository import FilesystemModelRepository
from bot_core.ai.trading_engine import (
    OpportunityDecision,
    OpportunitySnapshot,
    TradingOpportunityAI,
)
from bot_core.ai.trading_opportunity_shadow import (
    OpportunityOutcomeLabel,
    OpportunityShadowRepository,
)


def _build_samples(scale: float = 1.0) -> list[OpportunitySnapshot]:
    return [
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
            realized_return_bps=(8.0 + idx) * scale,
            as_of=datetime(2024, 1, 1, 12, idx, tzinfo=timezone.utc),
        )
        for idx in range(12)
    ]


def _shadow_decision(model_version: str, expected_edge_bps: float) -> OpportunityDecision:
    return OpportunityDecision(
        symbol="BTCUSDT",
        decision_source="model",
        model_version=model_version,
        expected_edge_bps=expected_edge_bps,
        success_probability=0.7,
        confidence=0.3,
        proposed_direction="long",
        accepted=True,
        rejection_reason=None,
        rank=1,
        provenance={"probability_method": "test_probability"},
    )


def _snapshot_metadata() -> dict[str, float]:
    return {
        "signal_strength": 0.8,
        "momentum_5m": 0.2,
        "volatility_30m": 0.3,
        "spread_bps": 2.0,
        "fee_bps": 1.0,
        "slippage_bps": 0.5,
        "liquidity_score": 0.85,
        "risk_penalty_bps": 0.2,
    }


def _temporal_eval(version: str, matched_outcomes: int) -> OpportunityTemporalEvaluation:
    return OpportunityTemporalEvaluation(
        model_version=version,
        sample_count=matched_outcomes,
        edge_mae_bps=1.0,
        edge_rmse_bps=1.0,
        directional_accuracy=0.6,
        success_probability_brier=0.2,
        success_probability_ece=0.1,
        avg_success_probability=0.6,
        probability_method="test",
        matched_outcomes=matched_outcomes,
        label_coverage=1.0,
        coverage=1.0,
    )


def _promotion_report(*, recommended: bool, matched_outcomes: int = 12) -> OpportunityPromotionReport:
    champion_eval = _temporal_eval("champion-v1", matched_outcomes)
    challenger_eval = _temporal_eval("challenger-v2", matched_outcomes)
    return OpportunityPromotionReport(
        champion_version="champion-v1",
        challenger_version="challenger-v2",
        evaluation_window_start=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        evaluation_window_end=datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc),
        temporal_comparison=OpportunityTemporalComparison(
            latest=challenger_eval,
            previous=champion_eval,
            common_sample_count=matched_outcomes,
            fairness_applied=True,
            delta_edge_mae_bps=-0.2,
            delta_success_probability_brier=-0.1,
        ),
        shadow_comparison=None,
        drift_report=OpportunityDriftReport(
            reference_count=matched_outcomes,
            evaluation_count=matched_outcomes,
            feature_distribution=(),
            prediction_distribution=(),
            realized_outcome=(),
        ),
        gate_config=OpportunityPromotionGateConfig(),
        gate_results=(
            OpportunityPromotionGateResult(
                gate="test_gate",
                passed=recommended,
                observed=1.0,
                expected=0.0,
                comparator=">=",
                message="test",
            ),
        ),
        promotion_recommended=recommended,
    )


def _readiness_report(
    *,
    activation_ready: bool,
    promotion_recommended: bool | None,
    degraded_reasons: tuple[str, ...] = (),
    matched_outcomes: int = 0,
) -> OpportunityPersistedPromotionReadinessReport:
    return OpportunityPersistedPromotionReadinessReport(
        champion_version="champion-v1",
        challenger_version="challenger-v2",
        champion_shadow_evaluation=None,
        challenger_shadow_evaluation=None,
        promotion_report=(
            _promotion_report(recommended=promotion_recommended, matched_outcomes=matched_outcomes)
            if promotion_recommended is not None
            else None
        ),
        activation_readiness=OpportunityActivationReadiness(
            activation_ready=activation_ready,
            recommendation="hold_current_aliases" if not activation_ready else "manual_alias_switch",
            reasons=() if activation_ready else ("not_ready",),
            alias_targets={"champion": "champion-v1", "challenger": "challenger-v2"},
        ),
        degraded_reasons=degraded_reasons,
        matched_outcomes={"champion": matched_outcomes, "challenger": matched_outcomes},
        evaluation_window_start=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        evaluation_window_end=datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc),
    )


def test_attach_outcome_label_requires_existing_shadow_record(tmp_path: Path) -> None:
    service = OpportunityLifecycleService()
    repository = OpportunityShadowRepository(tmp_path / "shadow")
    label = OpportunityOutcomeLabel(
        symbol="BTCUSDT",
        decision_timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        correlation_key="missing-key",
        horizon_minutes=30,
        realized_return_bps=3.0,
        max_favorable_excursion_bps=4.0,
        max_adverse_excursion_bps=-1.0,
    )
    attached, reason = service.attach_outcome_label(repository, label)
    assert attached is False
    assert reason == "missing_shadow_record:missing-key"


def test_build_persisted_promotion_readiness_uses_real_shadow_outcome_flow(tmp_path: Path) -> None:
    model_repo = FilesystemModelRepository(tmp_path / "models")
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    engine = TradingOpportunityAI(repository=model_repo)
    engine.fit(_build_samples(scale=1.0))
    engine.save_model(version="champion-v1", activate=False)
    engine.fit(_build_samples(scale=1.2))
    engine.save_model(version="challenger-v2", activate=False)

    decision_time = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    champion_records = engine.build_shadow_records(
        [_shadow_decision("champion-v1", 5.0)],
        decision_timestamp=decision_time,
        snapshot={"candidate_metadata": _snapshot_metadata()},
    )
    challenger_records = engine.build_shadow_records(
        [_shadow_decision("challenger-v2", 7.0)],
        decision_timestamp=decision_time,
        snapshot={"candidate_metadata": _snapshot_metadata()},
    )
    shadow_repo.append_shadow_records(champion_records)
    shadow_repo.append_shadow_records(challenger_records)
    shadow_repo.append_outcome_labels(
        [
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=decision_time,
                correlation_key=champion_records[0].record_key,
                horizon_minutes=30,
                realized_return_bps=4.0,
                max_favorable_excursion_bps=6.0,
                max_adverse_excursion_bps=-2.0,
                label_quality="final",
            ),
            OpportunityOutcomeLabel(
                symbol="BTCUSDT",
                decision_timestamp=decision_time,
                correlation_key=challenger_records[0].record_key,
                horizon_minutes=30,
                realized_return_bps=9.0,
                max_favorable_excursion_bps=10.0,
                max_adverse_excursion_bps=-1.0,
                label_quality="final",
            ),
        ]
    )

    service = OpportunityLifecycleService()
    report = service.build_persisted_promotion_readiness(
        model_repository=model_repo,
        shadow_repository=shadow_repo,
        champion_version="champion-v1",
        challenger_version="challenger-v2",
        gate_config=OpportunityPromotionGateConfig(require_shadow_no_regression=False),
    )

    assert report.champion_shadow_evaluation is not None
    assert report.challenger_shadow_evaluation is not None
    assert report.promotion_report is not None
    assert report.matched_outcomes["champion"] == 1
    assert report.matched_outcomes["challenger"] == 1


def test_build_persisted_promotion_readiness_is_explicitly_degraded_without_labels(
    tmp_path: Path,
) -> None:
    model_repo = FilesystemModelRepository(tmp_path / "models")
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    engine = TradingOpportunityAI(repository=model_repo)
    engine.fit(_build_samples())
    engine.save_model(version="single-v1", activate=False)

    shadow_repo.append_shadow_records(
        engine.build_shadow_records(
            [_shadow_decision("single-v1", 5.0)],
            decision_timestamp=datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc),
            snapshot={"candidate_metadata": _snapshot_metadata()},
        )
    )

    service = OpportunityLifecycleService()
    report = service.build_persisted_promotion_readiness(
        model_repository=model_repo,
        shadow_repository=shadow_repo,
        champion_version="single-v1",
        challenger_version="single-v1",
    )

    assert report.promotion_report is None
    assert report.activation_readiness.activation_ready is False
    assert report.degraded_reasons


def test_build_persisted_promotion_readiness_flags_incomplete_feature_context(
    tmp_path: Path,
) -> None:
    model_repo = FilesystemModelRepository(tmp_path / "models")
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    engine = TradingOpportunityAI(repository=model_repo)
    engine.fit(_build_samples(scale=1.0))
    engine.save_model(version="champion-v1", activate=False)
    engine.fit(_build_samples(scale=1.1))
    engine.save_model(version="challenger-v2", activate=False)

    decision_time = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    shadow_repo.append_shadow_records(
        engine.build_shadow_records(
            [_shadow_decision("champion-v1", 5.0)],
            decision_timestamp=decision_time,
            snapshot={"candidate_metadata": {"signal_strength": 0.8}},
        )
    )
    shadow_repo.append_shadow_records(
        engine.build_shadow_records(
            [_shadow_decision("challenger-v2", 6.0)],
            decision_timestamp=decision_time,
            snapshot={"candidate_metadata": _snapshot_metadata()},
        )
    )
    records = shadow_repo.load_shadow_records()
    shadow_repo.append_outcome_labels(
        [
            OpportunityOutcomeLabel(
                symbol=row.symbol,
                decision_timestamp=row.decision_timestamp,
                correlation_key=row.record_key,
                horizon_minutes=30,
                realized_return_bps=4.0,
                max_favorable_excursion_bps=5.0,
                max_adverse_excursion_bps=-2.0,
                label_quality="final",
            )
            for row in records
        ]
    )

    service = OpportunityLifecycleService()
    report = service.build_persisted_promotion_readiness(
        model_repository=model_repo,
        shadow_repository=shadow_repo,
        champion_version="champion-v1",
        challenger_version="challenger-v2",
        gate_config=OpportunityPromotionGateConfig(require_shadow_no_regression=False),
    )

    assert any(
        reason.startswith("incomplete_feature_context_for_snapshot_reconstruction:")
        for reason in report.degraded_reasons
    )


def test_build_persisted_promotion_readiness_excludes_proxy_only_outcomes(tmp_path: Path) -> None:
    model_repo = FilesystemModelRepository(tmp_path / "models")
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    engine = TradingOpportunityAI(repository=model_repo)
    engine.fit(_build_samples(scale=1.0))
    engine.save_model(version="champion-v1", activate=False)
    engine.fit(_build_samples(scale=1.1))
    engine.save_model(version="challenger-v2", activate=False)

    decision_time = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    champion = engine.build_shadow_records(
        [_shadow_decision("champion-v1", 5.0)],
        decision_timestamp=decision_time,
        snapshot={"candidate_metadata": _snapshot_metadata()},
    )[0]
    challenger = engine.build_shadow_records(
        [_shadow_decision("challenger-v2", 6.0)],
        decision_timestamp=decision_time,
        snapshot={"candidate_metadata": _snapshot_metadata()},
    )[0]
    shadow_repo.append_shadow_records([champion, challenger])
    shadow_repo.append_outcome_labels(
        [
            OpportunityOutcomeLabel(
                symbol=champion.symbol,
                decision_timestamp=champion.decision_timestamp,
                correlation_key=champion.record_key,
                horizon_minutes=0,
                realized_return_bps=0.0,
                max_favorable_excursion_bps=0.0,
                max_adverse_excursion_bps=0.0,
                label_quality="execution_proxy_pending_exit",
            ),
            OpportunityOutcomeLabel(
                symbol=challenger.symbol,
                decision_timestamp=challenger.decision_timestamp,
                correlation_key=challenger.record_key,
                horizon_minutes=0,
                realized_return_bps=0.0,
                max_favorable_excursion_bps=0.0,
                max_adverse_excursion_bps=0.0,
                label_quality="execution_proxy_pending_exit",
            ),
        ]
    )

    service = OpportunityLifecycleService()
    report = service.build_persisted_promotion_readiness(
        model_repository=model_repo,
        shadow_repository=shadow_repo,
        champion_version="champion-v1",
        challenger_version="challenger-v2",
    )
    assert report.promotion_report is None
    assert "proxy_only_outcomes_excluded_from_governance" in report.degraded_reasons


def test_build_persisted_promotion_readiness_degrades_partial_only_outcomes(tmp_path: Path) -> None:
    model_repo = FilesystemModelRepository(tmp_path / "models")
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    engine = TradingOpportunityAI(repository=model_repo)
    engine.fit(_build_samples(scale=1.0))
    engine.save_model(version="champion-v1", activate=False)
    engine.fit(_build_samples(scale=1.1))
    engine.save_model(version="challenger-v2", activate=False)

    decision_time = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    champion = engine.build_shadow_records(
        [_shadow_decision("champion-v1", 5.0)],
        decision_timestamp=decision_time,
        snapshot={"candidate_metadata": _snapshot_metadata()},
    )[0]
    challenger = engine.build_shadow_records(
        [_shadow_decision("challenger-v2", 6.0)],
        decision_timestamp=decision_time,
        snapshot={"candidate_metadata": _snapshot_metadata()},
    )[0]
    shadow_repo.append_shadow_records([champion, challenger])
    shadow_repo.append_outcome_labels(
        [
            OpportunityOutcomeLabel(
                symbol=champion.symbol,
                decision_timestamp=champion.decision_timestamp,
                correlation_key=champion.record_key,
                horizon_minutes=15,
                realized_return_bps=2.0,
                max_favorable_excursion_bps=0.0,
                max_adverse_excursion_bps=0.0,
                label_quality="partial_exit_unconfirmed",
            ),
            OpportunityOutcomeLabel(
                symbol=challenger.symbol,
                decision_timestamp=challenger.decision_timestamp,
                correlation_key=challenger.record_key,
                horizon_minutes=10,
                realized_return_bps=1.0,
                max_favorable_excursion_bps=0.0,
                max_adverse_excursion_bps=0.0,
                label_quality="partial_exit_unconfirmed",
            ),
        ]
    )

    service = OpportunityLifecycleService()
    report = service.build_persisted_promotion_readiness(
        model_repository=model_repo,
        shadow_repository=shadow_repo,
        champion_version="champion-v1",
        challenger_version="challenger-v2",
    )
    assert report.promotion_report is None
    assert "partial_only_outcomes_excluded_from_governance" in report.degraded_reasons


def test_build_persisted_promotion_readiness_flags_mixed_final_partial_evidence(tmp_path: Path) -> None:
    model_repo = FilesystemModelRepository(tmp_path / "models")
    shadow_repo = OpportunityShadowRepository(tmp_path / "shadow")
    engine = TradingOpportunityAI(repository=model_repo)
    engine.fit(_build_samples(scale=1.0))
    engine.save_model(version="champion-v1", activate=False)
    engine.fit(_build_samples(scale=1.1))
    engine.save_model(version="challenger-v2", activate=False)

    decision_time = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    champion = engine.build_shadow_records(
        [_shadow_decision("champion-v1", 5.0)],
        decision_timestamp=decision_time,
        snapshot={"candidate_metadata": _snapshot_metadata()},
    )[0]
    challenger = engine.build_shadow_records(
        [_shadow_decision("challenger-v2", 6.0)],
        decision_timestamp=decision_time,
        snapshot={"candidate_metadata": _snapshot_metadata()},
    )[0]
    shadow_repo.append_shadow_records([champion, challenger])
    shadow_repo.append_outcome_labels(
        [
            OpportunityOutcomeLabel(
                symbol=champion.symbol,
                decision_timestamp=champion.decision_timestamp,
                correlation_key=champion.record_key,
                horizon_minutes=30,
                realized_return_bps=4.0,
                max_favorable_excursion_bps=6.0,
                max_adverse_excursion_bps=-2.0,
                label_quality="final",
            ),
            OpportunityOutcomeLabel(
                symbol=challenger.symbol,
                decision_timestamp=challenger.decision_timestamp,
                correlation_key=challenger.record_key,
                horizon_minutes=10,
                realized_return_bps=1.0,
                max_favorable_excursion_bps=0.0,
                max_adverse_excursion_bps=0.0,
                label_quality="partial_exit_unconfirmed",
            ),
        ]
    )

    service = OpportunityLifecycleService()
    report = service.build_persisted_promotion_readiness(
        model_repository=model_repo,
        shadow_repository=shadow_repo,
        champion_version="champion-v1",
        challenger_version="challenger-v2",
    )
    assert any(
        reason.startswith("mixed_final_partial_outcomes_degraded:")
        for reason in report.degraded_reasons
    )


def test_autonomy_gate_proxy_only_evidence_is_denied() -> None:
    service = OpportunityLifecycleService()
    decision = service.evaluate_autonomy_gate(
        readiness_report=_readiness_report(
            activation_ready=False,
            promotion_recommended=None,
            degraded_reasons=("proxy_only_outcomes_excluded_from_governance",),
            matched_outcomes=0,
        )
    )
    assert decision.mode == OpportunityAutonomyMode.DENIED


def test_autonomy_gate_partial_only_evidence_is_shadow_by_default() -> None:
    service = OpportunityLifecycleService()
    decision = service.evaluate_autonomy_gate(
        readiness_report=_readiness_report(
            activation_ready=False,
            promotion_recommended=None,
            degraded_reasons=(
                "partial_only_outcomes_excluded_from_governance",
                "non_final_outcomes_excluded:partial_exit_unconfirmed:5",
            ),
            matched_outcomes=0,
        )
    )
    assert decision.mode == OpportunityAutonomyMode.SHADOW_ONLY


def test_autonomy_gate_partial_only_evidence_can_be_paper_with_explicit_opt_in() -> None:
    service = OpportunityLifecycleService()
    decision = service.evaluate_autonomy_gate(
        readiness_report=_readiness_report(
            activation_ready=False,
            promotion_recommended=None,
            degraded_reasons=(
                "partial_only_outcomes_excluded_from_governance",
                "non_final_outcomes_excluded:partial_exit_unconfirmed:5",
            ),
            matched_outcomes=0,
        ),
        gate_config=OpportunityAutonomyGateConfig(
            allow_partial_only_for_shadow_or_paper=True,
            min_observed_outcomes_for_paper_autonomy=3,
        ),
    )
    assert decision.mode == OpportunityAutonomyMode.PAPER_AUTONOMOUS


def test_autonomy_gate_mixed_final_partial_keeps_full_observed_evidence_summary() -> None:
    service = OpportunityLifecycleService()
    decision = service.evaluate_autonomy_gate(
        readiness_report=_readiness_report(
            activation_ready=True,
            promotion_recommended=True,
            degraded_reasons=(
                "mixed_final_partial_outcomes_degraded:final:4,partial:3",
                "non_final_outcomes_excluded:final:4,partial_exit_unconfirmed:3",
            ),
            matched_outcomes=4,
        ),
    )
    assert decision.evidence_summary["final_outcomes"] == 4
    assert decision.evidence_summary["partial_outcomes"] == 3
    assert decision.evidence_summary["observed_outcomes"] == 7


def test_autonomy_gate_final_evidence_without_activation_readiness_blocks_live() -> None:
    service = OpportunityLifecycleService()
    decision = service.evaluate_autonomy_gate(
        readiness_report=_readiness_report(
            activation_ready=False,
            promotion_recommended=True,
            degraded_reasons=(),
            matched_outcomes=12,
        )
    )
    assert decision.mode != OpportunityAutonomyMode.LIVE_AUTONOMOUS
    assert decision.mode == OpportunityAutonomyMode.PAPER_AUTONOMOUS


def test_autonomy_gate_live_assisted_when_single_non_blocking_degradation_present() -> None:
    service = OpportunityLifecycleService()
    decision = service.evaluate_autonomy_gate(
        readiness_report=_readiness_report(
            activation_ready=True,
            promotion_recommended=True,
            degraded_reasons=("mixed_final_partial_outcomes_degraded:final:12,partial:1",),
            matched_outcomes=12,
        )
    )
    assert decision.mode == OpportunityAutonomyMode.LIVE_ASSISTED


def test_autonomy_gate_promotion_fail_blocks_live_autonomous() -> None:
    service = OpportunityLifecycleService()
    decision = service.evaluate_autonomy_gate(
        readiness_report=_readiness_report(
            activation_ready=True,
            promotion_recommended=False,
            degraded_reasons=("promotion_gate_failed",),
            matched_outcomes=12,
        )
    )
    assert decision.mode != OpportunityAutonomyMode.LIVE_AUTONOMOUS


def test_autonomy_gate_blocking_degraded_reason_denies_execution() -> None:
    service = OpportunityLifecycleService()
    decision = service.evaluate_autonomy_gate(
        readiness_report=_readiness_report(
            activation_ready=True,
            promotion_recommended=True,
            degraded_reasons=("promotion_gate_failed",),
            matched_outcomes=12,
        )
    )
    assert decision.mode == OpportunityAutonomyMode.DENIED


def test_autonomy_gate_clean_strong_case_allows_live_autonomous() -> None:
    service = OpportunityLifecycleService()
    decision = service.evaluate_autonomy_gate(
        readiness_report=_readiness_report(
            activation_ready=True,
            promotion_recommended=True,
            degraded_reasons=(),
            matched_outcomes=12,
        )
    )
    assert decision.mode == OpportunityAutonomyMode.LIVE_AUTONOMOUS


def test_autonomy_gate_decision_shape_is_serializable() -> None:
    service = OpportunityLifecycleService()
    decision = service.evaluate_autonomy_gate(
        readiness_report=_readiness_report(
            activation_ready=True,
            promotion_recommended=True,
            degraded_reasons=(),
            matched_outcomes=12,
        )
    )
    payload = decision.to_dict()
    assert payload["mode"] == "live_autonomous"
    assert payload["autonomous_execution_allowed"] is True
    assert isinstance(payload["reasons"], list)
    assert isinstance(payload["evidence_summary"], dict)
