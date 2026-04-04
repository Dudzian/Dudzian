from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bot_core.ai import (
    FilesystemModelRepository,
    OpportunityPromotionGateConfig,
    OpportunityOutcomeLabel,
    OpportunityShadowRecord,
    OpportunitySnapshot,
    OpportunityThresholdConfig,
    OpportunityTemporalEvaluator,
    TradingOpportunityAI,
)
from bot_core.ai.models import ModelArtifact


def _build_samples(scale: float = 1.0) -> list[OpportunitySnapshot]:
    samples: list[OpportunitySnapshot] = []
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for idx in range(30):
        signal = (idx - 15) / 10.0
        momentum = signal * 0.8
        volatility = 0.2 + (idx % 4) * 0.05
        spread = 0.5 + (idx % 3) * 0.1
        fee = 0.2
        slippage = 0.1 + (idx % 2) * 0.1
        liquidity = 0.9 - (idx % 5) * 0.1
        risk_penalty = max(0.0, volatility * 2.0 - liquidity)
        realized = scale * (signal * 7.5 + momentum * 2.5 - volatility * 1.2)
        samples.append(
            OpportunitySnapshot(
                symbol=f"PAIR{idx % 3}",
                signal_strength=signal,
                momentum_5m=momentum,
                volatility_30m=volatility,
                spread_bps=spread,
                fee_bps=fee,
                slippage_bps=slippage,
                liquidity_score=liquidity,
                risk_penalty_bps=risk_penalty,
                realized_return_bps=realized,
                as_of=base + timedelta(minutes=idx),
            )
        )
    return samples


def test_evaluate_uses_classifier_path_when_available() -> None:
    engine = TradingOpportunityAI()
    artifact = engine.fit(_build_samples())
    evaluator = OpportunityTemporalEvaluator()

    report = evaluator.evaluate(artifact, _build_samples())

    assert report.sample_count == 30
    assert report.probability_method == "model_success_classifier_calibrated"
    assert report.calibration_method == "platt_scaling"
    assert 0.0 <= report.avg_success_probability <= 1.0
    assert report.success_probability_ece >= 0.0
    assert report.reliability_summary


def test_evaluate_falls_back_for_legacy_artifact_without_classifier() -> None:
    engine = TradingOpportunityAI()
    artifact = engine.fit(_build_samples())
    legacy_state = {
        key: value for key, value in dict(artifact.model_state).items() if key != "classifier_head_state"
    }
    legacy = type(artifact)(
        feature_names=artifact.feature_names,
        model_state=legacy_state,
        trained_at=artifact.trained_at,
        metrics=artifact.metrics,
        metadata=artifact.metadata,
        target_scale=artifact.target_scale,
        training_rows=artifact.training_rows,
        validation_rows=artifact.validation_rows,
        test_rows=artifact.test_rows,
        feature_scalers=artifact.feature_scalers,
        decision_journal_entry_id=artifact.decision_journal_entry_id,
        backend=artifact.backend,
    )
    evaluator = OpportunityTemporalEvaluator()

    report = evaluator.evaluate(legacy, _build_samples())

    assert report.probability_method == "heuristic_sigmoid_scaled_edge_fallback"


def test_evaluate_reports_uncalibrated_when_metadata_declares_platt_but_state_missing() -> None:
    engine = TradingOpportunityAI()
    artifact = engine.fit(_build_samples())
    assert isinstance(artifact.metadata.get("probability_calibration"), dict)
    metadata_declares_platt = {
        **dict(artifact.metadata),
        "probability_calibration": {
            "method": "platt_scaling",
            "fit_split": "validation",
        },
    }
    state_without_calibrator = {
        key: value
        for key, value in dict(artifact.model_state).items()
        if key != "success_probability_calibrator"
    }
    inconsistent_artifact = type(artifact)(
        feature_names=artifact.feature_names,
        model_state=state_without_calibrator,
        trained_at=artifact.trained_at,
        metrics=artifact.metrics,
        metadata=metadata_declares_platt,
        target_scale=artifact.target_scale,
        training_rows=artifact.training_rows,
        validation_rows=artifact.validation_rows,
        test_rows=artifact.test_rows,
        feature_scalers=artifact.feature_scalers,
        decision_journal_entry_id=artifact.decision_journal_entry_id,
        backend=artifact.backend,
    )
    evaluator = OpportunityTemporalEvaluator()

    report = evaluator.evaluate(inconsistent_artifact, _build_samples())

    assert report.probability_method == "model_success_classifier_uncalibrated"
    assert report.calibration_method == "none"


def test_evaluate_reports_uncalibrated_when_calibrator_payload_is_invalid() -> None:
    engine = TradingOpportunityAI()
    artifact = engine.fit(_build_samples())
    metadata_declares_platt = {
        **dict(artifact.metadata),
        "probability_calibration": {
            "method": "platt_scaling",
            "fit_split": "validation",
        },
    }
    invalid_state = {
        **dict(artifact.model_state),
        "success_probability_calibrator": {
            "method": "platt_scaling",
            "slope": "invalid",
            "intercept": "invalid",
        },
    }
    inconsistent_artifact = type(artifact)(
        feature_names=artifact.feature_names,
        model_state=invalid_state,
        trained_at=artifact.trained_at,
        metrics=artifact.metrics,
        metadata=metadata_declares_platt,
        target_scale=artifact.target_scale,
        training_rows=artifact.training_rows,
        validation_rows=artifact.validation_rows,
        test_rows=artifact.test_rows,
        feature_scalers=artifact.feature_scalers,
        decision_journal_entry_id=artifact.decision_journal_entry_id,
        backend=artifact.backend,
    )
    evaluator = OpportunityTemporalEvaluator()

    report = evaluator.evaluate(inconsistent_artifact, _build_samples())

    assert report.probability_method == "model_success_classifier_uncalibrated"
    assert report.calibration_method == "none"


def test_evaluate_reports_heuristic_and_no_calibration_when_classifier_missing_but_calibrator_present() -> None:
    engine = TradingOpportunityAI()
    artifact = engine.fit(_build_samples())
    model_state_without_classifier = {
        key: value
        for key, value in dict(artifact.model_state).items()
        if key != "classifier_head_state"
    }
    assert "success_probability_calibrator" in model_state_without_classifier
    classifier_missing_artifact = type(artifact)(
        feature_names=artifact.feature_names,
        model_state=model_state_without_classifier,
        trained_at=artifact.trained_at,
        metrics=artifact.metrics,
        metadata=artifact.metadata,
        target_scale=artifact.target_scale,
        training_rows=artifact.training_rows,
        validation_rows=artifact.validation_rows,
        test_rows=artifact.test_rows,
        feature_scalers=artifact.feature_scalers,
        decision_journal_entry_id=artifact.decision_journal_entry_id,
        backend=artifact.backend,
    )
    evaluator = OpportunityTemporalEvaluator()

    report = evaluator.evaluate(classifier_missing_artifact, _build_samples())

    assert report.probability_method == "heuristic_sigmoid_scaled_edge_fallback"
    assert report.calibration_method == "none"


def test_evaluate_with_model_comparison_uses_common_subset_fairness() -> None:
    engine = TradingOpportunityAI()
    latest = engine.fit(_build_samples(scale=1.0))
    previous = engine.fit(_build_samples(scale=-1.0))
    evaluator = OpportunityTemporalEvaluator()

    duplicated = _build_samples() + [_build_samples()[0]]
    comparison = evaluator.evaluate_with_model_comparison(latest, previous, duplicated)

    assert comparison.fairness_applied is True
    assert comparison.common_sample_count == 30
    assert comparison.latest.sample_count == comparison.previous.sample_count == 30


def test_evaluate_with_model_comparison_aligns_real_scoreable_subset() -> None:
    engine = TradingOpportunityAI()
    latest = engine.fit(_build_samples(scale=1.0))
    previous = engine.fit(_build_samples(scale=-1.0))
    evaluator = OpportunityTemporalEvaluator()
    samples = _build_samples()
    samples.append(
        OpportunitySnapshot(
            symbol="PAIR0",
            signal_strength=float("nan"),
            momentum_5m=0.1,
            volatility_30m=0.2,
            spread_bps=0.3,
            fee_bps=0.1,
            slippage_bps=0.1,
            liquidity_score=0.8,
            risk_penalty_bps=0.0,
            realized_return_bps=0.2,
            as_of=datetime(2026, 1, 3, tzinfo=timezone.utc),
        )
    )

    comparison = evaluator.evaluate_with_model_comparison(latest, previous, samples)

    assert comparison.fairness_applied is True
    assert comparison.common_sample_count == 30


def test_evaluate_with_model_comparison_raises_on_empty_common_scoreable_subset() -> None:
    engine = TradingOpportunityAI()
    latest = engine.fit(_build_samples(scale=1.0))
    previous = engine.fit(_build_samples(scale=-1.0))
    latest_bad = ModelArtifact(
        feature_names=latest.feature_names,
        model_state={**dict(latest.model_state), "feature_names": ["ghost_feature"]},
        trained_at=latest.trained_at,
        metrics=latest.metrics,
        metadata=latest.metadata,
        target_scale=latest.target_scale,
        training_rows=latest.training_rows,
        validation_rows=latest.validation_rows,
        test_rows=latest.test_rows,
        feature_scalers=latest.feature_scalers,
        decision_journal_entry_id=latest.decision_journal_entry_id,
        backend=latest.backend,
    )
    evaluator = OpportunityTemporalEvaluator()

    with pytest.raises(ValueError, match="Brak wspólnego podzbioru"):
        evaluator.evaluate_with_model_comparison(latest_bad, previous, _build_samples())


def test_evaluate_with_model_comparison_rejects_inconsistent_feature_spec_metadata() -> None:
    engine = TradingOpportunityAI()
    latest = engine.fit(_build_samples(scale=1.0))
    previous = engine.fit(_build_samples(scale=-1.0))
    previous_bad_metadata = ModelArtifact(
        feature_names=previous.feature_names,
        model_state=previous.model_state,
        trained_at=previous.trained_at,
        metrics=previous.metrics,
        metadata={
            **dict(previous.metadata),
            "feature_spec": {
                **dict(previous.metadata["feature_spec"]),
                "names": ["signal_strength", "momentum_5m"],
            },
        },
        target_scale=previous.target_scale,
        training_rows=previous.training_rows,
        validation_rows=previous.validation_rows,
        test_rows=previous.test_rows,
        feature_scalers=previous.feature_scalers,
        decision_journal_entry_id=previous.decision_journal_entry_id,
        backend=previous.backend,
    )
    evaluator = OpportunityTemporalEvaluator()

    with pytest.raises(ValueError, match="Feature spec mismatch"):
        evaluator.evaluate_with_model_comparison(latest, previous_bad_metadata, _build_samples())


def test_evaluate_with_model_comparison_uses_common_subset_for_classifier_feature_requirements() -> None:
    engine = TradingOpportunityAI()
    latest = engine.fit(_build_samples(scale=1.0))
    previous = engine.fit(_build_samples(scale=-1.0))
    latest_with_unscoreable_classifier = ModelArtifact(
        feature_names=latest.feature_names,
        model_state={
            **dict(latest.model_state),
            "classifier_head_state": {
                **dict(latest.model_state["classifier_head_state"]),
                "feature_names": ["ghost_feature"],
            },
        },
        trained_at=latest.trained_at,
        metrics=latest.metrics,
        metadata=latest.metadata,
        target_scale=latest.target_scale,
        training_rows=latest.training_rows,
        validation_rows=latest.validation_rows,
        test_rows=latest.test_rows,
        feature_scalers=latest.feature_scalers,
        decision_journal_entry_id=latest.decision_journal_entry_id,
        backend=latest.backend,
    )
    evaluator = OpportunityTemporalEvaluator()

    with pytest.raises(ValueError, match="Brak wspólnego podzbioru"):
        evaluator.evaluate_with_model_comparison(
            latest_with_unscoreable_classifier,
            previous,
            _build_samples(),
        )


def test_evaluate_with_model_comparison_computes_deltas_on_aligned_rows_only() -> None:
    engine = TradingOpportunityAI()
    latest = engine.fit(_build_samples(scale=1.0))
    previous = engine.fit(_build_samples(scale=-1.0))
    evaluator = OpportunityTemporalEvaluator()
    base_samples = _build_samples()
    unscoreable = OpportunitySnapshot(
        symbol="PAIR0",
        signal_strength=float("nan"),
        momentum_5m=0.1,
        volatility_30m=0.2,
        spread_bps=0.3,
        fee_bps=0.1,
        slippage_bps=0.1,
        liquidity_score=0.8,
        risk_penalty_bps=0.0,
        realized_return_bps=0.2,
        as_of=datetime(2026, 1, 4, tzinfo=timezone.utc),
    )
    samples = base_samples + [unscoreable]

    comparison = evaluator.evaluate_with_model_comparison(latest, previous, samples)
    latest_aligned = evaluator.evaluate(latest, base_samples)
    previous_aligned = evaluator.evaluate(previous, base_samples)

    assert comparison.common_sample_count == len(base_samples)
    assert comparison.delta_edge_mae_bps == pytest.approx(
        latest_aligned.edge_mae_bps - previous_aligned.edge_mae_bps
    )
    assert comparison.delta_success_probability_brier == pytest.approx(
        latest_aligned.success_probability_brier - previous_aligned.success_probability_brier
    )


def test_evaluate_latest_vs_previous_loads_versions_from_repository(tmp_path) -> None:
    repository = FilesystemModelRepository(tmp_path / "repo")
    engine = TradingOpportunityAI(repository=repository)
    evaluator = OpportunityTemporalEvaluator()

    first_artifact = engine.fit(_build_samples(scale=-1.0))
    repository.publish(first_artifact, version="v1", aliases=("previous",), activate=False)
    second_artifact = engine.fit(_build_samples(scale=1.0))
    repository.publish(second_artifact, version="v2", aliases=("latest",), activate=True)

    comparison = evaluator.evaluate_latest_vs_previous(repository, _build_samples())

    assert comparison.latest.model_version == second_artifact.metadata["model_version"]
    assert comparison.previous.model_version == first_artifact.metadata["model_version"]
    assert comparison.fairness_applied is False


def test_detect_drift_report_contract_includes_feature_prediction_and_outcome_metrics() -> None:
    engine = TradingOpportunityAI()
    evaluator = OpportunityTemporalEvaluator()
    champion = engine.fit(_build_samples(scale=-1.0))
    challenger = engine.fit(_build_samples(scale=1.0))

    drift = evaluator.detect_drift(
        reference_artifact=champion,
        evaluation_artifact=challenger,
        reference_samples=_build_samples(scale=-1.0),
        evaluation_samples=_build_samples(scale=1.0),
    )

    assert drift.reference_count == 30
    assert drift.evaluation_count == 30
    assert drift.feature_distribution
    assert drift.prediction_distribution
    assert drift.realized_outcome
    assert all(metric.name for metric in drift.feature_distribution)


def test_build_promotion_report_rejects_worse_challenger() -> None:
    engine = TradingOpportunityAI()
    evaluator = OpportunityTemporalEvaluator()
    champion = engine.fit(_build_samples(scale=1.0))
    challenger = engine.fit(_build_samples(scale=-1.0))

    report = evaluator.build_promotion_report(
        champion_artifact=champion,
        challenger_artifact=challenger,
        evaluation_samples=_build_samples(scale=1.0),
        reference_samples=_build_samples(scale=1.0),
    )

    assert report.promotion_recommended is False
    assert any(not gate.passed for gate in report.gate_results)


def test_build_promotion_report_recommends_better_challenger_without_regression() -> None:
    engine = TradingOpportunityAI()
    evaluator = OpportunityTemporalEvaluator()
    champion = engine.fit(_build_samples(scale=-1.0))
    challenger = engine.fit(_build_samples(scale=1.0))

    report = evaluator.build_promotion_report(
        champion_artifact=champion,
        challenger_artifact=challenger,
        evaluation_samples=_build_samples(scale=1.0),
        reference_samples=_build_samples(scale=1.0),
        gate_config=OpportunityPromotionGateConfig(
            min_edge_mae_improvement_bps=0.01,
            min_brier_improvement=0.0,
            max_directional_accuracy_regression=0.0,
            max_drift_metrics_triggered=20,
            require_shadow_no_regression=False,
        ),
    )
    metadata = report.to_metadata()

    assert report.promotion_recommended is True
    assert all(gate.passed for gate in report.gate_results)
    assert metadata["compared_versions"] == {
        "champion": champion.metadata["model_version"],
        "challenger": challenger.metadata["model_version"],
    }
    assert metadata["promotion"]["recommended"] is True


def test_detect_drift_rejects_artifact_with_inconsistent_feature_spec_metadata() -> None:
    engine = TradingOpportunityAI()
    evaluator = OpportunityTemporalEvaluator()
    reference = engine.fit(_build_samples(scale=-1.0))
    evaluation = engine.fit(_build_samples(scale=1.0))
    evaluation_bad_metadata = ModelArtifact(
        feature_names=evaluation.feature_names,
        model_state=evaluation.model_state,
        trained_at=evaluation.trained_at,
        metrics=evaluation.metrics,
        metadata={
            **dict(evaluation.metadata),
            "feature_spec": {
                **dict(evaluation.metadata["feature_spec"]),
                "names": ["signal_strength", "momentum_5m"],
            },
        },
        target_scale=evaluation.target_scale,
        training_rows=evaluation.training_rows,
        validation_rows=evaluation.validation_rows,
        test_rows=evaluation.test_rows,
        feature_scalers=evaluation.feature_scalers,
        decision_journal_entry_id=evaluation.decision_journal_entry_id,
        backend=evaluation.backend,
    )

    with pytest.raises(ValueError, match="Feature spec mismatch"):
        evaluator.detect_drift(
            reference_artifact=reference,
            evaluation_artifact=evaluation_bad_metadata,
            reference_samples=_build_samples(scale=-1.0),
            evaluation_samples=_build_samples(scale=1.0),
        )


def test_normalized_mean_shift_for_constant_distributions_with_different_means_is_non_zero() -> None:
    score = OpportunityTemporalEvaluator._normalized_mean_shift([1.0, 1.0, 1.0], [10.0, 10.0, 10.0])

    assert score > 0.0
    assert score == pytest.approx(9.0)


def test_temporal_split_and_walk_forward_contract() -> None:
    evaluator = OpportunityTemporalEvaluator()
    samples = _build_samples()

    train, test = evaluator.split_temporal(samples, train_ratio=0.7)
    splits = evaluator.walk_forward_splits(samples, folds=3, min_train_size=5)

    assert train
    assert test
    assert len(train) + len(test) == len(samples)
    assert len(splits) >= 1
    for train_fold, test_fold in splits:
        assert train_fold
        assert test_fold


def test_public_api_exports_temporal_evaluator_from_package() -> None:
    from bot_core import ai as ai_pkg

    evaluator_cls = getattr(ai_pkg, "OpportunityTemporalEvaluator")
    assert evaluator_cls.__module__.endswith("opportunity_evaluation")


def test_evaluate_from_shadow_labels_uses_record_outcome_contract() -> None:
    evaluator = OpportunityTemporalEvaluator()
    decision_timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    record = OpportunityShadowRecord(
        record_key="rk-1",
        symbol="BTC/USDT",
        decision_timestamp=decision_timestamp,
        model_version="opp-v1",
        decision_source="model",
        expected_edge_bps=2.4,
        success_probability=0.73,
        confidence=0.46,
        proposed_direction="long",
        accepted=True,
        rejection_reason=None,
        rank=1,
        provenance={"probability_method": "model_success_classifier_calibrated"},
        threshold_config=OpportunityThresholdConfig(),
        snapshot={},
    )
    label = OpportunityOutcomeLabel(
        symbol="BTC/USDT",
        decision_timestamp=decision_timestamp,
        correlation_key="rk-1",
        horizon_minutes=30,
        realized_return_bps=1.8,
        max_favorable_excursion_bps=2.6,
        max_adverse_excursion_bps=-0.7,
    )

    report = evaluator.evaluate_from_shadow_labels([record], [label])

    assert report.sample_count == 1
    assert report.matched_outcomes == 1
    assert report.label_coverage == pytest.approx(1.0)
    assert report.coverage == pytest.approx(1.0)
    assert report.abstention_rate == pytest.approx(0.0)
    assert report.accepted_sample_count == 1
    assert report.probability_method == "model_success_classifier_calibrated"
    assert report.success_probability_ece >= 0.0


def test_evaluate_from_shadow_labels_reports_abstention_and_conditional_quality() -> None:
    evaluator = OpportunityTemporalEvaluator()
    decision_timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    accepted_record = OpportunityShadowRecord(
        record_key="rk-accepted",
        symbol="BTC/USDT",
        decision_timestamp=decision_timestamp,
        model_version="opp-v1",
        decision_source="model",
        expected_edge_bps=2.0,
        success_probability=0.7,
        confidence=0.4,
        proposed_direction="long",
        accepted=True,
        rejection_reason=None,
        rank=1,
        provenance={"probability_method": "model_success_classifier_calibrated"},
        threshold_config=OpportunityThresholdConfig(),
        snapshot={},
    )
    abstained_record = OpportunityShadowRecord(
        record_key="rk-abstain",
        symbol="ETH/USDT",
        decision_timestamp=decision_timestamp,
        model_version="opp-v1",
        decision_source="model",
        expected_edge_bps=1.5,
        success_probability=0.52,
        confidence=0.04,
        proposed_direction="skip",
        accepted=False,
        rejection_reason="abstain_low_agreement",
        rank=2,
        provenance={"probability_method": "model_success_classifier_calibrated"},
        threshold_config=OpportunityThresholdConfig(),
        snapshot={},
    )
    labels = [
        OpportunityOutcomeLabel(
            symbol="BTC/USDT",
            decision_timestamp=decision_timestamp,
            correlation_key="rk-accepted",
            horizon_minutes=30,
            realized_return_bps=1.0,
            max_favorable_excursion_bps=2.0,
            max_adverse_excursion_bps=-0.5,
        ),
        OpportunityOutcomeLabel(
            symbol="ETH/USDT",
            decision_timestamp=decision_timestamp,
            correlation_key="rk-abstain",
            horizon_minutes=30,
            realized_return_bps=-1.2,
            max_favorable_excursion_bps=0.4,
            max_adverse_excursion_bps=-1.8,
        ),
    ]

    report = evaluator.evaluate_from_shadow_labels([accepted_record, abstained_record], labels)

    assert report.coverage == pytest.approx(1.0)
    assert report.abstention_rate == pytest.approx(0.5)
    assert report.accepted_sample_count == 1
    assert report.accepted_edge_mae_bps == pytest.approx(1.0)
    assert report.accepted_success_probability_brier == pytest.approx(0.09)


def test_evaluate_from_shadow_labels_uses_full_shadow_set_for_abstention_rate() -> None:
    evaluator = OpportunityTemporalEvaluator()
    decision_timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    shadow_records = [
        OpportunityShadowRecord(
            record_key="rk-accepted-labeled",
            symbol="BTC/USDT",
            decision_timestamp=decision_timestamp,
            model_version="opp-v1",
            decision_source="model",
            expected_edge_bps=2.0,
            success_probability=0.75,
            confidence=0.5,
            proposed_direction="long",
            accepted=True,
            rejection_reason=None,
            rank=1,
            provenance={"probability_method": "model_success_classifier_calibrated"},
            threshold_config=OpportunityThresholdConfig(),
            snapshot={},
        ),
        OpportunityShadowRecord(
            record_key="rk-abstain-unlabeled-1",
            symbol="ETH/USDT",
            decision_timestamp=decision_timestamp,
            model_version="opp-v1",
            decision_source="model",
            expected_edge_bps=1.2,
            success_probability=0.51,
            confidence=0.02,
            proposed_direction="skip",
            accepted=False,
            rejection_reason="abstain_low_agreement",
            rank=2,
            provenance={"probability_method": "model_success_classifier_calibrated"},
            threshold_config=OpportunityThresholdConfig(),
            snapshot={},
        ),
        OpportunityShadowRecord(
            record_key="rk-threshold-reject-unlabeled",
            symbol="SOL/USDT",
            decision_timestamp=decision_timestamp,
            model_version="opp-v1",
            decision_source="model",
            expected_edge_bps=0.8,
            success_probability=0.49,
            confidence=0.02,
            proposed_direction="skip",
            accepted=False,
            rejection_reason="edge_below_threshold",
            rank=3,
            provenance={"probability_method": "model_success_classifier_calibrated"},
            threshold_config=OpportunityThresholdConfig(),
            snapshot={},
        ),
    ]
    labels = [
        OpportunityOutcomeLabel(
            symbol="BTC/USDT",
            decision_timestamp=decision_timestamp,
            correlation_key="rk-accepted-labeled",
            horizon_minutes=30,
            realized_return_bps=1.0,
            max_favorable_excursion_bps=1.9,
            max_adverse_excursion_bps=-0.4,
        )
    ]

    report = evaluator.evaluate_from_shadow_labels(shadow_records, labels)

    assert report.sample_count == 1
    assert report.coverage == pytest.approx(1.0 / 3.0)
    assert report.abstention_rate == pytest.approx(1.0 / 3.0)
    assert report.accepted_sample_count == 1
