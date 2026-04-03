from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from bot_core.ai import (
    FilesystemModelRepository,
    ModelArtifact,
    OpportunityCandidate,
    OpportunityDecision,
    OpportunityOutcomeLabel,
    OpportunityShadowContext,
    OpportunityShadowRecord,
    OpportunityShadowRepository,
    OpportunityTemporalEvaluator,
    OpportunityThresholdConfig,
    OpportunitySnapshot,
    TradingOpportunityAI,
)


def _build_samples(scale: float = 1.0) -> list[OpportunitySnapshot]:
    samples: list[OpportunitySnapshot] = []
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
            )
        )
    return samples


def test_training_and_contract_ranking_fields(tmp_path) -> None:
    repository = FilesystemModelRepository(tmp_path / "models")
    engine = TradingOpportunityAI(repository=repository)

    artifact = engine.fit(_build_samples())
    path = engine.save_model(activate=True)

    assert artifact.training_rows == 30
    assert artifact.metadata["decision_source"] == "model"
    assert artifact.metadata["artifact_schema_version"] == "opportunity_dual_head_v1"
    assert path.endswith(".json")

    decisions = engine.rank(
        [
            OpportunityCandidate(
                symbol="BTC/USDT",
                signal_strength=1.1,
                momentum_5m=0.8,
                volatility_30m=0.25,
                spread_bps=0.5,
                fee_bps=0.2,
                slippage_bps=0.1,
                liquidity_score=0.95,
                risk_penalty_bps=0.0,
                direction_hint="long",
            ),
            OpportunityCandidate(
                symbol="ALT/USDT",
                signal_strength=-0.8,
                momentum_5m=-0.7,
                volatility_30m=0.6,
                spread_bps=1.2,
                fee_bps=0.3,
                slippage_bps=0.4,
                liquidity_score=0.2,
                risk_penalty_bps=1.0,
            ),
        ],
        min_expected_edge_bps=0.0,
        min_probability=0.55,
    )

    assert len(decisions) == 2
    assert decisions[0].rank == 1
    assert decisions[1].rank == 2
    assert decisions[0].decision_source == "model"
    assert decisions[0].model_version
    assert 0.0 <= decisions[0].success_probability <= 1.0
    assert decisions[1].proposed_direction in {"skip", "short", "long", "hold"}
    if not decisions[1].accepted:
        assert decisions[1].rejection_reason in {
            "edge_below_threshold",
            "probability_below_threshold",
        }


def test_save_load_keeps_prediction_consistent(tmp_path) -> None:
    repository = FilesystemModelRepository(tmp_path / "repo")
    trainer = TradingOpportunityAI(repository=repository)
    trainer.fit(_build_samples())
    trainer.save_model(version="vtest", activate=True)

    reloaded = TradingOpportunityAI(repository=repository)
    reloaded.load_model("vtest")

    candidate = OpportunityCandidate(
        symbol="ETH/USDT",
        signal_strength=0.75,
        momentum_5m=0.4,
        volatility_30m=0.3,
        spread_bps=0.5,
        fee_bps=0.2,
        slippage_bps=0.1,
        liquidity_score=0.9,
        risk_penalty_bps=0.0,
    )
    a = trainer.rank([candidate])[0]
    b = reloaded.rank([candidate])[0]

    assert abs(a.expected_edge_bps - b.expected_edge_bps) < 1e-9
    assert abs(a.success_probability - b.success_probability) < 1e-12


def test_fit_dual_head_contains_classifier_state_and_metadata() -> None:
    engine = TradingOpportunityAI()
    artifact = engine.fit(_build_samples())

    assert "classifier_head_state" in artifact.model_state
    assert artifact.metadata["artifact_schema_version"] == "opportunity_dual_head_v1"
    assert artifact.metadata["classification_target_definition"] == "1 if target_definition > 0 else 0"
    assert artifact.metadata["heads"]["success_classifier"]["prediction_field"] == "success_probability"
    summary = artifact.metrics.summary()
    assert "classifier_accuracy" in summary
    assert "classifier_brier_score" in summary


def test_save_load_dual_head_artifact_keeps_classifier_probability(tmp_path) -> None:
    repository = FilesystemModelRepository(tmp_path / "repo")
    trainer = TradingOpportunityAI(repository=repository)
    trainer.fit(_build_samples())
    trainer.save_model(version="dual-head", activate=True)

    candidate = OpportunityCandidate(
        symbol="UNI/USDT",
        signal_strength=0.95,
        momentum_5m=0.7,
        volatility_30m=0.25,
        spread_bps=0.4,
        fee_bps=0.2,
        slippage_bps=0.1,
        liquidity_score=0.9,
        risk_penalty_bps=0.0,
    )
    before = trainer.rank([candidate])[0]

    reloaded = TradingOpportunityAI(repository=repository)
    artifact = reloaded.load_model("dual-head")
    after = reloaded.rank([candidate])[0]

    assert "classifier_head_state" in artifact.model_state
    assert before.success_probability == pytest.approx(after.success_probability, rel=1e-12, abs=1e-12)


def test_model_retrains_and_changes_scores_when_distribution_changes() -> None:
    engine = TradingOpportunityAI()
    candidate = OpportunityCandidate(
        symbol="XRP/USDT",
        signal_strength=1.0,
        momentum_5m=0.7,
        volatility_30m=0.25,
        spread_bps=0.5,
        fee_bps=0.2,
        slippage_bps=0.2,
        liquidity_score=0.8,
        risk_penalty_bps=0.0,
    )

    engine.fit(_build_samples(scale=1.0))
    first = engine.rank([candidate])[0].expected_edge_bps

    engine.fit(_build_samples(scale=-1.0))
    second = engine.rank([candidate])[0].expected_edge_bps

    assert first > 0
    assert second < 0


def test_fit_rejects_nan_and_inf_values() -> None:
    engine = TradingOpportunityAI()
    samples = _build_samples()
    samples[0] = OpportunitySnapshot(
        symbol="BTC/USDT",
        signal_strength=float("nan"),
        momentum_5m=0.1,
        volatility_30m=0.2,
        spread_bps=0.3,
        fee_bps=0.2,
        slippage_bps=0.1,
        liquidity_score=0.8,
        risk_penalty_bps=0.1,
        realized_return_bps=1.2,
    )
    with pytest.raises(ValueError, match="Niefinitywna cecha"):
        engine.fit(samples)

    samples = _build_samples()
    samples[1] = OpportunitySnapshot(
        symbol="ETH/USDT",
        signal_strength=0.3,
        momentum_5m=0.1,
        volatility_30m=0.2,
        spread_bps=0.3,
        fee_bps=0.2,
        slippage_bps=0.1,
        liquidity_score=0.8,
        risk_penalty_bps=float("inf"),
        realized_return_bps=1.2,
    )
    with pytest.raises(ValueError, match="Niefinitywna cecha"):
        engine.fit(samples)


def test_fit_rejects_empty_symbol() -> None:
    engine = TradingOpportunityAI()
    samples = _build_samples()
    samples[0] = OpportunitySnapshot(
        symbol="   ",
        signal_strength=0.1,
        momentum_5m=0.1,
        volatility_30m=0.2,
        spread_bps=0.3,
        fee_bps=0.2,
        slippage_bps=0.1,
        liquidity_score=0.8,
        risk_penalty_bps=0.1,
        realized_return_bps=1.2,
    )
    with pytest.raises(ValueError, match="Nieprawidłowy symbol"):
        engine.fit(samples)

    samples = _build_samples()
    samples[0] = OpportunitySnapshot(
        symbol=None,  # type: ignore[arg-type]
        signal_strength=0.1,
        momentum_5m=0.1,
        volatility_30m=0.2,
        spread_bps=0.3,
        fee_bps=0.2,
        slippage_bps=0.1,
        liquidity_score=0.8,
        risk_penalty_bps=0.1,
        realized_return_bps=1.2,
    )
    with pytest.raises(ValueError, match="Nieprawidłowy symbol"):
        engine.fit(samples)

    samples = _build_samples()
    samples[0] = OpportunitySnapshot(
        symbol=123,  # type: ignore[arg-type]
        signal_strength=0.1,
        momentum_5m=0.1,
        volatility_30m=0.2,
        spread_bps=0.3,
        fee_bps=0.2,
        slippage_bps=0.1,
        liquidity_score=0.8,
        risk_penalty_bps=0.1,
        realized_return_bps=1.2,
    )
    with pytest.raises(ValueError, match="Nieprawidłowy symbol"):
        engine.fit(samples)


def test_fit_rejects_non_finite_target_with_finite_features() -> None:
    engine = TradingOpportunityAI()
    samples = _build_samples()
    samples[0] = OpportunitySnapshot(
        symbol="BNB/USDT",
        signal_strength=0.1,
        momentum_5m=0.1,
        volatility_30m=0.2,
        spread_bps=0.3,
        fee_bps=0.2,
        slippage_bps=0.1,
        liquidity_score=0.8,
        risk_penalty_bps=0.1,
        realized_return_bps=float("inf"),
    )
    with pytest.raises(ValueError, match="Niefinitywny target"):
        engine.fit(samples)


def test_rank_rejects_invalid_candidate_and_thresholds() -> None:
    engine = TradingOpportunityAI()
    engine.fit(_build_samples())

    invalid_candidate = OpportunityCandidate(
        symbol="BTC/USDT",
        signal_strength=float("-inf"),
        momentum_5m=0.1,
        volatility_30m=0.2,
        spread_bps=0.3,
        fee_bps=0.2,
        slippage_bps=0.1,
        liquidity_score=0.8,
        risk_penalty_bps=0.1,
    )
    with pytest.raises(ValueError, match="Niefinitywna cecha"):
        engine.rank([invalid_candidate])

    blank_symbol_candidate = OpportunityCandidate(
        symbol="   ",
        signal_strength=0.1,
        momentum_5m=0.1,
        volatility_30m=0.2,
        spread_bps=0.3,
        fee_bps=0.2,
        slippage_bps=0.1,
        liquidity_score=0.8,
        risk_penalty_bps=0.1,
    )
    with pytest.raises(ValueError, match="Nieprawidłowy symbol"):
        engine.rank([blank_symbol_candidate])

    none_symbol_candidate = OpportunityCandidate(
        symbol=None,  # type: ignore[arg-type]
        signal_strength=0.1,
        momentum_5m=0.1,
        volatility_30m=0.2,
        spread_bps=0.3,
        fee_bps=0.2,
        slippage_bps=0.1,
        liquidity_score=0.8,
        risk_penalty_bps=0.1,
    )
    with pytest.raises(ValueError, match="Nieprawidłowy symbol"):
        engine.rank([none_symbol_candidate])

    non_numeric_candidate = OpportunityCandidate(
        symbol="DOT/USDT",
        signal_strength="bad",  # type: ignore[arg-type]
        momentum_5m=0.1,
        volatility_30m=0.2,
        spread_bps=0.3,
        fee_bps=0.2,
        slippage_bps=0.1,
        liquidity_score=0.8,
        risk_penalty_bps=0.1,
    )
    with pytest.raises(ValueError, match="Niefinitywna cecha"):
        engine.rank([non_numeric_candidate])

    valid_candidate = OpportunityCandidate(
        symbol="SOL/USDT",
        signal_strength=0.1,
        momentum_5m=0.1,
        volatility_30m=0.2,
        spread_bps=0.3,
        fee_bps=0.2,
        slippage_bps=0.1,
        liquidity_score=0.8,
        risk_penalty_bps=0.1,
    )
    with pytest.raises(ValueError, match="min_probability poza zakresem"):
        engine.rank([valid_candidate], min_probability=1.1)
    with pytest.raises(ValueError, match="Nieprawidłowy min_probability"):
        engine.rank([valid_candidate], min_probability=float("nan"))
    with pytest.raises(ValueError, match="Nieprawidłowy min_expected_edge_bps"):
        engine.rank([valid_candidate], min_expected_edge_bps=float("inf"))


def test_rank_handles_extreme_inputs_without_probability_overflow() -> None:
    engine = TradingOpportunityAI()
    engine.fit(_build_samples())
    assert engine.artifact is not None
    hacked_metadata = dict(engine.artifact.metadata)
    hacked_metadata["probability_scale_bps"] = float("nan")
    engine._artifact = ModelArtifact(
        feature_names=engine.artifact.feature_names,
        model_state=engine.artifact.model_state,
        trained_at=engine.artifact.trained_at,
        metrics=engine.artifact.metrics,
        metadata=hacked_metadata,
        target_scale=engine.artifact.target_scale,
        training_rows=engine.artifact.training_rows,
        validation_rows=engine.artifact.validation_rows,
        test_rows=engine.artifact.test_rows,
        feature_scalers=engine.artifact.feature_scalers,
        decision_journal_entry_id=engine.artifact.decision_journal_entry_id,
        backend=engine.artifact.backend,
    )
    decision = engine.rank(
        [
            OpportunityCandidate(
                symbol="EXT/USDT",
                signal_strength=1e9,
                momentum_5m=1e9,
                volatility_30m=1e-9,
                spread_bps=0.0,
                fee_bps=0.0,
                slippage_bps=0.0,
                liquidity_score=1e9,
                risk_penalty_bps=0.0,
            )
        ]
    )[0]
    assert 0.0 <= decision.success_probability <= 1.0
    assert math.isfinite(decision.confidence)


def test_rank_uses_classifier_probability_when_available() -> None:
    engine = TradingOpportunityAI()
    engine.fit(_build_samples())
    decision = engine.rank(
        [
            OpportunityCandidate(
                symbol="ADA/USDT",
                signal_strength=0.5,
                momentum_5m=0.3,
                volatility_30m=0.2,
                spread_bps=0.4,
                fee_bps=0.2,
                slippage_bps=0.1,
                liquidity_score=0.9,
                risk_penalty_bps=0.0,
            )
        ]
    )[0]
    provenance = decision.provenance
    assert provenance["probability_method"] == "model_success_classifier"
    assert provenance["confidence_method"] == "distance_from_probability_midpoint"
    assert isinstance(provenance["calibration"], dict)
    assert provenance["calibration"]["type"] == "classifier"  # type: ignore[index]


def test_rank_falls_back_to_heuristic_probability_without_classifier() -> None:
    engine = TradingOpportunityAI()
    engine.fit(_build_samples())
    assert engine.artifact is not None
    model_state_without_classifier = {
        key: value
        for key, value in dict(engine.artifact.model_state).items()
        if key != "classifier_head_state"
    }
    engine._artifact = ModelArtifact(
        feature_names=engine.artifact.feature_names,
        model_state=model_state_without_classifier,
        trained_at=engine.artifact.trained_at,
        metrics=engine.artifact.metrics,
        metadata=engine.artifact.metadata,
        target_scale=engine.artifact.target_scale,
        training_rows=engine.artifact.training_rows,
        validation_rows=engine.artifact.validation_rows,
        test_rows=engine.artifact.test_rows,
        feature_scalers=engine.artifact.feature_scalers,
        decision_journal_entry_id=engine.artifact.decision_journal_entry_id,
        backend=engine.artifact.backend,
    )
    engine._classifier_model = None

    decision = engine.rank(
        [
            OpportunityCandidate(
                symbol="ADA/USDT",
                signal_strength=0.5,
                momentum_5m=0.3,
                volatility_30m=0.2,
                spread_bps=0.4,
                fee_bps=0.2,
                slippage_bps=0.1,
                liquidity_score=0.9,
                risk_penalty_bps=0.0,
            )
        ]
    )[0]
    provenance = decision.provenance
    assert provenance["probability_method"] == "heuristic_sigmoid_scaled_edge_fallback"
    assert provenance["calibration"]["type"] == "heuristic_fallback"  # type: ignore[index]
    assert math.isfinite(float(provenance["calibration"]["scale_bps"]))  # type: ignore[index]


def test_temporal_evaluator_uses_classifier_probability_when_available() -> None:
    engine = TradingOpportunityAI()
    artifact = engine.fit(_build_samples())
    evaluator = OpportunityTemporalEvaluator()

    evaluation = evaluator.evaluate(artifact, _build_samples())

    assert evaluation.sample_count == 30
    assert evaluation.probability_method == "model_success_classifier"
    assert 0.0 <= evaluation.avg_success_probability <= 1.0
    assert evaluation.success_probability_brier >= 0.0


def test_temporal_evaluator_falls_back_to_heuristic_for_legacy_artifact() -> None:
    engine = TradingOpportunityAI()
    artifact = engine.fit(_build_samples())
    legacy_model_state = {
        key: value for key, value in dict(artifact.model_state).items() if key != "classifier_head_state"
    }
    legacy_artifact = ModelArtifact(
        feature_names=artifact.feature_names,
        model_state=legacy_model_state,
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

    evaluation = evaluator.evaluate(legacy_artifact, _build_samples())

    assert evaluation.probability_method == "heuristic_sigmoid_scaled_edge_fallback"
    assert 0.0 <= evaluation.avg_success_probability <= 1.0


def test_save_load_keeps_classifier_based_ranking_and_evaluation(tmp_path) -> None:
    repository = FilesystemModelRepository(tmp_path / "repo")
    trainer = TradingOpportunityAI(repository=repository)
    original_artifact = trainer.fit(_build_samples())
    trainer.save_model(version="dual-head-eval", activate=True)

    reloaded = TradingOpportunityAI(repository=repository)
    reloaded_artifact = reloaded.load_model("dual-head-eval")
    evaluator = OpportunityTemporalEvaluator()
    sample = OpportunityCandidate(
        symbol="AVAX/USDT",
        signal_strength=0.7,
        momentum_5m=0.5,
        volatility_30m=0.3,
        spread_bps=0.4,
        fee_bps=0.2,
        slippage_bps=0.1,
        liquidity_score=0.85,
        risk_penalty_bps=0.0,
    )

    original_decision = trainer.rank([sample])[0]
    reloaded_decision = reloaded.rank([sample])[0]
    original_eval = evaluator.evaluate(original_artifact, _build_samples())
    reloaded_eval = evaluator.evaluate(reloaded_artifact, _build_samples())

    assert original_decision.provenance["probability_method"] == "model_success_classifier"
    assert reloaded_decision.provenance["probability_method"] == "model_success_classifier"
    assert reloaded_decision.success_probability == pytest.approx(
        original_decision.success_probability,
        rel=1e-12,
        abs=1e-12,
    )
    assert reloaded_eval.probability_method == "model_success_classifier"
    assert reloaded_eval.avg_success_probability == pytest.approx(
        original_eval.avg_success_probability,
        rel=1e-12,
        abs=1e-12,
    )


def test_temporal_evaluator_model_comparison_uses_matching_probability_method() -> None:
    engine = TradingOpportunityAI()
    latest_artifact = engine.fit(_build_samples(scale=1.0))
    previous_artifact = engine.fit(_build_samples(scale=-1.0))
    evaluator = OpportunityTemporalEvaluator()

    comparison = evaluator.compare_latest_vs_previous(
        latest_artifact=latest_artifact,
        previous_artifact=previous_artifact,
        samples=_build_samples(),
    )

    assert comparison.latest.probability_method == "model_success_classifier"
    assert comparison.previous.probability_method == "model_success_classifier"
    assert math.isfinite(comparison.delta_edge_mae_bps)
    assert math.isfinite(comparison.delta_success_probability_brier)


def test_temporal_evaluator_model_comparison_handles_legacy_previous_artifact() -> None:
    engine = TradingOpportunityAI()
    latest_artifact = engine.fit(_build_samples(scale=1.0))
    previous_full = engine.fit(_build_samples(scale=-1.0))
    legacy_previous = ModelArtifact(
        feature_names=previous_full.feature_names,
        model_state={
            key: value
            for key, value in dict(previous_full.model_state).items()
            if key != "classifier_head_state"
        },
        trained_at=previous_full.trained_at,
        metrics=previous_full.metrics,
        metadata=previous_full.metadata,
        target_scale=previous_full.target_scale,
        training_rows=previous_full.training_rows,
        validation_rows=previous_full.validation_rows,
        test_rows=previous_full.test_rows,
        feature_scalers=previous_full.feature_scalers,
        decision_journal_entry_id=previous_full.decision_journal_entry_id,
        backend=previous_full.backend,
    )
    evaluator = OpportunityTemporalEvaluator()

    comparison = evaluator.compare_latest_vs_previous(
        latest_artifact=latest_artifact,
        previous_artifact=legacy_previous,
        samples=_build_samples(),
    )

    assert comparison.latest.probability_method == "model_success_classifier"
    assert comparison.previous.probability_method == "heuristic_sigmoid_scaled_edge_fallback"


def test_shadow_record_building_from_ranked_decisions() -> None:
    engine = TradingOpportunityAI()
    engine.fit(_build_samples())
    decisions = engine.rank(
        [
            OpportunityCandidate(
                symbol="BTC/USDT",
                signal_strength=0.9,
                momentum_5m=0.6,
                volatility_30m=0.3,
                spread_bps=0.4,
                fee_bps=0.2,
                slippage_bps=0.1,
                liquidity_score=0.9,
                risk_penalty_bps=0.1,
            )
        ]
    )
    timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    records = TradingOpportunityAI.build_shadow_records(
        decisions,
        decision_timestamp=timestamp,
        threshold_config=OpportunityThresholdConfig(min_expected_edge_bps=2.0, min_probability=0.6),
        snapshot={"market_regime": "range"},
        context=OpportunityShadowContext(run_id="test-run-1", environment="shadow"),
    )
    assert len(records) == 1
    record = records[0]
    assert record.symbol == decisions[0].symbol
    assert record.record_key
    assert record.threshold_config.min_probability == 0.6
    assert record.snapshot["market_regime"] == "range"
    assert record.context.run_id == "test-run-1"


def test_shadow_and_outcome_contracts_roundtrip_with_repository(tmp_path) -> None:
    repo = OpportunityShadowRepository(tmp_path / "shadow")
    decision_timestamp = datetime(2026, 1, 2, 10, 30, tzinfo=timezone.utc)
    record_key = OpportunityShadowRecord.build_record_key(
        symbol="ETH/USDT",
        decision_timestamp=decision_timestamp,
        model_version="opp-v1",
        rank=1,
    )
    records = TradingOpportunityAI.build_shadow_records(
        [
            OpportunityDecision(
                symbol="ETH/USDT",
                decision_source="model",
                model_version="opp-v1",
                expected_edge_bps=4.2,
                success_probability=0.63,
                confidence=0.26,
                proposed_direction="long",
                accepted=True,
                rejection_reason=None,
                rank=1,
                provenance={"trained_at": "2026-01-01T00:00:00+00:00"},
            )
        ],
        decision_timestamp=decision_timestamp,
    )
    assert records[0].record_key == record_key
    repo.append_shadow_records(records)

    label = OpportunityOutcomeLabel(
        symbol="ETH/USDT",
        decision_timestamp=decision_timestamp,
        correlation_key=record_key,
        horizon_minutes=30,
        realized_return_bps=3.5,
        max_favorable_excursion_bps=5.1,
        max_adverse_excursion_bps=-1.8,
        hit_take_profit=True,
        hit_stop_loss=False,
        provenance={"feed": "backtest"},
        label_quality="high",
    )
    repo.append_outcome_labels([label])

    loaded_records = repo.load_shadow_records()
    loaded_labels = repo.load_outcome_labels()
    assert loaded_records == records
    assert loaded_labels == [label]


def test_record_key_is_canonical_for_same_instant_in_different_timezones() -> None:
    utc_instant = datetime(2026, 1, 2, 10, 30, tzinfo=timezone.utc)
    plus_two = utc_instant.astimezone(timezone(timedelta(hours=2)))
    minus_five = utc_instant.astimezone(timezone(timedelta(hours=-5)))

    key_utc = OpportunityShadowRecord.build_record_key(
        symbol="BTC/USDT",
        decision_timestamp=utc_instant,
        model_version="opp-v1",
        rank=1,
    )
    key_plus_two = OpportunityShadowRecord.build_record_key(
        symbol="BTC/USDT",
        decision_timestamp=plus_two,
        model_version="opp-v1",
        rank=1,
    )
    key_minus_five = OpportunityShadowRecord.build_record_key(
        symbol="BTC/USDT",
        decision_timestamp=minus_five,
        model_version="opp-v1",
        rank=1,
    )
    assert key_utc == key_plus_two == key_minus_five


def test_record_key_stable_after_persistence_roundtrip(tmp_path) -> None:
    repo = OpportunityShadowRepository(tmp_path / "shadow")
    decision_timestamp = datetime(2026, 3, 10, 9, 15, tzinfo=timezone.utc)
    record = TradingOpportunityAI.build_shadow_records(
        [
            OpportunityDecision(
                symbol="SOL/USDT",
                decision_source="model",
                model_version="opp-v2",
                expected_edge_bps=2.3,
                success_probability=0.58,
                confidence=0.16,
                proposed_direction="long",
                accepted=True,
                rejection_reason=None,
                rank=1,
                provenance={"source": "unit_test"},
            )
        ],
        decision_timestamp=decision_timestamp,
    )[0]
    repo.append_shadow_records([record])
    loaded = repo.load_shadow_records()[0]
    recomputed = OpportunityShadowRecord.build_record_key(
        symbol=loaded.symbol,
        decision_timestamp=loaded.decision_timestamp,
        model_version=loaded.model_version,
        rank=loaded.rank,
    )
    assert loaded.record_key == record.record_key
    assert loaded.record_key == recomputed


def test_from_dict_accepts_only_real_bool_fields() -> None:
    decision_timestamp = datetime(2026, 1, 2, 10, 30, tzinfo=timezone.utc)
    record = OpportunityShadowRecord.from_dict(
        {
            "record_key": "rk",
            "symbol": "ETH/USDT",
            "decision_timestamp": decision_timestamp.isoformat(),
            "model_version": "opp-v1",
            "decision_source": "model",
            "expected_edge_bps": 1.2,
            "success_probability": 0.55,
            "confidence": 0.10,
            "proposed_direction": "long",
            "accepted": True,
            "rejection_reason": None,
            "rank": 1,
            "provenance": {},
            "threshold_config": {"min_expected_edge_bps": 0.0, "min_probability": 0.5},
            "snapshot": {},
            "context": {"run_id": None, "environment": "shadow", "notes": {}},
        }
    )
    assert record.accepted is True

    label = OpportunityOutcomeLabel.from_dict(
        {
            "symbol": "ETH/USDT",
            "decision_timestamp": decision_timestamp.isoformat(),
            "correlation_key": "rk",
            "horizon_minutes": 30,
            "realized_return_bps": 2.2,
            "max_favorable_excursion_bps": 3.1,
            "max_adverse_excursion_bps": -1.0,
            "hit_take_profit": False,
            "hit_stop_loss": True,
            "provenance": {},
            "label_quality": "high",
        }
    )
    assert label.hit_take_profit is False
    assert label.hit_stop_loss is True


def test_from_dict_rejects_string_pseudo_booleans() -> None:
    decision_timestamp = datetime(2026, 1, 2, 10, 30, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="accepted"):
        OpportunityShadowRecord.from_dict(
            {
                "record_key": "rk",
                "symbol": "ETH/USDT",
                "decision_timestamp": decision_timestamp.isoformat(),
                "model_version": "opp-v1",
                "decision_source": "model",
                "expected_edge_bps": 1.2,
                "success_probability": 0.55,
                "confidence": 0.10,
                "proposed_direction": "long",
                "accepted": "true",
                "rejection_reason": None,
                "rank": 1,
                "provenance": {},
                "threshold_config": {"min_expected_edge_bps": 0.0, "min_probability": 0.5},
                "snapshot": {},
                "context": {"run_id": None, "environment": "shadow", "notes": {}},
            }
        )
    with pytest.raises(ValueError, match="hit_take_profit"):
        OpportunityOutcomeLabel.from_dict(
            {
                "symbol": "ETH/USDT",
                "decision_timestamp": decision_timestamp.isoformat(),
                "correlation_key": "rk",
                "horizon_minutes": 30,
                "realized_return_bps": 2.2,
                "max_favorable_excursion_bps": 3.1,
                "max_adverse_excursion_bps": -1.0,
                "hit_take_profit": "false",
                "hit_stop_loss": None,
                "provenance": {},
                "label_quality": "high",
            }
        )


def test_build_shadow_records_uses_deep_defensive_copies_for_nested_payloads() -> None:
    engine = TradingOpportunityAI()
    engine.fit(_build_samples())
    decisions = engine.rank(
        [
            OpportunityCandidate(
                symbol="BTC/USDT",
                signal_strength=1.0,
                momentum_5m=0.7,
                volatility_30m=0.2,
                spread_bps=0.3,
                fee_bps=0.1,
                slippage_bps=0.1,
                liquidity_score=0.9,
                risk_penalty_bps=0.0,
            ),
            OpportunityCandidate(
                symbol="ETH/USDT",
                signal_strength=0.8,
                momentum_5m=0.6,
                volatility_30m=0.25,
                spread_bps=0.35,
                fee_bps=0.1,
                slippage_bps=0.1,
                liquidity_score=0.85,
                risk_penalty_bps=0.0,
            ),
        ]
    )
    input_snapshot = {"regime": {"name": "trend", "weights": [1, 2]}}
    input_context = OpportunityShadowContext(
        run_id="run-42",
        environment="shadow",
        notes={"risk": {"bucket": ["low", "mid"]}},
    )
    decisions = [
        OpportunityDecision(
            symbol=decision.symbol,
            decision_source=decision.decision_source,
            model_version=decision.model_version,
            expected_edge_bps=decision.expected_edge_bps,
            success_probability=decision.success_probability,
            confidence=decision.confidence,
            proposed_direction=decision.proposed_direction,
            accepted=decision.accepted,
            rejection_reason=decision.rejection_reason,
            rank=decision.rank,
            provenance={
                "meta": {"source": decision.provenance.get("objective", "unknown")},
                "flags": ["a", "b"],
            },
        )
        for decision in decisions
    ]
    records = TradingOpportunityAI.build_shadow_records(
        decisions,
        snapshot=input_snapshot,
        context=input_context,
    )

    records[0].snapshot["regime"]["name"] = "mutated"  # type: ignore[index]
    records[0].snapshot["regime"]["weights"].append(999)  # type: ignore[index]
    records[0].context.notes["risk"]["bucket"].append("high")  # type: ignore[index]
    records[0].provenance["meta"]["source"] = "mutated"  # type: ignore[index]
    records[0].provenance["flags"].append("c")  # type: ignore[index]

    assert records[1].snapshot["regime"]["name"] == "trend"
    assert records[1].snapshot["regime"]["weights"] == [1, 2]
    assert records[1].context.notes["risk"]["bucket"] == ["low", "mid"]
    assert records[1].provenance["meta"]["source"] != "mutated"
    assert records[1].provenance["flags"] == ["a", "b"]


def test_build_shadow_records_normalizes_timestamp_to_utc_in_memory() -> None:
    decision_timestamp = datetime(
        2026,
        4,
        1,
        12,
        0,
        tzinfo=timezone(timedelta(hours=2)),
    )
    records = TradingOpportunityAI.build_shadow_records(
        [
            OpportunityDecision(
                symbol="BTC/USDT",
                decision_source="model",
                model_version="opp-v3",
                expected_edge_bps=1.0,
                success_probability=0.51,
                confidence=0.02,
                proposed_direction="long",
                accepted=True,
                rejection_reason=None,
                rank=1,
                provenance={},
            )
        ],
        decision_timestamp=decision_timestamp,
    )
    assert records[0].decision_timestamp.tzinfo == timezone.utc
    assert records[0].decision_timestamp.isoformat() == "2026-04-01T10:00:00+00:00"
