from __future__ import annotations

import math

import pytest

from bot_core.ai import (
    FilesystemModelRepository,
    ModelArtifact,
    OpportunityCandidate,
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


def test_rank_provenance_contains_heuristic_probability_markers() -> None:
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
    assert provenance["probability_method"] == "heuristic_sigmoid_scaled_edge"
    assert provenance["confidence_method"] == "distance_from_probability_midpoint"
    assert isinstance(provenance["calibration"], dict)
    assert math.isfinite(float(provenance["calibration"]["scale_bps"]))  # type: ignore[index]
