"""Tests for dynamic ensemble weighting in AI manager."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bot_core.ai.manager import AIManager, EnsembleDefinition
from bot_core.ai.pipeline import train_gradient_boosting_model
from bot_core.ai.regime import MarketRegime, MarketRegimeAssessment


def test_ensemble_definition_resolve_weights_with_regime_and_meta() -> None:
    definition = EnsembleDefinition(
        name="combo",
        components=("alpha", "beta"),
        aggregation="weighted",
        weights=(0.5, 0.5),
        regime_weights={"trend": (0.8, 0.2)},
        meta_weight_floor=0.1,
    )
    meta_scores = {"alpha": 0.2, "beta": 0.8}
    weights = definition.resolve_weights(
        regime=MarketRegime.TREND,
        meta_confidence=meta_scores,
    )
    assert np.isclose(weights.sum(), 1.0)
    # base regime weights 0.8/0.2 combined with meta scores 0.2/0.8 -> equalised
    assert np.allclose(weights, np.array([0.5, 0.5]), atol=1e-6)


def test_manager_aggregate_ensemble_uses_dynamic_weights(tmp_path: Path) -> None:
    manager = AIManager(model_dir=tmp_path)
    definition = manager.register_ensemble(
        "combo",
        ["alpha", "beta"],
        aggregation="weighted",
        weights=[0.6, 0.4],
        regime_weights={"trend": (0.9, 0.1)},
        meta_weight_floor=0.05,
        override=True,
    )

    class DummyInference:
        def __init__(self, confidence: float) -> None:
            self.meta_labeling_confidence = confidence

    manager._repository_models[manager._model_key("btc", "alpha")] = DummyInference(0.3)
    manager._repository_models[manager._model_key("btc", "beta")] = DummyInference(0.9)
    manager._latest_regimes["btc"] = MarketRegimeAssessment(
        regime=MarketRegime.TREND,
        confidence=0.8,
        risk_score=0.2,
        metrics={},
        symbol="btc",
    )

    series_alpha = pd.Series([0.1, 0.2, 0.3])
    series_beta = pd.Series([0.4, 0.5, 0.6])
    combined = manager._aggregate_ensemble_predictions(  # type: ignore[protected-access]
        (series_alpha, series_beta),
        definition,
        regime=MarketRegime.TREND,
        meta_scores=manager._collect_meta_confidence("btc", definition.components),  # type: ignore[protected-access]
    )
    base = np.asarray([0.9, 0.1], dtype=float)
    meta = np.asarray([0.3, 0.9], dtype=float)
    expected_weights = (base * meta) / np.sum(base * meta)
    expected = series_alpha.to_numpy() * expected_weights[0] + series_beta.to_numpy() * expected_weights[1]
    assert np.allclose(combined.to_numpy(), expected)


def test_manager_resolve_meta_confidence_from_repository_metadata(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "feature": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
            "target": [0.2, -0.3, 0.4, -0.5, 0.6, -0.7],
        }
    )
    artifact_path = train_gradient_boosting_model(
        frame,
        ["feature"],
        "target",
        output_dir=tmp_path,
        model_name="swing",
        metadata={"symbol": "btc", "model_type": "swing"},
    )

    manager = AIManager(model_dir=tmp_path)
    manager.ingest_model_repository(tmp_path)

    key = manager._model_key("btc", "swing")  # type: ignore[protected-access]
    manager._repository_models.pop(key, None)  # type: ignore[protected-access]
    manager._meta_confidence_cache.pop(key, None)  # type: ignore[protected-access]

    confidence = manager._resolve_meta_confidence("btc", "swing")  # type: ignore[protected-access]
    assert confidence is not None
    assert 0.0 <= confidence <= 1.0
