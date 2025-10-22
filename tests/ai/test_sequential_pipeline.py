"""Integration tests for the sequential AI training pipeline."""

from __future__ import annotations

import math
from datetime import datetime, timezone

from bot_core.ai.feature_engineering import FeatureDataset, FeatureVector
from bot_core.ai.sequential import (
    BUILTIN_HEURISTICS,
    HistoricalFeatureRepository,
    SequentialOnlineScorer,
    SequentialTrainingPipeline,
)


def _build_synthetic_dataset() -> FeatureDataset:
    vectors: list[FeatureVector] = []
    start = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
    for idx in range(80):
        momentum = math.sin(idx / 4.0)
        lagged = math.sin((idx - 1) / 4.0)
        volatility = 0.4 + 0.2 * abs(math.cos(idx / 6.0))
        features = {
            "momentum_1": momentum,
            "return_1": lagged,
            "volatility_1": volatility,
        }
        target = 200.0 * (0.9 * lagged - 0.3 * volatility)
        vectors.append(
            FeatureVector(
                timestamp=start + idx * 3600.0,
                symbol="BTC_USDT",
                features=features,
                target_bps=target,
            )
        )
    metadata = {
        "symbols": ["BTC_USDT"],
        "interval": "1h",
        "start": int(start),
        "end": int(start + 80 * 3600.0),
    }
    return FeatureDataset(vectors=tuple(vectors), metadata=metadata)


def test_sequential_pipeline_beats_heuristics(tmp_path) -> None:
    dataset = _build_synthetic_dataset()
    repository = HistoricalFeatureRepository((tmp_path / "repo").resolve())
    pipeline = SequentialTrainingPipeline(
        repository=repository,
        heuristics=BUILTIN_HEURISTICS,
        min_directional_accuracy=0.55,
    )

    report = pipeline.train_offline(
        dataset,
        top_k_features=3,
        folds=5,
        learning_rate=0.08,
        discount_factor=0.85,
    )

    model_accuracy = sum(report.walk_forward_metrics.directional_accuracy) / len(
        report.walk_forward_metrics.directional_accuracy
    )
    heuristic_accuracy = sum(report.heuristic_metrics.directional_accuracy) / len(
        report.heuristic_metrics.directional_accuracy
    )

    assert model_accuracy > heuristic_accuracy + 0.02
    assert report.artifact.backend == "sequential_td"
    assert repository.list(), "training should persist dataset in repository"


def test_online_scorer_uses_fallback_when_probability_low(tmp_path) -> None:
    dataset = _build_synthetic_dataset()
    repository = HistoricalFeatureRepository((tmp_path / "repo2").resolve())
    pipeline = SequentialTrainingPipeline(
        repository=repository,
        heuristics=BUILTIN_HEURISTICS,
    )
    report = pipeline.train_offline(dataset, top_k_features=3, folds=4)
    model = report.artifact.build_model()

    base_features = dataset.features[10]
    features = {name: -float(value) for name, value in base_features.items()}
    features["volatility_1"] = abs(base_features.get("volatility_1", 1.0)) * 3.0
    low_confidence = SequentialOnlineScorer(
        model=model,
        heuristics=BUILTIN_HEURISTICS,
        min_probability=0.99,
    ).score(features)
    assert low_confidence.source == "heuristic"
    assert "heuristic_probability" in low_confidence.diagnostics

    confident_features = dataset.features[10]
    confident = SequentialOnlineScorer(
        model=model,
        heuristics=BUILTIN_HEURISTICS,
        min_probability=0.4,
    ).score(confident_features)
    assert confident.source == "model"
    assert confident.score.success_probability >= 0.4
