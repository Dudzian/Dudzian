"""Integration tests for the sequential AI training pipeline."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from bot_core.ai.feature_engineering import FeatureDataset, FeatureVector
from bot_core.ai.sequential import (
    BUILTIN_HEURISTICS,
    build_heuristic_registry,
    HistoricalFeatureRepository,
    select_heuristics,
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
        heuristic_names=("momentum", "volatility"),
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
    heuristics_used = report.artifact.metadata.get("heuristics_used")
    assert heuristics_used is not None
    assert set(heuristics_used) == set(BUILTIN_HEURISTICS)
    assert set(report.heuristic_weights) == set(BUILTIN_HEURISTICS)
    assert pytest.approx(sum(report.heuristic_weights.values()), rel=1e-6) == 1.0
    assert report.suppressed_heuristics == {}
    assert report.heuristic_confidence
    assert set(report.heuristic_confidence) == set(BUILTIN_HEURISTICS)
    detail = report.artifact.metadata.get("heuristics_detail")
    assert isinstance(detail, dict)
    assert detail.keys() == set(BUILTIN_HEURISTICS)
    suppressed_meta = report.artifact.metadata.get("heuristics_suppressed", {})
    assert suppressed_meta == {}
    confidence_meta = report.artifact.metadata.get("heuristics_confidence")
    assert isinstance(confidence_meta, dict)
    assert set(confidence_meta) == set(BUILTIN_HEURISTICS)
    for name, value in report.heuristic_confidence.items():
        assert pytest.approx(confidence_meta[name], rel=1e-6) == value


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
        heuristic_names=("momentum",),
        min_probability=0.99,
        heuristic_weights=report.heuristic_weights,
        heuristic_confidence=report.heuristic_confidence,
    ).score(features)
    assert low_confidence.source == "heuristic"
    assert "heuristic_probability" in low_confidence.diagnostics

    confident_features = dataset.features[10]
    confident = SequentialOnlineScorer(
        model=model,
        heuristics=BUILTIN_HEURISTICS,
        heuristic_names=("momentum", "volatility"),
        min_probability=0.4,
        heuristic_weights=report.heuristic_weights,
        heuristic_confidence=report.heuristic_confidence,
    ).score(confident_features)
    assert confident.source == "model"
    assert confident.score.success_probability >= 0.4


def test_heuristic_selection_merges_custom_registry() -> None:
    registry = build_heuristic_registry({"custom": lambda features: features.get("bias", 0.0)})
    selected = select_heuristics(["custom", "momentum"], registry=registry)
    assert set(selected) == {"custom", "momentum"}


def test_pipeline_rejects_unknown_heuristic(tmp_path) -> None:
    repository = HistoricalFeatureRepository((tmp_path / "repo3").resolve())
    with pytest.raises(ValueError):
        SequentialTrainingPipeline(repository=repository, heuristic_names=("unknown",))


def test_select_heuristics_raises_for_unknown() -> None:
    with pytest.raises(ValueError):
        select_heuristics(["ghost"])


def test_online_scorer_applies_heuristic_weights() -> None:
    heuristics = {
        "strong": lambda _: 20.0,
        "weak": lambda _: -10.0,
        "ignored": lambda _: -100.0,
    }
    scorer = SequentialOnlineScorer(
        model=None,
        heuristics=heuristics,
        heuristic_names=("strong", "weak", "ignored"),
        heuristic_weights={"strong": 0.9, "weak": 0.1, "ignored": 0.0},
        min_probability=0.6,
        heuristic_confidence={"strong": 0.9, "weak": 0.55, "ignored": 0.4},
    )
    result = scorer.score({})
    assert result.source == "heuristic"
    assert result.score.expected_return_bps > 0.0
    expected_probability = max(0.5, min((0.9 * 0.9 + 0.1 * 0.55), 0.99))
    assert pytest.approx(result.score.success_probability, rel=1e-6) == expected_probability
    diagnostics = dict(result.diagnostics)
    assert "heuristic_probability" in diagnostics
    assert pytest.approx(diagnostics["heuristic_probability"], rel=1e-6) == expected_probability
    assert pytest.approx(diagnostics["heuristic_confidence.strong"], rel=1e-6) == 0.9
    assert pytest.approx(diagnostics["heuristic_confidence.weak"], rel=1e-6) == 0.55


def test_pipeline_suppresses_underperforming_heuristics(tmp_path) -> None:
    dataset = _build_synthetic_dataset()
    repository = HistoricalFeatureRepository((tmp_path / "repo4").resolve())
    bad_heuristic = lambda features: -float(features.get("momentum_1", 0.0))
    heuristics = dict(BUILTIN_HEURISTICS)
    heuristics["inverse_momentum"] = bad_heuristic
    pipeline = SequentialTrainingPipeline(
        repository=repository,
        heuristics=heuristics,
        heuristic_names=("momentum", "inverse_momentum"),
        min_directional_accuracy=0.9,
    )

    report = pipeline.train_offline(
        dataset,
        top_k_features=3,
        folds=5,
        learning_rate=0.08,
        discount_factor=0.85,
    )

    assert "inverse_momentum" in report.heuristic_weights
    assert report.heuristic_weights["inverse_momentum"] == 0.0
    assert "inverse_momentum" in report.suppressed_heuristics
    suppressed_meta = report.artifact.metadata.get("heuristics_suppressed")
    assert suppressed_meta is not None
    assert pytest.approx(
        suppressed_meta["inverse_momentum"], rel=1e-6
    ) == report.suppressed_heuristics["inverse_momentum"]
    assert set(report.artifact.metadata["heuristics_used"]) == {"momentum"}
    confidence_meta = report.artifact.metadata.get("heuristics_confidence", {})
    assert confidence_meta
    assert "inverse_momentum" in confidence_meta
    assert pytest.approx(confidence_meta["inverse_momentum"], rel=1e-6) == report.heuristic_confidence[
        "inverse_momentum"
    ]

    scorer = SequentialOnlineScorer(
        model=None,
        heuristics=heuristics,
        heuristic_names=("momentum", "inverse_momentum"),
        heuristic_weights=report.heuristic_weights,
        heuristic_confidence=report.heuristic_confidence,
    )
    sample_features = dataset.features[0]
    fallback = scorer.score(sample_features)
    expected = BUILTIN_HEURISTICS["momentum"](sample_features)
    assert fallback.source == "heuristic"
    assert pytest.approx(fallback.score.expected_return_bps, rel=1e-6) == expected


def test_online_scorer_uses_trained_heuristic_confidence(tmp_path) -> None:
    dataset = _build_synthetic_dataset()
    repository = HistoricalFeatureRepository((tmp_path / "repo5").resolve())
    pipeline = SequentialTrainingPipeline(
        repository=repository,
        heuristics=BUILTIN_HEURISTICS,
        heuristic_names=tuple(BUILTIN_HEURISTICS),
    )

    report = pipeline.train_offline(dataset, top_k_features=3, folds=4)
    scorer = SequentialOnlineScorer(
        model=None,
        heuristics=BUILTIN_HEURISTICS,
        heuristic_names=tuple(BUILTIN_HEURISTICS),
        heuristic_weights=report.heuristic_weights,
        heuristic_confidence=report.heuristic_confidence,
    )

    features = dataset.features[5]
    result = scorer.score(features)
    assert result.source == "heuristic"
    positive_weights = {
        name: weight for name, weight in report.heuristic_weights.items() if weight > 0.0
    }
    total_weight = sum(positive_weights.values())
    if total_weight > 0.0:
        weighted_confidence = sum(
            report.heuristic_confidence.get(name, 0.0) * weight
            for name, weight in positive_weights.items()
        )
        expected_probability = max(0.5, min(weighted_confidence / total_weight, 0.99))
    else:
        expected_probability = 0.5

    assert pytest.approx(result.score.success_probability, rel=1e-6) == expected_probability
    diagnostics = dict(result.diagnostics)
    assert pytest.approx(diagnostics["heuristic_probability"], rel=1e-6) == expected_probability
    for name, weight in positive_weights.items():
        key = f"heuristic_confidence.{name}"
        assert key in diagnostics
        assert pytest.approx(diagnostics[key], rel=1e-6) == report.heuristic_confidence[name]
