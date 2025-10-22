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
    assert report.heuristic_trend
    assert set(report.heuristic_trend) == set(BUILTIN_HEURISTICS)
    assert report.heuristic_volatility
    assert set(report.heuristic_volatility) == set(BUILTIN_HEURISTICS)
    assert report.heuristic_consistency
    assert set(report.heuristic_consistency) == set(BUILTIN_HEURISTICS)
    assert report.heuristic_drift
    assert set(report.heuristic_drift) == set(BUILTIN_HEURISTICS)
    assert report.heuristic_correlation
    assert set(report.heuristic_correlation) == set(BUILTIN_HEURISTICS)
    assert report.heuristic_sharpe
    assert set(report.heuristic_sharpe) == set(BUILTIN_HEURISTICS)
    assert report.heuristic_sortino
    assert set(report.heuristic_sortino) == set(BUILTIN_HEURISTICS)
    assert report.heuristic_omega
    assert set(report.heuristic_omega) == set(BUILTIN_HEURISTICS)
    assert report.heuristic_calmar
    assert set(report.heuristic_calmar) == set(BUILTIN_HEURISTICS)
    assert report.heuristic_sterling
    assert set(report.heuristic_sterling) == set(BUILTIN_HEURISTICS)
    assert report.heuristic_burke
    assert set(report.heuristic_burke) == set(BUILTIN_HEURISTICS)
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
    trend_meta = report.artifact.metadata.get("heuristics_trend")
    assert isinstance(trend_meta, dict)
    assert set(trend_meta) == set(BUILTIN_HEURISTICS)
    for name, value in report.heuristic_trend.items():
        assert pytest.approx(trend_meta[name], rel=1e-6) == value
    volatility_meta = report.artifact.metadata.get("heuristics_volatility")
    assert isinstance(volatility_meta, dict)
    assert set(volatility_meta) == set(BUILTIN_HEURISTICS)
    for name, value in report.heuristic_volatility.items():
        assert pytest.approx(volatility_meta[name], rel=1e-6) == value
    consistency_meta = report.artifact.metadata.get("heuristics_consistency")
    assert isinstance(consistency_meta, dict)
    assert set(consistency_meta) == set(BUILTIN_HEURISTICS)
    for name, value in report.heuristic_consistency.items():
        assert pytest.approx(consistency_meta[name], rel=1e-6) == value
    drift_meta = report.artifact.metadata.get("heuristics_drift")
    assert isinstance(drift_meta, dict)
    assert set(drift_meta) == set(BUILTIN_HEURISTICS)
    for name, value in report.heuristic_drift.items():
        assert pytest.approx(drift_meta[name], rel=1e-6) == value
    correlation_meta = report.artifact.metadata.get("heuristics_correlation")
    assert isinstance(correlation_meta, dict)
    assert set(correlation_meta) == set(BUILTIN_HEURISTICS)
    for name, value in report.heuristic_correlation.items():
        assert pytest.approx(correlation_meta[name], rel=1e-6) == value
    sharpe_meta = report.artifact.metadata.get("heuristics_sharpe")
    assert isinstance(sharpe_meta, dict)
    assert set(sharpe_meta) == set(BUILTIN_HEURISTICS)
    for name, value in report.heuristic_sharpe.items():
        assert pytest.approx(sharpe_meta[name], rel=1e-6) == value
    sortino_meta = report.artifact.metadata.get("heuristics_sortino")
    assert isinstance(sortino_meta, dict)
    assert set(sortino_meta) == set(BUILTIN_HEURISTICS)
    for name, value in report.heuristic_sortino.items():
        assert pytest.approx(sortino_meta[name], rel=1e-6) == value
    omega_meta = report.artifact.metadata.get("heuristics_omega")
    assert isinstance(omega_meta, dict)
    assert set(omega_meta) == set(BUILTIN_HEURISTICS)
    for name, value in report.heuristic_omega.items():
        assert pytest.approx(omega_meta[name], rel=1e-6) == value
    calmar_meta = report.artifact.metadata.get("heuristics_calmar")
    assert isinstance(calmar_meta, dict)
    assert set(calmar_meta) == set(BUILTIN_HEURISTICS)
    for name, value in report.heuristic_calmar.items():
        assert pytest.approx(calmar_meta[name], rel=1e-6) == value
    sterling_meta = report.artifact.metadata.get("heuristics_sterling")
    assert isinstance(sterling_meta, dict)
    assert set(sterling_meta) == set(BUILTIN_HEURISTICS)
    for name, value in report.heuristic_sterling.items():
        assert pytest.approx(sterling_meta[name], rel=1e-6) == value
    burke_meta = report.artifact.metadata.get("heuristics_burke")
    assert isinstance(burke_meta, dict)
    assert set(burke_meta) == set(BUILTIN_HEURISTICS)
    for name, value in report.heuristic_burke.items():
        assert pytest.approx(burke_meta[name], rel=1e-6) == value


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
        heuristic_trend=report.heuristic_trend,
        heuristic_volatility=report.heuristic_volatility,
        heuristic_consistency=report.heuristic_consistency,
        heuristic_drift=report.heuristic_drift,
        heuristic_correlation=report.heuristic_correlation,
        heuristic_sharpe=report.heuristic_sharpe,
        heuristic_sortino=report.heuristic_sortino,
        heuristic_omega=report.heuristic_omega,
        heuristic_calmar=report.heuristic_calmar,
        heuristic_sterling=report.heuristic_sterling,
        heuristic_burke=report.heuristic_burke,
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
        heuristic_trend=report.heuristic_trend,
        heuristic_volatility=report.heuristic_volatility,
        heuristic_consistency=report.heuristic_consistency,
        heuristic_drift=report.heuristic_drift,
        heuristic_correlation=report.heuristic_correlation,
        heuristic_sharpe=report.heuristic_sharpe,
        heuristic_sortino=report.heuristic_sortino,
        heuristic_omega=report.heuristic_omega,
        heuristic_calmar=report.heuristic_calmar,
        heuristic_sterling=report.heuristic_sterling,
        heuristic_burke=report.heuristic_burke,
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
    volatility = {"strong": 0.05, "weak": 0.2, "ignored": 0.8}
    consistency = {"strong": 0.9, "weak": 0.6, "ignored": 0.3}
    drift = {"strong": 0.2, "weak": -0.1, "ignored": -0.5}
    correlation = {"strong": 0.55, "weak": 0.15, "ignored": -0.4}
    sharpe = {"strong": 1.4, "weak": 0.2, "ignored": -0.3}
    sortino = {"strong": 1.1, "weak": 0.35, "ignored": -0.4}
    omega = {"strong": 2.5, "weak": 0.85, "ignored": 0.2}
    calmar = {"strong": 1.8, "weak": 0.4, "ignored": -0.6}
    sterling = {"strong": 1.5, "weak": 0.2, "ignored": -0.7}
    burke = {"strong": 1.1, "weak": 0.25, "ignored": -0.8}
    scorer = SequentialOnlineScorer(
        model=None,
        heuristics=heuristics,
        heuristic_names=("strong", "weak", "ignored"),
        heuristic_weights={"strong": 0.9, "weak": 0.1, "ignored": 0.0},
        min_probability=0.6,
        heuristic_confidence={"strong": 0.9, "weak": 0.55, "ignored": 0.4},
        heuristic_trend={"strong": 0.1, "weak": -0.2},
        heuristic_volatility=volatility,
        heuristic_consistency=consistency,
        heuristic_drift=drift,
        heuristic_correlation=correlation,
        heuristic_sharpe=sharpe,
        heuristic_sortino=sortino,
        heuristic_omega=omega,
        heuristic_calmar=calmar,
        heuristic_sterling=sterling,
        heuristic_burke=burke,
    )
    result = scorer.score({})
    assert result.source == "heuristic"
    assert result.score.expected_return_bps > 0.0
    strong_factor = SequentialOnlineScorer._trend_factor(0.1)
    weak_factor = SequentialOnlineScorer._trend_factor(-0.2)
    strong_vol = SequentialOnlineScorer._volatility_factor(volatility["strong"])
    weak_vol = SequentialOnlineScorer._volatility_factor(volatility["weak"])
    strong_consistency = SequentialOnlineScorer._consistency_factor(consistency["strong"])
    weak_consistency = SequentialOnlineScorer._consistency_factor(consistency["weak"])
    strong_drift = SequentialOnlineScorer._drift_factor(drift["strong"])
    weak_drift = SequentialOnlineScorer._drift_factor(drift["weak"])
    strong_corr = SequentialOnlineScorer._correlation_factor(correlation["strong"])
    weak_corr = SequentialOnlineScorer._correlation_factor(correlation["weak"])
    strong_sharpe = SequentialOnlineScorer._sharpe_factor(sharpe["strong"])
    weak_sharpe = SequentialOnlineScorer._sharpe_factor(sharpe["weak"])
    strong_sortino = SequentialOnlineScorer._sortino_factor(sortino["strong"])
    weak_sortino = SequentialOnlineScorer._sortino_factor(sortino["weak"])
    strong_omega = SequentialOnlineScorer._omega_factor(omega["strong"])
    weak_omega = SequentialOnlineScorer._omega_factor(omega["weak"])
    strong_calmar = SequentialOnlineScorer._calmar_factor(calmar["strong"])
    weak_calmar = SequentialOnlineScorer._calmar_factor(calmar["weak"])
    strong_sterling = SequentialOnlineScorer._sterling_factor(sterling["strong"])
    weak_sterling = SequentialOnlineScorer._sterling_factor(sterling["weak"])
    strong_burke = SequentialOnlineScorer._burke_factor(burke["strong"])
    weak_burke = SequentialOnlineScorer._burke_factor(burke["weak"])
    total_weight = (
        0.9
        * strong_factor
        * strong_vol
        * strong_consistency
        * strong_drift
        * strong_corr
        * strong_sharpe
        * strong_sortino
        * strong_omega
        * strong_calmar
        * strong_sterling
        * strong_burke
        + 0.1
        * weak_factor
        * weak_vol
        * weak_consistency
        * weak_drift
        * weak_corr
        * weak_sharpe
        * weak_sortino
        * weak_omega
        * weak_calmar
        * weak_sterling
        * weak_burke
    )
    weighted_confidence = (
        0.9
        * strong_factor
        * strong_vol
        * strong_consistency
        * strong_drift
        * strong_corr
        * strong_sharpe
        * strong_sortino
        * strong_omega
        * strong_calmar
        * strong_sterling
        * strong_burke
        * 0.9
        + 0.1
        * weak_factor
        * weak_vol
        * weak_consistency
        * weak_drift
        * weak_corr
        * weak_sharpe
        * weak_sortino
        * weak_omega
        * weak_calmar
        * weak_sterling
        * weak_burke
        * 0.55
    )
    expected_probability = max(0.5, min(weighted_confidence / total_weight, 0.99))
    assert pytest.approx(result.score.success_probability, rel=1e-6) == expected_probability
    diagnostics = dict(result.diagnostics)
    assert "heuristic_probability" in diagnostics
    assert pytest.approx(diagnostics["heuristic_probability"], rel=1e-6) == expected_probability
    assert pytest.approx(diagnostics["heuristic_confidence.strong"], rel=1e-6) == 0.9
    assert pytest.approx(diagnostics["heuristic_confidence.weak"], rel=1e-6) == 0.55
    assert pytest.approx(diagnostics["heuristic_trend.strong"], rel=1e-6) == 0.1
    assert pytest.approx(diagnostics["heuristic_trend.weak"], rel=1e-6) == -0.2
    assert pytest.approx(diagnostics["heuristic_volatility.strong"], rel=1e-6) == volatility[
        "strong"
    ]
    assert pytest.approx(diagnostics["heuristic_volatility.weak"], rel=1e-6) == volatility[
        "weak"
    ]
    assert pytest.approx(diagnostics["heuristic_consistency.strong"], rel=1e-6) == consistency[
        "strong"
    ]
    assert pytest.approx(diagnostics["heuristic_consistency.weak"], rel=1e-6) == consistency[
        "weak"
    ]
    assert pytest.approx(diagnostics["heuristic_drift.strong"], rel=1e-6) == drift["strong"]
    assert pytest.approx(diagnostics["heuristic_drift.weak"], rel=1e-6) == drift["weak"]
    assert pytest.approx(diagnostics["heuristic_correlation.strong"], rel=1e-6) == correlation[
        "strong"
    ]
    assert pytest.approx(diagnostics["heuristic_correlation.weak"], rel=1e-6) == correlation[
        "weak"
    ]
    assert pytest.approx(diagnostics["heuristic_sharpe.strong"], rel=1e-6) == sharpe["strong"]
    assert pytest.approx(diagnostics["heuristic_sharpe.weak"], rel=1e-6) == sharpe["weak"]
    assert pytest.approx(diagnostics["heuristic_sortino.strong"], rel=1e-6) == sortino["strong"]
    assert pytest.approx(diagnostics["heuristic_sortino.weak"], rel=1e-6) == sortino["weak"]
    assert pytest.approx(diagnostics["heuristic_omega.strong"], rel=1e-6) == omega["strong"]
    assert pytest.approx(diagnostics["heuristic_omega.weak"], rel=1e-6) == omega["weak"]
    assert pytest.approx(diagnostics["heuristic_calmar.strong"], rel=1e-6) == calmar["strong"]
    assert pytest.approx(diagnostics["heuristic_calmar.weak"], rel=1e-6) == calmar["weak"]
    assert pytest.approx(diagnostics["heuristic_sterling.strong"], rel=1e-6) == sterling["strong"]
    assert pytest.approx(diagnostics["heuristic_sterling.weak"], rel=1e-6) == sterling["weak"]
    assert pytest.approx(diagnostics["heuristic_burke.strong"], rel=1e-6) == burke["strong"]
    assert pytest.approx(diagnostics["heuristic_burke.weak"], rel=1e-6) == burke["weak"]


def test_online_scorer_penalizes_negative_trend() -> None:
    heuristics = {"positive": lambda _: 5.0, "negative": lambda _: -5.0}
    weights = {"positive": 0.5, "negative": 0.5}
    confidences = {"positive": 0.9, "negative": 0.6}
    trend = {"positive": 0.3, "negative": -0.45}
    volatility = {"positive": 0.05, "negative": 0.3}
    consistency = {"positive": 0.85, "negative": 0.45}
    drift = {"positive": 0.15, "negative": -0.4}
    correlation = {"positive": 0.5, "negative": -0.2}
    sharpe = {"positive": 1.2, "negative": -0.6}
    sortino = {"positive": 0.9, "negative": -0.45}
    omega = {"positive": 1.8, "negative": 0.4}
    calmar = {"positive": 1.2, "negative": -0.3}
    sterling = {"positive": 1.0, "negative": -0.5}
    burke = {"positive": 0.95, "negative": -0.6}
    scorer = SequentialOnlineScorer(
        model=None,
        heuristics=heuristics,
        heuristic_names=tuple(heuristics),
        heuristic_weights=weights,
        heuristic_confidence=confidences,
        heuristic_trend=trend,
        heuristic_volatility=volatility,
        heuristic_consistency=consistency,
        heuristic_drift=drift,
        heuristic_correlation=correlation,
        heuristic_sharpe=sharpe,
        heuristic_sortino=sortino,
        heuristic_omega=omega,
        heuristic_calmar=calmar,
        heuristic_sterling=sterling,
        heuristic_burke=burke,
        min_probability=0.95,
    )
    result = scorer.score({})
    assert result.source == "heuristic"
    positive_factor = SequentialOnlineScorer._trend_factor(trend["positive"])
    negative_factor = SequentialOnlineScorer._trend_factor(trend["negative"])
    positive_vol = SequentialOnlineScorer._volatility_factor(volatility["positive"])
    negative_vol = SequentialOnlineScorer._volatility_factor(volatility["negative"])
    positive_consistency = SequentialOnlineScorer._consistency_factor(consistency["positive"])
    negative_consistency = SequentialOnlineScorer._consistency_factor(consistency["negative"])
    positive_drift = SequentialOnlineScorer._drift_factor(drift["positive"])
    negative_drift = SequentialOnlineScorer._drift_factor(drift["negative"])
    positive_corr = SequentialOnlineScorer._correlation_factor(correlation["positive"])
    negative_corr = SequentialOnlineScorer._correlation_factor(correlation["negative"])
    positive_sharpe = SequentialOnlineScorer._sharpe_factor(sharpe["positive"])
    negative_sharpe = SequentialOnlineScorer._sharpe_factor(sharpe["negative"])
    positive_sortino = SequentialOnlineScorer._sortino_factor(sortino["positive"])
    negative_sortino = SequentialOnlineScorer._sortino_factor(sortino["negative"])
    positive_omega = SequentialOnlineScorer._omega_factor(omega["positive"])
    negative_omega = SequentialOnlineScorer._omega_factor(omega["negative"])
    positive_calmar = SequentialOnlineScorer._calmar_factor(calmar["positive"])
    negative_calmar = SequentialOnlineScorer._calmar_factor(calmar["negative"])
    positive_sterling = SequentialOnlineScorer._sterling_factor(sterling["positive"])
    negative_sterling = SequentialOnlineScorer._sterling_factor(sterling["negative"])
    positive_burke = SequentialOnlineScorer._burke_factor(burke["positive"])
    negative_burke = SequentialOnlineScorer._burke_factor(burke["negative"])
    total_weight = (
        weights["positive"]
        * positive_factor
        * positive_vol
        * positive_consistency
        * positive_drift
        * positive_corr
        * positive_sharpe
        * positive_sortino
        * positive_omega
        * positive_calmar
        * positive_sterling
        * positive_burke
        + weights["negative"]
        * negative_factor
        * negative_vol
        * negative_consistency
        * negative_drift
        * negative_corr
        * negative_sharpe
        * negative_sortino
        * negative_omega
        * negative_calmar
        * negative_sterling
        * negative_burke
    )
    weighted_confidence = (
        weights["positive"]
        * positive_factor
        * positive_vol
        * positive_consistency
        * positive_drift
        * positive_corr
        * positive_sharpe
        * positive_sortino
        * positive_omega
        * positive_calmar
        * positive_sterling
        * positive_burke
        * confidences["positive"]
        + weights["negative"]
        * negative_factor
        * negative_vol
        * negative_consistency
        * negative_drift
        * negative_corr
        * negative_sharpe
        * negative_sortino
        * negative_omega
        * negative_calmar
        * negative_sterling
        * negative_burke
        * confidences["negative"]
    )
    expected_probability = max(0.5, min(weighted_confidence / total_weight, 0.99))
    assert pytest.approx(result.score.success_probability, rel=1e-6) == expected_probability
    diagnostics = dict(result.diagnostics)
    assert pytest.approx(diagnostics["heuristic_probability"], rel=1e-6) == expected_probability
    assert pytest.approx(diagnostics["heuristic_trend.positive"], rel=1e-6) == trend["positive"]
    assert pytest.approx(diagnostics["heuristic_trend.negative"], rel=1e-6) == trend["negative"]
    assert pytest.approx(diagnostics["heuristic_volatility.positive"], rel=1e-6) == volatility[
        "positive"
    ]
    assert pytest.approx(diagnostics["heuristic_volatility.negative"], rel=1e-6) == volatility[
        "negative"
    ]
    assert pytest.approx(
        diagnostics["heuristic_consistency.positive"], rel=1e-6
    ) == consistency["positive"]
    assert pytest.approx(
        diagnostics["heuristic_consistency.negative"], rel=1e-6
    ) == consistency["negative"]
    assert pytest.approx(diagnostics["heuristic_drift.positive"], rel=1e-6) == drift["positive"]
    assert pytest.approx(diagnostics["heuristic_drift.negative"], rel=1e-6) == drift["negative"]
    assert pytest.approx(diagnostics["heuristic_correlation.positive"], rel=1e-6) == correlation[
        "positive"
    ]
    assert pytest.approx(diagnostics["heuristic_correlation.negative"], rel=1e-6) == correlation[
        "negative"
    ]
    assert pytest.approx(diagnostics["heuristic_sharpe.positive"], rel=1e-6) == sharpe["positive"]
    assert pytest.approx(diagnostics["heuristic_sharpe.negative"], rel=1e-6) == sharpe["negative"]
    assert pytest.approx(diagnostics["heuristic_sortino.positive"], rel=1e-6) == sortino["positive"]
    assert pytest.approx(diagnostics["heuristic_sortino.negative"], rel=1e-6) == sortino["negative"]
    assert pytest.approx(diagnostics["heuristic_omega.positive"], rel=1e-6) == omega["positive"]
    assert pytest.approx(diagnostics["heuristic_omega.negative"], rel=1e-6) == omega["negative"]
    assert pytest.approx(diagnostics["heuristic_calmar.positive"], rel=1e-6) == calmar["positive"]
    assert pytest.approx(diagnostics["heuristic_calmar.negative"], rel=1e-6) == calmar["negative"]
    assert pytest.approx(diagnostics["heuristic_sterling.positive"], rel=1e-6) == sterling["positive"]
    assert pytest.approx(diagnostics["heuristic_sterling.negative"], rel=1e-6) == sterling["negative"]
    assert pytest.approx(diagnostics["heuristic_burke.positive"], rel=1e-6) == burke["positive"]
    assert pytest.approx(diagnostics["heuristic_burke.negative"], rel=1e-6) == burke["negative"]


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
        heuristic_trend=report.heuristic_trend,
        heuristic_volatility=report.heuristic_volatility,
        heuristic_consistency=report.heuristic_consistency,
        heuristic_drift=report.heuristic_drift,
        heuristic_correlation=report.heuristic_correlation,
        heuristic_sharpe=report.heuristic_sharpe,
        heuristic_sortino=report.heuristic_sortino,
        heuristic_omega=report.heuristic_omega,
        heuristic_calmar=report.heuristic_calmar,
        heuristic_sterling=report.heuristic_sterling,
        heuristic_burke=report.heuristic_burke,
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
        heuristic_trend=report.heuristic_trend,
        heuristic_volatility=report.heuristic_volatility,
        heuristic_consistency=report.heuristic_consistency,
        heuristic_drift=report.heuristic_drift,
        heuristic_correlation=report.heuristic_correlation,
        heuristic_sharpe=report.heuristic_sharpe,
        heuristic_sortino=report.heuristic_sortino,
        heuristic_omega=report.heuristic_omega,
        heuristic_calmar=report.heuristic_calmar,
        heuristic_sterling=report.heuristic_sterling,
        heuristic_burke=report.heuristic_burke,
    )

    features = dataset.features[5]
    result = scorer.score(features)
    assert result.source == "heuristic"
    positive_weights = {
        name: weight for name, weight in report.heuristic_weights.items() if weight > 0.0
    }
    total_weight = 0.0
    weighted_confidence = 0.0
    for name, weight in positive_weights.items():
        trend_factor = SequentialOnlineScorer._trend_factor(
            report.heuristic_trend.get(name)
        )
        volatility_factor = SequentialOnlineScorer._volatility_factor(
            report.heuristic_volatility.get(name)
        )
        consistency_factor = SequentialOnlineScorer._consistency_factor(
            report.heuristic_consistency.get(name)
        )
        drift_factor = SequentialOnlineScorer._drift_factor(
            report.heuristic_drift.get(name)
        )
        correlation_factor = SequentialOnlineScorer._correlation_factor(
            report.heuristic_correlation.get(name)
        )
        sharpe_factor = SequentialOnlineScorer._sharpe_factor(
            report.heuristic_sharpe.get(name)
        )
        sortino_factor = SequentialOnlineScorer._sortino_factor(
            report.heuristic_sortino.get(name)
        )
        omega_factor = SequentialOnlineScorer._omega_factor(
            report.heuristic_omega.get(name)
        )
        calmar_factor = SequentialOnlineScorer._calmar_factor(
            report.heuristic_calmar.get(name)
        )
        sterling_factor = SequentialOnlineScorer._sterling_factor(
            report.heuristic_sterling.get(name)
        )
        burke_factor = SequentialOnlineScorer._burke_factor(
            report.heuristic_burke.get(name)
        )
        total_weight += (
            weight
            * trend_factor
            * volatility_factor
            * consistency_factor
            * drift_factor
            * correlation_factor
            * sharpe_factor
            * sortino_factor
            * omega_factor
            * calmar_factor
            * sterling_factor
            * burke_factor
        )
        weighted_confidence += (
            weight
            * trend_factor
            * volatility_factor
            * consistency_factor
            * drift_factor
            * correlation_factor
            * sharpe_factor
            * sortino_factor
            * omega_factor
            * calmar_factor
            * sterling_factor
            * burke_factor
            * report.heuristic_confidence.get(name, 0.0)
        )
    if total_weight > 0.0:
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
        trend_key = f"heuristic_trend.{name}"
        if name in report.heuristic_trend:
            assert trend_key in diagnostics
            assert pytest.approx(diagnostics[trend_key], rel=1e-6) == report.heuristic_trend[name]
        volatility_key = f"heuristic_volatility.{name}"
        if name in report.heuristic_volatility:
            assert volatility_key in diagnostics
            assert pytest.approx(diagnostics[volatility_key], rel=1e-6) == report.heuristic_volatility[
                name
            ]
        consistency_key = f"heuristic_consistency.{name}"
        if name in report.heuristic_consistency:
            assert consistency_key in diagnostics
            assert pytest.approx(diagnostics[consistency_key], rel=1e-6) == report.heuristic_consistency[
                name
            ]
        drift_key = f"heuristic_drift.{name}"
        if name in report.heuristic_drift:
            assert drift_key in diagnostics
            assert pytest.approx(diagnostics[drift_key], rel=1e-6) == report.heuristic_drift[name]
        correlation_key = f"heuristic_correlation.{name}"
        if name in report.heuristic_correlation:
            assert correlation_key in diagnostics
            assert pytest.approx(
                diagnostics[correlation_key], rel=1e-6
            ) == report.heuristic_correlation[name]
        sharpe_key = f"heuristic_sharpe.{name}"
        if name in report.heuristic_sharpe:
            assert sharpe_key in diagnostics
            assert pytest.approx(diagnostics[sharpe_key], rel=1e-6) == report.heuristic_sharpe[name]
        sortino_key = f"heuristic_sortino.{name}"
        if name in report.heuristic_sortino:
            assert sortino_key in diagnostics
            assert pytest.approx(diagnostics[sortino_key], rel=1e-6) == report.heuristic_sortino[name]
        omega_key = f"heuristic_omega.{name}"
        if name in report.heuristic_omega:
            assert omega_key in diagnostics
            assert pytest.approx(diagnostics[omega_key], rel=1e-6) == report.heuristic_omega[name]
        calmar_key = f"heuristic_calmar.{name}"
        if name in report.heuristic_calmar:
            assert calmar_key in diagnostics
            assert pytest.approx(diagnostics[calmar_key], rel=1e-6) == report.heuristic_calmar[name]
        sterling_key = f"heuristic_sterling.{name}"
        if name in report.heuristic_sterling:
            assert sterling_key in diagnostics
            assert pytest.approx(diagnostics[sterling_key], rel=1e-6) == report.heuristic_sterling[name]
        burke_key = f"heuristic_burke.{name}"
        if name in report.heuristic_burke:
            assert burke_key in diagnostics
            assert pytest.approx(diagnostics[burke_key], rel=1e-6) == report.heuristic_burke[name]
