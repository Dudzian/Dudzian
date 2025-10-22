"""Integration test comparing sequential AI outputs against heuristics."""

from __future__ import annotations

from pathlib import Path

import pytest

from bot_core.ai.sequential import (
    BUILTIN_HEURISTICS,
    HistoricalFeatureRepository,
    SequentialTrainingPipeline,
)

from tests.ai.test_sequential_pipeline import _build_synthetic_dataset


def test_sequential_model_beats_builtin_heuristics(tmp_path) -> None:
    dataset = _build_synthetic_dataset()
    repository_path = Path(tmp_path) / "repo"
    repository = HistoricalFeatureRepository(repository_path)
    pipeline = SequentialTrainingPipeline(
        repository=repository,
        heuristics=BUILTIN_HEURISTICS,
    )

    report = pipeline.train_offline(dataset, top_k_features=3, folds=4)

    model_accuracy = sum(report.walk_forward_metrics.directional_accuracy) / len(
        report.walk_forward_metrics.directional_accuracy
    )
    heuristic_accuracy = sum(report.heuristic_metrics.directional_accuracy) / len(
        report.heuristic_metrics.directional_accuracy
    )

    assert model_accuracy >= heuristic_accuracy
    assert repository.list(), "repository should retain historical datasets"
    assert report.heuristic_weights
    assert pytest.approx(sum(report.heuristic_weights.values()), rel=1e-6) == 1.0
    assert report.suppressed_heuristics == {}
    assert report.heuristic_confidence
    assert report.heuristic_trend
    assert report.heuristic_volatility
    assert report.heuristic_consistency
    assert report.heuristic_drift
