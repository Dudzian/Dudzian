from __future__ import annotations

import pytest

from bot_core.ai.feature_engineering import FeatureDataset, FeatureVector
from bot_core.ai.training import ModelTrainer


def _build_dataset(rows: int = 60) -> FeatureDataset:
    vectors = []
    for idx in range(rows):
        vectors.append(
            FeatureVector(
                timestamp=float(idx),
                symbol="BTC",
                features={"f1": float(idx), "f2": float(idx % 5)},
                target_bps=float((idx % 7) - 3),
            )
        )
    return FeatureDataset(vectors=tuple(vectors), metadata={})


def test_model_trainer_emits_test_metrics() -> None:
    dataset = _build_dataset()
    trainer = ModelTrainer(
        learning_rate=0.1,
        n_estimators=10,
        validation_split=0.2,
        test_split=0.1,
    )
    artifact = trainer.train(dataset)

    total_rows = (
        artifact.metadata["training_rows"]
        + artifact.metadata["validation_rows"]
        + artifact.metadata["test_rows"]
    )
    assert total_rows == len(dataset.vectors)
    assert artifact.metadata["validation_rows"] > 0
    assert artifact.metadata["test_rows"] > 0
    assert "dataset_split" in artifact.metadata
    assert artifact.metadata["dataset_split"]["validation_ratio"] == pytest.approx(0.2)
    assert artifact.metadata["dataset_split"]["test_ratio"] == pytest.approx(0.1)

    assert "validation_mae" in artifact.metrics
    assert "test_mae" in artifact.metrics
    assert "test_metrics" in artifact.metadata
    assert artifact.metadata["test_metrics"]["mae"] == pytest.approx(
        artifact.metrics["test_mae"]
    )

