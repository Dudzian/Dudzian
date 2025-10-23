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

    total_rows = artifact.training_rows + artifact.validation_rows + artifact.test_rows
    assert total_rows == len(dataset.vectors)
    assert artifact.validation_rows > 0
    assert artifact.test_rows > 0
    assert "dataset_split" in artifact.metadata
    assert artifact.metadata["dataset_split"]["validation_ratio"] == pytest.approx(0.2)
    assert artifact.metadata["dataset_split"]["test_ratio"] == pytest.approx(0.1)

    assert "validation" in artifact.metrics
    assert "test" in artifact.metrics
    assert "summary" in artifact.metrics
    assert artifact.metrics["validation"].get("mae", 0.0) >= 0.0
    assert artifact.metrics["test"].get("mae", 0.0) >= 0.0
    assert artifact.metrics["summary"]["mae"] == pytest.approx(
        artifact.metrics["train"]["mae"]
    )
    assert artifact.metrics["summary"]["test_mae"] == pytest.approx(
        artifact.metrics["test"]["mae"]
    )
    assert artifact.metrics["summary"]["validation_mae"] == pytest.approx(
        artifact.metrics["validation"]["mae"]
    )

