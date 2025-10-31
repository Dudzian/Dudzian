from __future__ import annotations

import json
from pathlib import Path

import pytest

from bot_core.ai.feature_engineering import FeatureDataset, FeatureVector
from core.data.validators import DatasetValidationError, DatasetValidator
from core.ml.training_pipeline import TrainingPipeline


def _build_valid_dataset() -> FeatureDataset:
    vectors = (
        FeatureVector(
            timestamp=1_700_000_000.0,
            symbol="BTCUSDT",
            features={"momentum": 0.1, "volatility": 0.5},
            target_bps=0.05,
        ),
        FeatureVector(
            timestamp=1_700_000_060.0,
            symbol="BTCUSDT",
            features={"momentum": 0.2, "volatility": 0.6},
            target_bps=0.06,
        ),
    )
    return FeatureDataset(vectors=vectors, metadata={"source": "unit-test"})


def test_dataset_validator_produces_success_report(tmp_path: Path) -> None:
    dataset = _build_valid_dataset()
    validator = DatasetValidator()

    report = validator.validate(dataset)
    assert not report.has_errors

    path = validator.log_report(report, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["dataset_metadata"]["row_count"] == 2


def test_dataset_validator_detects_missing_values(tmp_path: Path) -> None:
    vectors = (
        FeatureVector(
            timestamp=1_700_000_000.0,
            symbol="BTCUSDT",
            features={"momentum": float("nan")},
            target_bps=float("nan"),
        ),
    )
    dataset = FeatureDataset(vectors=vectors, metadata={})
    validator = DatasetValidator()

    report = validator.validate(dataset)
    assert report.has_errors

    path = validator.log_report(report, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    rules = {issue["rule"] for issue in payload["issues"]}
    assert "missing_data" in rules


def test_training_pipeline_aborts_on_validation_error(tmp_path: Path) -> None:
    vectors = (
        FeatureVector(
            timestamp=1_700_000_000.0,
            symbol="BTCUSDT",
            features={"momentum": 0.1, "volatility": float("nan")},
            target_bps=0.05,
        ),
    )
    dataset = FeatureDataset(vectors=vectors, metadata={})

    pipeline = TrainingPipeline(
        preferred_backends=("reference",),
        fallback_log_dir=tmp_path / "fallback",
        validation_log_dir=tmp_path / "validation",
    )

    with pytest.raises(DatasetValidationError) as exc:
        pipeline.train(dataset)

    assert exc.value.log_path is not None
    log_files = sorted((tmp_path / "validation").glob("dataset_validation_*.json"))
    assert log_files, "Validation log should be generated"
