from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import jsonschema

import pytest

from bot_core.ai.models import ModelArtifact
from bot_core.ai.validation import (
    ModelArtifactValidationError,
    validate_model_artifact_schema,
)


def test_model_artifact_matches_schema() -> None:
    schema_path = Path("docs/schemas/model_artifact.schema.json")
    schema = json.loads(schema_path.read_text())

    artifact = ModelArtifact(
        feature_names=("f1", "f2"),
        model_state={"weights": [0.1, -0.2]},
        trained_at=datetime.now(timezone.utc),
        metrics={
            "summary": {"mae": 1.0, "directional_accuracy": 0.55},
            "train": {"mae": 1.0, "directional_accuracy": 0.55},
            "validation": {"mae": 1.2, "directional_accuracy": 0.53},
            "test": {},
        },
        metadata={"quality_thresholds": {"min_directional_accuracy": 0.5}},
        target_scale=1.0,
        training_rows=120,
        validation_rows=30,
        test_rows=0,
        feature_scalers={"f1": (0.0, 1.0), "f2": (0.5, 0.25)},
        decision_journal_entry_id="AI-2024-10-01",
        backend="builtin",
    )

    payload = artifact.to_dict()
    jsonschema.validate(instance=payload, schema=schema)


def test_validate_model_artifact_schema_accepts_model_artifact() -> None:
    artifact = ModelArtifact(
        feature_names=("f1",),
        model_state={"weights": [0.5]},
        trained_at=datetime.now(timezone.utc),
        metrics={
            "summary": {"mae": 0.1, "rmse": 0.2},
            "train": {"mae": 0.1, "rmse": 0.2},
            "validation": {"mae": 0.12, "rmse": 0.22},
            "test": {"mae": 0.15, "rmse": 0.25},
        },
        metadata={},
        target_scale=1.0,
        training_rows=10,
        validation_rows=5,
        test_rows=2,
        feature_scalers={"f1": (0.0, 1.0)},
        backend="builtin",
    )

    validate_model_artifact_schema(artifact)


def test_validate_model_artifact_schema_raises_on_missing_metrics() -> None:
    payload = {
        "feature_names": ["f1"],
        "model_state": {"weights": [0.1]},
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {"summary": {}, "train": {}, "validation": {}, "test": {}},
        "metadata": {},
        "target_scale": 1.0,
        "training_rows": 0,
        "validation_rows": 0,
        "test_rows": 0,
        "feature_scalers": {},
        "backend": "builtin",
    }

    payload["metrics"].pop("summary")  # type: ignore[index]
    with pytest.raises(ModelArtifactValidationError):
        validate_model_artifact_schema(payload)
