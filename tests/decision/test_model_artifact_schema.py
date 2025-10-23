from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import jsonschema

from bot_core.ai.models import ModelArtifact


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
