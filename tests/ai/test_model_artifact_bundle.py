from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from bot_core.ai.models import (
    ModelArtifact,
    ModelArtifactIntegrityError,
    generate_model_artifact_bundle,
    load_model_artifact_bundle,
)
from bot_core.ai.validation import ModelArtifactValidationError


def _build_sample_artifact() -> ModelArtifact:
    return ModelArtifact(
        feature_names=("f1",),
        model_state={"weights": [0.0]},
        trained_at=datetime.now(timezone.utc),
        metrics={"summary": {"mae": 0.0, "rmse": 0.0}},
        metadata={"model_version": "test-1", "symbol": "BTCUSDT"},
        target_scale=1.0,
        training_rows=1,
        validation_rows=1,
        test_rows=1,
        feature_scalers={"f1": (0.0, 1.0)},
    )


def test_load_model_artifact_bundle_valid(tmp_path: Path) -> None:
    artifact = _build_sample_artifact()
    bundle = generate_model_artifact_bundle(artifact, tmp_path, name="model")

    loaded = load_model_artifact_bundle(tmp_path, expected_artifact=bundle.artifact_path.name)
    assert loaded.artifact.metadata["model_version"] == "test-1"
    assert loaded.checksums


def test_load_model_artifact_bundle_detects_checksum_mismatch(tmp_path: Path) -> None:
    artifact = _build_sample_artifact()
    generate_model_artifact_bundle(artifact, tmp_path, name="model")

    artifact_path = tmp_path / "model.json"
    artifact_path.write_text("tampered", encoding="utf-8")

    with pytest.raises(ModelArtifactIntegrityError):
        load_model_artifact_bundle(tmp_path, expected_artifact="model.json")


def test_load_model_artifact_bundle_requires_version(tmp_path: Path) -> None:
    artifact = ModelArtifact(
        feature_names=("f1",),
        model_state={"weights": [1.0]},
        trained_at=datetime.now(timezone.utc),
        metrics={"summary": {"mae": 0.1}},
        metadata={},
        target_scale=1.0,
        training_rows=1,
        validation_rows=1,
        test_rows=1,
        feature_scalers={"f1": (0.0, 1.0)},
    )
    generate_model_artifact_bundle(artifact, tmp_path, name="model")

    with pytest.raises(ModelArtifactIntegrityError):
        load_model_artifact_bundle(tmp_path, expected_artifact="model.json")


def test_load_model_artifact_bundle_rejects_invalid_signature(tmp_path: Path) -> None:
    artifact = _build_sample_artifact()
    bundle = generate_model_artifact_bundle(
        artifact,
        tmp_path,
        name="model",
        signing_key=b"correct-signing-key",
        signing_key_id="primary",
    )

    with pytest.raises(ModelArtifactIntegrityError):
        load_model_artifact_bundle(
            tmp_path,
            expected_artifact=bundle.artifact_path.name,
            signing_keys={"primary": b"invalid-key"},
        )


def test_load_model_artifact_bundle_rejects_schema_violation(tmp_path: Path) -> None:
    artifact = ModelArtifact(
        feature_names=(),
        model_state={"weights": [0.0]},
        trained_at=datetime.now(timezone.utc),
        metrics={"summary": {"mae": 0.0}, "train": {}, "validation": {}, "test": {}},
        metadata={"model_version": "invalid-schema"},
        target_scale=1.0,
        training_rows=1,
        validation_rows=1,
        test_rows=1,
        feature_scalers={"f1": (0.0, 1.0)},
    )

    bundle = generate_model_artifact_bundle(artifact, tmp_path, name="model")

    with pytest.raises(ModelArtifactValidationError):
        load_model_artifact_bundle(
            tmp_path,
            expected_artifact=bundle.artifact_path.name,
        )
