from __future__ import annotations

import json
from pathlib import Path

import pytest

from bot_core.runtime.state_manager import RuntimeStateManager
from core.ml.model_registry import ModelRegistry, ModelRegistryError
import scripts.manage_models as manage_models


def _create_artifact(tmp_path: Path, name: str, payload: str) -> Path:
    artifact = tmp_path / name
    artifact.write_text(payload, encoding="utf-8")
    return artifact


def _create_state_manager(tmp_path: Path) -> RuntimeStateManager:
    state_root = tmp_path / "state"
    manager = RuntimeStateManager(root=state_root)
    manager.record_checkpoint(entrypoint="demo", mode="demo", config_path="runtime.yml")
    return manager


def test_publish_updates_registry_and_state_manager(tmp_path: Path) -> None:
    manager = _create_state_manager(tmp_path)
    registry_dir = tmp_path / "models"
    registry = ModelRegistry(root=registry_dir, state_manager=manager)

    artifact = _create_artifact(tmp_path, "model.bin", "payload-1")
    metadata = registry.publish_model(artifact, backend="reference", dataset_metadata={"rows": 42})

    assert metadata.backend == "reference"
    assert metadata.dataset_metadata["rows"] == 42

    registry_payload = json.loads((registry_dir / "registry.json").read_text(encoding="utf-8"))
    assert registry_payload["active_model_id"] == metadata.model_id

    checkpoint_payload = json.loads(manager.path.read_text(encoding="utf-8"))
    active_model = checkpoint_payload["metadata"]["active_model"]
    assert active_model["model_id"] == metadata.model_id
    assert active_model["backend"] == "reference"


def test_list_and_rollback_changes_active_model(tmp_path: Path) -> None:
    manager = _create_state_manager(tmp_path)
    registry_root = tmp_path / "models"
    registry = ModelRegistry(root=registry_root, state_manager=manager)

    artifact_a = _create_artifact(tmp_path, "a.bin", "a")
    artifact_b = _create_artifact(tmp_path, "b.bin", "bb")

    meta_a = registry.publish_model(artifact_a, backend="reference")
    meta_b = registry.publish_model(artifact_b, backend="reference")

    models = registry.list_models()
    assert models[0].model_id == meta_b.model_id
    assert models[1].model_id == meta_a.model_id

    restored = registry.rollback(meta_a.model_id)
    assert restored.model_id == meta_a.model_id

    payload = json.loads((registry_root / "registry.json").read_text(encoding="utf-8"))
    assert payload["active_model_id"] == meta_a.model_id


def test_cli_publish_list_and_rollback(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    state_root = tmp_path / "state"
    manager = RuntimeStateManager(root=state_root)
    manager.record_checkpoint(entrypoint="demo", mode="demo", config_path="runtime.yml")

    registry_dir = tmp_path / "models"
    artifact = _create_artifact(tmp_path, "cli.bin", "cli")

    exit_code = manage_models.main(
        [
            "publish",
            "--artifact",
            str(artifact),
            "--backend",
            "reference",
            "--metadata",
            "rows=12",
            "--registry-dir",
            str(registry_dir),
            "--state-dir",
            str(state_root),
        ]
    )
    assert exit_code == 0
    publish_payload = json.loads(capsys.readouterr().out)

    exit_code = manage_models.main(
        [
            "list",
            "--registry-dir",
            str(registry_dir),
            "--output",
            "json",
        ]
    )
    assert exit_code == 0
    list_output = json.loads(capsys.readouterr().out)
    assert list_output[0]["model_id"] == publish_payload["model_id"]

    exit_code = manage_models.main(
        [
            "rollback",
            "--model-id",
            publish_payload["model_id"],
            "--registry-dir",
            str(registry_dir),
            "--state-dir",
            str(state_root),
        ]
    )
    assert exit_code == 0
    rollback_output = json.loads(capsys.readouterr().out)
    assert rollback_output["model_id"] == publish_payload["model_id"]


def test_metadata_parser_rejects_invalid_entries() -> None:
    with pytest.raises(ModelRegistryError):
        manage_models._parse_metadata(["invalid-entry"])  # type: ignore[attr-defined]
