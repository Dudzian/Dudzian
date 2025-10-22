from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bot_core.ai.inference import ModelRepository
from bot_core.ai.models import ModelArtifact


def _make_artifact(*, metadata: dict[str, object] | None = None) -> ModelArtifact:
    payload = {"feature_scalers": {}}
    if metadata:
        payload.update(metadata)
    return ModelArtifact(
        feature_names=("f1", "f2"),
        model_state={"weights": [0.1, 0.2], "bias": 0.0},
        trained_at=datetime.now(timezone.utc),
        metrics={"mae": 1.23, "directional_accuracy": 0.6},
        metadata=payload,
        backend="builtin",
    )


def test_publish_registers_version_and_aliases(tmp_path: Path) -> None:
    repository = ModelRepository(tmp_path)
    artifact = _make_artifact()

    destination = repository.publish(
        artifact,
        version="1.0.0",
        filename="model-1.0.0.json",
        aliases=("production", "stable"),
    )

    assert destination.name == "model-1.0.0.json"
    assert repository.get_active_version() == "1.0.0"

    loaded = repository.load("production")
    assert loaded.metadata.get("model_version") == "1.0.0"

    manifest = repository.get_manifest()
    assert manifest["aliases"]["production"] == "1.0.0"
    assert manifest["versions"]["1.0.0"]["file"] == "model-1.0.0.json"
    assert set(manifest["versions"]["1.0.0"]["aliases"]) == {"production", "stable"}


def test_publish_updates_aliases_and_allows_active_switch(tmp_path: Path) -> None:
    repository = ModelRepository(tmp_path)
    v1 = repository.publish(
        _make_artifact(),
        version="1.0.0",
        filename="model-1.0.0.json",
        aliases=("latest", "production"),
    )
    assert v1.exists()

    repository.publish(
        _make_artifact(metadata={"extra": True}),
        version="1.1.0",
        filename="model-1.1.0.json",
        aliases=("latest",),
        activate=True,
    )

    assert repository.resolve("latest").name == "model-1.1.0.json"
    assert repository.list_versions() == ("1.0.0", "1.1.0")

    repository.set_active_version("1.0.0")
    assert repository.get_active_version() == "1.0.0"

    repository.set_active_version("1.1.0")
    loaded_active = repository.load("@active")
    assert loaded_active.metadata.get("model_version") == "1.1.0"


def test_save_without_version_does_not_touch_manifest(tmp_path: Path) -> None:
    repository = ModelRepository(tmp_path)
    destination = repository.save(_make_artifact(), "manual.json")
    assert destination.exists()
    manifest = repository.get_manifest()
    assert manifest["versions"] == {}


def test_get_version_entry_returns_copy(tmp_path: Path) -> None:
    repository = ModelRepository(tmp_path)
    repository.publish(
        _make_artifact(),
        version="5.0.0",
        filename="model-5.0.0.json",
        aliases=("latest",),
    )

    entry = repository.get_version_entry("5.0.0")
    assert entry is not None
    assert entry["file"] == "model-5.0.0.json"
    entry["file"] = "mutated.json"
    manifest = repository.get_manifest()
    assert manifest["versions"]["5.0.0"]["file"] == "model-5.0.0.json"


def test_remove_alias_updates_manifest(tmp_path: Path) -> None:
    repository = ModelRepository(tmp_path)
    repository.publish(
        _make_artifact(),
        version="2.0.0",
        filename="model-2.0.0.json",
        aliases=("production", "stable"),
    )

    assert repository.get_alias_target("production") == "2.0.0"
    repository.remove_alias("production")

    manifest = repository.get_manifest()
    assert "production" not in manifest["aliases"]
    assert manifest["versions"]["2.0.0"]["aliases"] == ["stable"]


def test_assign_alias_creates_mapping(tmp_path: Path) -> None:
    repository = ModelRepository(tmp_path)
    repository.publish(
        _make_artifact(),
        version="4.0.0",
        filename="model-4.0.0.json",
        aliases=("baseline",),
    )

    repository.assign_alias("production", "4.0.0")

    manifest = repository.get_manifest()
    assert manifest["aliases"]["production"] == "4.0.0"
    assert set(manifest["versions"]["4.0.0"]["aliases"]) == {"baseline", "production"}


def test_assign_alias_replaces_previous_target(tmp_path: Path) -> None:
    repository = ModelRepository(tmp_path)
    repository.publish(
        _make_artifact(),
        version="6.0.0",
        filename="model-6.0.0.json",
        aliases=("latest",),
    )
    repository.publish(
        _make_artifact(),
        version="6.1.0",
        filename="model-6.1.0.json",
        aliases=("beta",),
    )

    repository.assign_alias("latest", "6.1.0")

    manifest = repository.get_manifest()
    assert manifest["aliases"]["latest"] == "6.1.0"
    assert set(manifest["versions"]["6.1.0"]["aliases"]) == {"beta", "latest"}
    assert manifest["versions"]["6.0.0"]["aliases"] == []


def test_assign_alias_rejects_missing_version(tmp_path: Path) -> None:
    repository = ModelRepository(tmp_path)
    with pytest.raises(KeyError):
        repository.assign_alias("missing", "9.9.9")


def test_activate_alias_sets_active_version(tmp_path: Path) -> None:
    repository = ModelRepository(tmp_path)
    repository.publish(
        _make_artifact(),
        version="1.0.0",
        filename="model-1.0.0.json",
        aliases=("production",),
        activate=True,
    )
    repository.publish(
        _make_artifact(),
        version="1.1.0",
        filename="model-1.1.0.json",
        aliases=("candidate",),
        activate=False,
    )

    repository.assign_alias("production", "1.1.0")

    activated = repository.activate_alias("production")

    assert activated == "1.1.0"
    assert repository.get_active_version() == "1.1.0"


def test_activate_alias_prunes_stale_entry(tmp_path: Path) -> None:
    repository = ModelRepository(tmp_path)
    repository.publish(
        _make_artifact(),
        version="2.0.0",
        filename="model-2.0.0.json",
        aliases=("staging",),
        activate=False,
    )

    # Zostaw cel aliasu, ale usuń wpis wersji, aby zasymulować nieaktualny manifest.
    manifest_path = repository.base_path / repository.manifest_name
    payload = json.loads(manifest_path.read_text("utf-8"))
    payload["versions"].pop("2.0.0", None)
    payload["aliases"]["staging"] = "2.0.0"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    repository._manifest_cache = None  # type: ignore[attr-defined]

    result = repository.activate_alias("staging", missing_ok=True)

    assert result is None
    manifest = repository.get_manifest()
    assert "staging" not in manifest["aliases"]


def test_promote_version_sets_active_and_aliases(tmp_path: Path) -> None:
    repository = ModelRepository(tmp_path)
    repository.publish(
        _make_artifact(),
        version="1.0.0",
        filename="model-1.0.0.json",
        aliases=("latest",),
    )
    repository.publish(
        _make_artifact(),
        version="1.1.0",
        filename="model-1.1.0.json",
        aliases=("candidate",),
    )

    repository.promote_version("1.1.0", aliases=("latest", "production"))

    manifest = repository.get_manifest()
    assert manifest["active"] == "1.1.0"
    assert manifest["aliases"]["latest"] == "1.1.0"
    assert manifest["aliases"]["production"] == "1.1.0"
    assert set(manifest["versions"]["1.1.0"]["aliases"]) == {
        "candidate",
        "latest",
        "production",
    }
    assert manifest["versions"]["1.0.0"]["aliases"] == []


def test_promote_version_clears_aliases_when_empty_sequence(tmp_path: Path) -> None:
    repository = ModelRepository(tmp_path)
    repository.publish(
        _make_artifact(),
        version="2.0.0",
        filename="model-2.0.0.json",
        aliases=("production", "stable"),
    )

    repository.promote_version("2.0.0", aliases=())

    manifest = repository.get_manifest()
    assert manifest["active"] == "2.0.0"
    assert manifest["versions"]["2.0.0"]["aliases"] == []
    assert manifest["aliases"] == {}


def test_remove_version_clears_aliases_and_optionally_deletes_file(tmp_path: Path) -> None:
    repository = ModelRepository(tmp_path)
    path = repository.publish(
        _make_artifact(),
        version="3.1.0",
        filename="model-3.1.0.json",
        aliases=("latest", "canary"),
    )
    repository.publish(
        _make_artifact(),
        version="3.2.0",
        filename="model-3.2.0.json",
        aliases=("latest",),
        activate=True,
    )

    assert path.exists()
    repository.remove_version("3.1.0", delete_file=True)

    manifest = repository.get_manifest()
    assert "3.1.0" not in manifest["versions"]
    assert "canary" not in manifest["aliases"]
    assert repository.get_active_version() == "3.2.0"
    assert not path.exists()


def test_prune_versions_removes_oldest_entries(tmp_path: Path) -> None:
    repository = ModelRepository(tmp_path)
    for index in range(1, 6):
        repository.publish(
            _make_artifact(),
            version=f"1.0.{index}",
            filename=f"model-1.0.{index}.json",
            aliases=("latest",) if index == 5 else (),
            activate=index == 5,
        )

    removed = repository.prune_versions(3)

    assert removed == ("1.0.1", "1.0.2")
    assert repository.list_versions() == ("1.0.3", "1.0.4", "1.0.5")
    assert repository.get_active_version() == "1.0.5"


def test_prune_versions_keeps_alias_targets(tmp_path: Path) -> None:
    repository = ModelRepository(tmp_path)
    for version in ("1.0.0", "1.1.0", "1.2.0", "1.3.0"):
        repository.publish(
            _make_artifact(),
            version=version,
            filename=f"model-{version}.json",
            aliases=("production",) if version == "1.0.0" else (),
        )
    repository.assign_alias("canary", "1.2.0")

    removed = repository.prune_versions(2)

    assert removed == ("1.1.0",)
    assert repository.list_versions() == ("1.0.0", "1.2.0", "1.3.0")
    manifest = repository.get_manifest()
    assert manifest["aliases"]["production"] == "1.0.0"
    assert manifest["aliases"]["canary"] == "1.2.0"


def test_prune_versions_respects_keep_versions_and_allows_zero(tmp_path: Path) -> None:
    repository = ModelRepository(tmp_path)
    for version in ("0.9.0", "1.0.0"):
        repository.publish(
            _make_artifact(),
            version=version,
            filename=f"model-{version}.json",
        )

    removed = repository.prune_versions(
        0,
        keep_versions=("1.0.0",),
        keep_active=False,
    )

    assert removed == ("0.9.0",)
    assert repository.list_versions() == ("1.0.0",)


def test_prune_versions_rejects_negative_limit(tmp_path: Path) -> None:
    repository = ModelRepository(tmp_path)
    with pytest.raises(ValueError):
        repository.prune_versions(-1)
