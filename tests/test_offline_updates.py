from __future__ import annotations

import json
import tarfile
from pathlib import Path

from core.update.installer import (
    create_release_archive,
    install_release_archive,
    verify_release_archive,
)
import pytest

from scripts import offline_update, package_offline_release


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_create_and_verify_release(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    strategies_dir = tmp_path / "strategies"
    _write_file(models_dir / "model.bin", "MODEL")
    _write_file(strategies_dir / "grid.json", json.dumps({"name": "grid"}))

    archive = tmp_path / "release.tar.gz"
    key = b"sign-key"
    create_release_archive(
        version="1.0.0",
        output_path=archive,
        models_dir=models_dir,
        strategies_dir=strategies_dir,
        signing_key=key,
        signing_key_id="offline",
    )

    manifest = verify_release_archive(archive, signing_key=key)
    assert manifest["version"] == "1.0.0"
    paths = {entry["path"] for entry in manifest["artifacts"]}
    assert "models/model.bin" in paths
    assert "strategies/grid.json" in paths


def test_install_release_creates_backups(tmp_path: Path) -> None:
    staging_models = tmp_path / "src_models"
    staging_strategies = tmp_path / "src_strategies"
    _write_file(staging_models / "model.bin", "NEW")
    _write_file(staging_strategies / "ai.json", json.dumps({"name": "ai"}))

    archive = tmp_path / "release.tar.gz"
    key = b"install-key"
    create_release_archive(
        version="2.0.0",
        output_path=archive,
        models_dir=staging_models,
        strategies_dir=staging_strategies,
        signing_key=key,
        signing_key_id="offline",
    )

    target_models = tmp_path / "models"
    target_strategies = tmp_path / "strategies"
    _write_file(target_models / "model.bin", "OLD")
    _write_file(target_strategies / "sunset.json", json.dumps({"name": "sunset"}))

    backup_dir = tmp_path / "backups"
    result = install_release_archive(
        archive,
        signing_key=key,
        models_target=target_models,
        strategies_target=target_strategies,
        backup_dir=backup_dir,
    )

    assert target_models.joinpath("model.bin").read_text(encoding="utf-8") == "NEW"
    assert target_strategies.joinpath("ai.json").exists()
    backup_names = {name for name, _ in result["backups"]}
    assert "models" in backup_names
    assert "strategies" in backup_names


def test_offline_update_cli_prepare_and_verify(tmp_path: Path, capsys) -> None:
    models_dir = tmp_path / "models"
    _write_file(models_dir / "model.bin", "DATA")
    archive = tmp_path / "release.tar.gz"

    offline_update.main(
        [
            "prepare-release",
            "--version",
            "3.1.0",
            "--output",
            str(archive),
            "--models-dir",
            str(models_dir),
            "--signing-key",
            "cli-key",
        ]
    )
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["archive"] == str(archive)

    offline_update.main(
        [
            "verify-release",
            "--archive",
            str(archive),
            "--signing-key",
            "cli-key",
        ]
    )
    verified = json.loads(capsys.readouterr().out)
    assert verified["version"] == "3.1.0"


def test_package_offline_release_cli(tmp_path: Path, capsys) -> None:
    models_dir = tmp_path / "models"
    strategies_dir = tmp_path / "strategies"
    _write_file(models_dir / "model.bin", "M")
    _write_file(strategies_dir / "grid.json", json.dumps({"name": "grid"}))

    archive = tmp_path / "bundle.tar.gz"
    manifest_path = tmp_path / "manifest.json"

    package_offline_release.main(
        [
            "--version",
            "4.0.0",
            "--output",
            str(archive),
            "--models-dir",
            str(models_dir),
            "--strategies-dir",
            str(strategies_dir),
            "--signing-key",
            "pack-key",
            "--manifest-output",
            str(manifest_path),
        ]
    )

    stdout = json.loads(capsys.readouterr().out)
    assert stdout["archive"] == str(archive)
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == "4.0.0"
    assert any(entry["path"] == "models/model.bin" for entry in manifest["artifacts"])


def test_verify_release_detects_manifest_tampering(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    _write_file(models_dir / "model.bin", "DATA")
    archive = tmp_path / "release.tar.gz"
    key = b"tamper-key"

    create_release_archive(
        version="5.0.0",
        output_path=archive,
        models_dir=models_dir,
        strategies_dir=None,
        signing_key=key,
        signing_key_id="offline",
    )

    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(extract_dir)

    manifest_path = extract_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["sha384"] = "0" * 96
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    tampered = tmp_path / "tampered.tar.gz"
    with tarfile.open(tampered, "w:gz") as bundle:
        for entry in sorted(extract_dir.iterdir()):
            bundle.add(entry, arcname=entry.name)

    with pytest.raises(RuntimeError):
        verify_release_archive(tampered, signing_key=key)
