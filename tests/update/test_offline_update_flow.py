from __future__ import annotations

import errno
import json
import shutil
from pathlib import Path

import pytest

from core.update.installer import (
    create_release_archive,
    install_release_archive,
    verify_release_archive,
)


def _write_file(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


@pytest.mark.timeout(5)
def test_install_and_rollback_offline_release(tmp_path: Path) -> None:
    app_root = tmp_path / "app"
    models_target = app_root / "models"
    strategies_target = app_root / "strategies"

    _write_file(models_target / "model.bin", "MODEL-ORIGINAL")
    _write_file(strategies_target / "grid.json", json.dumps({"name": "sunset"}))

    staging_root = tmp_path / "staging"
    models_source = staging_root / "models"
    strategies_source = staging_root / "strategies"
    _write_file(models_source / "model.bin", "MODEL-UPDATED")
    _write_file(strategies_source / "grid.json", json.dumps({"name": "grid"}))

    archive_path = tmp_path / "release.tar"
    signing_key = b"rollback-secret"
    create_release_archive(
        version="9.9.9",
        output_path=archive_path,
        models_dir=models_source,
        strategies_dir=strategies_source,
        signing_key=signing_key,
        signing_key_id="offline",
    )

    manifest = verify_release_archive(archive_path, signing_key=signing_key)
    assert manifest["version"] == "9.9.9"

    backup_dir = tmp_path / "backups"
    install_result = install_release_archive(
        archive_path,
        signing_key=signing_key,
        models_target=models_target,
        strategies_target=strategies_target,
        backup_dir=backup_dir,
    )

    assert install_result["installed_models"] is True
    assert install_result["installed_strategies"] is True

    backups = {name: Path(path) for name, path in install_result["backups"]}
    assert "models" in backups and backups["models"].exists()
    assert "strategies" in backups and backups["strategies"].exists()

    assert models_target.joinpath("model.bin").read_text(encoding="utf-8") == "MODEL-UPDATED"
    assert json.loads(strategies_target.joinpath("grid.json").read_text(encoding="utf-8"))["name"] == "grid"

    shutil.rmtree(models_target)
    shutil.copytree(backups["models"], models_target)
    shutil.rmtree(strategies_target)
    shutil.copytree(backups["strategies"], strategies_target)

    assert models_target.joinpath("model.bin").read_text(encoding="utf-8") == "MODEL-ORIGINAL"
    assert json.loads(strategies_target.joinpath("grid.json").read_text(encoding="utf-8"))["name"] == "sunset"


@pytest.mark.timeout(5)
def test_install_release_archive_fails_when_disk_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app_root = tmp_path / "app"
    models_target = app_root / "models"
    backup_dir = tmp_path / "backups"

    _write_file(models_target / "model.bin", "OLD")

    staging_root = tmp_path / "staging"
    models_source = staging_root / "models"
    _write_file(models_source / "model.bin", "NEW")

    archive_path = tmp_path / "release.tar"
    create_release_archive(
        version="1.0",
        output_path=archive_path,
        models_dir=models_source,
        strategies_dir=None,
        signing_key=None,
    )

    original_copytree = shutil.copytree
    call_count = {"value": 0}

    def failing_copytree(src, dst, *args, **kwargs):  # type: ignore[override]
        call_count["value"] += 1
        if call_count["value"] == 1:
            return original_copytree(src, dst, *args, **kwargs)
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(shutil, "copytree", failing_copytree)

    with pytest.raises(OSError):
        install_release_archive(
            archive_path,
            models_target=models_target,
            strategies_target=None,
            backup_dir=backup_dir,
        )

    # Backup istnieje, a dane źródłowe nie zostały nadpisane.
    backups = list(backup_dir.glob("models-*"))
    assert backups and backups[0].is_dir()
    assert models_target.joinpath("model.bin").read_text(encoding="utf-8") == "OLD"


@pytest.mark.timeout(5)
def test_install_release_archive_partial_failure_allows_manual_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app_root = tmp_path / "app"
    models_target = app_root / "models"
    strategies_target = app_root / "strategies"
    backup_dir = tmp_path / "backups"

    _write_file(models_target / "model.bin", "LEGACY")
    _write_file(strategies_target / "grid.json", json.dumps({"name": "sunset"}))

    staging_root = tmp_path / "staging"
    models_source = staging_root / "models"
    strategies_source = staging_root / "strategies"
    _write_file(models_source / "model.bin", "UPDATED")
    _write_file(strategies_source / "grid.json", json.dumps({"name": "fresh"}))

    archive_path = tmp_path / "release.tar"
    create_release_archive(
        version="2.0",
        output_path=archive_path,
        models_dir=models_source,
        strategies_dir=strategies_source,
        signing_key=None,
    )

    original_copytree = shutil.copytree
    call_order: list[str] = []

    def partially_failing_copytree(src, dst, *args, **kwargs):  # type: ignore[override]
        call_order.append(str(dst))
        # Pozwól wykonać backup modeli, instalację modeli oraz backup strategii
        if len(call_order) <= 3:
            return original_copytree(src, dst, *args, **kwargs)
        # Przy próbie kopiowania strategii zgłoś błąd
        raise RuntimeError("Interrupt during strategies deployment")

    monkeypatch.setattr(shutil, "copytree", partially_failing_copytree)

    with pytest.raises(RuntimeError):
        install_release_archive(
            archive_path,
            models_target=models_target,
            strategies_target=strategies_target,
            backup_dir=backup_dir,
        )

    monkeypatch.setattr(shutil, "copytree", original_copytree)

    backups = sorted(backup_dir.glob("*"))
    assert backups and all(path.is_dir() for path in backups)

    # Modele zostały nadpisane, ale możemy je przywrócić z kopii.
    assert models_target.joinpath("model.bin").read_text(encoding="utf-8") == "UPDATED"
    shutil.rmtree(models_target)
    shutil.copytree(backups[0], models_target)
    assert models_target.joinpath("model.bin").read_text(encoding="utf-8") == "LEGACY"
