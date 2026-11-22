from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

import pytest

from core.update import offline_updater
from core.update.offline_updater import OfflinePackageError, import_offline_package
from scripts.package_update import build_offline_package


def _build_test_package(tmp_path: Path) -> Path:
    package_path = tmp_path / "update.dudzianpkg"
    build_offline_package(
        package_id="demo",
        version="1.2.3",
        payload_dir=_create_payload(tmp_path),
        output_path=package_path,
        signing_key=b"test-import-key",
    )
    return package_path


@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize("existing", [False, True])
def test_import_copy_failure_does_not_publish_partial_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupted: bool, existing: bool
) -> None:
    package_path = _build_test_package(tmp_path)
    destination = tmp_path / "packages"
    target = destination / "demo-1.2.3"
    if existing:
        target.mkdir(parents=True)
        (target / "previous.bin").write_bytes(b"previous-verified-package")

    failure = KeyboardInterrupt if interrupted else OSError

    def fail_copy(source: Path, destination: Path) -> None:
        Path(destination).write_bytes(b"partial-copy")
        raise failure("copy interrupted")

    monkeypatch.setattr(shutil, "copy2", fail_copy)
    with pytest.raises(failure, match="copy interrupted"):
        import_offline_package(package_path, destination, hmac_key=b"test-import-key")

    if existing:
        assert {path.name for path in target.iterdir()} == {"previous.bin"}
        assert (target / "previous.bin").read_bytes() == b"previous-verified-package"
    else:
        assert not target.exists()
    assert not list(destination.glob(".offline-import-*"))


def test_import_interruption_after_backup_rename_restores_previous_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_path = _build_test_package(tmp_path)
    destination = tmp_path / "packages"
    target = destination / "demo-1.2.3"
    target.mkdir(parents=True)
    (target / "previous.bin").write_bytes(b"previous-verified-package")
    original_rename = Path.rename

    def interrupted_rename(source: Path, destination: Path) -> Path:
        result = original_rename(source, destination)
        if source == target:
            raise KeyboardInterrupt("interrupted after backup rename")
        return result

    monkeypatch.setattr(Path, "rename", interrupted_rename)
    with pytest.raises(KeyboardInterrupt, match="interrupted after backup rename"):
        import_offline_package(package_path, destination, hmac_key=b"test-import-key")

    assert (target / "previous.bin").read_bytes() == b"previous-verified-package"
    assert not list(destination.glob(".offline-import-*"))


def test_import_rollback_failure_preserves_verified_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_path = _build_test_package(tmp_path)
    destination = tmp_path / "packages"
    target = destination / "demo-1.2.3"
    target.mkdir(parents=True)
    (target / "previous.bin").write_bytes(b"previous-verified-package")
    original_rename = Path.rename

    def fail_publish_and_rollback(source: Path, destination: Path) -> Path:
        if source.name in {"prepared", "previous"}:
            raise OSError("rename denied")
        return original_rename(source, destination)

    monkeypatch.setattr(Path, "rename", fail_publish_and_rollback)
    with pytest.raises(OfflinePackageError, match="zachowano kopię"):
        import_offline_package(package_path, destination, hmac_key=b"test-import-key")

    assert not target.exists()
    backups = list(destination.glob(".offline-import-*/previous/previous.bin"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"previous-verified-package"


def test_import_extraction_interruption_cleans_partial_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_path = _build_test_package(tmp_path)
    extracted: list[Path] = []

    def interrupt_extract(archive: tarfile.TarFile, path: Path) -> None:
        extracted.append(path)
        (path / "partial.bin").write_bytes(b"partial")
        raise KeyboardInterrupt("extraction interrupted")

    monkeypatch.setattr(tarfile.TarFile, "extractall", interrupt_extract)
    with pytest.raises(KeyboardInterrupt, match="extraction interrupted"):
        import_offline_package(package_path, tmp_path / "packages", hmac_key=b"test-import-key")

    assert extracted and all(not path.exists() for path in extracted)
    assert not (tmp_path / "packages").exists()


def test_import_verification_interruption_cleans_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_path = _build_test_package(tmp_path)
    extracted: list[Path] = []

    def interrupt_manifest(directory: Path) -> None:
        extracted.append(directory)
        raise KeyboardInterrupt("verification interrupted")

    monkeypatch.setattr(offline_updater, "_load_manifest", interrupt_manifest)
    with pytest.raises(KeyboardInterrupt, match="verification interrupted"):
        import_offline_package(package_path, tmp_path / "packages", hmac_key=b"test-import-key")

    assert extracted and all(not path.exists() for path in extracted)
    assert not (tmp_path / "packages").exists()


@pytest.mark.parametrize("interrupted", [False, True])
def test_import_publish_failure_restores_previous_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupted: bool
) -> None:
    package_path = _build_test_package(tmp_path)
    destination = tmp_path / "packages"
    target = destination / "demo-1.2.3"
    target.mkdir(parents=True)
    (target / "previous.bin").write_bytes(b"previous-verified-package")
    original_rename = Path.rename
    failure = KeyboardInterrupt if interrupted else OSError

    def fail_publish(source: Path, destination: Path) -> Path:
        if source.name == "prepared":
            raise failure("publication interrupted")
        return original_rename(source, destination)

    monkeypatch.setattr(Path, "rename", fail_publish)
    with pytest.raises(failure, match="publication interrupted"):
        import_offline_package(package_path, destination, hmac_key=b"test-import-key")

    assert {path.name for path in target.iterdir()} == {"previous.bin"}
    assert (target / "previous.bin").read_bytes() == b"previous-verified-package"
    assert not list(destination.glob(".offline-import-*"))


def test_import_destination_failure_cleans_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_path = _build_test_package(tmp_path)
    destination = tmp_path / "packages"
    destination.write_text("not a directory", encoding="utf-8")
    extracted: list[Path] = []
    original_extract = offline_updater._extract_package

    def record_extraction(path: Path) -> Path:
        staging = original_extract(path)
        extracted.append(staging)
        return staging

    monkeypatch.setattr(offline_updater, "_extract_package", record_extraction)
    with pytest.raises(OSError):
        import_offline_package(package_path, destination, hmac_key=b"test-import-key")

    assert extracted and all(not path.exists() for path in extracted)


def test_import_success_replaces_previous_package_after_preparation(tmp_path: Path) -> None:
    package_path = _build_test_package(tmp_path)
    destination = tmp_path / "packages"
    target = destination / "demo-1.2.3"
    target.mkdir(parents=True)
    (target / "previous.bin").write_bytes(b"previous-verified-package")

    result = import_offline_package(package_path, destination, hmac_key=b"test-import-key")

    assert result.target_directory == target
    assert (target / "payload.tar").is_file()
    assert not (target / "previous.bin").exists()
    assert all(path.is_file() and path.is_relative_to(target) for path in result.artifacts.values())
    assert not list(destination.glob(".offline-import-*"))


def _create_payload(tmp_path: Path) -> Path:
    payload_dir = tmp_path / "payload"
    (payload_dir / "models").mkdir(parents=True)
    (payload_dir / "models" / "model.bin").write_bytes(b"binary-model")
    (payload_dir / "strategies").mkdir()
    (payload_dir / "strategies" / "alpha.json").write_text("{}", encoding="utf-8")
    return payload_dir


def _read_manifest(path: Path) -> dict[str, object]:
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def test_import_offline_package_success(tmp_path: Path) -> None:
    payload_dir = _create_payload(tmp_path)
    package_path = tmp_path / "update.dudzianpkg"
    key = b"super-secret"

    build_offline_package(
        package_id="demo",
        version="1.2.3",
        payload_dir=payload_dir,
        output_path=package_path,
        fingerprint="fp-123",
        metadata={"channel": "stable"},
        signing_key=key,
        signing_key_id="demo-key",
    )

    destination = tmp_path / "packages"
    result = import_offline_package(
        package_path,
        destination,
        expected_fingerprint="fp-123",
        hmac_key=key,
    )

    assert result.manifest.package_id == "demo"
    assert result.manifest.version == "1.2.3"
    assert result.target_directory.exists()
    manifest = _read_manifest(result.target_directory)
    assert manifest["id"] == "demo"
    assert manifest["version"] == "1.2.3"
    assert manifest.get("signature", "").startswith(result.manifest.artifacts[0].sha256[:32])
    assert (result.target_directory / "payload.tar").exists()


def test_import_offline_package_invalid_signature(tmp_path: Path) -> None:
    payload_dir = _create_payload(tmp_path)
    package_path = tmp_path / "update.dudzianpkg"
    build_offline_package(
        package_id="demo",
        version="1.0.0",
        payload_dir=payload_dir,
        output_path=package_path,
        signing_key=b"correct",
    )

    with pytest.raises(OfflinePackageError):
        import_offline_package(package_path, tmp_path / "packages", hmac_key=b"wrong")


def test_import_offline_package_fingerprint_mismatch(tmp_path: Path) -> None:
    payload_dir = _create_payload(tmp_path)
    package_path = tmp_path / "update.dudzianpkg"
    key = b"fingerprint"
    build_offline_package(
        package_id="demo",
        version="1.0.0",
        payload_dir=payload_dir,
        output_path=package_path,
        fingerprint="expected",
        signing_key=key,
    )

    with pytest.raises(OfflinePackageError):
        import_offline_package(
            package_path,
            tmp_path / "packages",
            expected_fingerprint="other",
            hmac_key=key,
        )


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_import_error_after_completed_publication_restores_previous_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure
) -> None:
    package_path = _build_test_package(tmp_path)
    destination = tmp_path / "packages"
    target = destination / "demo-1.2.3"
    target.mkdir(parents=True)
    (target / "previous.bin").write_bytes(b"previous-verified-package")
    original_rename = Path.rename

    def delivered_after_publication(source: Path, destination: Path) -> Path:
        result = original_rename(source, destination)
        if source.name == "prepared":
            raise failure("delivered after publication")
        return result

    monkeypatch.setattr(Path, "rename", delivered_after_publication)
    with pytest.raises(failure, match="delivered after publication"):
        import_offline_package(package_path, destination, hmac_key=b"test-import-key")
    assert {path.name for path in target.iterdir()} == {"previous.bin"}
    assert (target / "previous.bin").read_bytes() == b"previous-verified-package"
    assert not list(destination.glob(".offline-import-*"))


def test_import_failed_restore_preserves_interrupt_and_recovery_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_path = _build_test_package(tmp_path)
    destination = tmp_path / "packages"
    target = destination / "demo-1.2.3"
    target.mkdir(parents=True)
    (target / "previous.bin").write_bytes(b"previous-verified-package")
    original_rename = Path.rename

    def fail_publish_and_restore(source: Path, destination: Path) -> Path:
        if source.name == "prepared":
            raise KeyboardInterrupt("publication interrupted")
        if source.name == "previous":
            raise OSError("restore unavailable")
        return original_rename(source, destination)

    monkeypatch.setattr(Path, "rename", fail_publish_and_restore)
    with pytest.raises(KeyboardInterrupt, match="publication interrupted") as caught:
        import_offline_package(package_path, destination, hmac_key=b"test-import-key")
    previous = next(destination.glob(".offline-import-*/previous"))
    assert (previous / "previous.bin").read_bytes() == b"previous-verified-package"
    assert any(str(previous) in note for note in caught.value.__notes__)
