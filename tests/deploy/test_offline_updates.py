import base64
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import pytest

from bot_core.security.signing import build_hmac_signature
from bot_core.security.hwid import HwIdProvider
from bot_core.security.update import verify_update_bundle
from bot_core.update.differential import DifferentialUpdateManager
from scripts import build_oem_release


def _sha384(path: Path) -> str:
    digest = hashlib.sha384()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_fingerprint_document(path: Path, fingerprint: str, key: bytes) -> None:
    payload = {"fingerprint": fingerprint, "generated_at": "2024-06-01T00:00:00Z"}
    signature = build_hmac_signature(payload, key=key)
    document = {"payload": payload, "signature": signature}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_verify_update_bundle_reports_embedded_signature_failure(tmp_path: Path) -> None:
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    payload_path = package_dir / "payload.tar.gz"
    payload_bytes = b"archive-content"
    payload_path.write_bytes(payload_bytes)

    manifest_payload = {
        "version": "1.2.3",
        "platform": "linux",
        "runtime": "python311",
        "artifacts": [
            {
                "path": "payload.tar.gz",
                "sha384": hashlib.sha384(payload_bytes).hexdigest(),
                "sha256": hashlib.sha256(payload_bytes).hexdigest(),
                "size": len(payload_bytes),
            }
        ],
        "signature": {
            "algorithm": "HMAC-SHA256",
            "value": "deadbeef",
        },
    }
    manifest_path = package_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    result = verify_update_bundle(
        manifest_path=manifest_path,
        base_dir=package_dir,
        hmac_key=b"verification-key",
    )

    assert result.signature_valid is False
    assert result.signature_checked is True
    assert "Podpis manifestu aktualizacji jest niepoprawny." in result.errors
    assert result.is_successful is False


def test_verify_update_bundle_requires_signature_when_key_is_configured(tmp_path: Path) -> None:
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    payload_path = package_dir / "payload.tar.gz"
    payload_bytes = b"archive-content"
    payload_path.write_bytes(payload_bytes)

    manifest_payload = {
        "version": "1.2.3",
        "platform": "linux",
        "runtime": "python311",
        "artifacts": [
            {
                "path": "payload.tar.gz",
                "sha384": hashlib.sha384(payload_bytes).hexdigest(),
                "sha256": hashlib.sha256(payload_bytes).hexdigest(),
                "size": len(payload_bytes),
            }
        ],
    }
    manifest_path = package_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    result = verify_update_bundle(
        manifest_path=manifest_path,
        base_dir=package_dir,
        hmac_key=b"verification-key",
    )

    assert result.signature_valid is False
    assert result.signature_checked is True
    assert result.is_successful is False
    assert (
        "Manifest aktualizacji nie zawiera podpisu, mimo że oczekiwano go w konfiguracji." in result.errors
    )


def test_verify_update_bundle_skips_signature_without_key(tmp_path: Path) -> None:
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    payload_path = package_dir / "payload.tar.gz"
    payload_bytes = b"archive-content"
    payload_path.write_bytes(payload_bytes)

    manifest_payload = {
        "version": "1.2.3",
        "platform": "linux",
        "runtime": "python311",
        "artifacts": [
            {
                "path": "payload.tar.gz",
                "sha384": hashlib.sha384(payload_bytes).hexdigest(),
                "sha256": hashlib.sha256(payload_bytes).hexdigest(),
                "size": len(payload_bytes),
            }
        ],
        "signature": {
            "algorithm": "HMAC-SHA256",
            "value": base64.b64encode(b"dummy").decode("ascii"),
        },
    }
    manifest_path = package_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    result = verify_update_bundle(
        manifest_path=manifest_path,
        base_dir=package_dir,
    )

    assert result.signature_valid is True
    assert result.signature_checked is False
    assert any("nie został zweryfikowany" in warning for warning in result.warnings)
    assert result.is_successful is True


def test_delta_manifest_without_key(tmp_path: Path) -> None:
    manifest_payload = {
        "base_version": "1.0.0",
        "target_version": "1.1.0",
        "bundle": "core-oem",
        "platform": "linux",
        "changed_files": ["daemon/app.bin"],
        "removed_files": [],
    }
    manifest_path = tmp_path / "delta.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    signature_payload = {"algorithm": "HS256", "signature": "deadbeef", "key_id": "unused"}
    signature_path = tmp_path / "delta.json.sig"
    signature_path.write_text(json.dumps(signature_payload), encoding="utf-8")

    manager = DifferentialUpdateManager(storage_dir=tmp_path)
    result = manager.validate_manifest(manifest_path)

    assert result.signature_valid is None
    assert result.issues == []


def test_delta_manifest_reports_missing_metadata(tmp_path: Path) -> None:
    manifest_payload = {
        "bundle": " ",
        "base_version": 101,
        "target_version": "",
        "platform": None,
    }
    manifest_path = tmp_path / "delta.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    manager = DifferentialUpdateManager(storage_dir=tmp_path)
    result = manager.validate_manifest(manifest_path)

    assert set(result.issues) == {
        "missing-bundle",
        "missing-base-version",
        "missing-target-version",
        "missing-platform",
        "missing-changed-files",
        "missing-removed-files",
    }


def test_delta_manifest_reports_invalid_file_lists(tmp_path: Path) -> None:
    manifest_payload = {
        "bundle": "core-oem",
        "base_version": "1.0.0",
        "target_version": "1.1.0",
        "platform": "linux",
        "changed_files": ["daemon/app.bin", 123, " "],
        "removed_files": "daemon/obsolete.bin",
    }
    manifest_path = tmp_path / "delta.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    manager = DifferentialUpdateManager(storage_dir=tmp_path)
    result = manager.validate_manifest(manifest_path)

    assert set(result.issues) == {"invalid-changed-files", "invalid-removed-files"}


def test_delta_manifest_reports_duplicate_entries(tmp_path: Path) -> None:
    manifest_payload = {
        "bundle": "core-oem",
        "base_version": "1.0.0",
        "target_version": "1.1.0",
        "platform": "linux",
        "changed_files": ["daemon/app.bin", "daemon/app.bin", "daemon/config.json"],
        "removed_files": ["daemon/obsolete.bin", "daemon/obsolete.bin"],
    }
    manifest_path = tmp_path / "delta.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    manager = DifferentialUpdateManager(storage_dir=tmp_path)
    result = manager.validate_manifest(manifest_path)

    assert set(result.issues) == {"duplicate-changed-files", "duplicate-removed-files"}


def test_delta_manifest_reports_conflicting_entries(tmp_path: Path) -> None:
    manifest_payload = {
        "bundle": "core-oem",
        "base_version": "1.0.0",
        "target_version": "1.1.0",
        "platform": "linux",
        "changed_files": ["daemon/app.bin"],
        "removed_files": ["daemon/app.bin", "daemon/obsolete.bin"],
    }
    manifest_path = tmp_path / "delta.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    manager = DifferentialUpdateManager(storage_dir=tmp_path)
    result = manager.validate_manifest(manifest_path)

    assert "conflicting-file-lists" in result.issues


def test_delta_manifest_rejects_invalid_versions(tmp_path: Path) -> None:
    manifest_payload = {
        "bundle": "core-oem",
        "base_version": "invalid-version",
        "target_version": "1.1.0",
        "platform": "linux",
        "changed_files": ["daemon/app.bin"],
        "removed_files": ["daemon/obsolete.bin"],
    }
    manifest_path = tmp_path / "delta.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    manager = DifferentialUpdateManager(storage_dir=tmp_path)
    result = manager.validate_manifest(manifest_path)

    assert "invalid-base-version" in result.issues


def test_delta_manifest_requires_increasing_versions(tmp_path: Path) -> None:
    manifest_payload = {
        "bundle": "core-oem",
        "base_version": "1.1.0",
        "target_version": "1.0.0",
        "platform": "linux",
        "changed_files": ["daemon/app.bin"],
        "removed_files": ["daemon/obsolete.bin"],
    }
    manifest_path = tmp_path / "delta.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    manager = DifferentialUpdateManager(storage_dir=tmp_path)
    result = manager.validate_manifest(manifest_path)

    assert "non-incremental-version" in result.issues


def test_delta_manifest_requires_at_least_one_file(tmp_path: Path) -> None:
    manifest_payload = {
        "bundle": "core-oem",
        "base_version": "1.0.0",
        "target_version": "1.1.0",
        "platform": "linux",
        "changed_files": [],
        "removed_files": [],
    }
    manifest_path = tmp_path / "delta.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    manager = DifferentialUpdateManager(storage_dir=tmp_path)
    result = manager.validate_manifest(manifest_path)

    assert "empty-file-lists" in result.issues


def test_delta_manifest_rejects_unsafe_paths(tmp_path: Path) -> None:
    manifest_payload = {
        "bundle": "core-oem",
        "base_version": "1.0.0",
        "target_version": "1.1.0",
        "platform": "linux",
        "changed_files": ["../etc/passwd", "daemon/app.bin"],
        "removed_files": ["/absolute/path", "daemon\\old.dll"],
    }
    manifest_path = tmp_path / "delta.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    manager = DifferentialUpdateManager(storage_dir=tmp_path)
    result = manager.validate_manifest(manifest_path)

    assert "invalid-changed-file-path" in result.issues
    assert "invalid-removed-file-path" in result.issues


def test_delta_manifest_rejects_non_normalized_paths(tmp_path: Path) -> None:
    manifest_payload = {
        "bundle": "core-oem",
        "base_version": "1.0.0",
        "target_version": "1.1.0",
        "platform": "linux",
        "changed_files": ["./daemon/app.bin", "daemon/app.bin/"],
        "removed_files": ["daemon//obsolete.bin"],
    }
    manifest_path = tmp_path / "delta.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    manager = DifferentialUpdateManager(storage_dir=tmp_path)
    result = manager.validate_manifest(manifest_path)

    assert result.issues.count("invalid-changed-file-path") == 1
    assert result.issues.count("invalid-removed-file-path") == 1


def test_delta_manifest_rejects_casefold_duplicates(tmp_path: Path) -> None:
    manifest_payload = {
        "bundle": "core-oem",
        "base_version": "1.0.0",
        "target_version": "1.1.0",
        "platform": "linux",
        "changed_files": ["daemon/App.bin", "Daemon/app.bin"],
        "removed_files": ["config/archive.cfg"],
    }
    manifest_path = tmp_path / "delta.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    manager = DifferentialUpdateManager(storage_dir=tmp_path)
    result = manager.validate_manifest(manifest_path)

    assert "duplicate-changed-files-casefold" in result.issues


def test_delta_manifest_rejects_casefold_conflicts_between_lists(tmp_path: Path) -> None:
    manifest_payload = {
        "bundle": "core-oem",
        "base_version": "1.0.0",
        "target_version": "1.1.0",
        "platform": "linux",
        "changed_files": ["daemon/app.bin"],
        "removed_files": ["Daemon/App.bin"],
    }
    manifest_path = tmp_path / "delta.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    manager = DifferentialUpdateManager(storage_dir=tmp_path)
    result = manager.validate_manifest(manifest_path)

    assert "conflicting-file-lists-casefold" in result.issues


@pytest.mark.parametrize("embed_hwid", [True])
def test_offline_update_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embed_hwid: bool) -> None:
    fingerprint_value = "OEM-FP-999"
    monkeypatch.setattr("bot_core.security.hwid.get_local_fingerprint", lambda: fingerprint_value)

    hmac_key = b"oem-hmac-key"
    delta_key = b"delta-secret"
    package_key = b"package-secret"
    monkeypatch.setenv("OEM_PIPELINE_HMAC", base64.b64encode(hmac_key).decode("ascii"))
    monkeypatch.setenv("OEM_PIPELINE_DELTA_KEY", base64.b64encode(delta_key).decode("ascii"))
    monkeypatch.setenv("OEM_PIPELINE_PACKAGE_KEY", base64.b64encode(package_key).decode("ascii"))

    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "daemon").mkdir()
    base_app = base_dir / "daemon" / "app.bin"
    base_app.write_bytes(b"OLD")
    base_fingerprint_path = base_dir / "config" / "fingerprint.expected.json"
    _write_fingerprint_document(base_fingerprint_path, fingerprint_value, hmac_key)

    base_manifest_dir = tmp_path / "base-bundle"
    base_manifest_dir.mkdir()
    base_manifest = {
        "version": "1.0.0",
        "files": [
            {"path": "daemon/app.bin", "sha384": _sha384(base_app)},
            {"path": "config/fingerprint.expected.json", "sha384": _sha384(base_fingerprint_path)},
        ],
    }
    (base_manifest_dir / "manifest.json").write_text(json.dumps(base_manifest), encoding="utf-8")

    staging_root = tmp_path / "staging"
    (staging_root / "daemon").mkdir(parents=True)
    new_app = staging_root / "daemon" / "app.bin"
    new_app.write_bytes(b"NEW")
    fingerprint_doc = staging_root / "config" / "fingerprint.expected.json"
    _write_fingerprint_document(fingerprint_doc, fingerprint_value, hmac_key)

    archive_path = tmp_path / "installer.zip"
    archive_path.write_bytes(b"installer-binary")

    manifest_payload = {
        "bundle": "core-oem",
        "version": "1.1.0",
        "platform": "linux",
        "files": [
            {"path": "daemon/app.bin", "sha384": _sha384(new_app)},
            {"path": "config/fingerprint.expected.json", "sha384": _sha384(fingerprint_doc)},
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    pipeline_config = {
        "fingerprint_validation": {
            "expected": fingerprint_value,
            "hmac_key": "env:OEM_PIPELINE_HMAC",
        },
        "delta_updates": {
            "bases": [base_manifest_dir.name],
            "output_dir": "delta",
            "manifest_dir": "delta/manifests",
            "manifest_signing_key": "env:OEM_PIPELINE_DELTA_KEY",
            "manifest_signing_key_id": "delta-key",
            "embed_hwid": embed_hwid,
        },
        "update_package": {
            "output_dir": "packages",
            "package_id": "core-oem",
            "runtime": "python311",
            "signing_key": "env:OEM_PIPELINE_PACKAGE_KEY",
            "signing_key_id": "pkg-key",
            "embed_hwid": embed_hwid,
        },
        "code_signing": {
            "command": ["codesign", "--sign", "OEM", "{target}"],
            "targets": ["archive"],
            "dry_run": True,
        },
    }
    pipeline_config_path = tmp_path / "pipeline.json"
    pipeline_config_path.write_text(json.dumps(pipeline_config), encoding="utf-8")

    release_dir = tmp_path / "releases"

    build_oem_release.main(
        [
            "--pipeline-config",
            str(pipeline_config_path),
            "--manifest",
            str(manifest_path),
            "--staging-root",
            str(staging_root),
            "--archive",
            str(archive_path),
            "--release-dir",
            str(release_dir),
        ]
    )

    release_manifest_path = release_dir / "core-oem-1.1.0-linux.json"
    assert release_manifest_path.exists()
    release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))

    delta_entry = release_manifest["report"]["delta_manifests"][0]
    manager = DifferentialUpdateManager(
        storage_dir=tmp_path / "downloads",
        manifest_key=delta_key,
        package_key=package_key,
        hwid_provider=HwIdProvider(),
    )
    delta_validation = manager.validate_manifest(
        Path(delta_entry["manifest_path"]),
        signature_path=Path(delta_entry["signature_path"]),
    )
    assert delta_validation.signature_valid is True
    assert delta_validation.fingerprint_ok is True
    assert delta_validation.payload["fingerprint"] == fingerprint_value

    package_entry = release_manifest["report"]["update_packages"][0]
    package_path = Path(package_entry["package_path"])
    verification = manager.verify_package(package_path.parent)
    assert verification.is_successful
    assert verification.manifest.metadata.get("fingerprint") == fingerprint_value

    install_dir = tmp_path / "install"
    with tarfile.open(package_path, mode="r:gz") as archive:
        archive.extractall(install_dir)
    assert (install_dir / "daemon" / "app.bin").read_bytes() == b"NEW"

    runtime_dir = tmp_path / "runtime"
    shutil.copytree(base_dir, runtime_dir)
    shutil.copy2(install_dir / "daemon" / "app.bin", runtime_dir / "daemon" / "app.bin")
    assert (runtime_dir / "daemon" / "app.bin").read_bytes() == b"NEW"

    manager.rollback(base_dir, runtime_dir)
    assert (runtime_dir / "daemon" / "app.bin").read_bytes() == b"OLD"
    restored_doc = json.loads((runtime_dir / "config" / "fingerprint.expected.json").read_text(encoding="utf-8"))
    assert restored_doc["payload"]["fingerprint"] == fingerprint_value
