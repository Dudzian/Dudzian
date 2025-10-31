from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.update.offline_updater import OfflinePackageError, import_kbot_package
from scripts.package_update import build_kbot_package


def _create_payload(tmp_path: Path) -> Path:
    payload_dir = tmp_path / "payload"
    (payload_dir / "models").mkdir(parents=True)
    (payload_dir / "models" / "model.bin").write_bytes(b"binary-model")
    (payload_dir / "strategies").mkdir()
    (payload_dir / "strategies" / "alpha.json").write_text("{}", encoding="utf-8")
    return payload_dir


def _read_manifest(path: Path) -> dict[str, object]:
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def test_import_kbot_package_success(tmp_path: Path) -> None:
    payload_dir = _create_payload(tmp_path)
    package_path = tmp_path / "update.kbot"
    key = b"super-secret"

    build_kbot_package(
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
    result = import_kbot_package(
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


def test_import_kbot_package_invalid_signature(tmp_path: Path) -> None:
    payload_dir = _create_payload(tmp_path)
    package_path = tmp_path / "update.kbot"
    build_kbot_package(
        package_id="demo",
        version="1.0.0",
        payload_dir=payload_dir,
        output_path=package_path,
        signing_key=b"correct",
    )

    with pytest.raises(OfflinePackageError):
        import_kbot_package(package_path, tmp_path / "packages", hmac_key=b"wrong")


def test_import_kbot_package_fingerprint_mismatch(tmp_path: Path) -> None:
    payload_dir = _create_payload(tmp_path)
    package_path = tmp_path / "update.kbot"
    key = b"fingerprint"
    build_kbot_package(
        package_id="demo",
        version="1.0.0",
        payload_dir=payload_dir,
        output_path=package_path,
        fingerprint="expected",
        signing_key=key,
    )

    with pytest.raises(OfflinePackageError):
        import_kbot_package(
            package_path,
            tmp_path / "packages",
            expected_fingerprint="other",
            hmac_key=key,
        )
