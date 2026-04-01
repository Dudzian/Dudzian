from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from deploy.packaging.offline_package import build_offline_package


def _extract_member(archive_path: Path, member: str) -> bytes:
    with tarfile.open(archive_path, mode="r:gz") as archive:
        handle = archive.extractfile(member)
        assert handle is not None
        return handle.read()


def test_build_offline_package_contract_with_signing_key(tmp_path: Path) -> None:
    payload_dir = tmp_path / "payload"
    payload_dir.mkdir()
    (payload_dir / "bin" / "run.sh").parent.mkdir(parents=True)
    (payload_dir / "bin" / "run.sh").write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    (payload_dir / "config.yaml").write_text("mode: demo\n", encoding="utf-8")

    package_path = tmp_path / "signed.dudzianpkg"
    output = build_offline_package(
        package_id="offline-demo",
        version="1.2.3",
        payload_dir=payload_dir,
        output_path=package_path,
        fingerprint="HW-XYZ",
        metadata={"channel": "stable"},
        signing_key=b"secret-key",
        signing_key_id="k1",
    )

    assert output == package_path
    assert package_path.exists()

    with tarfile.open(package_path, mode="r:gz") as archive:
        members = {member.name for member in archive.getmembers()}

    assert "manifest.json" in members
    assert "payload.tar" in members
    assert "manifest.sig" in members

    manifest = json.loads(_extract_member(package_path, "manifest.json").decode("utf-8"))
    assert manifest["id"] == "offline-demo"
    assert manifest["version"] == "1.2.3"
    assert manifest["fingerprint"] == "HW-XYZ"
    assert manifest["artifacts"]

    payload_tar_bytes = _extract_member(package_path, "payload.tar")
    payload_tar_size = len(payload_tar_bytes)
    payload_tar_sha256 = hashlib.sha256(payload_tar_bytes).hexdigest()
    payload_artifact = next(
        (artifact for artifact in manifest["artifacts"] if artifact["path"] == "payload.tar"),
        None,
    )
    assert payload_artifact is not None
    assert payload_artifact["size"] == payload_tar_size
    assert payload_artifact["sha256"] == payload_tar_sha256

    payload_tar_path = tmp_path / "payload.tar"
    payload_tar_path.write_bytes(payload_tar_bytes)
    with tarfile.open(payload_tar_path, mode="r") as payload_tar:
        payload_members = {member.name for member in payload_tar.getmembers() if member.isfile()}

    assert payload_members == {"bin/run.sh", "config.yaml"}


def test_build_offline_package_omits_signature_without_key(tmp_path: Path) -> None:
    payload_dir = tmp_path / "payload"
    payload_dir.mkdir()
    (payload_dir / "data.txt").write_text("demo", encoding="utf-8")

    package_path = tmp_path / "unsigned.dudzianpkg"
    build_offline_package(
        package_id="offline-demo",
        version="1.2.4",
        payload_dir=payload_dir,
        output_path=package_path,
        fingerprint=None,
        metadata=None,
        signing_key=None,
        signing_key_id=None,
    )

    with tarfile.open(package_path, mode="r:gz") as archive:
        members = {member.name for member in archive.getmembers()}

    assert "manifest.json" in members
    assert "payload.tar" in members
    assert "manifest.sig" not in members


def test_build_offline_package_rejects_missing_payload_directory(tmp_path: Path) -> None:
    missing_payload = tmp_path / "missing_payload"
    package_path = tmp_path / "missing.dudzianpkg"

    with pytest.raises(FileNotFoundError) as excinfo:
        build_offline_package(
            package_id="offline-demo",
            version="1.2.5",
            payload_dir=missing_payload,
            output_path=package_path,
        )

    message = str(excinfo.value)
    assert str(missing_payload) in message
    assert "nie istnieje lub nie jest katalogiem" in message
