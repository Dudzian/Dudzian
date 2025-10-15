"""Tests for the Stage4 observability bundle exporter."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/export_observability_bundle.py"), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=REPO_ROOT,
    )


def _run_verify(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/verify_signature.py"), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=REPO_ROOT,
    )


def _read_manifest(archive_path: Path) -> dict:
    with tarfile.open(archive_path, "r:gz") as archive:
        with archive.extractfile("./manifest.json") as handle:
            assert handle is not None
            return json.load(handle)


def test_export_observability_bundle_creates_signed_archive(tmp_path: Path) -> None:
    signing_key = tmp_path / "sign.key"
    signing_key.write_bytes(os.urandom(48))
    signing_key.chmod(0o600)

    output_dir = tmp_path / "dist"
    result = _run_cli(
        [
            "--version",
            "1.2.3",
            "--output-dir",
            str(output_dir),
            "--signing-key",
            str(signing_key),
            "--key-id",
            "test-key",
        ]
    )

    assert result.returncode == 0

    archive_path = output_dir / "observability-bundle-1.2.3.tar.gz"
    assert archive_path.exists()

    with tarfile.open(archive_path, "r:gz") as archive:
        members = {member.name for member in archive.getmembers()}
        normalized = {name.lstrip("./") for name in members}
        assert "manifest.json" in normalized
        assert "manifest.sig" in normalized
        assert any(entry.startswith("dashboards/") for entry in normalized)
        assert any(entry.startswith("alert_rules/") for entry in normalized)

        manifest_bytes = archive.extractfile("./manifest.json")
        assert manifest_bytes is not None
        (tmp_path / "manifest.json").write_bytes(manifest_bytes.read())

        signature_bytes = archive.extractfile("./manifest.sig")
        assert signature_bytes is not None
        (tmp_path / "manifest.sig").write_bytes(signature_bytes.read())

    verify_result = _run_verify(
        [
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--signature",
            str(tmp_path / "manifest.sig"),
            "--signing-key",
            str(signing_key),
            "--digest",
            "sha384",
        ]
    )
    assert verify_result.returncode == 0

    manifest = _read_manifest(archive_path)
    assert manifest["version"] == "1.2.3"
    assert manifest["dashboards"]
    assert manifest["alert_rules"]
    assert "alert_rules/stage6_alerts.yaml" in manifest["alert_rules"]
    for entry in manifest["files"]:
        assert "sha256" in entry
        assert entry["path"]

def test_export_observability_bundle_prevents_overwrite(tmp_path: Path) -> None:
    signing_key = tmp_path / "sign.key"
    signing_key.write_bytes(os.urandom(48))
    signing_key.chmod(0o600)

    output_dir = tmp_path / "dist"
    _run_cli(
        [
            "--version",
            "9.9.9",
            "--output-dir",
            str(output_dir),
            "--signing-key",
            str(signing_key),
        ]
    )

    with pytest.raises(subprocess.CalledProcessError):
        _run_cli(
            [
                "--version",
                "9.9.9",
                "--output-dir",
                str(output_dir),
                "--signing-key",
                str(signing_key),
            ]
        )
