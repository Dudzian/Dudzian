"""Tests for the Stage6 resilience bundle exporter and verifier."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_export(args: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/export_resilience_bundle.py"), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )


def _run_verify(args: list[str], *, env: dict[str, str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/verify_resilience_bundle.py"), *args],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )


def _extract_member(archive: Path, name: str) -> bytes:
    with tarfile.open(archive, "r:gz") as handle:
        member = handle.getmember(name)
        extracted = handle.extractfile(member)
        assert extracted is not None
        return extracted.read()


def test_export_resilience_bundle_creates_signed_archive(tmp_path: Path) -> None:
    signing_key = "A" * 64
    env = os.environ.copy()
    env["RESILIENCE_SIGNING_KEY"] = signing_key

    report_path = tmp_path / "resilience_failover_report.json"
    report_path.write_text(json.dumps({"drills": ["binance_failover"], "status": "ok"}))

    signature_path = tmp_path / "resilience_failover_report.json.sig"
    signature_path.write_text("dummy-signature")

    extra_file = tmp_path / "operator_notes.txt"
    extra_file.write_text("All drills succeeded")

    output_dir = tmp_path / "dist"

    result = _run_export(
        [
            "--version",
            "1.0.0",
            "--report",
            str(report_path),
            "--signature",
            str(signature_path),
            "--include",
            str(extra_file),
            "--output-dir",
            str(output_dir),
            "--signing-key-env",
            "RESILIENCE_SIGNING_KEY",
            "--key-id",
            "stage6-test",
        ],
        env=env,
    )

    assert result.returncode == 0

    archive_path = output_dir / "resilience-bundle-1.0.0.tar.gz"
    assert archive_path.exists()

    with tarfile.open(archive_path, "r:gz") as archive:
        members = {member.name.lstrip("./") for member in archive.getmembers()}
        assert "manifest.json" in members
        assert "manifest.sig" in members
        assert "reports/resilience_failover_report.json" in members
        assert "extras/operator_notes.txt" in members

    manifest = json.loads(_extract_member(archive_path, "./manifest.json"))
    assert manifest["version"] == "1.0.0"
    paths = {entry["path"] for entry in manifest["files"]}
    assert "reports/resilience_failover_report.json" in paths
    assert "extras/operator_notes.txt" in paths

    verify_env = env.copy()
    verify_result = _run_verify(
        [
            "--bundle",
            str(archive_path),
            "--signing-key-env",
            "RESILIENCE_SIGNING_KEY",
        ],
        env=verify_env,
    )
    assert verify_result.returncode == 0


def test_verify_resilience_bundle_detects_tampering(tmp_path: Path) -> None:
    signing_key = "B" * 64
    env = os.environ.copy()
    env["RESILIENCE_SIGNING_KEY"] = signing_key

    report_path = tmp_path / "resilience_failover_report.json"
    report_path.write_text(json.dumps({"drills": ["kraken_failover"], "status": "ok"}))
    output_dir = tmp_path / "dist"

    _run_export(
        [
            "--version",
            "2.0.0",
            "--report",
            str(report_path),
            "--output-dir",
            str(output_dir),
            "--signing-key-env",
            "RESILIENCE_SIGNING_KEY",
        ],
        env=env,
    )

    archive_path = output_dir / "resilience-bundle-2.0.0.tar.gz"
    assert archive_path.exists()

    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(staging_dir)

    report_copy = staging_dir / "reports" / "resilience_failover_report.json"
    original = report_copy.read_text()
    report_copy.write_text(original + "\n# tampered")

    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(staging_dir, arcname=".")

    verify_env = env.copy()
    result = _run_verify(
        [
            "--bundle",
            str(archive_path),
            "--signing-key-env",
            "RESILIENCE_SIGNING_KEY",
        ],
        env=verify_env,
        check=False,
    )

    assert result.returncode != 0
    assert "Digest mismatch" in result.stderr
