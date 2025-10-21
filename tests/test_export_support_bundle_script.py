"""Integration tests for the export_support_bundle CLI."""
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "export_support_bundle.py"


@pytest.fixture
def sample_root(tmp_path: Path) -> Path:
    """Populate a temporary repository-style tree for the CLI to pack."""
    # Create directories that mirror defaults.
    (tmp_path / "logs").mkdir(parents=True)
    (tmp_path / "var" / "reports").mkdir(parents=True)
    (tmp_path / "var" / "licenses").mkdir(parents=True)
    (tmp_path / "var" / "metrics").mkdir(parents=True)
    (tmp_path / "var" / "audit").mkdir(parents=True)

    # Populate with some files to ensure counting logic works.
    (tmp_path / "logs" / "engine.log").write_text("engine ready\n", encoding="utf-8")
    (tmp_path / "var" / "reports" / "summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "var" / "licenses" / "oem.lic").write_text("dummy", encoding="utf-8")
    (tmp_path / "var" / "metrics" / "fps.prom").write_text("fps 60", encoding="utf-8")
    (tmp_path / "var" / "audit" / "audit.log").write_text("audit", encoding="utf-8")

    extra_dir = tmp_path / "artifacts"
    extra_dir.mkdir()
    (extra_dir / "notes.txt").write_text("bundle note", encoding="utf-8")

    return tmp_path


def _run_cli(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        cwd=str(cwd or REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )


def test_export_support_bundle_creates_archive(sample_root: Path, tmp_path: Path) -> None:
    output = tmp_path / "bundle-out"
    args = [
        "--root",
        str(sample_root),
        "--output",
        str(output / "bundle.tar.gz"),
        "--include",
        "artifacts=" + str(sample_root / "artifacts"),
        "--metadata",
        "operator=qa",
    ]

    result = _run_cli(args)

    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"

    bundle_path = Path(payload["bundle_path"])
    assert bundle_path.exists()
    assert bundle_path.suffixes[-2:] == [".tar", ".gz"]

    # Validate manifest basics.
    manifest_entries = payload["entries"]
    labels = {entry["label"] for entry in manifest_entries}
    assert {"logs", "reports", "licenses", "metrics", "audit", "artifacts"}.issubset(labels)

    # Ensure archive contains the expected files.
    with tarfile.open(bundle_path, mode="r:gz") as archive:
        members = {member.name for member in archive.getmembers()}
        assert any(name.startswith("logs/") for name in members)
        assert any(name.startswith("artifacts/") for name in members)
        assert "bundle_manifest.json" in members


def test_export_support_bundle_respects_disable_and_dry_run(sample_root: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "dist"
    args = [
        "--root",
        str(sample_root),
        "--output-dir",
        str(output_dir),
        "--basename",
        "support",
        "--format",
        "zip",
        "--disable",
        "metrics",
        "--dry-run",
        "--metadata",
        "region=eu-central",
    ]

    result = _run_cli(args)

    payload = json.loads(result.stdout)
    assert payload["status"] == "preview"
    assert payload["format"] == "zip"
    assert payload["metadata"]["region"] == "eu-central"
    assert payload["metadata"]["origin"] == "desktop_ui"
    labels = {entry["label"] for entry in payload["entries"]}
    assert "metrics" not in labels

    # Dry-run should not create files.
    assert not list(output_dir.glob("support-*.zip"))

    # Execute real run and inspect archive content names.
    real_args = [arg for arg in args if arg != "--dry-run"]
    result = _run_cli(real_args)
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"

    bundle_path = Path(payload["bundle_path"])
    assert bundle_path.exists()

    with zipfile.ZipFile(bundle_path, mode="r") as archive:
        members = set(archive.namelist())
        assert "metrics/fps.prom" not in members
        assert "licenses/oem.lic" in members
        assert "bundle_manifest.json" in members

