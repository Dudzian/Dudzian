from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import update_package


def _write_file(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_build_and_verify_package(tmp_path: Path) -> None:
    payload_path = _write_file(tmp_path / "payload.tar", "payload-data")
    diff_path = _write_file(tmp_path / "payload.diff", "diff-data")
    integrity_path = _write_file(tmp_path / "integrity.json", json.dumps({"files": []}))

    output_dir = tmp_path / "package"
    key_hex = "00112233445566778899AABBCCDDEEFF"

    exit_code = update_package.main(
        [
            "build",
            "--output-dir",
            str(output_dir),
            "--package-id",
            "pkg-1",
            "--version",
            "2024.04",
            "--platform",
            "linux-x86_64",
            "--runtime",
            "desktop",
            "--payload",
            str(payload_path),
            "--diff",
            str(diff_path),
            "--base-id",
            "2024.03",
            "--integrity-manifest",
            str(integrity_path),
            "--metadata",
            json.dumps({"required_modules": ["oem_updater"]}),
            "--key",
            key_hex,
            "--key-id",
            "build-key",
        ]
    )

    assert exit_code == 0
    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifacts"][0]["type"] == "full"
    assert any(artifact["type"] == "diff" for artifact in manifest["artifacts"])
    assert manifest["signature"]["key_id"] == "build-key"

    exit_code_verify = update_package.main(
        [
            "verify",
            "--package-dir",
            str(output_dir),
            "--key",
            key_hex,
        ]
    )

    assert exit_code_verify == 0


def test_verify_package_fails_without_manifest(tmp_path: Path) -> None:
    package_dir = tmp_path / "missing"
    package_dir.mkdir()

    exit_code = update_package.main([
        "verify",
        "--package-dir",
        str(package_dir),
    ])

    assert exit_code == 1


def test_scan_packages_reports_results(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    payload_path = _write_file(tmp_path / "payload.tar", "payload")
    package_dir = tmp_path / "pkg"

    exit_code = update_package.main(
        [
            "build",
            "--output-dir",
            str(package_dir),
            "--package-id",
            "pkg-123",
            "--version",
            "1.0.0",
            "--platform",
            "linux",
            "--runtime",
            "desktop",
            "--payload",
            str(payload_path),
        ]
    )
    assert exit_code == 0
    capfd.readouterr()  # wyczyść stdout po komendzie build

    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    broken_dir.joinpath("manifest.json").write_text("{\"invalid\": }", encoding="utf-8")

    scan_code = update_package.main([
        "scan",
        "--packages-dir",
        str(tmp_path),
    ])
    assert scan_code == 0

    captured = capfd.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "ok"
    entries = {entry["id"]: entry for entry in payload["packages"] if entry.get("status") == "ok"}
    assert "pkg-123" in entries
    assert entries["pkg-123"]["version"] == "1.0.0"

    errors = [entry for entry in payload["packages"] if entry.get("status") == "error"]
    assert errors and "manifest" in errors[0]["error"].lower()
