from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.packaging import validate_desktop_manifest as validator


@pytest.fixture
def manifest_payload(tmp_path: Path) -> Path:
    py_dir = tmp_path / "py"
    py_dir.mkdir()
    exe = py_dir / "app" / "app"
    exe.parent.mkdir()
    exe.write_text("bin", encoding="utf-8")

    brief_dir = tmp_path / "brief"
    brief_dir.mkdir()
    pkg = brief_dir / "FinanceApp-1.0.zip"
    pkg.write_bytes(b"zip")

    payload = {
        "pyinstaller": [
            {
                "name": "app/app",
                "size_bytes": exe.stat().st_size,
                "sha256": validator._sha256(exe),  # noqa: SLF001 - helper for fixtures
            }
        ],
        "briefcase": [
            {
                "name": "FinanceApp-1.0.zip",
                "size_bytes": pkg.stat().st_size,
                "sha256": validator._sha256(pkg),  # noqa: SLF001 - helper for fixtures
            }
        ],
    }

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def test_main_validates_manifest(monkeypatch, tmp_path: Path, manifest_payload: Path) -> None:
    args = [
        str(manifest_payload),
        "--pyinstaller-root",
        str(tmp_path / "py"),
        "--briefcase-root",
        str(tmp_path / "brief"),
    ]

    assert validator.main(args) == 0


def test_main_detects_missing_file(manifest_payload: Path, tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"
    missing_root.mkdir()

    with pytest.raises(SystemExit) as excinfo:
        validator.main([
            str(manifest_payload),
            "--pyinstaller-root",
            str(missing_root),
            "--briefcase-root",
            str(tmp_path / "brief"),
        ])

    assert "plik nie istnieje" in str(excinfo.value)


def test_main_detects_mismatched_hash(monkeypatch, tmp_path: Path, manifest_payload: Path) -> None:
    manifest = json.loads(manifest_payload.read_text(encoding="utf-8"))
    manifest["briefcase"][0]["sha256"] = "0" * 64
    manifest_payload.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        validator.main([
            str(manifest_payload),
            "--pyinstaller-root",
            str(tmp_path / "py"),
            "--briefcase-root",
            str(tmp_path / "brief"),
        ])

    assert "hash" in str(excinfo.value)
