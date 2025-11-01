from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from core.support.diagnostics import DiagnosticsError, create_diagnostics_package


@pytest.fixture()
def sample_project(tmp_path: Path) -> Path:
    base = tmp_path / "project"
    (base / "logs").mkdir(parents=True)
    (base / "config").mkdir(parents=True)
    (base / "reports").mkdir(parents=True)
    (base / "logs" / "app.log").write_text("log entry", encoding="utf-8")
    (base / "config" / "settings.yml").write_text("key: value", encoding="utf-8")
    (base / "reports" / "summary.txt").write_text("report", encoding="utf-8")
    return base


def test_create_diagnostics_package(sample_project: Path, tmp_path: Path) -> None:
    package = create_diagnostics_package(
        destination=tmp_path,
        base_path=sample_project,
        metadata={"ticket": "ABC-1"},
    )

    assert package.archive_path.exists()

    with zipfile.ZipFile(package.archive_path) as archive:
        names = set(archive.namelist())
        assert "logs/app.log" in names
        assert "config/settings.yml" in names
        assert "reports/summary.txt" in names
        assert "manifest/diagnostics.json" in names
        manifest = json.loads(archive.read("manifest/diagnostics.json").decode("utf-8"))
        assert manifest["ticket"] == "ABC-1"
        assert "files" in manifest and len(manifest["files"]) >= 3


def test_create_diagnostics_package_without_sources(tmp_path: Path) -> None:
    empty_base = tmp_path / "empty"
    empty_base.mkdir()

    with pytest.raises(DiagnosticsError):
        create_diagnostics_package(destination=tmp_path, base_path=empty_base)
