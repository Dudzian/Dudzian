from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest
import yaml

from scripts.verify_windows_artifact import artifact_data_root, validate_artifact
from scripts.windows_build_probe import build_report, main
from ui.pyside_app.app import AppOptions
from ui.pyside_app.runtime_paths import default_config_path, default_qml_path, qml_import_roots

ROOT = Path(__file__).resolve().parents[2]


def _create_valid_artifact(root: Path, *, internal: bool = True) -> Path:
    data_root = root / "_internal" if internal else root
    (data_root / "ui/config").mkdir(parents=True)
    (data_root / "ui/config/preview_local.yaml").write_text(
        "endpoint: in-process\n", encoding="utf-8"
    )
    (data_root / "ui/pyside_app/qml").mkdir(parents=True)
    (data_root / "ui/pyside_app/qml/MainWindow.qml").write_text(
        "import QtQuick\nItem {}\n", encoding="utf-8"
    )
    (data_root / "ui/qml").mkdir(parents=True)
    (data_root / "ui/qml/Icon.qml").write_text("import QtQuick\nItem {}\n", encoding="utf-8")
    (data_root / "PySide6/Qt/plugins/platforms").mkdir(parents=True)
    (data_root / "PySide6/Qt/plugins/platforms/qwindows.dll").write_bytes(b"dll")
    (root / "GymOS.exe").write_bytes(b"exe")
    (root / "BUILD_INFO.txt").write_text("application=GymOS\n", encoding="utf-8")
    return data_root


def test_probe_reports_required_build_contract(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    assert main() == 0
    report_path = ROOT / "build" / "reports" / "windows_environment_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["desktop_entrypoint"] == "ui.pyside_app"
    assert report["output_name"] == "GymOS.exe"
    assert report["application_version"] == "0.1.0"
    assert "ui/pyside_app/qml/MainWindow.qml" in report["qml_assets"]["ui/pyside_app/qml"]
    assert "platforms" in report["required_qt_plugins"]
    assert any(item["path"] == "requirements.txt" for item in report["dependency_and_lock_files"])


def test_source_resource_resolution_uses_repo_resources():
    assert default_config_path() == (ROOT / "ui/config/preview_local.yaml").resolve()
    assert default_qml_path() == (ROOT / "ui/pyside_app/qml/MainWindow.qml").resolve()
    assert (ROOT / "ui/qml").resolve() in qml_import_roots(default_qml_path())


def test_simulated_frozen_resource_resolution(monkeypatch, tmp_path):
    artifact = tmp_path / "artifact"
    bundle = artifact / "_internal"
    (bundle / "ui/config").mkdir(parents=True)
    (bundle / "ui/config/preview_local.yaml").write_text("endpoint: in-process\n", encoding="utf-8")
    (bundle / "ui/pyside_app/qml").mkdir(parents=True)
    (bundle / "ui/pyside_app/qml/MainWindow.qml").write_text(
        "import QtQuick\nItem {}\n", encoding="utf-8"
    )
    (bundle / "ui/qml").mkdir(parents=True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    assert default_config_path() == (bundle / "ui/config/preview_local.yaml").resolve()
    assert default_qml_path() == (bundle / "ui/pyside_app/qml/MainWindow.qml").resolve()
    assert (bundle / "ui/qml").resolve() in qml_import_roots(default_qml_path())


def test_default_options_use_bundled_preview_config_and_qml_not_example():
    options = AppOptions.parse([])
    assert options.config_path.name == "preview_local.yaml"
    assert "example.yaml" not in options.config_path.as_posix()
    assert options.qml_path is None


def test_explicit_config_and_qml_still_work(tmp_path):
    config = tmp_path / "custom.yaml"
    qml = tmp_path / "Custom.qml"
    config.write_text("endpoint: in-process\n", encoding="utf-8")
    qml.write_text("import QtQuick\nItem {}\n", encoding="utf-8")
    options = AppOptions.parse(["--config", str(config), "--qml", str(qml)])
    assert options.config_path == config.resolve()
    assert options.qml_path == qml.resolve()


def test_smoke_test_alias_uses_safe_smoke_mode():
    options = AppOptions.parse(["--smoke-test", "--offscreen"])
    assert options.smoke is True
    assert options.offscreen is True
    assert options.enable_cloud_runtime is False


def test_workflow_uses_isolated_artifact_smoke_and_lockfile():
    workflow_path = ROOT / ".github" / "workflows" / "windows-build.yml"
    workflow = yaml.safe_load(workflow_path.read_text())
    assert workflow["on"]["workflow_dispatch"] is None
    job = workflow["jobs"]["build-windows"]
    assert job["runs-on"] == "windows-latest"
    text = workflow_path.read_text()
    assert "python -m pip install -r deploy/packaging/requirements-desktop.lock" in text
    assert "python -m pip install -e .[dev,desktop]" in text
    assert "python -m pip install -e .[dev,desktop] --no-deps" not in text
    assert "python -m pip check" in text
    assert "$env:RUNNER_TEMP" in text
    assert "Set-Location $isolated" in text
    assert ".\\GymOS.exe --smoke-test --offscreen" in text
    assert "GymOS.exe --config ui/config/preview_local.yaml" not in text
    assert "Compress-Archive -Path build/output/GymOS/*" in text
    assert "Get-FileHash -Algorithm SHA256 -Path $zipPath" in text
    assert "build/reports/ZIP_SHA256.txt" in text
    assert "actions/create-release" not in text
    assert "softprops/action-gh-release" not in text


def test_artifact_scanner_accepts_pyinstaller6_internal_layout(tmp_path):
    data_root = _create_valid_artifact(tmp_path)
    result = validate_artifact(tmp_path)
    assert artifact_data_root(tmp_path) == data_root
    assert result.data_root == data_root.resolve().as_posix()
    assert result.ok, result.to_json()


def test_artifact_scanner_accepts_flat_fallback_layout(tmp_path):
    data_root = _create_valid_artifact(tmp_path, internal=False)
    result = validate_artifact(tmp_path)
    assert artifact_data_root(tmp_path) == data_root
    assert result.data_root == tmp_path.resolve().as_posix()
    assert result.ok, result.to_json()


@pytest.mark.parametrize(
    "relative,expected",
    [
        ("ui/config/preview_local.yaml", "ui/config/preview_local.yaml"),
        ("ui/pyside_app/qml/MainWindow.qml", "ui/pyside_app/qml/MainWindow.qml"),
        ("ui/qml", "ui/qml"),
        ("PySide6/Qt/plugins/platforms/qwindows.dll", "Qt qwindows.dll platform plugin"),
    ],
)
def test_artifact_scanner_requires_internal_data_files(tmp_path, relative, expected):
    data_root = _create_valid_artifact(tmp_path)
    target = data_root / relative
    if target.is_dir():
        for child in sorted(target.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            else:
                child.rmdir()
        target.rmdir()
    else:
        target.unlink()
    result = validate_artifact(tmp_path)
    assert not result.ok
    assert expected in result.missing


def test_root_level_source_ui_does_not_mask_missing_internal_resources(tmp_path):
    data_root = _create_valid_artifact(tmp_path)
    (data_root / "ui/config/preview_local.yaml").unlink()
    (tmp_path / "ui/config").mkdir(parents=True)
    (tmp_path / "ui/config/preview_local.yaml").write_text(
        "endpoint: in-process\n", encoding="utf-8"
    )
    result = validate_artifact(tmp_path)
    assert not result.ok
    assert "ui/config/preview_local.yaml" in result.missing


@pytest.mark.parametrize(
    "relative",
    [".env", "credentials.json", "_internal/.env", "_internal/secrets/token.txt"],
)
def test_artifact_scanner_rejects_forbidden_paths_in_root_and_internal(tmp_path, relative):
    _create_valid_artifact(tmp_path)
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("", encoding="utf-8")
    result = validate_artifact(tmp_path)
    assert not result.ok
    assert relative in result.forbidden
