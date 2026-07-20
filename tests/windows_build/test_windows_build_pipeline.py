from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from scripts.verify_windows_artifact import (
    REQUIRED_ROOT_FILES,
    artifact_data_root,
    validate_artifact,
)
from scripts.windows_build_names import (
    EXE_NAME,
    ONEDIR_NAME,
    PRODUCT_NAME,
    PYTHON_PACKAGE_NAME,
    SPEC_FILE_NAME,
    WINDOWS_ARTIFACT_PREFIX,
    WINDOWS_REPORTS_ARTIFACT_PREFIX,
)
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
    (root / EXE_NAME).write_bytes(b"exe")
    (root / "BUILD_INFO.txt").write_text(f"application={PRODUCT_NAME}\n", encoding="utf-8")
    return data_root


def test_probe_reports_required_build_contract(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    assert main() == 0
    report_path = ROOT / "build" / "reports" / "windows_environment_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["desktop_entrypoint"] == "ui.pyside_app"
    assert report["output_name"] == EXE_NAME
    assert report["application_name"] == PRODUCT_NAME
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
    assert f".\\{EXE_NAME} `" in text
    assert "--smoke-report $smokeReport" in text
    assert "*>" not in text[text.index("Smoke test isolated artifact") : text.index("Package ZIP")]
    assert "verify_windows_smoke_report.py" in text
    assert f"{EXE_NAME} --config ui/config/preview_local.yaml" not in text
    assert f"Compress-Archive -Path build/output/{ONEDIR_NAME}/*" in text
    assert "Get-FileHash -Algorithm SHA256 -Path $zipPath" in text
    assert "build/reports/ZIP_SHA256.txt" in text
    assert "actions/create-release" not in text
    assert "softprops/action-gh-release" not in text
    assert "console=False" in (ROOT / SPEC_FILE_NAME).read_text(encoding="utf-8")


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


def test_windows_pipeline_files_do_not_reference_legacy_product_names():
    checked = [
        ROOT / ".github/workflows/windows-build.yml",
        ROOT / SPEC_FILE_NAME,
        ROOT / "scripts/build_windows.ps1",
        ROOT / "scripts/windows_build_probe.py",
        ROOT / "scripts/verify_windows_artifact.py",
    ]
    for path in checked:
        text = path.read_text(encoding="utf-8")
        for legacy_name in ("GymOS", "dudzian-bot-preview"):
            assert legacy_name not in text, path


def test_authoritative_windows_product_name_is_consistent():
    workflow_text = (ROOT / ".github/workflows/windows-build.yml").read_text(encoding="utf-8")
    spec_text = (ROOT / SPEC_FILE_NAME).read_text(encoding="utf-8")
    build_script_text = (ROOT / "scripts/build_windows.ps1").read_text(encoding="utf-8")
    report = build_report()

    assert PYTHON_PACKAGE_NAME == "dudzian-bot"
    assert PRODUCT_NAME == "CryptoHunter"
    assert EXE_NAME == "CryptoHunter.exe"
    assert ONEDIR_NAME == "CryptoHunter"
    assert SPEC_FILE_NAME == "CryptoHunter.spec"
    assert WINDOWS_ARTIFACT_PREFIX == "CryptoHunter-windows"
    assert WINDOWS_REPORTS_ARTIFACT_PREFIX == "CryptoHunter-windows-reports"
    assert PRODUCT_NAME == ONEDIR_NAME
    assert EXE_NAME == f"{PRODUCT_NAME}.exe"
    assert SPEC_FILE_NAME == f"{PRODUCT_NAME}.spec"
    assert report["application_name"] == PRODUCT_NAME
    assert report["output_name"] == EXE_NAME
    assert REQUIRED_ROOT_FILES[0] == EXE_NAME
    assert f"application={PRODUCT_NAME}" in workflow_text
    assert f'artifactName = "{WINDOWS_ARTIFACT_PREFIX}-' in workflow_text
    assert f'zipName = "{WINDOWS_ARTIFACT_PREFIX}-' in workflow_text
    assert f"name: {WINDOWS_ARTIFACT_PREFIX}-" in workflow_text
    assert f"name: {WINDOWS_REPORTS_ARTIFACT_PREFIX}-" in workflow_text
    assert f"build/output/{ONEDIR_NAME}/{EXE_NAME}" in workflow_text
    assert f'name="{PRODUCT_NAME}"' in spec_text
    assert SPEC_FILE_NAME in build_script_text


def test_smoke_report_option_is_parsed_as_absolute_path(tmp_path):
    report_path = tmp_path / "nested" / "smoke.json"
    options = AppOptions.parse(["--smoke-test", "--smoke-report", str(report_path)])
    assert options.smoke is True
    assert options.smoke_report_path == report_path.resolve()
    assert options.smoke_report_path.is_absolute()


def test_smoke_report_without_smoke_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="--smoke-report"):
        AppOptions.parse(["--smoke-report", str(tmp_path / "smoke.json")])


def test_smoke_report_file_is_created_and_stdout_none_is_safe(monkeypatch, tmp_path):
    from ui.pyside_app import app as app_module
    import ui.pyside_app.smoke as smoke_module

    report_path = tmp_path / "reports" / "smoke-test.log"
    calls: dict[str, object] = {}
    captured_output = []

    def fake_run_smoke(options, *, output, force_offscreen):
        captured_output.append(output)
        calls["output_closed_during_run"] = output.closed
        calls["force_offscreen"] = force_offscreen
        output.write('{"status":"ok"}\n')
        return 0

    def fake_terminate(exit_code: int) -> None:
        calls["terminated"] = exit_code
        calls["report_closed_before_terminate"] = captured_output[0].closed

    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr(smoke_module, "run_smoke", fake_run_smoke)
    monkeypatch.setattr(app_module, "_terminate_offscreen_smoke", fake_terminate)

    assert app_module.main(["--smoke-test", "--offscreen", "--smoke-report", str(report_path)]) == 0
    assert report_path.parent.is_dir()
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "ok"
    assert calls == {
        "output_closed_during_run": False,
        "force_offscreen": True,
        "terminated": 0,
        "report_closed_before_terminate": True,
    }


def test_smoke_without_report_uses_devnull_when_stdout_is_none(monkeypatch):
    from ui.pyside_app import app as app_module
    import ui.pyside_app.smoke as smoke_module

    calls: dict[str, object] = {}

    def fake_run_smoke(options, *, output, force_offscreen):
        calls["output_is_none"] = output is None
        output.write('{"status":"ok"}\n')
        return 0

    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr(smoke_module, "run_smoke", fake_run_smoke)
    monkeypatch.setattr(app_module, "_terminate_offscreen_smoke", lambda exit_code: None)

    assert app_module.main(["--smoke-test", "--offscreen"]) == 0
    assert calls == {"output_is_none": False}


def test_verify_windows_smoke_report_requires_ok_status(tmp_path):
    from scripts.verify_windows_smoke_report import load_smoke_report

    report = tmp_path / "smoke.json"
    report.write_text('{"status":"ok"}\n', encoding="utf-8")
    assert load_smoke_report(report)["status"] == "ok"

    report.write_text('{"status":"blocked"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not ok"):
        load_smoke_report(report)
