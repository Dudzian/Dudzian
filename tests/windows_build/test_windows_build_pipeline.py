from __future__ import annotations

import ast
import json
import runpy
import sys
from pathlib import Path

import pytest
import yaml
from PyInstaller.utils.hooks import collect_submodules

import ui.backend as ui_backend

from scripts.verify_windows_artifact import (
    NUMPY_OPENBLAS_PATTERN,
    QT_PLATFORM_CANDIDATES,
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


def _is_collect_ui_backend_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "collect_submodules"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "ui.backend"
    )


def _contains_collect_ui_backend_call(node: ast.AST) -> bool:
    return any(_is_collect_ui_backend_call(child) for child in ast.walk(node))


def _is_collect_delvewheel_numpy_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "collect_delvewheel_libs_directory"
        and len(node.args) >= 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "numpy"
    )


def _keyword_name(call: ast.Call, name: str) -> str | None:
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Name):
            return keyword.value.id
    return None


def _is_name_tuple(node: ast.AST, names: tuple[str, ...]) -> bool:
    return (
        isinstance(node, ast.Tuple)
        and len(node.elts) == len(names)
        and all(isinstance(elt, ast.Name) and elt.id == name for elt, name in zip(node.elts, names))
    )


def _analysis_call(tree: ast.AST) -> ast.Call:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Analysis"
        ):
            return node
    raise AssertionError("Analysis(...) call not found")


def test_spec_collects_dynamic_ui_backend_submodules():
    spec_path = ROOT / SPEC_FILE_NAME
    tree = ast.parse(spec_path.read_text(encoding="utf-8"), filename=str(spec_path))

    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "PyInstaller.utils.hooks"
        and any(alias.name == "collect_submodules" for alias in node.names)
        for node in ast.walk(tree)
    )
    assert any(_is_collect_ui_backend_call(node) for node in ast.walk(tree))
    assert any(
        isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "hiddenimports"
        and isinstance(node.op, ast.Add)
        and _contains_collect_ui_backend_call(node.value)
        for node in ast.walk(tree)
    )


def test_spec_collects_numpy_delvewheel_runtime_libs_for_pyinstaller():
    spec_path = ROOT / SPEC_FILE_NAME
    tree = ast.parse(spec_path.read_text(encoding="utf-8"), filename=str(spec_path))

    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "PyInstaller.utils.hooks"
        and any(alias.name == "collect_delvewheel_libs_directory" for alias in node.names)
        for node in ast.walk(tree)
    )

    collect_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and _is_name_tuple(node.targets[0], ("app_datas", "app_binaries"))
        and _is_collect_delvewheel_numpy_call(node.value)
    ]
    assert collect_assignments

    collect_call = collect_assignments[0].value
    assert isinstance(collect_call, ast.Call)
    assert _keyword_name(collect_call, "datas") == "app_datas"
    assert _keyword_name(collect_call, "binaries") == "app_binaries"

    analysis = _analysis_call(tree)
    assert _keyword_name(analysis, "binaries") == "app_binaries"
    assert _keyword_name(analysis, "datas") == "app_datas"


def test_pyinstaller_collects_every_dynamic_ui_backend_export():
    collected = set(collect_submodules("ui.backend"))
    required = {
        f"ui.backend.{module_path.removeprefix('.')}"
        for module_path in ui_backend._MODULE_BY_EXPORT.values()
    }

    assert required <= collected


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
    (data_root / "PySide6/plugins/platforms").mkdir(parents=True)
    (data_root / "PySide6/plugins/platforms/qwindows.dll").write_bytes(b"dll")
    numpy_libs = data_root / "numpy.libs"
    numpy_libs.mkdir(parents=True)
    (numpy_libs / "libopenblas-test.dll").write_bytes(b"dll")
    (root / EXE_NAME).write_bytes(b"exe")
    (root / "BUILD_INFO.txt").write_text(f"application={PRODUCT_NAME}\n", encoding="utf-8")
    return data_root


def test_pyside_entrypoint_can_load_without_package_execution_context():
    entrypoint = ROOT / "ui/pyside_app/__main__.py"

    namespace = runpy.run_path(
        str(entrypoint),
        run_name="cryptohunter_packaged_entrypoint_probe",
    )

    assert callable(namespace["main"])


def test_pyside_entrypoint_imports_app_with_absolute_package_name():
    entrypoint = ROOT / "ui/pyside_app/__main__.py"
    tree = ast.parse(entrypoint.read_text(encoding="utf-8"), filename=str(entrypoint))

    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "ui.pyside_app.app"
    ]

    assert any(alias.name == "main" for node in imports for alias in node.names)
    assert not any(isinstance(node, ast.ImportFrom) and node.level > 0 for node in ast.walk(tree))


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
    isolated_smoke_section = text[
        text.index("Smoke test isolated artifact") : text.index("Package ZIP")
    ]
    assert "$env:RUNNER_TEMP" in isolated_smoke_section
    assert "Set-Location $isolated" in isolated_smoke_section
    assert "Start-Process" in isolated_smoke_section
    assert f'-FilePath ".\\{EXE_NAME}"' in isolated_smoke_section
    assert "-Wait" in isolated_smoke_section
    assert "-PassThru" in isolated_smoke_section
    assert "$process.ExitCode" in isolated_smoke_section
    assert "Remove-Item $smokeReport -Force" in isolated_smoke_section
    assert "Test-Path $smokeReport -PathType Leaf" in isolated_smoke_section
    assert "verify_windows_smoke_report.py" in isolated_smoke_section
    assert "$code = $LASTEXITCODE" not in isolated_smoke_section
    assert "--smoke-report" in isolated_smoke_section
    assert "*>" not in isolated_smoke_section
    assert "verify_windows_smoke_report.py" in text
    assert f"{EXE_NAME} --config ui/config/preview_local.yaml" not in text
    assert f"Compress-Archive -Path build/output/{ONEDIR_NAME}/*" in text
    assert "Get-FileHash -Algorithm SHA256 -Path $zipPath" in text
    assert "build/reports/ZIP_SHA256.txt" in text
    assert "actions/create-release" not in text
    assert "softprops/action-gh-release" not in text
    assert "console=False" in (ROOT / SPEC_FILE_NAME).read_text(encoding="utf-8")


def test_workflow_materializes_checksums_before_writing_output():
    workflow_text = (ROOT / ".github/workflows/windows-build.yml").read_text(encoding="utf-8")
    checksum_section = workflow_text[
        workflow_text.index("$artifactRoot = (Resolve-Path") : workflow_text.index(
            "      - name: Verify artifact contents"
        )
    ]

    assert '$artifactRoot = (Resolve-Path "build/output/CryptoHunter").Path' in checksum_section
    assert '$checksumPath = Join-Path $artifactRoot "SHA256SUMS.txt"' in checksum_section
    assert "if (Test-Path $checksumPath)" in checksum_section
    assert "Remove-Item $checksumPath -Force" in checksum_section
    assert "$filesToHash = @(" in checksum_section
    assert "Get-ChildItem $artifactRoot -Recurse -File |" in checksum_section
    assert "Where-Object { $_.FullName -ne $checksumPath } |" in checksum_section
    assert "Sort-Object FullName" in checksum_section
    assert "$checksumLines = @(" in checksum_section
    assert "foreach ($file in $filesToHash)" in checksum_section
    assert "Get-FileHash -Algorithm SHA256 -Path $file.FullName" in checksum_section
    assert ".Replace('\\', '/')" in checksum_section
    assert '"$($hash.Hash.ToLower())  $relativePath"' in checksum_section
    assert "$checksumLines | Out-File -FilePath $checksumPath -Encoding utf8" in checksum_section
    assert "Copy-Item $checksumPath build/reports/SHA256SUMS.txt -Force" in checksum_section
    assert "-ErrorAction SilentlyContinue" not in checksum_section
    assert "Start-Sleep" not in checksum_section
    assert (
        "Get-ChildItem build/output/CryptoHunter -Recurse -File | ForEach-Object"
        not in workflow_text
    )
    assert "} | Out-File -FilePath build/output/CryptoHunter/SHA256SUMS.txt" not in workflow_text
    assert workflow_text.index("$filesToHash = @(") < workflow_text.index(
        "$checksumLines | Out-File"
    )
    assert workflow_text.index("$checksumLines = @(") < workflow_text.index(
        "$checksumLines | Out-File"
    )


def test_artifact_scanner_accepts_pyinstaller6_internal_layout(tmp_path):
    data_root = _create_valid_artifact(tmp_path)
    result = validate_artifact(tmp_path)
    assert artifact_data_root(tmp_path) == data_root
    assert result.data_root == data_root.resolve().as_posix()
    assert result.ok, result.to_json()


def test_artifact_scanner_accepts_actual_pyside6_platform_layout(tmp_path):
    data_root = _create_valid_artifact(tmp_path)
    plugin = data_root / "PySide6/plugins/platforms/qwindows.dll"
    assert plugin.is_file()

    result = validate_artifact(tmp_path)

    assert result.ok, result.to_json()
    assert "Qt qwindows.dll platform plugin" not in result.missing


def test_artifact_scanner_allows_grpc_root_certificate_bundle(tmp_path):
    data_root = _create_valid_artifact(tmp_path)
    roots = data_root / "grpc/_cython/_credentials/roots.pem"
    roots.parent.mkdir(parents=True, exist_ok=True)
    roots.write_text("public root certificate bundle", encoding="utf-8")

    result = validate_artifact(tmp_path)

    assert result.ok, result.to_json()
    assert "_internal/grpc/_cython/_credentials" not in result.forbidden
    assert "_internal/grpc/_cython/_credentials/roots.pem" not in result.forbidden


@pytest.mark.parametrize(
    "platform_plugin",
    [
        "PySide6/plugins/platforms/qwindows.dll",
        "PySide6/Qt/plugins/platforms/qwindows.dll",
    ],
)
def test_artifact_scanner_accepts_supported_qt_platform_layouts(tmp_path, platform_plugin):
    data_root = _create_valid_artifact(tmp_path)
    for candidate in QT_PLATFORM_CANDIDATES:
        path = data_root / candidate
        if path.is_file():
            path.unlink()
    plugin = data_root / platform_plugin
    plugin.parent.mkdir(parents=True, exist_ok=True)
    plugin.write_bytes(b"dll")

    result = validate_artifact(tmp_path)

    assert result.ok, result.to_json()
    assert "Qt qwindows.dll platform plugin" not in result.missing


def test_artifact_scanner_reports_missing_numpy_openblas_runtime(tmp_path):
    data_root = _create_valid_artifact(tmp_path)

    for candidate in (data_root / "numpy.libs").glob("libopenblas*.dll"):
        candidate.unlink()

    result = validate_artifact(tmp_path)

    assert not result.ok
    assert "NumPy OpenBLAS runtime DLL" in result.missing


def test_artifact_scanner_rejects_scipy_openblas_as_numpy_runtime(tmp_path):
    data_root = _create_valid_artifact(tmp_path)

    for candidate in (data_root / "numpy.libs").glob("libopenblas*.dll"):
        candidate.unlink()
    scipy_libs = data_root / "scipy.libs"
    scipy_libs.mkdir(parents=True)
    (scipy_libs / "libscipy_openblas-example.dll").write_bytes(b"dll")

    result = validate_artifact(tmp_path)

    assert not result.ok
    assert "NumPy OpenBLAS runtime DLL" in result.missing
    assert NUMPY_OPENBLAS_PATTERN == "numpy.libs/libopenblas*.dll"


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
        ("PySide6/plugins/platforms/qwindows.dll", "Qt qwindows.dll platform plugin"),
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
    [
        ".env",
        "credentials.json",
        "_internal/.env",
        "_internal/credentials.json",
        "_internal/secrets/token.txt",
        "_internal/private_credentials/secret.pem",
    ],
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
