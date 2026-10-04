from __future__ import annotations

import ast
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from deployment.windows_installer.build import (
    InstallerBuildError,
    ENTRYPOINTS,
    STAGE9_SCHEMA_NAMES,
    build_executables,
    numpy_openblas_dll,
    qualify_pyinstaller_warnings,
    qualify_win32_imports,
    smoke_executable,
)
from deployment.windows_installer.dependency_contract import (
    AI_DEFAULTS_PACKAGE,
    AI_DEFAULTS_RESOURCE,
    AI_DEFAULTS_SMOKE_MARKER,
    REQUIRED_WIN32_MODULES,
)
from deployment.windows_installer.service_base import (
    build_smoke_requested,
    qualify_ai_defaults_resource,
)

ROOT = Path(__file__).parents[2]


def test_win32_import_preflight_requires_complete_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []

    def importer(name: str) -> object:
        observed.append(name)
        if name == "win32timezone":
            raise ModuleNotFoundError(name)
        return object()

    monkeypatch.setattr("deployment.windows_installer.build.importlib.import_module", importer)
    with pytest.raises(InstallerBuildError, match="win32timezone"):
        qualify_win32_imports()
    assert observed == list(REQUIRED_WIN32_MODULES)


def test_required_surface_covers_all_production_pywin32_imports() -> None:
    sources = (
        "service_base.py",
        "postgresql_service.py",
        "backend_service.py",
        "verifier_service.py",
        "provision.py",
    )
    imported: set[str] = set()
    for source in sources:
        tree = ast.parse(
            (ROOT / "deployment/windows_installer" / source).read_text(encoding="utf-8")
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
    used = {name for name in imported if name == "servicemanager" or name.startswith("win32")}
    assert used <= set(REQUIRED_WIN32_MODULES)
    assert set(REQUIRED_WIN32_MODULES) == {
        "servicemanager",
        "win32api",
        "win32con",
        "win32event",
        "win32job",
        "win32process",
        "win32security",
        "win32service",
        "win32serviceutil",
        "win32timezone",
    }


def test_pyinstaller_warning_qualifier_rejects_only_required_win32(
    tmp_path: Path,
) -> None:
    warnings = tmp_path / "warn.txt"
    warnings.write_text("missing module named optional_database_backend\n", encoding="utf-8")
    qualify_pyinstaller_warnings(warnings)
    warnings.write_text("missing module named 'win32security'\n", encoding="utf-8")
    with pytest.raises(InstallerBuildError, match="win32security"):
        qualify_pyinstaller_warnings(warnings)


def test_numpy_openblas_qualification_requires_wheel_owned_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "site-packages/numpy"
    package.mkdir(parents=True)
    init = package / "__init__.py"
    init.write_text("", encoding="utf-8")
    runtime = tmp_path / "site-packages/numpy.libs/libopenblas64__test.dll"
    runtime.parent.mkdir()
    runtime.write_bytes(b"dll")
    monkeypatch.setattr(
        "deployment.windows_installer.build.importlib.import_module",
        lambda name: SimpleNamespace(__file__=str(init)),
    )
    assert numpy_openblas_dll() == runtime
    runtime.unlink()
    with pytest.raises(InstallerBuildError, match="found 0"):
        numpy_openblas_dll()


def test_packaged_smoke_is_read_only_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "CryptoHunterProvision.exe"
    executable.write_bytes(b"exe")
    calls: list[object] = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=(
                f"{AI_DEFAULTS_SMOKE_MARKER}\nBUILD_SMOKE = PASS\nSTAGE9_SCHEMA_RESOURCES = PASS\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("deployment.windows_installer.build.subprocess.run", run)
    smoke_executable(executable)
    assert calls[0][0] == [str(executable), "--build-smoke"]
    assert build_smoke_requested(["install"]) is False


def test_packaged_smoke_rejects_missing_ai_defaults_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "CryptoHunterBackend.exe"
    executable.write_bytes(b"exe")
    monkeypatch.setattr(
        "deployment.windows_installer.build.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=f"{AI_DEFAULTS_SMOKE_MARKER}\nBUILD_SMOKE = PASS\n",
            stderr=("Default risk thresholds resource missing; using minimal built-in defaults\n"),
        ),
    )

    with pytest.raises(InstallerBuildError, match="packaged executable smoke failed"):
        smoke_executable(executable)


def test_build_smoke_reads_canonical_ai_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[tuple[str, str]] = []

    class Resource:
        def joinpath(self, name: str) -> Resource:
            opened.append((AI_DEFAULTS_PACKAGE, name))
            return self

        def open(self, mode: str, encoding: str):
            return __import__("io").StringIO("market_regime: {}\n")

    monkeypatch.setattr(
        "deployment.windows_installer.service_base.resources.files", lambda package: Resource()
    )
    monkeypatch.setitem(
        sys.modules,
        "yaml",
        SimpleNamespace(safe_load=lambda stream: {"market_regime": {}}),
    )

    qualify_ai_defaults_resource()
    assert opened == [(AI_DEFAULTS_PACKAGE, AI_DEFAULTS_RESOURCE)]


def test_packaged_smoke_failure_preserves_bounded_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "CryptoHunterBackend.exe"
    executable.write_bytes(b"exe")
    stderr = "discarded-prefix:" + ("x" * 4096) + "diagnostic-tail"
    monkeypatch.setattr(
        "deployment.windows_installer.build.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="smoke stdout", stderr=stderr),
    )

    with pytest.raises(InstallerBuildError) as failure:
        smoke_executable(executable)

    message = str(failure.value)
    assert "CryptoHunterBackend.exe (exit=1)" in message
    assert "stdout='smoke stdout'" in message
    assert "diagnostic-tail" in message
    assert "discarded-prefix" not in message


def test_builder_smokes_all_four_packaged_executables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = tmp_path / "payload"
    payload.mkdir()
    work = tmp_path / "work"
    openblas = tmp_path / "numpy.libs/libopenblas64__test.dll"
    openblas.parent.mkdir()
    openblas.write_bytes(b"dll")
    smoked: list[str] = []
    pyinstaller_commands: list[list[str]] = []
    pyinstaller_environments: list[dict[str, str]] = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        pyinstaller_commands.append(command)
        pyinstaller_environments.append(kwargs["env"])  # type: ignore[arg-type]
        name = command[command.index("--name") + 1]
        warning = work / name / name / f"warn-{name}.txt"
        warning.parent.mkdir(parents=True)
        warning.write_text("optional backend missing\n", encoding="utf-8")
        (payload / f"{name}.exe").write_bytes(b"MZ")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("deployment.windows_installer.build.qualify_win32_imports", lambda: None)
    monkeypatch.setattr("deployment.windows_installer.build.numpy_openblas_dll", lambda: openblas)
    monkeypatch.setattr("deployment.windows_installer.build.subprocess.run", run)
    monkeypatch.setattr("deployment.windows_installer.build.qualify_pe_x64", lambda path: None)
    monkeypatch.setattr(
        "deployment.windows_installer.build.smoke_executable",
        lambda path: smoked.append(path.name),
    )
    monkeypatch.setenv("PYTHONHASHSEED", "attacker-controlled")
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1")
    build_executables(
        payload,
        work,
        {
            "python_hash_seed": "0",
            "source_date_epoch": 1790962380,
            "upx": False,
        },
    )
    assert len(pyinstaller_commands) == len(ENTRYPOINTS) == 4
    for command in pyinstaller_commands:
        assert "--noupx" in command
        hidden_imports = [
            command[index + 1]
            for index, argument in enumerate(command)
            if argument == "--hidden-import"
        ]
        assert hidden_imports == [*REQUIRED_WIN32_MODULES, AI_DEFAULTS_PACKAGE]
        collect_data = [
            command[index + 1]
            for index, argument in enumerate(command)
            if argument == "--collect-data"
        ]
        assert collect_data == [AI_DEFAULTS_PACKAGE]
        add_data = [
            command[index + 1] for index, argument in enumerate(command) if argument == "--add-data"
        ]
        if command[command.index("--name") + 1] == "CryptoHunterProvision":
            assert len(add_data) == len(STAGE9_SCHEMA_NAMES)
            assert {Path(value.split(os.pathsep, 1)[0]).name for value in add_data} == set(
                STAGE9_SCHEMA_NAMES
            )
            assert all(value.endswith(f"{os.pathsep}deployment") for value in add_data)
        else:
            assert add_data == []
    assert all(environment["PYTHONHASHSEED"] == "0" for environment in pyinstaller_environments)
    assert all(
        environment["SOURCE_DATE_EPOCH"] == "1790962380" for environment in pyinstaller_environments
    )
    assert smoked == [f"{name}.exe" for name in ENTRYPOINTS]
