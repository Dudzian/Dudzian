"""Semantic contracts for the exact desktop release dependency authority."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from scripts.ci.bootstrap_dependency import project_requirements
from scripts.ci.require_python311 import require_python311
from scripts.ci.validate_locked_resolution import locked_versions, validate_target_resolution

LOCK = Path("deploy/packaging/requirements-desktop.lock")
LOCK_REPOSITORY_TEXT = LOCK.as_posix()
LOCK_NATIVE = os.fspath(LOCK)
PYSIDE_STACK = {"pyside6", "pyside6-addons", "pyside6-essentials", "shiboken6"}
MACOS_BRIEFCASE_CLOSURE = {
    "dmgbuild": "1.6.7",
    "ds-store": "1.3.3",
    "mac-alias": "2.2.3",
}
PIP_BOOTSTRAP = f"python scripts/ci/bootstrap_locked_pip.py {LOCK_REPOSITORY_TEXT}"


def _text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_release_workflows_install_and_audit_canonical_lock() -> None:
    windows = _text(".github/workflows/windows-build.yml")
    packaging = _text(".github/workflows/main.yml")
    cross_installer = _text("deploy/ci/github_actions_cross_installer.yml")
    audit = _text(".github/workflows/ci.yml")
    assert f"pip install --no-deps -r {LOCK_REPOSITORY_TEXT}" in windows
    assert packaging.count(f"pip install --no-deps -r {LOCK_REPOSITORY_TEXT}") == 3
    assert cross_installer.count(f"pip install --no-deps -r {LOCK_REPOSITORY_TEXT}") == 2
    assert packaging.count(f"validate_locked_resolution.py {LOCK_REPOSITORY_TEXT}") == 3
    assert cross_installer.count(f"validate_locked_resolution.py {LOCK_REPOSITORY_TEXT}") == 2
    assert f"--requirement {LOCK_REPOSITORY_TEXT}" in audit


def test_mypy_uses_fixed_vendor_pyside_stubs_and_checks_project_sources() -> None:
    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["tool"]["mypy"]
    assert config["follow_imports"] == "skip"
    assert "mypy_path" not in config
    assert "ui" in config["files"]
    assert not Path("typings/PySide6").exists()
    assert locked_versions(LOCK)["pyside6-essentials"] == "6.7.2"


def test_vendor_qtcore_typing_still_rejects_invalid_api_use(tmp_path: Path) -> None:
    pytest.importorskip("PySide6.QtCore")
    pytest.importorskip("mypy")
    config = tmp_path / "mypy.ini"
    config.write_text("[mypy]\nfollow_imports = normal\n", encoding="utf-8")
    fixture = tmp_path / "invalid_qtcore.py"
    fixture.write_text(
        'from PySide6.QtCore import QTimer\nQTimer().setInterval("not-an-int")\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--config-file", str(config), str(fixture)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert 'incompatible type "str"; expected "int"' in result.stdout


def test_release_build_environments_have_no_loose_python_resolver_mutations() -> None:
    windows = _text(".github/workflows/windows-build.yml")
    packaging = _text(".github/workflows/main.yml")
    cross_installer = _text("deploy/ci/github_actions_cross_installer.yml")
    ci = _text(".github/workflows/ci.yml")
    wheelhouse_job = ci[ci.index("  prepare-wheelhouse:") : ci.index("  sbom:")]

    assert windows.count(PIP_BOOTSTRAP) == 1
    assert packaging.count(PIP_BOOTSTRAP) == 3
    assert cross_installer.count(PIP_BOOTSTRAP) == 2
    assert wheelhouse_job.count(PIP_BOOTSTRAP) == 1
    bootstrap = wheelhouse_job.index("- name: Bootstrap canonical pip")
    validation = wheelhouse_job.index("- name: Validate exact lock resolution")
    build = wheelhouse_job.index("- name: Build wheelhouse")
    assert bootstrap < validation < build
    for release_path in (windows, packaging, cross_installer, wheelhouse_job):
        assert "pip install --upgrade pip" not in release_path
        assert "pip install -U pip" not in release_path

    loose_tools = "python -m pip install pytest mypy ruff openpyxl"
    exact_tools = (
        "python -m pip install --no-deps -r deploy/packaging/requirements-windows-build-tools.lock"
    )
    assert loose_tools not in windows
    assert exact_tools in windows


def test_cross_installer_binds_release_install_to_bootstrapped_python() -> None:
    cross_installer = _text("deploy/ci/github_actions_cross_installer.yml")
    bound_install = f"python -m pip install --no-deps -r {LOCK_REPOSITORY_TEXT}"
    assert cross_installer.count(bound_install) == 2
    assert not re.search(
        rf"(?m)^\s+pip install --no-deps -r {re.escape(LOCK_REPOSITORY_TEXT)}$", cross_installer
    )


def test_windows_build_tooling_authority_is_exact_and_separate_from_runtime() -> None:
    tooling_lock = Path("deploy/packaging/requirements-windows-build-tools.lock")
    tooling = locked_versions(tooling_lock)
    assert tooling == {
        "iniconfig": "2.3.0",
        "pluggy": "1.6.0",
        "pytest": "9.0.2",
    }
    runtime = locked_versions(LOCK)
    assert "pytest" not in runtime


def test_windows_colorama_dependency_is_owned_by_canonical_desktop_authority() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    desktop = [Requirement(raw) for raw in project["optional-dependencies"]["desktop"]]
    colorama = next(requirement for requirement in desktop if requirement.name == "colorama")
    assert colorama.marker is not None
    assert colorama.marker.evaluate({"platform_system": "Windows"})
    assert not colorama.marker.evaluate({"platform_system": "Linux"})
    assert colorama.specifier.contains("0.4.6")
    assert locked_versions(LOCK)["colorama"] == "0.4.6"
    assert "colorama" not in locked_versions(
        Path("deploy/packaging/requirements-windows-runtime.lock")
    )
    assert "colorama" not in locked_versions(
        Path("deploy/packaging/requirements-windows-build-tools.lock")
    )


def test_macos_briefcase_closure_is_owned_by_canonical_desktop_authority() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    desktop = {
        canonicalize_name(requirement.name): requirement
        for raw in project["optional-dependencies"]["desktop"]
        if canonicalize_name((requirement := Requirement(raw)).name) in MACOS_BRIEFCASE_CLOSURE
    }
    assert set(desktop) == set(MACOS_BRIEFCASE_CLOSURE)
    for name, version in MACOS_BRIEFCASE_CLOSURE.items():
        requirement = desktop[name]
        assert requirement.marker is not None
        assert requirement.marker.evaluate({"platform_system": "Darwin"})
        assert not requirement.marker.evaluate({"platform_system": "Windows"})
        assert not requirement.marker.evaluate({"platform_system": "Linux"})
        assert requirement.specifier.contains(version)
    assert {name: locked_versions(LOCK)[name] for name in MACOS_BRIEFCASE_CLOSURE} == (
        MACOS_BRIEFCASE_CLOSURE
    )


def test_canonical_pyside_stack_has_one_version_everywhere() -> None:
    locked = locked_versions(LOCK)
    versions = {locked[name] for name in PYSIDE_STACK}
    direct = locked_versions(Path("deploy/packaging/requirements-desktop.txt"))
    assert versions == {direct["pyside6"]}
    workflow = _text(".github/workflows/ci.yml")
    configured = re.search(r'^  PYSIDE6_VERSION: "([^"]+)"$', workflow, re.MULTILINE)
    assert configured and configured.group(1) in versions
    builder = _text("scripts/ci/build_wheelhouse.py")
    assert "--pyside6-version" not in builder
    assert "pyside_packages" not in builder
    assert "PYSIDE6_VERSION" not in _text(".github/workflows/main.yml")


def test_generator_rejects_every_non_python311_runtime() -> None:
    require_python311((3, 11, 9))
    for version in ((3, 10, 14), (3, 12, 0), (3, 14, 0)):
        with pytest.raises(SystemExit, match="requires Python 3.11"):
            require_python311(version)


def test_lock_generation_authority_is_self_consistent() -> None:
    header = LOCK.read_text(encoding="utf-8").splitlines()[:2]
    generator = _text("scripts/compile_desktop_lock.sh")
    assert header == [
        "# Generated by scripts/compile_desktop_lock.sh.",
        "# Python 3.11 is required; that script is the canonical generation authority.",
    ]
    assert "scripts/ci/require_python311.py" in generator
    assert "--no-header" in generator
    for package, version in {
        "click": "8.3.3",
        "GitPython": "3.1.60",
        "Pygments": "2.20.0",
        "idna": "3.15",
        "requests": "2.33.0",
        "urllib3": "2.8.0",
    }.items():
        assert f"--upgrade-package {package}=={version}" in generator


def test_cross_platform_jobs_validate_native_plan_before_wheelhouse() -> None:
    workflow = _text(".github/workflows/ci.yml")
    assert "os: [ubuntu-latest, windows-latest, macos-latest]" in workflow
    validation = f"validate_locked_resolution.py {LOCK_REPOSITORY_TEXT}"
    assert validation in workflow
    assert '--target ".[desktop]"' in workflow
    assert workflow.index(validation) < workflow.index("Build wheelhouse")
    assert f"--requirements {LOCK_REPOSITORY_TEXT}" in workflow
    builder = _text("scripts/ci/build_wheelhouse.py")
    assert "validate_desktop_resolution(args.python, args.requirements)" in builder
    assert builder.count(".[desktop]") == 1  # validator CLI compatibility only
    for local_target in (".[tools]", ".[dev]", ".[test,codegen]"):
        assert local_target not in builder


@pytest.mark.parametrize(
    ("name", "version", "expected"),
    [
        ("colorama", "0.4.6", "colorama==0.4.6"),
        ("dmgbuild", "1.6.7", "dmgbuild==1.6.7"),
        ("ds-store", "1.3.3", "ds-store==1.3.3"),
        ("mac-alias", "2.2.3", "mac-alias==2.2.3"),
    ],
)
def test_actual_desktop_target_rejects_missing_or_mismatched_dependency(
    name: str, version: str, expected: str
) -> None:
    def fake_runner(command: list[str], *, check: bool) -> object:
        assert check is True
        report = Path(command[command.index("--report") + 1])
        if "--requirement" in command:
            assert command[command.index("--requirement") + 1] == LOCK_NATIVE
            assert "--no-deps" in command
            install = []
        else:
            assert command[command.index("--constraint") + 1] == LOCK_NATIVE
            assert ".[desktop]" not in command
            assert "pyarrow>=21.0.0" in command
            assert "pyinstaller>=6.5" in command
            install = [
                {"requested": True, "metadata": {"name": name, "version": version}},
            ]
        report.write_text(json.dumps({"install": install}))
        return object()

    with pytest.raises(SystemExit, match=expected):
        validate_target_resolution(LOCK, ".[desktop]", runner=fake_runner)


def test_lock_path_representations_keep_repository_text_separate_from_native_paths() -> None:
    assert LOCK_REPOSITORY_TEXT == "deploy/packaging/requirements-desktop.lock"
    assert "\\" not in LOCK_REPOSITORY_TEXT
    assert LOCK_NATIVE == str(LOCK) == os.fspath(LOCK)
    if os.name == "nt":
        assert LOCK_NATIVE == r"deploy\packaging\requirements-desktop.lock"
    else:
        assert LOCK_NATIVE == LOCK_REPOSITORY_TEXT


def test_every_desktop_direct_dependency_is_in_canonical_lock() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    desktop = project["optional-dependencies"]["desktop"]
    locked = locked_versions(LOCK)
    missing = [
        requirement.name
        for raw in desktop
        if canonicalize_name((requirement := Requirement(raw)).name) not in locked
    ]
    assert not missing


def test_every_build_system_requirement_is_satisfied_by_canonical_lock() -> None:
    build_requirements = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))[
        "build-system"
    ]["requires"]
    locked = locked_versions(LOCK)
    for raw in build_requirements:
        requirement = Requirement(raw)
        name = canonicalize_name(requirement.name)
        assert name in locked
        assert requirement.specifier.contains(locked[name], prereleases=True)


def test_validator_reads_static_base_and_desktop_dependencies() -> None:
    requirements = project_requirements(Path("pyproject.toml"), ("desktop",))
    assert "pyarrow>=21.0.0" in requirements
    assert "pyinstaller>=6.5" in requirements


def test_every_lock_entry_is_an_exact_pin() -> None:
    for raw in LOCK.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        assert canonicalize_name(requirement.name) in locked_versions(LOCK)
