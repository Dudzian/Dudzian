"""Optional Qt collection and required QML dependency failures stay distinct."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from xml.etree import ElementTree

import pytest

from tests.ui import _qt

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKBENCH = "tests/ui/test_strategy_workbench_demo.py"


@dataclass(frozen=True)
class _QtTestEnvironment:
    environment: dict[str, str] = field(repr=False)


@pytest.fixture
def missing_egl_environment(tmp_path: Path) -> _QtTestEnvironment:
    """A fresh child sees a Qt package whose native Widgets dependency is absent."""
    dependencies = tmp_path / "dependencies"
    package = dependencies / "PySide6"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "Path(os.environ['QT_IMPORT_TRACE_FILE']).write_text('PySide6 imported')\n",
        encoding="utf-8",
    )
    (package / "QtCore.py").write_text(
        "QObject = QUrl = QMetaObject = Qt = Q_ARG = object\n", encoding="utf-8"
    )
    (package / "QtQml.py").write_text("QQmlApplicationEngine = object\n", encoding="utf-8")
    (package / "QtWidgets.py").write_text(
        "raise ImportError('libEGL.so.1: cannot open shared object file: No such file or directory')\n",
        encoding="utf-8",
    )
    playwright = dependencies / "playwright"
    playwright.mkdir()
    (playwright / "__init__.py").write_text(
        "raise ModuleNotFoundError('controlled missing Playwright', name='playwright')\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=os.pathsep.join([str(dependencies), str(_REPO_ROOT)]),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_REQUIRE_QML="0",
        QT_IMPORT_TRACE_FILE=str(tmp_path / "qt-imported"),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    return _QtTestEnvironment(environment)


@pytest.mark.parametrize("ordinary_shard", [False, True])
def test_workbench_collects_without_loading_native_qt_or_playwright(
    tmp_path: Path, missing_egl_environment: _QtTestEnvironment, ordinary_shard: bool
) -> None:
    probe = tmp_path / "test_ordinary_probe.py"
    probe.write_text("def test_ordinary_probe():\n    assert True\n", encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--noconftest",
        "-c",
        str(_REPO_ROOT / "pyproject.toml"),
        "--collect-only",
        "-q",
        _WORKBENCH,
        str(probe),
    ]
    if ordinary_shard:
        command.extend(["-m", "not qml"])

    result = subprocess.run(
        command,
        cwd=_REPO_ROOT,
        env=missing_egl_environment.environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert any(line.endswith("::test_ordinary_probe") for line in result.stdout.splitlines())
    assert not Path(missing_egl_environment.environment["QT_IMPORT_TRACE_FILE"]).exists()
    if ordinary_shard:
        assert "2 deselected" in result.stdout
    else:
        assert f"{_WORKBENCH}::test_workbench_demo_mode_roundtrip" in result.stdout
        assert f"{_WORKBENCH}::test_workbench_demo_title_in_browser" in result.stdout
        assert "3 tests collected" in result.stdout


def test_required_workbench_fails_instead_of_skipping_missing_egl(
    tmp_path: Path, missing_egl_environment: _QtTestEnvironment
) -> None:
    environment = dict(missing_egl_environment.environment, PYTEST_REQUIRE_QML="1")
    report = tmp_path / "required-qml.xml"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--noconftest",
            "-c",
            str(_REPO_ROOT / "pyproject.toml"),
            "-q",
            f"{_WORKBENCH}::test_workbench_demo_mode_roundtrip",
            f"--junitxml={report}",
        ],
        cwd=_REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "libEGL.so.1" in result.stdout + result.stderr
    assert Path(environment["QT_IMPORT_TRACE_FILE"]).exists()
    cases = ElementTree.parse(report).findall(".//testcase")
    assert len(cases) == 1
    assert cases[0].find("failure") is not None or cases[0].find("error") is not None
    assert cases[0].find("skipped") is None


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("dependency", ["PySide6", "PySide6.QtWidgets", "shiboken6"])
def test_missing_qt_python_dependency_is_optional_only_outside_qml_job(
    monkeypatch: pytest.MonkeyPatch, required: bool, dependency: str
) -> None:
    monkeypatch.setenv("PYTEST_REQUIRE_QML", "1" if required else "0")
    failure = ModuleNotFoundError(f"No module named '{dependency}'", name=dependency)

    def unavailable(name: str) -> ModuleType:
        raise failure

    monkeypatch.setattr(_qt, "import_module", unavailable)

    expected = AssertionError if required else pytest.skip.Exception
    with pytest.raises(expected, match="PySide6/Qt"):
        _qt.require_pyside6(modules=("QtWidgets",))


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("library", ["libEGL.so.1", "libGL.so.1"])
def test_missing_qt_native_dependency_is_optional_only_outside_qml_job(
    monkeypatch: pytest.MonkeyPatch, required: bool, library: str
) -> None:
    monkeypatch.setenv("PYTEST_REQUIRE_QML", "1" if required else "0")
    monkeypatch.setattr(_qt, "sys", SimpleNamespace(platform="linux"))
    failure = ImportError(f"{library}: cannot open shared object file: No such file or directory")

    def unavailable(name: str) -> ModuleType:
        if name == "PySide6.QtWidgets":
            raise failure
        return ModuleType(name)

    monkeypatch.setattr(_qt, "import_module", unavailable)

    expected = AssertionError if required else pytest.skip.Exception
    with pytest.raises(expected, match=library):
        _qt.require_pyside6(modules=("QtCore", "QtQml", "QtWidgets"))


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize(
    "failure",
    [
        ImportError("libQt6Widgets.so.6: undefined symbol: broken_abi"),
        ImportError("cannot import name 'QApplication' from 'PySide6.QtWidgets'"),
        ModuleNotFoundError("No module named 'application_bug'", name="application_bug"),
        RuntimeError("Qt initialization failed"),
    ],
)
def test_qt_import_errors_that_are_not_missing_dependencies_propagate(
    monkeypatch: pytest.MonkeyPatch, required: bool, failure: Exception
) -> None:
    monkeypatch.setenv("PYTEST_REQUIRE_QML", "1" if required else "0")

    def broken(name: str) -> ModuleType:
        raise failure

    monkeypatch.setattr(_qt, "import_module", broken)

    with pytest.raises(type(failure)) as raised:
        _qt.require_pyside6(modules=("QtWidgets",))
    assert raised.value is failure


def test_required_qml_imports_all_requested_qt_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTEST_REQUIRE_QML", "1")
    modules = {
        name: ModuleType(name)
        for name in ("PySide6", "PySide6.QtCore", "PySide6.QtQml", "PySide6.QtWidgets")
    }
    imported: list[str] = []

    def available(name: str) -> ModuleType:
        imported.append(name)
        return modules[name]

    monkeypatch.setattr(_qt, "import_module", available)

    assert _qt.require_pyside6(modules=("QtCore", "QtQml", "QtWidgets")) is modules["PySide6"]
    assert imported == list(modules)


def test_required_qml_does_not_skip_missing_libgl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTEST_REQUIRE_QML", "1")
    monkeypatch.setattr(_qt, "sys", SimpleNamespace(platform="linux"))
    failure = OSError("libGL.so.1: cannot open shared object file: No such file or directory")

    def unavailable(name: str) -> None:
        raise failure

    monkeypatch.setattr(_qt.ctypes, "CDLL", unavailable)

    with pytest.raises(AssertionError, match="libGL.so.1") as raised:
        _qt.require_libgl()
    assert raised.value.__cause__ is failure
