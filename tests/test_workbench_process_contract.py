"""Workbench isolation must preserve native failures and required Qt coverage."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tests.ui.test_strategy_workbench_demo import (
    _OPTIONAL_BROWSER_SKIP,
    _assert_native_result,
    _is_missing_browser_executable,
    _qt_diagnostics,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_workbench_native_crash_cannot_be_reported_as_pass_or_optional_skip() -> None:
    code = """
import os
import signal
if os.name == 'posix':
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.kill(os.getpid(), signal.SIGSEGV)
else:
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    for library in ('ucrtbase', 'msvcrt'):
        try:
            runtime = ctypes.CDLL(library)
        except OSError:
            continue
        configure_abort = getattr(runtime, '_set_abort_behavior', None)
        if configure_abort is not None:
            configure_abort(0, 0x3)
            break
    print('WORKBENCH_NATIVE_ABORT_PROBE_READY', flush=True)
    os.abort()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode != 0
    if os.name == "posix":
        assert result.returncode == -11
    else:
        assert "WORKBENCH_NATIVE_ABORT_PROBE_READY" in result.stdout
        assert "Traceback (most recent call last)" not in result.stderr
    for browser in (False, True):
        with pytest.raises(AssertionError, match="native QML process failed"):
            _assert_native_result(result, browser=browser)


def test_workbench_child_assertion_failure_retains_diagnostics() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "raise AssertionError('actual QML assertion failed')"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    with pytest.raises(AssertionError, match="actual QML assertion failed"):
        _assert_native_result(result, browser=False)


def test_workbench_required_case_cannot_be_skipped_by_child() -> None:
    result = subprocess.CompletedProcess(
        ["native-case"],
        77,
        stdout=_OPTIONAL_BROWSER_SKIP + "missing browser\n",
        stderr="",
    )
    with pytest.raises(AssertionError, match="exit code 77"):
        _assert_native_result(result, browser=False)
    with pytest.raises(pytest.skip.Exception, match="missing browser"):
        _assert_native_result(result, browser=True)


def test_workbench_exit_77_without_browser_skip_reason_is_failure() -> None:
    result = subprocess.CompletedProcess(["native-case"], 77, stdout="", stderr="unexpected exit")
    with pytest.raises(AssertionError, match="unexpected exit"):
        _assert_native_result(result, browser=True)


@pytest.mark.parametrize(
    ("message", "missing"),
    [
        ("BrowserType.launch: Executable doesn't exist at /missing/chromium", True),
        ("BrowserType.launch: Target page, context or browser has been closed", False),
        ("BrowserType.launch: Invalid launch arguments", False),
        ("unexpected driver failure", False),
    ],
)
def test_only_absent_browser_executable_is_optional(message: str, missing: bool) -> None:
    assert _is_missing_browser_executable(RuntimeError(message)) is missing


@pytest.fixture
def qt_diagnostic_handler(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    from tests.ui import _qt

    state = SimpleNamespace(handler=None)

    def install_handler(handler):
        previous = state.handler
        state.handler = handler
        return previous

    qt_core = ModuleType("PySide6.QtCore")
    qt_core.QtMsgType = SimpleNamespace(
        QtWarningMsg="warning", QtCriticalMsg="critical", QtFatalMsg="fatal"
    )
    qt_core.qInstallMessageHandler = install_handler
    pyside = ModuleType("PySide6")
    pyside.QtCore = qt_core
    monkeypatch.setitem(sys.modules, "PySide6", pyside)
    monkeypatch.setitem(sys.modules, "PySide6.QtCore", qt_core)
    monkeypatch.setattr(_qt, "require_pyside6", lambda **kwargs: pyside)
    return state


@pytest.mark.parametrize("message_type", ["warning", "critical", "fatal"])
def test_optional_browser_skip_cannot_hide_qt_diagnostic(
    qt_diagnostic_handler: SimpleNamespace, message_type: str
) -> None:
    with pytest.raises(AssertionError, match="actual Qt diagnostic"):
        with _qt_diagnostics():
            qt_diagnostic_handler.handler(message_type, None, "actual Qt diagnostic")
            pytest.skip("optional browser executable absent")
    assert qt_diagnostic_handler.handler is None


def test_optional_browser_skip_without_qt_diagnostics_remains_optional(
    qt_diagnostic_handler: SimpleNamespace,
) -> None:
    with pytest.raises(pytest.skip.Exception, match="optional browser executable absent"):
        with _qt_diagnostics():
            pytest.skip("optional browser executable absent")
    assert qt_diagnostic_handler.handler is None


def test_qt_diagnostics_preserve_a_primary_test_failure(
    qt_diagnostic_handler: SimpleNamespace,
) -> None:
    with pytest.raises(ValueError, match="primary test failure"):
        with _qt_diagnostics():
            qt_diagnostic_handler.handler("warning", None, "additional Qt diagnostic")
            raise ValueError("primary test failure")
    assert qt_diagnostic_handler.handler is None


@pytest.mark.qml
def test_workbench_required_native_case_runs_with_core_parent_and_missing_playwright(
    tmp_path: Path,
) -> None:
    playwright = tmp_path / "playwright"
    playwright.mkdir()
    (playwright / "__init__.py").write_text(
        "raise ModuleNotFoundError('controlled missing Playwright', name='playwright')\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(
        PYTEST_REQUIRE_QML="1", QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software"
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(tmp_path), str(_REPOSITORY_ROOT), environment.get("PYTHONPATH", "")))
    )
    code = """
try:
    import playwright
except ModuleNotFoundError as exc:
    assert exc.name == 'playwright'
else:
    raise AssertionError('Playwright must be unavailable in this regression')
from PySide6.QtCore import QCoreApplication
from tests.ui.test_strategy_workbench_demo import _run_workbench_case
app = QCoreApplication([])
assert type(app) is QCoreApplication
_run_workbench_case('roundtrip')
assert QCoreApplication.instance() is app
"""
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code],
        cwd=_REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=75,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
