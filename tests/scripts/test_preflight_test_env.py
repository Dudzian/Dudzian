from __future__ import annotations

from scripts.ci.preflight_test_env import REQUIRED_MODULES, WINDOWS_REQUIRED_MODULES, _required_modules


def test_windows_preflight_requires_complete_pywin32_surface() -> None:
    required = _required_modules("win32")

    assert required.items() >= REQUIRED_MODULES.items()
    assert {"ntsecuritycon", "win32security", "win32serviceutil"} <= required.keys()
    assert required.items() >= WINDOWS_REQUIRED_MODULES.items()


def test_non_windows_preflight_does_not_require_pywin32() -> None:
    assert _required_modules("linux") == REQUIRED_MODULES
    assert _required_modules("darwin") == REQUIRED_MODULES
