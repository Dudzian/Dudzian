from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.ci import preflight_test_env
from scripts.ci.preflight_test_env import (
    REQUIRED_MODULES,
    WINDOWS_REQUIRED_MODULES,
    _required_modules,
)


def test_windows_preflight_requires_complete_pywin32_surface() -> None:
    required = _required_modules("win32")

    assert required.items() >= REQUIRED_MODULES.items()
    assert {"ntsecuritycon", "win32security", "win32serviceutil"} <= required.keys()
    assert required.items() >= WINDOWS_REQUIRED_MODULES.items()


def test_non_windows_preflight_does_not_require_pywin32() -> None:
    assert _required_modules("linux") == REQUIRED_MODULES
    assert _required_modules("darwin") == REQUIRED_MODULES


@pytest.mark.parametrize(
    ("missing_module", "require_fastapi", "exit_code"),
    [("fastapi", True, 1), ("httpx", False, 1), ("fastapi", False, 0)],
)
def test_preflight_checks_required_endpoint_dependencies(
    monkeypatch, capsys, missing_module, require_fastapi, exit_code
) -> None:
    def import_module(name):
        if name == missing_module:
            raise RuntimeError(name)
        return SimpleNamespace(__version__="test")

    monkeypatch.setattr(preflight_test_env, "_import_module", import_module)
    monkeypatch.setattr(
        preflight_test_env.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="", stderr="", returncode=0),
    )

    assert preflight_test_env.main(["--require-fastapi"] if require_fastapi else []) == exit_code
    output = capsys.readouterr().out
    if exit_code:
        assert f"- {missing_module}" in output
    else:
        assert "Preflight OK:" in output
