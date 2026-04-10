from __future__ import annotations

from pathlib import Path


def test_windows_qml_runner_isolation_contract() -> None:
    script = Path("scripts/run_qml_tests_windows.ps1").read_text(encoding="utf-8")
    assert "--boxed" not in script, "Windows QML runner must not pass unsupported --boxed flag."
    assert "--forked" not in script, "Windows QML runner must not force pytest-forked isolation."
    assert "import pytest_forked" not in script, (
        "Windows QML runner must not require pytest-forked."
    )
