from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only for Python < 3.11 in local tooling
    tomllib = None


def _read_test_extra_entries(pyproject_text: str) -> list[str]:
    if tomllib is not None:
        pyproject = tomllib.loads(pyproject_text)
        return list(pyproject["project"]["optional-dependencies"]["test"])

    entries: list[str] = []
    in_test_block = False
    for raw_line in pyproject_text.splitlines():
        line = raw_line.strip()
        if not in_test_block and line == "test = [":
            in_test_block = True
            continue
        if in_test_block:
            if line == "]":
                break
            if line.startswith('"') and line.endswith('",'):
                entries.append(line[1:-2])
    return entries


def test_windows_qml_runner_isolation_contract_matches_test_extras() -> None:
    script = Path("scripts/run_qml_tests_windows.ps1").read_text(encoding="utf-8")
    requires_boxed = "import xdist" in script and "--boxed" in script
    supports_forked_fallback = "import pytest_forked" in script and "--forked" in script

    assert requires_boxed or supports_forked_fallback, (
        "Windows QML runner must require at least one pytest isolation mechanism "
        "(xdist/--boxed or pytest-forked/--forked)."
    )
    assert "pytest isolation unavailable" in script

    test_extra = _read_test_extra_entries(Path("pyproject.toml").read_text(encoding="utf-8"))
    provides_xdist = any(dep.startswith("pytest-xdist") for dep in test_extra)
    provides_forked = any(dep.startswith("pytest-forked") for dep in test_extra)

    if requires_boxed:
        assert provides_xdist, (
            "project.optional-dependencies.test must include pytest-xdist for "
            "scripts/run_qml_tests_windows.ps1 isolation preflight"
        )
    else:
        assert supports_forked_fallback and provides_forked, (
            "project.optional-dependencies.test must include pytest-forked when "
            "Windows QML runner relies on --forked isolation"
        )
