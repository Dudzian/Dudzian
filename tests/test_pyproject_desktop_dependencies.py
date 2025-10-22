"""Assercje pilnujące spójności zależności w `pyproject.toml`."""
from __future__ import annotations

from pathlib import Path

import tomllib


PYPROJECT_PATH = Path("pyproject.toml")


def _load_pyproject() -> dict:
    return tomllib.loads(PYPROJECT_PATH.read_text())


def _find_dependency(dependencies: list[str], package: str) -> str | None:
    package_lower = package.lower()
    for entry in dependencies:
        normalized = entry.split(";", 1)[0].strip().lower()
        if normalized.startswith(package_lower):
            return entry
    return None


def test_numeric_stack_is_declared_in_core_dependencies() -> None:
    project = _load_pyproject()["project"]
    deps = project.get("dependencies", [])

    for package in ("numpy>=1.26", "pandas>=2.2", "joblib>=1.3"):
        found = _find_dependency(deps, package.split(">=", 1)[0])
        assert found is not None, f"Brak pakietu {package} w zależnościach podstawowych"
        assert found.startswith(package), (
            "Zależność ma inną wersję niż oczekiwana: "
            f"{package} (znaleziono: {found})"
        )


def test_dev_extra_keeps_numeric_stack() -> None:
    project = _load_pyproject()["project"]
    extras = project.get("optional-dependencies", {})
    dev = extras.get("dev", [])

    for package in ("numpy>=1.26", "pandas>=2.2", "joblib>=1.3"):
        found = _find_dependency(dev, package.split(">=", 1)[0])
        assert found is not None, f"Brak pakietu {package} w extras.dev"
        assert found.startswith(package), (
            "Pakiet ma inną wersję niż oczekiwana: "
            f"{package} (znaleziono: {found})"
        )


def test_desktop_extra_contains_bundling_tools() -> None:
    project = _load_pyproject()["project"]
    extras = project.get("optional-dependencies", {})
    desktop = extras.get("desktop", [])

    for package in ("pyinstaller>=6.5", "briefcase>=0.3.18"):
        found = _find_dependency(desktop, package.split(">=", 1)[0])
        assert found is not None, f"Brak pakietu {package} w extras.desktop"
        assert found.startswith(package), (
            "Pakiet ma inną wersję niż oczekiwana: "
            f"{package} (znaleziono: {found})"
        )
