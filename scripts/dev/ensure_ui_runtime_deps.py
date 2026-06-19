"""Repo-local preflight for imports required by PySide UI runtime smoke.

The helper only installs missing dependencies when --install is passed explicitly.
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from dataclasses import dataclass

DEFAULT_PYSIDE6_VERSION = "6.10.2"
RECOMMENDED_ENV = {
    "QT_QPA_PLATFORM": "offscreen",
    "QT_OPENGL": "software",
    "DUDZIAN_QML_FLUSH_DELETES": "0",
}


@dataclass(frozen=True)
class RuntimeDependency:
    module: str
    requirement: str
    reason: str


def _pyside6_requirement() -> str:
    version = os.environ.get("PYSIDE6_VERSION", DEFAULT_PYSIDE6_VERSION).strip()
    return f"PySide6=={version or DEFAULT_PYSIDE6_VERSION}"


def _runtime_dependencies() -> tuple[RuntimeDependency, ...]:
    return (
        RuntimeDependency(
            module="PySide6",
            requirement=_pyside6_requirement(),
            reason="Qt/PySide runtime used by tests/ui_pyside/test_source_smoke.py",
        ),
        RuntimeDependency(
            module="cryptography",
            requirement="cryptography>=48.0.1,<49",
            reason="imported by marketplace/security modules reached from ui.pyside_app bootstrap",
        ),
        RuntimeDependency(
            module="yaml",
            requirement="PyYAML>=6.0",
            reason="imported as yaml by UI config/runbook/controllers during smoke bootstrap",
        ),
        RuntimeDependency(
            module="nacl",
            requirement="PyNaCl>=1.5.0,<2",
            reason="imported by bot_core security/marketplace paths reached from UI bootstrap",
        ),
        RuntimeDependency(
            module="numpy",
            requirement="numpy>=1.26",
            reason="imported by bot_core runtime/trading paths reached from UI smoke bootstrap",
        ),
        RuntimeDependency(
            module="pandas",
            requirement="pandas>=2.2",
            reason="imported by bot_core runtime/trading paths reached from UI smoke bootstrap",
        ),
        RuntimeDependency(
            module="pydantic",
            requirement="pydantic>=2.5",
            reason="imported by bot_core marketplace/config/runtime models reached from UI smoke bootstrap",
        ),
        RuntimeDependency(
            module="jsonschema",
            requirement="jsonschema>=4.21",
            reason="declared core dependency imported by bot_core validation paths reached from UI smoke bootstrap",
        ),
        RuntimeDependency(
            module="grpc",
            requirement="grpcio>=1.62,<2",
            reason="imported by UI/backend gRPC bridge paths reached from UI smoke bootstrap",
        ),
    )


def _print_recommended_env() -> None:
    print("Zalecane zmienne środowiskowe dla lokalnego/offscreen smoke:")
    for key, value in RECOMMENDED_ENV.items():
        print(f"  {key}={value}")


def _missing_dependencies(dependencies: tuple[RuntimeDependency, ...]) -> list[RuntimeDependency]:
    missing: list[RuntimeDependency] = []
    for dependency in dependencies:
        try:
            importlib.import_module(dependency.module)
        except ImportError:
            missing.append(dependency)
    return missing


def _print_dependency_status(
    dependencies: tuple[RuntimeDependency, ...], missing: list[RuntimeDependency]
) -> None:
    missing_modules = {dependency.module for dependency in missing}
    for dependency in dependencies:
        status = "MISSING" if dependency.module in missing_modules else "OK"
        print(f"[{status}] import {dependency.module} -> pip: {dependency.requirement}")
        print(f"       powód: {dependency.reason}")


def _install_missing(missing: list[RuntimeDependency]) -> None:
    requirements = [dependency.requirement for dependency in missing]
    print("Instaluję brakujące dependency UI runtime smoke:")
    for requirement in requirements:
        print(f"  {requirement}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *requirements])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sprawdza minimalne importy wymagane przez lokalny PySide UI runtime smoke."
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="jawnie zainstaluj tylko brakujące dependency wymagane przez UI runtime smoke",
    )
    args = parser.parse_args(argv)

    dependencies = _runtime_dependencies()
    missing = _missing_dependencies(dependencies)
    _print_dependency_status(dependencies, missing)
    _print_recommended_env()

    if not missing:
        print("UI runtime dependency preflight: komplet importów dostępny.")
        return 0

    print("Brakujące dependency UI runtime smoke:")
    for dependency in missing:
        print(f"  - import {dependency.module}: zainstaluj {dependency.requirement}")
    print("Oficjalny repozytoryjny setup dev: python -m pip install '.[dev]'")

    if not args.install:
        print("Nie wykonano instalacji, bo wymagane jest jawne użycie flagi --install.")
        print(
            f"Aby doinstalować tylko braki: {sys.executable} scripts/dev/ensure_ui_runtime_deps.py --install"
        )
        return 1

    _install_missing(missing)
    remaining = _missing_dependencies(dependencies)
    if remaining:
        print("Po instalacji nadal brakuje dependency:", file=sys.stderr)
        for dependency in remaining:
            print(f"  - import {dependency.module}: {dependency.requirement}", file=sys.stderr)
        return 2

    print("UI runtime dependency preflight: komplet importów dostępny po instalacji.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
