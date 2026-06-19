"""Repo-local PySide6 preflight/bootstrap helper.

The helper is intentionally side-effect free by default. It only installs PySide6
when --install is passed explicitly.
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys

DEFAULT_PYSIDE6_VERSION = "6.10.2"
RECOMMENDED_ENV = {
    "QT_QPA_PLATFORM": "offscreen",
    "QT_OPENGL": "software",
    "DUDZIAN_QML_FLUSH_DELETES": "0",
}


def _target_version() -> str:
    return (
        os.environ.get("PYSIDE6_VERSION", DEFAULT_PYSIDE6_VERSION).strip()
        or DEFAULT_PYSIDE6_VERSION
    )


def _print_recommended_env() -> None:
    print("Zalecane zmienne środowiskowe dla lokalnego/offscreen smoke:")
    for key, value in RECOMMENDED_ENV.items():
        print(f"  {key}={value}")


def _pyside6_version() -> str | None:
    try:
        module = importlib.import_module("PySide6")
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def _install_pyside6(version: str) -> None:
    package = f"PySide6=={version}"
    print(f"Instaluję {package} przez bieżący interpreter: {sys.executable}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sprawdza dostępność PySide6 i opcjonalnie instaluje repozytoryjną wersję testową."
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="jawnie zainstaluj PySide6==${PYSIDE6_VERSION:-6.10.2}, jeśli brakuje importu",
    )
    args = parser.parse_args(argv)

    version = _target_version()
    installed_version = _pyside6_version()
    if installed_version is not None:
        print(f"PySide6 dostępny: {installed_version}")
        _print_recommended_env()
        return 0

    print("PySide6 nie jest dostępny w bieżącym środowisku Pythona.")
    print(f"Docelowa wersja repozytoryjna: PySide6=={version}")
    if not args.install:
        print("Nie wykonano instalacji, bo wymagane jest jawne użycie flagi --install.")
        print(f"Aby doinstalować: {sys.executable} scripts/dev/ensure_pyside6.py --install")
        _print_recommended_env()
        return 1

    _install_pyside6(version)
    installed_version = _pyside6_version()
    if installed_version is None:
        print("Instalacja zakończona, ale import PySide6 nadal nie działa.", file=sys.stderr)
        return 2

    print(f"PySide6 dostępny po instalacji: {installed_version}")
    _print_recommended_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
