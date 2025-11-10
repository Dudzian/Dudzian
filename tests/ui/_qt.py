"""Pomocnicy do wymuszania dostępności PySide6 w testach UI."""
from __future__ import annotations

import os
from importlib import import_module
from types import ModuleType

_REQUIRE_ENV = {"1", "true", "yes", "on"}


def _qml_required() -> bool:
    return os.getenv("PYTEST_REQUIRE_QML", "").lower() in _REQUIRE_ENV


def require_pyside6() -> ModuleType:
    """Zwróć moduł PySide6 lub przerwij test z komunikatem o brakującej zależności."""
    try:
        module = import_module("PySide6")
    except ModuleNotFoundError as exc:  # pragma: no cover - zależne od środowiska
        if _qml_required():
            raise AssertionError(
                "PySide6 musi być zainstalowany w środowisku testowym UI (job ui-tests)."
            ) from exc
        import pytest

        pytest.skip(
            "Pomijam testy QML: moduł PySide6 nie jest dostępny w bieżącym środowisku.",
            allow_module_level=True,
        )
    return module
