"""Pomocnicy do wymuszania dostępności PySide6 w testach UI."""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
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


def require_libgl() -> None:
    """Wymuś obecność biblioteki OpenGL w środowisku testowym.

    Sprawdzamy tylko platformy linuksowe, bo na Windows i macOS Qt zapewnia
    własne biblioteki lub używa innych nazw. Jeśli OpenGL nie jest obecny na
    Linuksie, spróbujemy go przygotować przy użyciu istniejącego helpera z
    ``tests.utils.libgl``. Dopiero jeśli to się nie uda, testy zostaną
    pominięte na poziomie modułu.
    """

    if not sys.platform.startswith("linux"):
        return

    try:
        ctypes.CDLL("libGL.so.1")
        return
    except OSError:
        pass

    try:
        from tests.utils.libgl import ensure_libgl_available
    except ImportError:  # pragma: no cover - brak helpera w nietypowej strukturze testów
        ensure_libgl_available = None  # type: ignore[assignment]

    if ensure_libgl_available is not None:
        try:
            ensure_libgl_available()
            ctypes.CDLL("libGL.so.1")
            return
        except (RuntimeError, OSError) as exc:  # pragma: no cover - zależne od środowiska CI
            failure_reason = str(exc)
    else:  # pragma: no cover - zależne od struktury repo
        failure_reason = "helper ensure_libgl_available niedostępny"

    import pytest

    pytest.skip(
        "Brak biblioteki libGL.so.1 wymaganej przez QtWidgets w trybie testowym"
        + (f" ({failure_reason})" if failure_reason else ""),
        allow_module_level=True,
    )


def teardown_qt_app(app: object | None) -> None:
    """Wykonaj bezpieczny teardown Qt (DeferredDelete + processEvents)."""
    if app is None:
        return

    try:
        from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop
    except Exception:
        return

    try:
        app.processEvents()
        for _ in range(3):
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            app.processEvents(QEventLoop.AllEvents, 50)
    except Exception:
        # Teardown nie powinien wysadzać testów.
        return
