"""Pomocnicy do wymuszania dostępności PySide6 w testach UI."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
import sys
from importlib import import_module
from types import ModuleType
from typing import Any, NoReturn

_REQUIRE_ENV = {"1", "true", "yes", "on"}


def _qml_required() -> bool:
    return os.getenv("PYTEST_REQUIRE_QML", "").lower() in _REQUIRE_ENV


def _missing_qt_dependency(exc: ImportError) -> bool:
    if isinstance(exc, ModuleNotFoundError):
        return bool(
            exc.name
            and (
                exc.name == "PySide6"
                or exc.name.startswith("PySide6.")
                or exc.name == "shiboken6"
                or exc.name.startswith("shiboken6.")
            )
        )
    # Missing ELF dependencies are optional only outside the required QML job.
    # ABI/symbol errors and unrelated import failures must remain visible.
    return bool(
        sys.platform.startswith("linux")
        and re.fullmatch(
            r"lib[^:/\n]+\.so(?:\.[\w.-]+)*: cannot open shared object file: No such file or directory",
            str(exc),
        )
    )


def _qt_unavailable(message: str, exc: BaseException) -> NoReturn:
    if _qml_required():
        raise AssertionError(message) from exc
    import pytest

    pytest.skip(message, allow_module_level=True)


def require_pyside6(*, modules: tuple[str, ...] = ()) -> ModuleType:
    """Sprawdź PySide6 i żądane moduły; wymagany job QML nie może ich pominąć."""
    try:
        module = import_module("PySide6")
        for name in modules:
            import_module(f"PySide6.{name}")
    except ImportError as exc:
        if not _missing_qt_dependency(exc):
            raise
        _qt_unavailable(
            f"Brak wymaganej zależności PySide6/Qt w środowisku testowym UI: {exc}", exc
        )
    return module


def require_libgl() -> None:
    """Wymuś obecność biblioteki OpenGL w środowisku testowym.

    Sprawdzamy tylko platformy linuksowe, bo na Windows i macOS Qt zapewnia
    własne biblioteki lub używa innych nazw. Wymagany job QML musi dostarczyć
    bibliotekę; jej brak jest błędem. Poza nim próbujemy istniejącego helpera
    ``tests.utils.libgl`` i pomijamy opcjonalny moduł, jeżeli nadal jej brakuje.
    """

    if not sys.platform.startswith("linux"):
        return

    try:
        ctypes.CDLL("libGL.so.1")
        return
    except OSError as exc:
        if _qml_required():
            raise AssertionError("Brak biblioteki libGL.so.1 wymaganej przez job QML") from exc

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


def qml_value_to_python(value: Any) -> Any:
    """Zamień QML/JS value na natywny typ Pythona (rekurencyjnie)."""
    try:
        from PySide6.QtQml import QJSValue
    except Exception:
        QJSValue = None  # type: ignore[assignment]

    if QJSValue is not None and isinstance(value, QJSValue):
        try:
            value = value.toVariant()
        except Exception:
            return value

    if isinstance(value, list):
        return [qml_value_to_python(item) for item in value]
    if isinstance(value, dict):
        return {key: qml_value_to_python(item) for key, item in value.items()}
    return value
