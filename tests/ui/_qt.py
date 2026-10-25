"""Pomocnicy do wymuszania dostępności PySide6 w testach UI."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, NoReturn

_REQUIRE_ENV = {"1", "true", "yes", "on"}


def prepare_windows_offscreen_fonts() -> Path | None:
    """Provide the Windows offscreen FreeType backend with an existing font directory."""

    if (
        sys.platform != "win32"
        or os.environ.get("QT_QPA_PLATFORM", "").split(":", 1)[0] != "offscreen"
    ):
        return None
    configured = os.environ.get("QT_QPA_FONTDIR")
    if configured is None:
        windows_root = os.environ.get("WINDIR") or os.environ.get("SystemRoot")
        if not windows_root:
            raise AssertionError("Windows offscreen Qt requires WINDIR or SystemRoot for fonts")
        fonts = Path(windows_root) / "Fonts"
    else:
        if not configured.strip():
            raise AssertionError("Windows offscreen Qt has an empty explicit QT_QPA_FONTDIR")
        fonts = Path(configured)
    # Qt 6.7.2 QFreeTypeFontDatabase scans precisely these filename extensions.
    # Actual font parsing remains Qt's responsibility after QApplication starts.
    if not fonts.is_dir() or not any(
        item.is_file() and item.suffix.lower() in {".ttf", ".pfa", ".pfb", ".otf"}
        for item in fonts.iterdir()
    ):
        raise AssertionError(
            f"Windows offscreen Qt requires an existing directory with supported font files: {fonts}"
        )
    if configured is None:
        os.environ["QT_QPA_FONTDIR"] = str(fonts)
    return fonts


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
    if not sys.platform.startswith("linux"):
        return False
    library, separator, detail = str(exc).partition(": ")
    if not separator or detail != "cannot open shared object file: No such file or directory":
        return False
    if not library.startswith("lib") or any(character in library for character in ":/\n"):
        return False
    if library.endswith(".so"):
        return len(library) > len("lib.so")
    # Leave a version character after the rightmost separator; scan it once.
    version_separator = library.rfind(".so.", 0, len(library) - 1)
    return version_separator > len("lib") and all(
        character.isalnum() or character in "_.-"
        for character in library[version_separator + len(".so.") :]
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
