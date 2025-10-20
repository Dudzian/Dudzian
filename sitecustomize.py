"""Umożliwia import lokalnych modułów bez instalacji pakietu."""
from __future__ import annotations

from pathlib import Path

import importlib

from pathbootstrap import ensure_repo_root_on_sys_path


def _stabilize_numpy_no_value() -> None:
    """Zapewnij spójność sentinela NumPy `_NoValue` po ewentualnych przeładowaniach."""

    try:
        import numpy as _np
        from numpy.core import _multiarray_umath as _umath  # type: ignore
        from numpy.core import _methods as _methods  # type: ignore
    except Exception:
        return

    sentinel = getattr(_umath, "_NoValue", None)
    if sentinel is None:
        return
    try:
        _np._NoValue = sentinel  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        if getattr(_methods, "_NoValue", None) is not sentinel:
            _methods._NoValue = sentinel  # type: ignore[attr-defined]
    except Exception:
        pass


ensure_repo_root_on_sys_path(Path(__file__).resolve().parent)
_stabilize_numpy_no_value()


_original_reload = importlib.reload


def _patched_reload(module):  # type: ignore[override]
    result = _original_reload(module)
    if getattr(module, "__name__", "") == "numpy" or getattr(module, "__name__", "").startswith("numpy."):
        _stabilize_numpy_no_value()
    return result


importlib.reload = _patched_reload
