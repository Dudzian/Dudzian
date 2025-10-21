"""Zestaw testów regresyjnych utrzymujących zgodność z modułami ``bot_core``."""

from __future__ import annotations

from importlib import import_module as _import_module
from typing import Any

_runtime = _import_module('tests')
__all__ = getattr(_runtime, '__all__', [])

def __getattr__(name: str) -> Any:  # pragma: no cover - delegacja do właściwego pakietu testowego
    return getattr(_runtime, name)


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(set(__all__) | set(dir(_runtime)))
