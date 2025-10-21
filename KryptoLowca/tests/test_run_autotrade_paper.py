"""Shim legacy przekierowujący do testów autotradera poprzez moduł egzekucji."""
from __future__ import annotations

from KryptoLowca.tests.test_execution_service import *  # noqa: F401,F403

__all__ = sorted(
    name
    for name in globals()
    if not name.startswith("_")
)
