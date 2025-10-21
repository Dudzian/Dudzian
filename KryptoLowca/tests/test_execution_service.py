"""Shim legacy delegujący testy serwisu egzekucji do nowej bazy."""
from __future__ import annotations

from tests.test_trading_controller import *  # noqa: F401,F403

__all__ = sorted(
    name
    for name in globals()
    if not name.startswith("_")
)
