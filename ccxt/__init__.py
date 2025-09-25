"""Minimalny stub biblioteki ccxt na potrzeby testów jednostkowych."""
from __future__ import annotations

import sys
import builtins
from types import SimpleNamespace


class AuthenticationError(Exception):
    """Zastępczy wyjątek ccxt AuthenticationError."""


base = SimpleNamespace(errors=SimpleNamespace(AuthenticationError=AuthenticationError))

__all__ = ["AuthenticationError", "base"]

# Ułatwienie dla testów odwołujących się do globalnego `ccxt` bez importu.
builtins.ccxt = sys.modules[__name__]
