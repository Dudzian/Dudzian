"""Zgodność wsteczna: re-eksportuje SecurityManager z nowego pakietu Stage6."""

from __future__ import annotations

import warnings

from bot_core.security.legacy import SecurityError, SecurityManager

warnings.warn(
    "KryptoLowca.security_manager jest przestarzałe – użyj bot_core.security.legacy.SecurityManager",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SecurityManager", "SecurityError"]
