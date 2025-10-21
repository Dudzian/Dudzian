"""Zgodność wsteczna: re-eksportuje ``AppState`` z nowego pakietu Stage6."""

from __future__ import annotations

import warnings

from bot_core.ui.trading.state import AppState

warnings.warn(
    "KryptoLowca.ui.trading.state jest przestarzałe – użyj bot_core.ui.trading.state",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["AppState"]
