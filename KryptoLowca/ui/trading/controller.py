"""Zgodność wsteczna: kontroler GUI deleguje do ``bot_core.ui.trading.controller``."""

from __future__ import annotations

import warnings

from bot_core.ui.trading.controller import TradingSessionController

warnings.warn(
    "KryptoLowca.ui.trading.controller jest przestarzałe – użyj bot_core.ui.trading.controller",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["TradingSessionController"]
