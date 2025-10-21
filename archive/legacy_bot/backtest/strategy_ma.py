"""Legacy compatibility shim delegating to :mod:`KryptoLowca.backtest.strategy_ma`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.backtest.strategy_ma", "backtest/strategy_ma.py")
