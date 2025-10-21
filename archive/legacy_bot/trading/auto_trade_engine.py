"""Legacy compatibility shim delegating to :mod:`KryptoLowca.trading.auto_trade_engine`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.trading.auto_trade_engine", "trading/auto_trade_engine.py")
