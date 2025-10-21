"""Legacy compatibility shim delegating to :mod:`KryptoLowca.backtest.engine`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.backtest.engine", "backtest/engine.py")
