"""Legacy compatibility shim delegating to :mod:`KryptoLowca.backtest.optimize`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.backtest.optimize", "backtest/optimize.py")
