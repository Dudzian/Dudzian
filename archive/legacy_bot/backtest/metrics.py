"""Legacy compatibility shim delegating to :mod:`KryptoLowca.backtest.metrics`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.backtest.metrics", "backtest/metrics.py")
