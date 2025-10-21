"""Legacy compatibility shim delegating to :mod:`KryptoLowca.backtest.scan_signals`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.backtest.scan_signals", "backtest/scan_signals.py")
