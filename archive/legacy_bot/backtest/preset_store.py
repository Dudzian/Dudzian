"""Legacy compatibility shim delegating to :mod:`KryptoLowca.backtest.preset_store`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.backtest.preset_store", "backtest/preset_store.py")
