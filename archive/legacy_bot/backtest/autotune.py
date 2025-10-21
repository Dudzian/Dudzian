"""Legacy compatibility shim delegating to :mod:`KryptoLowca.backtest.autotune`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.backtest.autotune", "backtest/autotune.py")
