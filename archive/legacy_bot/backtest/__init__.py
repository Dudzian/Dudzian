"""Legacy compatibility shim delegating to :mod:`KryptoLowca.backtest.__init__`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.backtest.__init__", "backtest/__init__.py")
