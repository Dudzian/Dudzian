"""Legacy compatibility shim delegating to :mod:`KryptoLowca.backtest.walkforward_service`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.backtest.walkforward_service", "backtest/walkforward_service.py")
