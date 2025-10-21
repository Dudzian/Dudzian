"""Legacy compatibility shim delegating to :mod:`KryptoLowca.core.trading_engine`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.core.trading_engine", "core/trading_engine.py")
