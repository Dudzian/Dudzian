"""Legacy compatibility shim delegating to :mod:`KryptoLowca.trading_gui`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.trading_gui", "trading_gui.py")
