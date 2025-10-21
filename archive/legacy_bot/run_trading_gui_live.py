"""Legacy compatibility shim delegating to :mod:`KryptoLowca.run_trading_gui_live`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.run_trading_gui_live", "run_trading_gui_live.py")
