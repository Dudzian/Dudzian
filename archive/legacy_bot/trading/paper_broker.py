"""Legacy compatibility shim delegating to :mod:`KryptoLowca.trading.paper_broker`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.trading.paper_broker", "trading/paper_broker.py")
