"""Legacy compatibility shim delegating to :mod:`KryptoLowca.paper_exchange`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.paper_exchange", "managers/paper_exchange.py")
