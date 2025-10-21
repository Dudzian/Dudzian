"""Legacy compatibility shim delegating to :mod:`KryptoLowca.services.marketdata`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.services.marketdata", "services/marketdata.py")
