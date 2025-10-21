"""Legacy compatibility shim delegating to :mod:`KryptoLowca.services.indicators`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.services.indicators", "services/indicators.py")
