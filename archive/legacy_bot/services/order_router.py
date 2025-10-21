"""Legacy compatibility shim delegating to :mod:`KryptoLowca.services.order_router`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.services.order_router", "services/order_router.py")
