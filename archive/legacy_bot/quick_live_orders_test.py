"""Legacy compatibility shim delegating to :mod:`KryptoLowca.quick_live_orders_test`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.quick_live_orders_test", "quick_live_orders_test.py")
