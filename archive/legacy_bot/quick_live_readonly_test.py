"""Legacy compatibility shim delegating to :mod:`KryptoLowca.quick_live_readonly_test`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.quick_live_readonly_test", "quick_live_readonly_test.py")
