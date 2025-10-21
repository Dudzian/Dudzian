"""Legacy compatibility shim delegating to :mod:`KryptoLowca.quick_db_test`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.quick_db_test", "quick_db_test.py")
