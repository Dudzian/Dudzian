"""Legacy compatibility shim delegating to :mod:`KryptoLowca.database_manager`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.database_manager", "managers/database_manager.py")
