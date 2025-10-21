"""Legacy compatibility shim delegating to :mod:`KryptoLowca.scripts.db_tools`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.scripts.db_tools", "scripts/db_tools.py")
