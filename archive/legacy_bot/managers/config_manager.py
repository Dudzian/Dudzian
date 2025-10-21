"""Legacy compatibility shim delegating to :mod:`KryptoLowca.config_manager`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.config_manager", "managers/config_manager.py")
