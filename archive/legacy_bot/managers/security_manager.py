"""Legacy compatibility shim delegating to :mod:`KryptoLowca.security_manager`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.security_manager", "managers/security_manager.py")
