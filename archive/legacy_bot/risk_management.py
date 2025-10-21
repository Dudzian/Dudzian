"""Legacy compatibility shim delegating to :mod:`KryptoLowca.risk_management`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.risk_management", "risk_management.py")
