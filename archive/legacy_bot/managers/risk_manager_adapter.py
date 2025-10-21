"""Legacy compatibility shim delegating to :mod:`KryptoLowca.risk_manager`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.risk_manager", "managers/risk_manager_adapter.py")
