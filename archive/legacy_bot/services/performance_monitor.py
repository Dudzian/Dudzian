"""Legacy compatibility shim delegating to :mod:`KryptoLowca.services.performance_monitor`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.services.performance_monitor", "services/performance_monitor.py")
