"""Legacy compatibility shim delegating to :mod:`KryptoLowca.report_manager`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.report_manager", "managers/report_manager.py")
