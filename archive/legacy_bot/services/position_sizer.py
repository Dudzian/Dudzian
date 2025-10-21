"""Legacy compatibility shim delegating to :mod:`KryptoLowca.services.position_sizer`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.services.position_sizer", "services/position_sizer.py")
