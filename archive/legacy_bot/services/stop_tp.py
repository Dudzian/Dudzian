"""Legacy compatibility shim delegating to :mod:`KryptoLowca.services.stop_tp`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "bot_core.services.stop_tp", "services/stop_tp.py")
