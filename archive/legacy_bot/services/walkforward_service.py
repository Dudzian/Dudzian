"""Legacy compatibility shim delegating to :mod:`KryptoLowca.services.walkforward_service`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "bot_core.services.walkforward_service", "services/walkforward_service.py")
