"""Legacy compatibility shim delegating to :mod:`KryptoLowca.services.persistence`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "bot_core.services.persistence", "services/persistence.py")
