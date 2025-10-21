"""Legacy compatibility shim delegating to :mod:`KryptoLowca.ai_manager`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.ai_manager", "managers/ai_manager.py")
