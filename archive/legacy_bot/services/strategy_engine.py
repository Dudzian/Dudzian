"""Legacy compatibility shim delegating to :mod:`KryptoLowca.services.strategy_engine`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.services.strategy_engine", "services/strategy_engine.py")
