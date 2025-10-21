"""Legacy compatibility shim delegating to :mod:`bot_core.auto_trader.app`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "bot_core.auto_trader.app", "auto_trader.py")
