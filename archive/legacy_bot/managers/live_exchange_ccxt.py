"""Legacy compatibility shim delegating to :mod:`bot_core.exchanges.ccxt_adapter`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "bot_core.exchanges.ccxt_adapter", "managers/live_exchange_ccxt.py")
