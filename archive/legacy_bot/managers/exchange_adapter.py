"""Legacy compatibility shim delegating to :mod:`KryptoLowca.exchange_adapter`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.exchange_adapter", "managers/exchange_adapter.py")
