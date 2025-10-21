"""Legacy compatibility shim delegating to :mod:`KryptoLowca.strategies.rules_v0`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.strategies.rules_v0", "strategies/rules_v0.py")
