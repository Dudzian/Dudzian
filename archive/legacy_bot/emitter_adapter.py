"""Legacy compatibility shim delegating to :mod:`KryptoLowca.emitter_adapter`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.emitter_adapter", "emitter_adapter.py")
