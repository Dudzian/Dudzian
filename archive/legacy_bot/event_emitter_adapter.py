"""Legacy compatibility shim delegating to :mod:`KryptoLowca.event_emitter_adapter`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.event_emitter_adapter", "event_emitter_adapter.py")
