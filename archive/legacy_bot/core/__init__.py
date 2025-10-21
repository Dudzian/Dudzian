"""Legacy compatibility shim delegating to :mod:`KryptoLowca.core.__init__`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.core.__init__", "core/__init__.py")
