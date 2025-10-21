"""Legacy compatibility shim delegating to :mod:`KryptoLowca.__init__`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.__init__", "__init__.py")
