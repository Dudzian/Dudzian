"""Legacy compatibility shim delegating to :mod:`KryptoLowca.tests.__init__`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.tests.__init__", "tests/__init__.py")
