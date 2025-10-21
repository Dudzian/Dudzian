"""Legacy compatibility shim delegating to :mod:`KryptoLowca.scanner`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.scanner", "managers/scanner.py")
