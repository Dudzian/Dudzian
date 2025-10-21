"""Legacy compatibility shim delegating to :mod:`KryptoLowca.data_preprocessor`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.data_preprocessor", "data_preprocessor.py")
