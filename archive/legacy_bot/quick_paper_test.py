"""Legacy compatibility shim delegating to :mod:`KryptoLowca.quick_paper_test`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.quick_paper_test", "quick_paper_test.py")
