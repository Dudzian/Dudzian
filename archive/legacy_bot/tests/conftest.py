"""Legacy compatibility shim delegating to :mod:`KryptoLowca.tests.conftest`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.tests.conftest", "tests/conftest.py")
