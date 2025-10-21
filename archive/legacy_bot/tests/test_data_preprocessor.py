"""Legacy compatibility shim delegating to :mod:`KryptoLowca.tests.test_data_preprocessor`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.tests.test_data_preprocessor", "tests/test_data_preprocessor.py")
