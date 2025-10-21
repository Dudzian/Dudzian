"""Legacy compatibility shim delegating to :mod:`KryptoLowca.tests.test_trading_engine`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.tests.test_trading_engine", "tests/test_trading_engine.py")
