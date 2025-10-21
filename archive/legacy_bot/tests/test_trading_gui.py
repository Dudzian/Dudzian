"""Legacy compatibility shim delegating to :mod:`KryptoLowca.tests.test_trading_gui`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.tests.test_trading_gui", "tests/test_trading_gui.py")
