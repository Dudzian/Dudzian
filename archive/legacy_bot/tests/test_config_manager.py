"""Legacy compatibility shim delegating to :mod:`KryptoLowca.tests.test_config_manager`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.tests.test_config_manager", "tests/test_config_manager.py")
