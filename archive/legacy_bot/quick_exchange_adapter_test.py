"""Legacy compatibility shim delegating to :mod:`KryptoLowca.quick_exchange_adapter_test`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.quick_exchange_adapter_test", "quick_exchange_adapter_test.py")
