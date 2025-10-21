"""Legacy compatibility shim delegating to :mod:`KryptoLowca.wfa_daemon`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.wfa_daemon", "wfa_daemon.py")
