"""Legacy compatibility shim delegating to :mod:`KryptoLowca.run_autotrade_paper`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.run_autotrade_paper", "run_autotrade_paper.py")
