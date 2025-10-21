"""Legacy compatibility shim delegating to :mod:`KryptoLowca.backtest.runner_preset`."""
from __future__ import annotations

from archive.legacy_bot._compat import proxy_globals

proxy_globals(globals(), "KryptoLowca.backtest.runner_preset", "backtest/runner_preset.py")
