"""Shim testowy delegujący do kanonicznych scenariuszy z pakietu ``tests``."""
from __future__ import annotations

from tests import (
    test_auto_trader_regime_flow as _test_auto_trader_regime_flow,
    test_license_capabilities as _test_license_capabilities,
    test_ohlcv_backfill as _test_ohlcv_backfill,
    test_self_healing as _test_self_healing,
)
from tests.test_auto_trader_regime_flow import *  # noqa: F401,F403 - re-eksportujemy testy
from tests.test_license_capabilities import *  # noqa: F401,F403 - kompatybilność legacy
from tests.test_ohlcv_backfill import *  # noqa: F401,F403 - pokrycie regresji
from tests.test_self_healing import *  # noqa: F401,F403 - sekwencje naprawcze

__all__ = sorted(
    name
    for name in globals()
    if not name.startswith("_")
)
