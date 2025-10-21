"""Shim agregujący scenariusze exchange z nowej bazy testów."""
from __future__ import annotations

from tests.runtime.test_bootstrap_license_validation import *  # noqa: F401,F403
from tests.test_binance_futures_adapter import *  # noqa: F401,F403
from tests.test_binance_spot_adapter import *  # noqa: F401,F403
from tests.test_kraken_spot_adapter import *  # noqa: F401,F403
from tests.test_live_execution_router import *  # noqa: F401,F403
from tests.test_live_router import *  # noqa: F401,F403
from tests.test_ohlcv_backfill import *  # noqa: F401,F403
from tests.test_pipeline_smoke_ccxt import *  # noqa: F401,F403
from tests.test_runtime_pipeline_offline import *  # noqa: F401,F403
from tests.test_self_healing import *  # noqa: F401,F403
from tests.test_zonda_spot_adapter import *  # noqa: F401,F403

__all__ = sorted(
    name
    for name in globals()
    if not name.startswith("_")
)
