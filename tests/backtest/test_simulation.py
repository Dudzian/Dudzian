import math
from datetime import datetime, timezone

import pytest

from bot_core.backtest.simulation import MatchingEngine


def test_slippage_scales_with_filled_size_and_direction(heavy_slippage_scenario):
    engine = MatchingEngine(heavy_slippage_scenario)
    now = datetime.now(timezone.utc)

    engine.submit_market_order(side="buy", size=10.0, index=0, timestamp=now)
    fills = engine.process_bar(index=0, timestamp=now, bar={"close": 100.0})

    assert len(fills) == 1
    assert math.isclose(fills[0].slippage, 10.0, rel_tol=1e-9)

    engine.submit_market_order(side="sell", size=2.0, index=1, timestamp=now)
    fills = engine.process_bar(index=1, timestamp=now, bar={"close": 50.0})

    assert len(fills) == 1
    assert math.isclose(fills[0].slippage, -1.0, rel_tol=1e-9)
