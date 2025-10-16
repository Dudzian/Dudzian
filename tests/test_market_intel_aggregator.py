from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping, MutableMapping, Sequence

import math
import statistics

import pytest

from bot_core.market_intel import MarketIntelAggregator, MarketIntelQuery


class MemoryCacheStorage:
    def __init__(self, payloads: Mapping[str, Mapping[str, Sequence[Sequence[float]]]]):
        self._payloads = dict(payloads)
        self._metadata: MutableMapping[str, str] = {}

    def read(self, key: str) -> Mapping[str, Sequence[Sequence[float]]]:
        return self._payloads[key]

    def write(self, key: str, payload: Mapping[str, Sequence[Sequence[float]]]) -> None:
        self._payloads[key] = payload

    def metadata(self) -> MutableMapping[str, str]:
        return self._metadata

    def latest_timestamp(self, key: str) -> float | None:
        payload = self._payloads.get(key)
        if not payload:
            return None
        rows = payload.get("rows", ())
        if not rows:
            return None
        return float(rows[-1][0])


def test_market_intel_aggregator_builds_metrics() -> None:
    rows = [
        [1_000.0, 100.0, 101.0, 99.0, 100.0, 10.0],
        [2_000.0, 101.0, 106.0, 100.0, 105.0, 12.0],
        [3_000.0, 106.0, 111.0, 105.0, 110.0, 15.0],
        [4_000.0, 112.0, 121.0, 111.0, 120.0, 20.0],
    ]
    storage = MemoryCacheStorage(
        {
            "BTC_USDT::1h": {
                "columns": ("open_time", "open", "high", "low", "close", "volume"),
                "rows": rows,
            }
        }
    )
    aggregator = MarketIntelAggregator(storage)

    snapshot = aggregator.build_snapshot(
        MarketIntelQuery(symbol="BTC_USDT", interval="1h", lookback_bars=4)
    )

    assert snapshot.symbol == "BTC_USDT"
    assert snapshot.interval == "1h"
    assert snapshot.bar_count == 4
    assert snapshot.start == datetime.fromtimestamp(1.0, tz=timezone.utc)
    assert snapshot.end == datetime.fromtimestamp(4.0, tz=timezone.utc)
    assert snapshot.price_change_pct == pytest.approx((120.0 / 100.0 - 1.0) * 100.0)

    returns = [
        105.0 / 100.0 - 1.0,
        110.0 / 105.0 - 1.0,
        120.0 / 110.0 - 1.0,
    ]
    expected_volatility = math.sqrt(len(returns)) * statistics.pstdev(returns) * 100.0
    assert snapshot.volatility_pct == pytest.approx(expected_volatility)
    assert snapshot.max_drawdown_pct == pytest.approx(0.0)
    assert snapshot.average_volume == pytest.approx(sum(row[-1] for row in rows) / len(rows))
    notional = [row[4] * row[5] for row in rows]
    assert snapshot.liquidity_usd == pytest.approx(sum(notional) / len(notional))
    assert snapshot.momentum_score is not None
    assert snapshot.metadata["bars_used"] == pytest.approx(4.0)


def test_market_intel_aggregator_respects_lookback() -> None:
    rows = [
        [1_000.0, 100.0, 101.0, 99.0, 100.0, 10.0],
        [2_000.0, 101.0, 106.0, 100.0, 105.0, 12.0],
        [3_000.0, 106.0, 111.0, 105.0, 110.0, 15.0],
        [4_000.0, 112.0, 121.0, 111.0, 120.0, 20.0],
    ]
    storage = MemoryCacheStorage(
        {
            "ETH_USDT::1h": {
                "columns": ("open_time", "open", "high", "low", "close", "volume"),
                "rows": rows,
            }
        }
    )
    aggregator = MarketIntelAggregator(storage)
    snapshot = aggregator.build_snapshot(
        MarketIntelQuery(symbol="ETH_USDT", interval="1h", lookback_bars=2)
    )

    assert snapshot.bar_count == 2
    assert snapshot.start == datetime.fromtimestamp(3.0, tz=timezone.utc)
    assert snapshot.end == datetime.fromtimestamp(4.0, tz=timezone.utc)
    assert snapshot.price_change_pct == pytest.approx((120.0 / 110.0 - 1.0) * 100.0)
