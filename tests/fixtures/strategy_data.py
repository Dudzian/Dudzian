"""Wspólne fikstury/stuby danych dla testów strategii."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from bot_core.strategies.base import MarketSnapshot


@dataclass(frozen=True)
class CrossExchangeFixture:
    """Zestaw przykładowych świec dla scenariuszy arbitrażowych."""

    entry: MarketSnapshot
    exit: MarketSnapshot
    timeout: MarketSnapshot


@dataclass(frozen=True)
class VolatilitySeriesFixture:
    """Seria cenowa do testów kontroli zmienności."""

    history: Sequence[MarketSnapshot]
    volatile_tick: MarketSnapshot
    follow_up: MarketSnapshot


def build_cross_exchange_fixture() -> CrossExchangeFixture:
    def snapshot(primary_bid: float, primary_ask: float, secondary_bid: float, secondary_ask: float, timestamp: int) -> MarketSnapshot:
        return MarketSnapshot(
            symbol="BTC_USDT_CROSS",
            timestamp=timestamp,
            open=primary_bid,
            high=primary_ask,
            low=primary_bid,
            close=primary_ask,
            volume=1_000_000.0,
            indicators={
                "primary_bid": primary_bid,
                "primary_ask": primary_ask,
                "secondary_bid": secondary_bid,
                "secondary_ask": secondary_ask,
                "secondary_timestamp": timestamp,
            },
        )

    entry = snapshot(100.0, 100.2, 101.0, 101.2, 1_000)
    exit_tick = snapshot(100.3, 100.35, 100.37, 100.4, 1_040)
    timeout_tick = snapshot(100.1, 100.3, 101.0, 101.2, 1_020)
    return CrossExchangeFixture(entry=entry, exit=exit_tick, timeout=timeout_tick)


def build_volatility_series_fixture() -> VolatilitySeriesFixture:
    def snapshot(price: float, timestamp: int) -> MarketSnapshot:
        return MarketSnapshot(
            symbol="ETH_USDT",
            timestamp=timestamp,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=1_000_000.0,
        )

    history = [snapshot(price, idx) for idx, price in enumerate([100.0, 100.5, 101.0, 102.0, 104.0])]
    volatile_tick = snapshot(110.0, 10)
    follow_up = snapshot(110.1, 11)
    return VolatilitySeriesFixture(history=tuple(history), volatile_tick=volatile_tick, follow_up=follow_up)

