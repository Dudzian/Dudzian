from __future__ import annotations

from bot_core.strategies.base import MarketSnapshot
from bot_core.strategies.cross_exchange_hedge import (
    CrossExchangeHedgeSettings,
    CrossExchangeHedgeStrategy,
)


def _snapshot(basis: float, inventory: float, latency: float) -> MarketSnapshot:
    return MarketSnapshot(
        symbol="ETH-USD",
        timestamp=0,
        open=1500.0,
        high=1510.0,
        low=1490.0,
        close=1505.0,
        volume=5.0,
        indicators={
            "spot_basis": basis,
            "inventory_skew": inventory,
            "latency_ms": latency,
        },
    )


def test_cross_exchange_hedge_outputs_ratio() -> None:
    strategy = CrossExchangeHedgeStrategy(
        CrossExchangeHedgeSettings(basis_scale=0.01, inventory_scale=0.35, latency_limit_ms=200.0, max_hedge_ratio=0.9)
    )
    strategy.warm_up([_snapshot(0.0, 0.0, 50.0)])

    signals = strategy.on_data(_snapshot(0.008, -0.1, 60.0))
    assert signals and signals[0].side == "rebalance_delta"
    assert 0.0 < signals[0].metadata["target_ratio"] <= 0.9


def test_cross_exchange_hedge_latency_penalty_reduces_ratio() -> None:
    strategy = CrossExchangeHedgeStrategy(
        CrossExchangeHedgeSettings(basis_scale=0.01, inventory_scale=0.2, latency_limit_ms=100.0, max_hedge_ratio=1.0)
    )
    fast_signal = strategy.on_data(_snapshot(0.01, 0.0, 30.0))[0].metadata["target_ratio"]
    slow_signal = strategy.on_data(_snapshot(0.01, 0.0, 120.0))[0].metadata["target_ratio"]
    assert abs(slow_signal) < abs(fast_signal)
