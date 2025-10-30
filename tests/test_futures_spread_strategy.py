from __future__ import annotations

from bot_core.strategies.base import MarketSnapshot
from bot_core.strategies.futures_spread import (
    FuturesSpreadSettings,
    FuturesSpreadStrategy,
)


def _snapshot(spread: float, basis: float, funding: float) -> MarketSnapshot:
    return MarketSnapshot(
        symbol="BTC-USD",
        timestamp=0,
        open=20000.0,
        high=20010.0,
        low=19950.0,
        close=20005.0,
        volume=12.0,
        indicators={
            "spread_zscore": spread,
            "basis": basis,
            "funding_rate": funding,
        },
    )


def test_futures_spread_strategy_generates_entry_and_exit() -> None:
    strategy = FuturesSpreadStrategy(
        FuturesSpreadSettings(entry_z=1.2, exit_z=0.3, max_bars=10, funding_exit=0.005, basis_exit=0.05)
    )

    warmup = [
        _snapshot(spread=0.2, basis=0.0, funding=0.0),
        _snapshot(spread=-0.1, basis=0.0, funding=0.0),
    ]
    strategy.warm_up(warmup)

    entry = strategy.on_data(_snapshot(spread=1.5, basis=0.01, funding=0.0005))
    assert entry and entry[0].side in {"short_front_long_back", "long_front_short_back"}

    # Spread mean-reverts -> exit
    exit_signals = strategy.on_data(_snapshot(spread=0.1, basis=0.0, funding=0.0002))
    assert exit_signals and exit_signals[0].side == "close_spread_position"


def test_futures_spread_strategy_triggers_exit_on_funding_risk() -> None:
    strategy = FuturesSpreadStrategy(
        FuturesSpreadSettings(entry_z=1.0, exit_z=0.3, max_bars=10, funding_exit=0.001, basis_exit=0.05)
    )
    strategy.on_data(_snapshot(spread=1.2, basis=0.0, funding=0.0))
    exit_signals = strategy.on_data(_snapshot(spread=0.8, basis=0.0, funding=0.002))
    assert exit_signals and exit_signals[0].metadata["exit_reason"] == "funding_risk"
