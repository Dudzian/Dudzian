from bot_core.strategies.base import MarketSnapshot
from bot_core.strategies.volatility_target import (
    VolatilityTargetSettings,
    VolatilityTargetStrategy,
)


def _snapshot(price: float, timestamp: int) -> MarketSnapshot:
    return MarketSnapshot(
        symbol="ETH_USDT",
        timestamp=timestamp,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=1_000_000.0,
    )


def test_volatility_target_rebalances_when_threshold_exceeded() -> None:
    settings = VolatilityTargetSettings(
        target_volatility=0.05,
        lookback=5,
        rebalance_threshold=0.05,
        min_allocation=0.2,
        max_allocation=1.0,
        floor_volatility=0.01,
    )
    strategy = VolatilityTargetStrategy(settings)

    base_prices = [100.0, 100.5, 101.0, 102.0, 104.0]
    history = [_snapshot(price, idx) for idx, price in enumerate(base_prices)]
    strategy.warm_up(history)

    next_price = _snapshot(110.0, 10)
    signals = strategy.on_data(next_price)
    assert signals and signals[0].side == "rebalance"
    metadata = signals[0].metadata
    assert "target_allocation" in metadata
    assert metadata["target_allocation"] <= settings.max_allocation

    # kolejny pomiar bez istotnej zmiany nie powinien generować sygnału
    follow_up = _snapshot(110.1, 11)
    assert not strategy.on_data(follow_up)
