from bot_core.strategies.base import MarketSnapshot
from bot_core.strategies.grid import GridTradingSettings, GridTradingStrategy


def _snapshot(price: float, timestamp: int) -> MarketSnapshot:
    return MarketSnapshot(
        symbol="ETHUSDT",
        timestamp=timestamp,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=5_000.0,
    )


def test_grid_strategy_generates_signals_on_level_change() -> None:
    settings = GridTradingSettings(grid_size=3, grid_spacing=0.01, max_inventory=1.0)
    strategy = GridTradingStrategy(settings)

    # warm-up base price
    strategy.on_data(_snapshot(100.0, 1))

    # move below one level -> buy
    buy_signals = strategy.on_data(_snapshot(98.5, 2))
    assert buy_signals and buy_signals[0].side == "buy"
    assert buy_signals[0].metadata["strategy"]["risk_label"] == "moderate"

    # move above base -> sell
    sell_signals = strategy.on_data(_snapshot(101.5, 3))
    assert sell_signals and sell_signals[0].side == "sell"
