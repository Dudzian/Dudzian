from bot_core.strategies.base import MarketSnapshot
from bot_core.strategies.statistical_arbitrage import (
    StatisticalArbitrageSettings,
    StatisticalArbitrageStrategy,
)


def _snapshot(close: float, paired: float, ts: int) -> MarketSnapshot:
    return MarketSnapshot(
        symbol="BTCUSDT",
        timestamp=ts,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1_000.0,
        indicators={"paired_price": paired, "paired_symbol": "ETHUSDT"},
    )


def test_statistical_arbitrage_opens_and_closes_position() -> None:
    settings = StatisticalArbitrageSettings(lookback=5, spread_entry_z=0.5, spread_exit_z=0.1)
    strategy = StatisticalArbitrageStrategy(settings)

    history_points = [
        (100.5, 100.0),
        (99.7, 100.1),
        (100.2, 100.0),
        (99.9, 100.2),
        (100.1, 99.95),
    ]
    history = [_snapshot(close=close, paired=paired, ts=idx) for idx, (close, paired) in enumerate(history_points)]
    strategy.warm_up(history)

    entry = strategy.on_data(_snapshot(close=120.0, paired=100.0, ts=10))
    assert entry and entry[0].side == "short_primary_long_secondary"

    exit_signal = strategy.on_data(_snapshot(close=101.0, paired=100.5, ts=11))
    assert exit_signal and exit_signal[0].side.startswith("close_")
