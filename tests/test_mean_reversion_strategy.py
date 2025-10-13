from bot_core.strategies.base import MarketSnapshot
from bot_core.strategies.mean_reversion import MeanReversionSettings, MeanReversionStrategy


def _snapshot(price: float, *, volume: float = 1_000_000.0, timestamp: int = 0) -> MarketSnapshot:
    return MarketSnapshot(
        symbol="BTC_USDT",
        timestamp=timestamp,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=volume,
    )


def test_mean_reversion_generates_entry_and_exit_signals() -> None:
    settings = MeanReversionSettings(lookback=10, entry_zscore=1.0, exit_zscore=0.2, min_volume_usd=0)
    strategy = MeanReversionStrategy(settings)

    history = [_snapshot(100.0 + i * 0.1, timestamp=i) for i in range(10)]
    strategy.warm_up(history)

    entry_snapshot = _snapshot(98.0, timestamp=11)
    signals = strategy.on_data(entry_snapshot)
    assert signals and signals[0].side == "buy"

    # kilka barów utrzymania pozycji
    for i in range(12, 15):
        strategy.on_data(_snapshot(98.5 + (i - 12) * 0.1, timestamp=i))

    exit_snapshot = _snapshot(101.0, timestamp=20)
    exit_signals = strategy.on_data(exit_snapshot)
    assert exit_signals and exit_signals[0].side == "sell"


def test_mean_reversion_respects_volatility_cap() -> None:
    settings = MeanReversionSettings(lookback=6, entry_zscore=1.0, exit_zscore=0.2, volatility_cap=0.01, min_volume_usd=0)
    strategy = MeanReversionStrategy(settings)

    base_history = [_snapshot(100.0, timestamp=i) for i in range(6)]
    strategy.warm_up(base_history)

    high_vol_snapshots = [
        _snapshot(price, timestamp=10 + idx)
        for idx, price in enumerate([110.0, 90.0, 115.0, 85.0, 120.0])
    ]
    for snap in high_vol_snapshots:
        strategy.on_data(snap)

    # Strategia nie powinna otworzyć pozycji przy wysokiej zmienności
    assert all(state.position == 0 for state in strategy._states.values())
