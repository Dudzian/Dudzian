from bot_core.strategies.base import MarketSnapshot
from bot_core.strategies.scalping import ScalpingSettings, ScalpingStrategy


def _snapshot(symbol: str, close: float, timestamp: int = 0) -> MarketSnapshot:
    return MarketSnapshot(
        symbol=symbol,
        timestamp=timestamp,
        open=close,
        high=close * 1.001,
        low=close * 0.999,
        close=close,
        volume=10_000.0,
    )


def test_scalping_strategy_enters_and_exits_long() -> None:
    strategy = ScalpingStrategy(ScalpingSettings(min_price_change=0.001, take_profit=0.001, stop_loss=0.002))

    enter = strategy.on_data(_snapshot("BTCUSDT", 100.0, timestamp=1))
    assert enter == []

    signals = strategy.on_data(_snapshot("BTCUSDT", 101.0, timestamp=2))
    assert signals and signals[0].side == "buy"

    exit_signals = strategy.on_data(_snapshot("BTCUSDT", 101.2, timestamp=3))
    assert exit_signals and exit_signals[0].side == "sell"
    assert exit_signals[0].metadata.get("exit_reason") == "take_profit"


def test_scalping_strategy_enters_and_exits_short() -> None:
    strategy = ScalpingStrategy(
        ScalpingSettings(min_price_change=0.001, take_profit=0.002, stop_loss=0.0015, max_hold_bars=10)
    )

    strategy.on_data(_snapshot("ETHUSDT", 200.0, timestamp=10))

    short_signals = strategy.on_data(_snapshot("ETHUSDT", 198.0, timestamp=11))
    assert short_signals and short_signals[0].side == "sell"

    # gwałtowny zwrot ceny powinien zamknąć pozycję krótką zleceniem kupna
    exit_signals = strategy.on_data(_snapshot("ETHUSDT", 202.0, timestamp=12))
    assert exit_signals and exit_signals[0].side == "buy"
    assert exit_signals[0].metadata.get("exit_reason") == "stop_loss"
