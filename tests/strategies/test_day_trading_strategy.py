from bot_core.strategies.base import MarketSnapshot
from bot_core.strategies.day_trading import DayTradingSettings, DayTradingStrategy


def _snapshot(symbol: str, close: float, atr: float, *, high: float | None = None, low: float | None = None, timestamp: int = 0) -> MarketSnapshot:
    high_price = high if high is not None else close
    low_price = low if low is not None else close
    return MarketSnapshot(
        symbol=symbol,
        timestamp=timestamp,
        open=close,
        high=high_price,
        low=low_price,
        close=close,
        volume=10_000.0,
        indicators={"atr": atr},
    )


def test_day_trading_strategy_generates_long_and_exit() -> None:
    settings = DayTradingSettings(
        momentum_window=1,
        volatility_window=2,
        entry_threshold=0.5,
        exit_threshold=0.1,
        take_profit_atr=1.5,
        stop_loss_atr=2.0,
        max_holding_bars=6,
        atr_floor=0.01,
        bias_strength=0.0,
    )
    strategy = DayTradingStrategy(settings)
    strategy.warm_up([_snapshot("BTCUSDT", 100.0, 0.01, high=101.0, low=99.0)])

    enter = strategy.on_data(_snapshot("BTCUSDT", 101.0, 0.01, high=101.5, low=100.2, timestamp=2))
    assert enter and enter[0].side == "buy"

    exit_signals = strategy.on_data(_snapshot("BTCUSDT", 104.0, 0.01, high=104.5, low=103.5, timestamp=3))
    assert exit_signals and exit_signals[0].side == "sell"
    assert exit_signals[0].metadata.get("exit_reason") == "take_profit"


def test_day_trading_strategy_short_stop_loss() -> None:
    settings = DayTradingSettings(
        momentum_window=1,
        volatility_window=2,
        entry_threshold=0.5,
        exit_threshold=0.1,
        take_profit_atr=1.5,
        stop_loss_atr=1.0,
        max_holding_bars=6,
        atr_floor=0.01,
        bias_strength=0.0,
    )
    strategy = DayTradingStrategy(settings)
    strategy.warm_up([_snapshot("ETHUSDT", 200.0, 0.01, high=201.0, low=199.0)])

    enter = strategy.on_data(_snapshot("ETHUSDT", 198.0, 0.01, high=199.0, low=197.0, timestamp=2))
    assert enter and enter[0].side == "sell"

    stop = strategy.on_data(_snapshot("ETHUSDT", 202.0, 0.01, high=203.0, low=201.5, timestamp=3))
    assert stop and stop[0].side == "buy"
    assert stop[0].metadata.get("exit_reason") == "stop_loss"
