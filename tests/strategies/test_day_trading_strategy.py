from bot_core.strategies.base import MarketSnapshot
from bot_core.strategies.day_trading import DayTradingSettings, DayTradingStrategy
import pytest


def _snapshot(
    symbol: str,
    close: float,
    atr: float,
    *,
    high: float | None = None,
    low: float | None = None,
    timestamp: int = 0,
) -> MarketSnapshot:
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
    strategy.prepare()
    strategy.warmup([_snapshot("BTCUSDT", 100.0, 0.01, high=101.0, low=99.0)])

    enter = strategy.decide(_snapshot("BTCUSDT", 101.0, 0.01, high=101.5, low=100.2, timestamp=2))
    assert enter and enter[0].side == "buy"

    exit_signals = strategy.decide(
        _snapshot("BTCUSDT", 104.0, 0.01, high=104.5, low=103.5, timestamp=3)
    )
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
    strategy.prepare()
    strategy.warmup([_snapshot("ETHUSDT", 200.0, 0.01, high=201.0, low=199.0)])

    enter = strategy.decide(_snapshot("ETHUSDT", 198.0, 0.01, high=199.0, low=197.0, timestamp=2))
    assert enter and enter[0].side == "sell"

    stop = strategy.decide(_snapshot("ETHUSDT", 202.0, 0.01, high=203.0, low=201.5, timestamp=3))
    assert stop and stop[0].side == "buy"
    assert stop[0].metadata.get("exit_reason") == "stop_loss"


def test_day_trading_settings_validation_and_from_parameters() -> None:
    with pytest.raises(ValueError, match="exit_threshold must be lower"):
        DayTradingSettings(entry_threshold=0.5, exit_threshold=0.5)

    with pytest.raises(ValueError, match="bias_strength must be in the range"):
        DayTradingSettings(bias_strength=1.2)

    settings = DayTradingSettings.from_parameters(
        {"momentum_window": "2", "entry_threshold": "0.4", "bias_strength": "0.3"}
    )
    assert settings.momentum_window == 2
    assert settings.entry_threshold == 0.4
    assert settings.bias_strength == 0.3


def test_day_trading_strategy_time_exit_and_teardown() -> None:
    strategy = DayTradingStrategy(
        DayTradingSettings(
            momentum_window=1,
            volatility_window=1,
            entry_threshold=0.1,
            exit_threshold=0.05,
            take_profit_atr=10.0,
            stop_loss_atr=10.0,
            max_holding_bars=1,
            atr_floor=0.01,
            bias_strength=0.0,
        )
    )
    strategy.prepare()
    assert strategy.decide(_snapshot("SOLUSDT", 10.0, 0.01)) == []
    enter = strategy.decide(_snapshot("SOLUSDT", 10.2, 0.01, timestamp=2))
    assert enter and enter[0].side == "buy"
    exit_signal = strategy.decide(_snapshot("SOLUSDT", 10.21, 0.01, timestamp=3))
    assert exit_signal and exit_signal[0].metadata.get("exit_reason") == "time_exit"
    strategy.teardown()
    assert strategy.decide(_snapshot("SOLUSDT", 10.3, 0.01, timestamp=4)) == []


def test_day_trading_strategy_momentum_fade_exit_and_zero_bias_midpoint() -> None:
    strategy = DayTradingStrategy(
        DayTradingSettings(
            momentum_window=1,
            volatility_window=1,
            entry_threshold=0.08,
            exit_threshold=0.06,
            take_profit_atr=10.0,
            stop_loss_atr=10.0,
            max_holding_bars=5,
            atr_floor=0.01,
            bias_strength=0.0,
        )
    )
    strategy.prepare()
    strategy.warmup([_snapshot("XRPUSDT", 1.0, 0.01)])
    assert strategy.decide(_snapshot("XRPUSDT", 1.1, 0.01, timestamp=2))
    faded = strategy.decide(_snapshot("XRPUSDT", 1.1005, 0.01, timestamp=3))
    assert faded and faded[0].metadata.get("exit_reason") == "momentum_fade"
    assert strategy._intraday_bias(_snapshot("XRPUSDT", 0.0, 0.01, high=0.0, low=0.0)) == 0.0
