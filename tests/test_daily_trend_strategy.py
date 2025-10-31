"""Testy strategii trend/momentum na danych dziennych."""
from __future__ import annotations

from pathlib import Path


from bot_core.strategies import MarketSnapshot
from bot_core.strategies.daily_trend import (
    DailyTrendMomentumSettings,
    DailyTrendMomentumStrategy,
)


def _snapshot(ts: int, price: float, high: float | None = None, low: float | None = None) -> MarketSnapshot:
    high = high if high is not None else price
    low = low if low is not None else price
    return MarketSnapshot(
        symbol="BTCUSDT",
        timestamp=ts,
        open=price,
        high=high,
        low=low,
        close=price,
        volume=1.0,
    )


def test_daily_trend_strategy_generates_entry_signal() -> None:
    settings = DailyTrendMomentumSettings(
        fast_ma=3,
        slow_ma=5,
        breakout_lookback=4,
        momentum_window=3,
        atr_window=3,
        atr_multiplier=1.5,
        min_trend_strength=0.0,
        min_momentum=0.0,
    )
    strategy = DailyTrendMomentumStrategy(settings)

    history = [
        _snapshot(1, 100, high=101, low=99),
        _snapshot(2, 101, high=102, low=100),
        _snapshot(3, 102, high=103, low=101),
        _snapshot(4, 103, high=104, low=102),
        _snapshot(5, 104, high=105, low=103),
    ]
    strategy.warm_up(history)

    signal_snapshot = _snapshot(6, 107, high=108, low=106)
    signals = strategy.on_data(signal_snapshot)

    assert len(signals) == 1
    signal = signals[0]
    assert signal.side == "buy"
    assert 0.0 <= signal.confidence <= 1.0
    assert signal.metadata["position"] == 1.0
    assert signal.metadata["breakout_high"] <= signal_snapshot.close


def test_daily_trend_strategy_generates_exit_on_stop() -> None:
    settings = DailyTrendMomentumSettings(
        fast_ma=3,
        slow_ma=5,
        breakout_lookback=4,
        momentum_window=3,
        atr_window=3,
        atr_multiplier=1.5,
        min_trend_strength=0.0,
        min_momentum=0.0,
    )
    strategy = DailyTrendMomentumStrategy(settings)

    history = [
        _snapshot(1, 100, high=101, low=99),
        _snapshot(2, 101, high=102, low=100),
        _snapshot(3, 102, high=103, low=101),
        _snapshot(4, 103, high=104, low=102),
        _snapshot(5, 104, high=105, low=103),
    ]
    strategy.warm_up(history)

    entry_snapshot = _snapshot(6, 108, high=109, low=107)
    entry_signals = strategy.on_data(entry_snapshot)
    assert entry_signals and entry_signals[0].side == "buy"

    exit_snapshot = _snapshot(7, 100, high=101, low=97)
    exit_signals = strategy.on_data(exit_snapshot)

    assert exit_signals, "Strategia powinna wygenerować sygnał wyjścia"
    exit_signal = exit_signals[0]
    assert exit_signal.side == "sell"
    assert 0.0 <= exit_signal.confidence <= 1.0
    assert exit_signal.metadata["position"] == 0.0


def test_daily_trend_strategy_requires_history() -> None:
    strategy = DailyTrendMomentumStrategy(
        DailyTrendMomentumSettings(fast_ma=3, slow_ma=5, breakout_lookback=4, momentum_window=3, atr_window=3)
    )
    snapshot = _snapshot(1, 100, high=101, low=99)

    assert strategy.on_data(snapshot) == []
