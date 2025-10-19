from collections import deque
from math import log, sqrt

import pytest

from bot_core.strategies.base import MarketSnapshot
from bot_core.strategies.volatility_target import (
    VolatilityTargetSettings,
    VolatilityTargetStrategy,
    _SymbolState,
)
from tests.fixtures import build_volatility_series_fixture


def make_snapshot(
    *,
    symbol: str = "BTC/USDT",
    timestamp: int = 0,
    open_price: float | None = None,
    high: float | None = None,
    low: float | None = None,
    close: float = 1.0,
) -> MarketSnapshot:
    """Build a minimal ``MarketSnapshot`` for helper tests."""

    open_value = close if open_price is None else open_price
    high_value = close if high is None else high
    low_value = close if low is None else low
    return MarketSnapshot(
        symbol=symbol,
        timestamp=timestamp,
        open=open_value,
        high=high_value,
        low=low_value,
        close=close,
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

    fixtures = build_volatility_series_fixture()
    strategy.warm_up(fixtures.history)

    signals = strategy.on_data(fixtures.volatile_tick)
    assert signals and signals[0].side == "rebalance"
    metadata = signals[0].metadata
    assert "target_allocation" in metadata
    assert metadata["target_allocation"] <= settings.max_allocation

    # kolejny pomiar bez istotnej zmiany nie powinien generować sygnału
    assert not strategy.on_data(fixtures.follow_up)


def test_volatility_target_waits_for_full_history() -> None:
    settings = VolatilityTargetSettings(lookback=4)
    strategy = VolatilityTargetStrategy(settings)

    fixtures = build_volatility_series_fixture()
    partial_history = fixtures.history[:2]

    strategy.warm_up(partial_history)

    next_snapshot = fixtures.history[2]
    signals = strategy.on_data(next_snapshot)

    assert signals == []


def test_realized_volatility_handles_empty_and_populated_returns() -> None:
    settings = VolatilityTargetSettings(lookback=5)
    strategy = VolatilityTargetStrategy(settings)
    state = _SymbolState(returns=deque(maxlen=settings.history_size()))

    assert strategy._realized_volatility(state) == 0.0

    sample_returns = [0.1, -0.05, 0.2]
    state.returns.extend(sample_returns)
    realized = strategy._realized_volatility(state)

    mean_ret = sum(sample_returns) / len(sample_returns)
    variance = sum((value - mean_ret) ** 2 for value in sample_returns) / (len(sample_returns) - 1)
    assert realized == pytest.approx(sqrt(variance))


def test_update_state_appends_returns_and_handles_non_positive_closes() -> None:
    settings = VolatilityTargetSettings(lookback=3)
    strategy = VolatilityTargetStrategy(settings)
    state = _SymbolState(returns=deque(maxlen=settings.history_size()))

    first = make_snapshot(close=100.0)
    strategy._update_state(state, first)

    assert list(state.returns) == []
    assert state.last_price == 100.0

    second = make_snapshot(close=110.0)
    strategy._update_state(state, second)

    expected_return = log(110.0 / 100.0)
    assert list(state.returns) == [pytest.approx(expected_return)]
    assert state.last_price == 110.0

    zero_close = make_snapshot(close=0.0)
    strategy._update_state(state, zero_close)

    assert list(state.returns) == [pytest.approx(expected_return)]
    assert state.last_price == 0.0

    negative_close = make_snapshot(close=-50.0)
    strategy._update_state(state, negative_close)

    assert list(state.returns) == [pytest.approx(expected_return)]
    assert state.last_price == -50.0

    positive_after_negative = make_snapshot(close=120.0)
    strategy._update_state(state, positive_after_negative)

    assert list(state.returns) == [pytest.approx(expected_return)]
    assert state.last_price == 120.0

    resumed_positive = make_snapshot(close=132.0)
    strategy._update_state(state, resumed_positive)

    resumed_return = log(132.0 / 120.0)
    assert list(state.returns) == [
        pytest.approx(expected_return),
        pytest.approx(resumed_return),
    ]
    assert state.last_price == 132.0


@pytest.mark.parametrize(
    "realized, expected",
    [
        (0.0, 1.0),  # floor volatility triggers max clamp
        (10.0, 0.1),  # high vol clamps to min allocation
        (0.2, 0.5),  # mid vol stays within bounds
    ],
)
def test_target_allocation_applies_floor_and_clamps(realized: float, expected: float) -> None:
    settings = VolatilityTargetSettings(
        target_volatility=0.1,
        lookback=5,
        min_allocation=0.1,
        max_allocation=1.0,
        floor_volatility=0.05,
    )
    strategy = VolatilityTargetStrategy(settings)

    assert strategy._target_allocation(realized) == pytest.approx(expected)


def test_should_rebalance_threshold_and_positive_target() -> None:
    settings = VolatilityTargetSettings(rebalance_threshold=0.1)
    strategy = VolatilityTargetStrategy(settings)

    assert strategy._should_rebalance(0.0, 1.0) is False
    assert strategy._should_rebalance(1.0, 0.05) is False
    assert strategy._should_rebalance(1.0, 0.1) is True
    assert strategy._should_rebalance(1.0, -0.1) is True
