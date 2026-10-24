"""Day trading abstains when a volatility estimate is unavailable."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from bot_core.trading.engine import TechnicalIndicators, TradingParameters
from bot_core.trading.strategies import DayTradingStrategy


def _indicators(atr: list[float]) -> TechnicalIndicators:
    index = pd.RangeIndex(len(atr))
    price = pd.Series(np.linspace(100.0, 110.0, len(atr)), index=index)
    return TechnicalIndicators(
        rsi=price,
        ema_fast=price,
        ema_slow=price,
        sma_trend=price,
        atr=pd.Series(atr, index=index, dtype=float),
        bollinger_upper=price,
        bollinger_lower=price,
        bollinger_middle=price / 1.5,
        macd=price,
        macd_signal=price,
        stochastic_k=price,
        stochastic_d=price,
    )


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("atr", [[], [float("nan")] * 4, [0.0] * 4, [float("inf")] * 4, [-1.0] * 4])
def test_day_trading_abstains_without_positive_finite_volatility(atr: list[float]) -> None:
    indicators = _indicators(atr)

    signal = DayTradingStrategy().generate(indicators, TradingParameters())

    assert signal.index.equals(indicators.ema_fast.index)
    assert np.isfinite(signal).all()
    assert (signal == 0.0).all()


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_day_trading_warmup_then_available_volatility() -> None:
    indicators = _indicators([float("nan")] * 3 + [1.0] * 5)

    signal = DayTradingStrategy().generate(indicators, TradingParameters())

    assert (signal.iloc[:3] == 0.0).all()
    assert (signal.iloc[3:] > 0.0).all()
    assert np.isfinite(signal).all()


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0, 0.0])
def test_day_trading_does_not_reuse_prior_volatility_for_invalid_latest_bar(invalid: float) -> None:
    indicators = _indicators([1.0] * 5 + [invalid])

    signal = DayTradingStrategy().generate(indicators, TradingParameters())

    assert signal.iloc[-2] > 0.0
    assert signal.iloc[-1] == 0.0
    assert np.isfinite(signal).all()


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_day_trading_retains_bias_with_valid_volatility() -> None:
    indicators = _indicators([1.0] * 5)
    price = pd.Series(100.0, index=indicators.ema_fast.index)
    indicators = replace(indicators, ema_fast=price, bollinger_middle=price / 1.5)

    signal = DayTradingStrategy().generate(indicators, TradingParameters())

    assert signal.to_list() == pytest.approx([0.1] * 5)
