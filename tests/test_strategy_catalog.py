"""Testy dla katalogu strategii pluginowych TradingParameters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bot_core.trading.engine import TechnicalIndicators, TradingParameters
from bot_core.trading.strategies import (
    ArbitrageStrategy,
    DayTradingStrategy,
    MeanReversionStrategy,
    StrategyCatalog,
    TrendFollowingStrategy,
)


def _dummy_indicators(rows: int = 16) -> TechnicalIndicators:
    index = pd.date_range("2024-01-01", periods=rows, freq="h")
    base = pd.Series(np.linspace(100.0, 102.0, rows), index=index)
    spreads = pd.Series(np.linspace(-1.0, 1.0, rows), index=index)
    atr = pd.Series(np.linspace(0.5, 1.5, rows), index=index)

    return TechnicalIndicators(
        rsi=pd.Series(np.linspace(20, 80, rows), index=index),
        ema_fast=base,
        ema_slow=base.rolling(window=5, min_periods=1).mean(),
        sma_trend=base.rolling(window=8, min_periods=1).mean(),
        atr=atr,
        bollinger_upper=base + 1,
        bollinger_lower=base - 1,
        bollinger_middle=base,
        macd=spreads,
        macd_signal=spreads.rolling(window=3, min_periods=1).mean(),
        stochastic_k=pd.Series(np.linspace(10, 90, rows), index=index),
        stochastic_d=pd.Series(np.linspace(15, 70, rows), index=index),
    )


def test_default_catalog_contains_builtin_strategies() -> None:
    catalog = StrategyCatalog.default()
    assert catalog.available() == (
        "arbitrage",
        "day_trading",
        "mean_reversion",
        "trend_following",
    )


def test_describe_exposes_metadata() -> None:
    catalog = StrategyCatalog.default()
    descriptions = {entry["name"]: entry["description"] for entry in catalog.describe()}
    assert descriptions["trend_following"].startswith("EMA")
    assert descriptions["arbitrage"]


def test_plugins_generate_series_with_matching_index() -> None:
    indicators = _dummy_indicators()
    params = TradingParameters()

    for plugin_cls in (
        TrendFollowingStrategy,
        DayTradingStrategy,
        MeanReversionStrategy,
        ArbitrageStrategy,
    ):
        plugin = plugin_cls()
        signal = plugin.generate(indicators, params)
        assert list(signal.index) == list(indicators.rsi.index)
        assert ((signal >= -1.000001) & (signal <= 1.000001)).all()
