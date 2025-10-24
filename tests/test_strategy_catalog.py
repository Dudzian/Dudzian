"""Testy dla katalogu strategii pluginowych TradingParameters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot_core.strategies.catalog import DEFAULT_STRATEGY_CATALOG
from bot_core.trading.engine import TechnicalIndicators, TradingParameters
from bot_core.trading.strategies import (
    ArbitrageStrategy,
    DayTradingStrategy,
    GridTradingStrategy,
    MeanReversionStrategy,
    OptionsIncomeStrategy,
    ScalpingStrategy,
    StatisticalArbitrageStrategy,
    StrategyCatalog,
    StrategyPlugin,
    TrendFollowingStrategy,
    VolatilityTargetStrategy,
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


def _dummy_market_data(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "implied_volatility": pd.Series(0.25, index=index),
            "realized_volatility": pd.Series(0.18, index=index),
        }
    )


def test_default_catalog_contains_builtin_strategies() -> None:
    catalog = StrategyCatalog.default()
    assert catalog.available() == (
        "arbitrage",
        "day_trading",
        "grid_trading",
        "mean_reversion",
        "options_income",
        "scalping",
        "statistical_arbitrage",
        "trend_following",
        "volatility_target",
    )


def test_describe_exposes_metadata() -> None:
    catalog = StrategyCatalog.default()
    descriptions = {entry["name"]: entry["description"] for entry in catalog.describe()}
    assert descriptions["trend_following"].startswith("EMA")
    assert descriptions["arbitrage"]
    assert "volatility" in descriptions["volatility_target"].lower()


def test_plugin_metadata_matches_strategy_catalog() -> None:
    catalog = StrategyCatalog.default()
    mapping = {
        "trend_following": "daily_trend_momentum",
        "day_trading": "day_trading",
        "mean_reversion": "mean_reversion",
        "arbitrage": "cross_exchange_arbitrage",
        "grid_trading": "grid_trading",
        "volatility_target": "volatility_target",
        "scalping": "scalping",
        "options_income": "options_income",
        "statistical_arbitrage": "statistical_arbitrage",
    }

    for plugin_name, engine in mapping.items():
        plugin_meta = catalog.metadata_for(plugin_name)
        spec = DEFAULT_STRATEGY_CATALOG.get(engine)
        assert plugin_meta["license_tier"] == spec.license_tier
        assert tuple(plugin_meta["risk_classes"]) == tuple(spec.risk_classes)
        assert tuple(plugin_meta["required_data"]) == tuple(spec.required_data)
        if spec.capability:
            assert plugin_meta["capability"] == spec.capability
        if spec.default_tags:
            assert tuple(plugin_meta["tags"]) == tuple(spec.default_tags)


def test_plugins_generate_series_with_matching_index() -> None:
    indicators = _dummy_indicators()
    params = TradingParameters()
    market_data = _dummy_market_data(indicators.rsi.index)

    for plugin_cls in (
        TrendFollowingStrategy,
        DayTradingStrategy,
        MeanReversionStrategy,
        ArbitrageStrategy,
        GridTradingStrategy,
        VolatilityTargetStrategy,
        ScalpingStrategy,
        OptionsIncomeStrategy,
        StatisticalArbitrageStrategy,
    ):
        plugin = plugin_cls()
        signal = plugin.generate(indicators, params, market_data=market_data)
        assert list(signal.index) == list(indicators.rsi.index)
        assert ((signal >= -1.000001) & (signal <= 1.000001)).all()


def test_plugin_requires_known_engine_key() -> None:
    with pytest.raises(ValueError):

        class _UnknownEngineStrategy(StrategyPlugin):
            engine_key = "missing_engine"
            name = "unknown_engine"

            def generate(
                self,
                indicators: TechnicalIndicators,
                params: TradingParameters,
                *,
                market_data: pd.DataFrame | None = None,
            ) -> pd.Series:
                return indicators.rsi.reindex(indicators.rsi.index).clip(-1.0, 1.0)


def test_plugin_rejects_duplicate_engine_key() -> None:
    with pytest.raises(ValueError):

        class _DuplicateDayTradingStrategy(StrategyPlugin):
            engine_key = "day_trading"
            name = "duplicate_day_trading"

            def generate(
                self,
                indicators: TechnicalIndicators,
                params: TradingParameters,
                *,
                market_data: pd.DataFrame | None = None,
            ) -> pd.Series:
                return indicators.ema_fast.reindex(indicators.ema_fast.index).clip(-1.0, 1.0)
