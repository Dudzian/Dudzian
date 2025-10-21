"""Testy katalogu strategii Multi-Strategy."""
from __future__ import annotations

import pytest

from bot_core.strategies.base import StrategyEngine
from bot_core.strategies.catalog import DEFAULT_STRATEGY_CATALOG, StrategyDefinition


def test_catalog_builds_trend_strategy() -> None:
    definition = StrategyDefinition(
        name="demo_trend",
        engine="daily_trend_momentum",
        parameters={
            "fast_ma": 5,
            "slow_ma": 20,
            "breakout_lookback": 10,
            "momentum_window": 10,
            "atr_window": 14,
            "atr_multiplier": 1.5,
        },
    )
    engine = DEFAULT_STRATEGY_CATALOG.create(definition)
    assert isinstance(engine, StrategyEngine)


def test_catalog_unknown_engine() -> None:
    definition = StrategyDefinition(name="unknown", engine="non_existing", parameters={})
    with pytest.raises(KeyError):
        DEFAULT_STRATEGY_CATALOG.create(definition)
