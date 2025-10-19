"""Warstwa zgodności delegująca do natywnego modułu ``bot_core.backtest.ma``."""
from __future__ import annotations

from bot_core.backtest.ma import (
    Bar,
    Trade,
    param_grid_fast_slow,
    simulate_trades_ma,
)

__all__ = ["Bar", "Trade", "param_grid_fast_slow", "simulate_trades_ma"]
