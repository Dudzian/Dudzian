"""Compatibility accessors for backtesting components under the bot_core namespace."""
from __future__ import annotations

from .simulation import BacktestFill, MatchingConfig, MatchingEngine

__all__ = ["BacktestFill", "MatchingConfig", "MatchingEngine"]
