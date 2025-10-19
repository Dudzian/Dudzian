"""Warstwa kompatybilnościowa delegująca do ``bot_core.trading.auto_trade``."""
from __future__ import annotations

from bot_core.trading.auto_trade import AutoTradeConfig, AutoTradeEngine

__all__ = ["AutoTradeConfig", "AutoTradeEngine"]
