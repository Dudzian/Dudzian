"""Auto-trading helpers exposed by the bot_core namespace."""
from __future__ import annotations

from .app import AutoTrader, EmitterLike, RiskDecision

__all__ = ["AutoTrader", "EmitterLike", "RiskDecision"]
