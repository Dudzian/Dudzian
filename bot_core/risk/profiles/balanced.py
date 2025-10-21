"""Profil zbalansowany będący ustawieniem domyślnym."""
from __future__ import annotations

from dataclasses import dataclass

from bot_core.risk.base import StaticRiskProfile


@dataclass(slots=True)
class BalancedProfile(StaticRiskProfile):
    """Profil o średniej tolerancji ryzyka."""

    name: str = "balanced"
    _max_positions: int = 5
    _max_leverage: float = 3.0
    _drawdown_limit: float = 0.10
    _daily_loss_limit: float = 0.015
    _max_position_pct: float = 0.05
    _target_volatility: float = 0.11
    _stop_loss_atr_multiple: float = 1.5

__all__ = ["BalancedProfile"]
