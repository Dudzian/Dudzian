"""Profil agresywny zgodny z wymaganiami klienta."""
from __future__ import annotations

from dataclasses import dataclass

from bot_core.risk.base import StaticRiskProfile


@dataclass(slots=True)
class AggressiveProfile(StaticRiskProfile):
    """Profil o wysokiej tolerancji ryzyka."""

    name: str = "aggressive"
    _max_positions: int = 10
    _max_leverage: float = 5.0
    _drawdown_limit: float = 0.20
    _daily_loss_limit: float = 0.03
    _max_position_pct: float = 0.10
    _target_volatility: float = 0.19
    _stop_loss_atr_multiple: float = 2.0

__all__ = ["AggressiveProfile"]
