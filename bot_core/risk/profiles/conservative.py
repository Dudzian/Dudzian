"""Profil konserwatywny zgodny z wymaganiami klienta."""
from __future__ import annotations

from dataclasses import dataclass

from bot_core.risk.base import StaticRiskProfile


@dataclass(slots=True)
class ConservativeProfile(StaticRiskProfile):
    """Profil o niskiej tolerancji ryzyka."""

    name: str = "conservative"
    _max_positions: int = 3
    _max_leverage: float = 2.0
    _drawdown_limit: float = 0.05
    _daily_loss_limit: float = 0.01
    _max_position_pct: float = 0.03
    _target_volatility: float = 0.07
    _stop_loss_atr_multiple: float = 1.0

__all__ = ["ConservativeProfile"]
