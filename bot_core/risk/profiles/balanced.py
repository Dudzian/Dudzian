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
    min_sortino_ratio: float = 1.2
    min_omega_ratio: float = 1.1
    max_risk_of_ruin_pct: float = 7.5
    min_hit_ratio_pct: float = 50.0

__all__ = ["BalancedProfile"]
