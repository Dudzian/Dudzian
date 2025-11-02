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
    _daily_loss_limit: float = 0.04
    _max_position_pct: float = 0.30
    _target_volatility: float = 0.19
    _stop_loss_atr_multiple: float = 2.0
    _trade_risk_pct_range: tuple[float, float] = (0.0075, 0.01)
    _instrument_alert_pct: float = 0.27
    _instrument_limit_pct: float = 0.30
    _portfolio_alert_pct: float = 0.60
    _portfolio_limit_pct: float = 0.70
    _daily_kill_switch_r_multiple: float = 2.0
    _daily_kill_switch_loss_pct: float = 0.04
    _weekly_kill_switch_loss_pct: float = 0.08
    _max_cost_to_profit_ratio: float = 0.25
    min_sortino_ratio: float = 0.85
    min_omega_ratio: float = 1.0
    max_risk_of_ruin_pct: float = 12.0
    min_hit_ratio_pct: float = 45.0

__all__ = ["AggressiveProfile"]
