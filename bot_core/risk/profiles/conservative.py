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
    _daily_loss_limit: float = 0.04
    _max_position_pct: float = 0.30
    _target_volatility: float = 0.07
    _stop_loss_atr_multiple: float = 1.0
    _trade_risk_pct_range: tuple[float, float] = (0.005, 0.0075)
    _instrument_alert_pct: float = 0.25
    _instrument_limit_pct: float = 0.30
    _portfolio_alert_pct: float = 0.50
    _portfolio_limit_pct: float = 0.60
    _daily_kill_switch_r_multiple: float = 2.0
    _daily_kill_switch_loss_pct: float = 0.04
    _weekly_kill_switch_loss_pct: float = 0.06
    _max_cost_to_profit_ratio: float = 0.25
    min_sortino_ratio: float = 1.6
    min_omega_ratio: float = 1.3
    max_risk_of_ruin_pct: float = 4.0
    min_hit_ratio_pct: float = 55.0

__all__ = ["ConservativeProfile"]
