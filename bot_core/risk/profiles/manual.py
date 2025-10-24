"""Profil ręczny pozwalający na własne limity."""
from __future__ import annotations

from dataclasses import dataclass

from bot_core.risk.base import StaticRiskProfile


@dataclass(slots=True, init=False)
class ManualProfile(StaticRiskProfile):
    """Profil konfigurowany przez użytkownika."""

    name: str
    _max_positions: int
    _max_leverage: float
    _drawdown_limit: float
    _daily_loss_limit: float
    _max_position_pct: float
    _target_volatility: float
    _stop_loss_atr_multiple: float
    min_sortino_ratio: float | None
    min_omega_ratio: float | None
    max_risk_of_ruin_pct: float | None
    min_hit_ratio_pct: float | None

    def __init__(
        self,
        *,
        name: str = "manual",
        max_positions: int,
        max_leverage: float,
        drawdown_limit: float,
        daily_loss_limit: float,
        max_position_pct: float,
        target_volatility: float,
        stop_loss_atr_multiple: float,
        min_sortino_ratio: float | None = None,
        min_omega_ratio: float | None = None,
        max_risk_of_ruin_pct: float | None = None,
        min_hit_ratio_pct: float | None = None,
    ) -> None:
        self.name = name
        self._max_positions = max_positions
        self._max_leverage = max_leverage
        self._drawdown_limit = drawdown_limit
        self._daily_loss_limit = daily_loss_limit
        self._max_position_pct = max_position_pct
        self._target_volatility = target_volatility
        self._stop_loss_atr_multiple = stop_loss_atr_multiple
        self.min_sortino_ratio = min_sortino_ratio
        self.min_omega_ratio = min_omega_ratio
        self.max_risk_of_ruin_pct = max_risk_of_ruin_pct
        self.min_hit_ratio_pct = min_hit_ratio_pct

__all__ = ["ManualProfile"]
