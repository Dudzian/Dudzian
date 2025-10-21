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
    ) -> None:
        self.name = name
        self._max_positions = max_positions
        self._max_leverage = max_leverage
        self._drawdown_limit = drawdown_limit
        self._daily_loss_limit = daily_loss_limit
        self._max_position_pct = max_position_pct
        self._target_volatility = target_volatility
        self._stop_loss_atr_multiple = stop_loss_atr_multiple

__all__ = ["ManualProfile"]
