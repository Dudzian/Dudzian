"""Profil konserwatywny zgodny z wymaganiami klienta."""
from __future__ import annotations

from dataclasses import dataclass

from bot_core.risk.base import RiskProfile


@dataclass(slots=True)
class ConservativeProfile(RiskProfile):
    """Profil o niskiej tolerancji ryzyka."""

    name: str = "conservative"
    _max_positions: int = 3
    _max_leverage: float = 2.0
    _drawdown_limit: float = 0.05
    _daily_loss_limit: float = 0.01
    _max_position_pct: float = 0.03
    _target_volatility: float = 0.07
    _stop_loss_atr_multiple: float = 1.0

    def max_positions(self) -> int:
        return self._max_positions

    def max_leverage(self) -> float:
        return self._max_leverage

    def drawdown_limit(self) -> float:
        return self._drawdown_limit

    def daily_loss_limit(self) -> float:
        return self._daily_loss_limit

    def max_position_exposure(self) -> float:
        return self._max_position_pct

    def target_volatility(self) -> float:
        return self._target_volatility

    def stop_loss_atr_multiple(self) -> float:
        return self._stop_loss_atr_multiple


__all__ = ["ConservativeProfile"]
