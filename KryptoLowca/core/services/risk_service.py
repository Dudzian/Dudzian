"""Moduł zarządzania ryzykiem – walidacja sygnałów przed wykonaniem."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from KryptoLowca.logging_utils import get_logger
from KryptoLowca.strategies.base import StrategyContext, StrategySignal

logger = get_logger(__name__)


@dataclass(slots=True)
class RiskAssessment:
    allow: bool
    reason: str
    size: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class RiskService:
    """Lekka warstwa logiki biznesowej."""

    def __init__(self, *, max_position_notional_pct: float = 0.02, max_daily_loss_pct: float = 0.05) -> None:
        self.max_position_notional_pct = max_position_notional_pct
        self.max_daily_loss_pct = max_daily_loss_pct

    def assess(self, signal: StrategySignal, context: StrategyContext, market_state: Mapping[str, float]) -> RiskAssessment:
        if signal is None:
            return RiskAssessment(allow=False, reason="Brak sygnału")

        notional_limit = context.portfolio_value * self.max_position_notional_pct
        proposed_size = signal.size or notional_limit
        if proposed_size > notional_limit:
            logger.warning(
                "Rozmiar pozycji %s przekracza limit %.2f (portfolio %.2f)",
                proposed_size,
                notional_limit,
                context.portfolio_value,
            )
            return RiskAssessment(allow=False, reason="Przekroczony limit pozycji")

        if signal.action == "HOLD":
            return RiskAssessment(allow=False, reason="Brak działania")

        last_loss_pct = float(market_state.get("daily_loss_pct", 0.0))
        if last_loss_pct <= -abs(self.max_daily_loss_pct):
            return RiskAssessment(allow=False, reason="Przekroczony dzienny limit strat")

        return RiskAssessment(
            allow=True,
            reason="OK",
            size=proposed_size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )


__all__ = ["RiskService", "RiskAssessment"]
