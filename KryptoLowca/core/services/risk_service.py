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

    def __init__(
        self,
        *,
        max_position_notional_pct: float = 0.02,
        max_daily_loss_pct: float = 0.05,
        max_portfolio_risk_pct: float | None = None,
        max_positions: int | None = None,
        emergency_stop_drawdown_pct: float | None = None,
    ) -> None:
        self.max_position_notional_pct = float(max_position_notional_pct)
        self.max_daily_loss_pct = float(max_daily_loss_pct)
        self.max_portfolio_risk_pct = (
            float(max_portfolio_risk_pct)
            if max_portfolio_risk_pct is not None and max_portfolio_risk_pct > 0
            else None
        )
        self.max_positions = int(max_positions) if max_positions else None
        self.emergency_stop_drawdown_pct = (
            float(emergency_stop_drawdown_pct)
            if emergency_stop_drawdown_pct is not None and emergency_stop_drawdown_pct > 0
            else None
        )

    def assess(self, signal: StrategySignal, context: StrategyContext, market_state: Mapping[str, float]) -> RiskAssessment:
        if signal is None:
            return RiskAssessment(allow=False, reason="Brak sygnału")

        portfolio_value = float(context.portfolio_value or market_state.get("portfolio_value") or 0.0)
        if portfolio_value <= 0:
            portfolio_value = 1.0

        notional_limit = portfolio_value * self.max_position_notional_pct
        proposed_size = float(signal.size or notional_limit)

        increase_exposure = self._increases_exposure(signal.action, context.position)

        if self.max_positions is not None and increase_exposure:
            current_positions = self._coerce_int(
                market_state.get("open_positions"),
                default=1 if context.position else 0,
            )
            if current_positions >= self.max_positions:
                logger.warning(
                    "Przekroczony limit liczby pozycji: %s >= %s", current_positions, self.max_positions
                )
                return RiskAssessment(allow=False, reason="Przekroczony limit liczby pozycji")

        if self.max_portfolio_risk_pct is not None and increase_exposure:
            current_exposure = self._coerce_float(
                market_state,
                ("portfolio_exposure_pct", "portfolio_risk_pct", "position_notional_pct"),
                default=0.0,
            )
            current_exposure = abs(current_exposure)
            additional_fraction = proposed_size / portfolio_value if portfolio_value else 0.0
            if current_exposure + additional_fraction > self.max_portfolio_risk_pct:
                available_fraction = self.max_portfolio_risk_pct - current_exposure
                if available_fraction <= 0:
                    logger.warning(
                        "Profil ryzyka blokuje handel: exposure %.4f przekracza limit %.4f",
                        current_exposure,
                        self.max_portfolio_risk_pct,
                    )
                    return RiskAssessment(allow=False, reason="Przekroczony łączny limit ekspozycji")
                proposed_size = min(proposed_size, max(0.0, available_fraction) * portfolio_value)

        if proposed_size > notional_limit:
            logger.warning(
                "Rozmiar pozycji %s przekracza limit %.2f (portfolio %.2f)",
                proposed_size,
                notional_limit,
                portfolio_value,
            )
            return RiskAssessment(allow=False, reason="Przekroczony limit pozycji")

        if self.emergency_stop_drawdown_pct is not None:
            drawdown = self._coerce_float(
                market_state,
                ("drawdown_pct", "max_drawdown_pct", "drawdown"),
                default=0.0,
            )
            drawdown = abs(drawdown)
            if drawdown >= self.emergency_stop_drawdown_pct:
                logger.warning(
                    "Aktywny awaryjny stop drawdown: %.4f >= %.4f",
                    drawdown,
                    self.emergency_stop_drawdown_pct,
                )
                return RiskAssessment(
                    allow=False,
                    reason="Osiągnięto limit awaryjnego drawdownu",
                )

        if proposed_size <= 0:
            return RiskAssessment(allow=False, reason="Brak dostępnego budżetu ryzyka")

        if signal.action == "HOLD":
            return RiskAssessment(allow=False, reason="Brak działania")

        last_loss_pct = float(market_state.get("daily_loss_pct", 0.0))
        if last_loss_pct <= -abs(self.max_daily_loss_pct):
            return RiskAssessment(allow=False, reason="Przekroczony dzienny limit strat")

        return RiskAssessment(
            allow=True,
            reason="OK",
            size=max(0.0, proposed_size),
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )

    @staticmethod
    def _increases_exposure(action: Optional[str], position: float) -> bool:
        if not action:
            return False
        if action.upper() == "BUY":
            return position >= 0
        if action.upper() == "SELL":
            return position <= 0
        return False

    @staticmethod
    def _coerce_float(
        payload: Mapping[str, object], keys: tuple[str, ...], *, default: float
    ) -> float:
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            if isinstance(value, (str, int, float)):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
            else:
                continue
        return float(default)

    @staticmethod
    def _coerce_int(value: object, *, default: int) -> int:
        if value is None:
            return default
        if not isinstance(value, (str, int, float)):
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default


__all__ = ["RiskService", "RiskAssessment"]
