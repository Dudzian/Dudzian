"""Implementacja strategii Dollar-Cost Averaging (DCA)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Mapping, MutableMapping, Sequence

from .base import MarketSnapshot, StrategyEngine, StrategySignal


@dataclass(slots=True)
class DollarCostAveragingSettings:
    """Parametry strategii DCA.

    Strategia działa w oparciu o regularne zakupy aktywa w zadanych
    odstępach czasu, z opcjonalnym zwiększaniem pozycji w przypadku
    silniejszych spadków.
    """

    cadence_days: int = 7
    max_allocation: float = 1.0
    drawdown_acceleration: float = 0.25
    min_drawdown: float = 0.02
    max_drawdown: float = 0.25


@dataclass(slots=True)
class _DCAState:
    last_purchase: datetime | None = None
    reference_price: float | None = None
    cumulative_allocation: float = 0.0


class DollarCostAveragingStrategy(StrategyEngine):
    """Prosty silnik DCA uwzględniający spadki i limity ekspozycji."""

    def __init__(self, settings: DollarCostAveragingSettings | None = None) -> None:
        self._settings = settings or DollarCostAveragingSettings()
        self._states: MutableMapping[str, _DCAState] = {}

    def warm_up(self, history: Sequence[MarketSnapshot]) -> None:
        for snapshot in history:
            state = self._ensure_state(snapshot.symbol)
            state.reference_price = snapshot.close

    def on_data(self, snapshot: MarketSnapshot) -> Sequence[StrategySignal]:
        state = self._ensure_state(snapshot.symbol)
        state.reference_price = state.reference_price or snapshot.close

        cadence = max(1, self._settings.cadence_days)
        next_due: datetime | None = None
        if state.last_purchase is not None:
            next_due = state.last_purchase + timedelta(days=cadence)

        now = datetime.fromtimestamp(snapshot.timestamp / 1000.0, tz=timezone.utc)
        due = next_due is None or now >= next_due

        if not due and snapshot.close >= state.reference_price:
            return []

        drawdown = (state.reference_price - snapshot.close) / max(1e-9, state.reference_price)
        drawdown = max(0.0, drawdown)

        if not due and drawdown < self._settings.min_drawdown:
            return []

        acceleration = 1.0 + min(self._settings.max_drawdown, drawdown) * self._settings.drawdown_acceleration
        allocation = min(self._settings.max_allocation - state.cumulative_allocation, acceleration * 0.1)
        if allocation <= 0.0:
            return []

        state.last_purchase = now
        state.cumulative_allocation += allocation
        state.reference_price = snapshot.close

        metadata: Mapping[str, float] = {
            "drawdown": float(drawdown),
            "allocation_used": float(state.cumulative_allocation),
            "allocation_step": float(allocation),
            "cadence_days": float(cadence),
        }

        signal = StrategySignal(
            symbol=snapshot.symbol,
            side="buy",
            confidence=min(1.0, 0.5 + drawdown),
            metadata=metadata,
        )
        return [signal]

    def _ensure_state(self, symbol: str) -> _DCAState:
        if symbol not in self._states:
            self._states[symbol] = _DCAState()
        return self._states[symbol]


__all__ = ["DollarCostAveragingSettings", "DollarCostAveragingStrategy"]

