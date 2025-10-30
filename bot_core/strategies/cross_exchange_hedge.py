"""Cross-exchange hedging engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .base import MarketSnapshot, StrategyEngine, StrategySignal


@dataclass(slots=True)
class CrossExchangeHedgeSettings:
    """Parametry hedgingu pomiędzy rynkiem kasowym i pochodnym."""

    basis_scale: float = 0.01
    inventory_scale: float = 0.35
    latency_limit_ms: float = 180.0
    max_hedge_ratio: float = 0.9


@dataclass(slots=True)
class _HedgeState:
    last_ratio: float = 0.0


class CrossExchangeHedgeStrategy(StrategyEngine):
    """Generuje docelowy hedge ratio w zależności od basisu i inventory."""

    def __init__(self, settings: CrossExchangeHedgeSettings | None = None) -> None:
        self._settings = settings or CrossExchangeHedgeSettings()
        self._states: Dict[str, _HedgeState] = {}

    def warm_up(self, history: Sequence[MarketSnapshot]) -> None:
        for snapshot in history:
            self._ensure_state(snapshot.symbol)

    def on_data(self, snapshot: MarketSnapshot) -> Sequence[StrategySignal]:
        state = self._ensure_state(snapshot.symbol)
        basis = float(snapshot.indicators.get("spot_basis", snapshot.indicators.get("basis", 0.0)))
        inventory = float(snapshot.indicators.get("inventory_skew", snapshot.indicators.get("inventory", 0.0)))
        latency_ms = float(snapshot.indicators.get("latency_ms", 0.0))

        hedge_ratio = self._target_ratio(basis, inventory)
        latency_penalty = min(1.0, max(0.0, latency_ms / max(self._settings.latency_limit_ms, 1.0)))
        hedge_ratio *= 1.0 - 0.5 * latency_penalty

        hedge_ratio = max(-self._settings.max_hedge_ratio, min(self._settings.max_hedge_ratio, hedge_ratio))
        hedge_ratio = round(hedge_ratio, 4)

        state.last_ratio = hedge_ratio

        signal = StrategySignal(
            symbol=snapshot.symbol,
            side="rebalance_delta",
            confidence=min(1.0, abs(hedge_ratio)),
            metadata={
                "target_ratio": hedge_ratio,
                "basis": basis,
                "inventory": inventory,
                "latency_ms": latency_ms,
            },
        )
        return [signal]

    def _ensure_state(self, symbol: str) -> _HedgeState:
        state = self._states.get(symbol)
        if state is None:
            state = _HedgeState()
            self._states[symbol] = state
        return state

    def _target_ratio(self, basis: float, inventory: float) -> float:
        basis_component = max(-1.0, min(1.0, basis / max(self._settings.basis_scale, 1e-6)))
        inventory_component = max(-1.0, min(1.0, inventory / max(self._settings.inventory_scale, 1e-6)))
        return (basis_component - inventory_component) / 2.0


__all__ = ["CrossExchangeHedgeSettings", "CrossExchangeHedgeStrategy"]
