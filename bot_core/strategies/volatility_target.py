"""Strategia kontroli zmienności portfela."""
from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Deque, Dict, List, Sequence

from collections import deque

from bot_core.strategies.base import MarketSnapshot, StrategyEngine, StrategySignal
from bot_core.strategies._volatility import realized_volatility


@dataclass(slots=True)
class VolatilityTargetSettings:
    """Ustawienia strategii dostosowującej ekspozycję do zmienności."""

    target_volatility: float = 0.1
    lookback: int = 60
    rebalance_threshold: float = 0.1
    min_allocation: float = 0.1
    max_allocation: float = 1.0
    floor_volatility: float = 0.02

    def history_size(self) -> int:
        return max(self.lookback, 2) + 2


@dataclass(slots=True)
class _SymbolState:
    returns: Deque[float]
    last_price: float | None = None
    allocation: float = 0.0


class VolatilityTargetStrategy(StrategyEngine):
    """Utrzymuje celowaną zmienność portfela poprzez dynamiczne wagi."""

    def __init__(self, settings: VolatilityTargetSettings | None = None) -> None:
        self._settings = settings or VolatilityTargetSettings()
        self._states: Dict[str, _SymbolState] = {}

    def warm_up(self, history: Sequence[MarketSnapshot]) -> None:
        for snapshot in history:
            state = self._state_for(snapshot.symbol)
            self._update_state(state, snapshot)

    def on_data(self, snapshot: MarketSnapshot) -> Sequence[StrategySignal]:
        state = self._state_for(snapshot.symbol)
        self._update_state(state, snapshot)

        if len(state.returns) < self._settings.lookback:
            return []

        realized_vol = self._realized_volatility(state)
        target = self._target_allocation(realized_vol)
        diff = target - state.allocation
        if not self._should_rebalance(target, diff):
            return []

        state.allocation = target
        confidence = min(1.0, abs(diff) / max(target, 1e-6))
        signal = StrategySignal(
            symbol=snapshot.symbol,
            side="rebalance",
            confidence=confidence,
            metadata={
                "target_allocation": target,
                "current_allocation": target - diff,
                "realized_volatility": realized_vol,
                "target_volatility": self._settings.target_volatility,
            },
        )
        return [signal]

    def _state_for(self, symbol: str) -> _SymbolState:
        if symbol not in self._states:
            window = self._settings.history_size()
            self._states[symbol] = _SymbolState(returns=deque(maxlen=window))
        return self._states[symbol]

    def _update_state(self, state: _SymbolState, snapshot: MarketSnapshot) -> None:
        if state.last_price and state.last_price > 0 and snapshot.close > 0:
            state.returns.append(log(snapshot.close / state.last_price))
        state.last_price = snapshot.close

    def _realized_volatility(self, state: _SymbolState) -> float:
        return realized_volatility(state.returns, lookback=self._settings.lookback)

    def _target_allocation(self, realized_vol: float) -> float:
        effective_vol = max(realized_vol, self._settings.floor_volatility)
        raw = self._settings.target_volatility / effective_vol if effective_vol else self._settings.max_allocation
        return min(self._settings.max_allocation, max(self._settings.min_allocation, raw))

    def _should_rebalance(self, target: float, diff: float) -> bool:
        if target <= 0:
            return False
        relative = abs(diff) / target
        return relative >= self._settings.rebalance_threshold


__all__ = ["VolatilityTargetSettings", "VolatilityTargetStrategy"]
