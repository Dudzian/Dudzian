"""Strategia grid trading zarządzająca ekspozycją wokół ceny bazowej."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

from bot_core.strategies.base import MarketSnapshot, StrategyEngine, StrategySignal


@dataclass(slots=True)
class GridTradingSettings:
    """Parametry strategii grid trading."""

    grid_size: int = 5
    grid_spacing: float = 0.004
    rebalance_threshold: float = 0.001
    max_inventory: float = 1.0


@dataclass(slots=True)
class _GridState:
    base_price: float | None = None
    inventory: float = 0.0
    last_level: int | None = None


class GridTradingStrategy(StrategyEngine):
    """Buduje siatkę zleceń i reaguje na przekroczenia poziomów cenowych."""

    def __init__(self, settings: GridTradingSettings | None = None) -> None:
        self._settings = settings or GridTradingSettings()
        self._states: Dict[str, _GridState] = {}

    def warm_up(self, history: Sequence[MarketSnapshot]) -> None:
        for snapshot in history:
            state = self._ensure_state(snapshot.symbol)
            if state.base_price is None and snapshot.close > 0:
                state.base_price = snapshot.close
                state.last_level = 0

    def on_data(self, snapshot: MarketSnapshot) -> Sequence[StrategySignal]:
        state = self._ensure_state(snapshot.symbol)
        if state.base_price is None:
            if snapshot.close > 0:
                state.base_price = snapshot.close
                state.last_level = 0
            return []

        normalized_move = (snapshot.close - state.base_price) / state.base_price
        current_level = int(math.floor(normalized_move / self._settings.grid_spacing))
        current_level = max(-self._settings.grid_size, min(self._settings.grid_size, current_level))

        signals: List[StrategySignal] = []
        if state.last_level is None:
            state.last_level = current_level
            return signals

        if current_level == state.last_level:
            return signals

        direction = "buy" if current_level < state.last_level else "sell"
        trade_size = max(0.0, min(self._settings.max_inventory, abs(current_level - state.last_level) * 0.1))
        inventory_change = trade_size if direction == "buy" else -trade_size
        new_inventory = state.inventory + inventory_change

        # rebalance when inventory drifts too much
        if abs(new_inventory) > self._settings.max_inventory:
            direction = "sell" if new_inventory > 0 else "buy"
            trade_size = abs(new_inventory) - self._settings.max_inventory
            inventory_change = trade_size if direction == "buy" else -trade_size
            new_inventory = state.inventory + inventory_change

        confidence = min(1.0, abs(normalized_move) / (self._settings.grid_spacing * self._settings.grid_size))
        metadata = {
            "strategy": {
                "type": "grid",
                "profile": "grid_balanced",
                "risk_label": "moderate",
            },
            "normalized_move": normalized_move,
            "grid_level": current_level,
            "previous_level": state.last_level,
            "trade_size": trade_size,
        }

        state.inventory = new_inventory
        state.last_level = current_level
        signals.append(
            StrategySignal(
                symbol=snapshot.symbol,
                side=direction,
                confidence=confidence,
                metadata=metadata,
            )
        )
        return signals

    def _ensure_state(self, symbol: str) -> _GridState:
        if symbol not in self._states:
            self._states[symbol] = _GridState()
        return self._states[symbol]


__all__ = ["GridTradingSettings", "GridTradingStrategy"]
