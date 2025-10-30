"""Lekki silnik market-making dla katalogu strategii."""
from __future__ import annotations

from dataclasses import dataclass
from typing import MutableMapping, Sequence

from .base import MarketSnapshot, StrategyEngine, StrategySignal


@dataclass(slots=True)
class MarketMakingSettings:
    """Parametry strategii market-making."""

    spread_bps: float = 12.0
    inventory_target: float = 0.0
    max_inventory: float = 5.0
    rebalance_threshold: float = 0.6


@dataclass(slots=True)
class _InventoryState:
    inventory: float = 0.0
    anchor_price: float | None = None


class MarketMakingStrategy(StrategyEngine):
    """Strategia równoważąca zlecenia bid/ask względem zapasu."""

    def __init__(self, settings: MarketMakingSettings | None = None) -> None:
        self._settings = settings or MarketMakingSettings()
        self._states: MutableMapping[str, _InventoryState] = {}

    def warm_up(self, history: Sequence[MarketSnapshot]) -> None:
        for snapshot in history:
            state = self._ensure_state(snapshot.symbol)
            state.anchor_price = snapshot.close if snapshot.close > 0 else state.anchor_price

    def on_data(self, snapshot: MarketSnapshot) -> Sequence[StrategySignal]:
        state = self._ensure_state(snapshot.symbol)
        if snapshot.close <= 0:
            return []

        state.anchor_price = state.anchor_price or snapshot.close
        mid_price = snapshot.close

        imbalance = state.inventory - self._settings.inventory_target
        imbalance_ratio = 0.0
        if self._settings.max_inventory > 0:
            imbalance_ratio = max(-1.0, min(1.0, imbalance / self._settings.max_inventory))

        spread = mid_price * (self._settings.spread_bps / 10_000)
        buy_price = max(0.0, mid_price - spread * (1.0 + max(0.0, imbalance_ratio)))
        sell_price = mid_price + spread * (1.0 + max(0.0, -imbalance_ratio))

        buy_confidence = max(0.1, 1.0 - max(0.0, imbalance_ratio))
        sell_confidence = max(0.1, 1.0 + min(0.0, imbalance_ratio))

        signals: list[StrategySignal] = []

        if state.inventory < self._settings.max_inventory * self._settings.rebalance_threshold:
            signals.append(
                StrategySignal(
                    symbol=snapshot.symbol,
                    side="buy",
                    confidence=min(1.0, buy_confidence),
                    metadata={
                        "quote_price": buy_price,
                        "spread_bps": self._settings.spread_bps,
                        "inventory": state.inventory,
                        "target": self._settings.inventory_target,
                    },
                )
            )

        if state.inventory > -self._settings.max_inventory * self._settings.rebalance_threshold:
            signals.append(
                StrategySignal(
                    symbol=snapshot.symbol,
                    side="sell",
                    confidence=min(1.0, sell_confidence),
                    metadata={
                        "quote_price": sell_price,
                        "spread_bps": self._settings.spread_bps,
                        "inventory": state.inventory,
                        "target": self._settings.inventory_target,
                    },
                )
            )

        state.anchor_price = (state.anchor_price * 0.9) + (mid_price * 0.1)
        return signals

    def _ensure_state(self, symbol: str) -> _InventoryState:
        if symbol not in self._states:
            self._states[symbol] = _InventoryState()
        return self._states[symbol]


__all__ = ["MarketMakingSettings", "MarketMakingStrategy"]

