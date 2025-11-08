"""Adaptacyjna strategia market making z kontrolą inventory i zmienności."""

from __future__ import annotations

from dataclasses import dataclass
from typing import MutableMapping, Sequence

from .base import MarketSnapshot, StrategyEngine, StrategySignal


@dataclass(slots=True)
class AdaptiveMarketMakingSettings:
    """Parametry sterujące adaptacyjną strategią MM."""

    base_spread_bps: float = 10.0
    volatility_sensitivity: float = 1.6
    inventory_skew_strength: float = 0.6
    max_inventory: float = 7.5
    target_inventory: float = 0.0
    cooldown_fraction: float = 0.25
    min_confidence: float = 0.15


@dataclass(slots=True)
class _AdaptiveInventoryState:
    inventory: float = 0.0
    last_volatility: float = 0.0
    last_price: float = 0.0


class AdaptiveMarketMakingStrategy(StrategyEngine):
    """Strategia równoważąca spread względem zmienności i inventory."""

    def __init__(self, settings: AdaptiveMarketMakingSettings | None = None) -> None:
        self._settings = settings or AdaptiveMarketMakingSettings()
        self._states: MutableMapping[str, _AdaptiveInventoryState] = {}

    def warm_up(self, history: Sequence[MarketSnapshot]) -> None:
        for snapshot in history:
            state = self._ensure_state(snapshot.symbol)
            state.last_price = snapshot.close if snapshot.close > 0 else state.last_price
            volatility = float(snapshot.indicators.get("realized_volatility", 0.0))
            state.last_volatility = max(state.last_volatility, volatility)

    def on_data(self, snapshot: MarketSnapshot) -> Sequence[StrategySignal]:
        state = self._ensure_state(snapshot.symbol)
        if snapshot.close <= 0:
            return []

        realized_volatility = float(snapshot.indicators.get("realized_volatility", state.last_volatility))
        inventory = float(snapshot.indicators.get("inventory", state.inventory))
        state.inventory = inventory
        state.last_price = snapshot.close
        state.last_volatility = realized_volatility

        spread_multiplier = 1.0 + realized_volatility * self._settings.volatility_sensitivity
        spread = snapshot.close * (self._settings.base_spread_bps / 10_000) * spread_multiplier

        if self._settings.max_inventory <= 0:
            imbalance_ratio = 0.0
        else:
            imbalance_ratio = max(
                -1.0,
                min(1.0, (inventory - self._settings.target_inventory) / self._settings.max_inventory),
            )

        skew = imbalance_ratio * self._settings.inventory_skew_strength
        buy_price = max(0.0, snapshot.close - spread * (1.0 + max(0.0, skew)))
        sell_price = snapshot.close + spread * (1.0 + max(0.0, -skew))

        buy_confidence = max(self._settings.min_confidence, 1.0 - max(0.0, skew))
        sell_confidence = max(self._settings.min_confidence, 1.0 + min(0.0, skew))

        signals: list[StrategySignal] = []

        if inventory < self._settings.max_inventory * (1.0 - self._settings.cooldown_fraction):
            signals.append(
                StrategySignal(
                    symbol=snapshot.symbol,
                    side="buy",
                    confidence=min(1.0, buy_confidence),
                    metadata={
                        "quote_price": buy_price,
                        "spread_bps": self._settings.base_spread_bps,
                        "volatility": realized_volatility,
                        "inventory": inventory,
                        "target_inventory": self._settings.target_inventory,
                        "mode": "adaptive",
                    },
                )
            )

        if inventory > -self._settings.max_inventory * (1.0 - self._settings.cooldown_fraction):
            signals.append(
                StrategySignal(
                    symbol=snapshot.symbol,
                    side="sell",
                    confidence=min(1.0, sell_confidence),
                    metadata={
                        "quote_price": sell_price,
                        "spread_bps": self._settings.base_spread_bps,
                        "volatility": realized_volatility,
                        "inventory": inventory,
                        "target_inventory": self._settings.target_inventory,
                        "mode": "adaptive",
                    },
                )
            )

        return signals

    def _ensure_state(self, symbol: str) -> _AdaptiveInventoryState:
        if symbol not in self._states:
            self._states[symbol] = _AdaptiveInventoryState()
        return self._states[symbol]


__all__ = ["AdaptiveMarketMakingSettings", "AdaptiveMarketMakingStrategy"]

