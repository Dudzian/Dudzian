"""Serwis do równoległego uruchamiania strategii i kierowania sygnałów do guardrails."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol


class Strategy(Protocol):
    """Minimalny kontrakt strategii tradingowej."""

    name: str

    async def generate_signal(self, market_snapshot: Dict[str, Any]) -> "StrategySignal":
        ...


@dataclass(slots=True)
class StrategySignal:
    """Standardowy opis sygnału generowanego przez strategię."""

    strategy: str
    action: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_tradeable(self) -> bool:
        return self.action.lower() in {"buy", "sell", "long", "short"}


GuardrailsCallback = Callable[[StrategySignal], Awaitable[bool] | bool]


class StrategyService:
    """Zarządza rejestrem strategii i równoległym generowaniem sygnałów."""

    def __init__(self, guardrails: GuardrailsCallback) -> None:
        self._guardrails = guardrails
        self._strategies: Dict[str, Strategy] = {}

    def register(self, strategy: Strategy) -> None:
        self._strategies[strategy.name] = strategy

    def unregister(self, name: str) -> None:
        self._strategies.pop(name, None)

    def list_strategies(self) -> List[str]:
        return sorted(self._strategies)

    async def run_all(self, market_snapshot: Dict[str, Any]) -> List[StrategySignal]:
        if not self._strategies:
            return []

        tasks = [
            asyncio.create_task(self._execute(name, strategy, market_snapshot))
            for name, strategy in list(self._strategies.items())
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        signals: List[StrategySignal] = []
        for result in results:
            if isinstance(result, StrategySignal):
                signals.append(result)
        return signals

    async def _execute(
        self, name: str, strategy: Strategy, market_snapshot: Dict[str, Any]
    ) -> Optional[StrategySignal]:
        signal = await strategy.generate_signal(market_snapshot)
        if signal.strategy != name:
            signal = StrategySignal(
                strategy=name,
                action=signal.action,
                confidence=signal.confidence,
                metadata=dict(signal.metadata),
            )
        decision = self._guardrails(signal)
        if asyncio.iscoroutine(decision) or asyncio.isfuture(decision):
            decision = await decision  # type: ignore[assignment]
        return signal if decision else None


__all__ = ["StrategyService", "Strategy", "StrategySignal", "GuardrailsCallback"]
