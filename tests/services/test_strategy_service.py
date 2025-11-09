from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List

from bot_core.services.strategy_service import GuardrailsCallback, StrategyService, StrategySignal


@dataclass
class DummyStrategy:
    name: str
    confidence: float

    async def generate_signal(self, snapshot: Dict[str, Any]) -> StrategySignal:
        await asyncio.sleep(0)
        trend = float(snapshot.get("trend", 0.0))
        action = "buy" if trend >= 0 else "sell"
        return StrategySignal(
            strategy=self.name,
            action=action,
            confidence=self.confidence,
            metadata={"trend": trend},
        )


def test_strategy_service_filters_by_guardrails() -> None:
    accepted: List[str] = []

    async def guard(signal: StrategySignal) -> bool:
        accepted.append(signal.strategy)
        return signal.confidence >= 0.5

    service = StrategyService(cast_guardrails(guard))
    service.register(DummyStrategy("trend", 0.7))
    service.register(DummyStrategy("scalp", 0.3))

    results = asyncio.run(service.run_all({"trend": 1.0}))

    assert [signal.strategy for signal in results] == ["trend"]
    assert accepted == ["trend", "scalp"]


def test_unregister_strategy() -> None:
    async def guard(_signal: StrategySignal) -> bool:
        return True

    service = StrategyService(cast_guardrails(guard))
    service.register(DummyStrategy("trend", 0.9))
    service.unregister("trend")

    results = asyncio.run(service.run_all({"trend": 1.0}))

    assert results == []


def cast_guardrails(cb: GuardrailsCallback) -> GuardrailsCallback:
    """Pomocniczy wrapper pozwalający na dokładne typowanie guardrails."""

    return cb
