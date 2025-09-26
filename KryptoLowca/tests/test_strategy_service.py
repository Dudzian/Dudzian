from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, List

import pytest

from KryptoLowca.services.strategy_service import StrategyService, StrategySignal


@dataclass
class DummyStrategy:
    name: str
    confidence: float

    async def generate_signal(self, snapshot: Dict[str, float]) -> StrategySignal:
        await asyncio.sleep(0)
        return StrategySignal(
            strategy=self.name,
            action="buy" if snapshot.get("trend", 0.0) >= 0 else "sell",
            confidence=self.confidence,
            metadata={"trend": snapshot.get("trend", 0.0)},
        )


@pytest.mark.asyncio
async def test_strategy_service_filters_by_guardrails():
    accepted: List[str] = []

    async def guard(signal: StrategySignal) -> bool:
        accepted.append(signal.strategy)
        return signal.confidence >= 0.5

    service = StrategyService(guard)
    service.register(DummyStrategy("trend", 0.7))
    service.register(DummyStrategy("scalp", 0.3))

    results = await service.run_all({"trend": 1.0})
    assert [signal.strategy for signal in results] == ["trend"]
    assert accepted == ["trend", "scalp"]


@pytest.mark.asyncio
async def test_unregister_strategy():
    async def guard(_signal: StrategySignal) -> bool:
        return True

    service = StrategyService(guard)
    service.register(DummyStrategy("trend", 0.9))
    service.unregister("trend")

    results = await service.run_all({"trend": 1.0})
    assert results == []
