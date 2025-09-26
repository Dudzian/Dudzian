"""Testy jednostkowe dla nowej warstwy bazowej strategii."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

import pytest

from KryptoLowca.strategies.base import (
    BaseStrategy,
    StrategyContext,
    StrategyError,
    StrategyMetadata,
    StrategySignal,
)
from KryptoLowca.strategies.base.registry import StrategyRegistry
from KryptoLowca.strategies.presets import get_builtin_preset, load_builtin_presets


@dataclass
class _StubProvider:
    payload: Mapping[str, Any]

    async def get_ohlcv(self, symbol: str, timeframe: str, *, limit: int = 500) -> Mapping[str, Any]:
        return {"symbol": symbol, "timeframe": timeframe, "limit": limit}

    async def get_ticker(self, symbol: str) -> Mapping[str, Any]:  # pragma: no cover - nieużyte
        return {"symbol": symbol, "price": self.payload.get("price", 0.0)}


class _DemoStrategy(BaseStrategy):
    metadata = StrategyMetadata(
        name="DemoStrategy",
        description="Strategia do testów jednostkowych",
        risk_level="balanced",
        exchanges=("binance",),
        timeframes=("1h",),
    )

    async def generate_signal(
        self,
        context: StrategyContext,
        market_payload: Mapping[str, Any],
    ) -> StrategySignal:
        price = float(market_payload.get("close", 0.0))
        action = "BUY" if price > 0 else "HOLD"
        return StrategySignal(symbol=context.symbol, action=action, confidence=0.8)


@pytest.mark.asyncio
async def test_strategy_prepare_requires_demo_mode() -> None:
    provider = _StubProvider({"close": 100.0})
    ctx = StrategyContext(
        symbol="BTC/USDT",
        timeframe="1h",
        portfolio_value=10_000.0,
        position=0.0,
        timestamp=datetime.utcnow(),
        metadata=_DemoStrategy.metadata,
        extra={"mode": "demo"},
    )
    strategy = _DemoStrategy()
    await strategy.prepare(ctx, provider)
    signal = await strategy.handle_market_data(ctx, {"close": 100.0})
    assert signal.action == "BUY"


@pytest.mark.asyncio
async def test_strategy_prepare_blocks_live_mode() -> None:
    provider = _StubProvider({"close": -1.0})
    ctx = StrategyContext(
        symbol="ETH/USDT",
        timeframe="1h",
        portfolio_value=5_000.0,
        position=0.0,
        timestamp=datetime.utcnow(),
        metadata=_DemoStrategy.metadata,
        extra={"mode": "live"},
    )
    strategy = _DemoStrategy()
    with pytest.raises(StrategyError):
        await strategy.prepare(ctx, provider)


def test_registry_registration() -> None:
    local_registry = StrategyRegistry()

    @local_registry.register
    class AnotherStrategy(_DemoStrategy):
        pass

    assert "anotherstrategy" in local_registry
    assert local_registry.get("AnotherStrategy") is AnotherStrategy


def test_builtin_presets_available() -> None:
    presets = list(load_builtin_presets())
    assert presets, "Spodziewamy się co najmniej jednego wbudowanego presetu"
    preset_ids = {preset.preset_id for preset in presets}
    for expected in {"DAILY_TREND", "INTRADAY_TREND"}:
        assert expected in preset_ids
        preset = get_builtin_preset(expected)
        assert preset.config["strategy"]["mode"] == "demo"
