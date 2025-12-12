from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from bot_core.strategies.base import MarketSnapshot, StrategyEngine, StrategySignal
from bot_core.strategies.catalog import DEFAULT_STRATEGY_CATALOG, StrategyDefinition
from bot_core.strategies.presets import StrategyPresetWizard


def _snapshot(
    symbol: str,
    close: float,
    *,
    ts: int,
    high: float | None = None,
    low: float | None = None,
    volume: float = 200_000.0,
    indicators: Mapping[str, float] | None = None,
) -> MarketSnapshot:
    price_high = high if high is not None else close
    price_low = low if low is not None else close
    return MarketSnapshot(
        symbol=symbol,
        timestamp=ts,
        open=close,
        high=price_high,
        low=price_low,
        close=close,
        volume=volume,
        indicators=indicators or {},
    )


def _definition_from_entry(entry: Mapping[str, Any]) -> StrategyDefinition:
    spec = DEFAULT_STRATEGY_CATALOG.get(str(entry.get("engine")))
    name = str(entry.get("name") or spec.key)
    parameters = dict(entry.get("parameters") or {})
    risk_classes = tuple(
        dict.fromkeys((*spec.risk_classes, *tuple(entry.get("risk_classes") or ())))
    )
    required_data = tuple(
        dict.fromkeys((*spec.required_data, *tuple(entry.get("required_data") or ())))
    )
    risk_hooks = tuple(dict.fromkeys((*spec.risk_hooks, *tuple(entry.get("risk_hooks") or ()))))
    metadata = dict(entry.get("metadata") or {})
    if risk_hooks and "risk_hooks" not in metadata:
        metadata["risk_hooks"] = risk_hooks
    metadata.setdefault("required_data", required_data)
    metadata.setdefault("risk_classes", risk_classes)
    return StrategyDefinition(
        name=name,
        engine=spec.key,
        license_tier=str(entry.get("license_tier") or spec.license_tier),
        risk_classes=risk_classes,
        required_data=required_data,
        risk_hooks=risk_hooks,
        parameters=parameters,
        risk_profile=entry.get("risk_profile"),
        tags=tuple(entry.get("tags") or ()),
        metadata=metadata,
    )


def _build_engine(entry: Mapping[str, Any]) -> tuple[StrategyEngine, StrategyDefinition]:
    definition = _definition_from_entry(entry)
    engine = DEFAULT_STRATEGY_CATALOG.create(definition)
    assert isinstance(engine, StrategyEngine)
    return engine, definition


def test_scalping_preset_lifecycle_generates_signal() -> None:
    wizard = StrategyPresetWizard(DEFAULT_STRATEGY_CATALOG)
    preset = wizard.build_preset(
        "scalping-basic",
        [
            {
                "engine": "scalping",
                "parameters": {"min_price_change": 0.001, "take_profit": 0.002},
                "tags": ["scalping", "intraday"],
            }
        ],
    )
    entry = preset["strategies"][0]
    engine, definition = _build_engine(entry)

    engine.prepare()
    engine.warmup([_snapshot("BTCUSDT", 100.0, ts=0)])
    signals = engine.decide(_snapshot("BTCUSDT", 100.3, ts=1))

    assert isinstance(signals, Sequence)
    assert signals and isinstance(signals[0], StrategySignal)
    assert signals[0].side in {"buy", "sell"}
    assert engine.metadata.get("required_data") == definition.required_data
    assert engine.metadata.get("risk_hooks", tuple()) == definition.risk_hooks


def test_trend_preset_lifecycle_breakout_entry() -> None:
    wizard = StrategyPresetWizard(DEFAULT_STRATEGY_CATALOG)
    preset = wizard.build_preset(
        "trend-intraday",
        [
            {
                "engine": "daily_trend_momentum",
                "parameters": {
                    "fast_ma": 3,
                    "slow_ma": 5,
                    "breakout_lookback": 4,
                    "momentum_window": 3,
                    "atr_window": 3,
                    "min_trend_strength": 0.0001,
                    "min_momentum": 0.0,
                },
                "tags": ["trend", "momentum"],
            }
        ],
    )
    entry = preset["strategies"][0]
    engine, definition = _build_engine(entry)

    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    history = [_snapshot("ETHUSDT", price, ts=index, high=price + 0.2, low=price - 0.2) for index, price in enumerate(prices)]
    engine.warmup(history)

    decision = engine.decide(_snapshot("ETHUSDT", 106.5, ts=len(prices), high=106.7, low=106.1))

    assert isinstance(decision, Sequence)
    assert decision and decision[0].side == "buy"
    assert decision[0].metadata.get("position") == 1.0
    assert engine.metadata.get("required_data") == definition.required_data


def test_mean_reversion_preset_lifecycle_short_signal() -> None:
    preset_path = Path("bot_core/strategies/marketplace/presets/mean_reversion_demo.json")
    raw = json.loads(preset_path.read_text())
    preset = raw["preset"]
    entry = preset["strategies"][0]
    entry["license_tier"] = DEFAULT_STRATEGY_CATALOG.get("mean_reversion").license_tier
    entry["parameters"] = {
        "lookback": 10,
        "entry_zscore": 1.2,
        "exit_zscore": 0.6,
        "max_holding_period": 6,
        "volatility_cap": 0.2,
        "min_volume_usd": 50_000.0,
    }
    engine, definition = _build_engine(entry)

    base_prices = [100.0 + 0.3 * i for i in range(13)]
    history = [_snapshot("SOLUSDT", price, ts=i) for i, price in enumerate(base_prices)]
    engine.warmup(history)

    signals = engine.decide(_snapshot("SOLUSDT", 106.0, ts=len(base_prices)))

    assert isinstance(signals, Sequence)
    assert signals and signals[0].side == "sell"
    assert "zscore" in signals[0].metadata
    assert engine.metadata.get("required_data") == definition.required_data
    assert engine.metadata.get("risk_hooks", tuple()) == definition.risk_hooks
