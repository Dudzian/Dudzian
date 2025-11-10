from __future__ import annotations

import math

import pytest
from typing import Mapping

from bot_core.ai.regime import MarketRegime
from bot_core.events import EmitterAdapter, Event
from bot_core.trading.auto_trade import AutoTradeConfig, AutoTradeEngine
from bot_core.trading.strategies import StrategyCatalog


_NEW_STRATEGIES = {
    "grid_trading",
    "options_income",
    "scalping",
    "statistical_arbitrage",
    "volatility_target",
}


class _WorkflowRecorder:
    def __init__(self) -> None:
        self.by_regime: dict[MarketRegime, list[dict[str, object]]] = {}
        self.metadata: dict[MarketRegime, list[dict[str, object]]] = {}
        self.emergency: list[dict[str, object]] = []

    def register_preset(
        self,
        regime: MarketRegime,
        *,
        name: str,
        entries,
        signing_key,
        metadata,
    ) -> None:
        self.by_regime.setdefault(regime, []).extend(entries)
        self.metadata.setdefault(regime, []).append(metadata)

    def register_emergency_preset(self, *, name: str, entries, signing_key, metadata) -> None:
        self.emergency = list(entries)


def _make_sync_adapter() -> EmitterAdapter:
    adapter = EmitterAdapter()
    bus = adapter.bus
    bus.stop()
    bus._closed = False  # type: ignore[attr-defined]
    bus._async_mode = False  # type: ignore[attr-defined]

    def _publish_sync(event_type, payload=None):
        evt = Event(type=bus._key(event_type), payload=payload)  # type: ignore[attr-defined]
        bus._dispatch(evt)  # type: ignore[attr-defined]

    bus.publish = _publish_sync  # type: ignore[assignment]
    bus.emit = _publish_sync  # type: ignore[assignment]
    bus.emit_event = _publish_sync  # type: ignore[assignment]
    bus.post = _publish_sync  # type: ignore[assignment]
    return adapter


def test_autotrade_engine_registers_extended_strategies() -> None:
    adapter = _make_sync_adapter()
    engine = AutoTradeEngine(adapter, lambda *args, **kwargs: None, AutoTradeConfig())

    recorder = _WorkflowRecorder()
    engine._register_strategy_presets(recorder)

    assert MarketRegime.TREND in recorder.by_regime
    trend_entries = recorder.by_regime[MarketRegime.TREND]
    trend_metadata = recorder.metadata[MarketRegime.TREND][0]
    names = {entry["name"] for entry in trend_entries}

    assert _NEW_STRATEGIES <= names
    weights_meta = trend_metadata["ensemble_weights"]
    assert math.isclose(sum(weights_meta.values()), 1.0, rel_tol=1e-9)
    assert _NEW_STRATEGIES <= set(weights_meta)
    for entry in trend_entries:
        name = entry["name"]
        if name in _NEW_STRATEGIES:
            assert entry["engine"] == name
            assert entry["metadata"]["strategy"] == name
            assert math.isclose(
                entry["metadata"]["ensemble_weight"],
                weights_meta[name],
                rel_tol=1e-9,
            )


def test_strategy_catalog_dynamic_preset_records_metrics() -> None:
    catalog = StrategyCatalog()

    class _Learner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, Mapping[str, float] | None]] = []

        def build_dynamic_preset(
            self,
            regime: str,
            *,
            metrics: Mapping[str, float] | None = None,
        ) -> Mapping[str, object]:
            self.calls.append((regime, metrics))
            return {
                "name": f"adaptive::{regime}",
                "regime": regime,
                "strategies": [{"name": "trend_following", "weight": 1.0}],
                "generated_at": "2024-01-01T00:00:00+00:00",
            }

    learner = _Learner()
    catalog.attach_adaptive_learner(learner)

    metrics = {"confidence": 0.72, "risk_score": 0.15}
    preset = catalog.dynamic_preset_for(MarketRegime.TREND, metrics=metrics)

    assert preset is not None
    assert learner.calls
    regime_key, passed_metrics = learner.calls[0]
    assert regime_key == MarketRegime.TREND.value
    assert passed_metrics is not None
    assert pytest.approx(passed_metrics["confidence"], rel=1e-6) == 0.72
    snapshot = catalog.dynamic_presets_snapshot()
    assert "trend" in snapshot
    stored = snapshot["trend"]
    assert "metrics" in stored
    assert pytest.approx(stored["metrics"]["confidence"], rel=1e-6) == 0.72
    assert "generated_at" in stored
