from __future__ import annotations

import math

from bot_core.ai.regime import MarketRegime
from bot_core.events import EmitterAdapter, Event
from bot_core.trading.auto_trade import AutoTradeConfig, AutoTradeEngine


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
