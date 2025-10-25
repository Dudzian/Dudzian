from __future__ import annotations

import math
from collections import Counter
from types import MappingProxyType, SimpleNamespace

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
        self.emergency_metadata: dict[str, object] | None = None

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
        self.emergency_metadata = dict(metadata)


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


def test_autotrade_presets_renormalize_missing_entries() -> None:
    adapter = _make_sync_adapter()
    engine = AutoTradeEngine(adapter, lambda *args, **kwargs: None, AutoTradeConfig())

    entries, normalized = engine._build_preset_entries(
        {"trend_following": 0.7, "nonexistent": 0.3}
    )

    assert len(entries) == 1
    entry = entries[0]
    assert entry["name"] == "trend_following"
    assert math.isclose(entry["metadata"]["ensemble_weight"], 1.0, rel_tol=1e-9)
    assert normalized == {"trend_following": 1.0}


def test_autotrade_presets_metadata_consistency_across_regimes() -> None:
    adapter = _make_sync_adapter()
    engine = AutoTradeEngine(adapter, lambda *args, **kwargs: None, AutoTradeConfig())

    recorder = _WorkflowRecorder()
    engine._register_strategy_presets(recorder)

    assert recorder.by_regime
    for regime, entries in recorder.by_regime.items():
        assert entries, f"Brak zarejestrowanych wpisów dla reżimu {regime}"
        meta_payloads = recorder.metadata.get(regime, [])
        assert meta_payloads, f"Brak metadanych dla reżimu {regime}"
        metadata = meta_payloads[0]
        weights = metadata["ensemble_weights"]
        assert math.isclose(sum(weights.values()), 1.0, rel_tol=1e-9)
        ordered_names = [entry["name"] for entry in entries]
        assert list(weights) == ordered_names
        assert set(weights) == {entry["name"] for entry in entries}
        for entry in entries:
            entry_weight = entry["metadata"]["ensemble_weight"]
            assert math.isclose(entry_weight, weights[entry["name"]], rel_tol=1e-9)

    assert recorder.emergency, "Powinien istnieć preset awaryjny"
    assert recorder.emergency_metadata is not None
    emergency_weights = recorder.emergency_metadata["ensemble_weights"]
    assert emergency_weights
    assert math.isclose(sum(emergency_weights.values()), 1.0, rel_tol=1e-9)
    assert list(emergency_weights) == [entry["name"] for entry in recorder.emergency]
    for entry in recorder.emergency:
        assert math.isclose(
            entry["metadata"]["ensemble_weight"],
            emergency_weights[entry["name"]],
            rel_tol=1e-9,
        )


def test_build_preset_entries_orders_by_weight_and_name() -> None:
    adapter = _make_sync_adapter()
    engine = AutoTradeEngine(adapter, lambda *args, **kwargs: None, AutoTradeConfig())

    entries, normalized = engine._build_preset_entries(
        {"day_trading": 2.0, "trend_following": 2.0, "arbitrage": 1.0}
    )

    assert [entry["name"] for entry in entries] == [
        "day_trading",
        "trend_following",
        "arbitrage",
    ]
    assert list(normalized) == ["day_trading", "trend_following", "arbitrage"]
    assert math.isclose(normalized["day_trading"], 0.4, rel_tol=1e-9)
    assert math.isclose(normalized["trend_following"], 0.4, rel_tol=1e-9)
    assert math.isclose(normalized["arbitrage"], 0.2, rel_tol=1e-9)


def test_build_preset_entries_uses_catalog_when_mapping_missing(monkeypatch) -> None:
    adapter = _make_sync_adapter()

    mapping_copy = dict(AutoTradeEngine._PRESET_ENGINE_MAPPING)
    mapping_copy.pop("grid_trading", None)
    monkeypatch.setattr(AutoTradeEngine, "_PRESET_ENGINE_MAPPING", mapping_copy, raising=False)

    engine = AutoTradeEngine(adapter, lambda *args, **kwargs: None, AutoTradeConfig())

    entries, normalized = engine._build_preset_entries({"grid_trading": 1.0})

    assert entries
    assert entries[0]["engine"] == "grid_trading"
    assert math.isclose(normalized["grid_trading"], 1.0, rel_tol=1e-9)


def test_resolve_engine_key_caches_catalog_results(monkeypatch) -> None:
    adapter = _make_sync_adapter()
    engine = AutoTradeEngine(adapter, lambda *args, **kwargs: None, AutoTradeConfig())

    mapping_copy = dict(AutoTradeEngine._PRESET_ENGINE_MAPPING)
    monkeypatch.setattr(AutoTradeEngine, "_PRESET_ENGINE_MAPPING", mapping_copy, raising=False)

    class _CountingCatalog:
        def __init__(self) -> None:
            self.metadata_calls = Counter()
            self.create_calls = Counter()

        def metadata_for(self, name: str):  # noqa: D401 - zgodność z interfejsem
            self.metadata_calls[name] += 1
            return MappingProxyType({})

        def create(self, name: str):  # noqa: D401 - zgodność z interfejsem
            self.create_calls[name] += 1
            return SimpleNamespace(engine_key="dummy_engine", name=name)

    catalog = _CountingCatalog()
    engine._strategy_catalog = catalog  # type: ignore[attr-defined]
    engine._engine_key_cache.clear()

    resolved_first = engine._resolve_engine_key("custom_strategy")
    resolved_second = engine._resolve_engine_key("custom_strategy")

    assert resolved_first == resolved_second == "dummy_engine"
    assert catalog.metadata_calls["custom_strategy"] == 1
    assert catalog.create_calls["custom_strategy"] == 1


def test_resolve_engine_key_caches_missing_catalog_entries(monkeypatch) -> None:
    adapter = _make_sync_adapter()
    engine = AutoTradeEngine(adapter, lambda *args, **kwargs: None, AutoTradeConfig())

    mapping_copy = dict(AutoTradeEngine._PRESET_ENGINE_MAPPING)
    monkeypatch.setattr(AutoTradeEngine, "_PRESET_ENGINE_MAPPING", mapping_copy, raising=False)

    class _MissingCatalog:
        def __init__(self) -> None:
            self.metadata_calls = Counter()
            self.create_calls = Counter()

        def metadata_for(self, name: str):  # noqa: D401 - zgodność z interfejsem
            self.metadata_calls[name] += 1
            return MappingProxyType({})

        def create(self, name: str):  # noqa: D401 - zgodność z interfejsem
            self.create_calls[name] += 1
            return None

    catalog = _MissingCatalog()
    engine._strategy_catalog = catalog  # type: ignore[attr-defined]
    engine._engine_key_cache.clear()

    resolved_first = engine._resolve_engine_key("missing_strategy")
    resolved_second = engine._resolve_engine_key("missing_strategy")

    assert resolved_first is None
    assert resolved_second is None
    assert catalog.metadata_calls["missing_strategy"] == 1
    assert catalog.create_calls["missing_strategy"] == 1
