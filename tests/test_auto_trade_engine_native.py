from __future__ import annotations

import pytest

import hashlib
import json
from copy import deepcopy
from dataclasses import replace

from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Iterable, Mapping

import pandas as pd

from bot_core.ai.models import ModelScore
from bot_core.ai.regime import (
    MarketRegime,
    MarketRegimeAssessment,
    MarketRegimeClassifier,
    RegimeHistory,
    RiskLevel,
)
from bot_core.events import EmitterAdapter, Event, EventType
from bot_core.trading.auto_trade import (
    AutoTradeConfig,
    AutoTradeEngine,
    AutoTradeSnapshot,
    RiskFreezeSnapshot,
)
from bot_core.trading.engine import TradingParameters
from bot_core.trading.strategies import StrategyCatalog, StrategyPlugin
from bot_core.strategies.regime_workflow import (
    PresetAvailability,
    PresetVersionInfo,
    RegimePresetActivation,
)
from bot_core.trading.regime_workflow import RegimeSwitchDecision
from bot_core.runtime.journal import InMemoryTradingDecisionJournal


_ENGINE_MAPPING = {
    "trend_following": "daily_trend_momentum",
    "day_trading": "day_trading",
    "mean_reversion": "mean_reversion",
    "arbitrage": "cross_exchange_arbitrage",
    "grid_trading": "grid_trading",
    "options_income": "options_income",
    "scalping": "scalping",
    "statistical_arbitrage": "statistical_arbitrage",
    "volatility_target": "volatility_target",
}


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


def _collect_status_payloads(adapter: EmitterAdapter) -> list[dict]:
    payloads: list[dict] = []

    def _collect(evt_or_batch):
        batch = evt_or_batch if isinstance(evt_or_batch, list) else [evt_or_batch]
        for evt in batch:
            payloads.append(evt.payload)

    adapter.subscribe(EventType.AUTOTRADE_STATUS, _collect)
    return payloads


def _activation_from_decision(decision: RegimeSwitchDecision) -> RegimePresetActivation:
    issued_at = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    version = PresetVersionInfo(
        hash="stub-hash",
        signature=MappingProxyType({"alg": "HMAC-SHA256", "key_id": "stub"}),
        issued_at=issued_at,
        metadata=MappingProxyType(
            {
                "name": "autotrade-stub",
                "strategy_keys": tuple(decision.weights.keys()),
                "strategy_names": tuple(decision.weights.keys()),
                "license_tiers": decision.license_tiers,
                "risk_classes": decision.risk_classes,
                "required_data": decision.required_data,
                "capabilities": decision.capabilities,
                "tags": decision.tags,
                "preset_metadata": {
                    "ensemble_weights": dict(decision.weights),
                },
            }
        ),
    )
    strategies: list[dict[str, object]] = []
    for name, meta in decision.strategy_metadata.items():
        metadata_payload = dict(meta)
        metadata_payload.setdefault("name", name)
        metadata_payload.setdefault("ensemble_weight", decision.weights.get(name, 0.0))
        entry: dict[str, object] = {
            "name": name,
            "engine": _ENGINE_MAPPING.get(name, name),
            "license_tier": meta.get("license_tier"),
            "risk_classes": list(meta.get("risk_classes", ())),
            "required_data": list(meta.get("required_data", ())),
            "capability": meta.get("capability"),
            "tags": list(meta.get("tags", ())),
            "metadata": metadata_payload,
        }
        strategies.append(entry)
    preset = MappingProxyType(
        {
            "name": "autotrade-stub",
            "strategies": strategies,
            "metadata": {"ensemble_weights": dict(decision.weights)},
        }
    )
    return RegimePresetActivation(
        regime=decision.regime,
        assessment=decision.assessment,
        summary=decision.summary,
        preset=preset,
        version=version,
        decision_candidates=(),
        activated_at=issued_at,
        preset_regime=decision.regime,
        used_fallback=False,
        missing_data=(),
        blocked_reason=None,
        recommendation=None,
        license_issues=(),
    )


def test_auto_trade_engine_generates_orders_and_signals(monkeypatch) -> None:
    adapter = _make_sync_adapter()
    orders: list[tuple[str, float]] = []
    signals: list[float] = []
    signal_payloads: list[dict] = []
    statuses: list[str] = []

    def _collect_signals(evt_or_batch):
        batch = evt_or_batch if isinstance(evt_or_batch, list) else [evt_or_batch]
        for evt in batch:
            signals.append(float(evt.payload["direction"]))
            signal_payloads.append(evt.payload)

    adapter.subscribe(EventType.SIGNAL, _collect_signals)

    adapter.subscribe(
        EventType.AUTOTRADE_STATUS,
        lambda evt: statuses.extend([ev.payload["status"] for ev in (evt if isinstance(evt, list) else [evt])]),
    )

    cfg = AutoTradeConfig(
        symbol="BTCUSDT",
        qty=0.5,
        regime_window=6,
        activation_threshold=0.0,
        breakout_window=3,
        mean_reversion_window=4,
        mean_reversion_z=0.5,
    )
    engine = AutoTradeEngine(adapter, lambda side, qty: orders.append((side, qty)), cfg)
    engine.apply_params({"fast": 2, "slow": 5})

    base_time = 1_700_000_000.0
    current_time = {"value": base_time}

    def fake_time() -> float:
        return current_time["value"]

    monkeypatch.setattr("bot_core.trading.auto_trade.time.time", fake_time)
    monkeypatch.setattr("bot_core.events.emitter.time.time", fake_time)

    closes = [10, 9, 8, 7, 6, 7, 8, 9, 10, 9, 8, 7, 6]
    for px in closes:
        adapter.push_market_tick("BTCUSDT", price=px)

    assert orders, "Expected autotrader to submit at least one order"
    assert any(abs(sig) > 0 for sig in signals)
    assert "params_applied" in statuses
    assert signal_payloads, "Expected at least one signal payload"
    signal_detail = signal_payloads[-1]["signals"]
    core_strategies = {"trend_following", "day_trading", "mean_reversion", "arbitrage"}
    assert core_strategies <= set(signal_detail)
    assert signal_detail["daily_breakout"] == signal_detail["day_trading"]
    metadata = signal_payloads[-1]["metadata"]
    assert "standard" in metadata["license_tiers"]
    assert "trend_d1" in metadata["capabilities"]
    assert "regime_summary" in metadata
    assert "risk_score" in metadata["regime_summary"]
    assert "risk_level" in metadata["regime_summary"]
    history_block = metadata["regime_summary"].get("history", [])
    assert isinstance(history_block, list)
    if history_block:
        assert isinstance(history_block[0], dict)
        assert set(history_block[0]) >= {"regime", "risk_score"}
    # serializacja JSON musi działać dla metadanych sygnału
    json.dumps(metadata["regime_summary"], sort_keys=True)
    json.dumps(metadata, sort_keys=True)


class _ConstantTrendStrategy(StrategyPlugin):
    name = "trend_following"
    description = "Stały sygnał dodatni do testów."
    license_tier = "standard"
    risk_classes = ("directional",)
    required_data = ("ohlcv",)
    capability = "trend_d1"
    tags = ("trend",)

    def generate(self, indicators, params, *, market_data=None):  # type: ignore[override]
        series = indicators.ema_fast.copy()
        series[:] = 0.75
        return series


class _ConstantMeanStrategy(StrategyPlugin):
    name = "mean_reversion"
    description = "Stały sygnał średni do testów."
    license_tier = "professional"
    risk_classes = ("statistical",)
    required_data = ("ohlcv", "spread_history")
    capability = "mean_reversion"
    tags = ("mean_reversion",)

    def generate(self, indicators, params, *, market_data=None):  # type: ignore[override]
        series = indicators.ema_fast.copy()
        series[:] = 0.4
        return series


class _WorkflowStub:
    def __init__(self, activation: RegimePresetActivation, catalog: StrategyCatalog) -> None:
        classifier = MarketRegimeClassifier()
        self.classifier = classifier
        self.history = RegimeHistory(thresholds_loader=classifier.thresholds_loader)
        self.history.reload_thresholds(thresholds=classifier.thresholds_snapshot())
        self.catalog = catalog
        self._activation = activation
        self.last_activation = activation
        self.calls: list[tuple[pd.DataFrame, tuple[str, ...], str | None]] = []
        self._availability: tuple[PresetAvailability, ...] = ()
        self._history_entries: list[RegimePresetActivation] = [activation]

    def activate(
        self,
        market_data: pd.DataFrame,
        *,
        available_data: Iterable[str] = (),
        symbol: str | None = None,
        now: datetime | None = None,
    ) -> RegimePresetActivation:
        self.calls.append((market_data.copy(), tuple(sorted(set(available_data))), symbol))
        self.history.update(self._activation.assessment)
        updated = replace(
            self._activation,
            activated_at=now or datetime.now(timezone.utc),
        )
        self.last_activation = updated
        self._history_entries.append(updated)
        return updated

    def set_availability(self, availability: Iterable[PresetAvailability]) -> None:
        self._availability = tuple(availability)

    def get_availability(self) -> tuple[PresetAvailability, ...]:
        return self._availability

    def inspect_presets(
        self, available_data: Iterable[str] = (), *, now: datetime | None = None
    ) -> tuple[PresetAvailability, ...]:
        return self._availability

    def activation_history(
        self, *, limit: int | None = None
    ) -> tuple[RegimePresetActivation, ...]:
        entries: tuple[RegimePresetActivation, ...] = tuple(self._history_entries)
        if limit is None:
            return entries
        try:
            parsed = int(limit)
        except (TypeError, ValueError):
            return entries
        if parsed <= 0:
            return entries
        return entries[-parsed:]

    def activation_history_frame(
        self, *, limit: int | None = None
    ) -> pd.DataFrame:
        entries = self.activation_history(limit=limit)
        columns = [
            "activated_at",
            "regime",
            "preset_regime",
            "preset_name",
            "preset_hash",
            "used_fallback",
            "blocked_reason",
            "missing_data",
            "license_issues",
            "recommendation",
        ]
        if not entries:
            return pd.DataFrame(columns=columns)
        rows = []
        for activation in entries:
            rows.append(
                {
                    "activated_at": activation.activated_at,
                    "regime": activation.regime,
                    "preset_regime": activation.preset_regime,
                    "preset_name": activation.preset.get("name"),
                    "preset_hash": activation.version.hash,
                    "used_fallback": activation.used_fallback,
                    "blocked_reason": activation.blocked_reason,
                    "missing_data": activation.missing_data,
                    "license_issues": activation.license_issues,
                    "recommendation": activation.recommendation,
                }
            )
        frame = pd.DataFrame(rows, columns=columns)
        if not frame.empty:
            frame["regime"] = frame["regime"].apply(
                lambda r: r.value if isinstance(r, MarketRegime) else r
            )
            frame["preset_regime"] = frame["preset_regime"].apply(
                lambda r: r.value if isinstance(r, MarketRegime) else r
            )
        return frame


class _InferenceStub:
    def __init__(self, score: ModelScore) -> None:
        self._score = score
        self.calls: list[dict[str, float]] = []
        self.contexts: list[dict[str, object]] = []
        self.is_ready = True

    def score(self, features, *, context=None):  # type: ignore[override]
        payload = {str(key): float(value) for key, value in features.items()}
        self.calls.append(payload)
        context_payload = {str(key): value for key, value in (context or {}).items()}
        self.contexts.append(context_payload)
        return self._score

    def set_availability(self, availability: Iterable[PresetAvailability]) -> None:
        self._availability = tuple(availability)

    def inspect_presets(
        self, available_data: Iterable[str] = (), *, now: datetime | None = None
    ) -> tuple[PresetAvailability, ...]:
        return self._availability

    def activation_history(
        self, *, limit: int | None = None
    ) -> tuple[RegimePresetActivation, ...]:
        entries: tuple[RegimePresetActivation, ...] = tuple(self._history_entries)
        if limit is None:
            return entries
        try:
            parsed = int(limit)
        except (TypeError, ValueError):
            return entries
        if parsed <= 0:
            return entries
        return entries[-parsed:]

    def activation_history_frame(
        self, *, limit: int | None = None
    ) -> pd.DataFrame:
        entries = self.activation_history(limit=limit)
        columns = [
            "activated_at",
            "regime",
            "preset_regime",
            "preset_name",
            "preset_hash",
            "used_fallback",
            "blocked_reason",
            "missing_data",
            "license_issues",
            "recommendation",
        ]
        if not entries:
            return pd.DataFrame(columns=columns)
        rows = []
        for activation in entries:
            rows.append(
                {
                    "activated_at": activation.activated_at,
                    "regime": activation.regime,
                    "preset_regime": activation.preset_regime,
                    "preset_name": activation.preset.get("name"),
                    "preset_hash": activation.version.hash,
                    "used_fallback": activation.used_fallback,
                    "blocked_reason": activation.blocked_reason,
                    "missing_data": activation.missing_data,
                    "license_issues": activation.license_issues,
                    "recommendation": activation.recommendation,
                }
            )
        frame = pd.DataFrame(rows, columns=columns)
        if not frame.empty:
            frame["regime"] = frame["regime"].apply(lambda r: r.value if isinstance(r, MarketRegime) else r)
            frame["preset_regime"] = frame["preset_regime"].apply(
                lambda r: r.value if isinstance(r, MarketRegime) else r
            )
        return frame


def test_auto_trade_engine_uses_strategy_catalog(monkeypatch) -> None:
    adapter = _make_sync_adapter()
    orders: list[tuple[str, float]] = []
    signal_payloads: list[dict] = []

    def _collect(evt_or_batch):
        batch = evt_or_batch if isinstance(evt_or_batch, list) else [evt_or_batch]
        for evt in batch:
            signal_payloads.append(evt.payload)

    adapter.subscribe(EventType.SIGNAL, _collect)

    catalog = StrategyCatalog(plugins=(_ConstantTrendStrategy,))
    cfg = AutoTradeConfig(
        symbol="BTCUSDT",
        qty=0.2,
        activation_threshold=0.5,
        regime_window=12,
        strategy_weights={
            MarketRegime.TREND.value: {"trend_following": 1.0},
        },
        breakout_window=3,
        mean_reversion_window=3,
    )
    engine = AutoTradeEngine(
        adapter,
        lambda side, qty: orders.append((side, qty)),
        cfg,
        strategy_catalog=catalog,
    )
    engine.apply_params({"fast": 2, "slow": 4})

    base_time = 1_700_100_000.0
    current_time = {"value": base_time}

    def fake_time() -> float:
        return current_time["value"]

    monkeypatch.setattr("bot_core.trading.auto_trade.time.time", fake_time)
    monkeypatch.setattr("bot_core.events.emitter.time.time", fake_time)

    for idx in range(30):
        price = 100.0 + idx * 0.2
        bar = {
            "open_time": float(idx),
            "close": price,
            "high": price * 1.001,
            "low": price * 0.999,
            "volume": 1500.0 + idx * 2,
        }
        current_time["value"] = base_time + idx * 1.0
        adapter.publish(EventType.MARKET_TICK, {"symbol": "BTCUSDT", "bar": bar})

    assert orders, "Expected at least one order from plugin driven signal"
    assert any(side == "buy" for side, _ in orders)
    assert signal_payloads, "Expected signal payloads"
    last_signals = signal_payloads[-1]["signals"]
    assert last_signals["trend_following"] == 0.75
    assert last_signals["daily_breakout"] == last_signals["day_trading"]
    metadata = signal_payloads[-1]["metadata"]
    assert metadata["per_strategy"]["trend_following"]["license_tier"] == "standard"
    assert "regime_summary" in metadata
    assert "risk_score" in metadata["regime_summary"]
    assert "risk_level" in metadata["regime_summary"]
    json.dumps(metadata["regime_summary"], sort_keys=True)
    json.dumps(metadata, sort_keys=True)


def test_auto_trade_engine_emits_regime_update_with_metrics(monkeypatch) -> None:
    adapter = _make_sync_adapter()
    statuses: list[dict] = []

    def _collect(evt_or_batch):
        batch = evt_or_batch if isinstance(evt_or_batch, list) else [evt_or_batch]
        for evt in batch:
            statuses.append(evt.payload)

    adapter.subscribe(EventType.AUTOTRADE_STATUS, _collect)

    cfg = AutoTradeConfig(
        symbol="BTCUSDT",
        regime_window=60,
        activation_threshold=1.0,
        breakout_window=5,
        mean_reversion_window=5,
        mean_reversion_z=1.0,
    )
    engine = AutoTradeEngine(adapter, lambda *_: None, cfg)
    engine.apply_params({"fast": 2, "slow": 4})

    base_price = 100.0
    base_time = 1_700_000_000.0
    current_time = {"value": base_time}

    def fake_time() -> float:
        return current_time["value"]

    monkeypatch.setattr("bot_core.trading.auto_trade.time.time", fake_time)
    monkeypatch.setattr("bot_core.events.emitter.time.time", fake_time)

    for idx in range(80):
        close = base_price + idx * 0.4
        bar = {
            "open_time": float(idx),
            "close": close,
            "high": close * 1.002,
            "low": close * 0.998,
            "volume": 1200.0 + idx * 5.0,
        }
        current_time["value"] = base_time + idx * 1.0
        adapter.publish(EventType.MARKET_TICK, {"symbol": "BTCUSDT", "bar": bar})

    regime_updates = [
        payload
        for payload in statuses
        if payload["status"] == "regime_update" and "trend_strength" in payload["detail"]
    ]
    assert regime_updates, "Expected regime_update status to be emitted"

    detail = regime_updates[-1]["detail"]
    assert detail["regime"] in {regime.value for regime in MarketRegime}
    assert 0.0 <= detail["risk_score"] <= 1.0
    for key in ("trend_strength", "volatility", "volume_trend", "return_skew"):
        assert key in detail


def test_auto_trade_engine_uses_regime_workflow_decision(monkeypatch) -> None:
    adapter = _make_sync_adapter()
    orders: list[tuple[str, float]] = []
    signal_payloads: list[dict] = []
    statuses: list[dict] = []

    def _collect_signals(evt_or_batch):
        batch = evt_or_batch if isinstance(evt_or_batch, list) else [evt_or_batch]
        for evt in batch:
            signal_payloads.append(evt.payload)

    def _collect_status(evt_or_batch):
        batch = evt_or_batch if isinstance(evt_or_batch, list) else [evt_or_batch]
        for evt in batch:
            statuses.append(evt.payload)

    adapter.subscribe(EventType.SIGNAL, _collect_signals)
    adapter.subscribe(EventType.AUTOTRADE_STATUS, _collect_status)

    catalog = StrategyCatalog(plugins=(_ConstantTrendStrategy, _ConstantMeanStrategy))
    base_params = TradingParameters()
    decision_params = replace(
        base_params,
        ema_fast_period=4,
        ema_slow_period=9,
        ensemble_weights={"trend_following": 0.2, "mean_reversion": 0.8},
        day_trading_momentum_window=6,
        day_trading_volatility_window=10,
    )
    assessment = MarketRegimeAssessment(
        regime=MarketRegime.MEAN_REVERSION,
        confidence=0.72,
        risk_score=0.33,
        metrics={},
        symbol="BTCUSDT",
    )
    strategy_metadata = MappingProxyType(
        {
            "trend_following": MappingProxyType(
                {
                    "license_tier": "standard",
                    "risk_classes": ("directional",),
                    "required_data": ("ohlcv",),
                    "capability": "trend_d1",
                    "tags": ("trend",),
                }
            ),
            "mean_reversion": MappingProxyType(
                {
                    "license_tier": "professional",
                    "risk_classes": ("statistical",),
                    "required_data": ("ohlcv", "spread_history"),
                    "capability": "mean_reversion",
                    "tags": ("mean_reversion",),
                }
            ),
        }
    )
    decision = RegimeSwitchDecision(
        regime=assessment.regime,
        assessment=assessment,
        summary=None,
        weights={"trend_following": 0.2, "mean_reversion": 0.8},
        parameters=decision_params,
        timestamp=pd.Timestamp.utcnow(),
        strategy_metadata=strategy_metadata,
        license_tiers=("standard", "professional"),
        risk_classes=("directional", "statistical"),
        required_data=("ohlcv", "spread_history"),
        capabilities=("trend_d1", "mean_reversion"),
        tags=("trend", "mean_reversion"),
    )
    activation = _activation_from_decision(decision)
    workflow = _WorkflowStub(activation, catalog)

    cfg = AutoTradeConfig(
        symbol="BTCUSDT",
        qty=0.3,
        activation_threshold=0.2,
        regime_window=12,
        breakout_window=4,
        mean_reversion_window=4,
    )
    engine = AutoTradeEngine(
        adapter,
        lambda side, qty: orders.append((side, qty)),
        cfg,
        strategy_catalog=catalog,
        regime_workflow=workflow,
    )
    engine.apply_params({"fast": 3, "slow": 8})
    expected_params = engine._build_base_trading_parameters()

    base_time = 1_700_200_000.0
    current_time = {"value": base_time}

    def fake_time() -> float:
        return current_time["value"]

    monkeypatch.setattr("bot_core.trading.auto_trade.time.time", fake_time)
    monkeypatch.setattr("bot_core.events.emitter.time.time", fake_time)

    for idx in range(25):
        price = 150.0 + idx * 0.4
        bar = {
            "open_time": float(idx),
            "close": price,
            "high": price * 1.001,
            "low": price * 0.999,
            "volume": 1000.0 + idx * 3,
        }
        current_time["value"] = base_time + idx
        adapter.publish(EventType.MARKET_TICK, {"symbol": "BTCUSDT", "bar": bar})

    assert workflow.calls, "Expected regime workflow to be invoked"
    _, available_data, call_symbol = workflow.calls[-1]
    assert call_symbol == cfg.symbol
    assert "ohlcv" in available_data
    assert orders, "Expected workflow-driven decision to result in an order"
    assert signal_payloads, "Expected workflow-driven signal payloads"
    latest_signal = signal_payloads[-1]
    assert latest_signal["weights"] == decision.weights
    params_payload = latest_signal["strategy_parameters"]
    assert params_payload["ema_fast_period"] == expected_params.ema_fast_period
    assert params_payload["ema_slow_period"] == expected_params.ema_slow_period
    assert params_payload["ensemble_weights"] == decision_params.ensemble_weights
    assert params_payload["day_trading_momentum_window"] == expected_params.day_trading_momentum_window
    assert params_payload["day_trading_volatility_window"] == expected_params.day_trading_volatility_window
    metadata_block = latest_signal["metadata"]
    assert metadata_block["license_tiers"] == ["standard", "professional"]
    assert "activation" in metadata_block
    activation_meta = metadata_block["activation"]
    assert activation_meta["preset_name"] == "autotrade-stub"
    assert activation_meta["used_fallback"] is False
    if "regime_summary" in metadata_block:
        history_block = metadata_block["regime_summary"].get("history", [])
        assert isinstance(history_block, list)
        if history_block:
            assert isinstance(history_block[0], dict)
            assert set(history_block[0]) >= {"regime", "risk_score"}
    assert engine.last_regime_decision is not None
    assert engine.last_regime_decision.weights == decision.weights
    expected_decision_params = replace(
        expected_params,
        ensemble_weights=decision_params.ensemble_weights,
    )
    assert engine.last_regime_decision.parameters == expected_decision_params
    assert engine.last_regime_activation is not None

    entry_statuses = [st for st in statuses if st.get("status") in {"entry_long", "entry_short"}]
    assert entry_statuses, "Expected entry status to include workflow metadata"
    last_entry = entry_statuses[-1]
    assert last_entry["detail"]["regime"]["regime"] == assessment.regime.value
    assert "summary" in last_entry["detail"]
    json.dumps(last_entry["detail"]["summary"], sort_keys=True)
    assert last_entry["detail"]["metadata"]["capabilities"] == ["trend_d1", "mean_reversion"]
    assert last_entry["detail"]["activation"]["preset_name"] == "autotrade-stub"


class _DummySummary:
    def __init__(self, level: RiskLevel, score: float) -> None:
        self.risk_level = level
        self.risk_score = score


class _DummyHistory:
    def __init__(self, summary: _DummySummary | None) -> None:
        self._summary = summary

    def summarise(self) -> _DummySummary | None:  # pragma: no cover - stub
        return self._summary

    def thresholds_snapshot(self) -> dict:
        return {}

    def reload_thresholds(self, *args, **kwargs) -> None:
        return None

    def update(self, *args, **kwargs) -> None:
        return None


def test_auto_risk_freeze_sync_state(monkeypatch) -> None:
    adapter = _make_sync_adapter()
    statuses: list[dict] = []
    journal = InMemoryTradingDecisionJournal()

    def _collect(evt_or_batch):
        batch = evt_or_batch if isinstance(evt_or_batch, list) else [evt_or_batch]
        for evt in batch:
            statuses.append(evt.payload)

    adapter.subscribe(EventType.AUTOTRADE_STATUS, _collect)

    cfg = AutoTradeConfig(
        symbol="BTCUSDT",
        risk_freeze_seconds=30,
        activation_threshold=1.0,
        auto_risk_freeze=True,
        auto_risk_freeze_level=RiskLevel.WATCH,
        auto_risk_freeze_score=0.3,
    )

    engine = AutoTradeEngine(
        adapter,
        lambda *_: None,
        cfg,
        decision_journal=journal,
        decision_journal_context={
            "environment": "paper",
            "portfolio": "autotrader",
            "risk_profile": "trend",
        },
    )

    base_time = 1_700_000_000.0
    current_time = {"value": base_time}

    def fake_time() -> float:
        return current_time["value"]

    monkeypatch.setattr("bot_core.trading.auto_trade.time.time", fake_time)
    monkeypatch.setattr("bot_core.events.emitter.time.time", fake_time)

    history = _DummyHistory(_DummySummary(RiskLevel.CRITICAL, 0.9))
    engine._regime_history = history  # type: ignore[assignment]

    engine._sync_freeze_state()
    assert engine._auto_risk_frozen is True
    assert engine._risk_frozen_until == pytest.approx(base_time + cfg.risk_freeze_seconds)
    assert statuses and statuses[-1]["status"] == "auto_risk_freeze"
    first_detail = statuses[-1]["detail"]
    assert first_detail["reason"] == "risk_level_and_score_threshold"
    assert first_detail["risk_level"] == RiskLevel.CRITICAL.value
    assert first_detail["risk_score"] == pytest.approx(0.9)
    assert first_detail["triggered_at"] == pytest.approx(base_time)
    assert first_detail["last_extension_at"] == pytest.approx(base_time)
    assert first_detail["released_at"] is None
    assert first_detail["frozen_for"] == pytest.approx(cfg.risk_freeze_seconds)
    assert first_detail["until"] == pytest.approx(engine._risk_frozen_until)

    history._summary = _DummySummary(RiskLevel.CRITICAL, 0.95)
    current_time["value"] = base_time + 10

    engine._sync_freeze_state()

    assert engine._auto_risk_frozen is True
    assert statuses[-1]["status"] == "auto_risk_freeze_extend"
    extend_detail = statuses[-1]["detail"]
    assert extend_detail["reason"] == "risk_score_increase"
    assert extend_detail["extended_from"] == pytest.approx(first_detail["until"])
    assert extend_detail["until"] == pytest.approx(engine._risk_frozen_until)
    assert extend_detail["triggered_at"] == pytest.approx(first_detail["triggered_at"])
    assert extend_detail["last_extension_at"] == pytest.approx(current_time["value"])

    history._summary = _DummySummary(RiskLevel.CALM, 0.1)
    current_time["value"] = engine._risk_frozen_until + 5

    engine._sync_freeze_state()

    assert engine._auto_risk_frozen is False
    assert engine._risk_frozen_until == 0.0
    assert statuses[-1]["status"] == "auto_risk_unfreeze"
    release_detail = statuses[-1]["detail"]
    assert release_detail["reason"] == "risk_recovered"
    assert release_detail["released_at"] == pytest.approx(current_time["value"])
    assert release_detail["triggered_at"] == pytest.approx(first_detail["triggered_at"])
    assert release_detail["frozen_for"] == pytest.approx(
        release_detail["released_at"] - release_detail["triggered_at"]
    )

    exported = list(journal.export())
    assert [event["event"] for event in exported][-3:] == [
        "auto_risk_freeze",
        "auto_risk_freeze_extend",
        "auto_risk_unfreeze",
    ]
    freeze_event = exported[-3]
    assert freeze_event["mode"] == "auto"
    assert freeze_event["risk_level"] == RiskLevel.CRITICAL.value
    extend_event = exported[-2]
    assert extend_event["reason"] == "risk_score_increase"
    assert extend_event["mode"] == "auto"
    unfreeze_event = exported[-1]
    assert unfreeze_event["mode"] == "auto"
    assert float(unfreeze_event["frozen_for"]) == pytest.approx(
        release_detail["frozen_for"]
    )


def test_manual_risk_freeze_telemetry(monkeypatch) -> None:
    adapter = _make_sync_adapter()
    statuses: list[dict] = []
    journal = InMemoryTradingDecisionJournal()

    def _collect(evt_or_batch):
        batch = evt_or_batch if isinstance(evt_or_batch, list) else [evt_or_batch]
        for evt in batch:
            statuses.append(evt.payload)

    adapter.subscribe(EventType.AUTOTRADE_STATUS, _collect)

    cfg = AutoTradeConfig(symbol="BTCUSDT", risk_freeze_seconds=120)
    engine = AutoTradeEngine(
        adapter,
        lambda *_: None,
        cfg,
        decision_journal=journal,
        decision_journal_context={
            "environment": "paper",
            "portfolio": "autotrader",
            "risk_profile": "trend",
        },
    )

    base_time = 1_701_000_000.0
    current_time = {"value": base_time}

    def fake_time() -> float:
        return current_time["value"]

    monkeypatch.setattr("bot_core.trading.auto_trade.time.time", fake_time)
    monkeypatch.setattr("bot_core.events.emitter.time.time", fake_time)

    engine.freeze_risk(duration=45, reason="manual_override")

    assert statuses and statuses[-1]["status"] == "risk_freeze"
    freeze_detail = statuses[-1]["detail"]
    assert freeze_detail["reason"] == "manual_override"
    assert freeze_detail["source_reason"] == "manual"
    assert freeze_detail["triggered_at"] == pytest.approx(base_time)
    assert freeze_detail["last_extension_at"] == pytest.approx(base_time)
    assert freeze_detail["released_at"] is None
    assert freeze_detail["frozen_for"] == pytest.approx(45.0)

    current_time["value"] += 10
    engine.freeze_risk(duration=90, reason="manual_override")

    assert statuses[-1]["status"] == "risk_freeze_extend"
    extend_detail = statuses[-1]["detail"]
    assert extend_detail["extended_from"] == pytest.approx(freeze_detail["until"])
    assert extend_detail["triggered_at"] == pytest.approx(freeze_detail["triggered_at"])
    assert extend_detail["last_extension_at"] == pytest.approx(current_time["value"])
    assert extend_detail["frozen_for"] == pytest.approx(
        extend_detail["until"] - extend_detail["triggered_at"]
    )

    current_time["value"] += 20
    engine.unfreeze_risk(reason="operator_clear")

    assert statuses[-1]["status"] == "risk_unfreeze"
    unfreeze_detail = statuses[-1]["detail"]
    assert unfreeze_detail["reason"] == "manual_override"
    assert unfreeze_detail["source_reason"] == "operator_clear"
    assert unfreeze_detail["released_at"] == pytest.approx(current_time["value"])
    assert unfreeze_detail["triggered_at"] == pytest.approx(freeze_detail["triggered_at"])
    assert unfreeze_detail["frozen_for"] == pytest.approx(
        unfreeze_detail["released_at"] - unfreeze_detail["triggered_at"]
    )

    exported = list(journal.export())
    assert [event["event"] for event in exported][-3:] == [
        "risk_freeze",
        "risk_freeze_extend",
        "risk_unfreeze",
    ]
    freeze_event = exported[-3]
    assert freeze_event["mode"] == "manual"
    assert freeze_event["risk_profile"] == "trend"
    assert float(freeze_event["frozen_for"]) == pytest.approx(45.0)
    extend_event = exported[-2]
    assert extend_event["mode"] == "manual"
    assert extend_event["source_reason"] == "manual"
    unfreeze_event = exported[-1]
    assert unfreeze_event["mode"] == "manual"
    assert unfreeze_event["source_reason"] == "operator_clear"
    assert unfreeze_event["risk_profile"] == "trend"


def test_auto_trade_snapshot_exposes_read_only_state(monkeypatch) -> None:
    adapter = _make_sync_adapter()
    orders: list[tuple[str, float]] = []

    catalog = StrategyCatalog(plugins=(_ConstantTrendStrategy, _ConstantMeanStrategy))
    base_params = TradingParameters()
    decision_params = replace(
        base_params,
        ema_fast_period=5,
        ema_slow_period=11,
        ensemble_weights={"trend_following": 0.3, "mean_reversion": 0.7},
        day_trading_momentum_window=6,
        day_trading_volatility_window=9,
    )
    assessment = MarketRegimeAssessment(
        regime=MarketRegime.TREND,
        confidence=0.61,
        risk_score=0.42,
        metrics={},
        symbol="BTCUSDT",
    )
    strategy_metadata = MappingProxyType(
        {
            "trend_following": MappingProxyType(
                {
                    "license_tier": "standard",
                    "risk_classes": ("directional",),
                    "required_data": ("ohlcv",),
                    "capability": "trend_d1",
                    "tags": ("trend",),
                }
            ),
            "mean_reversion": MappingProxyType(
                {
                    "license_tier": "professional",
                    "risk_classes": ("statistical",),
                    "required_data": ("ohlcv", "spread_history"),
                    "capability": "mean_reversion",
                    "tags": ("mean_reversion",),
                }
            ),
        }
    )
    decision = RegimeSwitchDecision(
        regime=assessment.regime,
        assessment=assessment,
        summary=None,
        weights=dict(decision_params.ensemble_weights),
        parameters=decision_params,
        timestamp=pd.Timestamp.utcnow(),
        strategy_metadata=strategy_metadata,
        license_tiers=("standard", "professional"),
        risk_classes=("directional", "statistical"),
        required_data=("ohlcv", "spread_history"),
        capabilities=("trend_d1", "mean_reversion"),
        tags=("trend", "mean_reversion"),
    )
    activation = _activation_from_decision(decision)
    workflow = _WorkflowStub(activation, catalog)

    cfg = AutoTradeConfig(
        symbol="BTCUSDT",
        qty=0.4,
        activation_threshold=0.1,
        regime_window=10,
        breakout_window=4,
        mean_reversion_window=4,
    )
    engine = AutoTradeEngine(
        adapter,
        lambda side, qty: orders.append((side, qty)),
        cfg,
        strategy_catalog=catalog,
        regime_workflow=workflow,
    )
    engine.apply_params({"fast": 4, "slow": 9})
    expected_params = engine._build_base_trading_parameters()

    base_time = 1_700_400_000.0
    current_time = {"value": base_time}

    def fake_time() -> float:
        return current_time["value"]

    monkeypatch.setattr("bot_core.trading.auto_trade.time.time", fake_time)
    monkeypatch.setattr("bot_core.events.emitter.time.time", fake_time)

    for idx in range(24):
        price = 200.0 + idx * 0.3
        bar = {
            "open_time": float(idx),
            "close": price,
            "high": price * 1.001,
            "low": price * 0.999,
            "volume": 1800.0 + idx,
        }
        current_time["value"] = base_time + idx
        adapter.publish(EventType.MARKET_TICK, {"symbol": "BTCUSDT", "bar": bar})

    adapter.publish(EventType.RISK_ALERT, {"symbol": "BTCUSDT", "kind": "stress"})

    engine._auto_risk_frozen = True
    engine._auto_risk_frozen_until = base_time + 600
    engine._auto_risk_state.risk_level = RiskLevel.CRITICAL
    engine._auto_risk_state.risk_score = 0.87
    engine._auto_risk_state.triggered_at = base_time
    engine._auto_risk_state.last_extension_at = base_time + 30
    current_time["value"] = base_time + 60

    snapshot = engine.snapshot()

    assert isinstance(snapshot, AutoTradeSnapshot)
    assert snapshot.symbol == cfg.symbol
    assert snapshot.enabled is True
    expected_snapshot_params = replace(
        expected_params,
        ensemble_weights=decision_params.ensemble_weights,
    )
    assert snapshot.trading_parameters == expected_snapshot_params
    assert snapshot.strategy_weights == expected_snapshot_params.ensemble_weights
    assert isinstance(snapshot.risk, RiskFreezeSnapshot)
    assert snapshot.risk.manual_active is True
    assert snapshot.risk.manual_reason == "stress"
    assert snapshot.risk.auto_active is True
    assert snapshot.risk.auto_risk_level is RiskLevel.CRITICAL
    assert snapshot.ai_inference is None
    assert snapshot.regime_decision is not None
    assert snapshot.regime_decision.weights == decision.weights
    assert snapshot.regime_decision.parameters == expected_snapshot_params
    assert snapshot.regime_activation is not None
    assert snapshot.regime_activation["preset_name"] == "autotrade-stub"
    assert snapshot.regime_thresholds == workflow.history.thresholds_snapshot()
    overrides = snapshot.regime_parameter_overrides
    assert MarketRegime.TREND.value in overrides
    assert overrides[MarketRegime.TREND.value]["day_trading_momentum_window"] == cfg.breakout_window
    assert {entry["name"] for entry in snapshot.strategy_catalog} == {
        "trend_following",
        "mean_reversion",
    }

    snapshot.strategy_weights["trend_following"] = 0.0
    refreshed = engine.snapshot()
    assert refreshed.strategy_weights == expected_snapshot_params.ensemble_weights
    assert refreshed.risk.combined_until >= snapshot.risk.combined_until
    assert refreshed.ai_inference is None


def test_auto_trade_engine_inspect_regime_presets_reports_availability() -> None:
    adapter = _make_sync_adapter()
    catalog = StrategyCatalog(plugins=(_ConstantTrendStrategy,))
    assessment = MarketRegimeAssessment(
        regime=MarketRegime.TREND,
        confidence=0.7,
        risk_score=0.2,
        metrics={},
        symbol="BTCUSDT",
    )
    base_params = TradingParameters()
    strategy_metadata = MappingProxyType(
        {
            "trend_following": MappingProxyType(
                {
                    "license_tier": "standard",
                    "risk_classes": ("directional",),
                    "required_data": ("ohlcv",),
                    "capability": "trend_d1",
                    "tags": ("trend",),
                }
            )
        }
    )
    decision = RegimeSwitchDecision(
        regime=assessment.regime,
        assessment=assessment,
        summary=None,
        weights={"trend_following": 1.0},
        parameters=base_params,
        timestamp=pd.Timestamp.utcnow(),
        strategy_metadata=strategy_metadata,
        license_tiers=("standard",),
        risk_classes=("directional",),
        required_data=("ohlcv",),
        capabilities=("trend_d1",),
        tags=("trend",),
    )
    activation = _activation_from_decision(decision)
    workflow = _WorkflowStub(activation, catalog)
    availability = PresetAvailability(
        regime=MarketRegime.TREND,
        version=activation.version,
        ready=False,
        blocked_reason="missing_data",
        missing_data=("spread_history",),
        license_issues=("pro_required",),
        schedule_blocked=False,
    )
    workflow.set_availability((availability,))

    cfg = AutoTradeConfig(symbol="BTCUSDT", qty=0.1, regime_window=6)
    engine = AutoTradeEngine(
        adapter,
        lambda *_: None,
        cfg,
        strategy_catalog=catalog,
        regime_workflow=workflow,
    )

    reports = engine.inspect_regime_presets(available_data={"ohlcv", "technical_indicators"})

    assert len(reports) == 1
    report = reports[0]
    assert report["regime"] == MarketRegime.TREND.value
    assert report["ready"] is False
    assert report["missing_data"] == ["spread_history"]
    assert report["license_issues"] == ["pro_required"]
    assert report["preset_hash"] == activation.version.hash
    assert report["preset_signature"]["alg"] == "HMAC-SHA256"
    assert report["license_tiers"] == ["standard"]
    assert report["preset_name"] == "autotrade-stub"


def test_auto_trade_engine_activation_history_records_and_frame() -> None:
    adapter = _make_sync_adapter()
    catalog = StrategyCatalog(plugins=(_ConstantTrendStrategy,))
    assessment = MarketRegimeAssessment(
        regime=MarketRegime.TREND,
        confidence=0.65,
        risk_score=0.18,
        metrics={},
        symbol="BTCUSDT",
    )
    base_params = TradingParameters()
    strategy_metadata = MappingProxyType(
        {
            "trend_following": MappingProxyType(
                {
                    "license_tier": "standard",
                    "risk_classes": ("directional",),
                    "required_data": ("ohlcv",),
                    "capability": "trend_d1",
                    "tags": ("trend",),
                }
            )
        }
    )
    decision = RegimeSwitchDecision(
        regime=assessment.regime,
        assessment=assessment,
        summary=None,
        weights={"trend_following": 1.0},
        parameters=base_params,
        timestamp=pd.Timestamp.utcnow(),
        strategy_metadata=strategy_metadata,
        license_tiers=("standard",),
        risk_classes=("directional",),
        required_data=("ohlcv",),
        capabilities=("trend_d1",),
        tags=("trend",),
    )
    activation = _activation_from_decision(decision)
    workflow = _WorkflowStub(activation, catalog)

    cfg = AutoTradeConfig(symbol="BTCUSDT", qty=0.2, regime_window=6)
    engine = AutoTradeEngine(
        adapter,
        lambda *_: None,
        cfg,
        strategy_catalog=catalog,
        regime_workflow=workflow,
    )

    records = engine.regime_activation_history_records()
    assert len(records) == 1
    first_record = records[0]
    assert first_record["preset_name"] == "autotrade-stub"
    assert first_record["used_fallback"] is False

    frame = engine.regime_activation_history_frame()
    assert not frame.empty
    assert frame.iloc[0]["preset_hash"] == activation.version.hash

    sample = pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.05, 1.15, 1.25],
            "volume": [1000, 1100, 1200],
        }
    )
    workflow.activate(sample, available_data=("ohlcv",))
    workflow.activation_history_frame = None  # type: ignore[assignment]

    tail_frame = engine.regime_activation_history_frame(limit=1)
    assert len(tail_frame) == 1
    assert tail_frame.iloc[0]["preset_name"] == "autotrade-stub"
    assert tail_frame.iloc[0]["regime"] == MarketRegime.TREND.value


def test_auto_trade_engine_summarizes_preset_availability() -> None:
    adapter = _make_sync_adapter()
    catalog = StrategyCatalog(plugins=(_ConstantTrendStrategy, _ConstantMeanStrategy))
    assessment = MarketRegimeAssessment(
        regime=MarketRegime.TREND,
        confidence=0.67,
        risk_score=0.22,
        metrics={},
        symbol="BTCUSDT",
    )
    base_params = TradingParameters()
    strategy_metadata = MappingProxyType(
        {
            "trend_following": MappingProxyType(
                {
                    "license_tier": "standard",
                    "risk_classes": ("directional",),
                    "required_data": ("ohlcv",),
                    "capability": "trend_d1",
                    "tags": ("trend",),
                }
            ),
            "mean_reversion": MappingProxyType(
                {
                    "license_tier": "professional",
                    "risk_classes": ("statistical",),
                    "required_data": ("ohlcv", "spread_history"),
                    "capability": "mean_reversion",
                    "tags": ("mean_reversion",),
                }
            ),
        }
    )
    decision = RegimeSwitchDecision(
        regime=assessment.regime,
        assessment=assessment,
        summary=None,
        weights={"trend_following": 0.6, "mean_reversion": 0.4},
        parameters=base_params,
        timestamp=pd.Timestamp.utcnow(),
        strategy_metadata=strategy_metadata,
        license_tiers=("standard", "professional"),
        risk_classes=("directional", "statistical"),
        required_data=("ohlcv", "spread_history"),
        capabilities=("trend_d1", "mean_reversion"),
        tags=("trend", "mean_reversion"),
    )
    activation = _activation_from_decision(decision)
    workflow = _WorkflowStub(activation, catalog)
    secondary_version = replace(
        activation.version,
        hash="mean-hash",
        issued_at=activation.version.issued_at + timedelta(minutes=5),
        metadata=MappingProxyType(
            {
                **dict(activation.version.metadata),
                "name": "autotrade-mean",
                "preset_metadata": {"ensemble_weights": {"mean_reversion": 1.0}},
            }
        ),
    )
    primary_availability = PresetAvailability(
        regime=MarketRegime.TREND,
        version=activation.version,
        ready=True,
        blocked_reason=None,
        missing_data=(),
        license_issues=(),
        schedule_blocked=False,
    )
    secondary_availability = PresetAvailability(
        regime=MarketRegime.MEAN_REVERSION,
        version=secondary_version,
        ready=False,
        blocked_reason="license_unavailable",
        missing_data=("order_book", "spread_history"),
        license_issues=("pro_tier_required",),
        schedule_blocked=True,
    )
    workflow.set_availability((primary_availability, secondary_availability))

    cfg = AutoTradeConfig(symbol="BTCUSDT", qty=0.1, regime_window=6)
    engine = AutoTradeEngine(
        adapter,
        lambda *_: None,
        cfg,
        strategy_catalog=catalog,
        regime_workflow=workflow,
    )

    summary = engine.summarize_regime_presets(available_data=("ohlcv", "spread_history"))

    assert summary["total_presets"] == 2
    assert summary["ready_presets"] == 1
    assert summary["blocked_presets"] == 1
    assert summary["schedule_blocked_presets"] == 1
    assert summary["missing_data_counts"]["order_book"] == 1
    assert summary["license_issue_counts"]["pro_tier_required"] == 1
    assert summary["blocked_reason_counts"]["license_unavailable"] == 1
    trend_stats = summary["regimes"][MarketRegime.TREND.value]
    assert trend_stats["ready_presets"] == 1
    mean_stats = summary["regimes"][MarketRegime.MEAN_REVERSION.value]
    assert mean_stats["blocked_presets"] == 1
    assert set(mean_stats["missing_data"]) == {"order_book", "spread_history"}
    assert mean_stats["license_issue_counts"]["pro_tier_required"] == 1


def test_auto_trade_engine_summarizes_activation_history() -> None:
    adapter = _make_sync_adapter()
    catalog = StrategyCatalog(plugins=(_ConstantTrendStrategy, _ConstantMeanStrategy))
    assessment = MarketRegimeAssessment(
        regime=MarketRegime.TREND,
        confidence=0.71,
        risk_score=0.28,
        metrics={},
        symbol="BTCUSDT",
    )
    base_params = TradingParameters()
    strategy_metadata = MappingProxyType(
        {
            "trend_following": MappingProxyType(
                {
                    "license_tier": "standard",
                    "risk_classes": ("directional",),
                    "required_data": ("ohlcv",),
                    "capability": "trend_d1",
                    "tags": ("trend",),
                }
            ),
            "mean_reversion": MappingProxyType(
                {
                    "license_tier": "professional",
                    "risk_classes": ("statistical",),
                    "required_data": ("ohlcv", "spread_history"),
                    "capability": "mean_reversion",
                    "tags": ("mean_reversion",),
                }
            ),
        }
    )
    decision = RegimeSwitchDecision(
        regime=assessment.regime,
        assessment=assessment,
        summary=None,
        weights={"trend_following": 0.5, "mean_reversion": 0.5},
        parameters=base_params,
        timestamp=pd.Timestamp.utcnow(),
        strategy_metadata=strategy_metadata,
        license_tiers=("standard", "professional"),
        risk_classes=("directional", "statistical"),
        required_data=("ohlcv", "spread_history"),
        capabilities=("trend_d1", "mean_reversion"),
        tags=("trend", "mean_reversion"),
    )
    activation = _activation_from_decision(decision)
    workflow = _WorkflowStub(activation, catalog)

    fallback_activation = replace(
        activation,
        regime=MarketRegime.MEAN_REVERSION,
        preset_regime=MarketRegime.MEAN_REVERSION,
        activated_at=activation.activated_at + timedelta(minutes=10),
        version=replace(
            activation.version,
            hash="mean-hash",
            issued_at=activation.version.issued_at + timedelta(minutes=10),
        ),
        used_fallback=True,
        missing_data=("spread_history",),
        blocked_reason="data_gap",
        license_issues=("pro_tier_required",),
        recommendation="backfill_spread",
    )
    daily_activation = replace(
        activation,
        regime=MarketRegime.DAILY,
        preset_regime=MarketRegime.DAILY,
        activated_at=activation.activated_at + timedelta(minutes=20),
        version=replace(
            activation.version,
            hash="arb-hash",
            issued_at=activation.version.issued_at + timedelta(minutes=20),
        ),
        missing_data=(),
        blocked_reason=None,
        license_issues=("compliance_hold",),
        recommendation="escalate_license",
    )
    workflow._history_entries = [activation, fallback_activation, daily_activation]
    workflow.last_activation = daily_activation

    cfg = AutoTradeConfig(symbol="BTCUSDT", qty=0.1, regime_window=6)
    engine = AutoTradeEngine(
        adapter,
        lambda *_: None,
        cfg,
        strategy_catalog=catalog,
        regime_workflow=workflow,
    )

    summary = engine.summarize_regime_activation_history()

    assert summary["total_activations"] == 3
    assert summary["fallback_activations"] == 1
    assert summary["license_issue_activations"] == 2
    assert summary["missing_data_counts"]["spread_history"] == 1
    assert summary["license_issue_counts"]["compliance_hold"] == 1
    assert summary["blocked_reason_counts"]["data_gap"] == 1
    assert summary["first_activation_at"] == activation.activated_at.isoformat()
    assert summary["last_activation"]["preset_hash"] == "arb-hash"
    trend_stats = summary["regimes"][MarketRegime.TREND.value]
    assert trend_stats["activations"] == 1
    mean_stats = summary["regimes"][MarketRegime.MEAN_REVERSION.value]
    assert mean_stats["fallback_activations"] == 1
    assert mean_stats["license_issue_counts"]["pro_tier_required"] == 1
    daily_stats = summary["regimes"][MarketRegime.DAILY.value]
    assert daily_stats["license_issue_activations"] == 1
    assert daily_stats["last_activation_at"] == daily_activation.activated_at.isoformat()


def test_auto_trade_engine_uses_ai_inference(monkeypatch) -> None:
    adapter = _make_sync_adapter()
    signals: list[float] = []
    signal_payloads: list[dict] = []

    def _collect_signals(evt_or_batch):
        batch = evt_or_batch if isinstance(evt_or_batch, list) else [evt_or_batch]
        for evt in batch:
            signal_payloads.append(evt.payload)
            signals.append(evt.payload["direction"])

    adapter.subscribe(EventType.SIGNAL, _collect_signals)

    journal = InMemoryTradingDecisionJournal()
    cfg = AutoTradeConfig(
        symbol="BTCUSDT",
        qty=0.25,
        regime_window=6,
        activation_threshold=0.0,
        breakout_window=3,
        mean_reversion_window=4,
        mean_reversion_z=0.5,
    )
    inference = _InferenceStub(
        ModelScore(expected_return_bps=120.0, success_probability=0.8)
    )
    engine = AutoTradeEngine(
        adapter,
        lambda *_: None,
        cfg,
        decision_inference=inference,
        decision_journal=journal,
        decision_journal_context={
            "environment": "paper",
            "portfolio": "autotrader",
            "risk_profile": "trend",
        },
    )
    engine.apply_params({"fast": 2, "slow": 5})

    base_time = 1_700_400_000.0
    current_time = {"value": base_time}

    def fake_time() -> float:
        return current_time["value"]

    monkeypatch.setattr("bot_core.trading.auto_trade.time.time", fake_time)
    monkeypatch.setattr("bot_core.events.emitter.time.time", fake_time)

    closes = [10, 9, 8, 7, 6, 7, 8, 9, 10, 9, 8, 7, 6]
    for px in closes:
        adapter.push_market_tick("BTCUSDT", price=px)

    assert inference.calls, "expected inference to be invoked"
    assert signal_payloads, "expected at least one signal payload"
    latest_payload = signal_payloads[-1]

    def _compute_base_signal(payload: dict) -> float:
        ai_meta_payload = payload["metadata"]["ai_inference"]
        scaling = float(ai_meta_payload["weight_scaling"])
        if scaling == 0.0:
            return 0.0
        scaled_weights = payload["weights"]
        base_weights_from_payload = {
            name: value / scaling for name, value in scaled_weights.items()
        }
        base_abs = sum(abs(value) for value in base_weights_from_payload.values())
        if base_abs == 0.0:
            return 0.0
        signals_payload = payload["signals"]
        base_numerator = sum(
            base_weights_from_payload.get(name, 0.0) * signals_payload.get(name, 0.0)
            for name in base_weights_from_payload
        )
        return base_numerator / base_abs if base_abs else 0.0

    ai_meta = latest_payload["metadata"]["ai_inference"]
    assert ai_meta["expected_return_bps"] == pytest.approx(120.0)
    assert ai_meta["weight_scaling"] > 1.0
    regime_key = latest_payload["regime"]
    base_weights = cfg.strategy_weights[regime_key]
    expected_trend = base_weights["trend_following"] * ai_meta["weight_scaling"]
    assert latest_payload["weights"]["trend_following"] == pytest.approx(expected_trend)
    base_signal = _compute_base_signal(latest_payload)
    assert ai_meta["signal_before_adjustment"] == pytest.approx(base_signal)
    assert ai_meta["signal_after_clamp"] == pytest.approx(latest_payload["direction"])
    assert ai_meta["signal_after_adjustment"] * base_signal >= 0
    assert latest_payload["direction"] * base_signal >= 0
    assert abs(ai_meta["signal_after_adjustment"]) > abs(base_signal)
    assert abs(latest_payload["direction"]) > abs(base_signal)

    inference._score = ModelScore(expected_return_bps=-150.0, success_probability=0.8)
    for px in closes:
        adapter.push_market_tick("BTCUSDT", price=px)

    negative_payload = signal_payloads[-1]
    negative_meta = negative_payload["metadata"]["ai_inference"]
    assert negative_meta["weight_scaling"] < 1.0
    negative_base_signal = _compute_base_signal(negative_payload)
    assert negative_meta["signal_before_adjustment"] == pytest.approx(negative_base_signal)
    assert negative_meta["signal_after_clamp"] == pytest.approx(negative_payload["direction"])
    assert negative_meta["signal_after_adjustment"] * negative_base_signal >= 0
    assert negative_payload["direction"] * negative_base_signal >= 0
    assert abs(negative_meta["signal_after_adjustment"]) < abs(negative_base_signal)
    assert abs(negative_payload["direction"]) < abs(negative_base_signal)
    exported = list(journal.export())
    assert len(exported) >= 2, "expected inference journal events"
    positive_event = next(
        event
        for event in reversed(exported)
        if float(event.get("expected_return_bps", 0.0)) > 0.0
    )
    assert positive_event["event"] == "ai_inference"
    assert positive_event["environment"] == "paper"
    assert float(positive_event["expected_return_bps"]) == pytest.approx(120.0)
    assert float(positive_event["signal_before_adjustment"]) == pytest.approx(base_signal)
    assert float(positive_event["signal_after_adjustment"]) == pytest.approx(
        ai_meta["signal_after_adjustment"]
    )
    assert float(positive_event["signal_after_clamp"]) == pytest.approx(
        ai_meta["signal_after_clamp"]
    )
    negative_event = next(
        event
        for event in reversed(exported)
        if float(event.get("expected_return_bps", 0.0)) < 0.0
    )
    assert negative_event["event"] == "ai_inference"
    assert negative_event["environment"] == "paper"
    assert float(negative_event["expected_return_bps"]) == pytest.approx(-150.0)
    assert float(negative_event["signal_before_adjustment"]) == pytest.approx(
        negative_base_signal
    )
    assert float(negative_event["signal_after_adjustment"]) == pytest.approx(
        negative_meta["signal_after_adjustment"]
    )
    assert float(negative_event["signal_after_clamp"]) == pytest.approx(
        negative_meta["signal_after_clamp"]
    )
    assert any(key.startswith("price_") for key in inference.calls[-1])
    assert inference.contexts[-1]["regime"] == regime_key
