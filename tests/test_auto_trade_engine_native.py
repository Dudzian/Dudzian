from __future__ import annotations

import pytest

from bot_core.ai.regime import MarketRegime, RiskLevel
from bot_core.events import EmitterAdapter, Event, EventType
from bot_core.trading.auto_trade import AutoTradeConfig, AutoTradeEngine


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


def test_auto_trade_engine_generates_orders_and_signals(monkeypatch) -> None:
    adapter = _make_sync_adapter()
    orders: list[tuple[str, float]] = []
    signals: list[float] = []
    statuses: list[str] = []

    def _collect_signals(evt_or_batch):
        batch = evt_or_batch if isinstance(evt_or_batch, list) else [evt_or_batch]
        for evt in batch:
            signals.append(float(evt.payload["direction"]))

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

    assert orders == [("buy", 0.5), ("sell", 0.5)]
    assert any(sig > 0 for sig in signals)
    assert any(sig < 0 for sig in signals)
    assert "params_applied" in statuses


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

    engine = AutoTradeEngine(adapter, lambda *_: None, cfg)

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
    initial_until = engine._risk_frozen_until
    assert initial_until == pytest.approx(base_time + cfg.risk_freeze_seconds)
    assert statuses and statuses[-1]["status"] == "auto_risk_freeze"
    first_detail = statuses[-1]["detail"]
    assert first_detail["until"] == pytest.approx(engine._risk_frozen_until)
    assert first_detail["reason"] == "risk_level_and_score_threshold"
    assert first_detail["triggered_at"] == pytest.approx(base_time)
    assert first_detail["last_extension_at"] == pytest.approx(base_time)
    assert first_detail["released_at"] is None
    assert first_detail["frozen_for"] is None

    statuses_before = len(statuses)
    history._summary = _DummySummary(RiskLevel.CRITICAL, 0.9)
    current_time["value"] = base_time + 10
    engine._sync_freeze_state()
    assert len(statuses) == statuses_before  # brak dodatkowego alertu przy niezmienionym ryzyku
    refreshed_until = current_time["value"] + cfg.risk_freeze_seconds
    assert engine._risk_frozen_until == pytest.approx(refreshed_until)
    assert engine._risk_frozen_until > initial_until

    history._summary = _DummySummary(RiskLevel.CRITICAL, 0.96)
    current_time["value"] = base_time + 18

    previous_until = engine._risk_frozen_until
    engine._sync_freeze_state()

    assert engine._auto_risk_frozen is True
    assert statuses[-1]["status"] == "auto_risk_freeze_extend"
    extend_detail = statuses[-1]["detail"]
    assert extend_detail["extended_from"] == pytest.approx(previous_until)
    assert extend_detail["extended_from"] > first_detail["until"]
    assert extend_detail["until"] == pytest.approx(engine._risk_frozen_until)
    assert extend_detail["reason"] == "risk_score_increase"
    assert extend_detail["triggered_at"] == pytest.approx(first_detail["triggered_at"])
    assert extend_detail["last_extension_at"] == pytest.approx(current_time["value"])
    assert extend_detail["released_at"] is None
    assert extend_detail["frozen_for"] is None

    history._summary = _DummySummary(RiskLevel.BALANCED, 0.12)
    current_time["value"] = base_time + 22

    engine._sync_freeze_state()

    assert engine._auto_risk_frozen is False
    assert engine._risk_frozen_until == 0.0
    assert statuses[-1]["status"] == "auto_risk_unfreeze"
    unfreeze_detail = statuses[-1]["detail"]
    assert unfreeze_detail["reason"] == "risk_recovered"
    assert unfreeze_detail["risk_level"] == RiskLevel.BALANCED.value
    assert unfreeze_detail["risk_score"] == pytest.approx(0.12)
    assert unfreeze_detail["triggered_at"] == pytest.approx(base_time)
    assert unfreeze_detail["released_at"] == pytest.approx(current_time["value"])
    assert unfreeze_detail["frozen_for"] == pytest.approx(current_time["value"] - base_time)
    assert unfreeze_detail["last_extension_at"] == pytest.approx(current_time["value"] - 4)


def test_auto_risk_freeze_extends_near_expiry(monkeypatch) -> None:
    adapter = _make_sync_adapter()
    statuses: list[dict] = []

    def _collect(evt_or_batch):
        batch = evt_or_batch if isinstance(evt_or_batch, list) else [evt_or_batch]
        for evt in batch:
            statuses.append(evt.payload)

    adapter.subscribe(EventType.AUTOTRADE_STATUS, _collect)

    cfg = AutoTradeConfig(
        symbol="BTCUSDT",
        risk_freeze_seconds=40,
        auto_risk_freeze=True,
        auto_risk_freeze_level=RiskLevel.WATCH,
        auto_risk_freeze_score=0.2,
    )

    engine = AutoTradeEngine(adapter, lambda *_: None, cfg)

    base_time = 1_700_500_000.0
    current_time = {"value": base_time}

    def fake_time() -> float:
        return current_time["value"]

    monkeypatch.setattr("bot_core.trading.auto_trade.time.time", fake_time)
    monkeypatch.setattr("bot_core.events.emitter.time.time", fake_time)

    history = _DummyHistory(_DummySummary(RiskLevel.ELEVATED, 0.6))
    engine._regime_history = history  # type: ignore[assignment]

    engine._sync_freeze_state()
    assert statuses[-1]["status"] == "auto_risk_freeze"

    current_time["value"] = base_time + 32
    engine._sync_freeze_state()

    extend_events = [payload for payload in statuses if payload["status"] == "auto_risk_freeze_extend"]
    assert extend_events, "powinno dojść do wydłużenia przy wygasającym zamrożeniu"
    extend_detail = extend_events[-1]["detail"]
    assert extend_detail["reason"] == "expiry_near"
    assert extend_detail["until"] == pytest.approx(engine._risk_frozen_until)
    assert extend_detail["triggered_at"] == pytest.approx(base_time)
    assert extend_detail["last_extension_at"] == pytest.approx(current_time["value"])
    assert extend_detail["released_at"] is None
    assert extend_detail["frozen_for"] is None


def test_auto_risk_freeze_unfreezes_on_expiry_when_disabled(monkeypatch) -> None:
    adapter = _make_sync_adapter()
    statuses: list[dict] = []

    def _collect(evt_or_batch):
        batch = evt_or_batch if isinstance(evt_or_batch, list) else [evt_or_batch]
        for evt in batch:
            statuses.append(evt.payload)

    adapter.subscribe(EventType.AUTOTRADE_STATUS, _collect)

    cfg = AutoTradeConfig(
        symbol="BTCUSDT",
        risk_freeze_seconds=20,
        auto_risk_freeze=True,
        auto_risk_freeze_level=RiskLevel.WATCH,
        auto_risk_freeze_score=0.2,
    )

    engine = AutoTradeEngine(adapter, lambda *_: None, cfg)

    base_time = 1_700_600_000.0
    current_time = {"value": base_time}

    def fake_time() -> float:
        return current_time["value"]

    monkeypatch.setattr("bot_core.trading.auto_trade.time.time", fake_time)
    monkeypatch.setattr("bot_core.events.emitter.time.time", fake_time)

    history = _DummyHistory(_DummySummary(RiskLevel.CRITICAL, 0.95))
    engine._regime_history = history  # type: ignore[assignment]

    engine._sync_freeze_state()
    assert statuses[-1]["status"] == "auto_risk_freeze"
    triggered_at = engine._auto_risk_state.triggered_at

    engine.cfg.auto_risk_freeze = False
    current_time["value"] = engine._risk_frozen_until + 5

    engine._sync_freeze_state()

    assert engine._auto_risk_frozen is False
    assert engine._risk_frozen_until == 0.0
    assert statuses[-1]["status"] == "auto_risk_unfreeze"
    detail = statuses[-1]["detail"]
    assert detail["reason"] == "expired"
    assert detail["risk_level"] == RiskLevel.CRITICAL.value
    assert detail["risk_score"] == pytest.approx(0.95)
    assert detail["triggered_at"] == pytest.approx(triggered_at)
    assert detail["released_at"] == pytest.approx(current_time["value"])
    assert detail["frozen_for"] == pytest.approx(current_time["value"] - triggered_at)
    assert detail["last_extension_at"] == pytest.approx(triggered_at)


def test_auto_risk_freeze_reason_selection(monkeypatch) -> None:
    cfg = AutoTradeConfig(
        symbol="BTCUSDT",
        risk_freeze_seconds=15,
        auto_risk_freeze=True,
        auto_risk_freeze_level=RiskLevel.CRITICAL,
        auto_risk_freeze_score=0.5,
    )

    base_time = 1_701_000_000.0
    current_time = {"value": base_time}

    def fake_time() -> float:
        return current_time["value"]

    monkeypatch.setattr("bot_core.trading.auto_trade.time.time", fake_time)
    monkeypatch.setattr("bot_core.events.emitter.time.time", fake_time)

    def _run_case(level: RiskLevel, score: float) -> dict:
        adapter = _make_sync_adapter()
        statuses: list[dict] = []

        def _collect(evt_or_batch):
            batch = evt_or_batch if isinstance(evt_or_batch, list) else [evt_or_batch]
            for evt in batch:
                statuses.append(evt.payload)

        adapter.subscribe(EventType.AUTOTRADE_STATUS, _collect)

        engine = AutoTradeEngine(adapter, lambda *_: None, cfg)
        engine._regime_history = _DummyHistory(_DummySummary(level, score))  # type: ignore[assignment]
        current_time["value"] = base_time
        engine._sync_freeze_state()
        assert statuses and statuses[-1]["status"] == "auto_risk_freeze"
        return statuses[-1]["detail"]

    score_detail = _run_case(RiskLevel.BALANCED, 0.72)
    assert score_detail["reason"] == "risk_score_threshold"
    assert score_detail["risk_level"] == RiskLevel.BALANCED.value

    level_detail = _run_case(RiskLevel.CRITICAL, 0.1)
    assert level_detail["reason"] == "risk_level_threshold"
    assert level_detail["risk_score"] == pytest.approx(0.1)


def test_manual_risk_freeze_telemetry(monkeypatch) -> None:
    adapter = _make_sync_adapter()
    statuses: list[dict] = []

    def _collect(evt_or_batch):
        batch = evt_or_batch if isinstance(evt_or_batch, list) else [evt_or_batch]
        for evt in batch:
            statuses.append(evt.payload)

    adapter.subscribe(EventType.AUTOTRADE_STATUS, _collect)

    cfg = AutoTradeConfig(
        symbol="BTCUSDT",
        risk_freeze_seconds=30,
        auto_risk_freeze=False,
    )

    engine = AutoTradeEngine(adapter, lambda *_: None, cfg)

    base_time = 1_702_000_000.0
    current_time = {"value": base_time}

    def fake_time() -> float:
        return current_time["value"]

    monkeypatch.setattr("bot_core.trading.auto_trade.time.time", fake_time)
    monkeypatch.setattr("bot_core.events.emitter.time.time", fake_time)

    adapter.publish(EventType.RISK_ALERT, {"symbol": "BTCUSDT", "kind": "risk_limit"})

    assert statuses[-1]["status"] == "risk_freeze"
    freeze_detail = statuses[-1]["detail"]
    assert freeze_detail["reason"] == "risk_limit"
    assert freeze_detail["until"] == pytest.approx(engine._manual_risk_frozen_until)
    assert freeze_detail["triggered_at"] == pytest.approx(base_time)
    assert freeze_detail["last_extension_at"] == pytest.approx(base_time)
    assert freeze_detail["released_at"] is None
    assert freeze_detail["frozen_for"] is None

    previous_until = engine._manual_risk_frozen_until
    current_time["value"] = base_time + 10
    adapter.publish(
        EventType.RISK_ALERT,
        {"symbol": "BTCUSDT", "kind": "risk_limit_escalated"},
    )

    assert statuses[-1]["status"] == "risk_freeze_extend"
    extend_detail = statuses[-1]["detail"]
    assert extend_detail["reason"] == "risk_limit_escalated"
    assert extend_detail["extended_from"] == pytest.approx(previous_until)
    assert extend_detail["until"] == pytest.approx(engine._manual_risk_frozen_until)
    assert extend_detail["triggered_at"] == pytest.approx(base_time)
    assert extend_detail["last_extension_at"] == pytest.approx(current_time["value"])
    assert extend_detail["released_at"] is None
    assert extend_detail["frozen_for"] is None
    assert extend_detail.get("previous_reason") == "risk_limit"

    current_time["value"] = engine._manual_risk_frozen_until + 5
    engine._sync_freeze_state()

    assert engine._manual_risk_frozen_until == 0.0
    assert statuses[-1]["status"] == "risk_unfreeze"
    unfreeze_detail = statuses[-1]["detail"]
    assert unfreeze_detail["reason"] == "expired"
    assert unfreeze_detail["triggered_at"] == pytest.approx(base_time)
    assert unfreeze_detail["released_at"] == pytest.approx(current_time["value"])
    assert unfreeze_detail["frozen_for"] == pytest.approx(
        current_time["value"] - base_time
    )
    assert unfreeze_detail["last_extension_at"] == pytest.approx(base_time + 10)
    assert unfreeze_detail["source_reason"] == "risk_limit_escalated"
