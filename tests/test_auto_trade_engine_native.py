from __future__ import annotations

import pytest

from dataclasses import replace
from typing import Any

import pandas as pd

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
from bot_core.trading.regime_workflow import RegimeSwitchDecision
from bot_core.trading.strategies import StrategyCatalog, StrategyPlugin


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
    assert {"trend_following", "day_trading", "mean_reversion", "arbitrage"} <= set(signal_detail)
    assert signal_detail["daily_breakout"] == signal_detail["day_trading"]


class _ConstantTrendStrategy(StrategyPlugin):
    name = "trend_following"
    description = "Stały sygnał dodatni do testów."

    def generate(self, indicators, params, *, market_data=None):  # type: ignore[override]
        series = indicators.ema_fast.copy()
        series[:] = 0.75
        return series


class _ConstantMeanStrategy(StrategyPlugin):
    name = "mean_reversion"
    description = "Stały sygnał średni do testów."

    def generate(self, indicators, params, *, market_data=None):  # type: ignore[override]
        series = indicators.ema_fast.copy()
        series[:] = 0.4
        return series


class _WorkflowStub:
    def __init__(self, decision: RegimeSwitchDecision, catalog: StrategyCatalog) -> None:
        classifier = MarketRegimeClassifier()
        self.classifier = classifier
        self.history = RegimeHistory(thresholds_loader=classifier.thresholds_loader)
        self.history.reload_thresholds(thresholds=classifier.thresholds_snapshot())
        self.catalog = catalog
        self._decision = decision
        self.last_decision = decision
        self.calls: list[dict[str, Any]] = []

    def decide(
        self,
        market_data: pd.DataFrame,
        base_parameters: TradingParameters,
        *,
        symbol: str | None = None,
        parameter_overrides=None,
    ) -> RegimeSwitchDecision:
        self.calls.append(
            {
                "market_data": market_data.copy(),
                "base_parameters": base_parameters,
                "symbol": symbol,
                "parameter_overrides": parameter_overrides,
            }
        )
        self.history.update(self._decision.assessment)
        return self._decision


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


def test_manual_risk_freeze_controls(monkeypatch) -> None:
    adapter = _make_sync_adapter()
    statuses = _collect_status_payloads(adapter)
    engine = AutoTradeEngine(adapter, lambda *args, **kwargs: None, AutoTradeConfig())

    base_time = 1_700_400_000.0
    now_ref = {"value": base_time}

    def fake_time() -> float:
        return now_ref["value"]

    monkeypatch.setattr("bot_core.trading.auto_trade.time.time", fake_time)
    monkeypatch.setattr("bot_core.events.emitter.time.time", fake_time)

    engine.freeze_risk(duration=30, reason="ui_hold")
    freeze_event = next(p for p in statuses if p["status"] == "risk_freeze")
    assert freeze_event["detail"]["reason"] == "ui_hold"
    assert freeze_event["detail"]["source"] == "manual"
    assert freeze_event["detail"]["until"] == pytest.approx(base_time + 30)

    snapshot = engine.snapshot()
    assert snapshot.risk.manual_active
    assert snapshot.risk.manual_reason == "ui_hold"
    assert snapshot.risk.manual_until == pytest.approx(base_time + 30)
    assert snapshot.risk.combined_until == pytest.approx(base_time + 30)

    now_ref["value"] = base_time + 10
    engine.freeze_risk(duration=45, reason="ui_extend")
    extend_event = next(p for p in statuses if p["status"] == "risk_freeze_extend")
    assert extend_event["detail"]["extended_from"] == pytest.approx(freeze_event["detail"]["until"])
    assert extend_event["detail"]["until"] == pytest.approx(now_ref["value"] + 45)
    assert extend_event["detail"]["previous_reason"] == "ui_hold"
    assert extend_event["detail"]["source"] == "manual"

    snapshot = engine.snapshot()
    assert snapshot.risk.manual_reason == "ui_extend"
    assert snapshot.risk.manual_until == pytest.approx(now_ref["value"] + 45)

    now_ref["value"] += 5
    engine.unfreeze_risk()
    unfreeze_event = next(p for p in statuses if p["status"] == "risk_unfreeze")
    assert unfreeze_event["detail"]["reason"] == "ui_extend"
    assert unfreeze_event["detail"]["source"] == "manual"
    assert unfreeze_event["detail"]["released_at"] == pytest.approx(now_ref["value"])

    snapshot = engine.snapshot()
    assert not snapshot.risk.manual_active
    assert snapshot.risk.manual_until is None
    assert snapshot.risk.combined_until == pytest.approx(snapshot.risk.auto_until or 0.0)


def test_manual_risk_freeze_invalid_duration(monkeypatch) -> None:
    adapter = _make_sync_adapter()
    engine = AutoTradeEngine(adapter, lambda *args, **kwargs: None, AutoTradeConfig())

    monkeypatch.setattr("bot_core.trading.auto_trade.time.time", lambda: 1_700_500_000.0)
    with pytest.raises(ValueError):
        engine.freeze_risk(duration=0)
    with pytest.raises(ValueError):
        engine.freeze_risk(duration=-5)


def test_configure_strategy_weights_updates_workflow() -> None:
    adapter = _make_sync_adapter()
    engine = AutoTradeEngine(adapter, lambda *args: None)

    engine.configure_strategy_weights(
        {
            MarketRegime.TREND: {"trend_following": 3.0, "arbitrage": 1.0},
            "mean_reversion": {"mean_reversion": 2.0, "arbitrage": 2.0},
        },
        replace=True,
    )

    workflow = engine._regime_workflow
    assert workflow is not None
    trend_weights = workflow.default_strategy_weights[MarketRegime.TREND]
    assert trend_weights == {"trend_following": 3.0, "arbitrage": 1.0}
    mean_weights = workflow.default_strategy_weights[MarketRegime.MEAN_REVERSION]
    assert mean_weights == {"mean_reversion": 2.0, "arbitrage": 2.0}

    snapshot = engine.snapshot()
    assert snapshot.strategy_weights["trend_following"] == pytest.approx(0.75)
    assert snapshot.strategy_weights["arbitrage"] == pytest.approx(0.25)
    assert engine.cfg.strategy_weights["trend"]["trend_following"] == pytest.approx(3.0)


def test_configure_strategy_weights_merges_without_replace() -> None:
    adapter = _make_sync_adapter()
    engine = AutoTradeEngine(adapter, lambda *args: None)

    base_trend = engine._strategy_weights.weights_for(MarketRegime.TREND, normalize=False)
    assert base_trend["arbitrage"] == pytest.approx(0.15)

    engine.configure_strategy_weights({"trend": {"arbitrage": 0.45}})

    trend_weights = engine._strategy_weights.weights_for(MarketRegime.TREND, normalize=False)
    assert trend_weights["arbitrage"] == pytest.approx(0.45)
    assert "day_trading" in trend_weights


def test_configure_parameter_overrides_updates_workflow() -> None:
    adapter = _make_sync_adapter()
    engine = AutoTradeEngine(adapter, lambda *args: None)

    engine.configure_parameter_overrides(
        {
            MarketRegime.TREND: {"signal_threshold": 0.21},
            "daily": {"day_trading_momentum_window": 12},
        }
    )

    workflow = engine._regime_workflow
    assert workflow is not None
    overrides = workflow.default_parameter_overrides
    assert overrides[MarketRegime.TREND]["signal_threshold"] == pytest.approx(0.21)
    assert overrides[MarketRegime.DAILY]["day_trading_momentum_window"] == 12
    assert engine.cfg.regime_parameter_overrides["trend"]["signal_threshold"] == pytest.approx(0.21)

    snapshot = engine.snapshot()
    assert (
        snapshot.regime_parameter_overrides[MarketRegime.TREND.value]["signal_threshold"]
        == pytest.approx(0.21)
    )


def test_configure_parameter_overrides_replace() -> None:
    adapter = _make_sync_adapter()
    engine = AutoTradeEngine(adapter, lambda *args: None)

    engine.configure_parameter_overrides(
        {MarketRegime.MEAN_REVERSION: {"rsi_oversold": 47, "rsi_overbought": 53}},
        replace=True,
    )

    overrides = engine._parameter_overrides[MarketRegime.MEAN_REVERSION]
    assert overrides == {"rsi_oversold": 47, "rsi_overbought": 53}
    workflow = engine._regime_workflow
    assert workflow is not None
    assert (
        workflow.default_parameter_overrides[MarketRegime.MEAN_REVERSION]
        == {"rsi_oversold": 47, "rsi_overbought": 53}
    )


def test_parameter_overrides_applied_without_workflow(monkeypatch) -> None:
    adapter = _make_sync_adapter()
    engine = AutoTradeEngine(adapter, lambda *args: None)

    engine.configure_parameter_overrides(
        {MarketRegime.TREND: {"signal_threshold": 0.44}},
        replace=True,
    )
    engine._regime_workflow = None
    engine._last_regime_decision = None

    assessment = MarketRegimeAssessment(
        regime=MarketRegime.TREND,
        confidence=0.5,
        risk_score=0.2,
        metrics={},
        symbol="BTCUSDT",
    )

    monkeypatch.setattr(engine, "_classify_regime", lambda frame: assessment)

    frame = pd.DataFrame({"close": [100.0, 100.5, 101.0]})
    base_params = engine._build_base_trading_parameters()
    _, _, _, params = engine._evaluate_regime_decision(frame, base_params)

    assert params.signal_threshold == pytest.approx(0.44)


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
    decision = RegimeSwitchDecision(
        regime=assessment.regime,
        assessment=assessment,
        summary=None,
        weights={"trend_following": 0.2, "mean_reversion": 0.8},
        parameters=decision_params,
        timestamp=pd.Timestamp.utcnow(),
    )
    workflow = _WorkflowStub(decision, catalog)

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
    last_call = workflow.calls[-1]
    call_params = last_call["base_parameters"]
    assert last_call["parameter_overrides"] == engine._parameter_overrides
    assert call_params.ema_fast_period == 3
    assert call_params.ema_slow_period == 8
    assert orders, "Expected workflow-driven decision to result in an order"
    assert signal_payloads, "Expected workflow-driven signal payloads"
    latest_signal = signal_payloads[-1]
    assert latest_signal["weights"] == decision.weights
    params_payload = latest_signal["strategy_parameters"]
    assert params_payload["ema_fast_period"] == decision_params.ema_fast_period
    assert params_payload["ema_slow_period"] == decision_params.ema_slow_period
    assert params_payload["ensemble_weights"] == decision_params.ensemble_weights
    assert engine.last_regime_decision is decision

    entry_statuses = [st for st in statuses if st.get("status") in {"entry_long", "entry_short"}]
    assert entry_statuses, "Expected entry status to include workflow metadata"
    last_entry = entry_statuses[-1]
    assert last_entry["detail"]["regime"]["regime"] == assessment.regime.value
    assert "summary" in last_entry["detail"]


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
    assert engine._risk_frozen_until == pytest.approx(base_time + cfg.risk_freeze_seconds)
    assert statuses and statuses[-1]["status"] == "auto_risk_freeze"
    first_detail = statuses[-1]["detail"]
    assert first_detail["until"] == pytest.approx(engine._risk_frozen_until)

    history._summary = _DummySummary(RiskLevel.CRITICAL, 0.95)
    current_time["value"] = base_time + 10

    engine._sync_freeze_state()

    assert engine._auto_risk_frozen is True
    assert statuses[-1]["status"] == "auto_risk_freeze_extend"
    extend_detail = statuses[-1]["detail"]
    assert extend_detail["extended_from"] == pytest.approx(first_detail["until"])
    assert extend_detail["until"] == pytest.approx(engine._risk_frozen_until)

    history._summary = _DummySummary(RiskLevel.CALM, 0.1)
    current_time["value"] = engine._risk_frozen_until + 5

    engine._sync_freeze_state()

    assert engine._auto_risk_frozen is False
    assert engine._risk_frozen_until == 0.0
    assert statuses[-1]["status"] == "auto_risk_unfreeze"


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
    decision = RegimeSwitchDecision(
        regime=assessment.regime,
        assessment=assessment,
        summary=None,
        weights=dict(decision_params.ensemble_weights),
        parameters=decision_params,
        timestamp=pd.Timestamp.utcnow(),
    )
    workflow = _WorkflowStub(decision, catalog)

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
    assert snapshot.trading_parameters == decision_params
    assert snapshot.strategy_weights == decision_params.ensemble_weights
    assert isinstance(snapshot.risk, RiskFreezeSnapshot)
    assert snapshot.risk.manual_active is True
    assert snapshot.risk.manual_reason == "stress"
    assert snapshot.risk.auto_active is True
    assert snapshot.risk.auto_risk_level is RiskLevel.CRITICAL
    assert snapshot.regime_decision == decision
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
    assert refreshed.strategy_weights == decision_params.ensemble_weights
    assert refreshed.risk.combined_until >= snapshot.risk.combined_until
