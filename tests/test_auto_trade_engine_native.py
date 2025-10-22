from __future__ import annotations

import time
from copy import deepcopy

import pandas as pd
import pytest

from bot_core.ai.regime import (
    MarketRegime,
    MarketRegimeAssessment,
    RegimeHistory,
    RegimeSnapshot,
    RegimeSummary,
    RiskLevel,
)
from bot_core.events import EmitterAdapter, EventType
from bot_core.trading.auto_trade import AutoTradeConfig, AutoTradeEngine


def test_auto_trade_engine_generates_orders_and_signals() -> None:
    adapter = EmitterAdapter()
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

    closes = [10, 9, 8, 7, 6, 7, 8, 9, 10, 9, 8, 7, 6]
    for px in closes:
        adapter.push_market_tick("BTCUSDT", price=px)
    time.sleep(0.2)

    assert orders == [("buy", 0.5), ("sell", 0.5)]
    assert any(sig > 0 for sig in signals)
    assert any(sig < 0 for sig in signals)
    assert "params_applied" in statuses


def test_auto_trade_engine_emits_regime_update_with_metrics() -> None:
    adapter = EmitterAdapter()
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
    for idx in range(80):
        close = base_price + idx * 0.4
        bar = {
            "open_time": float(idx),
            "close": close,
            "high": close * 1.002,
            "low": close * 0.998,
            "volume": 1200.0 + idx * 5.0,
        }
        adapter.publish(EventType.MARKET_TICK, {"symbol": "BTCUSDT", "bar": bar})

    time.sleep(0.4)

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
    summary = detail.get("summary")
    assert summary is not None
    assert summary["risk_level"] in {level.value for level in RiskLevel}
    assert "history" in summary and summary["history"]
    thresholds = detail.get("thresholds")
    assert thresholds is not None
    assert "market_regime" in thresholds
    assert "risk_level" in thresholds["market_regime"]


def _make_summary(
    *,
    risk_level: RiskLevel,
    risk_score: float,
    history: tuple[RegimeSnapshot, ...] | None = None,
) -> RegimeSummary:
    snapshot_history = history or (
        RegimeSnapshot(
            regime=MarketRegime.TREND,
            confidence=0.7,
            risk_score=risk_score,
            drawdown=0.25,
            volatility=0.04,
            volume_trend=0.0,
            volatility_ratio=1.5,
            return_skew=0.0,
            return_kurtosis=0.0,
            volume_imbalance=0.0,
        ),
    )
    return RegimeSummary(
        regime=MarketRegime.TREND,
        confidence=0.6,
        risk_score=risk_score,
        stability=0.3,
        risk_trend=0.1,
        risk_level=risk_level,
        risk_volatility=0.2,
        regime_persistence=0.4,
        transition_rate=0.6,
        confidence_trend=0.0,
        confidence_volatility=0.2,
        regime_streak=3,
        instability_score=0.7,
        confidence_decay=0.1,
        avg_drawdown=0.25,
        avg_volume_trend=-0.1,
        drawdown_pressure=0.7,
        liquidity_pressure=0.6,
        volatility_ratio=1.4,
        regime_entropy=0.5,
        tail_risk_index=0.6,
        shock_frequency=0.4,
        volatility_of_volatility=0.03,
        stress_index=0.7,
        severe_event_rate=0.4,
        cooldown_score=0.2,
        recovery_potential=0.3,
        resilience_score=0.35,
        stress_balance=0.5,
        liquidity_gap=0.2,
        confidence_resilience=0.4,
        stress_projection=0.3,
        stress_momentum=0.25,
        liquidity_trend=-0.05,
        confidence_fragility=0.2,
        volatility_trend=0.1,
        drawdown_trend=0.05,
        volume_trend_volatility=0.02,
        stability_projection=0.3,
        degradation_score=0.2,
        skewness_bias=0.0,
        kurtosis_excess=0.0,
        volume_imbalance=0.0,
        distribution_pressure=0.3,
        history=snapshot_history,
    )


def test_auto_trade_engine_emits_on_risk_level_change() -> None:
    adapter = EmitterAdapter()
    statuses: list[dict] = []

    def _collect(evt_or_batch):
        batch = evt_or_batch if isinstance(evt_or_batch, list) else [evt_or_batch]
        for evt in batch:
            statuses.append(evt.payload)

    adapter.subscribe(EventType.AUTOTRADE_STATUS, _collect)

    cfg = AutoTradeConfig(symbol="BTCUSDT", regime_window=20)
    engine = AutoTradeEngine(adapter, lambda *_: None, cfg)

    thresholds = {
        "market_regime": {
            "metrics": {},
            "risk_score": {},
            "risk_level": {
                "scales": {},
                "critical": {"risk_score": 0.8},
                "elevated": {"risk_score": 0.6},
                "watch": {"confidence_volatility": 0.2, "risk_score": 0.4},
                "balanced": {"risk_score": 0.45},
                "calm": {"risk_score": 0.25},
            },
        }
    }

    def _assessment(
        risk_score: float,
        *,
        drawdown: float,
        volatility: float,
        volume_trend: float,
        volatility_ratio: float,
    ) -> MarketRegimeAssessment:
        return MarketRegimeAssessment(
            regime=MarketRegime.TREND,
            confidence=0.6,
            risk_score=risk_score,
            metrics={
                "trend_strength": 0.02,
                "volatility": volatility,
                "momentum": 0.01,
                "autocorr": -0.05,
                "intraday_vol": volatility,
                "drawdown": drawdown,
                "volatility_ratio": volatility_ratio,
                "volume_trend": volume_trend,
                "return_skew": 0.0,
                "return_kurtosis": 0.0,
                "volume_imbalance": 0.0,
            },
            symbol="BTCUSDT",
        )

    assessments = [
        _assessment(0.15, drawdown=0.05, volatility=0.01, volume_trend=0.05, volatility_ratio=1.0),
        _assessment(0.95, drawdown=0.35, volatility=0.05, volume_trend=-0.35, volatility_ratio=1.65),
    ]

    class _ClassifierStub:
        def __init__(self) -> None:
            self._queue = list(assessments)
            self._thresholds = deepcopy(thresholds)

        @property
        def thresholds_loader(self):
            return lambda: deepcopy(self._thresholds)

        def thresholds_snapshot(self):
            return deepcopy(self._thresholds)

        def assess(
            self,
            market_data: pd.DataFrame,
            *,
            price_col: str = "close",
            symbol: str | None = None,
        ) -> MarketRegimeAssessment:
            assert self._queue, "No more assessments configured"
            return self._queue.pop(0)

    stub = _ClassifierStub()
    history = RegimeHistory(thresholds_loader=stub.thresholds_loader)
    history.decay = 0.1
    history.reload_thresholds(thresholds=stub.thresholds_snapshot())
    engine.set_regime_components(classifier=stub, history=history)

    frame = pd.DataFrame(
        {
            "close": [100.0 + idx for idx in range(4)],
            "high": [101.0 + idx for idx in range(4)],
            "low": [99.0 + idx for idx in range(4)],
            "volume": [1000.0 + idx * 10 for idx in range(4)],
        }
    )

    engine._classify_regime(frame)
    time.sleep(0.05)
    engine._classify_regime(frame)
    time.sleep(0.1)

    regime_updates = [payload for payload in statuses if payload["status"] == "regime_update"]
    assert len(regime_updates) == 2
    first_detail = regime_updates[0]["detail"]
    second_detail = regime_updates[1]["detail"]
    assert first_detail["summary"]["risk_level"] == RiskLevel.CALM.value
    assert second_detail["summary"]["risk_level"] == RiskLevel.CRITICAL.value


def test_auto_trade_config_validates_history_parameters() -> None:
    with pytest.raises(ValueError):
        AutoTradeConfig(symbol="BTCUSDT", regime_history_maxlen=0)
    with pytest.raises(ValueError):
        AutoTradeConfig(symbol="BTCUSDT", regime_history_decay=0.0)


def test_auto_trade_engine_uses_history_config_defaults() -> None:
    adapter = EmitterAdapter()
    cfg = AutoTradeConfig(
        symbol="NEARUSDT",
        regime_window=12,
        regime_history_maxlen=7,
        regime_history_decay=0.72,
    )
    engine = AutoTradeEngine(adapter, lambda *_: None, cfg)

    history = engine._regime_history  # type: ignore[attr-defined]
    assert isinstance(history, RegimeHistory)
    assert history.maxlen == 7
    assert history.decay == pytest.approx(0.72)


def test_auto_trade_engine_exposes_regime_state_accessors() -> None:
    adapter = EmitterAdapter()
    cfg = AutoTradeConfig(symbol="ETHUSDT", regime_window=10)
    engine = AutoTradeEngine(adapter, lambda *_: None, cfg)

    thresholds = {
        "market_regime": {
            "metrics": {},
            "risk_score": {"volatility_weight": 0.5},
            "risk_level": {
                "scales": {"skewness_bias": 1.0, "kurtosis_excess": 2.0, "volume_imbalance": 0.5},
                "critical": {"risk_score": 0.8},
                "elevated": {"risk_score": 0.6},
                "watch": {"risk_score": 0.4},
                "balanced": {"risk_score": 0.25},
                "calm": {"risk_score": 0.1},
            },
        }
    }

    metrics = {
        "trend_strength": 0.03,
        "volatility": 0.02,
        "momentum": 0.015,
        "autocorr": -0.1,
        "intraday_vol": 0.018,
        "drawdown": 0.05,
        "volatility_ratio": 1.05,
        "volume_trend": 0.12,
        "return_skew": 0.01,
        "return_kurtosis": 0.2,
        "volume_imbalance": 0.05,
    }

    assessment = MarketRegimeAssessment(
        regime=MarketRegime.TREND,
        confidence=0.7,
        risk_score=0.35,
        metrics=metrics,
        symbol="ETHUSDT",
    )

    class _ClassifierStub:
        def __init__(self) -> None:
            self._issued = False

        @property
        def thresholds_loader(self):
            return lambda: deepcopy(thresholds)

        def thresholds_snapshot(self):
            return deepcopy(thresholds)

        def assess(self, *_args, **_kwargs):
            if self._issued:
                raise AssertionError("Assessment already issued")
            self._issued = True
            return assessment

    stub = _ClassifierStub()
    history = RegimeHistory(thresholds_loader=stub.thresholds_loader)
    history.reload_thresholds(thresholds=stub.thresholds_snapshot())
    engine.set_regime_components(classifier=stub, history=history)

    frame = pd.DataFrame(
        {
            "close": [100.0 + idx for idx in range(6)],
            "high": [101.0 + idx for idx in range(6)],
            "low": [99.0 + idx for idx in range(6)],
            "volume": [1000.0 + idx * 10 for idx in range(6)],
        }
    )

    engine._classify_regime(frame)

    assessment_copy = engine.get_last_regime_assessment()
    assert assessment_copy is not None
    assert assessment_copy is not engine._last_regime
    assessment_copy.metrics = dict(assessment_copy.metrics)
    assessment_copy.metrics["volatility"] = 999.0
    assert engine._last_regime is not None
    assert engine._last_regime.metrics["volatility"] == metrics["volatility"]

    summary_one = engine.get_regime_summary()
    assert summary_one is not None
    summary_one.history[0].risk_score = 0.0
    assert engine._last_summary is not None
    summary_two = engine.get_regime_summary()
    assert summary_two is not None
    assert summary_two.history[0].risk_score == engine._last_summary.history[0].risk_score
    assert summary_two is not summary_one

    thresholds_copy = engine.get_regime_thresholds()
    thresholds_copy["market_regime"]["risk_level"]["critical"]["risk_score"] = 0.1
    fresh_thresholds = engine.get_regime_thresholds()
    assert (
        fresh_thresholds["market_regime"]["risk_level"]["critical"]["risk_score"]
        == thresholds["market_regime"]["risk_level"]["critical"]["risk_score"]
    )

    empty_engine = AutoTradeEngine(adapter, lambda *_: None, AutoTradeConfig(symbol="LTCUSDT"))
    assert empty_engine.get_last_regime_assessment() is None
    assert empty_engine.get_regime_summary() is None


def test_auto_trade_engine_accepts_custom_regime_components_on_init() -> None:
    adapter = EmitterAdapter()
    cfg = AutoTradeConfig(symbol="BNBUSDT", regime_window=8)

    thresholds = {
        "market_regime": {
            "metrics": {},
            "risk_score": {},
            "risk_level": {"critical": {"risk_score": 0.9}},
        }
    }

    assessments = [
        MarketRegimeAssessment(
            regime=MarketRegime.DAILY,
            confidence=0.55,
            risk_score=0.4,
            metrics={"trend_strength": 0.01, "volatility": 0.015},
            symbol="BNBUSDT",
        )
    ]

    class _ClassifierStub:
        def __init__(self) -> None:
            self._queue = list(assessments)

        @property
        def thresholds_loader(self):
            return lambda: deepcopy(thresholds)

        def thresholds_snapshot(self):
            return deepcopy(thresholds)

        def assess(self, *_args, **_kwargs):
            return self._queue.pop(0)

    stub = _ClassifierStub()
    history = RegimeHistory(thresholds_loader=stub.thresholds_loader)

    engine = AutoTradeEngine(
        adapter,
        lambda *_: None,
        cfg,
        regime_classifier=stub,
        regime_history=history,
    )
    assert engine._regime_history is history  # type: ignore[attr-defined]

    frame = pd.DataFrame(
        {
            "close": [50.0 + idx for idx in range(8)],
            "high": [50.5 + idx for idx in range(8)],
            "low": [49.5 + idx for idx in range(8)],
            "volume": [2000.0 + idx * 20 for idx in range(8)],
        }
    )

    assessment = engine._classify_regime(frame)
    assert assessment.regime is MarketRegime.DAILY
    assert engine.get_regime_thresholds()["market_regime"]["risk_level"]["critical"]["risk_score"] == 0.9


def test_configure_regime_history_updates_parameters_and_reset() -> None:
    adapter = EmitterAdapter()
    cfg = AutoTradeConfig(symbol="AVAXUSDT", regime_window=6)
    engine = AutoTradeEngine(adapter, lambda *_: None, cfg)

    engine.configure_regime_history(maxlen=4, decay=0.7)
    history = engine._regime_history  # type: ignore[attr-defined]
    assert history.maxlen == 4
    assert history.decay == pytest.approx(0.7)
    assert engine.cfg.regime_history_maxlen == 4
    assert engine.cfg.regime_history_decay == pytest.approx(0.7)

    thresholds = {
        "market_regime": {
            "metrics": {},
            "risk_score": {},
            "risk_level": {"critical": {"risk_score": 0.8}},
        }
    }

    assessment = MarketRegimeAssessment(
        regime=MarketRegime.TREND,
        confidence=0.6,
        risk_score=0.35,
        metrics={"trend_strength": 0.02, "volatility": 0.01},
        symbol="AVAXUSDT",
    )

    class _ClassifierStub:
        def __init__(self) -> None:
            self._issued = False

        @property
        def thresholds_loader(self):
            return lambda: deepcopy(thresholds)

        def thresholds_snapshot(self):
            return deepcopy(thresholds)

        def assess(self, *_args, **_kwargs):
            if self._issued:
                raise AssertionError("assessment already issued")
            self._issued = True
            return assessment

    stub = _ClassifierStub()
    history_override = RegimeHistory(thresholds_loader=stub.thresholds_loader)
    history_override.reload_thresholds(thresholds=stub.thresholds_snapshot())
    engine.set_regime_components(classifier=stub, history=history_override, reset_state=False)

    frame = pd.DataFrame(
        {
            "close": [50.0 + idx for idx in range(6)],
            "high": [50.5 + idx for idx in range(6)],
            "low": [49.5 + idx for idx in range(6)],
            "volume": [1200.0 + idx * 15 for idx in range(6)],
        }
    )
    engine._classify_regime(frame)
    assert engine.get_last_regime_assessment() is not None
    assert engine.get_regime_summary() is not None

    engine.configure_regime_history(reset=True)
    assert engine.get_last_regime_assessment() is None
    assert engine.get_regime_summary() is None


def test_auto_trade_engine_auto_risk_freeze_and_release() -> None:
    adapter = EmitterAdapter()
    statuses: list[dict] = []

    def _collect(evt_or_batch):
        batch = evt_or_batch if isinstance(evt_or_batch, list) else [evt_or_batch]
        for evt in batch:
            statuses.append(evt.payload)

    adapter.subscribe(EventType.AUTOTRADE_STATUS, _collect)

    cfg = AutoTradeConfig(
        symbol="BTCUSDT",
        regime_window=5,
        activation_threshold=1.0,
        risk_freeze_seconds=30,
        auto_risk_freeze=True,
        auto_risk_freeze_level=RiskLevel.ELEVATED,
        auto_risk_freeze_score=0.65,
    )

    high_assessment = MarketRegimeAssessment(
        regime=MarketRegime.TREND,
        confidence=0.7,
        risk_score=0.72,
        metrics={"trend_strength": 0.03, "volatility": 0.045, "drawdown": 0.26},
        symbol=cfg.symbol,
    )
    low_assessment = MarketRegimeAssessment(
        regime=MarketRegime.TREND,
        confidence=0.8,
        risk_score=0.35,
        metrics={"trend_strength": 0.02, "volatility": 0.015, "drawdown": 0.1},
        symbol=cfg.symbol,
    )

    class _Classifier:
        def __init__(self) -> None:
            self._assessments = [high_assessment, low_assessment]
            self._thresholds = {"market_regime": {"metrics": {}, "risk_score": {}, "risk_level": {}}}

        @property
        def thresholds_loader(self):
            return lambda: deepcopy(self._thresholds)

        def thresholds_snapshot(self):
            return deepcopy(self._thresholds)

        def assess(self, *_args, **_kwargs):
            if self._assessments:
                return self._assessments.pop(0)
            return low_assessment

    class _History:
        def __init__(self) -> None:
            self.maxlen = 5
            self.decay = 0.65
            self._thresholds = {}
            self._summaries = [
                _make_summary(risk_level=RiskLevel.CRITICAL, risk_score=0.82),
                _make_summary(risk_level=RiskLevel.BALANCED, risk_score=0.4),
            ]
            self._latest_summary: RegimeSummary | None = None

        def reload_thresholds(self, *, thresholds=None, loader=None):
            if thresholds is not None:
                self._thresholds = deepcopy(dict(thresholds))
            elif loader is not None:
                self._thresholds = deepcopy(dict(loader()))

        def thresholds_snapshot(self):
            return deepcopy(self._thresholds)

        def update(self, assessment):
            snapshot = RegimeSnapshot(
                regime=assessment.regime,
                confidence=assessment.confidence,
                risk_score=assessment.risk_score,
                drawdown=float(assessment.metrics.get("drawdown", assessment.risk_score)),
                volatility=float(assessment.metrics.get("volatility", assessment.risk_score)),
                volume_trend=float(assessment.metrics.get("volume_trend", 0.0)),
                volatility_ratio=float(assessment.metrics.get("volatility_ratio", 1.0)),
                return_skew=float(assessment.metrics.get("return_skew", 0.0)),
                return_kurtosis=float(assessment.metrics.get("return_kurtosis", 0.0)),
                volume_imbalance=float(assessment.metrics.get("volume_imbalance", 0.0)),
            )
            if self._summaries:
                self._latest_summary = self._summaries.pop(0)
            return snapshot

        def summarise(self):
            return self._latest_summary

        def reconfigure(self, *, maxlen=None, decay=None, keep_history=True):
            if maxlen is not None:
                self.maxlen = int(maxlen)
            if decay is not None:
                self.decay = float(decay)

        def clear(self):
            self._latest_summary = None

    classifier = _Classifier()
    history = _History()

    engine = AutoTradeEngine(adapter, lambda *_: None, cfg, regime_classifier=classifier, regime_history=history)

    frame = pd.DataFrame({
        "close": [100 + idx for idx in range(10)],
        "high": [101 + idx for idx in range(10)],
        "low": [99 + idx for idx in range(10)],
        "volume": [1500.0 + idx * 10.0 for idx in range(10)],
    })

    engine._classify_regime(frame)
    time.sleep(0.05)
    assert engine._auto_risk_frozen  # type: ignore[attr-defined]
    assert engine._risk_frozen_until > time.time()  # type: ignore[attr-defined]

    engine._classify_regime(frame)
    time.sleep(0.05)
    assert not engine._auto_risk_frozen  # type: ignore[attr-defined]
    assert engine._risk_frozen_until <= time.time() + 1  # type: ignore[attr-defined]

    freeze_events = [payload for payload in statuses if payload["status"] == "risk_freeze"]
    release_events = [payload for payload in statuses if payload["status"] == "risk_freeze_release"]

    assert freeze_events, "Expected auto risk freeze status"
    freeze_detail = freeze_events[-1]["detail"]
    assert freeze_detail["reason"] == "auto_risk_freeze"
    assert freeze_detail["risk_level"] == RiskLevel.CRITICAL.value

    assert release_events, "Expected auto risk release status"
    release_detail = release_events[-1]["detail"]
    assert release_detail["reason"] == "auto_risk_release"
    assert release_detail["risk_level"] == RiskLevel.BALANCED.value


def test_set_regime_components_can_reset_state() -> None:
    adapter = EmitterAdapter()
    cfg = AutoTradeConfig(symbol="SOLUSDT", regime_window=6)
    engine = AutoTradeEngine(adapter, lambda *_: None, cfg)

    thresholds = {
        "market_regime": {
            "metrics": {},
            "risk_score": {},
            "risk_level": {"critical": {"risk_score": 0.7}},
        }
    }

    def _make_stub(regime: MarketRegime) -> tuple[object, RegimeHistory]:
        assessment = MarketRegimeAssessment(
            regime=regime,
            confidence=0.6,
            risk_score=0.3,
            metrics={"trend_strength": 0.02, "volatility": 0.01},
            symbol="SOLUSDT",
        )

        class _ClassifierStub:
            def __init__(self) -> None:
                self._issued = False

            @property
            def thresholds_loader(self):
                return lambda: deepcopy(thresholds)

            def thresholds_snapshot(self):
                return deepcopy(thresholds)

            def assess(self, *_args, **_kwargs):
                if self._issued:
                    raise AssertionError("assessment already issued")
                self._issued = True
                return assessment

        stub = _ClassifierStub()
        history = RegimeHistory(thresholds_loader=stub.thresholds_loader)
        history.reload_thresholds(thresholds=stub.thresholds_snapshot())
        return stub, history

    stub_one, history_one = _make_stub(MarketRegime.TREND)
    engine.set_regime_components(classifier=stub_one, history=history_one)

    frame = pd.DataFrame(
        {
            "close": [100.0 + idx for idx in range(6)],
            "high": [101.0 + idx for idx in range(6)],
            "low": [99.0 + idx for idx in range(6)],
            "volume": [1500.0 + idx * 10 for idx in range(6)],
        }
    )

    engine._classify_regime(frame)
    assert engine.get_last_regime_assessment() is not None
    assert engine.get_regime_summary() is not None

    stub_two, history_two = _make_stub(MarketRegime.MEAN_REVERSION)
    engine.set_regime_components(classifier=stub_two, history=history_two)

    assert engine.get_last_regime_assessment() is None
    assert engine.get_regime_summary() is None


def test_set_regime_components_reuses_existing_history_without_reset() -> None:
    adapter = EmitterAdapter()
    cfg = AutoTradeConfig(symbol="ADAUSDT", regime_window=5)
    engine = AutoTradeEngine(adapter, lambda *_: None, cfg)

    thresholds_one = {
        "market_regime": {
            "metrics": {},
            "risk_score": {},
            "risk_level": {"critical": {"risk_score": 0.65}},
        }
    }

    class _ClassifierOne:
        def __init__(self) -> None:
            self._issued = False

        @property
        def thresholds_loader(self):
            return lambda: deepcopy(thresholds_one)

        def thresholds_snapshot(self):
            return deepcopy(thresholds_one)

        def assess(self, *_args, **_kwargs):
            if self._issued:
                raise AssertionError("assessment already issued")
            self._issued = True
            return MarketRegimeAssessment(
                regime=MarketRegime.DAILY,
                confidence=0.55,
                risk_score=0.35,
                metrics={"trend_strength": 0.015, "volatility": 0.008},
                symbol="ADAUSDT",
            )

    classifier_one = _ClassifierOne()
    history_one = RegimeHistory(thresholds_loader=classifier_one.thresholds_loader)
    history_one.reload_thresholds(thresholds=classifier_one.thresholds_snapshot())
    engine.set_regime_components(classifier=classifier_one, history=history_one)

    frame = pd.DataFrame({"close": [10 + idx for idx in range(5)]})
    engine._classify_regime(frame)
    history_ref = engine._regime_history  # type: ignore[attr-defined]
    assert isinstance(history_ref, RegimeHistory)
    assert len(history_ref.snapshots) == 1

    thresholds_two = {
        "market_regime": {
            "metrics": {},
            "risk_score": {},
            "risk_level": {"critical": {"risk_score": 0.85}},
        }
    }

    class _ClassifierTwo:
        @property
        def thresholds_loader(self):
            return lambda: deepcopy(thresholds_two)

        def thresholds_snapshot(self):
            return deepcopy(thresholds_two)

        def assess(self, *_args, **_kwargs):
            return MarketRegimeAssessment(
                regime=MarketRegime.MEAN_REVERSION,
                confidence=0.5,
                risk_score=0.4,
                metrics={"trend_strength": 0.01, "volatility": 0.01},
                symbol="ADAUSDT",
            )

    classifier_two = _ClassifierTwo()
    engine.set_regime_components(classifier=classifier_two, reset_state=False)

    assert engine._regime_history is history_ref  # type: ignore[attr-defined]
    assert len(engine._regime_history.snapshots) == 1  # type: ignore[attr-defined]
    thresholds_snapshot = engine.get_regime_thresholds()
    assert (
        thresholds_snapshot["market_regime"]["risk_level"]["critical"]["risk_score"]
        == 0.85
    )


def test_set_regime_components_clears_reused_history_when_resetting() -> None:
    adapter = EmitterAdapter()
    cfg = AutoTradeConfig(symbol="XRPUSDT", regime_window=5)
    engine = AutoTradeEngine(adapter, lambda *_: None, cfg)

    thresholds = {
        "market_regime": {
            "metrics": {},
            "risk_score": {},
            "risk_level": {"critical": {"risk_score": 0.75}},
        }
    }

    class _Classifier:
        def __init__(self) -> None:
            self._count = 0

        @property
        def thresholds_loader(self):
            return lambda: deepcopy(thresholds)

        def thresholds_snapshot(self):
            return deepcopy(thresholds)

        def assess(self, *_args, **_kwargs):
            self._count += 1
            return MarketRegimeAssessment(
                regime=MarketRegime.TREND,
                confidence=0.6,
                risk_score=0.45,
                metrics={"trend_strength": 0.02, "volatility": 0.015},
                symbol="XRPUSDT",
            )

    classifier = _Classifier()
    engine.set_regime_components(classifier=classifier)

    frame = pd.DataFrame({"close": [20 + idx for idx in range(6)]})
    engine._classify_regime(frame)
    assert len(engine._regime_history.snapshots) == 1  # type: ignore[attr-defined]

    engine.set_regime_components(classifier=classifier, reset_state=True)

    assert len(engine._regime_history.snapshots) == 0  # type: ignore[attr-defined]
    assert engine.get_last_regime_assessment() is None
    assert engine.get_regime_summary() is None
