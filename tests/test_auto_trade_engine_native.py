from __future__ import annotations

import time
from copy import deepcopy

import pandas as pd

from bot_core.ai.regime import (
    MarketRegime,
    MarketRegimeAssessment,
    RegimeHistory,
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
    engine._regime_classifier = stub  # type: ignore[attr-defined]
    engine._regime_history = RegimeHistory(thresholds_loader=stub.thresholds_loader)
    engine._regime_history.decay = 0.1
    engine._regime_history.reload_thresholds(thresholds=stub.thresholds_snapshot())
    engine._last_regime = None
    engine._last_summary = None

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
    engine._regime_classifier = stub  # type: ignore[attr-defined]
    engine._regime_history = RegimeHistory(thresholds_loader=stub.thresholds_loader)
    engine._regime_history.reload_thresholds(thresholds=stub.thresholds_snapshot())

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
