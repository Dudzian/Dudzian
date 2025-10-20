from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pytest

from bot_core.ai.regime import (
    MarketRegime,
    MarketRegimeAssessment,
    RegimeSnapshot,
    RegimeSummary,
    RiskLevel,
)
from bot_core.auto_trader.app import AutoTrader, RiskDecision
from tests.sample_data_loader import load_summary_for_regime


class _Emitter:
    def __init__(self) -> None:
        self.logs: list[str] = []
        self.events: list[tuple[str, dict[str, Any]]] = []

    def log(self, message: str, *_, **__) -> None:
        self.logs.append(message)

    def emit(self, event: str, **payload: Any) -> None:
        self.events.append((event, payload))


class _Var:
    def __init__(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        return self._value


class _GUI:
    def __init__(self) -> None:
        self.timeframe_var = _Var("1h")
        self.ai_mgr = None
        self._demo = True

    def is_demo_mode_active(self) -> bool:
        return self._demo


class _Provider:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.calls: list[tuple[str, str, int]] = []

    def get_historical(self, symbol: str, timeframe: str, limit: int = 256) -> pd.DataFrame:
        self.calls.append((symbol, timeframe, limit))
        return self.df


_MISSING = object()


@dataclass
class _DummyAssessment:
    regime: MarketRegime
    risk_score: float
    confidence: float = 0.8

    def to_assessment(self, symbol: str) -> MarketRegimeAssessment:
        return MarketRegimeAssessment(
            regime=self.regime,
            confidence=self.confidence,
            risk_score=self.risk_score,
            metrics={
                "trend_strength": 0.01,
                "volatility": 0.01,
                "momentum": 0.0,
                "autocorr": -0.1,
            "intraday_vol": 0.01,
            "drawdown": 0.05,
            "volatility_ratio": 1.1,
            "volume_trend": 0.02,
            "return_skew": 0.0,
            "return_kurtosis": 0.0,
            "volume_imbalance": 0.0,
        },
        symbol=symbol,
    )


class _AIManagerStub:
    def __init__(self, assessments: list[_DummyAssessment], summaries: dict[str, RegimeSummary] | None = None) -> None:
        self._queue = assessments
        self.calls: list[str] = []
        self._summaries = summaries or {}

    def assess_market_regime(self, symbol: str, market_data: pd.DataFrame, **_: Any) -> MarketRegimeAssessment:
        self.calls.append(symbol)
        next_assessment = self._queue.pop(0)
        return next_assessment.to_assessment(symbol)

    def get_regime_summary(self, symbol: str) -> RegimeSummary | None:
        return self._summaries.get(symbol)


def _assert_summary_details(
    decision: RiskDecision,
    summary: RegimeSummary,
    *,
    fields: Iterable[str] | None = None,
) -> None:
    summary_details = decision.details.get("summary")
    assert summary_details is not None, "summary details missing from risk decision"
    expected = summary.to_dict()
    if fields is None:
        keys = [key for key in expected.keys() if key != "history"]
    else:
        keys = list(fields)
    for key in keys:
        assert key in summary_details, f"field {key!r} missing from decision summary"
        assert key in expected, f"field {key!r} missing from expected summary payload"
        actual = summary_details[key]
        target = expected[key]
        if isinstance(target, float):
            assert actual == pytest.approx(target)
        else:
            assert actual == target


def _assert_risk_decision(
    decision: RiskDecision | None,
    *,
    should_trade: bool | None = None,
    state: str | None = None,
    cooldown_active: bool | None = None,
    cooldown_reason: str | set[str] | None = None,
    summary: RegimeSummary | None = None,
    summary_fields: Iterable[str] | None = None,
) -> RiskDecision:
    assert decision is not None, "risk decision missing"
    if should_trade is not None:
        assert decision.should_trade is should_trade
    if state is not None:
        assert decision.state == state
    if cooldown_active is not None:
        assert decision.cooldown_active is cooldown_active
    if cooldown_reason is not None:
        if isinstance(cooldown_reason, set):
            assert decision.cooldown_reason in cooldown_reason
        else:
            assert decision.cooldown_reason == cooldown_reason
    if summary is not None:
        _assert_summary_details(decision, summary, fields=summary_fields)
    return decision


def _load_summary(
    regime: MarketRegime,
    *,
    dataset: str | None = None,
    step: int = 24,
    **overrides: Any,
) -> RegimeSummary:
    """Fetch a calibrated summary for the requested regime and override selected fields."""

    resolved_overrides: dict[str, Any] = {}
    rename_map = {
        "risk": "risk_score",
    }
    for key, value in overrides.items():
        if value is None:
            continue
        mapped = rename_map.get(key, key)
        resolved_overrides[mapped] = value
    summary = load_summary_for_regime(regime, dataset=dataset, overrides=resolved_overrides, step=step)
    return summary


def _build_market_data() -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=120, freq="min")
    close = pd.Series(100 + (idx - idx[0]).total_seconds() / 3600.0, index=idx)
    high = close * 1.001
    low = close * 0.999
    volume = pd.Series(1_000, index=idx)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": volume})



def _prepare_trader(
    symbol: str,
    assessments: list[_DummyAssessment],
    *,
    summary: RegimeSummary | None = None,
    market_data: pd.DataFrame | None = None,
) -> tuple[AutoTrader, _Emitter, _Provider, _AIManagerStub]:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(market_data or _build_market_data())
    summaries = {symbol: summary} if summary is not None else None
    ai_manager = _AIManagerStub(assessments, summaries=summaries)

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: symbol,
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        market_data_provider=provider,
    )
    trader.ai_manager = ai_manager
    return trader, emitter, provider, ai_manager


def _run_auto_trade(trader: AutoTrader) -> RiskDecision:
    trader._auto_trade_loop()
    decision = trader._last_risk_decision
    assert isinstance(decision, RiskDecision), "AutoTrader did not record a risk decision"
    return decision


@dataclass(slots=True)
class AutoTradeResult:
    symbol: str
    decision: RiskDecision
    trader: AutoTrader
    emitter: _Emitter
    provider: _Provider
    ai_manager: _AIManagerStub

    def assert_decision(self, **kwargs: Any) -> RiskDecision:
        self.decision = _assert_risk_decision(self.decision, **kwargs)
        return self.decision

    def assert_provider_called(
        self,
        symbol: str | None = None,
        timeframe: str = "1h",
        limit: int = 256,
        *,
        total_calls: int | None = None,
    ) -> None:
        assert self.provider.calls, "market data provider was not called"
        expected_symbol = symbol or self.symbol
        assert self.provider.calls[-1] == (expected_symbol, timeframe, limit)
        if total_calls is not None:
            assert len(self.provider.calls) == total_calls

    def queue_assessments(self, *assessments: _DummyAssessment) -> None:
        self.ai_manager._queue.extend(assessments)

    def update_summary(self, summary: RegimeSummary | None) -> None:
        symbol = self.trader.symbol_getter()
        if summary is None:
            self.ai_manager._summaries.pop(symbol, None)
        else:
            self.ai_manager._summaries[symbol] = summary

    def run_followup(
        self,
        *,
        summary: RegimeSummary | None = _MISSING,
        market_data: pd.DataFrame | None = None,
        assessments: Iterable[_DummyAssessment] | None = None,
    ) -> RiskDecision:
        if assessments is not None:
            self.queue_assessments(*tuple(assessments))
        if summary is not _MISSING:
            self.update_summary(summary)
        if market_data is not None:
            self.provider.df = market_data
        self.decision = _run_auto_trade(self.trader)
        return self.decision

    def __iter__(self):
        yield from (self.decision, self.trader, self.emitter, self.provider, self.ai_manager)


def _execute_auto_trade(
    symbol: str,
    assessments: list[_DummyAssessment],
    *,
    summary: RegimeSummary | None = None,
    market_data: pd.DataFrame | None = None,
) -> AutoTradeResult:
    trader, emitter, provider, ai_manager = _prepare_trader(
        symbol,
        assessments,
        summary=summary,
        market_data=market_data,
    )
    decision = _run_auto_trade(trader)
    return AutoTradeResult(
        symbol=symbol,
        decision=decision,
        trader=trader,
        emitter=emitter,
        provider=provider,
        ai_manager=ai_manager,
    )


def test_auto_trader_maps_trend_assessment_to_buy_signal() -> None:
    result = _execute_auto_trade(
        "BTCUSDT", [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.35)]
    )
    decision, trader, _, provider, _ = result

    assert trader._last_signal == "buy"
    assert trader.current_strategy == "trend_following"
    assert trader.current_leverage > 1.0
    assert decision.should_trade is True
    result.assert_provider_called("BTCUSDT")


def test_auto_trader_respects_high_risk_regime() -> None:
    result = _execute_auto_trade(
        "ETHUSDT", [_DummyAssessment(regime=MarketRegime.DAILY, risk_score=0.85)]
    )
    decision, trader, _, provider, _ = result

    assert trader._last_signal == "hold"
    assert trader.current_strategy == "capital_preservation"
    assert trader.current_leverage == 0.0
    assert decision.should_trade is False
    result.assert_provider_called("ETHUSDT")


def test_auto_trader_uses_summary_to_lock_trading_on_high_risk() -> None:
    summary = _load_summary(
        MarketRegime.DAILY,
        confidence=0.7,
        risk=0.82,
        severe_event_rate=0.75,
        cooldown_score=0.82,
        stress_projection=0.8,
        liquidity_gap=0.75,
        confidence_resilience=0.25,
    )
    result = _execute_auto_trade(
        "ADAUSDT",
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.3)],
        summary=summary,
    )
    decision, trader, _, provider, _ = result

    decision = result.assert_decision(
        should_trade=False,
        cooldown_active=True,
        state="halted",
        summary=summary,
    )
    assert trader._last_signal == "hold"
    assert trader.current_strategy == "capital_preservation"
    assert trader.current_leverage == 0.0
    assert decision.details["effective_risk"] >= 0.8
    assert decision.details["cooldown_reason"] == decision.cooldown_reason
    assert summary.distribution_pressure <= 0.35


def test_auto_trader_throttles_on_liquidity_gap_and_confidence_drop() -> None:
    summary = _load_summary(
        MarketRegime.TREND,
        confidence=0.5,
        risk=0.42,
        liquidity_gap=0.72,
        stress_projection=0.58,
        confidence_resilience=0.38,
        distribution_pressure=0.4,
        resilience_score=0.42,
        cooldown_score=0.48,
    )
    result = _execute_auto_trade(
        "SOLUSDT",
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.32)],
        summary=summary,
    )
    decision, trader, _, provider, _ = result

    decision = result.assert_decision(
        should_trade=False,
        summary=summary,
        summary_fields=[
            "liquidity_gap",
            "confidence_resilience",
            "stress_projection",
            "distribution_pressure",
        ],
    )
    assert trader._last_signal == "hold"
    assert trader.current_leverage <= 0.45
    assert trader.current_strategy in {"capital_preservation", "trend_following_probing"}
    assert decision.details["effective_risk"] >= summary.risk_score * 0.7


def test_auto_trader_blocks_on_stress_momentum_and_fragility() -> None:
    summary = _load_summary(
        MarketRegime.DAILY,
        confidence=0.55,
        risk=0.58,
        risk_level=RiskLevel.ELEVATED,
        stress_index=0.58,
        stress_projection=0.58,
        stress_momentum=0.72,
        liquidity_pressure=0.62,
        liquidity_gap=0.58,
        liquidity_trend=0.68,
        confidence_resilience=0.35,
        confidence_fragility=0.64,
        degradation_score=0.58,
        cooldown_score=0.6,
        severe_event_rate=0.48,
        drawdown_pressure=0.55,
        volatility_ratio=1.32,
        volume_trend_volatility=0.22,
        distribution_pressure=0.52,
    )
    result = _execute_auto_trade(
        "MOMUSDT",
        [_DummyAssessment(regime=MarketRegime.MEAN_REVERSION, risk_score=0.4)],
        summary=summary,
    )
    decision, trader, _, provider, _ = result

    decision = result.assert_decision(
        should_trade=False,
        cooldown_active=True,
        cooldown_reason={"critical_risk", "elevated_risk"},
        summary=summary,
        summary_fields=[
            "stress_momentum",
            "liquidity_trend",
            "confidence_fragility",
            "confidence_resilience",
            "liquidity_gap",
            "distribution_pressure",
            "severe_event_rate",
        ],
    )
    assert trader._last_signal == "hold"
    assert decision.details["effective_risk"] >= summary.risk_score * 0.7
    assert trader.current_leverage <= 0.35


def test_auto_trader_cooldown_engages_and_recovers() -> None:
    severe_summary = _load_summary(
        MarketRegime.DAILY,
        confidence=0.6,
        risk=0.78,
        severe_event_rate=0.7,
        cooldown_score=0.8,
    )
    recovery_summary = _load_summary(
        MarketRegime.TREND,
        confidence=0.72,
        risk=0.42,
        risk_trend=-0.12,
        risk_level=RiskLevel.BALANCED,
        regime_persistence=0.7,
        stability=0.7,
        severe_event_rate=0.2,
        cooldown_score=0.25,
        recovery_potential=0.75,
        risk_volatility=0.08,
        stress_index=0.3,
    )
    result = _execute_auto_trade(
        "SOLUSDT",
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.32)],
        summary=severe_summary,
    )
    decision = result.decision
    trader = result.trader

    assert trader._last_signal == "hold"
    assert decision.cooldown_active is True
    assert decision.state == "halted"
    assert trader._cooldown_reason in {"critical_risk", "elevated_risk", "instability_spike"}

    result.queue_assessments(_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.34))
    recovery_decision = result.run_followup(summary=recovery_summary)

    assert trader._last_signal in {"buy", "sell", "hold"}
    assert recovery_decision.cooldown_active is False
    assert recovery_decision.state != "halted"
    assert trader._cooldown_reason is None
    if trader._last_signal in {"buy", "sell"}:
        assert recovery_decision.should_trade is True
    decision_summary = recovery_decision.details.get("summary")
    if decision_summary is not None:
        assert decision_summary["cooldown_score"] <= 0.4
        assert decision_summary["recovery_potential"] >= 0.4


def test_auto_trader_holds_when_confidence_low_despite_trend() -> None:
    summary = _load_summary(MarketRegime.TREND, confidence=0.3, risk=0.25)
    result = _execute_auto_trade(
        "SOLUSDT",
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.2, confidence=0.15)],
        summary=summary,
    )
    decision, trader, _, provider, _ = result

    decision = result.assert_decision(
        should_trade=False,
        summary=summary,
        summary_fields=[
            "confidence",
            "stability",
            "risk_trend",
            "risk_level",
            "risk_volatility",
            "regime_persistence",
            "transition_rate",
            "confidence_trend",
            "confidence_volatility",
            "regime_streak",
            "instability_score",
            "confidence_decay",
            "tail_risk_index",
            "shock_frequency",
            "volatility_of_volatility",
            "stress_index",
        ],
    )
    assert trader._last_signal == "hold"
    assert decision.details["confidence"] == 0.15


def test_auto_trader_waits_on_unstable_summary_even_in_trend() -> None:
    summary = _load_summary(
        MarketRegime.TREND,
        confidence=0.7,
        risk=0.35,
        stability=0.3,
        risk_trend=0.0,
    )
    result = _execute_auto_trade(
        "BNBUSDT",
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.3, confidence=0.65)],
        summary=summary,
    )
    decision, trader, _, provider, _ = result

    result.assert_decision(
        should_trade=False,
        summary=summary,
        summary_fields=["risk_level", "risk_volatility", "regime_persistence", "stability"],
    )
    assert trader._last_signal == "hold"
    assert trader.current_leverage <= 0.5


def test_auto_trader_holds_when_risk_volatility_spikes() -> None:
    summary = _load_summary(
        MarketRegime.TREND,
        confidence=0.65,
        risk=0.58,
        stability=0.7,
        risk_trend=0.04,
        risk_volatility=0.25,
        regime_persistence=0.55,
        confidence_trend=-0.05,
        confidence_volatility=0.025,
        regime_streak=2,
        transition_rate=0.45,
        instability_score=0.58,
        confidence_decay=0.05,
        risk_level=RiskLevel.ELEVATED,
    )
    result = _execute_auto_trade(
        "DOTUSDT",
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.38, confidence=0.7)],
        summary=summary,
    )
    decision, trader, _, provider, _ = result

    decision = result.assert_decision(
        should_trade=False,
        summary=summary,
        summary_fields=[
            "risk_volatility",
            "regime_persistence",
            "confidence_trend",
            "confidence_volatility",
            "regime_streak",
            "transition_rate",
            "instability_score",
            "confidence_decay",
        ],
    )
    assert trader._last_signal == "hold"
    assert trader.current_leverage <= 0.65
    assert decision.details["effective_risk"] >= summary.risk_score * 0.7


def test_auto_trader_blocks_when_stress_metrics_spike() -> None:
    summary = _load_summary(
        MarketRegime.MEAN_REVERSION,
        confidence=0.6,
        risk=0.48,
        stability=0.35,
        risk_trend=0.12,
        risk_level=RiskLevel.ELEVATED,
        tail_risk_index=0.65,
        shock_frequency=0.6,
        volatility_of_volatility=0.035,
        stress_index=0.78,
        drawdown_pressure=0.58,
        liquidity_pressure=0.55,
    )
    result = _execute_auto_trade(
        "AVAXUSDT",
        [_DummyAssessment(regime=MarketRegime.MEAN_REVERSION, risk_score=0.38, confidence=0.62)],
        summary=summary,
    )
    decision, trader, _, provider, _ = result

    decision = result.assert_decision(
        should_trade=False,
        summary=summary,
        summary_fields=[
            "stress_index",
            "tail_risk_index",
            "shock_frequency",
            "volatility_of_volatility",
            "drawdown_pressure",
            "liquidity_pressure",
        ],
    )
    assert trader._last_signal == "hold"
    assert trader.current_leverage <= 0.35
    assert decision.details["effective_risk"] >= 0.7


def test_auto_trader_blocks_on_degradation_metrics() -> None:
    summary = _load_summary(
        MarketRegime.TREND,
        confidence=0.58,
        risk=0.52,
        stability=0.4,
        risk_trend=0.1,
        risk_level=RiskLevel.ELEVATED,
        risk_volatility=0.22,
        regime_persistence=0.4,
        transition_rate=0.6,
        avg_drawdown=0.24,
        avg_volume_trend=-0.166,
        volatility_of_volatility=0.012,
        tail_risk_index=0.58,
        shock_frequency=0.58,
        stress_index=0.62,
        cooldown_score=0.5,
        severe_event_rate=0.42,
        volatility_trend=0.03,
        drawdown_trend=0.12,
        volume_trend_volatility=0.22,
        stability_projection=0.3,
        degradation_score=0.65,
    )
    result = _execute_auto_trade(
        "DEGUSDT",
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.42, confidence=0.58)],
        summary=summary,
    )
    decision, trader, _, provider, _ = result

    result.assert_decision(
        should_trade=False,
        cooldown_active=True,
        summary=summary,
        summary_fields=[
            "degradation_score",
            "stability_projection",
            "volatility_trend",
            "drawdown_trend",
        ],
    )
    assert trader._last_signal == "hold"
    assert trader.current_strategy == "capital_preservation"
    assert trader.current_leverage == 0.0


def test_auto_trader_blocks_on_distribution_pressure() -> None:
    summary = _load_summary(
        MarketRegime.TREND,
        confidence=0.62,
        risk=0.48,
        drawdown_pressure=0.45,
        liquidity_pressure=0.42,
        stress_index=0.5,
        severe_event_rate=0.38,
        cooldown_score=0.5,
        skewness_bias=1.6,
        kurtosis_excess=3.4,
        volume_imbalance=0.58,
        distribution_pressure=0.72,
        degradation_score=0.28,
    )
    result = _execute_auto_trade(
        "DOGEUSDT", [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.32)], summary=summary
    )
    decision, trader, _, provider, _ = result

    result.assert_decision(
        should_trade=False,
        cooldown_active=True,
        summary=summary,
        summary_fields=[
            "distribution_pressure",
            "skewness_bias",
            "kurtosis_excess",
            "volume_imbalance",
            "regime_entropy",
            "resilience_score",
            "stress_balance",
        ],
    )
    assert trader._last_signal == "hold"
    assert trader.current_leverage <= 0.35


def test_auto_trader_allows_trade_when_degradation_low() -> None:
    summary = _load_summary(
        MarketRegime.TREND,
        confidence=0.65,
        risk=0.38,
        stability=0.6,
        risk_trend=0.02,
        risk_level=RiskLevel.BALANCED,
        risk_volatility=0.12,
        regime_persistence=0.55,
        transition_rate=0.35,
        avg_drawdown=0.07,
        avg_volume_trend=0.05,
        volatility_of_volatility=0.006,
        tail_risk_index=0.25,
        shock_frequency=0.2,
        stress_index=0.3,
        cooldown_score=0.2,
        severe_event_rate=0.2,
        volatility_trend=-0.002,
        drawdown_trend=-0.02,
        volume_trend_volatility=0.05,
        stability_projection=0.72,
        degradation_score=0.18,
    )
    result = _execute_auto_trade(
        "POSUSDT",
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.35, confidence=0.68)],
        summary=summary,
    )
    decision, trader, _, provider, _ = result

    result.assert_decision(
        should_trade=True,
        cooldown_active=False,
        summary=summary,
        summary_fields=["degradation_score", "stability_projection", "cooldown_score", "risk_level"],
    )
    assert trader._last_signal == "buy"
    assert trader.current_strategy.startswith("trend_following")
    assert trader.current_leverage > 0.0


def test_auto_trader_blocks_on_high_instability() -> None:
    summary = _load_summary(
        MarketRegime.TREND,
        confidence=0.62,
        risk=0.68,
        stability=0.45,
        risk_trend=0.12,
        regime_persistence=0.25,
        transition_rate=0.75,
        instability_score=0.82,
        confidence_decay=0.25,
        confidence_trend=-0.1,
        confidence_volatility=0.04,
        risk_level=RiskLevel.CRITICAL,
    )
    result = _execute_auto_trade(
        "FTMUSDT",
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.4, confidence=0.6)],
        summary=summary,
    )
    decision, trader, _, provider, _ = result

    decision = result.assert_decision(
        should_trade=False,
        summary=summary,
        summary_fields=[
            "instability_score",
            "transition_rate",
            "confidence_decay",
            "confidence_trend",
        ],
    )
    assert trader._last_signal == "hold"
    assert trader.current_strategy == "capital_preservation"
    assert trader.current_leverage == 0.0
    assert decision.details["effective_risk"] >= summary.risk_score * 0.7


def test_auto_trader_blocks_on_drawdown_pressure() -> None:
    summary = _load_summary(
        MarketRegime.TREND,
        confidence=0.66,
        risk=0.66,
        stability=0.58,
        risk_trend=0.14,
        risk_level=RiskLevel.CRITICAL,
        avg_drawdown=0.32,
        drawdown_pressure=0.9,
        avg_volume_trend=-0.1,
        liquidity_pressure=0.58,
        volatility_ratio=1.45,
    )
    result = _execute_auto_trade(
        "LINKUSDT",
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.42, confidence=0.64)],
        summary=summary,
    )
    decision, trader, _, provider, _ = result

    decision = result.assert_decision(
        should_trade=False,
        summary=summary,
        summary_fields=[
            "drawdown_pressure",
            "avg_drawdown",
            "avg_volume_trend",
            "liquidity_pressure",
        ],
    )
    assert trader._last_signal == "hold"
    assert trader.current_strategy == "capital_preservation"
    assert trader.current_leverage == 0.0
    assert decision.details["effective_risk"] >= summary.risk_score * 0.7


def test_auto_trader_suppresses_leverage_on_liquidity_pressure() -> None:
    summary = _load_summary(
        MarketRegime.MEAN_REVERSION,
        confidence=0.6,
        risk=0.56,
        stability=0.62,
        risk_trend=0.06,
        risk_level=RiskLevel.ELEVATED,
        avg_drawdown=0.14,
        drawdown_pressure=0.56,
        avg_volume_trend=-0.28,
        liquidity_pressure=0.72,
        volatility_ratio=1.28,
    )
    result = _execute_auto_trade(
        "SOLUSDT",
        [_DummyAssessment(regime=MarketRegime.MEAN_REVERSION, risk_score=0.36, confidence=0.62)],
        summary=summary,
    )
    decision, trader, _, provider, _ = result

    decision = result.assert_decision(
        should_trade=False,
        summary=summary,
        summary_fields=["liquidity_pressure", "avg_volume_trend"],
    )
    assert trader._last_signal == "hold"
    assert trader.current_leverage <= 0.35
    assert trader.current_strategy in {"capital_preservation", "mean_reversion_probing"}
    assert decision.details["effective_risk"] >= summary.risk_score * 0.7


def test_auto_trader_holds_when_confidence_trend_collapses() -> None:
    summary = _load_summary(
        MarketRegime.TREND,
        confidence=0.28,
        risk=0.38,
        stability=0.68,
        risk_trend=0.02,
        confidence_trend=-0.44,
        confidence_volatility=0.18,
        confidence_decay=0.44,
    )
    result = _execute_auto_trade(
        "FTMUSDT",
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.34, confidence=0.74)],
        summary=summary,
    )
    decision, trader, _, provider, _ = result

    result.assert_decision(
        should_trade=False,
        summary=summary,
        summary_fields=[
            "confidence_trend",
            "confidence_volatility",
            "regime_streak",
            "transition_rate",
            "instability_score",
            "confidence_decay",
        ],
    )
    assert trader._last_signal == "hold"
    assert trader.current_leverage <= 0.6
    assert trader.current_strategy.endswith("probing") or trader.current_strategy == "capital_preservation"


def test_auto_trader_holds_when_regime_streak_short() -> None:
    summary = _load_summary(
        MarketRegime.TREND,
        confidence=0.68,
        risk=0.52,
        stability=0.52,
        risk_trend=0.03,
        regime_persistence=0.3,
        regime_streak=1,
        transition_rate=0.7,
        confidence_trend=-0.06,
        confidence_volatility=0.03,
        risk_level=RiskLevel.ELEVATED,
    )
    result = _execute_auto_trade(
        "NEARUSDT",
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.32, confidence=0.7)],
        summary=summary,
    )
    decision, trader, _, provider, _ = result

    result.assert_decision(
        should_trade=False,
        summary=summary,
        summary_fields=["regime_streak", "confidence_trend", "confidence_volatility"],
    )
    assert trader._last_signal == "hold"
    assert trader.current_leverage <= 0.4
    assert trader.current_strategy.endswith("probing") or trader.current_strategy == "capital_preservation"


def test_auto_trader_trims_when_persistence_collapses() -> None:
    summary = _load_summary(
        MarketRegime.MEAN_REVERSION,
        confidence=0.7,
        risk=0.6,
        stability=0.5,
        risk_trend=0.03,
        regime_persistence=0.2,
        risk_level=RiskLevel.ELEVATED,
    )
    result = _execute_auto_trade(
        "AVAXUSDT",
        [_DummyAssessment(regime=MarketRegime.MEAN_REVERSION, risk_score=0.44, confidence=0.72)],
        summary=summary,
    )
    decision, trader, _, provider, _ = result

    decision = result.assert_decision(
        should_trade=False,
        summary=summary,
        summary_fields=[
            "regime_persistence",
            "risk_volatility",
            "confidence_trend",
            "confidence_volatility",
            "regime_streak",
            "transition_rate",
            "instability_score",
            "confidence_decay",
        ],
    )
    assert trader._last_signal == "hold"
    assert trader.current_leverage <= 0.4
    assert trader.current_strategy.endswith("probing") or trader.current_strategy == "capital_preservation"
    assert decision.details["effective_risk"] >= summary.risk_score * 0.7


def test_auto_trader_scales_down_when_risk_trend_rising() -> None:
    summary = _load_summary(
        MarketRegime.TREND,
        confidence=0.8,
        risk=0.55,
        stability=0.9,
        risk_trend=0.2,
        risk_level=RiskLevel.ELEVATED,
    )
    result = _execute_auto_trade(
        "XRPUSDT",
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.28, confidence=0.8)],
        summary=summary,
    )
    decision, trader, _, provider, _ = result

    result.assert_decision(
        summary=summary,
        summary_fields=[
            "risk_trend",
            "risk_level",
            "risk_volatility",
            "regime_persistence",
            "confidence_trend",
            "confidence_volatility",
            "regime_streak",
            "transition_rate",
            "instability_score",
            "confidence_decay",
        ],
    )
    assert trader._last_signal == "hold"
    assert trader.current_leverage <= 1.7


def test_auto_trader_increases_risk_when_summary_calm() -> None:
    summary = _load_summary(
        MarketRegime.TREND,
        confidence=0.78,
        risk=0.18,
        stability=0.8,
        risk_trend=-0.05,
        risk_level=RiskLevel.CALM,
        confidence_trend=0.13,
        confidence_volatility=0.05,
        confidence_decay=0.0,
    )
    result = _execute_auto_trade(
        "LTCUSDT",
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.22, confidence=0.78)],
        summary=summary,
    )
    decision, trader, _, provider, _ = result

    result.assert_decision(
        should_trade=True,
        summary=summary,
        summary_fields=["risk_level", "risk_volatility", "regime_persistence"],
    )
    assert trader._last_signal == "buy"
    assert trader.current_strategy == "trend_following"
    assert trader.current_leverage > 2.0
