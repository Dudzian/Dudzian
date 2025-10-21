from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bot_core.ai import config_loader
from bot_core.ai.regime import (
    MarketRegime,
    MarketRegimeAssessment,
    MarketRegimeClassifier,
    RegimeSnapshot,
    RegimeSummary,
    RiskLevel,
)
from bot_core.auto_trader.app import AutoTrader, RiskDecision
from tests.sample_data_loader import load_summary_for_regime


_MISSING = object()


class _Approval(Enum):
    APPROVED = "approved"
    DENIED = "denied"

    def __bool__(self) -> bool:  # pragma: no cover - sugar for readability
        return self is _Approval.APPROVED


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


class _RiskServiceStub:
    def __init__(self, approval: Any) -> None:
        self._approval = approval
        self.calls: list[RiskDecision] = []

    def evaluate_decision(self, decision: RiskDecision) -> Any:
        self.calls.append(decision)
        return self._approval

    def __call__(self, decision: RiskDecision) -> Any:  # pragma: no cover - compatibility shim
        return self.evaluate_decision(decision)


class _RiskServiceResponseStub:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[RiskDecision] = []

    def _resolve(self) -> Any:
        if callable(self._response):
            return self._response()
        return self._response

    def evaluate_decision(self, decision: RiskDecision) -> Any:
        self.calls.append(decision)
        return self._resolve()

    def __call__(self, decision: RiskDecision) -> Any:  # pragma: no cover - compatibility shim
        return self.evaluate_decision(decision)


class _ExecutionServiceStub:
    def __init__(self) -> None:
        self.calls: list[RiskDecision] = []
        self.methods: list[str] = []

    def execute_decision(self, decision: RiskDecision) -> None:
        self.calls.append(decision)
        self.methods.append("execute_decision")

    def execute(self, decision: RiskDecision) -> None:
        self.calls.append(decision)
        self.methods.append("execute")


class _ExecutionServiceExecuteOnly:
    def __init__(self) -> None:
        self.calls: list[RiskDecision] = []
        self.methods: list[str] = []

    def execute(self, decision: RiskDecision) -> None:
        self.calls.append(decision)
        self.methods.append("execute")


def test_map_regime_to_signal_respects_config(monkeypatch: pytest.MonkeyPatch) -> None:
    config_loader.reset_threshold_cache()
    baseline = copy.deepcopy(config_loader.load_risk_thresholds())
    baseline["auto_trader"]["map_regime_to_signal"]["assessment_confidence"] = 0.9

    def _patched_loader(*_: object, **__: object) -> dict[str, object]:
        return baseline

    monkeypatch.setattr(config_loader, "load_risk_thresholds", _patched_loader)
    monkeypatch.setattr("bot_core.auto_trader.app.load_risk_thresholds", _patched_loader)

    try:
        emitter = _Emitter()
        gui = _GUI()
        trader = AutoTrader(emitter, gui, lambda: "BTCUSDT")
        assessment = MarketRegimeAssessment(
            regime=MarketRegime.TREND,
            confidence=0.5,
            risk_score=0.2,
            metrics={
                "trend_strength": 0.02,
                "volatility": 0.01,
                "momentum": 0.002,
                "autocorr": -0.05,
                "intraday_vol": 0.01,
                "drawdown": 0.05,
            },
        )

        assert (
            trader._thresholds["auto_trader"]["map_regime_to_signal"]["assessment_confidence"]
            == 0.9
        )
        assert trader._map_regime_to_signal(assessment, 0.01) == "hold"
    finally:
        config_loader.reset_threshold_cache()


def test_reload_thresholds_refreshes_classifier() -> None:
    config_loader.reset_threshold_cache()
    store = copy.deepcopy(config_loader.load_risk_thresholds())

    def _loader() -> dict[str, Any]:
        return copy.deepcopy(store)

    classifier = MarketRegimeClassifier(thresholds_loader=_loader)
    metrics_cfg = classifier._thresholds["market_regime"]["metrics"]
    assert metrics_cfg["short_span_min"] == store["market_regime"]["metrics"]["short_span_min"]

    store["market_regime"]["metrics"]["short_span_min"] = 9
    classifier.reload_thresholds()

    metrics_cfg = classifier._thresholds["market_regime"]["metrics"]
    assert metrics_cfg["short_span_min"] == 9
    config_loader.reset_threshold_cache()


def test_reload_thresholds_refreshes_auto_trader() -> None:
    store: dict[str, Any] = {
        "auto_trader": {"map_regime_to_signal": {"assessment_confidence": 0.3}}
    }

    def _loader() -> dict[str, Any]:
        return copy.deepcopy(store)

    emitter = _Emitter()
    gui = _GUI()
    trader = AutoTrader(emitter, gui, lambda: "ETHUSDT", thresholds_loader=_loader)

    assert (
        trader._thresholds["auto_trader"]["map_regime_to_signal"]["assessment_confidence"]
        == 0.3
    )

    store["auto_trader"]["map_regime_to_signal"]["assessment_confidence"] = 0.6
    trader.reload_thresholds()

    assert (
        trader._thresholds["auto_trader"]["map_regime_to_signal"]["assessment_confidence"]
        == 0.6
    )


def test_adjust_strategy_parameters_respects_summary_risk_cap() -> None:
    config_loader.reset_threshold_cache()
    try:
        base_thresholds = copy.deepcopy(config_loader.load_risk_thresholds())

        emitter_default = _Emitter()
        gui_default = _GUI()
        default_trader = AutoTrader(emitter_default, gui_default, lambda: "BTCUSDT")

        assessment = MarketRegimeAssessment(
            regime=MarketRegime.TREND,
            confidence=0.8,
            risk_score=0.32,
            metrics={
                "trend_strength": 0.02,
                "volatility": 0.01,
                "momentum": 0.002,
                "autocorr": -0.05,
                "intraday_vol": 0.01,
                "drawdown": 0.05,
            },
        )

        summary = _build_summary(
            MarketRegime.TREND,
            confidence=0.8,
            risk=0.35,
            stability=0.35,
            risk_trend=0.2,
            risk_level=RiskLevel.CALM,
            risk_volatility=0.1,
            regime_persistence=0.62,
            confidence_trend=0.1,
            confidence_volatility=0.05,
            regime_streak=3,
            transition_rate=0.2,
            instability_score=0.2,
            confidence_decay=0.1,
            drawdown_pressure=0.4,
            liquidity_pressure=0.4,
            volatility_ratio=1.1,
            tail_risk_index=0.3,
            shock_frequency=0.3,
            volatility_of_volatility=0.02,
            stress_index=0.3,
            severe_event_rate=0.1,
            cooldown_score=0.2,
            recovery_potential=0.6,
            volatility_trend=0.01,
            drawdown_trend=0.01,
            volume_trend_volatility=0.12,
            stability_projection=0.55,
            degradation_score=0.35,
            skewness_bias=0.4,
            kurtosis_excess=0.6,
            volume_imbalance=0.12,
            distribution_pressure=0.4,
            regime_entropy=0.4,
            resilience_score=0.7,
            stress_balance=0.7,
            liquidity_gap=0.45,
            confidence_resilience=0.6,
            stress_projection=0.4,
            stress_momentum=0.4,
            liquidity_trend=0.4,
            confidence_fragility=0.4,
        )

        default_trader._adjust_strategy_parameters(assessment, aggregated_risk=0.35, summary=summary)
        default_leverage = default_trader.current_leverage

        custom_thresholds = copy.deepcopy(base_thresholds)
        custom_thresholds["auto_trader"]["adjust_strategy_parameters"]["summary_risk_cap"] = 0.3

        def _loader() -> dict[str, Any]:
            return copy.deepcopy(custom_thresholds)

        emitter_custom = _Emitter()
        gui_custom = _GUI()
        custom_trader = AutoTrader(
            emitter_custom,
            gui_custom,
            lambda: "BTCUSDT",
            thresholds_loader=_loader,
        )

        custom_trader._adjust_strategy_parameters(assessment, aggregated_risk=0.35, summary=summary)

        assert default_leverage == pytest.approx(0.5)
        assert custom_trader.current_leverage > default_leverage
        assert custom_trader.current_leverage >= 2.0
    finally:
        config_loader.reset_threshold_cache()


def test_signal_guardrails_follow_configuration() -> None:
    config_loader.reset_threshold_cache()
    try:
        base_thresholds = copy.deepcopy(config_loader.load_risk_thresholds())

        emitter = _Emitter()
        gui = _GUI()
        trader = AutoTrader(emitter, gui, lambda: "BTCUSDT")

        summary = _build_summary(
            MarketRegime.TREND,
            confidence=0.75,
            risk=0.5,
            risk_level=RiskLevel.ELEVATED,
            stability=0.5,
            risk_trend=0.1,
            risk_volatility=0.2,
            regime_persistence=0.6,
            confidence_trend=0.05,
            confidence_volatility=0.05,
            regime_streak=5,
            transition_rate=0.2,
            instability_score=0.4,
            confidence_decay=0.05,
            drawdown_pressure=0.4,
            liquidity_pressure=0.4,
            volatility_ratio=1.1,
            tail_risk_index=0.3,
            shock_frequency=0.3,
            volatility_of_volatility=0.02,
            stress_index=0.7,
            severe_event_rate=0.2,
            cooldown_score=0.2,
            recovery_potential=0.6,
            volatility_trend=0.01,
            drawdown_trend=0.05,
            volume_trend_volatility=0.1,
            stability_projection=0.5,
            degradation_score=0.35,
            skewness_bias=0.4,
            kurtosis_excess=0.6,
            volume_imbalance=0.1,
            distribution_pressure=0.4,
            regime_entropy=0.5,
            resilience_score=0.55,
            stress_balance=0.55,
            liquidity_gap=0.4,
            confidence_resilience=0.55,
            stress_projection=0.4,
            stress_momentum=0.4,
            liquidity_trend=0.4,
            confidence_fragility=0.35,
        )

        assert trader._apply_signal_guardrails("buy", 0.8, summary) == "hold"
        assert trader._last_guardrail_reasons
        assert trader._last_guardrail_triggers
        assert all(isinstance(trigger, GuardrailTrigger) for trigger in trader._last_guardrail_triggers)
        assert any("effective risk" in reason for reason in trader._last_guardrail_reasons)
        assert trader._last_guardrail_triggers[0].name == "effective_risk"

        custom_thresholds = copy.deepcopy(base_thresholds)
        custom_thresholds["auto_trader"]["signal_guardrails"]["effective_risk_cap"] = 0.9
        custom_thresholds["auto_trader"]["signal_guardrails"]["stress_index"] = 0.75

        def _loader() -> dict[str, Any]:
            return copy.deepcopy(custom_thresholds)

        tuned_trader = AutoTrader(_Emitter(), _GUI(), lambda: "BTCUSDT", thresholds_loader=_loader)

        assert tuned_trader._apply_signal_guardrails("buy", 0.8, summary) == "buy"
        assert tuned_trader._last_guardrail_reasons == []
        assert tuned_trader._last_guardrail_triggers == []
    finally:
        config_loader.reset_threshold_cache()


def test_guardrail_reasons_propagate_to_decision() -> None:
    config_loader.reset_threshold_cache()
    try:
        emitter = _Emitter()
        gui = _GUI()
        trader = AutoTrader(emitter, gui, lambda: "BTCUSDT")
        assessment = MarketRegimeAssessment(
            regime=MarketRegime.TREND,
            confidence=0.7,
            risk_score=0.4,
            metrics={
                "trend_strength": 0.02,
                "volatility": 0.01,
                "momentum": 0.002,
                "autocorr": -0.05,
            },
        )
        summary = _build_summary(
            MarketRegime.TREND,
            confidence=0.7,
            risk=0.4,
            risk_level=RiskLevel.BALANCED,
            stability=0.55,
            risk_trend=0.02,
            risk_volatility=0.18,
            regime_persistence=0.6,
            confidence_trend=0.01,
            confidence_volatility=0.03,
            regime_streak=4,
            transition_rate=0.1,
            instability_score=0.2,
            confidence_decay=0.05,
            drawdown_pressure=0.3,
            liquidity_pressure=0.3,
            volatility_ratio=1.05,
            tail_risk_index=0.55,
            shock_frequency=0.5,
            volatility_of_volatility=0.02,
            stress_index=0.7,
            severe_event_rate=0.1,
            cooldown_score=0.2,
            recovery_potential=0.6,
            volatility_trend=0.02,
            drawdown_trend=0.08,
            volume_trend_volatility=0.18,
            stability_projection=0.4,
            degradation_score=0.4,
            skewness_bias=0.3,
            kurtosis_excess=0.6,
            volume_imbalance=0.1,
            distribution_pressure=0.3,
            regime_entropy=0.7,
            resilience_score=0.28,
            stress_balance=0.3,
            liquidity_gap=0.65,
            confidence_resilience=0.35,
            stress_projection=0.65,
            stress_momentum=0.7,
            liquidity_trend=0.65,
            confidence_fragility=0.6,
        )

        signal = trader._apply_signal_guardrails("buy", 0.7, summary)
        reasons = list(trader._last_guardrail_reasons)
        triggers = [trigger.to_dict() for trigger in trader._last_guardrail_triggers]

        assert signal == "hold"
        assert reasons
        assert any("stress index" in reason for reason in reasons)

        decision = trader._build_risk_decision(
            "BTCUSDT",
            signal,
            assessment,
            effective_risk=0.7,
            summary=summary,
            guardrail_reasons=reasons,
            guardrail_triggers=trader._last_guardrail_triggers,
        )

        assert decision.details["guardrail_reasons"] == reasons
        assert decision.details["guardrail_triggers"] == triggers
    finally:
        config_loader.reset_threshold_cache()


def test_load_risk_thresholds_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_loader.reset_threshold_cache()
    override = tmp_path / "risk_thresholds.yaml"
    override.write_text(
        """
auto_trader:
  map_regime_to_signal:
    assessment_confidence: 0.77
""".strip()
    )

    monkeypatch.setenv("BOT_CORE_RISK_THRESHOLDS_PATH", str(override))

    try:
        thresholds = config_loader.load_risk_thresholds()
        assert (
            thresholds["auto_trader"]["map_regime_to_signal"]["assessment_confidence"]
            == 0.77
        )
    finally:
        monkeypatch.delenv("BOT_CORE_RISK_THRESHOLDS_PATH", raising=False)
        config_loader.reset_threshold_cache()

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
    assert trader._last_risk_decision is not None
    assert trader._last_risk_decision.should_trade is True
    assert (
        trader._last_risk_decision.details["summary"]["risk_level"]
        == summary.risk_level.value
    )
    assert trader._last_risk_decision.details["summary"]["risk_volatility"] == pytest.approx(summary.risk_volatility)
    assert trader._last_risk_decision.details["summary"]["regime_persistence"] == pytest.approx(summary.regime_persistence)


@pytest.mark.parametrize(
    ("approval", "cooldown_active", "expected_execute"),
    [
        (True, False, True),
        (False, False, False),
        (True, True, False),
    ],
    ids=["approved", "rejected", "cooldown"],
)
def test_auto_trader_invokes_services_based_on_risk_approval(
    approval: bool,
    cooldown_active: bool,
    expected_execute: bool,
) -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())
    ai_manager = _AIManagerStub(
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.24, confidence=0.76)]
    )

    risk_service = _RiskServiceStub(approval)
    execution_service = _ExecutionServiceStub()

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "ETHUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        market_data_provider=provider,
        risk_service=risk_service,
        execution_service=execution_service,
    )
    trader.ai_manager = ai_manager

    if cooldown_active:
        trader._cooldown_until = time.monotonic() + 60.0
        trader._cooldown_reason = "test"

    trader._auto_trade_loop()

    assert len(risk_service.calls) == 1
    decision = risk_service.calls[0]
    assert decision is trader._last_risk_decision
    assert decision.details["symbol"] == "ETHUSDT"
    assert decision.details["signal"] == trader._last_signal
    assert decision.cooldown_active is cooldown_active
    assert decision.should_trade is (not cooldown_active)

    if expected_execute:
        assert execution_service.calls == [decision]
        assert execution_service.methods[0] in {"execute_decision", "execute"}
    else:
        assert execution_service.calls == []
        assert execution_service.methods == []


@pytest.mark.parametrize(
    "response, expected_execute",
    [
        (True, True),
        (False, False),
        ((True, {"reason": "ok"}), True),
        ((False, {"reason": "blocked"}), False),
        ({"approved": True}, True),
        ({"allow": 0}, False),
        (SimpleNamespace(approved=True), True),
        (SimpleNamespace(allow=0), False),
        (SimpleNamespace(should_trade=True), True),
        (SimpleNamespace(), False),
        ("approved", True),
        ("deny", False),
        (_Approval.APPROVED, True),
        (_Approval.DENIED, False),
    ],
    ids=[
        "bool_true",
        "bool_false",
        "tuple_true",
        "tuple_false",
        "dict_approved",
        "dict_allow_false",
        "ns_approved",
        "ns_allow_false",
        "ns_should_trade",
        "ns_unknown",
        "str_approved",
        "str_deny",
        "enum_approved",
        "enum_denied",
    ],
)
def test_auto_trader_handles_varied_risk_service_responses(
    response: Any,
    expected_execute: bool,
) -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())
    ai_manager = _AIManagerStub(
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.24, confidence=0.76)]
    )

    risk_service = _RiskServiceResponseStub(response)
    execution_service = _ExecutionServiceExecuteOnly()

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "BTCUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        market_data_provider=provider,
        risk_service=risk_service,
        execution_service=execution_service,
    )
    trader.ai_manager = ai_manager

    trader._auto_trade_loop()

    assert len(risk_service.calls) == 1
    decision = risk_service.calls[0]
    assert decision.should_trade is True
    metadata = decision.details.get("risk_service", {}).get("response")
    assert metadata is not None
    assert metadata["type"]
    if isinstance(response, str):
        expected_value = response.strip()
        if len(expected_value) > 120:
            expected_value = expected_value[:117] + "..."
        assert metadata.get("value") == expected_value
    elif isinstance(response, (bool, int, float)):
        assert metadata.get("value") == response
    elif isinstance(response, dict):
        assert "keys" in metadata
    elif isinstance(response, (list, tuple, set)):
        assert metadata.get("size") == len(response)
    else:
        assert "repr" in metadata or "value" in metadata

    if expected_execute:
        assert execution_service.calls == [decision]
    else:
        assert execution_service.calls == []


def test_auto_trader_records_risk_evaluation_history() -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())
    ai_manager = _AIManagerStub(
        [
            _DummyAssessment(regime=MarketRegime.TREND, risk_score=0.25, confidence=0.78),
            _DummyAssessment(regime=MarketRegime.TREND, risk_score=0.26, confidence=0.79),
        ]
    )

    responses = iter([True, False])
    risk_service = _RiskServiceResponseStub(lambda: next(responses))
    execution_service = _ExecutionServiceStub()

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "SOLUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        market_data_provider=provider,
        risk_service=risk_service,
        execution_service=execution_service,
    )
    trader.ai_manager = ai_manager

    trader._auto_trade_loop()
    trader._auto_trade_loop()

    evaluations = trader.get_risk_evaluations()
    assert len(evaluations) == 2

    first, second = evaluations
    assert first["approved"] is True
    assert first["normalized"] is True
    assert first["service"] == "_RiskServiceResponseStub"
    assert first["response"]["type"] == "bool"
    assert first["response"]["value"] is True
    assert first["decision"]["should_trade"] is True

    assert second["approved"] is False
    assert second["normalized"] is False
    assert second["response"]["type"] == "bool"
    assert second["response"]["value"] is False
    assert second["decision"]["should_trade"] is True

    evaluations[0]["approved"] = None
    fresh = trader.get_risk_evaluations()
    assert fresh[0]["approved"] is True


def test_auto_trader_limits_and_clears_risk_history() -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())
    ai_manager = _AIManagerStub(
        [
            _DummyAssessment(regime=MarketRegime.TREND, risk_score=0.21, confidence=0.76),
            _DummyAssessment(regime=MarketRegime.TREND, risk_score=0.22, confidence=0.77),
            _DummyAssessment(regime=MarketRegime.TREND, risk_score=0.23, confidence=0.78),
            _DummyAssessment(regime=MarketRegime.TREND, risk_score=0.24, confidence=0.79),
            _DummyAssessment(regime=MarketRegime.TREND, risk_score=0.25, confidence=0.8),
        ]
    )

    responses = iter([True, False, True, True, False])
    risk_service = _RiskServiceResponseStub(lambda: next(responses))
    execution_service = _ExecutionServiceStub()

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "ETHUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        market_data_provider=provider,
        risk_service=risk_service,
        execution_service=execution_service,
        risk_evaluations_limit=3,
    )
    trader.ai_manager = ai_manager

    for _ in range(5):
        trader._auto_trade_loop()

    evaluations = trader.get_risk_evaluations()
    assert len(evaluations) == 3
    assert [entry["response"]["value"] for entry in evaluations] == [True, True, False]
    assert [entry["normalized"] for entry in evaluations] == [True, True, False]

    trader.clear_risk_evaluations()
    assert trader.get_risk_evaluations() == []


def test_auto_trader_filters_and_summarizes_risk_history() -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())
    ai_manager = _AIManagerStub(
        [
            _DummyAssessment(regime=MarketRegime.TREND, risk_score=0.22, confidence=0.76),
            _DummyAssessment(regime=MarketRegime.TREND, risk_score=0.23, confidence=0.77),
            _DummyAssessment(regime=MarketRegime.TREND, risk_score=0.24, confidence=0.78),
            _DummyAssessment(regime=MarketRegime.TREND, risk_score=0.25, confidence=0.79),
        ]
    )

    responses = iter([True, False, RuntimeError("boom"), "approved"])

    def _next_response() -> Any:
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value

    risk_service = _RiskServiceResponseStub(_next_response)
    execution_service = _ExecutionServiceStub()

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "ADAUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        market_data_provider=provider,
        risk_service=risk_service,
        execution_service=execution_service,
        risk_evaluations_limit=None,
    )
    trader.ai_manager = ai_manager

    for _ in range(4):
        trader._auto_trade_loop()

    evaluations = trader.get_risk_evaluations()
    assert len(evaluations) == 4
    assert [entry["normalized"] for entry in evaluations] == [True, False, False, True]
    assert any("error" in entry for entry in evaluations)

    approved = trader.get_risk_evaluations(approved=True)
    assert [entry["approved"] for entry in approved] == [True, True]

    normalized_true = trader.get_risk_evaluations(normalized=True)
    assert len(normalized_true) == 2
    assert all(entry["normalized"] is True for entry in normalized_true)

    normalized_false_no_errors = trader.get_risk_evaluations(normalized=False, include_errors=False)
    assert len(normalized_false_no_errors) == 1
    assert "error" not in normalized_false_no_errors[0]

    latest = trader.get_risk_evaluations(limit=1, reverse=True)
    assert len(latest) == 1
    assert latest[0]["approved"] is True
    assert latest[0]["normalized"] is True

    normalized_true[0]["normalized"] = False
    assert trader.get_risk_evaluations(normalized=True)[0]["normalized"] is True

    summary = trader.summarize_risk_evaluations()
    assert summary["total"] == 4
    assert summary["approved"] == 2
    assert summary["rejected"] == 2
    assert summary["unknown"] == 0
    assert summary["errors"] == 1
    assert summary["raw_true"] == 2
    assert summary["raw_false"] == 2
    assert summary["raw_none"] == 0
    assert summary["approval_rate"] == pytest.approx(0.5)
    assert summary["error_rate"] == pytest.approx(0.25)
    assert summary["first_timestamp"] <= summary["last_timestamp"]


def test_auto_trader_risk_history_filters_by_service_and_time(monkeypatch: pytest.MonkeyPatch) -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "SOLUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=False,
        market_data_provider=provider,
    )

    decision = RiskDecision(
        should_trade=True,
        fraction=0.42,
        state="active",
        details={"origin": "unit-test"},
    )

    class _ServiceAlpha:
        ...

    class _ServiceBeta:
        ...

    alpha = _ServiceAlpha()
    beta = _ServiceBeta()

    timestamps = iter([1000.0, 1010.0, 1020.0, 1030.0])
    monkeypatch.setattr("bot_core.auto_trader.app.time.time", lambda: next(timestamps))

    trader._record_risk_evaluation(
        decision,
        approved=True,
        normalized=True,
        response=True,
        service=alpha,
        error=None,
    )
    rejected_decision = RiskDecision(
        should_trade=False,
        fraction=0.13,
        state="blocked",
        details={"origin": "unit-test"},
    )
    trader._record_risk_evaluation(
        rejected_decision,
        approved=False,
        normalized=False,
        response=False,
        service=beta,
        error=None,
    )
    trader._record_risk_evaluation(
        decision,
        approved=None,
        normalized=None,
        response=None,
        service=None,
        error=RuntimeError("risk failure"),
    )
    trader._record_risk_evaluation(
        decision,
        approved=True,
        normalized=True,
        response=True,
        service=alpha,
        error=None,
    )

    alpha_entries = trader.get_risk_evaluations(service="_ServiceAlpha")
    assert len(alpha_entries) == 2
    assert all(entry["service"] == "_ServiceAlpha" for entry in alpha_entries)

    unknown_entries = trader.get_risk_evaluations(service="<unknown>")
    assert len(unknown_entries) == 1
    assert "service" not in unknown_entries[0]

    window = trader.get_risk_evaluations(
        since=pd.Timestamp(1010.0, unit="s"),
        until=pd.Timestamp(1020.0, unit="s"),
    )
    assert len(window) == 2
    assert {entry.get("service", "<unknown>") for entry in window} == {"_ServiceBeta", "<unknown>"}

    summary = trader.summarize_risk_evaluations()
    assert summary["total"] == 4
    assert summary["services"]["_ServiceAlpha"]["total"] == 2
    assert summary["services"]["_ServiceAlpha"]["approval_rate"] == pytest.approx(1.0)
    assert summary["services"]["_ServiceBeta"]["rejected"] == 1
    assert summary["services"]["<unknown>"]["errors"] == 1

    filtered_summary = trader.summarize_risk_evaluations(service="_ServiceAlpha")
    assert filtered_summary["total"] == 2
    assert set(filtered_summary["services"].keys()) == {"_ServiceAlpha"}

    no_error_summary = trader.summarize_risk_evaluations(include_errors=False)
    assert no_error_summary["total"] == 3
    assert "<unknown>" not in no_error_summary["services"]

    df = trader.risk_evaluations_to_dataframe()
    assert len(df) == 4
    assert {"timestamp", "approved", "normalized", "decision"}.issubset(df.columns)
    assert df.loc[pd.isna(df["service"]), "error"].iloc[0].startswith("RuntimeError")

    alpha_df = trader.risk_evaluations_to_dataframe(service="_ServiceAlpha")
    assert len(alpha_df) == 2
    assert set(alpha_df["service"].unique()) == {"_ServiceAlpha"}

    window_df = trader.risk_evaluations_to_dataframe(
        since=pd.Timestamp(1010.0, unit="s"),
        until=pd.Timestamp(1020.0, unit="s"),
    )
    assert len(window_df) == 2
    assert set(window_df.get("service", pd.Series(index=window_df.index)).fillna("<unknown>").unique()) == {
        "_ServiceBeta",
        "<unknown>",
    }

    no_error_df = trader.risk_evaluations_to_dataframe(include_errors=False)
    assert len(no_error_df) == 3
    assert no_error_df.get("error").isna().all()

    flattened_df = trader.risk_evaluations_to_dataframe(flatten_decision=True)
    assert {
        "decision_should_trade",
        "decision_fraction",
        "decision_state",
    }.issubset(flattened_df.columns)
    assert bool(flattened_df.loc[0, "decision_should_trade"]) is True
    assert bool(flattened_df.loc[1, "decision_should_trade"]) is False
    assert flattened_df.loc[0, "decision_fraction"] == pytest.approx(0.42)
    assert flattened_df.loc[1, "decision_fraction"] == pytest.approx(0.13)
    assert flattened_df.loc[0, "decision_details"]["origin"] == "unit-test"

    prefixed_df = trader.risk_evaluations_to_dataframe(
        flatten_decision=True,
        decision_prefix="risk__",
    )
    assert {"risk__should_trade", "risk__fraction"}.issubset(prefixed_df.columns)
    assert "decision_should_trade" not in prefixed_df.columns

    subset_df = trader.risk_evaluations_to_dataframe(
        flatten_decision=True,
        decision_fields=["fraction", "details"],
    )
    subset_flattened_columns = [
        column for column in subset_df.columns if column.startswith("decision_")
    ]
    assert subset_flattened_columns == ["decision_fraction", "decision_details"]
    assert subset_df.loc[0, "decision_fraction"] == pytest.approx(0.42)
    assert subset_df.loc[1, "decision_fraction"] == pytest.approx(0.13)
    assert subset_df.loc[0, "decision_details"]["origin"] == "unit-test"

    drop_df = trader.risk_evaluations_to_dataframe(
        flatten_decision=True,
        drop_decision_column=True,
    )
    assert "decision" not in drop_df.columns
    assert {"decision_should_trade", "decision_fraction"}.issubset(drop_df.columns)

    fill_df = trader.risk_evaluations_to_dataframe(
        flatten_decision=True,
        decision_fields=["missing_field"],
        fill_value="missing",
    )
    assert (fill_df["decision_missing_field"] == "missing").all()

    flattened_df.loc[0, "decision_should_trade"] = False
    assert trader.get_risk_evaluations()[0]["decision"]["should_trade"] is True

    df.loc[pd.isna(df["service"]), "normalized"] = False
    assert trader.get_risk_evaluations()[2]["normalized"] is None
