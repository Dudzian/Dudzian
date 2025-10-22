from __future__ import annotations

from copy import deepcopy
from typing import Callable, Mapping

import pytest

import pandas as pd

from bot_core.ai.manager import AIManager
from bot_core.ai.regime import (
    MarketRegime,
    MarketRegimeAssessment,
    RegimeHistory,
    RiskLevel,
)


def _assessment(
    symbol: str,
    regime: MarketRegime,
    risk: float,
    confidence: float,
    *,
    drawdown: float = 0.05,
    volume_trend: float = 0.02,
    volatility_ratio: float = 1.05,
    volatility: float = 0.02,
    return_skew: float = 0.0,
    return_kurtosis: float = 0.0,
    volume_imbalance: float = 0.0,
) -> MarketRegimeAssessment:
    return MarketRegimeAssessment(
        regime=regime,
        confidence=confidence,
        risk_score=risk,
        metrics={
            "trend_strength": 0.01,
            "volatility": volatility,
            "momentum": 0.0,
            "autocorr": -0.1,
            "intraday_vol": 0.01,
            "drawdown": drawdown,
            "volatility_ratio": volatility_ratio,
            "volume_trend": volume_trend,
            "return_skew": return_skew,
            "return_kurtosis": return_kurtosis,
            "volume_imbalance": volume_imbalance,
        },
        symbol=symbol,
    )


def test_regime_history_validates_parameters() -> None:
    with pytest.raises(ValueError):
        RegimeHistory(maxlen=0)
    with pytest.raises(ValueError):
        RegimeHistory(decay=0.0)


def test_regime_history_emphasises_recent_regimes() -> None:
    history = RegimeHistory(maxlen=3, decay=0.5)
    history.update(_assessment("x", MarketRegime.TREND, risk=0.2, confidence=0.4))
    history.update(_assessment("x", MarketRegime.MEAN_REVERSION, risk=0.6, confidence=0.6))
    history.update(_assessment("x", MarketRegime.MEAN_REVERSION, risk=0.4, confidence=0.8))

    summary = history.summarise()
    assert summary is not None
    assert summary.regime is MarketRegime.MEAN_REVERSION
    assert summary.history[-1].regime is MarketRegime.MEAN_REVERSION
    assert pytest.approx(summary.risk_score, rel=1e-3) == 0.4285714286
    assert 0.7 <= summary.confidence <= 0.8
    assert 0.5 <= summary.stability <= 1.0
    assert summary.risk_trend > 0
    assert summary.risk_level in {RiskLevel.BALANCED, RiskLevel.WATCH, RiskLevel.ELEVATED}
    assert summary.risk_volatility > 0
    assert 0.0 <= summary.regime_persistence <= 1.0
    assert 0.0 <= summary.transition_rate <= 1.0
    assert summary.confidence_trend >= 0.0
    assert summary.confidence_volatility >= 0.0
    assert summary.regime_streak >= 1
    assert 0.0 <= summary.instability_score <= 1.0
    assert summary.confidence_decay >= 0.0
    assert summary.avg_drawdown >= 0.0
    assert 0.0 <= summary.drawdown_pressure <= 1.0
    assert -1.0 <= summary.avg_volume_trend <= 1.0
    assert 0.0 <= summary.liquidity_pressure <= 1.0
    assert summary.volatility_ratio >= 0.0
    assert 0.0 <= summary.regime_entropy <= 1.0
    assert 0.0 <= summary.tail_risk_index <= 1.0
    assert 0.0 <= summary.shock_frequency <= 1.0
    assert summary.volatility_of_volatility >= 0.0
    assert 0.0 <= summary.stress_index <= 1.0
    assert 0.0 <= summary.severe_event_rate <= 1.0
    assert 0.0 <= summary.cooldown_score <= 1.0
    assert 0.0 <= summary.recovery_potential <= 1.0
    assert 0.0 <= summary.resilience_score <= 1.0
    assert 0.0 <= summary.stress_balance <= 1.0
    assert 0.0 <= summary.liquidity_gap <= 1.0
    assert 0.0 <= summary.confidence_resilience <= 1.0
    assert 0.0 <= summary.stress_projection <= 1.0
    assert 0.0 <= summary.stress_momentum <= 1.0
    assert 0.0 <= summary.liquidity_trend <= 1.0
    assert 0.0 <= summary.confidence_fragility <= 1.0
    assert isinstance(summary.volatility_trend, float)
    assert isinstance(summary.drawdown_trend, float)
    assert summary.volume_trend_volatility >= 0.0
    assert 0.0 <= summary.stability_projection <= 1.0
    assert 0.0 <= summary.degradation_score <= 1.0
    assert -5.0 <= summary.skewness_bias <= 5.0
    assert -5.0 <= summary.kurtosis_excess <= 10.0
    assert -1.0 <= summary.volume_imbalance <= 1.0
    assert 0.0 <= summary.distribution_pressure <= 1.0


def test_regime_history_returns_none_when_empty() -> None:
    history = RegimeHistory()
    assert history.summarise() is None


def test_regime_history_reload_thresholds_and_snapshot() -> None:
    calls = 0

    def _loader() -> dict:
        nonlocal calls
        calls += 1
        return {
            "market_regime": {
                "risk_level": {
                    "critical": {"risk_score": 0.91},
                    "calm": {"risk_score": 0.2},
                }
            }
        }

    history = RegimeHistory(thresholds_loader=_loader)
    assert calls == 1

    snapshot = history.thresholds_snapshot()
    assert snapshot["market_regime"]["risk_level"]["critical"]["risk_score"] == 0.91
    snapshot["market_regime"]["risk_level"]["critical"]["risk_score"] = 0.5
    assert (
        history.thresholds_snapshot()["market_regime"]["risk_level"]["critical"]["risk_score"]
        == 0.91
    )

    history.reload_thresholds(
        thresholds={
            "market_regime": {
                "risk_level": {
                    "critical": {"risk_score": 0.72},
                    "calm": {"risk_score": 0.15},
                }
            }
        }
    )
    updated = history.thresholds_snapshot()
    assert updated["market_regime"]["risk_level"]["critical"]["risk_score"] == 0.72
    assert updated["market_regime"]["risk_level"]["calm"]["risk_score"] == 0.15

    history.reload_thresholds()
    assert calls == 2


class _ClassifierStub:
    def __init__(self, assessments: list[MarketRegimeAssessment]) -> None:
        self._queue = assessments
        self._thresholds = {
            "market_regime": {
                "metrics": {},
                "risk_score": {},
                "risk_level": {},
            }
        }

    @property
    def thresholds_loader(self) -> Callable[[], Mapping[str, object]]:
        return lambda: self._thresholds

    def thresholds_snapshot(self) -> Mapping[str, object]:
        return deepcopy(self._thresholds)

    def assess(self, market_data: pd.DataFrame, *, price_col: str = "close", symbol: str | None = None) -> MarketRegimeAssessment:
        assessment = self._queue.pop(0)
        return assessment


def test_ai_manager_exposes_regime_summary() -> None:
    symbol = "BTCUSDT"
    assessments = [
        _assessment(symbol, MarketRegime.TREND, risk=0.3, confidence=0.6),
        _assessment(symbol, MarketRegime.MEAN_REVERSION, risk=0.5, confidence=0.7),
    ]
    manager = AIManager()
    manager._regime_classifier = _ClassifierStub(assessments)  # type: ignore[attr-defined]
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})

    manager.assess_market_regime(symbol, df)
    manager.assess_market_regime(symbol, df)

    summary = manager.get_regime_summary(symbol)
    assert summary is not None
    assert summary.regime is MarketRegime.MEAN_REVERSION
    assert summary.risk_score > 0.4
    assert summary.confidence > 0.6
    assert 0.0 <= summary.stability <= 1.0
    assert isinstance(summary.risk_trend, float)
    assert summary.risk_level in {
        RiskLevel.BALANCED,
        RiskLevel.WATCH,
        RiskLevel.ELEVATED,
    }
    assert isinstance(summary.confidence_trend, float)
    assert isinstance(summary.confidence_volatility, float)
    assert isinstance(summary.regime_streak, int)
    assert isinstance(summary.transition_rate, float)
    assert isinstance(summary.instability_score, float)
    assert isinstance(summary.confidence_decay, float)
    assert isinstance(summary.stress_momentum, float)
    assert isinstance(summary.liquidity_trend, float)
    assert isinstance(summary.confidence_fragility, float)


def test_regime_history_risk_trend_reflects_direction() -> None:
    history = RegimeHistory(maxlen=4, decay=0.7)
    history.update(_assessment("x", MarketRegime.TREND, risk=0.6, confidence=0.4))
    history.update(_assessment("x", MarketRegime.TREND, risk=0.5, confidence=0.5))
    history.update(_assessment("x", MarketRegime.TREND, risk=0.4, confidence=0.6))

    summary = history.summarise()
    assert summary is not None
    assert summary.risk_trend < 0
    assert summary.risk_level in {RiskLevel.BALANCED, RiskLevel.WATCH, RiskLevel.CALM}
    assert summary.risk_volatility >= 0.0
    assert summary.regime_persistence == 1.0
    assert summary.transition_rate == 0.0
    assert summary.confidence_trend >= 0.0
    assert summary.confidence_volatility >= 0.0
    assert summary.regime_streak == len(history.snapshots)
    assert summary.instability_score <= 0.3
    assert summary.confidence_decay == pytest.approx(0.0)
    assert summary.drawdown_pressure <= 0.3
    assert summary.liquidity_pressure <= 0.5
    assert summary.recovery_potential >= 0.2
    assert summary.cooldown_score <= 0.5
    assert summary.degradation_score <= 0.5
    assert summary.stability_projection >= 0.3
    assert 0.0 <= summary.regime_entropy <= 1.0
    assert 0.0 <= summary.resilience_score <= 1.0
    assert 0.0 <= summary.stress_balance <= 1.0
    assert 0.0 <= summary.liquidity_gap <= 1.0
    assert 0.0 <= summary.confidence_resilience <= 1.0
    assert 0.0 <= summary.stress_projection <= 1.0
    baseline_degradation = summary.degradation_score

    history.update(_assessment("x", MarketRegime.TREND, risk=0.7, confidence=0.6))
    summary = history.summarise()
    assert summary is not None
    assert summary.risk_trend > 0
    assert summary.risk_level in {RiskLevel.ELEVATED, RiskLevel.CRITICAL}
    assert summary.risk_volatility >= 0.0
    assert summary.regime_persistence == 1.0
    assert summary.confidence_trend >= 0.0
    assert summary.confidence_volatility >= 0.0
    assert summary.drawdown_pressure >= 0.18
    assert summary.liquidity_pressure <= 0.6
    assert summary.cooldown_score >= 0.04
    assert summary.severe_event_rate >= 0.0
    assert summary.degradation_score >= baseline_degradation
    assert 0.0 <= summary.regime_entropy <= 1.0
    assert 0.0 <= summary.resilience_score <= 1.0
    assert 0.0 <= summary.stress_balance <= 1.0
    assert 0.0 <= summary.liquidity_gap <= 1.0
    assert 0.0 <= summary.confidence_resilience <= 1.0
    assert 0.0 <= summary.stress_projection <= 1.0
    # Stability projection can remain high for persistent regimes even as risk climbs


def test_regime_history_risk_level_extremes() -> None:
    calm_history = RegimeHistory(maxlen=4, decay=0.6)
    for _ in range(4):
        calm_history.update(
            _assessment(
                "calm",
                MarketRegime.DAILY,
                risk=0.18,
                confidence=0.7,
                drawdown=0.02,
                volume_trend=0.06,
                volatility_ratio=1.02,
            )
        )
    calm_summary = calm_history.summarise()
    assert calm_summary is not None
    assert calm_summary.risk_level is RiskLevel.CALM
    assert calm_summary.risk_volatility == 0.0
    assert calm_summary.regime_persistence == 1.0
    assert calm_summary.confidence_trend == pytest.approx(0.0)
    assert calm_summary.confidence_volatility == pytest.approx(0.0)
    assert calm_summary.regime_streak == calm_history.maxlen
    assert calm_summary.drawdown_pressure <= 0.35
    assert calm_summary.liquidity_pressure <= 0.35
    assert calm_summary.volatility_ratio <= 1.1
    assert calm_summary.tail_risk_index <= 0.3
    assert calm_summary.shock_frequency == pytest.approx(0.0)
    assert calm_summary.volatility_of_volatility == pytest.approx(0.0)
    assert calm_summary.stress_index <= 0.3
    assert calm_summary.cooldown_score <= 0.35
    assert calm_summary.severe_event_rate <= 0.35
    assert calm_summary.recovery_potential >= 0.3
    assert calm_summary.degradation_score <= 0.3
    assert calm_summary.stability_projection >= 0.5
    assert calm_summary.volatility_trend == pytest.approx(0.0)
    assert calm_summary.drawdown_trend == pytest.approx(0.0)
    assert calm_summary.volume_trend_volatility == pytest.approx(0.0)

    danger_history = RegimeHistory(maxlen=4, decay=0.8)
    danger_history.update(
        _assessment(
            "danger",
            MarketRegime.TREND,
            risk=0.9,
            confidence=0.6,
            drawdown=0.35,
            volume_trend=-0.25,
            volatility_ratio=1.55,
        )
    )
    danger_history.update(
        _assessment(
            "danger",
            MarketRegime.TREND,
            risk=0.92,
            confidence=0.6,
            drawdown=0.4,
            volume_trend=-0.3,
            volatility_ratio=1.6,
        )
    )
    danger_summary = danger_history.summarise()
    assert danger_summary is not None
    assert danger_summary.risk_level is RiskLevel.CRITICAL
    assert danger_summary.risk_volatility >= 0.0
    assert danger_summary.regime_persistence == 1.0
    assert danger_summary.confidence_trend == pytest.approx(0.0)
    assert danger_summary.confidence_volatility == pytest.approx(0.0)
    assert danger_summary.transition_rate == pytest.approx(0.0)
    assert danger_summary.instability_score <= 0.5
    assert danger_summary.confidence_decay == pytest.approx(0.0)
    assert danger_summary.drawdown_pressure >= 0.8
    assert danger_summary.liquidity_pressure >= 0.6
    assert danger_summary.volatility_ratio >= 1.5
    assert danger_summary.tail_risk_index >= 0.5
    assert danger_summary.stress_index >= 0.45
    assert danger_summary.cooldown_score >= 0.55
    assert danger_summary.severe_event_rate >= 0.5
    assert danger_summary.recovery_potential <= 0.4
    assert danger_summary.degradation_score >= max(calm_summary.degradation_score, 0.15)
    # Persistent high-risk streaks can still project elevated stability due to streak dominance
    assert danger_summary.volatility_trend >= 0.0
    assert danger_summary.drawdown_trend >= 0.0
    assert danger_summary.volume_trend_volatility >= 0.0


def test_regime_history_distribution_pressure_increases_risk() -> None:
    history = RegimeHistory(maxlen=5, decay=0.7)
    for _ in range(5):
        history.update(
            _assessment(
                "dist",
                MarketRegime.DAILY,
                risk=0.5,
                confidence=0.55,
                drawdown=0.12,
                volatility=0.03,
                volatility_ratio=1.3,
                volume_trend=-0.25,
                return_skew=2.2,
                return_kurtosis=4.5,
                volume_imbalance=0.65,
            )
        )

    summary = history.summarise()
    assert summary is not None
    assert summary.distribution_pressure >= 0.6
    assert summary.risk_level in {RiskLevel.ELEVATED, RiskLevel.CRITICAL}
    assert summary.skewness_bias > 1.0
    assert summary.kurtosis_excess > 3.0
    assert summary.volume_imbalance > 0.4


def test_regime_history_detects_choppy_market() -> None:
    history = RegimeHistory(maxlen=5, decay=0.6)
    history.update(_assessment("x", MarketRegime.TREND, risk=0.55, confidence=0.5))
    history.update(_assessment("x", MarketRegime.MEAN_REVERSION, risk=0.52, confidence=0.55))
    history.update(_assessment("x", MarketRegime.DAILY, risk=0.5, confidence=0.6))
    history.update(_assessment("x", MarketRegime.MEAN_REVERSION, risk=0.58, confidence=0.6))

    summary = history.summarise()
    assert summary is not None
    assert summary.risk_level in {RiskLevel.WATCH, RiskLevel.ELEVATED}
    assert summary.risk_volatility >= 0.02
    assert summary.regime_persistence < 0.5
    assert summary.transition_rate > 0.5
    assert abs(summary.confidence_trend) < 0.2
    assert summary.confidence_volatility <= 0.1
    assert summary.instability_score >= 0.35
    assert summary.confidence_decay == pytest.approx(0.0)
    assert summary.liquidity_pressure >= 0.0


def test_regime_history_confidence_metrics_drive_risk_level() -> None:
    history = RegimeHistory(maxlen=4, decay=0.6)
    history.update(_assessment("x", MarketRegime.DAILY, risk=0.42, confidence=0.75))
    history.update(_assessment("x", MarketRegime.DAILY, risk=0.44, confidence=0.5))
    history.update(_assessment("x", MarketRegime.MEAN_REVERSION, risk=0.46, confidence=0.35))

    summary = history.summarise()
    assert summary is not None
    assert summary.confidence_trend < 0
    assert summary.confidence_volatility > 0
    assert summary.regime_streak == 1
    assert summary.risk_level in {RiskLevel.WATCH, RiskLevel.ELEVATED}


def test_regime_history_drawdown_and_liquidity_escalate_risk() -> None:
    history = RegimeHistory(maxlen=3, decay=0.7)
    history.update(
        _assessment(
            "stress",
            MarketRegime.DAILY,
            risk=0.5,
            confidence=0.55,
            drawdown=0.3,
            volume_trend=-0.35,
            volatility_ratio=1.45,
        )
    )
    history.update(
        _assessment(
            "stress",
            MarketRegime.MEAN_REVERSION,
            risk=0.58,
            confidence=0.5,
            drawdown=0.38,
            volume_trend=-0.4,
            volatility_ratio=1.55,
        )
    )
    summary = history.summarise()
    assert summary is not None
    assert summary.drawdown_pressure >= 0.6
    assert summary.liquidity_pressure >= 0.5
    assert summary.volatility_ratio >= 1.45
    assert summary.risk_level in {RiskLevel.ELEVATED, RiskLevel.CRITICAL}
    assert summary.transition_rate >= 0.5
    assert summary.instability_score >= 0.35
    assert summary.confidence_decay > 0
    assert summary.cooldown_score >= 0.45
    assert summary.severe_event_rate >= 0.4


def test_regime_history_degradation_escalates_risk_level() -> None:
    history = RegimeHistory(maxlen=4, decay=0.7)
    history.update(
        _assessment(
            "deg",
            MarketRegime.DAILY,
            risk=0.45,
            confidence=0.55,
            drawdown=0.12,
            volume_trend=-0.05,
            volatility_ratio=1.2,
            volatility=0.02,
        )
    )
    history.update(
        _assessment(
            "deg",
            MarketRegime.DAILY,
            risk=0.58,
            confidence=0.52,
            drawdown=0.2,
            volume_trend=-0.18,
            volatility_ratio=1.4,
            volatility=0.03,
        )
    )
    history.update(
        _assessment(
            "deg",
            MarketRegime.TREND,
            risk=0.68,
            confidence=0.5,
            drawdown=0.26,
            volume_trend=-0.22,
            volatility_ratio=1.5,
            volatility=0.038,
        )
    )
    history.update(
        _assessment(
            "deg",
            MarketRegime.TREND,
            risk=0.72,
            confidence=0.48,
            drawdown=0.3,
            volume_trend=-0.28,
            volatility_ratio=1.55,
            volatility=0.04,
        )
    )

    summary = history.summarise()
    assert summary is not None
    assert summary.degradation_score >= 0.55
    assert summary.stability_projection <= 0.4
    assert summary.volatility_trend > 0
    assert summary.drawdown_trend > 0
    assert summary.volume_trend_volatility > 0
    assert summary.risk_level in {RiskLevel.ELEVATED, RiskLevel.CRITICAL}
    assert summary.cooldown_score >= 0.45


def test_regime_history_instability_raises_risk() -> None:
    history = RegimeHistory(maxlen=5, decay=0.7)
    history.update(_assessment("x", MarketRegime.DAILY, risk=0.42, confidence=0.72))
    history.update(_assessment("x", MarketRegime.TREND, risk=0.55, confidence=0.55))
    history.update(_assessment("x", MarketRegime.MEAN_REVERSION, risk=0.52, confidence=0.45))
    history.update(_assessment("x", MarketRegime.DAILY, risk=0.5, confidence=0.4))

    summary = history.summarise()
    assert summary is not None
    assert summary.transition_rate >= 0.5
    assert summary.instability_score >= 0.5
    assert summary.confidence_decay > 0
    assert summary.risk_level in {RiskLevel.ELEVATED, RiskLevel.CRITICAL}
    assert summary.cooldown_score >= 0.1


def test_regime_history_low_instability_confirms_calm() -> None:
    history = RegimeHistory(maxlen=4, decay=0.65)
    for confidence in (0.62, 0.65, 0.68, 0.7):
        history.update(
            MarketRegimeAssessment(
                regime=MarketRegime.DAILY,
                confidence=confidence,
                risk_score=0.2,
                metrics={
                    "trend_strength": 0.01,
                    "volatility": 0.01,
                    "momentum": 0.0,
                    "autocorr": -0.1,
                    "intraday_vol": 0.005,
                    "drawdown": 0.02,
                },
                symbol="calm",
            )
        )

    summary = history.summarise()
    assert summary is not None
    assert summary.risk_level is RiskLevel.CALM
    assert summary.transition_rate == pytest.approx(0.0)
    assert summary.instability_score <= 0.25
    assert summary.confidence_decay == pytest.approx(0.0)


def test_regime_history_tail_risk_and_shocks_raise_stress_index() -> None:
    history = RegimeHistory(maxlen=5, decay=0.7)
    history.update(
        _assessment(
            "stress",
            MarketRegime.TREND,
            risk=0.58,
            confidence=0.55,
            drawdown=0.28,
            volume_trend=-0.3,
            volatility_ratio=1.42,
        )
    )
    history.update(
        _assessment(
            "stress",
            MarketRegime.MEAN_REVERSION,
            risk=0.72,
            confidence=0.5,
            drawdown=0.32,
            volume_trend=-0.34,
            volatility_ratio=1.5,
        )
    )
    history.update(
        _assessment(
            "stress",
            MarketRegime.DAILY,
            risk=0.68,
            confidence=0.48,
            drawdown=0.35,
            volume_trend=-0.36,
            volatility_ratio=1.52,
        )
    )

    summary = history.summarise()
    assert summary is not None
    assert summary.tail_risk_index >= 0.5
    assert summary.shock_frequency >= 0.5
    assert summary.stress_index >= 0.6
    assert summary.risk_level in {RiskLevel.ELEVATED, RiskLevel.CRITICAL}
    assert summary.cooldown_score >= 0.5
    assert summary.severe_event_rate >= 0.5


def test_regime_history_recovery_potential_signals_improvement() -> None:
    history = RegimeHistory(maxlen=4, decay=0.7)
    history.update(
        _assessment(
            "rec",
            MarketRegime.TREND,
            risk=0.82,
            confidence=0.45,
            drawdown=0.32,
            volume_trend=-0.28,
            volatility_ratio=1.5,
        )
    )
    history.update(
        _assessment(
            "rec",
            MarketRegime.MEAN_REVERSION,
            risk=0.62,
            confidence=0.5,
            drawdown=0.22,
            volume_trend=-0.1,
            volatility_ratio=1.3,
        )
    )
    history.update(
        _assessment(
            "rec",
            MarketRegime.DAILY,
            risk=0.38,
            confidence=0.65,
            drawdown=0.12,
            volume_trend=0.08,
            volatility_ratio=1.1,
        )
    )

    summary = history.summarise()
    assert summary is not None
    assert summary.risk_trend < 0
    assert summary.recovery_potential >= 0.2
    assert summary.cooldown_score <= 0.6
    assert summary.severe_event_rate <= 0.7
