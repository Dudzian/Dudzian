from __future__ import annotations

import time
import enum
from dataclasses import dataclass
from typing import Any
from types import SimpleNamespace

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
    def __init__(self, approval: bool) -> None:
        self.approval = approval
        self.calls: list[RiskDecision] = []

    def evaluate_decision(self, decision: RiskDecision) -> bool:
        self.calls.append(decision)
        return self.approval


class _RiskServiceResponseStub:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[RiskDecision] = []

    def evaluate_decision(self, decision: RiskDecision) -> Any:
        self.calls.append(decision)
        if callable(self._response):
            return self._response()
        return self._response


class _ExecutionServiceStub:
    def __init__(self) -> None:
        self.calls: list[RiskDecision] = []
        self.methods: list[str] = []

    def execute_decision(self, decision: RiskDecision) -> None:
        self.methods.append("execute_decision")
        self.calls.append(decision)

    def execute(self, decision: RiskDecision) -> None:
        self.methods.append("execute")
        self.calls.append(decision)


class _ExecutionServiceExecuteOnly:
    def __init__(self) -> None:
        self.calls: list[RiskDecision] = []

    def execute(self, decision: RiskDecision) -> None:
        self.calls.append(decision)


class _Approval(enum.Enum):
    APPROVED = "approved"
    DENIED = "denied"


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


def _build_market_data() -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=120, freq="min")
    close = pd.Series(100 + (idx - idx[0]).total_seconds() / 3600.0, index=idx)
    high = close * 1.001
    low = close * 0.999
    volume = pd.Series(1_000, index=idx)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": volume})


def _build_summary(
    regime: MarketRegime,
    *,
    confidence: float,
    risk: float,
    size: int = 3,
    stability: float | None = None,
    risk_trend: float | None = None,
    risk_level: RiskLevel | None = None,
    risk_volatility: float | None = None,
    regime_persistence: float | None = None,
    confidence_path: tuple[float, ...] | None = None,
    regime_sequence: tuple[MarketRegime, ...] | None = None,
    confidence_trend: float | None = None,
    confidence_volatility: float | None = None,
    regime_streak: int | None = None,
    transition_rate: float | None = None,
    instability_score: float | None = None,
    confidence_decay: float | None = None,
    drawdown_path: tuple[float, ...] | None = None,
    volume_trend_path: tuple[float, ...] | None = None,
    volatility_ratio_path: tuple[float, ...] | None = None,
    volatility_path: tuple[float, ...] | None = None,
    avg_drawdown: float | None = None,
    avg_volume_trend: float | None = None,
    drawdown_pressure: float | None = None,
    liquidity_pressure: float | None = None,
    volatility_ratio: float | None = None,
    tail_risk_index: float | None = None,
    shock_frequency: float | None = None,
    volatility_of_volatility: float | None = None,
    stress_index: float | None = None,
    severe_event_rate: float | None = None,
    cooldown_score: float | None = None,
    recovery_potential: float | None = None,
    volatility_trend: float | None = None,
    drawdown_trend: float | None = None,
    volume_trend_volatility: float | None = None,
    stability_projection: float | None = None,
    degradation_score: float | None = None,
    skewness_path: tuple[float, ...] | None = None,
    kurtosis_path: tuple[float, ...] | None = None,
    volume_imbalance_path: tuple[float, ...] | None = None,
    skewness_bias: float | None = None,
    kurtosis_excess: float | None = None,
    volume_imbalance: float | None = None,
    distribution_pressure: float | None = None,
    regime_entropy: float | None = None,
    resilience_score: float | None = None,
    stress_balance: float | None = None,
    liquidity_gap: float | None = None,
    confidence_resilience: float | None = None,
    stress_projection: float | None = None,
    stress_momentum: float | None = None,
    liquidity_trend: float | None = None,
    confidence_fragility: float | None = None,
) -> RegimeSummary:
    if size < 1:
        size = 1
    resolved_risk_trend = 0.0 if risk_trend is None else risk_trend
    resolved_stability = 1.0 if stability is None else stability
    risk_values: list[float]
    if size >= 2 and risk_trend is not None:
        start_risk = max(0.0, min(1.0, risk - resolved_risk_trend))
        end_risk = max(0.0, min(1.0, risk))
        step = (end_risk - start_risk) / (size - 1) if size > 1 else 0.0
        risk_values = [start_risk + step * idx for idx in range(size)]
    else:
        risk_values = [risk for _ in range(size)]

    if confidence_path is not None and len(confidence_path) == size:
        confidences = list(confidence_path)
    else:
        confidences = [confidence for _ in range(size)]

    if regime_sequence is not None and len(regime_sequence) == size:
        regimes = list(regime_sequence)
    else:
        regimes = [regime for _ in range(size)]

    counts: dict[MarketRegime, float] = {}
    for entry in regimes:
        counts[entry] = counts.get(entry, 0.0) + 1.0
    distribution = np.asarray(list(counts.values()), dtype=float)
    if distribution.size:
        probabilities = distribution / distribution.sum()
        default_regime_entropy = float(
            np.clip(
                -np.nansum(probabilities * np.log(probabilities + 1e-12))
                / (np.log(len(MarketRegime)) or 1.0),
                0.0,
                1.0,
            )
        )
    else:
        default_regime_entropy = 0.0

    drawdowns = (
        list(drawdown_path)
        if drawdown_path is not None and len(drawdown_path) == size
        else [0.12 for _ in range(size)]
    )
    volume_trends = (
        list(volume_trend_path)
        if volume_trend_path is not None and len(volume_trend_path) == size
        else [0.02 for _ in range(size)]
    )
    volatility_ratios = (
        list(volatility_ratio_path)
        if volatility_ratio_path is not None and len(volatility_ratio_path) == size
        else [1.1 for _ in range(size)]
    )

    volatilities = (
        list(volatility_path)
        if volatility_path is not None and len(volatility_path) == size
        else [0.02 for _ in range(size)]
    )

    skew_values = (
        list(skewness_path)
        if skewness_path is not None and len(skewness_path) == size
        else [0.0 for _ in range(size)]
    )
    kurtosis_values = (
        list(kurtosis_path)
        if kurtosis_path is not None and len(kurtosis_path) == size
        else [0.0 for _ in range(size)]
    )
    volume_imbalance_values = (
        list(volume_imbalance_path)
        if volume_imbalance_path is not None and len(volume_imbalance_path) == size
        else [0.0 for _ in range(size)]
    )

    snapshots = tuple(
        RegimeSnapshot(
            regime=regimes[idx],
            confidence=confidences[idx],
            risk_score=risk_values[idx],
            drawdown=drawdowns[idx],
            volatility=volatilities[idx],
            volume_trend=volume_trends[idx],
            volatility_ratio=volatility_ratios[idx],
            return_skew=skew_values[idx],
            return_kurtosis=kurtosis_values[idx],
            volume_imbalance=volume_imbalance_values[idx],
        )
        for idx in range(size)
    )
    resolved_level = risk_level
    if resolved_level is None:
        if risk >= 0.85 or resolved_risk_trend >= 0.25:
            resolved_level = RiskLevel.CRITICAL
        elif risk >= 0.65 or (resolved_risk_trend >= 0.1 and resolved_stability < 0.7):
            resolved_level = RiskLevel.ELEVATED
        elif risk <= 0.25 and resolved_risk_trend <= 0.0 and resolved_stability >= 0.55 and confidence >= 0.5:
            resolved_level = RiskLevel.CALM
        elif risk <= 0.45 and resolved_risk_trend <= 0.05:
            resolved_level = RiskLevel.BALANCED
        else:
            resolved_level = RiskLevel.WATCH
    resolved_volatility = risk_volatility
    if resolved_volatility is None:
        if len(snapshots) <= 1:
            resolved_volatility = 0.0
        else:
            values = [snap.risk_score for snap in snapshots]
            mean = sum(values) / len(values)
            resolved_volatility = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
    resolved_persistence = regime_persistence
    if resolved_persistence is None:
        if len(snapshots) <= 1:
            resolved_persistence = 1.0
        else:
            changes = sum(
                1 for idx in range(1, len(snapshots)) if snapshots[idx].regime != snapshots[idx - 1].regime
            )
            resolved_persistence = 1.0 - (changes / (len(snapshots) - 1))
    resolved_conf_trend = confidence_trend
    if resolved_conf_trend is None:
        if len(snapshots) >= 2:
            resolved_conf_trend = snapshots[-1].confidence - snapshots[0].confidence
        else:
            resolved_conf_trend = 0.0
    resolved_conf_vol = confidence_volatility
    if resolved_conf_vol is None:
        if len(snapshots) <= 1:
            resolved_conf_vol = 0.0
        else:
            values = [snap.confidence for snap in snapshots]
            mean = sum(values) / len(values)
            resolved_conf_vol = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
    resolved_streak = regime_streak
    if resolved_streak is None:
        resolved_streak = 0
        latest = snapshots[-1].regime
        for snap in reversed(snapshots):
            if snap.regime is latest:
                resolved_streak += 1
            else:
                break
    resolved_conf_trend = float(resolved_conf_trend)
    resolved_conf_vol = float(resolved_conf_vol)
    resolved_streak = int(resolved_streak)
    resolved_transition_rate = (
        float(transition_rate)
        if transition_rate is not None
        else float(max(0.0, 1.0 - resolved_persistence))
    )
    resolved_conf_decay = (
        float(confidence_decay)
        if confidence_decay is not None
        else float(max(0.0, -resolved_conf_trend))
    )
    resolved_risk_volatility = resolved_volatility if resolved_volatility is not None else 0.0
    risk_vol_norm = min(resolved_risk_volatility / 0.3, 1.0)
    conf_vol_norm = min(resolved_conf_vol / 0.2, 1.0)
    if instability_score is None:
        resolved_instability = min(
            1.0,
            0.4 * risk_vol_norm
            + 0.3 * resolved_transition_rate
            + 0.2 * conf_vol_norm
            + 0.1 * min(resolved_conf_decay, 1.0),
        )
    else:
        resolved_instability = float(instability_score)
    resolved_avg_drawdown = float(avg_drawdown) if avg_drawdown is not None else float(sum(drawdowns) / len(drawdowns))
    resolved_avg_volume_trend = (
        float(avg_volume_trend)
        if avg_volume_trend is not None
        else float(sum(volume_trends) / len(volume_trends))
    )
    volume_trend_component = min(max(0.0, -resolved_avg_volume_trend) / 0.35, 1.0)
    resolved_volatility_ratio = (
        float(volatility_ratio)
        if volatility_ratio is not None
        else float(sum(volatility_ratios) / len(volatility_ratios))
    )
    resolved_drawdown_pressure = (
        float(drawdown_pressure)
        if drawdown_pressure is not None
        else min(1.0, resolved_avg_drawdown / 0.25)
    )
    resolved_liquidity_pressure = (
        float(liquidity_pressure)
        if liquidity_pressure is not None
        else min(
            1.0,
            max(0.0, -resolved_avg_volume_trend) / 0.4
            + max(0.0, resolved_volatility_ratio - 1.0) * 0.35
            + max(0.0, resolved_instability - 0.4) * 0.2,
        )
    )
    resolved_tail_risk_index = (
        float(tail_risk_index)
        if tail_risk_index is not None
        else min(
            1.0,
            sum(
                1
                for snap in snapshots
                if snap.drawdown >= 0.22
                or snap.volatility >= 0.035
                or snap.volatility_ratio >= 1.4
            )
            / max(len(snapshots), 1),
        )
    )
    resolved_shock_frequency = (
        float(shock_frequency)
        if shock_frequency is not None
        else (
            0.0
            if len(snapshots) <= 1
            else min(
                1.0,
                (
                    sum(
                        1
                        for idx in range(1, len(snapshots))
                        if abs(snapshots[idx].risk_score - snapshots[idx - 1].risk_score) >= 0.12
                    )
                    + sum(
                        1
                        for idx in range(1, len(snapshots))
                        if snapshots[idx].regime is not snapshots[idx - 1].regime
                    )
                )
                / (len(snapshots) - 1),
            )
        )
    )
    resolved_vol_of_vol = (
        float(volatility_of_volatility)
        if volatility_of_volatility is not None
        else (float(np.std(volatilities, dtype=float)) if len(volatilities) >= 2 else 0.0)
    )
    resolved_stress_index = (
        float(stress_index)
        if stress_index is not None
        else min(
            1.0,
            0.45 * resolved_tail_risk_index
            + 0.3 * resolved_shock_frequency
            + 0.25 * max(min(resolved_risk_volatility / 0.3, 1.0), min(resolved_vol_of_vol / 0.025, 1.0)),
        )
    )
    if severe_event_rate is not None:
        resolved_severe_event_rate = float(severe_event_rate)
    else:
        severe_events = sum(
            1
            for snap in snapshots
            if (
                snap.risk_score >= 0.75
                or snap.drawdown >= 0.22
                or snap.volatility >= 0.035
                or snap.volatility_ratio >= 1.45
            )
        )
        resolved_severe_event_rate = float(
            min(1.0, severe_events / max(len(snapshots), 1))
        )
    if recovery_potential is not None:
        resolved_recovery_potential = float(recovery_potential)
    else:
        resolved_recovery_potential = float(
            min(
                1.0,
                0.45 * max(0.0, -resolved_risk_trend)
                + 0.25 * max(0.0, confidence - 0.5)
                + 0.2 * max(0.0, resolved_persistence - 0.55)
                + 0.15 * max(0.0, 0.4 - resolved_risk_volatility)
                + 0.15 * max(0.0, 0.35 - resolved_drawdown_pressure)
                + 0.15 * max(0.0, 0.35 - resolved_liquidity_pressure),
            )
    )
    if cooldown_score is not None:
        resolved_cooldown_score = float(cooldown_score)
    else:
        resolved_cooldown_score = float(
            min(
                1.0,
                0.4 * resolved_severe_event_rate
                + 0.25 * resolved_stress_index
                + 0.2 * max(0.0, resolved_instability - 0.5)
                + 0.15 * max(0.0, resolved_conf_decay)
                + 0.15 * max(0.0, resolved_drawdown_pressure - 0.5)
                + 0.15 * max(0.0, resolved_liquidity_pressure - 0.5),
            )
        )
    if volatility_trend is not None:
        resolved_volatility_trend = float(volatility_trend)
    else:
        resolved_volatility_trend = float(
            np.clip(volatilities[-1] - volatilities[0], -0.05, 0.05)
        ) if len(volatilities) >= 2 else 0.0
    if drawdown_trend is not None:
        resolved_drawdown_trend = float(drawdown_trend)
    else:
        resolved_drawdown_trend = float(
            np.clip(drawdowns[-1] - drawdowns[0], -0.4, 0.4)
        ) if len(drawdowns) >= 2 else 0.0
    if volume_trend_volatility is not None:
        resolved_volume_trend_volatility = float(volume_trend_volatility)
    else:
        resolved_volume_trend_volatility = float(
            np.std(volume_trends, dtype=float)
        ) if len(volume_trends) >= 2 else 0.0
    vol_trend_intensity = min(1.0, max(0.0, resolved_volatility_trend) / 0.03)
    drawdown_trend_intensity = min(1.0, max(0.0, resolved_drawdown_trend) / 0.18)
    volume_trend_vol_norm = min(1.0, resolved_volume_trend_volatility / 0.25)
    if degradation_score is not None:
        resolved_degradation_score = float(degradation_score)
    else:
        resolved_degradation_score = float(
            min(
                1.0,
                0.35 * vol_trend_intensity
                + 0.35 * drawdown_trend_intensity
                + 0.2 * volume_trend_vol_norm
                + 0.1 * max(0.0, resolved_instability - 0.45)
                + 0.1 * max(0.0, resolved_tail_risk_index - 0.4)
                + 0.1 * max(0.0, resolved_stress_index - 0.45),
            )
        )
    if skewness_bias is not None:
        resolved_skewness_bias = float(skewness_bias)
    else:
        resolved_skewness_bias = float(np.mean(skew_values)) if skew_values else 0.0
    if kurtosis_excess is not None:
        resolved_kurtosis_excess = float(kurtosis_excess)
    else:
        resolved_kurtosis_excess = float(np.mean(kurtosis_values)) if kurtosis_values else 0.0
    if volume_imbalance is not None:
        resolved_volume_imbalance = float(volume_imbalance)
    else:
        resolved_volume_imbalance = float(np.mean(volume_imbalance_values)) if volume_imbalance_values else 0.0
    if distribution_pressure is not None:
        resolved_distribution_pressure = float(distribution_pressure)
    else:
        resolved_distribution_pressure = float(
            min(
                1.0,
                max(
                    0.0,
                    0.3 * min(abs(resolved_skewness_bias) / 1.5, 1.0)
                    + 0.3 * min(max(0.0, resolved_kurtosis_excess) / 3.0, 1.0)
                    + 0.2 * min(abs(resolved_volume_imbalance) / 0.6, 1.0)
                    + 0.1 * max(0.0, resolved_instability - 0.45),
                ),
            )
        )
    if stability_projection is not None:
        resolved_stability_projection = float(stability_projection)
    else:
        resolved_stability_projection = float(
            min(
                1.0,
                max(
                    0.0,
                    0.45 * resolved_persistence
                    + 0.25 * max(0.0, 1.0 - resolved_transition_rate)
                    + 0.2 * max(0.0, 1.0 - risk_vol_norm)
                    + 0.15 * resolved_recovery_potential
                    - 0.25 * vol_trend_intensity
                    - 0.2 * drawdown_trend_intensity
                    - 0.15 * volume_trend_vol_norm,
                ),
            )
        )
    if regime_entropy is not None:
        resolved_regime_entropy = float(regime_entropy)
    else:
        resolved_regime_entropy = default_regime_entropy
    if stress_balance is not None:
        resolved_stress_balance = float(stress_balance)
    else:
        resolved_stress_balance = float(
            np.clip(
                0.5
                + 0.5
                * (resolved_recovery_potential - resolved_stress_index),
                0.0,
                1.0,
            )
        )
    if resilience_score is not None:
        resolved_resilience_score = float(resilience_score)
    else:
        resolved_resilience_score = float(
            min(
                1.0,
                max(
                    0.0,
                    0.3 * resolved_recovery_potential
                    + 0.25 * resolved_stability_projection
                    + 0.2 * max(0.0, 1.0 - resolved_drawdown_pressure)
                    + 0.12 * max(0.0, 1.0 - resolved_liquidity_pressure)
                    + 0.08 * max(
                        0.0,
                        1.0
                        - min(
                            1.0,
                            max(0.0, -resolved_avg_volume_trend) / 0.35,
                        ),
                    )
                    + 0.05 * max(0.0, 1.0 - resolved_conf_decay)
                    + 0.05 * max(0.0, 1.0 - resolved_distribution_pressure),
                ),
            )
        )
    if liquidity_gap is not None:
        resolved_liquidity_gap = float(liquidity_gap)
    else:
        resolved_liquidity_gap = float(
            min(
                1.0,
                max(
                    0.0,
                    0.45 * resolved_liquidity_pressure
                    + 0.25 * min(max(0.0, -resolved_avg_volume_trend) / 0.35, 1.0)
                    + 0.2 * min(abs(resolved_volume_imbalance) / 0.6, 1.0)
                    + 0.1 * max(0.0, resolved_instability - 0.45)
                    + 0.1 * max(0.0, resolved_volatility_ratio - 1.0)
                    - 0.2 * resolved_recovery_potential,
                ),
            )
        )
    if confidence_resilience is not None:
        resolved_confidence_resilience = float(confidence_resilience)
    else:
        resolved_confidence_resilience = float(
            min(
                1.0,
                max(
                    0.0,
                    0.35 * confidence
                    + 0.2 * max(0.0, 1.0 - resolved_conf_decay)
                    + 0.15 * max(0.0, 1.0 - conf_vol_norm)
                    + 0.15 * max(0.0, resolved_conf_trend + 0.15)
                    + 0.15 * resolved_resilience_score
                    - 0.1 * resolved_distribution_pressure
                    - 0.1 * resolved_liquidity_gap,
                ),
            )
        )
    if stress_projection is not None:
        resolved_stress_projection = float(stress_projection)
    else:
        resolved_stress_projection = float(
            min(
                1.0,
                max(
                    0.0,
                    0.35 * resolved_stress_index
                    + 0.25 * resolved_degradation_score
                    + 0.15 * resolved_tail_risk_index
                    + 0.1 * resolved_shock_frequency
                    + 0.1 * resolved_distribution_pressure
                    + 0.1 * resolved_liquidity_gap
                    - 0.2 * resolved_recovery_potential
                    - 0.1 * resolved_resilience_score,
                ),
            )
        )
    if stress_momentum is not None:
        resolved_stress_momentum = float(stress_momentum)
    else:
        resolved_stress_momentum = float(
            min(
                1.0,
                max(
                    0.0,
                    0.4 * resolved_stress_index
                    + 0.3 * resolved_stress_projection
                    + 0.2 * resolved_tail_risk_index
                    + 0.15 * resolved_shock_frequency
                    + 0.1 * max(0.0, resolved_risk_trend)
                    - 0.2 * resolved_recovery_potential,
                ),
            )
        )
    if liquidity_trend is not None:
        resolved_liquidity_trend = float(liquidity_trend)
    else:
        resolved_liquidity_trend = float(
            min(
                1.0,
                max(
                    0.0,
                    0.5 * resolved_liquidity_pressure
                    + 0.3 * resolved_liquidity_gap
                    + 0.2 * volume_trend_component
                    + 0.1 * max(0.0, volume_trend_vol_norm - 0.4)
                    - 0.2 * resolved_recovery_potential,
                ),
            )
        )
    if confidence_fragility is not None:
        resolved_confidence_fragility = float(confidence_fragility)
    else:
        resolved_confidence_fragility = float(
            min(
                1.0,
                max(
                    0.0,
                    0.35 * conf_vol_norm
                    + 0.3 * resolved_conf_decay
                    + 0.2 * max(0.0, -resolved_conf_trend)
                    + 0.1 * resolved_regime_entropy
                    + 0.1 * resolved_distribution_pressure,
                ),
            )
        )
    return RegimeSummary(
        regime=regime,
        confidence=confidence,
        risk_score=risk,
        stability=resolved_stability,
        risk_trend=resolved_risk_trend,
        risk_level=resolved_level,
        risk_volatility=resolved_volatility,
        regime_persistence=resolved_persistence,
        transition_rate=resolved_transition_rate,
        confidence_trend=resolved_conf_trend,
        confidence_volatility=resolved_conf_vol,
        regime_streak=resolved_streak,
        instability_score=resolved_instability,
        confidence_decay=resolved_conf_decay,
        avg_drawdown=resolved_avg_drawdown,
        avg_volume_trend=resolved_avg_volume_trend,
        drawdown_pressure=resolved_drawdown_pressure,
        liquidity_pressure=resolved_liquidity_pressure,
        volatility_ratio=resolved_volatility_ratio,
        regime_entropy=resolved_regime_entropy,
        tail_risk_index=resolved_tail_risk_index,
        shock_frequency=resolved_shock_frequency,
        volatility_of_volatility=resolved_vol_of_vol,
        stress_index=resolved_stress_index,
        severe_event_rate=resolved_severe_event_rate,
        cooldown_score=resolved_cooldown_score,
        recovery_potential=resolved_recovery_potential,
        resilience_score=resolved_resilience_score,
        stress_balance=resolved_stress_balance,
        liquidity_gap=resolved_liquidity_gap,
        confidence_resilience=resolved_confidence_resilience,
        stress_projection=resolved_stress_projection,
        stress_momentum=resolved_stress_momentum,
        liquidity_trend=resolved_liquidity_trend,
        confidence_fragility=resolved_confidence_fragility,
        volatility_trend=resolved_volatility_trend,
        drawdown_trend=resolved_drawdown_trend,
        volume_trend_volatility=resolved_volume_trend_volatility,
        stability_projection=resolved_stability_projection,
        degradation_score=resolved_degradation_score,
        skewness_bias=resolved_skewness_bias,
        kurtosis_excess=resolved_kurtosis_excess,
        volume_imbalance=resolved_volume_imbalance,
        distribution_pressure=resolved_distribution_pressure,
        history=snapshots,
    )


def test_auto_trader_maps_trend_assessment_to_buy_signal() -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())
    ai_manager = _AIManagerStub([_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.35)])

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "BTCUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        market_data_provider=provider,
    )
    trader.ai_manager = ai_manager

    trader._auto_trade_loop()

    assert trader._last_signal == "buy"
    assert trader.current_strategy == "trend_following"
    assert trader.current_leverage > 1.0
    assert isinstance(trader._last_risk_decision, RiskDecision)
    assert trader._last_risk_decision.should_trade is True
    assert provider.calls == [("BTCUSDT", "1h", 256)]


def test_auto_trader_respects_high_risk_regime() -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())
    ai_manager = _AIManagerStub([_DummyAssessment(regime=MarketRegime.DAILY, risk_score=0.85)])

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "ETHUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        market_data_provider=provider,
    )
    trader.ai_manager = ai_manager

    trader._auto_trade_loop()

    assert trader._last_signal == "hold"
    assert trader.current_strategy == "capital_preservation"
    assert trader.current_leverage == 0.0
    assert isinstance(trader._last_risk_decision, RiskDecision)
    assert trader._last_risk_decision.should_trade is False
    assert provider.calls == [("ETHUSDT", "1h", 256)]


def test_auto_trader_uses_summary_to_lock_trading_on_high_risk() -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())
    summary = _build_summary(
        MarketRegime.DAILY,
        confidence=0.7,
        risk=0.82,
        severe_event_rate=0.75,
        cooldown_score=0.82,
        stress_projection=0.8,
        liquidity_gap=0.75,
        confidence_resilience=0.25,
    )
    ai_manager = _AIManagerStub(
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.3)],
        summaries={"ADAUSDT": summary},
    )

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "ADAUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        market_data_provider=provider,
    )
    trader.ai_manager = ai_manager

    trader._auto_trade_loop()

    assert trader._last_signal == "hold"
    assert trader.current_strategy == "capital_preservation"
    assert trader.current_leverage == 0.0
    assert trader._last_risk_decision is not None
    assert trader._last_risk_decision.should_trade is False
    assert trader._last_risk_decision.details["effective_risk"] >= 0.8
    assert trader._last_risk_decision.cooldown_active is True
    assert trader._last_risk_decision.state == "halted"
    assert (
        trader._last_risk_decision.details["cooldown_reason"]
        == trader._last_risk_decision.cooldown_reason
    )
    assert "summary" in trader._last_risk_decision.details
    assert trader._last_risk_decision.details["summary"]["stability"] == pytest.approx(summary.stability)
    assert trader._last_risk_decision.details["summary"]["risk_trend"] == pytest.approx(summary.risk_trend)
    assert (
        trader._last_risk_decision.details["summary"]["risk_level"]
        == summary.risk_level.value
    )
    assert trader._last_risk_decision.details["summary"]["risk_volatility"] == pytest.approx(summary.risk_volatility)
    assert trader._last_risk_decision.details["summary"]["regime_persistence"] == pytest.approx(summary.regime_persistence)
    assert trader._last_risk_decision.details["summary"]["stress_index"] == pytest.approx(summary.stress_index)
    assert trader._last_risk_decision.details["summary"]["tail_risk_index"] == pytest.approx(
        summary.tail_risk_index
    )
    assert trader._last_risk_decision.details["summary"]["shock_frequency"] == pytest.approx(
        summary.shock_frequency
    )
    assert trader._last_risk_decision.details["summary"]["severe_event_rate"] == pytest.approx(
        summary.severe_event_rate
    )
    assert trader._last_risk_decision.details["summary"]["cooldown_score"] == pytest.approx(
        summary.cooldown_score
    )
    assert trader._last_risk_decision.details["summary"]["recovery_potential"] == pytest.approx(
        summary.recovery_potential
    )
    assert trader._last_risk_decision.details["summary"]["confidence_trend"] == pytest.approx(
        summary.confidence_trend
    )
    assert trader._last_risk_decision.details["summary"]["confidence_volatility"] == pytest.approx(
        summary.confidence_volatility
    )
    assert (
        trader._last_risk_decision.details["summary"]["regime_streak"]
        == summary.regime_streak
    )
    assert trader._last_risk_decision.details["summary"]["transition_rate"] == pytest.approx(summary.transition_rate)
    assert trader._last_risk_decision.details["summary"]["instability_score"] == pytest.approx(
        summary.instability_score
    )
    assert trader._last_risk_decision.details["summary"]["confidence_decay"] == pytest.approx(
        summary.confidence_decay
    )
    assert trader._last_risk_decision.details["summary"]["stress_momentum"] == pytest.approx(
        summary.stress_momentum
    )
    assert trader._last_risk_decision.details["summary"]["liquidity_trend"] == pytest.approx(
        summary.liquidity_trend
    )
    assert trader._last_risk_decision.details["summary"]["confidence_fragility"] == pytest.approx(
        summary.confidence_fragility
    )
    assert trader._last_risk_decision.details["summary"]["volatility_trend"] == pytest.approx(
        summary.volatility_trend
    )
    assert trader._last_risk_decision.details["summary"]["drawdown_trend"] == pytest.approx(
        summary.drawdown_trend
    )
    assert trader._last_risk_decision.details["summary"]["volume_trend_volatility"] == pytest.approx(
        summary.volume_trend_volatility
    )
    assert trader._last_risk_decision.details["summary"]["stability_projection"] == pytest.approx(
        summary.stability_projection
    )
    assert trader._last_risk_decision.details["summary"]["distribution_pressure"] == pytest.approx(
        summary.distribution_pressure
    )
    assert summary.distribution_pressure <= 0.35
    assert trader._last_risk_decision.details["summary"]["degradation_score"] == pytest.approx(
        summary.degradation_score
    )
    assert trader._last_risk_decision.details["summary"]["stress_index"] == pytest.approx(summary.stress_index)
    assert trader._last_risk_decision.details["summary"]["tail_risk_index"] == pytest.approx(
        summary.tail_risk_index
    )
    assert trader._last_risk_decision.details["summary"]["shock_frequency"] == pytest.approx(
        summary.shock_frequency
    )
    assert trader._last_risk_decision.details["summary"]["stress_index"] == pytest.approx(summary.stress_index)
    assert trader._last_risk_decision.details["summary"]["tail_risk_index"] == pytest.approx(
        summary.tail_risk_index
    )
    assert trader._last_risk_decision.details["summary"]["shock_frequency"] == pytest.approx(
        summary.shock_frequency
    )
    assert trader._last_risk_decision.details["summary"]["avg_drawdown"] == pytest.approx(
        summary.avg_drawdown
    )
    assert trader._last_risk_decision.details["summary"]["avg_volume_trend"] == pytest.approx(
        summary.avg_volume_trend
    )
    assert trader._last_risk_decision.details["summary"]["drawdown_pressure"] == pytest.approx(
        summary.drawdown_pressure
    )
    assert trader._last_risk_decision.details["summary"]["liquidity_pressure"] == pytest.approx(
        summary.liquidity_pressure
    )
    assert trader._last_risk_decision.details["summary"]["volatility_ratio"] == pytest.approx(
        summary.volatility_ratio
    )
    assert trader._last_risk_decision.details["summary"]["tail_risk_index"] == pytest.approx(
        summary.tail_risk_index
    )
    assert trader._last_risk_decision.details["summary"]["shock_frequency"] == pytest.approx(
        summary.shock_frequency
    )
    assert trader._last_risk_decision.details["summary"]["volatility_of_volatility"] == pytest.approx(
        summary.volatility_of_volatility
    )
    assert trader._last_risk_decision.details["summary"]["stress_index"] == pytest.approx(
        summary.stress_index
    )
    assert trader._last_risk_decision.details["summary"]["confidence_trend"] == pytest.approx(
        summary.confidence_trend
    )
    assert trader._last_risk_decision.details["summary"]["confidence_volatility"] == pytest.approx(
        summary.confidence_volatility
    )
    assert (
        trader._last_risk_decision.details["summary"]["regime_streak"]
        == summary.regime_streak
    )
    assert trader._last_risk_decision.details["summary"]["transition_rate"] == pytest.approx(summary.transition_rate)


def test_auto_trader_throttles_on_liquidity_gap_and_confidence_drop() -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())
    summary = _build_summary(
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
    ai_manager = _AIManagerStub(
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.32)],
        summaries={"SOLUSDT": summary},
    )

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "SOLUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        market_data_provider=provider,
    )
    trader.ai_manager = ai_manager

    trader._auto_trade_loop()

    assert trader._last_signal == "hold"
    assert trader.current_leverage <= 0.45
    assert trader.current_strategy in {"capital_preservation", "trend_following_probing"}
    assert trader._last_risk_decision is not None
    assert trader._last_risk_decision.should_trade is False
    assert trader._last_risk_decision.details["summary"]["liquidity_gap"] == pytest.approx(summary.liquidity_gap)
    assert trader._last_risk_decision.details["summary"]["confidence_resilience"] == pytest.approx(
        summary.confidence_resilience
    )
    assert trader._last_risk_decision.details["effective_risk"] >= 0.6


def test_auto_trader_blocks_on_stress_momentum_and_fragility() -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())
    summary = _build_summary(
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
    ai_manager = _AIManagerStub(
        [_DummyAssessment(regime=MarketRegime.MEAN_REVERSION, risk_score=0.4)],
        summaries={"MOMUSDT": summary},
    )

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "MOMUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        market_data_provider=provider,
    )
    trader.ai_manager = ai_manager

    trader._auto_trade_loop()

    assert trader._last_signal == "hold"
    assert trader._last_risk_decision is not None
    assert trader._last_risk_decision.should_trade is False
    assert trader._last_risk_decision.details["effective_risk"] >= 0.75
    assert trader._last_risk_decision.cooldown_active is True
    assert trader._last_risk_decision.cooldown_reason in {"critical_risk", "elevated_risk"}
    assert trader.current_leverage <= 0.35
    summary_details = trader._last_risk_decision.details["summary"]
    assert summary_details["stress_momentum"] == pytest.approx(summary.stress_momentum)
    assert summary_details["liquidity_trend"] == pytest.approx(summary.liquidity_trend)
    assert summary_details["confidence_fragility"] == pytest.approx(summary.confidence_fragility)
    assert summary_details["confidence_resilience"] == pytest.approx(summary.confidence_resilience)
    assert summary_details["liquidity_gap"] == pytest.approx(summary.liquidity_gap)


def test_auto_trader_cooldown_engages_and_recovers() -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())
    severe_summary = _build_summary(
        MarketRegime.DAILY,
        confidence=0.6,
        risk=0.78,
        severe_event_rate=0.7,
        cooldown_score=0.8,
    )
    recovery_summary = _build_summary(
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
    ai_manager = _AIManagerStub(
        [
            _DummyAssessment(regime=MarketRegime.TREND, risk_score=0.32),
            _DummyAssessment(regime=MarketRegime.TREND, risk_score=0.34),
        ],
        summaries={"SOLUSDT": severe_summary},
    )

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "SOLUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        market_data_provider=provider,
    )
    trader.ai_manager = ai_manager

    trader._auto_trade_loop()

    assert trader._last_signal == "hold"
    assert trader._last_risk_decision.cooldown_active is True
    assert trader._last_risk_decision.state == "halted"
    assert trader._cooldown_reason in {"critical_risk", "elevated_risk", "instability_spike"}

    ai_manager._summaries["SOLUSDT"] = recovery_summary

    trader._auto_trade_loop()

    assert trader._last_signal in {"buy", "sell", "hold"}
    assert trader._last_risk_decision.cooldown_active is False
    assert trader._last_risk_decision.state != "halted"
    assert trader._cooldown_reason is None
    if trader._last_signal in {"buy", "sell"}:
        assert trader._last_risk_decision.should_trade is True
    decision_summary = trader._last_risk_decision.details.get("summary")
    if decision_summary is not None:
        assert decision_summary["cooldown_score"] <= 0.4
        assert decision_summary["recovery_potential"] >= 0.4


def test_auto_trader_holds_when_confidence_low_despite_trend() -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())
    summary = _build_summary(MarketRegime.TREND, confidence=0.3, risk=0.25)
    ai_manager = _AIManagerStub(
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.2, confidence=0.15)],
        summaries={"SOLUSDT": summary},
    )

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "SOLUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        market_data_provider=provider,
    )
    trader.ai_manager = ai_manager

    trader._auto_trade_loop()

    assert trader._last_signal == "hold"
    assert trader._last_risk_decision is not None
    assert trader._last_risk_decision.should_trade is False
    assert trader._last_risk_decision.details["confidence"] == 0.15
    assert trader._last_risk_decision.details["summary"]["confidence"] == 0.3
    assert trader._last_risk_decision.details["summary"]["stability"] == pytest.approx(summary.stability)
    assert trader._last_risk_decision.details["summary"]["risk_trend"] == pytest.approx(summary.risk_trend)
    assert (
        trader._last_risk_decision.details["summary"]["risk_level"]
        == summary.risk_level.value
    )
    assert trader._last_risk_decision.details["summary"]["risk_volatility"] == pytest.approx(summary.risk_volatility)
    assert trader._last_risk_decision.details["summary"]["regime_persistence"] == pytest.approx(summary.regime_persistence)
    assert trader._last_risk_decision.details["summary"]["transition_rate"] == pytest.approx(summary.transition_rate)
    assert trader._last_risk_decision.details["summary"]["confidence_trend"] == pytest.approx(
        summary.confidence_trend
    )
    assert trader._last_risk_decision.details["summary"]["confidence_volatility"] == pytest.approx(
        summary.confidence_volatility
    )
    assert (
        trader._last_risk_decision.details["summary"]["regime_streak"]
        == summary.regime_streak
    )
    assert trader._last_risk_decision.details["summary"]["instability_score"] == pytest.approx(
        summary.instability_score
    )
    assert trader._last_risk_decision.details["summary"]["confidence_decay"] == pytest.approx(
        summary.confidence_decay
    )
    assert trader._last_risk_decision.details["summary"]["tail_risk_index"] == pytest.approx(
        summary.tail_risk_index
    )
    assert trader._last_risk_decision.details["summary"]["shock_frequency"] == pytest.approx(
        summary.shock_frequency
    )
    assert trader._last_risk_decision.details["summary"]["volatility_of_volatility"] == pytest.approx(
        summary.volatility_of_volatility
    )
    assert trader._last_risk_decision.details["summary"]["stress_index"] == pytest.approx(
        summary.stress_index
    )


def test_auto_trader_waits_on_unstable_summary_even_in_trend() -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())
    summary = _build_summary(
        MarketRegime.TREND,
        confidence=0.7,
        risk=0.35,
        stability=0.3,
        risk_trend=0.0,
    )
    ai_manager = _AIManagerStub(
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.3, confidence=0.65)],
        summaries={"BNBUSDT": summary},
    )

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "BNBUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        market_data_provider=provider,
    )
    trader.ai_manager = ai_manager

    trader._auto_trade_loop()

    assert trader._last_signal == "hold"
    assert trader.current_leverage <= 0.5
    assert trader._last_risk_decision is not None
    assert trader._last_risk_decision.should_trade is False
    assert (
        trader._last_risk_decision.details["summary"]["risk_level"]
        == summary.risk_level.value
    )
    assert trader._last_risk_decision.details["summary"]["risk_volatility"] == pytest.approx(summary.risk_volatility)
    assert trader._last_risk_decision.details["summary"]["regime_persistence"] == pytest.approx(summary.regime_persistence)


def test_auto_trader_holds_when_risk_volatility_spikes() -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())
    summary = _build_summary(
        MarketRegime.TREND,
        confidence=0.65,
        risk=0.42,
        stability=0.7,
        risk_trend=0.04,
        risk_volatility=0.25,
        confidence_path=(0.65, 0.62, 0.6),
    )
    ai_manager = _AIManagerStub(
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.38, confidence=0.7)],
        summaries={"DOTUSDT": summary},
    )

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "DOTUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        market_data_provider=provider,
    )
    trader.ai_manager = ai_manager

    trader._auto_trade_loop()

    assert trader._last_signal == "hold"
    assert trader.current_leverage <= 0.65
    assert trader._last_risk_decision is not None
    assert trader._last_risk_decision.should_trade is False
    assert trader._last_risk_decision.details["effective_risk"] >= 0.6
    assert trader._last_risk_decision.details["summary"]["risk_volatility"] == pytest.approx(0.25)
    assert trader._last_risk_decision.details["summary"]["regime_persistence"] == pytest.approx(summary.regime_persistence)
    assert trader._last_risk_decision.details["summary"]["confidence_trend"] == pytest.approx(
        summary.confidence_trend
    )
    assert trader._last_risk_decision.details["summary"]["confidence_volatility"] == pytest.approx(
        summary.confidence_volatility
    )
    assert (
        trader._last_risk_decision.details["summary"]["regime_streak"]
        == summary.regime_streak
    )
    assert trader._last_risk_decision.details["summary"]["transition_rate"] == pytest.approx(summary.transition_rate)
    assert trader._last_risk_decision.details["summary"]["instability_score"] == pytest.approx(
        summary.instability_score
    )
    assert trader._last_risk_decision.details["summary"]["confidence_decay"] == pytest.approx(
        summary.confidence_decay
    )


def test_auto_trader_blocks_when_stress_metrics_spike() -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())
    summary = _build_summary(
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
    ai_manager = _AIManagerStub(
        [_DummyAssessment(regime=MarketRegime.MEAN_REVERSION, risk_score=0.38, confidence=0.62)],
        summaries={"AVAXUSDT": summary},
    )

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "AVAXUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        market_data_provider=provider,
    )
    trader.ai_manager = ai_manager

    trader._auto_trade_loop()

    assert trader._last_signal == "hold"
    assert trader.current_leverage <= 0.35
    assert trader._last_risk_decision is not None
    assert trader._last_risk_decision.should_trade is False
    assert trader._last_risk_decision.details["effective_risk"] >= 0.7
    assert trader._last_risk_decision.details["summary"]["stress_index"] == pytest.approx(0.78)
    assert trader._last_risk_decision.details["summary"]["tail_risk_index"] == pytest.approx(0.65)
    assert trader._last_risk_decision.details["summary"]["shock_frequency"] == pytest.approx(0.6)
    assert trader._last_risk_decision.details["summary"]["volatility_of_volatility"] == pytest.approx(
        summary.volatility_of_volatility
    )


def test_auto_trader_blocks_on_degradation_metrics() -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())
    summary = _build_summary(
        MarketRegime.TREND,
        confidence=0.58,
        risk=0.52,
        stability=0.4,
        risk_trend=0.1,
        risk_level=RiskLevel.ELEVATED,
        risk_volatility=0.22,
        regime_persistence=0.4,
        transition_rate=0.6,
        drawdown_path=(0.18, 0.24, 0.3),
        volatility_path=(0.02, 0.028, 0.035),
        volume_trend_path=(-0.1, -0.18, -0.22),
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
    ai_manager = _AIManagerStub(
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.42, confidence=0.58)],
        summaries={"DEGUSDT": summary},
    )

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "DEGUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        market_data_provider=provider,
    )
    trader.ai_manager = ai_manager

    trader._auto_trade_loop()

    assert trader._last_signal == "hold"
    assert trader.current_strategy == "capital_preservation"
    assert trader.current_leverage == 0.0
    assert trader._last_risk_decision is not None
    assert trader._last_risk_decision.should_trade is False
    assert trader._last_risk_decision.cooldown_active is True
    assert trader._last_risk_decision.details["summary"]["degradation_score"] == pytest.approx(
        summary.degradation_score
    )
    assert trader._last_risk_decision.details["summary"]["stability_projection"] == pytest.approx(
        summary.stability_projection
    )


def test_auto_trader_blocks_on_distribution_pressure() -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())
    summary = _build_summary(
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
    ai_manager = _AIManagerStub(
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.32)],
        summaries={"DOGEUSDT": summary},
    )

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "DOGEUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        market_data_provider=provider,
    )
    trader.ai_manager = ai_manager

    trader._auto_trade_loop()

    assert trader._last_signal == "hold"
    assert trader.current_leverage <= 0.35
    assert trader._last_risk_decision is not None
    assert trader._last_risk_decision.should_trade is False
    assert trader._last_risk_decision.cooldown_active is True
    summary_details = trader._last_risk_decision.details["summary"]
    assert summary_details["distribution_pressure"] == pytest.approx(summary.distribution_pressure)
    assert summary_details["skewness_bias"] == pytest.approx(summary.skewness_bias)
    assert summary_details["kurtosis_excess"] == pytest.approx(summary.kurtosis_excess)
    assert summary_details["volume_imbalance"] == pytest.approx(summary.volume_imbalance)
    assert summary_details["regime_entropy"] == pytest.approx(summary.regime_entropy)
    assert summary_details["resilience_score"] == pytest.approx(summary.resilience_score)
    assert summary_details["stress_balance"] == pytest.approx(summary.stress_balance)


def test_auto_trader_allows_trade_when_degradation_low() -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())
    summary = _build_summary(
        MarketRegime.TREND,
        confidence=0.65,
        risk=0.38,
        stability=0.6,
        risk_trend=0.02,
        risk_level=RiskLevel.BALANCED,
        risk_volatility=0.12,
        regime_persistence=0.55,
        transition_rate=0.35,
        drawdown_path=(0.08, 0.07, 0.06),
        volatility_path=(0.018, 0.017, 0.016),
        volume_trend_path=(0.04, 0.05, 0.06),
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
    ai_manager = _AIManagerStub(
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.35, confidence=0.68)],
        summaries={"POSUSDT": summary},
    )

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "POSUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        market_data_provider=provider,
    )
    trader.ai_manager = ai_manager

    trader._auto_trade_loop()

    assert trader._last_signal == "buy"
    assert trader.current_strategy.startswith("trend_following")
    assert trader.current_leverage > 0.0
    assert trader._last_risk_decision is not None
    assert trader._last_risk_decision.should_trade is True
    assert trader._last_risk_decision.cooldown_active is False
    assert trader._last_risk_decision.details["summary"]["degradation_score"] == pytest.approx(
        summary.degradation_score
    )
    assert trader._last_risk_decision.details["summary"]["stability_projection"] == pytest.approx(
        summary.stability_projection
    )


def test_auto_trader_blocks_on_high_instability() -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())
    summary = _build_summary(
        MarketRegime.TREND,
        confidence=0.62,
        risk=0.48,
        stability=0.45,
        risk_trend=0.12,
        regime_persistence=0.25,
        transition_rate=0.75,
        instability_score=0.82,
        confidence_decay=0.25,
        confidence_path=(0.72, 0.66, 0.62),
    )
    ai_manager = _AIManagerStub(
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.4, confidence=0.6)],
        summaries={"FTMUSDT": summary},
    )

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "FTMUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        market_data_provider=provider,
    )
    trader.ai_manager = ai_manager

    trader._auto_trade_loop()

    assert trader._last_signal == "hold"
    assert trader.current_strategy == "capital_preservation"
    assert trader.current_leverage == 0.0
    assert trader._last_risk_decision is not None
    assert trader._last_risk_decision.should_trade is False
    assert trader._last_risk_decision.details["effective_risk"] >= 0.75
    assert trader._last_risk_decision.details["summary"]["instability_score"] == pytest.approx(
        summary.instability_score
    )
    assert trader._last_risk_decision.details["summary"]["transition_rate"] == pytest.approx(
        summary.transition_rate
    )
    assert trader._last_risk_decision.details["summary"]["confidence_decay"] == pytest.approx(
        summary.confidence_decay
    )


def test_auto_trader_blocks_on_drawdown_pressure() -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())
    summary = _build_summary(
        MarketRegime.TREND,
        confidence=0.66,
        risk=0.46,
        stability=0.58,
        risk_trend=0.14,
        risk_level=RiskLevel.CRITICAL,
        drawdown_path=(0.28, 0.32, 0.35),
        avg_drawdown=0.32,
        drawdown_pressure=0.9,
        volume_trend_path=(-0.08, -0.1, -0.12),
        avg_volume_trend=-0.1,
        liquidity_pressure=0.58,
        volatility_ratio_path=(1.38, 1.45, 1.52),
        volatility_ratio=1.45,
    )
    ai_manager = _AIManagerStub(
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.42, confidence=0.64)],
        summaries={"LINKUSDT": summary},
    )

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "LINKUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        market_data_provider=provider,
    )
    trader.ai_manager = ai_manager

    trader._auto_trade_loop()

    assert trader._last_signal == "hold"
    assert trader.current_strategy == "capital_preservation"
    assert trader.current_leverage == 0.0
    assert trader._last_risk_decision is not None
    assert trader._last_risk_decision.should_trade is False
    assert trader._last_risk_decision.details["effective_risk"] >= 0.75
    assert trader._last_risk_decision.details["summary"]["drawdown_pressure"] == pytest.approx(
        summary.drawdown_pressure
    )
    assert trader._last_risk_decision.details["summary"]["avg_drawdown"] == pytest.approx(
        summary.avg_drawdown
    )


def test_auto_trader_suppresses_leverage_on_liquidity_pressure() -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())
    summary = _build_summary(
        MarketRegime.MEAN_REVERSION,
        confidence=0.6,
        risk=0.44,
        stability=0.62,
        risk_trend=0.06,
        risk_level=RiskLevel.ELEVATED,
        drawdown_path=(0.12, 0.14, 0.16),
        avg_drawdown=0.14,
        drawdown_pressure=0.56,
        volume_trend_path=(-0.25, -0.28, -0.3),
        avg_volume_trend=-0.28,
        liquidity_pressure=0.72,
        volatility_ratio_path=(1.25, 1.28, 1.32),
        volatility_ratio=1.28,
    )
    ai_manager = _AIManagerStub(
        [_DummyAssessment(regime=MarketRegime.MEAN_REVERSION, risk_score=0.36, confidence=0.62)],
        summaries={"SOLUSDT": summary},
    )

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "SOLUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        market_data_provider=provider,
    )
    trader.ai_manager = ai_manager

    trader._auto_trade_loop()

    assert trader._last_signal == "hold"
    assert trader.current_leverage <= 0.35
    assert trader.current_strategy in {"capital_preservation", "mean_reversion_probing"}
    assert trader._last_risk_decision is not None
    assert trader._last_risk_decision.should_trade is False
    assert trader._last_risk_decision.details["effective_risk"] >= 0.64
    assert trader._last_risk_decision.details["summary"]["liquidity_pressure"] == pytest.approx(
        summary.liquidity_pressure
    )
    assert trader._last_risk_decision.details["summary"]["avg_volume_trend"] == pytest.approx(
        summary.avg_volume_trend
    )


def test_auto_trader_holds_when_confidence_trend_collapses() -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())
    summary = _build_summary(
        MarketRegime.TREND,
        confidence=0.28,
        risk=0.38,
        stability=0.68,
        risk_trend=0.02,
        confidence_path=(0.72, 0.5, 0.28),
    )
    ai_manager = _AIManagerStub(
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.34, confidence=0.74)],
        summaries={"FTMUSDT": summary},
    )

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "FTMUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        market_data_provider=provider,
    )
    trader.ai_manager = ai_manager

    trader._auto_trade_loop()

    assert trader._last_signal == "hold"
    assert trader.current_leverage <= 0.6
    assert trader.current_strategy.endswith("probing") or trader.current_strategy == "capital_preservation"
    assert trader._last_risk_decision is not None
    assert trader._last_risk_decision.should_trade is False
    assert trader._last_risk_decision.details["summary"]["confidence_trend"] == pytest.approx(summary.confidence_trend)
    assert trader._last_risk_decision.details["summary"]["confidence_volatility"] == pytest.approx(
        summary.confidence_volatility
    )
    assert (
        trader._last_risk_decision.details["summary"]["regime_streak"]
        == summary.regime_streak
    )
    assert trader._last_risk_decision.details["summary"]["transition_rate"] == pytest.approx(summary.transition_rate)
    assert trader._last_risk_decision.details["summary"]["instability_score"] == pytest.approx(
        summary.instability_score
    )
    assert trader._last_risk_decision.details["summary"]["confidence_decay"] == pytest.approx(
        summary.confidence_decay
    )


def test_auto_trader_holds_when_regime_streak_short() -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())
    summary = _build_summary(
        MarketRegime.TREND,
        confidence=0.68,
        risk=0.36,
        stability=0.52,
        risk_trend=0.03,
        regime_sequence=(
            MarketRegime.MEAN_REVERSION,
            MarketRegime.DAILY,
            MarketRegime.TREND,
        ),
    )
    ai_manager = _AIManagerStub(
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.32, confidence=0.7)],
        summaries={"NEARUSDT": summary},
    )

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "NEARUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        market_data_provider=provider,
    )
    trader.ai_manager = ai_manager

    trader._auto_trade_loop()

    assert trader._last_signal == "hold"
    assert trader.current_leverage <= 0.4
    assert trader.current_strategy.endswith("probing") or trader.current_strategy == "capital_preservation"
    assert trader._last_risk_decision is not None
    assert trader._last_risk_decision.should_trade is False
    assert trader._last_risk_decision.details["summary"]["regime_streak"] == summary.regime_streak
    assert trader._last_risk_decision.details["summary"]["confidence_trend"] == pytest.approx(summary.confidence_trend)
    assert trader._last_risk_decision.details["summary"]["confidence_volatility"] == pytest.approx(
        summary.confidence_volatility
    )


def test_auto_trader_trims_when_persistence_collapses() -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())
    summary = _build_summary(
        MarketRegime.MEAN_REVERSION,
        confidence=0.7,
        risk=0.48,
        stability=0.5,
        risk_trend=0.03,
        regime_persistence=0.2,
    )
    ai_manager = _AIManagerStub(
        [_DummyAssessment(regime=MarketRegime.MEAN_REVERSION, risk_score=0.44, confidence=0.72)],
        summaries={"AVAXUSDT": summary},
    )

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "AVAXUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        market_data_provider=provider,
    )
    trader.ai_manager = ai_manager

    trader._auto_trade_loop()

    assert trader._last_signal == "hold"
    assert trader.current_leverage <= 0.4
    assert trader.current_strategy.endswith("probing") or trader.current_strategy == "capital_preservation"
    assert trader._last_risk_decision is not None
    assert trader._last_risk_decision.should_trade is False
    assert trader._last_risk_decision.details["effective_risk"] >= 0.7
    assert trader._last_risk_decision.details["summary"]["regime_persistence"] == pytest.approx(0.2)
    assert trader._last_risk_decision.details["summary"]["risk_volatility"] == pytest.approx(summary.risk_volatility)
    assert trader._last_risk_decision.details["summary"]["confidence_trend"] == pytest.approx(
        summary.confidence_trend
    )
    assert trader._last_risk_decision.details["summary"]["confidence_volatility"] == pytest.approx(
        summary.confidence_volatility
    )
    assert (
        trader._last_risk_decision.details["summary"]["regime_streak"]
        == summary.regime_streak
    )
    assert trader._last_risk_decision.details["summary"]["transition_rate"] == pytest.approx(summary.transition_rate)
    assert trader._last_risk_decision.details["summary"]["instability_score"] == pytest.approx(
        summary.instability_score
    )
    assert trader._last_risk_decision.details["summary"]["confidence_decay"] == pytest.approx(
        summary.confidence_decay
    )


def test_auto_trader_scales_down_when_risk_trend_rising() -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())
    summary = _build_summary(
        MarketRegime.TREND,
        confidence=0.8,
        risk=0.32,
        stability=0.9,
        risk_trend=0.2,
    )
    ai_manager = _AIManagerStub(
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.28, confidence=0.8)],
        summaries={"XRPUSDT": summary},
    )

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "XRPUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        market_data_provider=provider,
    )
    trader.ai_manager = ai_manager

    trader._auto_trade_loop()

    assert trader._last_signal == "hold"
    assert trader.current_leverage <= 1.7
    assert trader._last_risk_decision is not None
    assert trader._last_risk_decision.details["summary"]["risk_trend"] == pytest.approx(0.2)
    assert (
        trader._last_risk_decision.details["summary"]["risk_level"]
        == summary.risk_level.value
    )
    assert trader._last_risk_decision.details["summary"]["risk_volatility"] == pytest.approx(summary.risk_volatility)
    assert trader._last_risk_decision.details["summary"]["regime_persistence"] == pytest.approx(summary.regime_persistence)
    assert trader._last_risk_decision.details["summary"]["confidence_trend"] == pytest.approx(
        summary.confidence_trend
    )
    assert trader._last_risk_decision.details["summary"]["confidence_volatility"] == pytest.approx(
        summary.confidence_volatility
    )
    assert (
        trader._last_risk_decision.details["summary"]["regime_streak"]
        == summary.regime_streak
    )
    assert trader._last_risk_decision.details["summary"]["transition_rate"] == pytest.approx(summary.transition_rate)
    assert trader._last_risk_decision.details["summary"]["instability_score"] == pytest.approx(
        summary.instability_score
    )
    assert trader._last_risk_decision.details["summary"]["confidence_decay"] == pytest.approx(
        summary.confidence_decay
    )


def test_auto_trader_increases_risk_when_summary_calm() -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())
    summary = _build_summary(
        MarketRegime.TREND,
        confidence=0.78,
        risk=0.18,
        stability=0.8,
        risk_trend=-0.05,
        risk_level=RiskLevel.CALM,
        confidence_path=(0.65, 0.72, 0.78),
    )
    ai_manager = _AIManagerStub(
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.22, confidence=0.78)],
        summaries={"LTCUSDT": summary},
    )

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "LTCUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        market_data_provider=provider,
    )
    trader.ai_manager = ai_manager

    trader._auto_trade_loop()

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
