from __future__ import annotations

import copy
import enum
import json
import time
from collections.abc import Iterable
from dataclasses import MISSING as _MISSING, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
from bot_core.auto_trader.app import (
    AutoTrader,
    GuardrailTimelineRecords,
    GuardrailTrigger,
    RiskDecision,
)
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


def test_auto_trader_strategy_alias_overrides_extend_candidates() -> None:
    emitter = _Emitter()
    gui = _GUI()

    class _Catalog:
        def metadata_for(self, name: str):  # noqa: D401 - test double
            return {}

    trader = AutoTrader(
        emitter,
        gui,
        lambda: "BTCUSDT",
        strategy_catalog=_Catalog(),
        strategy_alias_map={"Legacy Breakout": "day_trading"},
        strategy_alias_suffixes=("_legacy",),
    )

    candidates = trader._strategy_metadata_candidates("Legacy Breakout_Legacy")
    assert "day_trading" in candidates
    resolver = trader._alias_resolver_instance()
    assert "_probing" in resolver.suffixes
    assert "_legacy" in resolver.suffixes


def test_auto_trader_strategy_alias_overrides_accept_canonical_collections() -> None:
    emitter = _Emitter()
    gui = _GUI()

    class _Catalog:
        def metadata_for(self, name: str):  # noqa: D401 - test double
            return {}

    trader = AutoTrader(
        emitter,
        gui,
        lambda: "BTCUSDT",
        strategy_catalog=_Catalog(),
        strategy_alias_map={
            "day_trading": ["Legacy Breakout", {"more": ["LegacyLegacy"]}]
        },
    )

    candidates = trader._strategy_metadata_candidates("LegacyLegacy")
    assert "day_trading" in candidates


def test_auto_trader_configure_aliases_updates_candidates() -> None:
    emitter = _Emitter()
    gui = _GUI()

    class _Catalog:
        def metadata_for(self, name: str):  # noqa: D401 - test double
            return {}

    trader = AutoTrader(
        emitter,
        gui,
        lambda: "BTCUSDT",
        strategy_catalog=_Catalog(),
    )

    baseline = trader._strategy_metadata_candidates("Legacy Breakout")
    assert "day_trading" not in baseline

    trader.configure_strategy_aliases(
        {"Legacy Breakout": "day_trading"}, suffixes=("_legacy",)
    )

    candidates = trader._strategy_metadata_candidates("Legacy Breakout_Legacy")
    assert "day_trading" in candidates
    suffixes = trader._alias_resolver_instance().suffixes
    assert "_legacy" in suffixes


class _RiskServiceStub:
    ...
def _prepare_guardrail_history(monkeypatch: pytest.MonkeyPatch) -> AutoTrader:
    emitter = _Emitter()
    gui = _GUI()
    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "SOLUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=False,
    )

    class _ServiceAlpha:
        ...

    class _ServiceBeta:
        ...

    alpha = _ServiceAlpha()
    beta = _ServiceBeta()

    def _decision(
        reasons: Iterable[str],
        triggers: Iterable[GuardrailTrigger],
    ) -> RiskDecision:
        return RiskDecision(
            should_trade=False,
            fraction=0.0,
            state="blocked",
            reason="guardrail-blocked",
            mode="auto",
            details={
                "origin": "guardrail-test",
                "guardrail_reasons": list(reasons),
                "guardrail_triggers": [trigger.to_dict() for trigger in triggers],
            },
        )

    timestamps = iter([2000.0, 2010.0, 2020.0, 2030.0])
    monkeypatch.setattr("bot_core.auto_trader.app.time.time", lambda: next(timestamps))

    with trader._decision_audit_scope(decision_id="guardrail-alpha-1"):
        trader._record_risk_evaluation(
            _decision(
                ["effective risk cap", "volatility spike"],
                [
                    GuardrailTrigger(
                        name="effective_risk",
                        label="Effective risk cap",
                        comparator=">=",
                        threshold=0.8,
                        unit="ratio",
                        value=0.83,
                    ),
                    GuardrailTrigger(
                        name="volatility_ratio",
                        label="Volatility ratio",
                        comparator=">=",
                        threshold=1.2,
                        unit="ratio",
                        value=1.25,
                    ),
                ],
            ),
            approved=False,
            normalized=False,
            response=False,
            service=alpha,
            error=None,
        )

    with trader._decision_audit_scope(decision_id="guardrail-alpha-2"):
        trader._record_risk_evaluation(
            _decision(
                ["effective risk cap"],
                [
                    GuardrailTrigger(
                        name="effective_risk",
                        label="Effective risk cap",
                        comparator=">=",
                        threshold=0.8,
                        unit="ratio",
                        value=0.81,
                    ),
                ],
            ),
            approved=False,
            normalized=False,
            response=False,
            service=alpha,
            error=None,
        )

    with trader._decision_audit_scope(decision_id="guardrail-beta-ok"):
        trader._record_risk_evaluation(
            RiskDecision(
                should_trade=True,
                fraction=0.5,
                state="active",
                details={"origin": "guardrail-test"},
            ),
            approved=True,
            normalized=True,
            response=True,
            service=beta,
            error=None,
        )

    with trader._decision_audit_scope(decision_id="guardrail-unknown-1"):
        trader._record_risk_evaluation(
            _decision(
                ["effective risk cap", "liquidity pressure"],
                [
                    GuardrailTrigger(
                        name="effective_risk",
                        label="Effective risk cap",
                        comparator=">=",
                        threshold=0.8,
                        unit="score",
                        value=0.79,
                    ),
                ],
            ),
            approved=False,
            normalized=False,
            response=False,
            service=None,
            error=RuntimeError("guardrail failure"),
        )

    return trader


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
class _RiskServiceStub:
    def __init__(self, approval: bool) -> None:
        self._approval = approval
        self.calls: list[RiskDecision] = []

    def __call__(self, decision: RiskDecision) -> bool:
        return self.evaluate_decision(decision)
    def __init__(self, approval: Any) -> None:
        self._approval = approval
        self.calls: list[RiskDecision] = []

    def evaluate_decision(self, decision: RiskDecision) -> Any:
        self.calls.append(decision)
        result = self._approval
        if callable(result):
            result = result()
        return result

    def __call__(self, decision: RiskDecision) -> Any:  # pragma: no cover - compatibility shim
        return self.evaluate_decision(decision)


class _RiskServiceResponseStub:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[RiskDecision] = []

    def evaluate_decision(self, decision: RiskDecision) -> Any:
        self.calls.append(decision)
        result = self._response
        if callable(result):
            result = result()
        return result
    def _resolve(self) -> Any:
        return self._response() if callable(self._response) else self._response

    def evaluate_decision(self, decision: RiskDecision) -> Any:
        self.calls.append(decision)
        return self._resolve()

    def __call__(self, decision: RiskDecision) -> Any:  # pragma: no cover - compatibility shim
        return self.evaluate_decision(decision)

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
        self.methods.append("execute_decision")
        self.calls.append(decision)

    def execute(self, decision: RiskDecision) -> None:
        self.methods.append("execute")
        self.calls.append(decision)


class _ExecutionServiceExecuteOnly:
    def __init__(self) -> None:
        self.calls: list[RiskDecision] = []
        self.methods: list[str] = []

    def execute(self, decision: RiskDecision) -> None:
        self.calls.append(decision)
        self.methods.append("execute")
        self.methods.append("execute")
        self.calls.append(decision)


_MISSING = object()


class _Approval(enum.Enum):
    APPROVED = "approved"
    DENIED = "denied"


def _build_summary(
    regime: MarketRegime,
    *,
    dataset: str | None = None,
    step: int = 24,
    **overrides: Any,
) -> RegimeSummary:
    rename_map = {"risk": "risk_score"}
    normalized: dict[str, Any] = {}
    for key, value in overrides.items():
        mapped = rename_map.get(key, key)
        normalized[mapped] = value
    overrides_map = normalized or None
    return load_summary_for_regime(
        regime,
        dataset=dataset,
        step=step,
        overrides=overrides_map,
    )


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


def test_load_risk_thresholds_signal_thresholds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_loader.reset_threshold_cache()
    override = tmp_path / "risk_thresholds.yaml"
    override.write_text(
        """
auto_trader:
  signal_thresholds:
    signal_after_adjustment: 0.42
    signal_after_clamp: 0.58
  strategy_signal_thresholds:
    BINANCE:
      Trend_Following:
        signal_after_clamp: 0.66
""".strip()
    )

    monkeypatch.setenv("BOT_CORE_RISK_THRESHOLDS_PATH", str(override))

    try:
        thresholds = config_loader.load_risk_thresholds()
        auto_cfg = thresholds["auto_trader"]
        signal_cfg = auto_cfg["signal_thresholds"]
        assert signal_cfg["signal_after_clamp"] == pytest.approx(0.58)
        assert signal_cfg["signal_after_adjustment"] == pytest.approx(0.42)
        strategy_cfg = auto_cfg["strategy_signal_thresholds"]
        binance_cfg = strategy_cfg["binance"]
        trend_cfg = binance_cfg["trend_following"]
        assert trend_cfg["signal_after_clamp"] == pytest.approx(0.66)
    finally:
        monkeypatch.delenv("BOT_CORE_RISK_THRESHOLDS_PATH", raising=False)
        config_loader.reset_threshold_cache()


def test_load_risk_thresholds_rejects_invalid_signal_threshold(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_loader.reset_threshold_cache()
    override = tmp_path / "risk_thresholds.yaml"
    override.write_text(
        """
auto_trader:
  signal_thresholds:
    signal_after_clamp: invalid
""".strip()
    )

    monkeypatch.setenv("BOT_CORE_RISK_THRESHOLDS_PATH", str(override))

    try:
        with pytest.raises(ValueError):
            config_loader.load_risk_thresholds()
    finally:
        monkeypatch.delenv("BOT_CORE_RISK_THRESHOLDS_PATH", raising=False)
        config_loader.reset_threshold_cache()

@dataclass
class _DummyAssessment:
    regime: MarketRegime
    risk_score: float
    confidence: float = 0.8
    ai_prediction: float = 0.015

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
        self.ai_threshold_bps = 5.0
        self.is_degraded = False
        self._last_prediction = assessments[0].ai_prediction if assessments else 0.0

    def assess_market_regime(self, symbol: str, market_data: pd.DataFrame, **_: Any) -> MarketRegimeAssessment:
        self.calls.append(symbol)
        next_assessment = self._queue.pop(0)
        self._last_prediction = next_assessment.ai_prediction
        return next_assessment.to_assessment(symbol)

    def get_regime_summary(self, symbol: str) -> RegimeSummary | None:
        return self._summaries.get(symbol)

    async def predict_series(
        self,
        symbol: str,
        df: pd.DataFrame,
        *,
        model_types: Any | None = None,
        feature_cols: Any | None = None,
    ) -> pd.Series:
        del symbol, model_types, feature_cols
        if not isinstance(df, pd.DataFrame) or df.empty:
            index = pd.RangeIndex(start=0, stop=1)
        else:
            index = df.index[-1:]
        return pd.Series([float(self._last_prediction)], index=index)

    def require_real_models(self) -> None:
        if self.is_degraded:
            raise RuntimeError("fallback backend")


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
        yield from (
            self.decision,
            self.trader,
            self.emitter,
            self.provider,
            self.ai_manager,
        )


def test_auto_trader_trusted_mode_auto_confirms(monkeypatch: pytest.MonkeyPatch) -> None:
    emitter = _Emitter()
    gui = _GUI()

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "BTCUSDT",
        enable_auto_trade=True,
        auto_trade_interval_s=0.0,
        trusted_auto_confirm=True,
    )

    started: list[bool] = []

    def _fake_start() -> None:
        started.append(True)

    monkeypatch.setattr(trader, "_start_auto_trade_thread_locked", _fake_start)
    trader.start()

    assert trader._auto_trade_user_confirmed is True
    assert started, "Trusted mode did not start the auto-trade loop"

    trader.stop()


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


def test_auto_trader_holds_when_ai_signal_below_threshold() -> None:
    result = _execute_auto_trade(
        "SOLUSDT",
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.32, ai_prediction=0.0)],
    )
    decision, trader, _, provider, _ = result

    assert decision.should_trade is False
    assert trader._last_signal == "hold"
    ai_details = decision.details.get("decision_engine", {}).get("ai")
    assert ai_details is not None
    assert ai_details.get("direction") == "hold"
    assert ai_details.get("prediction_bps") == pytest.approx(0.0)
    result.assert_provider_called("SOLUSDT")


def test_auto_trader_ai_conflict_forces_hold() -> None:
    result = _execute_auto_trade(
        "XRPUSDT",
        [_DummyAssessment(regime=MarketRegime.TREND, risk_score=0.3, ai_prediction=-0.02)],
    )
    decision, trader, _, provider, _ = result

    assert decision.should_trade is False
    assert trader._last_signal == "hold"
    ai_details = decision.details.get("decision_engine", {}).get("ai")
    assert ai_details is not None
    assert ai_details.get("direction") == "sell"
    assert ai_details.get("prediction_bps") == pytest.approx(-200.0)
    result.assert_provider_called("XRPUSDT")


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
    assert isinstance(first.get("decision_id"), str)
    assert first["decision_id"]
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

    filtered_by_id = trader.get_risk_evaluations(decision_id=first["decision_id"])
    assert [entry["decision_id"] for entry in filtered_by_id] == [first["decision_id"]]

    evaluations[0]["approved"] = None
    fresh = trader.get_risk_evaluations()
    assert fresh[0]["approved"] is True


def test_auto_trader_risk_evaluation_caches_guardrail_dimensions() -> None:
    emitter = _Emitter()
    gui = _GUI()
    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "BTCUSDT",
        enable_auto_trade=False,
    )

    decision = RiskDecision(
        should_trade=False,
        fraction=0.0,
        state="blocked",
        reason="guardrail",
        details={
            "guardrail_reasons": ["volatility spike"],
            "guardrail_triggers": [
                {
                    "name": "volatility_guard",
                    "label": "Volatility guard",
                    "comparator": ">=",
                    "threshold": "0.8",
                    "unit": "ratio",
                    "value": "0.95",
                }
            ],
        },
    )

    trader._record_risk_evaluation(
        decision,
        approved=False,
        normalized=False,
        response=None,
        service=None,
        error=None,
    )

    evaluations = trader.get_risk_evaluations()
    assert len(evaluations) == 1
    guardrail_dimensions = evaluations[0].get("guardrail_dimensions")
    assert guardrail_dimensions
    assert guardrail_dimensions["reasons"] == ("volatility spike",)
    triggers = guardrail_dimensions["triggers"]
    assert isinstance(triggers, tuple)
    assert triggers and triggers[0]["name"] == "volatility_guard"
    assert triggers[0]["threshold"] == "0.8"
    tokens = guardrail_dimensions["tokens"]
    assert isinstance(tokens, tuple)
    assert tokens and tokens[0]["name"] == "volatility_guard"
    assert tokens[0]["threshold"] == pytest.approx(0.8)
    assert tokens[0]["value"] == pytest.approx(0.95)


def test_auto_trader_traces_risk_evaluations_by_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitter = _Emitter()
    gui = _GUI()
    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "BTCUSDT",
        enable_auto_trade=False,
    )

    timestamps = iter([1000.0, 1002.5, 1005.5, 1006.0])

    def _fake_time() -> float:
        try:
            return next(timestamps)
        except StopIteration:  # pragma: no cover - defensive guard for flaky tests
            return 1006.0

    monkeypatch.setattr("bot_core.auto_trader.app.time.time", _fake_time)

    decision = RiskDecision(should_trade=True, fraction=0.3, state="active")

    class _Service:
        ...

    service = _Service()

    with trader._decision_audit_scope(decision_id="cycle-trace"):
        trader._record_risk_evaluation(
            decision,
            approved=True,
            normalized=True,
            response={"status": "ok"},
            service=service,
            error=None,
        )
        trader._record_risk_evaluation(
            RiskDecision(should_trade=False, fraction=0.0, state="blocked", reason="guardrail"),
            approved=False,
            normalized=False,
            response=None,
            service=None,
            error=RuntimeError("guardrail-blocked"),
        )

    timeline = trader.get_risk_evaluation_trace("cycle-trace")

    assert len(timeline) == 2
    assert timeline[0]["decision_id"] == "cycle-trace"
    assert isinstance(timeline[0]["timestamp"], datetime)
    assert timeline[1]["elapsed_since_first_s"] == pytest.approx(2.5)
    assert timeline[1]["elapsed_since_previous_s"] == pytest.approx(2.5)

    stripped = trader.get_risk_evaluation_trace(
        "cycle-trace",
        include_decision=False,
        include_service=False,
        include_response=False,
        include_error=False,
    )

    assert "decision" not in stripped[0]
    assert "service" not in stripped[0]
    assert "response" not in stripped[0]
    assert "error" not in stripped[1]

    no_errors = trader.get_risk_evaluation_trace("cycle-trace", include_errors=False)
    assert len(no_errors) == 1
    assert no_errors[0]["approved"] is True

    assert trader.get_risk_evaluation_trace("missing-id") == ()
    assert trader.get_risk_evaluation_trace(None) == ()


def test_auto_trader_groups_risk_evaluations_by_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitter = _Emitter()
    gui = _GUI()
    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "ETHUSDT",
        enable_auto_trade=False,
    )

    timestamps = iter([2000.0, 2001.0, 2002.0, 2003.0, 2004.0])

    def _fake_time_group() -> float:
        try:
            return next(timestamps)
        except StopIteration:  # pragma: no cover - defensive guard for flaky tests
            return 2004.0

    monkeypatch.setattr("bot_core.auto_trader.app.time.time", _fake_time_group)

    base_decision = RiskDecision(should_trade=True, fraction=0.2, state="active")

    with trader._decision_audit_scope(decision_id="cycle-alpha"):
        trader._record_risk_evaluation(
            base_decision,
            approved=True,
            normalized=True,
            response=True,
            service=_RiskServiceStub(True),
            error=None,
        )
        trader._record_risk_evaluation(
            RiskDecision(should_trade=False, fraction=0.0, state="blocked"),
            approved=False,
            normalized=False,
            response=False,
            service=_RiskServiceStub(False),
            error=None,
        )

    with trader._decision_audit_scope(decision_id="cycle-beta"):
        trader._record_risk_evaluation(
            base_decision,
            approved=True,
            normalized=True,
            response=True,
            service=_RiskServiceStub(True),
            error=None,
        )

    trader._store_risk_evaluation_entry(
        {
            "timestamp": 2004.0,
            "approved": None,
            "normalized": None,
            "decision": base_decision.to_dict(),
        },
        reference_time=2004.0,
    )

    grouped = trader.get_grouped_risk_evaluations(include_unidentified=True)

    assert list(grouped.keys()) == ["cycle-alpha", "cycle-beta", None]
    assert len(grouped["cycle-alpha"]) == 2
    assert grouped["cycle-beta"][0]["decision_id"] == "cycle-beta"
    assert grouped[None][0]["decision_id"] is None
    assert grouped["cycle-alpha"][0]["timestamp"].tzinfo is timezone.utc

    filtered = trader.get_grouped_risk_evaluations(include_unidentified=False)
    assert list(filtered.keys()) == ["cycle-alpha", "cycle-beta"]

    beta_only = trader.get_grouped_risk_evaluations(decision_id="cycle-beta")
    assert list(beta_only.keys()) == ["cycle-beta"]
    assert len(beta_only["cycle-beta"]) == 1

    raw_timestamps = trader.get_grouped_risk_evaluations(
        decision_id="cycle-alpha",
        coerce_timestamps=False,
    )
    assert isinstance(raw_timestamps["cycle-alpha"][0]["timestamp"], float)


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

    sample_decision_id = evaluations[0]["decision_id"]
    summary_by_id = trader.summarize_risk_evaluations(decision_id=sample_decision_id)
    assert summary_by_id["total"] == 1


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

    evaluations = trader.get_risk_evaluations()
    df = trader.risk_evaluations_to_dataframe()
    assert len(df) == 4
    assert {"timestamp", "approved", "normalized", "decision", "decision_id"}.issubset(
        df.columns
    )
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

    id_df = trader.risk_evaluations_to_dataframe(decision_id=evaluations[0]["decision_id"])
    assert id_df["decision_id"].unique().tolist() == [evaluations[0]["decision_id"]]

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
    assert subset_flattened_columns == [
        "decision_id",
        "decision_fraction",
        "decision_details",
    ]
    assert subset_df.loc[0, "decision_fraction"] == pytest.approx(0.42)
    assert subset_df.loc[1, "decision_fraction"] == pytest.approx(0.13)
    assert subset_df.loc[0, "decision_details"]["origin"] == "unit-test"

    drop_df = trader.risk_evaluations_to_dataframe(
        flatten_decision=True,
        drop_decision_column=True,
    )
    assert "decision" not in drop_df.columns
    assert {"decision_should_trade", "decision_fraction"}.issubset(drop_df.columns)
    assert "decision_id" in drop_df.columns

    fill_df = trader.risk_evaluations_to_dataframe(
        flatten_decision=True,
        decision_fields=["missing_field"],
        fill_value="missing",
    )
    assert (fill_df["decision_missing_field"] == "missing").all()

    coerced_df = trader.risk_evaluations_to_dataframe(coerce_timestamps=True)
    assert isinstance(coerced_df.loc[0, "timestamp"], datetime)
    assert coerced_df.loc[0, "timestamp"].tzinfo is timezone.utc

    naive_df = trader.risk_evaluations_to_dataframe(
        coerce_timestamps=True,
        tz=None,
    )
    assert naive_df.loc[0, "timestamp"].tzinfo is None

    records_snapshot = trader.risk_evaluations_to_records()
    assert len(records_snapshot) == 4
    assert {
        "timestamp",
        "approved",
        "normalized",
        "decision_id",
        "decision",
        "response",
        "error",
    }.issubset(
        records_snapshot[0].keys()
    )
    assert records_snapshot[0]["decision"]["details"]["origin"] == "unit-test"

    alpha_records = trader.risk_evaluations_to_records(service="_ServiceAlpha")
    assert len(alpha_records) == 2
    assert all(entry["service"] == "_ServiceAlpha" for entry in alpha_records)

    window_records = trader.risk_evaluations_to_records(
        since=pd.Timestamp(1010.0, unit="s"),
        until=pd.Timestamp(1020.0, unit="s"),
    )
    assert len(window_records) == 2
    assert {entry.get("service") or "<unknown>" for entry in window_records} == {
        "_ServiceBeta",
        "<unknown>",
    }

    id_records = trader.risk_evaluations_to_records(decision_id=evaluations[0]["decision_id"])
    assert [entry["decision_id"] for entry in id_records] == [evaluations[0]["decision_id"]]

    flattened_records = trader.risk_evaluations_to_records(flatten_decision=True)
    assert {"decision_should_trade", "decision_fraction", "decision_state"}.issubset(
        flattened_records[0].keys()
    )
    assert flattened_records[0]["decision_should_trade"] is True
    assert flattened_records[1]["decision_should_trade"] is False

    prefixed_records = trader.risk_evaluations_to_records(
        flatten_decision=True,
        decision_prefix="risk__",
    )
    assert "risk__should_trade" in prefixed_records[0]
    assert "decision_should_trade" not in prefixed_records[0]

    subset_records = trader.risk_evaluations_to_records(
        flatten_decision=True,
        decision_fields=["fraction", "details"],
    )
    subset_keys = [
        key for key in subset_records[0].keys() if key.startswith("decision_")
    ]
    assert subset_keys == ["decision_id", "decision_fraction", "decision_details"]

    dropped_records = trader.risk_evaluations_to_records(
        flatten_decision=True,
        drop_decision_column=True,
    )
    assert all("decision" not in entry for entry in dropped_records)
    assert "decision_should_trade" in dropped_records[0]

    filled_records = trader.risk_evaluations_to_records(
        flatten_decision=True,
        decision_fields=["missing_field"],
        fill_value="missing",
    )
    assert all(entry["decision_missing_field"] == "missing" for entry in filled_records)

    coerced_records = trader.risk_evaluations_to_records(coerce_timestamps=True)
    assert isinstance(coerced_records[0]["timestamp"], datetime)
    assert coerced_records[0]["timestamp"].tzinfo is timezone.utc

    naive_records = trader.risk_evaluations_to_records(
        coerce_timestamps=True,
        tz=None,
    )
    assert naive_records[0]["timestamp"].tzinfo is None

    records_snapshot[0]["decision"]["should_trade"] = False
    assert trader.get_risk_evaluations()[0]["decision"]["should_trade"] is True

    flattened_df.loc[0, "decision_should_trade"] = False
    assert trader.get_risk_evaluations()[0]["decision"]["should_trade"] is True

    df.loc[pd.isna(df["service"]), "normalized"] = False
    assert trader.get_risk_evaluations()[2]["normalized"] is None


def test_auto_trader_risk_history_emits_events(monkeypatch: pytest.MonkeyPatch) -> None:
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
        risk_evaluations_limit=2,
    )

    decision = RiskDecision(
        should_trade=True,
        fraction=0.25,
        state="active",
        details={"origin": "listener-test"},
    )

    payloads: list[dict[str, Any]] = []
    trader.add_risk_evaluation_listener(payloads.append)

    with pytest.raises(TypeError):
        trader.add_risk_evaluation_listener(object())  # type: ignore[arg-type]

    class _ServiceGamma:
        ...

    service = _ServiceGamma()

    timestamps = iter([1000.0, 1010.0, 1020.0, 1030.0])
    monkeypatch.setattr("bot_core.auto_trader.app.time.time", lambda: next(timestamps))

    trader._record_risk_evaluation(
        decision,
        approved=True,
        normalized=True,
        response=True,
        service=service,
        error=None,
    )
    trader._record_risk_evaluation(
        decision,
        approved=False,
        normalized=False,
        response=False,
        service=None,
        error=None,
    )
    trader._record_risk_evaluation(
        decision,
        approved=True,
        normalized=True,
        response=True,
        service=None,
        error=None,
    )

    risk_events = [payload for event, payload in emitter.events if event == "auto_trader.risk_evaluation"]
    assert len(risk_events) == 3
    assert len(payloads) == 3
    assert payloads[0]["service"] == "_ServiceGamma"
    assert payloads[0]["history_size"] == 1
    assert payloads[0]["history_trimmed_by_limit"] == 0
    assert payloads[-1]["history_size"] == 2
    assert payloads[-1]["history_trimmed_by_limit"] == 1
    assert payloads[-1]["history_limit"] == 2
    assert payloads[-1]["history_trimmed_by_ttl"] == 0

    trader.remove_risk_evaluation_listener(payloads.append)
    trader.remove_risk_evaluation_listener(payloads.append)

    trader._record_risk_evaluation(
        decision,
        approved=None,
        normalized=None,
        response=None,
        service=None,
        error=None,
    )

    risk_events_after = [payload for event, payload in emitter.events if event == "auto_trader.risk_evaluation"]
    assert len(risk_events_after) == 4
    assert len(payloads) == 3


def test_auto_trader_load_risk_evaluations_notifies_listeners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        fraction=0.5,
        state="active",
        details={"origin": "load-test"},
    )

    timestamps = iter([2000.0, 2010.0])
    monkeypatch.setattr("bot_core.auto_trader.app.time.time", lambda: next(timestamps))

    trader._record_risk_evaluation(
        decision,
        approved=True,
        normalized=True,
        response=True,
        service=None,
        error=None,
    )
    trader._record_risk_evaluation(
        decision,
        approved=False,
        normalized=False,
        response=False,
        service=None,
        error=None,
    )

    payload = trader.export_risk_evaluations()

    loader_emitter = _Emitter()
    loader = AutoTrader(
        loader_emitter,
        _GUI(),
        symbol_getter=lambda: "SOLUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=False,
        market_data_provider=_Provider(_build_market_data()),
    )

    default_notifications: list[dict[str, Any]] = []
    loader.add_risk_evaluation_listener(default_notifications.append)

    loader.load_risk_evaluations(payload)

    assert default_notifications == []
    default_risk_events = [
        event_payload
        for event_name, event_payload in loader_emitter.events
        if event_name == "auto_trader.risk_evaluation"
    ]
    assert len(default_risk_events) == len(payload.get("entries", []))

    notifying_emitter = _Emitter()
    notifying_loader = AutoTrader(
        notifying_emitter,
        _GUI(),
        symbol_getter=lambda: "SOLUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=False,
        market_data_provider=_Provider(_build_market_data()),
    )

    notified: list[dict[str, Any]] = []
    notifying_loader.add_risk_evaluation_listener(notified.append)

    notifying_loader.load_risk_evaluations(payload, notify_listeners=True)

    assert len(notified) == len(payload.get("entries", []))
    assert notified[0]["history_size"] == 1
    assert notified[-1]["history_size"] == len(payload.get("entries", []))
    assert notified[-1]["history_trimmed_by_limit"] == 0

    notified_events = [
        event_payload
        for event_name, event_payload in notifying_emitter.events
        if event_name == "auto_trader.risk_evaluation"
    ]
    assert notified_events == notified


def test_auto_trader_exports_and_dumps_risk_evaluations(tmp_path: Path) -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "ADAUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=False,
        market_data_provider=provider,
        risk_evaluations_limit=5,
        risk_evaluations_ttl_s=60.0,
    )

    class _ExportService:
        pass

    decision_active = RiskDecision(
        should_trade=True,
        fraction=0.4,
        state="active",
        reason="go",
        mode="auto",
        details={"origin": "export-test"},
    )
    decision_blocked = RiskDecision(
        should_trade=False,
        fraction=0.0,
        state="blocked",
        reason="cooldown",
        mode="demo",
        details={},
    )

    trader._record_risk_evaluation(
        decision_active,
        approved=True,
        normalized=True,
        response={"ok": True},
        service=_ExportService(),
        error=None,
    )
    trader._record_risk_evaluation(
        decision_blocked,
        approved=False,
        normalized=None,
        response=False,
        service=None,
        error=RuntimeError("failure"),
    )

    export = trader.export_risk_evaluations(
        flatten_decision=True,
        decision_fields=["state"],
        fill_value="missing",
        coerce_timestamps=True,
    )

    assert export["version"] == 1
    assert export["history_size"] == len(export["entries"]) >= 2
    assert export["retention"]["limit"] == 5
    assert export["retention"]["ttl_s"] == pytest.approx(60.0)
    assert export["filters"]["decision_fields"] == ["state"]
    assert export["filters"]["flatten_decision"] is True
    assert isinstance(export["entries"][0]["timestamp"], str)
    assert export["entries"][0]["decision_state"] == "active"
    assert export["entries"][0]["service"] == "_ExportService"
    assert "error" in export["entries"][1]

    destination = tmp_path / "risk_history.json"
    trader.dump_risk_evaluations(
        destination,
        flatten_decision=True,
        decision_fields=["state"],
        fill_value="missing",
        coerce_timestamps=True,
    )

    stored = json.loads(destination.read_text())
    assert stored == export


def test_auto_trader_loads_and_imports_risk_evaluations(tmp_path: Path) -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "MATICUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=False,
        market_data_provider=provider,
        risk_evaluations_limit=5,
        risk_evaluations_ttl_s=120.0,
    )

    class _ImportService:
        pass

    decision_active = RiskDecision(
        should_trade=True,
        fraction=0.3,
        state="active",
        reason="go",
        mode="auto",
        details={"origin": "load-test"},
    )
    decision_blocked = RiskDecision(
        should_trade=False,
        fraction=0.0,
        state="blocked",
        reason="cooldown",
        mode="demo",
        details={},
    )

    trader._record_risk_evaluation(
        decision_active,
        approved=True,
        normalized=True,
        response={"ok": True},
        service=_ImportService(),
        error=None,
    )
    trader._record_risk_evaluation(
        decision_blocked,
        approved=False,
        normalized=None,
        response=False,
        service=None,
        error=RuntimeError("boom"),
    )

    export = trader.export_risk_evaluations(
        flatten_decision=True,
        decision_fields=["state", "should_trade"],
        drop_decision_column=True,
        fill_value="<missing>",
        coerce_timestamps=True,
    )

    limited_export = copy.deepcopy(export)
    limited_export["retention"]["limit"] = 1

    trader.clear_risk_evaluations()
    loaded = trader.load_risk_evaluations(limited_export)
    assert loaded == len(export["entries"])

    restored = trader.get_risk_evaluations()
    assert len(restored) == 1
    restored_entry = restored[0]
    assert isinstance(restored_entry["timestamp"], float)
    assert restored_entry["decision"]["state"] == "blocked"
    assert restored_entry["decision"]["should_trade"] is False

    destination = tmp_path / "risk_history.json"
    destination.write_text(json.dumps(export, indent=2), encoding="utf-8")

    trader.clear_risk_evaluations()
    loaded_from_file = trader.import_risk_evaluations(destination)
    assert loaded_from_file == len(export["entries"])

    restored_all = trader.get_risk_evaluations()
    assert [entry["decision"]["state"] for entry in restored_all] == ["active", "blocked"]

    partial_payload = copy.deepcopy(export)
    partial_entry = copy.deepcopy(partial_payload["entries"][0])
    partial_entry["timestamp"] = (
        pd.Timestamp(partial_entry["timestamp"]) + pd.Timedelta(seconds=90)
    ).isoformat()
    partial_payload["entries"] = [partial_entry]
    partial_payload["retention"] = {}

    merged = trader.load_risk_evaluations(partial_payload, merge=True)
    assert merged == 1
    merged_states = [entry["decision"]["state"] for entry in trader.get_risk_evaluations()]
    assert merged_states.count("active") >= 1
    assert merged_states.count("blocked") >= 1
    assert len(merged_states) == len(export["entries"]) + 1


def test_auto_trader_filters_risk_history_by_decision_fields() -> None:
    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider(_build_market_data())

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "OPUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=False,
        market_data_provider=provider,
    )

    guardrail_decision = RiskDecision(
        should_trade=False,
        fraction=0.15,
        state="blocked",
        reason="guardrail",
        mode="auto",
        details={"origin": "decision-filter-test"},
    )
    missing_reason_decision = RiskDecision(
        should_trade=False,
        fraction=0.1,
        state="blocked",
        reason=None,
        mode="auto",
        details={"origin": "decision-filter-test"},
    )
    active_decision = RiskDecision(
        should_trade=True,
        fraction=0.5,
        state="active",
        reason="go",
        mode="demo",
        details={"origin": "decision-filter-test"},
    )
    cooldown_decision = RiskDecision(
        should_trade=False,
        fraction=0.05,
        state="cooldown",
        reason="cooldown",
        mode="paper",
        details={"origin": "decision-filter-test"},
    )

    trader._record_risk_evaluation(
        guardrail_decision,
        approved=False,
        normalized=False,
        response=False,
        service=None,
        error=None,
    )
    trader._record_risk_evaluation(
        missing_reason_decision,
        approved=False,
        normalized=False,
        response=False,
        service=None,
        error=None,
    )
    trader._record_risk_evaluation(
        active_decision,
        approved=True,
        normalized=True,
        response=True,
        service=None,
        error=None,
    )
    trader._record_risk_evaluation(
        cooldown_decision,
        approved=False,
        normalized=False,
        response=False,
        service=None,
        error=None,
    )

    blocked = trader.get_risk_evaluations(decision_state="blocked")
    assert len(blocked) == 2
    assert all(entry["decision"]["state"] == "blocked" for entry in blocked)

    guardrail_reason = trader.get_risk_evaluations(decision_reason="guardrail")
    assert [entry["decision"]["reason"] for entry in guardrail_reason] == [
        "guardrail",
    ]

    missing_reason = trader.get_risk_evaluations(decision_reason=None)
    assert len(missing_reason) == 1
    assert missing_reason[0]["decision"]["reason"] is None

    auto_mode = trader.get_risk_evaluations(decision_mode="auto")
    assert len(auto_mode) == 2
    assert all(entry["decision"]["mode"] == "auto" for entry in auto_mode)

    summary_active = trader.summarize_risk_evaluations(decision_state="active")
    assert summary_active["total"] == 1
    assert summary_active["approved"] == 1
    assert summary_active["rejected"] == 0

    paper_records = trader.risk_evaluations_to_records(
        decision_mode="paper",
        flatten_decision=True,
        decision_fields=("state", "mode"),
    )
    assert len(paper_records) == 1
    assert paper_records[0]["decision_state"] == "cooldown"
    assert paper_records[0]["decision_mode"] == "paper"

    guardrail_records = trader.risk_evaluations_to_records(
        decision_reason=["guardrail", "cooldown"],
    )
    assert len(guardrail_records) == 2
    assert {record["decision"]["reason"] for record in guardrail_records} == {
        "guardrail",
        "cooldown",
    }

    df_auto = trader.risk_evaluations_to_dataframe(decision_mode=["auto", "demo"])
    assert len(df_auto) == 3
    assert set(df_auto["decision"].apply(lambda payload: payload.get("mode"))) == {
        "auto",
        "demo",
    }


def test_auto_trader_summarizes_decision_dimensions() -> None:
    emitter = _Emitter()
    gui = _GUI()
    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "BNBUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=False,
    )

    class _ServiceAlpha:
        ...

    class _ServiceBeta:
        ...

    class _ServiceGamma:
        ...

    alpha = _ServiceAlpha()
    beta = _ServiceBeta()
    gamma = _ServiceGamma()

    active_auto = RiskDecision(
        should_trade=True,
        fraction=0.35,
        state="active",
        reason="momentum",
        mode="auto",
    )
    blocked_auto = RiskDecision(
        should_trade=False,
        fraction=0.25,
        state="blocked",
        reason="guardrail",
        mode="auto",
    )
    blocked_demo = RiskDecision(
        should_trade=False,
        fraction=0.2,
        state="blocked",
        reason="guardrail",
        mode="demo",
    )
    active_paper = RiskDecision(
        should_trade=True,
        fraction=0.15,
        state="active",
        reason=None,
        mode="paper",
    )
    blocked_cooldown = RiskDecision(
        should_trade=False,
        fraction=0.1,
        state="blocked",
        reason="cooldown",
        mode="auto",
    )

    trader._record_risk_evaluation(
        active_auto,
        approved=True,
        normalized=True,
        response=True,
        service=alpha,
        error=None,
    )
    trader._record_risk_evaluation(
        blocked_auto,
        approved=False,
        normalized=False,
        response=False,
        service=alpha,
        error=None,
    )
    trader._record_risk_evaluation(
        blocked_auto,
        approved=False,
        normalized=False,
        response=False,
        service=gamma,
        error=RuntimeError("blocked"),
    )
    trader._record_risk_evaluation(
        blocked_demo,
        approved=True,
        normalized=False,
        response=False,
        service=beta,
        error=None,
    )
    trader._record_risk_evaluation(
        active_paper,
        approved=None,
        normalized=None,
        response=True,
        service=beta,
        error=None,
    )
    trader._record_risk_evaluation(
        blocked_cooldown,
        approved=False,
        normalized=False,
        response=False,
        service=beta,
        error=None,
    )

    summary = trader.summarize_risk_decision_dimensions()
    assert summary["total"] == 6
    assert summary["services"] == {
        "_ServiceBeta": 3,
        "_ServiceAlpha": 2,
        "_ServiceGamma": 1,
    }
    assert summary["first_timestamp"] is not None
    assert summary["last_timestamp"] is not None
    assert summary["last_timestamp"] >= summary["first_timestamp"]

    blocked_state = summary["states"]["blocked"]
    assert blocked_state["total"] == 4
    assert blocked_state["rejected"] == 4
    assert blocked_state["services"] == {
        "_ServiceBeta": 2,
        "_ServiceAlpha": 1,
        "_ServiceGamma": 1,
    }
    assert blocked_state["error_rate"] == pytest.approx(0.25)

    active_state = summary["states"]["active"]
    assert active_state["total"] == 2
    assert active_state["approved"] == 1
    assert active_state["unknown"] == 1
    assert active_state["raw_none"] == 1

    guardrail_reason = summary["reasons"]["guardrail"]
    assert guardrail_reason["total"] == 3
    assert guardrail_reason["services"] == {
        "_ServiceAlpha": 1,
        "_ServiceBeta": 1,
        "_ServiceGamma": 1,
    }

    missing_reason = summary["reasons"]["<no-reason>"]
    assert missing_reason["total"] == 1
    assert missing_reason["unknown"] == 1

    auto_mode_summary = summary["modes"]["auto"]
    assert auto_mode_summary["total"] == 4
    assert auto_mode_summary["errors"] == 1

    combinations = summary["combinations"]
    assert combinations[0]["state"] == "blocked"
    assert combinations[0]["reason"] == "guardrail"
    assert combinations[0]["mode"] == "auto"
    assert combinations[0]["total"] == 2
    assert combinations[0]["services"] == {
        "_ServiceAlpha": 1,
        "_ServiceGamma": 1,
    }
    assert combinations[0]["error_rate"] == pytest.approx(0.5)

    blocked_only = trader.summarize_risk_decision_dimensions(decision_state="blocked")
    assert blocked_only["total"] == 4
    assert list(blocked_only["states"].keys()) == ["blocked"]
    assert all(row["state"] == "blocked" for row in blocked_only["combinations"])

    guardrail_only = trader.summarize_risk_decision_dimensions(decision_reason="guardrail")
    assert guardrail_only["total"] == 3
    assert all(row["reason"] == "guardrail" for row in guardrail_only["combinations"])

    demo_only = trader.summarize_risk_decision_dimensions(decision_mode="demo")
    assert demo_only["total"] == 1
    assert demo_only["combinations"][0]["mode"] == "demo"

    beta_only = trader.summarize_risk_decision_dimensions(service="_ServiceBeta")
    assert beta_only["total"] == 3
    assert beta_only["services"] == {"_ServiceBeta": 3}

    no_errors = trader.summarize_risk_decision_dimensions(include_errors=False)
    assert no_errors["total"] == 5
    assert no_errors["combinations"][0]["total"] == 1
    assert "_ServiceGamma" not in no_errors["combinations"][0]["services"]


def test_auto_trader_decision_timeline_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitter = _Emitter()
    gui = _GUI()
    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "BNBUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=False,
    )

    class _ServiceAlpha:
        ...

    class _ServiceBeta:
        ...

    class _ServiceGamma:
        ...

    alpha = _ServiceAlpha()
    beta = _ServiceBeta()
    gamma = _ServiceGamma()

    timestamps = iter([1000.0, 1015.0, 1039.0, 1078.0])
    monkeypatch.setattr("bot_core.auto_trader.app.time.time", lambda: next(timestamps))

    trader._record_risk_evaluation(
        RiskDecision(
            should_trade=True,
            fraction=0.5,
            state="active",
            reason="ok",
            mode="auto",
        ),
        approved=True,
        normalized=True,
        response=True,
        service=alpha,
        error=None,
    )
    trader._record_risk_evaluation(
        RiskDecision(
            should_trade=False,
            fraction=0.2,
            state="blocked",
            reason="guardrail",
            mode="auto",
        ),
        approved=False,
        normalized=False,
        response=False,
        service=alpha,
        error=None,
    )
    trader._record_risk_evaluation(
        RiskDecision(
            should_trade=True,
            fraction=0.1,
            state="monitoring",
            reason=None,
            mode="paper",
        ),
        approved=True,
        normalized=True,
        response=True,
        service=beta,
        error=None,
    )
    trader._record_risk_evaluation(
        RiskDecision(
            should_trade=False,
            fraction=0.15,
            state="blocked",
            reason="cooldown",
            mode="demo",
        ),
        approved=None,
        normalized=None,
        response=False,
        service=gamma,
        error=RuntimeError("cooldown"),
    )

    timeline = trader.summarize_risk_decision_timeline(
        bucket_s=20,
        include_decision_dimensions=True,
        fill_gaps=True,
    )
    assert timeline["bucket_s"] == pytest.approx(20.0)
    assert timeline["total"] == 4
    assert timeline["first_timestamp"] == pytest.approx(1000.0)
    assert timeline["last_timestamp"] == pytest.approx(1078.0)
    assert timeline["services"] == {
        "_ServiceAlpha": {"evaluations": 2, "errors": 0},
        "_ServiceBeta": {"evaluations": 1, "errors": 0},
        "_ServiceGamma": {"evaluations": 1, "errors": 1},
    }
    assert [bucket["start"] for bucket in timeline["buckets"]] == [
        1000.0,
        1020.0,
        1040.0,
        1060.0,
    ]

    first_bucket = timeline["buckets"][0]
    assert first_bucket["total"] == 2
    assert first_bucket["approved"] == 1
    assert first_bucket["rejected"] == 1
    assert first_bucket["services"] == {"_ServiceAlpha": 2}
    assert first_bucket["states"] == {"active": 1, "blocked": 1}
    assert first_bucket["reasons"] == {"guardrail": 1, "ok": 1}
    assert first_bucket["modes"] == {"auto": 2}

    second_bucket = timeline["buckets"][1]
    assert second_bucket["total"] == 1
    assert second_bucket["services"] == {"_ServiceBeta": 1}
    assert second_bucket["states"] == {"monitoring": 1}
    assert second_bucket["reasons"] == {"<no-reason>": 1}
    assert second_bucket["modes"] == {"paper": 1}

    empty_bucket = timeline["buckets"][2]
    assert empty_bucket["total"] == 0
    assert empty_bucket["services"] == {}
    assert empty_bucket["states"] == {}
    assert empty_bucket["reasons"] == {}
    assert empty_bucket["modes"] == {}

    last_bucket = timeline["buckets"][3]
    assert last_bucket["total"] == 1
    assert last_bucket["errors"] == 1
    assert last_bucket["services"] == {"_ServiceGamma": 1}
    assert last_bucket["reasons"] == {"cooldown": 1}
    assert last_bucket["states"] == {"blocked": 1}
    assert last_bucket["modes"] == {"demo": 1}
    assert timeline["approved"] == 2
    assert timeline["rejected"] == 1
    assert timeline["unknown"] == 1
    assert timeline["errors"] == 1
    assert timeline["services"] == {
        "_ServiceAlpha": {
            "evaluations": 2,
            "approved": 1,
            "rejected": 1,
            "unknown": 0,
            "errors": 0,
            "raw_true": 1,
            "raw_false": 1,
            "raw_none": 0,
        },
        "_ServiceBeta": {
            "evaluations": 1,
            "approved": 1,
            "rejected": 0,
            "unknown": 0,
            "errors": 0,
            "raw_true": 1,
            "raw_false": 0,
            "raw_none": 0,
        },
        "_ServiceGamma": {
            "evaluations": 1,
            "approved": 0,
            "rejected": 0,
            "unknown": 1,
            "errors": 1,
            "raw_true": 0,
            "raw_false": 0,
            "raw_none": 1,
        },
    }

    evaluations = trader.get_risk_evaluations()
    assert len(evaluations) == 4
    sample_decision_id = evaluations[0]["decision_id"]
    filtered_timeline = trader.summarize_risk_decision_timeline(
        bucket_s=20,
        decision_id=sample_decision_id,
        include_decision_dimensions=True,
        fill_gaps=True,
    )
    assert filtered_timeline["total"] == 1
    assert filtered_timeline["filters"]["decision_id"] == [sample_decision_id]
    assert sum(bucket["total"] for bucket in filtered_timeline["buckets"]) == 1

    blocked_only = trader.summarize_risk_decision_timeline(
        bucket_s=20,
        decision_state="blocked",
    )
    assert blocked_only["total"] == 2
    assert [bucket["start"] for bucket in blocked_only["buckets"]] == [
        1000.0,
        1060.0,
    ]

    timeline_no_services = trader.summarize_risk_decision_timeline(
        bucket_s=20,
        include_services=False,
    )
    assert "services" not in timeline_no_services
    assert "services" not in timeline_no_services["buckets"][0]

    timeline_coerced = trader.summarize_risk_decision_timeline(
        bucket_s=20,
        coerce_timestamps=True,
    )
    assert timeline_coerced["buckets"][0]["start"] == datetime.fromtimestamp(
        1000.0, tz=timezone.utc
    )

    with pytest.raises(ValueError):
        trader.summarize_risk_decision_timeline(bucket_s=0)


def test_auto_trader_decision_timeline_exports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitter = _Emitter()
    gui = _GUI()
    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "BNBUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=False,
    )

    class _ServiceAlpha:
        ...

    class _ServiceBeta:
        ...

    alpha = _ServiceAlpha()
    beta = _ServiceBeta()

    timestamps = iter([1200.0, 1210.0, 1250.0, float("nan")])
    monkeypatch.setattr("bot_core.auto_trader.app.time.time", lambda: next(timestamps))

    trader._record_risk_evaluation(
        RiskDecision(
            should_trade=True,
            fraction=0.5,
            state="active",
            reason="ok",
            mode="auto",
        ),
        approved=True,
        normalized=True,
        response=True,
        service=alpha,
        error=None,
    )
    trader._record_risk_evaluation(
        RiskDecision(
            should_trade=False,
            fraction=0.3,
            state="blocked",
            reason="guardrail",
            mode="auto",
        ),
        approved=False,
        normalized=False,
        response=False,
        service=alpha,
        error=None,
    )
    trader._record_risk_evaluation(
        RiskDecision(
            should_trade=True,
            fraction=0.2,
            state="monitoring",
            reason=None,
            mode="paper",
        ),
        approved=True,
        normalized=True,
        response=True,
        service=beta,
        error=None,
    )
    trader._record_risk_evaluation(
        RiskDecision(
            should_trade=False,
            fraction=0.1,
            state="blocked",
            reason="cooldown",
            mode="demo",
        ),
        approved=None,
        normalized=None,
        response=False,
        service=beta,
        error=RuntimeError("cooldown"),
    )

    evaluations = trader.get_risk_evaluations()
    target_decision_id = evaluations[0]["decision_id"]

    base_records = trader.risk_decision_timeline_to_records(
        bucket_s=20,
        fill_gaps=True,
        include_services=False,
    )
    assert [record["index"] for record in base_records] == [60, 61, 62]
    assert base_records[0]["bucket_type"] == "bucket"
    assert base_records[0]["total"] == 2
    assert base_records[1]["total"] == 0
    assert base_records[2]["total"] == 1

    enriched_records = trader.risk_decision_timeline_to_records(
        bucket_s=20,
        fill_gaps=True,
        include_services=True,
        include_decision_dimensions=True,
        include_missing_bucket=True,
    )
    assert isinstance(enriched_records, GuardrailTimelineRecords)
    assert [record["bucket_type"] for record in enriched_records] == [
        "bucket",
        "bucket",
        "bucket",
        "missing",
        "summary",
    ]
    first_bucket = enriched_records[0]
    assert first_bucket["index"] == 60
    assert first_bucket["start"] == pytest.approx(1200.0)
    assert first_bucket["services"] == {"_ServiceAlpha": 2}
    assert first_bucket["states"] == {"active": 1, "blocked": 1}
    assert first_bucket["reasons"] == {"guardrail": 1, "ok": 1}
    missing_record = enriched_records[-2]
    assert missing_record["bucket_type"] == "missing"
    assert missing_record["start"] is None
    assert missing_record["index"] is None
    summary_record = enriched_records[-1]
    assert summary_record["bucket_type"] == "summary"
    assert summary_record["errors"] == 1
    assert summary_record["services"]["_ServiceAlpha"]["evaluations"] == 2
    assert summary_record["services"]["_ServiceBeta"]["errors"] == 1
    assert "services" in enriched_records.summary
    assert enriched_records.summary["services"]["_ServiceAlpha"]["evaluations"] == 2

    filtered_records = trader.risk_decision_timeline_to_records(
        bucket_s=20,
        decision_id=target_decision_id,
        include_services=True,
        include_decision_dimensions=True,
    )
    assert isinstance(filtered_records, GuardrailTimelineRecords)
    assert filtered_records.summary["filters"]["decision_id"] == [target_decision_id]
    filtered_bucket_totals = [
        record.get("total", 0)
        for record in filtered_records
        if record.get("bucket_type") == "bucket"
    ]
    assert sum(filtered_bucket_totals) == 1

    df = trader.risk_decision_timeline_to_dataframe(
        bucket_s=20,
        fill_gaps=True,
        include_services=True,
        include_decision_dimensions=True,
        include_missing_bucket=True,
    )
    assert list(df["bucket_type"]) == [
        "bucket",
        "bucket",
        "bucket",
        "missing",
        "summary",
    ]
    assert df.loc[df["bucket_type"] == "bucket", "total"].sum() == 3
    assert df.loc[df["bucket_type"] == "missing", "total"].iloc[0] == 1
    assert df.loc[df["bucket_type"] == "summary", "errors"].iloc[0] == 1
    assert df.loc[0, "start"] == datetime.fromtimestamp(1200.0, tz=timezone.utc)
    assert "states" in df.columns
    assert "services" in df.columns

    filtered_df = trader.risk_decision_timeline_to_dataframe(
        bucket_s=20,
        decision_id=target_decision_id,
        include_services=True,
        include_decision_dimensions=True,
    )
    assert (
        filtered_df.loc[filtered_df["bucket_type"] == "bucket", "total"].sum() == 1
    )


def test_auto_trader_guardrail_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    trader = _prepare_guardrail_history(monkeypatch)

    summary = trader.summarize_risk_guardrails()
    assert summary["total"] == 4
    assert summary["guardrail_events"] == 3
    assert summary["reasons"]["effective risk cap"] == 3
    assert summary["reasons"]["volatility spike"] == 1
    assert summary["reasons"]["liquidity pressure"] == 1

    blocked_summary = trader.summarize_risk_guardrails(decision_state="blocked")
    assert blocked_summary["total"] == 3
    assert blocked_summary["guardrail_events"] == 3

    demo_mode_summary = trader.summarize_risk_guardrails(decision_mode="demo")
    assert demo_mode_summary["total"] == 1
    assert demo_mode_summary["guardrail_events"] == 0

    triggers = summary["triggers"]
    assert set(triggers.keys()) == {"effective_risk", "volatility_ratio"}

    effective = triggers["effective_risk"]
    assert effective["count"] == 3
    assert effective["label"] == "Effective risk cap"
    assert effective["comparator"] == ">="
    assert effective["unit"] == "ratio"
    assert effective["threshold"] == pytest.approx(0.8)
    assert effective["value_min"] == pytest.approx(0.79)
    assert effective["value_max"] == pytest.approx(0.83)
    assert effective["value_avg"] == pytest.approx((0.83 + 0.81 + 0.79) / 3)
    assert effective["value_last"] == pytest.approx(0.79)
    assert effective["services"]["_ServiceAlpha"] == 2
    assert effective["services"]["<unknown>"] == 1

    volatility = triggers["volatility_ratio"]
    assert volatility["count"] == 1
    assert volatility["threshold"] == pytest.approx(1.2)
    assert volatility["unit"] == "ratio"
    assert volatility["value_min"] == pytest.approx(1.25)
    assert volatility["value_max"] == pytest.approx(1.25)
    assert volatility["value_avg"] == pytest.approx(1.25)
    assert volatility["services"]["_ServiceAlpha"] == 1

    services = summary["services"]
    assert set(services.keys()) == {"_ServiceAlpha", "_ServiceBeta", "<unknown>"}
    alpha_bucket = services["_ServiceAlpha"]
    assert alpha_bucket["total"] == 2
    assert alpha_bucket["guardrail_events"] == 2
    assert alpha_bucket["reasons"]["effective risk cap"] == 2
    assert alpha_bucket["triggers"]["effective_risk"] == 2
    assert alpha_bucket["triggers"]["volatility_ratio"] == 1

    beta_bucket = services["_ServiceBeta"]
    assert beta_bucket["total"] == 1
    assert beta_bucket["guardrail_events"] == 0
    assert beta_bucket["reasons"] == {}
    assert beta_bucket["triggers"] == {}

    unknown_bucket = services["<unknown>"]
    assert unknown_bucket["total"] == 1
    assert unknown_bucket["guardrail_events"] == 1
    assert unknown_bucket["reasons"]["effective risk cap"] == 1
    assert unknown_bucket["triggers"]["effective_risk"] == 1

    filtered_without_errors = trader.summarize_risk_guardrails(include_errors=False)
    assert filtered_without_errors["total"] == 3
    assert filtered_without_errors["guardrail_events"] == 2
    assert set(filtered_without_errors["services"].keys()) == {"_ServiceAlpha", "_ServiceBeta"}

    alpha_only = trader.summarize_risk_guardrails(service="_ServiceAlpha")
    assert alpha_only["total"] == 2
    assert alpha_only["guardrail_events"] == 2
    assert set(alpha_only["services"].keys()) == {"_ServiceAlpha"}

    recent_only = trader.summarize_risk_guardrails(
        since=pd.Timestamp(2015.0, unit="s"),
    )
    assert recent_only["total"] == 2
    assert recent_only["guardrail_events"] == 1

    liquidity_only = trader.summarize_risk_guardrails(reason="liquidity pressure")
    assert liquidity_only["total"] == 1
    assert liquidity_only["guardrail_events"] == 1
    assert set(liquidity_only["services"].keys()) == {"<unknown>"}
    assert liquidity_only["services"]["<unknown>"]["total"] == 1
    assert liquidity_only["reasons"]["liquidity pressure"] == 1
    assert liquidity_only["triggers"]["effective_risk"]["count"] == 1

    trigger_only_summary = trader.summarize_risk_guardrails(trigger="volatility_ratio")
    assert trigger_only_summary["total"] == 1
    assert trigger_only_summary["guardrail_events"] == 1
    assert set(trigger_only_summary["services"].keys()) == {"_ServiceAlpha"}
    assert trigger_only_summary["triggers"]["volatility_ratio"]["count"] == 1
    assert trigger_only_summary["reasons"]["effective risk cap"] == 1

    unit_ratio_summary = trader.summarize_risk_guardrails(trigger_unit="ratio")
    assert unit_ratio_summary["total"] == 2
    assert unit_ratio_summary["guardrail_events"] == 2

    unit_score_summary = trader.summarize_risk_guardrails(trigger_unit="score")
    assert unit_score_summary["total"] == 1
    assert unit_score_summary["guardrail_events"] == 1
    assert set(unit_score_summary["services"].keys()) == {"<unknown>"}

    label_filtered_summary = trader.summarize_risk_guardrails(
        trigger_label="Volatility ratio"
    )
    assert label_filtered_summary["total"] == 1
    assert label_filtered_summary["guardrail_events"] == 1
    assert "volatility_ratio" in label_filtered_summary["triggers"]
    assert label_filtered_summary["triggers"]["volatility_ratio"]["count"] == 1

    comparator_filtered_summary = trader.summarize_risk_guardrails(
        trigger_comparator=">="
    )
    assert comparator_filtered_summary["total"] == 3
    assert comparator_filtered_summary["guardrail_events"] == 3

    threshold_filtered_summary = trader.summarize_risk_guardrails(
        trigger_threshold=0.8,
    )
    assert threshold_filtered_summary["total"] == 3
    assert threshold_filtered_summary["guardrail_events"] == 3

    high_threshold_summary = trader.summarize_risk_guardrails(
        trigger_threshold_min=1.0,
    )
    assert high_threshold_summary["total"] == 1
    assert high_threshold_summary["guardrail_events"] == 1

    threshold_range_summary = trader.summarize_risk_guardrails(
        trigger_threshold_min=0.8,
        trigger_threshold_max=0.8,
    )
    assert threshold_range_summary["total"] == 3
    assert threshold_range_summary["guardrail_events"] == 3

    value_exact_summary = trader.summarize_risk_guardrails(trigger_value=0.79)
    assert value_exact_summary["total"] == 1
    assert value_exact_summary["guardrail_events"] == 1

    value_range_summary = trader.summarize_risk_guardrails(
        trigger_value_min=0.8,
        trigger_value_max=0.85,
    )
    assert value_range_summary["total"] == 2
    assert value_range_summary["guardrail_events"] == 2

    high_value_summary = trader.summarize_risk_guardrails(
        trigger_value_min=1.2,
    )
    assert high_value_summary["total"] == 1
    assert high_value_summary["guardrail_events"] == 1

    records = trader.guardrail_events_to_records()
    assert [record["service"] for record in records] == [
        "_ServiceAlpha",
        "_ServiceAlpha",
        "<unknown>",
    ]
    assert all(record["decision_id"] for record in records)
    assert records[0]["guardrail_reasons"] == (
        "effective risk cap",
        "volatility spike",
    )
    assert records[0]["guardrail_trigger_count"] == 2
    assert records[2]["error"].startswith("RuntimeError")
    assert records[2]["guardrail_triggers"][0]["name"] == "effective_risk"
    assert records[2]["guardrail_triggers"][0]["unit"] == "score"
    assert records[0]["decision"]["details"]["guardrail_reasons"][0] == "effective risk cap"
    assert records[0]["guardrail_triggers"][0]["unit"] == "ratio"
    assert "guardrail_dimensions" in records[0]
    dimensions = records[0]["guardrail_dimensions"]
    assert dimensions["reasons"] == records[0]["guardrail_reasons"]
    assert dimensions["triggers"][0]["name"] == records[0]["guardrail_triggers"][0]["name"]
    assert dimensions["tokens"] and dimensions["tokens"][0]["name"]

    alpha_only_records = trader.guardrail_events_to_records(service="_ServiceAlpha")
    assert len(alpha_only_records) == 2
    assert all(record["service"] == "_ServiceAlpha" for record in alpha_only_records)

    recent_records = trader.guardrail_events_to_records(
        since=pd.Timestamp(2015.0, unit="s"),
    )
    assert len(recent_records) == 1
    assert recent_records[0]["guardrail_reasons"] == ("effective risk cap", "liquidity pressure")

    limited_records = trader.guardrail_events_to_records(limit=1)
    assert len(limited_records) == 1

    reversed_records = trader.guardrail_events_to_records(reverse=True)
    assert reversed_records[0]["service"] == "<unknown>"

    filtered_records = trader.guardrail_events_to_records(include_errors=False)
    assert len(filtered_records) == 2
    assert all(record["error"] is None for record in filtered_records)

    liquidity_records = trader.guardrail_events_to_records(reason="liquidity pressure")
    assert len(liquidity_records) == 1
    assert liquidity_records[0]["service"] == "<unknown>"
    assert liquidity_records[0]["guardrail_reason_count"] == 2

    blocked_records = trader.guardrail_events_to_records(decision_state="blocked")
    assert len(blocked_records) == 3

    demo_mode_records = trader.guardrail_events_to_records(decision_mode="demo")
    assert demo_mode_records == []

    trigger_records = trader.guardrail_events_to_records(trigger="volatility_ratio")
    assert len(trigger_records) == 1
    assert trigger_records[0]["service"] == "_ServiceAlpha"
    assert trigger_records[0]["guardrail_trigger_count"] == 2

    label_records = trader.guardrail_events_to_records(
        trigger_label="Volatility ratio"
    )
    assert len(label_records) == 1
    assert label_records[0]["service"] == "_ServiceAlpha"

    comparator_records = trader.guardrail_events_to_records(trigger_comparator=">=")
    assert len(comparator_records) == 3

    unit_records = trader.guardrail_events_to_records(trigger_unit="ratio")
    assert len(unit_records) == 2

    unit_score_records = trader.guardrail_events_to_records(trigger_unit="score")
    assert len(unit_score_records) == 1
    assert unit_score_records[0]["service"] == "<unknown>"

    threshold_records = trader.guardrail_events_to_records(trigger_threshold=0.8)
    assert len(threshold_records) == 3

    high_threshold_records = trader.guardrail_events_to_records(
        trigger_threshold_min=1.0,
    )
    assert len(high_threshold_records) == 1

    value_records = trader.guardrail_events_to_records(trigger_value=0.79)
    assert len(value_records) == 1

    value_range_records = trader.guardrail_events_to_records(
        trigger_value_min=0.8,
        trigger_value_max=0.85,
    )
    assert len(value_range_records) == 2

    coerced_records = trader.guardrail_events_to_records(coerce_timestamps=True)
    assert isinstance(coerced_records[0]["timestamp"], datetime)
    assert coerced_records[0]["timestamp"].tzinfo is timezone.utc

    naive_records = trader.guardrail_events_to_records(
        coerce_timestamps=True,
        tz=None,
    )
    assert naive_records[0]["timestamp"].tzinfo is None

    df = trader.guardrail_events_to_dataframe()
    assert "decision_id" in df.columns
    assert df["decision_id"].notna().all()
    assert list(df["service"]) == ["_ServiceAlpha", "_ServiceAlpha", "<unknown>"]
    assert df.loc[0, "guardrail_reason_count"] == 2
    assert df.loc[2, "guardrail_trigger_count"] == 1
    assert df.loc[2, "error"].startswith("RuntimeError")
    assert df.loc[0, "guardrail_triggers"][0]["name"] == "effective_risk"
    assert df.loc[0, "guardrail_triggers"][0]["unit"] == "ratio"
    assert isinstance(df.loc[0, "timestamp"], datetime)
    assert df.loc[0, "timestamp"].tzinfo is timezone.utc

    df_recent = trader.guardrail_events_to_dataframe(
        since=pd.Timestamp(2015.0, unit="s"),
    )
    assert len(df_recent) == 1

    df_blocked = trader.guardrail_events_to_dataframe(decision_state="blocked")
    assert len(df_blocked) == 3

    df_demo_mode = trader.guardrail_events_to_dataframe(decision_mode="demo")
    assert df_demo_mode.empty

    df_filtered = trader.guardrail_events_to_dataframe(include_errors=False)
    assert len(df_filtered) == 2

    df_liquidity = trader.guardrail_events_to_dataframe(reason="liquidity pressure")
    assert len(df_liquidity) == 1
    assert df_liquidity.loc[0, "service"] == "<unknown>"

    df_trigger = trader.guardrail_events_to_dataframe(trigger="volatility_ratio")
    assert len(df_trigger) == 1
    assert df_trigger.loc[0, "service"] == "_ServiceAlpha"

    df_label = trader.guardrail_events_to_dataframe(trigger_label="Volatility ratio")
    assert len(df_label) == 1
    assert df_label.loc[0, "service"] == "_ServiceAlpha"

    df_comparator = trader.guardrail_events_to_dataframe(trigger_comparator=">=")
    assert len(df_comparator) == 3

    df_unit = trader.guardrail_events_to_dataframe(trigger_unit="ratio")
    assert len(df_unit) == 2

    df_unit_score = trader.guardrail_events_to_dataframe(trigger_unit="score")
    assert list(df_unit_score["service"]) == ["<unknown>"]

    df_threshold = trader.guardrail_events_to_dataframe(trigger_threshold=0.8)
    assert len(df_threshold) == 3

    df_high_threshold = trader.guardrail_events_to_dataframe(
        trigger_threshold_min=1.0,
    )
    assert len(df_high_threshold) == 1

    df_value_exact = trader.guardrail_events_to_dataframe(trigger_value=0.79)
    assert len(df_value_exact) == 1

    df_value_range = trader.guardrail_events_to_dataframe(
        trigger_value_min=0.8,
        trigger_value_max=0.85,
    )
    assert len(df_value_range) == 2

    empty_reason_df = trader.guardrail_events_to_dataframe(reason="non-existent")
    assert empty_reason_df.empty

    df_reversed = trader.guardrail_events_to_dataframe(reverse=True)
    assert df_reversed.iloc[0]["service"] == "<unknown>"

    df_with_decision = trader.guardrail_events_to_dataframe(include_decision=True)
    assert "decision" in df_with_decision.columns
    assert df_with_decision.loc[0, "decision"]["details"]["guardrail_triggers"][0]["name"] == "effective_risk"

    empty_df = trader.guardrail_events_to_dataframe(limit=0)
    assert empty_df.empty


def test_guardrail_filters_support_decision_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trader = _prepare_guardrail_history(monkeypatch)

    records = trader.guardrail_events_to_records()
    assert all(isinstance(record.get("decision_id"), str) for record in records)
    target_id = records[0]["decision_id"]

    summary = trader.summarize_risk_guardrails(decision_id=target_id)
    assert summary["guardrail_events"] == 1

    filtered_records = trader.guardrail_events_to_records(decision_id=target_id)
    assert [entry["decision_id"] for entry in filtered_records] == [target_id]

    df = trader.guardrail_events_to_dataframe(decision_id=target_id)
    assert df["decision_id"].unique().tolist() == [target_id]

    timeline_summary = trader.summarize_guardrail_timeline(
        bucket_s=20.0,
        decision_id=target_id,
    )
    assert timeline_summary["filters"]["decision_id"] == [target_id]

    timeline_records = trader.guardrail_timeline_to_records(
        bucket_s=20.0,
        decision_id=target_id,
        include_summary_metadata=True,
    )
    assert isinstance(timeline_records, GuardrailTimelineRecords)
    assert timeline_records.summary["filters"]["decision_id"] == [target_id]

    df_timeline = trader.guardrail_timeline_to_dataframe(
        bucket_s=20.0,
        decision_id=target_id,
    )
    assert (
        df_timeline.attrs["guardrail_summary"]["filters"]["decision_id"]
        == [target_id]
    )


def test_auto_trader_guardrail_timeline_exports(monkeypatch: pytest.MonkeyPatch) -> None:
    trader = _prepare_guardrail_history(monkeypatch)

    summary = trader.summarize_guardrail_timeline(bucket_s=10.0)
    summary_filters = summary["filters"]
    assert summary_filters["approved"] is None
    assert summary_filters["normalized"] is None
    assert summary_filters["include_errors"] is True
    assert summary_filters["service"] is None
    assert summary_filters["trigger"] is None
    assert summary_filters["trigger_threshold"] is None
    assert summary_filters["trigger_value"] is None
    assert summary_filters["since"] is None
    assert summary_filters["until"] is None
    assert summary_filters["include_services"] is True
    assert summary_filters["include_guardrail_dimensions"] is True
    assert summary_filters["include_decision_dimensions"] is False
    assert summary_filters["fill_gaps"] is False
    assert summary_filters["coerce_timestamps"] is False
    assert summary_filters["tz"] == "UTC"
    threshold_totals = summary["guardrail_trigger_thresholds"]
    assert threshold_totals["count"] == 4
    assert threshold_totals["missing"] == 0
    assert threshold_totals["sum"] == pytest.approx(3.6)
    assert threshold_totals["min"] == pytest.approx(0.8)
    assert threshold_totals["max"] == pytest.approx(1.2)
    assert threshold_totals["average"] == pytest.approx(0.9)
    value_totals = summary["guardrail_trigger_values"]
    assert value_totals["count"] == 4
    assert value_totals["missing"] == 0
    assert value_totals["sum"] == pytest.approx(3.68)
    assert value_totals["min"] == pytest.approx(0.79)
    assert value_totals["max"] == pytest.approx(1.25)
    assert value_totals["average"] == pytest.approx(0.92)
    assert summary["total"] == 3
    assert summary["evaluations"] == 4
    assert summary["guardrail_rate"] == pytest.approx(
        summary["total"] / summary["evaluations"]
    )
    assert summary["approval_states"] == {"denied": 3, "approved": 1}
    assert summary["normalization_states"] == {"raw": 3, "normalized": 1}
    assert summary["first_timestamp"] == pytest.approx(2000.0)
    assert summary["last_timestamp"] == pytest.approx(2030.0)
    assert summary["services"] == {
        "<unknown>": {"evaluations": 1, "guardrail_events": 1},
        "_ServiceAlpha": {"evaluations": 2, "guardrail_events": 2},
        "_ServiceBeta": {"evaluations": 1, "guardrail_events": 0},
    }
    assert summary["guardrail_reasons"] == {
        "effective risk cap": 3,
        "liquidity pressure": 1,
        "volatility spike": 1,
    }
    assert summary["guardrail_triggers"] == {
        "effective_risk": 3,
        "volatility_ratio": 1,
    }
    assert summary["guardrail_trigger_labels"] == {
        "Effective risk cap": 3,
        "Volatility ratio": 1,
    }
    assert summary["guardrail_trigger_comparators"] == {">=": 4}
    assert summary["guardrail_trigger_units"] == {"ratio": 3, "score": 1}

    bucket_indices = [bucket["index"] for bucket in summary["buckets"]]
    assert bucket_indices == [200, 201, 202, 203]

    first_bucket = summary["buckets"][0]
    assert first_bucket["guardrail_events"] == 1
    assert first_bucket["evaluations"] == 1
    assert first_bucket["guardrail_rate"] == pytest.approx(1.0)
    assert first_bucket["approval_states"] == {"denied": 1}
    assert first_bucket["normalization_states"] == {"raw": 1}
    assert first_bucket["services"]["_ServiceAlpha"]["guardrail_events"] == 1
    assert first_bucket["services"]["_ServiceAlpha"]["evaluations"] == 1
    assert first_bucket["guardrail_reasons"] == {
        "effective risk cap": 1,
        "volatility spike": 1,
    }
    assert first_bucket["guardrail_triggers"] == {
        "effective_risk": 1,
        "volatility_ratio": 1,
    }
    assert first_bucket["guardrail_trigger_labels"] == {
        "Effective risk cap": 1,
        "Volatility ratio": 1,
    }
    assert first_bucket["guardrail_trigger_comparators"] == {">=": 2}
    assert first_bucket["guardrail_trigger_units"] == {"ratio": 2}
    first_thresholds = first_bucket["guardrail_trigger_thresholds"]
    assert first_thresholds["count"] == 2
    assert first_thresholds["missing"] == 0
    assert first_thresholds["sum"] == pytest.approx(2.0)
    assert first_thresholds["min"] == pytest.approx(0.8)
    assert first_thresholds["max"] == pytest.approx(1.2)
    assert first_thresholds["average"] == pytest.approx(1.0)
    first_values = first_bucket["guardrail_trigger_values"]
    assert first_values["count"] == 2
    assert first_values["missing"] == 0
    assert first_values["sum"] == pytest.approx(2.08)
    assert first_values["min"] == pytest.approx(0.83)
    assert first_values["max"] == pytest.approx(1.25)
    assert first_values["average"] == pytest.approx(1.04)

    neutral_bucket = next(bucket for bucket in summary["buckets"] if bucket["index"] == 202)
    assert neutral_bucket["guardrail_events"] == 0
    assert neutral_bucket["evaluations"] == 1
    assert neutral_bucket["guardrail_rate"] == pytest.approx(0.0)
    assert neutral_bucket["approval_states"] == {"approved": 1}
    assert neutral_bucket["normalization_states"] == {"normalized": 1}
    assert neutral_bucket["services"]["_ServiceBeta"]["evaluations"] == 1
    assert neutral_bucket["services"]["_ServiceBeta"]["guardrail_events"] == 0
    assert neutral_bucket["guardrail_reasons"] == {}
    assert neutral_bucket["guardrail_triggers"] == {}
    assert neutral_bucket["guardrail_trigger_labels"] == {}
    assert neutral_bucket["guardrail_trigger_comparators"] == {}
    assert neutral_bucket["guardrail_trigger_units"] == {}
    assert neutral_bucket["guardrail_trigger_thresholds"] == {}
    assert neutral_bucket["guardrail_trigger_values"] == {}

    last_bucket = summary["buckets"][-1]
    assert last_bucket["guardrail_events"] == 1
    assert last_bucket["guardrail_rate"] == pytest.approx(1.0)
    assert last_bucket["approval_states"] == {"denied": 1}
    assert last_bucket["normalization_states"] == {"raw": 1}
    assert last_bucket["services"]["<unknown>"]["guardrail_events"] == 1
    assert last_bucket["guardrail_reasons"] == {
        "effective risk cap": 1,
        "liquidity pressure": 1,
    }
    assert last_bucket["guardrail_triggers"] == {"effective_risk": 1}
    assert last_bucket["guardrail_trigger_labels"] == {"Effective risk cap": 1}
    assert last_bucket["guardrail_trigger_comparators"] == {">=": 1}
    assert last_bucket["guardrail_trigger_units"] == {"score": 1}
    last_thresholds = last_bucket["guardrail_trigger_thresholds"]
    assert last_thresholds["count"] == 1
    assert last_thresholds["missing"] == 0
    assert last_thresholds["sum"] == pytest.approx(0.8)
    assert last_thresholds["min"] == pytest.approx(0.8)
    assert last_thresholds["max"] == pytest.approx(0.8)
    assert last_thresholds["average"] == pytest.approx(0.8)
    last_values = last_bucket["guardrail_trigger_values"]
    assert last_values["count"] == 1
    assert last_values["missing"] == 0
    assert last_values["sum"] == pytest.approx(0.79)
    assert last_values["min"] == pytest.approx(0.79)
    assert last_values["max"] == pytest.approx(0.79)
    assert last_values["average"] == pytest.approx(0.79)

    decision_summary = trader.summarize_guardrail_timeline(
        bucket_s=10.0,
        include_decision_dimensions=True,
    )
    assert decision_summary["decision_states"] == {"blocked": 3}
    assert decision_summary["decision_reasons"] == {"guardrail-blocked": 3}
    assert decision_summary["decision_modes"] == {"auto": 3}
    first_decision_bucket = decision_summary["buckets"][0]
    assert first_decision_bucket["decision_states"] == {"blocked": 1}
    assert first_decision_bucket["decision_reasons"] == {"guardrail-blocked": 1}
    assert first_decision_bucket["decision_modes"] == {"auto": 1}

    filtered = trader.summarize_guardrail_timeline(
        bucket_s=10.0,
        service="_ServiceAlpha",
    )
    assert filtered["total"] == 2
    assert filtered["evaluations"] == 2
    assert [bucket["index"] for bucket in filtered["buckets"]] == [200, 201]

    summary_snapshot = trader.summarize_guardrail_timeline(
        bucket_s=20.0,
        fill_gaps=True,
        coerce_timestamps=True,
        tz=timezone.utc,
    )

    records = trader.guardrail_timeline_to_records(
        bucket_s=20.0,
        fill_gaps=True,
        coerce_timestamps=True,
        tz=timezone.utc,
    )
    assert len(records) == 2
    assert all(record["bucket_type"] == "bucket" for record in records)
    assert all(isinstance(record["start"], datetime) for record in records)
    assert hasattr(records, "summary")
    summary_metadata = records.summary
    assert summary_metadata["bucket_s"] == pytest.approx(20.0)
    assert summary_metadata["total"] == summary_snapshot["total"]
    assert summary_metadata["evaluations"] == summary_snapshot["evaluations"]
    assert summary_metadata["approval_states"]["denied"] == 3
    assert summary_metadata["normalization_states"]["raw"] == 3
    assert summary_metadata["first_timestamp"] == summary_snapshot["first_timestamp"]
    assert summary_metadata["last_timestamp"] == summary_snapshot["last_timestamp"]
    assert summary_metadata.get("missing_timestamp") == summary_snapshot.get(
        "missing_timestamp"
    )
    assert summary_metadata["services"]["_ServiceAlpha"]["guardrail_events"] == 2
    assert summary_metadata["services"]["_ServiceAlpha"]["evaluations"] == 2
    assert summary_metadata["guardrail_trigger_thresholds"]["count"] == 4
    first_record_thresholds = records[0]["guardrail_trigger_thresholds"]
    assert first_record_thresholds["count"] == 3
    assert first_record_thresholds["sum"] == pytest.approx(2.8)
    first_record_values = records[0]["guardrail_trigger_values"]
    assert first_record_values["count"] == 3
    assert first_record_values["sum"] == pytest.approx(2.89)
    second_record_thresholds = records[1]["guardrail_trigger_thresholds"]
    assert second_record_thresholds["count"] == 1
    second_record_values = records[1]["guardrail_trigger_values"]
    assert second_record_values["count"] == 1

    df_minimal = trader.guardrail_timeline_to_dataframe(
        bucket_s=20.0,
        fill_gaps=True,
        coerce_timestamps=True,
        tz=timezone.utc,
    )
    summary_records = trader.guardrail_timeline_to_records(
        bucket_s=20.0,
        fill_gaps=True,
        coerce_timestamps=True,
        tz=timezone.utc,
        include_summary_metadata=True,
    )
    assert len(summary_records) == 3
    summary_record = summary_records[-1]
    assert summary_record["bucket_type"] == "summary"
    assert summary_record["evaluations"] == summary_snapshot["evaluations"]
    assert summary_record["total"] == summary_snapshot["total"]
    assert summary_record["guardrail_events"] == summary_snapshot["total"]
    assert summary_record.get("services") == summary_snapshot.get("services")
    assert summary_record.get("guardrail_trigger_thresholds") == summary_snapshot.get(
        "guardrail_trigger_thresholds"
    )
    assert summary_record.get("guardrail_trigger_values") == summary_snapshot.get(
        "guardrail_trigger_values"
    )
    assert summary_record["approval_states"] == summary_snapshot["approval_states"]
    assert summary_record["normalization_states"] == summary_snapshot[
        "normalization_states"
    ]
    assert summary_record["guardrail_rate"] == pytest.approx(
        summary_snapshot["total"] / summary_snapshot["evaluations"]
    )
    assert isinstance(summary_record["first_timestamp"], datetime)
    assert isinstance(summary_record["last_timestamp"], datetime)
    assert summary_record["filters"]["fill_gaps"] is True
    assert summary_record["filters"]["coerce_timestamps"] is True
    assert summary_records.summary["filters"]["fill_gaps"] is True
    assert summary_records.summary["filters"]["tz"] == "UTC"
    assert summary_records.summary["guardrail_reasons"]["effective risk cap"] == 3
    assert summary_records.summary["guardrail_trigger_units"] == {
        "ratio": 3,
        "score": 1,
    }
    assert summary_records.summary["approval_states"] == summary_snapshot[
        "approval_states"
    ]
    assert summary_records.summary["normalization_states"] == summary_snapshot[
        "normalization_states"
    ]
    assert summary_records.summary["guardrail_rate"] == summary_snapshot[
        "guardrail_rate"
    ]

    df = trader.guardrail_timeline_to_dataframe(
        bucket_s=20.0,
        include_services=False,
        include_guardrail_dimensions=False,
    )
    assert list(df_minimal["guardrail_events"]) == [2, 1]
    for events, evals, rate in zip(
        df_minimal["guardrail_events"].tolist(),
        df_minimal["evaluations"].tolist(),
        df_minimal["guardrail_rate"].tolist(),
    ):
        expected_rate = events / evals if evals else 0.0
        assert rate == pytest.approx(expected_rate)
    assert "services" not in df_minimal.columns
    assert "guardrail_reasons" not in df_minimal.columns
    assert "guardrail_trigger_thresholds" not in df_minimal.columns
    assert "guardrail_trigger_values" not in df_minimal.columns

    df_with_metadata = trader.guardrail_timeline_to_dataframe(bucket_s=20.0)
    assert df_with_metadata.attrs["guardrail_summary"]["services"]["_ServiceAlpha"][
        "guardrail_events"
    ] == 2
    assert df_with_metadata.attrs["guardrail_summary"]["guardrail_trigger_thresholds"][
        "count"
    ] == 4
    assert df_with_metadata.attrs["guardrail_summary"]["guardrail_reasons"][
        "effective risk cap"
    ] == 3
    assert df_with_metadata.attrs["guardrail_summary"]["approval_states"][
        "denied"
    ] == 3
    assert df_with_metadata.attrs["guardrail_summary"]["normalization_states"][
        "raw"
    ] == 3
    assert df_with_metadata.attrs["guardrail_summary"]["guardrail_rate"] == (
        pytest.approx(summary["total"] / summary["evaluations"])
    )

    df_with_summary = trader.guardrail_timeline_to_dataframe(
        bucket_s=20.0,
        fill_gaps=True,
        include_summary_metadata=True,
        tz=timezone.utc,
    )
    assert df_with_summary["bucket_type"].iloc[-1] == "summary"
    summary_row = df_with_summary.iloc[-1]
    assert summary_row["evaluations"] == summary_snapshot["evaluations"]
    assert summary_row["total"] == summary_snapshot["total"]
    assert isinstance(summary_row["first_timestamp"], datetime)
    assert isinstance(summary_row["last_timestamp"], datetime)
    assert summary_row["guardrail_rate"] == pytest.approx(
        summary_snapshot["total"] / summary_snapshot["evaluations"]
    )
    if "services" in df_with_summary.columns:
        assert summary_row["services"] == summary_snapshot.get("services")
    if "guardrail_reasons" in df_with_summary.columns:
        assert summary_row["guardrail_reasons"] == summary_snapshot.get(
            "guardrail_reasons"
        )
    assert summary_row["approval_states"] == summary_snapshot["approval_states"]
    assert summary_row["normalization_states"] == summary_snapshot[
        "normalization_states"
    ]
    df_summary_metadata = df_with_summary.attrs.get("guardrail_summary")
    assert df_summary_metadata is not None
    assert df_summary_metadata["total"] == summary_snapshot["total"]
    assert df_summary_metadata["first_timestamp"] == summary_snapshot["first_timestamp"]
    assert df_summary_metadata["filters"]["fill_gaps"] is True
    assert df_summary_metadata["approval_states"] == summary_snapshot["approval_states"]
    assert df_summary_metadata["normalization_states"] == summary_snapshot[
        "normalization_states"
    ]
    assert df_summary_metadata["guardrail_rate"] == summary_snapshot["guardrail_rate"]

    records_with_missing = trader.guardrail_timeline_to_records(
        bucket_s=20.0,
        include_missing_bucket=True,
    )
    assert len(records_with_missing) == 2
    missing_record = next(
        (
            record
            for record in records_with_missing
            if record.get("bucket_type") == "missing"
        ),
        None,
    )
    if missing_record is not None:
        assert missing_record["guardrail_trigger_labels"] == {"Effective risk cap": 1}
        assert missing_record["guardrail_trigger_comparators"] == {">=": 1}
        assert missing_record["guardrail_trigger_units"] == {"score": 1}
        assert missing_record["services"]["<unknown>"]["guardrail_events"] == 1
        assert missing_record["services"]["<unknown>"]["evaluations"] == 1
        assert missing_record["guardrail_rate"] == pytest.approx(1.0)
        missing_thresholds = missing_record["guardrail_trigger_thresholds"]
        assert missing_thresholds["count"] == 1
        assert missing_thresholds["missing"] == 0
        assert missing_thresholds["sum"] == pytest.approx(0.8)
        assert missing_thresholds["average"] == pytest.approx(0.8)
        assert missing_thresholds["min"] == pytest.approx(0.8)
        assert missing_thresholds["max"] == pytest.approx(0.8)
        missing_values = missing_record["guardrail_trigger_values"]
        assert missing_values["count"] == 1
        assert missing_values["missing"] == 0
        assert missing_values["sum"] == pytest.approx(0.79)
        assert missing_values["average"] == pytest.approx(0.79)
        assert missing_values["min"] == pytest.approx(0.79)
        assert missing_values["max"] == pytest.approx(0.79)

    reason_filtered = trader.summarize_guardrail_timeline(
        bucket_s=10.0,
        reason="liquidity pressure",
    )
    assert reason_filtered["total"] == 1
    assert reason_filtered["evaluations"] == 4


def test_guardrail_filters_support_missing_label_and_comparator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitter = _Emitter()
    gui = _GUI()
    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "ETHUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=False,
    )

    timestamps = iter([3000.0, 3010.0, 3020.0])
    monkeypatch.setattr("bot_core.auto_trader.app.time.time", lambda: next(timestamps))

    def _decision(
        reasons: Iterable[str],
        triggers: Iterable[GuardrailTrigger],
    ) -> RiskDecision:
        return RiskDecision(
            should_trade=False,
            fraction=0.0,
            state="blocked",
            details={
                "origin": "guardrail-test-missing",
                "guardrail_reasons": list(reasons),
                "guardrail_triggers": [
                    trigger.to_dict() if hasattr(trigger, "to_dict") else dict(trigger)
                    for trigger in triggers
                ],
            },
        )

    trader._record_risk_evaluation(
        _decision(
            ["missing label"],
            [
                GuardrailTrigger(
                    name="missing_label",
                    label=None,
                    comparator=None,
                    threshold=1.0,
                    unit=None,
                    value=1.1,
                )
            ],
        ),
        approved=False,
        normalized=False,
        response=False,
        service=None,
        error=None,
    )

    trader._record_risk_evaluation(
        _decision(
            ["has label"],
            [
                GuardrailTrigger(
                    name="labeled",
                    label="Labeled guardrail",
                    comparator="<=",
                    threshold=2.0,
                    unit="ratio",
                    value=None,
                )
            ],
        ),
        approved=False,
        normalized=False,
        response=False,
        service="alpha",
        error=None,
    )

    trader._record_risk_evaluation(
        _decision(
            ["missing threshold"],
            [
                {
                    "name": "no_threshold",
                    "label": "Numeric guardrail",
                    "comparator": ">=",
                    "unit": "bps",
                    "value": 2.2,
                }
            ],
        ),
        approved=False,
        normalized=False,
        response=False,
        service="beta",
        error=None,
    )

    missing_label_summary = trader.summarize_risk_guardrails(trigger_label=None)
    assert missing_label_summary["total"] == 1
    assert missing_label_summary["guardrail_events"] == 1

    labeled_summary = trader.summarize_risk_guardrails(trigger_label="Labeled guardrail")
    assert labeled_summary["total"] == 1
    assert labeled_summary["guardrail_events"] == 1

    missing_comparator_summary = trader.summarize_risk_guardrails(trigger_comparator=None)
    assert missing_comparator_summary["total"] == 1
    assert missing_comparator_summary["guardrail_events"] == 1

    comparator_summary = trader.summarize_risk_guardrails(trigger_comparator="<=")
    assert comparator_summary["total"] == 1
    assert comparator_summary["guardrail_events"] == 1

    threshold_exact_summary = trader.summarize_risk_guardrails(trigger_threshold=1.0)
    assert threshold_exact_summary["total"] == 1
    assert threshold_exact_summary["guardrail_events"] == 1

    missing_threshold_summary = trader.summarize_risk_guardrails(trigger_threshold=None)
    assert missing_threshold_summary["total"] == 1
    assert missing_threshold_summary["guardrail_events"] == 1

    threshold_range_summary = trader.summarize_risk_guardrails(
        trigger_threshold_min=1.5,
    )
    assert threshold_range_summary["total"] == 1
    assert threshold_range_summary["guardrail_events"] == 1

    value_exact_summary = trader.summarize_risk_guardrails(trigger_value=1.1)
    assert value_exact_summary["total"] == 1
    assert value_exact_summary["guardrail_events"] == 1

    missing_value_summary = trader.summarize_risk_guardrails(trigger_value=None)
    assert missing_value_summary["total"] == 1
    assert missing_value_summary["guardrail_events"] == 1

    value_range_summary = trader.summarize_risk_guardrails(
        trigger_value_min=2.0,
    )
    assert value_range_summary["total"] == 1
    assert value_range_summary["guardrail_events"] == 1

    missing_unit_summary = trader.summarize_risk_guardrails(trigger_unit=None)
    assert missing_unit_summary["total"] == 1
    assert missing_unit_summary["guardrail_events"] == 1

    ratio_unit_summary = trader.summarize_risk_guardrails(trigger_unit="ratio")
    assert ratio_unit_summary["total"] == 1
    assert ratio_unit_summary["guardrail_events"] == 1

    bps_unit_summary = trader.summarize_risk_guardrails(trigger_unit="bps")
    assert bps_unit_summary["total"] == 1
    assert bps_unit_summary["guardrail_events"] == 1

    missing_label_records = trader.guardrail_events_to_records(trigger_label=None)
    assert len(missing_label_records) == 1
    assert missing_label_records[0]["guardrail_triggers"][0]["label"] is None

    comparator_records = trader.guardrail_events_to_records(trigger_comparator="<=")
    assert len(comparator_records) == 1
    assert comparator_records[0]["guardrail_triggers"][0]["comparator"] == "<="

    missing_comparator_records = trader.guardrail_events_to_records(
        trigger_comparator=None
    )
    assert len(missing_comparator_records) == 1
    assert missing_comparator_records[0]["guardrail_triggers"][0]["comparator"] is None

    threshold_records = trader.guardrail_events_to_records(trigger_threshold=1.0)
    assert len(threshold_records) == 1

    missing_threshold_records = trader.guardrail_events_to_records(trigger_threshold=None)
    assert len(missing_threshold_records) == 1
    assert "threshold" not in missing_threshold_records[0]["guardrail_triggers"][0]

    threshold_range_records = trader.guardrail_events_to_records(
        trigger_threshold_min=1.5,
    )
    assert len(threshold_range_records) == 1

    value_records = trader.guardrail_events_to_records(trigger_value=1.1)
    assert len(value_records) == 1

    missing_value_records = trader.guardrail_events_to_records(trigger_value=None)
    assert len(missing_value_records) == 1
    assert "value" not in missing_value_records[0]["guardrail_triggers"][0]

    value_range_records = trader.guardrail_events_to_records(
        trigger_value_min=2.0,
    )
    assert len(value_range_records) == 1

    missing_unit_records = trader.guardrail_events_to_records(trigger_unit=None)
    assert len(missing_unit_records) == 1
    assert missing_unit_records[0]["guardrail_triggers"][0].get("unit") is None

    ratio_unit_records = trader.guardrail_events_to_records(trigger_unit="ratio")
    assert len(ratio_unit_records) == 1
    assert ratio_unit_records[0]["guardrail_triggers"][0]["unit"] == "ratio"

    bps_unit_records = trader.guardrail_events_to_records(trigger_unit="bps")
    assert len(bps_unit_records) == 1
    assert bps_unit_records[0]["guardrail_triggers"][0]["unit"] == "bps"

    missing_label_df = trader.guardrail_events_to_dataframe(trigger_label=None)
    assert len(missing_label_df) == 1

    comparator_df = trader.guardrail_events_to_dataframe(trigger_comparator="<=")
    assert len(comparator_df) == 1

    missing_comparator_df = trader.guardrail_events_to_dataframe(trigger_comparator=None)
    assert len(missing_comparator_df) == 1

    df_threshold = trader.guardrail_events_to_dataframe(trigger_threshold=1.0)
    assert len(df_threshold) == 1

    df_missing_threshold = trader.guardrail_events_to_dataframe(trigger_threshold=None)
    assert len(df_missing_threshold) == 1

    df_threshold_range = trader.guardrail_events_to_dataframe(
        trigger_threshold_min=1.5,
    )
    assert len(df_threshold_range) == 1

    df_value = trader.guardrail_events_to_dataframe(trigger_value=1.1)
    assert len(df_value) == 1

    df_missing_value = trader.guardrail_events_to_dataframe(trigger_value=None)
    assert len(df_missing_value) == 1

    df_value_range = trader.guardrail_events_to_dataframe(
        trigger_value_min=2.0,
    )
    assert len(df_value_range) == 1

    df_missing_unit = trader.guardrail_events_to_dataframe(trigger_unit=None)
    assert len(df_missing_unit) == 1
    assert df_missing_unit.loc[0, "guardrail_triggers"][0].get("unit") is None

    df_ratio_unit = trader.guardrail_events_to_dataframe(trigger_unit="ratio")
    assert len(df_ratio_unit) == 1
    assert df_ratio_unit.loc[0, "guardrail_triggers"][0]["unit"] == "ratio"

    df_bps_unit = trader.guardrail_events_to_dataframe(trigger_unit="bps")
    assert len(df_bps_unit) == 1
    assert df_bps_unit.loc[0, "guardrail_triggers"][0]["unit"] == "bps"


def test_extract_guardrail_dimensions_prefers_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitter = _Emitter()
    gui = _GUI()
    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "BTCUSDT",
        enable_auto_trade=False,
    )

    entry: dict[str, Any] = {
        "guardrail_dimensions": {
            "reasons": ("cached reason",),
            "triggers": (
                {
                    "name": "cached_guard",
                    "label": None,
                    "comparator": None,
                    "unit": None,
                },
            ),
            "tokens": (
                {
                    "name": "cached_guard",
                    "label": "<no-label>",
                    "comparator": "<no-comparator>",
                    "unit": "<no-unit>",
                    "threshold": None,
                    "value": None,
                },
            ),
        },
        "decision": {"details": {"guardrail_triggers": [{"name": "should_not_be_used"}]}},
    }

    calls: list[Any] = []

    def _patched(payload: object) -> list[tuple[Any, dict[str, Any]]]:
        calls.append(payload)
        return []

    monkeypatch.setattr(
        "bot_core.auto_trader.app.normalize_guardrail_triggers",
        _patched,
    )

    reasons, triggers, tokens = trader._extract_guardrail_dimensions(entry)

    assert reasons == ("cached reason",)
    assert triggers == (
        {"name": "cached_guard", "label": None, "comparator": None, "unit": None},
    )
    assert tokens == [
        {
            "name": "cached_guard",
            "label": "<no-label>",
            "comparator": "<no-comparator>",
            "unit": "<no-unit>",
            "threshold": None,
            "value": None,
        }
    ]
    assert calls == []


def test_guardrail_events_export_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    trader = _prepare_guardrail_history(monkeypatch)

    payload = trader.export_guardrail_events(
        include_service=False,
        include_response=False,
        include_error=False,
    )

    assert payload["version"] == 1
    retention = payload["retention"]
    assert isinstance(retention["limit"], (int, type(None)))
    assert retention["ttl_s"] == trader.get_risk_evaluations_ttl()
    assert payload["filters"]["include_service"] is False
    assert len(payload["entries"]) == 3
    assert {entry["decision_id"] for entry in payload["entries"]} == {
        "guardrail-alpha-1",
        "guardrail-alpha-2",
        "guardrail-unknown-1",
    }
    first_entry = payload["entries"][0]
    assert "guardrail_dimensions" in first_entry
    dimensions_snapshot = first_entry["guardrail_dimensions"]
    assert dimensions_snapshot["reasons"]
    assert dimensions_snapshot["tokens"] and dimensions_snapshot["tokens"][0]["name"]

    fresh = AutoTrader(
        _Emitter(),
        _GUI(),
        symbol_getter=lambda: "SOLUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=False,
    )

    assert not fresh.guardrail_events_to_records()

    loaded = fresh.load_guardrail_events(payload)
    assert loaded == 3

    records = fresh.guardrail_events_to_records()
    assert len(records) == 3
    assert {record["decision_id"] for record in records} == {
        "guardrail-alpha-1",
        "guardrail-alpha-2",
        "guardrail-unknown-1",
    }
    restored_dimensions = records[0]["guardrail_dimensions"]
    assert restored_dimensions["tokens"] and restored_dimensions["tokens"][0]["name"]

    restored_entries = fresh.get_risk_evaluations()
    assert restored_entries
    snapshot = restored_entries[0]["guardrail_dimensions"]
    assert snapshot["reasons"]
    assert isinstance(snapshot["tokens"], tuple)
    assert snapshot["tokens"] and snapshot["tokens"][0]["name"]


def test_guardrail_events_dump_and_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trader = _prepare_guardrail_history(monkeypatch)
    export_path = tmp_path / "guardrail_events.json"

    trader.dump_guardrail_events(export_path, ensure_ascii=True)
    assert export_path.exists()

    imported = AutoTrader(
        _Emitter(),
        _GUI(),
        symbol_getter=lambda: "SOLUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=False,
    )

    count = imported.import_guardrail_events(export_path)
    assert count == 3

    summary = imported.summarize_guardrail_timeline(bucket_s=60.0)
    assert summary["total"] == 3


def test_guardrail_event_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    trader = _prepare_guardrail_history(monkeypatch)

    trace = trader.get_guardrail_event_trace(
        "guardrail-alpha-1",
        include_service=False,
        include_response=False,
        include_error=False,
        include_guardrail_dimensions=False,
    )

    assert len(trace) == 1
    first = trace[0]
    assert first["decision_id"] == "guardrail-alpha-1"
    assert first["step_index"] == 0
    assert first["elapsed_since_first_s"] == pytest.approx(0.0)
    assert first["elapsed_since_previous_s"] == pytest.approx(0.0)
    assert isinstance(first["timestamp"], datetime)
    assert "guardrail_dimensions" not in first

    missing = trader.get_guardrail_event_trace("non-existent")
    assert missing == ()


def test_guardrail_event_grouping(monkeypatch: pytest.MonkeyPatch) -> None:
    trader = _prepare_guardrail_history(monkeypatch)

    grouped = trader.get_grouped_guardrail_events(
        include_decision=False,
        include_service=False,
        include_response=False,
        include_error=False,
        include_guardrail_dimensions=False,
    )

    assert set(grouped.keys()) == {
        "guardrail-alpha-1",
        "guardrail-alpha-2",
        "guardrail-unknown-1",
    }

    sample_group = grouped["guardrail-alpha-1"]
    assert sample_group
    assert "guardrail_dimensions" not in sample_group[0]

    alpha_records = grouped["guardrail-alpha-1"]
    assert len(alpha_records) == 1
    assert alpha_records[0]["decision_id"] == "guardrail-alpha-1"
    assert "service" not in alpha_records[0]


def test_auto_trader_prunes_risk_history_by_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    emitter = _Emitter()
    gui = _GUI()
    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "ADAUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        risk_evaluations_limit=None,
        risk_evaluations_ttl_s=10.0,
    )

    decision = RiskDecision(should_trade=True, fraction=0.25, state="ok")

    class _TimeStub:
        def __init__(self, values: list[float]) -> None:
            self._values = values
            self._index = 0
            self.last_value: float | None = None

        def __call__(self) -> float:
            if self._index < len(self._values):
                value = self._values[self._index]
                self._index += 1
            else:
                value = self._values[-1]
            self.last_value = value
            return value

    time_stub = _TimeStub([1000.0, 1002.0, 1010.0, 1012.0, 1014.0, 1016.0, 1018.0, 1020.0, 1022.0])
    monkeypatch.setattr("bot_core.auto_trader.app.time.time", time_stub)

    trader._record_risk_evaluation(
        decision,
        approved=True,
        normalized=True,
        response=True,
        service=None,
        error=None,
    )
    trader._record_risk_evaluation(
        decision,
        approved=False,
        normalized=False,
        response=False,
        service=None,
        error=None,
    )
    trader._record_risk_evaluation(
        decision,
        approved=True,
        normalized=True,
        response=True,
        service=None,
        error=None,
    )

    evaluations = trader.get_risk_evaluations()
    assert len(evaluations) == 2
    assert [entry["normalized"] for entry in evaluations] == [False, True]
    assert evaluations[0]["timestamp"] == pytest.approx(1002.0)
    assert evaluations[1]["timestamp"] == pytest.approx(1010.0)
    assert trader.get_risk_evaluations_ttl() == pytest.approx(10.0)

    new_ttl = trader.set_risk_evaluations_ttl(1.0)
    assert new_ttl == pytest.approx(1.0)
    assert trader.get_risk_evaluations_ttl() == pytest.approx(1.0)

    assert trader.get_risk_evaluations() == []

    disabled_ttl = trader.set_risk_evaluations_ttl(None)
    assert disabled_ttl is None
    assert trader.get_risk_evaluations_ttl() is None

    trader._record_risk_evaluation(
        decision,
        approved=True,
        normalized=True,
        response=True,
        service=None,
        error=None,
    )

    refreshed = trader.get_risk_evaluations()
    assert len(refreshed) == 1
    assert refreshed[0]["timestamp"] == pytest.approx(time_stub.last_value or 0.0)
