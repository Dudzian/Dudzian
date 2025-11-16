"""Dane demonstracyjne dla panelu decyzji AI."""
from __future__ import annotations

from typing import Iterable, Mapping

from bot_core.ai.regime import MarketRegime, MarketRegimeAssessment
from bot_core.auto_trader import AutoTraderAIGovernor


def _build_assessment(
    regime: MarketRegime,
    *,
    risk_score: float,
    confidence: float = 0.72,
    metrics: Mapping[str, float] | None = None,
) -> MarketRegimeAssessment:
    payload = dict(metrics or {})
    return MarketRegimeAssessment(
        regime=regime,
        confidence=float(confidence),
        risk_score=float(risk_score),
        metrics=payload,
    )


def build_demo_ai_governor_snapshot() -> dict[str, object]:
    """Zwraca snapshot zasilany przykładowymi decyzjami AI Governora."""

    governor = AutoTraderAIGovernor(history_limit=12)
    samples: Iterable[tuple[MarketRegime, float, dict[str, float], dict[str, float], float]] = (
        (
            MarketRegime.TREND,
            0.42,
            {"risk_score": 0.42, "guardrail_active": 0.0},
            {"cycle_latency_p95_ms": 1480.0, "cycle_latency_p50_ms": 920.0},
            9.5,
        ),
        (
            MarketRegime.MEAN_REVERSION,
            0.63,
            {"risk_score": 0.63, "guardrail_active": 1.0},
            {"cycle_latency_p95_ms": 2200.0, "cycle_latency_p50_ms": 1210.0},
            17.0,
        ),
        (
            MarketRegime.DAILY,
            0.37,
            {"risk_score": 0.37, "cooldown_active": 0.0},
            {"cycle_latency_p95_ms": 980.0, "cycle_latency_p50_ms": 640.0},
            6.5,
        ),
        (
            MarketRegime.TREND,
            0.51,
            {"risk_score": 0.51, "guardrail_active": 0.0},
            {"cycle_latency_p95_ms": 1825.0, "cycle_latency_p50_ms": 1030.0},
            12.0,
        ),
    )
    last_risk_metrics: dict[str, float] = {}
    last_cycle_metrics: dict[str, float] = {}
    for regime, risk_score, risk_metrics, cycle_metrics, cost in samples:
        assessment = _build_assessment(regime, risk_score=risk_score, metrics=risk_metrics)
        governor.update_context(
            assessment=assessment,
            risk_metrics=risk_metrics,
            cycle_metrics=cycle_metrics,
            transaction_cost_bps=cost,
        )
        last_risk_metrics = dict(risk_metrics)
        last_cycle_metrics = dict(cycle_metrics)

    snapshot = governor.snapshot()
    telemetry = {
        "riskMetrics": last_risk_metrics,
        "cycleMetrics": last_cycle_metrics,
    }
    snapshot.setdefault("telemetry", telemetry)
    return snapshot


__all__ = ["build_demo_ai_governor_snapshot"]
