from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bot_core.ai import ModelScore
from bot_core.config.models import DecisionEngineConfig, DecisionOrchestratorThresholds
from bot_core.decision.models import (
    DecisionCandidate,
    ModelSelectionMetadata,
    RiskSnapshot,
)
from bot_core.decision.orchestrator import DecisionOrchestrator


class StubInference:
    def __init__(self, score: ModelScore) -> None:
        self._score = score
        self.is_ready = True
        self.called_with: dict[str, float] | None = None
        self.calls: int = 0

    def score(self, features: dict[str, float]) -> ModelScore:
        self.called_with = features
        self.calls += 1
        return self._score


def _config() -> DecisionEngineConfig:
    thresholds = DecisionOrchestratorThresholds(
        max_cost_bps=15.0,
        min_net_edge_bps=3.0,
        max_daily_loss_pct=0.03,
        max_drawdown_pct=0.12,
        max_position_ratio=0.5,
        max_open_positions=6,
        max_latency_ms=250.0,
        max_trade_notional=50_000.0,
    )
    return DecisionEngineConfig(
        orchestrator=thresholds,
        profile_overrides={},
        stress_tests=None,
        min_probability=0.5,
        require_cost_data=False,
        penalty_cost_bps=0.0,
    )


def _snapshot() -> RiskSnapshot:
    return RiskSnapshot(
        profile="balanced",
        start_of_day_equity=200_000.0,
        daily_realized_pnl=0.0,
        peak_equity=205_000.0,
        last_equity=202_500.0,
        gross_notional=50_000.0,
        active_positions=2,
        symbols=("BTCUSDT", "ETHUSDT"),
    )


def test_orchestrator_uses_inference_score() -> None:
    stub = StubInference(ModelScore(expected_return_bps=14.0, success_probability=0.75))
    orchestrator = DecisionOrchestrator(_config(), inference=stub)

    candidate = DecisionCandidate(
        strategy="ai_trend",
        action="enter",
        risk_profile="balanced",
        symbol="SOLUSDT",
        notional=10_000.0,
        expected_return_bps=0.0,
        expected_probability=0.0,
        cost_bps_override=2.0,
        metadata={"model_features": {"momentum": 1.0, "ret_volatility": 0.2}},
    )

    evaluation = orchestrator.evaluate_candidate(candidate, _snapshot())

    assert stub.called_with == {"momentum": 1.0, "ret_volatility": 0.2}
    assert evaluation.model_success_probability == pytest.approx(0.75)
    assert evaluation.model_expected_return_bps == pytest.approx(14.0)
    assert evaluation.model_name == "__default__"
    expected_value = 14.0 * 0.75
    assert evaluation.net_edge_bps == pytest.approx(expected_value - 2.0)
    assert evaluation.accepted is True
    assert orchestrator.model_usage_statistics() == {"__default__": 1}
    assert isinstance(evaluation.model_selection, ModelSelectionMetadata)
    assert evaluation.model_selection.selected == "__default__"
    default_detail = evaluation.model_selection.find("__default__")
    assert default_detail is not None
    assert default_detail.available is True
    assert default_detail.reason is None


def test_orchestrator_switches_models_based_on_history() -> None:
    slow = StubInference(ModelScore(expected_return_bps=4.0, success_probability=0.55))
    fast = StubInference(ModelScore(expected_return_bps=9.0, success_probability=0.68))
    orchestrator = DecisionOrchestrator(_config())
    orchestrator.attach_named_inference("slow", slow, set_default=True)
    orchestrator.attach_named_inference("fast", fast)
    orchestrator.update_model_performance(
        "slow",
        {"mae": 8.0, "directional_accuracy": 0.45},
        strategy="ai_trend",
        risk_profile="balanced",
    )
    orchestrator.update_model_performance(
        "fast",
        {"mae": 3.0, "directional_accuracy": 0.72},
        strategy="ai_trend",
        risk_profile="balanced",
    )

    candidate = DecisionCandidate(
        strategy="ai_trend",
        action="enter",
        risk_profile="balanced",
        symbol="SOLUSDT",
        notional=12_000.0,
        expected_return_bps=0.0,
        expected_probability=0.0,
        metadata={"model_features": {"momentum": 1.2, "ret_volatility": 0.15}},
    )

    evaluation = orchestrator.evaluate_candidate(candidate, _snapshot())

    assert fast.called_with == {"momentum": 1.2, "ret_volatility": 0.15}
    assert slow.called_with is None
    assert evaluation.model_expected_return_bps == pytest.approx(9.0)
    assert evaluation.model_success_probability == pytest.approx(0.68)
    assert evaluation.model_name == "fast"
    assert evaluation.accepted is True
    assert orchestrator.model_usage_statistics() == {"fast": 1}
    assert orchestrator.model_usage_statistics(reset=True) == {"fast": 1}
    assert orchestrator.model_usage_statistics() == {}
    assert evaluation.model_selection is not None
    assert evaluation.model_selection.selected == "fast"
    fast_detail = evaluation.model_selection.find("fast")
    assert fast_detail is not None
    assert fast_detail.reason is None


def test_orchestrator_prefers_recent_performance() -> None:
    base_time = datetime(2024, 5, 1, 12, tzinfo=timezone.utc)
    
    def _clock() -> datetime:
        return base_time

    stale = StubInference(ModelScore(expected_return_bps=12.0, success_probability=0.82))
    fresh = StubInference(ModelScore(expected_return_bps=7.0, success_probability=0.64))
    orchestrator = DecisionOrchestrator(
        _config(),
        performance_half_life_hours=1.0,
        clock=_clock,
    )
    orchestrator.attach_named_inference("stale", stale, set_default=True)
    orchestrator.attach_named_inference("fresh", fresh)
    orchestrator.update_model_performance(
        "stale",
        {"mae": 2.0, "directional_accuracy": 0.9},
        strategy="ai_trend",
        risk_profile="balanced",
        timestamp=base_time - timedelta(hours=4),
    )
    orchestrator.update_model_performance(
        "fresh",
        {"mae": 3.0, "directional_accuracy": 0.7},
        strategy="ai_trend",
        risk_profile="balanced",
        timestamp=base_time,
    )

    candidate = DecisionCandidate(
        strategy="ai_trend",
        action="enter",
        risk_profile="balanced",
        symbol="SOLUSDT",
        notional=15_000.0,
        expected_return_bps=0.0,
        expected_probability=0.0,
        metadata={"model_features": {"momentum": 0.7, "ret_volatility": 0.25}},
    )

    evaluation = orchestrator.evaluate_candidate(candidate, _snapshot())

    assert fresh.called_with == {"momentum": 0.7, "ret_volatility": 0.25}
    assert stale.called_with is None
    assert evaluation.model_expected_return_bps == pytest.approx(7.0)
    assert evaluation.model_success_probability == pytest.approx(0.64)
    assert evaluation.model_name == "fresh"

    snapshot = orchestrator.performance_snapshot("ai_trend", "balanced")
    assert "fresh" in snapshot
    assert "stale" in snapshot
    assert snapshot["stale"].weight < snapshot["fresh"].weight


def test_orchestrator_prunes_stale_performance() -> None:
    base_time = datetime(2024, 5, 1, 8, tzinfo=timezone.utc)

    def _clock() -> datetime:
        return base_time

    orchestrator = DecisionOrchestrator(_config(), clock=_clock)
    orchestrator.update_model_performance(
        "alpha",
        {"mae": 4.0, "directional_accuracy": 0.6},
        strategy="ai_trend",
        risk_profile="balanced",
        timestamp=base_time - timedelta(days=3),
    )

    orchestrator.prune_model_performance(older_than=timedelta(days=1))
    snapshot = orchestrator.performance_snapshot("ai_trend", "balanced")
    assert snapshot == {}


def test_orchestrator_reports_missing_features_in_trace() -> None:
    stub = StubInference(ModelScore(expected_return_bps=6.0, success_probability=0.55))
    orchestrator = DecisionOrchestrator(_config(), inference=stub)

    candidate = DecisionCandidate(
        strategy="ai_trend",
        action="enter",
        risk_profile="balanced",
        symbol="SOLUSDT",
        notional=9_000.0,
        expected_return_bps=10.0,
        expected_probability=0.7,
        metadata={},
    )

    evaluation = orchestrator.evaluate_candidate(candidate, _snapshot())

    assert evaluation.model_name is None
    assert evaluation.model_selection is not None
    assert evaluation.model_selection.selected == "__default__"
    default_detail = evaluation.model_selection.find("__default__")
    assert default_detail is not None
    assert default_detail.available is True
    assert default_detail.reason == "brak cech numerycznych"


def test_orchestrator_traces_unavailable_models() -> None:
    class UnreadyInference(StubInference):
        def __init__(self, score: ModelScore) -> None:
            super().__init__(score)
            self.is_ready = False

    slow = UnreadyInference(
        ModelScore(expected_return_bps=12.0, success_probability=0.9)
    )
    fast = StubInference(ModelScore(expected_return_bps=5.0, success_probability=0.6))
    orchestrator = DecisionOrchestrator(_config())
    orchestrator.attach_named_inference("slow", slow, set_default=True)
    orchestrator.attach_named_inference("fast", fast)
    orchestrator.update_model_performance(
        "slow",
        {"mae": 1.0, "directional_accuracy": 0.9},
        strategy="ai_trend",
        risk_profile="balanced",
    )
    orchestrator.update_model_performance(
        "fast",
        {"mae": 3.0, "directional_accuracy": 0.6},
        strategy="ai_trend",
        risk_profile="balanced",
    )

    candidate = DecisionCandidate(
        strategy="ai_trend",
        action="enter",
        risk_profile="balanced",
        symbol="SOLUSDT",
        notional=8_000.0,
        expected_return_bps=0.0,
        expected_probability=0.0,
        metadata={"model_features": {"momentum": 1.1}},
    )

    evaluation = orchestrator.evaluate_candidate(candidate, _snapshot())

    assert evaluation.model_name == "fast"
    assert evaluation.model_selection is not None
    assert evaluation.model_selection.selected == "fast"
    slow_detail = evaluation.model_selection.find("slow")
    assert slow_detail is not None
    assert slow_detail.available is False
    assert slow_detail.reason == "model niedostępny"
    fast_detail = evaluation.model_selection.find("fast")
    assert fast_detail is not None
    assert fast_detail.available is True
    assert fast_detail.reason is None
