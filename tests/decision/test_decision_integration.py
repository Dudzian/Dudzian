from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bot_core.ai import ModelScore
from bot_core.config.models import DecisionEngineConfig, DecisionOrchestratorThresholds
from bot_core.decision.models import DecisionCandidate, RiskSnapshot
from bot_core.decision.orchestrator import DecisionOrchestrator


class StubInference:
    def __init__(self, score: ModelScore) -> None:
        self._score = score
        self.is_ready = True
        self.called_with: dict[str, float] | None = None

    def score(self, features: dict[str, float]) -> ModelScore:
        self.called_with = features
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
    expected_value = 14.0 * 0.75
    assert evaluation.net_edge_bps == pytest.approx(expected_value - 2.0)
    assert evaluation.accepted is True
