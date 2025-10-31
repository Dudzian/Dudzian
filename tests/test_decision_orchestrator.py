from __future__ import annotations

import logging
from pathlib import Path


import pytest

from bot_core.config.models import (
    DecisionEngineConfig,
    DecisionOrchestratorThresholds,
    DecisionStressTestConfig,
)
from bot_core.decision.models import DecisionCandidate, RiskSnapshot
from bot_core.decision.orchestrator import DecisionOrchestrator


def _make_config(**overrides: object) -> DecisionEngineConfig:
    thresholds = DecisionOrchestratorThresholds(
        max_cost_bps=12.0,
        min_net_edge_bps=3.0,
        max_daily_loss_pct=0.02,
        max_drawdown_pct=0.08,
        max_position_ratio=0.35,
        max_open_positions=6,
        max_latency_ms=240.0,
        max_trade_notional=30000.0,
    )
    stress = DecisionStressTestConfig(cost_shock_bps=2.0, latency_spike_ms=40.0, slippage_multiplier=1.2)
    return DecisionEngineConfig(
        orchestrator=thresholds,
        profile_overrides={},
        stress_tests=stress,
        min_probability=overrides.get("min_probability", 0.5),
        require_cost_data=overrides.get("require_cost_data", False),
        penalty_cost_bps=overrides.get("penalty_cost_bps", 0.0),
    )


def _snapshot() -> RiskSnapshot:
    return RiskSnapshot(
        profile="balanced",
        start_of_day_equity=100_000.0,
        daily_realized_pnl=0.0,
        peak_equity=105_000.0,
        last_equity=102_000.0,
        gross_notional=25_000.0,
        active_positions=3,
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
    )


def test_accepts_candidate_with_positive_edge() -> None:
    orchestrator = DecisionOrchestrator(_make_config())
    candidate = DecisionCandidate(
        strategy="mean_reversion_alpha",
        action="enter",
        risk_profile="balanced",
        symbol="ADAUSDT",
        notional=5_000.0,
        expected_return_bps=12.0,
        expected_probability=0.8,
        cost_bps_override=2.0,
        latency_ms=180.0,
    )

    evaluation = orchestrator.evaluate_candidate(candidate, _snapshot())

    assert evaluation.accepted is True
    assert evaluation.net_edge_bps is not None and evaluation.net_edge_bps > 3.0
    assert evaluation.reasons == ()
    assert evaluation.stress_failures == ()
    assert evaluation.thresholds_snapshot is not None
    assert evaluation.thresholds_snapshot["min_probability"] == pytest.approx(0.5)


def test_rejects_when_cost_exceeds_limit() -> None:
    orchestrator = DecisionOrchestrator(_make_config())
    candidate = DecisionCandidate(
        strategy="mean_reversion_alpha",
        action="enter",
        risk_profile="balanced",
        symbol="ADAUSDT",
        notional=5_000.0,
        expected_return_bps=9.0,
        expected_probability=0.7,
        cost_bps_override=20.0,
        latency_ms=120.0,
    )

    evaluation = orchestrator.evaluate_candidate(candidate, _snapshot())

    assert evaluation.accepted is False
    assert any("koszt" in reason for reason in evaluation.reasons)
    assert evaluation.thresholds_snapshot is not None
    assert evaluation.thresholds_snapshot["max_cost_bps"] == pytest.approx(12.0)


def test_rejects_on_risk_limits() -> None:
    orchestrator = DecisionOrchestrator(_make_config())
    snapshot = RiskSnapshot(
        profile="balanced",
        start_of_day_equity=100_000.0,
        daily_realized_pnl=-3_000.0,
        peak_equity=105_000.0,
        last_equity=96_000.0,
        gross_notional=38_000.0,
        active_positions=6,
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT", "DOGEUSDT"),
    )
    candidate = DecisionCandidate(
        strategy="mean_reversion_alpha",
        action="enter",
        risk_profile="balanced",
        symbol="TRXUSDT",
        notional=8_000.0,
        expected_return_bps=10.0,
        expected_probability=0.9,
        cost_bps_override=2.0,
        latency_ms=150.0,
    )

    evaluation = orchestrator.evaluate_candidate(candidate, snapshot)

    assert evaluation.accepted is False
    assert evaluation.thresholds_snapshot is not None


def test_logs_threshold_snapshot_on_rejection(caplog: pytest.LogCaptureFixture) -> None:
    orchestrator = DecisionOrchestrator(_make_config())
    candidate = DecisionCandidate(
        strategy="mean_reversion_alpha",
        action="enter",
        risk_profile="balanced",
        symbol="ADAUSDT",
        notional=5_000.0,
        expected_return_bps=4.0,
        expected_probability=0.6,
        cost_bps_override=25.0,
        latency_ms=200.0,
    )

    with caplog.at_level(logging.INFO):
        orchestrator.evaluate_candidate(candidate, _snapshot())

    record = next(
        (entry for entry in caplog.records if "DecisionOrchestrator rejected candidate" in entry.message),
        None,
    )
    assert record is not None
    assert "thresholds" in record.message


def test_stress_failure_blocks_candidate() -> None:
    orchestrator = DecisionOrchestrator(_make_config())
    candidate = DecisionCandidate(
        strategy="mean_reversion_alpha",
        action="enter",
        risk_profile="balanced",
        symbol="ADAUSDT",
        notional=5_000.0,
        expected_return_bps=8.0,
        expected_probability=0.6,
        cost_bps_override=5.0,
        latency_ms=220.0,
    )

    evaluation = orchestrator.evaluate_candidate(candidate, _snapshot())

    assert evaluation.accepted is False
    assert evaluation.stress_failures != ()


def test_costs_loaded_from_tco_report_dict() -> None:
    config = _make_config()
    orchestrator = DecisionOrchestrator(config)
    orchestrator.update_costs_from_report(
        {
            "strategies": {
                "mean_reversion_alpha": {
                    "profiles": {
                        "balanced": {"cost_bps": 4.2},
                    },
                    "total": {"cost_bps": 5.0},
                }
            },
            "total": {"cost_bps": 6.1},
        }
    )
    candidate = DecisionCandidate(
        strategy="mean_reversion_alpha",
        action="enter",
        risk_profile="balanced",
        symbol="ADAUSDT",
        notional=4_000.0,
        expected_return_bps=14.0,
        expected_probability=0.85,
        latency_ms=120.0,
    )

    evaluation = orchestrator.evaluate_candidate(candidate, _snapshot())

    assert evaluation.accepted is True
    assert evaluation.cost_bps == 4.2


def test_missing_cost_requires_data_when_configured() -> None:
    config = _make_config(require_cost_data=True)
    orchestrator = DecisionOrchestrator(config)
    candidate = DecisionCandidate(
        strategy="mean_reversion_alpha",
        action="enter",
        risk_profile="balanced",
        symbol="ADAUSDT",
        notional=4_000.0,
        expected_return_bps=10.0,
        expected_probability=0.8,
        latency_ms=120.0,
    )

    evaluation = orchestrator.evaluate_candidate(candidate, _snapshot())

    assert evaluation.accepted is False
    assert any("brak danych" in reason for reason in evaluation.reasons)


def test_penalty_cost_used_when_configured() -> None:
    config = _make_config(penalty_cost_bps=1.5)
    orchestrator = DecisionOrchestrator(config)
    candidate = DecisionCandidate(
        strategy="mean_reversion_alpha",
        action="enter",
        risk_profile="balanced",
        symbol="ADAUSDT",
        notional=4_000.0,
        expected_return_bps=10.0,
        expected_probability=0.8,
        latency_ms=120.0,
    )

    evaluation = orchestrator.evaluate_candidate(candidate, _snapshot())

    assert evaluation.cost_bps == 1.5


def test_evaluate_candidates_handles_missing_snapshot() -> None:
    orchestrator = DecisionOrchestrator(_make_config())
    candidate = DecisionCandidate(
        strategy="mean_reversion_alpha",
        action="enter",
        risk_profile="balanced",
        symbol="ADAUSDT",
        notional=4_000.0,
        expected_return_bps=10.0,
        expected_probability=0.8,
        latency_ms=120.0,
    )

    evaluations = orchestrator.evaluate_candidates([candidate], risk_snapshots={})
    assert evaluations[0].accepted is False
    assert any("snapshot" in reason for reason in evaluations[0].reasons)
