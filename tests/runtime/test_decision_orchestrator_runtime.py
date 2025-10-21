from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Deque, Mapping, Sequence

import pytest

from bot_core.decision import DecisionEvaluation
from bot_core.runtime.pipeline import DecisionAwareSignalSink, InMemoryStrategySignalSink
from bot_core.strategies.base import StrategySignal


class _StubDecisionOrchestrator:
    def __init__(self, responses: Sequence[Mapping[str, object]]) -> None:
        self._responses: Deque[Mapping[str, object]] = deque(responses)
        self.calls: list[Mapping[str, object]] = []

    def evaluate_candidate(self, candidate, risk_snapshot):  # pragma: no cover - prosty stub
        if not self._responses:
            raise AssertionError("Brak przygotowanych odpowiedzi orchestratora")
        response = dict(self._responses.popleft())
        evaluation = DecisionEvaluation(
            candidate=candidate,
            accepted=bool(response.get("accepted", False)),
            cost_bps=response.get("cost_bps"),
            net_edge_bps=response.get("net_edge_bps"),
            reasons=tuple(response.get("reasons", ())),
            risk_flags=tuple(response.get("risk_flags", ())),
            stress_failures=tuple(response.get("stress_failures", ())),
            model_expected_return_bps=response.get("model_expected_return_bps"),
            model_success_probability=response.get("model_success_probability"),
            model_name=response.get("model_name"),
            model_selection=response.get("model_selection"),
            thresholds_snapshot=response.get("thresholds_snapshot"),
        )
        self.calls.append({"candidate": candidate, "snapshot": dict(risk_snapshot)})
        return evaluation


class _StubRiskEngine:
    def __init__(self, snapshot: Mapping[str, object] | None = None) -> None:
        self._snapshot = dict(snapshot or {})
        self.requested_profiles: list[str] = []

    def snapshot_state(self, profile: str) -> Mapping[str, object]:  # pragma: no cover - prosty stub
        self.requested_profiles.append(profile)
        return dict(self._snapshot)


def test_decision_orchestrator_runtime_filters_and_summarizes() -> None:
    orchestrator = _StubDecisionOrchestrator(
        (
            {
                "accepted": True,
                "net_edge_bps": 14.0,
                "cost_bps": 1.25,
                "model_name": "xgboost-v2",
                "model_success_probability": 0.72,
                "model_expected_return_bps": 18.0,
                "thresholds_snapshot": {"min_probability": 0.6},
                "risk_flags": ("limits_ok",),
            },
            {
                "accepted": False,
                "net_edge_bps": 4.0,
                "cost_bps": 2.1,
                "model_name": "xgboost-v2",
                "model_success_probability": 0.58,
                "model_expected_return_bps": 9.0,
                "thresholds_snapshot": {"min_probability": 0.61},
                "reasons": ("probability_below_threshold",),
                "risk_flags": ("limit_usage",),
                "stress_failures": ("stress_drawdown",),
            },
        )
    )
    risk_engine = _StubRiskEngine({"equity": 100_000, "positions": {"BTCUSDT": {"notional": 5_000}}})
    base_sink = InMemoryStrategySignalSink()
    sink = DecisionAwareSignalSink(
        base_sink=base_sink,
        orchestrator=orchestrator,
        risk_engine=risk_engine,
        default_notional=2_500.0,
        environment="paper",
        exchange="BINANCE",
        min_probability=0.6,
        journal=None,
        evaluation_history_limit=64,
    )

    timestamp = datetime(2024, 1, 1, 12, 5, tzinfo=timezone.utc)
    signals = (
        StrategySignal(
            symbol="BTCUSDT",
            side="BUY",
            confidence=0.82,
            metadata={
                "expected_probability": 0.72,
                "expected_return_bps": 18.0,
                "generated_at": "2024-01-01T11:59:00Z",
                "ai_manager": {"expected_return_bps": 18.0, "success_probability": 0.72},
            },
        ),
        StrategySignal(
            symbol="ETHUSDT",
            side="SELL",
            confidence=0.66,
            metadata={
                "expected_probability": 0.68,
                "expected_return_bps": 10.0,
                "generated_at": "2024-01-01T11:59:30Z",
                "ai_manager": {"expected_return_bps": 10.0, "success_probability": 0.68},
            },
        ),
    )

    sink.submit(
        strategy_name="trend-d1",
        schedule_name="trend-d1",
        risk_profile="balanced",
        timestamp=timestamp,
        signals=signals,
    )

    records = sink.export()
    assert len(records) == 1
    schedule_name, accepted_signals = records[0]
    assert schedule_name == "trend-d1"
    assert [signal.symbol for signal in accepted_signals] == ["BTCUSDT"]
    assert risk_engine.requested_profiles == ["balanced"]
    assert len(orchestrator.calls) == 2
    assert orchestrator.calls[0]["candidate"].symbol == "BTCUSDT"
    assert orchestrator.calls[1]["candidate"].symbol == "ETHUSDT"

    history = sink.evaluation_history(include_candidates=True)
    assert len(history) == 2
    first, second = history
    assert first["accepted"] is True
    assert first["candidate"]["symbol"] == "BTCUSDT"
    assert second["accepted"] is False
    assert second["candidate"]["symbol"] == "ETHUSDT"
    assert second["thresholds"]["min_probability"] == pytest.approx(0.61)

    latest_only = sink.evaluation_history(limit=1, include_candidates=True)
    assert len(latest_only) == 1
    assert latest_only[0]["candidate"]["symbol"] == "ETHUSDT"

    summary = sink.evaluation_summary()
    assert summary["total"] == 2
    assert summary["accepted"] == 1
    assert summary["rejected"] == 1
    assert summary["latest_status"] == "rejected"
    assert summary["current_acceptance_streak"] == 0
    assert summary["current_rejection_streak"] == 1
    assert summary["model_usage"]["xgboost-v2"] == 2
    assert summary["unique_models"] == 1
    assert summary["models_with_accepts"] == 1
    assert summary["model_breakdown"]["xgboost-v2"]["accepted"] == 1
    assert summary["rejection_reasons"]["probability_below_threshold"] == 1
    assert summary["unique_rejection_reasons"] == 1
    assert summary["latest_thresholds"]["min_probability"] == pytest.approx(0.61)
    assert summary["history_start_generated_at"] == "2024-01-01T11:59:00Z"
    assert summary["latest_candidate"]["symbol"] == "ETHUSDT"
    assert summary["latest_candidate"]["expected_value_bps"] == pytest.approx(6.8)
    assert summary["latest_expected_value_bps"] == pytest.approx(6.8)
    assert summary["latest_expected_value_minus_cost_bps"] == pytest.approx(4.7)
    assert summary["latest_net_edge_bps"] == pytest.approx(4.0)
    assert summary["latest_cost_bps"] == pytest.approx(2.1)
    assert summary["latest_expected_probability"] == pytest.approx(0.68)
    assert summary["latest_expected_return_bps"] == pytest.approx(10.0)
    assert summary["latest_notional"] == pytest.approx(2_500.0)
    assert summary["latest_model_expected_value_bps"] == pytest.approx(5.22)
    assert summary["latest_model_expected_value_minus_cost_bps"] == pytest.approx(3.12)
    assert summary["latest_model_expected_return_bps"] == pytest.approx(9.0)
    assert summary["latest_model_success_probability"] == pytest.approx(0.58)
    assert summary["avg_expected_value_bps"] == pytest.approx(9.88)
    assert summary["sum_expected_return_bps"] == pytest.approx(28.0)
    assert summary["sum_expected_value_bps"] == pytest.approx(19.76)
    assert summary["median_expected_value_bps"] == pytest.approx(9.88)
    assert summary["min_expected_value_bps"] == pytest.approx(6.8)
    assert summary["max_expected_value_bps"] == pytest.approx(12.96)
    assert summary["avg_expected_value_minus_cost_bps"] == pytest.approx(8.205)
    assert summary["sum_expected_value_minus_cost_bps"] == pytest.approx(16.41)
    assert summary["accepted_avg_net_edge_bps"] == pytest.approx(14.0)
    assert summary["accepted_median_net_edge_bps"] == pytest.approx(14.0)
    assert summary["accepted_p10_net_edge_bps"] == pytest.approx(14.0)
    assert summary["accepted_p90_net_edge_bps"] == pytest.approx(14.0)
    assert summary["accepted_min_net_edge_bps"] == pytest.approx(14.0)
    assert summary["accepted_max_net_edge_bps"] == pytest.approx(14.0)
    assert summary["accepted_std_net_edge_bps"] == pytest.approx(0.0)
    assert summary["accepted_sum_net_edge_bps"] == pytest.approx(14.0)
    assert summary["accepted_net_edge_bps_count"] == 1
    assert summary["rejected_avg_net_edge_bps"] == pytest.approx(4.0)
    assert summary["rejected_median_net_edge_bps"] == pytest.approx(4.0)
    assert summary["rejected_p10_net_edge_bps"] == pytest.approx(4.0)
    assert summary["rejected_p90_net_edge_bps"] == pytest.approx(4.0)
    assert summary["rejected_min_net_edge_bps"] == pytest.approx(4.0)
    assert summary["rejected_max_net_edge_bps"] == pytest.approx(4.0)
    assert summary["rejected_std_net_edge_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_net_edge_bps"] == pytest.approx(4.0)
    assert summary["rejected_net_edge_bps_count"] == 1
    assert summary["accepted_avg_cost_bps"] == pytest.approx(1.25)
    assert summary["accepted_median_cost_bps"] == pytest.approx(1.25)
    assert summary["accepted_p10_cost_bps"] == pytest.approx(1.25)
    assert summary["accepted_p90_cost_bps"] == pytest.approx(1.25)
    assert summary["accepted_min_cost_bps"] == pytest.approx(1.25)
    assert summary["accepted_max_cost_bps"] == pytest.approx(1.25)
    assert summary["accepted_std_cost_bps"] == pytest.approx(0.0)
    assert summary["accepted_sum_cost_bps"] == pytest.approx(1.25)
    assert summary["accepted_cost_bps_count"] == 1
    assert summary["rejected_avg_cost_bps"] == pytest.approx(2.1)
    assert summary["rejected_median_cost_bps"] == pytest.approx(2.1)
    assert summary["rejected_p10_cost_bps"] == pytest.approx(2.1)
    assert summary["rejected_p90_cost_bps"] == pytest.approx(2.1)
    assert summary["rejected_min_cost_bps"] == pytest.approx(2.1)
    assert summary["rejected_max_cost_bps"] == pytest.approx(2.1)
    assert summary["rejected_std_cost_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_cost_bps"] == pytest.approx(2.1)
    assert summary["rejected_cost_bps_count"] == 1
    assert summary["accepted_avg_expected_probability"] == pytest.approx(0.72)
    assert summary["accepted_median_expected_probability"] == pytest.approx(0.72)
    assert summary["accepted_std_expected_probability"] == pytest.approx(0.0)
    assert summary["accepted_expected_probability_count"] == 1
    assert summary["rejected_avg_expected_probability"] == pytest.approx(0.68)
    assert summary["rejected_median_expected_probability"] == pytest.approx(0.68)
    assert summary["rejected_std_expected_probability"] == pytest.approx(0.0)
    assert summary["rejected_expected_probability_count"] == 1
    assert summary["accepted_avg_expected_return_bps"] == pytest.approx(18.0)
    assert summary["accepted_median_expected_return_bps"] == pytest.approx(18.0)
    assert summary["accepted_p90_expected_return_bps"] == pytest.approx(18.0)
    assert summary["accepted_std_expected_return_bps"] == pytest.approx(0.0)
    assert summary["accepted_sum_expected_return_bps"] == pytest.approx(18.0)
    assert summary["accepted_expected_return_bps_count"] == 1
    assert summary["rejected_avg_expected_return_bps"] == pytest.approx(10.0)
    assert summary["rejected_median_expected_return_bps"] == pytest.approx(10.0)
    assert summary["rejected_p90_expected_return_bps"] == pytest.approx(10.0)
    assert summary["rejected_std_expected_return_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_expected_return_bps"] == pytest.approx(10.0)
    assert summary["rejected_expected_return_bps_count"] == 1
    assert summary["accepted_avg_expected_value_bps"] == pytest.approx(12.96)
    assert summary["accepted_median_expected_value_bps"] == pytest.approx(12.96)
    assert summary["accepted_p90_expected_value_bps"] == pytest.approx(12.96)
    assert summary["accepted_std_expected_value_bps"] == pytest.approx(0.0)
    assert summary["accepted_sum_expected_value_bps"] == pytest.approx(12.96)
    assert summary["accepted_expected_value_bps_count"] == 1
    assert summary["rejected_avg_expected_value_bps"] == pytest.approx(6.8)
    assert summary["rejected_median_expected_value_bps"] == pytest.approx(6.8)
    assert summary["rejected_p90_expected_value_bps"] == pytest.approx(6.8)
    assert summary["rejected_std_expected_value_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_expected_value_bps"] == pytest.approx(6.8)
    assert summary["rejected_expected_value_bps_count"] == 1
    assert summary["accepted_avg_expected_value_minus_cost_bps"] == pytest.approx(11.71)
    assert summary["accepted_median_expected_value_minus_cost_bps"] == pytest.approx(11.71)
    assert summary["accepted_p90_expected_value_minus_cost_bps"] == pytest.approx(11.71)
    assert summary["accepted_min_expected_value_minus_cost_bps"] == pytest.approx(11.71)
    assert summary["accepted_max_expected_value_minus_cost_bps"] == pytest.approx(11.71)
    assert summary["accepted_std_expected_value_minus_cost_bps"] == pytest.approx(0.0)
    assert summary["accepted_sum_expected_value_minus_cost_bps"] == pytest.approx(11.71)
    assert summary["accepted_expected_value_minus_cost_bps_count"] == 1
    assert summary["rejected_avg_expected_value_minus_cost_bps"] == pytest.approx(4.7)
    assert summary["rejected_median_expected_value_minus_cost_bps"] == pytest.approx(4.7)
    assert summary["rejected_p90_expected_value_minus_cost_bps"] == pytest.approx(4.7)
    assert summary["rejected_min_expected_value_minus_cost_bps"] == pytest.approx(4.7)
    assert summary["rejected_max_expected_value_minus_cost_bps"] == pytest.approx(4.7)
    assert summary["rejected_std_expected_value_minus_cost_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_expected_value_minus_cost_bps"] == pytest.approx(4.7)
    assert summary["rejected_expected_value_minus_cost_bps_count"] == 1
    assert summary["accepted_avg_notional"] == pytest.approx(2_500.0)
    assert summary["accepted_median_notional"] == pytest.approx(2_500.0)
    assert summary["accepted_p90_notional"] == pytest.approx(2_500.0)
    assert summary["accepted_min_notional"] == pytest.approx(2_500.0)
    assert summary["accepted_max_notional"] == pytest.approx(2_500.0)
    assert summary["accepted_std_notional"] == pytest.approx(0.0)
    assert summary["accepted_sum_notional"] == pytest.approx(2_500.0)
    assert summary["accepted_notional_count"] == 1
    assert summary["rejected_avg_notional"] == pytest.approx(2_500.0)
    assert summary["rejected_sum_notional"] == pytest.approx(2_500.0)
    assert summary["rejected_notional_count"] == 1
    assert summary["accepted_avg_model_success_probability"] == pytest.approx(0.72)
    assert summary["accepted_model_success_probability_count"] == 1
    assert summary["rejected_avg_model_success_probability"] == pytest.approx(0.58)
    assert summary["rejected_model_success_probability_count"] == 1
    assert summary["accepted_avg_model_expected_return_bps"] == pytest.approx(18.0)
    assert summary["accepted_sum_model_expected_return_bps"] == pytest.approx(18.0)
    assert summary["accepted_model_expected_return_bps_count"] == 1
    assert summary["rejected_avg_model_expected_return_bps"] == pytest.approx(9.0)
    assert summary["rejected_sum_model_expected_return_bps"] == pytest.approx(9.0)
    assert summary["rejected_model_expected_return_bps_count"] == 1
    assert summary["accepted_avg_model_expected_value_bps"] == pytest.approx(12.96)
    assert summary["accepted_sum_model_expected_value_bps"] == pytest.approx(12.96)
    assert summary["accepted_model_expected_value_bps_count"] == 1
    assert summary["rejected_avg_model_expected_value_bps"] == pytest.approx(5.22)
    assert summary["rejected_sum_model_expected_value_bps"] == pytest.approx(5.22)
    assert summary["rejected_model_expected_value_bps_count"] == 1
    assert summary["accepted_avg_model_expected_value_minus_cost_bps"] == pytest.approx(
        11.71
    )
    assert summary["accepted_sum_model_expected_value_minus_cost_bps"] == pytest.approx(
        11.71
    )
    assert summary["accepted_model_expected_value_minus_cost_bps_count"] == 1
    assert summary["rejected_avg_model_expected_value_minus_cost_bps"] == pytest.approx(
        3.12
    )
    assert summary["rejected_sum_model_expected_value_minus_cost_bps"] == pytest.approx(
        3.12
    )
    assert summary["rejected_model_expected_value_minus_cost_bps_count"] == 1
    assert summary["median_expected_value_minus_cost_bps"] == pytest.approx(8.205)
    assert summary["min_expected_value_minus_cost_bps"] == pytest.approx(4.7)
    assert summary["max_expected_value_minus_cost_bps"] == pytest.approx(11.71)
    assert summary["avg_model_expected_value_bps"] == pytest.approx(9.09)
    assert summary["sum_model_expected_return_bps"] == pytest.approx(27.0)
    assert summary["sum_model_expected_value_bps"] == pytest.approx(18.18)
    assert summary["median_model_expected_value_bps"] == pytest.approx(9.09)
    assert summary["min_model_expected_value_bps"] == pytest.approx(5.22)
    assert summary["max_model_expected_value_bps"] == pytest.approx(12.96)
    assert summary["avg_model_expected_value_minus_cost_bps"] == pytest.approx(7.415)
    assert summary["sum_model_expected_value_minus_cost_bps"] == pytest.approx(14.83)
    assert summary["median_model_expected_value_minus_cost_bps"] == pytest.approx(7.415)
    assert summary["min_model_expected_value_minus_cost_bps"] == pytest.approx(3.12)
    assert summary["max_model_expected_value_minus_cost_bps"] == pytest.approx(11.71)
    assert summary["std_net_edge_bps"] == pytest.approx(5.0)
    assert summary["std_cost_bps"] == pytest.approx(0.425)
    assert summary["std_expected_probability"] == pytest.approx(0.02)
    assert summary["std_expected_return_bps"] == pytest.approx(4.0)
    assert summary["std_expected_value_bps"] == pytest.approx(3.08)
    assert summary["std_expected_value_minus_cost_bps"] == pytest.approx(3.505)
    assert summary["std_model_success_probability"] == pytest.approx(0.07)
    assert summary["std_model_expected_return_bps"] == pytest.approx(4.5)
    assert summary["std_model_expected_value_bps"] == pytest.approx(3.87)
    assert summary["std_model_expected_value_minus_cost_bps"] == pytest.approx(4.295)

