from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bot_core.runtime.journal import JsonlTradingDecisionJournal, TradingDecisionEvent

from scripts.export_decision_engine_summary import main as export_summary


def _record_event(
    journal: JsonlTradingDecisionJournal,
    *,
    timestamp: datetime,
    status: str,
    symbol: str,
    strategy: str,
    schedule: str,
    side: str,
    metadata: dict[str, str],
) -> None:
    latency_raw = metadata.pop("latency_ms", None)
    if isinstance(latency_raw, str):
        latency_value = float(latency_raw)
    else:
        latency_value = latency_raw

    event = TradingDecisionEvent(
        event_type="decision_evaluation",
        timestamp=timestamp,
        environment="paper",
        portfolio="paper-01",
        risk_profile="balanced",
        symbol=symbol,
        side=side,
        side="BUY",
        strategy=strategy,
        schedule=schedule,
        status=status,
        latency_ms=latency_value,
        metadata=metadata,
    )
    journal.record(event)


def test_export_decision_engine_summary(tmp_path: Path) -> None:
    journal = JsonlTradingDecisionJournal(directory=tmp_path)
    base_time = datetime(2024, 5, 1, tzinfo=timezone.utc)

    _record_event(
        journal,
        timestamp=base_time,
        status="accepted",
        symbol="BTC/USDT",
        strategy="daily",
        schedule="d1",
        side="BUY",
        metadata={
            "expected_probability": "0.68",
            "expected_return_bps": "12.0",
            "notional": "1000",
            "cost_bps": "1.2",
            "net_edge_bps": "6.5",
            "model_success_probability": "0.72",
            "model_expected_return_bps": "8.5",
            "model_name": "gbm_v5",
            "decision_thresholds": json.dumps(
                {
                    "min_probability": 0.6,
                    "max_cost_bps": 18.0,
                    "min_net_edge_bps": 5.0,
                    "max_latency_ms": 60.0,
                    "max_trade_notional": 1_200.0,
                }
            ),
            "generated_at": "2024-05-01T00:00:00Z",
            "latency_ms": "42.0",
            "risk_flags": "volatility_spike",
            "decision_thresholds": json.dumps({"min_probability": 0.6, "max_cost_bps": 18.0}),
            "generated_at": "2024-05-01T00:00:00Z",
            "latency_ms": "42.0",
        },
    )

    _record_event(
        journal,
        timestamp=base_time.replace(day=2),
        status="rejected",
        symbol="ETH/USDT",
        strategy="daily",
        schedule="d1",
        side="SELL",
        metadata={
            "expected_probability": "0.4",
            "expected_return_bps": "3.0",
            "notional": "500",
            "cost_bps": "2.5",
            "net_edge_bps": "1.0",
            "model_success_probability": "0.41",
            "model_expected_return_bps": "2.5",
            "model_name": "gbm_v6",
            "decision_thresholds": json.dumps(
                {
                    "min_probability": 0.65,
                    "max_cost_bps": 2.0,
                    "min_net_edge_bps": 2.0,
                    "max_latency_ms": 50.0,
                    "max_trade_notional": 450.0,
                }
            ),
            "decision_reasons": "too_costly",
            "latency_ms": "55.0",
            "risk_flags": "liquidity",
            "stress_failures": "stress_liquidity",
            "decision_thresholds": json.dumps({"min_probability": 0.65, "max_cost_bps": 12.0}),
            "decision_reasons": "too_costly",
            "latency_ms": "55.0",
        },
    )

    output_path = tmp_path / "summary.json"
    exit_code = export_summary(
        [
            "--ledger",
            str(tmp_path),
            "--output",
            str(output_path),
            "--environment",
            "paper",
            "--portfolio",
            "paper-01",
            "--history-limit",
            "10",
            "--include-history",
            "--history-size",
            "2",
            "--pretty",
        ]
    )
    assert exit_code == 0

    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["type"] == "decision_engine_summary"
    assert summary["total"] == 2
    assert summary["accepted"] == 1
    assert summary["rejected"] == 1
    assert summary["filters"]["portfolio"] == "paper-01"
    assert summary["rejection_reasons"]["too_costly"] == 1
    assert summary["unique_rejection_reasons"] == 1
    assert summary["latest_model"] == "gbm_v6"
    assert summary["history_limit"] == 10
    assert summary["model_usage"]["gbm_v6"] == 1
    assert summary["unique_models"] == 2
    assert summary["models_with_accepts"] == 1
    model_breakdown = summary["model_breakdown"]
    assert model_breakdown["gbm_v5"]["accepted"] == 1
    gbm_v5_metrics = model_breakdown["gbm_v5"]["metrics"]
    assert gbm_v5_metrics["net_edge_bps"]["accepted_sum"] == pytest.approx(6.5)
    assert gbm_v5_metrics["expected_value_minus_cost_bps"]["accepted_sum"] == pytest.approx(6.96)
    assert gbm_v5_metrics["expected_value_bps"]["total_sum"] == pytest.approx(8.16)
    assert gbm_v5_metrics["notional"]["accepted_sum"] == pytest.approx(1000)
    assert gbm_v5_metrics["latency_ms"]["accepted_avg"] == pytest.approx(42.0)

    assert model_breakdown["gbm_v6"]["rejected"] == 1
    gbm_v6_metrics = model_breakdown["gbm_v6"]["metrics"]
    assert gbm_v6_metrics["net_edge_bps"]["rejected_sum"] == pytest.approx(1.0)
    assert gbm_v6_metrics["expected_value_minus_cost_bps"]["rejected_sum"] == pytest.approx(-1.3)
    assert gbm_v6_metrics["expected_value_bps"]["total_sum"] == pytest.approx(1.2)
    assert gbm_v6_metrics["cost_bps"]["total_sum"] == pytest.approx(2.5)
    assert gbm_v6_metrics["latency_ms"]["rejected_avg"] == pytest.approx(55.0)
    assert summary["longest_acceptance_streak"] == 1
    assert summary["longest_rejection_streak"] == 1
    assert summary["current_acceptance_streak"] == 0
    assert summary["current_rejection_streak"] == 1
    assert summary["risk_flag_counts"] == {"volatility_spike": 1, "liquidity": 1}
    assert summary["risk_flags_with_accepts"] == 1
    assert summary["risk_flag_breakdown"] == {
        "volatility_spike": {
            "total": 1,
            "accepted": 1,
            "rejected": 0,
            "acceptance_rate": pytest.approx(1.0),
        },
        "liquidity": {
            "total": 1,
            "accepted": 0,
            "rejected": 1,
            "acceptance_rate": pytest.approx(0.0),
        },
    }
    assert summary["unique_risk_flags"] == 2
    assert summary["stress_failure_counts"] == {"stress_liquidity": 1}
    assert summary["stress_failures_with_accepts"] == 0
    assert summary["stress_failure_breakdown"] == {
        "stress_liquidity": {
            "total": 1,
            "accepted": 0,
            "rejected": 1,
            "acceptance_rate": pytest.approx(0.0),
        }
    }
    assert summary["unique_stress_failures"] == 1
    assert summary["latest_risk_flags"] == ["liquidity"]
    assert summary["latest_stress_failures"] == ["stress_liquidity"]
    assert summary["history_start_generated_at"] == "2024-05-01T00:00:00Z"
    assert summary["median_net_edge_bps"] == pytest.approx(3.75)
    assert summary["p90_net_edge_bps"] == pytest.approx(5.95)
    assert summary["p95_net_edge_bps"] == pytest.approx(6.225)
    assert summary["median_cost_bps"] == pytest.approx(1.85)
    assert summary["p90_cost_bps"] == pytest.approx(2.37)
    assert summary["median_latency_ms"] == pytest.approx(48.5)
    assert summary["p90_latency_ms"] == pytest.approx(53.7)
    assert summary["p95_latency_ms"] == pytest.approx(54.35)
    assert summary["median_expected_probability"] == pytest.approx(0.54)
    assert summary["median_expected_return_bps"] == pytest.approx(7.5)
    assert summary["avg_expected_value_bps"] == pytest.approx(4.68)
    assert summary["sum_expected_return_bps"] == pytest.approx(15.0)
    assert summary["sum_expected_value_bps"] == pytest.approx(9.36)
    assert summary["median_expected_value_bps"] == pytest.approx(4.68)
    assert summary["min_expected_value_bps"] == pytest.approx(1.2)
    assert summary["max_expected_value_bps"] == pytest.approx(8.16)
    assert summary["avg_expected_value_minus_cost_bps"] == pytest.approx(2.83)
    assert summary["sum_expected_value_minus_cost_bps"] == pytest.approx(5.66)
    assert summary["probability_threshold_margin_count"] == 2
    assert summary["avg_probability_threshold_margin"] == pytest.approx(-0.085)
    assert summary["probability_threshold_breaches"] == 1
    assert summary["accepted_probability_threshold_margin_count"] == 1
    assert summary["accepted_avg_probability_threshold_margin"] == pytest.approx(0.08)
    assert summary["accepted_probability_threshold_breaches"] == 0
    assert summary["accepted_min_probability_threshold_margin"] == pytest.approx(0.08)
    assert summary["accepted_max_probability_threshold_margin"] == pytest.approx(0.08)
    assert summary["accepted_median_probability_threshold_margin"] == pytest.approx(0.08)
    assert summary["accepted_p10_probability_threshold_margin"] == pytest.approx(0.08)
    assert summary["accepted_p90_probability_threshold_margin"] == pytest.approx(0.08)
    assert summary["accepted_std_probability_threshold_margin"] == pytest.approx(0.0)
    assert summary["rejected_probability_threshold_margin_count"] == 1
    assert summary["rejected_avg_probability_threshold_margin"] == pytest.approx(-0.25)
    assert summary["rejected_probability_threshold_breaches"] == 1
    assert summary["rejected_min_probability_threshold_margin"] == pytest.approx(-0.25)
    assert summary["rejected_max_probability_threshold_margin"] == pytest.approx(-0.25)
    assert summary["rejected_median_probability_threshold_margin"] == pytest.approx(-0.25)
    assert summary["rejected_p10_probability_threshold_margin"] == pytest.approx(-0.25)
    assert summary["rejected_p90_probability_threshold_margin"] == pytest.approx(-0.25)
    assert summary["rejected_std_probability_threshold_margin"] == pytest.approx(0.0)
    assert summary["cost_threshold_margin_count"] == 2
    assert summary["avg_cost_threshold_margin"] == pytest.approx(8.15)
    assert summary["cost_threshold_breaches"] == 1
    assert summary["accepted_cost_threshold_margin_count"] == 1
    assert summary["accepted_avg_cost_threshold_margin"] == pytest.approx(16.8)
    assert summary["accepted_cost_threshold_breaches"] == 0
    assert summary["accepted_min_cost_threshold_margin"] == pytest.approx(16.8)
    assert summary["accepted_max_cost_threshold_margin"] == pytest.approx(16.8)
    assert summary["accepted_median_cost_threshold_margin"] == pytest.approx(16.8)
    assert summary["accepted_p10_cost_threshold_margin"] == pytest.approx(16.8)
    assert summary["accepted_p90_cost_threshold_margin"] == pytest.approx(16.8)
    assert summary["accepted_std_cost_threshold_margin"] == pytest.approx(0.0)
    assert summary["rejected_cost_threshold_margin_count"] == 1
    assert summary["rejected_avg_cost_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_cost_threshold_breaches"] == 1
    assert summary["rejected_min_cost_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_max_cost_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_median_cost_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_p10_cost_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_p90_cost_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_std_cost_threshold_margin"] == pytest.approx(0.0)
    assert summary["net_edge_threshold_margin_count"] == 2
    assert summary["avg_net_edge_threshold_margin"] == pytest.approx(0.25)
    assert summary["net_edge_threshold_breaches"] == 1
    assert summary["accepted_net_edge_threshold_margin_count"] == 1
    assert summary["accepted_avg_net_edge_threshold_margin"] == pytest.approx(1.5)
    assert summary["accepted_net_edge_threshold_breaches"] == 0
    assert summary["accepted_min_net_edge_threshold_margin"] == pytest.approx(1.5)
    assert summary["accepted_max_net_edge_threshold_margin"] == pytest.approx(1.5)
    assert summary["accepted_median_net_edge_threshold_margin"] == pytest.approx(1.5)
    assert summary["accepted_p10_net_edge_threshold_margin"] == pytest.approx(1.5)
    assert summary["accepted_p90_net_edge_threshold_margin"] == pytest.approx(1.5)
    assert summary["accepted_std_net_edge_threshold_margin"] == pytest.approx(0.0)
    assert summary["rejected_net_edge_threshold_margin_count"] == 1
    assert summary["rejected_avg_net_edge_threshold_margin"] == pytest.approx(-1.0)
    assert summary["rejected_net_edge_threshold_breaches"] == 1
    assert summary["rejected_min_net_edge_threshold_margin"] == pytest.approx(-1.0)
    assert summary["rejected_max_net_edge_threshold_margin"] == pytest.approx(-1.0)
    assert summary["rejected_median_net_edge_threshold_margin"] == pytest.approx(-1.0)
    assert summary["rejected_p10_net_edge_threshold_margin"] == pytest.approx(-1.0)
    assert summary["rejected_p90_net_edge_threshold_margin"] == pytest.approx(-1.0)
    assert summary["rejected_std_net_edge_threshold_margin"] == pytest.approx(0.0)
    assert summary["latency_threshold_margin_count"] == 2
    assert summary["avg_latency_threshold_margin"] == pytest.approx(6.5)
    assert summary["latency_threshold_breaches"] == 1
    assert summary["accepted_latency_threshold_margin_count"] == 1
    assert summary["accepted_avg_latency_threshold_margin"] == pytest.approx(18.0)
    assert summary["accepted_latency_threshold_breaches"] == 0
    assert summary["accepted_min_latency_threshold_margin"] == pytest.approx(18.0)
    assert summary["accepted_max_latency_threshold_margin"] == pytest.approx(18.0)
    assert summary["accepted_median_latency_threshold_margin"] == pytest.approx(18.0)
    assert summary["accepted_p10_latency_threshold_margin"] == pytest.approx(18.0)
    assert summary["accepted_p90_latency_threshold_margin"] == pytest.approx(18.0)
    assert summary["accepted_std_latency_threshold_margin"] == pytest.approx(0.0)
    assert summary["rejected_latency_threshold_margin_count"] == 1
    assert summary["rejected_avg_latency_threshold_margin"] == pytest.approx(-5.0)
    assert summary["rejected_latency_threshold_breaches"] == 1
    assert summary["rejected_min_latency_threshold_margin"] == pytest.approx(-5.0)
    assert summary["rejected_max_latency_threshold_margin"] == pytest.approx(-5.0)
    assert summary["rejected_median_latency_threshold_margin"] == pytest.approx(-5.0)
    assert summary["rejected_p10_latency_threshold_margin"] == pytest.approx(-5.0)
    assert summary["rejected_p90_latency_threshold_margin"] == pytest.approx(-5.0)
    assert summary["rejected_std_latency_threshold_margin"] == pytest.approx(0.0)
    assert summary["notional_threshold_margin_count"] == 2
    assert summary["avg_notional_threshold_margin"] == pytest.approx(75.0)
    assert summary["notional_threshold_breaches"] == 1
    assert summary["accepted_notional_threshold_margin_count"] == 1
    assert summary["accepted_avg_notional_threshold_margin"] == pytest.approx(200.0)
    assert summary["accepted_notional_threshold_breaches"] == 0
    assert summary["accepted_min_notional_threshold_margin"] == pytest.approx(200.0)
    assert summary["accepted_max_notional_threshold_margin"] == pytest.approx(200.0)
    assert summary["accepted_median_notional_threshold_margin"] == pytest.approx(200.0)
    assert summary["accepted_p10_notional_threshold_margin"] == pytest.approx(200.0)
    assert summary["accepted_p90_notional_threshold_margin"] == pytest.approx(200.0)
    assert summary["accepted_std_notional_threshold_margin"] == pytest.approx(0.0)
    assert summary["rejected_notional_threshold_margin_count"] == 1
    assert summary["rejected_avg_notional_threshold_margin"] == pytest.approx(-50.0)
    assert summary["rejected_notional_threshold_breaches"] == 1
    assert summary["rejected_min_notional_threshold_margin"] == pytest.approx(-50.0)
    assert summary["rejected_max_notional_threshold_margin"] == pytest.approx(-50.0)
    assert summary["rejected_median_notional_threshold_margin"] == pytest.approx(-50.0)
    assert summary["rejected_p10_notional_threshold_margin"] == pytest.approx(-50.0)
    assert summary["rejected_p90_notional_threshold_margin"] == pytest.approx(-50.0)
    assert summary["rejected_std_notional_threshold_margin"] == pytest.approx(0.0)
    assert summary["median_expected_value_minus_cost_bps"] == pytest.approx(2.83)
    assert summary["min_expected_value_minus_cost_bps"] == pytest.approx(-1.3)
    assert summary["max_expected_value_minus_cost_bps"] == pytest.approx(6.96)
    assert summary["avg_model_expected_value_bps"] == pytest.approx(3.5725)
    assert summary["sum_model_expected_return_bps"] == pytest.approx(11.0)
    assert summary["sum_model_expected_value_bps"] == pytest.approx(7.145)
    assert summary["median_model_expected_value_bps"] == pytest.approx(3.5725)
    assert summary["min_model_expected_value_bps"] == pytest.approx(1.025)
    assert summary["max_model_expected_value_bps"] == pytest.approx(6.12)
    assert summary["avg_model_expected_value_minus_cost_bps"] == pytest.approx(1.7225)
    assert summary["sum_model_expected_value_minus_cost_bps"] == pytest.approx(3.445)
    assert summary["median_model_expected_value_minus_cost_bps"] == pytest.approx(1.7225)
    assert summary["min_model_expected_value_minus_cost_bps"] == pytest.approx(-1.475)
    assert summary["max_model_expected_value_minus_cost_bps"] == pytest.approx(4.92)
    assert summary["sum_cost_bps"] == pytest.approx(3.7)
    assert summary["sum_net_edge_bps"] == pytest.approx(7.5)
    assert summary["sum_latency_ms"] == pytest.approx(97.0)
    assert summary["sum_notional"] == pytest.approx(1_500.0)
    assert summary["accepted_avg_net_edge_bps"] == pytest.approx(6.5)
    assert summary["accepted_median_net_edge_bps"] == pytest.approx(6.5)
    assert summary["accepted_min_net_edge_bps"] == pytest.approx(6.5)
    assert summary["accepted_max_net_edge_bps"] == pytest.approx(6.5)
    assert summary["accepted_p10_net_edge_bps"] == pytest.approx(6.5)
    assert summary["accepted_p90_net_edge_bps"] == pytest.approx(6.5)
    assert summary["accepted_std_net_edge_bps"] == pytest.approx(0.0)
    assert summary["accepted_sum_net_edge_bps"] == pytest.approx(6.5)
    assert summary["accepted_net_edge_bps_count"] == 1
    assert summary["rejected_avg_net_edge_bps"] == pytest.approx(1.0)
    assert summary["rejected_median_net_edge_bps"] == pytest.approx(1.0)
    assert summary["rejected_min_net_edge_bps"] == pytest.approx(1.0)
    assert summary["rejected_max_net_edge_bps"] == pytest.approx(1.0)
    assert summary["rejected_p10_net_edge_bps"] == pytest.approx(1.0)
    assert summary["rejected_p90_net_edge_bps"] == pytest.approx(1.0)
    assert summary["rejected_std_net_edge_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_net_edge_bps"] == pytest.approx(1.0)
    assert summary["rejected_net_edge_bps_count"] == 1
    assert summary["accepted_avg_cost_bps"] == pytest.approx(1.2)
    assert summary["accepted_median_cost_bps"] == pytest.approx(1.2)
    assert summary["accepted_min_cost_bps"] == pytest.approx(1.2)
    assert summary["accepted_max_cost_bps"] == pytest.approx(1.2)
    assert summary["accepted_p10_cost_bps"] == pytest.approx(1.2)
    assert summary["accepted_p90_cost_bps"] == pytest.approx(1.2)
    assert summary["accepted_std_cost_bps"] == pytest.approx(0.0)
    assert summary["accepted_sum_cost_bps"] == pytest.approx(1.2)
    assert summary["accepted_cost_bps_count"] == 1
    assert summary["rejected_avg_cost_bps"] == pytest.approx(2.5)
    assert summary["rejected_median_cost_bps"] == pytest.approx(2.5)
    assert summary["rejected_min_cost_bps"] == pytest.approx(2.5)
    assert summary["rejected_max_cost_bps"] == pytest.approx(2.5)
    assert summary["rejected_p10_cost_bps"] == pytest.approx(2.5)
    assert summary["rejected_p90_cost_bps"] == pytest.approx(2.5)
    assert summary["rejected_std_cost_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_cost_bps"] == pytest.approx(2.5)
    assert summary["rejected_cost_bps_count"] == 1
    assert summary["accepted_avg_expected_probability"] == pytest.approx(0.68)
    assert summary["accepted_median_expected_probability"] == pytest.approx(0.68)
    assert summary["accepted_std_expected_probability"] == pytest.approx(0.0)
    assert summary["accepted_expected_probability_count"] == 1
    assert summary["rejected_avg_expected_probability"] == pytest.approx(0.4)
    assert summary["rejected_median_expected_probability"] == pytest.approx(0.4)
    assert summary["rejected_std_expected_probability"] == pytest.approx(0.0)
    assert summary["rejected_expected_probability_count"] == 1
    assert summary["accepted_avg_expected_return_bps"] == pytest.approx(12.0)
    assert summary["accepted_median_expected_return_bps"] == pytest.approx(12.0)
    assert summary["accepted_p90_expected_return_bps"] == pytest.approx(12.0)
    assert summary["accepted_std_expected_return_bps"] == pytest.approx(0.0)
    assert summary["accepted_sum_expected_return_bps"] == pytest.approx(12.0)
    assert summary["accepted_expected_return_bps_count"] == 1
    assert summary["rejected_avg_expected_return_bps"] == pytest.approx(3.0)
    assert summary["rejected_median_expected_return_bps"] == pytest.approx(3.0)
    assert summary["rejected_p90_expected_return_bps"] == pytest.approx(3.0)
    assert summary["rejected_std_expected_return_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_expected_return_bps"] == pytest.approx(3.0)
    assert summary["rejected_expected_return_bps_count"] == 1
    assert summary["accepted_avg_expected_value_bps"] == pytest.approx(8.16)
    assert summary["accepted_median_expected_value_bps"] == pytest.approx(8.16)
    assert summary["accepted_p90_expected_value_bps"] == pytest.approx(8.16)
    assert summary["accepted_std_expected_value_bps"] == pytest.approx(0.0)
    assert summary["accepted_sum_expected_value_bps"] == pytest.approx(8.16)
    assert summary["accepted_expected_value_bps_count"] == 1
    assert summary["rejected_avg_expected_value_bps"] == pytest.approx(1.2)
    assert summary["rejected_median_expected_value_bps"] == pytest.approx(1.2)
    assert summary["rejected_p90_expected_value_bps"] == pytest.approx(1.2)
    assert summary["rejected_std_expected_value_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_expected_value_bps"] == pytest.approx(1.2)
    assert summary["rejected_expected_value_bps_count"] == 1
    assert summary["accepted_avg_expected_value_minus_cost_bps"] == pytest.approx(6.96)
    assert summary["accepted_median_expected_value_minus_cost_bps"] == pytest.approx(6.96)
    assert summary["accepted_p90_expected_value_minus_cost_bps"] == pytest.approx(6.96)
    assert summary["accepted_std_expected_value_minus_cost_bps"] == pytest.approx(0.0)
    assert summary["accepted_sum_expected_value_minus_cost_bps"] == pytest.approx(6.96)
    assert summary["accepted_expected_value_minus_cost_bps_count"] == 1
    assert summary["rejected_avg_expected_value_minus_cost_bps"] == pytest.approx(-1.3)
    assert summary["rejected_median_expected_value_minus_cost_bps"] == pytest.approx(-1.3)
    assert summary["rejected_p90_expected_value_minus_cost_bps"] == pytest.approx(-1.3)
    assert summary["rejected_std_expected_value_minus_cost_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_expected_value_minus_cost_bps"] == pytest.approx(-1.3)
    assert summary["rejected_expected_value_minus_cost_bps_count"] == 1
    assert summary["accepted_avg_notional"] == pytest.approx(1_000.0)
    assert summary["accepted_median_notional"] == pytest.approx(1_000.0)
    assert summary["accepted_p90_notional"] == pytest.approx(1_000.0)
    assert summary["accepted_min_notional"] == pytest.approx(1_000.0)
    assert summary["accepted_max_notional"] == pytest.approx(1_000.0)
    assert summary["accepted_std_notional"] == pytest.approx(0.0)
    assert summary["accepted_sum_notional"] == pytest.approx(1_000.0)
    assert summary["accepted_notional_count"] == 1
    assert summary["rejected_avg_notional"] == pytest.approx(500.0)
    assert summary["rejected_median_notional"] == pytest.approx(500.0)
    assert summary["rejected_p90_notional"] == pytest.approx(500.0)
    assert summary["rejected_min_notional"] == pytest.approx(500.0)
    assert summary["rejected_max_notional"] == pytest.approx(500.0)
    assert summary["rejected_std_notional"] == pytest.approx(0.0)
    assert summary["rejected_sum_notional"] == pytest.approx(500.0)
    assert summary["rejected_notional_count"] == 1
    assert summary["accepted_avg_latency_ms"] == pytest.approx(42.0)
    assert summary["accepted_median_latency_ms"] == pytest.approx(42.0)
    assert summary["accepted_p90_latency_ms"] == pytest.approx(42.0)
    assert summary["accepted_p95_latency_ms"] == pytest.approx(42.0)
    assert summary["accepted_min_latency_ms"] == pytest.approx(42.0)
    assert summary["accepted_max_latency_ms"] == pytest.approx(42.0)
    assert summary["accepted_std_latency_ms"] == pytest.approx(0.0)
    assert summary["accepted_sum_latency_ms"] == pytest.approx(42.0)
    assert summary["accepted_latency_ms_count"] == 1
    assert summary["rejected_avg_latency_ms"] == pytest.approx(55.0)
    assert summary["rejected_median_latency_ms"] == pytest.approx(55.0)
    assert summary["rejected_p90_latency_ms"] == pytest.approx(55.0)
    assert summary["rejected_p95_latency_ms"] == pytest.approx(55.0)
    assert summary["rejected_min_latency_ms"] == pytest.approx(55.0)
    assert summary["rejected_max_latency_ms"] == pytest.approx(55.0)
    assert summary["rejected_std_latency_ms"] == pytest.approx(0.0)
    assert summary["rejected_sum_latency_ms"] == pytest.approx(55.0)
    assert summary["rejected_latency_ms_count"] == 1
    assert summary["accepted_avg_model_success_probability"] == pytest.approx(0.72)
    assert summary["accepted_median_model_success_probability"] == pytest.approx(0.72)
    assert summary["accepted_std_model_success_probability"] == pytest.approx(0.0)
    assert summary["accepted_model_success_probability_count"] == 1
    assert summary["rejected_avg_model_success_probability"] == pytest.approx(0.41)
    assert summary["rejected_median_model_success_probability"] == pytest.approx(0.41)
    assert summary["rejected_std_model_success_probability"] == pytest.approx(0.0)
    assert summary["rejected_model_success_probability_count"] == 1
    assert summary["accepted_avg_model_expected_return_bps"] == pytest.approx(8.5)
    assert summary["accepted_median_model_expected_return_bps"] == pytest.approx(8.5)
    assert summary["accepted_p90_model_expected_return_bps"] == pytest.approx(8.5)
    assert summary["accepted_std_model_expected_return_bps"] == pytest.approx(0.0)
    assert summary["accepted_sum_model_expected_return_bps"] == pytest.approx(8.5)
    assert summary["accepted_model_expected_return_bps_count"] == 1
    assert summary["rejected_avg_model_expected_return_bps"] == pytest.approx(2.5)
    assert summary["rejected_median_model_expected_return_bps"] == pytest.approx(2.5)
    assert summary["rejected_p90_model_expected_return_bps"] == pytest.approx(2.5)
    assert summary["rejected_std_model_expected_return_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_model_expected_return_bps"] == pytest.approx(2.5)
    assert summary["rejected_model_expected_return_bps_count"] == 1
    assert summary["accepted_avg_model_expected_value_bps"] == pytest.approx(6.12)
    assert summary["accepted_median_model_expected_value_bps"] == pytest.approx(6.12)
    assert summary["accepted_p90_model_expected_value_bps"] == pytest.approx(6.12)
    assert summary["accepted_std_model_expected_value_bps"] == pytest.approx(0.0)
    assert summary["accepted_sum_model_expected_value_bps"] == pytest.approx(6.12)
    assert summary["accepted_model_expected_value_bps_count"] == 1
    assert summary["rejected_avg_model_expected_value_bps"] == pytest.approx(1.025)
    assert summary["rejected_median_model_expected_value_bps"] == pytest.approx(1.025)
    assert summary["rejected_p90_model_expected_value_bps"] == pytest.approx(1.025)
    assert summary["rejected_std_model_expected_value_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_model_expected_value_bps"] == pytest.approx(1.025)
    assert summary["rejected_model_expected_value_bps_count"] == 1
    assert summary["accepted_avg_model_expected_value_minus_cost_bps"] == pytest.approx(
        4.92
    )
    assert summary["accepted_median_model_expected_value_minus_cost_bps"] == pytest.approx(
        4.92
    )
    assert summary["accepted_p90_model_expected_value_minus_cost_bps"] == pytest.approx(
        4.92
    )
    assert summary["accepted_std_model_expected_value_minus_cost_bps"] == pytest.approx(0.0)
    assert summary["accepted_sum_model_expected_value_minus_cost_bps"] == pytest.approx(
        4.92
    )
    assert summary["accepted_model_expected_value_minus_cost_bps_count"] == 1
    assert summary["rejected_avg_model_expected_value_minus_cost_bps"] == pytest.approx(
        -1.475
    )
    assert summary["rejected_median_model_expected_value_minus_cost_bps"] == pytest.approx(
        -1.475
    )
    assert summary["rejected_p90_model_expected_value_minus_cost_bps"] == pytest.approx(
        -1.475
    )
    assert summary["rejected_std_model_expected_value_minus_cost_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_model_expected_value_minus_cost_bps"] == pytest.approx(
        -1.475
    )
    assert summary["rejected_model_expected_value_minus_cost_bps_count"] == 1
    assert summary["action_usage"] == {"BUY": 1, "SELL": 1}
    assert summary["unique_actions"] == 2
    assert summary["actions_with_accepts"] == 1
    action_breakdown = summary["action_breakdown"]
    assert action_breakdown["BUY"]["metrics"]["expected_value_minus_cost_bps"]["accepted_sum"] == pytest.approx(6.96)
    assert action_breakdown["SELL"]["metrics"]["net_edge_bps"]["rejected_sum"] == pytest.approx(1.0)
    assert summary["strategy_usage"] == {"daily": 2}
    assert summary["unique_strategies"] == 1
    assert summary["strategies_with_accepts"] == 1
    strategy_breakdown = summary["strategy_breakdown"]
    daily_metrics = strategy_breakdown["daily"]["metrics"]
    assert daily_metrics["expected_value_minus_cost_bps"]["total_sum"] == pytest.approx(5.66)
    assert daily_metrics["expected_value_minus_cost_bps"]["accepted_sum"] == pytest.approx(6.96)
    assert daily_metrics["expected_value_minus_cost_bps"]["rejected_sum"] == pytest.approx(-1.3)
    assert daily_metrics["net_edge_bps"]["total_sum"] == pytest.approx(7.5)
    assert daily_metrics["notional"]["total_sum"] == pytest.approx(1500)
    assert summary["symbol_usage"] == {"BTC/USDT": 1, "ETH/USDT": 1}
    assert summary["unique_symbols"] == 2
    assert summary["symbols_with_accepts"] == 1
    symbol_breakdown = summary["symbol_breakdown"]
    assert symbol_breakdown["BTC/USDT"]["metrics"]["notional"]["accepted_sum"] == pytest.approx(1000)
    assert symbol_breakdown["ETH/USDT"]["metrics"]["expected_value_minus_cost_bps"]["rejected_sum"] == pytest.approx(-1.3)
    assert summary["latest_candidate"]["expected_value_bps"] == pytest.approx(1.2)
    assert summary["latest_expected_value_bps"] == pytest.approx(1.2)
    assert summary["latest_expected_value_minus_cost_bps"] == pytest.approx(-1.3)
    assert summary["latest_net_edge_bps"] == pytest.approx(1.0)
    assert summary["latest_cost_bps"] == pytest.approx(2.5)
    assert summary["latest_latency_ms"] == pytest.approx(55.0)
    assert summary["latest_expected_probability"] == pytest.approx(0.4)
    assert summary["latest_expected_return_bps"] == pytest.approx(3.0)
    assert summary["latest_notional"] == pytest.approx(500.0)
    assert summary["latest_probability_threshold_margin"] == pytest.approx(-0.25)
    assert summary["latest_cost_threshold_margin"] == pytest.approx(-0.5)
    assert summary["latest_net_edge_threshold_margin"] == pytest.approx(-1.0)
    assert summary["latest_latency_threshold_margin"] == pytest.approx(-5.0)
    assert summary["latest_notional_threshold_margin"] == pytest.approx(-50.0)
    assert summary["latest_model_expected_value_bps"] == pytest.approx(1.025)
    assert summary["latest_model_expected_value_minus_cost_bps"] == pytest.approx(-1.475)
    assert summary["latest_model_expected_return_bps"] == pytest.approx(2.5)
    assert summary["latest_model_success_probability"] == pytest.approx(0.41)
    assert summary["std_net_edge_bps"] == pytest.approx(2.75)
    assert summary["std_cost_bps"] == pytest.approx(0.65)
    assert summary["std_latency_ms"] == pytest.approx(6.5)
    assert summary["std_expected_probability"] == pytest.approx(0.14)
    assert summary["std_expected_return_bps"] == pytest.approx(4.5)
    assert summary["std_expected_value_bps"] == pytest.approx(3.48)
    assert summary["std_expected_value_minus_cost_bps"] == pytest.approx(4.13)
    assert summary["std_notional"] == pytest.approx(250.0)
    assert summary["std_model_success_probability"] == pytest.approx(0.155)
    assert summary["std_model_expected_return_bps"] == pytest.approx(3.0)
    assert summary["std_model_expected_value_bps"] == pytest.approx(2.5475)
    assert summary["std_model_expected_value_minus_cost_bps"] == pytest.approx(3.1975)
    assert summary["latest_model"] == "gbm_v6"
    assert summary["history_limit"] == 10
    assert len(summary["history"]) == 2
    assert summary["history"][0]["candidate"]["symbol"] == "BTC/USDT"


def test_export_decision_engine_summary_requires_data(tmp_path: Path) -> None:
    output_path = tmp_path / "summary.json"
    exit_code = export_summary(
        [
            "--ledger",
            str(tmp_path),
            "--output",
            str(output_path),
            "--require-evaluations",
        ]
    )
    assert exit_code == 2
    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["total"] == 0
    assert summary["accepted"] == 0
