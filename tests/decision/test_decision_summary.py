from __future__ import annotations

import pytest

from bot_core.decision.summary import DecisionSummaryAggregator


def _build_summary(
    evaluations: list[dict[str, object]], *, history_limit: int | None = None
) -> dict[str, object]:
    aggregator = DecisionSummaryAggregator(evaluations, history_limit=history_limit)
    return dict(aggregator.build_summary())
from bot_core.decision.summary import (
    DecisionEngineSummary,
    summarize_evaluation_payloads,
)
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING


_REPO_ROOT = Path(__file__).resolve().parents[2]
if "bot_core" not in sys.modules:
    bot_core_stub = types.ModuleType("bot_core")
    bot_core_stub.__path__ = [str(_REPO_ROOT / "bot_core")]
    sys.modules["bot_core"] = bot_core_stub

if "bot_core.decision" not in sys.modules:
    decision_stub = types.ModuleType("bot_core.decision")
    decision_stub.__path__ = [str(_REPO_ROOT / "bot_core" / "decision")]
    sys.modules["bot_core.decision"] = decision_stub

from bot_core.decision.summary import summarize_evaluation_payloads
from bot_core.decision.schemas import DecisionEngineSummary


if TYPE_CHECKING:
    from bot_core.decision.models import DecisionEngineSummary


def _summarize(*args, **kwargs) -> dict[str, object]:
    return summarize_evaluation_payloads(*args, **kwargs).model_dump(
        exclude_none=True
    )


def _build_full_evaluations() -> list[dict[str, object]]:
    return [
        {
            "accepted": True,
            "cost_bps": 1.0,
            "net_edge_bps": 5.0,
            "risk_flags": ("latency",),
            "model_name": "gbm_v1",
            "model_expected_return_bps": 8.5,
            "model_success_probability": 0.72,
            "latency_ms": 42.0,
            "thresholds": {
                "min_probability": 0.6,
                "max_cost_bps": 12.0,
                "min_net_edge_bps": 4.0,
                "max_latency_ms": 60.0,
                "max_trade_notional": 2_000.0,
            },
            "candidate": {
                "symbol": "BTC/USDT",
                "action": "BUY",
                "strategy": "daily",
                "expected_probability": 0.7,
                "expected_return_bps": 10.0,
                "notional": 1_500,
                "latency_ms": 42.0,
                "metadata": {"generated_at": "2024-04-01T00:00:00Z"},
            },
        },
        {
            "accepted": False,
            "reasons": ("too_costly",),
            "risk_flags": ("volatility", "latency"),
            "stress_failures": ["liquidity"],
            "model_name": "gbm_v2",
            "thresholds": {
                "min_probability": 0.6,
                "max_cost_bps": 2.0,
                "min_net_edge_bps": 1.5,
                "max_latency_ms": 52.0,
                "max_trade_notional": 750.0,
            },
            "cost_bps": 2.5,
            "net_edge_bps": 1.0,
            "model_expected_return_bps": 4.5,
            "model_success_probability": 0.41,
            "latency_ms": 55.0,
            "candidate": {
                "symbol": "ETH/USDT",
                "action": "SELL",
                "strategy": "intraday",
                "expected_probability": 0.55,
                "expected_return_bps": 4.0,
                "notional": 800,
                "latency_ms": 55.0,
                "metadata": {"generated_at": "2024-05-01T00:00:00Z"},
            },
            "model_selection": {"selected": "gbm_v2", "score": 0.62},
        },
    ]

    summary = summarize_evaluation_payloads(evaluations, history_limit=5)
    DecisionEngineSummary.model_validate({"type": "decision_engine_summary", **summary})
    summary = _build_summary(evaluations, history_limit=5)
    summary_model = summarize_evaluation_payloads(evaluations, history_limit=5)
    assert isinstance(summary_model, DecisionEngineSummary)
    summary = summary_model.model_dump()

def test_summarize_evaluation_payloads_counts_and_latest_fields() -> None:
    evaluations = _build_full_evaluations()

    summary = _summarize(evaluations, history_limit=5)

    assert summary["total"] == 2
    assert summary["accepted"] == 1
    assert summary["rejected"] == 1
    assert summary["history_window"] == 2
    assert summary["rejection_reasons"] == {"too_costly": 1}
    assert summary["unique_rejection_reasons"] == 1
    assert summary["risk_flag_counts"] == {"latency": 2, "volatility": 1}
    assert summary["risk_flags_with_accepts"] == 1
    assert summary["risk_flag_breakdown"] == {
        "latency": {
            "total": 2,
            "accepted": 1,
            "rejected": 1,
            "acceptance_rate": pytest.approx(0.5),
        },
        "volatility": {
            "total": 1,
            "accepted": 0,
            "rejected": 1,
            "acceptance_rate": pytest.approx(0.0),
        },
    }
    assert summary["unique_risk_flags"] == 2
    assert summary["stress_failure_counts"] == {"liquidity": 1}
    assert summary["stress_failures_with_accepts"] == 0
    assert summary["stress_failure_breakdown"] == {
        "liquidity": {
            "total": 1,
            "accepted": 0,
            "rejected": 1,
            "acceptance_rate": pytest.approx(0.0),
        }
    }
    assert summary["unique_stress_failures"] == 1
    assert summary["history_generated_at_count"] == 2
    assert summary["history_missing_generated_at"] == 0
    assert summary["history_generated_at_coverage"] == pytest.approx(1.0)
    assert summary["full_history_generated_at_count"] == 2
    assert summary["full_history_missing_generated_at"] == 0
    assert summary["full_history_generated_at_coverage"] == pytest.approx(1.0)
    assert summary["model_usage"] == {"gbm_v1": 1, "gbm_v2": 1}
    assert summary["unique_models"] == 2
    assert summary["models_with_accepts"] == 1
    model_breakdown = summary["model_breakdown"]
    assert model_breakdown["gbm_v1"]["total"] == 1
    assert model_breakdown["gbm_v1"]["accepted"] == 1
    assert model_breakdown["gbm_v1"]["rejected"] == 0
    assert model_breakdown["gbm_v1"]["acceptance_rate"] == pytest.approx(1.0)
    gbm_v1_metrics = model_breakdown["gbm_v1"]["metrics"]
    assert gbm_v1_metrics["net_edge_bps"]["total_sum"] == pytest.approx(5.0)
    assert gbm_v1_metrics["net_edge_bps"]["accepted_sum"] == pytest.approx(5.0)
    assert gbm_v1_metrics["net_edge_bps"]["rejected_sum"] == pytest.approx(0.0)
    assert gbm_v1_metrics["expected_value_minus_cost_bps"]["accepted_sum"] == pytest.approx(6.0)
    assert gbm_v1_metrics["expected_value_bps"]["total_sum"] == pytest.approx(7.0)
    assert gbm_v1_metrics["cost_bps"]["accepted_sum"] == pytest.approx(1.0)
    assert gbm_v1_metrics["notional"]["total_sum"] == pytest.approx(1500)
    assert gbm_v1_metrics["latency_ms"]["total_avg"] == pytest.approx(42.0)

    assert model_breakdown["gbm_v2"]["total"] == 1
    assert model_breakdown["gbm_v2"]["accepted"] == 0
    assert model_breakdown["gbm_v2"]["rejected"] == 1
    assert model_breakdown["gbm_v2"]["acceptance_rate"] == pytest.approx(0.0)
    gbm_v2_metrics = model_breakdown["gbm_v2"]["metrics"]
    assert gbm_v2_metrics["net_edge_bps"]["rejected_sum"] == pytest.approx(1.0)
    assert gbm_v2_metrics["expected_value_minus_cost_bps"]["rejected_sum"] == pytest.approx(-0.3)
    assert gbm_v2_metrics["expected_value_bps"]["total_sum"] == pytest.approx(2.2)
    assert gbm_v2_metrics["cost_bps"]["total_sum"] == pytest.approx(2.5)
    assert gbm_v2_metrics["notional"]["rejected_sum"] == pytest.approx(800)
    assert gbm_v2_metrics["latency_ms"]["rejected_avg"] == pytest.approx(55.0)
    assert summary["longest_acceptance_streak"] == 1
    assert summary["longest_rejection_streak"] == 1
    assert summary["current_acceptance_streak"] == 0
    assert summary["current_rejection_streak"] == 1
    assert summary["action_usage"] == {"BUY": 1, "SELL": 1}
    assert summary["unique_actions"] == 2
    assert summary["actions_with_accepts"] == 1
    action_breakdown = summary["action_breakdown"]
    assert action_breakdown["BUY"]["accepted"] == 1
    assert action_breakdown["BUY"]["metrics"]["expected_value_minus_cost_bps"]["accepted_sum"] == pytest.approx(6.0)
    assert action_breakdown["SELL"]["rejected"] == 1
    assert action_breakdown["SELL"]["metrics"]["net_edge_bps"]["rejected_sum"] == pytest.approx(1.0)
    assert summary["strategy_usage"] == {"daily": 1, "intraday": 1}
    assert summary["unique_strategies"] == 2
    assert summary["strategies_with_accepts"] == 1
    strategy_breakdown = summary["strategy_breakdown"]
    assert strategy_breakdown["daily"]["metrics"]["expected_value_bps"]["accepted_sum"] == pytest.approx(7.0)
    assert summary["history_start_generated_at"] == "2024-04-01T00:00:00Z"
    assert summary["history_end_generated_at"] == "2024-05-01T00:00:00Z"
    assert summary["history_span_seconds"] == pytest.approx(2_592_000.0)
    assert summary["full_history_start_generated_at"] == "2024-04-01T00:00:00Z"
    assert summary["full_history_end_generated_at"] == "2024-05-01T00:00:00Z"
    assert summary["full_history_span_seconds"] == pytest.approx(2_592_000.0)
    assert strategy_breakdown["intraday"]["metrics"]["cost_bps"]["rejected_sum"] == pytest.approx(2.5)
    assert summary["symbol_usage"] == {"BTC/USDT": 1, "ETH/USDT": 1}
    assert summary["unique_symbols"] == 2
    assert summary["symbols_with_accepts"] == 1
    symbol_breakdown = summary["symbol_breakdown"]
    assert symbol_breakdown["BTC/USDT"]["metrics"]["notional"]["accepted_sum"] == pytest.approx(1500)
    assert symbol_breakdown["ETH/USDT"]["metrics"]["expected_value_minus_cost_bps"]["rejected_sum"] == pytest.approx(-0.3)
    assert summary["history_start_generated_at"] == "2024-04-01T00:00:00Z"
    assert summary["latest_model"] == "gbm_v2"
    assert summary["latest_status"] == "rejected"
    assert summary["latest_risk_flags"] == ["volatility", "latency"]
    assert summary["latest_stress_failures"] == ["liquidity"]
    assert summary["latest_model_selection"] == {"selected": "gbm_v2", "score": 0.62}
    assert summary["latest_candidate"]["symbol"] == "ETH/USDT"
    assert summary["latest_candidate"]["expected_value_bps"] == pytest.approx(2.2)
    assert summary["latest_generated_at"] == "2024-05-01T00:00:00Z"
    assert summary["latest_expected_value_bps"] == pytest.approx(2.2)
    assert summary["latest_expected_value_minus_cost_bps"] == pytest.approx(-0.3)
    assert summary["latest_net_edge_bps"] == pytest.approx(1.0)
    assert summary["latest_cost_bps"] == pytest.approx(2.5)
    assert summary["latest_latency_ms"] == pytest.approx(55.0)
    assert summary["latest_expected_probability"] == pytest.approx(0.55)
    assert summary["latest_expected_return_bps"] == pytest.approx(4.0)
    assert summary["latest_notional"] == pytest.approx(800.0)
    assert summary["latest_probability_threshold_margin"] == pytest.approx(-0.05)
    assert summary["latest_cost_threshold_margin"] == pytest.approx(-0.5)
    assert summary["latest_net_edge_threshold_margin"] == pytest.approx(-0.5)
    assert summary["latest_latency_threshold_margin"] == pytest.approx(-3.0)
    assert summary["latest_notional_threshold_margin"] == pytest.approx(-50.0)
    assert summary["latest_model_expected_value_bps"] == pytest.approx(1.845)
    assert summary["latest_model_expected_value_minus_cost_bps"] == pytest.approx(
        -0.655
    )
    assert summary["latest_model_expected_return_bps"] == pytest.approx(4.5)
    assert summary["latest_model_success_probability"] == pytest.approx(0.41)
    assert summary["avg_expected_probability"] == pytest.approx(0.625)
    assert summary["avg_cost_bps"] == pytest.approx(1.75)
    assert summary["avg_net_edge_bps"] == pytest.approx(3.0)
    assert summary["sum_cost_bps"] == pytest.approx(3.5)
    assert summary["sum_net_edge_bps"] == pytest.approx(6.0)
    assert summary["avg_latency_ms"] == pytest.approx(48.5)
    assert summary["sum_latency_ms"] == pytest.approx(97.0)
    assert summary["avg_model_success_probability"] == pytest.approx(0.565)
    assert summary["avg_model_expected_return_bps"] == pytest.approx(6.5)
    assert summary["avg_expected_value_bps"] == pytest.approx(4.6)
    assert summary["sum_expected_return_bps"] == pytest.approx(14.0)
    assert summary["sum_expected_value_bps"] == pytest.approx(9.2)
    assert summary["avg_expected_value_minus_cost_bps"] == pytest.approx(2.85)
    assert summary["sum_expected_value_minus_cost_bps"] == pytest.approx(5.7)
    assert summary["probability_threshold_margin_count"] == 2
    assert summary["avg_probability_threshold_margin"] == pytest.approx(0.025)
    assert summary["sum_probability_threshold_margin"] == pytest.approx(0.05)
    assert summary["median_probability_threshold_margin"] == pytest.approx(0.025)
    assert summary["p10_probability_threshold_margin"] == pytest.approx(-0.035)
    assert summary["p90_probability_threshold_margin"] == pytest.approx(0.085)
    assert summary["min_probability_threshold_margin"] == pytest.approx(-0.05)
    assert summary["max_probability_threshold_margin"] == pytest.approx(0.1)
    assert summary["std_probability_threshold_margin"] == pytest.approx(0.075)
    assert summary["probability_threshold_breaches"] == 1
    assert summary["probability_threshold_breach_rate"] == pytest.approx(0.5)
    assert summary["accepted_probability_threshold_margin_count"] == 1
    assert summary["accepted_avg_probability_threshold_margin"] == pytest.approx(0.1)
    assert summary["accepted_sum_probability_threshold_margin"] == pytest.approx(0.1)
    assert summary["accepted_probability_threshold_breaches"] == 0
    assert summary["accepted_probability_threshold_breach_rate"] == pytest.approx(0.0)
    assert summary["accepted_min_probability_threshold_margin"] == pytest.approx(0.1)
    assert summary["accepted_max_probability_threshold_margin"] == pytest.approx(0.1)
    assert summary["accepted_median_probability_threshold_margin"] == pytest.approx(0.1)
    assert summary["accepted_p10_probability_threshold_margin"] == pytest.approx(0.1)
    assert summary["accepted_p90_probability_threshold_margin"] == pytest.approx(0.1)
    assert summary["accepted_std_probability_threshold_margin"] == pytest.approx(0.0)
    assert summary["rejected_probability_threshold_margin_count"] == 1
    assert summary["rejected_avg_probability_threshold_margin"] == pytest.approx(-0.05)
    assert summary["rejected_sum_probability_threshold_margin"] == pytest.approx(-0.05)
    assert summary["rejected_probability_threshold_breaches"] == 1
    assert summary["rejected_probability_threshold_breach_rate"] == pytest.approx(1.0)
    assert summary["rejected_min_probability_threshold_margin"] == pytest.approx(-0.05)
    assert summary["rejected_max_probability_threshold_margin"] == pytest.approx(-0.05)
    assert summary["rejected_median_probability_threshold_margin"] == pytest.approx(-0.05)
    assert summary["rejected_p10_probability_threshold_margin"] == pytest.approx(-0.05)
    assert summary["rejected_p90_probability_threshold_margin"] == pytest.approx(-0.05)
    assert summary["rejected_std_probability_threshold_margin"] == pytest.approx(0.0)
    assert summary["cost_threshold_margin_count"] == 2
    assert summary["avg_cost_threshold_margin"] == pytest.approx(5.25)
    assert summary["sum_cost_threshold_margin"] == pytest.approx(10.5)
    assert summary["median_cost_threshold_margin"] == pytest.approx(5.25)
    assert summary["p10_cost_threshold_margin"] == pytest.approx(0.65)
    assert summary["p90_cost_threshold_margin"] == pytest.approx(9.85)
    assert summary["min_cost_threshold_margin"] == pytest.approx(-0.5)
    assert summary["max_cost_threshold_margin"] == pytest.approx(11.0)
    assert summary["std_cost_threshold_margin"] == pytest.approx(5.75)
    assert summary["cost_threshold_breaches"] == 1
    assert summary["cost_threshold_breach_rate"] == pytest.approx(0.5)
    assert summary["accepted_cost_threshold_margin_count"] == 1
    assert summary["accepted_avg_cost_threshold_margin"] == pytest.approx(11.0)
    assert summary["accepted_sum_cost_threshold_margin"] == pytest.approx(11.0)
    assert summary["accepted_cost_threshold_breaches"] == 0
    assert summary["accepted_cost_threshold_breach_rate"] == pytest.approx(0.0)
    assert summary["accepted_min_cost_threshold_margin"] == pytest.approx(11.0)
    assert summary["accepted_max_cost_threshold_margin"] == pytest.approx(11.0)
    assert summary["accepted_median_cost_threshold_margin"] == pytest.approx(11.0)
    assert summary["accepted_p10_cost_threshold_margin"] == pytest.approx(11.0)
    assert summary["accepted_p90_cost_threshold_margin"] == pytest.approx(11.0)
    assert summary["accepted_std_cost_threshold_margin"] == pytest.approx(0.0)
    assert summary["rejected_cost_threshold_margin_count"] == 1
    assert summary["rejected_avg_cost_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_sum_cost_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_cost_threshold_breaches"] == 1
    assert summary["rejected_cost_threshold_breach_rate"] == pytest.approx(1.0)
    assert summary["rejected_min_cost_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_max_cost_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_median_cost_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_p10_cost_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_p90_cost_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_std_cost_threshold_margin"] == pytest.approx(0.0)
    assert summary["net_edge_threshold_margin_count"] == 2
    assert summary["avg_net_edge_threshold_margin"] == pytest.approx(0.25)
    assert summary["sum_net_edge_threshold_margin"] == pytest.approx(0.5)
    assert summary["median_net_edge_threshold_margin"] == pytest.approx(0.25)
    assert summary["p10_net_edge_threshold_margin"] == pytest.approx(-0.35)
    assert summary["p90_net_edge_threshold_margin"] == pytest.approx(0.85)
    assert summary["min_net_edge_threshold_margin"] == pytest.approx(-0.5)
    assert summary["max_net_edge_threshold_margin"] == pytest.approx(1.0)
    assert summary["std_net_edge_threshold_margin"] == pytest.approx(0.75)
    assert summary["net_edge_threshold_breaches"] == 1
    assert summary["net_edge_threshold_breach_rate"] == pytest.approx(0.5)
    assert summary["accepted_net_edge_threshold_margin_count"] == 1
    assert summary["accepted_avg_net_edge_threshold_margin"] == pytest.approx(1.0)
    assert summary["accepted_sum_net_edge_threshold_margin"] == pytest.approx(1.0)
    assert summary["accepted_net_edge_threshold_breaches"] == 0
    assert summary["accepted_net_edge_threshold_breach_rate"] == pytest.approx(0.0)
    assert summary["accepted_min_net_edge_threshold_margin"] == pytest.approx(1.0)
    assert summary["accepted_max_net_edge_threshold_margin"] == pytest.approx(1.0)
    assert summary["accepted_median_net_edge_threshold_margin"] == pytest.approx(1.0)
    assert summary["accepted_p10_net_edge_threshold_margin"] == pytest.approx(1.0)
    assert summary["accepted_p90_net_edge_threshold_margin"] == pytest.approx(1.0)
    assert summary["accepted_std_net_edge_threshold_margin"] == pytest.approx(0.0)
    assert summary["rejected_net_edge_threshold_margin_count"] == 1
    assert summary["rejected_avg_net_edge_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_sum_net_edge_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_net_edge_threshold_breaches"] == 1
    assert summary["rejected_net_edge_threshold_breach_rate"] == pytest.approx(1.0)
    assert summary["rejected_min_net_edge_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_max_net_edge_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_median_net_edge_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_p10_net_edge_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_p90_net_edge_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_std_net_edge_threshold_margin"] == pytest.approx(0.0)
    assert summary["latency_threshold_margin_count"] == 2
    assert summary["avg_latency_threshold_margin"] == pytest.approx(7.5)
    assert summary["sum_latency_threshold_margin"] == pytest.approx(15.0)
    assert summary["median_latency_threshold_margin"] == pytest.approx(7.5)
    assert summary["p10_latency_threshold_margin"] == pytest.approx(-0.9)
    assert summary["p90_latency_threshold_margin"] == pytest.approx(15.9)
    assert summary["min_latency_threshold_margin"] == pytest.approx(-3.0)
    assert summary["max_latency_threshold_margin"] == pytest.approx(18.0)
    assert summary["std_latency_threshold_margin"] == pytest.approx(10.5)
    assert summary["latency_threshold_breaches"] == 1
    assert summary["latency_threshold_breach_rate"] == pytest.approx(0.5)
    assert summary["accepted_latency_threshold_margin_count"] == 1
    assert summary["accepted_avg_latency_threshold_margin"] == pytest.approx(18.0)
    assert summary["accepted_sum_latency_threshold_margin"] == pytest.approx(18.0)
    assert summary["accepted_latency_threshold_breaches"] == 0
    assert summary["accepted_latency_threshold_breach_rate"] == pytest.approx(0.0)
    assert summary["accepted_min_latency_threshold_margin"] == pytest.approx(18.0)
    assert summary["accepted_max_latency_threshold_margin"] == pytest.approx(18.0)
    assert summary["accepted_median_latency_threshold_margin"] == pytest.approx(18.0)
    assert summary["accepted_p10_latency_threshold_margin"] == pytest.approx(18.0)
    assert summary["accepted_p90_latency_threshold_margin"] == pytest.approx(18.0)
    assert summary["accepted_std_latency_threshold_margin"] == pytest.approx(0.0)
    assert summary["rejected_latency_threshold_margin_count"] == 1
    assert summary["rejected_avg_latency_threshold_margin"] == pytest.approx(-3.0)
    assert summary["rejected_sum_latency_threshold_margin"] == pytest.approx(-3.0)
    assert summary["rejected_latency_threshold_breaches"] == 1
    assert summary["rejected_latency_threshold_breach_rate"] == pytest.approx(1.0)
    assert summary["rejected_min_latency_threshold_margin"] == pytest.approx(-3.0)
    assert summary["rejected_max_latency_threshold_margin"] == pytest.approx(-3.0)
    assert summary["rejected_median_latency_threshold_margin"] == pytest.approx(-3.0)
    assert summary["rejected_p10_latency_threshold_margin"] == pytest.approx(-3.0)
    assert summary["rejected_p90_latency_threshold_margin"] == pytest.approx(-3.0)
    assert summary["rejected_std_latency_threshold_margin"] == pytest.approx(0.0)
    assert summary["notional_threshold_margin_count"] == 2
    assert summary["avg_notional_threshold_margin"] == pytest.approx(225.0)
    assert summary["sum_notional_threshold_margin"] == pytest.approx(450.0)
    assert summary["median_notional_threshold_margin"] == pytest.approx(225.0)
    assert summary["p10_notional_threshold_margin"] == pytest.approx(5.0)
    assert summary["p90_notional_threshold_margin"] == pytest.approx(445.0)
    assert summary["min_notional_threshold_margin"] == pytest.approx(-50.0)
    assert summary["max_notional_threshold_margin"] == pytest.approx(500.0)
    assert summary["std_notional_threshold_margin"] == pytest.approx(275.0)
    assert summary["notional_threshold_breaches"] == 1
    assert summary["notional_threshold_breach_rate"] == pytest.approx(0.5)
    assert summary["accepted_notional_threshold_margin_count"] == 1
    assert summary["accepted_avg_notional_threshold_margin"] == pytest.approx(500.0)
    assert summary["accepted_sum_notional_threshold_margin"] == pytest.approx(500.0)
    assert summary["accepted_notional_threshold_breaches"] == 0
    assert summary["accepted_notional_threshold_breach_rate"] == pytest.approx(0.0)
    assert summary["accepted_min_notional_threshold_margin"] == pytest.approx(500.0)
    assert summary["accepted_max_notional_threshold_margin"] == pytest.approx(500.0)
    assert summary["accepted_median_notional_threshold_margin"] == pytest.approx(500.0)
    assert summary["accepted_p10_notional_threshold_margin"] == pytest.approx(500.0)
    assert summary["accepted_p90_notional_threshold_margin"] == pytest.approx(500.0)
    assert summary["accepted_std_notional_threshold_margin"] == pytest.approx(0.0)
    assert summary["rejected_notional_threshold_margin_count"] == 1
    assert summary["rejected_avg_notional_threshold_margin"] == pytest.approx(-50.0)
    assert summary["rejected_sum_notional_threshold_margin"] == pytest.approx(-50.0)
    assert summary["rejected_notional_threshold_breaches"] == 1
    assert summary["rejected_notional_threshold_breach_rate"] == pytest.approx(1.0)
    assert summary["rejected_min_notional_threshold_margin"] == pytest.approx(-50.0)
    assert summary["rejected_max_notional_threshold_margin"] == pytest.approx(-50.0)
    assert summary["rejected_median_notional_threshold_margin"] == pytest.approx(-50.0)
    assert summary["rejected_p10_notional_threshold_margin"] == pytest.approx(-50.0)
    assert summary["rejected_p90_notional_threshold_margin"] == pytest.approx(-50.0)
    assert summary["rejected_std_notional_threshold_margin"] == pytest.approx(0.0)
    assert summary["avg_model_expected_value_bps"] == pytest.approx(3.9825)
    assert summary["sum_model_expected_return_bps"] == pytest.approx(13.0)
    assert summary["sum_model_expected_value_bps"] == pytest.approx(7.965)
    assert summary["avg_model_expected_value_minus_cost_bps"] == pytest.approx(
        2.2325
    )
    assert summary["sum_model_expected_value_minus_cost_bps"] == pytest.approx(
        4.465
    )
    assert summary["sum_notional"] == pytest.approx(2_300.0)
    assert summary["accepted_avg_net_edge_bps"] == pytest.approx(5.0)
    assert summary["accepted_median_net_edge_bps"] == pytest.approx(5.0)
    assert summary["accepted_p90_net_edge_bps"] == pytest.approx(5.0)
    assert summary["accepted_p10_net_edge_bps"] == pytest.approx(5.0)
    assert summary["accepted_min_net_edge_bps"] == pytest.approx(5.0)
    assert summary["accepted_max_net_edge_bps"] == pytest.approx(5.0)
    assert summary["accepted_std_net_edge_bps"] == pytest.approx(0.0)
    assert summary["accepted_sum_net_edge_bps"] == pytest.approx(5.0)
    assert summary["accepted_net_edge_bps_count"] == 1
    assert summary["rejected_avg_net_edge_bps"] == pytest.approx(1.0)
    assert summary["rejected_median_net_edge_bps"] == pytest.approx(1.0)
    assert summary["rejected_p90_net_edge_bps"] == pytest.approx(1.0)
    assert summary["rejected_p10_net_edge_bps"] == pytest.approx(1.0)
    assert summary["rejected_min_net_edge_bps"] == pytest.approx(1.0)
    assert summary["rejected_max_net_edge_bps"] == pytest.approx(1.0)
    assert summary["rejected_std_net_edge_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_net_edge_bps"] == pytest.approx(1.0)
    assert summary["rejected_net_edge_bps_count"] == 1
    assert summary["accepted_avg_cost_bps"] == pytest.approx(1.0)
    assert summary["accepted_median_cost_bps"] == pytest.approx(1.0)
    assert summary["accepted_p90_cost_bps"] == pytest.approx(1.0)
    assert summary["accepted_p10_cost_bps"] == pytest.approx(1.0)
    assert summary["accepted_min_cost_bps"] == pytest.approx(1.0)
    assert summary["accepted_max_cost_bps"] == pytest.approx(1.0)
    assert summary["accepted_std_cost_bps"] == pytest.approx(0.0)
    assert summary["accepted_sum_cost_bps"] == pytest.approx(1.0)
    assert summary["accepted_cost_bps_count"] == 1
    assert summary["rejected_avg_cost_bps"] == pytest.approx(2.5)
    assert summary["rejected_median_cost_bps"] == pytest.approx(2.5)
    assert summary["rejected_p90_cost_bps"] == pytest.approx(2.5)
    assert summary["rejected_p10_cost_bps"] == pytest.approx(2.5)
    assert summary["rejected_min_cost_bps"] == pytest.approx(2.5)
    assert summary["rejected_max_cost_bps"] == pytest.approx(2.5)
    assert summary["rejected_std_cost_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_cost_bps"] == pytest.approx(2.5)
    assert summary["rejected_cost_bps_count"] == 1
    assert summary["accepted_avg_expected_probability"] == pytest.approx(0.7)
    assert summary["accepted_median_expected_probability"] == pytest.approx(0.7)
    assert summary["accepted_std_expected_probability"] == pytest.approx(0.0)
    assert summary["accepted_expected_probability_count"] == 1
    assert summary["rejected_avg_expected_probability"] == pytest.approx(0.55)
    assert summary["rejected_median_expected_probability"] == pytest.approx(0.55)
    assert summary["rejected_std_expected_probability"] == pytest.approx(0.0)
    assert summary["rejected_expected_probability_count"] == 1
    assert summary["accepted_avg_expected_return_bps"] == pytest.approx(10.0)
    assert summary["accepted_median_expected_return_bps"] == pytest.approx(10.0)
    assert summary["accepted_p90_expected_return_bps"] == pytest.approx(10.0)
    assert summary["accepted_std_expected_return_bps"] == pytest.approx(0.0)
    assert summary["accepted_sum_expected_return_bps"] == pytest.approx(10.0)
    assert summary["accepted_expected_return_bps_count"] == 1
    assert summary["rejected_avg_expected_return_bps"] == pytest.approx(4.0)
    assert summary["rejected_median_expected_return_bps"] == pytest.approx(4.0)
    assert summary["rejected_p90_expected_return_bps"] == pytest.approx(4.0)
    assert summary["rejected_std_expected_return_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_expected_return_bps"] == pytest.approx(4.0)
    assert summary["rejected_expected_return_bps_count"] == 1
    assert summary["accepted_avg_expected_value_bps"] == pytest.approx(7.0)
    assert summary["accepted_median_expected_value_bps"] == pytest.approx(7.0)
    assert summary["accepted_p90_expected_value_bps"] == pytest.approx(7.0)
    assert summary["accepted_std_expected_value_bps"] == pytest.approx(0.0)
    assert summary["accepted_sum_expected_value_bps"] == pytest.approx(7.0)
    assert summary["accepted_expected_value_bps_count"] == 1
    assert summary["rejected_avg_expected_value_bps"] == pytest.approx(2.2)
    assert summary["rejected_median_expected_value_bps"] == pytest.approx(2.2)
    assert summary["rejected_p90_expected_value_bps"] == pytest.approx(2.2)
    assert summary["rejected_std_expected_value_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_expected_value_bps"] == pytest.approx(2.2)
    assert summary["rejected_expected_value_bps_count"] == 1
    assert summary["accepted_avg_expected_value_minus_cost_bps"] == pytest.approx(6.0)
    assert summary["accepted_median_expected_value_minus_cost_bps"] == pytest.approx(6.0)
    assert summary["accepted_p90_expected_value_minus_cost_bps"] == pytest.approx(6.0)
    assert summary["accepted_std_expected_value_minus_cost_bps"] == pytest.approx(0.0)
    assert summary["accepted_sum_expected_value_minus_cost_bps"] == pytest.approx(6.0)
    assert summary["accepted_expected_value_minus_cost_bps_count"] == 1
    assert summary["rejected_avg_expected_value_minus_cost_bps"] == pytest.approx(-0.3)
    assert summary["rejected_median_expected_value_minus_cost_bps"] == pytest.approx(-0.3)
    assert summary["rejected_p90_expected_value_minus_cost_bps"] == pytest.approx(-0.3)
    assert summary["rejected_std_expected_value_minus_cost_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_expected_value_minus_cost_bps"] == pytest.approx(-0.3)
    assert summary["rejected_expected_value_minus_cost_bps_count"] == 1
    assert summary["accepted_avg_notional"] == pytest.approx(1_500.0)
    assert summary["accepted_median_notional"] == pytest.approx(1_500.0)
    assert summary["accepted_p90_notional"] == pytest.approx(1_500.0)
    assert summary["accepted_std_notional"] == pytest.approx(0.0)
    assert summary["accepted_sum_notional"] == pytest.approx(1_500.0)
    assert summary["accepted_notional_count"] == 1
    assert summary["rejected_avg_notional"] == pytest.approx(800.0)
    assert summary["rejected_median_notional"] == pytest.approx(800.0)
    assert summary["rejected_p90_notional"] == pytest.approx(800.0)
    assert summary["rejected_std_notional"] == pytest.approx(0.0)
    assert summary["rejected_sum_notional"] == pytest.approx(800.0)
    assert summary["rejected_notional_count"] == 1
    assert summary["accepted_avg_latency_ms"] == pytest.approx(42.0)
    assert summary["accepted_median_latency_ms"] == pytest.approx(42.0)
    assert summary["accepted_p90_latency_ms"] == pytest.approx(42.0)
    assert summary["accepted_p95_latency_ms"] == pytest.approx(42.0)
    assert summary["accepted_std_latency_ms"] == pytest.approx(0.0)
    assert summary["accepted_sum_latency_ms"] == pytest.approx(42.0)
    assert summary["accepted_latency_ms_count"] == 1
    assert summary["rejected_avg_latency_ms"] == pytest.approx(55.0)
    assert summary["rejected_median_latency_ms"] == pytest.approx(55.0)
    assert summary["rejected_p90_latency_ms"] == pytest.approx(55.0)
    assert summary["rejected_p95_latency_ms"] == pytest.approx(55.0)
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
    assert summary["rejected_avg_model_expected_return_bps"] == pytest.approx(4.5)
    assert summary["rejected_median_model_expected_return_bps"] == pytest.approx(4.5)
    assert summary["rejected_p90_model_expected_return_bps"] == pytest.approx(4.5)
    assert summary["rejected_std_model_expected_return_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_model_expected_return_bps"] == pytest.approx(4.5)
    assert summary["rejected_model_expected_return_bps_count"] == 1
    assert summary["accepted_avg_model_expected_value_bps"] == pytest.approx(6.12)
    assert summary["accepted_median_model_expected_value_bps"] == pytest.approx(6.12)
    assert summary["accepted_p90_model_expected_value_bps"] == pytest.approx(6.12)
    assert summary["accepted_std_model_expected_value_bps"] == pytest.approx(0.0)
    assert summary["accepted_sum_model_expected_value_bps"] == pytest.approx(6.12)
    assert summary["accepted_model_expected_value_bps_count"] == 1
    assert summary["rejected_avg_model_expected_value_bps"] == pytest.approx(1.845)
    assert summary["rejected_median_model_expected_value_bps"] == pytest.approx(1.845)
    assert summary["rejected_p90_model_expected_value_bps"] == pytest.approx(1.845)
    assert summary["rejected_std_model_expected_value_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_model_expected_value_bps"] == pytest.approx(1.845)
    assert summary["rejected_model_expected_value_bps_count"] == 1
    assert summary["accepted_avg_model_expected_value_minus_cost_bps"] == pytest.approx(
        5.12
    )
    assert summary["accepted_median_model_expected_value_minus_cost_bps"] == pytest.approx(
        5.12
    )
    assert summary["accepted_p90_model_expected_value_minus_cost_bps"] == pytest.approx(
        5.12
    )
    assert summary["accepted_std_model_expected_value_minus_cost_bps"] == pytest.approx(0.0)
    assert summary["accepted_sum_model_expected_value_minus_cost_bps"] == pytest.approx(
        5.12
    )
    assert summary["accepted_model_expected_value_minus_cost_bps_count"] == 1
    assert summary["rejected_avg_model_expected_value_minus_cost_bps"] == pytest.approx(
        -0.655
    )
    assert summary["rejected_median_model_expected_value_minus_cost_bps"] == pytest.approx(
        -0.655
    )
    assert summary["rejected_p90_model_expected_value_minus_cost_bps"] == pytest.approx(
        -0.655
    )
    assert summary["rejected_std_model_expected_value_minus_cost_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_model_expected_value_minus_cost_bps"] == pytest.approx(
        -0.655
    )
    assert summary["rejected_model_expected_value_minus_cost_bps_count"] == 1
    assert summary["median_net_edge_bps"] == pytest.approx(3.0)
    assert summary["p90_net_edge_bps"] == pytest.approx(4.6)
    assert summary["p95_net_edge_bps"] == pytest.approx(4.8)
    assert summary["min_net_edge_bps"] == pytest.approx(1.0)
    assert summary["max_net_edge_bps"] == pytest.approx(5.0)
    assert summary["median_cost_bps"] == pytest.approx(1.75)
    assert summary["p90_cost_bps"] == pytest.approx(2.35)
    assert summary["min_cost_bps"] == pytest.approx(1.0)
    assert summary["max_cost_bps"] == pytest.approx(2.5)
    assert summary["median_latency_ms"] == pytest.approx(48.5)
    assert summary["p90_latency_ms"] == pytest.approx(53.7)
    assert summary["p95_latency_ms"] == pytest.approx(54.35)
    assert summary["min_latency_ms"] == pytest.approx(42.0)
    assert summary["max_latency_ms"] == pytest.approx(55.0)
    assert summary["median_expected_probability"] == pytest.approx(0.625)
    assert summary["median_expected_return_bps"] == pytest.approx(7.0)
    assert summary["min_expected_return_bps"] == pytest.approx(4.0)
    assert summary["max_expected_return_bps"] == pytest.approx(10.0)
    assert summary["median_expected_value_bps"] == pytest.approx(4.6)
    assert summary["min_expected_value_bps"] == pytest.approx(2.2)
    assert summary["max_expected_value_bps"] == pytest.approx(7.0)
    assert summary["median_expected_value_minus_cost_bps"] == pytest.approx(2.85)
    assert summary["min_expected_value_minus_cost_bps"] == pytest.approx(-0.3)
    assert summary["max_expected_value_minus_cost_bps"] == pytest.approx(6.0)
    assert summary["median_notional"] == pytest.approx(1_150.0)
    assert summary["min_notional"] == pytest.approx(800.0)
    assert summary["max_notional"] == pytest.approx(1_500.0)
    assert summary["median_model_success_probability"] == pytest.approx(0.565)
    assert summary["median_model_expected_return_bps"] == pytest.approx(6.5)
    assert summary["min_model_expected_return_bps"] == pytest.approx(4.5)
    assert summary["max_model_expected_return_bps"] == pytest.approx(8.5)
    assert summary["median_model_expected_value_bps"] == pytest.approx(3.9825)
    assert summary["min_model_expected_value_bps"] == pytest.approx(1.845)
    assert summary["max_model_expected_value_bps"] == pytest.approx(6.12)
    assert summary["median_model_expected_value_minus_cost_bps"] == pytest.approx(
        2.2325
    )
    assert summary["min_model_expected_value_minus_cost_bps"] == pytest.approx(-0.655)
    assert summary["max_model_expected_value_minus_cost_bps"] == pytest.approx(5.12)
    assert summary["std_net_edge_bps"] == pytest.approx(2.0)
    assert summary["std_cost_bps"] == pytest.approx(0.75)
    assert summary["std_latency_ms"] == pytest.approx(6.5)
    assert summary["std_expected_probability"] == pytest.approx(0.075)
    assert summary["std_expected_return_bps"] == pytest.approx(3.0)
    assert summary["std_expected_value_bps"] == pytest.approx(2.4)
    assert summary["std_expected_value_minus_cost_bps"] == pytest.approx(3.15)
    assert summary["std_notional"] == pytest.approx(350.0)
    assert summary["std_model_success_probability"] == pytest.approx(0.155)
    assert summary["std_model_expected_return_bps"] == pytest.approx(2.0)
    assert summary["std_model_expected_value_bps"] == pytest.approx(2.1375)
    assert summary["std_model_expected_value_minus_cost_bps"] == pytest.approx(2.8875)
    assert summary["full_total"] == 2


def test_summarize_evaluation_payloads_respects_history_limit() -> None:
    evaluations = [
        {
            "accepted": True,
            "model_name": "gbm_v1",
            "net_edge_bps": 2.0,
            "candidate": {
                "expected_probability": 0.6,
                "expected_return_bps": 6.0,
                "metadata": {"generated_at": "2024-04-01T00:00:00Z"},
            },
        },
        {
            "accepted": False,
            "reasons": ("too_costly",),
            "model_name": "gbm_v2",
            "net_edge_bps": 1.0,
            "candidate": {
                "expected_probability": 0.55,
                "expected_return_bps": 5.0,
                "metadata": {"generated_at": "2024-04-02T00:00:00Z"},
            },
        },
        {
            "accepted": True,
            "model_name": "gbm_v3",
            "net_edge_bps": 3.0,
            "candidate": {
                "expected_probability": 0.7,
                "expected_return_bps": 9.0,
                "metadata": {"generated_at": "2024-04-03T00:00:00Z"},
            },
        },
    ]

    summary = summarize_evaluation_payloads(evaluations, history_limit=2)
    DecisionEngineSummary.model_validate({"type": "decision_engine_summary", **summary})
    summary = _build_summary(evaluations, history_limit=2)
    summary_model = summarize_evaluation_payloads(evaluations, history_limit=2)
    assert isinstance(summary_model, DecisionEngineSummary)
    summary = summary_model.model_dump()
    summary = _summarize(evaluations, history_limit=2)

    assert summary["total"] == 2
    assert summary["accepted"] == 1
    assert summary["rejected"] == 1
    assert summary["history_limit"] == 2
    assert summary["history_window"] == 2
    assert summary["rejection_reasons"] == {"too_costly": 1}
    assert summary["history_start_generated_at"] == "2024-04-02T00:00:00Z"
    assert summary["latest_generated_at"] == "2024-04-03T00:00:00Z"
    assert summary["history_end_generated_at"] == "2024-04-03T00:00:00Z"
    assert summary["history_span_seconds"] == pytest.approx(86_400.0)
    assert summary["full_history_start_generated_at"] == "2024-04-01T00:00:00Z"
    assert summary["full_history_end_generated_at"] == "2024-04-03T00:00:00Z"
    assert summary["full_history_span_seconds"] == pytest.approx(172_800.0)
    assert summary["latest_model"] == "gbm_v3"
    assert summary["median_expected_probability"] == pytest.approx(0.625)
    assert summary["median_expected_return_bps"] == pytest.approx(7.0)
    assert summary["full_total"] == 3
    assert summary["full_accepted"] == 2
    assert summary["full_rejected"] == 1
    assert summary["full_acceptance_rate"] == pytest.approx(2 / 3)
    assert summary["history_generated_at_count"] == 2
    assert summary["history_missing_generated_at"] == 0
    assert summary["history_generated_at_coverage"] == pytest.approx(1.0)
    assert summary["full_history_generated_at_count"] == 3
    assert summary["full_history_missing_generated_at"] == 0
    assert summary["full_history_generated_at_coverage"] == pytest.approx(1.0)
    assert summary["longest_acceptance_streak"] == 1
    assert summary["longest_rejection_streak"] == 1
    assert summary["current_acceptance_streak"] == 1
    assert summary["current_rejection_streak"] == 0


def test_summarize_evaluation_payloads_reports_generated_at_coverage() -> None:
    evaluations = [
        {
            "accepted": True,
            "model_name": "gbm_v1",
            "candidate": {
                "metadata": {"generated_at": "2024-04-01T00:00:00Z"},
                "expected_probability": 0.5,
                "expected_return_bps": 5.0,
            },
        },
        {
            "accepted": False,
            "model_name": "gbm_v2",
            "candidate": {"metadata": {}},
        },
        {
            "accepted": True,
            "model_name": "gbm_v3",
            "candidate": {
                "metadata": {"timestamp": "2024-04-03T12:00:00Z"},
                "expected_probability": 0.6,
                "expected_return_bps": 6.0,
            },
        },
    ]

    summary = _summarize(evaluations)

    assert summary["history_generated_at_count"] == 2
    assert summary["history_missing_generated_at"] == 1
    assert summary["history_generated_at_coverage"] == pytest.approx(2 / 3)
    assert summary["full_history_generated_at_count"] == 2
    assert summary["full_history_missing_generated_at"] == 1
    assert summary["full_history_generated_at_coverage"] == pytest.approx(2 / 3)
    assert summary["history_start_generated_at"] == "2024-04-01T00:00:00Z"
    assert summary["history_end_generated_at"] == "2024-04-03T12:00:00Z"


def test_summarize_evaluation_payloads_includes_trimmed_history() -> None:
    evaluations = [
        {"accepted": True, "model_name": "gbm_v1"},
        {
            "accepted": False,
            "model_name": "gbm_v2",
            "reasons": ("latency",),
            "candidate": {"strategy": "swing"},
        },
        {
            "accepted": True,
            "model_name": "gbm_v3",
            "reasons": ["spread"],
            "candidate": {"strategy": "intraday"},
        },
    ]

    summary = _summarize(evaluations, history_limit=2)

    history = summary["history"]
    assert len(history) == 2
    assert [entry["model_name"] for entry in history] == ["gbm_v2", "gbm_v3"]
    assert history[0]["reasons"] == ["latency"]
    assert isinstance(history[1]["candidate"], dict)


def test_summarize_evaluation_payloads_tracks_longest_streaks() -> None:
    evaluations = [
        {"accepted": True},
        {"accepted": True},
        {"accepted": False},
        {"accepted": False},
        {"accepted": False},
        {"accepted": True},
    ]

    summary = _build_summary(evaluations)
    summary_model = summarize_evaluation_payloads(evaluations)
    assert isinstance(summary_model, DecisionEngineSummary)
    summary = summary_model.model_dump()
    summary = _summarize(evaluations)

    assert summary["longest_acceptance_streak"] == 2
    assert summary["longest_rejection_streak"] == 3
    assert summary["current_acceptance_streak"] == 1
    assert summary["current_rejection_streak"] == 0


def test_decision_engine_summary_model_validates_full_payload() -> None:
    summary_model = summarize_evaluation_payloads(
        _build_full_evaluations(), history_limit=5
    )

    summary_type = summary_model.__class__
    assert summary_type.__name__ == "DecisionEngineSummary"
    assert summary_model.total == 2
    assert summary_model.accepted == 1
    assert summary_model.model_breakdown is not None
    gbm_v1 = summary_model.model_breakdown["gbm_v1"]
    assert gbm_v1.metrics is not None
    assert gbm_v1.metrics["net_edge_bps"].accepted_sum == pytest.approx(5.0)


def test_decision_engine_summary_model_validates_minimal_payload() -> None:
    empty_summary = summarize_evaluation_payloads([], history_limit=3)
    summary_type = empty_summary.__class__
    assert summary_type.__name__ == "DecisionEngineSummary"
    assert empty_summary.total == 0
    assert empty_summary.accepted == 0
    assert empty_summary.model_dump()["rejection_reasons"] == {}

    single_summary = summarize_evaluation_payloads(
        [{"accepted": True, "model_name": "gbm_v1"}], history_limit=1
    )
    assert isinstance(single_summary, summary_type)
    assert single_summary.total == 1
    assert single_summary.accepted == 1
    assert single_summary.rejected == 0

    clone = summary_type.model_validate(single_summary.model_dump())
    assert clone.accepted == 1
