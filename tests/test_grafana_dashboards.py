from __future__ import annotations

import json
from pathlib import Path

import pytest

STAGE4_DASHBOARD = Path("deploy/grafana/provisioning/dashboards/stage4_multi_strategy.json")
STAGE5_DASHBOARD = Path("deploy/grafana/provisioning/dashboards/stage5_compliance_cost.json")
STAGE6_DASHBOARD = Path("deploy/grafana/provisioning/dashboards/stage6_resilience_operations.json")


@pytest.mark.parametrize(
    "metric_substring",
    [
        "bot_core_multi_strategy_latency_ms",
        "bot_core_multi_strategy_signals",
        "bot_core_multi_strategy_avg_abs_zscore",
        "bot_core_multi_strategy_allocation_error_pct",
        "bot_core_multi_strategy_spread_capture_bps",
        "bot_core_multi_strategy_secondary_delay_ms",
        "bot_core_multi_strategy_pnl_drawdown_pct",
        "bot_core_multi_strategy_risk_exposure_deviation_pct",
    ],
)
def test_stage4_dashboard_contains_key_metrics(metric_substring: str) -> None:
    dashboard = json.loads(STAGE4_DASHBOARD.read_text(encoding="utf-8"))
    assert dashboard["title"] == "Stage4 – Multi-Strategy Operations"
    panel_expressions: list[str] = []
    for panel in dashboard.get("panels", []):
        for target in panel.get("targets", []):
            expr = target.get("expr")
            if expr:
                panel_expressions.append(expr)
    assert any(metric_substring in expr for expr in panel_expressions), metric_substring


def test_stage4_dashboard_threshold_configuration() -> None:
    dashboard = json.loads(STAGE4_DASHBOARD.read_text(encoding="utf-8"))
    stat_panels = [panel for panel in dashboard["panels"] if panel["type"] == "stat"]
    assert stat_panels, "expected stat panels for arbitrage metrics"
    for panel in stat_panels:
        field_config = panel.get("fieldConfig", {}).get("defaults", {})
        thresholds = field_config.get("thresholds", {})
        steps = thresholds.get("steps", [])
        assert steps, "stat panel should define threshold steps"
        assert steps == sorted(steps, key=lambda step: step.get("value", 0))


@pytest.mark.parametrize(
    "metric_substring",
    [
        "bot_core_decision_latency_ms",
        "bot_core_trade_cost_bps",
        "bot_core_fill_rate_pct",
        "bot_core_key_rotation_due_in_days",
    ],
)
def test_stage5_dashboard_contains_expected_metrics(metric_substring: str) -> None:
    dashboard = json.loads(STAGE5_DASHBOARD.read_text(encoding="utf-8"))
    assert dashboard["title"] == "Stage5 – Compliance & Cost Control"
    expressions: list[str] = []
    for panel in dashboard.get("panels", []):
        for target in panel.get("targets", []):
            expr = target.get("expr")
            if expr:
                expressions.append(expr)
    assert any(metric_substring in expr for expr in expressions), metric_substring


@pytest.mark.parametrize(
    "metric_substring",
    [
        "bot_core_stage6_portfolio_weight",
        "bot_core_stage6_portfolio_signal_factor",
        "bot_core_stage6_failover_latency_ms",
        "bot_core_stage6_failover_success_ratio_pct",
        "bot_core_stage6_stress_lab_failure_count",
        "bot_core_stage6_slo_breach_rate_pct",
        "bot_core_stage6_stream_reconnect_attempt_rate",
        "bot_core_stage6_stream_reconnect_duration_p95_seconds",
    ],
)
def test_stage6_dashboard_contains_expected_metrics(metric_substring: str) -> None:
    dashboard = json.loads(STAGE6_DASHBOARD.read_text(encoding="utf-8"))
    assert dashboard["title"] == "Stage6 – Resilience & Portfolio Intelligence"
    expressions: list[str] = []
    for panel in dashboard.get("panels", []):
        for target in panel.get("targets", []):
            expr = target.get("expr")
            if expr:
                expressions.append(expr)
    assert any(metric_substring in expr for expr in expressions), metric_substring


def test_stage6_dashboard_stat_thresholds_sorted() -> None:
    dashboard = json.loads(STAGE6_DASHBOARD.read_text(encoding="utf-8"))
    stat_panels = [panel for panel in dashboard.get("panels", []) if panel.get("type") == "stat"]
    assert stat_panels, "expected stat panels for stage6 observability"
    for panel in stat_panels:
        thresholds = (
            panel.get("fieldConfig", {})
            .get("defaults", {})
            .get("thresholds", {})
            .get("steps", [])
        )
        assert thresholds, "each stat panel should define threshold steps"
        values = [step.get("value", 0) for step in thresholds]
        assert values == sorted(values), panel.get("title", "<stat panel>")
