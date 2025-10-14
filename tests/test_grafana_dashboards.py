from __future__ import annotations

import json
from pathlib import Path

import pytest

DASHBOARD_PATH = Path("deploy/grafana/provisioning/dashboards/stage4_multi_strategy.json")


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
    dashboard = json.loads(DASHBOARD_PATH.read_text(encoding="utf-8"))
    assert dashboard["title"] == "Stage4 – Multi-Strategy Operations"
    panel_expressions: list[str] = []
    for panel in dashboard.get("panels", []):
        for target in panel.get("targets", []):
            expr = target.get("expr")
            if expr:
                panel_expressions.append(expr)
    assert any(metric_substring in expr for expr in panel_expressions), metric_substring


def test_stage4_dashboard_threshold_configuration() -> None:
    dashboard = json.loads(DASHBOARD_PATH.read_text(encoding="utf-8"))
    stat_panels = [panel for panel in dashboard["panels"] if panel["type"] == "stat"]
    assert stat_panels, "expected stat panels for arbitrage metrics"
    for panel in stat_panels:
        field_config = panel.get("fieldConfig", {}).get("defaults", {})
        thresholds = field_config.get("thresholds", {})
        steps = thresholds.get("steps", [])
        assert steps, "stat panel should define threshold steps"
        assert steps == sorted(steps, key=lambda step: step.get("value", 0))
