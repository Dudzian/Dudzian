from __future__ import annotations

import json
from pathlib import Path

from bot_core.observability.ui_metrics import RiskJournalMetricsExporter


def test_risk_journal_grafana_dashboard_matches_metadata() -> None:
    metadata = RiskJournalMetricsExporter().metadata()
    metric_names = {entry["metric"] for entry in metadata.values()}

    dashboard_path = Path("docs/observability/grafana/risk_journal_health.json")
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))

    expressions: list[str] = []
    for panel in dashboard.get("panels", []):
        for target in panel.get("targets", []):
            expr = target.get("expr")
            if isinstance(expr, str):
                expressions.append(expr)

    assert expressions, "Brak zapytań PromQL w dashboardzie"

    for metric in metric_names:
        assert any(metric in expr for expr in expressions), metric

    risk_flag_metric = metadata["riskFlagCounts"]["metric"]
    risk_flag_exprs = [expr for expr in expressions if risk_flag_metric in expr]
    assert risk_flag_exprs, "Brak panelu dla riskFlagCounts"
    assert any("riskFlag" in expr for expr in risk_flag_exprs)
