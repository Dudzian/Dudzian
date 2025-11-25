from __future__ import annotations

from bot_core.observability.metrics import MetricsRegistry
from bot_core.observability.ui_metrics import RiskJournalMetricsExporter


def test_risk_flag_counts_clear_missing_series() -> None:
    registry = MetricsRegistry()
    exporter = RiskJournalMetricsExporter(registry=registry)

    labels = {"environment": "prod"}

    exporter.record(
        state="ok",
        incomplete_entries=0,
        incomplete_samples=0,
        risk_flag_counts={"drawdown_watch": 2, "stress_override": 1},
        labels=labels,
    )

    exporter.record(
        state="ok",
        incomplete_entries=0,
        incomplete_samples=0,
        risk_flag_counts={"stress_override": 3},
        labels=labels,
    )

    metric = registry.get("bot_ui_risk_journal_risk_flag_entries_total")
    assert metric.value(
        labels={
            "channel": "risk_journal",
            "environment": "prod",
            "riskFlag": "drawdown_watch",
        }
    ) == 0.0
    assert metric.value(
        labels={
            "channel": "risk_journal",
            "environment": "prod",
            "riskFlag": "stress_override",
        }
    ) == 3.0


def test_risk_flag_counts_clear_when_payload_missing() -> None:
    registry = MetricsRegistry()
    exporter = RiskJournalMetricsExporter(registry=registry)

    labels = {"environment": "dev"}

    exporter.record(
        state="warning",
        incomplete_entries=0,
        incomplete_samples=0,
        risk_flag_counts={"fat_finger": 1},
        labels=labels,
    )

    exporter.record(
        state="ok",
        incomplete_entries=0,
        incomplete_samples=0,
        risk_flag_counts=None,
        labels=labels,
    )

    metric = registry.get("bot_ui_risk_journal_risk_flag_entries_total")
    assert metric.value(
        labels={
            "channel": "risk_journal",
            "environment": "dev",
            "riskFlag": "fat_finger",
        }
    ) == 0.0
