from __future__ import annotations

import re
from pathlib import Path

from bot_core.observability.ui_metrics import RiskJournalMetricsExporter


def test_risk_journal_metric_metadata_matches_qml() -> None:
    exporter = RiskJournalMetricsExporter()
    metadata = exporter.metadata()

    qml_text = Path("ui/qml/dashboard/RiskJournalPanel.qml").read_text(
        encoding="utf-8"
    )
    qml_metric_keys = set(re.findall(r"metricValue\(\"([A-Za-z0-9_]+)\"", qml_text))

    qml_metric_keys_required = {"incompleteEntries", "riskFlagCounts"}

    assert qml_metric_keys_required.issubset(qml_metric_keys)
    assert qml_metric_keys_required.issubset(metadata.keys())

    assert metadata["incompleteEntries"]["labels"] == ("channel", "environment")
    assert metadata["incompleteSamples"]["labels"] == ("channel", "environment")
    assert metadata["riskFlagCounts"]["labels"] == (
        "channel",
        "environment",
        "riskFlag",
    )
