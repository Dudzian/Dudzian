from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "exchange, expected_fragments",
    [
        (
            "kraken",
            [
                "# Kraken – Runbook go-live",
                "## Runbook go-live",
                "## Checklist licencyjna",
                "paper_exchange_metrics",
                "paper_exchange_metrics.summary.status",
                "paper_exchange_metrics.summary.breached_entrypoint_names",
                "paper_exchange_metrics.summary.breached_thresholds_entrypoints",
                "paper_exchange_metrics.summary.missing_metrics_entrypoints",
                "paper_exchange_metrics.summary.breach_counts_by_metric",
                "paper_exchange_metrics.summary.threshold_breach_counts",
                "paper_exchange_metrics.summary.missing_metric_counts",
                "paper_exchange_metrics.summary.monitored_entrypoint_names",
                "paper_exchange_metrics.summary.metric_coverage_entrypoints.bot_exchange_health_status",
                "paper_exchange_metrics.summary.metric_coverage_ratio",
                "paper_exchange_metrics.summary.metric_coverage_score",
                "paper_exchange_metrics.summary.threshold_coverage_score",
                "paper_exchange_metrics.summary.monitored_metric_names",
                "paper_exchange_metrics.summary.network_error_severity_totals",
                "paper_exchange_metrics.summary.network_error_severity_coverage_entrypoints",
                "paper_exchange_metrics.summary.network_error_severity_coverage_ratio",
                "paper_exchange_metrics.summary.missing_error_severity_counts",
                "paper_exchange_metrics.summary.missing_error_severity_entrypoints",
                "paper_exchange_metrics.kraken_desktop_paper.status",
                "breaches",
                "warnings",
                "tests/exchanges/test_kraken_signing.py",
                "execution.paper_profiles.kraken_paper",
                "execution.paper_profiles.kraken_paper.metrics.thresholds",
                "execution.trading_profiles.kraken_desktop",
            ],
        ),
        (
            "okx",
            [
                "# OKX – Runbook go-live",
                "## Checklist specyficzna",
                "paper_exchange_metrics.okx_desktop_paper",
                "paper_exchange_metrics.summary.status",
                "paper_exchange_metrics.summary.breached_entrypoint_names",
                "paper_exchange_metrics.summary.breached_thresholds_entrypoints",
                "paper_exchange_metrics.summary.missing_metrics_entrypoints",
                "paper_exchange_metrics.summary.breach_counts_by_metric",
                "paper_exchange_metrics.summary.threshold_breach_counts",
                "paper_exchange_metrics.summary.missing_metric_counts",
                "paper_exchange_metrics.summary.monitored_entrypoint_names",
                "paper_exchange_metrics.summary.metric_coverage_entrypoints.bot_exchange_errors_total",
                "paper_exchange_metrics.summary.metric_coverage_entrypoints.bot_exchange_health_status",
                "paper_exchange_metrics.summary.metric_coverage_ratio",
                "paper_exchange_metrics.summary.metric_coverage_score",
                "paper_exchange_metrics.summary.threshold_coverage_score",
                "paper_exchange_metrics.summary.network_error_severity_totals",
                "paper_exchange_metrics.summary.network_error_severity_coverage_entrypoints",
                "paper_exchange_metrics.summary.network_error_severity_coverage_ratio",
                "paper_exchange_metrics.summary.missing_error_severity_counts",
                "paper_exchange_metrics.summary.missing_error_severity_entrypoints",
                "paper_exchange_metrics.okx_desktop_paper.status",
                "breaches",
                "tests/integration/exchanges/test_okx.py",
                "tests/exchanges/test_okx_signing.py",
                "execution.paper_profiles.okx_paper",
                "execution.paper_profiles.okx_paper.metrics.thresholds",
            ],
        ),
        (
            "bybit",
            [
                "# Bybit – Runbook go-live",
                "## Checklist licencyjna",
                "io_queue.exchanges.bybit_spot",
                "tests/integration/exchanges/test_bybit.py",
                "tests/exchanges/test_bybit_signing.py",
                "execution.paper_profiles.bybit_paper",
                "execution.paper_profiles.bybit_paper.metrics.thresholds",
                "paper_exchange_metrics.summary.status",
                "paper_exchange_metrics.summary.breached_entrypoint_names",
                "paper_exchange_metrics.summary.breached_thresholds_entrypoints",
                "paper_exchange_metrics.summary.missing_metrics_entrypoints",
                "paper_exchange_metrics.summary.breach_counts_by_metric",
                "paper_exchange_metrics.summary.threshold_breach_counts",
                "paper_exchange_metrics.summary.missing_metric_counts",
                "paper_exchange_metrics.summary.monitored_entrypoint_names",
                "paper_exchange_metrics.summary.metric_coverage_entrypoints.bot_exchange_rate_limited_total",
                "paper_exchange_metrics.summary.metric_coverage_entrypoints.bot_exchange_health_status",
                "paper_exchange_metrics.summary.metric_coverage_ratio",
                "paper_exchange_metrics.summary.metric_coverage_score",
                "paper_exchange_metrics.summary.threshold_coverage_score",
                "paper_exchange_metrics.summary.network_error_severity_totals",
                "paper_exchange_metrics.summary.network_error_severity_coverage_entrypoints",
                "paper_exchange_metrics.summary.network_error_severity_coverage_ratio",
                "paper_exchange_metrics.summary.missing_error_severity_counts",
                "paper_exchange_metrics.summary.missing_error_severity_entrypoints",
                "paper_exchange_metrics.bybit_desktop_paper.status",
                "breaches",
            ],
        ),
    ],
)
def test_runbook_documents(exchange: str, expected_fragments: list[str]) -> None:
    path = Path("docs/deployment") / f"{exchange}_go_live.md"
    assert path.exists(), f"Brak runbooka dla {exchange}"
    content = path.read_text(encoding="utf-8")
    for fragment in expected_fragments:
        assert fragment in content, f"Nie znaleziono fragmentu '{fragment}' w runbooku {exchange}"
    assert "pytest tests/integration/exchanges/test_" + exchange in content
