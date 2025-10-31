from pathlib import Path

import pytest

from bot_core.observability.metrics import MetricsRegistry
from core.monitoring.metrics import AsyncIOMetricSet
from core.reporting.guardrails_reporter import GuardrailReport


def _write_log(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_guardrail_reporter_generates_summary(tmp_path):
    registry = MetricsRegistry()
    metrics = AsyncIOMetricSet(registry=registry)
    labels = {"queue": "binance_spot", "environment": "paper"}
    metrics.rate_limit_wait_total.inc(labels=labels)
    metrics.rate_limit_wait_seconds.observe(0.5, labels=labels)
    metrics.timeout_total.inc(labels=labels)
    metrics.timeout_duration.observe(12.0, labels=labels)

    log_path = tmp_path / "logs" / "events.log"
    _write_log(
        log_path,
        "2025-01-01T10:00:00+0000 WARNING RATE_LIMIT queue=binance_spot waited=0.500000s streak=1\n",
    )

    report = GuardrailReport.from_sources(
        registry=registry,
        log_directory=log_path.parent,
        environment_hint="paper",
    )

    assert report.summaries, "Powinien istnieć co najmniej jeden wpis podsumowania"
    summary = report.summaries[0]
    assert summary.queue == "binance_spot"
    assert summary.timeout_total == pytest.approx(1.0)
    assert summary.timeout_avg_seconds == pytest.approx(12.0)
    assert summary.rate_limit_wait_total == pytest.approx(1.0)
    assert summary.rate_limit_wait_avg_seconds == pytest.approx(0.5)

    assert report.logs, "Oczekiwano parsowania wpisów logu guardrail"
    log_record = report.logs[0]
    assert log_record.event == "RATE_LIMIT"
    assert log_record.metadata.get("waited") == pytest.approx(0.5)

    markdown = report.to_markdown()
    assert "Raport guardrail'i" in markdown
    output_path = report.write_markdown(tmp_path / "reports")
    assert output_path.exists()
    assert "binance_spot" in output_path.read_text(encoding="utf-8")
