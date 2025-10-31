from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from core.monitoring.events import (
    DataDriftDetected,
    MissingDataDetected,
    RetrainingCycleCompleted,
    RetrainingDelayInjected,
)
from core.reporting.retraining_reporter import RetrainingReport
from core.runtime.retraining_scheduler import RetrainingRunOutcome
from core.ml.training_pipeline import TrainingPipelineResult


pytestmark = pytest.mark.retraining


class _DummyModel:
    def fit(self, samples, targets):  # pragma: no cover - nieużywane w testach
        raise NotImplementedError

    def batch_predict(self, samples):  # pragma: no cover - nieużywane w testach
        raise NotImplementedError


def test_retraining_report_generates_markdown_and_json(tmp_path) -> None:
    started = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)
    finished = started + timedelta(seconds=42)

    fallback_chain = (
        {
            "backend": "lightgbm",
            "message": "module lightgbm nie znaleziony",
            "install_hint": "pip install lightgbm",
        },
    )
    training_result = TrainingPipelineResult(
        backend="reference",
        model=_DummyModel(),
        fallback_chain=fallback_chain,
        log_path=None,
        validation_log_path=None,
    )

    events = [
        MissingDataDetected(source="pipeline", missing_batches=2),
        RetrainingDelayInjected(reason="chaos_delay", delay_seconds=1.5),
        DataDriftDetected(source="pipeline", drift_score=0.42, drift_threshold=0.35),
        RetrainingCycleCompleted(
            source="scheduler",
            status="completed",
            duration_seconds=12.34,
            drift_score=0.42,
            metadata={"delay_seconds": 1.5},
        ),
    ]

    outcome = RetrainingRunOutcome(
        status="completed",
        result=training_result,
        reason=None,
        delay_seconds=1.5,
        drift_score=0.42,
        events=tuple(events),
    )

    report = RetrainingReport.from_execution(
        started_at=started,
        finished_at=finished,
        outcome=outcome,
        training_result=training_result,
        events=events,
        dataset_metadata={"row_count": 10, "feature_names": ["a", "b", "c"]},
    )

    markdown = report.to_markdown()
    assert "Raport cyklu retreningu" in markdown
    assert "Dryf danych" in markdown
    assert "Aktywowano fallback backendów" in "\n".join(report.alerts)

    md_path = report.write_markdown(tmp_path)
    json_path = report.write_json(tmp_path)
    assert md_path.exists()
    assert json_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["kpi"]["fallback_count"] == 1
    assert any(event["name"] == "DataDriftDetected" for event in payload["events"])


def test_retraining_report_captures_failure_reason() -> None:
    outcome = RetrainingRunOutcome(
        status="skipped",
        result=None,
        reason="missing_data",
        delay_seconds=0.0,
        drift_score=None,
        events=(),
    )

    report = RetrainingReport.from_execution(
        started_at=None,
        finished_at=None,
        outcome=outcome,
        training_result=None,
        events=(),
        dataset_metadata={},
    )

    assert any("missing_data" in error for error in report.errors)
