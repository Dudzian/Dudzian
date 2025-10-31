"""Testy guardrail'i retrainingu i obsługi zdarzeń monitorujących."""
from __future__ import annotations

from pathlib import Path

import pytest

from bot_core.observability.metrics import MetricsRegistry

from core.monitoring import (
    AsyncIOGuardrails,
    AsyncIOMetricSet,
    DataDriftDetected,
    MissingDataDetected,
    RetrainingCycleCompleted,
    RetrainingMetricSet,
)


@pytest.fixture()
def guardrails_factory(tmp_path: Path):
    def _factory(**overrides):
        registry = MetricsRegistry()
        io_metrics = AsyncIOMetricSet(registry=registry)
        retrain_metrics = RetrainingMetricSet(registry=registry)
        ui_events: list[tuple[str, dict[str, object]]] = []
        guardrails = AsyncIOGuardrails(
            environment="demo",
            metrics=io_metrics,
            log_directory=tmp_path / "io",
            retraining_metrics=retrain_metrics,
            retraining_log_directory=tmp_path / "retraining",
            retraining_duration_warning_threshold=overrides.get(
                "retraining_duration_warning_threshold", 0.01
            ),
            drift_warning_threshold=overrides.get("drift_warning_threshold"),
            ui_notifier=lambda event, payload: ui_events.append((event, dict(payload))),
        )
        return guardrails, retrain_metrics, ui_events

    return _factory


def test_guardrails_records_retraining_completion(guardrails_factory) -> None:
    guardrails, retrain_metrics, ui_events = guardrails_factory()

    event = RetrainingCycleCompleted(
        source="scheduler",
        status="completed",
        duration_seconds=0.02,
        drift_score=0.15,
        metadata={"delay_seconds": 0.01},
    )

    guardrails(event)

    labels = {"environment": "demo", "status": "completed"}
    snapshot = retrain_metrics.duration_seconds.snapshot(labels=labels)
    assert snapshot.count == 1
    assert pytest.approx(snapshot.sum, rel=1e-3) == 0.02

    drift_snapshot = retrain_metrics.drift_score.snapshot(labels=labels)
    assert drift_snapshot.count == 1
    assert pytest.approx(drift_snapshot.sum, rel=1e-3) == 0.15

    assert ui_events, "Oczekiwano wygenerowania alertu UI dla retrainingu"
    event_name, payload = ui_events[0]
    assert event_name == "retraining_cycle_completed"
    assert payload["duration_seconds"] == pytest.approx(0.02, rel=1e-3)
    assert payload["metadata"] == {"delay_seconds": 0.01}

    log_file = guardrails._retraining_log_path / "events.log"  # noqa: SLF001 - test integracyjny
    assert log_file.exists()
    assert "RETRAINING duration" in log_file.read_text(encoding="utf-8")


def test_guardrails_handles_drift_and_missing_data(guardrails_factory) -> None:
    guardrails, retrain_metrics, ui_events = guardrails_factory(drift_warning_threshold=0.3)

    drift_event = DataDriftDetected(
        source="pipeline",
        drift_score=0.45,
        drift_threshold=0.35,
    )
    missing_event = MissingDataDetected(source="pipeline", missing_batches=5)

    guardrails.handle_monitoring_event(drift_event)
    guardrails.handle_monitoring_event(missing_event)

    drift_labels = {"environment": "demo", "source": "pipeline"}
    drift_snapshot = retrain_metrics.drift_score.snapshot(labels=drift_labels)
    assert drift_snapshot.count == 1
    assert drift_snapshot.sum == pytest.approx(0.45)

    assert any(event == "retraining_drift_detected" for event, _ in ui_events)
    assert any(event == "retraining_missing_data" for event, _ in ui_events)

    log_file = guardrails._retraining_log_path / "events.log"  # noqa: SLF001 - test integracyjny
    contents = log_file.read_text(encoding="utf-8")
    assert "RETRAINING DRIFT" in contents
    assert "RETRAINING MISSING_DATA" in contents
