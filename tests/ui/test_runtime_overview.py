import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6", reason="Wymagany PySide6 do testów UI")

from PySide6.QtCore import QObject, QUrl  # type: ignore[attr-defined]
from PySide6.QtQml import QQmlApplicationEngine  # type: ignore[attr-defined]

try:  # pragma: no cover - zależne od środowiska CI
    from PySide6.QtWidgets import QApplication  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - brak bibliotek systemowych
    pytest.skip(f"Brak zależności QtWidgets: {exc}", allow_module_level=True)

from core.monitoring.metrics_api import (
    GuardrailOverview,
    IOQueueTelemetry,
    RetrainingTelemetry,
    RuntimeTelemetrySnapshot,
)
from ui.backend.telemetry_provider import TelemetryProvider


def _sample_snapshot() -> RuntimeTelemetrySnapshot:
    generated = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    io_entries = (
        IOQueueTelemetry(
            environment="prod",
            queue="binance.spot",
            timeout_total=2.0,
            timeout_avg_seconds=1.25,
            rate_limit_wait_total=4.0,
            rate_limit_wait_avg_seconds=0.8,
            severity="warning",
        ),
    )
    guardrail = GuardrailOverview(
        total_queues=1,
        normal_queues=0,
        info_queues=0,
        warning_queues=1,
        error_queues=0,
        total_timeouts=2.0,
        total_rate_limit_waits=4.0,
    )
    retraining_entries = (
        RetrainingTelemetry(
            status="completed",
            runs=3,
            average_duration_seconds=42.5,
            average_drift_score=0.12,
        ),
    )
    return RuntimeTelemetrySnapshot(
        generated_at=generated,
        io_queues=io_entries,
        guardrail_overview=guardrail,
        retraining=retraining_entries,
    )


@pytest.mark.timeout(30)
def test_runtime_overview_renders_snapshot(tmp_path: Path) -> None:
    provider = TelemetryProvider(snapshot_loader=_sample_snapshot)
    app = QApplication.instance() or QApplication([])
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("telemetryProvider", provider)
    qml_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "dashboard" / "RuntimeOverview.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    assert engine.rootObjects(), "Nie udało się załadować RuntimeOverview.qml"
    root = engine.rootObjects()[0]

    ok = provider.refreshTelemetry()
    assert ok is True
    app.processEvents()

    last_updated = root.findChild(QObject, "runtimeOverviewLastUpdated")
    assert last_updated is not None
    assert "2025" in last_updated.property("text")

    guardrail_card = root.findChild(QObject, "runtimeOverviewGuardrailCard")
    assert guardrail_card is not None
    manual_button = root.findChild(QObject, "manualRefreshButton")
    assert manual_button is not None and manual_button.property("enabled") is True

    engine.deleteLater()
    app.quit()


def test_telemetry_provider_reports_errors() -> None:
    calls: list[int] = []

    def _failing_loader() -> RuntimeTelemetrySnapshot:
        calls.append(1)
        raise RuntimeError("registry unreachable")

    provider = TelemetryProvider(snapshot_loader=_failing_loader)
    result = provider.refreshTelemetry()
    assert result is False
    assert provider.errorMessage == "registry unreachable"
    assert len(calls) == 1
