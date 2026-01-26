import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytestmark = pytest.mark.qml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:  # pragma: no cover - zależne od środowiska CI
    from PySide6.QtCore import QObject, QUrl  # type: ignore[attr-defined]
    from PySide6.QtQml import QQmlApplicationEngine  # type: ignore[attr-defined]
    from PySide6.QtWidgets import QApplication  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - brak Qt
    QObject = QUrl = QQmlApplicationEngine = QApplication = None  # type: ignore[assignment]

from bot_core.observability.metrics import MetricsRegistry
from core.monitoring.metrics import AsyncIOMetricSet
from core.reporting.guardrails_reporter import (
    GuardrailLogRecord,
    GuardrailQueueSummary,
    GuardrailReport,
    GuardrailReportEndpoint,
)
from ui.backend.runbook_controller import RunbookController


def _build_sample_report() -> GuardrailReport:
    generated_at = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    summaries = (
        GuardrailQueueSummary(
            environment="paper",
            queue="binance_spot",
            rate_limit_wait_total=3.0,
            rate_limit_wait_avg_seconds=1.5,
            timeout_total=2.0,
            timeout_avg_seconds=5.0,
        ),
    )
    logs = (
        GuardrailLogRecord(
            timestamp=generated_at,
            level="ERROR",
            message="TIMEOUT queue=binance_spot waited=5.000000s",
            event="TIMEOUT",
            metadata={"queue": "binance_spot", "environment": "paper"},
        ),
    )
    recommendations = (
        "Rozważ zwiększenie współbieżności lub obniżenie burst dla kolejki binance_spot.",
    )
    return GuardrailReport(
        generated_at=generated_at,
        summaries=summaries,
        logs=logs,
        recommendations=recommendations,
    )


class _StaticEndpoint(GuardrailReportEndpoint):
    def __init__(self, report: GuardrailReport) -> None:
        super().__init__(report_factory=lambda: report)


@pytest.mark.skipif(QObject is None, reason="Wymagany PySide6 do testów UI")
def test_runbook_panel_qml_load(tmp_path: Path) -> None:
    report = _build_sample_report()
    runbook_dir = tmp_path / "runbooks"
    runbook_dir.mkdir()
    (runbook_dir / "strategy_incident_playbook.md").write_text("# Strategia L1/L2\n", encoding="utf-8")
    (runbook_dir / "autotrade_threshold_calibration.md").write_text("# Kalibracja progów\n", encoding="utf-8")
    (runbook_dir / "oem_license_provisioning.md").write_text("# Provisioning OEM\n", encoding="utf-8")

    controller = RunbookController(report_endpoint=_StaticEndpoint(report), runbook_directory=runbook_dir)
    assert controller.refreshAlerts()

    app = QApplication.instance() or QApplication([])
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("runbookController", controller)
    qml_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "dashboard" / "RunbookPanel.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    assert engine.rootObjects(), "Nie udało się załadować RunbookPanel.qml"

    root = engine.rootObjects()[0]
    repeater = root.findChild(QObject, "runbookPanelRepeater")
    assert repeater is not None, "Brak kontenera alertów"
    assert repeater.property("count") >= 1

    label = root.findChild(QObject, "runbookPanelLastUpdated")
    assert label is not None
    text = label.property("text")
    assert "Ostatnia aktualizacja" in text


def test_runbook_controller_mapping(tmp_path: Path) -> None:
    report = _build_sample_report()
    runbook_dir = tmp_path / "ops"
    runbook_dir.mkdir()
    (runbook_dir / "strategy_incident_playbook.md").write_text("# Strategia L1/L2\n", encoding="utf-8")
    (runbook_dir / "autotrade_threshold_calibration.md").write_text("# Kalibracja progów\n", encoding="utf-8")
    (runbook_dir / "oem_license_provisioning.md").write_text("# Provisioning OEM\n", encoding="utf-8")

    controller = RunbookController(report_endpoint=_StaticEndpoint(report), runbook_directory=runbook_dir)
    assert controller.refreshAlerts()

    alerts = controller.alerts
    assert len(alerts) >= 2
    assert any(alert.get("runbookTitle") == "Strategia L1/L2" for alert in alerts)
    assert controller.lastUpdated


def test_guardrail_report_endpoint_fastapi(tmp_path: Path) -> None:
    fastapi = pytest.importorskip("fastapi", reason="Wymagany fastapi do testu endpointu")
    testclient = pytest.importorskip("fastapi.testclient", reason="Wymagany fastapi.testclient")

    registry = MetricsRegistry()
    metrics = AsyncIOMetricSet(registry=registry)
    labels = {"queue": "binance_spot", "environment": "paper"}
    metrics.rate_limit_wait_total.inc(labels=labels)
    metrics.timeout_total.inc(labels=labels)
    metrics.timeout_duration.observe(3.5, labels=labels)

    log_path = tmp_path / "logs" / "events.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "2025-01-01T10:00:00+0000 ERROR TIMEOUT queue=binance_spot waited=3.500000s\n",
        encoding="utf-8",
    )

    endpoint = GuardrailReportEndpoint(
        registry=registry,
        log_directory=log_path.parent,
        environment_hint="paper",
    )

    app = fastapi.FastAPI()
    app.include_router(endpoint.as_fastapi_router())
    client = testclient.TestClient(app)

    response = client.get("/guardrails/report")
    assert response.status_code == 200
    payload = response.json()
    assert "summaries" in payload
    assert payload["summaries"], "Oczekiwano co najmniej jednego wpisu"
