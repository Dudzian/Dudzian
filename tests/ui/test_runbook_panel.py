import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytestmark = [pytest.mark.qml, pytest.mark.timeout(30)]

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
from tests.ui._qml_tree import find_by_object_name
from tests.ui._qt_utils import force_qt_cleanup, teardown_qml_engine, wait_for


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
    (runbook_dir / "strategy_incident_playbook.md").write_text(
        "# Strategia L1/L2\n", encoding="utf-8"
    )
    (runbook_dir / "autotrade_threshold_calibration.md").write_text(
        "# Kalibracja progów\n", encoding="utf-8"
    )
    (runbook_dir / "oem_license_provisioning.md").write_text(
        "# Provisioning OEM\n", encoding="utf-8"
    )

    controller = RunbookController(
        report_endpoint=_StaticEndpoint(report), runbook_directory=runbook_dir
    )
    assert controller.refreshAlerts()

    app = QApplication.instance() or QApplication([])
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("runbookController", controller)
    qml_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "dashboard" / "RunbookPanel.qml"
    try:
        engine.load(QUrl.fromLocalFile(str(qml_path)))
        assert engine.rootObjects(), "Nie udało się załadować RunbookPanel.qml"

        root = engine.rootObjects()[0]
        timeout = 10.0 if sys.platform.startswith("win") else 5.0

        def _repeater_ready() -> object | None:
            repeater_obj = find_by_object_name(root, "runbookPanelRepeater")
            count_value = repeater_obj.property("count") if repeater_obj is not None else None
            if isinstance(count_value, int) and count_value >= 1:
                return repeater_obj
            return None

        try:
            repeater = wait_for(
                _repeater_ready,
                timeout_s=timeout,
                step_ms=10,
                process_events=app.processEvents,
                description="runbookPanelRepeater count >= 1",
            )
        except TimeoutError as exc:
            alerts = getattr(controller, "alerts", None)
            alerts_len = "n/a"
            if alerts is not None and hasattr(alerts, "__len__"):
                try:
                    alerts_len = len(alerts)
                except Exception:
                    alerts_len = "error"
            pytest.fail(
                "Brak kontenera alertów lub pusty repeater. "
                f"reason={exc!r} "
                f"alerts_type={type(alerts).__name__} "
                f"alerts_len={alerts_len} "
                f"lastUpdated={getattr(controller, 'lastUpdated', None)!r} "
                f"errorMessage={getattr(controller, 'errorMessage', None)!r}"
            )
        assert repeater is not None

        label = find_by_object_name(root, "runbookPanelLastUpdated")
        assert label is not None
        text = label.property("text")
        text = "" if text is None else str(text)
        assert text
        assert ("Ostatnia aktualizacja" in text) or (text == "runbookPanel.lastUpdated")
    finally:
        # Deterministycznie domknij QML obiekty (timer refresh + połączenia sygnałów),
        # aby uniknąć wycieków event-loop między modułami testów UI.
        teardown_qml_engine(
            engine,
            process_events=app.processEvents,
            context_properties_to_clear=("runbookController",),
        )
        del engine
        force_qt_cleanup(process_events=app.processEvents)


def test_runbook_controller_mapping(tmp_path: Path) -> None:
    report = _build_sample_report()
    runbook_dir = tmp_path / "ops"
    runbook_dir.mkdir()
    (runbook_dir / "strategy_incident_playbook.md").write_text(
        "# Strategia L1/L2\n", encoding="utf-8"
    )
    (runbook_dir / "autotrade_threshold_calibration.md").write_text(
        "# Kalibracja progów\n", encoding="utf-8"
    )
    (runbook_dir / "oem_license_provisioning.md").write_text(
        "# Provisioning OEM\n", encoding="utf-8"
    )

    controller = RunbookController(
        report_endpoint=_StaticEndpoint(report), runbook_directory=runbook_dir
    )
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
