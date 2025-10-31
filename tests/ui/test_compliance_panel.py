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
    ComplianceTelemetry,
    GuardrailOverview,
    RuntimeTelemetrySnapshot,
)
from ui.backend.compliance_controller import ComplianceController
from ui.backend.telemetry_provider import TelemetryProvider


def _sample_snapshot() -> RuntimeTelemetrySnapshot:
    generated = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    return RuntimeTelemetrySnapshot(
        generated_at=generated,
        io_queues=(),
        guardrail_overview=GuardrailOverview(0, 0, 0, 0, 0, 0.0, 0.0),
        retraining=(),
        compliance=ComplianceTelemetry(
            total_violations=2.0,
            by_severity={"warning": 1.0, "critical": 1.0},
            by_rule={"KYC_MISSING_FIELDS": 1.0, "AML_BLOCKED_COUNTRY": 1.0},
        ),
    )


@pytest.mark.timeout(30)
def test_compliance_panel_renders_and_triggers_audit(tmp_path: Path) -> None:
    provider = TelemetryProvider(snapshot_loader=_sample_snapshot)
    controller = ComplianceController(
        strategy_provider=lambda: {"name": "grid", "tags": ["sanctioned"], "exchange": "binance"},
        datasource_provider=lambda: ("darkpool",),
        transactions_provider=lambda: (
            {
                "id": "tx-1",
                "value_usd": 50000,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        ),
        kyc_profile_provider=lambda: {"full_name": "Jan Kowalski", "country": "IR"},
    )

    app = QApplication.instance() or QApplication([])
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("telemetryProvider", provider)
    engine.rootContext().setContextProperty("complianceController", controller)
    qml_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "dashboard" / "CompliancePanel.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    assert engine.rootObjects(), "Nie udało się załadować CompliancePanel.qml"
    root = engine.rootObjects()[0]

    provider.refreshTelemetry()
    controller.refreshAudit()
    app.processEvents()

    audit_button = root.findChild(QObject, "complianceAuditButton")
    assert audit_button is not None
    assert audit_button.property("enabled") is True

    status_label = root.findChild(QObject, "complianceStatusValue_kycStatus")
    assert status_label is not None
    assert "Błąd" in status_label.property("text")

    total_label = root.findChild(QObject, "complianceTotalViolations")
    assert total_label is not None
    assert "2" in total_label.property("text")

    no_findings = root.findChild(QObject, "complianceNoFindings")
    assert no_findings is not None
    assert not no_findings.property("visible")

    engine.deleteLater()
    app.quit()
