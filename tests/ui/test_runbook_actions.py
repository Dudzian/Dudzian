import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytestmark = pytest.mark.qml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:  # pragma: no cover - zależne od środowiska CI
    from PySide6.QtCore import QObject, QMetaObject, QUrl
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtWidgets import QApplication
except Exception:  # pragma: no cover - brak Qt
    QObject = QMetaObject = QUrl = QQmlApplicationEngine = QApplication = None  # type: ignore[assignment]

from core.reporting.guardrails_reporter import (
    GuardrailLogRecord,
    GuardrailReport,
    GuardrailReportEndpoint,
)
from ui.backend.runbook_controller import RunbookController


def _build_sample_report() -> GuardrailReport:
    generated_at = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    logs = (
        GuardrailLogRecord(
            timestamp=generated_at,
            level="ERROR",
            message="TIMEOUT queue=binance_spot waited=5.000000s",
            event="TIMEOUT",
            metadata={"queue": "binance_spot", "environment": "paper"},
        ),
    )
    return GuardrailReport(
        generated_at=generated_at,
        summaries=(),
        logs=logs,
        recommendations=(),
    )


class _StaticEndpoint(GuardrailReportEndpoint):
    def __init__(self, report: GuardrailReport) -> None:
        super().__init__(report_factory=lambda: report)


@pytest.mark.skipif(QObject is None, reason="Wymagany PySide6 do testów UI")
def test_runbook_panel_action_button_executes_script(tmp_path: Path) -> None:
    runbook_dir = tmp_path / "runbooks"
    metadata_dir = runbook_dir / "metadata"
    actions_dir = tmp_path / "actions"
    runbook_dir.mkdir()
    metadata_dir.mkdir()
    actions_dir.mkdir()

    (runbook_dir / "strategy_incident_playbook.md").write_text("# Strategia L1/L2\n", encoding="utf-8")
    metadata_dir.joinpath("strategy_incident_playbook.yml").write_text(
        """
        id: strategy_incident_playbook
        automatic_actions:
          - id: restart_queue
            label: Restartuj kolejkę
            script: restart_queue.py
        manual_steps:
          - Sprawdź limit zapytań
        """,
        encoding="utf-8",
    )

    script_path = actions_dir / "restart_queue.py"
    script_path.write_text(
        """
from __future__ import annotations
from pathlib import Path

(Path(__file__).resolve().parent / "ui_action_invoked.txt").write_text("ok", encoding="utf-8")
        """,
        encoding="utf-8",
    )

    report = _build_sample_report()
    controller = RunbookController(
        report_endpoint=_StaticEndpoint(report),
        runbook_directory=runbook_dir,
        actions_directory=actions_dir,
    )
    assert controller.refreshAlerts()

    app = QApplication.instance() or QApplication([])
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("runbookController", controller)
    qml_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "dashboard" / "RunbookPanel.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    assert engine.rootObjects(), "Nie udało się załadować RunbookPanel.qml"
    app.processEvents()

    root = engine.rootObjects()[0]
    deadline = time.monotonic() + 5.0
    button = None
    while time.monotonic() < deadline:
        app.processEvents()
        button = root.findChild(QObject, "runbookActionButton_restart_queue")
        if button is not None:
            break
        candidates = [
            obj
            for obj in root.findChildren(QObject)
            if (obj.objectName() or "").startswith("runbookActionButton_")
        ]
        if candidates:
            preferred = None
            for obj in candidates:
                if obj.objectName() == "runbookActionButton_restart_queue":
                    preferred = obj
                    break
            button = preferred or candidates[0]
            break
        time.sleep(0.01)
    if button is None:
        alerts = getattr(controller, "alerts", None)
        first = alerts[0] if isinstance(alerts, list) and alerts else None
        if isinstance(first, dict):
            auto_actions = first.get("automaticActions")
        else:
            auto_actions = getattr(first, "automaticActions", None)
        created_names = []
        try:
            created_names = [
                obj.objectName()
                for obj in root.findChildren(QObject)
                if (obj.objectName() or "").startswith("runbookActionButton_")
            ]
        except Exception:
            created_names = ["(failed to enumerate)"]
        pytest.fail(
            "Przycisk akcji nie został wyrenderowany. "
            f"alerts_type={type(alerts).__name__} "
            f"alerts_len={(len(alerts) if isinstance(alerts, list) else 'n/a')} "
            f"first_alert_type={type(first).__name__} "
            f"automaticActions={auto_actions!r} "
            f"created_action_buttons={created_names!r}"
        )

    print(f"Using runbook action button: {button.objectName()}")
    QMetaObject.invokeMethod(button, "click")
    app.processEvents()

    assert (actions_dir / "ui_action_invoked.txt").exists(), "Skrypt nie został wykonany przez przycisk"
