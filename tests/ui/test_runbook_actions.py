import os
import time
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

from bot_core.observability.guardrail_models import GuardrailLogRecord, GuardrailReport, GuardrailSummary
from ui.backend.runbook_controller import RunbookController


class _StaticEndpoint:
    def __init__(self, report: GuardrailReport) -> None:
        self._report = report

    def build_report(self) -> GuardrailReport:
        return self._report


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

    # Deterministyczny raport: ma wskazać runbook "strategy_incident_playbook"
    report = GuardrailReport(
        summaries=[
            GuardrailSummary(
                severity="error",
                title="Strategy incident",
                description="Synthetic incident for UI test",
                affected_components=["strategy"],
            )
        ],
        log_records=[
            GuardrailLogRecord(
                severity="error",
                message="Synthetic error",
                timestamp="2024-03-01T10:00:00Z",
            )
        ],
    )
    controller = RunbookController(
        runbook_directory=runbook_dir, actions_directory=actions_dir, endpoint=_StaticEndpoint(report)
    )
    assert controller.refreshAlerts()

    app = QApplication.instance() or QApplication([])
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("runbookController", controller)
    qml_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "dashboard" / "RunbookPanel.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    assert engine.rootObjects(), "Nie udało się załadować RunbookPanel.qml"

    root = engine.rootObjects()[0]
    deadline = time.monotonic() + 5.0
    button = None
    while time.monotonic() < deadline:
        app.processEvents()
        button = root.findChild(QObject, "runbookActionButton_restart_queue")
        if button is not None:
            break
        time.sleep(0.05)
    assert button is not None, "Przycisk akcji nie został wyrenderowany"

    QMetaObject.invokeMethod(button, "click")
    app.processEvents()

    assert (actions_dir / "ui_action_invoked.txt").exists(), "Skrypt nie został wykonany przez przycisk"
