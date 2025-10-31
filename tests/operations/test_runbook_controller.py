from __future__ import annotations

import json
from pathlib import Path

import pytest

from ui.backend.runbook_controller import RunbookController


@pytest.fixture()
def runbook_environment(tmp_path: Path) -> tuple[Path, Path]:
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

    actions_dir.joinpath("restart_queue.py").write_text(
        """
from __future__ import annotations
from pathlib import Path

(Path(__file__).resolve().parent / "invoked.txt").write_text("ok", encoding="utf-8")
        """,
        encoding="utf-8",
    )

    return runbook_dir, actions_dir


def test_runbook_controller_executes_action(tmp_path: Path, runbook_environment: tuple[Path, Path]) -> None:
    runbook_dir, actions_dir = runbook_environment
    controller = RunbookController(runbook_directory=runbook_dir, actions_directory=actions_dir)

    assert controller.refreshAlerts(), "Oczekiwano odświeżenia alertów"
    alert = controller.alerts[0]
    assert alert["manualSteps"], "Manual steps powinny być wczytane"
    assert alert["automaticActions"], "Automatic actions powinny być wczytane"

    assert controller.runAction("strategy_incident_playbook", "restart_queue")
    status = json.loads(controller.actionStatus)
    assert status["status"] == "success"
    assert (actions_dir / "invoked.txt").exists(), "Skrypt powinien zostać uruchomiony"


def test_runbook_controller_missing_action(tmp_path: Path, runbook_environment: tuple[Path, Path]) -> None:
    runbook_dir, actions_dir = runbook_environment
    controller = RunbookController(runbook_directory=runbook_dir, actions_directory=actions_dir)

    controller.refreshAlerts()
    assert not controller.runAction("strategy_incident_playbook", "unknown"), "Akcja nie powinna istnieć"
    payload = json.loads(controller.actionStatus)
    assert payload["status"] == "error"
    assert "Nie znaleziono akcji" in payload["message"]
