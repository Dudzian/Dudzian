import ast
from pathlib import Path

import pytest

from ui.backend.runbook_controller import RunbookController, _load_metadata


def test_load_metadata_parses_actions_list(tmp_path: Path) -> None:
    pytest.importorskip("yaml", reason="PyYAML required for metadata parsing")
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    metadata_path = metadata_dir / "strategy_incident_playbook.yml"
    metadata_path.write_text(
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

    payload = _load_metadata(metadata_dir)

    assert "strategy_incident_playbook" in payload
    metadata = payload["strategy_incident_playbook"]
    actions = metadata["actions"]
    manual_steps = metadata["manual_steps"]

    assert manual_steps == ("Sprawdź limit zapytań",)
    assert len(actions) == 1
    action = actions[0]
    assert action.identifier == "restart_queue"
    assert action.label == "Restartuj kolejkę"
    assert action.script.name == "restart_queue.py"


def test_runbook_controller_exposes_action_methods() -> None:
    pytest.importorskip("PySide6", reason="PySide6 required for RunbookController class")
    assert hasattr(RunbookController, "runAction")
    assert hasattr(RunbookController, "openRunbook")
    assert hasattr(RunbookController, "_map_alerts")


def test_runbook_controller_ast_has_methods() -> None:
    source = (Path(__file__).resolve().parents[2] / "ui" / "backend" / "runbook_controller.py").read_text(
        encoding="utf-8"
    )
    module = ast.parse(source)
    class_node = next(
        node for node in ast.walk(module) if isinstance(node, ast.ClassDef) and node.name == "RunbookController"
    )
    method_names = {node.name for node in class_node.body if isinstance(node, ast.FunctionDef)}
    assert "refreshAlerts" in method_names
    assert "runAction" in method_names
    assert "openRunbook" in method_names
    assert "_map_alerts" in method_names
    refresh = next(
        node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "refreshAlerts"
    )
    nested_names = {node.name for node in refresh.body if isinstance(node, ast.FunctionDef)}
    assert "runAction" not in nested_names
    assert "openRunbook" not in nested_names
