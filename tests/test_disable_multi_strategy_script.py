from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.disable_multi_strategy import (
    COMPONENTS,
    DEFAULT_COMPONENT,
    DISABLE_FILENAME,
    main as disable_main,
    run as disable_scheduler,
)


ROOT = Path(__file__).resolve().parents[1]


def test_disable_scheduler_creates_override(tmp_path: Path) -> None:
    output_dir = tmp_path / "overrides"
    exit_code = disable_scheduler(
        [
            "--output-dir",
            str(output_dir),
            "--reason",
            "Awaria feedu danych",
            "--requested-by",
            "noc",
            "--ticket",
            "INC-2042",
            "--duration-minutes",
            "45",
        ]
    )

    assert exit_code == 0

    override_path = output_dir / DISABLE_FILENAME
    assert override_path.exists()
    payload = json.loads(override_path.read_text(encoding="utf-8"))
    assert payload["action"] == COMPONENTS[DEFAULT_COMPONENT].action
    assert payload["schema"] == COMPONENTS[DEFAULT_COMPONENT].schema
    assert payload["reason"] == "Awaria feedu danych"
    assert payload["ticket"] == "INC-2042"
    assert "expires_at" in payload

    if os.name != "nt":
        assert (override_path.stat().st_mode & 0o077) == 0

    with pytest.raises(FileExistsError):
        disable_scheduler([
            "--output-dir",
            str(output_dir),
            "--reason",
            "Ponowna próba",
        ])


def test_disable_decision_orchestrator_creates_override(tmp_path: Path) -> None:
    output_dir = tmp_path / "overrides"
    exit_code = disable_scheduler(
        [
            "--output-dir",
            str(output_dir),
            "--component",
            "decision_orchestrator",
            "--reason",
            "Fallback AI",
        ]
    )

    assert exit_code == 0

    override_path = output_dir / COMPONENTS["decision_orchestrator"].filename
    assert override_path.exists()
    payload = json.loads(override_path.read_text(encoding="utf-8"))
    assert payload["action"] == COMPONENTS["decision_orchestrator"].action
    assert payload["schema"] == COMPONENTS["decision_orchestrator"].schema
    assert payload["reason"] == "Fallback AI"


def test_cli_guard_annotates_component_in_error(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    output_dir = tmp_path / "overrides"
    output_dir.mkdir(parents=True)
    existing = output_dir / COMPONENTS["decision_orchestrator"].filename
    existing.write_text("{}", encoding="utf-8")

    with caplog.at_level("ERROR"):
        exit_code = disable_main(
            [
                "--output-dir",
                str(output_dir),
                "--component",
                "decision_orchestrator",
                "--reason",
                "Powtórna próba",
            ]
        )

    assert exit_code == 1
    assert any("komponentu decision_orchestrator" in message for message in caplog.messages)
