from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts import ui_preview_launch_plan

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ui_preview_launch_plan.py"
FORBIDDEN_SOURCE_TOKENS = (
    "create_order",
    "fetch_balance",
    "load_markets",
    "keyring",
    "dotenv",
    "os.environ",
    "shell=True",
    "subprocess.run",
)


def _run_plan() -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    result.stdout.encode("cp1252")
    return json.loads(result.stdout)


def test_launch_plan_contract_and_entrypoint_detection() -> None:
    payload = _run_plan()

    assert payload["safety_contract_version"] == "ui_preview_launch_plan.v1"
    assert payload["status"] == "ok"
    assert payload["ui_entrypoint_found"] is True
    assert payload["ui_entrypoint_path"] == "ui/pyside_app/__main__.py"
    assert payload["qml_entrypoint_found"] is True
    assert payload["qml_entrypoint_path"] == "ui/pyside_app/qml/MainWindow.qml"
    assert payload["ui_framework"] == "PySide6/QML"


def test_launch_plan_safety_boundary() -> None:
    payload = _run_plan()

    assert payload["demo_mode_required"] is True
    assert payload["live_mode_allowed"] is False
    assert payload["exchange_io"] == "disabled"
    assert payload["order_submission"] == "disabled"
    assert payload["real_orders_submitted"] is False
    assert payload["api_keys_required"] is False
    assert payload["secrets_read"] is False
    assert payload["keychain_read"] is False
    assert payload["env_values_read"] is False
    assert payload["dot_env_read"] is False
    assert payload["runtime_loop_started"] is False
    assert payload["production_runtime_loop_started"] is False
    assert payload["command_execution_allowed"] is False
    assert payload["command_executed"] is False
    assert payload["subprocess_invoked"] is False
    assert payload["shell_used"] is False


def test_launch_plan_command_is_render_only_source_entrypoint() -> None:
    payload = _run_plan()

    assert payload["ui_launch_command_preview"] == [
        sys.executable,
        "-m",
        "ui.pyside_app",
        "--config",
        "ui/config/example.yaml",
    ]
    assert payload["issues"] == []


def test_build_launch_plan_does_not_depend_on_cwd(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    payload = ui_preview_launch_plan.build_launch_plan()

    assert payload["ui_entrypoint_found"] is True
    assert payload["ui_entrypoint_path"] == "ui/pyside_app/__main__.py"
    assert payload["command_execution_allowed"] is False


def test_plan_only_source_has_no_forbidden_runtime_or_secret_calls() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    for token in FORBIDDEN_SOURCE_TOKENS:
        assert token not in source
