"""Source-level smoke checks for the existing PySide6/QML UI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ui.pyside_app.app import AppOptions

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_SOURCE = REPO_ROOT / "ui" / "pyside_app" / "smoke.py"
PLAN_SOURCE = REPO_ROOT / "scripts" / "ui_preview_launch_plan.py"
APP_SOURCE = REPO_ROOT / "ui" / "pyside_app" / "app.py"
FORBIDDEN_SOURCE_TOKENS = (
    "create_order",
    "fetch_balance",
    "load_markets",
    "keyring",
    "dotenv",
    "shell=True",
    "subprocess.run",
)


def _run_ui_smoke(*extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "ui.pyside_app",
            "--config",
            "ui/config/example.yaml",
            "--smoke",
            "--offscreen",
            *extra_args,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )


def _smoke_payload(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    result.stdout.encode("cp1252")
    assert result.stdout.strip(), result.stderr
    return json.loads(result.stdout)


def test_smoke_flags_are_available_in_parser() -> None:
    options = AppOptions.parse(["--config", "ui/config/example.yaml", "--smoke", "--offscreen"])

    assert options.smoke is True
    assert options.offscreen is True
    assert options.enable_cloud_runtime is False


def test_source_smoke_finishes_and_reports_safety_contract() -> None:
    result = _run_ui_smoke()
    payload = _smoke_payload(result)

    if result.returncode != 0 and any("libGL.so.1" in issue for issue in payload["issues"]):
        pytest.skip("Qt runtime unavailable in this headless environment: missing libGL.so.1")

    assert result.returncode == 0, result.stderr or result.stdout
    assert payload["status"] == "ok"
    assert payload["ui_loaded"] is True
    assert payload["qml_loaded"] is True
    assert payload["runtime_loop_started"] is False
    assert payload["production_runtime_loop_started"] is False
    assert payload["exchange_io"] == "disabled"
    assert payload["order_submission"] == "disabled"
    assert payload["api_keys_required"] is False
    assert payload["secrets_read"] is False
    assert payload["keychain_read"] is False
    assert payload["env_values_read"] is False
    assert payload["dot_env_read"] is False
    assert payload["issues"] == []


def test_smoke_blocks_live_runtime_flag_without_qt_bootstrap() -> None:
    result = _run_ui_smoke("--enable-cloud-runtime")
    payload = _smoke_payload(result)

    assert result.returncode == 2
    assert payload["status"] == "blocked"
    assert payload["ui_loaded"] is False
    assert payload["qml_loaded"] is False
    assert payload["runtime_loop_started"] is False
    assert payload["exchange_io"] == "disabled"
    assert payload["order_submission"] == "disabled"
    assert payload["api_keys_required"] is False
    assert payload["live_mode_allowed"] is False
    assert payload["issues"] == ["smoke_mode_blocks_enable_cloud_runtime"]


def test_smoke_and_plan_sources_have_no_forbidden_runtime_or_secret_calls() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in (SMOKE_SOURCE, PLAN_SOURCE, APP_SOURCE)
    )

    for token in FORBIDDEN_SOURCE_TOKENS:
        assert token not in source
