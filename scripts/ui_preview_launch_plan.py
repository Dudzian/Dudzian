"""Render a source-level UI preview launch plan without starting Qt.

The script is intentionally plan-only: it inspects repository paths and prints a
JSON safety contract for the existing PySide6/QML UI entrypoint. It does not
start a process, open a Qt window, read local credentials, or invoke runtime
loops.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Final

SAFETY_CONTRACT_VERSION: Final = "ui_preview_launch_plan.v1"
REPO_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_ENTRYPOINT: Final = REPO_ROOT / "ui" / "pyside_app" / "__main__.py"
DEFAULT_QML_ENTRYPOINT: Final = REPO_ROOT / "ui" / "pyside_app" / "qml" / "MainWindow.qml"
DEFAULT_CONFIG: Final = REPO_ROOT / "ui" / "config" / "example.yaml"


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _build_command_preview(entrypoint_found: bool, config_found: bool) -> list[str]:
    if not entrypoint_found or not config_found:
        return []
    return [
        sys.executable,
        "-m",
        "ui.pyside_app",
        "--config",
        _relative(DEFAULT_CONFIG),
    ]


def _build_smoke_command_preview(entrypoint_found: bool, config_found: bool) -> list[str]:
    if not entrypoint_found or not config_found:
        return []
    return [
        sys.executable,
        "-m",
        "ui.pyside_app",
        "--config",
        _relative(DEFAULT_CONFIG),
        "--smoke",
        "--offscreen",
    ]


def build_launch_plan() -> dict[str, Any]:
    """Return the UI preview safety contract as plain data."""

    entrypoint_found = DEFAULT_ENTRYPOINT.exists()
    qml_found = DEFAULT_QML_ENTRYPOINT.exists()
    config_found = DEFAULT_CONFIG.exists()
    issues: list[str] = []
    if not entrypoint_found:
        issues.append("missing_ui_entrypoint:ui/pyside_app/__main__.py")
    if not qml_found:
        issues.append("missing_qml_entrypoint:ui/pyside_app/qml/MainWindow.qml")
    if not config_found:
        issues.append("missing_ui_config:ui/config/example.yaml")

    return {
        "safety_contract_version": SAFETY_CONTRACT_VERSION,
        "status": "ok" if not issues else "blocked",
        "ui_entrypoint_found": entrypoint_found,
        "ui_entrypoint_path": _relative(DEFAULT_ENTRYPOINT) if entrypoint_found else "",
        "qml_entrypoint_found": qml_found,
        "qml_entrypoint_path": _relative(DEFAULT_QML_ENTRYPOINT) if qml_found else "",
        "ui_config_found": config_found,
        "ui_config_path": _relative(DEFAULT_CONFIG) if config_found else "",
        "ui_framework": "PySide6/QML" if entrypoint_found or qml_found else "unknown",
        "demo_mode_required": True,
        "live_mode_allowed": False,
        "exchange_io": "disabled",
        "order_submission": "disabled",
        "real_orders_submitted": False,
        "api_keys_required": False,
        "secrets_read": False,
        "keychain_read": False,
        "env_values_read": False,
        "dot_env_read": False,
        "runtime_loop_started": False,
        "production_runtime_loop_started": False,
        "ui_launch_command_preview": _build_command_preview(entrypoint_found, config_found),
        "ui_smoke_command_preview": _build_smoke_command_preview(entrypoint_found, config_found),
        "command_execution_allowed": False,
        "command_executed": False,
        "subprocess_invoked": False,
        "shell_used": False,
        "issues": issues,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the source UI preview launch plan.")
    parser.add_argument("--json", action="store_true", help="Emit the plan as JSON.")
    parser.parse_args(argv)
    print(json.dumps(build_launch_plan(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
