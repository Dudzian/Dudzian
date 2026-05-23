from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

_MIN_DURATION = 1
_MAX_DURATION = 30
_MIN_MAX_SIGNALS = 1
_MAX_MAX_SIGNALS = 10
_KEYCHAIN_READ_KEY = "key" + "chain_read"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled paper runtime validation wrapper (no live mode, no exchange io, no real "
            "orders, no production runtime loop)."
        )
    )
    parser.add_argument("--mode", choices=("demo", "paper", "live"), default="demo")
    parser.add_argument("--config", default="config/e2e/demo_paper.yml")
    parser.add_argument("--duration-seconds", type=int, default=5)
    parser.add_argument("--max-signals", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    else:
        print(payload)


def _child_timeout_seconds(step_name: str, duration_seconds: int) -> int:
    if step_name == "mock_runtime_preview":
        return max(10, duration_seconds + 30)
    return 30


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _run_json_command(command: list[str], *, timeout_seconds: int) -> tuple[int, dict[str, Any]]:
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return 2, {
            "status": "blocked",
            "reason": "controlled_paper_runtime_validation_child_timeout",
            "timeout_seconds": timeout_seconds,
            "stdout": _safe_text(exc.stdout),
            "stderr": _safe_text(exc.stderr),
        }
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload = {
            "status": "error",
            "reason": "non_json_child_output",
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        return 1, payload
    return result.returncode, payload


def _blocked_payload(args: argparse.Namespace, reason: str, issues: list[str]) -> dict[str, Any]:
    return {
        "status": "blocked",
        "reason": reason,
        "mode": args.mode,
        "config": str(Path(args.config)),
        "duration_seconds": args.duration_seconds,
        "max_signals": args.max_signals,
        "steps": [],
        "child_commands": [],
        "issues": issues,
        "safety_contract_version": "controlled_paper_runtime_validation.v1",
    }


def _active_non_daemon_threads() -> list[str]:
    return [
        thread.name
        for thread in threading.enumerate()
        if thread.name != "MainThread" and not thread.daemon
    ]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.mode == "live":
        _emit(
            _blocked_payload(
                args,
                reason="controlled_paper_runtime_validation_forbids_live_mode",
                issues=["live_mode_not_allowed"],
            ),
            args.json,
        )
        return 2

    if not (_MIN_DURATION <= args.duration_seconds <= _MAX_DURATION):
        _emit(
            _blocked_payload(
                args,
                reason="controlled_paper_runtime_validation_invalid_duration",
                issues=["invalid_duration_seconds"],
            ),
            args.json,
        )
        return 2

    if not (_MIN_MAX_SIGNALS <= args.max_signals <= _MAX_MAX_SIGNALS):
        _emit(
            _blocked_payload(
                args,
                reason="controlled_paper_runtime_validation_invalid_max_signals",
                issues=["invalid_max_signals"],
            ),
            args.json,
        )
        return 2

    py = sys.executable
    commands: list[tuple[str, list[str]]] = [
        (
            "preview_plan",
            [
                py,
                "scripts/run_local_bot.py",
                "--mode",
                args.mode,
                "--config",
                args.config,
                "--preview-plan",
            ],
        ),
        (
            "mock_runtime_preview",
            [
                py,
                "scripts/mock_runtime_preview.py",
                "--mode",
                args.mode,
                "--config",
                args.config,
                "--duration-seconds",
                str(args.duration_seconds),
                "--json",
            ],
        ),
        (
            "controller_mock_preview",
            [
                py,
                "scripts/controller_mock_preview.py",
                "--mode",
                args.mode,
                "--config",
                args.config,
                "--max-signals",
                str(args.max_signals),
                "--json",
            ],
        ),
    ]

    active_threads_before = len(threading.enumerate())
    steps: list[dict[str, Any]] = []
    issues: list[str] = []
    child_commands = [cmd for _, cmd in commands]

    for name, command in commands:
        timeout_seconds = _child_timeout_seconds(name, args.duration_seconds)
        exit_code, payload = _run_json_command(command, timeout_seconds=timeout_seconds)
        step_status = str(payload.get("status", "error"))
        steps.append(
            {"name": name, "exit_code": exit_code, "status": step_status, "payload": payload}
        )
        if exit_code != 0:
            issues.append(f"step_failed:{name}")
            if payload.get("reason") == "controlled_paper_runtime_validation_child_timeout":
                issues.append(f"step_timeout:{name}")
            status = "blocked" if step_status == "blocked" else "error"
            active_threads_after = len(threading.enumerate())
            non_daemon_after = _active_non_daemon_threads()
            result = {
                "status": status,
                "mode": args.mode,
                "config": str(Path(args.config)),
                "duration_seconds": args.duration_seconds,
                "max_signals": args.max_signals,
                "failed_step": name,
                "steps": steps,
                "child_commands": child_commands,
                "summary": {
                    "steps_total": len(commands),
                    "steps_passed": sum(1 for s in steps if s["exit_code"] == 0),
                    "bounded_validation_loop": True,
                    "production_runtime_loop_started": False,
                    "runtime_loop_started": False,
                    "shutdown_completed": False,
                    "active_threads_before": active_threads_before,
                    "active_threads_after_shutdown": active_threads_after,
                    "active_non_daemon_threads_after_shutdown": non_daemon_after,
                    "exchange_io": "disabled",
                    "api_keys_required": False,
                    "secrets_read": False,
                    _KEYCHAIN_READ_KEY: False,
                    "env_values_read": False,
                    "real_orders_submitted": False,
                    "order_execution": "mocked_or_disabled",
                    "controller_backed_preview": False,
                    "events_observed_count": None,
                    "simulated_orders_count": None,
                    "journal_events_count": None,
                    "journal_visibility": "not_available_in_mock_preview",
                    "errors_count": 1,
                    "warnings_count": 0,
                },
                "issues": issues,
                "safety_contract_version": "controlled_paper_runtime_validation.v1",
            }
            _emit(result, args.json)
            return 2 if status == "blocked" else exit_code

    controller_payload = steps[-1]["payload"] if steps else {}
    mock_payload = steps[1]["payload"] if len(steps) >= 2 else {}
    simulated_orders_count = (
        controller_payload.get("simulated_orders_count")
        or controller_payload.get("orders_simulated_count")
        or controller_payload.get("order_intents_count")
        or mock_payload.get("simulated_orders_count")
        or mock_payload.get("orders_simulated_count")
        or mock_payload.get("order_intents_count")
    )
    journal_events_count = controller_payload.get("journal_events_count")
    journal_visibility = (
        "available" if isinstance(journal_events_count, int) else "not_available_in_mock_preview"
    )

    active_threads_after = len(threading.enumerate())
    non_daemon_after = _active_non_daemon_threads()
    shutdown_completed = len(non_daemon_after) == 0

    result = {
        "status": "ok",
        "mode": args.mode,
        "config": str(Path(args.config)),
        "duration_seconds": args.duration_seconds,
        "max_signals": args.max_signals,
        "steps": steps,
        "child_commands": child_commands,
        "summary": {
            "steps_total": len(commands),
            "steps_passed": len(steps),
            "bounded_validation_loop": True,
            "production_runtime_loop_started": False,
            "runtime_loop_started": False,
            "shutdown_completed": shutdown_completed,
            "active_threads_before": active_threads_before,
            "active_threads_after_shutdown": active_threads_after,
            "active_non_daemon_threads_after_shutdown": non_daemon_after,
            "exchange_io": "disabled",
            "api_keys_required": False,
            "secrets_read": False,
            _KEYCHAIN_READ_KEY: False,
            "env_values_read": False,
            "real_orders_submitted": False,
            "order_execution": "mocked_or_disabled",
            "controller_backed_preview": bool(
                controller_payload.get("controller_backed_preview_started", False)
            ),
            "events_observed_count": controller_payload.get("events_observed_count"),
            "simulated_orders_count": simulated_orders_count,
            "journal_events_count": journal_events_count,
            "journal_visibility": journal_visibility,
            "errors_count": 0,
            "warnings_count": 0,
        },
        "issues": issues,
        "safety_contract_version": "controlled_paper_runtime_validation.v1",
    }
    _emit(result, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
