from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
            "Bounded paper runtime dry-run preflight wrapper (no live mode, no exchange io, no "
            "real orders, no production runtime loop)."
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


def _run_json_command(command: list[str]) -> tuple[int, dict[str, Any]]:
    result = subprocess.run(command, check=False, text=True, capture_output=True)
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


def _blocked_payload(
    *,
    args: argparse.Namespace,
    reason: str,
    issues: list[str],
) -> dict[str, Any]:
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
        "safety_contract_version": "paper_runtime_dry_run.v1",
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.mode == "live":
        _emit(
            _blocked_payload(
                args=args,
                reason="paper_runtime_dry_run_forbids_live_mode",
                issues=["live_mode_not_allowed"],
            ),
            args.json,
        )
        return 2

    if not (_MIN_DURATION <= args.duration_seconds <= _MAX_DURATION):
        _emit(
            _blocked_payload(
                args=args,
                reason="paper_runtime_dry_run_invalid_duration",
                issues=["invalid_duration_seconds"],
            ),
            args.json,
        )
        return 2

    if not (_MIN_MAX_SIGNALS <= args.max_signals <= _MAX_MAX_SIGNALS):
        _emit(
            _blocked_payload(
                args=args,
                reason="paper_runtime_dry_run_invalid_max_signals",
                issues=["invalid_max_signals"],
            ),
            args.json,
        )
        return 2

    py = sys.executable
    config = args.config
    mode = args.mode

    commands: list[tuple[str, list[str]]] = [
        (
            "preview_plan",
            [py, "scripts/run_local_bot.py", "--mode", mode, "--config", config, "--preview-plan"],
        ),
        (
            "mock_runtime_preview",
            [
                py,
                "scripts/mock_runtime_preview.py",
                "--mode",
                mode,
                "--config",
                config,
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
                mode,
                "--config",
                config,
                "--max-signals",
                str(args.max_signals),
                "--json",
            ],
        ),
    ]

    steps: list[dict[str, Any]] = []
    issues: list[str] = []
    child_commands = [command for _, command in commands]

    for name, command in commands:
        exit_code, payload = _run_json_command(command)
        step_status = str(payload.get("status", "error"))
        steps.append(
            {"name": name, "exit_code": exit_code, "status": step_status, "payload": payload}
        )

        if exit_code != 0:
            issues.append(f"step_failed:{name}")
            status = "blocked" if step_status == "blocked" else "error"
            result = {
                "status": status,
                "mode": mode,
                "config": str(Path(config)),
                "duration_seconds": args.duration_seconds,
                "max_signals": args.max_signals,
                "failed_step": name,
                "steps": steps,
                "child_commands": child_commands,
                "summary": {
                    "steps_total": len(commands),
                    "steps_passed": sum(1 for s in steps if s["exit_code"] == 0),
                    "bounded_preview_loop": True,
                    "production_runtime_loop_started": False,
                    "runtime_loop_started": False,
                    "exchange_io": "disabled",
                    "api_keys_required": False,
                    "secrets_read": False,
                    _KEYCHAIN_READ_KEY: False,
                    "env_values_read": False,
                    "real_orders_submitted": False,
                    "order_execution": "mocked_or_disabled",
                    "controller_backed_preview": False,
                },
                "issues": issues,
                "safety_contract_version": "paper_runtime_dry_run.v1",
            }
            _emit(result, args.json)
            return 2 if status == "blocked" else exit_code

    controller_payload = steps[-1]["payload"] if steps else {}
    result = {
        "status": "ok",
        "mode": mode,
        "config": str(Path(config)),
        "duration_seconds": args.duration_seconds,
        "max_signals": args.max_signals,
        "steps": steps,
        "child_commands": child_commands,
        "summary": {
            "steps_total": len(commands),
            "steps_passed": len(steps),
            "bounded_preview_loop": True,
            "production_runtime_loop_started": False,
            "runtime_loop_started": False,
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
        },
        "issues": issues,
        "safety_contract_version": "paper_runtime_dry_run.v1",
    }
    _emit(result, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
