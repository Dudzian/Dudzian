from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-command safe operator preview bundle (no live trading, no real exchange, no real orders, no api keys)."
        )
    )
    parser.add_argument("--config", default="config/e2e/demo_paper.yml")
    parser.add_argument("--mode", choices=("demo", "paper", "live"), default="demo")
    parser.add_argument("--duration-seconds", type=int, default=5)
    parser.add_argument("--max-signals", type=int, default=1)
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--side", choices=("BUY", "SELL"), default="BUY")
    parser.add_argument("--quantity", type=float, default=0.01)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    else:
        print(payload)


def _run_json_command(command: list[str]) -> tuple[int, dict[str, Any]]:
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    payload: dict[str, Any]
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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.mode == "live":
        payload = {
            "status": "blocked",
            "reason": "operator_preview_bundle_forbids_live_mode",
            "mode": args.mode,
            "config": str(Path(args.config)),
            "steps": [],
            "issues": ["live_mode_not_allowed"],
            "safety_contract_version": "operator_preview_bundle.v1",
        }
        _emit(payload, args.json)
        return 2

    py = sys.executable
    config = args.config
    mode = args.mode

    commands: list[tuple[str, list[str]]] = [
        (
            "demo_paper_precheck",
            [py, "scripts/demo_paper_precheck.py", "--config", config, "--json"],
        ),
        (
            "paper_adapter_readiness",
            [py, "scripts/paper_adapter_readiness.py", "--config", config, "--json"],
        ),
        (
            "preview_plan",
            [
                py,
                "scripts/run_local_bot.py",
                "--mode",
                mode,
                "--config",
                config,
                "--preview-plan",
            ],
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
                "--symbol",
                args.symbol,
                "--side",
                args.side,
                "--quantity",
                str(args.quantity),
                "--json",
            ],
        ),
    ]

    steps: list[dict[str, Any]] = []
    issues: list[str] = []

    for name, command in commands:
        exit_code, payload = _run_json_command(command)
        step_status = str(payload.get("status", "error"))
        step_entry = {
            "name": name,
            "exit_code": exit_code,
            "status": step_status,
            "payload": payload,
        }
        steps.append(step_entry)

        if exit_code != 0:
            issues.append(f"step_failed:{name}:exit_code={exit_code}")
            status = "blocked" if step_status == "blocked" else "error"
            result = {
                "status": status,
                "mode": mode,
                "config": str(Path(config)),
                "failed_step": name,
                "steps": steps,
                "summary": {
                    "steps_total": len(commands),
                    "steps_passed": sum(1 for step in steps if step["exit_code"] == 0),
                    "real_orders_submitted": False,
                    "exchange_io": "disabled",
                    "api_keys_required": False,
                    "runtime_loop_started": False,
                    "controller_backed_preview": False,
                },
                "issues": issues,
                "safety_contract_version": "operator_preview_bundle.v1",
            }
            _emit(result, args.json)
            return 2 if status == "blocked" else 1

    controller_payload = steps[-1]["payload"] if steps else {}
    result = {
        "status": "ok",
        "mode": mode,
        "config": str(Path(config)),
        "steps": steps,
        "summary": {
            "steps_total": len(commands),
            "steps_passed": len(steps),
            "real_orders_submitted": False,
            "exchange_io": "disabled",
            "api_keys_required": False,
            "runtime_loop_started": False,
            "controller_backed_preview": bool(
                controller_payload.get("controller_backed_preview_started", False)
            ),
        },
        "issues": issues,
        "safety_contract_version": "operator_preview_bundle.v1",
    }
    _emit(result, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
