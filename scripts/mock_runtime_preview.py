from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

import yaml

_EXPECTED_RULES: tuple[tuple[str, Any], ...] = (
    ("trading.enable_paper_mode", True),
    ("trading.enable_live_mode", False),
    ("execution.default_mode", "paper"),
    ("execution.force_paper_when_offline", True),
    ("execution.live.enabled", False),
)

_DEFAULT_DURATION_SECONDS = 5
_MAX_DURATION_SECONDS = 30


def _get_nested_mapping_value(payload: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for segment in dotted_path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            return None
        current = current[segment]
    return current


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled mock runtime preview (offline, no exchange io, no real orders, no api keys)."
        )
    )
    parser.add_argument("--config", default="config/e2e/demo_paper.yml")
    parser.add_argument("--mode", choices=("demo", "paper", "live"), default="demo")
    parser.add_argument("--duration-seconds", type=int, default=_DEFAULT_DURATION_SECONDS)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _validate_config(config_path: Path) -> tuple[dict[str, Any], list[str]]:
    loaded: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        return {}, ["config_not_mapping"]

    checks: dict[str, Any] = {}
    issues: list[str] = []
    for dotted_path, expected in _EXPECTED_RULES:
        observed = _get_nested_mapping_value(loaded, dotted_path)
        checks[dotted_path] = observed
        if observed != expected:
            issues.append(f"unsafe_config:{dotted_path}")
    return checks, issues


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    else:
        print(payload)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config_path = Path(args.config)
    simulated_max_sleep_seconds = min(args.duration_seconds, 1)

    if args.mode == "live":
        payload = {
            "status": "blocked",
            "reason": "mock_runtime_preview_forbids_live_mode",
            "mode": args.mode,
            "config": str(config_path),
            "runtime_preview_started": False,
            "bounded_duration_seconds": args.duration_seconds,
            "simulated_lifecycle_max_sleep_seconds": simulated_max_sleep_seconds,
            "exchange_io": "disabled",
            "order_execution": "disabled",
            "api_keys_required": False,
            "live_mode_allowed": False,
            "preview_steps": [],
            "issues": ["live_mode_not_allowed"],
        }
        _emit(payload, args.json)
        return 2

    if args.duration_seconds <= 0 or args.duration_seconds > _MAX_DURATION_SECONDS:
        payload = {
            "status": "blocked",
            "reason": "duration_out_of_bounds",
            "mode": args.mode,
            "config": str(config_path),
            "runtime_preview_started": False,
            "bounded_duration_seconds": args.duration_seconds,
            "max_duration_seconds": _MAX_DURATION_SECONDS,
            "simulated_lifecycle_max_sleep_seconds": simulated_max_sleep_seconds,
            "exchange_io": "disabled",
            "order_execution": "disabled",
            "api_keys_required": False,
            "live_mode_allowed": False,
            "preview_steps": [],
            "issues": ["duration_out_of_bounds"],
        }
        _emit(payload, args.json)
        return 2

    if not config_path.exists():
        payload = {
            "status": "error",
            "reason": "config_not_found",
            "mode": args.mode,
            "config": str(config_path),
            "runtime_preview_started": False,
            "bounded_duration_seconds": args.duration_seconds,
            "simulated_lifecycle_max_sleep_seconds": simulated_max_sleep_seconds,
            "exchange_io": "disabled",
            "order_execution": "disabled",
            "api_keys_required": False,
            "live_mode_allowed": False,
            "preview_steps": [],
            "issues": [f"config_not_found:{config_path}"],
        }
        _emit(payload, args.json)
        return 1

    checks, issues = _validate_config(config_path)
    if issues:
        payload = {
            "status": "blocked",
            "reason": "unsafe_config",
            "mode": args.mode,
            "config": str(config_path),
            "runtime_preview_started": False,
            "bounded_duration_seconds": args.duration_seconds,
            "simulated_lifecycle_max_sleep_seconds": simulated_max_sleep_seconds,
            "exchange_io": "disabled",
            "order_execution": "disabled",
            "api_keys_required": False,
            "live_mode_allowed": False,
            "checks": checks,
            "preview_steps": [],
            "issues": issues,
        }
        _emit(payload, args.json)
        return 2

    started_at = time.monotonic()
    preview_steps = [
        "safety_flags_validated",
        "mock_offline_preview_started",
        "mock_runtime_tick",
        "mock_offline_preview_finished",
    ]
    # bounded mock lifecycle only; no runtime start, no exchange io
    time.sleep(simulated_max_sleep_seconds)
    elapsed = round(time.monotonic() - started_at, 3)

    payload = {
        "status": "ok",
        "mode": args.mode,
        "config": str(config_path),
        "runtime_preview_started": True,
        "bounded_duration_seconds": args.duration_seconds,
        "simulated_lifecycle_max_sleep_seconds": simulated_max_sleep_seconds,
        "preview_elapsed_seconds": elapsed,
        "preview_kind": "controlled_mock_preview",
        "exchange_io": "disabled",
        "order_execution": "disabled",
        "api_keys_required": False,
        "live_mode_allowed": False,
        "checks": checks,
        "preview_steps": preview_steps,
        "issues": [],
    }
    _emit(payload, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
