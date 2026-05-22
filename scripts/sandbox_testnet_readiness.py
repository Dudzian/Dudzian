from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

EXPECTED_RULES: tuple[tuple[str, Any], ...] = (
    ("trading.enable_paper_mode", True),
    ("trading.enable_live_mode", False),
    ("execution.default_mode", "paper"),
    ("execution.force_paper_when_offline", True),
    ("execution.live.enabled", False),
)


def _get_nested(payload: dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for segment in dotted_path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return None
        current = current[segment]
    return current


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sandbox/testnet static readiness preflight (config-only, no exchange/API I/O, no secrets, no orders)."
        )
    )
    parser.add_argument("--config", default="config/e2e/demo_paper.yml")
    parser.add_argument("--environment", default="binance_paper")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    else:
        print(payload)


def _error_payload(config_path: Path, reason: str, issues: list[str]) -> dict[str, Any]:
    return {
        "status": "error",
        "reason": reason,
        "config": str(config_path),
        "environment": None,
        "config_shape": "unknown",
        "sandbox_testnet_readiness": {
            "enabled": False,
            "static_only": True,
            "exchange_io": "disabled",
            "order_submission": "disabled",
            "secrets_read": False,
            "api_keys_required": False,
            "runtime_loop_started": False,
            "live_mode_allowed": False,
            "environment_declared": False,
        },
        "checks": {},
        "issues": issues,
        "safety_contract_version": "sandbox_testnet_readiness.v1",
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config_path = Path(args.config)

    if not config_path.exists():
        payload = _error_payload(
            config_path,
            reason="config_not_found",
            issues=[f"config_not_found:{config_path}"],
        )
        _emit(payload, args.json)
        return 1

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        payload = _error_payload(
            config_path,
            reason="config_not_mapping",
            issues=["config_not_mapping"],
        )
        _emit(payload, args.json)
        return 1

    checks: dict[str, dict[str, Any]] = {}
    issues: list[str] = []

    for path, expected in EXPECTED_RULES:
        observed = _get_nested(loaded, path)
        ok = observed == expected
        checks[path] = {"expected": expected, "observed": observed, "ok": ok}
        if not ok:
            issues.append(f"unsafe_config:{path}")

    declared_env = _get_nested(loaded, "trading.entrypoints.demo_desktop.environment")
    environment_declared = declared_env == args.environment
    checks["trading.entrypoints.demo_desktop.environment"] = {
        "expected": args.environment,
        "observed": declared_env,
        "ok": environment_declared,
    }
    if not environment_declared:
        issues.append("environment_not_declared")

    is_ok = not issues
    status = "ok" if is_ok else "blocked"
    reason = None if is_ok else "sandbox_testnet_readiness_unsafe_config"
    payload = {
        "status": status,
        "reason": reason,
        "config": str(config_path),
        "environment": args.environment,
        "config_shape": "e2e_overlay",
        "sandbox_testnet_readiness": {
            "enabled": is_ok,
            "static_only": True,
            "exchange_io": "disabled",
            "order_submission": "disabled",
            "secrets_read": False,
            "api_keys_required": False,
            "runtime_loop_started": False,
            "live_mode_allowed": False,
            "environment_declared": environment_declared,
        },
        "checks": checks,
        "issues": issues,
        "safety_contract_version": "sandbox_testnet_readiness.v1",
    }
    _emit(payload, args.json)
    return 0 if is_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
