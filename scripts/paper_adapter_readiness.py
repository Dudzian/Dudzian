"""Statyczny preflight gotowości paper adaptera (no exchange I/O, no orders)."""

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
            "Paper adapter readiness preflight (config-only, no exchange I/O, no API keys, no orders)."
        )
    )
    parser.add_argument("--config", default="config/e2e/demo_paper.yml")
    parser.add_argument(
        "--environment",
        default=None,
        help=(
            "Opcjonalna nazwa środowiska wyłącznie do walidacji deklaratywnej (bez łączenia z exchange)."
        ),
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(payload)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config_path = Path(args.config)

    if not config_path.exists():
        payload = {
            "status": "error",
            "config": str(config_path),
            "checks": {},
            "adapter_readiness": {"enabled": False, "reason": "config_not_found"},
            "exchange_io": {"enabled": False, "reason": "contract_no_exchange_io"},
            "order_submission": {"enabled": False, "reason": "contract_no_order_submission"},
            "api_keys_required": False,
            "issues": [f"config_not_found:{config_path}"],
        }
        _emit(payload, args.json)
        return 1

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        payload = {
            "status": "error",
            "config": str(config_path),
            "checks": {},
            "adapter_readiness": {"enabled": False, "reason": "config_not_mapping"},
            "exchange_io": {"enabled": False, "reason": "contract_no_exchange_io"},
            "order_submission": {"enabled": False, "reason": "contract_no_order_submission"},
            "api_keys_required": False,
            "issues": ["config_not_mapping"],
        }
        _emit(payload, args.json)
        return 1

    checks: dict[str, dict[str, Any]] = {}
    issues: list[str] = []

    for path, expected in EXPECTED_RULES:
        observed = _get_nested(loaded, path)
        ok = observed == expected
        checks[path] = {"expected": expected, "observed": observed, "ok": ok}
        if not ok:
            issues.append(f"unsafe_flag:{path}:expected={expected!r}:observed={observed!r}")

    if args.environment:
        checks["environment.name"] = {
            "expected": args.environment,
            "observed": args.environment,
            "ok": True,
            "note": "validated_declaratively_only_no_exchange_connection",
        }

    is_ok = not issues
    payload = {
        "status": "ok" if is_ok else "error",
        "config": str(config_path),
        "checks": checks,
        "adapter_readiness": {
            "enabled": is_ok,
            "mode": "static_contract_preflight",
            "environment": args.environment,
        },
        "exchange_io": {"enabled": False, "reason": "contract_no_exchange_io"},
        "order_submission": {"enabled": False, "reason": "contract_no_order_submission"},
        "api_keys_required": False,
        "issues": issues,
    }

    if args.json:
        _emit(payload, True)
    elif is_ok:
        print(f"OK: {config_path} passed paper adapter readiness preflight")
    else:
        print("ERROR: paper adapter readiness preflight failed:")
        for issue in issues:
            print(f" - {issue}")

    return 0 if is_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
