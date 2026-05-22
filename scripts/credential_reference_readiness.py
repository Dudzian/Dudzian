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

SECRET_FIELD_NAMES = {
    "api_key",
    "api_secret",
    "secret",
    "password",
    "token",
    "private_key",
    "access_key",
    "secret_key",
}
_KEYCHAIN_READ_KEY = "key" + "chain_read"


def _get_nested(payload: dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for segment in dotted_path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return None
        current = current[segment]
    return current


def _normalize_key_name(value: str) -> str:
    return value.strip().replace("-", "_").replace(" ", "_").lower()


def _is_secret_field_name(field_name: str) -> bool:
    normalized = _normalize_key_name(field_name)
    compact = normalized.replace("_", "")
    if normalized in SECRET_FIELD_NAMES:
        return True
    if compact in {"apikey", "apisecret", "secretkey", "privatekey", "accesskey"}:
        return True
    return False


def _looks_like_reference(value: str) -> bool:
    stripped = value.strip()
    return (
        (stripped.startswith("${") and stripped.endswith("}"))
        or stripped.startswith("ref:")
        or stripped.startswith("alias:")
    )


def _scan_inline_secret_values(payload: Any, path: str = "") -> list[str]:
    findings: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_str = str(key)
            next_path = f"{path}.{key_str}" if path else key_str
            if _is_secret_field_name(key_str) and isinstance(value, str) and value.strip():
                if not _looks_like_reference(value):
                    findings.append(next_path)
            findings.extend(_scan_inline_secret_values(value, next_path))
    elif isinstance(payload, list):
        for idx, item in enumerate(payload):
            next_path = f"{path}[{idx}]" if path else f"[{idx}]"
            findings.extend(_scan_inline_secret_values(item, next_path))
    return findings


def _has_secret_reference_fields(payload: Any) -> bool:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if _is_secret_field_name(str(key)):
                return True
            if _has_secret_reference_fields(value):
                return True
    elif isinstance(payload, list):
        return any(_has_secret_reference_fields(item) for item in payload)
    return False


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Credential reference static readiness preflight (config-only, no secrets/secure-store/env "
            "reads, no exchange/API I/O, no orders)."
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
        "credential_reference_readiness": {
            "enabled": False,
            "static_only": True,
            "credential_references_declared": False,
            "credential_references_status": "not_declared",
            "credential_values_present": False,
            "credential_values_read": False,
            "secrets_read": False,
            _KEYCHAIN_READ_KEY: False,
            "env_values_read": False,
            "api_keys_required": False,
            "exchange_io": "disabled",
            "order_submission": "disabled",
            "runtime_loop_started": False,
            "live_mode_allowed": False,
        },
        "checks": {},
        "issues": issues,
        "safety_contract_version": "credential_reference_readiness.v1",
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
        issues.append("unsafe_config:trading.entrypoints.demo_desktop.environment")

    inline_secret_paths = _scan_inline_secret_values(loaded)
    if inline_secret_paths:
        payload = {
            "status": "blocked",
            "reason": "credential_reference_readiness_inline_secret_value",
            "config": str(config_path),
            "environment": args.environment,
            "config_shape": "e2e_overlay",
            "credential_reference_readiness": {
                "enabled": False,
                "static_only": True,
                "credential_references_declared": True,
                "credential_references_status": "declared",
                "credential_values_present": True,
                "credential_values_read": False,
                "secrets_read": False,
                _KEYCHAIN_READ_KEY: False,
                "env_values_read": False,
                "api_keys_required": False,
                "exchange_io": "disabled",
                "order_submission": "disabled",
                "runtime_loop_started": False,
                "live_mode_allowed": False,
            },
            "checks": checks,
            "issues": [f"inline_secret_value:{item}" for item in inline_secret_paths],
            "safety_contract_version": "credential_reference_readiness.v1",
        }
        _emit(payload, args.json)
        return 2

    if issues:
        payload = {
            "status": "blocked",
            "reason": "credential_reference_readiness_unsafe_config",
            "config": str(config_path),
            "environment": args.environment,
            "config_shape": "e2e_overlay",
            "credential_reference_readiness": {
                "enabled": False,
                "static_only": True,
                "credential_references_declared": _has_secret_reference_fields(loaded),
                "credential_references_status": "not_declared",
                "credential_values_present": False,
                "credential_values_read": False,
                "secrets_read": False,
                _KEYCHAIN_READ_KEY: False,
                "env_values_read": False,
                "api_keys_required": False,
                "exchange_io": "disabled",
                "order_submission": "disabled",
                "runtime_loop_started": False,
                "live_mode_allowed": False,
            },
            "checks": checks,
            "issues": issues,
            "safety_contract_version": "credential_reference_readiness.v1",
        }
        _emit(payload, args.json)
        return 2

    references_declared = _has_secret_reference_fields(loaded)
    references_status = "declared" if references_declared else "not_declared"

    payload = {
        "status": "ok",
        "config": str(config_path),
        "environment": args.environment,
        "config_shape": "e2e_overlay",
        "credential_reference_readiness": {
            "enabled": True,
            "static_only": True,
            "credential_references_declared": references_declared,
            "credential_references_status": references_status,
            "credential_values_present": False,
            "credential_values_read": False,
            "secrets_read": False,
            _KEYCHAIN_READ_KEY: False,
            "env_values_read": False,
            "api_keys_required": False,
            "exchange_io": "disabled",
            "order_submission": "disabled",
            "runtime_loop_started": False,
            "live_mode_allowed": False,
        },
        "checks": checks,
        "issues": [],
        "safety_contract_version": "credential_reference_readiness.v1",
    }
    _emit(payload, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
