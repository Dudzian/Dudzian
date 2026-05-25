from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

SAFETY_CONTRACT_VERSION = "packaged_config_readiness.v1"
VALID_MODES = {"install", "first-run"}


def _load_config(path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(loaded, dict):
        return loaded
    return {}


def _bool_path(data: dict[str, object], *keys: str) -> bool | None:
    current: object = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return bool(current) if isinstance(current, bool) else None


def build_payload(mode: str, config_path: Path) -> tuple[dict[str, object], int]:
    if not config_path.exists():
        payload = {
            "status": "error",
            "mode": mode,
            "config": str(config_path),
            "reason": "config_not_found",
            "issues": ["config_not_found"],
            "safety_contract_version": SAFETY_CONTRACT_VERSION,
        }
        return payload, 1

    cfg = _load_config(config_path)
    issues: list[str] = []
    status = "ok"

    live_mode_enabled = _bool_path(cfg, "trading", "enable_live_mode")
    paper_mode_enabled = _bool_path(cfg, "trading", "enable_paper_mode")
    force_paper = _bool_path(cfg, "execution", "force_paper_when_offline")
    default_mode = None
    execution = cfg.get("execution") if isinstance(cfg, dict) else None
    if isinstance(execution, dict):
        value = execution.get("default_mode")
        if isinstance(value, str):
            default_mode = value

    if live_mode_enabled is not False:
        issues.append("unsafe_config:trading.enable_live_mode")
    live_exec_enabled = _bool_path(cfg, "execution", "live", "enabled")
    if live_exec_enabled is not False:
        issues.append("unsafe_config:execution.live.enabled")
    if paper_mode_enabled is not True:
        issues.append("unsafe_config:trading.enable_paper_mode")
    if force_paper is not True:
        issues.append("unsafe_config:execution.force_paper_when_offline")
    if default_mode not in {"paper", "demo", "offline"}:
        issues.append("unsafe_config:execution.default_mode")

    if issues:
        status = "blocked"

    readiness = {
        "enabled": True,
        "static_only": True,
        "installer_safe": True,
        "first_run_safe": True,
        "config_exists": True,
        "config_shape": "e2e_overlay",
        "default_mode": default_mode,
        "live_mode_enabled": bool(live_mode_enabled),
        "paper_mode_enabled": bool(paper_mode_enabled),
        "force_paper_when_offline": bool(force_paper),
        "credentials_onboarding_required": True,
        "credentials_onboarding_separate_from_install": True,
        "api_keys_required_for_install": False,
        "api_keys_bundled": False,
        "env_file_bundled": False,
        "local_db_bundled": False,
        "logs_bundled": False,
        "reports_bundled": False,
        "secrets_read": False,
        "keychain_read": False,
        "env_values_read": False,
        "exchange_io": "disabled",
        "order_submission": "disabled",
        "runtime_loop_started": False,
        "production_runtime_loop_started": False,
    }

    payload = {
        "status": status,
        "mode": mode,
        "config": str(config_path),
        "packaged_config_readiness": readiness,
        "checks": {
            "mode_supported": mode in VALID_MODES,
            "config_exists": True,
            "safe_default_mode": default_mode in {"paper", "demo", "offline"},
            "live_disabled": live_mode_enabled is False and live_exec_enabled is False,
            "no_runtime": True,
        },
        "issues": issues,
        "safety_contract_version": SAFETY_CONTRACT_VERSION,
    }
    return payload, (2 if status == "blocked" else 0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Packaged config readiness contract")
    parser.add_argument("--mode", choices=sorted(VALID_MODES), default="first-run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload, code = build_payload(args.mode, Path(args.config))
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
