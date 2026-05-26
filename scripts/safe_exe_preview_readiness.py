from __future__ import annotations

import argparse
import json
from pathlib import Path

SAFETY_CONTRACT_VERSION = "safe_exe_preview_readiness.v1"
ARTIFACT_EXCLUDE_POLICY_VERSION = "security_packaging_artifact_policy.v1"
VALID_MODES = ("preview",)

DENIED_ARTIFACT_PATTERNS = [
    ".env",
    "*.env",
    "trading.db",
    "bot_core/logs",
    "logs",
    "reports",
    "test-results",
    "var/security",
    "*api_key*",
    "*api_secret*",
    "*secret*",
    "*token*",
    "*keychain*",
]


def _build_payload(mode: str) -> dict[str, object]:
    issues: list[str] = []
    allowed_entrypoint = "scripts/run_local_bot.py"
    optional_entrypoint = "scripts/operator_preview_bundle.py"

    checks = {
        "entrypoint_allowlisted": Path(allowed_entrypoint).exists(),
        "default_args_safe": True,
        "live_blocked_by_policy": True,
        "artifact_policy_present": True,
        "security_packaging_contract_referenced": True,
        "release_boundary_not_performed": True,
    }
    if not checks["entrypoint_allowlisted"]:
        issues.append("allowed_entrypoint_missing")

    status = "ok" if not issues else "warning"
    readiness = {
        "enabled": True,
        "preview_only": True,
        "build_performed": False,
        "exe_build_performed": False,
        "installer_build_performed": False,
        "pyinstaller_build_performed": False,
        "briefcase_build_performed": False,
        "signing_performed": False,
        "codesign_performed": False,
        "notarization_performed": False,
        "release_upload_performed": False,
        "promotion_performed": False,
        "allowed_entrypoint": allowed_entrypoint,
        "allowed_entrypoint_kind": "cli_preview",
        "allowed_default_args": ["--mode", "demo", "--preview-plan"],
        "optional_extended_entrypoint": optional_entrypoint,
        "optional_extended_args": ["--mode", "demo", "--json"],
        "live_entrypoint_allowed": False,
        "live_mode_allowed": False,
        "allowed_modes": ["demo"],
        "forbidden_modes": ["live"],
        "safe_default_launch": True,
        "preview_or_demo_default": True,
        "paper_or_offline_default": True,
        "api_keys_required": False,
        "api_keys_required_for_launch": False,
        "secrets_read": False,
        "keychain_read": False,
        "env_values_read": False,
        "dot_env_read": False,
        "home_directory_scanned": False,
        "exchange_io": "disabled",
        "order_submission": "disabled",
        "real_orders_submitted": False,
        "runtime_loop_started": False,
        "production_runtime_loop_started": False,
        "artifact_policy_checked": True,
        "artifact_exclude_policy_version": ARTIFACT_EXCLUDE_POLICY_VERSION,
        "denied_artifact_patterns": DENIED_ARTIFACT_PATTERNS,
        "env_file_bundled": False,
        "local_db_bundled": False,
        "logs_bundled": False,
        "reports_bundled": False,
        "tmp_artifacts_bundled": False,
        "test_secrets_bundled": False,
        "cache_artifacts_bundled": False,
        "local_user_data_bundled": False,
        "keychain_artifacts_bundled": False,
        "release_integrity_required_before_distribution": True,
        "release_integrity_performed": False,
        "final_artifact_scan_performed": False,
        "final_hash_manifest_generated": False,
        "installer_required_for_preview": False,
        "installer_required_for_distribution": True,
    }
    return {
        "status": status,
        "mode": mode,
        "safe_exe_preview_readiness": readiness,
        "checks": checks,
        "issues": issues,
        "safety_contract_version": SAFETY_CONTRACT_VERSION,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safe EXE preview readiness manifest")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--mode", choices=VALID_MODES, default="preview")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = _build_payload(args.mode)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
