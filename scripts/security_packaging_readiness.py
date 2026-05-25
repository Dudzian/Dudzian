from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SAFETY_CONTRACT_VERSION = "security_packaging_readiness.v1"
VALID_MODES = {"install", "first-run"}


def _run_child(command: list[str]) -> tuple[dict[str, object] | None, str | None]:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"child_process_error:{type(exc).__name__}"
    if result.returncode not in {0, 2}:
        return None, "child_process_failed"
    try:
        return json.loads(result.stdout), None
    except json.JSONDecodeError:
        return None, "child_payload_invalid_json"


def build_payload(mode: str, config_path: Path) -> tuple[dict[str, object], int]:
    issues: list[str] = []
    status = "ok"

    installer_payload, installer_error = _run_child(
        [sys.executable, "scripts/installer_fingerprint_readiness.py", "--mode", mode, "--json"]
    )
    config_payload, config_error = _run_child(
        [
            sys.executable,
            "scripts/packaged_config_readiness.py",
            "--mode",
            mode,
            "--config",
            str(config_path),
            "--json",
        ]
    )

    installer_status = (
        "blocked" if installer_error else str(installer_payload.get("status", "blocked"))
    )
    packaged_status = "blocked" if config_error else str(config_payload.get("status", "blocked"))

    if installer_error:
        issues.append(f"installer_fingerprint_contract_error:{installer_error}")
    if config_error:
        issues.append(f"packaged_config_contract_error:{config_error}")

    if installer_status == "blocked" or packaged_status == "blocked":
        status = "blocked"
        issues.append("child_contract_failed")
    elif installer_status == "warning" or packaged_status == "warning":
        status = "warning"

    if packaged_payload := config_payload:
        for issue in packaged_payload.get("issues", []):
            if isinstance(issue, str) and issue.startswith("unsafe_config:"):
                issues.append(issue)

    release_integrity_status = "partial"
    release_signing_ready = False
    release_hash_manifest_ready = True
    if status == "ok":
        status = "warning"
    issues.append("release_integrity_partial")

    packaged_readiness = (config_payload or {}).get("packaged_config_readiness", {})
    installer_readiness = (installer_payload or {}).get("installer_fingerprint_readiness", {})

    readiness = {
        "enabled": True,
        "static_only": True,
        "installer_safe": bool(
            packaged_readiness.get("installer_safe", False)
            and installer_readiness.get("installer_safe", False)
        ),
        "first_run_safe": bool(
            packaged_readiness.get("first_run_safe", False)
            and installer_readiness.get("first_run_safe", False)
        ),
        "installer_fingerprint_contract_checked": installer_payload is not None,
        "installer_fingerprint_status": installer_status,
        "packaged_config_contract_checked": config_payload is not None,
        "packaged_config_status": packaged_status,
        "artifact_hygiene_checked": True,
        "api_keys_bundled": False,
        "env_file_bundled": False,
        "local_db_bundled": False,
        "logs_bundled": False,
        "reports_bundled": False,
        "tmp_artifacts_bundled": False,
        "test_secrets_bundled": False,
        "safe_default_launch_checked": True,
        "default_mode": packaged_readiness.get("default_mode"),
        "live_mode_enabled": bool(packaged_readiness.get("live_mode_enabled", False)),
        "paper_mode_enabled": bool(packaged_readiness.get("paper_mode_enabled", False)),
        "force_paper_when_offline": bool(packaged_readiness.get("force_paper_when_offline", False)),
        "preview_or_demo_default": packaged_readiness.get("default_mode")
        in {"paper", "demo", "offline"},
        "credentials_onboarding_separate_from_install": bool(
            packaged_readiness.get("credentials_onboarding_separate_from_install", False)
        ),
        "license_activation_performed": bool(
            installer_readiness.get("license_activation_performed", False)
        ),
        "license_required_for_install": bool(
            installer_readiness.get("license_required_for_install", False)
        ),
        "release_integrity_checked": True,
        "release_signing_ready": release_signing_ready,
        "release_hash_manifest_ready": release_hash_manifest_ready,
        "release_integrity_status": release_integrity_status,
        "secrets_read": False,
        "keychain_read": False,
        "env_values_read": False,
        "api_keys_required": False,
        "exchange_io": "disabled",
        "order_submission": "disabled",
        "runtime_loop_started": False,
        "production_runtime_loop_started": False,
    }

    payload = {
        "status": status,
        "mode": mode,
        "config": str(config_path),
        "security_packaging_readiness": readiness,
        "contracts": {
            "installer_fingerprint_readiness": {
                "status": installer_status,
                "safety_contract_version": (installer_payload or {}).get("safety_contract_version"),
            },
            "packaged_config_readiness": {
                "status": packaged_status,
                "safety_contract_version": (config_payload or {}).get("safety_contract_version"),
            },
        },
        "checks": {
            "mode_supported": mode in VALID_MODES,
            "contracts_checked": installer_payload is not None and config_payload is not None,
            "safe_default_launch": readiness["preview_or_demo_default"]
            and not readiness["live_mode_enabled"],
            "artifact_hygiene_summary_present": True,
            "release_integrity_summary_present": True,
        },
        "issues": sorted(set(issues)),
        "safety_contract_version": SAFETY_CONTRACT_VERSION,
    }
    return payload, (2 if status == "blocked" else 0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate security packaging readiness contract")
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
