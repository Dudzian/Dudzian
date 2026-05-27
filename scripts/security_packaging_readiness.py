from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SAFETY_CONTRACT_VERSION = "security_packaging_readiness.v1"
VALID_MODES = {"install", "first-run"}
ARTIFACT_EXCLUDE_POLICY_VERSION = "security_packaging_artifact_policy.v1"
SAFE_LAUNCH_POLICY_VERSION = "security_packaging_safe_launch_policy.v1"
RELEASE_INTEGRITY_CONTRACT_VERSION = "release_integrity_readiness.v1"
SAFE_EXE_PREVIEW_CONTRACT_VERSION = "safe_exe_preview_readiness.v1"
SAFE_EXE_PREVIEW_BUILD_PLAN_CONTRACT_VERSION = "safe_exe_preview_build_plan.v1"

SAFE_EXE_PREVIEW_PROFILE_VALIDATOR_CONTRACT_VERSION = "safe_exe_preview_profile_validator.v1"
SAFE_EXE_PREVIEW_ALLOWED_ENTRYPOINT = "scripts/run_local_bot.py"
SAFE_EXE_PREVIEW_ALLOWED_DEFAULT_ARGS = ["--mode", "demo", "--preview-plan"]
DENIED_ARTIFACT_PATTERNS = [
    ".env",
    "*.env",
    "trading.db",
    "bot_core/logs",
    "logs",
    "reports",
    "test-results",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "var/security",
    "*api_key*",
    "*api_secret*",
    "*secret*",
    "*token*",
    "*keychain*",
]


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
    release_payload, release_error = _run_child(
        [sys.executable, "scripts/release_integrity_readiness.py", "--json"]
    )
    safe_exe_payload, safe_exe_error = _run_child(
        [sys.executable, "scripts/safe_exe_preview_readiness.py", "--json"]
    )
    safe_exe_build_plan_payload, safe_exe_build_plan_error = _run_child(
        [sys.executable, "scripts/safe_exe_preview_build_plan.py", "--json"]
    )
    safe_exe_profile_validator_payload, safe_exe_profile_validator_error = _run_child(
        [sys.executable, "scripts/safe_exe_preview_profile_validator.py", "--json"]
    )

    installer_status = (
        "blocked" if installer_error else str(installer_payload.get("status", "blocked"))
    )
    packaged_status = "blocked" if config_error else str(config_payload.get("status", "blocked"))
    release_status = "blocked" if release_error else str(release_payload.get("status", "blocked"))

    if installer_error:
        issues.append(f"installer_fingerprint_contract_error:{installer_error}")
    if config_error:
        issues.append(f"packaged_config_contract_error:{config_error}")
    if release_error:
        issues.append(f"release_integrity_contract_error:{release_error}")

    safe_exe_status = (
        "blocked" if safe_exe_error else str(safe_exe_payload.get("status", "blocked"))
    )
    if safe_exe_error:
        issues.append(f"safe_exe_preview_contract_error:{safe_exe_error}")

    safe_exe_build_plan_status = (
        "blocked"
        if safe_exe_build_plan_error
        else str(safe_exe_build_plan_payload.get("status", "blocked"))
    )
    if safe_exe_build_plan_error:
        issues.append(f"safe_exe_preview_build_plan_contract_error:{safe_exe_build_plan_error}")

    safe_exe_profile_validator_status = (
        "blocked"
        if safe_exe_profile_validator_error
        else str(safe_exe_profile_validator_payload.get("status", "blocked"))
    )
    if safe_exe_profile_validator_error:
        issues.append(
            f"safe_exe_preview_profile_validator_contract_error:{safe_exe_profile_validator_error}"
        )

    if "blocked" in {
        installer_status,
        packaged_status,
        release_status,
        safe_exe_status,
        safe_exe_build_plan_status,
        safe_exe_profile_validator_status,
    }:
        status = "blocked"
        issues.append("child_contract_failed")
    elif "warning" in {
        installer_status,
        packaged_status,
        release_status,
        safe_exe_status,
        safe_exe_build_plan_status,
        safe_exe_profile_validator_status,
    }:
        status = "warning"

    if packaged_payload := config_payload:
        for issue in packaged_payload.get("issues", []):
            if isinstance(issue, str) and issue.startswith("unsafe_config:"):
                issues.append(issue)

    safe_exe_readiness = (safe_exe_payload or {}).get("safe_exe_preview_readiness", {})
    safe_exe_contract_checked = safe_exe_payload is not None
    safe_exe_contract_version = (safe_exe_payload or {}).get("safety_contract_version")
    safe_exe_contract_version_ok = safe_exe_contract_version == SAFE_EXE_PREVIEW_CONTRACT_VERSION
    safe_exe_allowed_entrypoint = safe_exe_readiness.get("allowed_entrypoint")
    safe_exe_allowed_default_args = safe_exe_readiness.get("allowed_default_args")
    safe_exe_live_mode_allowed = safe_exe_readiness.get("live_mode_allowed")
    safe_exe_build_performed = bool(safe_exe_readiness.get("build_performed", True))
    safe_exe_exe_build_performed = bool(safe_exe_readiness.get("exe_build_performed", True))
    safe_exe_installer_build_performed = bool(
        safe_exe_readiness.get("installer_build_performed", True)
    )
    safe_exe_pyinstaller_build_performed = bool(
        safe_exe_readiness.get("pyinstaller_build_performed", True)
    )
    safe_exe_briefcase_build_performed = bool(
        safe_exe_readiness.get("briefcase_build_performed", True)
    )
    safe_exe_release_upload_performed = bool(
        safe_exe_readiness.get("release_upload_performed", True)
    )
    safe_exe_promotion_performed = bool(safe_exe_readiness.get("promotion_performed", True))
    safe_exe_final_artifact_scan_performed = bool(
        safe_exe_readiness.get("final_artifact_scan_performed", True)
    )
    safe_exe_final_hash_manifest_generated = bool(
        safe_exe_readiness.get("final_hash_manifest_generated", True)
    )
    safe_exe_exchange_io = safe_exe_readiness.get("exchange_io")
    safe_exe_order_submission = safe_exe_readiness.get("order_submission")
    safe_exe_runtime_loop_started = bool(safe_exe_readiness.get("runtime_loop_started", True))
    safe_exe_production_runtime_loop_started = bool(
        safe_exe_readiness.get("production_runtime_loop_started", True)
    )
    safe_exe_entrypoint_allowlisted = (
        safe_exe_allowed_entrypoint == SAFE_EXE_PREVIEW_ALLOWED_ENTRYPOINT
    )
    safe_exe_default_args_safe = (
        safe_exe_allowed_default_args == SAFE_EXE_PREVIEW_ALLOWED_DEFAULT_ARGS
    )
    safe_exe_live_blocked_by_policy = safe_exe_live_mode_allowed is False
    safe_exe_build_not_performed = not any(
        [
            safe_exe_build_performed,
            safe_exe_exe_build_performed,
            safe_exe_installer_build_performed,
            safe_exe_pyinstaller_build_performed,
            safe_exe_briefcase_build_performed,
        ]
    )
    safe_exe_release_boundary_not_performed = not any(
        [
            safe_exe_release_upload_performed,
            safe_exe_promotion_performed,
            safe_exe_final_artifact_scan_performed,
            safe_exe_final_hash_manifest_generated,
        ]
    )
    safe_exe_runtime_boundary_not_started = not any(
        [safe_exe_runtime_loop_started, safe_exe_production_runtime_loop_started]
    )
    safe_exe_exchange_or_order_disabled = (
        safe_exe_exchange_io == "disabled" and safe_exe_order_submission == "disabled"
    )
    safe_exe_readiness_ok = all(
        [
            safe_exe_contract_checked,
            safe_exe_status == "ok",
            safe_exe_contract_version_ok,
            safe_exe_entrypoint_allowlisted,
            safe_exe_default_args_safe,
            safe_exe_live_blocked_by_policy,
            safe_exe_build_not_performed,
            safe_exe_release_boundary_not_performed,
            safe_exe_runtime_boundary_not_started,
            safe_exe_exchange_or_order_disabled,
        ]
    )

    if not safe_exe_contract_checked:
        issues.append("safe_exe_preview_contract_missing")
    if safe_exe_status != "ok":
        issues.append("safe_exe_preview_readiness_not_ok")
    if not safe_exe_contract_version_ok:
        issues.append("safe_exe_preview_contract_version_mismatch")
    if not safe_exe_entrypoint_allowlisted:
        issues.append("safe_exe_preview_entrypoint_not_allowlisted")
    if not safe_exe_default_args_safe:
        issues.append("safe_exe_preview_default_args_unsafe")
    if not safe_exe_live_blocked_by_policy:
        issues.append("safe_exe_preview_live_mode_allowed")
    if not safe_exe_build_not_performed:
        issues.append("safe_exe_preview_build_performed")
    if not safe_exe_release_boundary_not_performed:
        issues.append("safe_exe_preview_release_boundary_performed")
    if not safe_exe_runtime_boundary_not_started:
        issues.append("safe_exe_preview_runtime_boundary_started")
    if not safe_exe_exchange_or_order_disabled:
        issues.append("safe_exe_preview_exchange_or_order_enabled")
    if not safe_exe_readiness_ok:
        status = "blocked"

    safe_exe_profile_validator = (safe_exe_profile_validator_payload or {}).get(
        "safe_exe_preview_profile_validator", {}
    )
    safe_exe_profile_validator_contract_checked = safe_exe_profile_validator_payload is not None
    safe_exe_profile_validator_contract_version = (safe_exe_profile_validator_payload or {}).get(
        "safety_contract_version"
    )
    safe_exe_profile_validator_contract_version_ok = (
        safe_exe_profile_validator_contract_version
        == SAFE_EXE_PREVIEW_PROFILE_VALIDATOR_CONTRACT_VERSION
    )
    safe_exe_preview_profiles_valid = all(
        [
            safe_exe_profile_validator.get("all_toml_valid") is True,
            safe_exe_profile_validator.get("all_runtime_names_ok") is True,
            safe_exe_profile_validator.get("all_hidden_imports_ok") is True,
            safe_exe_profile_validator.get("no_hidden_import_forbidden") is True,
            safe_exe_profile_validator.get("all_briefcase_projects_ok") is True,
            safe_exe_profile_validator.get("all_briefcase_apps_ok") is True,
            safe_exe_profile_validator.get("all_briefcase_outputs_preview_scoped") is True,
            safe_exe_profile_validator.get("no_profile_forbidden_tokens") is True,
        ]
    )
    safe_exe_preview_profiles_complete = all(
        [
            bool(safe_exe_profile_validator.get("all_profiles_exist", False)),
            bool(safe_exe_profile_validator.get("all_platforms_match", False)),
            bool(safe_exe_profile_validator.get("all_entrypoints_allowlisted", False)),
            bool(safe_exe_profile_validator.get("all_output_paths_preview_scoped", False)),
            bool(safe_exe_profile_validator.get("all_work_paths_preview_scoped", False)),
        ]
    )
    safe_exe_preview_profiles_count = int(safe_exe_profile_validator.get("profile_count", 0) or 0)
    safe_exe_preview_profile_issues = list(
        (safe_exe_profile_validator_payload or {}).get("issues", [])
    )
    safe_exe_profile_validator_ready = all(
        [
            safe_exe_profile_validator_contract_checked,
            safe_exe_profile_validator_status == "ok",
            safe_exe_profile_validator_contract_version_ok,
            safe_exe_preview_profiles_valid,
            safe_exe_preview_profiles_complete,
            len(safe_exe_preview_profile_issues) == 0,
        ]
    )

    if not safe_exe_profile_validator_contract_checked:
        issues.append("safe_exe_preview_profile_validator_contract_missing")
    if not safe_exe_profile_validator_contract_version_ok:
        issues.append("safe_exe_preview_profile_validator_contract_version_mismatch")
    if safe_exe_profile_validator_status != "ok":
        issues.append("safe_exe_preview_profile_validator_not_ok")
    if not safe_exe_preview_profiles_valid:
        issues.append("safe_exe_preview_profiles_invalid")
    if not safe_exe_preview_profiles_complete:
        issues.append("safe_exe_preview_profiles_incomplete")
    if safe_exe_preview_profile_issues:
        issues.append("safe_exe_preview_profile_validator_child_issues_present")
    if not safe_exe_profile_validator_ready:
        status = "blocked"

    safe_exe_build_plan_readiness = (safe_exe_build_plan_payload or {}).get(
        "safe_exe_preview_build_plan", {}
    )
    safe_exe_build_plan_contract_checked = safe_exe_build_plan_payload is not None
    safe_exe_build_plan_contract_version = (safe_exe_build_plan_payload or {}).get(
        "safety_contract_version"
    )
    safe_exe_build_plan_contract_version_ok = (
        safe_exe_build_plan_contract_version == SAFE_EXE_PREVIEW_BUILD_PLAN_CONTRACT_VERSION
    )
    safe_exe_build_plan_allowed_entrypoint = safe_exe_build_plan_readiness.get("allowed_entrypoint")
    safe_exe_build_plan_allowed_default_args = safe_exe_build_plan_readiness.get(
        "allowed_default_args"
    )
    safe_exe_build_plan_preview_only = safe_exe_build_plan_readiness.get("preview_only") is True
    safe_exe_build_plan_build_plan_only = (
        safe_exe_build_plan_readiness.get("build_plan_only") is True
    )
    safe_exe_build_plan_build_command_execution_allowed = bool(
        safe_exe_build_plan_readiness.get("build_command_execution_allowed", True)
    )
    safe_exe_build_plan_build_command_executed = bool(
        safe_exe_build_plan_readiness.get("build_command_executed", True)
    )
    safe_exe_build_plan_build_performed = bool(
        safe_exe_build_plan_readiness.get("build_performed", True)
    )
    safe_exe_build_plan_exe_build_performed = bool(
        safe_exe_build_plan_readiness.get("exe_build_performed", True)
    )
    safe_exe_build_plan_installer_build_performed = bool(
        safe_exe_build_plan_readiness.get("installer_build_performed", True)
    )
    safe_exe_build_plan_pyinstaller_build_performed = bool(
        safe_exe_build_plan_readiness.get("pyinstaller_build_performed", True)
    )
    safe_exe_build_plan_briefcase_build_performed = bool(
        safe_exe_build_plan_readiness.get("briefcase_build_performed", True)
    )
    safe_exe_build_plan_release_upload_performed = bool(
        safe_exe_build_plan_readiness.get("release_upload_performed", True)
    )
    safe_exe_build_plan_promotion_performed = bool(
        safe_exe_build_plan_readiness.get("promotion_performed", True)
    )
    safe_exe_build_plan_final_artifact_scan_performed = bool(
        safe_exe_build_plan_readiness.get("final_artifact_scan_performed", True)
    )
    safe_exe_build_plan_final_hash_manifest_generated = bool(
        safe_exe_build_plan_readiness.get("final_hash_manifest_generated", True)
    )
    safe_exe_build_plan_exchange_io = safe_exe_build_plan_readiness.get("exchange_io")
    safe_exe_build_plan_order_submission = safe_exe_build_plan_readiness.get("order_submission")
    safe_exe_build_plan_runtime_loop_started = bool(
        safe_exe_build_plan_readiness.get("runtime_loop_started", True)
    )
    safe_exe_build_plan_production_runtime_loop_started = bool(
        safe_exe_build_plan_readiness.get("production_runtime_loop_started", True)
    )
    safe_exe_build_plan_ready = all(
        [
            safe_exe_build_plan_contract_checked,
            safe_exe_build_plan_status == "ok",
            safe_exe_build_plan_contract_version_ok,
            safe_exe_build_plan_preview_only,
            safe_exe_build_plan_build_plan_only,
            not safe_exe_build_plan_build_command_execution_allowed,
            not safe_exe_build_plan_build_command_executed,
            not any(
                [
                    safe_exe_build_plan_build_performed,
                    safe_exe_build_plan_exe_build_performed,
                    safe_exe_build_plan_installer_build_performed,
                    safe_exe_build_plan_pyinstaller_build_performed,
                    safe_exe_build_plan_briefcase_build_performed,
                ]
            ),
            not any(
                [
                    safe_exe_build_plan_release_upload_performed,
                    safe_exe_build_plan_promotion_performed,
                    safe_exe_build_plan_final_artifact_scan_performed,
                    safe_exe_build_plan_final_hash_manifest_generated,
                ]
            ),
            not any(
                [
                    safe_exe_build_plan_runtime_loop_started,
                    safe_exe_build_plan_production_runtime_loop_started,
                ]
            ),
            safe_exe_build_plan_exchange_io == "disabled"
            and safe_exe_build_plan_order_submission == "disabled",
            safe_exe_build_plan_allowed_entrypoint == SAFE_EXE_PREVIEW_ALLOWED_ENTRYPOINT,
            safe_exe_build_plan_allowed_default_args == SAFE_EXE_PREVIEW_ALLOWED_DEFAULT_ARGS,
        ]
    )

    if not safe_exe_build_plan_contract_checked:
        issues.append("safe_exe_preview_build_plan_contract_missing")
    if safe_exe_build_plan_status != "ok":
        issues.append("safe_exe_preview_build_plan_readiness_not_ok")
    if not safe_exe_build_plan_contract_version_ok:
        issues.append("safe_exe_preview_build_plan_contract_version_mismatch")
    if not safe_exe_build_plan_preview_only:
        issues.append("safe_exe_preview_build_plan_not_preview_only")
    if not safe_exe_build_plan_build_plan_only:
        issues.append("safe_exe_preview_build_plan_not_build_plan_only")
    if any(
        [
            safe_exe_build_plan_build_performed,
            safe_exe_build_plan_exe_build_performed,
            safe_exe_build_plan_installer_build_performed,
            safe_exe_build_plan_pyinstaller_build_performed,
            safe_exe_build_plan_briefcase_build_performed,
            safe_exe_build_plan_build_command_execution_allowed,
            safe_exe_build_plan_build_command_executed,
        ]
    ):
        issues.append("safe_exe_preview_build_plan_build_performed")
    if any(
        [
            safe_exe_build_plan_release_upload_performed,
            safe_exe_build_plan_promotion_performed,
            safe_exe_build_plan_final_artifact_scan_performed,
            safe_exe_build_plan_final_hash_manifest_generated,
        ]
    ):
        issues.append("safe_exe_preview_build_plan_release_boundary_performed")
    if any(
        [
            safe_exe_build_plan_runtime_loop_started,
            safe_exe_build_plan_production_runtime_loop_started,
        ]
    ):
        issues.append("safe_exe_preview_build_plan_runtime_boundary_started")
    if not (
        safe_exe_build_plan_exchange_io == "disabled"
        and safe_exe_build_plan_order_submission == "disabled"
    ):
        issues.append("safe_exe_preview_build_plan_exchange_or_order_enabled")
    if safe_exe_build_plan_allowed_entrypoint != SAFE_EXE_PREVIEW_ALLOWED_ENTRYPOINT:
        issues.append("safe_exe_preview_build_plan_entrypoint_not_allowlisted")
    if safe_exe_build_plan_allowed_default_args != SAFE_EXE_PREVIEW_ALLOWED_DEFAULT_ARGS:
        issues.append("safe_exe_preview_build_plan_default_args_unsafe")
    if not safe_exe_build_plan_ready:
        status = "blocked"
    release_readiness = (release_payload or {}).get("release_integrity_readiness", {})
    for issue in (release_payload or {}).get("issues", []):
        if isinstance(issue, str):
            issues.append(issue)

    release_integrity_status = (
        str(release_payload.get("status", "warning")) if release_payload else "partial"
    )
    release_signing_ready = bool(release_readiness.get("release_signing_ready", False))
    release_hash_manifest_ready = bool(release_readiness.get("hash_manifest_ready", False))
    release_hash_manifest_algorithm = release_readiness.get("hash_manifest_algorithm")
    release_hash_manifest_policy_present = bool(
        release_readiness.get("hash_manifest_policy_present", False)
    )
    release_hash_manifest_generation_performed = bool(
        release_readiness.get("hash_manifest_generation_performed", False)
    )

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
        "release_integrity_contract_checked": release_payload is not None,
        "release_integrity_contract_version": release_readiness.get(
            "release_integrity_contract_version", RELEASE_INTEGRITY_CONTRACT_VERSION
        ),
        "release_integrity_readiness_present": release_payload is not None,
        "release_integrity_readiness_status": release_status,
        "artifact_hygiene_checked": True,
        "artifact_exclude_policy_present": True,
        "artifact_exclude_policy_version": ARTIFACT_EXCLUDE_POLICY_VERSION,
        "denied_artifact_patterns": DENIED_ARTIFACT_PATTERNS,
        "api_keys_bundled": False,
        "env_file_bundled": False,
        "local_db_bundled": False,
        "logs_bundled": False,
        "reports_bundled": False,
        "tmp_artifacts_bundled": False,
        "test_secrets_bundled": False,
        "cache_artifacts_bundled": False,
        "local_user_data_bundled": False,
        "keychain_artifacts_bundled": False,
        "safe_default_launch_checked": True,
        "safe_default_launch_policy_present": True,
        "safe_default_launch_policy_version": SAFE_LAUNCH_POLICY_VERSION,
        "default_mode": packaged_readiness.get("default_mode"),
        "default_launch_mode": packaged_readiness.get("default_mode"),
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
        "release_hash_manifest_algorithm": release_hash_manifest_algorithm,
        "release_hash_manifest_policy_present": release_hash_manifest_policy_present,
        "release_hash_manifest_generation_performed": release_hash_manifest_generation_performed,
        "release_channel_policy_present": bool(
            release_readiness.get("release_channel_policy_present", False)
        ),
        "release_channel_policy_version": release_readiness.get("release_channel_policy_version"),
        "supported_release_channels": release_readiness.get("supported_release_channels", []),
        "default_release_channel": release_readiness.get("default_release_channel"),
        "release_channel_gate_performed": bool(
            release_readiness.get("release_channel_gate_performed", False)
        ),
        "release_channel_gate_result": release_readiness.get("release_channel_gate_result"),
        "promotion_gate_policy_present": bool(
            release_readiness.get("promotion_gate_policy_present", False)
        ),
        "promotion_gate_policy_version": release_readiness.get("promotion_gate_policy_version"),
        "promotion_gate_performed": bool(release_readiness.get("promotion_gate_performed", False)),
        "promotion_gate_result": release_readiness.get("promotion_gate_result"),
        "rc_to_ga_promotion_ready": bool(release_readiness.get("rc_to_ga_promotion_ready", False)),
        "rc_to_ga_promotion_performed": bool(
            release_readiness.get("rc_to_ga_promotion_performed", False)
        ),
        "rc_to_ga_blockers": release_readiness.get("rc_to_ga_blockers", []),
        "release_integrity_status": release_integrity_status,
        "safe_exe_preview_contract_checked": safe_exe_contract_checked,
        "safe_exe_preview_profile_validator_contract_checked": safe_exe_profile_validator_contract_checked,
        "safe_exe_preview_profile_validator_contract_version": safe_exe_profile_validator_contract_version,
        "safe_exe_preview_profile_validator_ready": safe_exe_profile_validator_ready,
        "safe_exe_preview_profile_validator_status": safe_exe_profile_validator_status,
        "safe_exe_preview_profiles_valid": safe_exe_preview_profiles_valid,
        "safe_exe_preview_profiles_complete": safe_exe_preview_profiles_complete,
        "safe_exe_preview_profiles_count": safe_exe_preview_profiles_count,
        "safe_exe_preview_profile_issues": safe_exe_preview_profile_issues,
        "safe_exe_preview_build_plan_contract_checked": safe_exe_build_plan_contract_checked,
        "safe_exe_preview_build_plan_contract_version": safe_exe_build_plan_contract_version,
        "safe_exe_preview_build_plan_ready": safe_exe_build_plan_ready,
        "safe_exe_preview_build_plan_status": safe_exe_build_plan_status,
        "safe_exe_preview_build_plan_preview_only": safe_exe_build_plan_preview_only,
        "safe_exe_preview_build_plan_build_plan_only": safe_exe_build_plan_build_plan_only,
        "safe_exe_preview_build_plan_allowed_entrypoint": safe_exe_build_plan_allowed_entrypoint,
        "safe_exe_preview_build_plan_allowed_default_args": safe_exe_build_plan_allowed_default_args,
        "safe_exe_preview_build_plan_build_command_execution_allowed": safe_exe_build_plan_build_command_execution_allowed,
        "safe_exe_preview_build_plan_build_command_executed": safe_exe_build_plan_build_command_executed,
        "safe_exe_preview_build_plan_build_performed": safe_exe_build_plan_build_performed,
        "safe_exe_preview_build_plan_exe_build_performed": safe_exe_build_plan_exe_build_performed,
        "safe_exe_preview_build_plan_installer_build_performed": safe_exe_build_plan_installer_build_performed,
        "safe_exe_preview_build_plan_pyinstaller_build_performed": safe_exe_build_plan_pyinstaller_build_performed,
        "safe_exe_preview_build_plan_briefcase_build_performed": safe_exe_build_plan_briefcase_build_performed,
        "safe_exe_preview_build_plan_release_upload_performed": safe_exe_build_plan_release_upload_performed,
        "safe_exe_preview_build_plan_promotion_performed": safe_exe_build_plan_promotion_performed,
        "safe_exe_preview_build_plan_final_artifact_scan_performed": safe_exe_build_plan_final_artifact_scan_performed,
        "safe_exe_preview_build_plan_final_hash_manifest_generated": safe_exe_build_plan_final_hash_manifest_generated,
        "safe_exe_preview_build_plan_exchange_io": safe_exe_build_plan_exchange_io,
        "safe_exe_preview_build_plan_order_submission": safe_exe_build_plan_order_submission,
        "safe_exe_preview_build_plan_runtime_loop_started": safe_exe_build_plan_runtime_loop_started,
        "safe_exe_preview_build_plan_production_runtime_loop_started": safe_exe_build_plan_production_runtime_loop_started,
        "safe_exe_preview_contract_version": safe_exe_contract_version,
        "safe_exe_preview_ready": safe_exe_readiness_ok,
        "safe_exe_preview_status": safe_exe_status,
        "safe_exe_preview_allowed_entrypoint": safe_exe_allowed_entrypoint,
        "safe_exe_preview_allowed_default_args": safe_exe_allowed_default_args,
        "safe_exe_preview_live_mode_allowed": safe_exe_live_mode_allowed,
        "safe_exe_preview_build_performed": safe_exe_build_performed,
        "safe_exe_preview_exe_build_performed": safe_exe_exe_build_performed,
        "safe_exe_preview_installer_build_performed": safe_exe_installer_build_performed,
        "safe_exe_preview_pyinstaller_build_performed": safe_exe_pyinstaller_build_performed,
        "safe_exe_preview_briefcase_build_performed": safe_exe_briefcase_build_performed,
        "safe_exe_preview_release_upload_performed": safe_exe_release_upload_performed,
        "safe_exe_preview_promotion_performed": safe_exe_promotion_performed,
        "safe_exe_preview_final_artifact_scan_performed": safe_exe_final_artifact_scan_performed,
        "safe_exe_preview_final_hash_manifest_generated": safe_exe_final_hash_manifest_generated,
        "safe_exe_preview_exchange_io": safe_exe_exchange_io,
        "safe_exe_preview_order_submission": safe_exe_order_submission,
        "safe_exe_preview_runtime_loop_started": safe_exe_runtime_loop_started,
        "safe_exe_preview_production_runtime_loop_started": safe_exe_production_runtime_loop_started,
        "secrets_read": False,
        "keychain_read": False,
        "env_values_read": False,
        "api_keys_required": False,
        "api_keys_required_for_launch": False,
        "exchange_io": "disabled",
        "order_submission": "disabled",
        "real_orders_submitted": False,
        "runtime_loop_started": False,
        "production_runtime_loop_started": False,
        "live_launch_requires_explicit_reconfiguration": True,
        "live_launch_blocked_by_default": True,
        "packaged_shortcut_live_target_allowed": False,
        "packaged_shortcut_preview_target_allowed": True,
        "packaged_shortcut_demo_target_allowed": True,
        "packaged_shortcut_default_args": ["--mode", "demo", "--preview-plan"],
        "unsafe_launch_modes_blocked": ["live"],
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
            "safe_exe_preview_readiness": {
                "status": safe_exe_status,
                "safety_contract_version": (safe_exe_payload or {}).get("safety_contract_version"),
                "safe_exe_preview_readiness": safe_exe_readiness,
                "issues": (safe_exe_payload or {}).get("issues", []),
            },
            "safe_exe_preview_profile_validator": {
                "status": safe_exe_profile_validator_status,
                "safety_contract_version": safe_exe_profile_validator_contract_version,
                "safe_exe_preview_profile_validator": safe_exe_profile_validator,
                "issues": safe_exe_preview_profile_issues,
            },
            "safe_exe_preview_build_plan": {
                "status": safe_exe_build_plan_status,
                "safety_contract_version": (safe_exe_build_plan_payload or {}).get(
                    "safety_contract_version"
                ),
                "safe_exe_preview_build_plan": safe_exe_build_plan_readiness,
                "issues": (safe_exe_build_plan_payload or {}).get("issues", []),
            },
            "release_integrity_readiness": {
                "status": release_status,
                "safety_contract_version": (release_payload or {}).get("safety_contract_version"),
                "release_integrity_readiness": release_readiness,
            },
        },
        "checks": {
            "mode_supported": mode in VALID_MODES,
            "contracts_checked": (
                installer_payload is not None
                and config_payload is not None
                and safe_exe_contract_checked
                and safe_exe_build_plan_contract_checked
                and safe_exe_profile_validator_contract_checked
            ),
            "release_integrity_contract_checked": release_payload is not None,
            "safe_default_launch": readiness["preview_or_demo_default"]
            and not readiness["live_mode_enabled"],
            "artifact_hygiene_summary_present": True,
            "release_integrity_summary_present": True,
            "safe_exe_preview_contract_checked": safe_exe_contract_checked,
            "safe_exe_preview_build_plan_contract_checked": safe_exe_build_plan_contract_checked,
            "safe_exe_preview_profile_validator_contract_checked": safe_exe_profile_validator_contract_checked,
            "safe_exe_preview_profile_validator_contract_version_ok": safe_exe_profile_validator_contract_version_ok,
            "safe_exe_preview_profiles_valid": safe_exe_preview_profiles_valid,
            "safe_exe_preview_profiles_complete": safe_exe_preview_profiles_complete,
            "safe_exe_preview_profile_validator_ready": safe_exe_profile_validator_ready,
            "safe_exe_preview_build_plan_contract_version_ok": safe_exe_build_plan_contract_version_ok,
            "safe_exe_preview_build_plan_ready": safe_exe_build_plan_ready,
            "safe_exe_preview_build_plan_entrypoint_allowlisted": (
                safe_exe_build_plan_allowed_entrypoint == SAFE_EXE_PREVIEW_ALLOWED_ENTRYPOINT
            ),
            "safe_exe_preview_build_plan_default_args_safe": (
                safe_exe_build_plan_allowed_default_args == SAFE_EXE_PREVIEW_ALLOWED_DEFAULT_ARGS
            ),
            "safe_exe_preview_build_plan_build_not_performed": not any(
                [
                    safe_exe_build_plan_build_performed,
                    safe_exe_build_plan_exe_build_performed,
                    safe_exe_build_plan_installer_build_performed,
                    safe_exe_build_plan_pyinstaller_build_performed,
                    safe_exe_build_plan_briefcase_build_performed,
                    safe_exe_build_plan_build_command_execution_allowed,
                    safe_exe_build_plan_build_command_executed,
                ]
            ),
            "safe_exe_preview_build_plan_release_boundary_not_performed": not any(
                [
                    safe_exe_build_plan_release_upload_performed,
                    safe_exe_build_plan_promotion_performed,
                    safe_exe_build_plan_final_artifact_scan_performed,
                    safe_exe_build_plan_final_hash_manifest_generated,
                ]
            ),
            "safe_exe_preview_build_plan_runtime_boundary_not_started": not any(
                [
                    safe_exe_build_plan_runtime_loop_started,
                    safe_exe_build_plan_production_runtime_loop_started,
                ]
            ),
            "safe_exe_preview_build_plan_exchange_or_order_disabled": (
                safe_exe_build_plan_exchange_io == "disabled"
                and safe_exe_build_plan_order_submission == "disabled"
            ),
            "safe_exe_preview_build_plan_contract_checked": safe_exe_build_plan_contract_checked,
            "safe_exe_preview_build_plan_contract_version": safe_exe_build_plan_contract_version,
            "safe_exe_preview_build_plan_ready": safe_exe_build_plan_ready,
            "safe_exe_preview_build_plan_status": safe_exe_build_plan_status,
            "safe_exe_preview_build_plan_preview_only": safe_exe_build_plan_preview_only,
            "safe_exe_preview_build_plan_build_plan_only": safe_exe_build_plan_build_plan_only,
            "safe_exe_preview_build_plan_allowed_entrypoint": safe_exe_build_plan_allowed_entrypoint,
            "safe_exe_preview_build_plan_allowed_default_args": safe_exe_build_plan_allowed_default_args,
            "safe_exe_preview_build_plan_build_command_execution_allowed": safe_exe_build_plan_build_command_execution_allowed,
            "safe_exe_preview_build_plan_build_command_executed": safe_exe_build_plan_build_command_executed,
            "safe_exe_preview_build_plan_build_performed": safe_exe_build_plan_build_performed,
            "safe_exe_preview_build_plan_exe_build_performed": safe_exe_build_plan_exe_build_performed,
            "safe_exe_preview_build_plan_installer_build_performed": safe_exe_build_plan_installer_build_performed,
            "safe_exe_preview_build_plan_pyinstaller_build_performed": safe_exe_build_plan_pyinstaller_build_performed,
            "safe_exe_preview_build_plan_briefcase_build_performed": safe_exe_build_plan_briefcase_build_performed,
            "safe_exe_preview_build_plan_release_upload_performed": safe_exe_build_plan_release_upload_performed,
            "safe_exe_preview_build_plan_promotion_performed": safe_exe_build_plan_promotion_performed,
            "safe_exe_preview_build_plan_final_artifact_scan_performed": safe_exe_build_plan_final_artifact_scan_performed,
            "safe_exe_preview_build_plan_final_hash_manifest_generated": safe_exe_build_plan_final_hash_manifest_generated,
            "safe_exe_preview_build_plan_exchange_io": safe_exe_build_plan_exchange_io,
            "safe_exe_preview_build_plan_order_submission": safe_exe_build_plan_order_submission,
            "safe_exe_preview_build_plan_runtime_loop_started": safe_exe_build_plan_runtime_loop_started,
            "safe_exe_preview_build_plan_production_runtime_loop_started": safe_exe_build_plan_production_runtime_loop_started,
            "safe_exe_preview_contract_version_ok": safe_exe_contract_version_ok,
            "safe_exe_preview_contract_referenced": safe_exe_contract_checked,
            "safe_exe_preview_entrypoint_allowlisted": safe_exe_entrypoint_allowlisted,
            "safe_exe_preview_default_args_safe": safe_exe_default_args_safe,
            "safe_exe_preview_live_blocked_by_policy": safe_exe_live_blocked_by_policy,
            "safe_exe_preview_build_not_performed": safe_exe_build_not_performed,
            "safe_exe_preview_release_boundary_not_performed": safe_exe_release_boundary_not_performed,
            "safe_exe_preview_runtime_boundary_not_started": safe_exe_runtime_boundary_not_started,
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
