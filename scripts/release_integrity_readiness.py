from __future__ import annotations

import argparse
import json
from pathlib import Path

SAFETY_CONTRACT_VERSION = "release_integrity_readiness.v1"
VALID_MODES = {"prebuild", "release"}
RELEASE_PROCESS_DOC = Path("docs/deploy/release_process.md")


def _read_text_if_exists(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").lower()


def build_payload(mode: str) -> tuple[dict[str, object], int]:
    issues: list[str] = []
    release_process_docs_present = RELEASE_PROCESS_DOC.exists()
    release_process_docs_path = (
        RELEASE_PROCESS_DOC.as_posix() if release_process_docs_present else None
    )
    release_process_text = _read_text_if_exists(RELEASE_PROCESS_DOC)

    release_channel_policy_present = all(
        token in release_process_text for token in ("dev", "rc", "stabil")
    )
    dev_test_release_channels_documented = all(
        token in release_process_text for token in ("dev", "rc", "ga")
    )
    rc_ga_promotion_policy_present = "rc" in release_process_text and "ga" in release_process_text
    hash_manifest_policy_present = any(
        token in release_process_text
        for token in ("manifest", "checksum", "sha256", "sha384", "hash")
    )
    signing_policy_present = any(
        token in release_process_text for token in ("sign", "signature", "certificate", "codesign")
    )

    if not release_process_docs_present:
        issues.append("release_process_docs_missing")
    if not release_channel_policy_present:
        issues.append("release_channel_policy_missing")

    source_commit_available = bool(Path(".git").exists())

    issues.extend(
        [
            "release_signing_not_ready",
            "codesign_not_ready",
            "notarization_not_ready",
            "artifact_scan_not_performed",
            "release_hash_manifest_prebuild_only",
            "final_artifact_scan_not_performed",
            "release_channel_gate_not_performed",
            "ga_release_not_ready",
        ]
    )

    status = "warning"
    if not release_process_docs_present:
        status = "blocked"

    readiness = {
        "enabled": True,
        "static_only": True,
        "local_only": True,
        "release_integrity_contract_present": True,
        "release_integrity_contract_version": SAFETY_CONTRACT_VERSION,
        "release_process_docs_present": release_process_docs_present,
        "release_process_docs_path": release_process_docs_path,
        "release_channel_policy_present": release_channel_policy_present,
        "release_channel_policy_version": "release_channel_policy.v1",
        "supported_release_channels": ["dev", "test", "rc", "ga"],
        "default_release_channel": "dev",
        "current_release_channel": "dev",
        "dev_channel_allowed_without_signing": True,
        "test_channel_allowed_without_signing": True,
        "rc_channel_requires_hash_manifest": True,
        "rc_channel_requires_artifact_scan": True,
        "rc_channel_requires_signing_decision": True,
        "ga_channel_requires_hash_manifest": True,
        "ga_channel_requires_artifact_scan": True,
        "ga_channel_requires_signing": True,
        "ga_channel_requires_release_notes": True,
        "ga_channel_requires_source_commit": True,
        "ga_channel_requires_build_id": True,
        "promotion_policy_present": rc_ga_promotion_policy_present,
        "rc_to_ga_promotion_requires_clean_security_manifest": True,
        "rc_to_ga_promotion_requires_no_known_blockers": True,
        "release_channel_gate_performed": False,
        "release_channel_gate_result": "not_performed",
        "release_channel_is_prebuild_policy_only": True,
        "dev_test_release_channels_documented": dev_test_release_channels_documented,
        "rc_ga_promotion_policy_present": rc_ga_promotion_policy_present,
        "hash_manifest_policy_present": hash_manifest_policy_present,
        "hash_manifest_algorithm": "sha256" if hash_manifest_policy_present else None,
        "hash_manifest_ready": hash_manifest_policy_present,
        "hash_manifest_required_for_release": True,
        "hash_manifest_generation_performed": False,
        "hash_manifest_artifact_path": None,
        "hash_manifest_artifact_exists": False,
        "hash_manifest_includes_source_commit": True,
        "hash_manifest_includes_build_id": True,
        "hash_manifest_includes_artifact_size": True,
        "hash_manifest_includes_artifact_sha": True,
        "hash_manifest_is_prebuild_policy_only": True,
        "hash_manifest_final_artifact_scan_required": True,
        "hash_manifest_final_artifact_scan_performed": False,
        "source_commit_required": True,
        "source_commit_available": source_commit_available,
        "build_id_required": True,
        "build_id_available": False,
        "signing_policy_present": signing_policy_present,
        "release_signing_ready": False,
        "codesign_ready": False,
        "notarization_ready": False,
        "certificate_material_required": True,
        "certificate_material_read": False,
        "artifact_scan_required": True,
        "artifact_scan_performed": False,
        "artifact_exists": False,
        "artifact_built": False,
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
        "release_integrity_readiness": readiness,
        "checks": {
            "mode_supported": mode in VALID_MODES,
            "docs_present": release_process_docs_present,
            "release_signing_unready_expected": True,
            "artifact_scan_unperformed_expected": True,
            "hash_manifest_policy": hash_manifest_policy_present,
            "hash_manifest_algorithm": ("sha256" if hash_manifest_policy_present else None),
            "hash_manifest_generation": False,
            "final_artifact_scan": False,
        },
        "issues": sorted(set(issues)),
        "safety_contract_version": SAFETY_CONTRACT_VERSION,
    }
    return payload, (2 if status == "blocked" else 0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Static release integrity readiness contract")
    parser.add_argument("--mode", choices=sorted(VALID_MODES), default="prebuild")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload, code = build_payload(args.mode)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
