"""Evaluate independent platform gates from the canonical deployment contract."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from deployment.windows_stage9_evidence_contract import PRODUCTION_TRUST_CEREMONY_ID

CONTRACT = Path(__file__).with_name("platform_readiness.json")
PASS = "PASS"
PLATFORMS = ("WINDOWS", "LINUX", "MACOS")
RELEASE_PRIORITY = {"WINDOWS": "PRIMARY", "LINUX": "SUPPORTED", "MACOS": "DEFERRED"}
RELEASE_PROFILES = {
    "WINDOWS_BETA": ["WINDOWS"],
    "WINDOWS_LINUX_BETA": ["WINDOWS", "LINUX"],
    "CRYPTOHUNTER_V1": ["WINDOWS", "LINUX"],
}
POLICY_FIELDS = {
    "schema_version",
    "reference_platform",
    "platforms",
    "release_priority",
    "compatibility_platforms",
    "release_profiles",
    "allowed_statuses",
    "semantic_invariants",
    "mechanisms",
    "matrix",
    "acceptance",
    "release_gates",
    "legacy_flags",
}
ALLOWED_STATUSES = {
    "PASS",
    "FAIL",
    "NOT_IMPLEMENTED",
    "NOT_TESTED",
    "UNVERIFIED_ENVIRONMENT_LIMITATION",
    "NOT_APPLICABLE",
}
STATIC_ITEMS = {
    "WINDOWS_PATH_LAYOUT_STATIC_CONTRACT",
    "LINUX_STATIC_QUALIFICATION",
    "LINUX_INSTALLER_STATIC_CONTRACT",
}
# Frozen native requirements survive a release-scope change, including deferred macOS.
NATIVE_REQUIREMENTS = {
    "WINDOWS_PRODUCTION_READY": [
        "CORE_REQUIRED_SUITES",
        "WINDOWS_CLEAN_INSTALL",
        "WINDOWS_SERVICE_INSTALLATION",
        "WINDOWS_SERVICE_START",
        "WINDOWS_GRACEFUL_STOP",
        "WINDOWS_MANUAL_RESTART",
        "WINDOWS_AUTOMATIC_CRASH_RESTART",
        "WINDOWS_REBOOT_RECOVERY",
        "WINDOWS_AUTOSTART",
        "WINDOWS_LONG_RUNNING_LIFECYCLE",
        "WINDOWS_PERSISTENT_LOGGING",
        "WINDOWS_PROTECTED_CONFIGURATION",
        "WINDOWS_PROTECTED_STATE_PATHS",
        "WINDOWS_ACL_QUALIFICATION",
        "WINDOWS_LOCAL_PRINCIPAL_AUTHENTICATION",
        "WINDOWS_SQLITE_CRASH_INTEGRITY",
        "WINDOWS_FILE_LOCKING",
        "WINDOWS_PATH_LAYOUT_STATIC_CONTRACT",
        "WINDOWS_NATIVE_PATH_INTEGRATION",
        "WINDOWS_PROCESS_TREE",
        "WINDOWS_UPDATE_RESTART",
        "WINDOWS_FAILED_UPDATE_SAFE_BEHAVIOR",
        "WINDOWS_NO_ORPHAN_CHILDREN",
        "WINDOWS_NETWORK_RECOVERY",
        "WINDOWS_SAFE_OS_SHUTDOWN",
        "WINDOWS_POSTGRESQL_SUBSTRATE",
    ],
    "LINUX_PRODUCTION_READY": [
        "CORE_REQUIRED_SUITES",
        "LINUX_STATIC_QUALIFICATION",
        "LINUX_SYSTEMD_INTEGRATION",
        "LINUX_INSTALLER_STATIC_CONTRACT",
        "LINUX_CLEAN_INSTALL",
        "LINUX_REBOOT_LIFECYCLE_UPDATE",
    ],
    "MACOS_PRODUCTION_READY": [
        "CORE_REQUIRED_SUITES",
        "MACOS_LAUNCHD_INTEGRATION",
        "MACOS_PACKAGE_PERMISSIONS_AUTH_IPC",
        "MACOS_REBOOT_LIFECYCLE_UPDATE",
    ],
}
SEMANTIC_INVARIANTS = [
    "distinct runtime, verifier, and administrator principals",
    "database observes authenticated identity; caller claims are never authority",
    "least privilege and no PUBLIC grants",
    "protected local-only IPC and protected state/configuration",
    "fail-closed runtime qualification only reads and verifies permissions",
    "bounded graceful shutdown, crash supervision, autostart, and process-tree cleanup",
    "core update state is separate from privileged platform handoff",
]
PLATFORM_MATRIX = {
    "core_tests": {"WINDOWS": "REQUIRED", "LINUX": "REQUIRED", "MACOS": "REQUIRED"},
    "platform_integration": {
        "WINDOWS": "REQUIRED",
        "LINUX": "REQUIRED_BEFORE_LINUX_RELEASE",
        "MACOS": "REQUIRED_BEFORE_MACOS_RELEASE",
    },
    "installer_package": {
        "WINDOWS": "REQUIRED",
        "LINUX": "REQUIRED_BEFORE_LINUX_RELEASE",
        "MACOS": "REQUIRED_BEFORE_MACOS_RELEASE",
    },
    "long_running_lifecycle": {
        "WINDOWS": "REQUIRED",
        "LINUX": "REQUIRED_BEFORE_LINUX_RELEASE",
        "MACOS": "REQUIRED_BEFORE_MACOS_RELEASE",
    },
    "reboot_recovery": {
        "WINDOWS": "REQUIRED",
        "LINUX": "REQUIRED_BEFORE_LINUX_RELEASE",
        "MACOS": "REQUIRED_BEFORE_MACOS_RELEASE",
    },
    "update_restart": {
        "WINDOWS": "REQUIRED",
        "LINUX": "REQUIRED_BEFORE_LINUX_RELEASE",
        "MACOS": "REQUIRED_BEFORE_MACOS_RELEASE",
    },
}

REQUIRES_EXECUTION_EVIDENCE = {
    "LIVE_WINDOWS_INTEGRATION",
    "LIVE_LINUX_INTEGRATION",
    "LIVE_MACOS_INTEGRATION",
    "CROSS_OS_CI_MATRIX",
}
BASE_EVIDENCE_FIELDS = {
    "schema_version",
    "platform",
    "source_revision",
    "ci_provider",
    "ci_run_id",
    "runner_os",
    "runner_arch",
    "generated_at_utc",
    "results",
}
PRODUCTION_TRUST_FIELDS = {
    "ceremony_id",
    "package_manifest_sha256",
    "artifact_run_id",
    "frozen_production_trust",
}


class EvidenceValidationError(ValueError):
    pass


class PolicyValidationError(ValueError):
    """The policy is incomplete, unsupported, or weakens frozen requirements."""


def validate_contract(data: object) -> None:
    """Validate policy V2; historical V1 is never silently reinterpreted.

    Evidence envelopes remain V1. This validates policy semantics, not the
    provenance of supplied policy files or authorization to publish a release.
    """
    if not isinstance(data, dict):
        raise PolicyValidationError("INVALID_POLICY_FIELDS")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 2:
        raise PolicyValidationError("UNSUPPORTED_POLICY_VERSION")
    if set(data) != POLICY_FIELDS:
        raise PolicyValidationError("INVALID_POLICY_FIELDS")
    if (
        data["reference_platform"] != "WINDOWS"
        or data["platforms"] != list(PLATFORMS)
        or data["compatibility_platforms"] != list(PLATFORMS)
        or data["release_priority"] != RELEASE_PRIORITY
        or data["release_profiles"] != RELEASE_PROFILES
        or data["semantic_invariants"] != SEMANTIC_INVARIANTS
        or data["matrix"] != PLATFORM_MATRIX
    ):
        raise PolicyValidationError("INVALID_PLATFORM_POLICY")
    statuses = data["allowed_statuses"]
    if (
        not isinstance(statuses, list)
        or not all(isinstance(value, str) for value in statuses)
        or len(statuses) != len(ALLOWED_STATUSES)
        or set(statuses) != ALLOWED_STATUSES
    ):
        raise PolicyValidationError("INVALID_POLICY_STATUSES")
    if data["release_gates"] != NATIVE_REQUIREMENTS:
        raise PolicyValidationError("INVALID_NATIVE_REQUIREMENTS")
    acceptance = data["acceptance"]
    expected_items = {item for items in NATIVE_REQUIREMENTS.values() for item in items}
    expected_items.add("WINDOWS_GUI_BACKEND_LIFECYCLE")
    if not isinstance(acceptance, dict) or set(acceptance) != expected_items:
        raise PolicyValidationError("INVALID_ACCEPTANCE_INVENTORY")
    for name, item in acceptance.items():
        owner = "CROSS_PLATFORM_CORE" if name == "CORE_REQUIRED_SUITES" else name.split("_", 1)[0]
        evidence_class = (
            "CROSS_OS_CI_MATRIX"
            if name == "CORE_REQUIRED_SUITES"
            else "STATIC_EXECUTABLE_CONTRACT"
            if name in STATIC_ITEMS
            else f"LIVE_{owner}_INTEGRATION"
        )
        if (
            not isinstance(item, dict)
            or set(item) != {"owner", "status", "evidence", "evidence_class"}
            or item["owner"] != owner
            or item["evidence_class"] != evidence_class
            or not isinstance(item["status"], str)
            or item["status"] not in ALLOWED_STATUSES
            or not isinstance(item["evidence"], str)
            or not item["evidence"]
        ):
            raise PolicyValidationError("INVALID_ACCEPTANCE_ITEM")
    mechanisms = data["mechanisms"]
    mechanism_fields = {
        "service_manager",
        "identity",
        "ipc",
        "permissions",
        "principal_auth",
        "paths",
    }
    if not isinstance(mechanisms, dict) or set(mechanisms) != set(PLATFORMS):
        raise PolicyValidationError("INVALID_MECHANISMS")
    for mechanism in mechanisms.values():
        if (
            not isinstance(mechanism, dict)
            or set(mechanism) != mechanism_fields
            or not all(isinstance(value, str) and value for value in mechanism.values())
        ):
            raise PolicyValidationError("INVALID_MECHANISMS")
    flags = data["legacy_flags"]
    flag_names = {
        "FreshnessAuthority_implemented",
        "ROOT_PROOF_ISSUER_IMPLEMENTED",
        "production_substrate_implemented",
        "PRODUCTION_LOCAL_RUNTIME_AVAILABLE",
    }
    if (
        not isinstance(flags, dict)
        or set(flags) != flag_names | {"policy"}
        or not isinstance(flags["policy"], str)
        or not flags["policy"]
        or any(flags[name] is not False for name in flag_names)
    ):
        raise PolicyValidationError("INVALID_LEGACY_FLAGS")


def _policy_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PolicyValidationError("DUPLICATE_POLICY_FIELD")
        result[key] = value
    return result


def load_contract(path: Path = CONTRACT) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_policy_object)
    validate_contract(data)
    return cast(dict[str, Any], data)


def _validated_results(
    data: dict[str, Any],
    evidence: list[dict[str, Any]],
    current_revision: str,
    expected_ci_provider: str,
    expected_ci_run_id: str,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    expected_os = {
        "LIVE_WINDOWS_INTEGRATION": ("WINDOWS", "Windows"),
        "LIVE_LINUX_INTEGRATION": ("LINUX", "Linux"),
        "LIVE_MACOS_INTEGRATION": ("MACOS", "macOS"),
        "CROSS_OS_CI_MATRIX": ("CROSS_PLATFORM_CORE", None),
    }
    for document in evidence:
        if not isinstance(document, dict) or not set(document).issubset(
            BASE_EVIDENCE_FIELDS | {"production_trust"}
        ):
            raise EvidenceValidationError("malformed evidence document")
        if not isinstance(document.get("results"), list):
            raise EvidenceValidationError("results must be a list")
        contains_clean_install = any(
            isinstance(result, dict) and result.get("item") == "WINDOWS_CLEAN_INSTALL"
            for result in document["results"]
        )
        required = BASE_EVIDENCE_FIELDS | (
            {"production_trust"} if contains_clean_install else set()
        )
        if set(document) != required:
            raise EvidenceValidationError("malformed evidence document")
        if contains_clean_install:
            trust = document["production_trust"]
            if (
                not isinstance(trust, dict)
                or set(trust) != PRODUCTION_TRUST_FIELDS
                or trust["ceremony_id"] != PRODUCTION_TRUST_CEREMONY_ID
                or not isinstance(trust["package_manifest_sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", trust["package_manifest_sha256"])
                or not isinstance(trust["artifact_run_id"], str)
                or not re.fullmatch(r"[1-9][0-9]*", trust["artifact_run_id"])
                or trust["frozen_production_trust"] != PASS
            ):
                raise EvidenceValidationError("malformed production trust evidence")
        if (
            type(document["schema_version"]) is not int
            or document["schema_version"] != 1
            or document["source_revision"] != current_revision
        ):
            raise EvidenceValidationError("unsupported schema or stale source revision")
        if document["ci_provider"] != expected_ci_provider:
            raise EvidenceValidationError("untrusted CI provider")
        if document["ci_run_id"] != expected_ci_run_id:
            raise EvidenceValidationError("stale CI run")
        identity_fields = BASE_EVIDENCE_FIELDS - {"schema_version", "results"}
        if not all(isinstance(document[name], str) and document[name] for name in identity_fields):
            raise EvidenceValidationError("empty evidence identity field")
        try:
            if not document["generated_at_utc"].endswith("Z"):
                raise ValueError
            timestamp = datetime.fromisoformat(document["generated_at_utc"].replace("Z", "+00:00"))
            if timestamp.utcoffset() is None:
                raise ValueError
        except ValueError as exc:
            raise EvidenceValidationError("invalid evidence timestamp") from exc
        for result in document["results"]:
            result_fields = {"item", "status", "evidence_class", "test_or_probe", "details"}
            if not isinstance(result, dict) or set(result) != result_fields:
                raise EvidenceValidationError("malformed evidence result")
            if not all(isinstance(result[name], str) and result[name] for name in result_fields):
                raise EvidenceValidationError("empty evidence result field")
            item = result["item"]
            if item not in data["acceptance"]:
                raise EvidenceValidationError(f"unknown evidence item: {item}")
            if item in results:
                raise EvidenceValidationError(f"duplicate evidence item: {item}")
            evidence_class = result["evidence_class"]
            if evidence_class not in REQUIRES_EXECUTION_EVIDENCE:
                raise EvidenceValidationError(
                    f"unsupported execution evidence class: {evidence_class}"
                )
            if evidence_class != data["acceptance"][item]["evidence_class"]:
                raise EvidenceValidationError(f"evidence class mismatch: {item}")
            expected_platform, runner_os = expected_os[evidence_class]
            if document["platform"] != expected_platform:
                raise EvidenceValidationError(f"platform mismatch: {item}")
            if runner_os is not None and document["runner_os"] != runner_os:
                raise EvidenceValidationError(f"runner OS mismatch: {item}")
            if result["status"] not in data["allowed_statuses"]:
                raise EvidenceValidationError(f"invalid evidence status: {item}")
            results[item] = result
    return results


def production_ready(
    platform: str,
    contract: dict[str, Any] | None = None,
    evidence: list[dict[str, Any]] | None = None,
    current_revision: str | None = None,
    expected_ci_provider: str = "test",
    expected_ci_run_id: str = "1",
) -> bool:
    return not blocking_items(
        platform, contract, evidence, current_revision, expected_ci_provider, expected_ci_run_id
    )


def blocking_items(
    platform: str,
    contract: dict[str, Any] | None = None,
    evidence: list[dict[str, Any]] | None = None,
    current_revision: str | None = None,
    expected_ci_provider: str = "test",
    expected_ci_run_id: str = "1",
) -> list[str]:
    """Return required items without PASS evidence for the selected platform."""
    try:
        data = load_contract() if contract is None else contract
        validate_contract(data)
    except (PolicyValidationError, OSError, UnicodeError, json.JSONDecodeError):
        return ["INVALID_POLICY"]
    if type(platform) is not str or platform not in PLATFORMS:
        return ["UNKNOWN_PLATFORM"]
    if data["release_priority"][platform] == "DEFERRED":
        return ["PLATFORM_DEFERRED"]
    if evidence is not None and not isinstance(evidence, list):
        return ["INVALID_EVIDENCE:malformed evidence collection"]
    required = data["release_gates"][f"{platform}_PRODUCTION_READY"]
    try:
        results = _validated_results(
            data,
            evidence or [],
            current_revision or "",
            expected_ci_provider,
            expected_ci_run_id,
        )
    except EvidenceValidationError as exc:
        return [f"INVALID_EVIDENCE:{exc}"]
    blockers = []
    for name in required:
        item = data["acceptance"][name]
        if item["evidence_class"] == "STATIC_EXECUTABLE_CONTRACT":
            if item["status"] != PASS:
                blockers.append(name)
        elif name not in results or results[name]["status"] != PASS:
            blockers.append(name)
    return blockers


def release_blocking_items(
    profile: str,
    contract: dict[str, Any] | None = None,
    evidence: list[dict[str, Any]] | None = None,
    current_revision: str | None = None,
    expected_ci_provider: str = "test",
    expected_ci_run_id: str = "1",
) -> list[str]:
    """Evaluate exactly the platforms claimed by a closed release profile."""
    try:
        data = load_contract() if contract is None else contract
        validate_contract(data)
    except (PolicyValidationError, OSError, UnicodeError, json.JSONDecodeError):
        return ["INVALID_POLICY"]
    if type(profile) is not str or profile not in RELEASE_PROFILES:
        return ["UNKNOWN_RELEASE_PROFILE"]
    return [
        f"{platform}:{item}"
        for platform in data["release_profiles"][profile]
        for item in blocking_items(
            platform, data, evidence, current_revision, expected_ci_provider, expected_ci_run_id
        )
    ]


def release_ready(
    profile: str,
    contract: dict[str, Any] | None = None,
    evidence: list[dict[str, Any]] | None = None,
    current_revision: str | None = None,
    expected_ci_provider: str = "test",
    expected_ci_run_id: str = "1",
) -> bool:
    """Platform qualification only; never grants enrollment, Stage 10 or LIVE."""
    return not release_blocking_items(
        profile, contract, evidence, current_revision, expected_ci_provider, expected_ci_run_id
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-closed platform release gate")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--platform", choices=PLATFORMS)
    selector.add_argument("--release-profile", choices=tuple(RELEASE_PROFILES))
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--evidence", action="append", type=Path, default=[])
    parser.add_argument("--current-revision", default=os.environ.get("GITHUB_SHA"))
    parser.add_argument("--expected-ci-provider", default=os.environ.get("GITHUB_SERVER_URL"))
    parser.add_argument("--expected-ci-run-id", default=os.environ.get("GITHUB_RUN_ID"))
    args = parser.parse_args(argv)
    try:
        data = load_contract(args.contract)
    except (PolicyValidationError, OSError, UnicodeError, json.JSONDecodeError):
        print("INVALID_POLICY")
        return 1
    if not args.current_revision or not args.expected_ci_provider or not args.expected_ci_run_id:
        parser.error("current revision, CI provider and CI run id are required")
    try:
        evidence = [json.loads(path.read_text(encoding="utf-8")) for path in args.evidence]
    except (OSError, json.JSONDecodeError) as exc:
        print(f"invalid evidence file: {exc}")
        return 1
    evaluate = release_blocking_items if args.release_profile else blocking_items
    blockers = evaluate(
        args.release_profile or args.platform,
        data,
        evidence,
        args.current_revision,
        args.expected_ci_provider,
        args.expected_ci_run_id,
    )
    label = (
        f"{args.release_profile}_RELEASE_READY"
        if args.release_profile
        else f"{args.platform}_PRODUCTION_READY"
    )
    if blockers:
        for name in blockers:
            if name not in data["acceptance"]:
                print(name)
                continue
            item = data["acceptance"][name]
            status = (
                item["status"]
                if item["evidence_class"] == "STATIC_EXECUTABLE_CONTRACT"
                else "MISSING_CURRENT_EVIDENCE"
            )
            print(f"{name}={status} [{item['evidence_class']}]")
        print(f"{label}=NOT_READY")
        return 1
    print(f"{label}=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
