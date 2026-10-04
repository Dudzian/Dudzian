"""Revision-bound execution evidence producers for deployment release gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any

from deployment.core_test_plan import MANIFEST, canonical_plan_digest, load_manifest
from deployment.windows_installer.contract import is_reviewed_wix_version
from deployment.windows_stage9_evidence_contract import (
    CLEAN_INSTALL_PRECEREMONY_PROOFS,
    CLEAN_INSTALL_RECEIPT_KEYS,
    POST_ENROLLMENT_QUALIFICATION_STATE,
    PRODUCTION_TRUST_CEREMONY_ID,
)

SCHEMA_VERSION = 1
SCM_ITEMS = (
    "WINDOWS_NATIVE_PATH_INTEGRATION",
    "WINDOWS_PROTECTED_CONFIGURATION",
    "WINDOWS_PROTECTED_STATE_PATHS",
    "WINDOWS_ACL_QUALIFICATION",
    "WINDOWS_SERVICE_INSTALLATION",
    "WINDOWS_AUTOSTART",
    "WINDOWS_SERVICE_START",
    "WINDOWS_GRACEFUL_STOP",
    "WINDOWS_MANUAL_RESTART",
    "WINDOWS_AUTOMATIC_CRASH_RESTART",
    "WINDOWS_PROCESS_TREE",
    "WINDOWS_NO_ORPHAN_CHILDREN",
    "WINDOWS_PERSISTENT_LOGGING",
)
WINDOWS_STAGE6_ITEMS = (
    "WINDOWS_FILE_LOCKING",
    "WINDOWS_SQLITE_CRASH_INTEGRITY",
)
WINDOWS_STAGE7_ITEMS = (
    "WINDOWS_NETWORK_RECOVERY",
    "WINDOWS_SAFE_OS_SHUTDOWN",
)
WINDOWS_STAGE8_ITEMS = (
    "WINDOWS_POSTGRESQL_SUBSTRATE",
    "WINDOWS_LOCAL_PRINCIPAL_AUTHENTICATION",
)
WINDOWS_STAGE9_ITEMS = ("WINDOWS_CLEAN_INSTALL",)
WINDOWS_STAGE10_ITEMS = (
    "WINDOWS_REBOOT_RECOVERY",
    "WINDOWS_LONG_RUNNING_LIFECYCLE",
)
WINDOWS_STAGE10_CONTRACT = Path(__file__).with_name("windows_stage10_lifecycle_contract.json")
WINDOWS_STAGE10_PROBE_ID = "cryptohunter.windows.stage10.lifecycle.v1"
WINDOWS_LIVE_ITEMS = (
    *SCM_ITEMS,
    *WINDOWS_STAGE6_ITEMS,
    *WINDOWS_STAGE7_ITEMS,
    *WINDOWS_STAGE8_ITEMS,
)


class EvidenceProductionError(RuntimeError):
    pass


def _identity(revision: str | None) -> tuple[str, str, str]:
    source_revision = revision or os.environ.get("GITHUB_SHA", "")
    if not source_revision:
        raise EvidenceProductionError("source revision is required")
    return (
        source_revision,
        os.environ.get("GITHUB_SERVER_URL", "LOCAL_REVIEWED_EXECUTION"),
        os.environ.get("GITHUB_RUN_ID", "LOCAL"),
    )


def evidence_document(
    *,
    platform_name: str,
    source_revision: str,
    ci_provider: str,
    ci_run_id: str,
    runner_os: str,
    runner_arch: str,
    results: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "platform": platform_name,
        "source_revision": source_revision,
        "ci_provider": ci_provider,
        "ci_run_id": ci_run_id,
        "runner_os": runner_os,
        "runner_arch": runner_arch,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "results": results,
    }


def aggregate_core_markers(
    revision: str | None,
    markers: list[Path],
    output: Path,
    *,
    manifest_path: Path = MANIFEST,
    ci_run_id: str | None = None,
    ci_provider: str | None = None,
) -> None:
    """Emit one core result only after all three revision-bound runner markers exist."""
    source_revision, default_provider, default_run_id = _identity(revision)
    provider = ci_provider or default_provider
    run_id = ci_run_id or default_run_id
    manifest = load_manifest(manifest_path)
    expected_plan_id = manifest["test_plan_id"]
    expected_digest = canonical_plan_digest(manifest)
    if len(markers) != 3:
        raise EvidenceProductionError("exactly three core markers are required")
    observed: dict[str, Path] = {}
    for marker in markers:
        value = json.loads(marker.read_text(encoding="utf-8"))
        required = {
            "schema_version",
            "source_revision",
            "runner_os",
            "test_plan_id",
            "test_plan_digest",
            "ci_run_id",
            "ci_provider",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise EvidenceProductionError(f"malformed core marker: {marker}")
        if (
            value["schema_version"] != 1
            or value["source_revision"] != source_revision
            or value["test_plan_id"] != expected_plan_id
            or value["test_plan_digest"] != expected_digest
            or value["ci_run_id"] != run_id
            or value["ci_provider"] != provider
        ):
            raise EvidenceProductionError(f"invalid or stale core marker: {marker}")
        runner_os = value["runner_os"]
        if runner_os in observed:
            raise EvidenceProductionError(f"duplicate core runner marker: {runner_os}")
        observed[runner_os] = marker
    if set(observed) != {"Linux", "Windows", "macOS"}:
        raise EvidenceProductionError(f"incomplete core runner set: {sorted(observed)}")
    result = {
        "item": "CORE_REQUIRED_SUITES",
        "status": "PASS",
        "evidence_class": "CROSS_OS_CI_MATRIX",
        "test_or_probe": "core CI matrix",
        "details": f"{expected_plan_id}:{expected_digest}; Linux, Windows and macOS passed",
    }
    output.write_text(
        json.dumps(
            evidence_document(
                platform_name="CROSS_PLATFORM_CORE",
                source_revision=source_revision,
                ci_provider=provider,
                ci_run_id=run_id,
                runner_os="Windows",
                runner_arch=platform.machine(),
                results=[result],
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def produce_windows_clean_install_evidence(
    revision: str | None,
    receipt: Path,
    output: Path,
    *,
    msi: Path,
    manifest: Path,
    production_trust_package: Path,
) -> None:
    """Publish Stage-9 PASS only from the exact live-MSI proof receipt."""
    source_revision, provider, run_id = _identity(revision)
    if provider != "https://github.com" or os.environ.get("RUNNER_OS") != "Windows":
        raise EvidenceProductionError("clean-install PASS requires a GitHub Windows runner")
    value = json.loads(receipt.read_text(encoding="utf-8"))
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    if production_trust_package.name != PRODUCTION_TRUST_CEREMONY_ID:
        raise EvidenceProductionError("Production Trust ceremony directory differs")
    package_manifest_sha256 = hashlib.sha256(
        (production_trust_package / "package_manifest.json").read_bytes()
    ).hexdigest()
    if not isinstance(value, dict) or set(value) != CLEAN_INSTALL_RECEIPT_KEYS:
        raise EvidenceProductionError("malformed clean-install receipt")
    if (
        value["schema_version"] != 1
        or value["source_revision"] != source_revision
        or value["ci_provider"] != provider
        or value["ci_run_id"] != run_id
        or value["runner_os"] != "Windows"
        or value["runner_arch"] != os.environ.get("RUNNER_ARCH")
        or value["probe_id"] != "cryptohunter.windows.clean-install.v1"
        or value["install_exit_code"] != 0
        or value["uninstall_exit_code"] != 0
        or value["post_enrollment_live_qualification"] != POST_ENROLLMENT_QUALIFICATION_STATE
        or value["production_trust_ceremony_id"] != PRODUCTION_TRUST_CEREMONY_ID
        or not re.fullmatch(r"[0-9a-f]{64}", value["production_trust_package_manifest_sha256"])
        or value["production_trust_package_manifest_sha256"] != package_manifest_sha256
        or not re.fullmatch(r"[1-9][0-9]*", value["production_trust_artifact_run_id"])
        or not re.fullmatch(r"[0-9a-f]{64}", value["msi_sha256"])
        or not re.fullmatch(r"[0-9a-f]{64}", value["manifest_sha256"])
        or value["msi_sha256"] != hashlib.sha256(msi.read_bytes()).hexdigest()
        or value["manifest_sha256"] != hashlib.sha256(manifest.read_bytes()).hexdigest()
        or value["product_version"] != manifest_value.get("product_version")
        or value["msi_sha256"] != manifest_value.get("msi", {}).get("sha256")
        or manifest_value.get("architecture") != "x64"
        or not is_reviewed_wix_version(manifest_value.get("wix_version"))
        or manifest_value.get("postgresql_version") != "17.11"
        or manifest_value.get("postgresql_packaging_revision") != "4"
        or not isinstance(value["proofs"], dict)
        or set(value["proofs"]) != set(CLEAN_INSTALL_PRECEREMONY_PROOFS)
        or any(result != "PASS" for result in value["proofs"].values())
    ):
        raise EvidenceProductionError("clean-install receipt is incomplete or failed")
    details = (
        f"MSI sha256={value['msi_sha256']}; version={value['product_version']}; "
        f"architecture={value['runner_arch']}"
    )
    document = evidence_document(
        platform_name="WINDOWS",
        source_revision=source_revision,
        ci_provider=provider,
        ci_run_id=run_id,
        runner_os="Windows",
        runner_arch=value["runner_arch"],
        results=[
            {
                "item": WINDOWS_STAGE9_ITEMS[0],
                "status": "PASS",
                "evidence_class": "LIVE_WINDOWS_INTEGRATION",
                "test_or_probe": "canonical Stage-9 MSI clean-install proof",
                "details": details,
            }
        ],
    )
    document["production_trust"] = {
        "ceremony_id": value["production_trust_ceremony_id"],
        "package_manifest_sha256": value["production_trust_package_manifest_sha256"],
        "artifact_run_id": value["production_trust_artifact_run_id"],
        "frozen_production_trust": value["proofs"]["frozen_production_trust"],
    }
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _current_windows_machine_guid() -> str:
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as key:
            value, _ = winreg.QueryValueEx(key, "MachineGuid")
    except OSError as exc:
        raise EvidenceProductionError("current Windows MachineGuid is unavailable") from exc
    if not isinstance(value, str) or not value.strip():
        raise EvidenceProductionError("current Windows MachineGuid is missing or invalid")
    return value


def produce_windows_stage10_evidence(
    revision: str | None,
    receipt: Path,
    output: Path,
    *,
    contract_path: Path = WINDOWS_STAGE10_CONTRACT,
) -> None:
    """Publish lifecycle evidence from the native, reboot-capable Stage-10 host."""
    source_revision, provider, run_id = _identity(revision)
    if provider != "https://github.com" or os.environ.get("RUNNER_OS") != "Windows":
        raise EvidenceProductionError("Stage-10 PASS requires a GitHub Windows runner")
    value = json.loads(receipt.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "source_revision",
        "ci_provider",
        "ci_run_id",
        "runner_os",
        "runner_arch",
        "runner_name",
        "machine_guid",
        "probe_id",
        "runtime_provenance",
        "results",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise EvidenceProductionError("malformed Stage-10 receipt")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        not isinstance(contract, dict)
        or set(contract) != {"schema_version", "probe_id", "results"}
        or contract["schema_version"] != 1
        or contract["probe_id"] != WINDOWS_STAGE10_PROBE_ID
        or not isinstance(contract["results"], dict)
        or set(contract["results"]) != set(WINDOWS_STAGE10_ITEMS)
        or any(
            not isinstance(proofs, list)
            or not proofs
            or any(not isinstance(proof, str) or not proof for proof in proofs)
            or len(proofs) != len(set(proofs))
            for proofs in contract["results"].values()
        )
    ):
        raise EvidenceProductionError("malformed Stage-10 lifecycle contract")
    expected_results = {item: set(proofs) for item, proofs in contract["results"].items()}
    current_machine_guid = _current_windows_machine_guid()
    runtime_provenance = value.get("runtime_provenance")
    provenance_keys = {
        "source_revision",
        "ci_run_id",
        "product_version",
        "qualified_manifest_sha256",
        "qualified_msi_sha256",
        "installed_backend_sha256",
    }
    provenance_ok = (
        isinstance(runtime_provenance, dict)
        and set(runtime_provenance) == provenance_keys
        and runtime_provenance.get("source_revision") == source_revision
        and runtime_provenance.get("ci_run_id") == run_id
        and isinstance(runtime_provenance.get("product_version"), str)
        and bool(runtime_provenance.get("product_version"))
        and all(
            isinstance(runtime_provenance.get(key), str)
            and re.fullmatch(r"[0-9a-f]{64}", runtime_provenance[key]) is not None
            for key in (
                "qualified_manifest_sha256",
                "qualified_msi_sha256",
                "installed_backend_sha256",
            )
        )
    )
    identity_ok = (
        value.get("schema_version") == 1
        and value.get("source_revision") == source_revision
        and value.get("ci_provider") == provider
        and value.get("ci_run_id") == run_id
        and value.get("runner_os") == "Windows"
        and value.get("runner_arch") == os.environ.get("RUNNER_ARCH")
        and value.get("runner_name") == os.environ.get("RUNNER_NAME")
        and value.get("probe_id") == contract["probe_id"]
        and value.get("machine_guid") == current_machine_guid
        and provenance_ok
        and isinstance(value.get("results"), dict)
        and set(value["results"]) == set(expected_results)
    )
    if not identity_ok:
        raise EvidenceProductionError("Stage-10 receipt identity is stale or incomplete")
    for item, proofs in expected_results.items():
        result = value["results"][item]
        if (
            not isinstance(result, dict)
            or set(result) != {"status", "proofs", "details"}
            or result["status"] != "PASS"
            or not isinstance(result["details"], str)
            or not result["details"]
            or not isinstance(result["proofs"], dict)
            or set(result["proofs"]) != proofs
            or any(proof != "PASS" for proof in result["proofs"].values())
        ):
            raise EvidenceProductionError(f"Stage-10 receipt failed: {item}")
    results = [
        {
            "item": item,
            "status": "PASS",
            "evidence_class": "LIVE_WINDOWS_INTEGRATION",
            "test_or_probe": "native Stage-10 reboot and lifecycle qualification",
            "details": value["results"][item]["details"],
        }
        for item in WINDOWS_STAGE10_ITEMS
    ]
    output.write_text(
        json.dumps(
            evidence_document(
                platform_name="WINDOWS",
                source_revision=source_revision,
                ci_provider=provider,
                ci_run_id=run_id,
                runner_os="Windows",
                runner_arch=value["runner_arch"],
                results=results,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Executable deployment evidence producer")
    sub = parser.add_subparsers(dest="command", required=True)
    aggregate = sub.add_parser("aggregate-core")
    aggregate.add_argument("--source-revision")
    aggregate.add_argument("--marker", action="append", type=Path, required=True)
    aggregate.add_argument("--manifest", type=Path, default=MANIFEST)
    aggregate.add_argument("--output", type=Path, required=True)
    clean = sub.add_parser("windows-clean-install")
    clean.add_argument("--source-revision")
    clean.add_argument("--receipt", type=Path, required=True)
    clean.add_argument("--output", type=Path, required=True)
    clean.add_argument("--msi", type=Path, required=True)
    clean.add_argument("--manifest", type=Path, required=True)
    clean.add_argument("--production-trust-package", type=Path, required=True)
    stage10 = sub.add_parser("windows-stage10")
    stage10.add_argument("--source-revision")
    stage10.add_argument("--receipt", type=Path, required=True)
    stage10.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "aggregate-core":
            aggregate_core_markers(
                args.source_revision,
                args.marker,
                args.output,
                manifest_path=args.manifest,
            )
        elif args.command == "windows-clean-install":
            produce_windows_clean_install_evidence(
                args.source_revision,
                args.receipt,
                args.output,
                msi=args.msi,
                manifest=args.manifest,
                production_trust_package=args.production_trust_package,
            )
        else:
            produce_windows_stage10_evidence(args.source_revision, args.receipt, args.output)
    except (EvidenceProductionError, OSError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
