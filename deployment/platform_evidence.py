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
) -> None:
    """Publish Stage-9 PASS only from the exact live-MSI proof receipt."""
    source_revision, provider, run_id = _identity(revision)
    if provider != "https://github.com" or os.environ.get("RUNNER_OS") != "Windows":
        raise EvidenceProductionError("clean-install PASS requires a GitHub Windows runner")
    value = json.loads(receipt.read_text(encoding="utf-8"))
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "source_revision",
        "ci_provider",
        "ci_run_id",
        "runner_os",
        "runner_arch",
        "probe_id",
        "msi_sha256",
        "manifest_sha256",
        "product_version",
        "install_exit_code",
        "proofs",
        "uninstall_exit_code",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise EvidenceProductionError("malformed clean-install receipt")
    proofs = {
        "files",
        "services",
        "dacl",
        "postgresql",
        "mtls_matrix",
        "backend",
        "logging",
        "uninstall",
        "acceptance_cleanup",
    }
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
        or not re.fullmatch(r"[0-9a-f]{64}", value["msi_sha256"])
        or not re.fullmatch(r"[0-9a-f]{64}", value["manifest_sha256"])
        or value["msi_sha256"] != hashlib.sha256(msi.read_bytes()).hexdigest()
        or value["manifest_sha256"] != hashlib.sha256(manifest.read_bytes()).hexdigest()
        or value["product_version"] != manifest_value.get("product_version")
        or value["msi_sha256"] != manifest_value.get("msi", {}).get("sha256")
        or manifest_value.get("architecture") != "x64"
        or manifest_value.get("wix_version") != "7.0.0"
        or manifest_value.get("postgresql_version") != "17.11"
        or manifest_value.get("postgresql_packaging_revision") != "4"
        or not isinstance(value["proofs"], dict)
        or set(value["proofs"]) != proofs
        or any(result != "PASS" for result in value["proofs"].values())
    ):
        raise EvidenceProductionError("clean-install receipt is incomplete or failed")
    details = (
        f"MSI sha256={value['msi_sha256']}; version={value['product_version']}; "
        f"architecture={value['runner_arch']}"
    )
    output.write_text(
        json.dumps(
            evidence_document(
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
    args = parser.parse_args(argv)
    try:
        if args.command == "aggregate-core":
            aggregate_core_markers(
                args.source_revision,
                args.marker,
                args.output,
                manifest_path=args.manifest,
            )
        else:
            produce_windows_clean_install_evidence(
                args.source_revision,
                args.receipt,
                args.output,
                msi=args.msi,
                manifest=args.manifest,
            )
    except (EvidenceProductionError, OSError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
