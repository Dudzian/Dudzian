"""Read-only Stage-9 public-authority preflight and ceremony entry gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from deployment.windows_stage9_policy_material import PolicyVectorError, canonical_json_bytes
from deployment.windows_stage9_production_ceremony import (
    verify_pdsa_public_bundle,
    verify_recovery_public_bundle,
    verify_root_anchor_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MANIFEST = Path(__file__).with_name("stage9_expected_public_authorities.json")
CURRENT_STATUS = Path(__file__).with_name("stage9_current_status.json")
DEFAULT_AUTHORITY_DIRECTORY = Path(r"C:\CryptoHunter-Production-Authority\public")
TOOL_VERSION = 1
EVIDENCE_SCHEMA = "CryptoHunter.Stage9AuthorityPreflightEvidenceV1"


class ReadinessError(ValueError):
    """A fail-closed, public-only readiness failure."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReadinessError(f"MISSING_OR_MALFORMED_PUBLIC_ARTIFACT:{path}") from exc
    if not isinstance(value, dict):
        raise ReadinessError(f"MALFORMED_PUBLIC_ARTIFACT:{path}")
    return value


def validate_expected_manifest(value: Mapping[str, Any]) -> None:
    if set(value) != {
        "schema",
        "version",
        "generation",
        "source_freeze_artifact",
        "serialization_profile",
        "product_release_root",
        "pdsa",
        "k_recovery",
    }:
        raise ReadinessError("EXPECTED_MANIFEST_FIELDS")
    if (
        value["schema"] != "CryptoHunter.Stage9ExpectedPublicAuthoritiesV1"
        or value["version"] != 1
        or value["generation"] != "PRODUCTION_AUTHORITY_GENERATION_1"
        or value["source_freeze_artifact"]
        != "docs/windows_stage9_production_root_of_trust_freeze.md"
        or value["serialization_profile"] != "CRYPTOHUNTER_CANONICAL_JSON_V1"
    ):
        raise ReadinessError("EXPECTED_MANIFEST_PROFILE")
    root, pdsa, recovery = value["product_release_root"], value["pdsa"], value["k_recovery"]
    common = {"algorithm", "encoding", "threshold", "key_count", "canonical_key_set_sha256"}
    if set(root) != common | {"purpose", "environment"} or set(pdsa) != common | {"purpose"}:
        raise ReadinessError("EXPECTED_ED25519_FIELDS")
    if root["purpose"] != "PRODUCTION" or root["environment"] != "PRODUCTION":
        raise ReadinessError("EXPECTED_ROOT_PROFILE")
    if pdsa["purpose"] != "PRODUCTION":
        raise ReadinessError("EXPECTED_PDSA_PROFILE")
    for label, item in (("ROOT", root), ("PDSA", pdsa)):
        if (item["algorithm"], item["encoding"], item["threshold"], item["key_count"]) != (
            "Ed25519",
            "RFC8032_RAW_32_BYTES_LOWER_HEX",
            2,
            3,
        ):
            raise ReadinessError(f"EXPECTED_{label}_PROFILE")
        _digest(item["canonical_key_set_sha256"], f"EXPECTED_{label}_DIGEST")
    required_recovery = {
        "type",
        "name_algorithm",
        "curve",
        "scheme",
        "scheme_hash",
        "object_attributes",
        "tpmt_public_sha256",
        "name",
    }
    if set(recovery) != required_recovery or tuple(
        recovery[k]
        for k in ("type", "name_algorithm", "curve", "scheme", "scheme_hash", "object_attributes")
    ) != (
        "TPM_ALG_ECC",
        "TPM_ALG_SHA256",
        "TPM_ECC_NIST_P256",
        "TPM_ALG_ECDSA",
        "TPM_ALG_SHA256",
        "00040040",
    ):
        raise ReadinessError("EXPECTED_RECOVERY_PROFILE")
    digest = _digest(recovery["tpmt_public_sha256"], "EXPECTED_RECOVERY_DIGEST")
    if recovery["name"] != "000b" + digest:
        raise ReadinessError("EXPECTED_RECOVERY_NAME")


def _digest(value: Any, code: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ReadinessError(code)
    return value


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(manifest))).hexdigest()


def public_authority_preflight(
    authority_directory: Path,
    *,
    manifest_path: Path = EXPECTED_MANIFEST,
    revision: str,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Recompute public identities; this function never reads or accepts private material."""
    expected = _load(manifest_path)
    validate_expected_manifest(expected)
    return _verify_public_authorities(
        root_bundle=_load(authority_directory / "product_root_anchor_bundle.json"),
        pdsa_bundle=_load(authority_directory / "pdsa_public_bundle.json"),
        recovery_bundle=_load(authority_directory / "recovery_public_bundle.json"),
        expected=expected,
        revision=revision,
        timestamp=timestamp,
    )


def final_package_public_preflight(
    final_package_directory: Path,
    *,
    manifest_path: Path = EXPECTED_MANIFEST,
    revision: str,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Verify the historical public-authority filenames in a final ceremony package."""
    expected = _load(manifest_path)
    validate_expected_manifest(expected)
    return _verify_public_authorities(
        root_bundle=_load(final_package_directory / "root_anchor_bundle.json"),
        pdsa_bundle=_load(final_package_directory / "pdsa_public_bundle.json"),
        recovery_bundle=_load(final_package_directory / "recovery_public_bundle.json"),
        expected=expected,
        revision=revision,
        timestamp=timestamp,
    )


def _verify_public_authorities(
    *,
    root_bundle: Mapping[str, Any],
    pdsa_bundle: Mapping[str, Any],
    recovery_bundle: Mapping[str, Any],
    expected: Mapping[str, Any],
    revision: str,
    timestamp: str | None,
) -> dict[str, Any]:
    """Apply the shared production profiles and expected authority identities."""
    try:
        root = verify_root_anchor_bundle(root_bundle)
        pdsa = verify_pdsa_public_bundle(pdsa_bundle, expected_purpose="PRODUCTION")
        recovery = verify_recovery_public_bundle(
            recovery_bundle, expected_purpose="PRODUCTION"
        )
    except (PolicyVectorError, KeyError, TypeError, ValueError) as exc:
        raise ReadinessError(f"INVALID_PUBLIC_AUTHORITY:{exc}") from exc
    root_profile = {
        "purpose": root.purpose,
        "environment": root.environment,
        "algorithm": root.algorithm,
        "encoding": root.encoding,
        "threshold": root.threshold,
        "key_count": len(root.keys),
    }
    expected_root_profile = {
        key: expected["product_release_root"][key]
        for key in ("purpose", "environment", "algorithm", "encoding", "threshold", "key_count")
    }
    if root_profile != expected_root_profile:
        raise ReadinessError("ROOT_PROFILE_MISMATCH")
    pdsa_profile = {
        "purpose": pdsa["purpose"],
        "algorithm": pdsa["keys"][0]["algorithm"],
        "encoding": pdsa["keys"][0]["encoding"],
        "threshold": pdsa["threshold"],
        "key_count": len(pdsa["keys"]),
    }
    expected_pdsa_profile = {
        key: expected["pdsa"][key]
        for key in ("purpose", "algorithm", "encoding", "threshold", "key_count")
    }
    if pdsa_profile != expected_pdsa_profile:
        raise ReadinessError("PDSA_PROFILE_MISMATCH")
    actual = {
        "product_release_root_sha256": root.key_set_digest,
        "pdsa_sha256": pdsa["canonical_key_set_digest"],
        "k_recovery_tpmt_public_sha256": recovery["tpmt_public_sha256"],
        "k_recovery_name": recovery["derived_name"],
    }
    wanted = {
        "product_release_root_sha256": expected["product_release_root"]["canonical_key_set_sha256"],
        "pdsa_sha256": expected["pdsa"]["canonical_key_set_sha256"],
        "k_recovery_tpmt_public_sha256": expected["k_recovery"]["tpmt_public_sha256"],
        "k_recovery_name": expected["k_recovery"]["name"],
    }
    mismatches = [name for name in wanted if actual[name] != wanted[name]]
    if mismatches:
        raise ReadinessError("AUTHORITY_MISMATCH:" + ",".join(mismatches))
    return {
        "schema": EVIDENCE_SCHEMA,
        "version": 1,
        "tool_version": TOOL_VERSION,
        "git_revision": revision,
        "expected_manifest_sha256": manifest_digest(expected),
        "actual_public_projection_digests": actual,
        "actual_public_authority_profiles": {
            "product_release_root": root_profile,
            "pdsa": pdsa_profile,
        },
        "result": "PASS",
        "timestamp_utc": timestamp or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def validate_current_status(status: Mapping[str, Any]) -> list[str]:
    reasons = []
    expected = {
        "version": 1,
        "windows_clean_install": "LIVE_PASS",
        "windows_tpm_activation_bridge": "LIVE_PASS",
        "windows_0_14": "10/15 DONE",
        "stage_9": "IN_PROGRESS",
        "a_01_dependency_release_security": "PASS",
        "a_02_external_provisioning": "PASS",
        "production_root_material": "PROVISIONED_LOCALLY",
        "production_ceremony": "NOT_STARTED",
        "production_provisioning_ready": False,
        "windows_production_ready": "NOT_READY",
        "stage_10_implementation": "EXISTS",
        "stage_10_production_lifecycle_live": "NOT_STARTED",
        "stage_10_prerequisite": "BLOCKED_UNTIL_CEREMONY_AND_LEGAL_ENROLLMENT",
    }
    if (
        status.get("schema") != "CryptoHunter.Stage9CurrentStatusV1"
        or status.get("authority") != "CURRENT_STATUS"
    ):
        reasons.append("CURRENT_STATUS_SCHEMA")
    reasons.extend(
        f"CURRENT_STATUS_{key.upper()}"
        for key, value in expected.items()
        if status.get(key) != value
    )
    if (
        status.get("production_ceremony") == "NOT_STARTED"
        and status.get("production_provisioning_ready") is not False
    ):
        reasons.append("PROVISIONING_BEFORE_CEREMONY")
    if (
        status.get("windows_clean_install") == "LIVE_PASS"
        and status.get("windows_0_14") == "9/15 DONE"
    ):
        reasons.append("STALE_WINDOWS_COUNT")
    if (
        status.get("stage_10_production_lifecycle_live") == "LIVE_PASS"
        and status.get("production_ceremony") != "COMPLETE"
    ):
        reasons.append("STAGE10_BEFORE_CEREMONY")
    return sorted(set(reasons))


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if result.returncode:
        raise ReadinessError("GIT_INSPECTION_FAILED")
    return result.stdout.strip()


def _revision_bound_file(repo: Path, revision: str, relative_path: str, failure: str) -> list[str]:
    """Require a working file to be byte-identical to the reviewed Git object."""
    result = subprocess.run(
        ["git", "show", f"{revision}:{relative_path}"], cwd=repo, capture_output=True
    )
    if result.returncode:
        return [failure]
    try:
        working_bytes = (repo / relative_path).read_bytes()
    except OSError:
        return [failure]
    return [] if working_bytes == result.stdout else [failure]


def ceremony_entry_gate(
    *,
    repo: Path,
    reviewed_revision: str,
    authority_directory: Path,
    manifest_path: Path | None = None,
    status_path: Path | None = None,
) -> list[str]:
    reasons = []
    canonical_manifest = repo / "deployment/stage9_expected_public_authorities.json"
    canonical_status = repo / "deployment/stage9_current_status.json"
    if manifest_path is not None and manifest_path.resolve() != canonical_manifest.resolve():
        reasons.append("NONCANONICAL_MANIFEST_PATH")
    if status_path is not None and status_path.resolve() != canonical_status.resolve():
        reasons.append("NONCANONICAL_STATUS_PATH")
    head = _git(repo, "rev-parse", "HEAD")
    if head != reviewed_revision:
        reasons.append("REVISION_MISMATCH")
    if _git(repo, "status", "--porcelain"):
        reasons.append("WORKTREE_NOT_CLEAN")
    manifest_binding = _revision_bound_file(
        repo,
        reviewed_revision,
        "deployment/stage9_expected_public_authorities.json",
        "MANIFEST_REVISION_BINDING_FAILED",
    )
    status_binding = _revision_bound_file(
        repo,
        reviewed_revision,
        "deployment/stage9_current_status.json",
        "STATUS_REVISION_BINDING_FAILED",
    )
    reasons.extend(manifest_binding)
    reasons.extend(status_binding)
    if not status_binding:
        try:
            reasons.extend(validate_current_status(_load(canonical_status)))
        except ReadinessError as exc:
            reasons.append(str(exc))
    if not manifest_binding:
        try:
            public_authority_preflight(
                authority_directory, manifest_path=canonical_manifest, revision=head
            )
        except ReadinessError as exc:
            reasons.append(str(exc))
    return sorted(set(reasons))


def final_package_public_preflight_gate(
    *,
    repo: Path,
    reviewed_revision: str,
    final_package_directory: Path,
    manifest_path: Path | None = None,
) -> list[str]:
    """Bind final-package authority validation to the canonical reviewed manifest."""
    reasons = []
    canonical_manifest = repo / "deployment/stage9_expected_public_authorities.json"
    if manifest_path is not None and manifest_path.resolve() != canonical_manifest.resolve():
        reasons.append("NONCANONICAL_MANIFEST_PATH")
    head = _git(repo, "rev-parse", "HEAD")
    if head != reviewed_revision:
        reasons.append("REVISION_MISMATCH")
    if _git(repo, "status", "--porcelain"):
        reasons.append("WORKTREE_NOT_CLEAN")
    manifest_binding = _revision_bound_file(
        repo,
        reviewed_revision,
        "deployment/stage9_expected_public_authorities.json",
        "MANIFEST_REVISION_BINDING_FAILED",
    )
    reasons.extend(manifest_binding)
    if not manifest_binding and "NONCANONICAL_MANIFEST_PATH" not in reasons:
        try:
            final_package_public_preflight(
                final_package_directory,
                manifest_path=canonical_manifest,
                revision=head,
            )
        except ReadinessError as exc:
            reasons.append(str(exc))
    return sorted(set(reasons))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("public-preflight")
    final_preflight = sub.add_parser("final-package-public-preflight")
    gate = sub.add_parser("entry-gate")
    for command in (preflight, gate):
        command.add_argument("--authority-dir", type=Path, default=DEFAULT_AUTHORITY_DIRECTORY)
    for command in (preflight, final_preflight, gate):
        command.add_argument("--reviewed-revision", required=True)
    preflight.add_argument("--expected-manifest", type=Path, default=EXPECTED_MANIFEST)
    preflight.add_argument("--evidence-output", type=Path)
    final_preflight.add_argument("--final-package-dir", type=Path, required=True)
    final_preflight.add_argument("--repo", type=Path, default=ROOT)
    gate.add_argument("--repo", type=Path, default=ROOT)
    gate.add_argument("--expected-manifest", type=Path)
    gate.add_argument("--status", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "public-preflight":
            evidence = public_authority_preflight(
                args.authority_dir,
                manifest_path=args.expected_manifest,
                revision=args.reviewed_revision,
            )
            if args.evidence_output:
                if args.evidence_output.exists():
                    raise ReadinessError("EVIDENCE_REFUSES_OVERWRITE")
                args.evidence_output.write_bytes(canonical_json_bytes(evidence) + b"\n")
            print(json.dumps(evidence, sort_keys=True))
            return 0
        if args.command == "final-package-public-preflight":
            reasons = final_package_public_preflight_gate(
                repo=args.repo,
                reviewed_revision=args.reviewed_revision,
                final_package_directory=args.final_package_dir,
            )
            if reasons:
                print("BLOCKED " + " ".join(reasons))
                return 1
            evidence = final_package_public_preflight(
                args.final_package_dir,
                manifest_path=args.repo / "deployment/stage9_expected_public_authorities.json",
                revision=args.reviewed_revision,
            )
            print(json.dumps(evidence, sort_keys=True))
            return 0
        reasons = ceremony_entry_gate(
            repo=args.repo,
            reviewed_revision=args.reviewed_revision,
            authority_directory=args.authority_dir,
            manifest_path=args.expected_manifest,
            status_path=args.status,
        )
    except ReadinessError as exc:
        reasons = [str(exc)]
    if reasons:
        print("BLOCKED " + " ".join(reasons))
        return 1
    print("READY_FOR_PRODUCTION_CEREMONY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
