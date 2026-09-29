"""Public-only, fail-closed Stage-9 production ceremony tooling.

This module deliberately has no operation which accepts secret key material.  It
turns externally supplied public authorities and detached signatures into the
objects consumed by the canonical Stage-9 verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from deployment.windows_stage9_policy_material import (
    RELEASE_SCHEMA,
    PolicyVectorError,
    canonical_digest,
    canonical_json_bytes,
    validate_schema,
)
from deployment.windows_stage9_root_of_trust_freeze import (
    RELEASE_DOMAIN,
    REVOCATION_DOMAIN,
    PinnedProductReleaseRootV1,
    VerifiedReleasePolicyV1,
    build_frozen_manifest,
    load_pinned_product_release_root_v1,
    source_revision,
    stage9_revocation_genesis_v1,
    validate_tpmt_public,
    verify_freeze_manifest,
    verify_revocation_state,
    verify_signed_release_policy,
    verify_test_only_signed_release_policy,
)

TOOL_VERSION = 1
SERIALIZATION_PROFILE = "CRYPTOHUNTER_CANONICAL_JSON_V1"
ED25519_ENCODING = "RFC8032_RAW_32_BYTES_LOWER_HEX"
RELEASE_SIGNATURE_PROFILE = "Ed25519-SHA256-DIGEST-CH-STAGE9-RELEASE-V1"
REVOCATION_SIGNATURE_PROFILE = "Ed25519-SHA256-DIGEST-CH-STAGE9-REVOCATION-V1"
CEREMONY_ID_DOMAIN = b"CryptoHunter.Stage9.CeremonyIdV1\x00"
RECOVERY_PROFILE = {
    "role": "OFFLINE_BOOTSTRAP_RECOVERY_POLICY_AUTHORITY",
    "type": "TPM_ALG_ECC",
    "name_algorithm": "TPM_ALG_SHA256",
    "object_attributes": "00040040",
    "auth_policy_size": 0,
    "scheme": "TPM_ALG_ECDSA",
    "scheme_hash": "TPM_ALG_SHA256",
    "curve": "TPM_ECC_NIST_P256",
    "kdf": "TPM_ALG_NULL",
}


def _require_exact_keys(value: Mapping[str, Any], required: set[str]) -> None:
    if set(value) != required:
        raise PolicyVectorError("artifact has missing or unexpected fields")


def _hex(value: Any, size: int, label: str) -> bytes:
    try:
        raw = bytes.fromhex(value)
    except (TypeError, ValueError) as exc:
        raise PolicyVectorError(f"invalid {label}") from exc
    if raw.hex() != value or len(raw) != size:
        raise PolicyVectorError(f"noncanonical {label}")
    return raw


def build_root_anchor_bundle(
    key_records: Sequence[Mapping[str, str]], *, purpose: str, environment: str
) -> tuple[dict[str, Any], PinnedProductReleaseRootV1]:
    """Validate and canonically package the externally pinned public root."""
    records = [dict(record) for record in key_records]
    pinned = load_pinned_product_release_root_v1(records, purpose=purpose, environment=environment)
    bundle = {
        "schema": "CryptoHunter.Stage9ProductReleaseRootAnchorBundleV1",
        "version": 1,
        "purpose": purpose,
        "environment": environment,
        "algorithm": "Ed25519",
        "encoding": ED25519_ENCODING,
        "threshold": 2,
        "keys": records,
        "canonical_key_set_digest": pinned.key_set_digest,
        "bootstrap_root_trust": "EXTERNAL_OPERATOR/CUSTODY DECISION",
    }
    return bundle, pinned


def verify_root_anchor_bundle(bundle: Mapping[str, Any]) -> PinnedProductReleaseRootV1:
    _require_exact_keys(
        bundle,
        {
            "schema",
            "version",
            "purpose",
            "environment",
            "algorithm",
            "encoding",
            "threshold",
            "keys",
            "canonical_key_set_digest",
            "bootstrap_root_trust",
        },
    )
    if (
        bundle["schema"],
        bundle["version"],
        bundle["algorithm"],
        bundle["encoding"],
        bundle["threshold"],
        bundle["bootstrap_root_trust"],
    ) != (
        "CryptoHunter.Stage9ProductReleaseRootAnchorBundleV1",
        1,
        "Ed25519",
        ED25519_ENCODING,
        2,
        "EXTERNAL_OPERATOR/CUSTODY DECISION",
    ):
        raise PolicyVectorError("invalid root anchor bundle contract")
    _, pinned = build_root_anchor_bundle(
        bundle["keys"], purpose=bundle["purpose"], environment=bundle["environment"]
    )
    if bundle["canonical_key_set_digest"] != pinned.key_set_digest:
        raise PolicyVectorError("root key-set digest mismatch")
    return pinned


def build_pdsa_public_bundle(
    key_records: Sequence[Mapping[str, str]], *, threshold: int, purpose: str
) -> dict[str, Any]:
    records = [dict(record) for record in key_records]
    ids = [r.get("key_id") for r in records]
    values = [r.get("public_key_hex") for r in records]
    if (
        not records
        or ids != sorted(ids)
        or len(ids) != len(set(ids))
        or len(values) != len(set(values))
    ):
        raise PolicyVectorError("PDSA keys must be unique and lexically ordered")
    if purpose not in ("PRODUCTION", "TEST_ONLY") or not 1 <= threshold <= len(records):
        raise PolicyVectorError("invalid PDSA purpose or threshold")
    for record in records:
        _require_exact_keys(record, {"key_id", "algorithm", "encoding", "public_key_hex"})
        if record["algorithm"] != "Ed25519" or record["encoding"] != ED25519_ENCODING:
            raise PolicyVectorError("invalid PDSA profile")
        Ed25519PublicKey.from_public_bytes(_hex(record["public_key_hex"], 32, "PDSA public key"))
    return {
        "schema": "CryptoHunter.Stage9PdsaPublicAuthorityBundleV1",
        "version": 1,
        "purpose": purpose,
        "threshold": threshold,
        "keys": records,
        "canonical_key_set_digest": canonical_digest(records).hex(),
    }


def verify_pdsa_public_bundle(
    bundle: Mapping[str, Any], *, expected_purpose: str
) -> dict[str, Any]:
    _require_exact_keys(
        bundle, {"schema", "version", "purpose", "threshold", "keys", "canonical_key_set_digest"}
    )
    if (
        bundle["schema"] != "CryptoHunter.Stage9PdsaPublicAuthorityBundleV1"
        or bundle["version"] != 1
        or bundle["purpose"] != expected_purpose
    ):
        raise PolicyVectorError("invalid PDSA bundle identity")
    rebuilt = build_pdsa_public_bundle(
        bundle["keys"], threshold=bundle["threshold"], purpose=bundle["purpose"]
    )
    if dict(bundle) != rebuilt:
        raise PolicyVectorError("PDSA bundle digest or canonical content mismatch")
    return rebuilt


def build_recovery_public_bundle(
    *, key_id: str, tpmt_public_hex: str, purpose: str, provenance: str
) -> dict[str, Any]:
    expected = "PRODUCTION_PROVISIONED" if purpose == "PRODUCTION" else "TEST_FIXTURE"
    if purpose not in ("PRODUCTION", "TEST_ONLY") or provenance != expected or not key_id:
        raise PolicyVectorError("recovery provenance cannot be relabelled")
    parsed = validate_tpmt_public(tpmt_public_hex, RECOVERY_PROFILE)
    return {
        "schema": "CryptoHunter.Stage9RecoveryAuthorityPublicBundleV1",
        "version": 1,
        "purpose": purpose,
        "key_id": key_id,
        "role": RECOVERY_PROFILE["role"],
        "tpmt_public_hex": tpmt_public_hex,
        "tpmt_public_sha256": parsed.digest,
        "derived_name": parsed.name,
        "profile": dict(RECOVERY_PROFILE),
        "provenance": provenance,
    }


def verify_recovery_public_bundle(
    bundle: Mapping[str, Any], *, expected_purpose: str
) -> dict[str, Any]:
    rebuilt = build_recovery_public_bundle(
        key_id=bundle.get("key_id", ""),
        tpmt_public_hex=bundle.get("tpmt_public_hex", ""),
        purpose=expected_purpose,
        provenance=bundle.get("provenance", ""),
    )
    if dict(bundle) != rebuilt:
        raise PolicyVectorError("recovery public digest, Name, profile, or purpose mismatch")
    return rebuilt


def build_unsigned_release_policy(
    *,
    root_bundle: Mapping[str, Any],
    pdsa_bundle: Mapping[str, Any],
    recovery_bundle: Mapping[str, Any],
    release_policy_id: str,
    release_version: int,
    valid_from: str,
    valid_until: str,
    k_psa_profile: Mapping[str, Any],
    policy_refs: Mapping[str, Any],
    nv_template: Mapping[str, Any],
    branch_order: Sequence[str],
) -> dict[str, Any]:
    root = verify_root_anchor_bundle(root_bundle)
    pdsa = verify_pdsa_public_bundle(pdsa_bundle, expected_purpose=root.purpose)
    recovery = verify_recovery_public_bundle(recovery_bundle, expected_purpose=root.purpose)
    keys = pdsa["keys"]
    payload = {
        "schema": "CryptoHunter.ReleasePolicyV1",
        "purpose": root.purpose,
        "release_policy_id": release_policy_id,
        "serialization_profile": SERIALIZATION_PROFILE,
        "product_release_root": {
            "trust_anchor_source": "EXTERNALLY_PINNED",
            "algorithm": "Ed25519",
            "encoding": ED25519_ENCODING,
            "threshold": 2,
            "key_ids": list(root.keys),
            "public_keys_hex": list(root.keys.values()),
        },
        "pdsa_verification_keys": [item["public_key_hex"] for item in keys],
        "pdsa_verification_key_set": keys,
        "production_contract": {
            "release_version": release_version,
            "pdsa_threshold": pdsa["threshold"],
            "pdsa_key_ids": [item["key_id"] for item in keys],
            "valid_from": valid_from,
            "valid_until": valid_until,
            "rotation": "NEWER_ROOT_QUORUM_AND_MONOTONIC_VERSION",
            "revocation": "SIGNED_APPEND_ONLY_DENY_LIST",
            "compromise_recovery": "OFFLINE_BREAK_GLASS_QUORUM_NO_DOWNGRADE",
        },
        "k_recovery": {
            **recovery["profile"],
            "key_id": recovery["key_id"],
            "public_hex": recovery["tpmt_public_hex"],
            "provenance": recovery["provenance"],
        },
        "k_psa_profile": dict(k_psa_profile),
        "policy_refs": dict(policy_refs),
        "nv_template": dict(nv_template),
        "branch_order": list(branch_order),
    }
    validate_schema(payload, RELEASE_SCHEMA)
    canonical_json_bytes(payload)
    return payload


def ceremony_id(
    root_key_set_digest: str, release_payload: Mapping[str, Any], environment: str
) -> str:
    material = {
        "schema": "CryptoHunter.Stage9CeremonyIdMaterialV1",
        "version": 1,
        "root_key_set_digest": root_key_set_digest,
        "release_payload_digest": canonical_digest(dict(release_payload)).hex(),
        "release_version": release_payload["production_contract"]["release_version"],
        "environment": environment,
        "ceremony_profile": "STAGE9_PRODUCTION_CEREMONY_V1",
    }
    return hashlib.sha256(CEREMONY_ID_DOMAIN + canonical_json_bytes(material)).hexdigest()


def build_signing_request(
    *,
    artifact_type: str,
    payload: Mapping[str, Any],
    pinned_root: PinnedProductReleaseRootV1,
    ceremony: str,
) -> dict[str, Any]:
    profiles = {
        "RELEASE_POLICY": (RELEASE_DOMAIN, RELEASE_SIGNATURE_PROFILE),
        "INITIAL_REVOCATION": (REVOCATION_DOMAIN, REVOCATION_SIGNATURE_PROFILE),
    }
    if artifact_type not in profiles:
        raise PolicyVectorError("unsupported signing artifact type")
    domain, _ = profiles[artifact_type]
    digest = canonical_digest(dict(payload))
    return {
        "schema": "CryptoHunter.Stage9SigningRequestV1",
        "version": 1,
        "purpose": pinned_root.purpose,
        "artifact_type": artifact_type,
        "signature_domain": domain.hex(),
        "payload_digest": digest.hex(),
        "message_to_sign_hex": (domain + digest).hex(),
        "allowed_signer_ids": list(pinned_root.keys),
        "required_threshold": pinned_root.threshold,
        "root_key_set_digest": pinned_root.key_set_digest,
        "ceremony_id": ceremony,
    }


def verify_signing_request(
    request: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    pinned_root: PinnedProductReleaseRootV1,
) -> None:
    """Recompute every security-relevant request field from payload and pins."""
    _require_exact_keys(
        request,
        {
            "schema",
            "version",
            "purpose",
            "artifact_type",
            "signature_domain",
            "payload_digest",
            "message_to_sign_hex",
            "allowed_signer_ids",
            "required_threshold",
            "root_key_set_digest",
            "ceremony_id",
        },
    )
    if request["schema"] != "CryptoHunter.Stage9SigningRequestV1" or request["version"] != 1:
        raise PolicyVectorError("invalid signing request identity")
    expected = build_signing_request(
        artifact_type=request["artifact_type"],
        payload=payload,
        pinned_root=pinned_root,
        ceremony=request["ceremony_id"],
    )
    if dict(request) != expected:
        raise PolicyVectorError("signing request invariant failed")


def build_initial_revocation_payload(*, effective_at: str) -> dict[str, Any]:
    # Parse now, rather than silently sourcing a clock value.
    datetime.strptime(effective_at, "%Y-%m-%dT%H:%M:%SZ")
    return {
        "schema": "CryptoHunter.Stage9RevocationPayloadV1",
        "version": 1,
        "sequence": 1,
        "previous_state_digest": "00" * 32,
        "revoked_root_signer_ids": [],
        "revoked_pdsa_signer_ids": [],
        "effective_at": effective_at,
        "authority": "PRODUCT_RELEASE_ROOT_QUORUM",
    }


def import_detached_signatures(
    request: Mapping[str, Any],
    signatures: Sequence[Mapping[str, Any]],
    pinned_root: PinnedProductReleaseRootV1,
    *,
    revoked_signer_ids: Sequence[str] = (),
) -> list[dict[str, str]]:
    domain = bytes.fromhex(request["signature_domain"])
    digest = _hex(request["payload_digest"], 32, "payload digest")
    if (
        request["message_to_sign_hex"] != (domain + digest).hex()
        or request["root_key_set_digest"] != pinned_root.key_set_digest
    ):
        raise PolicyVectorError("signing request invariant failed")
    if (
        request.get("purpose") != pinned_root.purpose
        or request.get("allowed_signer_ids") != list(pinned_root.keys)
        or request.get("required_threshold") != pinned_root.threshold
    ):
        raise PolicyVectorError("signing request authority mismatch")
    accepted: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in signatures:
        _require_exact_keys(
            item,
            {
                "schema",
                "version",
                "ceremony_id",
                "artifact_type",
                "payload_digest",
                "signer_id",
                "signature_profile",
                "signature_hex",
            },
        )
        if (
            item["schema"] != "CryptoHunter.Stage9DetachedSignatureV1"
            or item["version"] != 1
            or item["ceremony_id"] != request["ceremony_id"]
            or item["artifact_type"] != request["artifact_type"]
            or item["payload_digest"] != request["payload_digest"]
        ):
            raise PolicyVectorError("detached signature is from another signing request")
        signer = item["signer_id"]
        if signer in seen:
            raise PolicyVectorError("duplicate signer")
        if signer not in pinned_root.keys or signer in revoked_signer_ids:
            raise PolicyVectorError("unknown or revoked signer")
        expected_profile = (
            RELEASE_SIGNATURE_PROFILE
            if request["artifact_type"] == "RELEASE_POLICY"
            else REVOCATION_SIGNATURE_PROFILE
        )
        if item["signature_profile"] != expected_profile:
            raise PolicyVectorError("wrong signature profile")
        signature = _hex(item["signature_hex"], 64, "signature")
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(pinned_root.keys[signer])).verify(
                signature, domain + digest
            )
        except InvalidSignature as exc:
            raise PolicyVectorError("wrong signature") from exc
        seen.add(signer)
        accepted.append({"signer_id": signer, "signature_hex": item["signature_hex"]})
    if len(accepted) < request["required_threshold"]:
        raise PolicyVectorError("insufficient signature threshold")
    return sorted(accepted, key=lambda entry: entry["signer_id"])


def assemble_signed_artifact(
    *,
    request: Mapping[str, Any],
    payload: Mapping[str, Any],
    signatures: Sequence[Mapping[str, Any]],
    pinned_root: PinnedProductReleaseRootV1,
) -> dict[str, Any]:
    verify_signing_request(request, payload=payload, pinned_root=pinned_root)
    accepted = import_detached_signatures(request, signatures, pinned_root)
    release = request["artifact_type"] == "RELEASE_POLICY"
    return {
        "schema": "CryptoHunter.SignedReleasePolicyV1"
        if release
        else "CryptoHunter.Stage9RevocationStateV1",
        "version": 1,
        "serialization_profile": SERIALIZATION_PROFILE,
        "payload": dict(payload),
        "payload_digest": request["payload_digest"],
        "signature_profile": RELEASE_SIGNATURE_PROFILE if release else REVOCATION_SIGNATURE_PROFILE,
        "threshold": 2,
        "signatures": accepted,
    }


def verify_ceremony(
    *,
    signed_release: Mapping[str, Any],
    signed_revocation: Mapping[str, Any],
    root_bundle: Mapping[str, Any],
    verification_time: datetime,
) -> VerifiedReleasePolicyV1:
    pinned = verify_root_anchor_bundle(root_bundle)
    revocation = verify_revocation_state(
        canonical_json_bytes(dict(signed_revocation)),
        pinned,
        retained_head=stage9_revocation_genesis_v1(),
        verification_time=verification_time,
    )
    verifier = (
        verify_signed_release_policy
        if pinned.purpose == "PRODUCTION"
        else verify_test_only_signed_release_policy
    )
    return verifier(
        canonical_json_bytes(dict(signed_release)),
        pinned,
        verification_time=verification_time,
        revocations=revocation,
    )


def build_audit_transcript(
    *,
    ceremony: str,
    release: VerifiedReleasePolicyV1,
    manifest: Mapping[str, Any],
    source_revision_value: str,
    environment: str,
) -> dict[str, Any]:
    release_envelope = json.loads(release.serialized_envelope)
    revocation_envelope = json.loads(release.revocations.serialized_envelope)
    return {
        "schema": "CryptoHunter.Stage9CeremonyAuditV1",
        "version": 1,
        "ceremony_id": ceremony,
        "tool_version": TOOL_VERSION,
        "source_revision": source_revision_value,
        "environment": environment,
        "root_key_set_digest": release.pinned_root.key_set_digest,
        "root_key_ids": list(release.root_keys),
        "pdsa_key_set_digest": release.pdsa_key_set_digest,
        "pdsa_key_ids": list(release.pdsa_keys),
        "k_recovery_digest": release.recovery_public_digest,
        "k_recovery_name": release.recovery_name,
        "release_payload_digest": release.payload_digest,
        "signed_release_digest": hashlib.sha256(release.serialized_envelope).hexdigest(),
        "revocation_state_digest": release.revocations.state_digest,
        "revocation_sequence": release.revocations.sequence,
        "accepted_release_signer_ids": [s["signer_id"] for s in release_envelope["signatures"]],
        "accepted_revocation_signer_ids": [
            s["signer_id"] for s in revocation_envelope["signatures"]
        ],
        "thresholds": {"root": release.root_threshold, "pdsa": release.pdsa_threshold},
        "freeze_manifest_digest": canonical_digest(dict(manifest)).hex(),
        "timestamps_supplied_by_ceremony_inputs": {
            "valid_from": release_envelope["payload"]["production_contract"]["valid_from"],
            "valid_until": release_envelope["payload"]["production_contract"]["valid_until"],
            "revocation_effective_at": revocation_envelope["payload"]["effective_at"],
        },
        "final_status": manifest["status"],
    }


def publish_final(
    output: Path,
    *,
    ceremony: str,
    verified_release: VerifiedReleasePolicyV1,
    manifest: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> Path:
    """Atomically publish only a completely verified final directory; never overwrite."""
    verify_freeze_manifest(dict(manifest), verified_release=verified_release)
    if (
        audit.get("schema") != "CryptoHunter.Stage9CeremonyAuditV1"
        or audit.get("ceremony_id") != ceremony
        or audit.get("freeze_manifest_digest") != canonical_digest(dict(manifest)).hex()
        or audit.get("final_status") != manifest["status"]
    ):
        raise PolicyVectorError("audit transcript is not bound to the verified final manifest")
    final = output / "final" / ceremony
    if final.exists():
        raise PolicyVectorError("completed ceremony already exists")
    staging_root = output / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{ceremony}.", dir=staging_root))
    try:
        artifacts = {"freeze_manifest": manifest, "audit": audit}
        for name, document in artifacts.items():
            (temporary / f"{name}.json").write_bytes(canonical_json_bytes(dict(document)) + b"\n")
        (output / "final").mkdir(parents=True, exist_ok=True)
        os.rename(temporary, final)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final


def _read(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: str, document: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise PolicyVectorError("refusing to overwrite ceremony artifact")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_json_bytes(dict(document)) + b"\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    for name in (
        "prepare-root-anchor",
        "prepare-pdsa-bundle",
        "prepare-recovery-bundle",
        "prepare-release",
        "prepare-signing-request",
        "prepare-initial-revocation",
        "assemble-signed-release",
        "assemble-signed-revocation",
        "verify-ceremony",
        "build-freeze-manifest",
        "verify-freeze-manifest",
    ):
        command = sub.add_parser(name)
        command.add_argument("--input", required=True, help="phase-specific canonical JSON input")
        command.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    data = _read(args.input)
    # The CLI's phase documents provide named arguments.  Keeping phases separate
    # prevents an accidental online all-in-one signer path.
    if args.phase == "prepare-root-anchor":
        result, _ = build_root_anchor_bundle(**data)
    elif args.phase == "prepare-pdsa-bundle":
        result = build_pdsa_public_bundle(**data)
    elif args.phase == "prepare-recovery-bundle":
        result = build_recovery_public_bundle(**data)
    elif args.phase == "prepare-release":
        result = build_unsigned_release_policy(**data)
    elif args.phase == "prepare-signing-request":
        pinned = verify_root_anchor_bundle(data.pop("root_bundle"))
        result = build_signing_request(pinned_root=pinned, **data)
    elif args.phase == "prepare-initial-revocation":
        result = build_initial_revocation_payload(**data)
    elif args.phase in ("assemble-signed-release", "assemble-signed-revocation"):
        pinned = verify_root_anchor_bundle(data.pop("root_bundle"))
        result = assemble_signed_artifact(pinned_root=pinned, **data)
    elif args.phase == "verify-ceremony":
        data["verification_time"] = datetime.fromisoformat(
            data["verification_time"].replace("Z", "+00:00")
        )
        release = verify_ceremony(**data)
        result = {"status": "PASS", "release_payload_digest": release.payload_digest}
    elif args.phase == "build-freeze-manifest":
        data["verification_time"] = datetime.fromisoformat(
            data["verification_time"].replace("Z", "+00:00")
        )
        release = verify_ceremony(**data)
        result = build_frozen_manifest(release, artifact_source_revision=source_revision())
    else:
        manifest = data.pop("manifest")
        data["verification_time"] = datetime.fromisoformat(
            data["verification_time"].replace("Z", "+00:00")
        )
        release = verify_ceremony(**data)
        verify_freeze_manifest(manifest, verified_release=release)
        result = {"status": "PASS"}
    _write(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
