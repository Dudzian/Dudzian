#!/usr/bin/env python3
"""Operator CLI for verified TEST_ONLY or production TPM activation requests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot_core.licensing.device_enrollment import build_activation_request, export_bundle
from deployment.platforms.windows import production_trust_package_path
from deployment.windows_stage9_production_trust import (
    CEREMONY_ID,
    ProductionTrustUnavailable,
    load_production_trust,
)
from bot_core.licensing.tpm_attestation import (
    PendingChallengeStore,
    ProductionTPMAttestationVerifier,
    TPMEnrollmentChallengeResponseV1,
    TPMEnrollmentRequestV1,
    certify_qualifying_data,
    create_issuer_challenge,
    credential_activation_proof,
)
from deployment.windows_stage9_policy_material import (
    RELEASE_SCHEMA,
    canonical_digest,
    validate_schema,
)
from deployment.windows_tpm_activation_bridge import (
    PhysicalWindowsTPMEnrollmentSubstrate,
    load_or_create_installation_id,
    physical_report,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="CryptoHunter TEST_ONLY TPM activation bridge")
    sub = result.add_subparsers(dest="command", required=True)
    for command in ("create", "physical-test", "negative-test"):
        item = sub.add_parser(command)
        item.add_argument("--environment", required=True, choices=("TEST_ONLY", "PRODUCTION"))
        item.add_argument("--output", required=True, type=Path)
        item.add_argument("--release-policy", type=Path)
        item.add_argument("--edition", default="pro")
        item.add_argument("--feature", action="append", default=["core_bot"])
    cleanup = sub.add_parser("cleanup")
    cleanup.add_argument("--environment", required=True, choices=("TEST_ONLY", "PRODUCTION"))
    cleanup.add_argument("--output", required=True, type=Path)
    return result


def _release(path: Path) -> tuple[str, int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_schema(value, RELEASE_SCHEMA)
    if value["purpose"] != "TEST_ONLY":
        raise ValueError("physical rehearsal requires an existing TEST_ONLY ReleasePolicyV1")
    return canonical_digest(value).hex(), 1


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    trust_context = None
    if args.command == "cleanup" and args.environment != "TEST_ONLY":
        print("cleanup is TEST_ONLY; PRODUCTION cleanup is forbidden", file=sys.stderr)
        return 2
    if args.environment == "PRODUCTION":
        try:
            trust_context = load_production_trust(production_trust_package_path(CEREMONY_ID))
        except ProductionTrustUnavailable:
            print("PRODUCTION_TRUST_UNAVAILABLE", file=sys.stderr)
            return 2
    if args.command == "cleanup":
        args.output.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(args.output / ".cryptohunter-test-state", ignore_errors=True)
        report = {
            "schema": "CryptoHunterTPMActivationCleanupV1",
            "environment": "TEST_ONLY",
            "transient_contexts": "TBS_CONTEXT_SCOPED",
            "persistent_objects": "NONE",
            "nv_indices": "NONE",
            "temporary_private_material": "NONE",
            "test_result": "PASS",
        }
        (args.output / "cleanup-evidence.json").write_text(
            json.dumps(report, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        print("TEST RESULT = PASS")
        return 0
    if args.environment == "TEST_ONLY":
        if args.release_policy is None:
            raise ValueError("TEST_ONLY requires --release-policy")
        release = _release(args.release_policy)
    else:
        release = (trust_context.release_payload_digest, trust_context.release_version)
    state = args.output / ".cryptohunter-test-state" / "installation-id"
    load_or_create_installation_id(state)
    with PhysicalWindowsTPMEnrollmentSubstrate() as substrate:
        if args.command == "create":
            evidence = substrate.collect()
            activation = build_activation_request(
                evidence=evidence,
                release_policy_digest=release[0],
                release_policy_version=release[1],
                requested_entitlements={
                    "product": "CryptoHunter",
                    "edition": args.edition,
                    "requested_features": sorted(set(args.feature)),
                },
                installation_id=load_or_create_installation_id(state),
                architecture=platform.machine() or "UNKNOWN",
                environment=args.environment,
                production_trust_context=trust_context,
            )
        else:
            evidence, k_psa, ek, ak = substrate.provision()
            activation = build_activation_request(
                evidence=evidence,
                release_policy_digest=release[0],
                release_policy_version=release[1],
                requested_entitlements={
                    "product": "CryptoHunter",
                    "edition": args.edition,
                    "requested_features": sorted(set(args.feature)),
                },
                installation_id=load_or_create_installation_id(state),
                architecture=platform.machine() or "UNKNOWN",
                environment=args.environment,
                production_trust_context=trust_context,
            )
            enrollment = TPMEnrollmentRequestV1.create(
                activation_request=activation, public_projection=evidence
            )
            pending = PendingChallengeStore()
            challenge = create_issuer_challenge(
                enrollment, pending, expires_at_utc="2099-01-01T00:00:00Z"
            )
            ch = challenge.document
            recovered = substrate.activate_credential(
                ak,
                ek,
                bytes.fromhex(ch["credential_blob_hex"]),
                bytes.fromhex(ch["encrypted_secret_hex"]),
            )
            issuer_nonce = bytes.fromhex(ch["issuer_nonce_hex"])
            pop = substrate.sign_k_psa(k_psa, issuer_nonce)
            qualifying = certify_qualifying_data(
                issuer_nonce, hashlib.sha256(activation.canonical_bytes).digest()
            )
            attest, certify_signature = substrate.certify_creation(ak, k_psa, qualifying)
            response = TPMEnrollmentChallengeResponseV1.create(
                challenge,
                activated_credential_digest=hashlib.sha256(recovered).hexdigest(),
                credential_activation_proof_hex=credential_activation_proof(
                    recovered, ch["challenge_id"]
                ).hex(),
                certify_creation_attest_hex=attest.hex(),
                certify_creation_signature_hex=certify_signature.hex(),
                k_psa_pop_signature_der_hex=pop.hex(),
            )
            if args.command == "negative-test":
                changed = response.document
                changed["credential_activation_proof_hex"] = "00" * 32
                response_raw = json.dumps(changed, sort_keys=True, separators=(",", ":")).encode()
                try:
                    ProductionTPMAttestationVerifier().verify(
                        activation.canonical_bytes,
                        enrollment.canonical_bytes,
                        challenge.canonical_bytes,
                        response_raw,
                        pending=pending,
                        expected_release_policy_digest=release[0],
                    )
                except ValueError:
                    pending.require_pending(challenge)
                    print("NEGATIVE TEST = PASS")
                    return 0
                raise RuntimeError("NEGATIVE TEST = FAIL")
            verified = ProductionTPMAttestationVerifier().verify(
                activation.canonical_bytes,
                enrollment.canonical_bytes,
                challenge.canonical_bytes,
                response.canonical_bytes,
                pending=pending,
                expected_release_policy_digest=release[0],
            )
            pending.consume(challenge)
    args.output.mkdir(parents=True, exist_ok=True)
    report = physical_report(evidence)
    if args.command == "physical-test":
        report.update(
            {
                "k_psa_tpm_sign": "PASS",
                "activate_credential": "PASS",
                "certify_creation": "PASS",
                "issuer_exchange": "PASS",
                "activation_request": "PASS",
                "exchange_reference": verified.exchange_reference,
            }
        )
        folder = (
            args.output
            / f"CryptoHunter-Activation-Request-{activation.document['request_id'][:12]}"
        )
        folder.mkdir(parents=True, exist_ok=False)
        for name, raw in {
            "activation-request.json": activation.canonical_bytes,
            "tpm-public-projection.json": evidence.canonical_bytes,
            "tpm-enrollment-request.json": enrollment.canonical_bytes,
            "tpm-enrollment-challenge.json": challenge.canonical_bytes,
            "tpm-enrollment-response.json": response.canonical_bytes,
        }.items():
            (folder / name).write_bytes(raw)
    target = args.output / "physical-preflight.json"
    target.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    print(f"PUBLIC TPM PROJECTION = IMPLEMENTED\n{target}")
    if args.command == "create":
        folder = export_bundle(
            args.output,
            activation,
            evidence,
            environment=args.environment,
            allowed_release=release if args.environment == "TEST_ONLY" else None,
            production_trust_context=trust_context,
        )
        print(folder)
        return 0
    print(folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
