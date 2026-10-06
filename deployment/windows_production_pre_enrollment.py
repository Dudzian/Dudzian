"""Installed Windows boundary for production pre-enrollment authentication.

The frozen contract requires a signed, retained PDSA challenge and independent
TPM creation/custody attestation. Neither may be replaced with local provider
properties or a request signature. See docs/windows_production_pre_enrollment.md.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any

from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.licensing.device_enrollment import TPMPublicProjectionV1
from bot_core.licensing.pdsa_enrollment_challenge import PDSAChallengeStore
from bot_core.licensing.production_pre_enrollment import (
    AuthenticatedProductionPreEnrollment,
    authenticate_production_pre_enrollment,
    build_local_production_pre_enrollment_request,
    require_authenticated_pre_enrollment,
)
from bot_core.licensing.production_tpm_custody import ProductionTPMChallengeStore
from bot_core.licensing.tpm_attestation import TPMEnrollmentRequestV1
from deployment.platforms.windows import production_trust_package_path, resolve_paths
from deployment.production_enrollment_issuer import (
    require_production_pdsa_store,
    require_production_tpm_store,
)
from deployment.windows_stage9_production_trust import (
    CEREMONY_ID,
    load_production_trust,
    require_current_production_trust_context,
)

REQUIRED_AUTHENTICATION_ARTIFACTS = (
    "RETAINED_ISSUED_QUORUM_SIGNED_PDSA_ENROLLMENT_CHALLENGE",
    "QUORUM_SIGNED_MANUFACTURER_VERIFIED_TPM_EK_ENDORSEMENT",
    "RETAINED_PRODUCTION_TPM_ENROLLMENT_EXCHANGE",
    "REQUEST_BOUND_TPM_CERTIFY_CREATION_CUSTODY_EVIDENCE",
    "EXACT_PRE_ENROLLMENT_REQUEST_PROOF_OF_POSSESSION",
)


def require_production_request_authentication(
    value: object,
) -> AuthenticatedProductionPreEnrollment:
    """Accept only the actual composed verifier's unchanged issued capability."""
    return require_authenticated_pre_enrollment(value)


@dataclass(frozen=True, slots=True)
class PreparedProductionPreEnrollment:
    """Transport data only; the issuer independently re-verifies every byte."""

    request_raw: bytes
    signature: bytes
    custody_evidence_raw: bytes


def prepare_installed_production_request(
    *,
    challenge_raw: bytes,
    activation_request_raw: bytes,
    tpm_request_raw: bytes,
    tpm_challenge_raw: bytes,
    tpm_response_raw: bytes,
    ak_key_name: str,
) -> PreparedProductionPreEnrollment:
    """Local Windows boundary: qualify A, collect creation B and sign request C.

    No issuer database, trust key, native backend or private-key handle is
    accepted from the caller. AK name is a locator, exact-bound to the retained
    projection by the open-only native bridge.
    """
    from deployment.windows_cng_custody_bridge import certify_pre_enrollment_key_creation
    from deployment.windows_cng_pre_enrollment import WindowsCNGPreEnrollmentKey

    paths = resolve_paths()
    trust = require_current_production_trust_context(
        load_production_trust(production_trust_package_path(CEREMONY_ID))
    )
    projection = TPMPublicProjectionV1.verify(
        TPMEnrollmentRequestV1.from_canonical_bytes(tpm_request_raw).document["public_projection"]
    )
    with WindowsCNGPreEnrollmentKey.open_or_create(paths.state / "PreEnrollment") as key:
        request = build_local_production_pre_enrollment_request(
            context=trust,
            challenge_raw=challenge_raw,
            activation_request_raw=activation_request_raw,
            tpm_request_raw=tpm_request_raw,
            tpm_challenge_raw=tpm_challenge_raw,
            tpm_response_raw=tpm_response_raw,
            key=key,
        )
        custody = certify_pre_enrollment_key_creation(
            key,
            request,
            ak_key_name=ak_key_name,
            target_tpm_projection=projection,
        )
        signature = key.sign_request(request, production_trust_context=trust)
        return PreparedProductionPreEnrollment(
            request.canonical_bytes, signature, custody.canonical_bytes
        )


def accept_retained_production_request(
    request_raw: bytes,
    signature: bytes,
    *,
    challenge_raw: bytes,
    challenge_store: PDSAChallengeStore,
    activation_request_raw: bytes,
    tpm_request_raw: bytes,
    tpm_challenge_raw: bytes,
    tpm_response_raw: bytes,
    pending: ProductionTPMChallengeStore,
    endorsement_raw: bytes,
    custody_evidence_raw: bytes,
    context: object,
) -> bytes:
    """Issuer boundary: reverify received B+C and atomically accept the request.

    This internal integration boundary requires issuer-owned protected stores;
    it does not deploy the PDSA service or turn a client-chosen database into
    issuer authority. A returned receipt records authentication only.
    """
    issuer = require_production_pdsa_store(challenge_store, context=context)
    require_production_tpm_store(
        pending, context=context, pdsa_store=challenge_store, issuer=issuer
    )
    previous = challenge_store.retry_exact_accepted(
        request_raw=request_raw, challenge_raw=challenge_raw
    )
    if previous is not None:
        return previous
    trust = require_current_production_trust_context(context)
    accepted = authenticate_production_pre_enrollment(
        request_raw,
        signature,
        challenge_raw=challenge_raw,
        challenge_store=challenge_store,
        activation_request_raw=activation_request_raw,
        tpm_request_raw=tpm_request_raw,
        tpm_challenge_raw=tpm_challenge_raw,
        tpm_response_raw=tpm_response_raw,
        pending=pending,
        endorsement_raw=endorsement_raw,
        custody_evidence_raw=custody_evidence_raw,
        context=trust,
    )
    return challenge_store.consume_authenticated_request(accepted)


def qualify_installed_production_key() -> dict[str, Any]:
    """Resolve the fixed production key using installed, verified trust only.

    This persists a machine identity. It does not construct an authenticated
    request, consume a PDSA challenge, enroll a device, or create LPPI authority.
    No caller-selected trust roots, PDSA keys, key name, or backend are accepted.
    """
    from deployment.windows_cng_pre_enrollment import WindowsCNGPreEnrollmentKey

    paths = resolve_paths()
    trust = require_current_production_trust_context(
        load_production_trust(production_trust_package_path(CEREMONY_ID))
    )
    with WindowsCNGPreEnrollmentKey.open_or_create(paths.state / "PreEnrollment") as key:
        return {
            "schema_version": "WindowsProductionPreEnrollmentQualificationV1",
            "environment": "PRODUCTION",
            "purpose": "LOCAL_PROVIDER_QUALIFICATION_ONLY",
            "release_policy_digest_sha256": trust.release_payload_digest,
            "release_policy_generation": trust.release_version,
            "key": key.public_evidence,
            "request_authentication": "REQUIRES_PRODUCTION_ARTIFACTS",
            "required_authentication_artifacts": list(REQUIRED_AUTHENTICATION_ARTIFACTS),
            "legal_enrollment": "NOT_PERFORMED",
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qualify-key",
        action="store_true",
        required=True,
        help="create or reuse the persistent production key; emit public qualification only",
    )
    parser.parse_args(argv)
    try:
        result = qualify_installed_production_key()
    except (OSError, RuntimeError, ValueError):
        # Native failures may include machine-specific paths. No exception or
        # secret-bearing raw input is forwarded to operator evidence/stdout.
        print("PRODUCTION_PRE_ENROLLMENT_QUALIFICATION_FAILED", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
