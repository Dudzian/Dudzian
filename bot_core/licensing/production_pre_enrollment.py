"""Compose production pre-enrollment authentication without issuing a license.

The trusted PDSA service owns both issuer stores. Their protected deployment,
and the installed public Production Trust package are prerequisites. The local
Windows producer separately qualifies the key before producing raw evidence.
This module does not expose an HTTP service or grant a caller
authority merely because it can choose a database path or parse public artifacts.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from weakref import WeakKeyDictionary

from deployment.production_enrollment_issuer import (
    ProductionEnrollmentIssuerContext,
    require_production_pdsa_store,
    require_production_tpm_store,
)
from deployment.windows_cng_pre_enrollment import require_verified_production_cng_key
from deployment.windows_stage9_production_trust import (
    ProductionTrustContext,
    require_current_production_trust_context,
)

from .activation_request import ActivationRequestV1
from .canonical import parse_canonical
from .device_enrollment import TPMPublicProjectionV1
from .pdsa_enrollment_challenge import (
    PDSAChallengeStore,
    VerifiedIssuedPDSAChallenge,
    require_verified_issued_challenge,
    verify_signed_production_pdsa_challenge,
)
from .pre_enrollment import (
    ALGORITHM_PROFILE,
    PDSA_TRUST_DOMAIN,
    SCHEMA_VERSION,
    PreEnrollmentRequestV1,
    public_key_fingerprint,
)
from .product_profile import PRODUCT_NAME, PRODUCTION_PRODUCT_PROFILE
from .production_tpm_custody import (
    ProductionPreEnrollmentKeyCustodyVerifier,
    ProductionTPMChallengeStore,
    ProductionTPMEnrollmentVerifier,
    VerifiedProductionPreEnrollmentKeyCustody,
    VerifiedProductionTPMExchange,
    require_verified_pre_enrollment_custody,
    require_verified_production_tpm_exchange,
    verify_production_tpm_endorsement,
)
from .tpm_attestation import (
    EXCHANGE_REFERENCE_DOMAIN,
    TPMEnrollmentChallengeV1,
    TPMEnrollmentRequestV1,
)


class ProductionPreEnrollmentError(ValueError):
    """One required production authentication boundary failed closed."""


def build_production_pre_enrollment_request(
    *,
    context: object,
    challenge: object,
    challenge_store: PDSAChallengeStore,
    exchange: object,
    pending: ProductionTPMChallengeStore,
    key: object,
) -> PreEnrollmentRequestV1:
    """Derive the exact frozen request from verified sources and fresh CSPRNG.

    Creation custody is collected afterwards, binding this complete request
    digest. The public evidence-reference field remains the projection reference
    to avoid a circular request/evidence digest dependency.
    """
    issuer = require_production_pdsa_store(challenge_store, context=context)
    require_production_tpm_store(
        pending, context=context, pdsa_store=challenge_store, issuer=issuer
    )
    trusted = require_current_production_trust_context(context)
    issued = require_verified_issued_challenge(
        challenge, store=challenge_store, context=trusted, issuer=issuer
    )
    verified = require_verified_production_tpm_exchange(
        exchange, context=trusted, pending=pending, issuer=issuer
    )
    qualified = require_verified_production_cng_key(key)
    activation_raw, request_raw, challenge_raw, response_raw = verified.retained_bytes
    result = build_local_production_pre_enrollment_request(
        context=trusted,
        challenge_raw=issued.canonical_bytes,
        activation_request_raw=activation_raw,
        tpm_request_raw=request_raw,
        tpm_challenge_raw=challenge_raw,
        tpm_response_raw=response_raw,
        key=qualified,
    )
    issued.require_request_binding(result)
    return result


def build_local_production_pre_enrollment_request(
    *,
    context: object,
    challenge_raw: bytes,
    activation_request_raw: bytes,
    tpm_request_raw: bytes,
    tpm_challenge_raw: bytes,
    tpm_response_raw: bytes,
    key: object,
) -> PreEnrollmentRequestV1:
    """Prepare public bytes locally; only the issuer verifies retained state.

    The client verifies signed challenge authority and exact byte bindings. It
    never manufactures an issuer-issued capability from transmitted artifacts.
    """
    trusted = require_current_production_trust_context(context)
    signed = verify_signed_production_pdsa_challenge(challenge_raw, trusted)
    qualified = require_verified_production_cng_key(key)
    activation_raw, request_raw, challenge_raw, response_raw = (
        activation_request_raw,
        tpm_request_raw,
        tpm_challenge_raw,
        tpm_response_raw,
    )
    activation = ActivationRequestV1.from_mapping(parse_canonical(activation_raw))
    tpm_request = TPMEnrollmentRequestV1.from_canonical_bytes(request_raw)
    projection = TPMPublicProjectionV1.verify(tpm_request.document["public_projection"])
    payload = signed.document["payload"]
    reference = hashlib.sha256(
        EXCHANGE_REFERENCE_DOMAIN
        + b"".join(
            hashlib.sha256(raw).digest()
            for raw in (activation_raw, request_raw, challenge_raw, response_raw)
        )
    ).hexdigest()
    result = PreEnrollmentRequestV1.from_mapping(
        {
            "schema_version": SCHEMA_VERSION,
            "environment": "PRODUCTION",
            "product": PRODUCT_NAME,
            "product_profile": PRODUCTION_PRODUCT_PROFILE,
            "pdsa_trust_domain": PDSA_TRUST_DOMAIN,
            "pdsa_challenge_id": payload["challenge_id"],
            "pdsa_challenge_digest_sha256": signed.digest_sha256,
            "pdsa_challenge_nonce_digest_sha256": signed.nonce_digest_sha256,
            "tpm_enrollment_request_digest_sha256": hashlib.sha256(request_raw).hexdigest(),
            "tpm_enrollment_challenge_digest_sha256": hashlib.sha256(challenge_raw).hexdigest(),
            "tpm_enrollment_response_digest_sha256": hashlib.sha256(response_raw).hexdigest(),
            "verified_tpm_exchange_reference": reference,
            "verified_tpm_public_projection_id": projection.evidence_reference,
            "ek_public_digest": projection.document["ek"]["public_digest"],
            "ak_public_digest": projection.document["ak"]["public_digest"],
            "tpm_attestation_evidence_reference": activation.document["tpm"]["evidence_reference"],
            "pre_enrollment_public_key_algorithm_profile": ALGORITHM_PROFILE,
            "pre_enrollment_public_key_canonical_bytes": qualified.public_key_bytes.hex(),
            "pre_enrollment_public_key_fingerprint_sha256": public_key_fingerprint(
                qualified.public_key_bytes
            ),
            "release_policy_digest_sha256": trusted.release_payload_digest,
            "release_policy_generation": trusted.release_version,
            "request_nonce_hex": secrets.token_bytes(32).hex(),
        }
    )
    result.require_production_trust_binding(trusted)
    result.compare_tpm_exchange_bindings(
        activation_request_raw=activation_raw,
        request_raw=request_raw,
        challenge_raw=challenge_raw,
        response_raw=response_raw,
    )
    return result


class AuthenticatedProductionPreEnrollment:
    """Verifier-issued acceptance capability, never legal enrollment authority."""

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("authenticated pre-enrollment comes only from production verification")

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("authenticated pre-enrollment is immutable")

    @property
    def request_raw(self) -> bytes:
        return _snapshot(self).request_raw

    @property
    def challenge_raw(self) -> bytes:
        return _snapshot(self).challenge_raw

    @property
    def challenge_store(self) -> PDSAChallengeStore:
        return _snapshot(self).challenge_store

    @property
    def context(self) -> ProductionTrustContext:
        return _snapshot(self).context

    @property
    def challenge(self) -> VerifiedIssuedPDSAChallenge:
        return _snapshot(self).challenge


@dataclass(frozen=True, slots=True)
class _AuthenticationSnapshot:
    request_raw: bytes
    signature: bytes
    challenge_raw: bytes
    challenge: VerifiedIssuedPDSAChallenge
    challenge_store: PDSAChallengeStore
    challenge_store_path: Path
    context: ProductionTrustContext
    exchange: VerifiedProductionTPMExchange
    pending: ProductionTPMChallengeStore
    custody: VerifiedProductionPreEnrollmentKeyCustody
    issuer: ProductionEnrollmentIssuerContext


_AUTHENTICATED: WeakKeyDictionary[AuthenticatedProductionPreEnrollment, _AuthenticationSnapshot] = (
    WeakKeyDictionary()
)


def _snapshot(value: object) -> _AuthenticationSnapshot:
    if type(value) is not AuthenticatedProductionPreEnrollment:
        raise ProductionPreEnrollmentError("AUTHENTICATED_PRODUCTION_PRE_ENROLLMENT_REQUIRED")
    result = _AUTHENTICATED.get(value)
    if result is None:
        raise ProductionPreEnrollmentError("AUTHENTICATED_PRODUCTION_PRE_ENROLLMENT_REQUIRED")
    return result


def authenticate_production_pre_enrollment(
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
) -> AuthenticatedProductionPreEnrollment:
    """Reverify retained raw artifacts, production TPM custody and request PoP.

    No preverified public projection or production-named historical verifier is
    accepted in place of the dedicated production exchange/custody verifiers.
    Only the store's subsequent atomic consume commits acceptance.
    """
    issuer = require_production_pdsa_store(challenge_store, context=context)
    require_production_tpm_store(
        pending, context=context, pdsa_store=challenge_store, issuer=issuer
    )
    trusted = require_current_production_trust_context(context)
    request = PreEnrollmentRequestV1.from_canonical_bytes(request_raw)
    request.require_production_trust_binding(trusted)
    challenge = challenge_store.verify_issued(challenge_raw, trusted)
    challenge.require_request_binding(request)
    endorsement = verify_production_tpm_endorsement(endorsement_raw, context=trusted)
    exchange = ProductionTPMEnrollmentVerifier().verify(
        activation_request_raw,
        tpm_request_raw,
        tpm_challenge_raw,
        tpm_response_raw,
        pending=pending,
        pdsa_challenge=challenge,
        pdsa_store=challenge_store,
        endorsement=endorsement,
        context=trusted,
    )
    request.compare_tpm_exchange_bindings(
        activation_request_raw=activation_request_raw,
        request_raw=tpm_request_raw,
        challenge_raw=tpm_challenge_raw,
        response_raw=tpm_response_raw,
    )
    request.verify_signature(signature)
    custody = ProductionPreEnrollmentKeyCustodyVerifier().verify(
        custody_evidence_raw,
        request=request,
        exchange=exchange,
        endorsement=endorsement,
        context=trusted,
    )
    accepted = object.__new__(AuthenticatedProductionPreEnrollment)
    _AUTHENTICATED[accepted] = _AuthenticationSnapshot(
        request_raw,
        signature,
        challenge_raw,
        challenge,
        challenge_store,
        challenge_store.path,
        trusted,
        exchange,
        pending,
        custody,
        issuer,
    )
    return require_authenticated_pre_enrollment(accepted, challenge_store=challenge_store)


def require_authenticated_pre_enrollment(
    value: object, *, challenge_store: PDSAChallengeStore | None = None
) -> AuthenticatedProductionPreEnrollment:
    """Recheck provenance and bindings before the issuer's atomic consume.

    The store rechecks live ISSUED/expiry inside its transaction. This guard does
    not reactivate CONSUMED challenges; exact retries only retrieve prior results.
    """
    snapshot = _snapshot(value)
    require_production_pdsa_store(
        snapshot.challenge_store, context=snapshot.context, issuer=snapshot.issuer
    )
    require_production_tpm_store(
        snapshot.pending,
        context=snapshot.context,
        issuer=snapshot.issuer,
        pdsa_store=snapshot.challenge_store,
    )
    if snapshot.challenge_store.path != snapshot.challenge_store_path or (
        challenge_store is not None and challenge_store is not snapshot.challenge_store
    ):
        raise ProductionPreEnrollmentError("PRE_ENROLLMENT_ISSUER_STORE_MISMATCH")
    trusted = require_current_production_trust_context(snapshot.context)
    request = PreEnrollmentRequestV1.from_canonical_bytes(snapshot.request_raw)
    request.require_production_trust_binding(trusted)
    request.verify_signature(snapshot.signature)
    exchange = require_verified_production_tpm_exchange(
        snapshot.exchange, context=trusted, pending=snapshot.pending, issuer=snapshot.issuer
    )
    require_verified_pre_enrollment_custody(
        snapshot.custody, context=trusted, request=request, exchange=exchange
    )
    # Exact type and issuance have already been checked; public properties are
    # derived from the private registered snapshot and cannot be copied to mint it.
    return cast(AuthenticatedProductionPreEnrollment, value)


def authenticated_package_expiry(value: object) -> str:
    """Derive the initial authorization deadline from live verified sources.

    The retained TPM challenge already caps its deadline to both the signed PDSA
    challenge and the curated endorsement. Parsing caller JSON cannot create the
    registered exchange capability used here.
    """
    require_authenticated_pre_enrollment(value)
    snapshot = _snapshot(value)
    exchange = require_verified_production_tpm_exchange(
        snapshot.exchange,
        context=snapshot.context,
        pending=snapshot.pending,
        issuer=snapshot.issuer,
    )
    pdsa_expiry: str = snapshot.challenge.document["payload"]["expires_at_utc"]
    tpm_challenge = TPMEnrollmentChallengeV1.from_canonical_bytes(exchange.retained_bytes[2])
    tpm_expiry: str = tpm_challenge.document["expires_at_utc"]
    # Both artifacts require exact canonical UTC seconds, so lexical ordering
    # agrees with chronological ordering after their trusted parsers validate.
    return min(pdsa_expiry, tpm_expiry)
