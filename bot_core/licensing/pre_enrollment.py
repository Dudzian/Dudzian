"""Executable canonical PreEnrollmentRequestV1 payload and public proof of possession.

The frozen architecture defines the exact payload fields, but leaves the
``schema_version`` literal unspecified.  This executable profile uses the artifact
name ``PreEnrollmentRequestV1``, following existing enrollment version conventions.
The implementation uses the existing digest-shaped exchange/projection/evidence
references, a 32-byte lowercase-hex request nonce, positive exact-integer release
generation, and the proposal's canonical ``pchal_<UUIDv7>`` challenge identifier.
These are executable field-encoding conventions where the freeze lists a field
without an explicit encoding; they do not change the frozen field set.
Parsing, public-key proof of possession and byte-binding comparisons do not confer
production authorization or authenticate PDSA challenges / TPM custody evidence.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils

from .canonical import canonical_json_bytes, parse_canonical
from .product_profile import PRODUCT_NAME, PRODUCTION_PRODUCT_PROFILE

SCHEMA_VERSION = "PreEnrollmentRequestV1"
ALGORITHM_PROFILE = "ECDSA-P256-SHA256"
PDSA_TRUST_DOMAIN = "PDSA_PRODUCTION_2_OF_3_ED25519"
SIGNATURE_DOMAIN = b"CryptoHunter.Stage9.PreEnrollmentRequest.v1\x00"
P256_ORDER = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
MAX_EXACT_INTEGER = 9_007_199_254_740_991
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX130 = re.compile(r"[0-9a-f]{130}\Z")
_CHALLENGE_ID = re.compile(
    r"pchal_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"
)

PAYLOAD_FIELDS = frozenset(
    {
        "schema_version",
        "environment",
        "product",
        "product_profile",
        "pdsa_trust_domain",
        "pdsa_challenge_id",
        "pdsa_challenge_digest_sha256",
        "pdsa_challenge_nonce_digest_sha256",
        "tpm_enrollment_request_digest_sha256",
        "tpm_enrollment_challenge_digest_sha256",
        "tpm_enrollment_response_digest_sha256",
        "verified_tpm_exchange_reference",
        "verified_tpm_public_projection_id",
        "ek_public_digest",
        "ak_public_digest",
        "tpm_attestation_evidence_reference",
        "pre_enrollment_public_key_algorithm_profile",
        "pre_enrollment_public_key_canonical_bytes",
        "pre_enrollment_public_key_fingerprint_sha256",
        "release_policy_digest_sha256",
        "release_policy_generation",
        "request_nonce_hex",
    }
)
_DIGEST_FIELDS = frozenset(
    {
        "pdsa_challenge_digest_sha256",
        "pdsa_challenge_nonce_digest_sha256",
        "tpm_enrollment_request_digest_sha256",
        "tpm_enrollment_challenge_digest_sha256",
        "tpm_enrollment_response_digest_sha256",
        "verified_tpm_exchange_reference",
        "verified_tpm_public_projection_id",
        "ek_public_digest",
        "ak_public_digest",
        "tpm_attestation_evidence_reference",
        "pre_enrollment_public_key_fingerprint_sha256",
        "release_policy_digest_sha256",
        "request_nonce_hex",
    }
)


class PreEnrollmentError(ValueError):
    """Malformed or mismatched public pre-enrollment contract."""


def validate_public_key(raw: bytes) -> ec.EllipticCurvePublicKey:
    """Accept only the frozen SEC1 uncompressed, finite NIST P-256 point."""
    if type(raw) is not bytes or len(raw) != 65 or raw[0] != 4:
        raise PreEnrollmentError("INVALID_PRE_ENROLLMENT_PUBLIC_KEY")
    try:
        return ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), raw)
    except ValueError as exc:
        raise PreEnrollmentError("INVALID_PRE_ENROLLMENT_PUBLIC_KEY") from exc


def public_key_fingerprint(raw: bytes) -> str:
    """SHA-256 of exact canonical SEC1 bytes, never of a provider blob or SPKI."""
    validate_public_key(raw)
    return hashlib.sha256(raw).hexdigest()


def _validate_payload(value: dict[str, Any]) -> None:
    if set(value) != PAYLOAD_FIELDS:
        raise PreEnrollmentError("PRE_ENROLLMENT_SCHEMA_MISMATCH")
    required = {
        "schema_version": SCHEMA_VERSION,
        "environment": "PRODUCTION",
        "product": PRODUCT_NAME,
        "product_profile": PRODUCTION_PRODUCT_PROFILE,
        "pdsa_trust_domain": PDSA_TRUST_DOMAIN,
        "pre_enrollment_public_key_algorithm_profile": ALGORITHM_PROFILE,
    }
    for field, expected in required.items():
        if type(value[field]) is not str or value[field] != expected:
            raise PreEnrollmentError(f"INVALID_{field.upper()}")
    for field in _DIGEST_FIELDS:
        if type(value[field]) is not str or _HEX64.fullmatch(value[field]) is None:
            raise PreEnrollmentError(f"INVALID_{field.upper()}")
    challenge_id = value["pdsa_challenge_id"]
    if type(challenge_id) is not str or _CHALLENGE_ID.fullmatch(challenge_id) is None:
        raise PreEnrollmentError("INVALID_PDSA_CHALLENGE_ID")
    # UUID variant/version are additionally checked rather than trusted from text.
    parsed_id = uuid.UUID(challenge_id.removeprefix("pchal_"))
    if parsed_id.version != 7 or str(parsed_id) != challenge_id[6:]:
        raise PreEnrollmentError("INVALID_PDSA_CHALLENGE_ID")
    generation = value["release_policy_generation"]
    if type(generation) is not int or not 1 <= generation <= MAX_EXACT_INTEGER:
        raise PreEnrollmentError("INVALID_RELEASE_POLICY_GENERATION")
    key_hex = value["pre_enrollment_public_key_canonical_bytes"]
    if type(key_hex) is not str or _HEX130.fullmatch(key_hex) is None:
        raise PreEnrollmentError("INVALID_PRE_ENROLLMENT_PUBLIC_KEY")
    fingerprint = public_key_fingerprint(bytes.fromhex(key_hex))
    if fingerprint != value["pre_enrollment_public_key_fingerprint_sha256"]:
        raise PreEnrollmentError("PRE_ENROLLMENT_PUBLIC_KEY_FINGERPRINT_MISMATCH")


def _parse_payload(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or len(raw) > 16_384:
        raise PreEnrollmentError("INVALID_PRE_ENROLLMENT_CANONICAL_BYTES")
    try:
        payload: dict[str, Any] = parse_canonical(raw)
    except (ValueError, TypeError) as exc:
        raise PreEnrollmentError("NONCANONICAL_PRE_ENROLLMENT_REQUEST") from exc
    _validate_payload(payload)
    return payload


@dataclass(frozen=True, init=False)
class PreEnrollmentRequestV1:
    """Immutable public payload; validation alone is not legal enrollment."""

    canonical_bytes: bytes

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("use from_mapping() or from_canonical_bytes()")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PreEnrollmentRequestV1:
        if not isinstance(value, Mapping):
            raise PreEnrollmentError("PRE_ENROLLMENT_SCHEMA_MISMATCH")
        payload = dict(value)
        _validate_payload(payload)
        result = object.__new__(cls)
        object.__setattr__(result, "canonical_bytes", canonical_json_bytes(payload))
        return result

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> PreEnrollmentRequestV1:
        return cls.from_mapping(_parse_payload(raw))

    @property
    def document(self) -> dict[str, Any]:
        # Frozen dataclasses are a convenience, not an authority capability.
        # Revalidation also protects consequential consumers from bypassed init.
        return _parse_payload(self.canonical_bytes)

    @property
    def digest_sha256(self) -> str:
        _parse_payload(self.canonical_bytes)
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @property
    def signing_bytes(self) -> bytes:
        return SIGNATURE_DOMAIN + bytes.fromhex(self.digest_sha256)

    def verify_signature(self, signature: bytes) -> None:
        """Verify strict low-S request PoP; this does not verify TPM custody."""
        payload = self.document  # Recompute the exact SEC1 fingerprint before verification.
        if type(signature) is not bytes or not 8 <= len(signature) <= 72:
            raise PreEnrollmentError("INVALID_PRE_ENROLLMENT_SIGNATURE")
        try:
            r, s = utils.decode_dss_signature(signature)
            if (
                utils.encode_dss_signature(r, s) != signature
                or not 1 <= r < P256_ORDER
                or not 1 <= s <= P256_ORDER // 2
            ):
                raise ValueError("noncanonical signature")
            key = validate_public_key(
                bytes.fromhex(payload["pre_enrollment_public_key_canonical_bytes"])
            )
            key.verify(signature, self.signing_bytes, ec.ECDSA(hashes.SHA256()))
        except (ValueError, InvalidSignature) as exc:
            raise PreEnrollmentError("INVALID_PRE_ENROLLMENT_SIGNATURE") from exc

    def require_production_trust_binding(self, context: object) -> None:
        """Require the canonical loader's capability and its exact release identity."""
        from deployment.windows_stage9_production_trust import (
            require_verified_production_trust_context,
        )

        trusted = require_verified_production_trust_context(context)
        payload = self.document
        if (
            payload["release_policy_digest_sha256"] != trusted.release_payload_digest
            or payload["release_policy_generation"] != trusted.release_version
        ):
            raise PreEnrollmentError("PRE_ENROLLMENT_PRODUCTION_TRUST_BINDING_MISMATCH")

    def compare_tpm_exchange_bindings(
        self,
        *,
        activation_request_raw: bytes,
        request_raw: bytes,
        challenge_raw: bytes,
        response_raw: bytes,
    ) -> None:
        """Compare exact canonical retained exchange bytes, without authenticating them.

        This deliberately does not invoke the existing production-named verifier:
        its creation/PoP domains remain TEST_ONLY.  Production composition must
        independently reject unauthenticated challenge and custody evidence.
        Existing ActivationRequestV1 evidence_reference is the public projection
        evidence ID; this envelope uses that same public evidence reference.
        """
        from .activation_request import ActivationRequestV1
        from .device_enrollment import TPMPublicProjectionV1
        from .tpm_attestation import (
            EXCHANGE_REFERENCE_DOMAIN,
            TPMEnrollmentChallengeResponseV1,
            TPMEnrollmentChallengeV1,
            TPMEnrollmentRequestV1,
            _validate_activation_projection_binding,
        )

        raw_items = (activation_request_raw, request_raw, challenge_raw, response_raw)
        if any(type(raw) is not bytes for raw in raw_items):
            raise PreEnrollmentError("INVALID_TPM_EXCHANGE_BYTES")
        try:
            activation = ActivationRequestV1.from_mapping(parse_canonical(activation_request_raw))
            request = TPMEnrollmentRequestV1.from_canonical_bytes(request_raw)
            challenge = TPMEnrollmentChallengeV1.from_canonical_bytes(challenge_raw)
            response = TPMEnrollmentChallengeResponseV1.from_canonical_bytes(response_raw)
            req, ch, rsp = request.document, challenge.document, response.document
            projection = TPMPublicProjectionV1.verify(req["public_projection"])
            _validate_activation_projection_binding(activation, projection)
        except (ValueError, TypeError, KeyError) as exc:
            raise PreEnrollmentError("INVALID_TPM_EXCHANGE_BYTES") from exc
        digests = tuple(hashlib.sha256(raw).hexdigest() for raw in raw_items)
        reference = hashlib.sha256(
            EXCHANGE_REFERENCE_DOMAIN + b"".join(bytes.fromhex(item) for item in digests)
        ).hexdigest()
        payload = self.document
        expected = {
            "tpm_enrollment_request_digest_sha256": digests[1],
            "tpm_enrollment_challenge_digest_sha256": digests[2],
            "tpm_enrollment_response_digest_sha256": digests[3],
            "verified_tpm_exchange_reference": reference,
            "verified_tpm_public_projection_id": projection.evidence_reference,
            "ek_public_digest": projection.document["ek"]["public_digest"],
            "ak_public_digest": projection.document["ak"]["public_digest"],
            "tpm_attestation_evidence_reference": activation.document["tpm"]["evidence_reference"],
            "release_policy_digest_sha256": req["release_policy_digest"],
            "release_policy_generation": activation.document["release"]["release_policy_version"],
        }
        if (
            any(payload[field] != expected_value for field, expected_value in expected.items())
            or req["activation_request_id"] != activation.document["request_id"]
            or req["activation_request_digest"] != digests[0]
            or req["installation_id"] != activation.document["installation_id"]
            or req["requested_entitlements"] != activation.document["requested_entitlements"]
            or req["release_policy_digest"]
            != activation.document["release"]["release_policy_digest"]
            or ch["request_id"] != req["request_id"]
            or ch["public_projection_id"] != projection.evidence_reference
            or rsp["request_id"] != req["request_id"]
            or rsp["challenge_id"] != ch["challenge_id"]
            or rsp["issuer_nonce_hex"] != ch["issuer_nonce_hex"]
        ):
            raise PreEnrollmentError("PRE_ENROLLMENT_TPM_EXCHANGE_BINDING_MISMATCH")
