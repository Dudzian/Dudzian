"""Frozen initial LPPI successor binding and independent signature gates."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .lppi_authority_custody import LPPIAuthorityKeyCustodyEvidenceV1

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils

from .canonical import canonical_json_bytes, parse_canonical
from .pre_enrollment import (
    P256_ORDER,
    PDSA_TRUST_DOMAIN,
    public_key_fingerprint,
    validate_public_key,
)

ALGORITHM_PROFILE = "ECDSA-P256-SHA256"
CUSTODY_PROFILE = "WINDOWS_PLATFORM_CRYPTO_PROVIDER_TPM_ECDSA_P256_V1"
PROVIDER_NAME = "Microsoft Platform Crypto Provider"
KEY_NAME = "CryptoHunter.Stage9.Production.LPPI.Authority.v1"
CONTINUITY_DOMAIN = b"CryptoHunter.Stage9.LPPIAuthorityKeyContinuity.v1\x00"
AUTHORITY_POP_DOMAIN = b"CryptoHunter.Stage9.LPPIAuthorityKeyPoP.v1\x00"
BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "environment",
        "pdsa_trust_domain",
        "pdsa_package_digest_sha256",
        "pre_enrollment_request_digest_sha256",
        "pre_enrollment_public_key_algorithm_profile",
        "pre_enrollment_public_key_fingerprint_sha256",
        "lppi_authority_public_key_algorithm_profile",
        "lppi_authority_public_key_fingerprint_sha256",
        "custody_profile",
        "lppi_authority_key_tpm_name",
        "lppi_authority_tpmt_public_sha256",
        "cng_provider_name",
        "cng_key_name",
        "cng_key_unique_name",
        "tpm_creation_attestation_sha256",
        "verified_tpm_public_projection_id",
        "verified_tpm_exchange_reference",
        "provisioning_subject_id",
        "enrollment_reference",
        "generation",
        "created_at_utc",
    }
)
POP_FIELDS = frozenset(
    {
        "lppi_authority_key_binding_digest_sha256",
        "pdsa_package_digest_sha256",
        "pre_enrollment_request_digest_sha256",
        "lppi_authority_public_key_fingerprint_sha256",
        "verified_tpm_public_projection_id",
        "verified_tpm_exchange_reference",
    }
)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_UNIQUE_NAME = re.compile(r"[A-Za-z0-9._\\:\-]{1,256}\Z")


class LPPIAuthorityKeyError(ValueError):
    """A frozen authority-key binding gate rejected its exact inputs."""


def _timestamp(value: object) -> datetime:
    if type(value) is not str:
        raise LPPIAuthorityKeyError("INVALID_LPPI_AUTHORITY_CREATED_AT")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise LPPIAuthorityKeyError("INVALID_LPPI_AUTHORITY_CREATED_AT") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise LPPIAuthorityKeyError("INVALID_LPPI_AUTHORITY_CREATED_AT")
    return parsed


@dataclass(frozen=True, slots=True)
class LPPIAuthorityKeyBindingV1:
    """Canonical transport bytes; construction alone grants no authority."""

    canonical_bytes: bytes

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> LPPIAuthorityKeyBindingV1:
        if type(raw) is not bytes or len(raw) > 16384:
            raise LPPIAuthorityKeyError("INVALID_LPPI_AUTHORITY_BINDING")
        try:
            payload = parse_canonical(raw)
            if type(payload) is not dict or set(payload) != BINDING_FIELDS:
                raise ValueError("binding schema mismatch")
            exact_values = {
                "schema_version": 1,
                "environment": "PRODUCTION",
                "pdsa_trust_domain": PDSA_TRUST_DOMAIN,
                "pre_enrollment_public_key_algorithm_profile": ALGORITHM_PROFILE,
                "lppi_authority_public_key_algorithm_profile": ALGORITHM_PROFILE,
                "custody_profile": CUSTODY_PROFILE,
                "cng_provider_name": PROVIDER_NAME,
                "cng_key_name": KEY_NAME,
                "generation": 1,
            }
            if any(
                type(payload[field]) is not type(expected) or payload[field] != expected
                for field, expected in exact_values.items()
            ):
                raise ValueError("binding required value mismatch")
            for field in BINDING_FIELDS:
                if field.endswith("sha256") or field in {
                    "verified_tpm_public_projection_id",
                    "verified_tpm_exchange_reference",
                }:
                    if type(payload[field]) is not str or _HEX64.fullmatch(payload[field]) is None:
                        raise ValueError("binding digest mismatch")
            fingerprint = payload["lppi_authority_public_key_fingerprint_sha256"]
            if (
                payload["lppi_authority_tpmt_public_sha256"] != fingerprint
                or payload["lppi_authority_key_tpm_name"] != "000b" + fingerprint
            ):
                raise ValueError("binding TPM name mismatch")
            if (
                type(payload["cng_key_unique_name"]) is not str
                or _UNIQUE_NAME.fullmatch(payload["cng_key_unique_name"]) is None
            ):
                raise ValueError("binding unique name mismatch")
            for field, prefix in (
                ("provisioning_subject_id", "psub_"),
                ("enrollment_reference", "penr_"),
            ):
                if (
                    type(payload[field]) is not str
                    or re.fullmatch(
                        prefix
                        + r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
                        payload[field],
                    )
                    is None
                ):
                    raise ValueError("binding identity mismatch")
            _timestamp(payload["created_at_utc"])
        except (ValueError, TypeError, KeyError, RecursionError) as exc:
            raise LPPIAuthorityKeyError("INVALID_LPPI_AUTHORITY_BINDING") from exc
        return cls(raw)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> LPPIAuthorityKeyBindingV1:
        return cls.from_canonical_bytes(canonical_json_bytes(payload))

    @property
    def document(self) -> dict[str, Any]:
        return cast(dict[str, Any], parse_canonical(self.canonical_bytes))

    @property
    def digest_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


def continuity_signed_bytes(binding_raw: bytes) -> bytes:
    binding = LPPIAuthorityKeyBindingV1.from_canonical_bytes(binding_raw)
    return CONTINUITY_DOMAIN + hashlib.sha256(binding.canonical_bytes).digest()


def authority_pop_challenge(binding: LPPIAuthorityKeyBindingV1) -> dict[str, str]:
    validated = LPPIAuthorityKeyBindingV1.from_canonical_bytes(binding.canonical_bytes)
    payload = validated.document
    return {
        field: validated.digest_sha256
        if field == "lppi_authority_key_binding_digest_sha256"
        else payload[field]
        for field in POP_FIELDS
    }


def authority_pop_signed_bytes(binding_raw: bytes) -> bytes:
    binding = LPPIAuthorityKeyBindingV1.from_canonical_bytes(binding_raw)
    return (
        AUTHORITY_POP_DOMAIN
        + hashlib.sha256(canonical_json_bytes(authority_pop_challenge(binding))).digest()
    )


def verify_low_s_signature(public: bytes, signature: bytes, message: bytes, *, error: str) -> None:
    try:
        validate_public_key(public)
        if type(signature) is not bytes or not 8 <= len(signature) <= 72:
            raise ValueError("invalid DER")
        r, s = utils.decode_dss_signature(signature)
        if (
            not 0 < r < P256_ORDER
            or not 0 < s <= P256_ORDER // 2
            or utils.encode_dss_signature(r, s) != signature
        ):
            raise ValueError("invalid DER")
        ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), public).verify(
            signature, message, ec.ECDSA(hashes.SHA256())
        )
    except (ValueError, TypeError, InvalidSignature) as exc:
        raise LPPIAuthorityKeyError(error) from exc


def require_binding_targets(
    binding: LPPIAuthorityKeyBindingV1,
    *,
    accepted: object,
    key: object,
    evidence: LPPIAuthorityKeyCustodyEvidenceV1,
) -> None:
    from deployment.windows_lppi_authority_key import require_verified_production_lppi_authority_key

    from .lppi_authority_custody import (
        LPPIAuthorityKeyCustodyEvidenceV1,
        parse_lppi_authority_public,
    )
    from .lppi_package_acceptance import require_verified_lppi_package_acceptance

    accepted = require_verified_lppi_package_acceptance(accepted)
    key = require_verified_production_lppi_authority_key(key)
    binding = LPPIAuthorityKeyBindingV1.from_canonical_bytes(binding.canonical_bytes)
    if type(evidence) is not LPPIAuthorityKeyCustodyEvidenceV1:
        raise LPPIAuthorityKeyError("REJECT_LPPI_AUTHORITY_CUSTODY_MISSING")
    document = evidence.document
    public = parse_lppi_authority_public(bytes.fromhex(document["subject_tpmt_public_hex"]))
    payload = binding.document
    source = accepted.package.payload
    expected = {
        "pdsa_package_digest_sha256": accepted.package.package_digest_sha256,
        "pre_enrollment_request_digest_sha256": accepted.request.digest_sha256,
        "pre_enrollment_public_key_algorithm_profile": source[
            "pre_enrollment_public_key_algorithm_profile"
        ],
        "pre_enrollment_public_key_fingerprint_sha256": source[
            "pre_enrollment_public_key_fingerprint_sha256"
        ],
        "verified_tpm_public_projection_id": source["verified_tpm_public_projection_id"],
        "verified_tpm_exchange_reference": source["verified_tpm_exchange_reference"],
        "provisioning_subject_id": source["provisioning_subject_id"],
        "enrollment_reference": source["enrollment_reference"],
        "cng_key_unique_name": key.key_unique_name,
        "lppi_authority_public_key_fingerprint_sha256": hashlib.sha256(public.raw).hexdigest(),
        "lppi_authority_tpmt_public_sha256": hashlib.sha256(public.raw).hexdigest(),
        "lppi_authority_key_tpm_name": public.name.hex(),
        "tpm_creation_attestation_sha256": hashlib.sha256(
            bytes.fromhex(document["certify_creation_attest_hex"])
        ).hexdigest(),
    }
    if (
        any(payload[field] != value for field, value in expected.items())
        or public.sec1 != key.public_key_bytes
    ):
        raise LPPIAuthorityKeyError("LPPI_AUTHORITY_KEY_LIFECYCLE_CONFLICT")
    if (
        public.sec1 == accepted.key.public_key_bytes
        or key.key_unique_name == accepted.pre_enrollment_key_unique_name
        or payload["lppi_authority_public_key_fingerprint_sha256"]
        == payload["pre_enrollment_public_key_fingerprint_sha256"]
    ):
        raise LPPIAuthorityKeyError("REJECT_LPPI_AUTHORITY_PRE_ENROLLMENT_KEY_REUSE")


def build_lppi_authority_key_binding(
    *,
    accepted: object,
    key: object,
    evidence: LPPIAuthorityKeyCustodyEvidenceV1,
    created_at_utc: str,
) -> LPPIAuthorityKeyBindingV1:
    from .lppi_authority_custody import parse_lppi_authority_public
    from .lppi_package_acceptance import require_verified_lppi_package_acceptance

    accepted = require_verified_lppi_package_acceptance(accepted)
    from deployment.windows_lppi_authority_key import require_verified_production_lppi_authority_key

    key = require_verified_production_lppi_authority_key(key)
    source = accepted.package.payload
    public = parse_lppi_authority_public(
        bytes.fromhex(evidence.document["subject_tpmt_public_hex"])
    )
    fingerprint = hashlib.sha256(public.raw).hexdigest()
    payload = {
        "schema_version": 1,
        "environment": "PRODUCTION",
        "pdsa_trust_domain": source["pdsa_trust_domain"],
        "pdsa_package_digest_sha256": accepted.package.package_digest_sha256,
        "pre_enrollment_request_digest_sha256": accepted.request.digest_sha256,
        "pre_enrollment_public_key_algorithm_profile": source[
            "pre_enrollment_public_key_algorithm_profile"
        ],
        "pre_enrollment_public_key_fingerprint_sha256": source[
            "pre_enrollment_public_key_fingerprint_sha256"
        ],
        "lppi_authority_public_key_algorithm_profile": ALGORITHM_PROFILE,
        "lppi_authority_public_key_fingerprint_sha256": fingerprint,
        "custody_profile": CUSTODY_PROFILE,
        "lppi_authority_key_tpm_name": public.name.hex(),
        "lppi_authority_tpmt_public_sha256": fingerprint,
        "cng_provider_name": PROVIDER_NAME,
        "cng_key_name": KEY_NAME,
        "cng_key_unique_name": key.key_unique_name,
        "tpm_creation_attestation_sha256": hashlib.sha256(
            bytes.fromhex(evidence.document["certify_creation_attest_hex"])
        ).hexdigest(),
        "verified_tpm_public_projection_id": source["verified_tpm_public_projection_id"],
        "verified_tpm_exchange_reference": source["verified_tpm_exchange_reference"],
        "provisioning_subject_id": source["provisioning_subject_id"],
        "enrollment_reference": source["enrollment_reference"],
        "generation": 1,
        "created_at_utc": created_at_utc,
    }
    binding = LPPIAuthorityKeyBindingV1.from_mapping(payload)
    require_binding_targets(binding, accepted=accepted, key=key, evidence=evidence)
    return binding


def verify_authority_key_continuity(
    binding: LPPIAuthorityKeyBindingV1, signature: bytes, *, accepted: object
) -> None:
    from .lppi_package_acceptance import require_verified_lppi_package_acceptance

    accepted = require_verified_lppi_package_acceptance(accepted)
    binding = LPPIAuthorityKeyBindingV1.from_canonical_bytes(binding.canonical_bytes)
    source = accepted.request.document
    public = bytes.fromhex(source["pre_enrollment_public_key_canonical_bytes"])
    fingerprint = public_key_fingerprint(public)
    if (
        any(
            value != fingerprint
            for value in (
                source["pre_enrollment_public_key_fingerprint_sha256"],
                accepted.package.payload["pre_enrollment_public_key_fingerprint_sha256"],
                binding.document["pre_enrollment_public_key_fingerprint_sha256"],
                public_key_fingerprint(accepted.key.public_key_bytes),
            )
        )
        or public != accepted.key.public_key_bytes
    ):
        raise LPPIAuthorityKeyError("REJECT_CONTINUITY_SIGNATURE_KEY_MISMATCH")
    verify_low_s_signature(
        public,
        signature,
        continuity_signed_bytes(binding.canonical_bytes),
        error="REJECT_CONTINUITY_SIGNATURE_KEY_MISMATCH",
    )


def verify_authority_key_pop(
    binding: LPPIAuthorityKeyBindingV1,
    signature: bytes,
    *,
    key: object,
    evidence: LPPIAuthorityKeyCustodyEvidenceV1,
) -> None:
    from deployment.windows_lppi_authority_key import require_verified_production_lppi_authority_key

    from .lppi_authority_custody import parse_lppi_authority_public

    if not signature:
        raise LPPIAuthorityKeyError("REJECT_AUTHORITY_KEY_POP_MISSING")
    key = require_verified_production_lppi_authority_key(key)
    binding = LPPIAuthorityKeyBindingV1.from_canonical_bytes(binding.canonical_bytes)
    public = parse_lppi_authority_public(
        bytes.fromhex(evidence.document["subject_tpmt_public_hex"])
    )
    if (
        public.sec1 != key.public_key_bytes
        or hashlib.sha256(public.raw).hexdigest()
        != binding.document["lppi_authority_public_key_fingerprint_sha256"]
    ):
        raise LPPIAuthorityKeyError("REJECT_AUTHORITY_KEY_POP_KEY_MISMATCH")
    verify_low_s_signature(
        public.sec1,
        signature,
        authority_pop_signed_bytes(binding.canonical_bytes),
        error="REJECT_AUTHORITY_KEY_POP_INVALID",
    )
