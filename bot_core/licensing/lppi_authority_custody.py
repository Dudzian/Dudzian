"""Successor-only LPPI authority creation custody.

Public evidence is never an authority capability. Verification requires both a
live client acceptance and the locally qualified successor CNG key. Existing
pre-enrollment creation evidence cannot satisfy this distinct domain and schema.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, cast
from weakref import WeakKeyDictionary

from .canonical import canonical_json_bytes
from .production_tpm_custody import (
    ParsedProductionTPMPublic,
    ProductionTPMCustodyError,
    _exact,
    _hex,
    _parse,
    _Reader,
    _verify_creation,
    parse_production_creation_attestation,
    parse_production_ecc_public,
)

CUSTODY_CREATION_DOMAIN = b"CryptoHunter.Stage9.LPPIAuthorityKeyCustody.CertifyCreation.v1\x00"
CUSTODY_PROFILE = "WINDOWS_PLATFORM_CRYPTO_PROVIDER_TPM_ECDSA_P256_V1"
ALGORITHM_PROFILE = "ECDSA-P256-SHA256"
PROVIDER_NAME = "Microsoft Platform Crypto Provider"
KEY_NAME = "CryptoHunter.Stage9.Production.LPPI.Authority.v1"
CUSTODY_FIELDS = frozenset(
    {
        "schema_version",
        "environment",
        "custody_profile",
        "cng_provider_name",
        "cng_key_name",
        "cng_key_unique_name",
        "cng_public_blob_sha256",
        "pdsa_package_digest_sha256",
        "pre_enrollment_request_digest_sha256",
        "verified_tpm_public_projection_id",
        "verified_tpm_exchange_reference",
        "ek_public_digest",
        "ak_public_digest",
        "pre_enrollment_public_key_fingerprint_sha256",
        "provisioning_subject_id",
        "enrollment_reference",
        "release_policy_digest_sha256",
        "release_policy_generation",
        "lppi_authority_public_key_algorithm_profile",
        "lppi_authority_public_key_fingerprint_sha256",
        "lppi_authority_tpmt_public_sha256",
        "subject_tpmt_public_hex",
        "subject_tpm_name",
        "subject_creation_hash",
        "subject_creation_ticket_hex",
        "retained_ak_tpmt_public_hex",
        "retained_ak_tpm_name",
        "retained_ak_qualified_name",
        "certify_creation_attest_hex",
        "certify_creation_signature_hex",
        "tpm_creation_attestation_sha256",
    }
)
BINDING_FIELDS = CUSTODY_FIELDS - {
    "certify_creation_attest_hex",
    "certify_creation_signature_hex",
    "tpm_creation_attestation_sha256",
}


class LPPIAuthorityCustodyError(ProductionTPMCustodyError):
    """Exact successor custody, native identity or cryptography failed closed."""


def parse_lppi_authority_public(raw: bytes) -> ParsedProductionTPMPublic:
    """Use the strict signing-key parser, with ECDSA-SHA256 as the sole scheme.

    Only the existing fixedTPM/fixedParent/sensitiveDataOrigin/userWithAuth/sign
    profile, optionally noDA, is accepted. NULL scheme is intentionally rejected;
    no Windows hardware evidence has qualified a different successor profile.
    """
    parsed = parse_production_ecc_public(raw, role="pre_enrollment")
    reader = _Reader(raw)
    reader.take(8)
    reader.blob()
    if (reader.number(2), reader.number(2), reader.number(2)) != (0x0010, 0x0018, 0x000B):
        raise LPPIAuthorityCustodyError("UNSUPPORTED_LPPI_AUTHORITY_TPM_SCHEME")
    return parsed


def lppi_authority_custody_qualifying_data(binding: Mapping[str, Any]) -> bytes:
    document = dict(binding)
    _exact(document, BINDING_FIELDS)
    return hashlib.sha256(CUSTODY_CREATION_DOMAIN + canonical_json_bytes(document)).digest()


def _ticket(raw: bytes) -> bytes:
    # Reuse the native bridge's strict marshaled TPMT_TK_CREATION codec.
    from deployment.windows_cng_custody_bridge import (
        WindowsCNGCustodyBridgeError,
        _creation_ticket,
    )

    try:
        return cast(bytes, _creation_ticket(raw))
    except WindowsCNGCustodyBridgeError as exc:
        raise LPPIAuthorityCustodyError("INVALID_LPPI_AUTHORITY_CREATION_TICKET") from exc


@dataclass(frozen=True, init=False)
class LPPIAuthorityKeyCustodyEvidenceV1:
    """Canonical public artifact; its constructor never grants production trust."""

    canonical_bytes: bytes

    def __init__(self) -> None:
        raise TypeError("use from_mapping() or from_canonical_bytes()")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> LPPIAuthorityKeyCustodyEvidenceV1:
        return cls.from_canonical_bytes(canonical_json_bytes(dict(value)))

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> LPPIAuthorityKeyCustodyEvidenceV1:
        document = _parse(raw)
        _exact(document, CUSTODY_FIELDS)
        literals = {
            "schema_version": "LPPIAuthorityKeyCustodyEvidenceV1",
            "environment": "PRODUCTION",
            "custody_profile": CUSTODY_PROFILE,
            "cng_provider_name": PROVIDER_NAME,
            "cng_key_name": KEY_NAME,
            "lppi_authority_public_key_algorithm_profile": ALGORITHM_PROFILE,
        }
        if any(
            type(document[field]) is not str or document[field] != literal
            for field, literal in literals.items()
        ):
            raise LPPIAuthorityCustodyError("INVALID_LPPI_AUTHORITY_CUSTODY_PROFILE")
        for field in ("cng_key_unique_name", "provisioning_subject_id", "enrollment_reference"):
            value = document[field]
            if (
                type(value) is not str
                or not 1 <= len(value) <= 256
                or any(ord(character) < 32 for character in value)
            ):
                raise LPPIAuthorityCustodyError("INVALID_LPPI_AUTHORITY_IDENTITY")
        for field, prefix in (
            ("provisioning_subject_id", "psub_"),
            ("enrollment_reference", "penr_"),
        ):
            if not re.fullmatch(
                prefix + r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
                document[field],
            ):
                raise LPPIAuthorityCustodyError("INVALID_LPPI_AUTHORITY_AUTHORIZATION_IDENTITY")
        if re.fullmatch(r"[A-Za-z0-9._\\:-]{1,256}", document["cng_key_unique_name"]) is None:
            raise LPPIAuthorityCustodyError("INVALID_LPPI_AUTHORITY_IDENTITY")
        textual = set(literals) | {
            "cng_key_unique_name",
            "provisioning_subject_id",
            "enrollment_reference",
        }
        names = {"subject_tpm_name", "retained_ak_tpm_name", "retained_ak_qualified_name"}
        variable = {
            "subject_tpmt_public_hex",
            "subject_creation_ticket_hex",
            "retained_ak_tpmt_public_hex",
            "certify_creation_attest_hex",
            "certify_creation_signature_hex",
        }
        for field in CUSTODY_FIELDS - textual - {"release_policy_generation"}:
            _hex(document[field], size=34 if field in names else None if field in variable else 32)
        if (
            type(document["release_policy_generation"]) is not int
            or not 1 <= document["release_policy_generation"] <= 9007199254740991
        ):
            raise LPPIAuthorityCustodyError("INVALID_LPPI_AUTHORITY_CUSTODY_GENERATION")
        subject = parse_lppi_authority_public(_hex(document["subject_tpmt_public_hex"]))
        ak = parse_production_ecc_public(_hex(document["retained_ak_tpmt_public_hex"]), role="ak")
        fingerprint = hashlib.sha256(subject.raw).hexdigest()
        attest = _hex(document["certify_creation_attest_hex"])
        parsed = parse_production_creation_attestation(attest)
        _ticket(_hex(document["subject_creation_ticket_hex"]))
        qualifier = lppi_authority_custody_qualifying_data(
            {field: document[field] for field in BINDING_FIELDS}
        )
        if (
            document["lppi_authority_public_key_fingerprint_sha256"] != fingerprint
            or document["lppi_authority_tpmt_public_sha256"] != fingerprint
            or subject.name.hex() != document["subject_tpm_name"]
            or ak.name.hex() != document["retained_ak_tpm_name"]
            or hashlib.sha256(ak.raw).hexdigest() != document["ak_public_digest"]
            or subject.sec1 == ak.sec1
            or hashlib.sha256(attest).hexdigest() != document["tpm_creation_attestation_sha256"]
            or parsed.qualified_signer.hex() != document["retained_ak_qualified_name"]
            or parsed.object_name != subject.name
            or parsed.creation_hash != _hex(document["subject_creation_hash"], size=32)
            or parsed.extra_data != qualifier
        ):
            raise LPPIAuthorityCustodyError("LPPI_AUTHORITY_CUSTODY_PUBLIC_BINDING_MISMATCH")
        instance = object.__new__(cls)
        object.__setattr__(instance, "canonical_bytes", raw)
        return instance

    @property
    def document(self) -> dict[str, Any]:
        return cast(dict[str, Any], _parse(self.canonical_bytes))

    @property
    def digest_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


def _trusted_inputs(accepted: object, key: object) -> tuple[Any, Any]:
    # Lazy imports keep the capability factories and native collector acyclic.
    from deployment.windows_lppi_authority_key import require_verified_production_lppi_authority_key

    from .lppi_package_acceptance import require_verified_lppi_package_acceptance

    return (
        require_verified_lppi_package_acceptance(accepted),
        require_verified_production_lppi_authority_key(key),
    )


def lppi_authority_custody_binding(
    *,
    accepted: object,
    key: object,
    subject_tpmt_public: bytes,
    subject_name: bytes,
    creation_hash: bytes,
    creation_ticket: bytes,
    retained_ak_public: bytes,
    retained_ak_name: bytes,
    retained_ak_qualified_name: bytes,
) -> dict[str, Any]:
    trusted, qualified = _trusted_inputs(accepted, key)
    package, request = trusted.package, trusted.request
    payload, req = package.payload, request.document
    subject = parse_lppi_authority_public(subject_tpmt_public)
    if subject.sec1 != qualified.public_key_bytes or subject.name != subject_name:
        raise LPPIAuthorityCustodyError("EXACT_LPPI_AUTHORITY_CNG_TPM_IDENTITY_REQUIRED")
    fingerprint = hashlib.sha256(subject.raw).hexdigest()
    return {
        "schema_version": "LPPIAuthorityKeyCustodyEvidenceV1",
        "environment": "PRODUCTION",
        "custody_profile": CUSTODY_PROFILE,
        "cng_provider_name": qualified.provider_name,
        "cng_key_name": qualified.key_name,
        "cng_key_unique_name": qualified.key_unique_name,
        "cng_public_blob_sha256": qualified.public_blob_sha256,
        "pdsa_package_digest_sha256": package.package_digest_sha256,
        "pre_enrollment_request_digest_sha256": request.digest_sha256,
        **{
            field: req[field]
            for field in (
                "verified_tpm_public_projection_id",
                "verified_tpm_exchange_reference",
                "ek_public_digest",
                "ak_public_digest",
                "pre_enrollment_public_key_fingerprint_sha256",
                "release_policy_digest_sha256",
                "release_policy_generation",
            )
        },
        "provisioning_subject_id": payload["provisioning_subject_id"],
        "enrollment_reference": payload["enrollment_reference"],
        "lppi_authority_public_key_algorithm_profile": ALGORITHM_PROFILE,
        "lppi_authority_public_key_fingerprint_sha256": fingerprint,
        "lppi_authority_tpmt_public_sha256": fingerprint,
        "subject_tpmt_public_hex": subject_tpmt_public.hex(),
        "subject_tpm_name": subject_name.hex(),
        "subject_creation_hash": creation_hash.hex(),
        "subject_creation_ticket_hex": _ticket(creation_ticket).hex(),
        "retained_ak_tpmt_public_hex": retained_ak_public.hex(),
        "retained_ak_tpm_name": retained_ak_name.hex(),
        "retained_ak_qualified_name": retained_ak_qualified_name.hex(),
    }


class VerifiedLPPIAuthorityKeyCustody:
    """Private-registry capability retaining the exact accepted and native key."""

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("verified LPPI custody comes only from verification")

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("verified LPPI custody is immutable")

    @property
    def evidence(self) -> LPPIAuthorityKeyCustodyEvidenceV1:
        return LPPIAuthorityKeyCustodyEvidenceV1.from_canonical_bytes(_snapshot(self)["raw"])

    @property
    def canonical_bytes(self) -> bytes:
        return cast(bytes, _snapshot(self)["raw"])

    @property
    def document(self) -> dict[str, Any]:
        return self.evidence.document

    @property
    def digest_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @property
    def subject_tpmt_public(self) -> bytes:
        return cast(bytes, _hex(self.document["subject_tpmt_public_hex"]))

    @property
    def public_key_bytes(self) -> bytes:
        return cast(bytes, parse_lppi_authority_public(self.subject_tpmt_public).sec1)

    @property
    def tpm_name(self) -> bytes:
        return cast(bytes, _hex(self.document["subject_tpm_name"], size=34))


_ISSUED: WeakKeyDictionary[VerifiedLPPIAuthorityKeyCustody, Mapping[str, Any]] = WeakKeyDictionary()


def _verify(raw: bytes, *, accepted: object, key: object) -> None:
    from deployment import windows_cng_custody_bridge as bridge

    trusted, qualified = _trusted_inputs(accepted, key)
    evidence = LPPIAuthorityKeyCustodyEvidenceV1.from_canonical_bytes(raw)
    document = evidence.document
    native = qualified._native
    if (
        bridge._creation_hash(native.property(qualified._key, "PCP_KEY_CREATIONHASH")).hex()
        != document["subject_creation_hash"]
        or bridge._creation_ticket(native.property(qualified._key, "PCP_KEY_CREATIONTICKET")).hex()
        != document["subject_creation_ticket_hex"]
    ):
        raise LPPIAuthorityCustodyError("LPPI_AUTHORITY_NATIVE_CREATION_BINDING_MISMATCH")
    expected = lppi_authority_custody_binding(
        accepted=trusted,
        key=qualified,
        subject_tpmt_public=_hex(document["subject_tpmt_public_hex"]),
        subject_name=_hex(document["subject_tpm_name"], size=34),
        creation_hash=_hex(document["subject_creation_hash"], size=32),
        creation_ticket=_hex(document["subject_creation_ticket_hex"]),
        retained_ak_public=_hex(document["retained_ak_tpmt_public_hex"]),
        retained_ak_name=_hex(document["retained_ak_tpm_name"], size=34),
        retained_ak_qualified_name=_hex(document["retained_ak_qualified_name"], size=34),
    )
    projection = trusted.projection.document
    if any(document[field] != value for field, value in expected.items()) or (
        document["retained_ak_tpmt_public_hex"] != projection["ak"]["public_area"]["hex"]
        or document["retained_ak_tpm_name"] != projection["ak"]["name"]
        or parse_lppi_authority_public(_hex(document["subject_tpmt_public_hex"])).sec1
        == trusted.key.public_key_bytes
    ):
        raise LPPIAuthorityCustodyError("LPPI_AUTHORITY_CUSTODY_ACCEPTANCE_OR_IDENTITY_MISMATCH")
    _verify_creation(
        attest=_hex(document["certify_creation_attest_hex"]),
        signature=_hex(document["certify_creation_signature_hex"]),
        ak=parse_production_ecc_public(_hex(document["retained_ak_tpmt_public_hex"]), role="ak"),
        expected_name=_hex(document["subject_tpm_name"], size=34),
        expected_creation_hash=_hex(document["subject_creation_hash"], size=32),
        expected_qualifier=lppi_authority_custody_qualifying_data(expected),
    )


def verify_lppi_authority_key_custody(
    evidence: bytes | LPPIAuthorityKeyCustodyEvidenceV1, *, accepted: object, key: object
) -> VerifiedLPPIAuthorityKeyCustody:
    raw = (
        evidence.canonical_bytes
        if type(evidence) is LPPIAuthorityKeyCustodyEvidenceV1
        else evidence
    )
    if type(raw) is not bytes:
        raise LPPIAuthorityCustodyError("INVALID_LPPI_AUTHORITY_CUSTODY_ARTIFACT")
    _verify(cast(bytes, raw), accepted=accepted, key=key)
    result = object.__new__(VerifiedLPPIAuthorityKeyCustody)
    _ISSUED[result] = MappingProxyType({"raw": raw, "accepted": accepted, "key": key})
    return result


def _snapshot(value: object) -> Mapping[str, Any]:
    if type(value) is not VerifiedLPPIAuthorityKeyCustody:
        raise LPPIAuthorityCustodyError("VERIFIED_LPPI_AUTHORITY_CUSTODY_REQUIRED")
    result = _ISSUED.get(value)
    if result is None:
        raise LPPIAuthorityCustodyError("VERIFIED_LPPI_AUTHORITY_CUSTODY_REQUIRED")
    _verify(result["raw"], accepted=result["accepted"], key=result["key"])
    return result


def require_verified_lppi_authority_key_custody(
    value: object, *, accepted: object | None = None, key: object | None = None
) -> VerifiedLPPIAuthorityKeyCustody:
    snapshot = _snapshot(value)
    if (accepted is not None and accepted is not snapshot["accepted"]) or (
        key is not None and key is not snapshot["key"]
    ):
        raise LPPIAuthorityCustodyError("LPPI_AUTHORITY_CUSTODY_CAPABILITY_BINDING_MISMATCH")
    return cast(VerifiedLPPIAuthorityKeyCustody, value)
