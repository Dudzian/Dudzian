"""Target-Windows package acceptance with live retained-key possession and custody.

Public package verification and issuer pre-enrollment capabilities have different
purposes. This boundary never manufactures an issuer store capability. PDSA signs
the exact retained exchange; the client independently verifies its public proofs
and observes the exact nonmigratable AK and request key on the current TPM.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, cast
from weakref import WeakKeyDictionary

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils

from deployment import windows_cng_custody_bridge as bridge
from deployment.windows_cng_pre_enrollment import (
    KEY_NAME,
    WindowsCNGPreEnrollmentKey,
    require_verified_production_cng_key,
)
from deployment.windows_stage9_production_trust import (
    ProductionTrustContext,
    require_current_production_trust_context,
)

from .canonical import parse_canonical
from .device_enrollment import TPMPublicProjectionV1
from .external_provisioning import (
    ProductionProvisioningPackageVerifier,
    VerifiedProvisioningPackage,
)
from .pdsa_enrollment_challenge import verify_signed_production_pdsa_challenge
from .pre_enrollment import (
    P256_ORDER,
    PreEnrollmentRequestV1,
    public_key_fingerprint,
    validate_public_key,
)
from .production_tpm_custody import (
    ProductionPreEnrollmentKeyCustodyEvidenceV1,
    _activation_and_request,
    _production_projection,
    _verify_creation,
    parse_production_creation_attestation,
    parse_production_ecc_public,
    pre_enrollment_custody_qualifying_data,
    production_exchange_qualifying_data,
    production_k_psa_pop_digest,
    verify_production_tpm_endorsement,
)
from .tpm_attestation import TPMEnrollmentChallengeResponseV1, TPMEnrollmentChallengeV1

PACKAGE_ACCEPTANCE_DOMAIN = b"CryptoHunter.Stage9.LPPI.PackageAcceptancePoP.v1\x00"
PACKAGE_ACCEPTANCE_CUSTODY_DOMAIN = b"CryptoHunter.Stage9.LPPI.PackageAcceptanceCustody.v1\x00"
PACKAGE_ACCEPTANCE_FIELDS = frozenset(
    {
        "pdsa_package_digest_sha256",
        "pre_enrollment_request_digest_sha256",
        "pre_enrollment_public_key_fingerprint_sha256",
        "verified_tpm_public_projection_id",
        "verified_tpm_exchange_reference",
        "nonce_hex",
    }
)
_HEX32 = re.compile(r"[0-9a-f]{64}\Z")


class LPPIPackageAcceptanceError(ValueError):
    """A target-host acceptance or current provenance check failed closed."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_package_acceptance_challenge(raw: bytes) -> dict[str, str]:
    if type(raw) is not bytes or len(raw) > 4096:
        raise LPPIPackageAcceptanceError("INVALID_PACKAGE_ACCEPTANCE_CHALLENGE")
    value = parse_canonical(raw)
    if (
        type(value) is not dict
        or set(value) != PACKAGE_ACCEPTANCE_FIELDS
        or any(type(item) is not str or _HEX32.fullmatch(item) is None for item in value.values())
    ):
        raise LPPIPackageAcceptanceError("INVALID_PACKAGE_ACCEPTANCE_CHALLENGE")
    return cast(dict[str, str], value)


def validate_continuity_payload(raw: bytes) -> dict[str, Any]:
    """Use the one frozen binding parser for the retained pre-key operation."""
    from .lppi_authority_key import LPPIAuthorityKeyBindingV1

    return LPPIAuthorityKeyBindingV1.from_canonical_bytes(raw).document


def _verify_low_s(public: bytes, signature: bytes, signing_bytes: bytes) -> None:
    try:
        if type(signature) is not bytes or not 8 <= len(signature) <= 72:
            raise ValueError("invalid signature")
        r, s = utils.decode_dss_signature(signature)
        if (
            utils.encode_dss_signature(r, s) != signature
            or not 1 <= r < P256_ORDER
            or not 1 <= s <= P256_ORDER // 2
        ):
            raise ValueError("noncanonical signature")
        validate_public_key(public).verify(signature, signing_bytes, ec.ECDSA(hashes.SHA256()))
    except (ValueError, InvalidSignature) as exc:
        raise LPPIPackageAcceptanceError("INVALID_PACKAGE_ACCEPTANCE_POP") from exc


@dataclass(frozen=True, slots=True)
class _AcceptanceSnapshot:
    context: ProductionTrustContext
    key: WindowsCNGPreEnrollmentKey
    package_raw: bytes
    request_raw: bytes
    request_signature: bytes
    challenge_raw: bytes
    exchange_raw: tuple[bytes, bytes, bytes, bytes]
    endorsement_raw: bytes
    custody_raw: bytes
    projection_raw: bytes
    ak_key_name: str
    pop_challenge_raw: bytes
    pop_signature: bytes
    live_attestation_raw: bytes
    live_signature: bytes


class VerifiedLPPIClientPackageAcceptance:
    """Verifier-issued target-host authority to begin the successor lifecycle."""

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("client package acceptance comes only from live production verification")

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("client package acceptance is immutable")

    @property
    def package(self) -> VerifiedProvisioningPackage:
        source = _snapshot(self)
        return ProductionProvisioningPackageVerifier(source.context).verify(
            source.package_raw,
            expected_device_key=public_key_fingerprint(source.key.public_key_bytes),
            now=_utc_now(),
        )

    @property
    def request(self) -> PreEnrollmentRequestV1:
        return PreEnrollmentRequestV1.from_canonical_bytes(_snapshot(self).request_raw)

    @property
    def projection(self) -> TPMPublicProjectionV1:
        return cast(
            TPMPublicProjectionV1,
            TPMPublicProjectionV1.verify(parse_canonical(_snapshot(self).projection_raw)),
        )

    @property
    def context(self) -> ProductionTrustContext:
        return _snapshot(self).context

    @property
    def key(self) -> WindowsCNGPreEnrollmentKey:
        return _snapshot(self).key

    @property
    def ak_key_name(self) -> str:
        return _snapshot(self).ak_key_name

    @property
    def pre_enrollment_public_key_bytes(self) -> bytes:
        return cast(bytes, _snapshot(self).key.public_key_bytes)

    @property
    def pre_enrollment_key_unique_name(self) -> str:
        return cast(str, _snapshot(self).key.public_evidence["cng_key_unique_name"])

    @property
    def pop_challenge_raw(self) -> bytes:
        return _snapshot(self).pop_challenge_raw

    @property
    def pop_signature(self) -> bytes:
        return _snapshot(self).pop_signature

    @property
    def live_attestation_raw(self) -> bytes:
        return _snapshot(self).live_attestation_raw

    @property
    def live_signature(self) -> bytes:
        return _snapshot(self).live_signature


_ACCEPTED: WeakKeyDictionary[VerifiedLPPIClientPackageAcceptance, _AcceptanceSnapshot] = (
    WeakKeyDictionary()
)


def _snapshot(value: object) -> _AcceptanceSnapshot:
    if type(value) is not VerifiedLPPIClientPackageAcceptance:
        raise LPPIPackageAcceptanceError("VERIFIED_LPPI_CLIENT_PACKAGE_ACCEPTANCE_REQUIRED")
    result = _ACCEPTED.get(value)
    if result is None:
        raise LPPIPackageAcceptanceError("VERIFIED_LPPI_CLIENT_PACKAGE_ACCEPTANCE_REQUIRED")
    return result


def _validate_sources(
    source: _AcceptanceSnapshot,
) -> tuple[VerifiedProvisioningPackage, PreEnrollmentRequestV1, TPMPublicProjectionV1]:
    trusted = require_current_production_trust_context(source.context)
    key = require_verified_production_cng_key(source.key)
    request = PreEnrollmentRequestV1.from_canonical_bytes(source.request_raw)
    request.require_production_trust_binding(trusted)
    request.verify_signature(source.request_signature)
    req = request.document
    if req["pre_enrollment_public_key_canonical_bytes"] != key.public_key_bytes.hex():
        raise LPPIPackageAcceptanceError("REJECT_PACKAGE_TARGET_MISMATCH")
    package = ProductionProvisioningPackageVerifier(trusted).verify(
        source.package_raw,
        expected_device_key=public_key_fingerprint(key.public_key_bytes),
        now=_utc_now(),
    )
    signed = verify_signed_production_pdsa_challenge(source.challenge_raw, trusted)
    ch = signed.document["payload"]
    expected_request = {
        "pdsa_challenge_id": ch["challenge_id"],
        "pdsa_challenge_digest_sha256": signed.digest_sha256,
        "pdsa_challenge_nonce_digest_sha256": signed.nonce_digest_sha256,
        "release_policy_digest_sha256": ch["release_policy_digest_sha256"],
        "release_policy_generation": ch["release_policy_generation"],
        "pdsa_trust_domain": ch["pdsa_trust_domain"],
    }
    if any(req[field] != expected for field, expected in expected_request.items()):
        raise LPPIPackageAcceptanceError("REJECT_PACKAGE_TARGET_MISMATCH")
    activation_raw, tpm_request_raw, tpm_challenge_raw, response_raw = source.exchange_raw
    request.compare_tpm_exchange_bindings(
        activation_request_raw=activation_raw,
        request_raw=tpm_request_raw,
        challenge_raw=tpm_challenge_raw,
        response_raw=response_raw,
    )
    endorsement = verify_production_tpm_endorsement(source.endorsement_raw, context=trusted)
    activation, _tpm_request, projection = _activation_and_request(
        activation_raw, tpm_request_raw, trusted, endorsement
    )
    if source.projection_raw and source.projection_raw != projection.canonical_bytes:
        raise LPPIPackageAcceptanceError("REJECT_PACKAGE_TARGET_MISMATCH")
    _ek, ak, k_psa = _production_projection(projection, endorsement)
    tpm_challenge = TPMEnrollmentChallengeV1.from_canonical_bytes(tpm_challenge_raw)
    response = TPMEnrollmentChallengeResponseV1.from_canonical_bytes(response_raw).document
    nonce = bytes.fromhex(tpm_challenge.document["issuer_nonce_hex"])
    pdsa_digest = bytes.fromhex(signed.digest_sha256)
    _verify_creation(
        attest=bytes.fromhex(response["certify_creation_attest_hex"]),
        signature=bytes.fromhex(response["certify_creation_signature_hex"]),
        ak=ak,
        expected_name=k_psa.name,
        expected_creation_hash=bytes.fromhex(projection.document["k_psa"]["creation_hash"]),
        expected_qualifier=production_exchange_qualifying_data(
            nonce, hashlib.sha256(activation.canonical_bytes).digest(), pdsa_digest
        ),
    )
    # ActivateCredential's issuer secret is never present on the client. The
    # authenticated PDSA package binds the exact response accepted by its issuer.
    pop = bytes.fromhex(response["k_psa_pop_signature_der_hex"])
    try:
        r, s = utils.decode_dss_signature(pop)
        if utils.encode_dss_signature(r, s) != pop or not (
            1 <= r < P256_ORDER and 1 <= s < P256_ORDER
        ):
            raise ValueError("noncanonical signature")
        validate_public_key(k_psa.sec1).verify(
            pop,
            production_k_psa_pop_digest(nonce, pdsa_digest),
            ec.ECDSA(utils.Prehashed(hashes.SHA256())),
        )
    except (ValueError, InvalidSignature) as exc:
        raise LPPIPackageAcceptanceError("INVALID_RETAINED_PRODUCTION_K_PSA_POP") from exc
    custody = ProductionPreEnrollmentKeyCustodyEvidenceV1.from_canonical_bytes(source.custody_raw)
    evidence = custody.document
    expected_custody = {
        field: req[field]
        for field in (
            "verified_tpm_public_projection_id",
            "verified_tpm_exchange_reference",
            "ek_public_digest",
            "ak_public_digest",
            "pdsa_challenge_digest_sha256",
            "release_policy_digest_sha256",
            "release_policy_generation",
            "pre_enrollment_public_key_canonical_bytes",
            "pre_enrollment_public_key_fingerprint_sha256",
        )
    }
    expected_custody["pre_enrollment_request_digest_sha256"] = request.digest_sha256
    if any(evidence[field] != expected for field, expected in expected_custody.items()):
        raise LPPIPackageAcceptanceError("REJECT_PACKAGE_TARGET_MISMATCH")
    subject = parse_production_ecc_public(
        bytes.fromhex(evidence["subject_tpmt_public_hex"]), role="pre_enrollment"
    )
    _verify_creation(
        attest=bytes.fromhex(evidence["certify_creation_attest_hex"]),
        signature=bytes.fromhex(evidence["certify_creation_signature_hex"]),
        ak=ak,
        expected_name=subject.name,
        expected_creation_hash=bytes.fromhex(evidence["subject_creation_hash"]),
        expected_qualifier=pre_enrollment_custody_qualifying_data(request),
    )
    expected_package = {
        field: req[field]
        for field in (
            "pdsa_challenge_id",
            "pdsa_challenge_digest_sha256",
            "verified_tpm_exchange_reference",
            "verified_tpm_public_projection_id",
            "pre_enrollment_public_key_algorithm_profile",
            "pre_enrollment_public_key_fingerprint_sha256",
            "release_policy_digest_sha256",
            "release_policy_generation",
            "product_profile",
            "pdsa_trust_domain",
        )
    }
    expected_package.update(
        pre_enrollment_request_digest_sha256=request.digest_sha256,
        target_tpm_ek_public_digest=req["ek_public_digest"],
        target_tpm_ak_public_digest=req["ak_public_digest"],
        expires_at_utc=min(ch["expires_at_utc"], tpm_challenge.document["expires_at_utc"]),
    )
    if any(package.payload[field] != expected for field, expected in expected_package.items()):
        raise LPPIPackageAcceptanceError("REJECT_PACKAGE_TARGET_MISMATCH")
    return package, request, projection


def _challenge(package: VerifiedProvisioningPackage) -> bytes:
    from .canonical import canonical_json_bytes

    fields = {
        field: package.payload[field]
        for field in PACKAGE_ACCEPTANCE_FIELDS - {"pdsa_package_digest_sha256", "nonce_hex"}
    }
    fields["pdsa_package_digest_sha256"] = package.package_digest_sha256
    fields["nonce_hex"] = secrets.token_bytes(32).hex()
    return cast(bytes, canonical_json_bytes(fields))


def _live_custody(
    source: _AcceptanceSnapshot,
    projection: TPMPublicProjectionV1,
    challenge_raw: bytes,
) -> tuple[bytes, bytes]:
    key = require_verified_production_cng_key(source.key)
    if (
        not re.fullmatch(r"[A-Za-z0-9._\\:-]{1,256}", source.ak_key_name)
        or source.ak_key_name == KEY_NAME
    ):
        raise LPPIPackageAcceptanceError("INVALID_RETAINED_AK_KEY_NAME")
    native = key._native
    context = bridge._platform_handle(native, key._provider, provider=True)
    handle = bridge._platform_handle(native, key._key, provider=False)
    creation_hash = bridge._creation_hash(native.property(key._key, "PCP_KEY_CREATIONHASH"))
    ticket = bridge._creation_ticket(native.property(key._key, "PCP_KEY_CREATIONTICKET"))
    evidence = ProductionPreEnrollmentKeyCustodyEvidenceV1.from_canonical_bytes(
        source.custody_raw
    ).document
    if creation_hash.hex() != evidence["subject_creation_hash"]:
        raise LPPIPackageAcceptanceError("LIVE_PRE_ENROLLMENT_CREATION_MISMATCH")
    tbs = bridge._load_tbs_native()
    if type(tbs) is not bridge._TBSCustodyAPI:
        raise LPPIPackageAcceptanceError("EXACT_NATIVE_TBS_BOUNDARY_REQUIRED")
    tbs.require_tpm20()
    ak_handle = bridge._open_retained_ak(native, key._provider, source.ak_key_name)
    try:
        bridge._require_ak_profile(native, ak_handle, source.ak_key_name)
        ak_tpm_handle = bridge._platform_handle(native, ak_handle, provider=False)
        if ak_tpm_handle == handle:
            raise LPPIPackageAcceptanceError("DISTINCT_RETAINED_AK_REQUIRED")
        ak_raw, ak_name, ak_qualified = bridge._read_public(tbs, context, ak_tpm_handle)
        ak = parse_production_ecc_public(ak_raw, role="ak")
        target = projection.document["ak"]
        if (
            ak.name != ak_name
            or ak_raw.hex() != target["public_area"]["hex"]
            or ak_name.hex() != target["name"]
        ):
            raise LPPIPackageAcceptanceError("LIVE_TARGET_AK_MISMATCH")
        raw, name, _qualified = bridge._read_public(tbs, context, handle)
        subject = parse_production_ecc_public(raw, role="pre_enrollment")
        if (
            subject.sec1 != key.public_key_bytes
            or subject.name != name
            or raw.hex() != evidence["subject_tpmt_public_hex"]
            or name.hex() != evidence["subject_tpm_name"]
        ):
            raise LPPIPackageAcceptanceError("LIVE_PRE_ENROLLMENT_IDENTITY_MISMATCH")
        qualifier = hashlib.sha256(
            PACKAGE_ACCEPTANCE_CUSTODY_DOMAIN + hashlib.sha256(challenge_raw).digest()
        ).digest()
        parameters = (
            bridge._tpm2b(qualifier) + bridge._tpm2b(creation_hash) + b"\x00\x18\x00\x0b" + ticket
        )
        result = tbs.submit(
            context,
            bridge._packet(
                bridge.TPM_CC_CERTIFY_CREATION, (ak_tpm_handle, handle), parameters, auth=True
            ),
            auth=True,
        )
        attest, offset = bridge._read_2b(result, 0)
        signature = result[offset:]
        if parse_production_creation_attestation(attest).qualified_signer != ak_qualified:
            raise LPPIPackageAcceptanceError("LIVE_TARGET_AK_QUALIFIED_NAME_MISMATCH")
        _verify_creation(
            attest=attest,
            signature=signature,
            ak=ak,
            expected_name=name,
            expected_creation_hash=creation_hash,
            expected_qualifier=qualifier,
        )
        return attest, signature
    finally:
        native.free(ak_handle)


def accept_production_lppi_package(
    package_raw: bytes,
    *,
    context: object,
    key: object,
    request_raw: bytes,
    request_signature: bytes,
    challenge_raw: bytes,
    activation_request_raw: bytes,
    tpm_request_raw: bytes,
    tpm_challenge_raw: bytes,
    tpm_response_raw: bytes,
    endorsement_raw: bytes,
    custody_evidence_raw: bytes,
    ak_key_name: str,
) -> VerifiedLPPIClientPackageAcceptance:
    """Verify authorization, live target and fresh possession before successor creation."""
    trusted = require_current_production_trust_context(context)
    qualified = require_verified_production_cng_key(key)
    if (
        any(
            type(raw) is not bytes
            for raw in (
                package_raw,
                request_raw,
                request_signature,
                challenge_raw,
                activation_request_raw,
                tpm_request_raw,
                tpm_challenge_raw,
                tpm_response_raw,
                endorsement_raw,
                custody_evidence_raw,
            )
        )
        or type(ak_key_name) is not str
    ):
        raise LPPIPackageAcceptanceError("INVALID_LPPI_PACKAGE_ACCEPTANCE_INPUT")
    source = _AcceptanceSnapshot(
        trusted,
        qualified,
        package_raw,
        request_raw,
        request_signature,
        challenge_raw,
        (activation_request_raw, tpm_request_raw, tpm_challenge_raw, tpm_response_raw),
        endorsement_raw,
        custody_evidence_raw,
        b"",
        ak_key_name,
        b"",
        b"",
        b"",
        b"",
    )
    package, _request, projection = _validate_sources(source)
    pop_raw = _challenge(package)
    pop = qualified.sign_package_acceptance(pop_raw, production_trust_context=trusted)
    _verify_low_s(
        qualified.public_key_bytes,
        pop,
        PACKAGE_ACCEPTANCE_DOMAIN + hashlib.sha256(pop_raw).digest(),
    )
    attest, signature = _live_custody(source, projection, pop_raw)
    value = object.__new__(VerifiedLPPIClientPackageAcceptance)
    _ACCEPTED[value] = _AcceptanceSnapshot(
        trusted,
        qualified,
        package_raw,
        request_raw,
        request_signature,
        challenge_raw,
        source.exchange_raw,
        endorsement_raw,
        custody_evidence_raw,
        projection.canonical_bytes,
        ak_key_name,
        pop_raw,
        pop,
        attest,
        signature,
    )
    return value


def require_verified_lppi_package_acceptance(value: object) -> VerifiedLPPIClientPackageAcceptance:
    """Recheck current authorization, native identity and fresh possession on every use."""
    source = _snapshot(value)
    package, _request, projection = _validate_sources(source)
    challenge = validate_package_acceptance_challenge(source.pop_challenge_raw)
    expected = validate_package_acceptance_challenge(_challenge(package))
    if any(
        challenge[field] != expected[field] for field in PACKAGE_ACCEPTANCE_FIELDS - {"nonce_hex"}
    ):
        raise LPPIPackageAcceptanceError("PACKAGE_ACCEPTANCE_POP_BINDING_MISMATCH")
    _verify_low_s(
        source.key.public_key_bytes,
        source.pop_signature,
        PACKAGE_ACCEPTANCE_DOMAIN + hashlib.sha256(source.pop_challenge_raw).digest(),
    )
    fresh = _challenge(package)
    signature = source.key.sign_package_acceptance(fresh, production_trust_context=source.context)
    _verify_low_s(
        source.key.public_key_bytes,
        signature,
        PACKAGE_ACCEPTANCE_DOMAIN + hashlib.sha256(fresh).digest(),
    )
    _live_custody(source, projection, fresh)
    return cast(VerifiedLPPIClientPackageAcceptance, value)
