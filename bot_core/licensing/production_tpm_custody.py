"""Production TPM verification using PDSA curated hardware EK endorsements.

The client trusts the genuine Production Trust PDSA quorum's independent
manufacturer-chain approval. A signed chain digest is an approval reference,
not a claim that this client independently validated an OEM certificate chain.
Local CNG qualification and request PoP remain distinct acceptance requirements.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterator, Mapping, TypeVar, cast
from weakref import WeakKeyDictionary

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils

from bot_core.licensing.activation_request import ActivationRequestV1
from bot_core.licensing.canonical import canonical_json_bytes, parse_canonical
from bot_core.licensing.device_enrollment import TPMPublicProjectionV1
from bot_core.licensing.pre_enrollment import (
    P256_ORDER,
    PDSA_TRUST_DOMAIN,
    PreEnrollmentRequestV1,
    public_key_fingerprint,
    validate_public_key,
)
from bot_core.licensing.product_profile import PRODUCT_NAME, PRODUCTION_PRODUCT_PROFILE
from bot_core.licensing.tpm_attestation import (
    EXCHANGE_REFERENCE_DOMAIN,
    TPMEnrollmentChallengeResponseV1,
    TPMEnrollmentChallengeV1,
    TPMEnrollmentRequestV1,
    _validate_activation_projection_binding,
    credential_activation_proof,
    make_credential_ecc,
    test_only_sign_policy_digest,
)
from deployment.production_enrollment_issuer import (
    ProductionEnrollmentIssuerContext,
    require_production_tpm_store,
)
from deployment.windows_stage9_production_trust import (
    PDSA_KEY_SET_DIGEST,
    ProductionTrustContext,
    require_current_production_trust_context as require_verified_production_trust_context,
)

if TYPE_CHECKING:
    from .pdsa_enrollment_challenge import PDSAChallengeStore

ENDORSEMENT_DOMAIN = b"CryptoHunter.Stage9.ProductionTPMEndorsement.v1\x00"
EXCHANGE_CREATION_DOMAIN = b"CryptoHunter.Stage9.ProductionTPMEnrollment.CertifyCreation.v1\x00"
EXCHANGE_POP_DOMAIN = b"CryptoHunter.Stage9.ProductionTPMEnrollment.K_PSA.ProofOfPossession.v1\x00"
CUSTODY_CREATION_DOMAIN = (
    b"CryptoHunter.Stage9.ProductionPreEnrollmentKeyCustody.CertifyCreation.v1\x00"
)
ENDORSEMENT_PROFILE = "PDSA_CURATED_MANUFACTURER_VERIFIED_TPM2_ECC_EK_V1"
CUSTODY_PROFILE = "WINDOWS_PLATFORM_CRYPTO_PROVIDER_TPM_ECDSA_P256_PRE_ENROLLMENT_V1"
PROJECTION_PROFILE = "WindowsTPM2-PCP-Stage9-PRODUCTION-v1"
PROJECTION_SOURCE_PROFILE = "Stage9-PCP-persistent-production-v1"
K_PSA_PROFILE = "Stage9.K_PSA.ECC_P256_SHA256.PRODUCTION"
_HEX = re.compile(r"[0-9a-f]+\Z")
_ENDORSEMENT_ID = re.compile(
    r"ptpm_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"
)
ENDORSEMENT_FIELDS = frozenset(
    {
        "schema_version",
        "environment",
        "product",
        "product_profile",
        "pdsa_trust_domain",
        "endorsement_profile",
        "endorsement_id",
        "ek_public_digest",
        "ek_name",
        "ek_certificate_digest_sha256",
        "manufacturer_chain_digest_sha256",
        "manufacturer_chain_verification_record_digest_sha256",
        "hardware_origin_verification",
        "release_policy_digest_sha256",
        "release_policy_generation",
        "issued_at_utc",
        "expires_at_utc",
        "signature_algorithm_profile",
        "signer_key_ids",
    }
)
CUSTODY_FIELDS = frozenset(
    {
        "schema_version",
        "environment",
        "custody_profile",
        "provider_name",
        "pre_enrollment_request_digest_sha256",
        "verified_tpm_public_projection_id",
        "verified_tpm_exchange_reference",
        "ek_public_digest",
        "ak_public_digest",
        "pdsa_challenge_digest_sha256",
        "release_policy_digest_sha256",
        "release_policy_generation",
        "pre_enrollment_public_key_canonical_bytes",
        "pre_enrollment_public_key_fingerprint_sha256",
        "subject_tpmt_public_hex",
        "subject_tpm_name",
        "subject_creation_hash",
        "certify_creation_attest_hex",
        "certify_creation_signature_hex",
        "creation_attestation_digest_sha256",
    }
)


class ProductionTPMCustodyError(ValueError):
    """A production TPM proof or retained binding could not be established."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _hex(value: object, *, size: int | None = None, maximum: int = 16384) -> bytes:
    if type(value) is not str or not value or _HEX.fullmatch(value) is None or len(value) % 2:
        raise ProductionTPMCustodyError("INVALID_PRODUCTION_TPM_HEX")
    raw = bytes.fromhex(value)
    if len(raw) > maximum or (size is not None and len(raw) != size):
        raise ProductionTPMCustodyError("INVALID_PRODUCTION_TPM_HEX")
    return raw


def _time(value: object) -> datetime:
    if type(value) is not str or len(value) != 20:
        raise ProductionTPMCustodyError("INVALID_PRODUCTION_TPM_TIMESTAMP")
    try:
        result = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ProductionTPMCustodyError("INVALID_PRODUCTION_TPM_TIMESTAMP") from exc
    if result.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ProductionTPMCustodyError("INVALID_PRODUCTION_TPM_TIMESTAMP")
    return result


def _parse(raw: bytes, *, maximum: int = 131072) -> dict[str, Any]:
    if type(raw) is not bytes or len(raw) > maximum:
        raise ProductionTPMCustodyError("INVALID_PRODUCTION_TPM_ARTIFACT")
    try:
        result = parse_canonical(raw)
    except (ValueError, TypeError) as exc:
        raise ProductionTPMCustodyError("NONCANONICAL_PRODUCTION_TPM_ARTIFACT") from exc
    if type(result) is not dict:
        raise ProductionTPMCustodyError("INVALID_PRODUCTION_TPM_ARTIFACT")
    return result


def _exact(value: object, fields: frozenset[str] | set[str]) -> None:
    if type(value) is not dict or set(value) != fields:
        raise ProductionTPMCustodyError("PRODUCTION_TPM_SCHEMA_MISMATCH")


class _Reader:
    def __init__(self, raw: bytes) -> None:
        if type(raw) is not bytes or not raw or len(raw) > 16384:
            raise ProductionTPMCustodyError("MALFORMED_PRODUCTION_TPM_STRUCTURE")
        self.raw, self.position = raw, 0

    def take(self, count: int) -> bytes:
        end = self.position + count
        if end > len(self.raw):
            raise ProductionTPMCustodyError("MALFORMED_PRODUCTION_TPM_STRUCTURE")
        result, self.position = self.raw[self.position : end], end
        return result

    def number(self, count: int) -> int:
        return int.from_bytes(self.take(count), "big")

    def blob(self) -> bytes:
        return self.take(self.number(2))

    def finish(self) -> None:
        if self.position != len(self.raw):
            raise ProductionTPMCustodyError("MALFORMED_PRODUCTION_TPM_STRUCTURE")


@dataclass(frozen=True, slots=True)
class ParsedProductionTPMPublic:
    raw: bytes
    name: bytes
    sec1: bytes
    attributes: int
    auth_policy: bytes


def parse_production_ecc_public(raw: bytes, *, role: str) -> ParsedProductionTPMPublic:
    """Parse all bytes; enforce nonmigratable TPM origin and role-specific attributes."""
    reader = _Reader(raw)
    key_type, name_alg, attributes = reader.number(2), reader.number(2), reader.number(4)
    policy = reader.blob()
    symmetric = reader.number(2)
    if role == "ek":
        if symmetric != 0x0006 or (reader.number(2), reader.number(2)) != (128, 0x0043):
            raise ProductionTPMCustodyError("PRODUCTION_TPM_PUBLIC_PROFILE_REJECTED")
    elif symmetric != 0x0010:
        raise ProductionTPMCustodyError("PRODUCTION_TPM_PUBLIC_PROFILE_REJECTED")
    scheme = reader.number(2)
    if scheme == 0x0018:
        if reader.number(2) != 0x000B:
            raise ProductionTPMCustodyError("PRODUCTION_TPM_PUBLIC_PROFILE_REJECTED")
    elif scheme != 0x0010 or role in {"ak", "k_psa"}:
        raise ProductionTPMCustodyError("PRODUCTION_TPM_PUBLIC_PROFILE_REJECTED")
    curve, kdf = reader.number(2), reader.number(2)
    x, y = reader.blob(), reader.blob()
    reader.finish()
    if (key_type, name_alg, curve, kdf, len(x), len(y)) != (0x23, 0x0B, 3, 0x10, 32, 32):
        raise ProductionTPMCustodyError("PRODUCTION_TPM_PUBLIC_PROFILE_REJECTED")
    # fixedTPM + fixedParent + sensitiveDataOrigin cannot be asserted by an imported private key.
    expected_attributes = {
        "pre_enrollment": 0x40072,
        "ak": 0x50072,
        "k_psa": 0x400B2,
        "ek": 0x300B2,
    }
    if role not in expected_attributes or attributes not in {
        expected_attributes[role],
        expected_attributes[role] | 0x400,
    }:
        raise ProductionTPMCustodyError("PRODUCTION_TPM_PUBLIC_PROFILE_REJECTED")
    if role in {"pre_enrollment", "ak"} and policy:
        raise ProductionTPMCustodyError("PRODUCTION_TPM_PUBLIC_PROFILE_REJECTED")
    if role == "k_psa" and (
        len(policy) != 32 or policy in {bytes(32), test_only_sign_policy_digest()}
    ):
        raise ProductionTPMCustodyError("TEST_ONLY_OR_INVALID_TPM_POLICY")
    if role == "ek" and (
        policy != bytes.fromhex("837197674484b3f81a90cc8d46a5d724fd52d76e06520b64f2a1da1b331469aa")
        or scheme != 0x0010
    ):
        raise ProductionTPMCustodyError("PRODUCTION_TPM_PUBLIC_PROFILE_REJECTED")
    sec1 = b"\x04" + x + y
    validate_public_key(sec1)
    return ParsedProductionTPMPublic(
        raw, b"\x00\x0b" + hashlib.sha256(raw).digest(), sec1, attributes, policy
    )


@dataclass(frozen=True, slots=True)
class ParsedProductionCreationAttestation:
    qualified_signer: bytes
    extra_data: bytes
    object_name: bytes
    creation_hash: bytes


def parse_production_creation_attestation(raw: bytes) -> ParsedProductionCreationAttestation:
    reader = _Reader(raw)
    if (reader.number(4), reader.number(2)) != (0xFF544347, 0x801A):
        raise ProductionTPMCustodyError("WRONG_PRODUCTION_CREATION_ATTESTATION_TYPE")
    signer, extra = reader.blob(), reader.blob()
    reader.number(8)  # clock
    reader.number(4)  # resetCount
    reader.number(4)  # restartCount
    safe = reader.number(1)
    reader.number(8)  # firmwareVersion
    name, creation_hash = reader.blob(), reader.blob()
    reader.finish()
    if (
        len(signer) != 34
        or signer[:2] != b"\x00\x0b"
        or len(extra) != 32
        or safe not in {0, 1}
        or len(name) != 34
        or name[:2] != b"\x00\x0b"
        or len(creation_hash) != 32
    ):
        raise ProductionTPMCustodyError("MALFORMED_PRODUCTION_CREATION_ATTESTATION")
    # qualifiedSigner is a Qualified Name, not the AK Name. The authenticated,
    # restricted AK signature supplies signer identity; do not guess its parent QN.
    return ParsedProductionCreationAttestation(signer, extra, name, creation_hash)


def _verify_creation(
    *,
    attest: bytes,
    signature: bytes,
    ak: ParsedProductionTPMPublic,
    expected_name: bytes,
    expected_creation_hash: bytes,
    expected_qualifier: bytes,
) -> None:
    parsed = parse_production_creation_attestation(attest)
    if (parsed.object_name, parsed.creation_hash, parsed.extra_data) != (
        expected_name,
        expected_creation_hash,
        expected_qualifier,
    ):
        raise ProductionTPMCustodyError("PRODUCTION_CERTIFY_CREATION_BINDING_MISMATCH")
    reader = _Reader(signature)
    if (reader.number(2), reader.number(2)) != (0x0018, 0x000B):
        raise ProductionTPMCustodyError("WRONG_PRODUCTION_CERTIFY_SIGNATURE_PROFILE")
    r_raw, s_raw = reader.blob(), reader.blob()
    reader.finish()
    r, s = int.from_bytes(r_raw, "big"), int.from_bytes(s_raw, "big")
    if not (
        1 <= len(r_raw) <= 32
        and 1 <= len(s_raw) <= 32
        and 1 <= r < P256_ORDER
        and 1 <= s < P256_ORDER
    ):
        raise ProductionTPMCustodyError("MALFORMED_PRODUCTION_CERTIFY_SIGNATURE")
    try:
        validate_public_key(ak.sec1).verify(
            utils.encode_dss_signature(r, s), attest, ec.ECDSA(hashes.SHA256())
        )
    except InvalidSignature as exc:
        raise ProductionTPMCustodyError("INVALID_PRODUCTION_CERTIFY_SIGNATURE") from exc


def production_exchange_qualifying_data(
    issuer_nonce: bytes, activation_digest: bytes, pdsa_challenge_digest: bytes
) -> bytes:
    if any(
        type(item) is not bytes or len(item) != 32
        for item in (issuer_nonce, activation_digest, pdsa_challenge_digest)
    ):
        raise ProductionTPMCustodyError("INVALID_PRODUCTION_EXCHANGE_BINDING")
    return hashlib.sha256(
        EXCHANGE_CREATION_DOMAIN + issuer_nonce + activation_digest + pdsa_challenge_digest
    ).digest()


def production_k_psa_pop_digest(issuer_nonce: bytes, pdsa_challenge_digest: bytes) -> bytes:
    if any(
        type(item) is not bytes or len(item) != 32 for item in (issuer_nonce, pdsa_challenge_digest)
    ):
        raise ProductionTPMCustodyError("INVALID_PRODUCTION_EXCHANGE_BINDING")
    return hashlib.sha256(EXCHANGE_POP_DOMAIN + issuer_nonce + pdsa_challenge_digest).digest()


def pre_enrollment_custody_qualifying_data(request: PreEnrollmentRequestV1) -> bytes:
    if type(request) is not PreEnrollmentRequestV1:
        raise ProductionTPMCustodyError("EXACT_PRE_ENROLLMENT_REQUEST_REQUIRED")
    return hashlib.sha256(CUSTODY_CREATION_DOMAIN + bytes.fromhex(request.digest_sha256)).digest()


class _VerifiedCapability:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("production TPM capabilities come only from verification")

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("production TPM capabilities are immutable")

    @property
    def canonical_bytes(self) -> bytes:
        return cast(bytes, _snapshot(self)["canonical_bytes"])

    @property
    def document(self) -> dict[str, Any]:
        return _parse(self.canonical_bytes)

    @property
    def digest_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


class VerifiedProductionTPMEndorsement(_VerifiedCapability):
    __slots__ = ()


class VerifiedProductionTPMExchange(_VerifiedCapability):
    __slots__ = ()

    @property
    def exchange_reference(self) -> str:
        return cast(str, _snapshot(self)["exchange_reference"])

    @property
    def projection(self) -> TPMPublicProjectionV1:
        return TPMPublicProjectionV1.from_canonical_bytes(_snapshot(self)["projection_raw"])

    @property
    def retained_bytes(self) -> tuple[bytes, bytes, bytes, bytes]:
        return cast(tuple[bytes, bytes, bytes, bytes], _snapshot(self)["retained_bytes"])


class VerifiedProductionPreEnrollmentKeyCustody(_VerifiedCapability):
    __slots__ = ()


_ISSUED: WeakKeyDictionary[_VerifiedCapability, Mapping[str, Any]] = WeakKeyDictionary()
_CapabilityT = TypeVar("_CapabilityT", bound=_VerifiedCapability)


def _issue_capability(kind: type[_CapabilityT], **values: Any) -> _CapabilityT:
    item = object.__new__(kind)
    _ISSUED[item] = MappingProxyType(values)
    return item


def _snapshot(value: object) -> Mapping[str, Any]:
    if type(value) not in {
        VerifiedProductionTPMEndorsement,
        VerifiedProductionTPMExchange,
        VerifiedProductionPreEnrollmentKeyCustody,
    }:
        raise ProductionTPMCustodyError("VERIFIED_PRODUCTION_TPM_CAPABILITY_REQUIRED")
    result = _ISSUED.get(cast(_VerifiedCapability, value))
    if result is None:
        raise ProductionTPMCustodyError("VERIFIED_PRODUCTION_TPM_CAPABILITY_REQUIRED")
    require_verified_production_trust_context(result["context"])
    return result


def _require_context(
    value: object, kind: type[_VerifiedCapability], context: object
) -> Mapping[str, Any]:
    require_verified_production_trust_context(context)
    snapshot = _snapshot(value)
    if type(value) is not kind or snapshot["context"] is not context:
        raise ProductionTPMCustodyError("PRODUCTION_TPM_CAPABILITY_CONTEXT_MISMATCH")
    return snapshot


def verify_production_tpm_endorsement(
    raw: bytes, *, context: object
) -> VerifiedProductionTPMEndorsement:
    trusted = require_verified_production_trust_context(context)
    envelope = _parse(raw)
    _exact(envelope, {"payload", "payload_digest_sha256", "signatures"})
    payload = envelope["payload"]
    _exact(payload, ENDORSEMENT_FIELDS)
    constants = {
        "schema_version": "ProductionTPMEndorsementV1",
        "environment": "PRODUCTION",
        "product": PRODUCT_NAME,
        "product_profile": PRODUCTION_PRODUCT_PROFILE,
        "pdsa_trust_domain": PDSA_TRUST_DOMAIN,
        "endorsement_profile": ENDORSEMENT_PROFILE,
        "hardware_origin_verification": "MANUFACTURER_CHAIN_AND_EK_CERTIFICATE_BINDING_VERIFIED",
        "signature_algorithm_profile": "PDSA-2-OF-3-ED25519",
        "release_policy_digest_sha256": trusted.release_payload_digest,
    }
    if any(
        type(payload[field]) is not str or payload[field] != expected
        for field, expected in constants.items()
    ):
        raise ProductionTPMCustodyError("PRODUCTION_TPM_ENDORSEMENT_PROFILE_REJECTED")
    if (
        type(payload["release_policy_generation"]) is not int
        or payload["release_policy_generation"] != trusted.release_version
    ):
        raise ProductionTPMCustodyError("PRODUCTION_TPM_ENDORSEMENT_RELEASE_MISMATCH")
    if type(payload["endorsement_id"]) is not str or not _ENDORSEMENT_ID.fullmatch(
        payload["endorsement_id"]
    ):
        raise ProductionTPMCustodyError("INVALID_PRODUCTION_TPM_ENDORSEMENT_ID")
    for field in (
        "ek_public_digest",
        "ek_certificate_digest_sha256",
        "manufacturer_chain_digest_sha256",
        "manufacturer_chain_verification_record_digest_sha256",
        "release_policy_digest_sha256",
    ):
        _hex(payload[field], size=32)
    if _hex(payload["ek_name"], size=34) != b"\x00\x0b" + _hex(
        payload["ek_public_digest"], size=32
    ):
        raise ProductionTPMCustodyError("PRODUCTION_TPM_ENDORSEMENT_EK_NAME_MISMATCH")
    issued, expires, now = (
        _time(payload["issued_at_utc"]),
        _time(payload["expires_at_utc"]),
        _utc_now(),
    )
    if not issued <= now < expires or not 0 < (expires - issued).total_seconds() <= 604800:
        raise ProductionTPMCustodyError("PRODUCTION_TPM_ENDORSEMENT_EXPIRED_OR_INVALID_WINDOW")
    payload_digest = hashlib.sha256(canonical_json_bytes(payload)).digest()
    if _hex(envelope["payload_digest_sha256"], size=32) != payload_digest:
        raise ProductionTPMCustodyError("PRODUCTION_TPM_ENDORSEMENT_DIGEST_MISMATCH")
    signatures, signer_ids = envelope["signatures"], payload["signer_key_ids"]
    if (
        type(signatures) is not list
        or not 2 <= len(signatures) <= 3
        or type(signer_ids) is not list
        or any(type(item) is not str for item in signer_ids)
        or signer_ids != sorted(set(signer_ids))
        or len(signer_ids) != len(signatures)
    ):
        raise ProductionTPMCustodyError("PRODUCTION_TPM_ENDORSEMENT_QUORUM_REQUIRED")
    for key_id, signature in zip(signer_ids, signatures, strict=True):
        _exact(signature, {"algorithm", "key_id", "authority_key_set_digest", "signature_hex"})
        if (
            signature["algorithm"] != "Ed25519"
            or signature["key_id"] != key_id
            or signature["authority_key_set_digest"] != PDSA_KEY_SET_DIGEST
            or key_id not in trusted.pdsa_keys
        ):
            raise ProductionTPMCustodyError("UNAUTHORIZED_PRODUCTION_TPM_ENDORSEMENT_SIGNER")
        try:
            trusted.pdsa_keys[key_id].verify(
                _hex(signature["signature_hex"], size=64), ENDORSEMENT_DOMAIN + payload_digest
            )
        except InvalidSignature as exc:
            raise ProductionTPMCustodyError("INVALID_PRODUCTION_TPM_ENDORSEMENT_SIGNATURE") from exc
    return _issue_capability(VerifiedProductionTPMEndorsement, canonical_bytes=raw, context=trusted)


def _require_endorsement(value: object, context: object) -> VerifiedProductionTPMEndorsement:
    snapshot = _require_context(value, VerifiedProductionTPMEndorsement, context)
    # Reverify expiry, exact signatures and current Production Trust on every consequential use.
    verify_production_tpm_endorsement(snapshot["canonical_bytes"], context=context)
    return cast(VerifiedProductionTPMEndorsement, value)


def _production_projection(
    projection: TPMPublicProjectionV1, endorsement: VerifiedProductionTPMEndorsement
) -> tuple[ParsedProductionTPMPublic, ParsedProductionTPMPublic, ParsedProductionTPMPublic]:
    value, approved = projection.document, endorsement.document["payload"]
    if (
        value["evidence_profile"] != PROJECTION_PROFILE
        or value["source"] != {"substrate": "Windows-TBS", "profile": PROJECTION_SOURCE_PROFILE}
        or value["k_psa"]["algorithm_profile"] != K_PSA_PROFILE
    ):
        raise ProductionTPMCustodyError("TEST_ONLY_OR_UNSUPPORTED_PRODUCTION_TPM_PROJECTION")
    ek, ak, k_psa = tuple(
        parse_production_ecc_public(_hex(value[role]["public_area"]["hex"]), role=role)
        for role in ("ek", "ak", "k_psa")
    )
    if (
        value["ek"]["public_digest"] != approved["ek_public_digest"]
        or ek.name.hex() != approved["ek_name"]
        or value["ek"]["manufacturer_certificate"] != "AVAILABLE"
        or value["ek"]["manufacturer_certificate_digest"]
        != approved["ek_certificate_digest_sha256"]
    ):
        raise ProductionTPMCustodyError("PRODUCTION_TPM_ENDORSEMENT_TARGET_MISMATCH")
    return ek, ak, k_psa


def _activation_and_request(
    activation_raw: bytes,
    request_raw: bytes,
    context: ProductionTrustContext,
    endorsement: VerifiedProductionTPMEndorsement,
) -> tuple[ActivationRequestV1, TPMEnrollmentRequestV1, TPMPublicProjectionV1]:
    activation = ActivationRequestV1.from_mapping(_parse(activation_raw))
    request = cast(TPMEnrollmentRequestV1, TPMEnrollmentRequestV1.from_canonical_bytes(request_raw))
    req, act = request.document, activation.document
    projection = TPMPublicProjectionV1.verify(req["public_projection"])
    _validate_activation_projection_binding(activation, projection)
    if (
        req["activation_request_digest"] != hashlib.sha256(activation_raw).hexdigest()
        or req["activation_request_id"] != act["request_id"]
        or req["installation_id"] != act["installation_id"]
        or req["requested_entitlements"] != act["requested_entitlements"]
        or req["release_policy_digest"] != context.release_payload_digest
        or act["release"]
        != {
            "release_policy_digest": context.release_payload_digest,
            "release_policy_version": context.release_version,
        }
    ):
        raise ProductionTPMCustodyError("PRODUCTION_TPM_ACTIVATION_BINDING_MISMATCH")
    _production_projection(projection, endorsement)
    return activation, request, projection


class ProductionTPMChallengeStore:
    """SQLite mechanics; production authority requires the installed issuer factory."""

    __slots__ = ("_path", "__weakref__")
    _path: Path

    def __init__(self, path: Path) -> None:
        object.__setattr__(self, "_path", Path(path).resolve())
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as database:
            database.execute("""CREATE TABLE IF NOT EXISTS tpm_challenges (
                challenge_id TEXT PRIMARY KEY, challenge BLOB NOT NULL,
                credential_secret BLOB NOT NULL,
                activation BLOB NOT NULL, request BLOB NOT NULL, pdsa_digest TEXT NOT NULL,
                endorsement_digest TEXT NOT NULL, release_digest TEXT NOT NULL,
                generation INTEGER NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('ISSUED','VERIFIED','EXPIRED')),
                exchange_reference TEXT, response BLOB)""")
            database.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS exact_tpm_issuance "
                "ON tpm_challenges(request,pdsa_digest)"
            )
        self._path.chmod(0o600)

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("production TPM challenge store configuration is immutable")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        database = sqlite3.connect(self._path, timeout=30)
        try:
            database.execute("PRAGMA journal_mode=WAL")
            database.execute("PRAGMA synchronous=FULL")
            with database:
                yield database
        finally:
            database.close()

    def issue(
        self,
        activation_request_raw: bytes,
        request_raw: bytes,
        *,
        pdsa_challenge: object,
        pdsa_store: PDSAChallengeStore,
        endorsement: object,
        context: object,
    ) -> TPMEnrollmentChallengeV1:
        from .pdsa_enrollment_challenge import require_verified_issued_challenge

        issuer = require_production_tpm_store(self, context=context, pdsa_store=pdsa_store)
        trusted = require_verified_production_trust_context(context)
        approved = _require_endorsement(endorsement, trusted)
        pdsa = require_verified_issued_challenge(pdsa_challenge, store=pdsa_store, context=trusted)
        _activation, request, projection = _activation_and_request(
            activation_request_raw, request_raw, trusted, approved
        )
        public = projection.document
        expiry = min(
            _time(pdsa.document["payload"]["expires_at_utc"]),
            _time(approved.document["payload"]["expires_at_utc"]),
        )
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            # Recheck live authority after waiting for the issuer write lock.
            require_production_tpm_store(
                self, context=trusted, issuer=issuer, pdsa_store=pdsa_store
            )
            require_verified_production_trust_context(trusted)
            pdsa = require_verified_issued_challenge(
                pdsa_challenge, store=pdsa_store, context=trusted
            )
            approved = _require_endorsement(endorsement, trusted)
            recorded = database.execute(
                "SELECT challenge,endorsement_digest,state FROM tpm_challenges "
                "WHERE request=? AND pdsa_digest=?",
                (request_raw, pdsa.digest_sha256),
            ).fetchone()
            if recorded is not None:
                if recorded[1] != approved.digest_sha256:
                    raise ProductionTPMCustodyError("TPM_CHALLENGE_ISSUANCE_CONFLICT")
                if recorded[2] == "EXPIRED" or expiry <= _utc_now():
                    raise ProductionTPMCustodyError("PRODUCTION_TPM_CHALLENGE_EXPIRED")
                return cast(
                    TPMEnrollmentChallengeV1,
                    TPMEnrollmentChallengeV1.from_canonical_bytes(recorded[0]),
                )
            made = make_credential_ecc(
                public["ek"]["public_area"]["hex"], bytes.fromhex(public["ak"]["name"])
            )
            challenge = TPMEnrollmentChallengeV1.create(
                request,
                expires_at_utc=expiry.strftime("%Y-%m-%dT%H:%M:%SZ"),
                credential_blob_hex=made.credential_blob.hex(),
                encrypted_secret_hex=made.encrypted_secret.hex(),
            )
            require_verified_production_trust_context(trusted)
            require_verified_issued_challenge(pdsa_challenge, store=pdsa_store, context=trusted)
            _require_endorsement(approved, trusted)
            if expiry <= _utc_now():
                raise ProductionTPMCustodyError("PRODUCTION_TPM_CHALLENGE_EXPIRED")
            database.execute(
                "INSERT INTO tpm_challenges VALUES (?,?,?,?,?,?,?,?,?,'ISSUED',NULL,NULL)",
                (
                    challenge.document["challenge_id"],
                    challenge.canonical_bytes,
                    made.credential_secret,
                    activation_request_raw,
                    request_raw,
                    pdsa.digest_sha256,
                    approved.digest_sha256,
                    trusted.release_payload_digest,
                    trusted.release_version,
                ),
            )
        return challenge

    def _record(self, challenge: TPMEnrollmentChallengeV1) -> tuple[Any, ...]:
        issuer = require_production_tpm_store(self)
        with self._connect() as database:
            require_production_tpm_store(self, issuer=issuer)
            row = database.execute(
                "SELECT * FROM tpm_challenges WHERE challenge_id=?",
                (challenge.document["challenge_id"],),
            ).fetchone()
        if row is None or row[1] != challenge.canonical_bytes:
            raise ProductionTPMCustodyError("UNKNOWN_OR_CHANGED_PRODUCTION_TPM_CHALLENGE")
        if _time(challenge.document["expires_at_utc"]) <= _utc_now() or row[9] == "EXPIRED":
            with self._connect() as database:
                database.execute("BEGIN IMMEDIATE")
                require_production_tpm_store(self, issuer=issuer)
                database.execute(
                    "UPDATE tpm_challenges SET state='EXPIRED' "
                    "WHERE challenge_id=? AND state='ISSUED'",
                    (row[0],),
                )
            raise ProductionTPMCustodyError("PRODUCTION_TPM_CHALLENGE_EXPIRED")
        return cast(tuple[Any, ...], row)

    def _retain_verified(
        self,
        challenge: TPMEnrollmentChallengeV1,
        response: bytes,
        reference: str,
        *,
        expected_record: tuple[Any, ...],
        pdsa_challenge: object,
        pdsa_store: PDSAChallengeStore,
        endorsement: object,
        context: object,
    ) -> None:
        from .pdsa_enrollment_challenge import require_verified_issued_challenge

        issuer = require_production_tpm_store(self, context=context, pdsa_store=pdsa_store)
        with self._connect() as database:
            database.execute("BEGIN IMMEDIATE")
            require_production_tpm_store(
                self, context=context, issuer=issuer, pdsa_store=pdsa_store
            )
            require_verified_production_trust_context(context)
            pdsa = require_verified_issued_challenge(
                pdsa_challenge, store=pdsa_store, context=context
            )
            approved = _require_endorsement(endorsement, context)
            row = database.execute(
                "SELECT * FROM tpm_challenges WHERE challenge_id=?",
                (challenge.document["challenge_id"],),
            ).fetchone()
            if (
                row is None
                or row[9] == "EXPIRED"
                or _time(challenge.document["expires_at_utc"]) <= _utc_now()
            ):
                raise ProductionTPMCustodyError("UNKNOWN_OR_EXPIRED_PRODUCTION_TPM_CHALLENGE")
            if (
                row[1:9] != expected_record[1:9]
                or row[1] != challenge.canonical_bytes
                or row[5] != pdsa.digest_sha256
                or row[6] != approved.digest_sha256
            ):
                raise ProductionTPMCustodyError("PRODUCTION_TPM_RETAINED_RECORD_CHANGED")
            if row[9] == "VERIFIED" and (row[11], row[10]) != (response, reference):
                raise ProductionTPMCustodyError("TPM_EXCHANGE_REPLAY_CONFLICT")
            database.execute(
                "UPDATE tpm_challenges SET state='VERIFIED',response=?,exchange_reference=? "
                "WHERE challenge_id=?",
                (response, reference, challenge.document["challenge_id"]),
            )


class ProductionTPMEnrollmentVerifier:
    def verify(
        self,
        activation_raw: bytes,
        request_raw: bytes,
        challenge_raw: bytes,
        response_raw: bytes,
        *,
        pending: ProductionTPMChallengeStore,
        pdsa_challenge: object,
        pdsa_store: PDSAChallengeStore,
        endorsement: object,
        context: object,
    ) -> VerifiedProductionTPMExchange:
        from .pdsa_enrollment_challenge import require_verified_issued_challenge

        if (
            type(self) is not ProductionTPMEnrollmentVerifier
            or type(pending) is not ProductionTPMChallengeStore
        ):
            raise ProductionTPMCustodyError("EXACT_PRODUCTION_TPM_VERIFIER_AND_STORE_REQUIRED")
        issuer = require_production_tpm_store(pending, context=context, pdsa_store=pdsa_store)
        trusted = require_verified_production_trust_context(context)
        approved = _require_endorsement(endorsement, trusted)
        pdsa = require_verified_issued_challenge(pdsa_challenge, store=pdsa_store, context=trusted)
        activation, request, projection = _activation_and_request(
            activation_raw, request_raw, trusted, approved
        )
        challenge = cast(
            TPMEnrollmentChallengeV1,
            TPMEnrollmentChallengeV1.from_canonical_bytes(challenge_raw),
        )
        response = TPMEnrollmentChallengeResponseV1.from_canonical_bytes(response_raw)
        row = pending._record(challenge)
        req, ch, rsp = request.document, challenge.document, response.document
        if (
            row[3:9]
            != (
                activation_raw,
                request_raw,
                pdsa.digest_sha256,
                approved.digest_sha256,
                trusted.release_payload_digest,
                trusted.release_version,
            )
            or ch["request_id"] != req["request_id"]
            or ch["public_projection_id"] != projection.evidence_reference
            or (rsp["request_id"], rsp["challenge_id"], rsp["issuer_nonce_hex"])
            != (req["request_id"], ch["challenge_id"], ch["issuer_nonce_hex"])
        ):
            raise ProductionTPMCustodyError("PRODUCTION_TPM_EXCHANGE_BINDING_MISMATCH")
        if row[9] == "VERIFIED" and row[11] != response_raw:
            raise ProductionTPMCustodyError("TPM_EXCHANGE_REPLAY_CONFLICT")
        secret = row[2]
        if not hmac.compare_digest(
            hashlib.sha256(secret).hexdigest(), rsp["activated_credential_digest"]
        ) or not hmac.compare_digest(
            credential_activation_proof(secret, ch["challenge_id"]),
            _hex(rsp["credential_activation_proof_hex"], size=32),
        ):
            raise ProductionTPMCustodyError("PRODUCTION_ACTIVATE_CREDENTIAL_PROOF_MISMATCH")
        _ek, ak, k_psa = _production_projection(projection, approved)
        nonce, pdsa_digest = (
            _hex(ch["issuer_nonce_hex"], size=32),
            bytes.fromhex(pdsa.digest_sha256),
        )
        _verify_creation(
            attest=_hex(rsp["certify_creation_attest_hex"]),
            signature=_hex(rsp["certify_creation_signature_hex"]),
            ak=ak,
            expected_name=k_psa.name,
            expected_creation_hash=_hex(projection.document["k_psa"]["creation_hash"], size=32),
            expected_qualifier=production_exchange_qualifying_data(
                nonce, hashlib.sha256(activation.canonical_bytes).digest(), pdsa_digest
            ),
        )
        pop = _hex(rsp["k_psa_pop_signature_der_hex"], maximum=72)
        try:
            r, s = utils.decode_dss_signature(pop)
            if (
                utils.encode_dss_signature(r, s) != pop
                or not 1 <= r < P256_ORDER
                or not 1 <= s < P256_ORDER
            ):
                raise ValueError("invalid DER signature")
            validate_public_key(k_psa.sec1).verify(
                pop,
                production_k_psa_pop_digest(nonce, pdsa_digest),
                ec.ECDSA(utils.Prehashed(hashes.SHA256())),
            )
        except (InvalidSignature, ValueError) as exc:
            raise ProductionTPMCustodyError("INVALID_PRODUCTION_K_PSA_POP") from exc
        raw_items = (activation_raw, request_raw, challenge_raw, response_raw)
        reference = hashlib.sha256(
            EXCHANGE_REFERENCE_DOMAIN + b"".join(hashlib.sha256(raw).digest() for raw in raw_items)
        ).hexdigest()
        pending._retain_verified(
            challenge,
            response_raw,
            reference,
            expected_record=row,
            pdsa_challenge=pdsa_challenge,
            pdsa_store=pdsa_store,
            endorsement=approved,
            context=trusted,
        )
        return _issue_capability(
            VerifiedProductionTPMExchange,
            canonical_bytes=request_raw,
            context=trusted,
            retained_bytes=raw_items,
            exchange_reference=reference,
            projection_raw=projection.canonical_bytes,
            pending=pending,
            pending_path=pending._path,
            issuer=issuer,
            pdsa_store=pdsa_store,
            endorsement=approved,
            pdsa_digest=pdsa.digest_sha256,
        )


def require_verified_production_tpm_exchange(
    value: object,
    *,
    context: object,
    pending: ProductionTPMChallengeStore | None = None,
    issuer: ProductionEnrollmentIssuerContext | None = None,
) -> VerifiedProductionTPMExchange:
    snapshot = _require_context(value, VerifiedProductionTPMExchange, context)
    source = snapshot["pending"]
    if (
        type(source) is not ProductionTPMChallengeStore
        or source._path != snapshot["pending_path"]
        or (pending is not None and source is not pending)
    ):
        raise ProductionTPMCustodyError("PRODUCTION_TPM_EXCHANGE_STORE_MISMATCH")
    retained_issuer = snapshot["issuer"]
    if issuer is not None and issuer is not retained_issuer:
        raise ProductionTPMCustodyError("PRODUCTION_TPM_EXCHANGE_ISSUER_MISMATCH")
    require_production_tpm_store(
        source,
        context=context,
        issuer=retained_issuer,
        pdsa_store=snapshot["pdsa_store"],
    )
    _require_endorsement(snapshot["endorsement"], context)
    challenge = cast(
        TPMEnrollmentChallengeV1,
        TPMEnrollmentChallengeV1.from_canonical_bytes(snapshot["retained_bytes"][2]),
    )
    row = source._record(challenge)
    if row[9] != "VERIFIED" or (row[10], row[11]) != (
        snapshot["exchange_reference"],
        snapshot["retained_bytes"][3],
    ):
        raise ProductionTPMCustodyError("RETAINED_PRODUCTION_TPM_EXCHANGE_REQUIRED")
    return cast(VerifiedProductionTPMExchange, value)


@dataclass(frozen=True, init=False)
class ProductionPreEnrollmentKeyCustodyEvidenceV1:
    """Canonical public evidence, never an authority capability by itself."""

    canonical_bytes: bytes

    def __init__(self) -> None:
        raise TypeError("use from_mapping() or from_canonical_bytes()")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ProductionPreEnrollmentKeyCustodyEvidenceV1:
        raw = canonical_json_bytes(dict(value))
        return cls.from_canonical_bytes(raw)

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> ProductionPreEnrollmentKeyCustodyEvidenceV1:
        value = _parse(raw)
        _exact(value, CUSTODY_FIELDS)
        constants = {
            "schema_version": "ProductionPreEnrollmentKeyCustodyEvidenceV1",
            "environment": "PRODUCTION",
            "custody_profile": CUSTODY_PROFILE,
            "provider_name": "Microsoft Platform Crypto Provider",
        }
        if any(
            type(value[field]) is not str or value[field] != expected
            for field, expected in constants.items()
        ):
            raise ProductionTPMCustodyError("TEST_ONLY_OR_INVALID_PRODUCTION_CUSTODY_PROFILE")
        for field in CUSTODY_FIELDS - set(constants) - {"release_policy_generation"}:
            size = (
                32
                if field.endswith("sha256")
                or field
                in {
                    "verified_tpm_public_projection_id",
                    "verified_tpm_exchange_reference",
                    "ek_public_digest",
                    "ak_public_digest",
                    "subject_creation_hash",
                }
                else None
            )
            _hex(value[field], size=size)
        if (
            type(value["release_policy_generation"]) is not int
            or not 1 <= value["release_policy_generation"] <= 9007199254740991
        ):
            raise ProductionTPMCustodyError("INVALID_PRODUCTION_CUSTODY_GENERATION")
        subject = parse_production_ecc_public(
            _hex(value["subject_tpmt_public_hex"]), role="pre_enrollment"
        )
        sec1 = _hex(value["pre_enrollment_public_key_canonical_bytes"], size=65)
        if (
            subject.sec1 != sec1
            or subject.name.hex() != value["subject_tpm_name"]
            or public_key_fingerprint(sec1) != value["pre_enrollment_public_key_fingerprint_sha256"]
            or hashlib.sha256(_hex(value["certify_creation_attest_hex"])).hexdigest()
            != value["creation_attestation_digest_sha256"]
        ):
            raise ProductionTPMCustodyError("PRODUCTION_CUSTODY_PUBLIC_BINDING_MISMATCH")
        parse_production_creation_attestation(_hex(value["certify_creation_attest_hex"]))
        instance = object.__new__(cls)
        object.__setattr__(instance, "canonical_bytes", raw)
        return instance

    @property
    def document(self) -> dict[str, Any]:
        return _parse(self.canonical_bytes)


def make_pre_enrollment_custody_evidence(
    *,
    request: PreEnrollmentRequestV1,
    target_tpm_projection: TPMPublicProjectionV1,
    subject_tpmt_public: bytes,
    subject_name: bytes,
    creation_hash: bytes,
    attest: bytes,
    signature: bytes,
) -> ProductionPreEnrollmentKeyCustodyEvidenceV1:
    document = request.document
    if target_tpm_projection.evidence_reference != document["verified_tpm_public_projection_id"]:
        raise ProductionTPMCustodyError("PRODUCTION_CUSTODY_TARGET_PROJECTION_MISMATCH")
    fields = {
        field: document[field]
        for field in (
            "verified_tpm_public_projection_id",
            "ek_public_digest",
            "ak_public_digest",
            "pdsa_challenge_digest_sha256",
            "release_policy_digest_sha256",
            "release_policy_generation",
            "pre_enrollment_public_key_canonical_bytes",
            "pre_enrollment_public_key_fingerprint_sha256",
        )
    }
    return ProductionPreEnrollmentKeyCustodyEvidenceV1.from_mapping(
        {
            **fields,
            "schema_version": "ProductionPreEnrollmentKeyCustodyEvidenceV1",
            "environment": "PRODUCTION",
            "custody_profile": CUSTODY_PROFILE,
            "provider_name": "Microsoft Platform Crypto Provider",
            "pre_enrollment_request_digest_sha256": request.digest_sha256,
            "verified_tpm_exchange_reference": document["verified_tpm_exchange_reference"],
            "subject_tpmt_public_hex": subject_tpmt_public.hex(),
            "subject_tpm_name": subject_name.hex(),
            "subject_creation_hash": creation_hash.hex(),
            "certify_creation_attest_hex": attest.hex(),
            "certify_creation_signature_hex": signature.hex(),
            "creation_attestation_digest_sha256": hashlib.sha256(attest).hexdigest(),
        }
    )


class ProductionPreEnrollmentKeyCustodyVerifier:
    def verify(
        self,
        evidence_raw: bytes,
        *,
        request: PreEnrollmentRequestV1,
        exchange: object,
        endorsement: object,
        context: object,
    ) -> VerifiedProductionPreEnrollmentKeyCustody:
        if (
            type(self) is not ProductionPreEnrollmentKeyCustodyVerifier
            or type(request) is not PreEnrollmentRequestV1
        ):
            raise ProductionTPMCustodyError("EXACT_PRODUCTION_CUSTODY_VERIFIER_REQUIRED")
        trusted = require_verified_production_trust_context(context)
        approved = _require_endorsement(endorsement, trusted)
        verified = require_verified_production_tpm_exchange(exchange, context=trusted)
        snapshot = _snapshot(verified)
        if snapshot["endorsement"].canonical_bytes != approved.canonical_bytes:
            raise ProductionTPMCustodyError("PRODUCTION_CUSTODY_ENDORSEMENT_MISMATCH")
        evidence = ProductionPreEnrollmentKeyCustodyEvidenceV1.from_canonical_bytes(evidence_raw)
        payload, req = evidence.document, request.document
        request.require_production_trust_binding(trusted)
        request.compare_tpm_exchange_bindings(
            activation_request_raw=verified.retained_bytes[0],
            request_raw=verified.retained_bytes[1],
            challenge_raw=verified.retained_bytes[2],
            response_raw=verified.retained_bytes[3],
        )
        if (
            req["pdsa_challenge_digest_sha256"] != snapshot["pdsa_digest"]
            or req["verified_tpm_exchange_reference"] != verified.exchange_reference
        ):
            raise ProductionTPMCustodyError("PRODUCTION_CUSTODY_EXCHANGE_MISMATCH")
        expected = {
            field: req[field]
            for field in (
                "verified_tpm_public_projection_id",
                "ek_public_digest",
                "ak_public_digest",
                "pdsa_challenge_digest_sha256",
                "release_policy_digest_sha256",
                "release_policy_generation",
                "pre_enrollment_public_key_canonical_bytes",
                "pre_enrollment_public_key_fingerprint_sha256",
            )
        }
        expected.update(
            pre_enrollment_request_digest_sha256=request.digest_sha256,
            verified_tpm_exchange_reference=verified.exchange_reference,
        )
        if any(payload[field] != value for field, value in expected.items()):
            raise ProductionTPMCustodyError("PRODUCTION_CUSTODY_REQUEST_BINDING_MISMATCH")
        _ek, ak, _k_psa = _production_projection(verified.projection, approved)
        subject = parse_production_ecc_public(
            _hex(payload["subject_tpmt_public_hex"]), role="pre_enrollment"
        )
        _verify_creation(
            attest=_hex(payload["certify_creation_attest_hex"]),
            signature=_hex(payload["certify_creation_signature_hex"]),
            ak=ak,
            expected_name=subject.name,
            expected_creation_hash=_hex(payload["subject_creation_hash"], size=32),
            expected_qualifier=pre_enrollment_custody_qualifying_data(request),
        )
        return _issue_capability(
            VerifiedProductionPreEnrollmentKeyCustody,
            canonical_bytes=evidence_raw,
            context=trusted,
            request_raw=request.canonical_bytes,
            exchange=verified,
            endorsement=approved,
        )


def require_verified_pre_enrollment_custody(
    value: object,
    *,
    context: object,
    request: PreEnrollmentRequestV1 | None = None,
    exchange: object | None = None,
) -> VerifiedProductionPreEnrollmentKeyCustody:
    snapshot = _require_context(value, VerifiedProductionPreEnrollmentKeyCustody, context)
    verified = require_verified_production_tpm_exchange(snapshot["exchange"], context=context)
    if (request is not None and request.canonical_bytes != snapshot["request_raw"]) or (
        exchange is not None and verified is not exchange
    ):
        raise ProductionTPMCustodyError("PRODUCTION_CUSTODY_CAPABILITY_BINDING_MISMATCH")
    return cast(VerifiedProductionPreEnrollmentKeyCustody, value)
