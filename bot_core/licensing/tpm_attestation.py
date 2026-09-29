"""Canonical TPM challenge contracts and issuer-side verification boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import secrets
import struct
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils
from cryptography.hazmat.decrepit.ciphers.modes import CFB
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

from .activation_request import ActivationRequestV1
from .canonical import canonical_json_bytes, digest, exact
from .device_enrollment import TPMPublicProjectionV1, derive_device_id

POP_DOMAIN = b"CryptoHunter.Stage9.K_PSA.ProofOfPossession.TEST_ONLY.v1\x00"
TPM_CC_SIGN = 0x0000015D
TPM_CC_POLICY_COMMAND_CODE = 0x0000016C
EXCHANGE_REFERENCE_DOMAIN = b"CryptoHunter.TPMEnrollmentExchangeReferenceV1\x00"
CERTIFY_QUALIFYING_DOMAIN = b"CryptoHunter.TPM.CertifyCreation.TEST_ONLY.v1\x00"
ACTIVATION_PROOF_DOMAIN = b"CryptoHunter.TPM.ActivateCredential.Proof.v1\x00"
TPM_GENERATED_VALUE = 0xFF544347
TPM_ST_ATTEST_CREATION = 0x801A


def test_only_sign_policy_digest() -> bytes:
    """TPM 2.0 PolicyCommandCode(Sign), starting from a zero SHA-256 policyDigest."""
    return hashlib.sha256(
        bytes(32) + struct.pack(">II", TPM_CC_POLICY_COMMAND_CODE, TPM_CC_SIGN)
    ).digest()


def proof_of_possession_digest(nonce: bytes) -> bytes:
    if len(nonce) != 32:
        raise ValueError("proof-of-possession nonce must be 256 bits")
    return hashlib.sha256(POP_DOMAIN + nonce).digest()


def _hex(value: Any, name: str, size: int | None = None) -> bytes:
    if not isinstance(value, str):
        raise ValueError(f"invalid {name}")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"invalid {name}") from exc
    if raw.hex() != value or (size is not None and len(raw) != size):
        raise ValueError(f"invalid {name}")
    return raw


def _ecc_public_key(public_hex: str) -> ec.EllipticCurvePublicKey:
    raw = bytes.fromhex(public_hex)
    if len(raw) < 68 or int.from_bytes(raw[-34:-32], "big") != 32:
        raise ValueError("invalid K_PSA Y coordinate")
    x_size_offset = len(raw) - 68
    if int.from_bytes(raw[x_size_offset : x_size_offset + 2], "big") != 32:
        raise ValueError("invalid K_PSA X coordinate")
    return ec.EllipticCurvePublicNumbers(
        int.from_bytes(raw[x_size_offset + 2 : x_size_offset + 34], "big"),
        int.from_bytes(raw[-32:], "big"),
        ec.SECP256R1(),
    ).public_key()


def _read_u16(data: bytes, offset: int) -> tuple[int, int]:
    if offset + 2 > len(data):
        raise ValueError("truncated UINT16")
    return int.from_bytes(data[offset : offset + 2], "big"), offset + 2


def _read_u32(data: bytes, offset: int) -> tuple[int, int]:
    if offset + 4 > len(data):
        raise ValueError("truncated UINT32")
    return int.from_bytes(data[offset : offset + 4], "big"), offset + 4


def _read_2b(data: bytes, offset: int) -> tuple[bytes, int]:
    size, offset = _read_u16(data, offset)
    end = offset + size
    if end > len(data):
        raise ValueError("truncated TPM2B")
    return data[offset:end], end


def _tpm2b(data: bytes) -> bytes:
    if len(data) > 0xFFFF:
        raise ValueError("TPM2B too large")
    return len(data).to_bytes(2, "big") + data


def _kdfa(key: bytes, label: bytes, context_u: bytes, context_v: bytes, bits: int) -> bytes:
    result = b""
    counter = 1
    while len(result) * 8 < bits:
        result += hmac.new(
            key,
            struct.pack(">I", counter)
            + label
            + b"\x00"
            + context_u
            + context_v
            + struct.pack(">I", bits),
            hashlib.sha256,
        ).digest()
        counter += 1
    return result[: (bits + 7) // 8]


def _kdfe(z: bytes, label: bytes, party_u: bytes, party_v: bytes, bits: int) -> bytes:
    result = b""
    counter = 1
    while len(result) * 8 < bits:
        result += hashlib.sha256(
            struct.pack(">I", counter) + z + label + b"\x00" + party_u + party_v
        ).digest()
        counter += 1
    return result[: (bits + 7) // 8]


@dataclass(frozen=True)
class MakeCredentialResult:
    credential_blob: bytes
    encrypted_secret: bytes
    credential_secret: bytes


def make_credential_ecc(
    ek_public_hex: str,
    ak_name: bytes,
    *,
    credential_secret: bytes | None = None,
    test_only_ephemeral_scalar: int | None = None,
) -> MakeCredentialResult:
    """TPM 2.0 ECC MakeCredential using KDFe/KDFa, AES-128-CFB and HMAC-SHA256."""
    ek = _ecc_public_key(ek_public_hex)
    ephemeral = (
        ec.derive_private_key(test_only_ephemeral_scalar, ec.SECP256R1())
        if test_only_ephemeral_scalar is not None
        else ec.generate_private_key(ec.SECP256R1())
    )
    ephemeral_numbers = ephemeral.public_key().public_numbers()
    ek_numbers = ek.public_numbers()
    ephemeral_x = ephemeral_numbers.x.to_bytes(32, "big")
    ephemeral_y = ephemeral_numbers.y.to_bytes(32, "big")
    ek_x = ek_numbers.x.to_bytes(32, "big")
    shared = ephemeral.exchange(ec.ECDH(), ek)
    seed = _kdfe(shared, b"IDENTITY", ephemeral_x, ek_x, 256)
    secret = credential_secret or secrets.token_bytes(32)
    if not 16 <= len(secret) <= 32:
        raise ValueError("credential secret must contain 16..32 bytes")
    storage_key = _kdfa(seed, b"STORAGE", ak_name, b"", 128)
    encrypted_identity = (
        Cipher(algorithms.AES(storage_key), CFB(bytes(16))).encryptor().update(_tpm2b(secret))
    )
    integrity_key = _kdfa(seed, b"INTEGRITY", b"", b"", 256)
    integrity = hmac.new(integrity_key, encrypted_identity + ak_name, hashlib.sha256).digest()
    return MakeCredentialResult(
        _tpm2b(integrity) + encrypted_identity,
        _tpm2b(ephemeral_x) + _tpm2b(ephemeral_y),
        secret,
    )


def certify_qualifying_data(issuer_nonce: bytes, activation_digest: bytes) -> bytes:
    if len(issuer_nonce) != 32 or len(activation_digest) != 32:
        raise ValueError("invalid CertifyCreation binding input")
    return hashlib.sha256(CERTIFY_QUALIFYING_DOMAIN + issuer_nonce + activation_digest).digest()


def credential_activation_proof(secret: bytes, challenge_id: str) -> bytes:
    return hmac.new(
        secret,
        ACTIVATION_PROOF_DOMAIN + bytes.fromhex(challenge_id),
        hashlib.sha256,
    ).digest()


@dataclass(frozen=True)
class CreationAttestation:
    extra_data: bytes
    object_name: bytes
    creation_hash: bytes


def parse_creation_attestation(attest: bytes) -> CreationAttestation:
    magic, offset = _read_u32(attest, 0)
    attest_type, offset = _read_u16(attest, offset)
    if magic != TPM_GENERATED_VALUE:
        raise ValueError("wrong TPMS_ATTEST magic")
    if attest_type != TPM_ST_ATTEST_CREATION:
        raise ValueError("wrong TPMS_ATTEST type")
    _qualified_signer, offset = _read_2b(attest, offset)
    extra_data, offset = _read_2b(attest, offset)
    if offset + 25 > len(attest):
        raise ValueError("truncated TPMS_ATTEST clock/firmware")
    offset += 25
    object_name, offset = _read_2b(attest, offset)
    creation_hash, offset = _read_2b(attest, offset)
    if offset != len(attest):
        raise ValueError("TPMS_ATTEST trailing bytes")
    return CreationAttestation(extra_data, object_name, creation_hash)


def _parse_ecdsa_signature(signature: bytes) -> bytes:
    algorithm, offset = _read_u16(signature, 0)
    hash_algorithm, offset = _read_u16(signature, offset)
    if algorithm != 0x0018 or hash_algorithm != 0x000B:
        raise ValueError("unsupported TPMT_SIGNATURE algorithm")
    r, offset = _read_2b(signature, offset)
    s, offset = _read_2b(signature, offset)
    if offset != len(signature) or not r or not s:
        raise ValueError("malformed TPMT_SIGNATURE")
    return utils.encode_dss_signature(int.from_bytes(r, "big"), int.from_bytes(s, "big"))


def tpm_ecdsa_signature_to_der(signature: bytes) -> bytes:
    return _parse_ecdsa_signature(signature)


def verify_certify_creation(
    *,
    attest: bytes,
    signature: bytes,
    ak_public_hex: str,
    expected_qualifying_data: bytes,
    expected_name: bytes,
    expected_creation_hash: bytes,
) -> None:
    parsed = parse_creation_attestation(attest)
    if parsed.extra_data != expected_qualifying_data:
        raise ValueError("CertifyCreation qualifyingData mismatch")
    if parsed.object_name != expected_name:
        raise ValueError("CertifyCreation object Name mismatch")
    if parsed.creation_hash != expected_creation_hash:
        raise ValueError("CertifyCreation creationHash mismatch")
    try:
        _ecc_public_key(ak_public_hex).verify(
            _parse_ecdsa_signature(signature),
            hashlib.sha256(attest).digest(),
            ec.ECDSA(utils.Prehashed(hashes.SHA256())),
        )
    except InvalidSignature as exc:
        raise ValueError("CertifyCreation AK signature invalid") from exc


def verify_k_psa_proof_of_possession(
    evidence: TPMPublicProjectionV1, nonce: bytes, signature: bytes
) -> None:
    verify_k_psa_public_signature(
        evidence.document["k_psa"]["public_area"]["hex"], nonce, signature
    )


def verify_k_psa_public_signature(public_hex: str, nonce: bytes, signature: bytes) -> None:
    key = _ecc_public_key(public_hex)
    try:
        key.verify(
            signature,
            proof_of_possession_digest(nonce),
            ec.ECDSA(utils.Prehashed(hashes.SHA256())),
        )
    except InvalidSignature as exc:
        raise ValueError("K_PSA_PROOF_OF_POSSESSION_FAILED") from exc


@dataclass(frozen=True, init=False)
class _CanonicalContract:
    canonical_bytes: bytes

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("use create() or from_canonical_bytes()")

    @classmethod
    def _from_value(cls, value: Mapping[str, Any]) -> "_CanonicalContract":
        instance = object.__new__(cls)
        object.__setattr__(instance, "canonical_bytes", canonical_json_bytes(dict(value)))
        return instance

    @property
    def document(self) -> dict[str, Any]:
        value = json.loads(self.canonical_bytes)
        assert isinstance(value, dict)
        return value

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> "_CanonicalContract":
        value = json.loads(raw)
        if canonical_json_bytes(value) != raw:
            raise ValueError("noncanonical enrollment artifact")
        cls._validate(value)
        return cls._from_value(value)

    @classmethod
    def _validate(cls, value: Mapping[str, Any]) -> None:
        raise NotImplementedError


@dataclass(frozen=True, init=False)
class TPMEnrollmentRequestV1(_CanonicalContract):
    @classmethod
    def create(
        cls,
        *,
        activation_request: ActivationRequestV1,
        public_projection: TPMPublicProjectionV1,
        client_nonce: bytes | None = None,
    ) -> "TPMEnrollmentRequestV1":
        _validate_activation_projection_binding(activation_request, public_projection)
        activation = activation_request.document
        body = {
            "schema": "TPMEnrollmentRequestV1",
            "version": 1,
            "activation_request_id": activation["request_id"],
            "activation_request_digest": hashlib.sha256(
                activation_request.canonical_bytes
            ).hexdigest(),
            "installation_id": activation["installation_id"],
            "release_policy_digest": activation["release"]["release_policy_digest"],
            "requested_entitlements": activation["requested_entitlements"],
            "client_nonce_hex": (client_nonce or secrets.token_bytes(32)).hex(),
            "public_projection": public_projection.document,
        }
        value = {**body, "request_id": digest(body)}
        cls._validate(value)
        return cls._from_value(value)  # type: ignore[return-value]

    @classmethod
    def _validate(cls, value: Mapping[str, Any]) -> None:
        exact(
            dict(value),
            {
                "schema",
                "version",
                "request_id",
                "activation_request_id",
                "activation_request_digest",
                "installation_id",
                "release_policy_digest",
                "requested_entitlements",
                "client_nonce_hex",
                "public_projection",
            },
            "TPM enrollment request",
        )
        if value["schema"] != "TPMEnrollmentRequestV1" or value["version"] != 1:
            raise ValueError("unsupported TPM enrollment request")
        _hex(value["request_id"], "request_id", 32)
        for name in ("activation_request_id", "activation_request_digest", "release_policy_digest"):
            _hex(value[name], name, 32)
        _hex(value["client_nonce_hex"], "client nonce", 32)
        projection = TPMPublicProjectionV1.verify(value["public_projection"])
        body = dict(value)
        request_id = body.pop("request_id")
        if digest(body) != request_id or projection.document != value["public_projection"]:
            raise ValueError("TPM enrollment request binding mismatch")


def _validate_activation_projection_binding(
    activation_request: ActivationRequestV1, projection: TPMPublicProjectionV1
) -> None:
    activation, public = activation_request.document, projection.document
    expected_tpm = {
        "evidence_profile": public["evidence_profile"],
        "ek_public_digest": public["ek"]["public_digest"],
        "ak_public_digest": public["ak"]["public_digest"],
        "evidence_reference": projection.evidence_reference,
        "manufacturer": public["tpm"]["manufacturer"],
        "model": public["tpm"]["model"],
    }
    if (
        activation["k_psa"]
        != {key: public["k_psa"][key] for key in ("public_area", "name", "algorithm_profile")}
        or activation["tpm"] != expected_tpm
        or activation["device"]["device_id"] != derive_device_id(projection)
    ):
        raise ValueError("ACTIVATION_REQUEST_PROJECTION_BINDING_MISMATCH")


def _parse_activation_request(raw: bytes) -> ActivationRequestV1:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid activation request JSON") from exc
    if canonical_json_bytes(value) != raw:
        raise ValueError("noncanonical activation request")
    return ActivationRequestV1.from_mapping(value)


@dataclass(frozen=True, init=False)
class TPMEnrollmentChallengeV1(_CanonicalContract):
    @classmethod
    def create(
        cls,
        request: TPMEnrollmentRequestV1,
        *,
        expires_at_utc: str,
        credential_blob_hex: str,
        encrypted_secret_hex: str,
    ) -> "TPMEnrollmentChallengeV1":
        req = request.document
        body = {
            "schema": "TPMEnrollmentChallengeV1",
            "version": 1,
            "request_id": req["request_id"],
            "public_projection_id": req["public_projection"]["evidence_id"],
            "issuer_nonce_hex": secrets.token_bytes(32).hex(),
            "credential_blob_hex": credential_blob_hex,
            "encrypted_secret_hex": encrypted_secret_hex,
            "expires_at_utc": expires_at_utc,
        }
        value = {**body, "challenge_id": digest(body)}
        cls._validate(value)
        return cls._from_value(value)  # type: ignore[return-value]

    @classmethod
    def _validate(cls, value: Mapping[str, Any]) -> None:
        exact(
            dict(value),
            {
                "schema",
                "version",
                "challenge_id",
                "request_id",
                "public_projection_id",
                "issuer_nonce_hex",
                "credential_blob_hex",
                "encrypted_secret_hex",
                "expires_at_utc",
            },
            "TPM enrollment challenge",
        )
        if value["schema"] != "TPMEnrollmentChallengeV1" or value["version"] != 1:
            raise ValueError("unsupported TPM enrollment challenge")
        for name in ("challenge_id", "request_id", "public_projection_id"):
            _hex(value[name], name, 32)
        _hex(value["issuer_nonce_hex"], "issuer nonce", 32)
        if not _hex(value["credential_blob_hex"], "credentialBlob"):
            raise ValueError("empty credentialBlob")
        if not _hex(value["encrypted_secret_hex"], "encryptedSecret"):
            raise ValueError("empty encryptedSecret")
        datetime.strptime(value["expires_at_utc"], "%Y-%m-%dT%H:%M:%SZ")
        body = dict(value)
        challenge_id = body.pop("challenge_id")
        if digest(body) != challenge_id:
            raise ValueError("challenge_id mismatch")


@dataclass(frozen=True, init=False)
class TPMEnrollmentChallengeResponseV1(_CanonicalContract):
    @classmethod
    def create(
        cls,
        challenge: TPMEnrollmentChallengeV1,
        *,
        activated_credential_digest: str,
        credential_activation_proof_hex: str,
        certify_creation_attest_hex: str,
        certify_creation_signature_hex: str,
        k_psa_pop_signature_der_hex: str,
    ) -> "TPMEnrollmentChallengeResponseV1":
        item = challenge.document
        value = {
            "schema": "TPMEnrollmentChallengeResponseV1",
            "version": 1,
            "request_id": item["request_id"],
            "challenge_id": item["challenge_id"],
            "issuer_nonce_hex": item["issuer_nonce_hex"],
            "activated_credential_digest": activated_credential_digest,
            "credential_activation_proof_hex": credential_activation_proof_hex,
            "certify_creation_attest_hex": certify_creation_attest_hex,
            "certify_creation_signature_hex": certify_creation_signature_hex,
            "k_psa_pop_signature_der_hex": k_psa_pop_signature_der_hex,
        }
        cls._validate(value)
        return cls._from_value(value)  # type: ignore[return-value]

    @classmethod
    def _validate(cls, value: Mapping[str, Any]) -> None:
        exact(
            dict(value),
            {
                "schema",
                "version",
                "request_id",
                "challenge_id",
                "issuer_nonce_hex",
                "activated_credential_digest",
                "credential_activation_proof_hex",
                "certify_creation_attest_hex",
                "certify_creation_signature_hex",
                "k_psa_pop_signature_der_hex",
            },
            "TPM challenge response",
        )
        if value["schema"] != "TPMEnrollmentChallengeResponseV1" or value["version"] != 1:
            raise ValueError("unsupported TPM challenge response")
        for name in ("request_id", "challenge_id", "issuer_nonce_hex"):
            _hex(value[name], name, 32)
        for name in (
            "activated_credential_digest",
            "credential_activation_proof_hex",
            "certify_creation_attest_hex",
            "certify_creation_signature_hex",
            "k_psa_pop_signature_der_hex",
        ):
            if not _hex(value[name], name):
                raise ValueError(f"empty {name}")


class PendingChallengeStore:
    """Transport-neutral single-use state; offline tools and servers can persist this model."""

    def __init__(self) -> None:
        self._pending: dict[str, tuple[bytes, bytes]] = {}
        self._used: set[str] = set()

    def add(self, challenge: TPMEnrollmentChallengeV1, credential_secret: bytes) -> None:
        challenge_id = challenge.document["challenge_id"]
        if challenge_id in self._pending or challenge_id in self._used:
            raise ValueError("challenge already recorded")
        if not credential_secret:
            raise ValueError("credential secret is required")
        self._pending[challenge_id] = (challenge.canonical_bytes, bytes(credential_secret))

    def require_pending(self, challenge: TPMEnrollmentChallengeV1) -> None:
        challenge_id = challenge.document["challenge_id"]
        if challenge_id in self._used:
            raise ValueError("TPM_ENROLLMENT_CHALLENGE_REPLAY")
        record = self._pending.get(challenge_id)
        if record is None or record[0] != challenge.canonical_bytes:
            raise ValueError("unknown or changed challenge")

    def credential_secret(self, challenge: TPMEnrollmentChallengeV1) -> bytes:
        self.require_pending(challenge)
        return self._pending[challenge.document["challenge_id"]][1]

    def consume(self, challenge: TPMEnrollmentChallengeV1) -> None:
        self.require_pending(challenge)
        challenge_id = challenge.document["challenge_id"]
        del self._pending[challenge_id]
        self._used.add(challenge_id)


def create_issuer_challenge(
    request: TPMEnrollmentRequestV1,
    pending: PendingChallengeStore,
    *,
    expires_at_utc: str,
) -> TPMEnrollmentChallengeV1:
    return _create_issuer_challenge(request, pending, expires_at_utc=expires_at_utc)


def create_test_only_issuer_challenge(
    request: TPMEnrollmentRequestV1,
    pending: PendingChallengeStore,
    *,
    expires_at_utc: str,
    credential_secret: bytes,
    ephemeral_scalar: int,
) -> TPMEnrollmentChallengeV1:
    return _create_issuer_challenge(
        request,
        pending,
        expires_at_utc=expires_at_utc,
        credential_secret=credential_secret,
        ephemeral_scalar=ephemeral_scalar,
    )


def _create_issuer_challenge(
    request: TPMEnrollmentRequestV1,
    pending: PendingChallengeStore,
    *,
    expires_at_utc: str,
    credential_secret: bytes | None = None,
    ephemeral_scalar: int | None = None,
) -> TPMEnrollmentChallengeV1:
    projection = request.document["public_projection"]
    made = make_credential_ecc(
        projection["ek"]["public_area"]["hex"],
        bytes.fromhex(projection["ak"]["name"]),
        credential_secret=credential_secret,
        test_only_ephemeral_scalar=ephemeral_scalar,
    )
    challenge = TPMEnrollmentChallengeV1.create(
        request,
        expires_at_utc=expires_at_utc,
        credential_blob_hex=made.credential_blob.hex(),
        encrypted_secret_hex=made.encrypted_secret.hex(),
    )
    pending.add(challenge, made.credential_secret)
    return challenge


@dataclass(frozen=True)
class VerifiedTPMEnrollmentExchangeV1:
    """Convenience result only; consequential boundaries must reverify raw bytes."""

    activation_request_digest: str
    request_digest: str
    challenge_digest: str
    response_digest: str
    exchange_reference: str


class ProductionTPMAttestationVerifier:
    """Concrete issuer verifier for canonical activation and TPM attestation artifacts."""

    def verify(
        self,
        activation_request_raw: bytes,
        request_raw: bytes,
        challenge_raw: bytes,
        response_raw: bytes,
        *,
        pending: PendingChallengeStore,
        expected_release_policy_digest: str,
    ) -> VerifiedTPMEnrollmentExchangeV1:
        activation, request, challenge, response = _parse_and_bind_exchange(
            activation_request_raw,
            request_raw,
            challenge_raw,
            response_raw,
            pending,
            expected_release_policy_digest,
        )
        _verify_hardware_exchange(activation, request, challenge, response, pending)
        return _exchange_result(activation_request_raw, request_raw, challenge_raw, response_raw)


def _parse_and_bind_exchange(
    activation_request_raw: bytes,
    request_raw: bytes,
    challenge_raw: bytes,
    response_raw: bytes,
    pending: PendingChallengeStore,
    expected_release_policy_digest: str,
) -> tuple[
    ActivationRequestV1,
    TPMEnrollmentRequestV1,
    TPMEnrollmentChallengeV1,
    TPMEnrollmentChallengeResponseV1,
]:
    activation = _parse_activation_request(activation_request_raw)
    request = TPMEnrollmentRequestV1.from_canonical_bytes(request_raw)
    challenge = TPMEnrollmentChallengeV1.from_canonical_bytes(challenge_raw)
    response = TPMEnrollmentChallengeResponseV1.from_canonical_bytes(response_raw)
    req, ch, rsp = request.document, challenge.document, response.document
    pending.require_pending(challenge)
    activation_digest = hashlib.sha256(activation_request_raw).hexdigest()
    if (
        req["activation_request_id"] != activation.document["request_id"]
        or req["activation_request_digest"] != activation_digest
    ):
        raise ValueError("ACTIVATION_REQUEST_EXCHANGE_BINDING_MISMATCH")
    projection = TPMPublicProjectionV1.verify(req["public_projection"])
    _validate_activation_projection_binding(activation, projection)
    if (
        req["installation_id"] != activation.document["installation_id"]
        or req["release_policy_digest"] != activation.document["release"]["release_policy_digest"]
        or req["requested_entitlements"] != activation.document["requested_entitlements"]
    ):
        raise ValueError("ACTIVATION_REQUEST_EXCHANGE_BINDING_MISMATCH")
    if req["release_policy_digest"] != expected_release_policy_digest:
        raise ValueError("release policy binding mismatch")
    if (
        ch["request_id"] != req["request_id"]
        or ch["public_projection_id"] != req["public_projection"]["evidence_id"]
    ):
        raise ValueError("request/challenge binding mismatch")
    if (rsp["request_id"], rsp["challenge_id"], rsp["issuer_nonce_hex"]) != (
        req["request_id"],
        ch["challenge_id"],
        ch["issuer_nonce_hex"],
    ):
        raise ValueError("challenge/response binding mismatch")
    expires = datetime.strptime(ch["expires_at_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    if expires <= datetime.now(timezone.utc):
        raise ValueError("expired TPM enrollment challenge")
    return activation, request, challenge, response


def _verify_hardware_exchange(
    activation: ActivationRequestV1,
    request: TPMEnrollmentRequestV1,
    challenge: TPMEnrollmentChallengeV1,
    response: TPMEnrollmentChallengeResponseV1,
    pending: PendingChallengeStore,
) -> None:
    req, ch, rsp = request.document, challenge.document, response.document
    projection = TPMPublicProjectionV1.verify(req["public_projection"])
    secret = pending.credential_secret(challenge)
    if not hmac.compare_digest(
        hashlib.sha256(secret).hexdigest(), rsp["activated_credential_digest"]
    ):
        raise ValueError("ActivateCredential recovered credential mismatch")
    expected_proof = credential_activation_proof(secret, ch["challenge_id"])
    if not hmac.compare_digest(
        expected_proof, bytes.fromhex(rsp["credential_activation_proof_hex"])
    ):
        raise ValueError("ActivateCredential proof mismatch")
    activation_digest = hashlib.sha256(activation.canonical_bytes).digest()
    qualifying = certify_qualifying_data(bytes.fromhex(ch["issuer_nonce_hex"]), activation_digest)
    verify_certify_creation(
        attest=bytes.fromhex(rsp["certify_creation_attest_hex"]),
        signature=bytes.fromhex(rsp["certify_creation_signature_hex"]),
        ak_public_hex=projection.document["ak"]["public_area"]["hex"],
        expected_qualifying_data=qualifying,
        expected_name=bytes.fromhex(projection.document["k_psa"]["name"]),
        expected_creation_hash=bytes.fromhex(projection.document["k_psa"]["creation_hash"]),
    )
    verify_k_psa_proof_of_possession(
        projection,
        bytes.fromhex(ch["issuer_nonce_hex"]),
        bytes.fromhex(rsp["k_psa_pop_signature_der_hex"]),
    )


def _exchange_result(
    activation_raw: bytes, request_raw: bytes, challenge_raw: bytes, response_raw: bytes
) -> VerifiedTPMEnrollmentExchangeV1:
    digests = tuple(
        hashlib.sha256(raw).hexdigest()
        for raw in (activation_raw, request_raw, challenge_raw, response_raw)
    )
    reference = hashlib.sha256(
        EXCHANGE_REFERENCE_DOMAIN + b"".join(bytes.fromhex(item) for item in digests)
    ).hexdigest()
    return VerifiedTPMEnrollmentExchangeV1(*digests, reference)


class TestOnlyTPMAttestationVerifier:
    """Explicit unit-test harness; it is rejected by every runtime issuance boundary."""

    __test__ = False

    def verify_for_tests(
        self,
        activation_request_raw: bytes,
        request_raw: bytes,
        challenge_raw: bytes,
        response_raw: bytes,
        *,
        pending: PendingChallengeStore,
        expected_release_policy_digest: str,
    ) -> VerifiedTPMEnrollmentExchangeV1:
        activation, request, challenge, response = _parse_and_bind_exchange(
            activation_request_raw,
            request_raw,
            challenge_raw,
            response_raw,
            pending,
            expected_release_policy_digest,
        )
        _verify_hardware_exchange(activation, request, challenge, response, pending)
        return _exchange_result(activation_request_raw, request_raw, challenge_raw, response_raw)

    verify = verify_for_tests
