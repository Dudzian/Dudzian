"""Stage 9 external provisioning runtime (positive paths are TEST_ONLY).

The module implements the frozen authority split: PDSA authorizes, LPPI owns
``prvop`` and the CryptoHunter Account Authority (CHA) alone mints accounts.
SQLite is used through one transactional repository so uniqueness constraints,
not check-before-insert code, are the final idempotency fence.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.hazmat.primitives.asymmetric.utils import (
    decode_dss_signature,
    encode_dss_signature,
)

from .canonical import canonical_json_bytes

PDSA_DOMAIN = b"CryptoHunter.Stage9.PDSAEnrollmentAuthorization.v1\x00"
MEMBERSHIP_DOMAIN = b"CryptoHunter.Stage9.ProvisioningMembershipBinding.v1\x00"
PACKAGE_SCHEMA = "CryptoHunter.Stage9.PDSAEnrollmentAuthorizationPackage.v1"
P256_ORDER = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
HEX64 = set("0123456789abcdef")

PACKAGE_FIELDS = frozenset(
    {
        "schema_version",
        "environment",
        "pdsa_trust_domain",
        "provisioning_subject_id",
        "enrollment_reference",
        "pdsa_challenge_id",
        "pdsa_challenge_digest_sha256",
        "pre_enrollment_request_digest_sha256",
        "verified_tpm_exchange_reference",
        "verified_tpm_public_projection_id",
        "target_tpm_ek_public_digest",
        "target_tpm_ak_public_digest",
        "pre_enrollment_public_key_algorithm_profile",
        "pre_enrollment_public_key_fingerprint_sha256",
        "authorization_generation",
        "authorization_version",
        "issued_at_utc",
        "expires_at_utc",
        "release_policy_digest_sha256",
        "release_policy_generation",
        "product_profile",
        "predecessor_package_digest_or_null",
        "lineage_generation",
    }
)


class ProvisioningError(RuntimeError):
    """Fail-closed provisioning error carrying a stable reason code."""


class ConflictError(ProvisioningError):
    pass


class ProductionVerifierUnavailable(ProvisioningError):
    pass


class MembershipSignerUnavailable(ProvisioningError):
    pass


class SagaState(str, Enum):
    PREPARED = "PREPARED"
    ACCOUNT_COMMITTED = "ACCOUNT_COMMITTED"
    FIRST_DEVICE_COMMITTED = "FIRST_DEVICE_COMMITTED"
    MEMBERSHIP_COMMITTED = "MEMBERSHIP_COMMITTED"
    MATERIALIZED = "MATERIALIZED"
    CONSUMED = "CONSUMED"


@dataclass(frozen=True, slots=True)
class VerifiedProvisioningPackage:
    payload: Mapping[str, Any]
    package_digest_sha256: str
    canonical_package: bytes


@dataclass(frozen=True, slots=True)
class ProvisioningOutcome:
    provisioning_operation_id: str
    logical_operation_id: str
    account_id: str
    membership_digest_sha256: str
    state: SagaState


@dataclass(frozen=True, slots=True)
class PDSAChallengeV1:
    """Strict, immutable freshness challenge presented before authorization."""

    challenge_id: str
    nonce_hex: str
    pdsa_trust_domain: str
    issued_at_utc: str
    expires_at_utc: str

    def __post_init__(self) -> None:
        _bounded(self.challenge_id, "challenge_id")
        _bounded(self.pdsa_trust_domain, "pdsa_trust_domain")
        if len(self.nonce_hex) != 64 or any(c not in HEX64 for c in self.nonce_hex):
            raise ProvisioningError("INVALID_CHALLENGE_NONCE")
        if _iso(self.issued_at_utc) >= _iso(self.expires_at_utc):
            raise ProvisioningError("INVALID_CHALLENGE_WINDOW")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PDSAChallengeV1":
        fields = {
            "schema_version",
            "challenge_id",
            "nonce_hex",
            "pdsa_trust_domain",
            "issued_at_utc",
            "expires_at_utc",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != fields
            or value.get("schema_version") != "PDSAChallengeV1"
        ):
            raise ProvisioningError("CHALLENGE_SCHEMA_MISMATCH")
        return cls(**{name: value[name] for name in fields - {"schema_version"}})

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "schema_version": "PDSAChallengeV1",
                "challenge_id": self.challenge_id,
                "nonce_hex": self.nonce_hex,
                "pdsa_trust_domain": self.pdsa_trust_domain,
                "issued_at_utc": self.issued_at_utc,
                "expires_at_utc": self.expires_at_utc,
            }
        )


@dataclass(frozen=True, slots=True)
class PDSAAuthorizationRequestV1:
    """Strict digest-level request linking challenge, TPM exchange and device key."""

    pdsa_challenge_id: str
    pdsa_challenge_digest_sha256: str
    pre_enrollment_request_digest_sha256: str
    verified_tpm_exchange_reference: str
    pre_enrollment_public_key_fingerprint_sha256: str

    def __post_init__(self) -> None:
        _bounded(self.pdsa_challenge_id, "pdsa_challenge_id")
        _bounded(self.verified_tpm_exchange_reference, "verified_tpm_exchange_reference")
        for name in (
            "pdsa_challenge_digest_sha256",
            "pre_enrollment_request_digest_sha256",
            "pre_enrollment_public_key_fingerprint_sha256",
        ):
            _hex_digest(getattr(self, name), name)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PDSAAuthorizationRequestV1":
        fields = {
            "schema_version",
            "pdsa_challenge_id",
            "pdsa_challenge_digest_sha256",
            "pre_enrollment_request_digest_sha256",
            "verified_tpm_exchange_reference",
            "pre_enrollment_public_key_fingerprint_sha256",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != fields
            or value.get("schema_version") != "PDSAAuthorizationRequestV1"
        ):
            raise ProvisioningError("AUTHORIZATION_REQUEST_SCHEMA_MISMATCH")
        return cls(**{name: value[name] for name in fields - {"schema_version"}})

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "schema_version": "PDSAAuthorizationRequestV1",
                "pdsa_challenge_id": self.pdsa_challenge_id,
                "pdsa_challenge_digest_sha256": self.pdsa_challenge_digest_sha256,
                "pre_enrollment_request_digest_sha256": self.pre_enrollment_request_digest_sha256,
                "verified_tpm_exchange_reference": self.verified_tpm_exchange_reference,
                "pre_enrollment_public_key_fingerprint_sha256": self.pre_enrollment_public_key_fingerprint_sha256,
            }
        )


def _iso(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ProvisioningError("INVALID_CANONICAL_UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ProvisioningError("INVALID_CANONICAL_UTC") from exc
    if parsed.tzinfo != timezone.utc or parsed.isoformat().replace("+00:00", "Z") != value:
        raise ProvisioningError("INVALID_CANONICAL_UTC")
    return parsed


def _hex_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in HEX64 for c in value):
        raise ProvisioningError(f"INVALID_{name.upper()}")
    return value


def _bounded(value: object, name: str, limit: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value.encode()) > limit:
        raise ProvisioningError(f"INVALID_{name.upper()}")
    return value


def _uuid7(prefix: str) -> str:
    millis = int(time.time() * 1000) & ((1 << 48) - 1)
    raw = (millis << 80) | (0x7 << 76) | (secrets.randbits(12) << 64)
    raw |= (0b10 << 62) | secrets.randbits(62)
    return f"{prefix}_{uuid.UUID(int=raw)}"


def _strict_package(raw: bytes) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if len(raw) > 32_768:
        raise ProvisioningError("PACKAGE_TOO_LARGE")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProvisioningError("INVALID_PACKAGE_JSON") from exc
    if canonical_json_bytes(value) != raw or not isinstance(value, dict):
        raise ProvisioningError("NONCANONICAL_PACKAGE")
    if set(value) != {"payload", "signatures"}:
        raise ProvisioningError("PACKAGE_SCHEMA_MISMATCH")
    payload, signatures = value["payload"], value["signatures"]
    if not isinstance(payload, dict) or set(payload) != PACKAGE_FIELDS:
        raise ProvisioningError("PACKAGE_PAYLOAD_SCHEMA_MISMATCH")
    if payload["schema_version"] != PACKAGE_SCHEMA:
        raise ProvisioningError("PACKAGE_VERSION_MISMATCH")
    if not isinstance(signatures, list) or len(signatures) != 2:
        raise ProvisioningError("PDSA_THRESHOLD_NOT_MET")
    expected = {"algorithm", "key_id", "signature_hex"}
    if any(not isinstance(item, dict) or set(item) != expected for item in signatures):
        raise ProvisioningError("SIGNATURE_SCHEMA_MISMATCH")
    return payload, signatures


class ProvisioningPackageVerifier(ABC):
    @abstractmethod
    def verify(
        self, raw: bytes, *, expected_device_key: str, now: datetime
    ) -> VerifiedProvisioningPackage: ...


class ProvisioningMembershipSigner(ABC):
    """Authentication boundary for an enrolled LPPI membership key."""

    @property
    @abstractmethod
    def identity(self) -> str: ...

    @property
    @abstractmethod
    def profile(self) -> str: ...

    @abstractmethod
    def sign(self, message: bytes) -> bytes: ...

    @abstractmethod
    def verify(self, message: bytes, signature: bytes) -> None: ...


class TestOnlyMembershipSigner(ProvisioningMembershipSigner):
    """Deterministic P-256 fixture available only through explicit test composition."""

    __test__ = False
    _PROFILE = "TEST_ONLY_ECDSA_P256_SHA256_LOW_S"

    def __init__(self, scalar: int = 1) -> None:
        if type(scalar) is not int or not 1 <= scalar < P256_ORDER:
            raise ValueError("invalid TEST_ONLY P-256 scalar")
        self._key = ec.derive_private_key(scalar, ec.SECP256R1())

    @property
    def identity(self) -> str:
        public = self._key.public_key().public_bytes(
            serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
        )
        return "TEST_ONLY_" + hashlib.sha256(public).hexdigest()

    @property
    def profile(self) -> str:
        return self._PROFILE

    def sign(self, message: bytes) -> bytes:
        encoded = self._key.sign(message, ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(encoded)
        return encode_dss_signature(r, min(s, P256_ORDER - s))

    def verify(self, message: bytes, signature: bytes) -> None:
        try:
            r, s = decode_dss_signature(signature)
            if encode_dss_signature(r, s) != signature:
                raise ProvisioningError("MEMBERSHIP_SIGNATURE_NONCANONICAL_DER")
            if not 1 <= r < P256_ORDER or not 1 <= s <= P256_ORDER // 2:
                raise ProvisioningError("MEMBERSHIP_SIGNATURE_NONCANONICAL_LOW_S")
            self._key.public_key().verify(signature, message, ec.ECDSA(hashes.SHA256()))
        except (InvalidSignature, ValueError) as exc:
            raise ProvisioningError("MEMBERSHIP_SIGNATURE_INVALID") from exc


class ProductionMembershipSignerUnavailable(ProvisioningMembershipSigner):
    """No production key exists until legal TPM successor-key enrollment."""

    @property
    def identity(self) -> str:
        raise MembershipSignerUnavailable("LEGAL_ENROLLMENT_NOT_COMPLETED")

    @property
    def profile(self) -> str:
        raise MembershipSignerUnavailable("LEGAL_ENROLLMENT_NOT_COMPLETED")

    def sign(self, message: bytes) -> bytes:
        del message
        raise MembershipSignerUnavailable("LEGAL_ENROLLMENT_NOT_COMPLETED")

    def verify(self, message: bytes, signature: bytes) -> None:
        del message, signature
        raise MembershipSignerUnavailable("LEGAL_ENROLLMENT_NOT_COMPLETED")


class ProductionProvisioningPackageVerifier(ProvisioningPackageVerifier):
    """Production stays unavailable until the offline ceremony activates trust."""

    def verify(
        self, raw: bytes, *, expected_device_key: str, now: datetime
    ) -> VerifiedProvisioningPackage:
        del raw, expected_device_key, now
        raise ProductionVerifierUnavailable(
            "CEREMONY_NOT_COMPLETED: PRODUCTION_VERIFIER_UNAVAILABLE"
        )


class TestOnlyProvisioningAuthority:
    """Deterministic 2-of-3 Ed25519 fixture; cannot be promoted to production."""

    __test__ = False

    def __init__(self) -> None:
        self._keys = {
            f"TEST_ONLY_PDSA_{i}": ed25519.Ed25519PrivateKey.from_private_bytes(
                hashlib.sha256(f"Stage9 TEST_ONLY authority {i}".encode()).digest()
            )
            for i in range(1, 4)
        }

    @property
    def public_keys(self) -> dict[str, ed25519.Ed25519PublicKey]:
        return {key_id: key.public_key() for key_id, key in self._keys.items()}

    def issue(self, payload: Mapping[str, Any]) -> bytes:
        if set(payload) != PACKAGE_FIELDS or payload.get("schema_version") != PACKAGE_SCHEMA:
            raise ProvisioningError("PACKAGE_PAYLOAD_SCHEMA_MISMATCH")
        message = PDSA_DOMAIN + hashlib.sha256(canonical_json_bytes(payload)).digest()
        signatures = [
            {
                "algorithm": "Ed25519",
                "key_id": key_id,
                "signature_hex": self._keys[key_id].sign(message).hex(),
            }
            for key_id in sorted(self._keys)[:2]
        ]
        return canonical_json_bytes({"payload": dict(payload), "signatures": signatures})


class TestOnlyProvisioningPackageVerifier(ProvisioningPackageVerifier):
    __test__ = False

    def __init__(self, keys: Mapping[str, ed25519.Ed25519PublicKey]) -> None:
        if len(keys) != 3 or any(not key.startswith("TEST_ONLY_") for key in keys):
            raise ValueError("TEST_ONLY verifier requires three labelled keys")
        self._keys = dict(keys)

    def verify(
        self, raw: bytes, *, expected_device_key: str, now: datetime
    ) -> VerifiedProvisioningPackage:
        payload, signatures = _strict_package(raw)
        for field in PACKAGE_FIELDS - {"predecessor_package_digest_or_null"}:
            if payload[field] is None:
                raise ProvisioningError(f"NULL_{field.upper()}")
        for field in (
            "pdsa_challenge_digest_sha256",
            "pre_enrollment_request_digest_sha256",
            "target_tpm_ek_public_digest",
            "target_tpm_ak_public_digest",
            "pre_enrollment_public_key_fingerprint_sha256",
            "release_policy_digest_sha256",
        ):
            _hex_digest(payload[field], field)
        for field in ("provisioning_subject_id", "enrollment_reference", "pdsa_challenge_id"):
            _bounded(payload[field], field)
        expected_literals = {
            "environment": "TEST_ONLY",
            "pdsa_trust_domain": "TEST_ONLY_2_OF_3_ED25519",
            "product_profile": "TEST_ONLY",
        }
        for field, expected in expected_literals.items():
            if payload[field] != expected:
                raise ProvisioningError(f"INVALID_{field.upper()}")
        for field in (
            "verified_tpm_exchange_reference",
            "verified_tpm_public_projection_id",
        ):
            _bounded(payload[field], field)
        for field in (
            "authorization_generation",
            "authorization_version",
            "release_policy_generation",
            "lineage_generation",
        ):
            value = payload[field]
            if type(value) is not int or not 1 <= value <= 2**63 - 1:
                raise ProvisioningError(f"INVALID_{field.upper()}")
        predecessor = payload["predecessor_package_digest_or_null"]
        if predecessor is not None:
            _hex_digest(predecessor, "predecessor_package_digest_or_null")
        if payload["pre_enrollment_public_key_algorithm_profile"] != "ECDSA-P256-SHA256":
            raise ProvisioningError("WRONG_DEVICE_KEY_ALGORITHM")
        if payload["pre_enrollment_public_key_fingerprint_sha256"] != expected_device_key:
            raise ProvisioningError("REJECT_PACKAGE_TARGET_MISMATCH")
        issued_at = _iso(payload["issued_at_utc"])
        expires_at = _iso(payload["expires_at_utc"])
        if issued_at >= expires_at:
            raise ProvisioningError("INVALID_PACKAGE_VALIDITY_WINDOW")
        if now.tzinfo != timezone.utc or now < issued_at:
            raise ProvisioningError("PACKAGE_NOT_YET_VALID")
        if now >= expires_at:
            raise ProvisioningError("PACKAGE_EXPIRED")
        message = PDSA_DOMAIN + hashlib.sha256(canonical_json_bytes(payload)).digest()
        seen: set[str] = set()
        try:
            for signature in signatures:
                key_id = signature["key_id"]
                if (
                    signature["algorithm"] != "Ed25519"
                    or key_id in seen
                    or key_id not in self._keys
                ):
                    raise ProvisioningError("UNKNOWN_OR_DUPLICATE_PDSA_SIGNER")
                seen.add(key_id)
                self._keys[key_id].verify(bytes.fromhex(signature["signature_hex"]), message)
        except (InvalidSignature, ValueError) as exc:
            raise ProvisioningError("INVALID_PDSA_SIGNATURE") from exc
        return VerifiedProvisioningPackage(payload, hashlib.sha256(raw).hexdigest(), raw)


class ProvisioningRepository:
    """Transactional durable store for LPPI, CHA mapping and membership entities."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS provisioning_operations (
                  prvop TEXT PRIMARY KEY, package_digest TEXT NOT NULL UNIQUE,
                  subject_id TEXT NOT NULL, enrollment_reference TEXT NOT NULL,
                  device_key TEXT NOT NULL, state TEXT NOT NULL, logical_operation_id TEXT UNIQUE,
                  account_id TEXT UNIQUE, membership_digest TEXT UNIQUE,
                  created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS consumed_authorizations (
                  enrollment_reference TEXT PRIMARY KEY, prvop TEXT NOT NULL UNIQUE
                    REFERENCES provisioning_operations(prvop));
                CREATE TABLE IF NOT EXISTS cha_genesis (
                  logical_operation_id TEXT PRIMARY KEY, prvop TEXT NOT NULL UNIQUE,
                  binding_digest TEXT NOT NULL, account_id TEXT NOT NULL UNIQUE);
                CREATE TABLE IF NOT EXISTS first_device_memberships (
                  prvop TEXT PRIMARY KEY, account_id TEXT NOT NULL UNIQUE,
                  subject_id TEXT NOT NULL, package_digest TEXT NOT NULL,
                  enrollment_reference TEXT NOT NULL, device_key TEXT NOT NULL,
                  logical_operation_id TEXT NOT NULL UNIQUE, payload BLOB NOT NULL,
                  signature BLOB NOT NULL, membership_digest TEXT NOT NULL UNIQUE,
                  signer_identity TEXT NOT NULL, signer_profile TEXT NOT NULL,
                  created_at TEXT NOT NULL);
            """)

    def reserve(self, package: VerifiedProvisioningPackage, now: str) -> sqlite3.Row:
        p = package.payload
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT * FROM provisioning_operations WHERE package_digest=?",
                (package.package_digest_sha256,),
            ).fetchone()
            if existing is not None:
                self._assert_binding(existing, package)
                db.commit()
                return existing
            prvop = _uuid7("prvop")
            try:
                db.execute(
                    "INSERT INTO provisioning_operations VALUES(?,?,?,?,?,'PREPARED',NULL,NULL,NULL,?,?)",
                    (
                        prvop,
                        package.package_digest_sha256,
                        p["provisioning_subject_id"],
                        p["enrollment_reference"],
                        p["pre_enrollment_public_key_fingerprint_sha256"],
                        now,
                        now,
                    ),
                )
                db.execute(
                    "INSERT INTO consumed_authorizations VALUES(?,?)",
                    (p["enrollment_reference"], prvop),
                )
            except sqlite3.IntegrityError as exc:
                db.rollback()
                raise ConflictError("AUTHORIZATION_ALREADY_CONSUMED") from exc
            row = db.execute(
                "SELECT * FROM provisioning_operations WHERE prvop=?", (prvop,)
            ).fetchone()
            db.commit()
            assert row is not None
            return row

    @staticmethod
    def _assert_binding(row: sqlite3.Row, package: VerifiedProvisioningPackage) -> None:
        p = package.payload
        actual = (
            row["package_digest"],
            row["subject_id"],
            row["enrollment_reference"],
            row["device_key"],
        )
        expected = (
            package.package_digest_sha256,
            p["provisioning_subject_id"],
            p["enrollment_reference"],
            p["pre_enrollment_public_key_fingerprint_sha256"],
        )
        if actual != expected:
            raise ConflictError("PROVISIONING_OPERATION_BINDING_CONFLICT")

    def operation(self, prvop: str) -> sqlite3.Row:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM provisioning_operations WHERE prvop=?", (prvop,)
            ).fetchone()
        if row is None:
            raise ProvisioningError("UNKNOWN_PROVISIONING_OPERATION")
        return row

    def bind_account(self, prvop: str, logical: str, account: str, now: str) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM provisioning_operations WHERE prvop=?", (prvop,)
            ).fetchone()
            if row is None:
                raise ConflictError("ILLEGAL_ACCOUNT_TRANSITION")
            if row["account_id"] not in (None, account) or row["logical_operation_id"] not in (
                None,
                logical,
            ):
                raise ConflictError("ACCOUNT_OUTCOME_CONFLICT")
            if row["state"] == "PREPARED":
                db.execute(
                    "UPDATE provisioning_operations SET logical_operation_id=?,account_id=?,state='ACCOUNT_COMMITTED',updated_at=? WHERE prvop=?",
                    (logical, account, now, prvop),
                )
            db.commit()

    def transition(
        self,
        prvop: str,
        before: SagaState,
        after: SagaState,
        now: str,
        membership: str | None = None,
    ) -> None:
        progression = tuple(SagaState)
        if progression.index(after) != progression.index(before) + 1:
            raise ConflictError("ILLEGAL_SAGA_TRANSITION")
        with self._connect() as db:
            values: tuple[Any, ...]
            if membership:
                sql = "UPDATE provisioning_operations SET state=?,membership_digest=?,updated_at=? WHERE prvop=? AND state=?"
                values = (after.value, membership, now, prvop, before.value)
            else:
                sql = "UPDATE provisioning_operations SET state=?,updated_at=? WHERE prvop=? AND state=?"
                values = (after.value, now, prvop, before.value)
            changed = db.execute(sql, values).rowcount
            if changed != 1:
                current = db.execute(
                    "SELECT state FROM provisioning_operations WHERE prvop=?", (prvop,)
                ).fetchone()
                if current is None or progression.index(SagaState(current[0])) < progression.index(
                    after
                ):
                    raise ConflictError("ILLEGAL_SAGA_TRANSITION")


class Stage9AccountGenesisAuthority:
    """The sole account-id minting boundary."""

    def __init__(self, repository: ProvisioningRepository) -> None:
        self.repository = repository

    def genesis(self, operation: sqlite3.Row) -> tuple[str, str]:
        binding = hashlib.sha256(
            "\x00".join(
                (
                    operation["prvop"],
                    operation["package_digest"],
                    operation["subject_id"],
                    operation["device_key"],
                )
            ).encode()
        ).hexdigest()
        with self.repository._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute(
                "SELECT * FROM cha_genesis WHERE prvop=?", (operation["prvop"],)
            ).fetchone()
            if old:
                if old["binding_digest"] != binding:
                    raise ConflictError("CHA_LOGICAL_OPERATION_CONFLICT")
                db.commit()
                return old["logical_operation_id"], old["account_id"]
            logical, account = _uuid7("ago"), _uuid7("acct")
            db.execute(
                "INSERT INTO cha_genesis VALUES(?,?,?,?)",
                (logical, operation["prvop"], binding, account),
            )
            db.commit()
            return logical, account


class Stage9ProvisioningService:
    """Recoverable LPPI saga. Verification occurs before any repository access."""

    def __init__(
        self,
        repository: ProvisioningRepository,
        verifier: ProvisioningPackageVerifier,
        membership_signer: ProvisioningMembershipSigner,
        *,
        crash_hook: Callable[[str], None] | None = None,
    ) -> None:
        if isinstance(verifier, ProductionProvisioningPackageVerifier) and isinstance(
            membership_signer, TestOnlyMembershipSigner
        ):
            raise ValueError("TEST_ONLY membership signer forbidden with production verifier")
        self.repository, self.verifier = repository, verifier
        self.membership_signer = membership_signer
        self.cha = Stage9AccountGenesisAuthority(repository)
        self.crash_hook = crash_hook or (lambda _: None)

    def provision(
        self, raw: bytes, *, expected_device_key: str, now: datetime
    ) -> ProvisioningOutcome:
        verified = self.verifier.verify(raw, expected_device_key=expected_device_key, now=now)
        stamp = now.isoformat().replace("+00:00", "Z")
        operation = self.repository.reserve(verified, stamp)
        self.crash_hook("AFTER_PRVOP_BEFORE_CHA")
        return self._resume(operation["prvop"], stamp)

    def _resume(self, prvop: str, stamp: str) -> ProvisioningOutcome:
        op = self.repository.operation(prvop)
        if tuple(SagaState).index(SagaState(op["state"])) >= tuple(SagaState).index(
            SagaState.MEMBERSHIP_COMMITTED
        ):
            self._verify_membership(op)
        if op["state"] == SagaState.PREPARED.value:
            logical, account = self.cha.genesis(op)
            self.crash_hook("AFTER_CHA_BEFORE_ACCOUNT_OBSERVED")
            self.repository.bind_account(prvop, logical, account, stamp)
            op = self.repository.operation(prvop)
        if op["state"] == SagaState.ACCOUNT_COMMITTED.value:
            self.crash_hook("AFTER_ACCOUNT_BEFORE_MEMBERSHIP")
            self.repository.transition(
                prvop, SagaState.ACCOUNT_COMMITTED, SagaState.FIRST_DEVICE_COMMITTED, stamp
            )
            op = self.repository.operation(prvop)
        if op["state"] == SagaState.FIRST_DEVICE_COMMITTED.value:
            membership = self._membership(op, stamp)
            self.repository.transition(
                prvop,
                SagaState.FIRST_DEVICE_COMMITTED,
                SagaState.MEMBERSHIP_COMMITTED,
                stamp,
                membership,
            )
            self.crash_hook("AFTER_MEMBERSHIP_BEFORE_COMPLETED")
            op = self.repository.operation(prvop)
        if op["state"] == SagaState.MEMBERSHIP_COMMITTED.value:
            self._verify_membership(op)
            self.repository.transition(
                prvop, SagaState.MEMBERSHIP_COMMITTED, SagaState.MATERIALIZED, stamp
            )
            op = self.repository.operation(prvop)
        if op["state"] == SagaState.MATERIALIZED.value:
            self._verify_membership(op)
            self.repository.transition(prvop, SagaState.MATERIALIZED, SagaState.CONSUMED, stamp)
            op = self.repository.operation(prvop)
        if op["state"] != SagaState.CONSUMED.value:
            raise ProvisioningError("UNRECOVERABLE_SAGA_STATE")
        self._verify_membership(op)
        return ProvisioningOutcome(
            prvop,
            op["logical_operation_id"],
            op["account_id"],
            op["membership_digest"],
            SagaState.CONSUMED,
        )

    @staticmethod
    def _membership_payload(op: sqlite3.Row, issued_at: str) -> bytes:
        return canonical_json_bytes(
            {
                "schema_version": "ProvisioningMembershipBinding.v1",
                "environment": "TEST_ONLY",
                "trust_domain": "TEST_ONLY",
                "provisioning_subject_id": op["subject_id"],
                "provisioning_operation_id": op["prvop"],
                "logical_operation_id": op["logical_operation_id"],
                "account_id": op["account_id"],
                "device_installation_id": op["device_key"],
                "claim_fingerprint_sha256": op["package_digest"],
                "authority_source": "LPPI_TEST_ONLY",
                "issuance_generation": 1,
                "issued_at_utc": issued_at,
                "expires_at_utc_or_null": None,
                "predecessor_digest_or_null": None,
            }
        )

    def _verify_membership(self, op: sqlite3.Row) -> str:
        with self.repository._connect() as db:
            old = db.execute(
                "SELECT * FROM first_device_memberships WHERE prvop=?", (op["prvop"],)
            ).fetchone()
        if old is None:
            raise ProvisioningError("MEMBERSHIP_MISSING")
        payload = bytes(old["payload"])
        expected_payload = self._membership_payload(op, old["created_at"])
        stored_digest = old["membership_digest"]
        computed_digest = hashlib.sha256(payload).hexdigest()
        relational_projection = (
            old["prvop"],
            old["account_id"],
            old["subject_id"],
            old["package_digest"],
            old["enrollment_reference"],
            old["device_key"],
            old["logical_operation_id"],
        )
        canonical_operation = (
            op["prvop"],
            op["account_id"],
            op["subject_id"],
            op["package_digest"],
            op["enrollment_reference"],
            op["device_key"],
            op["logical_operation_id"],
        )
        if (
            relational_projection != canonical_operation
            or payload != expected_payload
            or stored_digest != computed_digest
            or op["membership_digest"] not in (None, stored_digest)
            or old["signer_identity"] != self.membership_signer.identity
            or old["signer_profile"] != self.membership_signer.profile
        ):
            raise ConflictError("MEMBERSHIP_AUTHENTICATION_BINDING_CONFLICT")
        self.membership_signer.verify(
            MEMBERSHIP_DOMAIN + bytes.fromhex(stored_digest), bytes(old["signature"])
        )
        return stored_digest

    def _membership(self, op: sqlite3.Row, stamp: str) -> str:
        payload = self._membership_payload(op, stamp)
        digest = hashlib.sha256(payload).hexdigest()
        message = MEMBERSHIP_DOMAIN + bytes.fromhex(digest)
        with self.repository._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute(
                "SELECT * FROM first_device_memberships WHERE prvop=?", (op["prvop"],)
            ).fetchone()
            if old:
                db.commit()
                return self._verify_membership(op)
            signature = self.membership_signer.sign(message)
            self.membership_signer.verify(message, signature)
            try:
                db.execute(
                    "INSERT INTO first_device_memberships VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        op["prvop"],
                        op["account_id"],
                        op["subject_id"],
                        op["package_digest"],
                        op["enrollment_reference"],
                        op["device_key"],
                        op["logical_operation_id"],
                        payload,
                        signature,
                        digest,
                        self.membership_signer.identity,
                        self.membership_signer.profile,
                        stamp,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                db.rollback()
                raise ConflictError("MEMBERSHIP_CONSTRAINT_CONFLICT") from exc
            db.commit()
        return digest
