"""Durable, local CHA root-proof issuance attempt storage.

This adapter implements only the frozen local persistence boundary.  It does
not decide entitlement state, validate or issue root proofs, mint accounts, or
turn missing issuer data into authoritative evidence.  The database must live
on a controlled host filesystem with working SQLite locking; shared/network
filesystems and protection from a privileged host administrator are outside
the supported threat boundary.
"""

from __future__ import annotations

from contextlib import contextmanager
import base64
import binascii
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import time
from typing import Iterator

from bot_core.root_proof_issuer_substrate import (
    CredentialRoleIdentity,
    ProviderCapabilities,
    ProviderIdentity,
    ProviderRole,
    SecurityProfile,
    SecurityProfileIdentity,
)


_SCHEMA_IDENTITY = "CRYPT0HUNTER_CHA_ATTEMPT_STORE"
_SCHEMA_VERSION = 3
_ATTEMPT_DOMAIN = b"CRYPTOHUNTER_ACCOUNT_GENESIS_ROOT_PROOF_ISSUANCE_ATTEMPT_IDENTITY_V1\x00"
_IDEMPOTENCY_DOMAIN = b"CRYPTOHUNTER_CHA_ATTEMPT_RESERVATION_IDEMPOTENCY_V1\x00"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_RPA_ID = re.compile(
    r"rpa_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"
)
_ACCOUNT_ID = re.compile(
    r"acct_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"
)
_REQUESTER_ROLE = "ACCOUNT_GENESIS_ROOT_PROOF_ISSUANCE_REQUESTER_V1"
_ISSUANCE_PROFILE = (
    "CRYPTOHUNTER_ACCOUNT_GENESIS_ROOT_PROOF_ISSUANCE_REQUEST_V1/"
    "JCS-SHA256-Ed25519-v1"
)
_CLAIMANT_PROFILE = (
    "CRYPTOHUNTER_ACCOUNT_GENESIS_ROOT_PROOF_ENTITLEMENT_CLAIM_V1/"
    "JCS-SHA256-Ed25519-v1"
)


class AttemptStoreError(RuntimeError):
    """Base class for typed, fail-closed store failures."""


class AttemptNotFoundError(AttemptStoreError):
    pass


class AttemptConflictError(AttemptStoreError):
    pass


class AttemptCorruptError(AttemptStoreError):
    pass


class AttemptStoreUnavailableError(AttemptStoreError):
    pass


class AttemptSchemaUnsupportedError(AttemptStoreError):
    pass


class AttemptState(str, Enum):
    RESERVED_AWAITING_SIGNATURES = "RESERVED_AWAITING_SIGNATURES"
    SIGNED_IMMUTABLE_DURABLE_NOT_SENT = "SIGNED_IMMUTABLE_DURABLE_NOT_SENT"
    MAY_HAVE_BEEN_SENT_OUTCOME_UNKNOWN = "MAY_HAVE_BEEN_SENT_OUTCOME_UNKNOWN"
    EXACT_BOUND_RECOVERED = "EXACT_BOUND_RECOVERED"
    SUPERSEDED_AFTER_AUTHORITATIVE_UNBOUND_RECONCILIATION = (
        "SUPERSEDED_AFTER_AUTHORITATIVE_UNBOUND_RECONCILIATION"
    )


@dataclass(frozen=True, slots=True)
class AttemptAuthorization:
    environment: str
    trust_domain: str
    logical_operation_id: str
    account_id: str
    canonical_genesis_request_fingerprint_sha256: str
    initial_binding_reference: str
    initial_binding_digest_sha256: str
    bootstrap_entitlement_id: str
    entitlement_generation: int
    requester_principal_id: str
    requester_credential_role: str
    requester_key_id: str
    requester_key_version: int
    provisioning_principal_id: str
    claimant_key_id: str
    claimant_key_version: int

    def __post_init__(self) -> None:
        if type(self.account_id) is not str or _ACCOUNT_ID.fullmatch(self.account_id) is None:
            raise ValueError("account_id must be a canonical lowercase UUIDv7 acct id")
        if type(self.requester_credential_role) is not str or self.requester_credential_role != _REQUESTER_ROLE:
            raise ValueError("requester_credential_role is not the frozen value")
        for name, value in asdict(self).items():
            if name.endswith("_sha256"):
                if type(value) is not str or _HEX64.fullmatch(value) is None:
                    raise ValueError(f"{name} must be lowercase SHA-256 hex")
            elif name.endswith("_generation") or name.endswith("_version"):
                if type(value) is not int or value < 1:
                    raise ValueError(f"{name} must be a positive exact int")
            elif type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be a non-empty exact str")


@dataclass(frozen=True, slots=True)
class AttemptReservation:
    authorization: AttemptAuthorization
    exact_idempotency_key: str
    issuance_attempt_id: str
    reservation_state: AttemptState = AttemptState.RESERVED_AWAITING_SIGNATURES

    def __post_init__(self) -> None:
        if type(self.authorization) is not AttemptAuthorization:
            raise TypeError("authorization must be an exact AttemptAuthorization")
        if type(self.exact_idempotency_key) is not str or _HEX64.fullmatch(self.exact_idempotency_key) is None:
            raise ValueError("idempotency key must be lowercase SHA-256 hex")
        if type(self.issuance_attempt_id) is not str or _RPA_ID.fullmatch(self.issuance_attempt_id) is None:
            raise ValueError("issuance_attempt_id must be a canonical lowercase UUIDv7 rpa id")
        if type(self.reservation_state) is not AttemptState:
            raise TypeError("reservation_state must be an exact AttemptState")


@dataclass(frozen=True, slots=True)
class AttemptIdentity:
    authorization: AttemptAuthorization
    issuance_attempt_id: str
    root_proof_issuance_request_signed_payload_digest_sha256: str
    root_proof_issuance_request_canonical_bytes_reference: str
    requester_signature_base64url: str
    claimant_authorization_signature_base64url: str
    issuance_request_domain_and_profile_version: str
    claimant_authorization_domain_and_profile_version: str

    def __post_init__(self) -> None:
        if type(self.authorization) is not AttemptAuthorization:
            raise TypeError("authorization must be an exact AttemptAuthorization")
        if _RPA_ID.fullmatch(self.issuance_attempt_id) is None:
            raise ValueError("issuance_attempt_id must be a canonical lowercase UUIDv7 rpa id")
        if _HEX64.fullmatch(
            self.root_proof_issuance_request_signed_payload_digest_sha256
        ) is None:
            raise ValueError("signed payload digest must be lowercase SHA-256 hex")
        if (
            type(self.issuance_request_domain_and_profile_version) is not str
            or self.issuance_request_domain_and_profile_version != _ISSUANCE_PROFILE
        ):
            raise ValueError("issuance request domain/profile is not the frozen value")
        if (
            type(self.claimant_authorization_domain_and_profile_version) is not str
            or self.claimant_authorization_domain_and_profile_version != _CLAIMANT_PROFILE
        ):
            raise ValueError("claimant authorization domain/profile is not the frozen value")
        for signature in (
            self.requester_signature_base64url,
            self.claimant_authorization_signature_base64url,
        ):
            _validate_ed25519_signature_representation(signature)
        for value in (
            self.root_proof_issuance_request_canonical_bytes_reference,
        ):
            if type(value) is not str or not value.strip():
                raise ValueError("attempt identity string fields cannot be blank")

    def payload(self) -> dict[str, str | int]:
        result: dict[str, str | int] = {
            "schema_version": "1",
            **asdict(self.authorization),
            "issuance_attempt_id": self.issuance_attempt_id,
            "root_proof_issuance_request_signed_payload_digest_sha256": self.root_proof_issuance_request_signed_payload_digest_sha256,
            "root_proof_issuance_request_canonical_bytes_reference": self.root_proof_issuance_request_canonical_bytes_reference,
            "requester_signature_base64url": self.requester_signature_base64url,
            "claimant_authorization_signature_base64url": self.claimant_authorization_signature_base64url,
            "issuance_request_domain_and_profile_version": self.issuance_request_domain_and_profile_version,
            "claimant_authorization_domain_and_profile_version": self.claimant_authorization_domain_and_profile_version,
        }
        return result

    @property
    def digest_sha256(self) -> str:
        return hashlib.sha256(_ATTEMPT_DOMAIN + _canonical(self.payload())).hexdigest()


@dataclass(frozen=True, slots=True)
class AuthoritativeUnboundEvidence:
    """Evidence already verified by the trusted reconciliation layer.

    The store validates exact bindings and persists the evidence identity.  It
    does not cryptographically authenticate raw issuer evidence and therefore
    does not become an issuer or reconciliation authority.
    """
    schema_version: str
    environment: str
    trust_domain: str
    issuer_authority_identity: str
    issuer_registry_identity: str
    bootstrap_entitlement_id: str
    entitlement_generation: int
    logical_operation_id: str
    account_id: str
    canonical_genesis_request_fingerprint_sha256: str
    old_issuance_attempt_id: str
    initial_binding_reference: str
    initial_binding_digest_sha256: str
    outcome: str
    authoritative_state_identity: str
    authoritative_state_revision: str
    authority_authenticated_evidence_reference: str
    authority_authenticated_evidence_digest_sha256: str
    verification_profile_version: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != "1":
            raise ValueError("evidence schema_version is not the frozen value")
        if type(self.outcome) is not str or self.outcome != "AUTHORITATIVELY_UNBOUND":
            raise ValueError("only positive authoritative UNBOUND evidence permits replacement")
        if type(self.old_issuance_attempt_id) is not str or _RPA_ID.fullmatch(self.old_issuance_attempt_id) is None:
            raise ValueError("old attempt id is malformed")
        if _HEX64.fullmatch(self.authority_authenticated_evidence_digest_sha256) is None:
            raise ValueError("evidence digest must be lowercase SHA-256 hex")
        if _HEX64.fullmatch(self.canonical_genesis_request_fingerprint_sha256) is None:
            raise ValueError("request fingerprint must be lowercase SHA-256 hex")
        if _HEX64.fullmatch(self.initial_binding_digest_sha256) is None:
            raise ValueError("initial binding digest must be lowercase SHA-256 hex")
        if type(self.account_id) is not str or _ACCOUNT_ID.fullmatch(self.account_id) is None:
            raise ValueError("evidence account_id is malformed")
        if type(self.entitlement_generation) is not int or self.entitlement_generation < 1:
            raise ValueError("entitlement generation must be a positive exact int")
        for value in (
            self.schema_version,
            self.environment,
            self.trust_domain,
            self.issuer_authority_identity,
            self.issuer_registry_identity,
            self.bootstrap_entitlement_id,
            self.logical_operation_id,
            self.account_id,
            self.initial_binding_reference,
            self.authoritative_state_identity,
            self.authoritative_state_revision,
            self.authority_authenticated_evidence_reference,
            self.verification_profile_version,
        ):
            if type(value) is not str or not value.strip():
                raise ValueError("evidence fields cannot be blank")


@dataclass(frozen=True, slots=True)
class RecoveryResolution:
    issuance_attempt_id: str
    outcome: AttemptState
    authenticated_reference: str
    authenticated_digest_sha256: str

    def __post_init__(self) -> None:
        if type(self.outcome) is not AttemptState:
            raise TypeError("recovery outcome must be an exact AttemptState")
        if self.outcome not in {
            AttemptState.MAY_HAVE_BEEN_SENT_OUTCOME_UNKNOWN,
            AttemptState.EXACT_BOUND_RECOVERED,
        }:
            raise ValueError("recovery cannot establish authoritative UNBOUND")
        if type(self.issuance_attempt_id) is not str or _RPA_ID.fullmatch(self.issuance_attempt_id) is None:
            raise ValueError("attempt id is malformed")
        if type(self.authenticated_reference) is not str or not self.authenticated_reference.strip() or type(self.authenticated_digest_sha256) is not str or _HEX64.fullmatch(
            self.authenticated_digest_sha256
        ) is None:
            raise ValueError("authenticated recovery evidence is malformed")


@dataclass(frozen=True, slots=True)
class CurrentAttempt:
    reservation: AttemptReservation
    state: AttemptState
    fence: int
    immutable_attempt_digest_sha256: str | None
    identity: AttemptIdentity | None

    def __post_init__(self) -> None:
        if type(self.reservation) is not AttemptReservation:
            raise TypeError("reservation must be exact AttemptReservation")
        if type(self.state) is not AttemptState:
            raise TypeError("state must be exact AttemptState")
        if type(self.fence) is not int or self.fence < 1:
            raise ValueError("fence must be a positive exact int")


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _validate_ed25519_signature_representation(value: object) -> None:
    if type(value) is not str or "=" in value or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise ValueError("signature must be unpadded canonical base64url")
    try:
        decoded = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("signature must be unpadded canonical base64url") from exc
    if len(decoded) != 64 or base64.urlsafe_b64encode(decoded).rstrip(b"=").decode() != value:
        raise ValueError("signature must canonically encode one 64-byte Ed25519 signature")


def _idempotency_key(auth: AttemptAuthorization) -> str:
    return hashlib.sha256(_IDEMPOTENCY_DOMAIN + _canonical(asdict(auth))).hexdigest()


def _decision_key(kind: str, value: object) -> str:
    return hashlib.sha256(
        b"CRYPTOHUNTER_CHA_ATTEMPT_DECISION_V1\x00"
        + kind.encode("ascii")
        + b"\x00"
        + _canonical(value)
    ).hexdigest()


def _new_rpa_id() -> str:
    millis = int(time.time() * 1000) & ((1 << 48) - 1)
    random_bits = secrets.randbits(74)
    value = (millis << 80) | (0x7 << 76) | ((random_bits >> 62) << 64)
    value |= 0b10 << 62
    value |= random_bits & ((1 << 62) - 1)
    raw = f"{value:032x}"
    return f"rpa_{raw[:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:]}"


class SQLiteCHAAttemptStore:
    """Dedicated PRODUCTION_LOCAL SQLite implementation of the CHA store."""

    def __init__(self, path: Path, trust_domain: str, *, timeout: float = 5.0) -> None:
        if not isinstance(path, Path) or str(path) == ":memory:" or not path.is_absolute():
            raise ValueError("attempt database must be an absolute dedicated file path")
        if type(trust_domain) is not str or not trust_domain.strip():
            raise ValueError("trust_domain must be a non-empty exact str")
        self._path = path
        self._security = SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, trust_domain)
        try:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._connection = sqlite3.connect(path, timeout=timeout, isolation_level=None)
            if os.name == "posix":
                os.chmod(path, 0o600)
            self._configure()
            self._open_schema()
        except (OSError, sqlite3.Error) as exc:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            raise AttemptStoreUnavailableError("unable to open durable CHA attempt store") from exc

    @property
    def identity(self) -> ProviderIdentity:
        return ProviderIdentity(ProviderRole.CHA_ATTEMPT_STORE, self._security, "cha-attempt-sqlite-v1")

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(True, authoritative_reads=True, durable_state=True, compare_and_swap=True)

    def credential_identities(self) -> tuple[CredentialRoleIdentity, ...]:
        return ()

    @property
    def effective_pragmas(self) -> tuple[str, int, int]:
        try:
            journal = str(self._connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            synchronous = int(self._connection.execute("PRAGMA synchronous").fetchone()[0])
            foreign_keys = int(self._connection.execute("PRAGMA foreign_keys").fetchone()[0])
        except sqlite3.Error as exc:
            raise AttemptStoreUnavailableError("cannot read SQLite durability settings") from exc
        return journal, synchronous, foreign_keys

    def _configure(self) -> None:
        journal = str(self._connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        effective = self.effective_pragmas
        if journal != "wal" or effective != ("wal", 2, 1):
            raise AttemptStoreUnavailableError("required WAL/FULL/foreign_keys configuration is unavailable")

    def _open_schema(self) -> None:
        tables = {row[0] for row in self._connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not tables:
            self._create_schema()
            return
        if "store_metadata" not in tables:
            raise AttemptSchemaUnsupportedError("database has no CHA store schema identity")
        rows = self._connection.execute(
            "SELECT schema_identity, schema_version, profile, trust_domain FROM store_metadata"
        ).fetchall()
        if len(rows) != 1:
            raise AttemptCorruptError("store identity cardinality is corrupt")
        identity, version, profile, trust_domain = rows[0]
        if identity != _SCHEMA_IDENTITY or type(version) is not int or version != _SCHEMA_VERSION:
            raise AttemptSchemaUnsupportedError("CHA attempt store schema is unsupported")
        if profile != self._security.profile.value or trust_domain != self._security.trust_domain:
            raise AttemptConflictError("persisted profile or trust_domain does not match")
        required_tables = {
            "store_metadata",
            "reservations",
            "immutable_attempts",
            "current_attempts",
            "replacement_relations",
            "recovery_resolutions",
            "attempt_transitions",
        }
        required_triggers = {
            f"{table}_immutable_{operation}"
            for table in (
                "store_metadata",
                "reservations",
                "attempts",
                "replacements",
                "recoveries",
                "transitions",
            )
            for operation in ("update", "delete")
        }
        triggers = {
            row[0]: row[1]
            for row in self._connection.execute(
                "SELECT name,sql FROM sqlite_master WHERE type='trigger'"
            )
        }
        if not required_tables.issubset(tables) or not required_triggers.issubset(triggers):
            raise AttemptCorruptError("security-critical CHA schema object is missing")
        trigger_tables = {
            "store_metadata": ("store_metadata", "metadata"),
            "reservations": ("reservations", "reservation"),
            "attempts": ("immutable_attempts", "attempt"),
            "replacements": ("replacement_relations", "replacement"),
            "recoveries": ("recovery_resolutions", "recovery"),
            "transitions": ("attempt_transitions", "transition"),
        }
        for prefix, (table, label) in trigger_tables.items():
            for operation in ("update", "delete"):
                name = f"{prefix}_immutable_{operation}"
                actual = self._normalize_schema_sql(str(triggers[name]))
                expected = self._normalize_schema_sql(
                    f"CREATE TRIGGER {name} BEFORE {operation.upper()} ON {table} "
                    f"BEGIN SELECT RAISE(ABORT,'immutable {label}'); END"
                )
                if actual != expected:
                    raise AttemptCorruptError("security-critical immutability trigger is malformed")
        expected_foreign_keys = {
            ("immutable_attempts", "reservations", "attempt_id", "attempt_id"),
            ("current_attempts", "reservations", "attempt_id", "attempt_id"),
            ("replacement_relations", "reservations", "old_attempt_id", "attempt_id"),
            ("replacement_relations", "reservations", "new_attempt_id", "attempt_id"),
            ("recovery_resolutions", "reservations", "attempt_id", "attempt_id"),
            ("attempt_transitions", "reservations", "attempt_id", "attempt_id"),
        }
        actual_foreign_keys: set[tuple[str, str, str, str]] = set()
        for table in required_tables:
            for row in self._connection.execute(f"PRAGMA foreign_key_list({table})"):
                actual_foreign_keys.add((table, row[2], row[3], row[4]))
        if not expected_foreign_keys.issubset(actual_foreign_keys):
            raise AttemptCorruptError("security-critical CHA foreign key is missing")
        required_unique_keys = {
            "current_attempts": {("operation_id",)},
            "reservations": {("attempt_id",), ("decision_key",)},
            "immutable_attempts": {("attempt_id",), ("digest",)},
            "replacement_relations": {
                ("old_attempt_id",),
                ("new_attempt_id",),
                ("decision_key",),
            },
            "recovery_resolutions": {("recovery_sequence",), ("decision_key",)},
            "attempt_transitions": {("transition_id",), ("decision_key",)},
        }
        for table, required in required_unique_keys.items():
            if not required.issubset(self._unique_keys(table)):
                raise AttemptCorruptError("security-critical UNIQUE/PRIMARY KEY is missing")

    @staticmethod
    def _normalize_schema_sql(sql: str) -> str:
        return " ".join(sql.rstrip("; ").upper().split())

    def _unique_keys(self, table: str) -> set[tuple[str, ...]]:
        keys = {
            tuple(
                row[2]
                for row in sorted(
                    self._connection.execute(f"PRAGMA index_info({index[1]})"),
                    key=lambda row: row[0],
                )
            )
            for index in self._connection.execute(f"PRAGMA index_list({table})")
            if index[2] == 1
        }
        primary = tuple(
            row[1]
            for row in sorted(
                (row for row in self._connection.execute(f"PRAGMA table_info({table})") if row[5]),
                key=lambda row: row[5],
            )
        )
        if primary:
            keys.add(primary)
        return keys

    def _create_schema(self) -> None:
        # Schema identifiers are constants; all caller data below is bound.
        self._connection.executescript(
            """
            BEGIN IMMEDIATE;
            CREATE TABLE store_metadata(schema_identity TEXT NOT NULL, schema_version INTEGER NOT NULL, profile TEXT NOT NULL, trust_domain TEXT NOT NULL);
            CREATE TABLE reservations(
              attempt_id TEXT PRIMARY KEY, operation_id TEXT NOT NULL, idempotency_key TEXT NOT NULL,
              authorization_json BLOB NOT NULL, reservation_state TEXT NOT NULL,
              reservation_kind TEXT NOT NULL CHECK(reservation_kind IN ('INITIAL','REPLACEMENT')),
              decision_key TEXT NOT NULL UNIQUE);
            CREATE TABLE immutable_attempts(
              attempt_id TEXT PRIMARY KEY REFERENCES reservations(attempt_id), digest TEXT NOT NULL UNIQUE,
              identity_json BLOB NOT NULL);
            CREATE TABLE current_attempts(
              operation_id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL REFERENCES reservations(attempt_id),
              fence INTEGER NOT NULL CHECK(fence > 0), state TEXT NOT NULL,
              digest_status TEXT NOT NULL CHECK(digest_status IN ('NOT_YET_DEFINED','DEFINED')),
              digest TEXT REFERENCES immutable_attempts(digest),
              CHECK((digest_status='NOT_YET_DEFINED' AND digest IS NULL) OR (digest_status='DEFINED' AND digest IS NOT NULL)));
            CREATE TABLE replacement_relations(
              old_attempt_id TEXT NOT NULL UNIQUE REFERENCES reservations(attempt_id),
              new_attempt_id TEXT PRIMARY KEY REFERENCES reservations(attempt_id), evidence_json BLOB NOT NULL,
              authorization_json BLOB NOT NULL, evidence_digest TEXT NOT NULL,
              decision_key TEXT NOT NULL UNIQUE);
            CREATE TABLE recovery_resolutions(
              recovery_sequence INTEGER PRIMARY KEY,
              attempt_id TEXT NOT NULL REFERENCES reservations(attempt_id),
              predecessor_state TEXT NOT NULL, outcome TEXT NOT NULL,
              decision_key TEXT NOT NULL UNIQUE, resolution_json BLOB NOT NULL);
            CREATE TABLE attempt_transitions(
              transition_id INTEGER PRIMARY KEY, attempt_id TEXT NOT NULL REFERENCES reservations(attempt_id),
              state TEXT NOT NULL, evidence_reference TEXT, evidence_digest TEXT,
              decision_key TEXT NOT NULL UNIQUE);
            CREATE TRIGGER store_metadata_immutable_update BEFORE UPDATE ON store_metadata BEGIN SELECT RAISE(ABORT,'immutable metadata'); END;
            CREATE TRIGGER store_metadata_immutable_delete BEFORE DELETE ON store_metadata BEGIN SELECT RAISE(ABORT,'immutable metadata'); END;
            CREATE TRIGGER reservations_immutable_update BEFORE UPDATE ON reservations BEGIN SELECT RAISE(ABORT,'immutable reservation'); END;
            CREATE TRIGGER reservations_immutable_delete BEFORE DELETE ON reservations BEGIN SELECT RAISE(ABORT,'immutable reservation'); END;
            CREATE TRIGGER attempts_immutable_update BEFORE UPDATE ON immutable_attempts BEGIN SELECT RAISE(ABORT,'immutable attempt'); END;
            CREATE TRIGGER attempts_immutable_delete BEFORE DELETE ON immutable_attempts BEGIN SELECT RAISE(ABORT,'immutable attempt'); END;
            CREATE TRIGGER replacements_immutable_update BEFORE UPDATE ON replacement_relations BEGIN SELECT RAISE(ABORT,'immutable replacement'); END;
            CREATE TRIGGER replacements_immutable_delete BEFORE DELETE ON replacement_relations BEGIN SELECT RAISE(ABORT,'immutable replacement'); END;
            CREATE TRIGGER recoveries_immutable_update BEFORE UPDATE ON recovery_resolutions BEGIN SELECT RAISE(ABORT,'immutable recovery'); END;
            CREATE TRIGGER recoveries_immutable_delete BEFORE DELETE ON recovery_resolutions BEGIN SELECT RAISE(ABORT,'immutable recovery'); END;
            CREATE TRIGGER transitions_immutable_update BEFORE UPDATE ON attempt_transitions BEGIN SELECT RAISE(ABORT,'immutable transition'); END;
            CREATE TRIGGER transitions_immutable_delete BEFORE DELETE ON attempt_transitions BEGIN SELECT RAISE(ABORT,'immutable transition'); END;
            """
        )
        try:
            self._connection.execute(
                "INSERT INTO store_metadata VALUES(?,?,?,?)",
                (_SCHEMA_IDENTITY, _SCHEMA_VERSION, self._security.profile.value, self._security.trust_domain),
            )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            yield self._connection
            self._connection.execute("COMMIT")
        except AttemptStoreError:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise
        except sqlite3.IntegrityError as exc:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise AttemptConflictError("immutable/CAS constraint rejected the write") from exc
        except sqlite3.Error as exc:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise AttemptStoreUnavailableError("SQLite authority transaction failed") from exc

    def _check_auth(self, auth: AttemptAuthorization) -> None:
        if auth.environment != self._security.profile.value or auth.trust_domain != self._security.trust_domain:
            raise AttemptConflictError("authorization environment/trust_domain mismatch")

    def reserve_or_resolve_attempt_id(self, auth: AttemptAuthorization) -> CurrentAttempt:
        self._check_auth(auth)
        key = _idempotency_key(auth)
        with self._write() as db:
            row = db.execute("SELECT attempt_id FROM current_attempts WHERE operation_id=?", (auth.logical_operation_id,)).fetchone()
            if row is not None:
                existing = self._read_current(auth.logical_operation_id, db)
                if existing.reservation.exact_idempotency_key != key:
                    raise AttemptConflictError("operation already has an incompatible current attempt")
                return existing
            attempt_id = _new_rpa_id()
            decision_key = _decision_key("INITIAL", asdict(auth))
            db.execute(
                "INSERT INTO reservations VALUES(?,?,?,?,?,?,?)",
                (
                    attempt_id,
                    auth.logical_operation_id,
                    key,
                    _canonical(asdict(auth)),
                    AttemptState.RESERVED_AWAITING_SIGNATURES.value,
                    "INITIAL",
                    decision_key,
                ),
            )
            db.execute(
                "INSERT INTO current_attempts VALUES(?,?,?,?,?,NULL)",
                (auth.logical_operation_id, attempt_id, 1, AttemptState.RESERVED_AWAITING_SIGNATURES.value, "NOT_YET_DEFINED"),
            )
            return self._read_current(auth.logical_operation_id, db)

    def finalize_attempt(self, identity: AttemptIdentity, *, expected_fence: int) -> CurrentAttempt:
        self._check_auth(identity.authorization)
        payload = _canonical(identity.payload())
        with self._write() as db:
            current = self._read_current(identity.authorization.logical_operation_id, db)
            if current.reservation.issuance_attempt_id != identity.issuance_attempt_id:
                raise AttemptConflictError("reserved attempt does not match finalization")
            existing = db.execute("SELECT digest, identity_json FROM immutable_attempts WHERE attempt_id=?", (identity.issuance_attempt_id,)).fetchone()
            if existing is not None:
                if existing != (identity.digest_sha256, payload):
                    raise AttemptCorruptError("same attempt id has unequal immutable identity")
                return current
            db.execute("INSERT INTO immutable_attempts VALUES(?,?,?)", (identity.issuance_attempt_id, identity.digest_sha256, payload))
            db.execute(
                "INSERT INTO attempt_transitions(attempt_id,state,decision_key) VALUES(?,?,?)",
                (
                    identity.issuance_attempt_id,
                    AttemptState.SIGNED_IMMUTABLE_DURABLE_NOT_SENT.value,
                    _decision_key("FINALIZATION", identity.payload()),
                ),
            )
            self._cas(db, identity.authorization.logical_operation_id, identity.issuance_attempt_id, expected_fence, identity.issuance_attempt_id, AttemptState.SIGNED_IMMUTABLE_DURABLE_NOT_SENT, identity.digest_sha256)
            return self._read_current(identity.authorization.logical_operation_id, db)

    def replace_after_authoritative_unbound(self, auth: AttemptAuthorization, evidence: AuthoritativeUnboundEvidence, *, expected_fence: int) -> CurrentAttempt:
        self._check_auth(auth)
        if evidence.environment != auth.environment or evidence.trust_domain != auth.trust_domain:
            raise AttemptConflictError("replacement evidence domain mismatch")
        if (
            evidence.logical_operation_id != auth.logical_operation_id
            or evidence.account_id != auth.account_id
            or evidence.bootstrap_entitlement_id != auth.bootstrap_entitlement_id
            or evidence.entitlement_generation != auth.entitlement_generation
            or evidence.canonical_genesis_request_fingerprint_sha256
            != auth.canonical_genesis_request_fingerprint_sha256
            or evidence.initial_binding_reference != auth.initial_binding_reference
            or evidence.initial_binding_digest_sha256 != auth.initial_binding_digest_sha256
        ):
            raise AttemptConflictError("replacement evidence authorization binding mismatch")
        replacement_identity = {
            "authorization": asdict(auth),
            "old_issuance_attempt_id": evidence.old_issuance_attempt_id,
            "evidence": asdict(evidence),
        }
        decision_key = _decision_key("REPLACEMENT", replacement_identity)
        with self._write() as db:
            previous = db.execute(
                "SELECT old_attempt_id,new_attempt_id,evidence_json,authorization_json,evidence_digest,decision_key FROM replacement_relations WHERE old_attempt_id=?",
                (evidence.old_issuance_attempt_id,),
            ).fetchone()
            if previous is not None:
                stored_auth, stored_evidence = self._validate_replacement_row(previous, db)
                if stored_evidence != evidence or stored_auth != auth:
                    raise AttemptConflictError("predecessor already has a different replacement decision")
                replay = self._read_current(auth.logical_operation_id, db)
                if replay.reservation.issuance_attempt_id != previous[1]:
                    raise AttemptCorruptError("committed replacement is not current")
                return replay
            old = self._read_current(auth.logical_operation_id, db)
            if old.reservation.issuance_attempt_id != evidence.old_issuance_attempt_id:
                raise AttemptConflictError("replacement evidence does not name current attempt")
            attempt_id = _new_rpa_id()
            key = _idempotency_key(auth)
            db.execute(
                "INSERT INTO reservations VALUES(?,?,?,?,?,?,?)",
                (
                    attempt_id,
                    auth.logical_operation_id,
                    key,
                    _canonical(asdict(auth)),
                    AttemptState.RESERVED_AWAITING_SIGNATURES.value,
                    "REPLACEMENT",
                    decision_key,
                ),
            )
            db.execute(
                "INSERT INTO replacement_relations VALUES(?,?,?,?,?,?)",
                (
                    evidence.old_issuance_attempt_id,
                    attempt_id,
                    _canonical(asdict(evidence)),
                    _canonical(asdict(auth)),
                    evidence.authority_authenticated_evidence_digest_sha256,
                    decision_key,
                ),
            )
            db.execute(
                "INSERT INTO attempt_transitions(attempt_id,state,evidence_reference,evidence_digest,decision_key) VALUES(?,?,?,?,?)",
                (
                    evidence.old_issuance_attempt_id,
                    AttemptState.SUPERSEDED_AFTER_AUTHORITATIVE_UNBOUND_RECONCILIATION.value,
                    evidence.authority_authenticated_evidence_reference,
                    evidence.authority_authenticated_evidence_digest_sha256,
                    decision_key,
                ),
            )
            self._cas(db, auth.logical_operation_id, evidence.old_issuance_attempt_id, expected_fence, attempt_id, AttemptState.RESERVED_AWAITING_SIGNATURES, None)
            return self._read_current(auth.logical_operation_id, db)

    def record_recovery_resolution(self, operation_id: str, resolution: RecoveryResolution, *, expected_fence: int) -> CurrentAttempt:
        resolution_json = _canonical(
            {
                "issuance_attempt_id": resolution.issuance_attempt_id,
                "outcome": resolution.outcome.value,
                "authenticated_reference": resolution.authenticated_reference,
                "authenticated_digest_sha256": resolution.authenticated_digest_sha256,
            }
        )
        decision_key = _decision_key("RECOVERY", json.loads(resolution_json))
        with self._write() as db:
            previous = db.execute(
                "SELECT recovery_sequence,attempt_id,predecessor_state,outcome,decision_key,resolution_json FROM recovery_resolutions WHERE decision_key=?",
                (decision_key,),
            ).fetchone()
            if previous is not None:
                stored = self._recovery_from_row(previous)
                if stored != resolution:
                    raise AttemptCorruptError("recovery decision identity collision or corruption")
                replay = self._read_current(operation_id, db)
                if replay.reservation.issuance_attempt_id != resolution.issuance_attempt_id:
                    raise AttemptCorruptError("committed recovery resolution does not match current")
                return replay
            current = self._read_current(operation_id, db)
            if current.reservation.issuance_attempt_id != resolution.issuance_attempt_id:
                raise AttemptConflictError("recovery does not resolve current attempt")
            if current.identity is None:
                raise AttemptConflictError("an unsigned reservation cannot have an issuer recovery outcome")
            legal = {
                AttemptState.SIGNED_IMMUTABLE_DURABLE_NOT_SENT: {
                    AttemptState.MAY_HAVE_BEEN_SENT_OUTCOME_UNKNOWN,
                    AttemptState.EXACT_BOUND_RECOVERED,
                },
                AttemptState.MAY_HAVE_BEEN_SENT_OUTCOME_UNKNOWN: {
                    AttemptState.EXACT_BOUND_RECOVERED,
                },
            }
            if resolution.outcome not in legal.get(current.state, set()):
                raise AttemptConflictError("illegal or non-monotonic recovery transition")
            db.execute(
                "INSERT INTO recovery_resolutions(attempt_id,predecessor_state,outcome,decision_key,resolution_json) VALUES(?,?,?,?,?)",
                (
                    resolution.issuance_attempt_id,
                    current.state.value,
                    resolution.outcome.value,
                    decision_key,
                    resolution_json,
                ),
            )
            db.execute(
                "INSERT INTO attempt_transitions(attempt_id,state,evidence_reference,evidence_digest,decision_key) VALUES(?,?,?,?,?)",
                (
                    resolution.issuance_attempt_id,
                    resolution.outcome.value,
                    resolution.authenticated_reference,
                    resolution.authenticated_digest_sha256,
                    decision_key,
                ),
            )
            self._cas(db, operation_id, resolution.issuance_attempt_id, expected_fence, resolution.issuance_attempt_id, resolution.outcome, current.immutable_attempt_digest_sha256)
            return self._read_current(operation_id, db)

    def _recovery_from_row(self, row: tuple[object, ...]) -> RecoveryResolution:
        _, attempt_id, _, outcome, decision_key, resolution_json = row
        try:
            data = json.loads(bytes(resolution_json))
            data["outcome"] = AttemptState(data["outcome"])
            resolution = RecoveryResolution(**data)
        except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise AttemptCorruptError("persisted recovery resolution is malformed") from exc
        if (
            attempt_id != resolution.issuance_attempt_id
            or outcome != resolution.outcome.value
            or decision_key != _decision_key("RECOVERY", json.loads(bytes(resolution_json)))
        ):
            raise AttemptCorruptError("persisted recovery resolution identity is inconsistent")
        return resolution

    def _validate_recovery_history(
        self, attempt_id: str, current_state: AttemptState, db: sqlite3.Connection
    ) -> None:
        rows = db.execute(
            "SELECT recovery_sequence,attempt_id,predecessor_state,outcome,decision_key,resolution_json "
            "FROM recovery_resolutions WHERE attempt_id=? ORDER BY recovery_sequence",
            (attempt_id,),
        ).fetchall()
        predecessor = AttemptState.SIGNED_IMMUTABLE_DURABLE_NOT_SENT
        legal = {
            AttemptState.SIGNED_IMMUTABLE_DURABLE_NOT_SENT: {
                AttemptState.MAY_HAVE_BEEN_SENT_OUTCOME_UNKNOWN,
                AttemptState.EXACT_BOUND_RECOVERED,
            },
            AttemptState.MAY_HAVE_BEEN_SENT_OUTCOME_UNKNOWN: {
                AttemptState.EXACT_BOUND_RECOVERED,
            },
        }
        for row in rows:
            resolution = self._recovery_from_row(row)
            try:
                stored_predecessor = AttemptState(row[2])
            except ValueError as exc:
                raise AttemptCorruptError("recovery predecessor state is malformed") from exc
            if stored_predecessor is not predecessor or resolution.outcome not in legal.get(predecessor, set()):
                raise AttemptCorruptError("recovery history progression is inconsistent")
            predecessor = resolution.outcome
        if rows and predecessor is not current_state:
            raise AttemptCorruptError("recovery history head differs from current state")

    def _validate_replacement_row(
        self, row: tuple[object, ...], db: sqlite3.Connection
    ) -> tuple[AttemptAuthorization, AuthoritativeUnboundEvidence]:
        old_id, new_id, evidence_json, authorization_json, evidence_digest, decision_key = row
        try:
            auth = AttemptAuthorization(**json.loads(bytes(authorization_json)))
            evidence = AuthoritativeUnboundEvidence(**json.loads(bytes(evidence_json)))
        except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise AttemptCorruptError("persisted replacement relation is malformed") from exc
        identity = {
            "authorization": asdict(auth),
            "old_issuance_attempt_id": evidence.old_issuance_attempt_id,
            "evidence": asdict(evidence),
        }
        reservation = db.execute(
            "SELECT reservation_kind,decision_key FROM reservations WHERE attempt_id=?",
            (new_id,),
        ).fetchone()
        if (
            type(old_id) is not str
            or _RPA_ID.fullmatch(old_id) is None
            or type(new_id) is not str
            or _RPA_ID.fullmatch(new_id) is None
            or evidence.old_issuance_attempt_id != old_id
            or evidence_digest != evidence.authority_authenticated_evidence_digest_sha256
            or decision_key != _decision_key("REPLACEMENT", identity)
            or reservation != ("REPLACEMENT", decision_key)
        ):
            raise AttemptCorruptError("persisted replacement relation identity is inconsistent")
        self._validate_evidence_binding(auth, evidence)
        return auth, evidence

    @staticmethod
    def _validate_evidence_binding(
        auth: AttemptAuthorization, evidence: AuthoritativeUnboundEvidence
    ) -> None:
        if (
            evidence.environment != auth.environment
            or evidence.trust_domain != auth.trust_domain
            or evidence.logical_operation_id != auth.logical_operation_id
            or evidence.account_id != auth.account_id
            or evidence.bootstrap_entitlement_id != auth.bootstrap_entitlement_id
            or evidence.entitlement_generation != auth.entitlement_generation
            or evidence.canonical_genesis_request_fingerprint_sha256
            != auth.canonical_genesis_request_fingerprint_sha256
            or evidence.initial_binding_reference != auth.initial_binding_reference
            or evidence.initial_binding_digest_sha256 != auth.initial_binding_digest_sha256
        ):
            raise AttemptCorruptError("persisted replacement evidence binding is inconsistent")

    def _cas(self, db: sqlite3.Connection, operation_id: str, old_attempt: str, fence: int, new_attempt: str, state: AttemptState, digest: str | None) -> None:
        status = "DEFINED" if digest is not None else "NOT_YET_DEFINED"
        cursor = db.execute(
            "UPDATE current_attempts SET attempt_id=?,fence=fence+1,state=?,digest_status=?,digest=? WHERE operation_id=? AND attempt_id=? AND fence=?",
            (new_attempt, state.value, status, digest, operation_id, old_attempt, fence),
        )
        if cursor.rowcount == 0:
            raise AttemptConflictError("current attempt fence CAS lost")
        if cursor.rowcount != 1:
            raise AttemptCorruptError("current attempt CAS affected impossible row count")

    def attempt(self, operation_id: str) -> CurrentAttempt:
        return self._read_current(operation_id, self._connection)

    def _read_current(self, operation_id: str, db: sqlite3.Connection) -> CurrentAttempt:
        rows = db.execute("SELECT attempt_id,fence,state,digest_status,digest FROM current_attempts WHERE operation_id=?", (operation_id,)).fetchall()
        if not rows:
            raise AttemptNotFoundError("operation has no current attempt")
        if len(rows) != 1:
            raise AttemptCorruptError("duplicate current attempt relation")
        attempt_id, fence, state_raw, digest_status, digest = rows[0]
        reservation_row = db.execute(
            "SELECT operation_id,idempotency_key,authorization_json,reservation_state,reservation_kind,decision_key "
            "FROM reservations WHERE attempt_id=?",
            (attempt_id,),
        ).fetchone()
        if reservation_row is None:
            raise AttemptCorruptError("current pointer target is missing")
        try:
            auth_data = json.loads(bytes(reservation_row[2]))
            auth = AttemptAuthorization(**auth_data)
            reservation_state = AttemptState(reservation_row[3])
            state = AttemptState(state_raw)
        except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise AttemptCorruptError("persisted reservation/state is malformed") from exc
        try:
            reservation = AttemptReservation(auth, reservation_row[1], attempt_id, reservation_state)
        except (TypeError, ValueError) as exc:
            raise AttemptCorruptError("persisted reservation identity is malformed") from exc
        if (
            reservation_row[0] != operation_id
            or auth.logical_operation_id != operation_id
            or auth.environment != self._security.profile.value
            or auth.trust_domain != self._security.trust_domain
            or reservation.exact_idempotency_key != _idempotency_key(auth)
            or reservation.reservation_state is not AttemptState.RESERVED_AWAITING_SIGNATURES
            or type(fence) is not int
            or fence < 1
        ):
            raise AttemptCorruptError("persisted reservation/current binding is inconsistent")
        reservation_kind, reservation_decision_key = reservation_row[4], reservation_row[5]
        if reservation_kind == "INITIAL":
            if reservation_decision_key != _decision_key("INITIAL", asdict(auth)):
                raise AttemptCorruptError("initial reservation decision identity is inconsistent")
        elif reservation_kind == "REPLACEMENT":
            replacement_row = db.execute(
                "SELECT old_attempt_id,new_attempt_id,evidence_json,authorization_json,evidence_digest,decision_key "
                "FROM replacement_relations WHERE new_attempt_id=?",
                (attempt_id,),
            ).fetchone()
            if replacement_row is None:
                raise AttemptCorruptError("replacement reservation has no immutable relation")
            replacement_auth, _ = self._validate_replacement_row(replacement_row, db)
            if replacement_auth != auth or reservation_decision_key != replacement_row[5]:
                raise AttemptCorruptError("replacement reservation decision identity is inconsistent")
        else:
            raise AttemptCorruptError("reservation kind is malformed")
        identity = None
        if digest_status == "DEFINED":
            if _HEX64.fullmatch(digest or "") is None:
                raise AttemptCorruptError("defined immutable digest is malformed")
            attempt_row = db.execute("SELECT identity_json FROM immutable_attempts WHERE attempt_id=? AND digest=?", (attempt_id, digest)).fetchone()
            if attempt_row is None:
                raise AttemptCorruptError("current immutable attempt target is missing")
            try:
                data = json.loads(bytes(attempt_row[0]))
                auth_fields = {name: data.pop(name) for name in AttemptAuthorization.__dataclass_fields__}
                data.pop("schema_version")
                identity = AttemptIdentity(AttemptAuthorization(**auth_fields), **data)
            except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
                raise AttemptCorruptError("persisted immutable attempt is malformed") from exc
            if identity.digest_sha256 != digest:
                raise AttemptCorruptError("immutable attempt digest mismatch")
            if identity.authorization != reservation.authorization:
                raise AttemptCorruptError("immutable attempt authorization differs from reservation")
        elif digest_status != "NOT_YET_DEFINED" or digest is not None:
            raise AttemptCorruptError("current digest union is malformed")
        if identity is not None:
            self._validate_recovery_history(attempt_id, state, db)
        return CurrentAttempt(reservation, state, fence, digest, identity)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> SQLiteCHAAttemptStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
