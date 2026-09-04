"""Purpose-separated authentication authority for physical backup artifacts.

Authority metadata is deliberately held in its own SQLite database.  Secret bytes are
staged in secure custody before a metadata reference can become authoritative.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import sqlite3
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol
from uuid import uuid4

from bot_core.security.base import SecretStorage

BACKUP_AUTHENTICATION_PURPOSE = "CRYPTOHUNTER_M0_11_PHYSICAL_BACKUP_AUTH_V1"
BACKUP_AUTHENTICATION_ALGORITHM = "HMAC-SHA-256"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(
    r"^[a-z][a-z0-9]*_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_ENVIRONMENTS = frozenset({"PAPER", "TESTNET", "LIVE"})


class BackupAuthenticationError(RuntimeError):
    """Base error with a stable, non-secret semantic code."""

    code = "AUTHORITY_UNAVAILABLE"

    def __init__(self) -> None:
        super().__init__(self.code)


class AuthorityNotProvisionedError(BackupAuthenticationError):
    code = "AUTHORITY_NOT_PROVISIONED"


class NoActiveKeyError(BackupAuthenticationError):
    code = "NO_ACTIVE_KEY"


class StaleAuthorityRevisionError(BackupAuthenticationError):
    code = "STALE_AUTHORITY_REVISION"


class AuthorityUnavailableError(BackupAuthenticationError):
    code = "AUTHORITY_UNAVAILABLE"


class UnknownAdminMutationOutcomeError(BackupAuthenticationError):
    code = "UNKNOWN_ADMIN_MUTATION_OUTCOME"


class AlreadyProvisionedError(BackupAuthenticationError):
    code = "AUTHORITY_ALREADY_PROVISIONED"


class InvalidAuthorityOperationError(BackupAuthenticationError):
    code = "INVALID_AUTHORITY_OPERATION"


def _canonical_id(value: object, prefix: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not _ID_RE.fullmatch(value)
        or not value.startswith(prefix + "_")
    ):
        raise ValueError(f"{name} must be a canonical M0.2 identifier")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256 hexadecimal")
    return value


def _validate_authority_revision(value: object) -> int:
    """Return an exact positive integer revision; bool and coercions are forbidden."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("authority_revision must be a positive non-boolean integer")
    return value


@dataclass(frozen=True, slots=True)
class BackupArtifactAuthenticationScope:
    account_id: str
    device_installation_id: str
    environment: str
    purpose: str = BACKUP_AUTHENTICATION_PURPOSE

    def __post_init__(self) -> None:
        _canonical_id(self.account_id, "acct", "account_id")
        _canonical_id(self.device_installation_id, "dev", "device_installation_id")
        if self.environment not in _ENVIRONMENTS:
            raise ValueError("environment must be PAPER, TESTNET, or LIVE")
        if self.purpose != BACKUP_AUTHENTICATION_PURPOSE:
            raise ValueError("purpose is fixed by the M0.11 authority")


class AuthorityKeyState(str, Enum):
    ACTIVE = "ACTIVE"
    VERIFY_ONLY = "VERIFY_ONLY"
    REVOKED = "REVOKED"


class AuthorityScopeCondition(str, Enum):
    UNPROVISIONED_SCOPE = "UNPROVISIONED_SCOPE"
    PROVISIONED_WITH_ACTIVE = "PROVISIONED_WITH_ACTIVE"
    PROVISIONED_WITHOUT_ACTIVE = "PROVISIONED_WITHOUT_ACTIVE"


@dataclass(frozen=True, slots=True)
class AuthorityKeyMetadata:
    authority_key_id: str
    state: AuthorityKeyState

    def __post_init__(self) -> None:
        if not isinstance(self.authority_key_id, str) or not self.authority_key_id:
            raise ValueError("authority_key_id must be a non-empty string")
        if not isinstance(self.state, AuthorityKeyState):
            raise ValueError("state must be an AuthorityKeyState")


@dataclass(frozen=True, slots=True)
class BackupAuthenticationAuthorityScopeSnapshot:
    scope: BackupArtifactAuthenticationScope
    authority_revision: int
    keys: tuple[AuthorityKeyMetadata, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scope, BackupArtifactAuthenticationScope):
            raise ValueError("scope must be a BackupArtifactAuthenticationScope")
        _validate_authority_revision(self.authority_revision)
        if not isinstance(self.keys, tuple) or not all(
            isinstance(key, AuthorityKeyMetadata) for key in self.keys
        ):
            raise ValueError("keys must be an exact tuple of AuthorityKeyMetadata")
        if sum(key.state is AuthorityKeyState.ACTIVE for key in self.keys) > 1:
            raise ValueError("snapshot cannot contain multiple ACTIVE keys")

    @property
    def condition(self) -> AuthorityScopeCondition:
        active = sum(key.state is AuthorityKeyState.ACTIVE for key in self.keys)
        return (
            AuthorityScopeCondition.PROVISIONED_WITH_ACTIVE
            if active == 1
            else AuthorityScopeCondition.PROVISIONED_WITHOUT_ACTIVE
        )


@dataclass(frozen=True, slots=True)
class PhysicalSQLiteArtifactManifest:
    artifact_format_version: int
    account_id: str
    device_installation_id: str
    environment: str
    state_store_identity_fingerprint_sha256: str
    state_store_schema_version: int
    local_protected_freshness_generation: int
    state_fingerprint_sha256: str
    transaction_fingerprint_sha256: str
    backup_envelope_fingerprint_sha256: str
    physical_artifact_sha256: str
    physical_artifact_byte_length: int
    sqlite_schema_fingerprint_sha256: str
    manifest_fingerprint_sha256: str

    def __post_init__(self) -> None:
        _canonical_id(self.account_id, "acct", "account_id")
        _canonical_id(self.device_installation_id, "dev", "device_installation_id")
        if self.environment not in _ENVIRONMENTS:
            raise ValueError("invalid environment")
        for name in (
            "artifact_format_version",
            "state_store_schema_version",
            "local_protected_freshness_generation",
            "physical_artifact_byte_length",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "state_store_identity_fingerprint_sha256",
            "state_fingerprint_sha256",
            "transaction_fingerprint_sha256",
            "backup_envelope_fingerprint_sha256",
            "physical_artifact_sha256",
            "sqlite_schema_fingerprint_sha256",
            "manifest_fingerprint_sha256",
        ):
            _sha(getattr(self, name), name)
        if not hmac.compare_digest(
            self.manifest_fingerprint_sha256, self.computed_fingerprint()
        ):
            raise ValueError(
                "manifest fingerprint does not match canonical manifest projection"
            )

    def projection(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        result = asdict(self)
        if not include_fingerprint:
            del result["manifest_fingerprint_sha256"]
        return result

    def computed_fingerprint(self) -> str:
        return hashlib.sha256(
            _canonical_json(self.projection(include_fingerprint=False))
        ).hexdigest()

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PhysicalSQLiteArtifactManifest:
        expected = {field.name for field in fields(cls)}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("manifest mapping must contain the exact fourteen fields")
        return cls(**{name: value[name] for name in expected})  # type: ignore[arg-type]

    @classmethod
    def create(cls, **values: object) -> PhysicalSQLiteArtifactManifest:
        expected = {field.name for field in fields(cls)} - {"manifest_fingerprint_sha256"}
        if set(values) != expected:
            raise ValueError("manifest creation requires the exact thirteen source fields")
        fingerprint = hashlib.sha256(_canonical_json(values)).hexdigest()
        return cls(**values, manifest_fingerprint_sha256=fingerprint)  # type: ignore[arg-type]


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_authentication_payload(manifest: PhysicalSQLiteArtifactManifest) -> bytes:
    return (
        BACKUP_AUTHENTICATION_PURPOSE.encode()
        + b"\x00"
        + _canonical_json(manifest.projection())
    )


@dataclass(frozen=True, slots=True)
class PhysicalArtifactAuthenticationProof:
    proof_version: int
    algorithm: str
    purpose: str
    authority_key_id: str
    manifest_fingerprint_sha256: str
    authentication_tag_hex: str

    def __post_init__(self) -> None:
        if self.proof_version != 1 or isinstance(self.proof_version, bool):
            raise ValueError("proof_version must be 1")
        if self.algorithm != BACKUP_AUTHENTICATION_ALGORITHM:
            raise ValueError("algorithm must be HMAC-SHA-256")
        if self.purpose != BACKUP_AUTHENTICATION_PURPOSE:
            raise ValueError("invalid proof purpose")
        if not isinstance(self.authority_key_id, str) or not self.authority_key_id:
            raise ValueError("authority_key_id must be non-empty")
        _sha(self.manifest_fingerprint_sha256, "manifest_fingerprint_sha256")
        _sha(self.authentication_tag_hex, "authentication_tag_hex")

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> PhysicalArtifactAuthenticationProof:
        expected = {field.name for field in fields(cls)}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("proof mapping must contain the exact six fields")
        return cls(**{name: value[name] for name in expected})  # type: ignore[arg-type]


class BackupArtifactVerificationResult(str, Enum):
    VERIFIED = "VERIFIED"
    INVALID_PROOF = "INVALID_PROOF"
    UNKNOWN_KEY = "UNKNOWN_KEY"
    REVOKED_KEY = "REVOKED_KEY"
    AUTHORITY_UNAVAILABLE = "AUTHORITY_UNAVAILABLE"
    AUTHORITY_NOT_PROVISIONED = "AUTHORITY_NOT_PROVISIONED"


class BackupArtifactAuthenticationSecureCustody(Protocol):
    """Opaque custody: callers can request MAC operations, never key bytes."""

    def persist(self, custody_handle: str, key_material: bytes) -> None: ...
    def digest(self, custody_handle: str, payload: bytes) -> bytes | None: ...


class SecretStorageBackupAuthenticationSecureCustody:
    """Purpose-separated adapter over generic OS/encrypted SecretStorage mechanics."""

    _PREFIX = "dudzian.backup-authentication.v1:"

    def __init__(self, storage: SecretStorage) -> None:
        self._storage = storage

    def persist(self, custody_handle: str, key_material: bytes) -> None:
        if len(key_material) != 32:
            raise AuthorityUnavailableError
        storage_key = self._PREFIX + custody_handle
        try:
            if self._storage.get_secret(storage_key) is not None:
                raise AuthorityUnavailableError
            self._storage.set_secret(
                storage_key, base64.b64encode(key_material).decode("ascii")
            )
        except BackupAuthenticationError:
            raise
        except Exception:
            raise AuthorityUnavailableError from None

    def digest(self, custody_handle: str, payload: bytes) -> bytes | None:
        try:
            encoded = self._storage.get_secret(self._PREFIX + custody_handle)
        except Exception:
            raise AuthorityUnavailableError from None
        if encoded is None:
            return None
        try:
            key = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError):
            return None
        if len(key) != 32:
            return None
        return hmac.digest(key, payload, "sha256")


@dataclass(frozen=True, slots=True)
class _StoredKey:
    authority_key_id: str
    state: AuthorityKeyState
    custody_handle: str


class SQLiteBackupAuthenticationMetadataStore:
    """Independent transactional authority metadata store."""

    def __init__(
        self, path: str | Path, *, state_store_path: str | Path | None = None
    ) -> None:
        self.path = Path(path)
        if (
            state_store_path is not None
            and self.path.resolve() == Path(state_store_path).resolve()
        ):
            raise ValueError("authority metadata must not alias StateStore")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS authority_scope (
                  account_id TEXT NOT NULL, device_installation_id TEXT NOT NULL,
                  environment TEXT NOT NULL, purpose TEXT NOT NULL, revision INTEGER NOT NULL CHECK(revision >= 1),
                  PRIMARY KEY(account_id, device_installation_id, environment, purpose));
                CREATE TABLE IF NOT EXISTS authority_key (
                  account_id TEXT NOT NULL, device_installation_id TEXT NOT NULL,
                  environment TEXT NOT NULL, purpose TEXT NOT NULL, authority_key_id TEXT NOT NULL,
                  lifecycle_state TEXT NOT NULL CHECK(lifecycle_state IN ('ACTIVE','VERIFY_ONLY','REVOKED')),
                  custody_handle TEXT NOT NULL UNIQUE, PRIMARY KEY(account_id,device_installation_id,environment,purpose,authority_key_id),
                  FOREIGN KEY(account_id,device_installation_id,environment,purpose)
                    REFERENCES authority_scope(account_id,device_installation_id,environment,purpose) ON DELETE CASCADE);
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_per_scope ON authority_key(account_id,device_installation_id,environment,purpose)
                  WHERE lifecycle_state='ACTIVE';
            """)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _params(scope: BackupArtifactAuthenticationScope) -> tuple[str, str, str, str]:
        return (
            scope.account_id,
            scope.device_installation_id,
            scope.environment,
            scope.purpose,
        )

    def _read(
        self, connection: sqlite3.Connection, scope: BackupArtifactAuthenticationScope
    ) -> tuple[int, tuple[_StoredKey, ...]] | None:
        params = self._params(scope)
        row = connection.execute(
            "SELECT revision FROM authority_scope WHERE account_id=? AND device_installation_id=? AND environment=? AND purpose=?",
            params,
        ).fetchone()
        if row is None:
            return None
        rows = connection.execute(
            "SELECT authority_key_id,lifecycle_state,custody_handle FROM authority_key WHERE account_id=? AND device_installation_id=? AND environment=? AND purpose=? ORDER BY authority_key_id",
            params,
        ).fetchall()
        try:
            revision = _validate_authority_revision(row[0])
        except ValueError:
            raise AuthorityUnavailableError
        try:
            keys = tuple(
                _StoredKey(item[0], AuthorityKeyState(item[1]), item[2])
                for item in rows
            )
        except (TypeError, ValueError, IndexError):
            raise AuthorityUnavailableError from None
        if any(
            not isinstance(key.authority_key_id, str)
            or not key.authority_key_id
            or not isinstance(key.custody_handle, str)
            or not key.custody_handle
            for key in keys
        ):
            raise AuthorityUnavailableError
        if sum(key.state is AuthorityKeyState.ACTIVE for key in keys) > 1:
            raise AuthorityUnavailableError
        return revision, keys

    def read(
        self, scope: BackupArtifactAuthenticationScope
    ) -> tuple[int, tuple[_StoredKey, ...]] | None:
        try:
            with self._connect() as connection:
                connection.execute("BEGIN")
                result = self._read(connection, scope)
                connection.commit()
                return result
        except BackupAuthenticationError:
            raise
        except sqlite3.Error as exc:
            raise AuthorityUnavailableError from exc

    def create(
        self,
        scope: BackupArtifactAuthenticationScope,
        key: _StoredKey,
        *,
        _before_commit: Callable[[], None] | None = None,
    ) -> tuple[int, tuple[_StoredKey, ...]]:
        params = self._params(scope)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("INSERT INTO authority_scope VALUES (?,?,?,?,1)", params)
            connection.execute(
                "INSERT INTO authority_key VALUES (?,?,?,?,?,?,?)",
                (*params, key.authority_key_id, key.state.value, key.custody_handle),
            )
            if _before_commit is not None:
                _before_commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            connection.close()
            raise AlreadyProvisionedError from exc
        except Exception as exc:
            connection.rollback()
            connection.close()
            if isinstance(exc, BackupAuthenticationError):
                raise
            raise AuthorityUnavailableError from exc
        try:
            connection.commit()
        except sqlite3.Error:
            raise UnknownAdminMutationOutcomeError from None
        finally:
            connection.close()
        return 1, (key,)

    def replace(
        self,
        scope: BackupArtifactAuthenticationScope,
        expected_revision: int,
        keys: tuple[_StoredKey, ...],
        *,
        _before_commit: Callable[[], None] | None = None,
    ) -> tuple[int, tuple[_StoredKey, ...]]:
        _validate_authority_revision(expected_revision)
        params = self._params(scope)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                "UPDATE authority_scope SET revision=revision+1 WHERE account_id=? AND device_installation_id=? AND environment=? AND purpose=? AND revision=?",
                (*params, expected_revision),
            ).rowcount
            if changed != 1:
                connection.rollback()
                raise StaleAuthorityRevisionError
            connection.execute(
                "DELETE FROM authority_key WHERE account_id=? AND device_installation_id=? AND environment=? AND purpose=?",
                params,
            )
            connection.executemany(
                "INSERT INTO authority_key VALUES (?,?,?,?,?,?,?)",
                [
                    (*params, key.authority_key_id, key.state.value, key.custody_handle)
                    for key in keys
                ],
            )
            if _before_commit is not None:
                _before_commit()
        except BackupAuthenticationError:
            connection.close()
            raise
        except Exception as exc:
            connection.rollback()
            connection.close()
            raise AuthorityUnavailableError from exc
        try:
            connection.commit()
        except sqlite3.Error:
            raise UnknownAdminMutationOutcomeError from None
        finally:
            connection.close()
        return expected_revision + 1, tuple(
            sorted(keys, key=lambda key: key.authority_key_id)
        )


class BackupArtifactAuthenticationAuthority:
    def __init__(
        self,
        metadata: SQLiteBackupAuthenticationMetadataStore,
        custody: BackupArtifactAuthenticationSecureCustody,
        *,
        _fault_hook: Callable[[str], None] | None = None,
    ) -> None:
        self._metadata, self._custody = metadata, custody
        self._fault_hook = _fault_hook or (lambda _point: None)

    def _after_staging(self) -> None:
        try:
            self._fault_hook("AFTER_KEY_DURABLE_PERSIST_BEFORE_METADATA_COMMIT")
        except Exception:
            raise AuthorityUnavailableError from None

    def _metadata_before_commit(self) -> None:
        self._fault_hook("BEFORE_METADATA_COMMIT")

    def _after_commit(self) -> None:
        try:
            self._fault_hook("AFTER_METADATA_COMMIT_BEFORE_RETURN")
        except Exception:
            raise UnknownAdminMutationOutcomeError from None

    @staticmethod
    def _public_snapshot(
        scope: BackupArtifactAuthenticationScope,
        committed: tuple[int, tuple[_StoredKey, ...]],
    ) -> BackupAuthenticationAuthorityScopeSnapshot:
        revision, keys = committed
        return BackupAuthenticationAuthorityScopeSnapshot(
            scope,
            revision,
            tuple(
                AuthorityKeyMetadata(key.authority_key_id, key.state) for key in keys
            ),
        )

    @staticmethod
    def _new_key() -> tuple[_StoredKey, bytes]:
        key_id = "bak_" + uuid4().hex
        return _StoredKey(
            key_id, AuthorityKeyState.ACTIVE, "material-" + uuid4().hex
        ), secrets.token_bytes(32)

    def read_snapshot(
        self, scope: BackupArtifactAuthenticationScope
    ) -> BackupAuthenticationAuthorityScopeSnapshot | None:
        stored = self._metadata.read(scope)
        if stored is None:
            return None
        revision, keys = stored
        snapshot = BackupAuthenticationAuthorityScopeSnapshot(
            scope,
            revision,
            tuple(
                AuthorityKeyMetadata(key.authority_key_id, key.state) for key in keys
            ),
        )
        snapshot.condition
        return snapshot

    def provision(
        self, scope: BackupArtifactAuthenticationScope
    ) -> BackupAuthenticationAuthorityScopeSnapshot:
        if self._metadata.read(scope) is not None:
            raise AlreadyProvisionedError
        key, material = self._new_key()
        self._custody.persist(key.custody_handle, material)
        self._after_staging()
        committed = self._metadata.create(
            scope, key, _before_commit=self._metadata_before_commit
        )
        result = self._public_snapshot(scope, committed)
        self._after_commit()
        return result

    def _existing(
        self, scope: BackupArtifactAuthenticationScope, expected_revision: int
    ) -> tuple[_StoredKey, ...]:
        try:
            _validate_authority_revision(expected_revision)
        except ValueError:
            raise InvalidAuthorityOperationError from None
        current = self._metadata.read(scope)
        if current is None:
            raise AuthorityNotProvisionedError
        revision, keys = current
        if revision != expected_revision:
            raise StaleAuthorityRevisionError
        return keys

    def rotate(
        self, scope: BackupArtifactAuthenticationScope, expected_revision: int
    ) -> BackupAuthenticationAuthorityScopeSnapshot:
        keys = self._existing(scope, expected_revision)
        if sum(key.state is AuthorityKeyState.ACTIVE for key in keys) != 1:
            raise NoActiveKeyError
        new, material = self._new_key()
        self._custody.persist(new.custody_handle, material)
        updated = tuple(
            _StoredKey(
                key.authority_key_id,
                (
                    AuthorityKeyState.VERIFY_ONLY
                    if key.state is AuthorityKeyState.ACTIVE
                    else key.state
                ),
                key.custody_handle,
            )
            for key in keys
        ) + (new,)
        self._after_staging()
        committed = self._metadata.replace(
            scope,
            expected_revision,
            updated,
            _before_commit=self._metadata_before_commit,
        )
        result = self._public_snapshot(scope, committed)
        self._after_commit()
        return result

    def rekey(
        self, scope: BackupArtifactAuthenticationScope, expected_revision: int
    ) -> BackupAuthenticationAuthorityScopeSnapshot:
        keys = self._existing(scope, expected_revision)
        if any(key.state is AuthorityKeyState.ACTIVE for key in keys):
            raise InvalidAuthorityOperationError
        new, material = self._new_key()
        self._custody.persist(new.custody_handle, material)
        self._after_staging()
        committed = self._metadata.replace(
            scope,
            expected_revision,
            keys + (new,),
            _before_commit=self._metadata_before_commit,
        )
        result = self._public_snapshot(scope, committed)
        self._after_commit()
        return result

    def revoke(
        self,
        scope: BackupArtifactAuthenticationScope,
        authority_key_id: str,
        expected_revision: int,
    ) -> BackupAuthenticationAuthorityScopeSnapshot:
        keys = self._existing(scope, expected_revision)
        matches = [key for key in keys if key.authority_key_id == authority_key_id]
        if not matches or matches[0].state is AuthorityKeyState.REVOKED:
            raise InvalidAuthorityOperationError
        updated = tuple(
            _StoredKey(
                key.authority_key_id,
                (
                    AuthorityKeyState.REVOKED
                    if key.authority_key_id == authority_key_id
                    else key.state
                ),
                key.custody_handle,
            )
            for key in keys
        )
        committed = self._metadata.replace(
            scope,
            expected_revision,
            updated,
            _before_commit=self._metadata_before_commit,
        )
        result = self._public_snapshot(scope, committed)
        self._after_commit()
        return result

    def authenticate(
        self,
        scope: BackupArtifactAuthenticationScope,
        manifest: PhysicalSQLiteArtifactManifest,
    ) -> PhysicalArtifactAuthenticationProof:
        if (
            manifest.account_id,
            manifest.device_installation_id,
            manifest.environment,
        ) != (scope.account_id, scope.device_installation_id, scope.environment):
            raise AuthorityUnavailableError
        stored = self._metadata.read(scope)
        if stored is None:
            raise AuthorityNotProvisionedError
        active = [key for key in stored[1] if key.state is AuthorityKeyState.ACTIVE]
        if len(active) > 1:
            raise AuthorityUnavailableError
        if not active:
            raise NoActiveKeyError
        try:
            tag = self._custody.digest(
                active[0].custody_handle, canonical_authentication_payload(manifest)
            )
        except Exception:
            raise AuthorityUnavailableError from None
        if tag is None:
            raise AuthorityUnavailableError
        return PhysicalArtifactAuthenticationProof(
            1,
            BACKUP_AUTHENTICATION_ALGORITHM,
            BACKUP_AUTHENTICATION_PURPOSE,
            active[0].authority_key_id,
            manifest.manifest_fingerprint_sha256,
            tag.hex(),
        )

    def verify(
        self,
        scope: BackupArtifactAuthenticationScope,
        manifest: PhysicalSQLiteArtifactManifest,
        proof: PhysicalArtifactAuthenticationProof,
    ) -> BackupArtifactVerificationResult:
        if (
            manifest.account_id,
            manifest.device_installation_id,
            manifest.environment,
        ) != (scope.account_id, scope.device_installation_id, scope.environment):
            return BackupArtifactVerificationResult.INVALID_PROOF
        if not hmac.compare_digest(
            proof.manifest_fingerprint_sha256, manifest.manifest_fingerprint_sha256
        ):
            return BackupArtifactVerificationResult.INVALID_PROOF
        try:
            stored = self._metadata.read(scope)
        except BackupAuthenticationError:
            return BackupArtifactVerificationResult.AUTHORITY_UNAVAILABLE
        if stored is None:
            return BackupArtifactVerificationResult.AUTHORITY_NOT_PROVISIONED
        matches = [
            key for key in stored[1] if key.authority_key_id == proof.authority_key_id
        ]
        if not matches:
            return BackupArtifactVerificationResult.UNKNOWN_KEY
        key = matches[0]
        if key.state is AuthorityKeyState.REVOKED:
            return BackupArtifactVerificationResult.REVOKED_KEY
        try:
            expected = self._custody.digest(
                key.custody_handle, canonical_authentication_payload(manifest)
            )
        except Exception:
            return BackupArtifactVerificationResult.AUTHORITY_UNAVAILABLE
        if expected is None:
            return BackupArtifactVerificationResult.AUTHORITY_UNAVAILABLE
        return (
            BackupArtifactVerificationResult.VERIFIED
            if hmac.compare_digest(expected.hex(), proof.authentication_tag_hex)
            else BackupArtifactVerificationResult.INVALID_PROOF
        )


class BackupArtifactAuthenticator:
    """Least-privilege creation capability."""

    def __init__(self, authority: BackupArtifactAuthenticationAuthority) -> None:
        self._authority = authority

    def authenticate(
        self,
        scope: BackupArtifactAuthenticationScope,
        manifest: PhysicalSQLiteArtifactManifest,
    ) -> PhysicalArtifactAuthenticationProof:
        return self._authority.authenticate(scope, manifest)


class BackupArtifactVerifier:
    """Least-privilege restore-admission capability."""

    def __init__(self, authority: BackupArtifactAuthenticationAuthority) -> None:
        self._authority = authority

    def verify(
        self,
        scope: BackupArtifactAuthenticationScope,
        manifest: PhysicalSQLiteArtifactManifest,
        proof: PhysicalArtifactAuthenticationProof,
    ) -> BackupArtifactVerificationResult:
        return self._authority.verify(scope, manifest, proof)
