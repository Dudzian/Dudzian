"""Protected local Ed25519 custody for isolated M0.5 signing authorities.

Provisioning is deliberately separated from runtime use.  Public metadata is a
crash-safe file, while the private seed is stored through the repository's
native-keyring ``SecretStorage`` boundary.  A runtime provider receives only a
read capability and can neither create nor replace key material.

This module is a custody/signing foundation.  The semantic root-proof and
history-head canonical signing domains have not been frozen, so each operation
signs the already-canonical bytes supplied by its distinct frozen role port.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
import os
from pathlib import Path
import re
import secrets
import stat
import uuid

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from bot_core.root_proof_issuer_substrate import (
    CredentialRoleIdentity,
    CredentialSemanticRole,
    ProviderCapabilities,
    ProviderIdentity,
    ProviderRole,
    SecurityProfile,
    SecurityProfileIdentity,
    SigningCapabilities,
    public_key_material_identity,
)
from bot_core.security.keyring_storage import KeyringSecretStorage


_SCHEMA = "cryptohunter.production-local-signing-custody"
_SCHEMA_VERSION = 1
_KEYRING_SERVICE_PREFIX = "dudzian.root-proof-issuer.production-local"
_LOCK_FILENAME = ".signing-custody.lock"
_RECORD_FILENAME = {
    ProviderRole.ROOT_PROOF_SIGNING: "root-proof-signing.json",
    ProviderRole.HISTORY_ATTESTATION_SIGNING: "history-attestation-signing.json",
    ProviderRole.FRESHNESS_AUTHORITY_FINALIZATION_SIGNING: (
        "freshness-authority-finalization-signing.json"
    ),
    ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING: (
        "cha-freshness-proposer-signing.json"
    ),
}
_SEMANTIC_ROLE = {
    ProviderRole.ROOT_PROOF_SIGNING: CredentialSemanticRole.ROOT_PROOF_ISSUER_SIGNING,
    ProviderRole.HISTORY_ATTESTATION_SIGNING: CredentialSemanticRole.HISTORY_ATTESTATION_SIGNING,
    ProviderRole.FRESHNESS_AUTHORITY_FINALIZATION_SIGNING: (
        CredentialSemanticRole.ACCOUNT_GENESIS_FRESHNESS_AUTHORITY_FINALIZATION_SIGNING_V1
    ),
    ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING: (
        CredentialSemanticRole.ACCOUNT_GENESIS_FRESHNESS_PROPOSER_SIGNING_V1
    ),
}


class SigningKeyLifecycle(str, Enum):
    ACTIVE = "ACTIVE"
    VERIFY_ONLY = "VERIFY_ONLY"
    REVOKED = "REVOKED"


class LocalSigningCustodyError(RuntimeError):
    """Fail-closed custody loading or signing failure."""


class LocalSigningProvisioningConflict(LocalSigningCustodyError):
    """An existing authority conflicts with the requested provisioning."""


class LocalSigningLifecycleError(LocalSigningCustodyError):
    """An illegal lifecycle transition or signing attempt."""


@dataclass(frozen=True, slots=True)
class _AuthoritySecretScope:
    security: SecurityProfileIdentity
    role: ProviderRole
    scope_identity: str
    service_name: str
    reference_prefix: str


@dataclass(frozen=True, slots=True)
class _CanonicalSigningIdentity:
    credential_id: str
    provider_namespace: str
    key_handle: str
    key_version: int
    lifecycle_namespace: str
    protected_private_material_reference: str


def _exact_path_snapshot(candidate: Path, *, label: str) -> Path:
    if type(candidate) is not type(Path()):
        raise LocalSigningCustodyError(f"{label} must be an exact platform Path")
    snapshot = Path(os.fspath(candidate))
    if not snapshot.is_absolute():
        raise LocalSigningCustodyError(f"{label} must be absolute")
    return snapshot


def _authority_secret_scope(
    security: SecurityProfileIdentity, role: ProviderRole
) -> _AuthoritySecretScope:
    security = _snapshot_security(security)
    if type(role) is not ProviderRole or role not in _RECORD_FILENAME:
        raise LocalSigningCustodyError("invalid signing authority role")
    canonical = json.dumps(
        ["production-local-signing-secret-v1", security.trust_domain, role.value],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    scope_identity = hashlib.sha256(canonical).hexdigest()
    return _AuthoritySecretScope(
        security,
        role,
        scope_identity,
        f"{_KEYRING_SERVICE_PREFIX}.{scope_identity}",
        f"signing-secret:v1:{scope_identity}:",
    )


def _expected_secret_reference(
    scope: _AuthoritySecretScope, credential_identity: str
) -> str:
    if type(credential_identity) is not str or not credential_identity.strip():
        raise LocalSigningCustodyError("invalid credential identity")
    binding = hashlib.sha256(
        json.dumps(
            [scope.scope_identity, credential_identity], separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()[:32]
    return scope.reference_prefix + binding


def _canonical_signing_identity(
    security: SecurityProfileIdentity,
    role: ProviderRole,
    unique: str,
    *,
    key_version: int = 1,
) -> _CanonicalSigningIdentity:
    scope = _authority_secret_scope(security, role)
    if (
        type(unique) is not str
        or re.fullmatch(r"[0-9a-f]{32}", unique) is None
        or type(key_version) is not int
        or key_version < 1
    ):
        raise LocalSigningCustodyError("invalid provisioning identity")
    role_token = role.value.lower().replace("_", "-")
    credential_id = f"{role_token}-{unique}"
    provider_namespace = f"{security.trust_domain}.production-local.{role_token}"
    return _CanonicalSigningIdentity(
        credential_id=credential_id,
        provider_namespace=provider_namespace,
        key_handle=f"{role_token}-key-{unique}",
        key_version=key_version,
        lifecycle_namespace=f"{provider_namespace}.lifecycle",
        protected_private_material_reference=_expected_secret_reference(
            scope, credential_id
        ),
    )


def _provisioning_unique_from_credential_id(
    credential_id: object, role: ProviderRole
) -> str:
    role_token = role.value.lower().replace("_", "-")
    prefix = f"{role_token}-"
    if type(credential_id) is not str or not credential_id.startswith(prefix):
        raise LocalSigningCustodyError("custody material unavailable/corrupt")
    unique = credential_id[len(prefix) :]
    if re.fullmatch(r"[0-9a-f]{32}", unique) is None:
        raise LocalSigningCustodyError("custody material unavailable/corrupt")
    return unique


class NativeKeyringSigningSecretReader:
    """Sealed read-only runtime facade over the reviewed native-keyring store."""

    __slots__ = ("__storage", "__scope", "__sealed")

    def __init__(
        self,
        *,
        index_path: Path,
        security: SecurityProfileIdentity,
        role: ProviderRole,
    ) -> None:
        path = _exact_path_snapshot(index_path, label="keyring index path")
        scope = _authority_secret_scope(security, role)
        storage = KeyringSecretStorage(
            service_name=scope.service_name, index_path=path
        )
        object.__setattr__(self, "_NativeKeyringSigningSecretReader__storage", storage)
        object.__setattr__(self, "_NativeKeyringSigningSecretReader__scope", scope)
        object.__setattr__(self, "_NativeKeyringSigningSecretReader__sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("runtime signing secret reader is immutable")

    @property
    def service_namespace(self) -> str:
        return self.__scope.service_name

    def read_authority_secret(self, expected_reference: str) -> str | None:
        if (
            type(expected_reference) is not str
            or not expected_reference.startswith(self.__scope.reference_prefix)
            or len(expected_reference) != len(self.__scope.reference_prefix) + 32
        ):
            raise LocalSigningCustodyError("secret reference is outside authority scope")
        storage = self.__storage
        if (
            type(storage) is not KeyringSecretStorage
            or storage._service_name != self.__scope.service_name
        ):
            raise LocalSigningCustodyError("runtime signing secret reader is invalid")
        return storage.get_secret(expected_reference)


class NativeKeyringSigningSecretAdministrator:
    """Offline-only create/delete facade over the reviewed native-keyring store."""

    __slots__ = ("__storage", "__scope", "__sealed")

    def __init__(
        self,
        *,
        index_path: Path,
        security: SecurityProfileIdentity,
        role: ProviderRole,
    ) -> None:
        path = _exact_path_snapshot(index_path, label="keyring index path")
        scope = _authority_secret_scope(security, role)
        storage = KeyringSecretStorage(
            service_name=scope.service_name, index_path=path
        )
        object.__setattr__(self, "_NativeKeyringSigningSecretAdministrator__storage", storage)
        object.__setattr__(self, "_NativeKeyringSigningSecretAdministrator__scope", scope)
        object.__setattr__(self, "_NativeKeyringSigningSecretAdministrator__sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("signing secret administrator is immutable")

    @property
    def service_namespace(self) -> str:
        return self.__scope.service_name

    def matches_authority(
        self, security: SecurityProfileIdentity, role: ProviderRole
    ) -> bool:
        try:
            return self.__scope == _authority_secret_scope(security, role)
        except LocalSigningCustodyError:
            return False

    def new_secret_reference(self, credential_identity: str) -> str:
        return _expected_secret_reference(self.__scope, credential_identity)

    def read_authority_secret(self, expected_reference: str) -> str | None:
        self._validate_reference(expected_reference)
        return self.__storage.get_secret(expected_reference)

    def create_authority_secret(self, expected_reference: str, value: str) -> None:
        self._validate_reference(expected_reference)
        self.__storage.set_secret(expected_reference, value)

    def delete_authority_secret(self, expected_reference: str) -> None:
        self._validate_reference(expected_reference)
        self.__storage.delete_secret(expected_reference)

    def _validate_reference(self, expected_reference: str) -> None:
        if (
            type(expected_reference) is not str
            or not expected_reference.startswith(self.__scope.reference_prefix)
            or len(expected_reference) != len(self.__scope.reference_prefix) + 32
            or type(self.__storage) is not KeyringSecretStorage
            or self.__storage._service_name != self.__scope.service_name
        ):
            raise LocalSigningCustodyError("secret reference is outside authority scope")


@dataclass(frozen=True, slots=True)
class LocalSigningMetadata:
    security_profile: SecurityProfile
    trust_domain: str
    provider_role: ProviderRole
    credential_semantic_role: CredentialSemanticRole
    credential_id: str
    provider_namespace: str
    key_handle: str
    key_version: int
    lifecycle_namespace: str
    lifecycle_generation: int
    lifecycle_state: SigningKeyLifecycle
    public_key: bytes
    protected_private_material_reference: str
    storage_schema: str = _SCHEMA
    storage_schema_version: int = _SCHEMA_VERSION

    @property
    def key_material_identity(self) -> str:
        return public_key_material_identity(self.public_key)

    def credential_identity(self) -> CredentialRoleIdentity:
        opaque_handle = self.key_handle
        if self.provider_role in {
            ProviderRole.ROOT_PROOF_SIGNING,
            ProviderRole.HISTORY_ATTESTATION_SIGNING,
        }:
            # Preserve the accepted legacy identity byte-for-byte.  It remains
            # opaque; numeric key_version is sourced only from its typed field.
            opaque_handle = f"{self.key_handle}:v{self.key_version}"
        return CredentialRoleIdentity(
            self.credential_semantic_role,
            self.credential_id,
            self.provider_namespace,
            opaque_handle,
            self.lifecycle_namespace,
            self.key_material_identity,
        )


@dataclass(frozen=True, slots=True)
class SigningCredentialSnapshot:
    """Exact provider-originated facts; the opaque handle is never parsed."""

    provider_identity: ProviderIdentity
    credential_identity: CredentialRoleIdentity
    key_version: int
    lifecycle_generation: int
    lifecycle_state: SigningKeyLifecycle
    public_key: bytes
    key_material_identity: str

    def __post_init__(self) -> None:
        if (
            type(self.provider_identity) is not ProviderIdentity
            or type(self.credential_identity) is not CredentialRoleIdentity
            or type(self.key_version) is not int
            or self.key_version < 1
            or type(self.lifecycle_generation) is not int
            or self.lifecycle_generation < 1
            or type(self.lifecycle_state) is not SigningKeyLifecycle
            or type(self.public_key) is not bytes
            or len(self.public_key) != 32
            or type(self.key_material_identity) is not str
            or self.key_material_identity
            != public_key_material_identity(self.public_key)
        ):
            raise TypeError("invalid exact signing credential snapshot")


@dataclass(frozen=True, slots=True)
class LocalSigningResult:
    """Signature and the exact ACTIVE snapshot linearized with its private key."""

    snapshot: SigningCredentialSnapshot
    signature: bytes

    def __post_init__(self) -> None:
        if (
            type(self.snapshot) is not SigningCredentialSnapshot
            or self.snapshot.lifecycle_state is not SigningKeyLifecycle.ACTIVE
            or type(self.signature) is not bytes
            or len(self.signature) != 64
        ):
            raise TypeError("invalid local signing result")


_LOCAL_CAPABILITIES = ProviderCapabilities(
    implemented=True,
    signing=SigningCapabilities(
        ed25519=True,
        durable_key_identity=True,
        hardware_or_equivalent_secure_custody=False,
        plaintext_private_key_export_forbidden=False,
        role_isolation=True,
        lifecycle_support=True,
        stable_provider_namespace=True,
        stable_key_handle_or_version_identity=True,
    ),
)


def _validate_directory(directory: Path, *, create: bool) -> Path:
    path = _exact_path_snapshot(directory, label="custody directory")
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.exists() or not path.is_dir() or path.is_symlink():
        raise LocalSigningCustodyError("custody directory is unavailable or unsafe")
    if os.name == "posix":
        info = path.stat()
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o022:
            raise LocalSigningCustodyError("custody directory ownership or permissions are unsafe")
    return path


def _snapshot_security(security: SecurityProfileIdentity) -> SecurityProfileIdentity:
    if (
        type(security) is not SecurityProfileIdentity
        or type(security.profile) is not SecurityProfile
        or type(security.trust_domain) is not str
        or not security.trust_domain.strip()
    ):
        raise LocalSigningCustodyError("invalid security profile identity")
    return SecurityProfileIdentity(security.profile, security.trust_domain)


@contextmanager
def _custody_lock(directory: Path, *, exclusive: bool):
    """Cross-process advisory lock shared by both signing roles and lifecycle."""

    if os.name != "posix":
        raise LocalSigningCustodyError("production-local custody locking is unavailable")
    import fcntl

    path = directory / _LOCK_FILENAME
    try:
        descriptor = os.open(
            path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600
        )
    except OSError as exc:
        raise LocalSigningCustodyError("custody locking failed") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise LocalSigningCustodyError("custody lock ownership or permissions are unsafe")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        except OSError as exc:
            raise LocalSigningCustodyError("custody locking failed") from exc
        try:
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError as exc:
                raise LocalSigningCustodyError("custody unlock failed") from exc
    finally:
        os.close(descriptor)


def _record_path(directory: Path, role: ProviderRole) -> Path:
    try:
        filename = _RECORD_FILENAME[role]
    except KeyError as exc:
        raise LocalSigningCustodyError("unsupported signing authority role") from exc
    return directory / filename


def _stored_public_keys(directory: Path) -> tuple[bytes, ...]:
    """Read public material from every current or retained custody record."""

    keys: list[bytes] = []
    for candidate in directory.glob("*signing*.json"):
        try:
            document = json.loads(candidate.read_text(encoding="utf-8"))
            if type(document) is not dict or "public_key_b64" not in document:
                raise ValueError
            public_key = base64.b64decode(
                document["public_key_b64"], validate=True
            )
            if len(public_key) != 32:
                raise ValueError
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LocalSigningCustodyError(
                "custody material unavailable/corrupt"
            ) from exc
        keys.append(public_key)
    return tuple(keys)


def _encode_record(metadata: LocalSigningMetadata) -> bytes:
    document = {
        "storage_schema": metadata.storage_schema,
        "storage_schema_version": metadata.storage_schema_version,
        "security_profile": metadata.security_profile.value,
        "trust_domain": metadata.trust_domain,
        "provider_role": metadata.provider_role.value,
        "credential_semantic_role": metadata.credential_semantic_role.value,
        "credential_id": metadata.credential_id,
        "provider_namespace": metadata.provider_namespace,
        "key_handle": metadata.key_handle,
        "key_version": metadata.key_version,
        "lifecycle_namespace": metadata.lifecycle_namespace,
        "lifecycle_generation": metadata.lifecycle_generation,
        "lifecycle_state": metadata.lifecycle_state.value,
        "public_key_b64": base64.b64encode(metadata.public_key).decode("ascii"),
        "protected_private_material_reference": metadata.protected_private_material_reference,
    }
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _decode_record(raw: bytes, *, expected_security: SecurityProfileIdentity, expected_role: ProviderRole) -> LocalSigningMetadata:
    try:
        value = json.loads(raw.decode("utf-8"))
        if type(value) is not dict or set(value) != {
            "storage_schema", "storage_schema_version", "security_profile", "trust_domain",
            "provider_role", "credential_semantic_role", "credential_id", "provider_namespace",
            "key_handle", "key_version", "lifecycle_namespace", "lifecycle_generation",
            "lifecycle_state", "public_key_b64", "protected_private_material_reference",
        }:
            raise ValueError
        metadata = LocalSigningMetadata(
            security_profile=SecurityProfile(value["security_profile"]),
            trust_domain=value["trust_domain"],
            provider_role=ProviderRole(value["provider_role"]),
            credential_semantic_role=CredentialSemanticRole(value["credential_semantic_role"]),
            credential_id=value["credential_id"], provider_namespace=value["provider_namespace"],
            key_handle=value["key_handle"], key_version=value["key_version"],
            lifecycle_namespace=value["lifecycle_namespace"],
            lifecycle_generation=value["lifecycle_generation"],
            lifecycle_state=SigningKeyLifecycle(value["lifecycle_state"]),
            public_key=base64.b64decode(value["public_key_b64"], validate=True),
            protected_private_material_reference=value["protected_private_material_reference"],
            storage_schema=value["storage_schema"], storage_schema_version=value["storage_schema_version"],
        )
    except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise LocalSigningCustodyError("custody material unavailable/corrupt") from exc
    _validate_metadata_identity_binding(
        metadata,
        expected_security=expected_security,
        expected_role=expected_role,
    )
    return metadata


def _validate_metadata_identity_binding(
    metadata: LocalSigningMetadata,
    *,
    expected_security: SecurityProfileIdentity,
    expected_role: ProviderRole,
) -> None:
    """Prove that one record could have been emitted by current provisioning."""

    try:
        unique = _provisioning_unique_from_credential_id(
            metadata.credential_id, expected_role
        )
        canonical = _canonical_signing_identity(
            expected_security,
            expected_role,
            unique,
            key_version=metadata.key_version,
        )
    except (
        AttributeError,
        KeyError,
        LocalSigningCustodyError,
        TypeError,
        ValueError,
    ) as exc:
        raise LocalSigningCustodyError("custody material unavailable/corrupt") from exc
    strings = (
        metadata.trust_domain, metadata.credential_id, metadata.provider_namespace,
        metadata.key_handle, metadata.lifecycle_namespace,
        metadata.protected_private_material_reference,
    )
    lifecycle_generation_is_legal = {
        SigningKeyLifecycle.ACTIVE: {1},
        SigningKeyLifecycle.VERIFY_ONLY: {2},
        SigningKeyLifecycle.REVOKED: {2, 3},
    }.get(metadata.lifecycle_state, set())
    if (
        type(metadata) is not LocalSigningMetadata
        or metadata.storage_schema != _SCHEMA
        or type(metadata.storage_schema_version) is not int
        or metadata.storage_schema_version != _SCHEMA_VERSION
        or metadata.security_profile is not SecurityProfile.PRODUCTION_LOCAL
        or metadata.security_profile is not expected_security.profile
        or metadata.trust_domain != expected_security.trust_domain
        or metadata.provider_role is not expected_role
        or metadata.credential_semantic_role is not _SEMANTIC_ROLE[expected_role]
        or any(type(item) is not str or not item.strip() for item in strings)
        or metadata.credential_id != canonical.credential_id
        or metadata.provider_namespace != canonical.provider_namespace
        or metadata.key_handle != canonical.key_handle
        or type(metadata.key_version) is not int
        or metadata.key_version < 1
        or (
            expected_role
            in {
                ProviderRole.ROOT_PROOF_SIGNING,
                ProviderRole.HISTORY_ATTESTATION_SIGNING,
            }
            and metadata.key_version != 1
        )
        or metadata.key_version != canonical.key_version
        or metadata.lifecycle_namespace != canonical.lifecycle_namespace
        or metadata.protected_private_material_reference
        != canonical.protected_private_material_reference
        or type(metadata.lifecycle_generation) is not int
        or metadata.lifecycle_generation not in lifecycle_generation_is_legal
        or type(metadata.public_key) is not bytes
        or len(metadata.public_key) != 32
    ):
        raise LocalSigningCustodyError("custody material unavailable/corrupt")


def _read_record(path: Path, security: SecurityProfileIdentity, role: ProviderRole) -> LocalSigningMetadata:
    try:
        if path.is_symlink():
            raise LocalSigningCustodyError("custody material unavailable/corrupt")
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise LocalSigningCustodyError("custody material unavailable/corrupt")
        if os.name == "posix" and (info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077):
            raise LocalSigningCustodyError("custody material unavailable/corrupt")
        raw = path.read_bytes()
    except (OSError, LocalSigningCustodyError) as exc:
        raise LocalSigningCustodyError("custody material unavailable/corrupt") from exc
    return _decode_record(raw, expected_security=security, expected_role=role)


def _load_private_key(
    metadata: LocalSigningMetadata, secret_reader: NativeKeyringSigningSecretReader | NativeKeyringSigningSecretAdministrator
) -> Ed25519PrivateKey:
    try:
        encoded = secret_reader.read_authority_secret(
            metadata.protected_private_material_reference
        )
        if type(encoded) is not str:
            raise ValueError
        seed = base64.b64decode(encoded, validate=True)
        if len(seed) != 32:
            raise ValueError
        private_key = Ed25519PrivateKey.from_private_bytes(seed)
        derived = private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        if not secrets.compare_digest(derived, metadata.public_key):
            raise ValueError
    except Exception as exc:
        raise LocalSigningCustodyError("custody material unavailable/corrupt") from exc
    return private_key


def _fsync_directory(directory: Path) -> None:
    if os.name == "posix":
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _write_new_record(path: Path, payload: bytes) -> bool:
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
            won = True
        except FileExistsError:
            won = False
        _fsync_directory(path.parent)
        return won
    finally:
        temporary.unlink(missing_ok=True)


def _replace_record(path: Path, payload: bytes) -> None:
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def provision_local_signing_authority(
    directory: Path,
    secret_store: NativeKeyringSigningSecretAdministrator,
    *,
    security: SecurityProfileIdentity,
    role: ProviderRole,
    private_seed: bytes | None = None,
) -> LocalSigningMetadata:
    """Offline/admin-only, idempotent and concurrent-safe authority provisioning."""

    root = _validate_directory(directory, create=True)
    security = _snapshot_security(security)
    if type(secret_store) is not NativeKeyringSigningSecretAdministrator:
        raise LocalSigningCustodyError("unqualified production signing secret administrator")
    if security.profile is not SecurityProfile.PRODUCTION_LOCAL or type(role) is not ProviderRole or role not in _RECORD_FILENAME:
        raise LocalSigningProvisioningConflict("only a production-local signing role may be provisioned")
    if not secret_store.matches_authority(security, role):
        raise LocalSigningCustodyError("signing secret administrator authority mismatch")
    if private_seed is not None and (type(private_seed) is not bytes or len(private_seed) != 32):
        raise LocalSigningProvisioningConflict("invalid Ed25519 provisioning material")
    with _custody_lock(root, exclusive=True):
        path = _record_path(root, role)
        if path.exists() or path.is_symlink():
            existing = _read_record(path, security, role)
            _load_private_key(existing, secret_store)
            if private_seed is not None:
                requested_public = Ed25519PrivateKey.from_private_bytes(private_seed).public_key().public_bytes_raw()
                if not secrets.compare_digest(existing.public_key, requested_public):
                    raise LocalSigningProvisioningConflict("authority is already provisioned differently")
            return existing

        seed = private_seed if private_seed is not None else Ed25519PrivateKey.generate().private_bytes_raw()
        public_key = Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes_raw()
        if any(
            secrets.compare_digest(observed, public_key)
            for observed in _stored_public_keys(root)
        ):
            raise LocalSigningProvisioningConflict(
                "physical key material aliases another authority"
            )

        unique = uuid.uuid4().hex
        canonical = _canonical_signing_identity(security, role, unique)
        metadata = LocalSigningMetadata(
            security.profile, security.trust_domain, role, _SEMANTIC_ROLE[role],
            canonical.credential_id, canonical.provider_namespace,
            canonical.key_handle, canonical.key_version,
            canonical.lifecycle_namespace, 1, SigningKeyLifecycle.ACTIVE,
            public_key, canonical.protected_private_material_reference,
        )
        secret_store.create_authority_secret(
            canonical.protected_private_material_reference,
            base64.b64encode(seed).decode("ascii"),
        )
        try:
            if not _write_new_record(path, _encode_record(metadata)):
                raise LocalSigningProvisioningConflict("concurrent provisioning conflict")
            return metadata
        except Exception:
            secret_store.delete_authority_secret(
                canonical.protected_private_material_reference
            )
            raise


def transition_local_signing_lifecycle(
    directory: Path,
    *,
    security: SecurityProfileIdentity,
    role: ProviderRole,
    target: SigningKeyLifecycle,
) -> LocalSigningMetadata:
    """Offline/admin lifecycle update with the frozen one-way transition graph."""

    root = _validate_directory(directory, create=False)
    security = _snapshot_security(security)
    if type(role) is not ProviderRole or role not in _RECORD_FILENAME:
        raise LocalSigningLifecycleError("invalid signing authority role")
    if type(target) is not SigningKeyLifecycle:
        raise LocalSigningLifecycleError("illegal signing-key lifecycle transition")
    allowed = {
        (SigningKeyLifecycle.ACTIVE, SigningKeyLifecycle.VERIFY_ONLY),
        (SigningKeyLifecycle.ACTIVE, SigningKeyLifecycle.REVOKED),
        (SigningKeyLifecycle.VERIFY_ONLY, SigningKeyLifecycle.REVOKED),
    }
    with _custody_lock(root, exclusive=True):
        path = _record_path(root, role)
        current = _read_record(path, security, role)
        if (current.lifecycle_state, target) not in allowed:
            raise LocalSigningLifecycleError("illegal signing-key lifecycle transition")
        successor = LocalSigningMetadata(
            current.security_profile, current.trust_domain, current.provider_role,
            current.credential_semantic_role, current.credential_id, current.provider_namespace,
            current.key_handle, current.key_version, current.lifecycle_namespace,
            current.lifecycle_generation + 1, target, current.public_key,
            current.protected_private_material_reference,
        )
        _replace_record(path, _encode_record(successor))
        return successor


def rotate_local_freshness_signing_authority(
    directory: Path,
    secret_store: NativeKeyringSigningSecretAdministrator,
    *,
    security: SecurityProfileIdentity,
    role: ProviderRole,
    private_seed: bytes | None = None,
) -> LocalSigningMetadata:
    """Offline planned rotation: retain old VERIFY_ONLY, publish new ACTIVE."""

    freshness_roles = {
        ProviderRole.FRESHNESS_AUTHORITY_FINALIZATION_SIGNING,
        ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING,
    }
    root = _validate_directory(directory, create=False)
    security = _snapshot_security(security)
    if type(role) is not ProviderRole or role not in freshness_roles:
        raise LocalSigningLifecycleError("rotation is limited to freshness signing roles")
    if (
        type(secret_store) is not NativeKeyringSigningSecretAdministrator
        or not secret_store.matches_authority(security, role)
    ):
        raise LocalSigningCustodyError("signing secret administrator authority mismatch")
    if private_seed is not None and (
        type(private_seed) is not bytes or len(private_seed) != 32
    ):
        raise LocalSigningProvisioningConflict("invalid Ed25519 provisioning material")
    with _custody_lock(root, exclusive=True):
        path = _record_path(root, role)
        current = _read_record(path, security, role)
        if current.lifecycle_state is not SigningKeyLifecycle.ACTIVE:
            raise LocalSigningLifecycleError("only ACTIVE credential may be rotated")
        seed = (
            private_seed
            if private_seed is not None
            else Ed25519PrivateKey.generate().private_bytes_raw()
        )
        public_key = (
            Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes_raw()
        )
        if any(
            secrets.compare_digest(observed, public_key)
            for observed in _stored_public_keys(root)
        ):
            raise LocalSigningProvisioningConflict(
                "physical key material aliases another authority"
            )
        retained = LocalSigningMetadata(
            current.security_profile,
            current.trust_domain,
            current.provider_role,
            current.credential_semantic_role,
            current.credential_id,
            current.provider_namespace,
            current.key_handle,
            current.key_version,
            current.lifecycle_namespace,
            current.lifecycle_generation + 1,
            SigningKeyLifecycle.VERIFY_ONLY,
            current.public_key,
            current.protected_private_material_reference,
        )
        archive = root / f"{path.stem}.v{current.key_version}.verify-only.json"
        if archive.exists() or not _write_new_record(archive, _encode_record(retained)):
            raise LocalSigningProvisioningConflict("rotation archive already exists")
        unique = uuid.uuid4().hex
        canonical = _canonical_signing_identity(
            security, role, unique, key_version=current.key_version + 1
        )
        successor = LocalSigningMetadata(
            security.profile,
            security.trust_domain,
            role,
            _SEMANTIC_ROLE[role],
            canonical.credential_id,
            canonical.provider_namespace,
            canonical.key_handle,
            canonical.key_version,
            canonical.lifecycle_namespace,
            1,
            SigningKeyLifecycle.ACTIVE,
            public_key,
            canonical.protected_private_material_reference,
        )
        secret_store.create_authority_secret(
            canonical.protected_private_material_reference,
            base64.b64encode(seed).decode("ascii"),
        )
        try:
            _replace_record(path, _encode_record(successor))
        except Exception:
            secret_store.delete_authority_secret(
                canonical.protected_private_material_reference
            )
            archive.unlink(missing_ok=True)
            raise
        return successor


class _LocalSigningProviderBase:
    __slots__ = ("__directory", "__secret_reader", "__security", "__role")

    def __init__(
        self,
        directory: Path,
        *,
        keyring_index_path: Path,
        security: SecurityProfileIdentity,
        role: ProviderRole,
    ) -> None:
        trusted_directory = _validate_directory(directory, create=False)
        trusted_security = _snapshot_security(security)
        reader = NativeKeyringSigningSecretReader(
            index_path=keyring_index_path,
            security=trusted_security,
            role=role,
        )
        object.__setattr__(self, "_LocalSigningProviderBase__directory", trusted_directory)
        object.__setattr__(self, "_LocalSigningProviderBase__security", trusted_security)
        object.__setattr__(self, "_LocalSigningProviderBase__role", role)
        object.__setattr__(self, "_LocalSigningProviderBase__secret_reader", reader)
        self._load_material()

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("local signing provider is immutable")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(role={self.__role.value!r}, trust_domain={self.__security.trust_domain!r})"

    @property
    def identity(self) -> ProviderIdentity:
        metadata, _ = self._load_material()
        return ProviderIdentity(self.__role, self.__security, metadata.provider_namespace)

    @property
    def capabilities(self) -> ProviderCapabilities:
        return _LOCAL_CAPABILITIES

    def credential_identities(self) -> tuple[CredentialRoleIdentity, ...]:
        metadata, _ = self._load_material()
        return (metadata.credential_identity(),)

    def active_credential_identity(self) -> CredentialRoleIdentity:
        with _custody_lock(self.__directory, exclusive=False):
            metadata, _ = self._load_material()
            if metadata.lifecycle_state is not SigningKeyLifecycle.ACTIVE:
                raise LocalSigningLifecycleError("signing credential is not ACTIVE")
            return metadata.credential_identity()

    def public_key(self, credential_identity: str) -> bytes:
        metadata, _ = self._load_material()
        if credential_identity != metadata.credential_id:
            raise LocalSigningCustodyError("unknown signing credential")
        return metadata.public_key

    def lifecycle_generation(self) -> int:
        metadata, _ = self._load_material()
        return metadata.lifecycle_generation

    def lifecycle_state(self) -> SigningKeyLifecycle:
        metadata, _ = self._load_material()
        return metadata.lifecycle_state

    def credential_snapshot(self) -> SigningCredentialSnapshot:
        """Return one deep, exact observation of identity and lifecycle facts."""

        with _custody_lock(self.__directory, exclusive=False):
            metadata, _ = self._load_material()
            return self._snapshot(metadata)

    def _snapshot(self, metadata: LocalSigningMetadata) -> SigningCredentialSnapshot:
        return SigningCredentialSnapshot(
            ProviderIdentity(
                self.__role, self.__security, metadata.provider_namespace
            ),
            metadata.credential_identity(),
            metadata.key_version,
            metadata.lifecycle_generation,
            metadata.lifecycle_state,
            bytes(metadata.public_key),
            metadata.key_material_identity,
        )

    def _sign(self, payload: bytes) -> bytes:
        return self._sign_result(payload).signature

    def _sign_result(self, payload: bytes) -> LocalSigningResult:
        if type(payload) is not bytes:
            raise TypeError("canonical signing payload must be exact bytes")
        with _custody_lock(self.__directory, exclusive=False):
            metadata, private_key = self._load_material()
            if metadata.lifecycle_state is not SigningKeyLifecycle.ACTIVE:
                raise LocalSigningLifecycleError("signing credential is not ACTIVE")
            signature = private_key.sign(payload)
            if len(signature) != 64:
                raise LocalSigningCustodyError("invalid Ed25519 signature result")
            return LocalSigningResult(self._snapshot(metadata), bytes(signature))

    def verify_historical(
        self,
        canonical_payload: bytes,
        signature: bytes,
        expected_snapshot: SigningCredentialSnapshot,
    ) -> bool:
        """Verify retained material without allowing REVOKED to create trust."""

        if (
            type(canonical_payload) is not bytes
            or type(signature) is not bytes
            or type(expected_snapshot) is not SigningCredentialSnapshot
        ):
            raise TypeError("historical verification requires exact values")
        with _custody_lock(self.__directory, exclusive=False):
            metadata = _read_record(
                _record_path(self.__directory, self.__role),
                self.__security,
                self.__role,
            )
            current = self._snapshot(metadata)
            if current.credential_identity != expected_snapshot.credential_identity:
                path = _record_path(self.__directory, self.__role)
                archives = self.__directory.glob(
                    f"{path.stem}.v*.verify-only.json"
                )
                matches = []
                for archive in archives:
                    candidate = _read_record(
                        archive, self.__security, self.__role
                    )
                    if (
                        candidate.credential_identity()
                        == expected_snapshot.credential_identity
                    ):
                        matches.append(self._snapshot(candidate))
                if len(matches) == 1:
                    current = matches[0]
            same_credential_facts = (
                current.provider_identity == expected_snapshot.provider_identity
                and current.credential_identity
                == expected_snapshot.credential_identity
                and current.key_version == expected_snapshot.key_version
                and current.public_key == expected_snapshot.public_key
                and current.key_material_identity
                == expected_snapshot.key_material_identity
            )
            if not same_credential_facts:
                raise LocalSigningCustodyError(
                    "historical credential facts are stale or do not match"
                )
            if current.lifecycle_state is SigningKeyLifecycle.REVOKED:
                raise LocalSigningLifecycleError(
                    "REVOKED credential cannot establish historical trust"
                )
            try:
                Ed25519PublicKey.from_public_bytes(current.public_key).verify(
                    signature, canonical_payload
                )
            except InvalidSignature:
                return False
            return True

    def _load_material(self) -> tuple[LocalSigningMetadata, Ed25519PrivateKey]:
        metadata = _read_record(
            _record_path(self.__directory, self.__role), self.__security, self.__role
        )
        return metadata, _load_private_key(metadata, self.__secret_reader)


class LocalRootProofSigningProvider(_LocalSigningProviderBase):
    __slots__ = ()

    def __init__(self, directory: Path, *, keyring_index_path: Path, security: SecurityProfileIdentity) -> None:
        super().__init__(directory, keyring_index_path=keyring_index_path, security=security, role=ProviderRole.ROOT_PROOF_SIGNING)

    def sign_root_proof(self, canonical_payload: bytes) -> bytes:
        return self._sign(canonical_payload)


class LocalHistoryAttestationSigningProvider(_LocalSigningProviderBase):
    __slots__ = ()

    def __init__(self, directory: Path, *, keyring_index_path: Path, security: SecurityProfileIdentity) -> None:
        super().__init__(directory, keyring_index_path=keyring_index_path, security=security, role=ProviderRole.HISTORY_ATTESTATION_SIGNING)

    def sign_history_head(self, canonical_head: bytes) -> bytes:
        return self._sign(canonical_head)


class LocalFreshnessAuthorityFinalizationSigningProvider(_LocalSigningProviderBase):
    """Local finalization signer only; this is not a FreshnessAuthority."""

    __slots__ = ()

    def __init__(
        self,
        directory: Path,
        *,
        keyring_index_path: Path,
        security: SecurityProfileIdentity,
    ) -> None:
        super().__init__(
            directory,
            keyring_index_path=keyring_index_path,
            security=security,
            role=ProviderRole.FRESHNESS_AUTHORITY_FINALIZATION_SIGNING,
        )

    def sign_finalization(self, canonical_payload: bytes) -> LocalSigningResult:
        return self._sign_result(canonical_payload)


class LocalCHAFreshnessProposerSigningProvider(_LocalSigningProviderBase):
    """Local CHA proposer signer; it conveys no FreshnessAuthority trust."""

    __slots__ = ()

    def __init__(
        self,
        directory: Path,
        *,
        keyring_index_path: Path,
        security: SecurityProfileIdentity,
    ) -> None:
        super().__init__(
            directory,
            keyring_index_path=keyring_index_path,
            security=security,
            role=ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING,
        )

    def sign_freshness_proposal(
        self, canonical_payload: bytes
    ) -> LocalSigningResult:
        return self._sign_result(canonical_payload)


__all__ = [
    "LocalCHAFreshnessProposerSigningProvider",
    "LocalFreshnessAuthorityFinalizationSigningProvider",
    "LocalHistoryAttestationSigningProvider", "LocalRootProofSigningProvider",
    "LocalSigningCustodyError", "LocalSigningLifecycleError", "LocalSigningMetadata",
    "LocalSigningResult", "SigningCredentialSnapshot",
    "LocalSigningProvisioningConflict", "SigningKeyLifecycle",
    "NativeKeyringSigningSecretAdministrator", "NativeKeyringSigningSecretReader",
    "provision_local_signing_authority", "rotate_local_freshness_signing_authority",
    "transition_local_signing_lifecycle",
]
