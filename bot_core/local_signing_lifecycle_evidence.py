"""Durable custody evidence for closed Root Proof/history lifecycle operations.

The PostgreSQL operation must already have durably closed its exact fence before
this module may change custody. An intent binds its complete operation descriptor
and exact before/after records before the existing CAS runs. Recovery accepts
only either exact record; later legal descendants are not completion evidence.

Receipts are process-local membership capabilities derived from reread durable
COMPLETE records, never caller-constructed DTOs or hashes. The verification
context retains the custody lock through PostgreSQL publication. Existing native
storage, process locking and fsync helpers remain the custody boundary; there is
no alternate private-key storage here.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from threading import Lock
from typing import Callable, Iterator, cast
from weakref import WeakKeyDictionary

from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.local_signing_custody import (
    LocalSigningCustodyError,
    LocalSigningLifecycleError,
    LocalSigningMetadata,
    SigningKeyLifecycle,
    _ADMISSION_LOCK,
    _LOCK_FILENAME,
    _MUTATION_ADMISSIONS,
    _CustodyLockLease,
    _LocalSigningMutationAdmission,
    _MutationAdmissionBinding,
    _custody_lock,
    _decode_record,
    _encode_record,
    _fsync_directory,
    _open_custody_descriptor,
    _record_path,
    _replace_record,
    _resolve_credential_record,
    _retained_path,
    _snapshot_security,
    _transition_exact_locked,
    _validate_directory,
    _write_new_record,
)
from bot_core.root_proof_issuer_substrate import (
    CredentialRoleIdentity,
    ProviderRole,
    SecurityProfile,
    SecurityProfileIdentity,
)

_DIRECTORY = ".signing-lifecycle-operations"
_SCHEMA = "cryptohunter.production-local-signing-lifecycle-operation"
_VERSION = 2
_ROLES = frozenset({ProviderRole.ROOT_PROOF_SIGNING, ProviderRole.HISTORY_ATTESTATION_SIGNING})
_ALLOWED = frozenset(
    {
        (SigningKeyLifecycle.ACTIVE, SigningKeyLifecycle.VERIFY_ONLY),
        (SigningKeyLifecycle.ACTIVE, SigningKeyLifecycle.REVOKED),
        (SigningKeyLifecycle.VERIFY_ONLY, SigningKeyLifecycle.REVOKED),
    }
)


def _canonical(value: object) -> bytes:
    return cast(bytes, canonical_json_bytes(value))


def _descriptor_snapshot(value: bytes) -> bytes:
    if type(value) is not bytes or not value or len(value) > 262144:
        raise LocalSigningLifecycleError("exact canonical lifecycle operation descriptor required")
    try:
        decoded = json.loads(value)
        if type(decoded) is not dict or not decoded or _canonical(decoded) != value:
            raise ValueError
    except (TypeError, ValueError, UnicodeError) as exc:
        raise LocalSigningLifecycleError(
            "exact canonical lifecycle operation descriptor required"
        ) from exc
    return value


def _operation_token(operation_id: str) -> str:
    if (
        type(operation_id) is not str
        or len(operation_id) > 128
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", operation_id) is None
    ):
        raise LocalSigningLifecycleError("invalid lifecycle operation identity")
    return hashlib.sha256(operation_id.encode("utf-8")).hexdigest()


def _require_close_authorization(authorization: object, descriptor: bytes) -> None:
    # Deferred to avoid a dependency cycle with the PostgreSQL coordinator.
    from bot_core.postgresql_custody_lifecycle import validate_durable_close_authorization

    validate_durable_close_authorization(authorization, descriptor)


def _request_binding(
    descriptor: bytes,
    root: Path,
    security: SecurityProfileIdentity,
    role: ProviderRole,
    identity: CredentialRoleIdentity,
    generation: int,
    target: SigningKeyLifecycle,
    operation_id: str,
) -> dict[str, object]:
    from bot_core.postgresql_custody_lifecycle import (
        CustodyLifecycleConflict,
        CustodyLifecycleRequest,
    )

    try:
        request = CustodyLifecycleRequest(root, security, role, descriptor)
    except (CustodyLifecycleConflict, TypeError, ValueError) as exc:
        raise LocalSigningLifecycleError("exact authorized custody request required") from exc
    operation = cast(dict[str, object], request.operation)
    if (
        request.credential_identity != identity
        or operation["lifecycle_generation"] != generation
        or operation["target_state"] != target.value
        or operation["operation_id"] != operation_id
    ):
        raise LocalSigningLifecycleError("custody mutation differs from exact authorized close")
    return operation


def _validate_before_binding(before: LocalSigningMetadata, operation: dict[str, object]) -> None:
    if (
        before.public_key.hex() != operation["public_key_hex"]
        or before.key_version != operation["key_version"]
        or before.lifecycle_state.value != operation["predecessor_state"]
    ):
        raise LocalSigningLifecycleError("exact custody predecessor differs from authorized close")


def _safe_directory(directory: Path, *, create: bool) -> Path:
    path = cast(Path, _validate_directory(directory, create=create))
    if os.name == "posix" and path.resolve(strict=True) != path:
        raise LocalSigningCustodyError("custody directory ancestry is unsafe")
    return path


def _directory_identity(path: Path) -> tuple[int, int]:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode) or path.is_symlink():
        raise LocalSigningCustodyError("custody directory identity is unsafe")
    return info.st_dev, info.st_ino


def _check_directory(path: Path, expected: tuple[int, int]) -> None:
    _safe_directory(path, create=False)
    if _directory_identity(path) != expected:
        raise LocalSigningCustodyError("custody directory identity changed")


def _unsafe_file_owner(info: os.stat_result) -> bool:
    if sys.platform == "win32":
        return False
    return os.name == "posix" and (
        info.st_uid != os.geteuid() or bool(stat.S_IMODE(info.st_mode) & 0o077)
    )


def _read_bytes(path: Path) -> bytes:
    try:
        with _open_custody_descriptor(path) as descriptor:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or _unsafe_file_owner(info):
                raise LocalSigningCustodyError(
                    "lifecycle journal ownership or permissions are unsafe"
                )
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                raw = stream.read(1048577)
            observed = path.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(observed.st_mode)
                or (observed.st_dev, observed.st_ino) != (info.st_dev, info.st_ino)
                or len(raw) > 1048576
            ):
                raise LocalSigningCustodyError("lifecycle journal path identity changed")
            return raw
    except OSError as exc:
        raise LocalSigningCustodyError("lifecycle journal unavailable/corrupt") from exc


def _sync_path(path: Path) -> None:
    """Re-establish durability after a prior writer's interrupted fsync/ack."""

    try:
        # Windows CRT _commit() rejects an O_RDONLY descriptor (EBADF).
        # Request writable access to the same pinned, NTFS-qualified file only
        # for durability; do not modify its bytes or relax any path/ACL checks.
        with _open_custody_descriptor(path, write=os.name == "nt") as descriptor:
            os.fsync(descriptor)
        _fsync_directory(path.parent)
    except OSError as exc:
        raise LocalSigningCustodyError("lifecycle evidence durability failed") from exc


@dataclass(frozen=True, slots=True)
class _OperationRecord:
    operation_id: str
    descriptor: bytes
    before: LocalSigningMetadata
    after: LocalSigningMetadata
    state: str


def _document(record: _OperationRecord) -> bytes:
    return _canonical(
        {
            "schema": _SCHEMA,
            "schema_version": _VERSION,
            "operation_id": record.operation_id,
            "operation_descriptor_b64": base64.b64encode(record.descriptor).decode("ascii"),
            "before_metadata": json.loads(_encode_record(record.before)),
            "after_metadata": json.loads(_encode_record(record.after)),
            "completion_state": record.state,
        }
    )


def _read_operation(
    path: Path, security: SecurityProfileIdentity, role: ProviderRole
) -> tuple[_OperationRecord, bytes]:
    raw = _read_bytes(path)
    return _decode_operation(raw, security, role), raw


def _decode_operation(
    raw: bytes, security: SecurityProfileIdentity, role: ProviderRole
) -> _OperationRecord:
    """Decode the existing V2 journal without reading, syncing or changing custody."""
    try:
        value = json.loads(raw)
        if (
            type(value) is not dict
            or set(value)
            != {
                "schema",
                "schema_version",
                "operation_id",
                "operation_descriptor_b64",
                "before_metadata",
                "after_metadata",
                "completion_state",
            }
            or value["schema"] != _SCHEMA
            or type(value["schema_version"]) is not int
            or value["schema_version"] != _VERSION
            or value["completion_state"] not in {"INTENT", "COMPLETE"}
            or _canonical(value) != raw
        ):
            raise ValueError
        _operation_token(value["operation_id"])
        encoded = value["operation_descriptor_b64"]
        descriptor = _descriptor_snapshot(base64.b64decode(encoded, validate=True))
        if base64.b64encode(descriptor).decode("ascii") != encoded:
            raise ValueError
        before = _decode_record(
            _canonical(value["before_metadata"]), expected_security=security, expected_role=role
        )
        after = _decode_record(
            _canonical(value["after_metadata"]), expected_security=security, expected_role=role
        )
        if (before.lifecycle_state, after.lifecycle_state) not in _ALLOWED or after != replace(
            before,
            lifecycle_generation=before.lifecycle_generation + 1,
            lifecycle_state=after.lifecycle_state,
        ):
            raise ValueError
        record = _OperationRecord(
            value["operation_id"], descriptor, before, after, value["completion_state"]
        )
        if _document(record) != raw:
            raise ValueError
    except (KeyError, TypeError, ValueError, UnicodeError) as exc:
        raise LocalSigningCustodyError("lifecycle journal unavailable/corrupt") from exc
    return record


@dataclass(frozen=True, slots=True)
class _CompletionBinding:
    directory: str
    directory_identity: tuple[int, int]
    journal_directory_identity: tuple[int, int]
    security: SecurityProfileIdentity
    role: ProviderRole
    evidence: bytes


class DurableSigningLifecycleCompletion:
    """Receipt with genuine process membership and reread durable evidence."""

    __slots__ = ("__weakref__",)

    def __new__(cls) -> DurableSigningLifecycleCompletion:
        raise TypeError("durable lifecycle completion is issued only by verified custody")

    @property
    def evidence_bytes(self) -> bytes:
        return _binding(self).evidence

    @property
    def operation_id(self) -> str:
        value = json.loads(self.evidence_bytes)["operation_id"]
        return str(value)

    @property
    def operation_descriptor(self) -> bytes:
        return base64.b64decode(
            json.loads(self.evidence_bytes)["operation_descriptor_b64"], validate=True
        )

    @property
    def completion_fingerprint_sha256(self) -> str:
        return hashlib.sha256(self.evidence_bytes).hexdigest()

    @property
    def metadata(self) -> LocalSigningMetadata:
        binding = _binding(self)
        return _decode_record(
            _canonical(json.loads(binding.evidence)["after_metadata"]),
            expected_security=binding.security,
            expected_role=binding.role,
        )


_COMPLETIONS: WeakKeyDictionary[DurableSigningLifecycleCompletion, _CompletionBinding] = (
    WeakKeyDictionary()
)
_MEMBERSHIP_LOCK = Lock()


def _binding(completion: object) -> _CompletionBinding:
    if type(completion) is not DurableSigningLifecycleCompletion:
        raise LocalSigningLifecycleError("genuine durable lifecycle completion required")
    with _MEMBERSHIP_LOCK:
        binding = _COMPLETIONS.get(completion)
    if binding is None:
        raise LocalSigningLifecycleError("genuine durable lifecycle completion required")
    return binding


def _verified_complete_locked(
    root: Path,
    security: SecurityProfileIdentity,
    role: ProviderRole,
    journal: Path,
    operation_id: str,
    descriptor: bytes,
) -> tuple[LocalSigningMetadata, bytes]:
    record, raw = _read_operation(journal, security, role)
    if (
        record.operation_id != operation_id
        or record.descriptor != descriptor
        or record.state != "COMPLETE"
    ):
        raise LocalSigningLifecycleError("exact durable lifecycle completion required")
    path, current = _resolve_credential_record(
        root, security, role, record.after.credential_identity()
    )
    if current != record.after:
        raise LocalSigningLifecycleError(
            "custody conflicts with exact durable lifecycle completion"
        )
    _sync_path(path)
    _sync_path(journal)
    # Never issue or use a capability for a pathname substituted during sync.
    if _read_operation(journal, security, role)[1] != raw:
        raise LocalSigningCustodyError("durable lifecycle completion changed")
    if (
        _resolve_credential_record(root, security, role, record.after.credential_identity())[1]
        != current
    ):
        raise LocalSigningCustodyError("custody lifecycle changed during verification")
    return current, raw


def durable_transition_local_signing_credential_lifecycle(
    directory: Path,
    *,
    security: SecurityProfileIdentity,
    role: ProviderRole,
    credential_identity: CredentialRoleIdentity,
    expected_lifecycle_generation: int,
    target: SigningKeyLifecycle,
    lifecycle_operation_id: str,
    operation_descriptor: bytes,
    close_authorization: object,
    fault_injector: Callable[[str], None] | None = None,
) -> DurableSigningLifecycleCompletion:
    """Recover one exact downgrade only after genuine durable fence closure."""

    root = _safe_directory(directory, create=False)
    security = _snapshot_security(security)
    descriptor = _descriptor_snapshot(operation_descriptor)
    token = _operation_token(lifecycle_operation_id)
    if (
        type(role) is not ProviderRole
        or role not in _ROLES
        or security.profile is not SecurityProfile.PRODUCTION_LOCAL
        or type(credential_identity) is not CredentialRoleIdentity
        or type(expected_lifecycle_generation) is not int
        or expected_lifecycle_generation < 1
        or type(target) is not SigningKeyLifecycle
        or target is SigningKeyLifecycle.ACTIVE
    ):
        raise LocalSigningLifecycleError("invalid exact root/history lifecycle request")
    _require_close_authorization(close_authorization, descriptor)
    operation = _request_binding(
        descriptor,
        root,
        security,
        role,
        credential_identity,
        expected_lifecycle_generation,
        target,
        lifecycle_operation_id,
    )
    root_identity = _directory_identity(root)
    with _custody_lock(root, exclusive=True) as lease:
        _check_directory(root, root_identity)
        journal_directory = _safe_directory(root / _DIRECTORY, create=True)
        journal_identity = _directory_identity(journal_directory)
        _fsync_directory(root)
        journal = journal_directory / f"{token}.json"
        if journal.exists() or journal.is_symlink():
            record, _ = _read_operation(journal, security, role)
            if (
                record.operation_id != lifecycle_operation_id
                or record.descriptor != descriptor
                or record.before.credential_identity() != credential_identity
                or record.before.lifecycle_generation != expected_lifecycle_generation
                or record.after.lifecycle_state is not target
            ):
                raise LocalSigningLifecycleError(
                    "lifecycle operation identity is bound to another request"
                )
        else:
            _, before = _resolve_credential_record(root, security, role, credential_identity)
            if before.lifecycle_generation != expected_lifecycle_generation:
                raise LocalSigningLifecycleError("stale signing-key lifecycle generation")
            if (before.lifecycle_state, target) not in _ALLOWED:
                raise LocalSigningLifecycleError("illegal signing-key lifecycle transition")
            _validate_before_binding(before, operation)
            record = _OperationRecord(
                lifecycle_operation_id,
                descriptor,
                before,
                replace(
                    before,
                    lifecycle_generation=before.lifecycle_generation + 1,
                    lifecycle_state=target,
                ),
                "INTENT",
            )
            if not _write_new_record(journal, _document(record)):
                raise LocalSigningCustodyError("lifecycle intent publication conflict")
            if fault_injector is not None:
                fault_injector("lifecycle_after_intent")
        _validate_before_binding(record.before, operation)
        if record.state == "INTENT":
            # Re-fsync an existing intent before recovering any state mutation.
            _sync_path(journal)
            _check_directory(root, root_identity)
            _check_directory(journal_directory, journal_identity)
            path, current = _resolve_credential_record(root, security, role, credential_identity)
            if current == record.before:
                _require_close_authorization(close_authorization, descriptor)

                def validate_admission() -> None:
                    _check_directory(root, root_identity)
                    _check_directory(journal_directory, journal_identity)
                    observed, _ = _read_operation(journal, security, role)
                    if observed != record or observed.state != "INTENT":
                        raise LocalSigningLifecycleError("exact durable lifecycle INTENT required")
                    _validate_before_binding(observed.before, operation)
                    _sync_path(journal)
                    _require_close_authorization(close_authorization, descriptor)

                # There is no custody API for minting admissions from a caller
                # DTO/validator. Only this authenticated, journal-verified owner
                # inserts the token. Arbitrary registry mutation is a change to
                # Python trust state and outside the OS isolation guarantee.
                admission = object.__new__(_LocalSigningMutationAdmission)
                with _ADMISSION_LOCK:
                    _MUTATION_ADMISSIONS[admission] = _MutationAdmissionBinding(
                        root,
                        role,
                        path,
                        record.before,
                        record.after,
                        close_authorization,
                        descriptor,
                        cast(_CustodyLockLease, lease),
                        validate_admission,
                    )
                try:
                    current = _transition_exact_locked(
                        root,
                        security,
                        role,
                        credential_identity,
                        expected_lifecycle_generation,
                        target,
                        close_authorization=close_authorization,
                        operation_descriptor=descriptor,
                        mutation_admission=admission,
                    )
                finally:
                    with _ADMISSION_LOCK:
                        _MUTATION_ADMISSIONS.pop(admission, None)
                if fault_injector is not None:
                    fault_injector("lifecycle_after_custody_transition")
            if current != record.after:
                raise LocalSigningLifecycleError("custody conflicts with durable lifecycle intent")
            _sync_path(path)
            _replace_record(journal, _document(replace(record, state="COMPLETE")))
            if fault_injector is not None:
                fault_injector("lifecycle_after_completion")
        _check_directory(root, root_identity)
        _check_directory(journal_directory, journal_identity)
        _, evidence = _verified_complete_locked(
            root, security, role, journal, lifecycle_operation_id, descriptor
        )
        receipt = object.__new__(DurableSigningLifecycleCompletion)
        with _MEMBERSHIP_LOCK:
            _COMPLETIONS[receipt] = _CompletionBinding(
                str(root), root_identity, journal_identity, security, role, evidence
            )
        return receipt


@contextmanager
def verified_durable_signing_lifecycle_completion(
    completion: object, *, operation_descriptor: bytes
) -> Iterator[LocalSigningMetadata]:
    """Hold exact verified custody completion locked through DB publication."""

    binding = _binding(completion)
    descriptor = _descriptor_snapshot(operation_descriptor)
    root = _safe_directory(Path(binding.directory), create=False)
    _check_directory(root, binding.directory_identity)
    operation = json.loads(binding.evidence)
    with _custody_lock(root, exclusive=True):
        _check_directory(root, binding.directory_identity)
        journal_directory = root / _DIRECTORY
        _check_directory(journal_directory, binding.journal_directory_identity)
        metadata, raw = _verified_complete_locked(
            root,
            binding.security,
            binding.role,
            journal_directory / f"{_operation_token(operation['operation_id'])}.json",
            operation["operation_id"],
            descriptor,
        )
        if raw != binding.evidence:
            raise LocalSigningLifecycleError("durable lifecycle capability evidence changed")
        yield metadata


@dataclass(frozen=True, slots=True)
class SigningLifecycleCompletionObservation:
    """Public journal facts, never a close authorization or completion capability.

    This observation supports the existing V2 journal only. It does not adapt
    the isolated PostgreSQL coordinator to production or attest who wrote the
    journal, private-key control, an independent authority, or prior fsync.
    """

    evidence_bytes: bytes
    metadata: LocalSigningMetadata

    @property
    def completion_fingerprint_sha256(self) -> str:
        return hashlib.sha256(self.evidence_bytes).hexdigest()


def _read_only_file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


class _ReadOnlyCustodyFiles:
    """Pin every path component and existing file without creating any object."""

    def __init__(self, root: Path) -> None:
        if len(root.parts) > 128:
            raise LocalSigningCustodyError("read-only custody ancestry exceeds limit")
        self.root = root
        self.directories: dict[Path, tuple[int, tuple[int, ...]]] = {}
        self.files: dict[Path, tuple[int, tuple[int, ...], bytes]] = {}
        self.descriptors: list[int] = []

    @staticmethod
    def _directory_identity(info: os.stat_result) -> tuple[int, ...]:
        return info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid

    @staticmethod
    def _no_acl(descriptor: int) -> None:
        if sys.platform != "linux":
            raise LocalSigningCustodyError("read-only custody ACL observation requires Linux")
        if any(name.startswith("system.posix_acl_") for name in os.listxattr(descriptor)):
            raise LocalSigningCustodyError("read-only custody ACL is unsafe")

    def directory(self, path: Path) -> int:
        if sys.platform != "linux":
            raise LocalSigningCustodyError("read-only custody directory observation requires Linux")
        if path in self.directories:
            return self.directories[path][0]
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        if path == Path(path.anchor):
            descriptor = os.open(path.anchor, flags)
        else:
            descriptor = os.open(path.name, flags, dir_fd=self.directory(path.parent))
        self.descriptors.append(descriptor)
        info = os.fstat(descriptor)
        if path == self.root or self.root in path.parents:
            if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
                raise LocalSigningCustodyError("read-only custody directory is unsafe")
            self._no_acl(descriptor)
        self.directories[path] = descriptor, self._directory_identity(info)
        return descriptor

    def read(self, path: Path, *, limit: int = 1048576) -> bytes:
        if sys.platform != "linux":
            raise LocalSigningCustodyError("read-only custody file observation requires Linux")
        if path in self.files:
            descriptor, expected, original = self.files[path]
        else:
            descriptor = os.open(
                path.name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=self.directory(path.parent),
            )
            self.descriptors.append(descriptor)
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_nlink != 1
                or info.st_size > limit
            ):
                raise LocalSigningCustodyError("read-only custody file is unsafe")
            self._no_acl(descriptor)
            expected, original = _read_only_file_identity(info), b""
        os.lseek(descriptor, 0, os.SEEK_SET)
        raw = bytearray()
        while len(raw) <= limit:
            part = os.read(descriptor, min(8192, limit + 1 - len(raw)))
            if not part:
                break
            raw.extend(part)
        if (
            len(raw) > limit
            or len(raw) != expected[6]
            or _read_only_file_identity(os.fstat(descriptor)) != expected
            or (path in self.files and bytes(raw) != original)
        ):
            raise LocalSigningCustodyError("read-only custody file changed")
        self.files[path] = descriptor, expected, bytes(raw)
        return bytes(raw)

    def recheck(self) -> None:
        for path, (descriptor, expected) in self.directories.items():
            observed = (
                os.stat(path.anchor, follow_symlinks=False)
                if path == Path(path.anchor)
                else os.stat(
                    path.name, dir_fd=self.directories[path.parent][0], follow_symlinks=False
                )
            )
            if (
                self._directory_identity(observed) != expected
                or self._directory_identity(os.fstat(descriptor)) != expected
            ):
                raise LocalSigningCustodyError("read-only custody directory changed")
            if path == self.root or self.root in path.parents:
                self._no_acl(descriptor)
        for path, (descriptor, expected, _) in self.files.items():
            observed = os.stat(
                path.name, dir_fd=self.directories[path.parent][0], follow_symlinks=False
            )
            if _read_only_file_identity(observed) != expected:
                raise LocalSigningCustodyError("read-only custody path changed")
            self._no_acl(descriptor)
            self.read(path)

    def close(self) -> None:
        for descriptor in reversed(self.descriptors):
            os.close(descriptor)


@contextmanager
def observe_durable_signing_lifecycle_completion(
    directory: Path,
    *,
    security: SecurityProfileIdentity,
    role: ProviderRole,
    operation_descriptor: bytes,
) -> Iterator[SigningLifecycleCompletionObservation]:
    """Read exact existing V2 COMPLETE evidence under an existing Linux lock.

    No close, recovery, fsync, secret read, capability issuance, or file creation
    occurs. The shared custody lock remains held through the caller's context;
    all pinned paths and bytes are checked again before releasing it. Missing
    evidence and INTENT stay unavailable. Independent production authentication
    and SQL publication are deliberately outside this diagnostic port.
    """
    if sys.platform != "linux":
        raise LocalSigningCustodyError("read-only custody observation requires Linux")
    from bot_core.postgresql_custody_lifecycle import (
        CustodyLifecycleConflict,
        CustodyLifecycleRequest,
    )

    import fcntl

    root = _safe_directory(directory, create=False)
    security = _snapshot_security(security)
    descriptor = _descriptor_snapshot(operation_descriptor)
    if (
        type(role) is not ProviderRole
        or role not in _ROLES
        or security.profile is not SecurityProfile.PRODUCTION_LOCAL
    ):
        raise LocalSigningLifecycleError("exact root/history custody observation required")
    try:
        request = CustodyLifecycleRequest(root, security, role, descriptor)
    except (CustodyLifecycleConflict, TypeError, ValueError) as exc:
        raise LocalSigningLifecycleError("exact custody observation descriptor required") from exc
    files = _ReadOnlyCustodyFiles(root)
    locked: int | None = None
    try:
        lock_path = root / _LOCK_FILENAME
        files.read(lock_path, limit=1)
        locked = files.files[lock_path][0]
        fcntl.flock(locked, fcntl.LOCK_SH | fcntl.LOCK_NB)
        files.recheck()
        journal = root / _DIRECTORY / f"{_operation_token(request.operation_id)}.json"
        evidence = files.read(journal)
        record = _decode_operation(evidence, security, role)
        if (
            record.state != "COMPLETE"
            or record.operation_id != request.operation_id
            or record.descriptor != descriptor
            or record.before.credential_identity() != request.credential_identity
            or record.before.lifecycle_generation != request.operation["lifecycle_generation"]
            or record.after.lifecycle_state.value != request.operation["target_state"]
        ):
            raise LocalSigningLifecycleError("exact existing COMPLETE evidence required")
        _validate_before_binding(record.before, request.operation)
        current_raw = files.read(_record_path(root, role), limit=65536)
        current = _decode_record(current_raw, expected_security=security, expected_role=role)
        if current.credential_identity() != request.credential_identity:
            current_raw = files.read(
                _retained_path(root, role, request.credential_identity), limit=65536
            )
            current = _decode_record(current_raw, expected_security=security, expected_role=role)
        if _encode_record(current) != current_raw or current != record.after:
            raise LocalSigningLifecycleError("custody differs from exact COMPLETE evidence")
        files.recheck()
        try:
            yield SigningLifecycleCompletionObservation(evidence, current)
        finally:
            files.recheck()
    except OSError as exc:
        raise LocalSigningCustodyError("read-only custody evidence unavailable") from exc
    finally:
        try:
            if locked is not None:
                fcntl.flock(locked, fcntl.LOCK_UN)
        finally:
            files.close()


__all__ = [
    "DurableSigningLifecycleCompletion",
    "SigningLifecycleCompletionObservation",
    "durable_transition_local_signing_credential_lifecycle",
    "observe_durable_signing_lifecycle_completion",
    "verified_durable_signing_lifecycle_completion",
]
