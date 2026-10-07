"""Durable, singleton initial operation owned by the installed ACTIVE LPPI.

The write lock only reserves and commits retained bytes. Native signing and
current ACTIVE requalification run outside it, under a separate nonblocking
signing lock. A signed result becomes authority only after its durable commit.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import secrets
import stat
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from bot_core.licensing.canonical import canonical_json_bytes, parse_canonical
from bot_core.licensing.lppi_authenticated_operation import (
    LPPIAuthenticatedProvisioningOperationBindingV1,
    VerifiedLPPIAuthenticatedProvisioningOperation,
    _active_operation_material,
    _issue_verified_operation,
    reservation_digest,
    verify_authenticated_operation_binding,
)
from bot_core.licensing.lppi_authority_key import LPPIAuthorityKeyBindingV1
from bot_core.licensing.pdsa_enrollment_authorization import (
    _mint_uuidv7,
    _reservation_epoch_milliseconds,
)
from deployment import windows_production_lppi_authority as lifecycle
from deployment.windows_cng_pre_enrollment import _safe_path
from deployment.windows_lppi_authority_key import (
    sign_active_lppi_authenticated_operation_binding,
)

STATUSES = ("PRVOP_RESERVED", "AUTHENTICATED_OPERATION_COMMITTED")
PURPOSE_DOMAIN = "CryptoHunter.Stage9.ProvisioningOperation.v1"
_SOURCE_FIELDS = frozenset(
    {
        "pdsa_trust_domain",
        "pdsa_package_digest_sha256",
        "provisioning_subject_id",
        "enrollment_reference",
        "lppi_authority_key_binding_digest_sha256",
        "lppi_authority_public_key_algorithm_profile",
        "lppi_authority_public_key_fingerprint_sha256",
        "custody_profile",
    }
)
_FIELDS = _SOURCE_FIELDS | frozenset(
    {
        "schema_version",
        "status",
        "history",
        "purpose_domain",
        "reservation_digest_sha256",
        "active_authority_binding_raw_hex",
        "provisioning_operation_id",
        "binding_generation",
        "created_at_utc",
        "binding_raw_hex",
        "binding_digest_sha256",
        "signature_hex",
    }
)
_SigningOwner = tuple[int, int, Path, object]
_SIGNING_OWNER: ContextVar[_SigningOwner | None] = ContextVar(
    "lppi_operation_signing_owner", default=None
)
_HELD_SIGNING_OWNER: _SigningOwner | None = None


class LPPIAuthenticatedOperationError(RuntimeError):
    """Installed operation identity, persistence or retained proof failed closed."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _state_path() -> Path:
    return cast(Path, lifecycle._state_path().with_name("initial-authenticated-operation.json"))


@contextmanager
def _lock(name: str) -> Iterator[Path]:
    path = _state_path()
    directory = path.parent
    _safe_path(directory, directory=True)
    directory.mkdir(parents=True, exist_ok=True)
    _safe_path(directory, directory=True)
    lock = directory / name
    _safe_path(lock)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "r+b") as stream:
        information = os.fstat(stream.fileno())
        if not stat.S_ISREG(information.st_mode) or information.st_nlink != 1:
            raise LPPIAuthenticatedOperationError("UNSAFE_LPPI_AUTHENTICATED_OPERATION_LOCK")
        if os.name == "nt":
            import msvcrt

            windows_lock: Any = msvcrt
            if information.st_size == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            try:
                windows_lock.locking(stream.fileno(), windows_lock.LK_NBLCK, 1)
            except OSError as exc:
                raise LPPIAuthenticatedOperationError("LPPI_AUTHENTICATED_OPERATION_BUSY") from exc
            try:
                yield path
            finally:
                stream.seek(0)
                windows_lock.locking(stream.fileno(), windows_lock.LK_UNLCK, 1)
        else:
            # Resolve the POSIX-only module dynamically. Static type checking runs on
            # Windows too, where typeshed intentionally exposes no usable flock API.
            posix_lock: Any = __import__("fcntl")
            try:
                posix_lock.flock(stream.fileno(), posix_lock.LOCK_EX | posix_lock.LOCK_NB)
            except OSError as exc:
                raise LPPIAuthenticatedOperationError("LPPI_AUTHENTICATED_OPERATION_BUSY") from exc
            try:
                yield path
            finally:
                posix_lock.flock(stream.fileno(), posix_lock.LOCK_UN)


@contextmanager
def _locked_state() -> Iterator[Path]:
    with _lock("initial-authenticated-operation.lock") as path:
        yield path


@contextmanager
def _locked_signing() -> Iterator[Path]:
    global _HELD_SIGNING_OWNER
    with _lock("initial-authenticated-operation-signing.lock") as path:
        owner = (os.getpid(), threading.get_ident(), path, object())
        token = _SIGNING_OWNER.set(owner)
        _HELD_SIGNING_OWNER = owner
        try:
            yield path
        finally:
            _HELD_SIGNING_OWNER = None
            _SIGNING_OWNER.reset(token)


def _require_signing_owner() -> None:
    owner = _SIGNING_OWNER.get()
    if (
        owner is None
        or owner is not _HELD_SIGNING_OWNER
        or owner[:3] != (os.getpid(), threading.get_ident(), _state_path())
    ):
        raise LPPIAuthenticatedOperationError("LPPI_AUTHENTICATED_OPERATION_SIGNING_LOCK_REQUIRED")


def _hex_bytes(value: object) -> bytes:
    if type(value) is not str:
        raise ValueError("state hex type")
    raw = bytes.fromhex(value)
    if not raw or raw.hex() != value:
        raise ValueError("state canonical hex")
    return raw


def _validate_state(state: object) -> dict[str, Any]:
    if (
        type(state) is not dict
        or set(state) != _FIELDS
        or state["schema_version"] != "LPPIInitialAuthenticatedOperationV1"
        or state["purpose_domain"] != PURPOSE_DOMAIN
    ):
        raise ValueError("state schema")
    status = state["status"]
    if status not in STATUSES or state["history"] != list(STATUSES[: STATUSES.index(status) + 1]):
        raise ValueError("state progression")
    if type(state["binding_generation"]) is not int or state["binding_generation"] != 1:
        raise ValueError("state initial generation")
    source = {field: state[field] for field in _SOURCE_FIELDS}
    if state["reservation_digest_sha256"] != reservation_digest(source):
        raise ValueError("state reservation digest")
    active_binding = LPPIAuthorityKeyBindingV1.from_canonical_bytes(
        _hex_bytes(state["active_authority_binding_raw_hex"])
    )
    active_document = active_binding.document
    for field in _SOURCE_FIELDS:
        expected = (
            active_binding.digest_sha256
            if field == "lppi_authority_key_binding_digest_sha256"
            else active_document[field]
        )
        if state[field] != expected:
            raise ValueError("state retained ACTIVE snapshot")
    binding = LPPIAuthenticatedProvisioningOperationBindingV1.from_canonical_bytes(
        _hex_bytes(state["binding_raw_hex"])
    )
    document = binding.document
    if state["binding_digest_sha256"] != binding.digest_sha256:
        raise ValueError("state binding digest")
    for field in (
        "pdsa_trust_domain",
        "pdsa_package_digest_sha256",
        "provisioning_subject_id",
        "enrollment_reference",
        "provisioning_operation_id",
        "binding_generation",
        "created_at_utc",
    ):
        if type(state[field]) is not type(document[field]) or state[field] != document[field]:
            raise ValueError("state operation tuple")
    if status == "PRVOP_RESERVED":
        if state["signature_hex"] is not None:
            raise ValueError("state premature signature")
    else:
        _hex_bytes(state["signature_hex"])
    return cast(dict[str, Any], state)


def _read(path: Path) -> dict[str, Any] | None:
    _safe_path(path)
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise LPPIAuthenticatedOperationError(
            "INVALID_RETAINED_LPPI_AUTHENTICATED_OPERATION"
        ) from exc
    try:
        with os.fdopen(fd, "rb") as stream:
            information = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(information.st_mode)
                or information.st_nlink != 1
                or information.st_size > 1_048_576
            ):
                raise ValueError("unsafe or oversized state")
            raw = stream.read(1_048_577)
        if len(raw) > 1_048_576:
            raise ValueError("oversized state")
        return _validate_state(parse_canonical(raw))
    except (ValueError, TypeError, KeyError, RecursionError, OSError) as exc:
        raise LPPIAuthenticatedOperationError(
            "INVALID_RETAINED_LPPI_AUTHENTICATED_OPERATION"
        ) from exc


def _replace_write_through(source: Path, destination: Path) -> None:
    if os.name == "nt":
        # FlushFileBuffers/fsync protects the file contents; MOVEFILE_WRITE_THROUGH
        # completes the filesystem rename before operation authority is returned.
        windows_abi: Any = ctypes
        kernel32 = windows_abi.WinDLL("kernel32", use_last_error=True)
        move = kernel32.MoveFileExW
        move.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
        move.restype = ctypes.c_int
        if not move(str(source), str(destination), 0x1 | 0x8):
            raise windows_abi.WinError(windows_abi.get_last_error())
    else:
        os.replace(source, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _write(path: Path, state: dict[str, Any]) -> None:
    _safe_path(path)
    try:
        _validate_state(state)
    except (ValueError, TypeError, KeyError, RecursionError) as exc:
        raise LPPIAuthenticatedOperationError(
            "INVALID_RETAINED_LPPI_AUTHENTICATED_OPERATION"
        ) from exc
    previous = _read(path)
    if previous is not None:
        mutable = {"status", "history", "signature_hex"}
        if any(previous[field] != state[field] for field in _FIELDS - mutable) or (
            previous["status"] == "AUTHENTICATED_OPERATION_COMMITTED" and previous != state
        ):
            raise LPPIAuthenticatedOperationError("LPPI_AUTHENTICATED_OPERATION_CONFLICT")
    temporary = path.with_name(path.name + "." + secrets.token_hex(16) + ".tmp")
    _safe_path(temporary)
    fd = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical_json_bytes(state))
            stream.flush()
            os.fsync(stream.fileno())
        _safe_path(path)
        _replace_write_through(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class _OperationSource:
    tuple_raw: bytes
    active_binding_raw: bytes

    @property
    def document(self) -> dict[str, Any]:
        return cast(dict[str, Any], parse_canonical(self.tuple_raw))


def _source(active_authority: object) -> _OperationSource:
    source, binding_raw, _ = _active_operation_material(active_authority)
    if set(source) != _SOURCE_FIELDS:
        raise LPPIAuthenticatedOperationError("LPPI_AUTHENTICATED_OPERATION_CONFLICT")
    return _OperationSource(canonical_json_bytes(source), binding_raw)


def _targets(state: dict[str, Any], source: _OperationSource) -> None:
    if any(state[field] != value for field, value in source.document.items()) or (
        state["active_authority_binding_raw_hex"] != source.active_binding_raw.hex()
    ):
        raise LPPIAuthenticatedOperationError("LPPI_AUTHENTICATED_OPERATION_CONFLICT")


def _reserve(path: Path, source: _OperationSource) -> dict[str, Any]:
    state = _read(path)
    if state is None:
        # One captured instant feeds both UUIDv7's full milliseconds and the
        # canonical timestamp. The reusable PDSA mint obtains CSPRNG random bits.
        captured = _utc_now()
        milliseconds = _reservation_epoch_milliseconds(captured)
        operation_id = _mint_uuidv7("prvop_", milliseconds)
        created = captured.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        document = source.document
        binding = LPPIAuthenticatedProvisioningOperationBindingV1.from_mapping(
            {
                "schema_version": 1,
                "environment": "PRODUCTION",
                **{
                    field: document[field]
                    for field in (
                        "pdsa_trust_domain",
                        "pdsa_package_digest_sha256",
                        "provisioning_subject_id",
                        "enrollment_reference",
                    )
                },
                "provisioning_operation_id": operation_id,
                "binding_generation": 1,
                "created_at_utc": created,
            }
        )
        state = {
            "schema_version": "LPPIInitialAuthenticatedOperationV1",
            "status": "PRVOP_RESERVED",
            "history": ["PRVOP_RESERVED"],
            "purpose_domain": PURPOSE_DOMAIN,
            "reservation_digest_sha256": reservation_digest(document),
            **document,
            "active_authority_binding_raw_hex": source.active_binding_raw.hex(),
            "provisioning_operation_id": operation_id,
            "binding_generation": 1,
            "created_at_utc": created,
            "binding_raw_hex": binding.canonical_bytes.hex(),
            "binding_digest_sha256": binding.digest_sha256,
            "signature_hex": None,
        }
        _write(path, state)
    _targets(state, source)
    return state


def require_reserved_operation_binding(binding_raw: bytes, active_authority: object) -> None:
    """Native effect gate: exact durably reserved bytes for the current ACTIVE."""
    _require_signing_owner()
    source = _source(active_authority)
    state = _read(_state_path())
    if state is None:
        raise LPPIAuthenticatedOperationError("DURABLE_LPPI_OPERATION_RESERVATION_REQUIRED")
    _targets(state, source)
    if state["status"] != "PRVOP_RESERVED":
        raise LPPIAuthenticatedOperationError("LPPI_AUTHENTICATED_OPERATION_ALREADY_COMMITTED")
    if type(binding_raw) is not bytes or state["binding_raw_hex"] != binding_raw.hex():
        raise LPPIAuthenticatedOperationError("LPPI_AUTHENTICATED_OPERATION_CONFLICT")


def validate_retained_operation(
    active_authority: object, path: Path, state_raw: bytes
) -> tuple[LPPIAuthenticatedProvisioningOperationBindingV1, bytes]:
    """Reverify current ACTIVE, exact retained bytes, tuple and strict signature."""
    source = _source(active_authority)
    if path != _state_path() or type(state_raw) is not bytes:
        raise LPPIAuthenticatedOperationError("LPPI_AUTHENTICATED_OPERATION_CONFLICT")
    state = _read(path)
    if state is None or canonical_json_bytes(state) != state_raw:
        raise LPPIAuthenticatedOperationError("LPPI_AUTHENTICATED_OPERATION_CONFLICT")
    _targets(state, source)
    if state["status"] != "AUTHENTICATED_OPERATION_COMMITTED":
        raise LPPIAuthenticatedOperationError("LPPI_AUTHENTICATED_OPERATION_NOT_COMMITTED")
    signature = _hex_bytes(state["signature_hex"])
    binding = verify_authenticated_operation_binding(
        _hex_bytes(state["binding_raw_hex"]), signature, active_authority
    )
    return binding, signature


def establish_installed_lppi_authenticated_operation(
    active_authority: object,
) -> VerifiedLPPIAuthenticatedProvisioningOperation:
    """Reserve or resume the single initial operation, retaining its first signature."""
    source = _source(active_authority)
    with _locked_signing():
        with _locked_state() as path:
            state = _reserve(path, source)
        if state["status"] == "AUTHENTICATED_OPERATION_COMMITTED":
            return _issue_verified_operation(active_authority, path, canonical_json_bytes(state))
        reserved_raw = canonical_json_bytes(state)
        binding_raw = _hex_bytes(state["binding_raw_hex"])
        signature = sign_active_lppi_authenticated_operation_binding(active_authority, binding_raw)
        verify_authenticated_operation_binding(binding_raw, signature, active_authority)
        # Repeat live source requalification before entering the brief finalize
        # lock; never retain a signature from a changed or no-longer-ACTIVE key.
        current_source = _source(active_authority)
        with _locked_state() as path:
            current = _read(path)
            if current is None:
                raise LPPIAuthenticatedOperationError("LPPI_AUTHENTICATED_OPERATION_CONFLICT")
            _targets(current, current_source)
            if current["status"] == "AUTHENTICATED_OPERATION_COMMITTED":
                # A retained first winner is immutable even if this invocation
                # obtained a different valid native ECDSA result before commit.
                state = current
            else:
                if canonical_json_bytes(current) != reserved_raw:
                    raise LPPIAuthenticatedOperationError("LPPI_AUTHENTICATED_OPERATION_CONFLICT")
                current["signature_hex"] = signature.hex()
                current["status"] = "AUTHENTICATED_OPERATION_COMMITTED"
                current["history"] = list(STATUSES)
                _write(path, current)
                state = current
        return _issue_verified_operation(active_authority, path, canonical_json_bytes(state))


def load_installed_lppi_authenticated_operation(
    active_authority: object,
) -> VerifiedLPPIAuthenticatedProvisioningOperation:
    """Restore committed authority after the existing loader reverified ACTIVE.

    This loader never mints an operation or trusts retained ACTIVE JSON alone.
    The caller must first obtain a current verifier-issued ACTIVE capability
    through the production package acceptance and successor lifecycle.
    """
    _source(active_authority)
    state = _read(_state_path())
    if state is None:
        raise LPPIAuthenticatedOperationError("LPPI_AUTHENTICATED_OPERATION_NOT_COMMITTED")
    return _issue_verified_operation(active_authority, _state_path(), canonical_json_bytes(state))
