"""Installed CHA owner of the immutable initial authenticated prvop-to-ago mapping.

The short nonblocking filesystem lock retains a reservation before committing
its mapping. No authority is published until the commit has been physically
fenced and the upstream LPPI capability has been requalified. Disk rollback
protection is not provided by this layer.
"""

from __future__ import annotations

import os
import re
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from bot_core.licensing.canonical import canonical_json_bytes, parse_canonical
from bot_core.licensing.cha_logical_operation import (
    VerifiedCHALogicalOperation,
    _issue_verified_cha_operation,
)
from bot_core.licensing.lppi_authenticated_operation import (
    _operation_snapshot,
    require_verified_lppi_authenticated_operation,
)
from bot_core.persistence.physical_durability import (
    atomic_write_bytes_durably,
    flush_created_directory_metadata,
)
from bot_core.uuid7 import UUID7Error, mint_uuid7, reservation_epoch_milliseconds
from deployment import (
    windows_production_lppi_authority as lifecycle,
    windows_production_lppi_operation as lppi_operation,
)
from deployment.windows_cng_pre_enrollment import WindowsCNGPreEnrollmentError, _safe_path

STATUSES = ("AGO_RESERVED", "PRVOP_AGO_BIJECTION_COMMITTED")
_MAX_BYTES = 4_194_304
_MAX_UPSTREAM_BYTES = 1_048_576
_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "history",
        "mapping_generation",
        "pdsa_trust_domain",
        "provisioning_operation_id",
        "logical_operation_id",
        "assigned_at_utc",
        "lppi_operation_state_raw_hex",
    }
)
_UUID7 = r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
_UTC = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z\Z")


class CHAOperationError(RuntimeError):
    """The installed CHA source, immutable identity or durable proof failed closed."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _state_path() -> Path:
    return cast(Path, lifecycle._state_path().with_name("initial-cha-logical-operation.json"))


def _safe(path: Path, *, directory: bool = False) -> None:
    try:
        _safe_path(path, directory=directory)
    except (OSError, WindowsCNGPreEnrollmentError) as exc:
        raise CHAOperationError("UNSAFE_CHA_LOGICAL_OPERATION_PATH") from exc


def _create_directory(directory: Path) -> None:
    """Fence parent metadata, including directories left by a failed earlier fence."""
    _safe(directory, directory=True)
    for component in (*reversed(directory.parents), directory):
        _safe(component, directory=True)
        try:
            component.lstat()
        except FileNotFoundError:
            try:
                component.mkdir(mode=0o700)
            except FileExistsError:
                # A concurrent creator must still have installed a safe directory.
                pass
        _safe(component, directory=True)
        if component.parent != component:
            flush_created_directory_metadata(component.parent)
    _safe(directory, directory=True)


@contextmanager
def _locked_state() -> Iterator[Path]:
    path = _state_path()
    if not path.is_absolute():
        raise CHAOperationError("UNSAFE_CHA_LOGICAL_OPERATION_PATH")
    try:
        _create_directory(path.parent)
        lock = path.with_suffix(".lock")
        _safe(lock)
        descriptor = os.open(lock, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(descriptor, "r+b") as stream:
            information = os.fstat(stream.fileno())
            if not stat.S_ISREG(information.st_mode) or information.st_nlink != 1:
                raise CHAOperationError("UNSAFE_CHA_LOGICAL_OPERATION_LOCK")
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
                    raise CHAOperationError("CHA_LOGICAL_OPERATION_BUSY") from exc
                try:
                    yield path
                finally:
                    stream.seek(0)
                    windows_lock.locking(stream.fileno(), windows_lock.LK_UNLCK, 1)
            elif sys.platform != "win32" and os.name == "posix":
                import fcntl

                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    raise CHAOperationError("CHA_LOGICAL_OPERATION_BUSY") from exc
                try:
                    yield path
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            else:
                raise CHAOperationError("UNSUPPORTED_CHA_LOGICAL_OPERATION_PLATFORM")
    except OSError as exc:
        raise CHAOperationError("CHA_LOGICAL_OPERATION_PERSISTENCE_FAILED") from exc


def _upstream_state(raw: object) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_UPSTREAM_BYTES:
        raise ValueError("invalid retained LPPI source bytes")
    upstream = lppi_operation._validate_state(parse_canonical(raw))
    if upstream["status"] != "AUTHENTICATED_OPERATION_COMMITTED":
        raise ValueError("LPPI source is not committed")
    return cast(dict[str, Any], upstream)


def _validate_state(value: object) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _FIELDS:
        raise ValueError("CHA operation schema")
    if (
        type(value["schema_version"]) is not str
        or value["schema_version"] != "CHAInitialLogicalOperationV1"
        or type(value["mapping_generation"]) is not int
        or value["mapping_generation"] != 1
    ):
        raise ValueError("CHA operation initial generation")
    status = value["status"]
    if (
        type(status) is not str
        or status not in STATUSES
        or (
            type(value["history"]) is not list
            or value["history"] != list(STATUSES[: STATUSES.index(status) + 1])
        )
    ):
        raise ValueError("CHA operation progression")
    for field, prefix in (
        ("provisioning_operation_id", "prvop_"),
        ("logical_operation_id", "ago_"),
    ):
        if type(value[field]) is not str or re.fullmatch(prefix + _UUID7, value[field]) is None:
            raise ValueError("CHA operation identity")
    assigned = value["assigned_at_utc"]
    if type(assigned) is not str or _UTC.fullmatch(assigned) is None:
        raise ValueError("CHA operation timestamp")
    captured = datetime.strptime(assigned, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    milliseconds = reservation_epoch_milliseconds(captured)
    # The first 48 UUID bits retain the same full millisecond instant as the wire time.
    uuid_hex = value["logical_operation_id"][4:].replace("-", "")
    if int(uuid_hex[:12], 16) != milliseconds:
        raise ValueError("CHA operation reservation instant mismatch")
    retained_hex = value["lppi_operation_state_raw_hex"]
    if type(retained_hex) is not str or len(retained_hex) > 2 * _MAX_UPSTREAM_BYTES:
        raise ValueError("CHA operation retained LPPI encoding")
    retained = bytes.fromhex(retained_hex)
    if retained.hex() != retained_hex:
        raise ValueError("CHA operation noncanonical LPPI encoding")
    upstream = _upstream_state(retained)
    for field in ("pdsa_trust_domain", "provisioning_operation_id"):
        if type(value[field]) is not str or value[field] != upstream[field]:
            raise ValueError("CHA operation retained LPPI source mismatch")
    return cast(dict[str, Any], value)


def _read(path: Path) -> dict[str, Any] | None:
    _safe(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CHAOperationError("INVALID_RETAINED_CHA_LOGICAL_OPERATION") from exc
    try:
        with os.fdopen(descriptor, "rb") as stream:
            information = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(information.st_mode)
                or information.st_nlink != 1
                or information.st_size > _MAX_BYTES
            ):
                raise ValueError("unsafe or oversized CHA operation state")
            raw = stream.read(_MAX_BYTES + 1)
        if len(raw) > _MAX_BYTES:
            raise ValueError("oversized CHA operation state")
        return _validate_state(parse_canonical(raw))
    except (ValueError, TypeError, KeyError, RecursionError, OSError) as exc:
        raise CHAOperationError("INVALID_RETAINED_CHA_LOGICAL_OPERATION") from exc


def _write(path: Path, state: dict[str, Any]) -> None:
    _safe(path)
    try:
        _validate_state(state)
        raw = canonical_json_bytes(state)
        if len(raw) > _MAX_BYTES:
            raise ValueError("oversized CHA operation state")
    except (ValueError, TypeError, KeyError, RecursionError) as exc:
        raise CHAOperationError("INVALID_RETAINED_CHA_LOGICAL_OPERATION") from exc
    previous = _read(path)
    if previous is not None and (
        any(previous[field] != state[field] for field in _FIELDS - {"status", "history"})
        or (previous["status"] == STATUSES[1] and previous != state)
    ):
        raise CHAOperationError("CHA_LOGICAL_OPERATION_CONFLICT")
    try:
        _safe(path)
        atomic_write_bytes_durably(path, raw)
    except OSError as exc:
        raise CHAOperationError("CHA_LOGICAL_OPERATION_PERSISTENCE_FAILED") from exc


def _source(upstream: object) -> bytes:
    """Obtain exact retained lineage only after the installed LPPI verifier guard."""
    try:
        verified = require_verified_lppi_authenticated_operation(upstream)
        return cast(bytes, _operation_snapshot(verified).state_raw)
    except (ValueError, RuntimeError, OSError) as exc:
        raise CHAOperationError("VERIFIED_LPPI_AUTHENTICATED_OPERATION_REQUIRED") from exc


def _targets(state: dict[str, Any], upstream_raw: bytes) -> None:
    if state["lppi_operation_state_raw_hex"] != upstream_raw.hex():
        raise CHAOperationError("CHA_LOGICAL_OPERATION_CONFLICT")


def _reserve(path: Path, upstream_raw: bytes) -> dict[str, Any]:
    state = _read(path)
    if state is not None:
        _targets(state, upstream_raw)
        return state
    try:
        upstream = _upstream_state(upstream_raw)
        captured = _utc_now()
        milliseconds = reservation_epoch_milliseconds(captured)
        logical_operation_id = mint_uuid7("ago_", milliseconds)
        assigned = captured.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    except (UUID7Error, ValueError, TypeError, KeyError, RecursionError) as exc:
        raise CHAOperationError("INVALID_CHA_LOGICAL_OPERATION_RESERVATION") from exc
    state = {
        "schema_version": "CHAInitialLogicalOperationV1",
        "status": STATUSES[0],
        "history": [STATUSES[0]],
        "mapping_generation": 1,
        "pdsa_trust_domain": upstream["pdsa_trust_domain"],
        "provisioning_operation_id": upstream["provisioning_operation_id"],
        "logical_operation_id": logical_operation_id,
        "assigned_at_utc": assigned,
        "lppi_operation_state_raw_hex": upstream_raw.hex(),
    }
    _write(path, state)
    return state


def _commit(path: Path, reserved_raw: bytes) -> dict[str, Any]:
    current = _read(path)
    if current is None or canonical_json_bytes(current) != reserved_raw:
        raise CHAOperationError("CHA_LOGICAL_OPERATION_CONFLICT")
    if current["status"] == STATUSES[0]:
        current["status"] = STATUSES[1]
        current["history"] = list(STATUSES)
    # Reassert the physical fence even for a retained winner. A preceding process
    # could have crashed after rename but before completing its publication fence.
    _write(path, current)
    return current


def validate_retained_cha_operation(
    upstream: object, path: Path, state_raw: bytes
) -> dict[str, Any]:
    """Requalify current LPPI and exact committed mapping on every capability use."""
    upstream_raw = _source(upstream)
    if path != _state_path() or type(state_raw) is not bytes:
        raise CHAOperationError("CHA_LOGICAL_OPERATION_CONFLICT")
    state = _read(path)
    if state is None or canonical_json_bytes(state) != state_raw:
        raise CHAOperationError("CHA_LOGICAL_OPERATION_CONFLICT")
    _targets(state, upstream_raw)
    if state["status"] != STATUSES[1]:
        raise CHAOperationError("CHA_LOGICAL_OPERATION_NOT_COMMITTED")
    return state


def establish_installed_cha_logical_operation(upstream: object) -> VerifiedCHALogicalOperation:
    """Reserve or resume the initial mapping; callers never supply its UUID or clock."""
    upstream_raw = _source(upstream)
    with _locked_state() as path:
        reserved = _reserve(path, upstream_raw)
        state = _commit(path, canonical_json_bytes(reserved))
    return _issue_verified_cha_operation(upstream, path, canonical_json_bytes(state))


def load_installed_cha_logical_operation(upstream: object) -> VerifiedCHALogicalOperation:
    """Restore committed CHA authority solely through reverified upstream LPPI."""
    upstream_raw = _source(upstream)
    with _locked_state() as path:
        state = _read(path)
        if state is None or state["status"] != STATUSES[1]:
            raise CHAOperationError("CHA_LOGICAL_OPERATION_NOT_COMMITTED")
        _targets(state, upstream_raw)
        _write(path, state)
    return _issue_verified_cha_operation(upstream, path, canonical_json_bytes(state))
