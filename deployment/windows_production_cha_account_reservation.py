"""Installed CryptoHunterAccountAuthority for initial candidate account reservation.

INITIAL_BINDING is neither authorization nor a genuine account. Protected
installed directory is required; disk rollback protection is not supplied.
"""

from __future__ import annotations

import hashlib
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
from bot_core.licensing.cha_account_reservation import (
    VerifiedAccountGenesisInitialBinding,
    _issue_verified_initial_binding,
)
from bot_core.licensing.cha_logical_operation import (
    _logical_operation_snapshot,
    require_verified_cha_logical_operation,
)
from bot_core.persistence.physical_durability import (
    atomic_write_bytes_durably,
    flush_created_directory_metadata,
)
from bot_core.uuid7 import UUID7Error, mint_uuid7, reservation_epoch_milliseconds
from deployment import windows_production_cha_operation as cha_operation
from deployment.windows_cng_pre_enrollment import WindowsCNGPreEnrollmentError, _safe_path

OWNER = "CryptoHunterAccountAuthority"
STATUS = "INITIAL_BINDING_COMMITTED"
_MAX_BYTES = 16_777_216
_MAX_UPSTREAM_BYTES = 4_194_304
_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "reservation_generation",
        "pdsa_trust_domain",
        "provisioning_operation_id",
        "logical_operation_id",
        "account_id",
        "assigned_at_utc",
        "canonical_request_raw_hex",
        "canonical_request_sha256",
        "initial_binding_sha256",
        "cha_operation_state_raw_hex",
    }
)
_UUID7 = r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
_UTC = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z\Z")


class AccountReservationError(RuntimeError):
    """The candidate binding or upstream authority failed closed."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _state_path() -> Path:
    return cast(Path, cha_operation._state_path().with_name("initial-cha-account-reservation.json"))


def _safe(path: Path, *, directory: bool = False) -> None:
    try:
        _safe_path(path, directory=directory)
    except (OSError, WindowsCNGPreEnrollmentError) as exc:
        raise AccountReservationError("UNSAFE_CHA_ACCOUNT_RESERVATION_PATH") from exc


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
        raise AccountReservationError("UNSAFE_CHA_ACCOUNT_RESERVATION_PATH")
    try:
        _create_directory(path.parent)
        lock = path.with_suffix(".lock")
        _safe(lock)
        descriptor = os.open(lock, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(descriptor, "r+b") as stream:
            information = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(information.st_mode)
                or information.st_nlink != 1
                or information.st_size > 1
            ):
                raise AccountReservationError("UNSAFE_CHA_ACCOUNT_RESERVATION_LOCK")
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
                    raise AccountReservationError("CHA_ACCOUNT_RESERVATION_BUSY") from exc
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
                    raise AccountReservationError("CHA_ACCOUNT_RESERVATION_BUSY") from exc
                try:
                    yield path
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            else:
                raise AccountReservationError("UNSUPPORTED_CHA_ACCOUNT_RESERVATION_PLATFORM")
    except OSError as exc:
        raise AccountReservationError("CHA_ACCOUNT_RESERVATION_PERSISTENCE_FAILED") from exc


def _upstream_state(raw: object) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_UPSTREAM_BYTES:
        raise ValueError("invalid retained CHA source bytes")
    upstream = cha_operation._validate_state(parse_canonical(raw))
    if upstream["status"] != "PRVOP_AGO_BIJECTION_COMMITTED":
        raise ValueError("CHA source is not committed")
    return cast(dict[str, Any], upstream)


def _canonical_request(upstream_raw: bytes, account_id: str) -> bytes:
    upstream = _upstream_state(upstream_raw)
    lppi = parse_canonical(bytes.fromhex(upstream["lppi_operation_state_raw_hex"]))
    source_binding = parse_canonical(bytes.fromhex(lppi["binding_raw_hex"]))
    return cast(
        bytes,
        canonical_json_bytes(
            {
                "schema_version": "AccountGenesisInitialRequestV1",
                "request_domain": "cryptohunter.m0.5.account-genesis-request.v1",
                "purpose": "INITIAL_INSTALLED_ACCOUNT_RESERVATION",
                "environment": source_binding["environment"],
                "intended_action": "ACCOUNT_GENESIS_BOOTSTRAP",
                "product_scope": "CryptoHunter",
                "pdsa_trust_domain": upstream["pdsa_trust_domain"],
                "logical_operation_id": upstream["logical_operation_id"],
                "provisioning_operation_id": upstream["provisioning_operation_id"],
                "account_id": account_id,
                "reservation_relation": "EXACT_OPERATION_ACCOUNT",
                "root_proof_handoff": {
                    "timing": "FROZEN_AFTER_INITIAL_BINDING_BEFORE_PREPARED",
                    "initial_binding_authenticity_model": (
                        "CHA_AUTHENTICATED_EXACT_INITIAL_BINDING_DIGEST_AND_REFERENCE"
                    ),
                },
                "cha_state_sha256": hashlib.sha256(upstream_raw).hexdigest(),
            }
        ),
    )


def _decode_hex(value: object, limit: int) -> bytes:
    if type(value) is not str or len(value) > limit * 2:
        raise ValueError("invalid retained encoding")
    raw = bytes.fromhex(cast(str, value))
    if raw.hex() != value:
        raise ValueError("noncanonical retained encoding")
    return raw


def _binding_integrity(state: dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {field: value for field, value in state.items() if field != "initial_binding_sha256"}
        )
    ).hexdigest()


def _validate_state(value: object) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _FIELDS:
        raise ValueError("account reservation closed schema")
    if (
        value["schema_version"] != "AccountGenesisInitialBindingV1"
        or value["status"] != STATUS
        or type(value["reservation_generation"]) is not int
        or value["reservation_generation"] != 1
    ):
        raise ValueError("account reservation state")
    if value["initial_binding_sha256"] != _binding_integrity(value):
        raise ValueError("account initial binding integrity mismatch")
    upstream_raw = _decode_hex(value["cha_operation_state_raw_hex"], _MAX_UPSTREAM_BYTES)
    upstream = _upstream_state(upstream_raw)
    for field in ("pdsa_trust_domain", "logical_operation_id", "provisioning_operation_id"):
        if type(value[field]) is not str or value[field] != upstream[field]:
            raise ValueError("account reservation source mismatch")
    account = value["account_id"]
    if type(account) is not str or re.fullmatch("acct_" + _UUID7, account) is None:
        raise ValueError("account candidate identity")
    assigned = value["assigned_at_utc"]
    if type(assigned) is not str or _UTC.fullmatch(assigned) is None:
        raise ValueError("account reservation instant")
    captured = datetime.strptime(assigned, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    milliseconds = reservation_epoch_milliseconds(captured)
    if int(account[5:].replace("-", "")[:12], 16) != milliseconds:
        raise ValueError("account reservation instant mismatch")
    request_raw = _decode_hex(value["canonical_request_raw_hex"], 4096)
    if request_raw != _canonical_request(upstream_raw, account):
        raise ValueError("account reservation request mismatch")
    if value["canonical_request_sha256"] != hashlib.sha256(request_raw).hexdigest():
        raise ValueError("account reservation request fingerprint mismatch")
    return cast(dict[str, Any], value)


def _read(path: Path) -> dict[str, Any] | None:
    _safe(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise AccountReservationError("INVALID_RETAINED_CHA_ACCOUNT_RESERVATION") from exc
    try:
        with os.fdopen(descriptor, "rb") as stream:
            information = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(information.st_mode)
                or information.st_nlink != 1
                or information.st_size > _MAX_BYTES
            ):
                raise ValueError("unsafe or oversized account reservation state")
            raw = stream.read(_MAX_BYTES + 1)
        if len(raw) > _MAX_BYTES:
            raise ValueError("oversized account reservation state")
        return _validate_state(parse_canonical(raw))
    except (ValueError, TypeError, KeyError, RecursionError, OSError) as exc:
        raise AccountReservationError("INVALID_RETAINED_CHA_ACCOUNT_RESERVATION") from exc


def _write(path: Path, state: dict[str, Any]) -> None:
    _safe(path)
    try:
        _validate_state(state)
        raw = canonical_json_bytes(state)
        if len(raw) > _MAX_BYTES:
            raise ValueError("oversized account reservation")
    except (ValueError, TypeError, KeyError, RecursionError) as exc:
        raise AccountReservationError("INVALID_RETAINED_CHA_ACCOUNT_RESERVATION") from exc
    previous = _read(path)
    if previous is not None and previous != state:
        raise AccountReservationError("CHA_ACCOUNT_RESERVATION_CONFLICT")
    try:
        _safe(path)
        atomic_write_bytes_durably(path, raw)
    except OSError as exc:
        # Replacement may have succeeded before a fence failed. Reconcile by
        # rereading, never publish on this failed attempt, and retain the winner
        # for a subsequent locked retry that completes all physical fences.
        retained = _read(path)
        if retained is not None and retained != state:
            raise AccountReservationError("CHA_ACCOUNT_RESERVATION_CONFLICT") from exc
        raise AccountReservationError("CHA_ACCOUNT_RESERVATION_PERSISTENCE_FAILED") from exc


def _source(upstream: object) -> bytes:
    try:
        verified = require_verified_cha_logical_operation(upstream)
        return cast(bytes, _logical_operation_snapshot(verified).state_raw)
    except (ValueError, RuntimeError, OSError) as exc:
        raise AccountReservationError("VERIFIED_CHA_LOGICAL_OPERATION_REQUIRED") from exc


def _targets(state: dict[str, Any], upstream_raw: bytes) -> None:
    upstream = _upstream_state(upstream_raw)
    if (
        state["cha_operation_state_raw_hex"] != upstream_raw.hex()
        or state["canonical_request_raw_hex"]
        != _canonical_request(upstream_raw, state["account_id"]).hex()
        or any(
            state[field] != upstream[field]
            for field in ("pdsa_trust_domain", "logical_operation_id", "provisioning_operation_id")
        )
    ):
        raise AccountReservationError("CHA_ACCOUNT_RESERVATION_CONFLICT")


def _reserve(path: Path, upstream_raw: bytes) -> dict[str, Any]:
    state = _read(path)
    if state is not None:
        _targets(state, upstream_raw)
    else:
        try:
            upstream = _upstream_state(upstream_raw)
            captured = _utc_now()
            account_id = mint_uuid7("acct_", reservation_epoch_milliseconds(captured))
            request_raw = _canonical_request(upstream_raw, account_id)
            assigned = captured.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        except (UUID7Error, ValueError, TypeError, KeyError, RecursionError) as exc:
            raise AccountReservationError("INVALID_CHA_ACCOUNT_RESERVATION") from exc
        state = {
            "schema_version": "AccountGenesisInitialBindingV1",
            "status": STATUS,
            "reservation_generation": 1,
            "pdsa_trust_domain": upstream["pdsa_trust_domain"],
            "logical_operation_id": upstream["logical_operation_id"],
            "provisioning_operation_id": upstream["provisioning_operation_id"],
            "account_id": account_id,
            "assigned_at_utc": assigned,
            "canonical_request_raw_hex": request_raw.hex(),
            "canonical_request_sha256": hashlib.sha256(request_raw).hexdigest(),
            "cha_operation_state_raw_hex": upstream_raw.hex(),
        }
        state["initial_binding_sha256"] = _binding_integrity(state)
    # Reassert exact retained winner fences after an ambiguous prior publication.
    _write(path, state)
    return state


def validate_retained_initial_binding(
    upstream: object, path: Path, state_raw: bytes
) -> dict[str, Any]:
    """Requalify upstream and exact durable request/source/candidate on every use."""
    upstream_raw = _source(upstream)
    if path != _state_path() or type(state_raw) is not bytes:
        raise AccountReservationError("CHA_ACCOUNT_RESERVATION_CONFLICT")
    state = _read(path)
    if state is None or canonical_json_bytes(state) != state_raw:
        raise AccountReservationError("CHA_ACCOUNT_RESERVATION_CONFLICT")
    _targets(state, upstream_raw)
    return state


def establish_installed_account_initial_binding(
    upstream: object,
) -> VerifiedAccountGenesisInitialBinding:
    """Reserve one candidate from guarded CHA; no caller IDs, time, path or DTO."""
    upstream_raw = _source(upstream)
    with _locked_state() as path:
        state = _reserve(path, upstream_raw)
    return _issue_verified_initial_binding(upstream, path, canonical_json_bytes(state))


def load_installed_account_initial_binding(
    upstream: object,
) -> VerifiedAccountGenesisInitialBinding:
    """Restore a retained winner without ever minting an account candidate."""
    upstream_raw = _source(upstream)
    with _locked_state() as path:
        state = _read(path)
        if state is None:
            raise AccountReservationError("CHA_ACCOUNT_RESERVATION_NOT_FOUND")
        _targets(state, upstream_raw)
        _write(path, state)
    return _issue_verified_initial_binding(upstream, path, canonical_json_bytes(state))
