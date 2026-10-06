"""Installed, durable initial LPPI successor lifecycle; no legal enrollment.

The singleton is machine state, never a public capability constructor. Every
runtime authority is issued by this loader after current local requalification
and independent re-verification of the complete retained cryptographic history.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
from weakref import WeakKeyDictionary

from bot_core.licensing.canonical import canonical_json_bytes, parse_canonical
from bot_core.licensing.lppi_authority_custody import (
    LPPIAuthorityKeyCustodyEvidenceV1,
    verify_lppi_authority_key_custody,
)
from bot_core.licensing.lppi_authority_key import (
    LPPIAuthorityKeyBindingV1,
    build_lppi_authority_key_binding,
    require_binding_targets,
    verify_authority_key_continuity,
    verify_authority_key_pop,
    verify_low_s_signature,
)
from bot_core.licensing.lppi_package_acceptance import (
    VerifiedLPPIClientPackageAcceptance,
    require_verified_lppi_package_acceptance,
)
from deployment.platforms.windows import resolve_paths
from deployment.windows_cng_pre_enrollment import _safe_path
from deployment.windows_lppi_authority_custody import collect_lppi_authority_key_custody
from deployment.windows_lppi_authority_key import (
    WindowsLPPIAuthorityKey,
    open_or_create_production_lppi_authority_key,
    probe_production_lppi_authority_key_presence,
    require_verified_production_lppi_authority_key,
)

STATUSES = (
    "CREATION_RESERVED",
    "CANDIDATE",
    "CONTINUITY_SIGNATURE_VERIFIED",
    "AUTHORITY_KEY_POP_VERIFIED",
    "CUSTODY_EVIDENCE_VERIFIED",
    "VERIFIED_CONTINUITY",
    "ACTIVE",
)
_FIELDS = frozenset(
    {
        "schema_version",
        "reservation_id",
        "status",
        "history",
        "created_at_utc",
        "package_raw_hex",
        "request_raw_hex",
        "projection_raw_hex",
        "pre_enrollment_key_unique_name",
        "acceptance_challenge_raw_hex",
        "acceptance_signature_hex",
        "acceptance_live_attestation_hex",
        "acceptance_live_signature_hex",
        "creation_attempted",
        "authority_sec1_hex",
        "authority_unique_name",
        "custody_raw_hex",
        "binding_raw_hex",
        "continuity_signature_hex",
        "authority_pop_signature_hex",
        "pdsa_trust_domain",
        "provisioning_subject_id",
        "generation",
        "ceremony_id",
        "release_policy_digest_sha256",
    }
)


class LPPIAuthorityLifecycleError(RuntimeError):
    """Durable identity, status history or live authority failed closed."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _state_path() -> Path:
    return cast(
        Path, (resolve_paths().state / "LPPIAuthority" / "initial-authority-key.json").absolute()
    )


@contextmanager
def _locked_state() -> Iterator[Path]:
    path = _state_path()
    directory = path.parent
    _safe_path(directory, directory=True)
    directory.mkdir(parents=True, exist_ok=True)
    _safe_path(directory, directory=True)
    lock = directory / "initial-authority-key.lock"
    _safe_path(lock)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "r+b") as stream:
        information = os.fstat(stream.fileno())
        if not stat.S_ISREG(information.st_mode) or information.st_nlink != 1:
            raise LPPIAuthorityLifecycleError("UNSAFE_LPPI_AUTHORITY_STATE_LOCK")
        if os.name == "nt":
            import msvcrt

            windows_lock: Any = msvcrt

            if not stream.seek(0, 2):
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            try:
                windows_lock.locking(stream.fileno(), windows_lock.LK_NBLCK, 1)
            except OSError as exc:
                raise LPPIAuthorityLifecycleError("LPPI_AUTHORITY_LIFECYCLE_BUSY") from exc
            try:
                yield path
            finally:
                stream.seek(0)
                windows_lock.locking(stream.fileno(), windows_lock.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise LPPIAuthorityLifecycleError("LPPI_AUTHORITY_LIFECYCLE_BUSY") from exc
            try:
                yield path
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _read(path: Path) -> dict[str, Any] | None:
    _safe_path(path)
    if not path.exists():
        return None
    try:
        if path.stat().st_size > 1_048_576:
            raise ValueError("oversized state")
        state = parse_canonical(path.read_bytes())
        if (
            type(state) is not dict
            or set(state) != _FIELDS
            or state["schema_version"] != "LPPIInitialAuthorityLifecycleV1"
        ):
            raise ValueError("state schema")
        status = state["status"]
        if status not in STATUSES or state["history"] != list(
            STATUSES[: STATUSES.index(status) + 1]
        ):
            raise ValueError("state progression")
        if (
            type(state["generation"]) is not int
            or state["generation"] != 1
            or type(state["creation_attempted"]) is not bool
        ):
            raise ValueError("state initial generation")
        if (
            type(state["reservation_id"]) is not str
            or len(bytes.fromhex(state["reservation_id"])) != 32
        ):
            raise ValueError("state reservation")
        public, unique = state["authority_sec1_hex"], state["authority_unique_name"]
        if (public is None) != (unique is None) or (
            public is not None and (type(public) is not str or type(unique) is not str)
        ):
            raise ValueError("state identity")
        stage = STATUSES.index(status)
        if stage >= 1:
            if public is None or not state["creation_attempted"]:
                raise ValueError("state candidate")
            binding = LPPIAuthorityKeyBindingV1.from_canonical_bytes(
                bytes.fromhex(state["binding_raw_hex"])
            )
            LPPIAuthorityKeyCustodyEvidenceV1.from_canonical_bytes(
                bytes.fromhex(state["custody_raw_hex"])
            )
            if binding.document["created_at_utc"] != state["created_at_utc"]:
                raise ValueError("state reservation timestamp")
        elif state["binding_raw_hex"] is not None or state["custody_raw_hex"] is not None:
            raise ValueError("state premature binding")
        for field, gate in (("continuity_signature_hex", 2), ("authority_pop_signature_hex", 3)):
            if stage >= gate:
                if type(state[field]) is not str or not bytes.fromhex(state[field]):
                    raise ValueError("state missing proof")
            elif state[field] is not None:
                raise ValueError("state premature proof")
        return state
    except (ValueError, TypeError, KeyError, RecursionError, OSError) as exc:
        raise LPPIAuthorityLifecycleError("INVALID_RETAINED_LPPI_AUTHORITY_LIFECYCLE") from exc


def _write(path: Path, state: dict[str, Any]) -> None:
    _safe_path(path)
    temporary = path.with_name(path.name + "." + secrets.token_hex(16) + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical_json_bytes(state))
            stream.flush()
            os.fsync(stream.fileno())
        _safe_path(path)
        os.replace(temporary, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _targets(state: dict[str, Any], accepted: VerifiedLPPIClientPackageAcceptance) -> None:
    package = accepted.package
    expected = {
        "package_raw_hex": package.canonical_package.hex(),
        "request_raw_hex": accepted.request.canonical_bytes.hex(),
        "projection_raw_hex": accepted.projection.canonical_bytes.hex(),
        "pre_enrollment_key_unique_name": accepted.pre_enrollment_key_unique_name,
        "pdsa_trust_domain": package.payload["pdsa_trust_domain"],
        "provisioning_subject_id": package.payload["provisioning_subject_id"],
        "generation": 1,
        "ceremony_id": accepted.context.ceremony_id,
        "release_policy_digest_sha256": accepted.context.release_payload_digest,
    }
    if any(state[field] != value for field, value in expected.items()):
        raise LPPIAuthorityLifecycleError("LPPI_AUTHORITY_KEY_LIFECYCLE_CONFLICT")
    from bot_core.licensing.lppi_authority_key import _timestamp
    from bot_core.licensing.lppi_package_acceptance import (
        PACKAGE_ACCEPTANCE_CUSTODY_DOMAIN,
        PACKAGE_ACCEPTANCE_DOMAIN,
        validate_package_acceptance_challenge,
    )
    from bot_core.licensing.production_tpm_custody import (
        _verify_creation,
        parse_production_creation_attestation,
        parse_production_ecc_public,
    )

    created = _timestamp(state["created_at_utc"])
    if (
        not _timestamp(package.payload["issued_at_utc"])
        <= created
        < _timestamp(package.payload["expires_at_utc"])
        or created > _utc_now()
    ):
        raise LPPIAuthorityLifecycleError("LPPI_AUTHORITY_KEY_LIFECYCLE_CONFLICT")
    challenge_raw = bytes.fromhex(state["acceptance_challenge_raw_hex"])
    challenge = validate_package_acceptance_challenge(challenge_raw)
    expected_challenge = {
        "pdsa_package_digest_sha256": package.package_digest_sha256,
        **{
            field: package.payload[field]
            for field in (
                "pre_enrollment_request_digest_sha256",
                "pre_enrollment_public_key_fingerprint_sha256",
                "verified_tpm_public_projection_id",
                "verified_tpm_exchange_reference",
            )
        },
    }
    if any(challenge[field] != value for field, value in expected_challenge.items()):
        raise LPPIAuthorityLifecycleError("LPPI_AUTHORITY_KEY_LIFECYCLE_CONFLICT")
    verify_low_s_signature(
        accepted.pre_enrollment_public_key_bytes,
        bytes.fromhex(state["acceptance_signature_hex"]),
        PACKAGE_ACCEPTANCE_DOMAIN + hashlib.sha256(challenge_raw).digest(),
        error="LPPI_AUTHORITY_KEY_LIFECYCLE_CONFLICT",
    )
    current = parse_production_creation_attestation(accepted.live_attestation_raw)
    retained = bytes.fromhex(state["acceptance_live_attestation_hex"])
    if parse_production_creation_attestation(retained).qualified_signer != current.qualified_signer:
        raise LPPIAuthorityLifecycleError("LPPI_AUTHORITY_KEY_LIFECYCLE_CONFLICT")
    _verify_creation(
        attest=retained,
        signature=bytes.fromhex(state["acceptance_live_signature_hex"]),
        ak=parse_production_ecc_public(
            bytes.fromhex(accepted.projection.document["ak"]["public_area"]["hex"]), role="ak"
        ),
        expected_name=current.object_name,
        expected_creation_hash=current.creation_hash,
        expected_qualifier=hashlib.sha256(
            PACKAGE_ACCEPTANCE_CUSTODY_DOMAIN + hashlib.sha256(challenge_raw).digest()
        ).digest(),
    )


class _CreationReservation:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("only installed lifecycle can reserve successor creation")


@dataclass(frozen=True, slots=True)
class _ReservationSnapshot:
    path: Path
    accepted: VerifiedLPPIClientPackageAcceptance
    reservation_id: str
    creation_authorized: bool


@dataclass(frozen=True, slots=True)
class LPPIAuthorityKeyReservationDescriptor:
    mode: str
    expected_authority_public_key: bytes | None
    expected_authority_unique_name: str | None
    pre_enrollment_public_key: bytes
    pre_enrollment_unique_name: str
    reservation_id: str


_RESERVATIONS: WeakKeyDictionary[_CreationReservation, _ReservationSnapshot] = WeakKeyDictionary()


def require_lppi_authority_key_reservation(
    value: object, accepted_package: object
) -> LPPIAuthorityKeyReservationDescriptor:
    accepted = require_verified_lppi_package_acceptance(accepted_package)
    if type(value) is not _CreationReservation:
        raise LPPIAuthorityLifecycleError("TRUSTED_LPPI_CREATION_RESERVATION_REQUIRED")
    snapshot = _RESERVATIONS.get(value)
    if snapshot is None or snapshot.accepted is not accepted or snapshot.path != _state_path():
        raise LPPIAuthorityLifecycleError("TRUSTED_LPPI_CREATION_RESERVATION_REQUIRED")
    state = _read(snapshot.path)
    if state is None or state["reservation_id"] != snapshot.reservation_id:
        raise LPPIAuthorityLifecycleError("LPPI_AUTHORITY_KEY_LIFECYCLE_CONFLICT")
    _targets(state, accepted)
    public = state["authority_sec1_hex"]
    mode = (
        "recover"
        if public is not None
        else "create"
        if snapshot.creation_authorized
        else "reconcile"
    )
    return LPPIAuthorityKeyReservationDescriptor(
        mode,
        bytes.fromhex(public) if public is not None else None,
        state["authority_unique_name"],
        accepted.pre_enrollment_public_key_bytes,
        accepted.pre_enrollment_key_unique_name,
        snapshot.reservation_id,
    )


def _reserve(
    path: Path, accepted: VerifiedLPPIClientPackageAcceptance
) -> tuple[_CreationReservation, dict[str, Any]]:
    state = _read(path)
    if state is None:
        if probe_production_lppi_authority_key_presence(accepted):
            raise LPPIAuthorityLifecycleError("PREEXISTING_UNRESERVED_LPPI_AUTHORITY_IDENTITY")
        package = accepted.package
        state = {
            "schema_version": "LPPIInitialAuthorityLifecycleV1",
            "reservation_id": secrets.token_hex(32),
            "status": "CREATION_RESERVED",
            "history": ["CREATION_RESERVED"],
            "created_at_utc": _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "package_raw_hex": package.canonical_package.hex(),
            "request_raw_hex": accepted.request.canonical_bytes.hex(),
            "projection_raw_hex": accepted.projection.canonical_bytes.hex(),
            "pre_enrollment_key_unique_name": accepted.pre_enrollment_key_unique_name,
            "acceptance_challenge_raw_hex": accepted.pop_challenge_raw.hex(),
            "acceptance_signature_hex": accepted.pop_signature.hex(),
            "acceptance_live_attestation_hex": accepted.live_attestation_raw.hex(),
            "acceptance_live_signature_hex": accepted.live_signature.hex(),
            "creation_attempted": False,
            "authority_sec1_hex": None,
            "authority_unique_name": None,
            "custody_raw_hex": None,
            "binding_raw_hex": None,
            "continuity_signature_hex": None,
            "authority_pop_signature_hex": None,
            "pdsa_trust_domain": package.payload["pdsa_trust_domain"],
            "provisioning_subject_id": package.payload["provisioning_subject_id"],
            "generation": 1,
            "ceremony_id": accepted.context.ceremony_id,
            "release_policy_digest_sha256": accepted.context.release_payload_digest,
        }
        _write(path, state)
    _targets(state, accepted)
    creation_authorized = not state["creation_attempted"]
    if creation_authorized:
        state["creation_attempted"] = True
        _write(path, state)
    reservation = object.__new__(_CreationReservation)
    _RESERVATIONS[reservation] = _ReservationSnapshot(
        path, accepted, state["reservation_id"], creation_authorized
    )
    return reservation, state


def _advance(path: Path, state: dict[str, Any], status: str) -> None:
    index = STATUSES.index(state["status"])
    if index + 1 >= len(STATUSES) or STATUSES[index + 1] != status:
        raise LPPIAuthorityLifecycleError("LPPI_AUTHORITY_STATUS_GATE_ORDER_VIOLATION")
    state["status"] = status
    state["history"] = [*state["history"], status]
    _write(path, state)


def _verify_complete(
    state: dict[str, Any], accepted: object, key: object
) -> LPPIAuthorityKeyBindingV1:
    key = require_verified_production_lppi_authority_key(key)
    binding = LPPIAuthorityKeyBindingV1.from_canonical_bytes(
        bytes.fromhex(state["binding_raw_hex"])
    )
    custody = LPPIAuthorityKeyCustodyEvidenceV1.from_canonical_bytes(
        bytes.fromhex(state["custody_raw_hex"])
    )
    verify_authority_key_continuity(
        binding, bytes.fromhex(state["continuity_signature_hex"]), accepted=accepted
    )
    verify_authority_key_pop(
        binding, bytes.fromhex(state["authority_pop_signature_hex"]), key=key, evidence=custody
    )
    verify_lppi_authority_key_custody(custody, accepted=accepted, key=key)
    require_binding_targets(binding, accepted=accepted, key=key, evidence=custody)
    # A retained attestation is history. Observe the currently opened successor
    # again through PCP/TBS and prove its current private-key possession before
    # restoring or consuming ACTIVE authority, without replacing retained bytes.
    live = collect_lppi_authority_key_custody(accepted=accepted, key=key)
    verify_lppi_authority_key_custody(live, accepted=accepted, key=key)
    from bot_core.licensing.lppi_authority_custody import BINDING_FIELDS

    if any(live.document[field] != custody.document[field] for field in BINDING_FIELDS):
        raise LPPIAuthorityLifecycleError("LPPI_AUTHORITY_KEY_LIFECYCLE_CONFLICT")
    verify_authority_key_pop(
        binding, key.sign_binding_pop(binding.canonical_bytes), key=key, evidence=custody
    )
    return binding


@dataclass(frozen=True, slots=True)
class _ActiveSnapshot:
    path: Path
    state_raw: bytes
    accepted: VerifiedLPPIClientPackageAcceptance
    key: WindowsLPPIAuthorityKey
    reservation: _CreationReservation


class VerifiedActiveLPPIAuthorityKey:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("ACTIVE LPPI authority comes only from installed verified lifecycle")

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("ACTIVE LPPI authority is immutable")

    @property
    def binding(self) -> LPPIAuthorityKeyBindingV1:
        snapshot = _active_snapshot(self)
        state = parse_canonical(snapshot.state_raw)
        return LPPIAuthorityKeyBindingV1.from_canonical_bytes(
            bytes.fromhex(state["binding_raw_hex"])
        )

    @property
    def active_key_record(self) -> dict[str, Any]:
        binding = self.binding
        document = binding.document
        return {
            "pdsa_trust_domain": document["pdsa_trust_domain"],
            "provisioning_subject_id": document["provisioning_subject_id"],
            "generation": document["generation"],
            "status": "ACTIVE",
            **{
                field: document[field]
                for field in (
                    "lppi_authority_public_key_algorithm_profile",
                    "lppi_authority_public_key_fingerprint_sha256",
                    "custody_profile",
                    "lppi_authority_key_tpm_name",
                    "lppi_authority_tpmt_public_sha256",
                )
            },
            "binding_digest_sha256": binding.digest_sha256,
        }

    def close(self) -> None:
        snapshot = _ACTIVE.pop(self, None)
        if snapshot is not None:
            try:
                snapshot.key.close()
            finally:
                pre_enrollment = _OWNED_PRE_ENROLLMENT.pop(self, None)
                if pre_enrollment is not None:
                    pre_enrollment.close()


_ACTIVE: WeakKeyDictionary[VerifiedActiveLPPIAuthorityKey, _ActiveSnapshot] = WeakKeyDictionary()
_OWNED_PRE_ENROLLMENT: WeakKeyDictionary[VerifiedActiveLPPIAuthorityKey, Any] = WeakKeyDictionary()


def _active_snapshot(value: object) -> _ActiveSnapshot:
    if type(value) is not VerifiedActiveLPPIAuthorityKey:
        raise LPPIAuthorityLifecycleError("VERIFIED_ACTIVE_LPPI_AUTHORITY_KEY_REQUIRED")
    snapshot = _ACTIVE.get(value)
    if snapshot is None or snapshot.path != _state_path():
        raise LPPIAuthorityLifecycleError("VERIFIED_ACTIVE_LPPI_AUTHORITY_KEY_REQUIRED")
    accepted = require_verified_lppi_package_acceptance(snapshot.accepted)
    state = _read(snapshot.path)
    if (
        state is None
        or state["status"] != "ACTIVE"
        or canonical_json_bytes(state) != snapshot.state_raw
    ):
        raise LPPIAuthorityLifecycleError("LPPI_AUTHORITY_KEY_LIFECYCLE_CONFLICT")
    _targets(state, accepted)
    require_verified_production_lppi_authority_key(snapshot.key)
    _verify_complete(state, accepted, snapshot.key)
    return snapshot


def require_verified_active_lppi_authority_key(value: object) -> VerifiedActiveLPPIAuthorityKey:
    _active_snapshot(value)
    return cast(VerifiedActiveLPPIAuthorityKey, value)


def establish_installed_lppi_authority_key(
    accepted_package: object,
) -> VerifiedActiveLPPIAuthorityKey:
    """Resume exactly one initial identity; never issue membership or operation IDs."""
    accepted = require_verified_lppi_package_acceptance(accepted_package)
    with _locked_state() as path:
        reservation, state = _reserve(path, accepted)
        key = open_or_create_production_lppi_authority_key(accepted, reservation)
        try:
            if state["authority_sec1_hex"] is None:
                state["authority_sec1_hex"] = key.public_key_bytes.hex()
                state["authority_unique_name"] = key.key_unique_name
                _write(path, state)
            if state["status"] == "CREATION_RESERVED":
                evidence = collect_lppi_authority_key_custody(accepted=accepted, key=key)
                binding = build_lppi_authority_key_binding(
                    accepted=accepted,
                    key=key,
                    evidence=evidence,
                    created_at_utc=state["created_at_utc"],
                )
                state["custody_raw_hex"] = evidence.canonical_bytes.hex()
                state["binding_raw_hex"] = binding.canonical_bytes.hex()
                _advance(path, state, "CANDIDATE")
            binding = LPPIAuthorityKeyBindingV1.from_canonical_bytes(
                bytes.fromhex(state["binding_raw_hex"])
            )
            evidence = LPPIAuthorityKeyCustodyEvidenceV1.from_canonical_bytes(
                bytes.fromhex(state["custody_raw_hex"])
            )
            if state["status"] == "CANDIDATE":
                signature = accepted.key.sign_authority_continuity(
                    binding.canonical_bytes, production_trust_context=accepted.context
                )
                verify_authority_key_continuity(binding, signature, accepted=accepted)
                state["continuity_signature_hex"] = signature.hex()
                _advance(path, state, "CONTINUITY_SIGNATURE_VERIFIED")
            verify_authority_key_continuity(
                binding, bytes.fromhex(state["continuity_signature_hex"]), accepted=accepted
            )
            if state["status"] == "CONTINUITY_SIGNATURE_VERIFIED":
                signature = key.sign_binding_pop(binding.canonical_bytes)
                verify_authority_key_pop(binding, signature, key=key, evidence=evidence)
                state["authority_pop_signature_hex"] = signature.hex()
                _advance(path, state, "AUTHORITY_KEY_POP_VERIFIED")
            verify_authority_key_pop(
                binding,
                bytes.fromhex(state["authority_pop_signature_hex"]),
                key=key,
                evidence=evidence,
            )
            verify_lppi_authority_key_custody(evidence, accepted=accepted, key=key)
            if state["status"] == "AUTHORITY_KEY_POP_VERIFIED":
                _advance(path, state, "CUSTODY_EVIDENCE_VERIFIED")
            require_binding_targets(binding, accepted=accepted, key=key, evidence=evidence)
            if state["status"] == "CUSTODY_EVIDENCE_VERIFIED":
                _advance(path, state, "VERIFIED_CONTINUITY")
            _verify_complete(state, accepted, key)
            if state["status"] == "VERIFIED_CONTINUITY":
                _advance(path, state, "ACTIVE")
            result = object.__new__(VerifiedActiveLPPIAuthorityKey)
            _ACTIVE[result] = _ActiveSnapshot(
                path, canonical_json_bytes(state), accepted, key, reservation
            )
            require_verified_active_lppi_authority_key(result)
            return result
        except BaseException:
            key.close()
            raise


def establish_installed_lppi_authority_from_artifacts(
    package_raw: bytes,
    *,
    request_raw: bytes,
    request_signature: bytes,
    challenge_raw: bytes,
    activation_request_raw: bytes,
    tpm_request_raw: bytes,
    tpm_challenge_raw: bytes,
    tpm_response_raw: bytes,
    endorsement_raw: bytes,
    custody_evidence_raw: bytes,
    ak_key_name: str,
) -> VerifiedActiveLPPIAuthorityKey:
    """Installed native entrypoint with fixed trust and open-only retained identity."""
    from bot_core.licensing.lppi_package_acceptance import accept_production_lppi_package
    from deployment.platforms.windows import production_trust_package_path
    from deployment.windows_cng_pre_enrollment import WindowsCNGPreEnrollmentKey
    from deployment.windows_stage9_production_trust import CEREMONY_ID, load_production_trust

    key = WindowsCNGPreEnrollmentKey.open_existing(resolve_paths().state / "PreEnrollment")
    try:
        accepted = accept_production_lppi_package(
            package_raw,
            context=load_production_trust(production_trust_package_path(CEREMONY_ID)),
            key=key,
            request_raw=request_raw,
            request_signature=request_signature,
            challenge_raw=challenge_raw,
            activation_request_raw=activation_request_raw,
            tpm_request_raw=tpm_request_raw,
            tpm_challenge_raw=tpm_challenge_raw,
            tpm_response_raw=tpm_response_raw,
            endorsement_raw=endorsement_raw,
            custody_evidence_raw=custody_evidence_raw,
            ak_key_name=ak_key_name,
        )
        active = establish_installed_lppi_authority_key(accepted)
        _OWNED_PRE_ENROLLMENT[active] = key
        return active
    except BaseException:
        key.close()
        raise
