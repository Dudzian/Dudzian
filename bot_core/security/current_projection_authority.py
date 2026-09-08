"""Core-internal monotonic publication of current security projections."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from types import MappingProxyType

from bot_core.security.initial_security import (
    DeviceTrustProjection,
    InitialSecurityAuthoritySnapshot,
    InitialSecurityError,
    InitialSecuritySemanticState,
    OperatorIdentitySecurityProjection,
    PinVerifierRecord,
    SessionSecurityState,
    _ID_RE,
    _SHA_RE,
    _make_fingerprint,
)


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _canonical_id(value: object, prefix: str) -> bool:
    return isinstance(value, str) and value.startswith(prefix) and bool(_ID_RE.fullmatch(value))


def _identity_valid(value: object) -> bool:
    return (
        isinstance(value, OperatorIdentitySecurityProjection)
        and _canonical_id(value.account_id, "acct_")
        and _canonical_id(value.operator_id, "op_")
        and value.state in {"ACTIVE", "REVOKED"}
        and _positive_int(value.identity_revision)
        and _positive_int(value.security_generation)
        and isinstance(value.content_fingerprint_sha256, str)
        and bool(_SHA_RE.fullmatch(value.content_fingerprint_sha256))
        and _make_fingerprint(value) == value.content_fingerprint_sha256
    )


def _device_valid(value: object) -> bool:
    return (
        isinstance(value, DeviceTrustProjection)
        and _canonical_id(value.account_id, "acct_")
        and _canonical_id(value.device_installation_id, "dev_")
        and value.state in {"ENROLLED_UNTRUSTED", "TRUSTED", "REVOKED", "REPLACED"}
        and _positive_int(value.trust_revision)
        and _positive_int(value.security_generation)
        and _positive_int(value.platform_enrollment_revision)
        and isinstance(value.content_fingerprint_sha256, str)
        and bool(_SHA_RE.fullmatch(value.content_fingerprint_sha256))
        and _make_fingerprint(value) == value.content_fingerprint_sha256
    )


def _canonical_utc(value: object) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return parsed.tzinfo == timezone.utc and parsed.isoformat().replace("+00:00", "Z") == value


def _pin_valid(value: object) -> bool:
    forbidden = (
        "api_key",
        "apikey",
        "secret",
        "password",
        "token",
        "private_key",
        "credential_value",
        "plaintext",
    )
    if not isinstance(value, PinVerifierRecord):
        return False
    locator = (
        value.salt_reference[len("secure-store://") :]
        if isinstance(value.salt_reference, str)
        and value.salt_reference.startswith("secure-store://")
        else ""
    )
    return (
        _canonical_id(value.account_id, "acct_")
        and _canonical_id(value.operator_id, "op_")
        and _canonical_id(value.device_installation_id, "dev_")
        and isinstance(value.algorithm_id, str)
        and bool(value.algorithm_id)
        and _positive_int(value.parameter_policy_version)
        and bool(locator)
        and not any(char.isspace() or char in "?#=" for char in locator)
        and not any(marker in locator.lower() for marker in forbidden)
        and isinstance(value.verifier, str)
        and bool(_SHA_RE.fullmatch(value.verifier))
        and _positive_int(value.pin_revision)
        and isinstance(value.failed_attempts, int)
        and not isinstance(value.failed_attempts, bool)
        and value.failed_attempts >= 0
        and (value.lockout_until_utc is None or _canonical_utc(value.lockout_until_utc))
        and _positive_int(value.security_generation)
        and isinstance(value.content_fingerprint_sha256, str)
        and bool(_SHA_RE.fullmatch(value.content_fingerprint_sha256))
        and _make_fingerprint(value) == value.content_fingerprint_sha256
    )


def _session_valid(value: object) -> bool:
    return (
        isinstance(value, SessionSecurityState)
        and _canonical_id(value.account_id, "acct_")
        and _canonical_id(value.operator_id, "op_")
        and _canonical_id(value.device_installation_id, "dev_")
        and _canonical_id(value.runtime_session_id, "run_")
        and value.state in {"LOCKED", "UNLOCKED", "LOGGED_OUT"}
        and _positive_int(value.session_generation)
        and _positive_int(value.security_generation)
        and isinstance(value.content_fingerprint_sha256, str)
        and bool(_SHA_RE.fullmatch(value.content_fingerprint_sha256))
        and _make_fingerprint(value) == value.content_fingerprint_sha256
    )


def _accept_identity_projection(
    state: InitialSecuritySemanticState, candidate: OperatorIdentitySecurityProjection
) -> bool:
    """Accept trusted Core identity state while preventing authority rollback."""
    if not _identity_valid(candidate):
        return False
    scope = (candidate.account_id, candidate.operator_id)
    with state.lock:
        before = state.snapshot
        fingerprint = before.current_identities.get(scope)
        current = None
        if fingerprint is not None:
            current = (
                before.accepted_identities.get(fingerprint)
                if isinstance(fingerprint, str)
                else None
            )
            if (
                not isinstance(current, OperatorIdentitySecurityProjection)
                or not _identity_valid(current)
                or current.content_fingerprint_sha256 != fingerprint
                or (current.account_id, current.operator_id) != scope
            ):
                raise InitialSecurityError("CONTRACT_INCONSISTENT")
        collision = before.accepted_identities.get(candidate.content_fingerprint_sha256)
        if collision is not None and collision != candidate:
            raise InitialSecurityError("CONTRACT_INCONSISTENT")
        if current == candidate:
            return True
        if current is not None and (
            current.state == "REVOKED"
            or candidate.identity_revision <= current.identity_revision
            or candidate.security_generation < current.security_generation
        ):
            return False
        accepted = dict(before.accepted_identities)
        current_map = dict(before.current_identities)
        accepted[candidate.content_fingerprint_sha256] = candidate
        current_map[scope] = candidate.content_fingerprint_sha256
        state.snapshot = replace(
            before,
            accepted_identities=MappingProxyType(accepted),
            current_identities=MappingProxyType(current_map),
        )
        return True


def _accept_device_projection(
    state: InitialSecuritySemanticState, candidate: DeviceTrustProjection
) -> bool:
    """Accept trusted Core device state while preventing authority rollback."""
    if not _device_valid(candidate):
        return False
    scope = (candidate.account_id, candidate.device_installation_id)
    with state.lock:
        before = state.snapshot
        fingerprint = before.current_devices.get(scope)
        current = None
        if fingerprint is not None:
            current = (
                before.accepted_devices.get(fingerprint) if isinstance(fingerprint, str) else None
            )
            if (
                not isinstance(current, DeviceTrustProjection)
                or not _device_valid(current)
                or current.content_fingerprint_sha256 != fingerprint
                or (current.account_id, current.device_installation_id) != scope
            ):
                raise InitialSecurityError("CONTRACT_INCONSISTENT")
        collision = before.accepted_devices.get(candidate.content_fingerprint_sha256)
        if collision is not None and collision != candidate:
            raise InitialSecurityError("CONTRACT_INCONSISTENT")
        if current == candidate:
            return True
        if current is not None and (
            current.state in {"REVOKED", "REPLACED"}
            or candidate.trust_revision <= current.trust_revision
            or candidate.security_generation < current.security_generation
            or candidate.platform_enrollment_revision < current.platform_enrollment_revision
        ):
            return False
        accepted = dict(before.accepted_devices)
        current_map = dict(before.current_devices)
        accepted[candidate.content_fingerprint_sha256] = candidate
        current_map[scope] = candidate.content_fingerprint_sha256
        state.snapshot = replace(
            before,
            accepted_devices=MappingProxyType(accepted),
            current_devices=MappingProxyType(current_map),
        )
        return True


def _validate_pin_successor(
    snapshot: InitialSecurityAuthoritySnapshot, candidate: PinVerifierRecord
) -> bool:
    """Validate one PIN successor against the frozen current-authority rule."""
    if not _pin_valid(candidate):
        return False
    accepted = snapshot.accepted_pins
    current_map = snapshot.current_pins
    scope = (candidate.account_id, candidate.operator_id, candidate.device_installation_id)
    designated = scope in current_map
    fingerprint = current_map.get(scope)
    current = None
    if designated:
        current = accepted.get(fingerprint) if isinstance(fingerprint, str) else None
        if (
            not isinstance(current, PinVerifierRecord)
            or not _pin_valid(current)
            or current.content_fingerprint_sha256 != fingerprint
            or (current.account_id, current.operator_id, current.device_installation_id) != scope
        ):
            raise InitialSecurityError("CONTRACT_INCONSISTENT")
    collision = accepted.get(candidate.content_fingerprint_sha256)
    if collision is not None and collision != candidate:
        raise InitialSecurityError("CONTRACT_INCONSISTENT")
    if current is None or current == candidate:
        return True
    if candidate.pin_revision < current.pin_revision:
        return False
    immutable = (
        "account_id",
        "operator_id",
        "device_installation_id",
        "algorithm_id",
        "parameter_policy_version",
        "salt_reference",
        "verifier",
        "pin_revision",
        "security_generation",
    )
    return candidate.pin_revision != current.pin_revision or all(
        getattr(candidate, field) == getattr(current, field) for field in immutable
    )


def _accept_pin_projection(
    state: InitialSecuritySemanticState, candidate: PinVerifierRecord
) -> bool:
    """Accept trusted Core PIN state while preventing revision rollback."""
    with state.lock:
        before = state.snapshot
        if not _validate_pin_successor(before, candidate):
            return False
        scope = (candidate.account_id, candidate.operator_id, candidate.device_installation_id)
        if before.accepted_pins.get(before.current_pins.get(scope)) == candidate:
            return True
        accepted = dict(before.accepted_pins)
        current = dict(before.current_pins)
        accepted[candidate.content_fingerprint_sha256] = candidate
        current[scope] = candidate.content_fingerprint_sha256
        state.snapshot = replace(
            before, accepted_pins=MappingProxyType(accepted), current_pins=MappingProxyType(current)
        )
        return True


def _validate_session_successor(
    snapshot: InitialSecurityAuthoritySnapshot, candidate: SessionSecurityState
) -> bool:
    """Validate one session successor against the frozen generation rule."""
    if not _session_valid(candidate):
        return False
    accepted = snapshot.accepted_sessions
    current_map = snapshot.current_sessions
    scope = (candidate.account_id, candidate.operator_id, candidate.device_installation_id)
    designated = scope in current_map
    fingerprint = current_map.get(scope)
    current = None
    if designated:
        current = accepted.get(fingerprint) if isinstance(fingerprint, str) else None
        if (
            not isinstance(current, SessionSecurityState)
            or not _session_valid(current)
            or current.content_fingerprint_sha256 != fingerprint
            or (current.account_id, current.operator_id, current.device_installation_id) != scope
        ):
            raise InitialSecurityError("CONTRACT_INCONSISTENT")
    collision = accepted.get(candidate.content_fingerprint_sha256)
    if collision is not None and collision != candidate:
        raise InitialSecurityError("CONTRACT_INCONSISTENT")
    return (
        current is None
        or current == candidate
        or candidate.session_generation > current.session_generation
    )


def _accept_session_projection(
    state: InitialSecuritySemanticState, candidate: SessionSecurityState
) -> bool:
    """Accept trusted Core session state while preventing generation rollback."""
    with state.lock:
        before = state.snapshot
        if not _validate_session_successor(before, candidate):
            return False
        scope = (candidate.account_id, candidate.operator_id, candidate.device_installation_id)
        if before.accepted_sessions.get(before.current_sessions.get(scope)) == candidate:
            return True
        accepted = dict(before.accepted_sessions)
        current = dict(before.current_sessions)
        accepted[candidate.content_fingerprint_sha256] = candidate
        current[scope] = candidate.content_fingerprint_sha256
        state.snapshot = replace(
            before,
            accepted_sessions=MappingProxyType(accepted),
            current_sessions=MappingProxyType(current),
        )
        return True
