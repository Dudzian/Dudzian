"""Core-internal monotonic publication of current identity and device projections."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

from bot_core.security.initial_security import (
    DeviceTrustProjection,
    InitialSecurityError,
    InitialSecuritySemanticState,
    OperatorIdentitySecurityProjection,
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
