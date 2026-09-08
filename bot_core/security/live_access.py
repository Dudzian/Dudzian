"""Core-owned authority for frozen M0.10 live-access grant transitions."""

from __future__ import annotations

from dataclasses import asdict, replace
from types import MappingProxyType
from typing import cast

from bot_core.persistence.fingerprints import canonical_json_sha256
from bot_core.security.authentication import (
    AuthenticationAuthority,
    AuthenticationError,
    AuthenticationProof,
    AuthorizationRequest,
    OPERATION_POLICY_REGISTRY,
)
from bot_core.security.authorization import AuthorizationAuthority, AuthorizationError
from bot_core.security.initial_security import (
    DeviceTrustProjection,
    LiveAccessGrantSecurityProjection,
    OperatorIdentitySecurityProjection,
    _ID_RE,
)

_LIVE_TRANSITIONS = MappingProxyType(
    {
        "GRANT_LIVE_ACCESS": "ACTIVE",
        "SUSPEND_LIVE_ACCESS": "SUSPENDED",
        "REVOKE_LIVE_ACCESS": "REVOKED",
    }
)
_GRANT_STATES = frozenset(_LIVE_TRANSITIONS.values())


def live_access_grant_fingerprint(grant: LiveAccessGrantSecurityProjection) -> str:
    """Return the canonical integrity hash; integrity alone is not authority."""
    return cast(
        str,
        canonical_json_sha256(
            {
                key: value
                for key, value in asdict(grant).items()
                if key != "content_fingerprint_sha256"
            }
        ),
    )


def _canonical_id(value: object, prefix: str) -> bool:
    return isinstance(value, str) and value.startswith(prefix) and bool(_ID_RE.fullmatch(value))


def _canonical_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_live_access_grant(value: object) -> bool:
    if not isinstance(value, LiveAccessGrantSecurityProjection):
        return False
    try:
        return bool(
            _canonical_id(value.live_access_grant_id, "lgrant_")
            and _canonical_id(value.account_id, "acct_")
            and _canonical_id(value.operator_id, "op_")
            and _canonical_id(value.device_installation_id, "dev_")
            and _canonical_sha256(value.policy_scope_fingerprint_sha256)
            and value.state in _GRANT_STATES
            and isinstance(value.grant_revision, int)
            and not isinstance(value.grant_revision, bool)
            and value.grant_revision >= 1
            and isinstance(value.security_generation, int)
            and not isinstance(value.security_generation, bool)
            and value.security_generation >= 1
            and _canonical_sha256(value.content_fingerprint_sha256)
            and live_access_grant_fingerprint(value) == value.content_fingerprint_sha256
        )
    except (AttributeError, TypeError, ValueError):
        return False


def live_mutation_fingerprint(
    request: AuthorizationRequest,
    grant_id: str,
    target_state: str,
    policy_scope_fingerprint_sha256: str,
    current_revision: int,
    next_revision: int,
    security_generation: int,
) -> str:
    """Bind authorization to the exact live-grant revision transition."""
    policy = OPERATION_POLICY_REGISTRY[request.operation]
    return cast(
        str,
        canonical_json_sha256(
            [
                "M010-LIVE-GRANT",
                request.account_id,
                request.operator_id,
                request.device_installation_id,
                grant_id,
                request.operation,
                target_state,
                policy_scope_fingerprint_sha256,
                current_revision,
                next_revision,
                security_generation,
                policy.authorization_scope,
            ]
        ),
    )


class LiveAccessGrantAuthority:
    """Validate current grants and atomically publish legal live transitions."""

    def __init__(self, authorization: AuthorizationAuthority) -> None:
        self._authorization = authorization
        self._authentication = authorization._authentication  # noqa: SLF001
        self._state = authorization._state  # noqa: SLF001 - exact shared Core plane

    def transition_live_grant(
        self,
        proof: object,
        request: object,
        now_utc: object,
        grant_id: object,
        policy_scope_fingerprint_sha256: object,
    ) -> tuple[str, LiveAccessGrantSecurityProjection | None]:
        if not isinstance(request, AuthorizationRequest):
            return "AUTHORIZATION_DENIED", None
        try:
            self._authentication._validate_request_structure(request)  # noqa: SLF001
        except AuthenticationError:
            return "AUTHORIZATION_DENIED", None
        if (
            request.operation not in _LIVE_TRANSITIONS
            or not _canonical_id(grant_id, "lgrant_")
            or not _canonical_sha256(policy_scope_fingerprint_sha256)
        ):
            return "AUTHORIZATION_DENIED", None
        canonical_grant_id = cast(str, grant_id)
        policy_scope = cast(str, policy_scope_fingerprint_sha256)

        with self._state.lock:
            before = self._state.snapshot
            if not isinstance(
                before.accepted_live_access_grants, MappingProxyType
            ) or not isinstance(before.current_live_access_grants, MappingProxyType):
                return "CONTRACT_INCONSISTENT", None
            history = tuple(
                item
                for item in before.accepted_live_access_grants.values()
                if item.live_access_grant_id == canonical_grant_id
            )
            parents = (
                request.account_id,
                request.operator_id,
                request.device_installation_id,
                policy_scope,
            )
            if any(
                (
                    item.account_id,
                    item.operator_id,
                    item.device_installation_id,
                    item.policy_scope_fingerprint_sha256,
                )
                != parents
                for item in history
            ):
                return "AUTHORIZATION_DENIED", None

            scope = (request.account_id, request.device_installation_id, policy_scope)
            current_fingerprint = before.current_live_access_grants.get(scope)
            current = before.accepted_live_access_grants.get(current_fingerprint or "")
            if current_fingerprint is not None and (
                current is None
                or not _valid_live_access_grant(current)
                or current.content_fingerprint_sha256 != current_fingerprint
            ):
                return "CONTRACT_INCONSISTENT", None

            target_state = _LIVE_TRANSITIONS[request.operation]
            if request.operation == "GRANT_LIVE_ACCESS":
                if history or (current is not None and current.state != "REVOKED"):
                    return "AUTHORIZATION_DENIED", None
                current_revision = 0
            else:
                allowed = {
                    "SUSPEND_LIVE_ACCESS": {"ACTIVE"},
                    "REVOKE_LIVE_ACCESS": {"ACTIVE", "SUSPENDED"},
                }[request.operation]
                if (
                    current is None
                    or current.live_access_grant_id != canonical_grant_id
                    or current.state not in allowed
                ):
                    return "AUTHORIZATION_DENIED", None
                current_revision = current.grant_revision

            generation = proof.security_generation if isinstance(proof, AuthenticationProof) else 0
            expected = live_mutation_fingerprint(
                request,
                canonical_grant_id,
                target_state,
                policy_scope,
                current_revision,
                current_revision + 1,
                generation,
            )
            if request.mutation_fingerprint_sha256 != expected:
                return "AUTHORIZATION_DENIED", None
            try:
                authorized = self._authorization.authorize(proof, request, now_utc)
            except AuthorizationError:
                return "AUTHORIZATION_DENIED", None
            if authorized != "AUTHORIZED" or not isinstance(proof, AuthenticationProof):
                return "AUTHORIZATION_DENIED", None

            post = LiveAccessGrantSecurityProjection(
                canonical_grant_id,
                request.account_id,
                request.operator_id,
                request.device_installation_id,
                policy_scope,
                target_state,
                current_revision + 1,
                proof.security_generation,
                "",
            )
            post = replace(post, content_fingerprint_sha256=live_access_grant_fingerprint(post))
            if not _valid_live_access_grant(post):
                return "CONTRACT_INCONSISTENT", None
            collision = before.accepted_live_access_grants.get(post.content_fingerprint_sha256)
            if collision is not None and collision != post:
                return "CONTRACT_INCONSISTENT", None

            accepted = dict(before.accepted_live_access_grants)
            current_designations = dict(before.current_live_access_grants)
            accepted[post.content_fingerprint_sha256] = post
            current_designations[scope] = post.content_fingerprint_sha256
            self._state.snapshot = replace(
                before,
                accepted_live_access_grants=MappingProxyType(accepted),
                current_live_access_grants=MappingProxyType(current_designations),
            )
            return post.state, post

    def validate_live_grant(self, untrusted: object) -> str:
        if not _valid_live_access_grant(untrusted):
            return "AUTHORIZATION_DENIED"
        grant = cast(LiveAccessGrantSecurityProjection, untrusted)
        scope = (
            grant.account_id,
            grant.device_installation_id,
            grant.policy_scope_fingerprint_sha256,
        )
        with self._state.lock:
            snapshot = self._state.snapshot
            fingerprint = snapshot.current_live_access_grants.get(scope)
            current = snapshot.accepted_live_access_grants.get(fingerprint or "")
            if current != grant or fingerprint != grant.content_fingerprint_sha256:
                return "PROOF_STALE"
            try:
                identity = self._authentication._resolve_current_projection(  # noqa: SLF001
                    snapshot.accepted_identities,
                    snapshot.current_identities,
                    (grant.account_id, grant.operator_id),
                    OperatorIdentitySecurityProjection,
                    (grant.account_id, grant.operator_id),
                    lambda value: (value.account_id, value.operator_id),
                    self._authentication._identity_intrinsically_valid,  # noqa: SLF001
                    "AUTHORIZATION_DENIED",
                )
                device = self._authentication._resolve_current_projection(  # noqa: SLF001
                    snapshot.accepted_devices,
                    snapshot.current_devices,
                    (grant.account_id, grant.device_installation_id),
                    DeviceTrustProjection,
                    (grant.account_id, grant.device_installation_id),
                    lambda value: (value.account_id, value.device_installation_id),
                    self._authentication._device_intrinsically_valid,  # noqa: SLF001
                    "AUTHORIZATION_DENIED",
                )
            except (AuthenticationError, AttributeError):
                return "AUTHORIZATION_DENIED"
            if identity.state != "ACTIVE" or device.state != "TRUSTED":
                return "AUTHORIZATION_DENIED"
            if (
                identity.security_generation != grant.security_generation
                or device.security_generation != grant.security_generation
            ):
                return "PROOF_STALE"
            return "GRANT_ACTIVE" if grant.state == "ACTIVE" else "AUTHORIZATION_DENIED"


__all__ = [
    "LiveAccessGrantAuthority",
    "live_access_grant_fingerprint",
    "live_mutation_fingerprint",
]
