"""Core-owned authorized LOCK_SESSION and LOGOUT_SESSION transitions."""

from __future__ import annotations

from dataclasses import asdict, replace
from types import MappingProxyType
from typing import cast

from bot_core.persistence.fingerprints import canonical_json_sha256
from bot_core.security.authentication import (
    AuthenticationError,
    AuthorizationRequest,
    session_mutation_fingerprint,
)
from bot_core.security.authorization import AuthorizationAuthority, AuthorizationError
from bot_core.security.current_projection_authority import _accept_session_projection
from bot_core.security.initial_security import InitialSecurityError, SessionSecurityState


def _session_fingerprint(value: SessionSecurityState) -> str:
    return cast(
        str,
        canonical_json_sha256(
            {
                key: item
                for key, item in asdict(value).items()
                if key != "content_fingerprint_sha256"
            }
        ),
    )


class SessionSecurityTransitionAuthority:
    """Atomically publishes ordinary session lock and logout revisions."""

    _TARGET_STATES = {
        "LOCK_SESSION": "LOCKED",
        "LOGOUT_SESSION": "LOGGED_OUT",
    }

    def __init__(self, authorization: AuthorizationAuthority) -> None:
        self._authorization = authorization
        self._authentication = authorization._authentication  # noqa: SLF001
        self._state = authorization._state  # noqa: SLF001 - exact shared Core plane

    def transition_session(self, proof: object, request: object, now_utc: object) -> str:
        """Authorize and publish one exact ordinary session-state transition."""
        if not isinstance(request, AuthorizationRequest):
            return "AUTHORIZATION_DENIED"
        try:
            self._authentication._validate_request_structure(request)  # noqa: SLF001
        except AuthenticationError:
            return "AUTHORIZATION_DENIED"
        target = self._TARGET_STATES.get(request.operation)
        if target is None:
            return "AUTHORIZATION_DENIED"

        with self._state.lock:
            before = self._state.snapshot
            if not isinstance(before.accepted_sessions, MappingProxyType) or not isinstance(
                before.current_sessions, MappingProxyType
            ):
                return "CONTRACT_INCONSISTENT"
            scope = (request.account_id, request.operator_id, request.device_installation_id)
            try:
                old = cast(
                    SessionSecurityState,
                    self._authentication._resolve_current_projection(  # noqa: SLF001
                        before.accepted_sessions,
                        before.current_sessions,
                        scope,
                        SessionSecurityState,
                        scope,
                        lambda value: (
                            value.account_id,
                            value.operator_id,
                            value.device_installation_id,
                        ),
                        self._authentication._session_intrinsically_valid,  # noqa: SLF001
                        "AUTHORIZATION_DENIED",
                    ),
                )
            except AuthenticationError:
                return "AUTHORIZATION_DENIED"

            next_generation = old.session_generation + 1
            if request.mutation_fingerprint_sha256 != session_mutation_fingerprint(
                request, target, old.session_generation, next_generation
            ):
                return "AUTHORIZATION_DENIED"
            try:
                authorized = self._authorization.authorize(proof, request, now_utc)
            except AuthorizationError:
                return "AUTHORIZATION_DENIED"
            if authorized != "AUTHORIZED":
                return "AUTHORIZATION_DENIED"

            post = SessionSecurityState(
                old.account_id,
                old.operator_id,
                old.device_installation_id,
                old.runtime_session_id,
                target,
                next_generation,
                old.security_generation,
                "",
            )
            post = replace(post, content_fingerprint_sha256=_session_fingerprint(post))
            if not self._authentication._session_intrinsically_valid(post):  # noqa: SLF001
                return "CONTRACT_INCONSISTENT"
            if (
                post.session_generation <= old.session_generation
                or post.session_generation != old.session_generation + 1
                or post.security_generation != old.security_generation
            ):
                return "CONTRACT_INCONSISTENT"
            try:
                accepted = _accept_session_projection(self._state, post)
            except InitialSecurityError:
                return "CONTRACT_INCONSISTENT"
            if not accepted:
                return "CONTRACT_INCONSISTENT"
            return cast(str, post.state)
