"""Dedicated Core-owned fresh-factor UNLOCK_SESSION transition."""

from __future__ import annotations

from dataclasses import asdict, replace
from types import MappingProxyType
from typing import cast

from bot_core.persistence.fingerprints import canonical_json_sha256
from bot_core.security.authentication import (
    AuthenticationAuthority,
    AuthenticationError,
    AuthorizationRequest,
    OPERATION_POLICY_REGISTRY,
    canonical_scope_fingerprint,
    session_mutation_fingerprint,
    _valid_utc,
)
from bot_core.security.authorization import (
    AuthorizationAuthority,
    OperationEntitlementProjection,
    _valid_entitlement,
)
from bot_core.security.current_projection_authority import _accept_session_projection
from bot_core.security.initial_security import (
    DeviceTrustProjection,
    InitialSecurityError,
    OperatorIdentitySecurityProjection,
    PinVerifierRecord,
    SessionSecurityState,
)


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


class SessionUnlockAuthority:
    """Unlocks exactly one locked session using fresh PIN and biometric factors."""

    def __init__(self, authorization: AuthorizationAuthority) -> None:
        self._authorization = authorization
        self._authentication = cast(
            AuthenticationAuthority,
            authorization._authentication,  # noqa: SLF001
        )
        self._state = authorization._state  # noqa: SLF001 - exact shared Core plane

    def unlock_session(
        self, request: object, now_utc: object, raw_pin: object, assertion: object
    ) -> str:
        if not isinstance(request, AuthorizationRequest) or not _valid_utc(now_utc):
            return "AUTHORIZATION_DENIED"
        try:
            self._authentication._validate_request_structure(request)  # noqa: SLF001
        except AuthenticationError:
            return "AUTHORIZATION_DENIED"
        if request.operation != "UNLOCK_SESSION":
            return "AUTHORIZATION_DENIED"
        policy = OPERATION_POLICY_REGISTRY["UNLOCK_SESSION"]
        if (
            request.environment not in policy.environments
            or policy.factor_policy != "PIN_AND_BIOMETRIC"
            or request.scope_fingerprint_sha256 != canonical_scope_fingerprint(request)
        ):
            return "AUTHORIZATION_DENIED"

        with self._state.lock:
            before = self._state.snapshot
            family_scope = (
                request.account_id,
                request.operator_id,
                request.device_installation_id,
            )
            try:
                identity = cast(
                    OperatorIdentitySecurityProjection,
                    self._authentication._resolve_current_projection(  # noqa: SLF001
                        before.accepted_identities,
                        before.current_identities,
                        (request.account_id, request.operator_id),
                        OperatorIdentitySecurityProjection,
                        (request.account_id, request.operator_id),
                        lambda value: (value.account_id, value.operator_id),
                        self._authentication._identity_intrinsically_valid,  # noqa: SLF001
                        "AUTHORIZATION_DENIED",
                    ),
                )
                device = cast(
                    DeviceTrustProjection,
                    self._authentication._resolve_current_projection(  # noqa: SLF001
                        before.accepted_devices,
                        before.current_devices,
                        (request.account_id, request.device_installation_id),
                        DeviceTrustProjection,
                        (request.account_id, request.device_installation_id),
                        lambda value: (value.account_id, value.device_installation_id),
                        self._authentication._device_intrinsically_valid,  # noqa: SLF001
                        "AUTHORIZATION_DENIED",
                    ),
                )
                pin = cast(
                    PinVerifierRecord,
                    self._authentication._resolve_current_projection(  # noqa: SLF001
                        before.accepted_pins,
                        before.current_pins,
                        family_scope,
                        PinVerifierRecord,
                        family_scope,
                        lambda value: (
                            value.account_id,
                            value.operator_id,
                            value.device_installation_id,
                        ),
                        lambda _value: True,
                        "AUTHORIZATION_DENIED",
                    ),
                )
                self._authentication._validate_pin(pin)  # noqa: SLF001
                session = cast(
                    SessionSecurityState,
                    self._authentication._resolve_current_projection(  # noqa: SLF001
                        before.accepted_sessions,
                        before.current_sessions,
                        family_scope,
                        SessionSecurityState,
                        family_scope,
                        lambda value: (
                            value.account_id,
                            value.operator_id,
                            value.device_installation_id,
                        ),
                        self._authentication._session_intrinsically_valid,  # noqa: SLF001
                        "AUTHORIZATION_DENIED",
                    ),
                )
            except AuthenticationError as error:
                return (
                    "CONTRACT_INCONSISTENT"
                    if error.reason == "CONTRACT_INCONSISTENT"
                    else "AUTHORIZATION_DENIED"
                )
            if identity.state != "ACTIVE" or device.state != "TRUSTED" or session.state != "LOCKED":
                return "AUTHORIZATION_DENIED"

            entitlement_scope = (
                request.account_id,
                request.operator_id,
                request.operation,
                request.environment,
                policy.authorization_scope,
            )
            if not isinstance(
                before.accepted_operation_entitlements, MappingProxyType
            ) or not isinstance(before.current_operation_entitlements, MappingProxyType):
                return "CONTRACT_INCONSISTENT"
            entitlement_key = before.current_operation_entitlements.get(entitlement_scope)
            if entitlement_key is None:
                return "AUTHORIZATION_DENIED"
            if not isinstance(entitlement_key, str):
                return "CONTRACT_INCONSISTENT"
            entitlement = before.accepted_operation_entitlements.get(entitlement_key)
            if (
                not isinstance(entitlement, OperationEntitlementProjection)
                or not _valid_entitlement(entitlement)
                or entitlement.content_fingerprint_sha256 != entitlement_key
                or self._authorization._entitlement_scope(entitlement) != entitlement_scope  # noqa: SLF001
            ):
                return "AUTHORIZATION_DENIED"
            if (
                len(
                    {
                        identity.security_generation,
                        device.security_generation,
                        pin.security_generation,
                        session.security_generation,
                        entitlement.security_generation,
                    }
                )
                != 1
            ):
                return "CONTRACT_INCONSISTENT"
            if request.mutation_fingerprint_sha256 != session_mutation_fingerprint(
                request, "UNLOCKED", session.session_generation, session.session_generation + 1
            ):
                return "AUTHORIZATION_DENIED"

            pin_result = self._authentication.verify_current_pin(
                request.account_id,
                request.operator_id,
                request.device_installation_id,
                raw_pin,
                now_utc,
            )
            if pin_result == "PIN_VERIFIER_DEPENDENCY_FAILURE":
                return pin_result
            if pin_result != "PIN_ACCEPTED":
                return "AUTHENTICATION_FAILED"
            try:
                biometric = self._authentication.verify_platform_assertion(
                    assertion, request, now_utc
                )
            except AuthenticationError:
                return "AUTHENTICATION_FAILED"
            if biometric != "BIOMETRIC_ACCEPTED":
                return "AUTHENTICATION_FAILED"

            post = SessionSecurityState(
                session.account_id,
                session.operator_id,
                session.device_installation_id,
                session.runtime_session_id,
                "UNLOCKED",
                session.session_generation + 1,
                session.security_generation,
                "",
            )
            post = replace(post, content_fingerprint_sha256=_session_fingerprint(post))
            if not self._authentication._session_intrinsically_valid(post):  # noqa: SLF001
                return "CONTRACT_INCONSISTENT"
            if (
                post.session_generation != session.session_generation + 1
                or post.security_generation != session.security_generation
                or post.runtime_session_id != session.runtime_session_id
            ):
                return "CONTRACT_INCONSISTENT"

            try:
                accepted = _accept_session_projection(self._state, post)
            except InitialSecurityError:
                return "CONTRACT_INCONSISTENT"
            if not accepted:
                return "CONTRACT_INCONSISTENT"
            return "UNLOCKED"
