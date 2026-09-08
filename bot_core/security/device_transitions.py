"""Core-owned authority for the frozen M0.10 device trust transition graph."""

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
    device_mutation_fingerprint,
)
from bot_core.security.authorization import AuthorizationAuthority, AuthorizationError
from bot_core.security.initial_security import DeviceTrustProjection


def _device_fingerprint(value: DeviceTrustProjection) -> str:
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


class DeviceTrustTransitionAuthority:
    """Publishes only TRUST_DEVICE and REVOKE_DEVICE transitions."""

    def __init__(self, authorization: AuthorizationAuthority) -> None:
        self._authorization = authorization
        self._authentication = authorization._authentication  # noqa: SLF001
        self._state = authorization._state  # noqa: SLF001 - exact shared Core plane

    def transition_device(
        self,
        proof: object,
        request: object,
        now_utc: object,
        target_device_id: object,
    ) -> str:
        """Authorize and atomically publish one legal device transition."""
        if not isinstance(request, AuthorizationRequest):
            return "AUTHORIZATION_DENIED"
        try:
            self._authentication._validate_request_structure(request)  # noqa: SLF001
        except AuthenticationError:
            return "AUTHORIZATION_DENIED"
        if request.operation not in {"TRUST_DEVICE", "REVOKE_DEVICE"} or not (
            AuthenticationAuthority._canonical_id(target_device_id, "dev_")  # noqa: SLF001
        ):
            return "AUTHORIZATION_DENIED"

        target = cast(str, target_device_id)
        with self._state.lock:
            before = self._state.snapshot
            if not isinstance(before.accepted_devices, MappingProxyType) or not isinstance(
                before.current_devices, MappingProxyType
            ):
                return "CONTRACT_INCONSISTENT"
            scope = (request.account_id, target)
            current_fingerprint = before.current_devices.get(scope)
            old: DeviceTrustProjection | None = None
            if current_fingerprint is not None:
                try:
                    old = cast(
                        DeviceTrustProjection,
                        self._authentication._resolve_current_projection(  # noqa: SLF001
                            before.accepted_devices,
                            before.current_devices,
                            scope,
                            DeviceTrustProjection,
                            scope,
                            lambda value: (value.account_id, value.device_installation_id),
                            self._authentication._device_intrinsically_valid,  # noqa: SLF001
                            "CONTRACT_INCONSISTENT",
                        ),
                    )
                except AuthenticationError:
                    return "CONTRACT_INCONSISTENT"

            current_revision = 0 if old is None else old.trust_revision
            if request.mutation_fingerprint_sha256 != device_mutation_fingerprint(
                request, target, current_revision, current_revision + 1
            ):
                return "AUTHORIZATION_DENIED"
            try:
                authorized = self._authorization.authorize(proof, request, now_utc)
            except AuthorizationError:
                return "AUTHORIZATION_DENIED"
            if authorized != "AUTHORIZED" or not isinstance(proof, AuthenticationProof):
                return "AUTHORIZATION_DENIED"

            if request.operation == "TRUST_DEVICE":
                if old is not None and old.state != "ENROLLED_UNTRUSTED":
                    return "AUTHORIZATION_DENIED"
                generation = proof.security_generation
                enrollment_revision = 1 if old is None else old.platform_enrollment_revision
                state = "TRUSTED"
            else:
                if old is None or old.state != "TRUSTED":
                    return "DEVICE_NOT_TRUSTED"
                generation = old.security_generation
                enrollment_revision = old.platform_enrollment_revision
                state = "REVOKED"

            post = DeviceTrustProjection(
                request.account_id,
                target,
                state,
                current_revision + 1,
                generation,
                enrollment_revision,
                "",
            )
            post = replace(post, content_fingerprint_sha256=_device_fingerprint(post))
            if not self._authentication._device_intrinsically_valid(post):  # noqa: SLF001
                return "CONTRACT_INCONSISTENT"
            if old is not None and (
                post.trust_revision <= old.trust_revision
                or post.security_generation < old.security_generation
                or post.platform_enrollment_revision < old.platform_enrollment_revision
                or old.state in {"REVOKED", "REPLACED"}
            ):
                return "CONTRACT_INCONSISTENT"
            collision = before.accepted_devices.get(post.content_fingerprint_sha256)
            if collision is not None and collision != post:
                return "CONTRACT_INCONSISTENT"

            accepted = dict(before.accepted_devices)
            current = dict(before.current_devices)
            accepted[post.content_fingerprint_sha256] = post
            current[scope] = post.content_fingerprint_sha256
            self._state.snapshot = replace(
                before,
                accepted_devices=MappingProxyType(accepted),
                current_devices=MappingProxyType(current),
            )
            return cast(str, post.state)
