"""Core-owned authorized CHANGE_PIN and RESET_PIN transitions."""

from __future__ import annotations

from dataclasses import asdict, replace
from types import MappingProxyType
from typing import cast

from bot_core.persistence.fingerprints import canonical_json_sha256
from bot_core.security.authentication import (
    AuthenticationError,
    AuthorizationRequest,
    pin_mutation_fingerprint,
)
from bot_core.security.authorization import AuthorizationAuthority, AuthorizationError
from bot_core.security.initial_security import (
    PinVerifierFactory,
    PinVerifierMaterial,
    PinVerifierRecord,
)


def _pin_fingerprint(value: PinVerifierRecord) -> str:
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


class PinSecurityTransitionAuthority:
    """Atomically replaces an existing current PIN after exact authorization."""

    def __init__(
        self, authorization: AuthorizationAuthority, pin_verifier_factory: PinVerifierFactory
    ) -> None:
        self._authorization = authorization
        self._authentication = authorization._authentication  # noqa: SLF001
        self._state = authorization._state  # noqa: SLF001 - exact shared Core plane
        self._pin_factory = pin_verifier_factory

    def transition_pin(
        self,
        proof: object,
        request: object,
        now_utc: object,
        new_pin: object,
    ) -> str:
        """Authorize and publish one exact PIN revision transition."""
        if not isinstance(request, AuthorizationRequest):
            return "AUTHORIZATION_DENIED"
        try:
            self._authentication._validate_request_structure(request)  # noqa: SLF001
        except AuthenticationError:
            return "AUTHORIZATION_DENIED"
        if request.operation not in {"CHANGE_PIN", "RESET_PIN"} or not isinstance(new_pin, str):
            return "AUTHORIZATION_DENIED"

        with self._state.lock:
            before = self._state.snapshot
            if not isinstance(before.accepted_pins, MappingProxyType) or not isinstance(
                before.current_pins, MappingProxyType
            ):
                return "CONTRACT_INCONSISTENT"
            scope = (request.account_id, request.operator_id, request.device_installation_id)
            try:
                old = cast(
                    PinVerifierRecord,
                    self._authentication._resolve_current_projection(  # noqa: SLF001
                        before.accepted_pins,
                        before.current_pins,
                        scope,
                        PinVerifierRecord,
                        scope,
                        lambda value: (
                            value.account_id,
                            value.operator_id,
                            value.device_installation_id,
                        ),
                        lambda value: self._pin_valid(value),
                        "CONTRACT_INCONSISTENT",
                    ),
                )
            except AuthenticationError:
                return "AUTHORIZATION_DENIED"

            next_revision = old.pin_revision + 1
            if request.mutation_fingerprint_sha256 != pin_mutation_fingerprint(
                request, old.pin_revision, next_revision
            ):
                return "AUTHORIZATION_DENIED"
            try:
                authorized = self._authorization.authorize(proof, request, now_utc)
            except AuthorizationError:
                return "AUTHORIZATION_DENIED"
            if authorized != "AUTHORIZED":
                return "AUTHORIZATION_DENIED"

            try:
                material = self._pin_factory.create(new_pin)
            except Exception:
                # Dependency exceptions may contain the supplied PIN; discard their graph.
                return "PIN_VERIFIER_DEPENDENCY_FAILURE"
            if not self._material_has_safe_runtime_shape(material):
                return "PIN_VERIFIER_DEPENDENCY_FAILURE"
            material = cast(PinVerifierMaterial, material)
            post = PinVerifierRecord(
                old.account_id,
                old.operator_id,
                old.device_installation_id,
                material.algorithm_id,
                material.parameter_policy_version,
                material.salt_reference,
                material.verifier,
                next_revision,
                0,
                None,
                old.security_generation,
                "",
            )
            post = replace(post, content_fingerprint_sha256=_pin_fingerprint(post))
            if not self._pin_valid(post):
                return "PIN_VERIFIER_DEPENDENCY_FAILURE"
            if post.pin_revision != old.pin_revision + 1:
                return "CONTRACT_INCONSISTENT"
            collision = before.accepted_pins.get(post.content_fingerprint_sha256)
            if collision is not None and collision != post:
                return "CONTRACT_INCONSISTENT"

            accepted = dict(before.accepted_pins)
            current = dict(before.current_pins)
            accepted[post.content_fingerprint_sha256] = post
            current[scope] = post.content_fingerprint_sha256
            self._state.snapshot = replace(
                before,
                accepted_pins=MappingProxyType(accepted),
                current_pins=MappingProxyType(current),
            )
            return "PIN_CHANGED"

    @staticmethod
    def _material_has_safe_runtime_shape(material: object) -> bool:
        """Reject malformed dependency output before canonical JSON serialization."""
        return bool(
            isinstance(material, PinVerifierMaterial)
            and isinstance(material.algorithm_id, str)
            and isinstance(material.parameter_policy_version, int)
            and not isinstance(material.parameter_policy_version, bool)
            and isinstance(material.salt_reference, str)
            and isinstance(material.verifier, str)
        )

    def _pin_valid(self, value: PinVerifierRecord) -> bool:
        try:
            self._authentication._validate_pin(value)  # noqa: SLF001
        except AuthenticationError:
            return False
        return bool(_pin_fingerprint(value) == value.content_fingerprint_sha256)
