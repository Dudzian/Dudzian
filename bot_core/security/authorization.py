"""Pure Core-owned M0.10 operation authorization decision authority."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import NoReturn, cast

from bot_core.persistence.fingerprints import canonical_json_sha256
from bot_core.security.authentication import (
    AuthenticationAuthority,
    AuthenticationError,
    AuthenticationProof,
    AuthorizationRequest,
    DownstreamOperationDefinition,
    OPERATION_OWNERSHIP,
    OPERATION_POLICY_REGISTRY,
    _parse_utc,
    _valid_utc,
    canonical_scope_fingerprint,
    downstream_mutation_fingerprint,
    downstream_scope_fingerprint,
)
from bot_core.security.initial_security import (
    DeviceTrustProjection,
    InitialSecurityAuthoritySnapshot,
    OperatorIdentitySecurityProjection,
    PinVerifierRecord,
    SessionSecurityState,
)


class AuthorizationError(RuntimeError):
    """Controlled fail-closed authorization decision error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _deny(reason: str) -> NoReturn:
    raise AuthorizationError(reason)


@dataclass(frozen=True, slots=True)
class OperationEntitlementProjection:
    account_id: str
    operator_id: str
    operation: str
    environment: str
    authorization_scope: str
    entitlement_revision: int
    security_generation: int
    content_fingerprint_sha256: str


def operation_entitlement_fingerprint(entitlement: OperationEntitlementProjection) -> str:
    """Return the canonical content hash; the hash establishes integrity, not authority."""
    return cast(
        str,
        canonical_json_sha256(
            {
                key: value
                for key, value in asdict(entitlement).items()
                if key != "content_fingerprint_sha256"
            }
        ),
    )


def _valid_entitlement(entitlement: object, policy: object = None) -> bool:
    if not isinstance(entitlement, OperationEntitlementProjection):
        return False
    try:
        policy = policy or OPERATION_POLICY_REGISTRY.get(entitlement.operation)
        return bool(
            AuthenticationAuthority._canonical_id(entitlement.account_id, "acct_")
            and AuthenticationAuthority._canonical_id(entitlement.operator_id, "op_")
            and policy is not None
            and entitlement.environment in {"PAPER", "TESTNET", "LIVE"}
            and entitlement.environment in policy.environments
            and entitlement.authorization_scope == policy.authorization_scope
            and isinstance(entitlement.entitlement_revision, int)
            and not isinstance(entitlement.entitlement_revision, bool)
            and entitlement.entitlement_revision >= 1
            and isinstance(entitlement.security_generation, int)
            and not isinstance(entitlement.security_generation, bool)
            and entitlement.security_generation >= 1
            and isinstance(entitlement.content_fingerprint_sha256, str)
            and len(entitlement.content_fingerprint_sha256) == 64
            and all(char in "0123456789abcdef" for char in entitlement.content_fingerprint_sha256)
            and operation_entitlement_fingerprint(entitlement)
            == entitlement.content_fingerprint_sha256
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _seed_trusted_operation_entitlement(
    authority: AuthorizationAuthority,
    entitlement: OperationEntitlementProjection,
    *,
    current: bool = True,
) -> None:
    """Module-private harness for authority already accepted by an upstream Core owner."""
    if not isinstance(authority, AuthorizationAuthority):
        _deny("CONTRACT_INCONSISTENT")
    with authority._state.lock:  # noqa: SLF001
        try:
            policy, _, _ = authority._authentication._resolve_operation(  # noqa: SLF001
                authority._state.snapshot,
                entitlement.operation,  # noqa: SLF001
            )
        except AuthenticationError:
            _deny("CONTRACT_INCONSISTENT")
    if not _valid_entitlement(entitlement, policy):
        _deny("CONTRACT_INCONSISTENT")
    scope = authority._entitlement_scope(entitlement)  # noqa: SLF001 - trusted owner boundary
    with authority._state.lock:  # noqa: SLF001 - trusted owner boundary
        before = authority._state.snapshot  # noqa: SLF001 - trusted owner boundary
        accepted = dict(before.accepted_operation_entitlements)
        designations = dict(before.current_operation_entitlements)
        accepted[entitlement.content_fingerprint_sha256] = entitlement
        if current:
            designations[scope] = entitlement.content_fingerprint_sha256
        authority._state.snapshot = replace(  # noqa: SLF001 - trusted owner boundary
            before,
            accepted_operation_entitlements=MappingProxyType(accepted),
            current_operation_entitlements=MappingProxyType(designations),
        )


class AuthorizationAuthority:
    """Makes a pure decision from genuine proof and fresh shared Core authority."""

    def __init__(self, authentication: AuthenticationAuthority) -> None:
        self._authentication = authentication
        self._state = authentication._state  # noqa: SLF001 - same exact semantic plane

    def _validate_authorization_inputs(
        self, proof: object, request: object, now_utc: object
    ) -> tuple[AuthenticationProof, AuthorizationRequest, datetime]:
        if not _valid_utc(now_utc):
            _deny("MALFORMED_UNTRUSTED_CONTEXT")
        if not isinstance(proof, AuthenticationProof):
            _deny("AUTHENTICATION_REQUIRED")
        if not isinstance(request, AuthorizationRequest):
            _deny("MALFORMED_UNTRUSTED_CONTEXT")
        try:
            self._authentication._validate_request_structure(request)  # noqa: SLF001
        except AuthenticationError as error:
            _deny(error.reason)
        return proof, request, cast(datetime, now_utc)

    def _authorize_against_snapshot_locked(
        self,
        proof: AuthenticationProof,
        request: AuthorizationRequest,
        now_utc: datetime,
    ) -> None:
        """Single authorization core; caller must hold the shared Core-state lock."""
        snapshot = self._state.snapshot
        try:
            policy, ownership, _ = (
                self._authentication._validate_operation_request_semantics_locked(  # noqa: SLF001
                    snapshot, request
                )
            )
        except AuthenticationError as error:
            _deny(error.reason)
        if request.environment not in policy.environments:
            _deny("AUTHORIZATION_DENIED")
        if (
            ownership == "M0.10_OWNED_TRANSITION"
            and request.scope_fingerprint_sha256 != canonical_scope_fingerprint(request)
        ):
            _deny("AUTHORIZATION_DENIED")
        try:
            accepted_proof = self._authentication._resolve_accepted_proof_membership(  # noqa: SLF001
                proof, snapshot
            )
        except AuthenticationError as error:
            _deny(error.reason)
        if not self._proof_matches_request(accepted_proof, request):
            _deny("AUTHORIZATION_DENIED")

        issued = self._proof_time(accepted_proof.issued_at_utc)
        expires = self._proof_time(accepted_proof.expires_at_utc)
        now = now_utc
        if expires <= issued:
            _deny("MALFORMED_UNTRUSTED_CONTEXT")
        if now < issued:
            _deny("AUTHENTICATION_REQUIRED")
        if now > expires or now - issued > timedelta(seconds=policy.freshness_seconds):
            _deny("PROOF_EXPIRED")

        expected_factors = {
            "PIN": ("PIN",),
            "BIOMETRIC": ("BIOMETRIC",),
            "PIN_AND_BIOMETRIC": ("PIN", "BIOMETRIC"),
        }.get(policy.factor_policy)
        if expected_factors is None:
            _deny("CONTRACT_INCONSISTENT")
        if accepted_proof.factor_set != expected_factors:
            _deny("PROOF_STALE")

        identity, device, pin, session = self._resolve_authorization_family(snapshot, request)
        if identity.state == "REVOKED":
            _deny("IDENTITY_REVOKED")
        if identity.state != "ACTIVE":
            _deny("CONTRACT_INCONSISTENT")
        if device.state != "TRUSTED":
            _deny("DEVICE_NOT_TRUSTED")
        if session.state != "UNLOCKED":
            _deny("PROOF_STALE")
        if (
            identity.identity_revision,
            device.trust_revision,
            pin.pin_revision,
            device.platform_enrollment_revision,
            identity.security_generation,
            session.session_generation,
        ) != (
            accepted_proof.identity_revision,
            accepted_proof.device_trust_revision,
            accepted_proof.pin_revision,
            accepted_proof.platform_enrollment_revision,
            accepted_proof.security_generation,
            accepted_proof.session_generation,
        ):
            _deny("PROOF_STALE")
        scope = (
            request.account_id,
            request.operator_id,
            request.operation,
            request.environment,
            policy.authorization_scope,
        )
        if not isinstance(
            snapshot.accepted_operation_entitlements, MappingProxyType
        ) or not isinstance(snapshot.current_operation_entitlements, MappingProxyType):
            _deny("CONTRACT_INCONSISTENT")
        fingerprint = snapshot.current_operation_entitlements.get(scope)
        if fingerprint is None:
            _deny("AUTHORIZATION_DENIED")
        if not isinstance(fingerprint, str):
            _deny("CONTRACT_INCONSISTENT")
        entitlement = snapshot.accepted_operation_entitlements.get(fingerprint)
        if entitlement is None:
            _deny("AUTHORIZATION_DENIED")
        if (
            not _valid_entitlement(entitlement, policy)
            or entitlement.content_fingerprint_sha256 != fingerprint
            or self._entitlement_scope(entitlement) != scope
        ):
            _deny("AUTHORIZATION_DENIED")
        current_generations = {
            identity.security_generation,
            device.security_generation,
            pin.security_generation,
            session.security_generation,
            entitlement.security_generation,
        }
        if len(current_generations) != 1:
            _deny("CONTRACT_INCONSISTENT")
        if current_generations != {accepted_proof.security_generation}:
            _deny("PROOF_STALE")

    def authorize(self, proof: object, request: object, now_utc: object) -> str:
        validated_proof, validated_request, validated_now = self._validate_authorization_inputs(
            proof, request, now_utc
        )
        with self._state.lock:
            self._authorize_against_snapshot_locked(
                validated_proof, validated_request, validated_now
            )
        return "AUTHORIZED"

    def validate_downstream_authorized_mutation(
        self,
        proof: object,
        request: object,
        now_utc: object,
        target_scope: object,
        mutation: object,
    ) -> str:
        """Return a point-in-time decision over one coherently locked Core snapshot."""
        if not isinstance(target_scope, dict) or not isinstance(mutation, dict):
            _deny("MALFORMED_UNTRUSTED_CONTEXT")
        validated_proof, validated_request, validated_now = self._validate_authorization_inputs(
            proof, request, now_utc
        )
        with self._state.lock:
            self._authorize_against_snapshot_locked(
                validated_proof, validated_request, validated_now
            )
            snapshot = self._state.snapshot
            fingerprint = snapshot.current_downstream_operation_definitions.get(
                validated_request.operation
            )
            definition = snapshot.accepted_downstream_operation_definitions.get(fingerprint)
            if not isinstance(definition, DownstreamOperationDefinition):
                _deny("CONTRACT_INCONSISTENT")
            if mutation.get("intent") != definition.declared_intent:
                _deny("AUTHORIZATION_DENIED")
            try:
                expected_scope = downstream_scope_fingerprint(
                    definition, validated_request, target_scope
                )
                expected_mutation = downstream_mutation_fingerprint(
                    definition, validated_request, target_scope, mutation
                )
            except AuthenticationError as error:
                _deny(error.reason)
            if (
                validated_request.scope_fingerprint_sha256 != expected_scope
                or validated_request.mutation_fingerprint_sha256 != expected_mutation
            ):
                _deny("AUTHORIZATION_DENIED")
        return "AUTHORIZED_MUTATION"

    def authorize_upstream_security_request(
        self,
        proof: object,
        request: object,
        now_utc: object,
    ) -> str:
        """Authorize, without executing, a request owned by an upstream authority."""
        if not isinstance(request, AuthorizationRequest):
            return "MALFORMED_UNTRUSTED_CONTEXT"
        try:
            self._authentication._validate_request_structure(request)  # noqa: SLF001
        except AuthenticationError as error:
            return cast(str, error.reason)
        if request.operation not in OPERATION_POLICY_REGISTRY:
            return "OPERATION_UNSUPPORTED"
        if (
            OPERATION_OWNERSHIP.get(request.operation)
            != "AUTHORIZED_SECURITY_REQUEST_TO_UPSTREAM_OWNER"
        ):
            return "OPERATION_UNSUPPORTED"
        try:
            decision = self.authorize(proof, request, now_utc)
        except AuthorizationError:
            return "AUTHORIZATION_DENIED"
        return "AUTHORIZED_SECURITY_REQUEST" if decision == "AUTHORIZED" else "AUTHORIZATION_DENIED"

    @staticmethod
    def _proof_time(value: str) -> datetime:
        try:
            parsed = _parse_utc(value)
        except AuthenticationError:
            _deny("MALFORMED_UNTRUSTED_CONTEXT")
        if parsed is None:
            _deny("MALFORMED_UNTRUSTED_CONTEXT")
        return cast(datetime, parsed)

    def _resolve_authorization_family(
        self,
        snapshot: InitialSecurityAuthoritySnapshot,
        request: AuthorizationRequest,
    ) -> tuple[
        OperatorIdentitySecurityProjection,
        DeviceTrustProjection,
        PinVerifierRecord,
        SessionSecurityState,
    ]:
        identity_scope = (request.account_id, request.operator_id)
        device_scope = (request.account_id, request.device_installation_id)
        family_scope = (*identity_scope, request.device_installation_id)
        try:
            identity = cast(
                OperatorIdentitySecurityProjection,
                self._authentication._resolve_current_projection(  # noqa: SLF001
                    snapshot.accepted_identities,
                    snapshot.current_identities,
                    identity_scope,
                    OperatorIdentitySecurityProjection,
                    identity_scope,
                    lambda value: (value.account_id, value.operator_id),
                    self._authentication._identity_intrinsically_valid,  # noqa: SLF001
                    "IDENTITY_REVOKED",
                ),
            )
            device = cast(
                DeviceTrustProjection,
                self._authentication._resolve_current_projection(  # noqa: SLF001
                    snapshot.accepted_devices,
                    snapshot.current_devices,
                    device_scope,
                    DeviceTrustProjection,
                    device_scope,
                    lambda value: (value.account_id, value.device_installation_id),
                    self._authentication._device_intrinsically_valid,  # noqa: SLF001
                    "DEVICE_NOT_TRUSTED",
                ),
            )
            pin = cast(
                PinVerifierRecord,
                self._authentication._resolve_current_projection(  # noqa: SLF001
                    snapshot.accepted_pins,
                    snapshot.current_pins,
                    family_scope,
                    PinVerifierRecord,
                    family_scope,
                    lambda value: (
                        value.account_id,
                        value.operator_id,
                        value.device_installation_id,
                    ),
                    lambda _value: True,
                    "PROOF_STALE",
                ),
            )
            session = cast(
                SessionSecurityState,
                self._authentication._resolve_current_projection(  # noqa: SLF001
                    snapshot.accepted_sessions,
                    snapshot.current_sessions,
                    family_scope,
                    SessionSecurityState,
                    family_scope,
                    lambda value: (
                        value.account_id,
                        value.operator_id,
                        value.device_installation_id,
                    ),
                    self._authentication._session_intrinsically_valid,  # noqa: SLF001
                    "PROOF_STALE",
                ),
            )
            self._authentication._validate_pin(pin)  # noqa: SLF001
        except AuthenticationError as error:
            _deny(error.reason)
        return identity, device, pin, session

    @staticmethod
    def _proof_matches_request(proof: AuthenticationProof, request: AuthorizationRequest) -> bool:
        return (
            proof.account_id,
            proof.operator_id,
            proof.device_installation_id,
            proof.environment,
            proof.operation,
            proof.scope_fingerprint_sha256,
            proof.mutation_fingerprint_sha256,
            proof.causation_id,
            proof.correlation_id,
        ) == (
            request.account_id,
            request.operator_id,
            request.device_installation_id,
            request.environment,
            request.operation,
            request.scope_fingerprint_sha256,
            request.mutation_fingerprint_sha256,
            request.causation_id,
            request.correlation_id,
        )

    @staticmethod
    def _entitlement_scope(
        entitlement: OperationEntitlementProjection,
    ) -> tuple[str, str, str, str, str]:
        return (
            entitlement.account_id,
            entitlement.operator_id,
            entitlement.operation,
            entitlement.environment,
            entitlement.authorization_scope,
        )


__all__ = [
    "AuthorizationAuthority",
    "AuthorizationError",
    "OperationEntitlementProjection",
    "operation_entitlement_fingerprint",
]
