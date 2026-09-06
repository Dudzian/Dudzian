"""Core-owned M0.10 PIN authentication-proof semantic authority.

This slice authenticates PIN-only session-operation requests.  It deliberately
does not authorize those requests and does not execute session transitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import re
from types import MappingProxyType
from typing import Any, NoReturn, Protocol, cast

from bot_core.persistence.fingerprints import canonical_json_sha256
from bot_core.runtime.runtime_session import RuntimeSession
from bot_core.security.initial_security import (
    DeviceTrustProjection,
    InitialSecurityAuthority,
    InitialSecurityAuthoritySnapshot,
    OperatorIdentitySecurityProjection,
    PinVerifierRecord,
    SessionSecurityState,
)

_ID_RE = re.compile(
    r"^[a-z][a-z0-9]*_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_AUTHORITY_SOURCE = "CoreHost"
_MAX_FAILED_ATTEMPTS = 3
_LOCKOUT_SECONDS = 300


class AuthenticationError(RuntimeError):
    """Controlled fail-closed error whose graph never retains a raw PIN."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _deny(reason: str) -> NoReturn:
    raise AuthenticationError(reason)


@dataclass(frozen=True, slots=True)
class AuthorizationRequest:
    account_id: str
    operator_id: str
    device_installation_id: str
    environment: str
    operation: str
    scope_fingerprint_sha256: str
    mutation_fingerprint_sha256: str
    causation_id: str
    correlation_id: str


@dataclass(frozen=True, slots=True)
class AuthenticationProof:
    account_id: str
    operator_id: str
    device_installation_id: str
    factor_set: tuple[str, ...]
    issued_at_utc: str
    expires_at_utc: str
    identity_revision: int
    device_trust_revision: int
    pin_revision: int
    platform_enrollment_revision: int
    security_generation: int
    session_generation: int
    environment: str
    operation: str
    scope_fingerprint_sha256: str
    mutation_fingerprint_sha256: str
    causation_id: str
    correlation_id: str
    proof_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class CoreIssuedAuthenticationProofBinding:
    proof_fingerprint_sha256: str
    complete_proof_content_fingerprint_sha256: str
    authority_source: str
    account_id: str
    operator_id: str
    device_installation_id: str
    identity_revision: int
    device_trust_revision: int
    pin_revision: int
    platform_enrollment_revision: int
    security_generation: int
    session_generation: int


@dataclass(frozen=True, slots=True)
class OperationPolicy:
    factor_policy: str
    freshness_seconds: int
    authorization_scope: str
    environments: tuple[str, ...]


OPERATION_POLICY_REGISTRY = MappingProxyType(
    {
        "LOCK_SESSION": OperationPolicy("PIN", 60, "lock_session", ("PAPER", "TESTNET")),
        "LOGOUT_SESSION": OperationPolicy("PIN", 60, "logout_session", ("PAPER", "TESTNET")),
        # Registered non-P1A operations remain visible so they fail closed rather than downgrade.
        "UNLOCK_SESSION": OperationPolicy(
            "PIN_AND_BIOMETRIC", 60, "unlock_session", ("PAPER", "TESTNET")
        ),
        "TRUST_DEVICE": OperationPolicy(
            "PIN_AND_BIOMETRIC", 60, "trust_device", ("PAPER", "TESTNET")
        ),
    }
)


class PinVerifierComparator(Protocol):
    """Production KDF/secure-store comparison boundary; it grants no authority."""

    def compare(self, raw_pin: str, record: PinVerifierRecord) -> bool: ...


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _valid_utc(value: object) -> bool:
    return (
        isinstance(value, datetime)
        and value.tzinfo is not None
        and value.utcoffset() == timedelta(0)
    )


def _parse_utc(value: str | None) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.endswith("Z"):
        _deny("CONTRACT_INCONSISTENT")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        _deny("CONTRACT_INCONSISTENT")
    if parsed.tzinfo != timezone.utc:
        _deny("CONTRACT_INCONSISTENT")
    return parsed


def _valid_canonical_utc_text(value: object) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except (TypeError, ValueError):
        return False
    return parsed.tzinfo == timezone.utc and _utc_text(parsed) == value


def _fingerprint_without(value: Any, field: str) -> str:
    return cast(str, canonical_json_sha256({k: v for k, v in asdict(value).items() if k != field}))


def authentication_proof_fingerprint(proof: AuthenticationProof) -> str:
    return _fingerprint_without(proof, "proof_fingerprint_sha256")


def complete_authentication_proof_fingerprint(proof: AuthenticationProof) -> str:
    return cast(str, canonical_json_sha256(asdict(proof)))


def canonical_scope_fingerprint(request: AuthorizationRequest) -> str:
    policy = OPERATION_POLICY_REGISTRY.get(request.operation)
    scope = policy.authorization_scope if policy is not None else "unsupported"
    return cast(
        str,
        canonical_json_sha256(
            [
                "M010-SCOPE",
                request.account_id,
                request.operator_id,
                request.device_installation_id,
                request.environment,
                request.operation,
                scope,
            ]
        ),
    )


def session_mutation_fingerprint(
    request: AuthorizationRequest,
    target_state: str,
    current_generation: int,
    next_generation: int,
) -> str:
    policy = OPERATION_POLICY_REGISTRY[request.operation]
    return cast(
        str,
        canonical_json_sha256(
            [
                "M010-SESSION",
                request.account_id,
                request.operator_id,
                request.device_installation_id,
                request.operation,
                target_state,
                current_generation,
                next_generation,
                policy.authorization_scope,
            ]
        ),
    )


class AuthenticationAuthority:
    """Issues and resolves genuine proofs over the shared initial-security state."""

    def __init__(
        self,
        security: InitialSecurityAuthority,
        comparator: PinVerifierComparator,
    ) -> None:
        self._state = security._state  # noqa: SLF001 -- trusted owner over the exact shared plane
        self._comparator = comparator
        self._runtime_sessions = security._runtime_sessions  # noqa: SLF001

    @property
    def snapshot(self) -> InitialSecurityAuthoritySnapshot:
        return self._state.snapshot

    def issue_authentication_proof(
        self, request: object, raw_pin: object, now: object
    ) -> AuthenticationProof:
        if not _valid_utc(now):
            _deny("MALFORMED_UNTRUSTED_CONTEXT")
        if not isinstance(request, AuthorizationRequest):
            _deny("MALFORMED_UNTRUSTED_CONTEXT")
        self._validate_request_structure(request)
        policy = OPERATION_POLICY_REGISTRY.get(request.operation)
        if policy is None:
            _deny("OPERATION_UNSUPPORTED")
        if request.environment not in policy.environments:
            _deny("AUTHORIZATION_DENIED")
        if policy.factor_policy != "PIN" or request.operation not in {
            "LOCK_SESSION",
            "LOGOUT_SESSION",
        }:
            _deny("OPERATION_UNSUPPORTED")
        if not isinstance(raw_pin, str):
            _deny("MALFORMED_UNTRUSTED_CONTEXT")

        current_time = cast(datetime, now)
        with self._state.lock:
            before = self._state.snapshot
            identity, device, pin, session = self._resolve_current_family(before, request)
            runtime = self._resolve_runtime(request, session)
            expected_scope = canonical_scope_fingerprint(request)
            target_state = {
                "LOCK_SESSION": "LOCKED",
                "LOGOUT_SESSION": "LOGGED_OUT",
            }[request.operation]
            expected_mutation = session_mutation_fingerprint(
                request, target_state, session.session_generation, session.session_generation + 1
            )
            if (
                request.scope_fingerprint_sha256 != expected_scope
                or request.mutation_fingerprint_sha256 != expected_mutation
            ):
                _deny("AUTHORIZATION_DENIED")

            lockout = _parse_utc(pin.lockout_until_utc)
            if lockout is not None and current_time < lockout:
                _deny("PIN_LOCKED")
            dependency_failed = False
            try:
                matched = self._comparator.compare(raw_pin, pin)
            except Exception:
                # Leave the dependency exception's possibly sensitive graph behind.
                dependency_failed = True
                matched = False
            if dependency_failed:
                _deny("PIN_VERIFIER_DEPENDENCY_FAILURE")
            if type(matched) is not bool:
                _deny("PIN_VERIFIER_DEPENDENCY_FAILURE")
            if not matched:
                self._publish_pin_failure(before, pin, current_time)
                _deny(
                    "PIN_LOCKED"
                    if pin.failed_attempts + 1 >= _MAX_FAILED_ATTEMPTS
                    else "AUTHENTICATION_FAILED"
                )

            effective_pin = pin
            if pin.failed_attempts or pin.lockout_until_utc is not None:
                effective_pin = self._updated_pin(pin, 0, None)
            proof = self._build_proof(
                request, policy, current_time, identity, device, effective_pin, session
            )
            binding = self._binding(proof)
            self._final_runtime_fence(request, session, runtime)
            accepted_pins = dict(before.accepted_pins)
            current_pins = dict(before.current_pins)
            if effective_pin != pin:
                accepted_pins[effective_pin.content_fingerprint_sha256] = effective_pin
                current_pins[(pin.account_id, pin.operator_id, pin.device_installation_id)] = (
                    effective_pin.content_fingerprint_sha256
                )
            proofs = dict(before.accepted_authentication_proofs)
            bindings = dict(before.accepted_authentication_proof_bindings)
            proofs[proof.proof_fingerprint_sha256] = proof
            bindings[proof.proof_fingerprint_sha256] = binding
            self._state.snapshot = replace(
                before,
                accepted_pins=MappingProxyType(accepted_pins),
                current_pins=MappingProxyType(current_pins),
                accepted_authentication_proofs=MappingProxyType(proofs),
                accepted_authentication_proof_bindings=MappingProxyType(bindings),
            )
            return proof

    def resolve_accepted_proof(self, candidate: object) -> AuthenticationProof:
        if not isinstance(candidate, AuthenticationProof):
            _deny("AUTHENTICATION_REQUIRED")
        self._validate_authentication_proof_structure(candidate)
        recomputed = authentication_proof_fingerprint(candidate)
        if candidate.proof_fingerprint_sha256 != recomputed:
            _deny("AUTHENTICATION_REQUIRED")
        with self._state.lock:
            snapshot = self._state.snapshot
            accepted = snapshot.accepted_authentication_proofs.get(recomputed)
            binding = snapshot.accepted_authentication_proof_bindings.get(recomputed)
        if accepted != candidate or not isinstance(binding, CoreIssuedAuthenticationProofBinding):
            _deny("AUTHENTICATION_REQUIRED")
        if binding != self._binding(candidate):
            _deny("CONTRACT_INCONSISTENT")
        request = AuthorizationRequest(
            candidate.account_id,
            candidate.operator_id,
            candidate.device_installation_id,
            candidate.environment,
            candidate.operation,
            candidate.scope_fingerprint_sha256,
            candidate.mutation_fingerprint_sha256,
            candidate.causation_id,
            candidate.correlation_id,
        )
        identity, device, pin, session = self._resolve_current_family(snapshot, request)
        if (
            identity.identity_revision,
            device.trust_revision,
            pin.pin_revision,
            device.platform_enrollment_revision,
            identity.security_generation,
            session.session_generation,
        ) != (
            candidate.identity_revision,
            candidate.device_trust_revision,
            candidate.pin_revision,
            candidate.platform_enrollment_revision,
            candidate.security_generation,
            candidate.session_generation,
        ):
            _deny("PROOF_STALE")
        return candidate

    @staticmethod
    def _validate_authentication_proof_structure(proof: AuthenticationProof) -> None:
        positive_epochs = (
            proof.identity_revision,
            proof.device_trust_revision,
            proof.pin_revision,
            proof.platform_enrollment_revision,
            proof.security_generation,
            proof.session_generation,
        )
        if (
            not all(
                isinstance(value, str)
                for value in (
                    proof.account_id,
                    proof.operator_id,
                    proof.device_installation_id,
                    proof.environment,
                    proof.operation,
                    proof.scope_fingerprint_sha256,
                    proof.mutation_fingerprint_sha256,
                    proof.causation_id,
                    proof.correlation_id,
                    proof.proof_fingerprint_sha256,
                )
            )
            or not _ID_RE.fullmatch(proof.account_id)
            or not proof.account_id.startswith("acct_")
            or not _ID_RE.fullmatch(proof.operator_id)
            or not proof.operator_id.startswith("op_")
            or not _ID_RE.fullmatch(proof.device_installation_id)
            or not proof.device_installation_id.startswith("dev_")
            or not isinstance(proof.factor_set, tuple)
            or proof.factor_set not in (("PIN",), ("BIOMETRIC",), ("PIN", "BIOMETRIC"))
            or not _valid_canonical_utc_text(proof.issued_at_utc)
            or not _valid_canonical_utc_text(proof.expires_at_utc)
            or not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1
                for value in positive_epochs
            )
            or proof.environment not in {"PAPER", "TESTNET", "LIVE"}
            or proof.operation not in OPERATION_POLICY_REGISTRY
            or not _SHA_RE.fullmatch(proof.scope_fingerprint_sha256)
            or not _SHA_RE.fullmatch(proof.mutation_fingerprint_sha256)
            or not _SHA_RE.fullmatch(proof.proof_fingerprint_sha256)
            or not proof.causation_id
            or not proof.correlation_id
        ):
            _deny("AUTHENTICATION_REQUIRED")

    def _resolve_runtime(
        self, request: AuthorizationRequest, session: SessionSecurityState
    ) -> RuntimeSession:
        dependency_failed = False
        try:
            runtime = self._runtime_sessions.resolve_current(
                request.account_id, request.device_installation_id
            )
        except Exception:
            dependency_failed = True
            runtime = None
        if dependency_failed or not self._runtime_matches(runtime, request, session):
            _deny("RUNTIME_SESSION_AUTHORITY_DENIED")
        return cast(RuntimeSession, runtime)

    @staticmethod
    def _runtime_matches(
        runtime: object, request: AuthorizationRequest, session: SessionSecurityState
    ) -> bool:
        return (
            isinstance(runtime, RuntimeSession)
            and type(runtime) is RuntimeSession
            and not runtime.closed
            and runtime.device_installation_id == request.device_installation_id
            and runtime.runtime_session_id == session.runtime_session_id
        )

    def _final_runtime_fence(
        self,
        request: AuthorizationRequest,
        session: SessionSecurityState,
        originally_resolved: RuntimeSession,
    ) -> None:
        dependency_failed = False
        try:
            current = self._runtime_sessions.resolve_current(
                request.account_id, request.device_installation_id
            )
        except Exception:
            dependency_failed = True
            current = None
        if (
            dependency_failed
            or current is not originally_resolved
            or not self._runtime_matches(originally_resolved, request, session)
            or not self._runtime_matches(current, request, session)
        ):
            _deny("RUNTIME_SESSION_AUTHORITY_DENIED")

    @staticmethod
    def _validate_request_structure(request: AuthorizationRequest) -> None:
        if (
            not all(
                isinstance(value, str)
                for value in (
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
            )
            or not _ID_RE.fullmatch(request.account_id)
            or not request.account_id.startswith("acct_")
            or not _ID_RE.fullmatch(request.operator_id)
            or not request.operator_id.startswith("op_")
            or not _ID_RE.fullmatch(request.device_installation_id)
            or not request.device_installation_id.startswith("dev_")
            or request.environment not in {"PAPER", "TESTNET", "LIVE"}
            or not _SHA_RE.fullmatch(request.scope_fingerprint_sha256)
            or not _SHA_RE.fullmatch(request.mutation_fingerprint_sha256)
            or not isinstance(request.causation_id, str)
            or not request.causation_id
            or not isinstance(request.correlation_id, str)
            or not request.correlation_id
        ):
            _deny("MALFORMED_UNTRUSTED_CONTEXT")

    def _resolve_current_family(
        self, snapshot: InitialSecurityAuthoritySnapshot, request: AuthorizationRequest
    ) -> tuple[
        OperatorIdentitySecurityProjection,
        DeviceTrustProjection,
        PinVerifierRecord,
        SessionSecurityState,
    ]:
        scope_i = (request.account_id, request.operator_id)
        scope_d = (request.account_id, request.device_installation_id)
        scope_p = (*scope_i, request.device_installation_id)

        def resolve(
            accepted: object, current: object, scope: object, expected_type: type[object]
        ) -> object:
            if not isinstance(accepted, MappingProxyType) or not isinstance(
                current, MappingProxyType
            ):
                _deny("CONTRACT_INCONSISTENT")
            fingerprint = current.get(scope)
            value = accepted.get(fingerprint) if isinstance(fingerprint, str) else None
            if (
                not isinstance(value, expected_type)
                or _fingerprint_without(value, "content_fingerprint_sha256") != fingerprint
            ):
                _deny("CONTRACT_INCONSISTENT")
            return value

        identity = cast(
            OperatorIdentitySecurityProjection,
            resolve(
                snapshot.accepted_identities,
                snapshot.current_identities,
                scope_i,
                OperatorIdentitySecurityProjection,
            ),
        )
        device = cast(
            DeviceTrustProjection,
            resolve(
                snapshot.accepted_devices,
                snapshot.current_devices,
                scope_d,
                DeviceTrustProjection,
            ),
        )
        pin = cast(
            PinVerifierRecord,
            resolve(snapshot.accepted_pins, snapshot.current_pins, scope_p, PinVerifierRecord),
        )
        session = cast(
            SessionSecurityState,
            resolve(
                snapshot.accepted_sessions,
                snapshot.current_sessions,
                scope_p,
                SessionSecurityState,
            ),
        )
        if not (
            isinstance(identity, OperatorIdentitySecurityProjection)
            and isinstance(device, DeviceTrustProjection)
            and isinstance(pin, PinVerifierRecord)
            and isinstance(session, SessionSecurityState)
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1
                for value in (
                    identity.identity_revision,
                    identity.security_generation,
                    device.trust_revision,
                    device.platform_enrollment_revision,
                    device.security_generation,
                    session.session_generation,
                    session.security_generation,
                )
            )
        ):
            _deny("CONTRACT_INCONSISTENT")
        if (
            (identity.account_id, identity.operator_id) != scope_i
            or (device.account_id, device.device_installation_id) != scope_d
            or (pin.account_id, pin.operator_id, pin.device_installation_id) != scope_p
            or (session.account_id, session.operator_id, session.device_installation_id) != scope_p
        ):
            _deny("CONTRACT_INCONSISTENT")
        if identity.state != "ACTIVE":
            _deny("IDENTITY_INVALID")
        if device.state != "TRUSTED":
            _deny("DEVICE_NOT_TRUSTED")
        if session.state != "UNLOCKED":
            _deny("AUTHENTICATION_REQUIRED")
        self._validate_pin(pin)
        if (
            len(
                {
                    identity.security_generation,
                    device.security_generation,
                    pin.security_generation,
                    session.security_generation,
                }
            )
            != 1
        ):
            _deny("CONTRACT_INCONSISTENT")
        return identity, device, pin, session

    @staticmethod
    def _validate_pin(pin: PinVerifierRecord) -> None:
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
        locator = (
            pin.salt_reference[len("secure-store://") :]
            if isinstance(pin.salt_reference, str)
            and pin.salt_reference.startswith("secure-store://")
            else ""
        )
        if (
            not isinstance(pin.algorithm_id, str)
            or not pin.algorithm_id
            or isinstance(pin.parameter_policy_version, bool)
            or not isinstance(pin.parameter_policy_version, int)
            or pin.parameter_policy_version < 1
            or not isinstance(pin.salt_reference, str)
            or not locator
            or any(char.isspace() or char in "?#=" for char in locator)
            or any(marker in locator.lower() for marker in forbidden)
            or not isinstance(pin.verifier, str)
            or not _SHA_RE.fullmatch(pin.verifier)
            or not isinstance(pin.pin_revision, int)
            or isinstance(pin.pin_revision, bool)
            or pin.pin_revision < 1
            or not isinstance(pin.failed_attempts, int)
            or isinstance(pin.failed_attempts, bool)
            or pin.failed_attempts < 0
            or not isinstance(pin.security_generation, int)
            or isinstance(pin.security_generation, bool)
            or pin.security_generation < 1
        ):
            _deny("CONTRACT_INCONSISTENT")
        _parse_utc(pin.lockout_until_utc)

    @staticmethod
    def _updated_pin(
        pin: PinVerifierRecord, failures: int, lockout: str | None
    ) -> PinVerifierRecord:
        updated = replace(
            pin, failed_attempts=failures, lockout_until_utc=lockout, content_fingerprint_sha256=""
        )
        return replace(
            updated,
            content_fingerprint_sha256=_fingerprint_without(updated, "content_fingerprint_sha256"),
        )

    def _publish_pin_failure(
        self, before: InitialSecurityAuthoritySnapshot, pin: PinVerifierRecord, now: datetime
    ) -> None:
        failures = pin.failed_attempts + 1
        lockout = (
            _utc_text(now + timedelta(seconds=_LOCKOUT_SECONDS))
            if failures >= _MAX_FAILED_ATTEMPTS
            else None
        )
        updated = self._updated_pin(pin, failures, lockout)
        accepted = dict(before.accepted_pins)
        current = dict(before.current_pins)
        accepted[updated.content_fingerprint_sha256] = updated
        current[(pin.account_id, pin.operator_id, pin.device_installation_id)] = (
            updated.content_fingerprint_sha256
        )
        self._state.snapshot = replace(
            before, accepted_pins=MappingProxyType(accepted), current_pins=MappingProxyType(current)
        )

    @staticmethod
    def _build_proof(
        request: AuthorizationRequest,
        policy: OperationPolicy,
        now: datetime,
        identity: OperatorIdentitySecurityProjection,
        device: DeviceTrustProjection,
        pin: PinVerifierRecord,
        session: SessionSecurityState,
    ) -> AuthenticationProof:
        values = dict(
            account_id=request.account_id,
            operator_id=request.operator_id,
            device_installation_id=request.device_installation_id,
            factor_set=("PIN",),
            issued_at_utc=_utc_text(now),
            expires_at_utc=_utc_text(now + timedelta(seconds=policy.freshness_seconds)),
            identity_revision=identity.identity_revision,
            device_trust_revision=device.trust_revision,
            pin_revision=pin.pin_revision,
            platform_enrollment_revision=device.platform_enrollment_revision,
            security_generation=identity.security_generation,
            session_generation=session.session_generation,
            environment=request.environment,
            operation=request.operation,
            scope_fingerprint_sha256=request.scope_fingerprint_sha256,
            mutation_fingerprint_sha256=request.mutation_fingerprint_sha256,
            causation_id=request.causation_id,
            correlation_id=request.correlation_id,
        )
        return AuthenticationProof(
            request.account_id,
            request.operator_id,
            request.device_installation_id,
            ("PIN",),
            _utc_text(now),
            _utc_text(now + timedelta(seconds=policy.freshness_seconds)),
            identity.identity_revision,
            device.trust_revision,
            pin.pin_revision,
            device.platform_enrollment_revision,
            identity.security_generation,
            session.session_generation,
            request.environment,
            request.operation,
            request.scope_fingerprint_sha256,
            request.mutation_fingerprint_sha256,
            request.causation_id,
            request.correlation_id,
            cast(str, canonical_json_sha256(values)),
        )

    @staticmethod
    def _binding(proof: AuthenticationProof) -> CoreIssuedAuthenticationProofBinding:
        return CoreIssuedAuthenticationProofBinding(
            proof.proof_fingerprint_sha256,
            complete_authentication_proof_fingerprint(proof),
            _AUTHORITY_SOURCE,
            proof.account_id,
            proof.operator_id,
            proof.device_installation_id,
            proof.identity_revision,
            proof.device_trust_revision,
            proof.pin_revision,
            proof.platform_enrollment_revision,
            proof.security_generation,
            proof.session_generation,
        )


__all__ = [
    "AuthenticationAuthority",
    "AuthenticationError",
    "AuthenticationProof",
    "AuthorizationRequest",
    "CoreIssuedAuthenticationProofBinding",
    "OPERATION_POLICY_REGISTRY",
    "PinVerifierComparator",
    "authentication_proof_fingerprint",
    "canonical_scope_fingerprint",
    "complete_authentication_proof_fingerprint",
    "session_mutation_fingerprint",
]
