"""Core-owned M0.10 authentication semantic authority.

This slice issues PIN-only and combined PIN-plus-biometric proofs by consuming
pre-existing external-platform biometric assertion membership.  It does not
authorize requests or execute security transitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import re
from types import MappingProxyType
from typing import Any, Callable, NoReturn, Protocol, cast

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
class PlatformBiometricAssertion:
    account_id: str
    device_installation_id: str
    platform_authenticator_source: object
    platform_enrollment_revision: int
    challenge_fingerprint_sha256: str
    outcome: str
    verified_at_utc: str
    expires_at_utc: str
    assertion_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class CoreAcceptedPlatformBiometricAssertionBinding:
    assertion_fingerprint_sha256: str
    complete_assertion_content_fingerprint_sha256: str
    authority_source: str
    account_id: str
    device_installation_id: str
    platform_enrollment_revision: int
    challenge_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class OperationPolicy:
    factor_policy: str
    freshness_seconds: int
    authorization_scope: str
    environments: tuple[str, ...]


OPERATION_POLICY_REGISTRY = MappingProxyType(
    {
        "TRUST_DEVICE": OperationPolicy(
            "PIN_AND_BIOMETRIC", 60, "trust_device", ("PAPER", "TESTNET")
        ),
        "REVOKE_DEVICE": OperationPolicy(
            "PIN_AND_BIOMETRIC", 60, "revoke_device", ("PAPER", "TESTNET")
        ),
        "SETUP_PIN": OperationPolicy("PIN_AND_BIOMETRIC", 60, "setup_pin", ("PAPER", "TESTNET")),
        "CHANGE_PIN": OperationPolicy("PIN_AND_BIOMETRIC", 60, "change_pin", ("PAPER", "TESTNET")),
        "RESET_PIN": OperationPolicy("PIN_AND_BIOMETRIC", 60, "reset_pin", ("PAPER", "TESTNET")),
        "LOCK_SESSION": OperationPolicy("PIN", 60, "lock_session", ("PAPER", "TESTNET")),
        "UNLOCK_SESSION": OperationPolicy(
            "PIN_AND_BIOMETRIC", 60, "unlock_session", ("PAPER", "TESTNET")
        ),
        "LOGOUT_SESSION": OperationPolicy("PIN", 60, "logout_session", ("PAPER", "TESTNET")),
        "ROTATE_SECRET_REFERENCE": OperationPolicy(
            "PIN_AND_BIOMETRIC", 60, "rotate_secret_reference", ("PAPER", "TESTNET")
        ),
        "REBIND_SECRET_REFERENCE": OperationPolicy(
            "PIN_AND_BIOMETRIC", 60, "rebind_secret_reference", ("PAPER", "TESTNET")
        ),
        "ACTIVATE_CREDENTIAL_PROFILE": OperationPolicy(
            "PIN_AND_BIOMETRIC", 60, "activate_credential_profile", ("PAPER", "TESTNET")
        ),
        "DEACTIVATE_CREDENTIAL_PROFILE": OperationPolicy(
            "PIN_AND_BIOMETRIC", 60, "deactivate_credential_profile", ("PAPER", "TESTNET")
        ),
        "CHANGE_RISK_POLICY": OperationPolicy(
            "PIN_AND_BIOMETRIC", 60, "change_risk_policy", ("PAPER", "TESTNET")
        ),
        "CHANGE_KILL_SWITCH": OperationPolicy(
            "PIN_AND_BIOMETRIC", 60, "change_kill_switch", ("PAPER", "TESTNET")
        ),
        "CHANGE_PRODUCT_CAPABILITIES": OperationPolicy(
            "PIN_AND_BIOMETRIC", 60, "change_product_capabilities", ("PAPER", "TESTNET")
        ),
        "GRANT_LIVE_ACCESS": OperationPolicy(
            "PIN_AND_BIOMETRIC", 60, "grant_live_access", ("LIVE",)
        ),
        "SUSPEND_LIVE_ACCESS": OperationPolicy(
            "PIN_AND_BIOMETRIC", 60, "suspend_live_access", ("LIVE",)
        ),
        "REVOKE_LIVE_ACCESS": OperationPolicy(
            "PIN_AND_BIOMETRIC", 60, "revoke_live_access", ("LIVE",)
        ),
    }
)

OPERATION_OWNERSHIP = MappingProxyType(
    {
        "TRUST_DEVICE": "M0.10_OWNED_TRANSITION",
        "REVOKE_DEVICE": "M0.10_OWNED_TRANSITION",
        "SETUP_PIN": "AUTHORIZED_SECURITY_REQUEST_TO_UPSTREAM_OWNER",
        "CHANGE_PIN": "M0.10_OWNED_TRANSITION",
        "RESET_PIN": "M0.10_OWNED_TRANSITION",
        "LOCK_SESSION": "M0.10_OWNED_TRANSITION",
        "UNLOCK_SESSION": "M0.10_OWNED_TRANSITION",
        "LOGOUT_SESSION": "M0.10_OWNED_TRANSITION",
        "ROTATE_SECRET_REFERENCE": "AUTHORIZED_SECURITY_REQUEST_TO_UPSTREAM_OWNER",
        "REBIND_SECRET_REFERENCE": "AUTHORIZED_SECURITY_REQUEST_TO_UPSTREAM_OWNER",
        "ACTIVATE_CREDENTIAL_PROFILE": "AUTHORIZED_SECURITY_REQUEST_TO_UPSTREAM_OWNER",
        "DEACTIVATE_CREDENTIAL_PROFILE": "AUTHORIZED_SECURITY_REQUEST_TO_UPSTREAM_OWNER",
        "CHANGE_RISK_POLICY": "AUTHORIZED_SECURITY_REQUEST_TO_UPSTREAM_OWNER",
        "CHANGE_KILL_SWITCH": "AUTHORIZED_SECURITY_REQUEST_TO_UPSTREAM_OWNER",
        "CHANGE_PRODUCT_CAPABILITIES": "AUTHORIZED_SECURITY_REQUEST_TO_UPSTREAM_OWNER",
        "GRANT_LIVE_ACCESS": "M0.10_OWNED_TRANSITION",
        "SUSPEND_LIVE_ACCESS": "M0.10_OWNED_TRANSITION",
        "REVOKE_LIVE_ACCESS": "M0.10_OWNED_TRANSITION",
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


def platform_biometric_assertion_fingerprint(assertion: PlatformBiometricAssertion) -> str:
    return _fingerprint_without(assertion, "assertion_fingerprint_sha256")


def complete_platform_biometric_assertion_fingerprint(
    assertion: PlatformBiometricAssertion,
) -> str:
    return cast(str, canonical_json_sha256(asdict(assertion)))


def core_expected_biometric_challenge(
    request: AuthorizationRequest,
    platform_enrollment_revision: int,
    security_generation: int,
    session_generation: int,
) -> str:
    return cast(
        str,
        canonical_json_sha256(
            [
                "M010-BIOMETRIC-CHALLENGE",
                request.account_id,
                request.operator_id,
                request.device_installation_id,
                request.environment,
                request.operation,
                request.scope_fingerprint_sha256,
                request.mutation_fingerprint_sha256,
                request.causation_id,
                request.correlation_id,
                platform_enrollment_revision,
                security_generation,
                session_generation,
            ]
        ),
    )


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


def device_mutation_fingerprint(
    request: AuthorizationRequest,
    target_device_id: str,
    current_revision: int,
    next_revision: int,
) -> str:
    """Bind a device transition to its exact target and revision edge."""
    policy = OPERATION_POLICY_REGISTRY[request.operation]
    return cast(
        str,
        canonical_json_sha256(
            [
                "M010-DEVICE",
                request.account_id,
                request.operator_id,
                request.device_installation_id,
                request.operation,
                target_device_id,
                current_revision,
                next_revision,
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

    def derive_platform_biometric_challenge(self, request: object) -> str:
        """Derive the challenge solely from current Core-owned semantic authority."""
        if not isinstance(request, AuthorizationRequest):
            _deny("MALFORMED_UNTRUSTED_CONTEXT")
        self._validate_request_structure(request)
        if request.operation not in OPERATION_POLICY_REGISTRY:
            _deny("OPERATION_UNSUPPORTED")
        with self._state.lock:
            identity, device, session = self._resolve_biometric_context(
                self._state.snapshot, request
            )
            return core_expected_biometric_challenge(
                request,
                device.platform_enrollment_revision,
                identity.security_generation,
                session.session_generation,
            )

    def verify_platform_assertion(self, assertion: object, request: object, now_utc: object) -> str:
        """Consume pre-existing external-platform membership as biometric evidence."""
        if not _valid_utc(now_utc) or not isinstance(request, AuthorizationRequest):
            _deny("MALFORMED_UNTRUSTED_CONTEXT")
        self._validate_request_structure(request)
        if request.operation not in OPERATION_POLICY_REGISTRY:
            _deny("OPERATION_UNSUPPORTED")
        if not self._valid_assertion_shape(assertion):
            _deny("MALFORMED_UNTRUSTED_CONTEXT")
        candidate = cast(PlatformBiometricAssertion, assertion)
        current_time = cast(datetime, now_utc)
        with self._state.lock:
            snapshot = self._state.snapshot
            identity, device, session = self._resolve_biometric_context(snapshot, request)
            expected_challenge = core_expected_biometric_challenge(
                request,
                device.platform_enrollment_revision,
                identity.security_generation,
                session.session_generation,
            )
            try:
                exact = CoreAcceptedPlatformBiometricAssertionBinding(
                    candidate.assertion_fingerprint_sha256,
                    complete_platform_biometric_assertion_fingerprint(candidate),
                    "external_platform_authenticator",
                    candidate.account_id,
                    candidate.device_installation_id,
                    candidate.platform_enrollment_revision,
                    candidate.challenge_fingerprint_sha256,
                )
            except (TypeError, ValueError):
                _deny("MALFORMED_UNTRUSTED_CONTEXT")
            binding = snapshot.accepted_platform_biometric_assertion_bindings.get(
                candidate.assertion_fingerprint_sha256
            )
            if binding != exact:
                _deny("AUTHENTICATION_FAILED")
            if candidate.outcome == "UNAVAILABLE":
                _deny("FACTOR_UNAVAILABLE")
            if candidate.outcome != "SUCCESS":
                _deny("AUTHENTICATION_FAILED")
            if (
                candidate.account_id,
                candidate.device_installation_id,
                candidate.platform_enrollment_revision,
                candidate.challenge_fingerprint_sha256,
            ) != (
                request.account_id,
                request.device_installation_id,
                device.platform_enrollment_revision,
                expected_challenge,
            ):
                _deny("AUTHENTICATION_FAILED")
            start = _parse_utc(candidate.verified_at_utc)
            end = _parse_utc(candidate.expires_at_utc)
            if start is None or end is None or not start <= current_time <= end:
                _deny("AUTHENTICATION_FAILED")
            return "BIOMETRIC_ACCEPTED"

    @staticmethod
    def _valid_assertion_shape(assertion: object) -> bool:
        if not isinstance(assertion, PlatformBiometricAssertion):
            return False
        try:
            return bool(
                isinstance(assertion.account_id, str)
                and assertion.account_id.startswith("acct_")
                and _ID_RE.fullmatch(assertion.account_id)
                and isinstance(assertion.device_installation_id, str)
                and assertion.device_installation_id.startswith("dev_")
                and _ID_RE.fullmatch(assertion.device_installation_id)
                and assertion.outcome in {"SUCCESS", "FAILED", "CANCELLED", "UNAVAILABLE"}
                and isinstance(assertion.platform_enrollment_revision, int)
                and not isinstance(assertion.platform_enrollment_revision, bool)
                and assertion.platform_enrollment_revision >= 1
                and isinstance(assertion.challenge_fingerprint_sha256, str)
                and _SHA_RE.fullmatch(assertion.challenge_fingerprint_sha256)
                and _valid_canonical_utc_text(assertion.verified_at_utc)
                and _valid_canonical_utc_text(assertion.expires_at_utc)
                and isinstance(assertion.assertion_fingerprint_sha256, str)
                and _SHA_RE.fullmatch(assertion.assertion_fingerprint_sha256)
                and platform_biometric_assertion_fingerprint(assertion)
                == assertion.assertion_fingerprint_sha256
            )
        except (TypeError, ValueError):
            return False

    def issue_authentication_proof(
        self,
        request: object,
        raw_pin: object,
        now: object,
        *,
        platform_assertion: object = None,
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
        pin_only = policy.factor_policy == "PIN" and request.operation in {
            "LOCK_SESSION",
            "LOGOUT_SESSION",
        }
        combined = policy.factor_policy == "PIN_AND_BIOMETRIC"
        if not pin_only and not combined:
            _deny("OPERATION_UNSUPPORTED")
        if combined and platform_assertion is None:
            _deny("AUTHENTICATION_FAILED")
        if not isinstance(raw_pin, str):
            _deny("AUTHENTICATION_FAILED" if combined else "MALFORMED_UNTRUSTED_CONTEXT")

        current_time = cast(datetime, now)
        with self._state.lock:
            before = self._state.snapshot
            identity, device, pin, session = self._resolve_current_family(before, request)
            runtime = self._resolve_runtime(request, session)
            ownership = OPERATION_OWNERSHIP.get(request.operation)
            if ownership is None:
                _deny("CONTRACT_INCONSISTENT")
            if (
                ownership == "M0.10_OWNED_TRANSITION"
                and request.scope_fingerprint_sha256 != canonical_scope_fingerprint(request)
            ):
                _deny("AUTHORIZATION_DENIED")
            if pin_only:
                target_state = {
                    "LOCK_SESSION": "LOCKED",
                    "LOGOUT_SESSION": "LOGGED_OUT",
                }[request.operation]
                expected_mutation = session_mutation_fingerprint(
                    request,
                    target_state,
                    session.session_generation,
                    session.session_generation + 1,
                )
                if request.mutation_fingerprint_sha256 != expected_mutation:
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

            if combined:
                try:
                    self.verify_platform_assertion(platform_assertion, request, current_time)
                except AuthenticationError as error:
                    if error.reason in {
                        "AUTHENTICATION_FAILED",
                        "FACTOR_UNAVAILABLE",
                        "MALFORMED_UNTRUSTED_CONTEXT",
                    }:
                        _deny("AUTHENTICATION_FAILED")
                    raise

            effective_pin = pin
            if pin.failed_attempts or pin.lockout_until_utc is not None:
                effective_pin = self._updated_pin(pin, 0, None)
            proof = self._build_proof(
                request,
                policy,
                current_time,
                identity,
                device,
                effective_pin,
                session,
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
        with self._state.lock:
            snapshot = self._state.snapshot
            accepted_proof = self._resolve_accepted_proof_membership(candidate, snapshot)
        request = AuthorizationRequest(
            accepted_proof.account_id,
            accepted_proof.operator_id,
            accepted_proof.device_installation_id,
            accepted_proof.environment,
            accepted_proof.operation,
            accepted_proof.scope_fingerprint_sha256,
            accepted_proof.mutation_fingerprint_sha256,
            accepted_proof.causation_id,
            accepted_proof.correlation_id,
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
            accepted_proof.identity_revision,
            accepted_proof.device_trust_revision,
            accepted_proof.pin_revision,
            accepted_proof.platform_enrollment_revision,
            accepted_proof.security_generation,
            accepted_proof.session_generation,
        ):
            _deny("PROOF_STALE")
        return accepted_proof

    def _resolve_accepted_proof_membership(
        self,
        candidate: object,
        snapshot: InitialSecurityAuthoritySnapshot,
    ) -> AuthenticationProof:
        """Resolve only genuine Core-issued proof membership and its exact binding."""
        if not isinstance(candidate, AuthenticationProof):
            _deny("AUTHENTICATION_REQUIRED")
        self._validate_authentication_proof_structure(candidate)
        recomputed = authentication_proof_fingerprint(candidate)
        if candidate.proof_fingerprint_sha256 != recomputed:
            _deny("AUTHENTICATION_REQUIRED")
        accepted = snapshot.accepted_authentication_proofs.get(recomputed)
        binding = snapshot.accepted_authentication_proof_bindings.get(recomputed)
        if accepted != candidate or not isinstance(binding, CoreIssuedAuthenticationProofBinding):
            _deny("AUTHENTICATION_REQUIRED")
        if binding != self._binding(candidate):
            _deny("CONTRACT_INCONSISTENT")
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

        identity = cast(
            OperatorIdentitySecurityProjection,
            self._resolve_current_projection(
                snapshot.accepted_identities,
                snapshot.current_identities,
                scope_i,
                OperatorIdentitySecurityProjection,
                scope_i,
                lambda value: (value.account_id, value.operator_id),
                self._identity_intrinsically_valid,
                "CONTRACT_INCONSISTENT",
            ),
        )
        device = cast(
            DeviceTrustProjection,
            self._resolve_current_projection(
                snapshot.accepted_devices,
                snapshot.current_devices,
                scope_d,
                DeviceTrustProjection,
                scope_d,
                lambda value: (value.account_id, value.device_installation_id),
                self._device_intrinsically_valid,
                "CONTRACT_INCONSISTENT",
            ),
        )
        pin = cast(
            PinVerifierRecord,
            self._resolve_current_projection(
                snapshot.accepted_pins,
                snapshot.current_pins,
                scope_p,
                PinVerifierRecord,
                scope_p,
                lambda value: (value.account_id, value.operator_id, value.device_installation_id),
                lambda _value: True,
                "CONTRACT_INCONSISTENT",
            ),
        )
        session = cast(
            SessionSecurityState,
            self._resolve_current_projection(
                snapshot.accepted_sessions,
                snapshot.current_sessions,
                scope_p,
                SessionSecurityState,
                scope_p,
                lambda value: (value.account_id, value.operator_id, value.device_installation_id),
                self._session_intrinsically_valid,
                "CONTRACT_INCONSISTENT",
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

    def _resolve_biometric_context(
        self, snapshot: InitialSecurityAuthoritySnapshot, request: AuthorizationRequest
    ) -> tuple[
        OperatorIdentitySecurityProjection,
        DeviceTrustProjection,
        SessionSecurityState,
    ]:
        identity_scope = (request.account_id, request.operator_id)
        device_scope = (request.account_id, request.device_installation_id)
        session_scope = (*identity_scope, request.device_installation_id)
        identity = cast(
            OperatorIdentitySecurityProjection,
            self._resolve_current_projection(
                snapshot.accepted_identities,
                snapshot.current_identities,
                identity_scope,
                OperatorIdentitySecurityProjection,
                identity_scope,
                lambda value: (value.account_id, value.operator_id),
                self._identity_intrinsically_valid,
                "AUTHENTICATION_FAILED",
            ),
        )
        device = cast(
            DeviceTrustProjection,
            self._resolve_current_projection(
                snapshot.accepted_devices,
                snapshot.current_devices,
                device_scope,
                DeviceTrustProjection,
                device_scope,
                lambda value: (value.account_id, value.device_installation_id),
                self._device_intrinsically_valid,
                "AUTHENTICATION_FAILED",
            ),
        )
        session = cast(
            SessionSecurityState,
            self._resolve_current_projection(
                snapshot.accepted_sessions,
                snapshot.current_sessions,
                session_scope,
                SessionSecurityState,
                session_scope,
                lambda value: (value.account_id, value.operator_id, value.device_installation_id),
                self._session_intrinsically_valid,
                "AUTHENTICATION_FAILED",
            ),
        )
        if (
            len(
                {
                    identity.security_generation,
                    device.security_generation,
                    session.security_generation,
                }
            )
            != 1
        ):
            _deny("CONTRACT_INCONSISTENT")
        return identity, device, session

    @staticmethod
    def _resolve_current_projection(
        accepted: object,
        current: object,
        designation_scope: object,
        expected_type: type[object],
        expected_payload_scope: object,
        payload_scope: Callable[[Any], object],
        intrinsic_validator: Callable[[Any], bool],
        missing_reason: str,
    ) -> object:
        if not isinstance(accepted, MappingProxyType) or not isinstance(current, MappingProxyType):
            _deny("CONTRACT_INCONSISTENT")
        fingerprint = current.get(designation_scope)
        if fingerprint is None:
            _deny(missing_reason)
        if not isinstance(fingerprint, str):
            _deny("CONTRACT_INCONSISTENT")
        value = accepted.get(fingerprint)
        if value is None:
            _deny(missing_reason)
        if not isinstance(value, expected_type):
            _deny("CONTRACT_INCONSISTENT")
        terminal_fingerprint = getattr(value, "content_fingerprint_sha256", None)
        if not isinstance(terminal_fingerprint, str) or not _SHA_RE.fullmatch(terminal_fingerprint):
            _deny("CONTRACT_INCONSISTENT")
        try:
            recomputed_fingerprint = _fingerprint_without(value, "content_fingerprint_sha256")
            valid = (
                terminal_fingerprint == fingerprint == recomputed_fingerprint
                and payload_scope(value) == expected_payload_scope
                and intrinsic_validator(value)
            )
        except (AttributeError, TypeError, ValueError):
            valid = False
        if not valid:
            _deny("CONTRACT_INCONSISTENT")
        return value

    @staticmethod
    def _canonical_id(value: object, prefix: str) -> bool:
        return isinstance(value, str) and value.startswith(prefix) and bool(_ID_RE.fullmatch(value))

    @staticmethod
    def _positive_int(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 1

    @classmethod
    def _identity_intrinsically_valid(cls, value: OperatorIdentitySecurityProjection) -> bool:
        return (
            cls._canonical_id(value.account_id, "acct_")
            and cls._canonical_id(value.operator_id, "op_")
            and value.state in {"ACTIVE", "REVOKED"}
            and cls._positive_int(value.identity_revision)
            and cls._positive_int(value.security_generation)
        )

    @classmethod
    def _device_intrinsically_valid(cls, value: DeviceTrustProjection) -> bool:
        return (
            cls._canonical_id(value.account_id, "acct_")
            and cls._canonical_id(value.device_installation_id, "dev_")
            and value.state in {"ENROLLED_UNTRUSTED", "TRUSTED", "REVOKED", "REPLACED"}
            and cls._positive_int(value.trust_revision)
            and cls._positive_int(value.security_generation)
            and cls._positive_int(value.platform_enrollment_revision)
        )

    @classmethod
    def _session_intrinsically_valid(cls, value: SessionSecurityState) -> bool:
        return (
            cls._canonical_id(value.account_id, "acct_")
            and cls._canonical_id(value.operator_id, "op_")
            and cls._canonical_id(value.device_installation_id, "dev_")
            and cls._canonical_id(value.runtime_session_id, "run_")
            and value.state in {"LOCKED", "UNLOCKED", "LOGGED_OUT"}
            and cls._positive_int(value.session_generation)
            and cls._positive_int(value.security_generation)
        )

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
            factor_set=(
                ("PIN", "BIOMETRIC") if policy.factor_policy == "PIN_AND_BIOMETRIC" else ("PIN",)
            ),
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
            (("PIN", "BIOMETRIC") if policy.factor_policy == "PIN_AND_BIOMETRIC" else ("PIN",)),
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
    "OPERATION_OWNERSHIP",
    "OPERATION_POLICY_REGISTRY",
    "PinVerifierComparator",
    "authentication_proof_fingerprint",
    "canonical_scope_fingerprint",
    "complete_authentication_proof_fingerprint",
    "session_mutation_fingerprint",
]
