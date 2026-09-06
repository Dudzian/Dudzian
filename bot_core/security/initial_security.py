"""Pure M0.10 initial-security establishment authority.

The module consumes an already durable M0.3 bootstrap transition.  It deliberately
does not persist the resulting security projections and mints no authentication,
entitlement, risk, lease, secret, or trading authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
import re
from threading import RLock
from types import MappingProxyType
from typing import Any, NoReturn, Protocol, cast

from bot_core.persistence.fingerprints import canonical_json_sha256
from bot_core.persistence.first_run_bootstrap import DurableFirstRunBootstrapRegistry
from bot_core.runtime.first_run_bootstrap import (
    AUTHORITY_SOURCE,
    INITIAL_SECURITY_ESTABLISHMENT_ONLY,
    BootstrapTransitionResult,
    ConsumedBootstrapAuthority,
    FirstRunBootstrapAuthority,
    FirstRunBootstrapClaim,
    FirstRunBootstrapError,
    ProvisioningBoundary,
    ProvisioningMembershipBinding,
    claim_content_fingerprint,
)
from bot_core.runtime.runtime_session import RuntimeSession

_ID_RE = re.compile(
    r"^[a-z][a-z0-9]*_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class InitialSecurityError(RuntimeError):
    """Controlled, PIN-safe, fail-closed M0.10 boundary error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _deny(reason: str) -> NoReturn:
    raise InitialSecurityError(reason)


@dataclass(frozen=True, slots=True)
class OperatorIdentitySecurityProjection:
    account_id: str
    operator_id: str
    state: str
    identity_revision: int
    security_generation: int
    content_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class DeviceTrustProjection:
    account_id: str
    device_installation_id: str
    state: str
    trust_revision: int
    security_generation: int
    platform_enrollment_revision: int
    content_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class PinVerifierRecord:
    account_id: str
    operator_id: str
    device_installation_id: str
    algorithm_id: str
    parameter_policy_version: int
    salt_reference: str
    verifier: str
    pin_revision: int
    failed_attempts: int
    lockout_until_utc: str | None
    security_generation: int
    content_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class SessionSecurityState:
    account_id: str
    operator_id: str
    device_installation_id: str
    runtime_session_id: str
    state: str
    session_generation: int
    security_generation: int
    content_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class InitialSecurityState:
    account_id: str
    operator_id: str
    device_installation_id: str
    security_generation: int
    session_generation: int
    state: str
    bootstrap_claim_fingerprint_sha256: str
    content_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class InitialSecurityEstablishmentResult:
    account_id: str
    operator_id: str
    device_installation_id: str
    security_generation: int
    session_generation: int
    bootstrap_claim_fingerprint_sha256: str
    result: str
    result_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class M03BootstrapAuthorityView:
    claim_fingerprint_sha256: str
    account_id: str
    device_installation_id: str
    operator_id: str
    bootstrap_generation: int
    bootstrap_revision: int
    purpose: str
    pre_state_fingerprint_sha256: str
    post_state_fingerprint_sha256: str
    consumed_authority_fingerprint_sha256: str
    consumed_claim_fingerprint_sha256: str
    consumed_challenge_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class M03AcceptedBootstrapAuthorityBinding:
    view_fingerprint_sha256: str
    complete_view_content_fingerprint_sha256: str
    claim_fingerprint_sha256: str
    account_id: str
    device_installation_id: str
    operator_id: str
    bootstrap_generation: int
    bootstrap_revision: int
    purpose: str
    accepted_pre_fingerprint_sha256: str
    current_post_fingerprint_sha256: str
    consumed_authority_fingerprint_sha256: str
    consumed_claim_fingerprint_sha256: str
    consumed_challenge_fingerprint_sha256: str
    authority_source: str


@dataclass(frozen=True, slots=True)
class PinVerifierMaterial:
    """Opaque output of the production PIN KDF/secure-salt boundary."""

    algorithm_id: str
    parameter_policy_version: int
    salt_reference: str
    verifier: str


class PinVerifierFactory(Protocol):
    def create(self, raw_pin: str) -> PinVerifierMaterial: ...


class RuntimeSessionAuthority(Protocol):
    def resolve_current(self, account_id: str, device_installation_id: str) -> RuntimeSession: ...


@dataclass(frozen=True, slots=True)
class _M03TrustedConsumedBootstrapEvidence:
    claim: FirstRunBootstrapClaim
    membership: ProvisioningMembershipBinding
    transition: BootstrapTransitionResult
    consumed: ConsumedBootstrapAuthority


def _make(cls: type, **values: object):  # type: ignore[type-arg,no-untyped-def]
    fingerprint_field = "content_fingerprint_sha256"
    # Construction helpers pass the terminal fingerprint placeholder last.
    payload = dict(values)
    payload.pop(fingerprint_field)
    values[fingerprint_field] = canonical_json_sha256(payload)
    return cls(**values)


def _make_fingerprint(value: object) -> str:
    payload = asdict(value)  # type: ignore[call-overload]
    return cast(
        str,
        canonical_json_sha256(
            {k: v for k, v in payload.items() if k != "content_fingerprint_sha256"}
        ),
    )


def _valid_salt_reference(value: str) -> bool:
    prefix = "secure-store://"
    if not value.startswith(prefix):
        return False
    locator = value[len(prefix) :]
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
    return (
        bool(locator)
        and not any(char.isspace() or char in "?#=" for char in locator)
        and not any(marker in locator.lower() for marker in forbidden)
    )


def _validate_bundle(view, runtime_session, identity, device, pin, session, initial) -> None:  # type: ignore[no-untyped-def]
    if not (
        _ID_RE.fullmatch(identity.account_id)
        and identity.account_id.startswith("acct_")
        and _ID_RE.fullmatch(identity.operator_id)
        and identity.operator_id.startswith("op_")
        and (
            identity.account_id,
            identity.operator_id,
            identity.state,
            identity.identity_revision,
            identity.security_generation,
        )
        == (view.account_id, view.operator_id, "ACTIVE", 1, 1)
        and (
            device.account_id,
            device.device_installation_id,
            device.state,
            device.trust_revision,
            device.platform_enrollment_revision,
            device.security_generation,
        )
        == (view.account_id, view.device_installation_id, "TRUSTED", 1, 1, 1)
        and (
            pin.account_id,
            pin.operator_id,
            pin.device_installation_id,
            pin.pin_revision,
            pin.failed_attempts,
            pin.lockout_until_utc,
            pin.security_generation,
        )
        == (view.account_id, view.operator_id, view.device_installation_id, 1, 0, None, 1)
        and (
            session.account_id,
            session.operator_id,
            session.device_installation_id,
            session.runtime_session_id,
            session.state,
            session.session_generation,
            session.security_generation,
        )
        == (
            view.account_id,
            view.operator_id,
            view.device_installation_id,
            runtime_session.runtime_session_id,
            "UNLOCKED",
            1,
            1,
        )
        and (
            initial.account_id,
            initial.operator_id,
            initial.device_installation_id,
            initial.security_generation,
            initial.session_generation,
            initial.state,
            initial.bootstrap_claim_fingerprint_sha256,
        )
        == (
            view.account_id,
            view.operator_id,
            view.device_installation_id,
            1,
            1,
            "ESTABLISHED",
            view.claim_fingerprint_sha256,
        )
        and all(
            _make_fingerprint(item) == item.content_fingerprint_sha256
            for item in (identity, device, pin, session, initial)
        )
    ):
        _deny("CONTRACT_INCONSISTENT")


class M03InitialSecurityBridge:
    """Read-only bridge from genuine durable M0.3 authority into M0.10 membership."""

    def __init__(
        self,
        registry: DurableFirstRunBootstrapRegistry,
        provisioning: ProvisioningBoundary,
    ) -> None:
        self._registry = registry
        self._provisioning = provisioning
        self._authority = FirstRunBootstrapAuthority(provisioning, registry)
        self._state = InitialSecuritySemanticState()

    def accept(self, transition: object) -> M03BootstrapAuthorityView:
        """Revalidate and register a durable transition; its transport shape is not authority."""

        if not isinstance(transition, BootstrapTransitionResult):
            _deny("M03_BOOTSTRAP_AUTHORITY_DENIED")
        try:
            post = self._authority.revalidate(transition, INITIAL_SECURITY_ESTABLISHMENT_ONLY)
            pre = self._registry.resolve_accepted_state(transition.pre_state_fingerprint_sha256)
            claim = self._provisioning.resolve_accepted_claim(
                transition.consumed_authority.claim_fingerprint_sha256
            )
            membership = self._provisioning.resolve_membership(
                transition.consumed_authority.claim_fingerprint_sha256
            )
        except (FirstRunBootstrapError, KeyError, LookupError, TypeError, ValueError) as exc:
            raise InitialSecurityError("M03_BOOTSTRAP_AUTHORITY_DENIED") from exc
        consumed = transition.consumed_authority
        if (
            not isinstance(claim, FirstRunBootstrapClaim)
            or not isinstance(membership, ProvisioningMembershipBinding)
            or claim_content_fingerprint(claim) != claim.claim_fingerprint_sha256
            or membership.claim_fingerprint_sha256 != claim.claim_fingerprint_sha256
            or membership.complete_claim_content_fingerprint_sha256
            != claim.claim_fingerprint_sha256
            or membership.provisioning_context_fingerprint_sha256
            != claim.provisioning_context_fingerprint_sha256
            or membership.authority_source != AUTHORITY_SOURCE
            or (
                claim.account_id,
                claim.device_installation_id,
                claim.intended_operator_id,
                claim.bootstrap_generation,
                claim.bootstrap_revision,
                claim.claim_fingerprint_sha256,
                claim.challenge_fingerprint_sha256,
            )
            != (
                post.account_id,
                post.device_installation_id,
                post.intended_operator_id,
                consumed.bootstrap_generation,
                consumed.bootstrap_revision,
                consumed.claim_fingerprint_sha256,
                consumed.challenge_fingerprint_sha256,
            )
            or (pre.expected_generation, pre.expected_revision)
            != (
                consumed.bootstrap_generation,
                consumed.bootstrap_revision,
            )
        ):
            _deny("M03_BOOTSTRAP_AUTHORITY_DENIED")
        evidence = _M03TrustedConsumedBootstrapEvidence(claim, membership, transition, consumed)
        consumed_fp = canonical_json_sha256(asdict(evidence.consumed))
        view = M03BootstrapAuthorityView(
            claim.claim_fingerprint_sha256,
            claim.account_id,
            claim.device_installation_id,
            claim.intended_operator_id,
            claim.bootstrap_generation,
            claim.bootstrap_revision,
            INITIAL_SECURITY_ESTABLISHMENT_ONLY,
            transition.pre_state_fingerprint_sha256,
            transition.post_state_fingerprint_sha256,
            consumed_fp,
            consumed.claim_fingerprint_sha256,
            consumed.challenge_fingerprint_sha256,
        )
        view_fp = canonical_json_sha256(asdict(view))
        binding = M03AcceptedBootstrapAuthorityBinding(
            view_fp,
            view_fp,
            view.claim_fingerprint_sha256,
            view.account_id,
            view.device_installation_id,
            view.operator_id,
            view.bootstrap_generation,
            view.bootstrap_revision,
            view.purpose,
            view.pre_state_fingerprint_sha256,
            view.post_state_fingerprint_sha256,
            view.consumed_authority_fingerprint_sha256,
            view.consumed_claim_fingerprint_sha256,
            view.consumed_challenge_fingerprint_sha256,
            evidence.membership.authority_source,
        )
        scope = (view.account_id, view.device_installation_id)
        with self._state.lock:
            before = self._state.snapshot
            if view.claim_fingerprint_sha256 in before.consumed_bootstrap_claims:
                _deny("BOOTSTRAP_REPLAY_DENIED")
            accepted_bindings = dict(before.accepted_bootstrap_bindings)
            accepted_views = dict(before.accepted_bootstrap_views)
            current = dict(before.current_bootstrap)
            accepted_bindings[view.claim_fingerprint_sha256] = binding
            accepted_views[view.claim_fingerprint_sha256] = view
            current[scope] = view.claim_fingerprint_sha256
            self._state.snapshot = replace(
                before,
                accepted_bootstrap_bindings=MappingProxyType(accepted_bindings),
                accepted_bootstrap_views=MappingProxyType(accepted_views),
                current_bootstrap=MappingProxyType(current),
            )
        return view

    def resolve(self, candidate: object) -> M03BootstrapAuthorityView:
        """Resolve accepted bridge membership and revalidate its upstream state again."""

        if not isinstance(candidate, M03BootstrapAuthorityView):
            _deny("AUTHORIZATION_DENIED")
        key = canonical_json_sha256(asdict(candidate))
        with self._state.lock:
            snapshot = self._state.snapshot
        view = snapshot.accepted_bootstrap_views.get(candidate.claim_fingerprint_sha256)
        binding = snapshot.accepted_bootstrap_bindings.get(candidate.claim_fingerprint_sha256)
        scope = (candidate.account_id, candidate.device_installation_id)
        if candidate.claim_fingerprint_sha256 in snapshot.consumed_bootstrap_claims:
            _deny("BOOTSTRAP_REPLAY_DENIED")
        if (
            view != candidate
            or binding is None
            or snapshot.current_bootstrap.get(scope) != candidate.claim_fingerprint_sha256
        ):
            _deny("AUTHORIZATION_DENIED")
        expected_binding = M03AcceptedBootstrapAuthorityBinding(
            key,
            key,
            view.claim_fingerprint_sha256,
            view.account_id,
            view.device_installation_id,
            view.operator_id,
            view.bootstrap_generation,
            view.bootstrap_revision,
            view.purpose,
            view.pre_state_fingerprint_sha256,
            view.post_state_fingerprint_sha256,
            view.consumed_authority_fingerprint_sha256,
            view.consumed_claim_fingerprint_sha256,
            view.consumed_challenge_fingerprint_sha256,
            binding.authority_source,
        )
        if binding != expected_binding or binding.authority_source != AUTHORITY_SOURCE:
            _deny("CONTRACT_INCONSISTENT")
        consumed = ConsumedBootstrapAuthority(
            view.account_id,
            view.device_installation_id,
            view.bootstrap_generation,
            view.bootstrap_revision,
            view.consumed_claim_fingerprint_sha256,
            view.consumed_challenge_fingerprint_sha256,
        )
        transition = BootstrapTransitionResult(
            "INITIAL_SECURITY_ESTABLISHMENT_AUTHORIZED",
            view.purpose,
            view.pre_state_fingerprint_sha256,
            view.post_state_fingerprint_sha256,
            consumed,
        )
        try:
            self._authority.revalidate(transition, INITIAL_SECURITY_ESTABLISHMENT_ONLY)
            claim = self._provisioning.resolve_accepted_claim(view.claim_fingerprint_sha256)
            membership = self._provisioning.resolve_membership(view.claim_fingerprint_sha256)
        except (FirstRunBootstrapError, KeyError, LookupError, TypeError, ValueError) as exc:
            raise InitialSecurityError("AUTHORIZATION_DENIED") from exc
        if (
            not isinstance(claim, FirstRunBootstrapClaim)
            or not isinstance(membership, ProvisioningMembershipBinding)
            or claim_content_fingerprint(claim) != claim.claim_fingerprint_sha256
            or membership.claim_fingerprint_sha256 != claim.claim_fingerprint_sha256
            or membership.complete_claim_content_fingerprint_sha256
            != claim.claim_fingerprint_sha256
            or membership.provisioning_context_fingerprint_sha256
            != claim.provisioning_context_fingerprint_sha256
            or membership.authority_source != binding.authority_source
            or canonical_json_sha256(asdict(consumed)) != view.consumed_authority_fingerprint_sha256
            or claim.challenge_fingerprint_sha256 != view.consumed_challenge_fingerprint_sha256
            or claim.claim_fingerprint_sha256 != view.consumed_claim_fingerprint_sha256
        ):
            _deny("AUTHORIZATION_DENIED")
        return view


@dataclass(frozen=True, slots=True)
class InitialSecurityAuthoritySnapshot:
    accepted_identities: Mapping[str, OperatorIdentitySecurityProjection]
    current_identities: Mapping[tuple[str, str], str]
    accepted_devices: Mapping[str, DeviceTrustProjection]
    current_devices: Mapping[tuple[str, str], str]
    accepted_pins: Mapping[str, PinVerifierRecord]
    current_pins: Mapping[tuple[str, str, str], str]
    accepted_sessions: Mapping[str, SessionSecurityState]
    current_sessions: Mapping[tuple[str, str, str], str]
    accepted_initial_states: Mapping[str, InitialSecurityState]
    current_initial_states: Mapping[tuple[str, str], str]
    accepted_bootstrap_bindings: Mapping[str, M03AcceptedBootstrapAuthorityBinding]
    accepted_bootstrap_views: Mapping[str, M03BootstrapAuthorityView]
    current_bootstrap: Mapping[tuple[str, str], str]
    consumed_bootstrap_claims: frozenset[str]
    consumed_bootstrap_authority_fingerprints: frozenset[str]


def _empty_snapshot() -> InitialSecurityAuthoritySnapshot:
    empty: Mapping[Any, Any] = MappingProxyType({})
    return InitialSecurityAuthoritySnapshot(
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        frozenset(),
        frozenset(),
    )


class InitialSecuritySemanticState:
    """Shared Core state and lock; bridge fencing and security publication are one plane."""

    def __init__(self) -> None:
        self.lock = RLock()
        self.snapshot = _empty_snapshot()


class InitialSecurityAuthority:
    """Single Core-owned semantic owner for atomic initial M0.10 publication."""

    def __init__(
        self,
        bridge: M03InitialSecurityBridge,
        pin_verifier_factory: PinVerifierFactory,
        runtime_sessions: RuntimeSessionAuthority,
    ) -> None:
        self._bridge = bridge
        self._pin_factory = pin_verifier_factory
        self._runtime_sessions = runtime_sessions
        self._state = bridge._state

    @property
    def snapshot(self) -> InitialSecurityAuthoritySnapshot:
        return self._state.snapshot

    @staticmethod
    def _resolve_current(accepted: Mapping[str, object], current: Mapping[Any, str], scope: object):
        fingerprint = current.get(scope)
        value = accepted.get(fingerprint) if fingerprint is not None else None
        if value is None or _make_fingerprint(value) != fingerprint:
            _deny("CONTRACT_INCONSISTENT")
        return value

    def resolve_current_identity(
        self, account_id: str, operator_id: str
    ) -> OperatorIdentitySecurityProjection:
        value = cast(
            OperatorIdentitySecurityProjection,
            self._resolve_current(
                self.snapshot.accepted_identities,
                self.snapshot.current_identities,
                (account_id, operator_id),
            ),
        )
        if (value.account_id, value.operator_id) != (account_id, operator_id):
            _deny("CONTRACT_INCONSISTENT")
        return value

    def resolve_current_device(self, account_id: str, device_id: str) -> DeviceTrustProjection:
        value = cast(
            DeviceTrustProjection,
            self._resolve_current(
                self.snapshot.accepted_devices,
                self.snapshot.current_devices,
                (account_id, device_id),
            ),
        )
        if (value.account_id, value.device_installation_id) != (account_id, device_id):
            _deny("CONTRACT_INCONSISTENT")
        return value

    def resolve_current_pin(
        self, account_id: str, operator_id: str, device_id: str
    ) -> PinVerifierRecord:
        value = cast(
            PinVerifierRecord,
            self._resolve_current(
                self.snapshot.accepted_pins,
                self.snapshot.current_pins,
                (account_id, operator_id, device_id),
            ),
        )
        if (value.account_id, value.operator_id, value.device_installation_id) != (
            account_id,
            operator_id,
            device_id,
        ):
            _deny("CONTRACT_INCONSISTENT")
        return value

    def resolve_current_session(
        self, account_id: str, operator_id: str, device_id: str
    ) -> SessionSecurityState:
        value = cast(
            SessionSecurityState,
            self._resolve_current(
                self.snapshot.accepted_sessions,
                self.snapshot.current_sessions,
                (account_id, operator_id, device_id),
            ),
        )
        if (value.account_id, value.operator_id, value.device_installation_id) != (
            account_id,
            operator_id,
            device_id,
        ):
            _deny("CONTRACT_INCONSISTENT")
        return value

    def resolve_current_initial_security(
        self, account_id: str, device_id: str
    ) -> InitialSecurityState:
        value = cast(
            InitialSecurityState,
            self._resolve_current(
                self.snapshot.accepted_initial_states,
                self.snapshot.current_initial_states,
                (account_id, device_id),
            ),
        )
        if (value.account_id, value.device_installation_id) != (account_id, device_id):
            _deny("CONTRACT_INCONSISTENT")
        return value

    def establish_initial_security(
        self, bootstrap_view: object, raw_pin: object
    ) -> InitialSecurityEstablishmentResult:
        if not isinstance(raw_pin, str) or not raw_pin:
            _deny("INVALID_PIN_INPUT")
        view = self._bridge.resolve(bootstrap_view)
        with self._state.lock:
            before = self._state.snapshot
            scope = (view.account_id, view.device_installation_id)
            if view.claim_fingerprint_sha256 in before.consumed_bootstrap_claims:
                _deny("BOOTSTRAP_REPLAY_DENIED")
            if before.current_bootstrap.get(scope) != view.claim_fingerprint_sha256:
                _deny("AUTHORIZATION_DENIED")
            if any(
                (
                    before.accepted_identities,
                    before.accepted_devices,
                    before.accepted_pins,
                    before.accepted_sessions,
                    before.accepted_initial_states,
                )
            ):
                _deny("PRE_EXISTING_SECURITY_AUTHORITY")
            dependency_failed = False
            try:
                runtime_session = self._runtime_sessions.resolve_current(
                    view.account_id, view.device_installation_id
                )
                if (
                    not isinstance(runtime_session, RuntimeSession)
                    or runtime_session.closed
                    or runtime_session.device_installation_id != view.device_installation_id
                    or not _ID_RE.fullmatch(runtime_session.runtime_session_id)
                    or not runtime_session.runtime_session_id.startswith("run_")
                ):
                    _deny("RUNTIME_SESSION_AUTHORITY_DENIED")
                material = self._pin_factory.create(raw_pin)
            except InitialSecurityError:
                raise
            except Exception:
                # Do not retain the dependency exception: it may contain the raw PIN.
                dependency_failed = True
                material = None
            if dependency_failed:
                _deny("PIN_VERIFIER_DEPENDENCY_FAILURE")
            if (
                not isinstance(material, PinVerifierMaterial)
                or not all(
                    isinstance(value, str) and value
                    for value in (
                        material.algorithm_id,
                        material.salt_reference,
                        material.verifier,
                    )
                )
                or isinstance(material.parameter_policy_version, bool)
                or not isinstance(material.parameter_policy_version, int)
                or material.parameter_policy_version < 1
                or not _SHA_RE.fullmatch(material.verifier)
                or not _valid_salt_reference(material.salt_reference)
            ):
                _deny("PIN_VERIFIER_DEPENDENCY_FAILURE")
            identity = _make(
                OperatorIdentitySecurityProjection,
                account_id=view.account_id,
                operator_id=view.operator_id,
                state="ACTIVE",
                identity_revision=1,
                security_generation=1,
                content_fingerprint_sha256="",
            )
            device = _make(
                DeviceTrustProjection,
                account_id=view.account_id,
                device_installation_id=view.device_installation_id,
                state="TRUSTED",
                trust_revision=1,
                security_generation=1,
                platform_enrollment_revision=1,
                content_fingerprint_sha256="",
            )
            pin = _make(
                PinVerifierRecord,
                account_id=view.account_id,
                operator_id=view.operator_id,
                device_installation_id=view.device_installation_id,
                algorithm_id=material.algorithm_id,
                parameter_policy_version=material.parameter_policy_version,
                salt_reference=material.salt_reference,
                verifier=material.verifier,
                pin_revision=1,
                failed_attempts=0,
                lockout_until_utc=None,
                security_generation=1,
                content_fingerprint_sha256="",
            )
            session = _make(
                SessionSecurityState,
                account_id=view.account_id,
                operator_id=view.operator_id,
                device_installation_id=view.device_installation_id,
                runtime_session_id=runtime_session.runtime_session_id,
                state="UNLOCKED",
                session_generation=1,
                security_generation=1,
                content_fingerprint_sha256="",
            )
            initial = _make(
                InitialSecurityState,
                account_id=view.account_id,
                operator_id=view.operator_id,
                device_installation_id=view.device_installation_id,
                security_generation=1,
                session_generation=1,
                state="ESTABLISHED",
                bootstrap_claim_fingerprint_sha256=view.claim_fingerprint_sha256,
                content_fingerprint_sha256="",
            )
            generations = {
                identity.security_generation,
                device.security_generation,
                pin.security_generation,
                session.security_generation,
                initial.security_generation,
            }
            if generations != {1} or (session.session_generation, initial.session_generation) != (
                1,
                1,
            ):
                _deny("CONTRACT_INCONSISTENT")
            _validate_bundle(view, runtime_session, identity, device, pin, session, initial)
            # One immutable pointer replacement publishes the complete shadow bundle and fence.
            current_bootstrap = dict(before.current_bootstrap)
            del current_bootstrap[scope]
            self._state.snapshot = replace(
                before,
                accepted_identities=MappingProxyType(
                    {identity.content_fingerprint_sha256: identity}
                ),
                current_identities=MappingProxyType(
                    {
                        (
                            identity.account_id,
                            identity.operator_id,
                        ): identity.content_fingerprint_sha256
                    }
                ),
                accepted_devices=MappingProxyType({device.content_fingerprint_sha256: device}),
                current_devices=MappingProxyType(
                    {
                        (
                            device.account_id,
                            device.device_installation_id,
                        ): device.content_fingerprint_sha256
                    }
                ),
                accepted_pins=MappingProxyType({pin.content_fingerprint_sha256: pin}),
                current_pins=MappingProxyType(
                    {
                        (
                            pin.account_id,
                            pin.operator_id,
                            pin.device_installation_id,
                        ): pin.content_fingerprint_sha256
                    }
                ),
                accepted_sessions=MappingProxyType({session.content_fingerprint_sha256: session}),
                current_sessions=MappingProxyType(
                    {
                        (
                            session.account_id,
                            session.operator_id,
                            session.device_installation_id,
                        ): session.content_fingerprint_sha256
                    }
                ),
                accepted_initial_states=MappingProxyType(
                    {initial.content_fingerprint_sha256: initial}
                ),
                current_initial_states=MappingProxyType(
                    {scope: initial.content_fingerprint_sha256}
                ),
                current_bootstrap=MappingProxyType(current_bootstrap),
                consumed_bootstrap_claims=before.consumed_bootstrap_claims
                | {view.claim_fingerprint_sha256},
                consumed_bootstrap_authority_fingerprints=before.consumed_bootstrap_authority_fingerprints
                | {view.consumed_authority_fingerprint_sha256},
            )
            result_values = {
                "account_id": view.account_id,
                "operator_id": view.operator_id,
                "device_installation_id": view.device_installation_id,
                "security_generation": 1,
                "session_generation": 1,
                "bootstrap_claim_fingerprint_sha256": view.claim_fingerprint_sha256,
                "result": "INITIAL_SECURITY_ESTABLISHED",
            }
            return InitialSecurityEstablishmentResult(
                account_id=view.account_id,
                operator_id=view.operator_id,
                device_installation_id=view.device_installation_id,
                security_generation=1,
                session_generation=1,
                bootstrap_claim_fingerprint_sha256=view.claim_fingerprint_sha256,
                result="INITIAL_SECURITY_ESTABLISHED",
                result_fingerprint_sha256=canonical_json_sha256(result_values),
            )


__all__ = [
    "DeviceTrustProjection",
    "InitialSecurityAuthority",
    "InitialSecurityAuthoritySnapshot",
    "InitialSecurityError",
    "InitialSecurityEstablishmentResult",
    "InitialSecurityState",
    "M03AcceptedBootstrapAuthorityBinding",
    "M03BootstrapAuthorityView",
    "M03InitialSecurityBridge",
    "OperatorIdentitySecurityProjection",
    "PinVerifierFactory",
    "PinVerifierMaterial",
    "PinVerifierRecord",
    "RuntimeSessionAuthority",
    "SessionSecurityState",
]
