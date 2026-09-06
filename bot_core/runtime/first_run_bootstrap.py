"""Pure M0.3 first-run bootstrap compare-and-consume authority.

Durable publication belongs to M0.11.  Objects returned here are transport values,
not authentication, device-trust, readiness, policy, lease, or trading authority.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import NoReturn, Protocol, cast

from bot_core.persistence.fingerprints import canonical_json_sha256

_ID_RE = re.compile(
    r"^[a-z][a-z0-9]*_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

AUTHORITY_SOURCE = "external_product_provisioning_boundary"
INITIAL_SECURITY_ESTABLISHMENT_ONLY = "INITIAL_SECURITY_ESTABLISHMENT_ONLY"


class FirstRunBootstrapError(RuntimeError):
    """Controlled fail-closed bootstrap denial."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _deny(reason: str) -> NoReturn:
    raise FirstRunBootstrapError(reason)


def _canonical_id(value: object, prefix: str, field: str) -> None:
    if (
        not isinstance(value, str)
        or not _ID_RE.fullmatch(value)
        or not value.startswith(prefix + "_")
    ):
        _deny(f"MALFORMED_{field.upper()}")


def _positive(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        _deny(f"MALFORMED_{field.upper()}")


def _sha(value: object, field: str) -> None:
    if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
        _deny(f"MALFORMED_{field.upper()}")


def _utc(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not _UTC_RE.fullmatch(value):
        _deny(f"MALFORMED_{field.upper()}")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        _deny(f"MALFORMED_{field.upper()}")


@dataclass(frozen=True, slots=True)
class FirstRunBootstrapClaim:
    account_id: str
    device_installation_id: str
    intended_operator_id: str
    bootstrap_generation: int
    bootstrap_revision: int
    issued_at_utc: str
    expires_at_utc: str
    challenge_fingerprint_sha256: str
    provisioning_context_fingerprint_sha256: str
    claim_fingerprint_sha256: str

    def __post_init__(self) -> None:
        _canonical_id(self.account_id, "acct", "account_id")
        _canonical_id(self.device_installation_id, "dev", "device_installation_id")
        _canonical_id(self.intended_operator_id, "op", "intended_operator_id")
        _positive(self.bootstrap_generation, "bootstrap_generation")
        _positive(self.bootstrap_revision, "bootstrap_revision")
        _utc(self.issued_at_utc, "issued_at_utc")
        _utc(self.expires_at_utc, "expires_at_utc")
        _sha(self.challenge_fingerprint_sha256, "challenge_fingerprint_sha256")
        _sha(
            self.provisioning_context_fingerprint_sha256,
            "provisioning_context_fingerprint_sha256",
        )
        _sha(self.claim_fingerprint_sha256, "claim_fingerprint_sha256")


@dataclass(frozen=True, slots=True)
class ProvisioningMembershipBinding:
    claim_fingerprint_sha256: str
    complete_claim_content_fingerprint_sha256: str
    authority_source: str
    provisioning_context_fingerprint_sha256: str

    def __post_init__(self) -> None:
        _sha(self.claim_fingerprint_sha256, "claim_fingerprint_sha256")
        _sha(
            self.complete_claim_content_fingerprint_sha256,
            "complete_claim_content_fingerprint_sha256",
        )
        if self.authority_source != AUTHORITY_SOURCE:
            _deny("BOOTSTRAP_AUTHORITY_DENIED")
        _sha(
            self.provisioning_context_fingerprint_sha256,
            "provisioning_context_fingerprint_sha256",
        )


@dataclass(frozen=True, slots=True)
class ConsumedBootstrapAuthority:
    account_id: str
    device_installation_id: str
    bootstrap_generation: int
    bootstrap_revision: int
    claim_fingerprint_sha256: str
    challenge_fingerprint_sha256: str

    def __post_init__(self) -> None:
        _canonical_id(self.account_id, "acct", "account_id")
        _canonical_id(self.device_installation_id, "dev", "device_installation_id")
        _positive(self.bootstrap_generation, "bootstrap_generation")
        _positive(self.bootstrap_revision, "bootstrap_revision")
        _sha(self.claim_fingerprint_sha256, "claim_fingerprint_sha256")
        _sha(self.challenge_fingerprint_sha256, "challenge_fingerprint_sha256")


@dataclass(frozen=True, slots=True)
class CoreCurrentBootstrapState:
    state_fingerprint_sha256: str
    account_id: str
    device_installation_id: str
    intended_operator_id: str
    startup_readiness: str
    initial_security_lifecycle: str
    first_operator_presence: str
    expected_generation: int
    expected_revision: int
    consumed_authorities: tuple[ConsumedBootstrapAuthority, ...]
    state_revision: int

    def __post_init__(self) -> None:
        _sha(self.state_fingerprint_sha256, "state_fingerprint_sha256")
        _canonical_id(self.account_id, "acct", "account_id")
        _canonical_id(self.device_installation_id, "dev", "device_installation_id")
        _canonical_id(self.intended_operator_id, "op", "intended_operator_id")
        if self.startup_readiness != "SETUP_REQUIRED":
            _deny("MALFORMED_CORE_BOOTSTRAP_STATE")
        presence = {"PRE_INITIAL_SECURITY": "ABSENT", "INITIAL_SECURITY_COMPLETED": "PRESENT"}
        if presence.get(self.initial_security_lifecycle) != self.first_operator_presence:
            _deny("MALFORMED_CORE_BOOTSTRAP_STATE")
        _positive(self.expected_generation, "expected_generation")
        _positive(self.expected_revision, "expected_revision")
        _positive(self.state_revision, "state_revision")
        if not isinstance(self.consumed_authorities, tuple):
            _deny("MALFORMED_CONSUMED_AUTHORITIES")
        _validate_history(self)


@dataclass(frozen=True, slots=True)
class BootstrapTransitionResult:
    outcome: str
    authority_purpose: str
    pre_state_fingerprint_sha256: str
    post_state_fingerprint_sha256: str
    consumed_authority: ConsumedBootstrapAuthority

    def __post_init__(self) -> None:
        if self.outcome != "INITIAL_SECURITY_ESTABLISHMENT_AUTHORIZED":
            _deny("BOOTSTRAP_SCOPE_DENIED")
        if self.authority_purpose != INITIAL_SECURITY_ESTABLISHMENT_ONLY:
            _deny("BOOTSTRAP_SCOPE_DENIED")
        _sha(self.pre_state_fingerprint_sha256, "pre_state_fingerprint_sha256")
        _sha(self.post_state_fingerprint_sha256, "post_state_fingerprint_sha256")
        if not isinstance(self.consumed_authority, ConsumedBootstrapAuthority):
            _deny("MALFORMED_CONSUMED_AUTHORITY")


class ProvisioningBoundary(Protocol):
    """Core-visible adapter over pre-existing external provisioning authority."""

    def resolve_membership(self, claim_reference: str) -> ProvisioningMembershipBinding: ...

    def resolve_accepted_claim(self, claim_reference: str) -> FirstRunBootstrapClaim: ...


class BootstrapStateRegistry(Protocol):
    """Core-owned accepted-state/current-designation lookup boundary."""

    def resolve_accepted_state(self, state_reference: str) -> CoreCurrentBootstrapState: ...

    def current_state_reference(self, account_id: str, device_installation_id: str) -> str: ...


def claim_content_fingerprint(claim: FirstRunBootstrapClaim) -> str:
    content = asdict(claim)
    del content["claim_fingerprint_sha256"]
    return cast(str, canonical_json_sha256(content))


def state_content_fingerprint(state: CoreCurrentBootstrapState) -> str:
    content = asdict(state)
    del content["state_fingerprint_sha256"]
    return cast(str, canonical_json_sha256(content))


def _validate_history(state: CoreCurrentBootstrapState) -> None:
    seen: set[ConsumedBootstrapAuthority] = set()
    generations: set[int] = set()
    previous: tuple[int, int, str, str] | None = None
    for item in state.consumed_authorities:
        if not isinstance(item, ConsumedBootstrapAuthority):
            _deny("MALFORMED_CONSUMED_HISTORY")
        if (item.account_id, item.device_installation_id) != (
            state.account_id,
            state.device_installation_id,
        ):
            _deny("MALFORMED_CONSUMED_HISTORY")
        key = (
            item.bootstrap_generation,
            item.bootstrap_revision,
            item.claim_fingerprint_sha256,
            item.challenge_fingerprint_sha256,
        )
        if (
            item in seen
            or item.bootstrap_generation in generations
            or (previous is not None and key <= previous)
        ):
            _deny("MALFORMED_CONSUMED_HISTORY")
        seen.add(item)
        generations.add(item.bootstrap_generation)
        previous = key


def validate_bootstrap_state_transition(
    pre: CoreCurrentBootstrapState,
    post: CoreCurrentBootstrapState,
    appended: ConsumedBootstrapAuthority,
) -> None:
    """Revalidate one exact P1A state edge without conferring registry authority."""

    stable = (
        "account_id",
        "device_installation_id",
        "intended_operator_id",
        "startup_readiness",
        "initial_security_lifecycle",
        "first_operator_presence",
        "expected_generation",
        "expected_revision",
    )
    if (
        pre.initial_security_lifecycle != "PRE_INITIAL_SECURITY"
        or pre.first_operator_presence != "ABSENT"
        or any(getattr(pre, field) != getattr(post, field) for field in stable)
        or post.state_revision != pre.state_revision + 1
        or post.consumed_authorities != pre.consumed_authorities + (appended,)
        or appended.bootstrap_generation != pre.expected_generation
        or appended.bootstrap_revision != pre.expected_revision
    ):
        _deny("BOOTSTRAP_SCOPE_DENIED")
    _validate_history(pre)
    _validate_history(post)


def _resolve_state(
    registry: BootstrapStateRegistry, reference: object
) -> CoreCurrentBootstrapState:
    _sha(reference, "core_bootstrap_state_reference")
    try:
        state = registry.resolve_accepted_state(reference)  # type: ignore[arg-type]
    except (KeyError, LookupError, TypeError, ValueError, FirstRunBootstrapError):
        _deny("BOOTSTRAP_AUTHORITY_DENIED")
    if not isinstance(state, CoreCurrentBootstrapState):
        _deny("BOOTSTRAP_AUTHORITY_DENIED")
    if state.state_fingerprint_sha256 != reference or state_content_fingerprint(state) != reference:
        _deny("CONTRACT_INCONSISTENT")
    return state


class FirstRunBootstrapAuthority:
    """Pure semantic owner; injected ports retain all accepted/current authority."""

    def __init__(
        self, provisioning_boundary: ProvisioningBoundary, state_registry: BootstrapStateRegistry
    ) -> None:
        self._provisioning = provisioning_boundary
        self._states = state_registry

    def consume(
        self,
        claim: object,
        current_state_reference: object,
        now_utc: object,
        purpose: object,
    ) -> tuple[BootstrapTransitionResult, CoreCurrentBootstrapState]:
        if purpose != INITIAL_SECURITY_ESTABLISHMENT_ONLY:
            _deny("BOOTSTRAP_SCOPE_DENIED")
        if not isinstance(claim, FirstRunBootstrapClaim):
            _deny("MALFORMED_BOOTSTRAP_CLAIM")
        state = _resolve_state(self._states, current_state_reference)
        try:
            current = self._states.current_state_reference(
                state.account_id, state.device_installation_id
            )
        except (KeyError, LookupError, TypeError, ValueError, FirstRunBootstrapError):
            _deny("BOOTSTRAP_AUTHORITY_DENIED")
        if current != current_state_reference:
            _deny("STALE_CORE_BOOTSTRAP_STATE")
        if (
            state.initial_security_lifecycle != "PRE_INITIAL_SECURITY"
            or state.first_operator_presence != "ABSENT"
        ):
            _deny("BOOTSTRAP_NOT_ELIGIBLE")
        issued = _utc(claim.issued_at_utc, "issued_at_utc")
        expires = _utc(claim.expires_at_utc, "expires_at_utc")
        now = _utc(now_utc, "now_utc")
        if expires <= issued:
            _deny("MALFORMED_BOOTSTRAP_CLAIM")
        if now < issued:
            _deny("BOOTSTRAP_NOT_YET_VALID")
        if now > expires:
            _deny("BOOTSTRAP_EXPIRED")
        computed_claim = claim_content_fingerprint(claim)
        if computed_claim != claim.claim_fingerprint_sha256:
            _deny("BOOTSTRAP_AUTHORITY_DENIED")
        try:
            binding = self._provisioning.resolve_membership(claim.claim_fingerprint_sha256)
            accepted_claim = self._provisioning.resolve_accepted_claim(
                claim.claim_fingerprint_sha256
            )
        except (KeyError, LookupError, TypeError, ValueError, FirstRunBootstrapError):
            _deny("BOOTSTRAP_AUTHORITY_DENIED")
        if not isinstance(binding, ProvisioningMembershipBinding) or accepted_claim != claim:
            _deny("BOOTSTRAP_AUTHORITY_DENIED")
        if (
            binding.claim_fingerprint_sha256 != claim.claim_fingerprint_sha256
            or binding.complete_claim_content_fingerprint_sha256 != computed_claim
            or binding.authority_source != AUTHORITY_SOURCE
            or binding.provisioning_context_fingerprint_sha256
            != claim.provisioning_context_fingerprint_sha256
        ):
            _deny("BOOTSTRAP_AUTHORITY_DENIED")
        if (
            claim.account_id,
            claim.device_installation_id,
            claim.intended_operator_id,
            claim.bootstrap_generation,
            claim.bootstrap_revision,
        ) != (
            state.account_id,
            state.device_installation_id,
            state.intended_operator_id,
            state.expected_generation,
            state.expected_revision,
        ):
            _deny("BOOTSTRAP_BINDING_DENIED")
        consumed = ConsumedBootstrapAuthority(
            claim.account_id,
            claim.device_installation_id,
            claim.bootstrap_generation,
            claim.bootstrap_revision,
            claim.claim_fingerprint_sha256,
            claim.challenge_fingerprint_sha256,
        )
        if consumed in state.consumed_authorities or any(
            item.bootstrap_generation == consumed.bootstrap_generation
            for item in state.consumed_authorities
        ):
            _deny("BOOTSTRAP_REPLAY_DENIED")
        post = replace(
            state,
            state_fingerprint_sha256="0" * 64,
            consumed_authorities=state.consumed_authorities + (consumed,),
            state_revision=state.state_revision + 1,
        )
        post = replace(post, state_fingerprint_sha256=state_content_fingerprint(post))
        result = BootstrapTransitionResult(
            "INITIAL_SECURITY_ESTABLISHMENT_AUTHORIZED",
            INITIAL_SECURITY_ESTABLISHMENT_ONLY,
            state.state_fingerprint_sha256,
            post.state_fingerprint_sha256,
            consumed,
        )
        return result, post

    def revalidate(self, result: object, purpose: object) -> CoreCurrentBootstrapState:
        if (
            not isinstance(result, BootstrapTransitionResult)
            or purpose != INITIAL_SECURITY_ESTABLISHMENT_ONLY
            or result.authority_purpose != purpose
        ):
            _deny("BOOTSTRAP_SCOPE_DENIED")
        pre = _resolve_state(self._states, result.pre_state_fingerprint_sha256)
        post = _resolve_state(self._states, result.post_state_fingerprint_sha256)
        try:
            current = self._states.current_state_reference(
                post.account_id, post.device_installation_id
            )
        except (KeyError, LookupError, TypeError, ValueError, FirstRunBootstrapError):
            _deny("BOOTSTRAP_SCOPE_DENIED")
        if current != result.post_state_fingerprint_sha256:
            _deny("BOOTSTRAP_SCOPE_DENIED")
        if not post.consumed_authorities:
            _deny("BOOTSTRAP_SCOPE_DENIED")
        appended = post.consumed_authorities[-1]
        if appended != result.consumed_authority:
            _deny("BOOTSTRAP_SCOPE_DENIED")
        validate_bootstrap_state_transition(pre, post, appended)
        return post


__all__ = [
    "AUTHORITY_SOURCE",
    "INITIAL_SECURITY_ESTABLISHMENT_ONLY",
    "BootstrapStateRegistry",
    "BootstrapTransitionResult",
    "ConsumedBootstrapAuthority",
    "CoreCurrentBootstrapState",
    "FirstRunBootstrapAuthority",
    "FirstRunBootstrapClaim",
    "FirstRunBootstrapError",
    "ProvisioningBoundary",
    "ProvisioningMembershipBinding",
    "claim_content_fingerprint",
    "state_content_fingerprint",
    "validate_bootstrap_state_transition",
]
