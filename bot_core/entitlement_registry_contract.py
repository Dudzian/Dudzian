"""Closed semantic values for the Account Genesis entitlement registry.

This is an executable port contract, not a storage adapter. It contains no
PostgreSQL, clock, signing, or network behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import re
from typing import Protocol, runtime_checkable

from bot_core.persistence.fingerprints import canonical_json


_UUID7 = r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
_PREFIXED_IDS = {
    "bootstrap_entitlement_id": re.compile(rf"ent_{_UUID7}\Z"),
    "logical_operation_id": re.compile(rf"ago_{_UUID7}\Z"),
    "account_id": re.compile(rf"acct_{_UUID7}\Z"),
    "issuance_attempt_id": re.compile(rf"rpa_{_UUID7}\Z"),
    "root_proof_id": re.compile(rf"rpf_{_UUID7}\Z"),
}
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_BINDING_DIGEST_DOMAIN = b"ENTITLEMENT_REGISTRY_BINDING_IDENTITY_V1\x00"


class ContractValidationError(ValueError):
    """Controlled fail-closed error for malformed or fabricated contract values."""


def _text(name: str, value: object) -> None:
    if type(value) is not str or not value.strip():
        raise ContractValidationError(f"{name} must be an exact non-empty str")


def _positive(name: str, value: object) -> None:
    if type(value) is not int or value < 1:
        raise ContractValidationError(f"{name} must be a positive exact int")


def _identifier(name: str, value: object) -> None:
    if type(value) is not str or _PREFIXED_IDS[name].fullmatch(value) is None:
        raise ContractValidationError(f"{name} must be its canonical lowercase UUIDv7 form")


def _digest(name: str, value: object) -> None:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise ContractValidationError(f"{name} must be lowercase SHA-256 hex")


def _slots(value: object, expected: type[object]) -> tuple[object, ...]:
    if type(value) is not expected:
        raise ContractValidationError(f"value must be exact {expected.__name__}")
    try:
        return tuple(getattr(value, field.name) for field in fields(expected))
    except (AttributeError, TypeError) as exc:
        raise ContractValidationError(f"malformed fabricated {expected.__name__}") from exc


class EntitlementLifecycle(str, Enum):
    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"
    SUPERSEDED = "SUPERSEDED"


class BindingKind(str, Enum):
    UNBOUND = "UNBOUND"
    BOUND = "BOUND"


class RegistryReadOutcome(str, Enum):
    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    CORRUPT = "CORRUPT"
    UNAVAILABLE = "UNAVAILABLE"


class BindOutcome(str, Enum):
    NEW_BIND_COMMITTED = "NEW_BIND_COMMITTED"
    EXACT_REPLAY = "EXACT_REPLAY"
    CONFLICT_BOUND_TO_DIFFERENT_TUPLE = "CONFLICT_BOUND_TO_DIFFERENT_TUPLE"
    STALE_PREDECESSOR = "STALE_PREDECESSOR"
    NOT_FOUND = "NOT_FOUND"
    INACTIVE_REVOKED = "INACTIVE_REVOKED"
    INACTIVE_SUPERSEDED = "INACTIVE_SUPERSEDED"
    CORRUPT = "CORRUPT"
    UNAVAILABLE = "UNAVAILABLE"
    RETRYABLE_SERIALIZATION_FAILURE = "RETRYABLE_SERIALIZATION_FAILURE"


class AdminOutcome(str, Enum):
    COMMITTED = "COMMITTED"
    CONFLICT = "CONFLICT"
    NOT_FOUND = "NOT_FOUND"
    CORRUPT = "CORRUPT"
    UNAVAILABLE = "UNAVAILABLE"
    RETRYABLE_SERIALIZATION_FAILURE = "RETRYABLE_SERIALIZATION_FAILURE"


class BindResolutionKind(str, Enum):
    NEW_BIND_ELIGIBLE = "NEW_BIND_ELIGIBLE"
    EXACT_REPLAY = "EXACT_REPLAY"
    CONFLICT_BOUND_TO_DIFFERENT_TUPLE = "CONFLICT_BOUND_TO_DIFFERENT_TUPLE"
    STALE_PREDECESSOR = "STALE_PREDECESSOR"
    NOT_FOUND = "NOT_FOUND"
    INACTIVE_REVOKED = "INACTIVE_REVOKED"
    INACTIVE_SUPERSEDED = "INACTIVE_SUPERSEDED"


def legal_lifecycle_transition(
    predecessor: EntitlementLifecycle, successor: EntitlementLifecycle
) -> bool:
    if type(predecessor) is not EntitlementLifecycle or type(successor) is not EntitlementLifecycle:
        raise ContractValidationError("lifecycle values must be exact EntitlementLifecycle members")
    return predecessor is EntitlementLifecycle.ACTIVE and successor in (
        EntitlementLifecycle.REVOKED,
        EntitlementLifecycle.SUPERSEDED,
    )


@dataclass(frozen=True, slots=True)
class RegistrySubject:
    lookup_handle: str
    environment: str
    trust_domain: str

    def __post_init__(self) -> None:
        for field in fields(self):
            _text(field.name, getattr(self, field.name))


@dataclass(frozen=True, slots=True)
class EntitlementIdentity:
    bootstrap_entitlement_id: str
    entitlement_generation: int
    environment: str
    trust_domain: str
    product_scope: str
    intended_action: str = "ACCOUNT_GENESIS_BOOTSTRAP"

    def __post_init__(self) -> None:
        _identifier("bootstrap_entitlement_id", self.bootstrap_entitlement_id)
        _positive("entitlement_generation", self.entitlement_generation)
        for name in ("environment", "trust_domain", "product_scope"):
            _text(name, getattr(self, name))
        if (
            type(self.intended_action) is not str
            or self.intended_action != "ACCOUNT_GENESIS_BOOTSTRAP"
        ):
            raise ContractValidationError("intended_action must be ACCOUNT_GENESIS_BOOTSTRAP")


@dataclass(frozen=True, slots=True)
class EntitlementProvenance:
    provisioning_principal_id: str
    claimant_key_id: str
    claimant_key_version: int
    creation_authority_identity: str
    authenticated_creation_reference: str
    authenticated_creation_digest_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "provisioning_principal_id",
            "claimant_key_id",
            "creation_authority_identity",
            "authenticated_creation_reference",
        ):
            _text(name, getattr(self, name))
        _positive("claimant_key_version", self.claimant_key_version)
        _digest("authenticated_creation_digest_sha256", self.authenticated_creation_digest_sha256)


@dataclass(frozen=True, slots=True)
class UnboundBinding:
    kind: BindingKind = BindingKind.UNBOUND

    def __post_init__(self) -> None:
        if type(self.kind) is not BindingKind or self.kind is not BindingKind.UNBOUND:
            raise ContractValidationError("UnboundBinding kind must be UNBOUND")


@dataclass(frozen=True, slots=True)
class BoundBinding:
    logical_operation_id: str
    account_id: str
    canonical_genesis_request_fingerprint_sha256: str
    entitlement_generation: int
    issuance_attempt_id: str
    requester_principal_id: str
    requester_key_id: str
    requester_key_version: int
    provisioning_principal_id: str
    claimant_key_id: str
    claimant_key_version: int
    signed_request_payload_digest_sha256: str
    signed_request_canonical_bytes_reference: str
    root_proof_id: str
    issuer_signing_credential_id: str
    issuer_signing_key_version: int
    kind: BindingKind = BindingKind.BOUND

    def __post_init__(self) -> None:
        for name in ("logical_operation_id", "account_id", "issuance_attempt_id", "root_proof_id"):
            _identifier(name, getattr(self, name))
        for name in (
            "canonical_genesis_request_fingerprint_sha256",
            "signed_request_payload_digest_sha256",
        ):
            _digest(name, getattr(self, name))
        for name in (
            "requester_principal_id",
            "requester_key_id",
            "provisioning_principal_id",
            "claimant_key_id",
            "signed_request_canonical_bytes_reference",
            "issuer_signing_credential_id",
        ):
            _text(name, getattr(self, name))
        for name in (
            "entitlement_generation",
            "requester_key_version",
            "claimant_key_version",
            "issuer_signing_key_version",
        ):
            _positive(name, getattr(self, name))
        if type(self.kind) is not BindingKind or self.kind is not BindingKind.BOUND:
            raise ContractValidationError("BoundBinding kind must be BOUND")


RegistryBinding = UnboundBinding | BoundBinding


@dataclass(frozen=True, slots=True)
class AuthoritativeEntitlementState:
    subject: RegistrySubject
    identity: EntitlementIdentity
    provenance: EntitlementProvenance
    lifecycle: EntitlementLifecycle
    binding: RegistryBinding
    authoritative_state_revision: int
    predecessor_revision: int | None

    def __post_init__(self) -> None:
        subject = validate_exact_snapshot(self.subject)
        identity = validate_exact_snapshot(self.identity)
        provenance = validate_exact_snapshot(self.provenance)
        binding = validate_exact_snapshot(self.binding)
        if not isinstance(subject, RegistrySubject) or not isinstance(
            identity, EntitlementIdentity
        ):
            raise ContractValidationError("invalid nested state identity")
        if not isinstance(provenance, EntitlementProvenance):
            raise ContractValidationError("invalid nested state provenance")
        if type(self.lifecycle) is not EntitlementLifecycle:
            raise ContractValidationError("lifecycle must be exact EntitlementLifecycle")
        if type(binding) not in (UnboundBinding, BoundBinding):
            raise ContractValidationError("invalid nested state binding")
        _positive("authoritative_state_revision", self.authoritative_state_revision)
        if self.predecessor_revision is not None:
            _positive("predecessor_revision", self.predecessor_revision)
            if self.predecessor_revision >= self.authoritative_state_revision:
                raise ContractValidationError(
                    "predecessor revision must be lower than current revision"
                )
        if (
            subject.environment != identity.environment
            or subject.trust_domain != identity.trust_domain
        ):
            raise ContractValidationError("subject security scope must equal identity scope")
        if type(binding) is BoundBinding:
            if binding.entitlement_generation != identity.entitlement_generation:
                raise ContractValidationError(
                    "binding generation must equal entitlement generation"
                )
            validate_claimant_anchor(provenance, binding)


@dataclass(frozen=True, slots=True)
class RegistryReadResult:
    outcome: RegistryReadOutcome
    state: AuthoritativeEntitlementState | None

    def __post_init__(self) -> None:
        if type(self.outcome) is not RegistryReadOutcome:
            raise ContractValidationError("outcome must be exact RegistryReadOutcome")
        if self.outcome is RegistryReadOutcome.FOUND:
            validate_exact_snapshot(self.state)
        elif self.state is not None:
            raise ContractValidationError("non-FOUND outcomes cannot manufacture a state")


@dataclass(frozen=True, slots=True)
class BindPredecessor:
    bootstrap_entitlement_id: str
    entitlement_generation: int
    authoritative_state_revision: int
    lifecycle: EntitlementLifecycle
    binding_identity_digest_sha256: str
    provisioning_principal_id: str
    claimant_key_id: str
    claimant_key_version: int

    def __post_init__(self) -> None:
        _identifier("bootstrap_entitlement_id", self.bootstrap_entitlement_id)
        _positive("entitlement_generation", self.entitlement_generation)
        _positive("authoritative_state_revision", self.authoritative_state_revision)
        if type(self.lifecycle) is not EntitlementLifecycle:
            raise ContractValidationError("lifecycle must be exact EntitlementLifecycle")
        _digest("binding_identity_digest_sha256", self.binding_identity_digest_sha256)
        _text("provisioning_principal_id", self.provisioning_principal_id)
        _text("claimant_key_id", self.claimant_key_id)
        _positive("claimant_key_version", self.claimant_key_version)


@dataclass(frozen=True, slots=True)
class AdminPredecessor:
    subject: RegistrySubject
    identity: EntitlementIdentity
    provenance: EntitlementProvenance
    lifecycle: EntitlementLifecycle
    authoritative_state_revision: int
    binding_identity_digest_sha256: str

    def __post_init__(self) -> None:
        validate_exact_snapshot(self.subject)
        validate_exact_snapshot(self.identity)
        validate_exact_snapshot(self.provenance)
        if type(self.lifecycle) is not EntitlementLifecycle:
            raise ContractValidationError("lifecycle must be exact EntitlementLifecycle")
        _positive("authoritative_state_revision", self.authoritative_state_revision)
        _digest("binding_identity_digest_sha256", self.binding_identity_digest_sha256)


@dataclass(frozen=True, slots=True)
class BindRequest:
    subject: RegistrySubject
    expected: BindPredecessor
    attempted_binding: BoundBinding

    def __post_init__(self) -> None:
        validate_exact_snapshot(self.subject)
        expected = validate_exact_snapshot(self.expected)
        binding = validate_exact_snapshot(self.attempted_binding)
        if not isinstance(expected, BindPredecessor) or not isinstance(binding, BoundBinding):
            raise ContractValidationError("invalid nested bind request")
        if expected.entitlement_generation != binding.entitlement_generation:
            raise ContractValidationError("bind request cannot change entitlement generation")
        validate_claimant_anchor_fields(
            expected.provisioning_principal_id,
            expected.claimant_key_id,
            expected.claimant_key_version,
            binding,
        )


@dataclass(frozen=True, slots=True)
class BindResult:
    outcome: BindOutcome
    authoritative_state: AuthoritativeEntitlementState | None

    def __post_init__(self) -> None:
        if type(self.outcome) is not BindOutcome:
            raise ContractValidationError("outcome must be exact BindOutcome, never bool")
        if self.outcome in (BindOutcome.NEW_BIND_COMMITTED, BindOutcome.EXACT_REPLAY):
            state = validate_exact_snapshot(self.authoritative_state)
            if (
                not isinstance(state, AuthoritativeEntitlementState)
                or type(state.binding) is not BoundBinding
                or state.lifecycle is not EntitlementLifecycle.ACTIVE
            ):
                raise ContractValidationError("successful bind requires exact ACTIVE/BOUND state")
        elif self.authoritative_state is not None:
            raise ContractValidationError("failure bind outcome cannot carry authoritative state")


@dataclass(frozen=True, slots=True)
class HistoricalStateResult:
    subject: RegistrySubject
    read: RegistryReadResult
    requested_authoritative_state_revision: int
    current_authoritative_state_revision: int | None
    retained_from_authoritative_state_revision: int | None

    def __post_init__(self) -> None:
        subject = validate_exact_snapshot(self.subject)
        read = validate_exact_snapshot(self.read)
        if not isinstance(subject, RegistrySubject) or not isinstance(read, RegistryReadResult):
            raise ContractValidationError("invalid historical result nesting")
        _positive(
            "requested_authoritative_state_revision", self.requested_authoritative_state_revision
        )
        if read.outcome is RegistryReadOutcome.FOUND:
            _positive(
                "current_authoritative_state_revision", self.current_authoritative_state_revision
            )
            _positive(
                "retained_from_authoritative_state_revision",
                self.retained_from_authoritative_state_revision,
            )
            assert self.current_authoritative_state_revision is not None
            assert self.retained_from_authoritative_state_revision is not None
            if (
                self.requested_authoritative_state_revision
                > self.current_authoritative_state_revision
            ):
                raise ContractValidationError(
                    "requested revision is beyond current subject revision"
                )
            if (
                self.retained_from_authoritative_state_revision
                > self.requested_authoritative_state_revision
            ):
                raise ContractValidationError("requested revision is outside retained history")
            assert read.state is not None
            if read.state.subject != subject:
                raise ContractValidationError("historical state belongs to another subject")
            if (
                read.state.authoritative_state_revision
                != self.requested_authoritative_state_revision
            ):
                raise ContractValidationError("historical state does not match requested revision")
        elif (
            self.current_authoritative_state_revision is not None
            or self.retained_from_authoritative_state_revision is not None
        ):
            raise ContractValidationError("non-FOUND history lookup cannot claim revision metadata")


@dataclass(frozen=True, slots=True)
class RetainedHistoryResult:
    outcome: RegistryReadOutcome
    subject: RegistrySubject
    states: tuple[AuthoritativeEntitlementState, ...]
    current_authoritative_state_revision: int | None
    retained_from_authoritative_state_revision: int | None

    def __post_init__(self) -> None:
        if type(self.outcome) is not RegistryReadOutcome:
            raise ContractValidationError("outcome must be exact RegistryReadOutcome")
        validate_exact_snapshot(self.subject)
        if type(self.states) is not tuple:
            raise ContractValidationError("history states must be exact tuple")
        if self.outcome is RegistryReadOutcome.FOUND:
            _positive(
                "current_authoritative_state_revision", self.current_authoritative_state_revision
            )
            _positive(
                "retained_from_authoritative_state_revision",
                self.retained_from_authoritative_state_revision,
            )
            assert self.current_authoritative_state_revision is not None
            assert self.retained_from_authoritative_state_revision is not None
            validate_complete_lineage(self.subject, self.states)
            if (
                self.states[-1].authoritative_state_revision
                != self.current_authoritative_state_revision
            ):
                raise ContractValidationError("history does not end at current subject revision")
            if (
                self.states[0].authoritative_state_revision
                != self.retained_from_authoritative_state_revision
            ):
                raise ContractValidationError(
                    "retained-from revision must identify first retained state"
                )
        elif (
            self.states
            or self.current_authoritative_state_revision is not None
            or self.retained_from_authoritative_state_revision is not None
        ):
            raise ContractValidationError(
                "non-FOUND history cannot carry states or revision metadata"
            )


@dataclass(frozen=True, slots=True)
class AuthoritativelyUnboundQuery:
    subject: RegistrySubject
    product_scope: str
    bootstrap_entitlement_id: str
    entitlement_generation: int
    logical_operation_id: str
    account_id: str
    canonical_genesis_request_fingerprint_sha256: str
    old_issuance_attempt_id: str

    def __post_init__(self) -> None:
        validate_exact_snapshot(self.subject)
        _text("product_scope", self.product_scope)
        for name in (
            "bootstrap_entitlement_id",
            "logical_operation_id",
            "account_id",
            "old_issuance_attempt_id",
        ):
            lookup = "issuance_attempt_id" if name == "old_issuance_attempt_id" else name
            _identifier(lookup, getattr(self, name))
        _positive("entitlement_generation", self.entitlement_generation)
        _digest(
            "canonical_genesis_request_fingerprint_sha256",
            self.canonical_genesis_request_fingerprint_sha256,
        )


@dataclass(frozen=True, slots=True)
class ProvisionEntitlementRequest:
    subject: RegistrySubject
    identity: EntitlementIdentity
    provenance: EntitlementProvenance

    def __post_init__(self) -> None:
        subject = validate_exact_snapshot(self.subject)
        identity = validate_exact_snapshot(self.identity)
        validate_exact_snapshot(self.provenance)
        if not isinstance(subject, RegistrySubject) or not isinstance(
            identity, EntitlementIdentity
        ):
            raise ContractValidationError("invalid provision request nesting")
        if identity.entitlement_generation != 1:
            raise ContractValidationError("initial provisioning requires generation 1")
        if (
            subject.environment != identity.environment
            or subject.trust_domain != identity.trust_domain
        ):
            raise ContractValidationError("provision subject and identity scope must match")


@dataclass(frozen=True, slots=True)
class SupersedeEntitlementRequest:
    expected: AdminPredecessor
    successor_identity: EntitlementIdentity
    successor_provenance: EntitlementProvenance

    def __post_init__(self) -> None:
        expected = validate_exact_snapshot(self.expected)
        successor = validate_exact_snapshot(self.successor_identity)
        validate_exact_snapshot(self.successor_provenance)
        if not isinstance(expected, AdminPredecessor) or not isinstance(
            successor, EntitlementIdentity
        ):
            raise ContractValidationError("invalid supersession nesting")
        prior = expected.identity
        if expected.lifecycle is not EntitlementLifecycle.ACTIVE:
            raise ContractValidationError("only ACTIVE generation can be superseded")
        if successor.entitlement_generation != prior.entitlement_generation + 1:
            raise ContractValidationError("supersession requires exactly next generation")
        immutable_scope = (
            "bootstrap_entitlement_id",
            "environment",
            "trust_domain",
            "product_scope",
            "intended_action",
        )
        if any(getattr(successor, name) != getattr(prior, name) for name in immutable_scope):
            raise ContractValidationError("supersession cannot change immutable lineage scope")


@dataclass(frozen=True, slots=True)
class RevokeEntitlementRequest:
    """Authenticated admin-port request; the exact predecessor is its decision input."""

    expected: AdminPredecessor

    def __post_init__(self) -> None:
        validate_exact_snapshot(self.expected)


@dataclass(frozen=True, slots=True)
class BindResolution:
    kind: BindResolutionKind
    historical_bound_state: AuthoritativeEntitlementState | None

    def __post_init__(self) -> None:
        if type(self.kind) is not BindResolutionKind:
            raise ContractValidationError("kind must be exact BindResolutionKind")
        if self.kind is BindResolutionKind.EXACT_REPLAY:
            state = validate_exact_snapshot(self.historical_bound_state)
            if (
                not isinstance(state, AuthoritativeEntitlementState)
                or type(state.binding) is not BoundBinding
            ):
                raise ContractValidationError("EXACT_REPLAY requires historical BOUND state")
        elif self.historical_bound_state is not None:
            raise ContractValidationError("only EXACT_REPLAY carries historical BOUND state")


@dataclass(frozen=True, slots=True)
class AdminResult:
    outcome: AdminOutcome
    state: AuthoritativeEntitlementState | None

    def __post_init__(self) -> None:
        if type(self.outcome) is not AdminOutcome:
            raise ContractValidationError("outcome must be exact AdminOutcome")
        if self.outcome is AdminOutcome.COMMITTED:
            validate_exact_snapshot(self.state)
        elif self.state is not None:
            raise ContractValidationError("non-COMMITTED admin outcome cannot carry state")


def validate_claimant_anchor_fields(
    provisioning_principal_id: str,
    claimant_key_id: str,
    claimant_key_version: int,
    binding: BoundBinding,
) -> None:
    _text("provisioning_principal_id", provisioning_principal_id)
    _text("claimant_key_id", claimant_key_id)
    _positive("claimant_key_version", claimant_key_version)
    validated = validate_exact_snapshot(binding)
    if not isinstance(validated, BoundBinding):
        raise ContractValidationError("claimant anchor requires exact BoundBinding")
    if (
        validated.provisioning_principal_id != provisioning_principal_id
        or validated.claimant_key_id != claimant_key_id
        or validated.claimant_key_version != claimant_key_version
    ):
        raise ContractValidationError("BOUND claimant anchor differs from provisioned authority")


def validate_claimant_anchor(provenance: EntitlementProvenance, binding: BoundBinding) -> None:
    trusted = validate_exact_snapshot(provenance)
    if not isinstance(trusted, EntitlementProvenance):
        raise ContractValidationError("claimant anchor requires exact provenance")
    validate_claimant_anchor_fields(
        trusted.provisioning_principal_id,
        trusted.claimant_key_id,
        trusted.claimant_key_version,
        binding,
    )


def validate_complete_lineage(
    subject: RegistrySubject,
    states: tuple[AuthoritativeEntitlementState, ...],
) -> None:
    trusted_subject = validate_exact_snapshot(subject)
    if not isinstance(trusted_subject, RegistrySubject) or type(states) is not tuple or not states:
        raise ContractValidationError("FOUND retained history requires a subject and states")
    trusted_states = tuple(validate_exact_snapshot(state) for state in states)
    if any(not isinstance(state, AuthoritativeEntitlementState) for state in trusted_states):
        raise ContractValidationError("history includes non-state value")
    first = trusted_states[0]
    assert isinstance(first, AuthoritativeEntitlementState)
    if (
        first.subject != trusted_subject
        or first.authoritative_state_revision != 1
        or first.predecessor_revision is not None
        or first.identity.entitlement_generation != 1
        or first.lifecycle is not EntitlementLifecycle.ACTIVE
        or type(first.binding) is not UnboundBinding
    ):
        raise ContractValidationError("history must begin with canonical generation-1 genesis")
    for previous, current in zip(trusted_states, trusted_states[1:]):
        assert isinstance(previous, AuthoritativeEntitlementState)
        assert isinstance(current, AuthoritativeEntitlementState)
        if current.subject != trusted_subject:
            raise ContractValidationError("history subject changed")
        if current.predecessor_revision != previous.authoritative_state_revision:
            raise ContractValidationError("history predecessor chain has a gap")
        if current.authoritative_state_revision != previous.authoritative_state_revision + 1:
            raise ContractValidationError("per-subject revision must increase by exactly one")
        prior_identity = previous.identity
        next_identity = current.identity
        immutable_scope = (
            "bootstrap_entitlement_id",
            "environment",
            "trust_domain",
            "product_scope",
            "intended_action",
        )
        if any(
            getattr(next_identity, name) != getattr(prior_identity, name)
            for name in immutable_scope
        ):
            raise ContractValidationError("immutable lineage scope changed")
        if next_identity.entitlement_generation == prior_identity.entitlement_generation:
            if current.provenance != previous.provenance:
                raise ContractValidationError("provenance changed within one generation")
            if current.lifecycle != previous.lifecycle:
                if not legal_lifecycle_transition(previous.lifecycle, current.lifecycle):
                    raise ContractValidationError("illegal lifecycle transition")
                if current.binding != previous.binding:
                    raise ContractValidationError("lifecycle mutation cannot change binding")
            elif (
                previous.lifecycle is EntitlementLifecycle.ACTIVE
                and type(previous.binding) is UnboundBinding
                and type(current.binding) is BoundBinding
            ):
                pass
            elif current.binding == previous.binding:
                raise ContractValidationError(
                    "authority revision cannot represent a semantic no-op"
                )
            else:
                raise ContractValidationError("illegal binding transition")
        else:
            if previous.lifecycle is not EntitlementLifecycle.SUPERSEDED:
                raise ContractValidationError("new generation requires superseded predecessor")
            if next_identity.entitlement_generation != prior_identity.entitlement_generation + 1:
                raise ContractValidationError("generation must increase by exactly one")
            if (
                current.lifecycle is not EntitlementLifecycle.ACTIVE
                or type(current.binding) is not UnboundBinding
            ):
                raise ContractValidationError("successor generation must start ACTIVE/UNBOUND")


def binding_identity_canonical_bytes(binding: RegistryBinding) -> bytes:
    trusted = validate_exact_snapshot(binding)
    if type(trusted) is UnboundBinding:
        payload: dict[str, object] = {"kind": trusted.kind.value}
    elif type(trusted) is BoundBinding:
        payload = {
            field.name: (
                getattr(trusted, field.name).value
                if isinstance(getattr(trusted, field.name), Enum)
                else getattr(trusted, field.name)
            )
            for field in fields(BoundBinding)
        }
    else:  # pragma: no cover - dispatcher is fail closed
        raise ContractValidationError("binding must be an exact binding variant")
    return canonical_json(payload).encode("utf-8")


def binding_identity_digest(binding: RegistryBinding) -> str:
    return hashlib.sha256(
        _BINDING_DIGEST_DOMAIN + binding_identity_canonical_bytes(binding)
    ).hexdigest()


def history_proves_authoritatively_unbound(
    history: RetainedHistoryResult,
    query: AuthoritativelyUnboundQuery,
    *,
    observed_current_revision: int,
) -> bool:
    trusted_history = validate_exact_snapshot(history)
    trusted_query = validate_exact_snapshot(query)
    if not isinstance(trusted_history, RetainedHistoryResult) or not isinstance(
        trusted_query, AuthoritativelyUnboundQuery
    ):
        raise ContractValidationError("invalid reconciliation inputs")
    _positive("observed_current_revision", observed_current_revision)
    if (
        trusted_history.outcome is not RegistryReadOutcome.FOUND
        or trusted_query.subject != trusted_history.subject
        or trusted_history.current_authoritative_state_revision != observed_current_revision
    ):
        return False
    current = trusted_history.states[-1]
    if (
        current.identity.environment != trusted_query.subject.environment
        or current.identity.trust_domain != trusted_query.subject.trust_domain
        or current.identity.product_scope != trusted_query.product_scope
        or current.identity.bootstrap_entitlement_id != trusted_query.bootstrap_entitlement_id
        or current.identity.entitlement_generation != trusted_query.entitlement_generation
        or type(current.binding) is not UnboundBinding
    ):
        return False
    return not any(
        type(state.binding) is BoundBinding
        and (
            state.binding.issuance_attempt_id == trusted_query.old_issuance_attempt_id
            or (
                state.binding.logical_operation_id == trusted_query.logical_operation_id
                and state.binding.account_id == trusted_query.account_id
                and state.binding.canonical_genesis_request_fingerprint_sha256
                == trusted_query.canonical_genesis_request_fingerprint_sha256
            )
        )
        for state in trusted_history.states
        if state.identity.entitlement_generation == trusted_query.entitlement_generation
    )


def authoritative_identity_key(identity: EntitlementIdentity) -> tuple[str, str, str, str]:
    """Return the frozen reverse-uniqueness scope for one authority identity."""

    trusted = validate_exact_snapshot(identity)
    if not isinstance(trusted, EntitlementIdentity):
        raise ContractValidationError("identity must be exact EntitlementIdentity")
    return (
        trusted.environment,
        trusted.trust_domain,
        trusted.product_scope,
        trusted.bootstrap_entitlement_id,
    )


def resolve_bind_request(
    history: RetainedHistoryResult,
    request: BindRequest,
) -> BindResolution:
    """Resolve a bind against validated lineage, preserving its historical winner."""

    trusted_history = validate_exact_snapshot(history)
    trusted_request = validate_exact_snapshot(request)
    if not isinstance(trusted_history, RetainedHistoryResult) or not isinstance(
        trusted_request, BindRequest
    ):
        raise ContractValidationError("invalid bind resolution values")
    if trusted_history.outcome is RegistryReadOutcome.NOT_FOUND:
        return BindResolution(BindResolutionKind.NOT_FOUND, None)
    if trusted_history.outcome is not RegistryReadOutcome.FOUND:
        raise ContractValidationError("unavailable/corrupt history cannot resolve a bind")
    if trusted_request.subject != trusted_history.subject:
        return BindResolution(BindResolutionKind.NOT_FOUND, None)

    winning_predecessor: AuthoritativeEntitlementState | None = None
    winning_successor: AuthoritativeEntitlementState | None = None
    for predecessor, successor in zip(trusted_history.states, trusted_history.states[1:]):
        if type(predecessor.binding) is UnboundBinding and type(successor.binding) is BoundBinding:
            winning_predecessor = predecessor
            winning_successor = successor
            break
    if winning_successor is not None:
        assert winning_predecessor is not None
        if trusted_request.attempted_binding != winning_successor.binding:
            return BindResolution(BindResolutionKind.CONFLICT_BOUND_TO_DIFFERENT_TUPLE, None)
        if trusted_request.expected != predecessor_for(winning_predecessor):
            return BindResolution(BindResolutionKind.STALE_PREDECESSOR, None)
        return BindResolution(BindResolutionKind.EXACT_REPLAY, winning_successor)

    current = trusted_history.states[-1]
    if trusted_request.expected != predecessor_for(current):
        return BindResolution(BindResolutionKind.STALE_PREDECESSOR, None)
    if current.lifecycle is EntitlementLifecycle.REVOKED:
        return BindResolution(BindResolutionKind.INACTIVE_REVOKED, None)
    if current.lifecycle is EntitlementLifecycle.SUPERSEDED:
        return BindResolution(BindResolutionKind.INACTIVE_SUPERSEDED, None)
    return BindResolution(BindResolutionKind.NEW_BIND_ELIGIBLE, None)


def initial_state_for(request: ProvisionEntitlementRequest) -> AuthoritativeEntitlementState:
    """Build the only legal authoritative genesis for a provision request."""

    trusted = validate_exact_snapshot(request)
    if not isinstance(trusted, ProvisionEntitlementRequest):
        raise ContractValidationError("request must be exact ProvisionEntitlementRequest")
    return AuthoritativeEntitlementState(
        trusted.subject,
        trusted.identity,
        trusted.provenance,
        EntitlementLifecycle.ACTIVE,
        UnboundBinding(),
        1,
        None,
    )


def supersession_states_for(
    current: AuthoritativeEntitlementState,
    request: SupersedeEntitlementRequest,
) -> tuple[AuthoritativeEntitlementState, AuthoritativeEntitlementState]:
    """Build the exact retired predecessor and ACTIVE/UNBOUND next generation."""

    trusted_current = validate_exact_snapshot(current)
    trusted_request = validate_exact_snapshot(request)
    if not isinstance(trusted_current, AuthoritativeEntitlementState) or not isinstance(
        trusted_request, SupersedeEntitlementRequest
    ):
        raise ContractValidationError("invalid supersession values")
    if admin_predecessor_for(trusted_current) != trusted_request.expected:
        raise ContractValidationError("supersession predecessor does not match current state")
    retired = AuthoritativeEntitlementState(
        trusted_current.subject,
        trusted_current.identity,
        trusted_current.provenance,
        EntitlementLifecycle.SUPERSEDED,
        trusted_current.binding,
        trusted_current.authoritative_state_revision + 1,
        trusted_current.authoritative_state_revision,
    )
    successor = AuthoritativeEntitlementState(
        trusted_current.subject,
        trusted_request.successor_identity,
        trusted_request.successor_provenance,
        EntitlementLifecycle.ACTIVE,
        UnboundBinding(),
        retired.authoritative_state_revision + 1,
        retired.authoritative_state_revision,
    )
    return retired, successor


def revoked_state_for(
    current: AuthoritativeEntitlementState,
    request: RevokeEntitlementRequest,
) -> AuthoritativeEntitlementState:
    """Build the sole legal ACTIVE -> REVOKED successor without changing binding."""

    trusted_current = validate_exact_snapshot(current)
    trusted_request = validate_exact_snapshot(request)
    if not isinstance(trusted_current, AuthoritativeEntitlementState) or not isinstance(
        trusted_request, RevokeEntitlementRequest
    ):
        raise ContractValidationError("invalid revocation values")
    if admin_predecessor_for(trusted_current) != trusted_request.expected:
        raise ContractValidationError("revocation predecessor does not match current state")
    if trusted_current.lifecycle is not EntitlementLifecycle.ACTIVE:
        raise ContractValidationError("only ACTIVE entitlement can be revoked")
    return AuthoritativeEntitlementState(
        trusted_current.subject,
        trusted_current.identity,
        trusted_current.provenance,
        EntitlementLifecycle.REVOKED,
        trusted_current.binding,
        trusted_current.authoritative_state_revision + 1,
        trusted_current.authoritative_state_revision,
    )


def validate_exact_snapshot(value: object) -> object:
    """Deeply reconstruct any public security record or fail with a controlled error."""

    try:
        if type(value) is RegistrySubject:
            return RegistrySubject(*_slots(value, RegistrySubject))
        if type(value) is EntitlementIdentity:
            return EntitlementIdentity(*_slots(value, EntitlementIdentity))
        if type(value) is EntitlementProvenance:
            return EntitlementProvenance(*_slots(value, EntitlementProvenance))
        if type(value) is UnboundBinding:
            return UnboundBinding(*_slots(value, UnboundBinding))
        if type(value) is BoundBinding:
            return BoundBinding(*_slots(value, BoundBinding))
        if type(value) is AuthoritativeEntitlementState:
            return AuthoritativeEntitlementState(*_slots(value, AuthoritativeEntitlementState))
        if type(value) is BindPredecessor:
            return BindPredecessor(*_slots(value, BindPredecessor))
        if type(value) is AdminPredecessor:
            return AdminPredecessor(*_slots(value, AdminPredecessor))
        if type(value) is BindRequest:
            return BindRequest(*_slots(value, BindRequest))
        if type(value) is RegistryReadResult:
            return RegistryReadResult(*_slots(value, RegistryReadResult))
        if type(value) is BindResult:
            return BindResult(*_slots(value, BindResult))
        if type(value) is HistoricalStateResult:
            return HistoricalStateResult(*_slots(value, HistoricalStateResult))
        if type(value) is RetainedHistoryResult:
            return RetainedHistoryResult(*_slots(value, RetainedHistoryResult))
        if type(value) is ProvisionEntitlementRequest:
            return ProvisionEntitlementRequest(*_slots(value, ProvisionEntitlementRequest))
        if type(value) is SupersedeEntitlementRequest:
            return SupersedeEntitlementRequest(*_slots(value, SupersedeEntitlementRequest))
        if type(value) is RevokeEntitlementRequest:
            return RevokeEntitlementRequest(*_slots(value, RevokeEntitlementRequest))
        if type(value) is BindResolution:
            return BindResolution(*_slots(value, BindResolution))
        if type(value) is AuthoritativelyUnboundQuery:
            return AuthoritativelyUnboundQuery(*_slots(value, AuthoritativelyUnboundQuery))
        if type(value) is AdminResult:
            return AdminResult(*_slots(value, AdminResult))
    except ContractValidationError:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        raise ContractValidationError("malformed fabricated contract value") from exc
    raise ContractValidationError("unsupported or non-exact contract value")


@runtime_checkable
class EntitlementProvisioningAdminProvider(Protocol):
    def provision_entitlement(self, request: ProvisionEntitlementRequest) -> AdminResult: ...
    def supersede_entitlement(self, request: SupersedeEntitlementRequest) -> AdminResult: ...
    def revoke_entitlement(self, request: RevokeEntitlementRequest) -> AdminResult: ...


def predecessor_for(state: AuthoritativeEntitlementState) -> BindPredecessor:
    trusted = validate_exact_snapshot(state)
    if not isinstance(trusted, AuthoritativeEntitlementState):
        raise ContractValidationError("state must be exact AuthoritativeEntitlementState")
    return BindPredecessor(
        trusted.identity.bootstrap_entitlement_id,
        trusted.identity.entitlement_generation,
        trusted.authoritative_state_revision,
        trusted.lifecycle,
        binding_identity_digest(trusted.binding),
        trusted.provenance.provisioning_principal_id,
        trusted.provenance.claimant_key_id,
        trusted.provenance.claimant_key_version,
    )


def admin_predecessor_for(state: AuthoritativeEntitlementState) -> AdminPredecessor:
    trusted = validate_exact_snapshot(state)
    if not isinstance(trusted, AuthoritativeEntitlementState):
        raise ContractValidationError("state must be exact AuthoritativeEntitlementState")
    return AdminPredecessor(
        trusted.subject,
        trusted.identity,
        trusted.provenance,
        trusted.lifecycle,
        trusted.authoritative_state_revision,
        binding_identity_digest(trusted.binding),
    )


__all__ = [
    "AdminOutcome",
    "AdminPredecessor",
    "AdminResult",
    "AuthoritativeEntitlementState",
    "AuthoritativelyUnboundQuery",
    "BindOutcome",
    "BindPredecessor",
    "BindRequest",
    "BindResolution",
    "BindResolutionKind",
    "BindResult",
    "BindingKind",
    "BoundBinding",
    "ContractValidationError",
    "EntitlementIdentity",
    "EntitlementLifecycle",
    "EntitlementProvenance",
    "EntitlementProvisioningAdminProvider",
    "HistoricalStateResult",
    "ProvisionEntitlementRequest",
    "RegistryReadOutcome",
    "RegistryReadResult",
    "RegistrySubject",
    "RetainedHistoryResult",
    "RevokeEntitlementRequest",
    "SupersedeEntitlementRequest",
    "UnboundBinding",
    "admin_predecessor_for",
    "authoritative_identity_key",
    "binding_identity_canonical_bytes",
    "binding_identity_digest",
    "history_proves_authoritatively_unbound",
    "initial_state_for",
    "legal_lifecycle_transition",
    "predecessor_for",
    "resolve_bind_request",
    "revoked_state_for",
    "supersession_states_for",
    "validate_claimant_anchor",
    "validate_complete_lineage",
    "validate_exact_snapshot",
]
