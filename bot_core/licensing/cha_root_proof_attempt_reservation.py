"""Stage 9 authorization and durable reservation from genuine public authorities.

Only the reviewed PostgreSQL composition may supply production authorization.
Registry ports and DTOs alone are not authority. This boundary stops at a local
reservation awaiting requester and claimant signatures.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Protocol, cast
from weakref import WeakKeyDictionary

from bot_core.cha_attempt_store import (
    AttemptAuthorization,
    AttemptState,
    CurrentAttempt,
    SQLiteCHAAttemptStore,
)
from bot_core.entitlement_registry_contract import (
    AuthoritativeEntitlementState,
    EntitlementLifecycle,
    RegistryReadOutcome,
    RegistryReadResult,
    RegistrySubject,
    UnboundBinding,
    validate_exact_snapshot,
)
from bot_core.postgresql_root_proof_issuance_authority import (
    PostgreSQLRootProofIssuanceAuthority,
    ProductionLocalIssuanceAuthorityError,
    _configured_root_proof_issuance_authority,
)
from bot_core.root_proof_issuer_substrate import (
    ClaimantIdentityRegistry,
    CredentialRoleIdentity,
    EntitlementRegistryProvider,
    ProviderCapabilities,
    ProviderIdentity,
    ProviderQualificationPolicy,
    ProviderRole,
    RequesterCredentialRegistry,
    RootProofIssuerCompositionGate,
    SecurityProfile,
    _ProviderQualificationSnapshot,
)

from .canonical import canonical_json_bytes, parse_canonical
from .cha_account_reservation import VerifiedAccountGenesisInitialBinding, _initial_binding_snapshot

_RESERVATION_RELATION_DOMAIN = b"CRYPTOHUNTER_STAGE9_INITIAL_BINDING_RESERVATION_RELATION_V1\x00"
_INITIAL_BINDING_REFERENCE_DOMAIN = (
    b"CRYPTOHUNTER_STAGE9_ROOT_PROOF_INITIAL_BINDING_REFERENCE_V1\x00"
)
_REQUESTER_ROLE = "ACCOUNT_GENESIS_ROOT_PROOF_ISSUANCE_REQUESTER_V1"
# Caller-supplied implementations and capability flags cannot register authority.
_TRUSTED_PROVIDER_TYPES: tuple[type, ...] = (PostgreSQLRootProofIssuanceAuthority,)


class RootProofAttemptReservationError(RuntimeError):
    """Exact live upstream, provider and retained reservation evidence is required."""


def _positive(value: object) -> None:
    if type(value) is not int or not 1 <= value <= 9_007_199_254_740_991:
        raise ValueError("provider version/revision must be a positive interoperable integer")


@dataclass(frozen=True, slots=True)
class _RequesterCredentialV1:
    requester_principal_id: str
    requester_credential_role: str
    requester_key_id: str
    requester_key_version: int
    lifecycle: str
    registry_revision: int

    def __post_init__(self) -> None:
        _validate_identity_record(self)
        if self.requester_credential_role != _REQUESTER_ROLE:
            raise ValueError("requester role mismatch")


@dataclass(frozen=True, slots=True)
class _ClaimantIdentityV1:
    provisioning_principal_id: str
    claimant_key_id: str
    claimant_key_version: int
    lifecycle: str
    registry_revision: int

    def __post_init__(self) -> None:
        _validate_identity_record(self)


def _validate_identity_record(record: _RequesterCredentialV1 | _ClaimantIdentityV1) -> None:
    for field in fields(record):
        value = getattr(record, field.name)
        if field.name.endswith("_version") or field.name == "registry_revision":
            _positive(value)
        elif type(value) is not str or not value.strip():
            raise ValueError("provider identity must be exact nonempty text")
    if record.lifecycle != "ACTIVE":
        raise ValueError("inactive identity cannot authorize a new reservation")


@dataclass(frozen=True, slots=True)
class _ProviderResolution:
    entitlement_subject: RegistrySubject
    requester_principal_id: str
    provisioning_principal_id: str

    def __post_init__(self) -> None:
        if type(self.entitlement_subject) is not RegistrySubject:
            raise TypeError("exact provider-originated registry subject required")
        validate_exact_snapshot(self.entitlement_subject)
        for value in (self.requester_principal_id, self.provisioning_principal_id):
            if type(value) is not str or not value.strip():
                raise ValueError("provider-originated identity lookup required")


class _IssuanceAuthorityProvider(Protocol):
    @property
    def entitlement_registry(self) -> EntitlementRegistryProvider: ...

    @property
    def requester_registry(self) -> RequesterCredentialRegistry: ...

    @property
    def claimant_registry(self) -> ClaimantIdentityRegistry: ...

    def requalify(self) -> object:
        """Return None after live substrate, privilege, lifecycle and durability checks."""
        ...

    def resolve_initial_binding(self, context_raw: bytes) -> _ProviderResolution: ...


def _issuance_authority_provider() -> _IssuanceAuthorityProvider:
    try:
        return _configured_root_proof_issuance_authority()
    except ProductionLocalIssuanceAuthorityError as exc:
        raise RootProofAttemptReservationError(str(exc)) from None


def _binding_context(binding: object) -> tuple[bytes, dict[str, Any]]:
    snapshot = _initial_binding_snapshot(binding)
    return snapshot.state_raw, _context_from_retained_binding(snapshot.state_raw)


def _context_from_retained_binding(raw: bytes) -> dict[str, Any]:
    """Identity computation only; raw bytes never establish capability provenance."""
    state = parse_canonical(raw)
    request = parse_canonical(bytes.fromhex(state["canonical_request_raw_hex"]))
    if request["environment"] != "PRODUCTION":
        raise RootProofAttemptReservationError("PRODUCTION_PROTOCOL_ENVIRONMENT_REQUIRED")
    relation = {
        "schema_version": "InitialBindingReservationRelationV1",
        "environment": request["environment"],
        "pdsa_trust_domain": state["pdsa_trust_domain"],
        "logical_operation_id": state["logical_operation_id"],
        "account_id": state["account_id"],
        "reservation_relation": request["reservation_relation"],
        "canonical_genesis_request_fingerprint_sha256": state["canonical_request_sha256"],
        "initial_binding_sha256": state["initial_binding_sha256"],
        "retained_initial_binding_sha256": hashlib.sha256(raw).hexdigest(),
    }
    reservation_identity = (
        "ibr_"
        + hashlib.sha256(_RESERVATION_RELATION_DOMAIN + canonical_json_bytes(relation)).hexdigest()
    )
    reference_payload = {
        **relation,
        "schema_version": "RootProofInitialBindingReferenceV1",
        "reservation_identity": reservation_identity,
    }
    digest = hashlib.sha256(
        _INITIAL_BINDING_REFERENCE_DOMAIN + canonical_json_bytes(reference_payload)
    ).hexdigest()
    return {
        **reference_payload,
        "product_scope": request["product_scope"],
        "initial_binding_reference": "initial-binding-v1:" + digest,
        "initial_binding_digest_sha256": digest,
    }


def _exact_record(value: object, record_type: type) -> Any:
    if type(value) is not record_type:
        raise TypeError("exact provider record type required")
    return record_type(
        *(object.__getattribute__(value, field.name) for field in fields(record_type))
    )


def _qualified_port(
    port: Any, role: ProviderRole, trust_domain: str
) -> _ProviderQualificationSnapshot:
    identity, capabilities = port.identity, port.capabilities
    if (
        type(identity) is not ProviderIdentity
        or type(capabilities) is not ProviderCapabilities
        or identity.role is not role
        or identity.security.trust_domain != trust_domain
        or ProviderQualificationPolicy().failures_for(
            SecurityProfile.PRODUCTION_LOCAL, identity, capabilities
        )
    ):
        raise RootProofAttemptReservationError("UNQUALIFIED_PRODUCTION_AUTHORIZATION_PROVIDER")
    credentials = port.credential_identities()
    if type(credentials) is not tuple:
        raise RootProofAttemptReservationError("EXACT_CREDENTIAL_EVIDENCE_REQUIRED")
    credentials = tuple(_exact_record(value, CredentialRoleIdentity) for value in credentials)
    snapshot = _ProviderQualificationSnapshot(port, identity, capabilities, credentials)
    if RootProofIssuerCompositionGate._credential_evidence_failures(snapshot):
        raise RootProofAttemptReservationError("CREDENTIAL_ROLE_OR_NAMESPACE_MISMATCH")
    return snapshot


def _resolve_from_provider(
    binding: object, provider: _IssuanceAuthorityProvider
) -> tuple[bytes, AttemptAuthorization, bytes]:
    if type(provider) not in _TRUSTED_PROVIDER_TYPES:
        raise RootProofAttemptReservationError("TRUSTED_PROVIDER_PROVENANCE_REQUIRED")
    if provider.requalify() is not None:
        raise RootProofAttemptReservationError("AUTHORIZATION_PROVIDER_REQUALIFICATION_FAILED")
    raw, context = _binding_context(binding)
    trust = context["pdsa_trust_domain"]
    entitlement_port = provider.entitlement_registry
    requester_port = provider.requester_registry
    claimant_port = provider.claimant_registry
    ports = {
        "entitlement": _qualified_port(entitlement_port, ProviderRole.ENTITLEMENT_REGISTRY, trust),
        "requester": _qualified_port(
            requester_port, ProviderRole.REQUESTER_CREDENTIAL_REGISTRY, trust
        ),
        "claimant": _qualified_port(claimant_port, ProviderRole.CLAIMANT_IDENTITY_REGISTRY, trust),
    }
    if RootProofIssuerCompositionGate._alias_failures(
        tuple(credential for port in ports.values() for credential in port.credentials)
    ):
        raise RootProofAttemptReservationError("FORBIDDEN_CREDENTIAL_ALIAS")
    resolution = _exact_record(
        provider.resolve_initial_binding(canonical_json_bytes(context)), _ProviderResolution
    )
    subject = cast(RegistrySubject, validate_exact_snapshot(resolution.entitlement_subject))
    if subject.environment != context["environment"] or subject.trust_domain != trust:
        raise RootProofAttemptReservationError("AUTHORIZATION_SCOPE_MISMATCH")
    result = entitlement_port.authoritative_state(subject)
    if (
        type(result) is not RegistryReadResult
        or type(result.state) is not AuthoritativeEntitlementState
    ):
        raise RootProofAttemptReservationError("AUTHORITATIVE_ENTITLEMENT_REQUIRED")
    result = cast(RegistryReadResult, validate_exact_snapshot(result))
    state = cast(AuthoritativeEntitlementState, result.state)
    requester = _exact_record(
        requester_port.active_requester_credential(resolution.requester_principal_id),
        _RequesterCredentialV1,
    )
    claimant = _exact_record(
        claimant_port.resolve_claimant(resolution.provisioning_principal_id),
        _ClaimantIdentityV1,
    )
    if (
        result.outcome is not RegistryReadOutcome.FOUND
        or state.subject != subject
        or state.lifecycle is not EntitlementLifecycle.ACTIVE
        or type(state.binding) is not UnboundBinding
        or state.identity.environment != context["environment"]
        or state.identity.trust_domain != trust
        or state.identity.product_scope != context["product_scope"]
        or requester.requester_principal_id != resolution.requester_principal_id
        or claimant.provisioning_principal_id != resolution.provisioning_principal_id
        or state.provenance.provisioning_principal_id != claimant.provisioning_principal_id
        or state.provenance.claimant_key_id != claimant.claimant_key_id
        or state.provenance.claimant_key_version != claimant.claimant_key_version
        or requester.requester_key_id == claimant.claimant_key_id
        or not any(
            credential.credential_identity == requester.requester_key_id
            for credential in ports["requester"].credentials
        )
        or not any(
            credential.credential_identity == claimant.claimant_key_id
            for credential in ports["claimant"].credentials
        )
    ):
        raise RootProofAttemptReservationError("INEXACT_PROVIDER_AUTHORIZATION")
    if type(provider) is PostgreSQLRootProofIssuanceAuthority:
        provider.validate_resolved_credentials(
            requester, claimant, ports["requester"].credentials, ports["claimant"].credentials
        )
    # Resolution comprises independent read-only authorities. Require the same
    # exact evidence on a second pass so a concurrent rotation or lifecycle
    # transition cannot combine captured old identities with a new ACTIVE DTO.
    for name, port, role in (
        ("entitlement", entitlement_port, ProviderRole.ENTITLEMENT_REGISTRY),
        ("requester", requester_port, ProviderRole.REQUESTER_CREDENTIAL_REGISTRY),
        ("claimant", claimant_port, ProviderRole.CLAIMANT_IDENTITY_REGISTRY),
    ):
        fresh = _qualified_port(port, role, trust)
        captured = ports[name]
        if (fresh.identity, fresh.capabilities, fresh.credentials) != (
            captured.identity,
            captured.capabilities,
            captured.credentials,
        ):
            raise RootProofAttemptReservationError(
                "AUTHORIZATION_EVIDENCE_CHANGED_DURING_RESOLUTION"
            )
    if (
        entitlement_port.authoritative_state(subject) != result
        or _exact_record(
            requester_port.active_requester_credential(resolution.requester_principal_id),
            _RequesterCredentialV1,
        )
        != requester
        or _exact_record(
            claimant_port.resolve_claimant(resolution.provisioning_principal_id),
            _ClaimantIdentityV1,
        )
        != claimant
    ):
        raise RootProofAttemptReservationError("AUTHORIZATION_EVIDENCE_CHANGED_DURING_RESOLUTION")
    evidence_raw = canonical_json_bytes(
        {
            "context": context,
            "resolution": asdict(resolution),
            "entitlement": asdict(state),
            "requester": asdict(requester),
            "claimant": asdict(claimant),
            "providers": {
                role: {
                    "identity": asdict(port.identity),
                    "capabilities": asdict(port.capabilities),
                    "credentials": [asdict(value) for value in port.credentials],
                }
                for role, port in ports.items()
            },
        }
    )
    authorization = AttemptAuthorization(
        environment=context["environment"],
        trust_domain=trust,
        product_scope=context["product_scope"],
        logical_operation_id=context["logical_operation_id"],
        account_id=context["account_id"],
        canonical_genesis_request_fingerprint_sha256=context[
            "canonical_genesis_request_fingerprint_sha256"
        ],
        initial_binding_reference=context["initial_binding_reference"],
        initial_binding_digest_sha256=context["initial_binding_digest_sha256"],
        bootstrap_entitlement_id=state.identity.bootstrap_entitlement_id,
        entitlement_generation=state.identity.entitlement_generation,
        requester_principal_id=requester.requester_principal_id,
        requester_credential_role=requester.requester_credential_role,
        requester_key_id=requester.requester_key_id,
        requester_key_version=requester.requester_key_version,
        provisioning_principal_id=claimant.provisioning_principal_id,
        claimant_key_id=claimant.claimant_key_id,
        claimant_key_version=claimant.claimant_key_version,
        reservation_identity=context["reservation_identity"],
        reservation_relation=context["reservation_relation"],
        initial_binding_sha256=context["initial_binding_sha256"],
        authorization_evidence_sha256=hashlib.sha256(evidence_raw).hexdigest(),
    )
    return raw, authorization, evidence_raw


@dataclass(frozen=True, slots=True)
class _AuthorizationSnapshot:
    binding: VerifiedAccountGenesisInitialBinding
    provider: _IssuanceAuthorityProvider
    state_raw: bytes
    authorization: AttemptAuthorization
    evidence_raw: bytes


class VerifiedRootProofIssuanceAuthorization:
    """Provider-originated eligible identity context; no signature authorization."""

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("authorization requires installed authoritative provider resolution")

    def __init_subclass__(cls) -> None:
        raise TypeError("authorization cannot be subclassed")

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("authorization is immutable")

    def __copy__(self) -> VerifiedRootProofIssuanceAuthorization:
        raise TypeError("authorization cannot be copied")

    def __deepcopy__(self, memo: object) -> VerifiedRootProofIssuanceAuthorization:
        raise TypeError("authorization cannot be copied")

    @property
    def authorization(self) -> AttemptAuthorization:
        return cast(
            AttemptAuthorization,
            _exact_record(_authorization_snapshot(self).authorization, AttemptAuthorization),
        )


_AUTHORIZATIONS: WeakKeyDictionary[
    VerifiedRootProofIssuanceAuthorization, _AuthorizationSnapshot
] = WeakKeyDictionary()


def _authorization_snapshot(value: object) -> _AuthorizationSnapshot:
    if type(value) is not VerifiedRootProofIssuanceAuthorization or value not in _AUTHORIZATIONS:
        raise RootProofAttemptReservationError("VERIFIED_ISSUANCE_AUTHORIZATION_REQUIRED")
    snapshot = _AUTHORIZATIONS[value]
    current_provider = _issuance_authority_provider()
    if current_provider is not snapshot.provider:
        raise RootProofAttemptReservationError("AUTHORIZATION_PROVIDER_CHANGED")
    raw, authorization, evidence = _resolve_from_provider(snapshot.binding, current_provider)
    if (raw, authorization, evidence) != (
        snapshot.state_raw,
        snapshot.authorization,
        snapshot.evidence_raw,
    ):
        raise RootProofAttemptReservationError("AUTHORIZATION_EVIDENCE_CHANGED")
    return snapshot


def require_verified_root_proof_issuance_authorization(
    value: object,
) -> VerifiedRootProofIssuanceAuthorization:
    _authorization_snapshot(value)
    return cast(VerifiedRootProofIssuanceAuthorization, value)


def resolve_root_proof_issuance_authorization(
    binding: object,
) -> VerifiedRootProofIssuanceAuthorization:
    _binding_context(binding)
    provider = _issuance_authority_provider()
    raw, authorization, evidence = _resolve_from_provider(binding, provider)
    result = object.__new__(VerifiedRootProofIssuanceAuthorization)
    _AUTHORIZATIONS[result] = _AuthorizationSnapshot(
        cast(VerifiedAccountGenesisInitialBinding, binding), provider, raw, authorization, evidence
    )
    return result


def _attempt_store_path() -> Path:
    from deployment.windows_production_cha_account_reservation import _state_path

    return cast(Path, _state_path().with_name("initial-cha-root-proof-attempts.sqlite3"))


def _open_store(trust_domain: str) -> SQLiteCHAAttemptStore:
    from deployment.windows_production_cha_account_reservation import _safe

    path = _attempt_store_path()
    if not path.is_absolute():
        raise RootProofAttemptReservationError("UNSAFE_ATTEMPT_STORE_PATH")
    _safe(path.parent, directory=True)
    for target in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
        _safe(target)
    return SQLiteCHAAttemptStore(path, trust_domain)


@dataclass(frozen=True, slots=True)
class _ReservationSnapshot:
    binding: VerifiedAccountGenesisInitialBinding
    authorization: VerifiedRootProofIssuanceAuthorization
    path: Path
    current: CurrentAttempt


class VerifiedRootProofIssuanceAttemptReservation:
    """Exact durable authority-owned reservation exists, awaiting both signatures."""

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("reservation comes only from the durable verifier")

    def __init_subclass__(cls) -> None:
        raise TypeError("reservation cannot be subclassed")

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("reservation is immutable")

    def __copy__(self) -> VerifiedRootProofIssuanceAttemptReservation:
        raise TypeError("reservation cannot be copied")

    def __deepcopy__(self, memo: object) -> VerifiedRootProofIssuanceAttemptReservation:
        raise TypeError("reservation cannot be copied")

    @property
    def issuance_attempt_id(self) -> str:
        return _reservation_snapshot(self).current.reservation.issuance_attempt_id

    @property
    def state(self) -> AttemptState:
        return _reservation_snapshot(self).current.state

    @property
    def fence(self) -> int:
        return _reservation_snapshot(self).current.fence

    @property
    def reservation_identity(self) -> str:
        return cast(
            str, _reservation_snapshot(self).current.reservation.authorization.reservation_identity
        )


_RESERVATIONS: WeakKeyDictionary[
    VerifiedRootProofIssuanceAttemptReservation, _ReservationSnapshot
] = WeakKeyDictionary()


def _exact_binding_authorization(binding: object, authorization: object) -> _AuthorizationSnapshot:
    snapshot = _authorization_snapshot(authorization)
    # The provider resolution just reverified this exact upstream capability.
    # A reconstructed capability must independently verify the same retained bytes.
    if binding is snapshot.binding:
        return snapshot
    raw, _ = _binding_context(binding)
    if raw != snapshot.state_raw:
        raise RootProofAttemptReservationError("AUTHORIZATION_INITIAL_BINDING_MISMATCH")
    return snapshot


def _reserved_current(current: CurrentAttempt, authorization: AttemptAuthorization) -> None:
    if (
        type(current) is not CurrentAttempt
        or current.state is not AttemptState.RESERVED_AWAITING_SIGNATURES
        or current.identity is not None
        or current.fence != 1
        or current.reservation.authorization != authorization
    ):
        raise RootProofAttemptReservationError("EXACT_CURRENT_RESERVED_ATTEMPT_REQUIRED")


def _reservation_snapshot(value: object) -> _ReservationSnapshot:
    if type(value) is not VerifiedRootProofIssuanceAttemptReservation or value not in _RESERVATIONS:
        raise RootProofAttemptReservationError("VERIFIED_ATTEMPT_RESERVATION_REQUIRED")
    snapshot = _RESERVATIONS[value]
    authorization = _exact_binding_authorization(
        snapshot.binding, snapshot.authorization
    ).authorization
    if snapshot.path != _attempt_store_path():
        raise RootProofAttemptReservationError("ATTEMPT_STORE_OWNER_CHANGED")
    with _open_store(authorization.trust_domain) as store:
        current = store.attempt(authorization.logical_operation_id)
    _reserved_current(current, authorization)
    if current != snapshot.current:
        raise RootProofAttemptReservationError("ATTEMPT_RESERVATION_CHANGED")
    return snapshot


def require_verified_root_proof_issuance_attempt_reservation(
    value: object,
) -> VerifiedRootProofIssuanceAttemptReservation:
    _reservation_snapshot(value)
    return cast(VerifiedRootProofIssuanceAttemptReservation, value)


def _issue_reservation(
    binding: object, authorization: object, current: CurrentAttempt
) -> VerifiedRootProofIssuanceAttemptReservation:
    result = object.__new__(VerifiedRootProofIssuanceAttemptReservation)
    _RESERVATIONS[result] = _ReservationSnapshot(
        cast(VerifiedAccountGenesisInitialBinding, binding),
        cast(VerifiedRootProofIssuanceAuthorization, authorization),
        _attempt_store_path(),
        current,
    )
    _reservation_snapshot(result)
    return result


def reserve_root_proof_issuance_attempt(
    binding: object, authorization: object
) -> VerifiedRootProofIssuanceAttemptReservation:
    snapshot = _exact_binding_authorization(binding, authorization)
    with _open_store(snapshot.authorization.trust_domain) as store:
        current = store.reserve_or_resolve_attempt_id(snapshot.authorization)
    _reserved_current(current, snapshot.authorization)
    return _issue_reservation(binding, authorization, current)


def load_root_proof_issuance_attempt(
    binding: object, authorization: object
) -> VerifiedRootProofIssuanceAttemptReservation:
    snapshot = _exact_binding_authorization(binding, authorization)
    with _open_store(snapshot.authorization.trust_domain) as store:
        current = store.attempt(snapshot.authorization.logical_operation_id)
    _reserved_current(current, snapshot.authorization)
    return _issue_reservation(binding, authorization, current)
