"""M0.5 Root-Proof Issuer provider contracts and fail-closed composition gate.

This module is deliberately a substrate, not an issuer implementation.  In
particular it contains no key storage, registry persistence, checkpoint
persistence, or issuance state machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Protocol, Sequence, runtime_checkable

from bot_core.entitlement_registry_contract import (
    BindRequest,
    BindResult,
    HistoricalStateResult,
    RetainedHistoryResult,
    RegistryReadResult,
    RegistrySubject,
)

if TYPE_CHECKING:
    from bot_core.cha_attempt_store import (
        AttemptAuthorization,
        AttemptIdentity,
        AuthoritativeUnboundEvidence,
        CurrentAttempt,
        RecoveryResolution,
    )


class SecurityProfile(str, Enum):
    DEVELOPMENT = "DEVELOPMENT"
    TEST = "TEST"
    PRODUCTION_LOCAL = "PRODUCTION_LOCAL"
    PRODUCTION_SERVER_READY = "PRODUCTION_SERVER_READY"


class ProviderRole(str, Enum):
    DEPLOYMENT_TRUST_ROOT = "DEPLOYMENT_TRUST_ROOT"
    ENTITLEMENT_REGISTRY = "ENTITLEMENT_REGISTRY"
    CLAIMANT_IDENTITY_REGISTRY = "CLAIMANT_IDENTITY_REGISTRY"
    REQUESTER_CREDENTIAL_REGISTRY = "REQUESTER_CREDENTIAL_REGISTRY"
    ROOT_PROOF_SIGNING = "ROOT_PROOF_SIGNING"
    HISTORY_ATTESTATION_SIGNING = "HISTORY_ATTESTATION_SIGNING"
    FRESHNESS_AUTHORITY_FINALIZATION_SIGNING = "FRESHNESS_AUTHORITY_FINALIZATION_SIGNING"
    CHA_FRESHNESS_PROPOSER_SIGNING = "CHA_FRESHNESS_PROPOSER_SIGNING"
    ISSUER_AUTHENTICATED_HISTORY = "ISSUER_AUTHENTICATED_HISTORY"
    CHECKPOINT_AUTHORITY = "CHECKPOINT_AUTHORITY"
    RECONCILIATION_EVIDENCE = "RECONCILIATION_EVIDENCE"
    CHA_ATTEMPT_STORE = "CHA_ATTEMPT_STORE"


class CredentialSemanticRole(str, Enum):
    ROOT_PROOF_REQUESTER = "ROOT_PROOF_REQUESTER"
    ROOT_PROOF_CLAIMANT = "ROOT_PROOF_CLAIMANT"
    ROOT_PROOF_ISSUER_SIGNING = "ROOT_PROOF_ISSUER_SIGNING"
    HISTORY_ATTESTATION_SIGNING = "HISTORY_ATTESTATION_SIGNING"
    ACCOUNT_GENESIS_FRESHNESS_AUTHORITY_FINALIZATION_SIGNING_V1 = (
        "ACCOUNT_GENESIS_FRESHNESS_AUTHORITY_FINALIZATION_SIGNING_V1"
    )
    ACCOUNT_GENESIS_FRESHNESS_PROPOSER_SIGNING_V1 = "ACCOUNT_GENESIS_FRESHNESS_PROPOSER_SIGNING_V1"
    # Retained for the already-frozen requester/claimant alias contract.  This
    # lineage role is deliberately not the local proposer signing role above.
    ACCOUNT_GENESIS_FRESHNESS_PROPOSER = "ACCOUNT_GENESIS_FRESHNESS_PROPOSER"
    CATALOG_AUTHORITY = "CATALOG_AUTHORITY"
    STORAGE_SECURITY_KEY = "STORAGE_SECURITY_KEY"


@dataclass(frozen=True, slots=True)
class SecurityProfileIdentity:
    profile: SecurityProfile
    trust_domain: str

    def __post_init__(self) -> None:
        if not _valid_security_profile_identity(self):
            raise TypeError("security profile identity has invalid runtime values")


@dataclass(frozen=True, slots=True)
class ProviderIdentity:
    role: ProviderRole
    security: SecurityProfileIdentity
    provider_namespace: str

    def __post_init__(self) -> None:
        if not _valid_provider_identity(self):
            raise TypeError("provider identity has invalid runtime values")


@dataclass(frozen=True, slots=True)
class CredentialRoleIdentity:
    semantic_role: CredentialSemanticRole
    credential_identity: str
    provider_namespace: str
    key_handle_or_version: str | None
    custody_lifecycle_namespace: str
    key_material_identity: str | None

    def __post_init__(self) -> None:
        if not _valid_credential_role_identity(self):
            raise TypeError("credential role identity has invalid runtime values")


_PUBLIC_KEY_FINGERPRINT_DOMAIN = b"CRYPT0HUNTER_ROOT_PROOF_CREDENTIAL_PUBLIC_KEY_V1\x00"


def public_key_material_identity(public_key: bytes) -> str:
    """Fingerprint canonical raw 32-byte Ed25519 public-key material."""

    if type(public_key) is not bytes or len(public_key) != 32:
        raise ValueError("canonical Ed25519 public key must be exactly 32 raw bytes")
    digest = hashlib.sha256(_PUBLIC_KEY_FINGERPRINT_DOMAIN + public_key).hexdigest()
    return f"sha256:{digest}"


@dataclass(frozen=True, slots=True)
class SigningCapabilities:
    ed25519: bool
    durable_key_identity: bool
    hardware_or_equivalent_secure_custody: bool
    plaintext_private_key_export_forbidden: bool
    role_isolation: bool
    lifecycle_support: bool
    stable_provider_namespace: bool
    stable_key_handle_or_version_identity: bool

    def __post_init__(self) -> None:
        if not _valid_signing_capabilities(self):
            raise TypeError("all signing capabilities must be exact bool values")


@dataclass(frozen=True, slots=True)
class CheckpointCapabilities:
    authenticated: bool
    monotonic: bool
    exact_history_head_binding: bool
    independent_rollback_domain: bool
    independent_admin_or_security_domain: bool
    retained_authenticated_history: bool
    historical_lookup_or_recovery: bool
    rollback_direction_proof: bool
    wall_clock_arbitration_forbidden: bool

    def __post_init__(self) -> None:
        if not _valid_checkpoint_capabilities(self):
            raise TypeError("all checkpoint capabilities must be exact bool values")


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """Provider-originated evidence; configuration cannot manufacture it."""

    implemented: bool
    authoritative_reads: bool = False
    durable_state: bool = False
    compare_and_swap: bool = False
    signing: SigningCapabilities | None = None
    checkpoint: CheckpointCapabilities | None = None

    def __post_init__(self) -> None:
        if not _valid_provider_capabilities(self):
            raise TypeError("provider capabilities contain invalid runtime evidence")


def _exact_nonempty_str(candidate: object) -> bool:
    return type(candidate) is str and bool(candidate.strip())


def _valid_security_profile_identity(candidate: object) -> bool:
    try:
        return (
            type(candidate) is SecurityProfileIdentity
            and type(candidate.profile) is SecurityProfile
            and _exact_nonempty_str(candidate.trust_domain)
        )
    except (AttributeError, TypeError):
        return False


def _valid_provider_identity(candidate: object) -> bool:
    try:
        return (
            type(candidate) is ProviderIdentity
            and type(candidate.role) is ProviderRole
            and _valid_security_profile_identity(candidate.security)
            and _exact_nonempty_str(candidate.provider_namespace)
        )
    except (AttributeError, TypeError):
        return False


def _valid_credential_role_identity(candidate: object) -> bool:
    try:
        return (
            type(candidate) is CredentialRoleIdentity
            and type(candidate.semantic_role) is CredentialSemanticRole
            and _exact_nonempty_str(candidate.credential_identity)
            and _exact_nonempty_str(candidate.provider_namespace)
            and _exact_nonempty_str(candidate.custody_lifecycle_namespace)
            and (
                candidate.key_handle_or_version is None
                or _exact_nonempty_str(candidate.key_handle_or_version)
            )
            and (
                candidate.key_material_identity is None
                or _exact_nonempty_str(candidate.key_material_identity)
            )
        )
    except (AttributeError, TypeError):
        return False


def _valid_signing_capabilities(candidate: object) -> bool:
    try:
        return type(candidate) is SigningCapabilities and all(
            type(value) is bool
            for value in (
                candidate.ed25519,
                candidate.durable_key_identity,
                candidate.hardware_or_equivalent_secure_custody,
                candidate.plaintext_private_key_export_forbidden,
                candidate.role_isolation,
                candidate.lifecycle_support,
                candidate.stable_provider_namespace,
                candidate.stable_key_handle_or_version_identity,
            )
        )
    except (AttributeError, TypeError):
        return False


def _valid_checkpoint_capabilities(candidate: object) -> bool:
    try:
        return type(candidate) is CheckpointCapabilities and all(
            type(value) is bool
            for value in (
                candidate.authenticated,
                candidate.monotonic,
                candidate.exact_history_head_binding,
                candidate.independent_rollback_domain,
                candidate.independent_admin_or_security_domain,
                candidate.retained_authenticated_history,
                candidate.historical_lookup_or_recovery,
                candidate.rollback_direction_proof,
                candidate.wall_clock_arbitration_forbidden,
            )
        )
    except (AttributeError, TypeError):
        return False


def _valid_provider_capabilities(candidate: object) -> bool:
    try:
        return (
            type(candidate) is ProviderCapabilities
            and all(
                type(value) is bool
                for value in (
                    candidate.implemented,
                    candidate.authoritative_reads,
                    candidate.durable_state,
                    candidate.compare_and_swap,
                )
            )
            and (candidate.signing is None or _valid_signing_capabilities(candidate.signing))
            and (
                candidate.checkpoint is None or _valid_checkpoint_capabilities(candidate.checkpoint)
            )
        )
    except (AttributeError, TypeError):
        return False


@runtime_checkable
class SecurityProvider(Protocol):
    @property
    def identity(self) -> ProviderIdentity: ...

    @property
    def capabilities(self) -> ProviderCapabilities: ...

    def credential_identities(self) -> tuple[CredentialRoleIdentity, ...]: ...


@runtime_checkable
class DeploymentTrustRootProvider(SecurityProvider, Protocol):
    def active_bundle(self) -> bytes: ...
    def verify_signed_successor(self, candidate: bytes) -> bool: ...


@runtime_checkable
class EntitlementRegistryProvider(SecurityProvider, Protocol):
    """Least-privilege runtime port; provisioning/admin is a separate authority."""

    def authoritative_state(self, subject: RegistrySubject) -> RegistryReadResult: ...
    def compare_and_swap_bind(self, request: BindRequest) -> BindResult: ...
    def state_at_revision(
        self, subject: RegistrySubject, authoritative_state_revision: int
    ) -> HistoricalStateResult: ...
    def retained_history(self, subject: RegistrySubject) -> RetainedHistoryResult: ...


@runtime_checkable
class ClaimantIdentityRegistry(SecurityProvider, Protocol):
    def resolve_claimant(self, claimant_id: str) -> object: ...
    def historical_claimant(self, claimant_id: str, generation: int) -> object: ...


@runtime_checkable
class RequesterCredentialRegistry(SecurityProvider, Protocol):
    def active_requester_credential(self, requester_id: str) -> object: ...
    def historical_requester_credential(self, credential_id: str) -> object: ...


@runtime_checkable
class SigningIdentityProvider(SecurityProvider, Protocol):
    def active_credential_identity(self) -> CredentialRoleIdentity: ...
    def public_key(self, credential_identity: str) -> bytes:
        """Return the canonical raw 32-byte Ed25519 public key."""
        ...

    def lifecycle_generation(self) -> int: ...


@runtime_checkable
class RootProofSigningProvider(SigningIdentityProvider, Protocol):
    def sign_root_proof(self, canonical_payload: bytes) -> bytes: ...


@runtime_checkable
class HistoryAttestationSigningProvider(SigningIdentityProvider, Protocol):
    """Independent signing authority for authenticated history heads."""

    def sign_history_head(self, canonical_head: bytes) -> bytes: ...


@runtime_checkable
class FreshnessAuthorityFinalizationSigningProvider(SigningIdentityProvider, Protocol):
    def sign_finalization(self, canonical_payload: bytes) -> object: ...


@runtime_checkable
class CHAFreshnessProposerSigningProvider(SigningIdentityProvider, Protocol):
    def sign_freshness_proposal(self, canonical_payload: bytes) -> object: ...


@runtime_checkable
class IssuerAuthenticatedHistory(SecurityProvider, Protocol):
    def current_head(self) -> object: ...
    def append_exact_successor(self, expected_head: object, record: object) -> object: ...
    def record_at(self, sequence: int) -> object: ...


@runtime_checkable
class CheckpointAuthorityProvider(SecurityProvider, Protocol):
    def current_checkpoint(self) -> object: ...
    def advance_exact_successor(self, expected: object, successor: object) -> bool: ...
    def authenticated_checkpoint_at(self, sequence: int) -> object: ...


@runtime_checkable
class RootProofReconciliationEvidenceSource(SecurityProvider, Protocol):
    def evidence_for(self, subject_id: str, history_head: bytes) -> object: ...
    def verify_evidence(self, evidence: object) -> bool: ...


@runtime_checkable
class CHAAttemptStore(SecurityProvider, Protocol):
    def reserve_or_resolve_attempt_id(
        self, authorization: AttemptAuthorization
    ) -> CurrentAttempt: ...
    def finalize_attempt(
        self, identity: AttemptIdentity, *, expected_fence: int
    ) -> CurrentAttempt: ...
    def replace_after_authoritative_unbound(
        self,
        authorization: AttemptAuthorization,
        evidence: AuthoritativeUnboundEvidence,
        *,
        expected_fence: int,
    ) -> CurrentAttempt: ...
    def record_recovery_resolution(
        self,
        operation_id: str,
        resolution: RecoveryResolution,
        *,
        expected_fence: int,
    ) -> CurrentAttempt: ...
    def attempt(self, operation_id: str) -> CurrentAttempt: ...


@dataclass(frozen=True, slots=True)
class RootProofIssuerSubstrateConfig:
    security: SecurityProfileIdentity
    provider_implementations: tuple[tuple[ProviderRole, str], ...]

    def __post_init__(self) -> None:
        if not _valid_security_profile_identity(self.security):
            raise TypeError("security contains invalid nested identity evidence")
        if type(self.provider_implementations) is not tuple:
            raise TypeError("provider_implementations must be an exact tuple")
        for entry in self.provider_implementations:
            if type(entry) is not tuple or len(entry) != 2:
                raise TypeError("each provider implementation entry must be an exact pair tuple")
            role, selection = entry
            if type(role) is not ProviderRole:
                raise TypeError("provider implementation role must be an exact ProviderRole")
            if type(selection) is not str:
                raise TypeError("provider implementation selection must be an exact str")
            if not selection.strip():
                raise ValueError("provider implementation selection cannot be blank")
        if len({role for role, _ in self.provider_implementations}) != len(
            self.provider_implementations
        ):
            raise ValueError("provider implementation roles must be unique")


class QualificationFailureCode(str, Enum):
    INVALID_COMPOSITION_SECURITY_IDENTITY = "INVALID_COMPOSITION_SECURITY_IDENTITY"
    MISSING_PROVIDER = "MISSING_PROVIDER"
    DUPLICATE_PROVIDER_ROLE = "DUPLICATE_PROVIDER_ROLE"
    PROFILE_IDENTITY_MISMATCH = "PROFILE_IDENTITY_MISMATCH"
    PROVIDER_NOT_IMPLEMENTED = "PROVIDER_NOT_IMPLEMENTED"
    PROVIDER_INTERFACE_MISMATCH = "PROVIDER_INTERFACE_MISMATCH"
    SIGNING_AUTHORITY_INTERFACE_ALIAS = "SIGNING_AUTHORITY_INTERFACE_ALIAS"
    INSUFFICIENT_CAPABILITIES = "INSUFFICIENT_CAPABILITIES"
    MISSING_CREDENTIAL_IDENTITY_EVIDENCE = "MISSING_CREDENTIAL_IDENTITY_EVIDENCE"
    CREDENTIAL_ROLE_MISMATCH = "CREDENTIAL_ROLE_MISMATCH"
    CREDENTIAL_NAMESPACE_MISMATCH = "CREDENTIAL_NAMESPACE_MISMATCH"
    MISSING_KEY_HANDLE_IDENTITY = "MISSING_KEY_HANDLE_IDENTITY"
    AMBIGUOUS_CREDENTIAL_IDENTITY_EVIDENCE = "AMBIGUOUS_CREDENTIAL_IDENTITY_EVIDENCE"
    ACTIVE_CREDENTIAL_IDENTITY_MISMATCH = "ACTIVE_CREDENTIAL_IDENTITY_MISMATCH"
    MISSING_KEY_MATERIAL_IDENTITY = "MISSING_KEY_MATERIAL_IDENTITY"
    KEY_MATERIAL_IDENTITY_MISMATCH = "KEY_MATERIAL_IDENTITY_MISMATCH"
    CREDENTIAL_IDENTITY_ALIAS = "CREDENTIAL_IDENTITY_ALIAS"
    KEY_HANDLE_ALIAS = "KEY_HANDLE_ALIAS"
    KEY_MATERIAL_ALIAS = "KEY_MATERIAL_ALIAS"
    PROVIDER_EVIDENCE_UNAVAILABLE = "PROVIDER_EVIDENCE_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class QualificationFailure:
    code: QualificationFailureCode
    role: ProviderRole | None
    detail: str


@dataclass(frozen=True, slots=True)
class QualificationResult:
    failures: tuple[QualificationFailure, ...]

    @property
    def qualified(self) -> bool:
        return not self.failures


_FORBIDDEN_ROLE_PAIRS = {
    frozenset(
        (CredentialSemanticRole.ROOT_PROOF_REQUESTER, CredentialSemanticRole.ROOT_PROOF_CLAIMANT)
    ),
    frozenset(
        (
            CredentialSemanticRole.ROOT_PROOF_REQUESTER,
            CredentialSemanticRole.ROOT_PROOF_ISSUER_SIGNING,
        )
    ),
    frozenset(
        (
            CredentialSemanticRole.ROOT_PROOF_CLAIMANT,
            CredentialSemanticRole.ROOT_PROOF_ISSUER_SIGNING,
        )
    ),
    frozenset(
        (
            CredentialSemanticRole.ROOT_PROOF_ISSUER_SIGNING,
            CredentialSemanticRole.HISTORY_ATTESTATION_SIGNING,
        )
    ),
    frozenset(
        (
            CredentialSemanticRole.ROOT_PROOF_REQUESTER,
            CredentialSemanticRole.ACCOUNT_GENESIS_FRESHNESS_PROPOSER,
        )
    ),
    frozenset(
        (CredentialSemanticRole.ROOT_PROOF_REQUESTER, CredentialSemanticRole.CATALOG_AUTHORITY)
    ),
    frozenset(
        (CredentialSemanticRole.ROOT_PROOF_REQUESTER, CredentialSemanticRole.STORAGE_SECURITY_KEY)
    ),
    frozenset(
        (
            CredentialSemanticRole.ROOT_PROOF_ISSUER_SIGNING,
            CredentialSemanticRole.ACCOUNT_GENESIS_FRESHNESS_PROPOSER,
        )
    ),
    frozenset(
        (CredentialSemanticRole.ROOT_PROOF_ISSUER_SIGNING, CredentialSemanticRole.CATALOG_AUTHORITY)
    ),
    *(
        frozenset((freshness, other))
        for freshness in (
            CredentialSemanticRole.ACCOUNT_GENESIS_FRESHNESS_AUTHORITY_FINALIZATION_SIGNING_V1,
            CredentialSemanticRole.ACCOUNT_GENESIS_FRESHNESS_PROPOSER_SIGNING_V1,
        )
        for other in (
            CredentialSemanticRole.ACCOUNT_GENESIS_FRESHNESS_AUTHORITY_FINALIZATION_SIGNING_V1,
            CredentialSemanticRole.ACCOUNT_GENESIS_FRESHNESS_PROPOSER_SIGNING_V1,
            CredentialSemanticRole.ROOT_PROOF_ISSUER_SIGNING,
            CredentialSemanticRole.HISTORY_ATTESTATION_SIGNING,
            CredentialSemanticRole.CATALOG_AUTHORITY,
            CredentialSemanticRole.STORAGE_SECURITY_KEY,
        )
        if freshness is not other
    ),
}

_ROLE_PORTS: Mapping[ProviderRole, type[SecurityProvider]] = MappingProxyType(
    {
        ProviderRole.DEPLOYMENT_TRUST_ROOT: DeploymentTrustRootProvider,
        ProviderRole.ENTITLEMENT_REGISTRY: EntitlementRegistryProvider,
        ProviderRole.CLAIMANT_IDENTITY_REGISTRY: ClaimantIdentityRegistry,
        ProviderRole.REQUESTER_CREDENTIAL_REGISTRY: RequesterCredentialRegistry,
        ProviderRole.ROOT_PROOF_SIGNING: RootProofSigningProvider,
        ProviderRole.HISTORY_ATTESTATION_SIGNING: HistoryAttestationSigningProvider,
        ProviderRole.FRESHNESS_AUTHORITY_FINALIZATION_SIGNING: FreshnessAuthorityFinalizationSigningProvider,
        ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING: CHAFreshnessProposerSigningProvider,
        ProviderRole.ISSUER_AUTHENTICATED_HISTORY: IssuerAuthenticatedHistory,
        ProviderRole.CHECKPOINT_AUTHORITY: CheckpointAuthorityProvider,
        ProviderRole.RECONCILIATION_EVIDENCE: RootProofReconciliationEvidenceSource,
        ProviderRole.CHA_ATTEMPT_STORE: CHAAttemptStore,
    }
)

_ROLE_METHODS: Mapping[ProviderRole, tuple[str, ...]] = MappingProxyType(
    {
        ProviderRole.DEPLOYMENT_TRUST_ROOT: ("active_bundle", "verify_signed_successor"),
        ProviderRole.ENTITLEMENT_REGISTRY: (
            "authoritative_state",
            "compare_and_swap_bind",
            "state_at_revision",
            "retained_history",
        ),
        ProviderRole.CLAIMANT_IDENTITY_REGISTRY: ("resolve_claimant", "historical_claimant"),
        ProviderRole.REQUESTER_CREDENTIAL_REGISTRY: (
            "active_requester_credential",
            "historical_requester_credential",
        ),
        ProviderRole.ROOT_PROOF_SIGNING: (
            "active_credential_identity",
            "public_key",
            "sign_root_proof",
            "lifecycle_generation",
        ),
        ProviderRole.HISTORY_ATTESTATION_SIGNING: (
            "active_credential_identity",
            "public_key",
            "lifecycle_generation",
            "sign_history_head",
        ),
        ProviderRole.FRESHNESS_AUTHORITY_FINALIZATION_SIGNING: (
            "active_credential_identity",
            "public_key",
            "lifecycle_generation",
            "sign_finalization",
        ),
        ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING: (
            "active_credential_identity",
            "public_key",
            "lifecycle_generation",
            "sign_freshness_proposal",
        ),
        ProviderRole.ISSUER_AUTHENTICATED_HISTORY: (
            "current_head",
            "append_exact_successor",
            "record_at",
        ),
        ProviderRole.CHECKPOINT_AUTHORITY: (
            "current_checkpoint",
            "advance_exact_successor",
            "authenticated_checkpoint_at",
        ),
        ProviderRole.RECONCILIATION_EVIDENCE: ("evidence_for", "verify_evidence"),
        ProviderRole.CHA_ATTEMPT_STORE: (
            "reserve_or_resolve_attempt_id",
            "finalize_attempt",
            "replace_after_authoritative_unbound",
            "record_recovery_resolution",
            "attempt",
        ),
    }
)

_CREDENTIAL_ROLES: Mapping[ProviderRole, CredentialSemanticRole] = MappingProxyType(
    {
        ProviderRole.REQUESTER_CREDENTIAL_REGISTRY: CredentialSemanticRole.ROOT_PROOF_REQUESTER,
        ProviderRole.CLAIMANT_IDENTITY_REGISTRY: CredentialSemanticRole.ROOT_PROOF_CLAIMANT,
        ProviderRole.ROOT_PROOF_SIGNING: CredentialSemanticRole.ROOT_PROOF_ISSUER_SIGNING,
        ProviderRole.HISTORY_ATTESTATION_SIGNING: CredentialSemanticRole.HISTORY_ATTESTATION_SIGNING,
        ProviderRole.FRESHNESS_AUTHORITY_FINALIZATION_SIGNING: CredentialSemanticRole.ACCOUNT_GENESIS_FRESHNESS_AUTHORITY_FINALIZATION_SIGNING_V1,
        ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING: CredentialSemanticRole.ACCOUNT_GENESIS_FRESHNESS_PROPOSER_SIGNING_V1,
    }
)

_SIGNING_ROLES = {
    ProviderRole.ROOT_PROOF_SIGNING,
    ProviderRole.HISTORY_ATTESTATION_SIGNING,
    ProviderRole.FRESHNESS_AUTHORITY_FINALIZATION_SIGNING,
    ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING,
}

_ROOT_PROOF_MANDATORY_PROVIDER_ROLES = tuple(
    role
    for role in ProviderRole
    if role
    not in {
        ProviderRole.FRESHNESS_AUTHORITY_FINALIZATION_SIGNING,
        ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING,
    }
)

_ROLE_CAPABILITIES: Mapping[ProviderRole, tuple[str, ...]] = MappingProxyType(
    {
        ProviderRole.DEPLOYMENT_TRUST_ROOT: ("authoritative_reads", "durable_state"),
        ProviderRole.ENTITLEMENT_REGISTRY: (
            "authoritative_reads",
            "durable_state",
            "compare_and_swap",
        ),
        ProviderRole.CLAIMANT_IDENTITY_REGISTRY: ("authoritative_reads", "durable_state"),
        ProviderRole.REQUESTER_CREDENTIAL_REGISTRY: ("authoritative_reads", "durable_state"),
        ProviderRole.ROOT_PROOF_SIGNING: (),
        ProviderRole.HISTORY_ATTESTATION_SIGNING: (),
        ProviderRole.FRESHNESS_AUTHORITY_FINALIZATION_SIGNING: (),
        ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING: (),
        ProviderRole.ISSUER_AUTHENTICATED_HISTORY: ("authoritative_reads", "durable_state"),
        ProviderRole.CHECKPOINT_AUTHORITY: ("durable_state",),
        ProviderRole.RECONCILIATION_EVIDENCE: ("authoritative_reads",),
        ProviderRole.CHA_ATTEMPT_STORE: (
            "authoritative_reads",
            "durable_state",
            "compare_and_swap",
        ),
    }
)


@dataclass(frozen=True, slots=True)
class _ProviderQualificationSnapshot:
    provider: object
    identity: ProviderIdentity
    capabilities: ProviderCapabilities
    credentials: tuple[CredentialRoleIdentity, ...]
    active_credential: CredentialRoleIdentity | None = None
    observed_key_material_identity: str | None = None


def _provider_qualification_failures(
    profile: SecurityProfile,
    identity: ProviderIdentity,
    evidence: ProviderCapabilities,
) -> tuple[str, ...]:
    """Apply the single frozen provider qualification policy implementation."""

    if type(profile) is not SecurityProfile:
        return ("security profile evidence has invalid type",)
    if not _valid_provider_identity(identity) or not _valid_provider_capabilities(evidence):
        return ("provider identity or capability evidence has invalid type",)
    if identity.security.profile is not profile:
        return ("target security profile does not match provider identity",)
    failures: list[str] = []
    if not evidence.implemented:
        failures.append("provider implementation is unavailable")
    role = identity.role
    missing = [name for name in _ROLE_CAPABILITIES[role] if not getattr(evidence, name)]
    if missing:
        failures.append(f"required foundation capabilities are absent: {', '.join(missing)}")
    if role in _SIGNING_ROLES:
        signing = evidence.signing
        if signing is None:
            failures.append("signing capability evidence is absent")
        else:
            common = (
                signing.ed25519,
                signing.durable_key_identity,
                signing.role_isolation,
                signing.lifecycle_support,
                signing.stable_provider_namespace,
                signing.stable_key_handle_or_version_identity,
            )
            if not all(common):
                failures.append("required signing capability is absent")
            if profile is SecurityProfile.PRODUCTION_SERVER_READY and not (
                signing.hardware_or_equivalent_secure_custody
                and signing.plaintext_private_key_export_forbidden
            ):
                failures.append("server-ready signing custody is insufficient")
    if role is ProviderRole.CHECKPOINT_AUTHORITY:
        checkpoint = evidence.checkpoint
        if checkpoint is None:
            failures.append("checkpoint capability evidence is absent")
        else:
            common = (
                checkpoint.authenticated,
                checkpoint.monotonic,
                checkpoint.exact_history_head_binding,
                checkpoint.retained_authenticated_history,
                checkpoint.historical_lookup_or_recovery,
                checkpoint.rollback_direction_proof,
                checkpoint.wall_clock_arbitration_forbidden,
            )
            if not all(common):
                failures.append("required checkpoint capability is absent")
            if profile is SecurityProfile.PRODUCTION_SERVER_READY and not (
                checkpoint.independent_rollback_domain
                and checkpoint.independent_admin_or_security_domain
            ):
                failures.append("server-ready checkpoint independence is insufficient")
    return tuple(failures)


class ProviderQualificationPolicy:
    """Stateless public wrapper over the frozen qualification policy."""

    __slots__ = ()

    def failures_for(
        self,
        profile: SecurityProfile,
        identity: ProviderIdentity,
        evidence: ProviderCapabilities,
    ) -> tuple[str, ...]:
        return _provider_qualification_failures(profile, identity, evidence)


class RootProofIssuerCompositionGate:
    __slots__ = ()

    def __init__(self) -> None:
        """Create a stateless gate with no policy replacement seam."""

    def qualify(
        self, security: SecurityProfileIdentity, providers: Sequence[SecurityProvider]
    ) -> QualificationResult:
        try:
            valid_security = self._valid_security_identity(security)
        except (AttributeError, TypeError):
            valid_security = False
        if not valid_security:
            return QualificationResult(
                (
                    QualificationFailure(
                        QualificationFailureCode.INVALID_COMPOSITION_SECURITY_IDENTITY,
                        None,
                        "composition security identity is invalid",
                    ),
                )
            )
        failures: list[QualificationFailure] = []
        snapshots: list[_ProviderQualificationSnapshot] = []
        for provider in providers:
            snapshot, capture_failure = self._capture_snapshot(provider)
            if capture_failure is not None:
                failures.append(capture_failure)
            if snapshot is not None:
                snapshots.append(snapshot)
        by_role: dict[ProviderRole, list[_ProviderQualificationSnapshot]] = {}
        for snapshot in snapshots:
            by_role.setdefault(snapshot.identity.role, []).append(snapshot)
        for role in _ROOT_PROOF_MANDATORY_PROVIDER_ROLES:
            matches = by_role.get(role, [])
            if not matches:
                failures.append(
                    QualificationFailure(
                        QualificationFailureCode.MISSING_PROVIDER,
                        role,
                        "mandatory provider is absent",
                    )
                )
            elif len(matches) > 1:
                failures.append(
                    QualificationFailure(
                        QualificationFailureCode.DUPLICATE_PROVIDER_ROLE,
                        role,
                        "semantic provider role occurs more than once",
                    )
                )
        for snapshot in snapshots:
            role = snapshot.identity.role
            try:
                expected_port_satisfied = isinstance(snapshot.provider, _ROLE_PORTS[role])
            except Exception:
                expected_port_satisfied = False
            methods_satisfied = all(
                callable(getattr(snapshot.provider, method, None)) for method in _ROLE_METHODS[role]
            )
            if not expected_port_satisfied or not methods_satisfied:
                failures.append(
                    QualificationFailure(
                        QualificationFailureCode.PROVIDER_INTERFACE_MISMATCH,
                        role,
                        "provider does not implement the runtime port required by its role",
                    )
                )
            role_operation = {
                ProviderRole.ROOT_PROOF_SIGNING: "sign_root_proof",
                ProviderRole.HISTORY_ATTESTATION_SIGNING: "sign_history_head",
                ProviderRole.FRESHNESS_AUTHORITY_FINALIZATION_SIGNING: "sign_finalization",
                ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING: "sign_freshness_proposal",
            }
            opposite_methods = {
                operation
                for operation in role_operation.values()
                if operation != role_operation.get(role)
            }
            if role in _SIGNING_ROLES and any(
                callable(getattr(snapshot.provider, operation, None))
                for operation in opposite_methods
            ):
                failures.append(
                    QualificationFailure(
                        QualificationFailureCode.SIGNING_AUTHORITY_INTERFACE_ALIAS,
                        role,
                        "signing provider exposes the opposite authority operation",
                    )
                )
            if snapshot.identity.security != security:
                failures.append(
                    QualificationFailure(
                        QualificationFailureCode.PROFILE_IDENTITY_MISMATCH,
                        role,
                        "provider profile or trust_domain differs from composition",
                    )
                )
            for detail in _provider_qualification_failures(
                security.profile, snapshot.identity, snapshot.capabilities
            ):
                code = (
                    QualificationFailureCode.PROVIDER_NOT_IMPLEMENTED
                    if "unavailable" in detail
                    else QualificationFailureCode.INSUFFICIENT_CAPABILITIES
                )
                failures.append(QualificationFailure(code, role, detail))
            failures.extend(self._credential_evidence_failures(snapshot))
        credentials = [identity for snapshot in snapshots for identity in snapshot.credentials]
        failures.extend(self._alias_failures(credentials))
        return QualificationResult(tuple(failures))

    @staticmethod
    def _valid_security_identity(candidate: object) -> bool:
        return _valid_security_profile_identity(candidate)

    @classmethod
    def _valid_provider_identity(cls, candidate: object) -> bool:
        return _valid_provider_identity(candidate)

    @staticmethod
    def _valid_capabilities(candidate: object) -> bool:
        return _valid_provider_capabilities(candidate)

    @staticmethod
    def _valid_credential_identity(candidate: object) -> bool:
        return _valid_credential_role_identity(candidate)

    @classmethod
    def _capture_snapshot(
        cls,
        provider: SecurityProvider,
    ) -> tuple[_ProviderQualificationSnapshot | None, QualificationFailure | None]:
        try:
            identity = provider.identity
        except Exception:
            return None, QualificationFailure(
                QualificationFailureCode.PROVIDER_EVIDENCE_UNAVAILABLE,
                None,
                "provider identity evidence is unavailable",
            )
        try:
            valid_identity = cls._valid_provider_identity(identity)
        except (AttributeError, TypeError):
            valid_identity = False
        if not valid_identity:
            return None, QualificationFailure(
                QualificationFailureCode.PROVIDER_EVIDENCE_UNAVAILABLE,
                None,
                "provider identity evidence has an invalid type",
            )
        try:
            capabilities = provider.capabilities
            credentials = provider.credential_identities()
        except Exception:
            return None, QualificationFailure(
                QualificationFailureCode.PROVIDER_EVIDENCE_UNAVAILABLE,
                identity.role,
                "provider capability or credential evidence is unavailable",
            )
        try:
            valid_evidence = (
                cls._valid_capabilities(capabilities)
                and type(credentials) is tuple
                and all(cls._valid_credential_identity(item) for item in credentials)
            )
        except (AttributeError, TypeError):
            valid_evidence = False
        if not valid_evidence:
            return None, QualificationFailure(
                QualificationFailureCode.PROVIDER_EVIDENCE_UNAVAILABLE,
                identity.role,
                "provider capability or credential evidence has an invalid type",
            )
        active: CredentialRoleIdentity | None = None
        observed: str | None = None
        interface_compatible = all(
            callable(getattr(provider, method, None)) for method in _ROLE_METHODS[identity.role]
        )
        if identity.role in _SIGNING_ROLES and interface_compatible and len(credentials) == 1:
            try:
                candidate = provider.active_credential_identity()  # type: ignore[attr-defined]
            except Exception:
                return None, QualificationFailure(
                    QualificationFailureCode.PROVIDER_EVIDENCE_UNAVAILABLE,
                    identity.role,
                    "active signing identity evidence is unavailable",
                )
            try:
                valid_active = cls._valid_credential_identity(candidate)
            except (AttributeError, TypeError):
                valid_active = False
            if not valid_active:
                return None, QualificationFailure(
                    QualificationFailureCode.ACTIVE_CREDENTIAL_IDENTITY_MISMATCH,
                    identity.role,
                    "active signing identity evidence has an invalid type",
                )
            active = candidate
            if active == credentials[0]:
                try:
                    public_key = provider.public_key(active.credential_identity)  # type: ignore[union-attr,attr-defined]
                    observed = public_key_material_identity(public_key)
                except Exception:
                    return None, QualificationFailure(
                        QualificationFailureCode.PROVIDER_EVIDENCE_UNAVAILABLE,
                        identity.role,
                        "signing public-key evidence is unavailable",
                    )
        return (
            _ProviderQualificationSnapshot(
                provider,
                identity,
                capabilities,
                credentials,
                active,
                observed,
            ),
            None,
        )

    @staticmethod
    def _credential_evidence_failures(
        snapshot: _ProviderQualificationSnapshot,
    ) -> list[QualificationFailure]:
        expected = _CREDENTIAL_ROLES.get(snapshot.identity.role)
        credentials = snapshot.credentials
        failures: list[QualificationFailure] = []
        if expected is not None and not credentials:
            failures.append(
                QualificationFailure(
                    QualificationFailureCode.MISSING_CREDENTIAL_IDENTITY_EVIDENCE,
                    snapshot.identity.role,
                    "credential-bearing provider supplied no composition identity",
                )
            )
        if snapshot.identity.role in _SIGNING_ROLES and len(credentials) != 1:
            failures.append(
                QualificationFailure(
                    QualificationFailureCode.AMBIGUOUS_CREDENTIAL_IDENTITY_EVIDENCE,
                    snapshot.identity.role,
                    "signing provider must supply exactly one composition identity",
                )
            )
        for credential in credentials:
            if credential.provider_namespace != snapshot.identity.provider_namespace:
                failures.append(
                    QualificationFailure(
                        QualificationFailureCode.CREDENTIAL_NAMESPACE_MISMATCH,
                        snapshot.identity.role,
                        "credential identity is not bound to the provider namespace",
                    )
                )
            if expected is not None and credential.semantic_role is not expected:
                failures.append(
                    QualificationFailure(
                        QualificationFailureCode.CREDENTIAL_ROLE_MISMATCH,
                        snapshot.identity.role,
                        "credential semantic role does not match provider role",
                    )
                )
            signing = snapshot.capabilities.signing
            if (
                snapshot.identity.role in _SIGNING_ROLES
                and signing is not None
                and signing.stable_key_handle_or_version_identity
                and credential.key_handle_or_version is None
            ):
                failures.append(
                    QualificationFailure(
                        QualificationFailureCode.MISSING_KEY_HANDLE_IDENTITY,
                        snapshot.identity.role,
                        "stable signing key handle/version capability lacks identity evidence",
                    )
                )
        if snapshot.identity.role in _SIGNING_ROLES and len(credentials) == 1:
            failures.extend(
                RootProofIssuerCompositionGate._active_signing_identity_failures(
                    snapshot, credentials[0]
                )
            )
        return failures

    @staticmethod
    def _active_signing_identity_failures(
        snapshot: _ProviderQualificationSnapshot, declared: CredentialRoleIdentity
    ) -> list[QualificationFailure]:
        role = snapshot.identity.role
        failures: list[QualificationFailure] = []
        active = snapshot.active_credential
        if not isinstance(active, CredentialRoleIdentity) or active != declared:
            failures.append(
                QualificationFailure(
                    QualificationFailureCode.ACTIVE_CREDENTIAL_IDENTITY_MISMATCH,
                    role,
                    "active signing identity does not exactly match composition evidence",
                )
            )
            return failures
        signing = snapshot.capabilities.signing
        material_required = bool(
            signing
            and signing.durable_key_identity
            and signing.stable_key_handle_or_version_identity
        )
        if material_required and active.key_material_identity is None:
            failures.append(
                QualificationFailure(
                    QualificationFailureCode.MISSING_KEY_MATERIAL_IDENTITY,
                    role,
                    "stable durable signing identity lacks public-key material identity",
                )
            )
            return failures
        observed = snapshot.observed_key_material_identity
        if observed is None or active.key_material_identity != observed:
            failures.append(
                QualificationFailure(
                    QualificationFailureCode.KEY_MATERIAL_IDENTITY_MISMATCH,
                    role,
                    "declared key material identity does not bind to provider public key",
                )
            )
        return failures

    @staticmethod
    def _alias_failures(
        credentials: Sequence[CredentialRoleIdentity],
    ) -> list[QualificationFailure]:
        failures: list[QualificationFailure] = []
        for index, left in enumerate(credentials):
            for right in credentials[index + 1 :]:
                if (
                    frozenset((left.semantic_role, right.semantic_role))
                    not in _FORBIDDEN_ROLE_PAIRS
                ):
                    continue
                if (
                    left.provider_namespace,
                    left.credential_identity,
                ) == (
                    right.provider_namespace,
                    right.credential_identity,
                ):
                    failures.append(
                        QualificationFailure(
                            QualificationFailureCode.CREDENTIAL_IDENTITY_ALIAS,
                            None,
                            "forbidden roles share credential identity",
                        )
                    )
                if left.key_handle_or_version is not None and (
                    left.provider_namespace,
                    left.key_handle_or_version,
                ) == (
                    right.provider_namespace,
                    right.key_handle_or_version,
                ):
                    failures.append(
                        QualificationFailure(
                            QualificationFailureCode.KEY_HANDLE_ALIAS,
                            None,
                            "forbidden roles share key handle/version",
                        )
                    )
                if (
                    left.key_material_identity is not None
                    and left.key_material_identity == right.key_material_identity
                ):
                    failures.append(
                        QualificationFailure(
                            QualificationFailureCode.KEY_MATERIAL_ALIAS,
                            None,
                            "forbidden roles share cryptographic key material identity",
                        )
                    )
        return failures


MANDATORY_PROVIDER_ROLES: Mapping[ProviderRole, str] = MappingProxyType(
    {role: role.value for role in ProviderRole}
)
