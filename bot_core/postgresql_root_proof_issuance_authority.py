"""Reviewed internal PRODUCTION_LOCAL pre-account public authority composition.

Configuration selects existing PostgreSQL authorities. Only live provider
evidence can establish credential, lifecycle, privilege, or durability facts.
This boundary performs reads and does not issue signatures or bind entitlement.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from psycopg import sql

from bot_core.entitlement_registry_contract import (
    AuthoritativeEntitlementState,
    EntitlementLifecycle,
    RegistryReadOutcome,
    RegistryReadResult,
    RegistrySubject,
    UnboundBinding,
    validate_exact_snapshot,
)
from bot_core.licensing.canonical import parse_canonical
from bot_core.postgresql_entitlement_registry import (
    PostgreSQLConnectionConfig,
    PostgreSQLEntitlementRegistryProvider,
)
from bot_core.postgresql_preaccount_credentials import (
    CLAIMANT_ADMIN_ROLE,
    CLAIMANT_OWNER_ROLE,
    CLAIMANT_RUNTIME_ROLE,
    CLAIMANT_SCHEMA,
    REQUESTER_ADMIN_ROLE,
    REQUESTER_OWNER_ROLE,
    REQUESTER_RUNTIME_ROLE,
    REQUESTER_SCHEMA,
    PostgreSQLClaimantIdentityRegistryProvider,
    PostgreSQLRequesterCredentialRegistryProvider,
)
from bot_core.root_proof_issuer_substrate import (
    CredentialRoleIdentity,
    ProviderCapabilities,
    ProviderIdentity,
    ProviderQualificationPolicy,
    ProviderRole,
    RootProofIssuerCompositionGate,
    SecurityProfile,
    SecurityProfileIdentity,
    public_key_material_identity,
)

if TYPE_CHECKING:
    from bot_core.licensing.cha_root_proof_attempt_reservation import _ProviderResolution

REQUESTER_PRINCIPAL = "CryptoHunterAccountAuthority"
_CONFIG_ENVIRONMENT = (
    "CH_ROOT_PROOF_ENTITLEMENT_RUNTIME_DSN",
    "CH_ROOT_PROOF_REQUESTER_RUNTIME_DSN",
    "CH_ROOT_PROOF_CLAIMANT_RUNTIME_DSN",
    "CH_ROOT_PROOF_TRUST_DOMAIN",
    "CH_ROOT_PROOF_ENTITLEMENT_SCHEMA",
    "CH_ROOT_PROOF_ENTITLEMENT_LOOKUP_HANDLE",
)
# Protect only process-local adapter identity. PostgreSQL remains the sole
# credential/lifecycle/concurrency authority.
_COMPOSITION_LOCK = threading.Lock()


class ProductionLocalIssuanceAuthorityError(RuntimeError):
    """Exact live production authority configuration or evidence is unavailable."""


@dataclass(frozen=True, slots=True)
class PostgreSQLRootProofIssuanceAuthority:
    """Exact aggregate of three genuine, independently qualified runtime ports."""

    entitlement_registry: PostgreSQLEntitlementRegistryProvider
    requester_registry: PostgreSQLRequesterCredentialRegistryProvider
    claimant_registry: PostgreSQLClaimantIdentityRegistryProvider
    _entitlement_subject: RegistrySubject

    def __post_init__(self) -> None:
        self.requalify()

    def __init_subclass__(cls) -> None:
        raise TypeError("production issuance authority cannot be subclassed")

    def requalify(self) -> None:
        """Recheck live PostgreSQL schema, principal, privileges and key material."""

        expected = (
            (self.entitlement_registry, PostgreSQLEntitlementRegistryProvider),
            (self.requester_registry, PostgreSQLRequesterCredentialRegistryProvider),
            (self.claimant_registry, PostgreSQLClaimantIdentityRegistryProvider),
        )
        if type(self) is not PostgreSQLRootProofIssuanceAuthority or any(
            type(port) is not implementation for port, implementation in expected
        ):
            raise ProductionLocalIssuanceAuthorityError("EXACT_GENUINE_AUTHORITY_TYPES_REQUIRED")
        if type(self._entitlement_subject) is not RegistrySubject:
            raise ProductionLocalIssuanceAuthorityError("EXACT_ENTITLEMENT_SUBJECT_REQUIRED")
        subject = self._entitlement_subject
        validate_exact_snapshot(subject)
        if subject.environment != "PRODUCTION":
            raise ProductionLocalIssuanceAuthorityError("PRODUCTION_PROTOCOL_ENVIRONMENT_REQUIRED")
        security = SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, subject.trust_domain)
        roles = (
            (self.entitlement_registry, ProviderRole.ENTITLEMENT_REGISTRY),
            (self.requester_registry, ProviderRole.REQUESTER_CREDENTIAL_REGISTRY),
            (self.claimant_registry, ProviderRole.CLAIMANT_IDENTITY_REGISTRY),
        )
        evidence: list[CredentialRoleIdentity] = []
        runtime_roles: list[str] = []
        for port, role in roles:
            port._qualify()
            with port._connect() as conn:
                runtime_identity = conn.execute("SELECT session_user,current_user").fetchone()
            if runtime_identity is None or runtime_identity[0] != runtime_identity[1]:
                raise ProductionLocalIssuanceAuthorityError("DIRECT_RUNTIME_AUTHORITY_REQUIRED")
            runtime_roles.append(runtime_identity[0])
            # These attributes originate in the exact reviewed implementation,
            # not caller DTOs or configuration-supplied capability declarations.
            if (
                port._environment != subject.environment
                or port._trust_domain != subject.trust_domain
            ):
                raise ProductionLocalIssuanceAuthorityError("AUTHORITY_SCOPE_MISMATCH")
            identity, capabilities = port.identity, port.capabilities
            if (
                type(identity) is not ProviderIdentity
                or type(capabilities) is not ProviderCapabilities
                or identity.role is not role
                or identity.security != security
                or ProviderQualificationPolicy().failures_for(
                    SecurityProfile.PRODUCTION_LOCAL, identity, capabilities
                )
            ):
                raise ProductionLocalIssuanceAuthorityError("UNQUALIFIED_PRODUCTION_AUTHORITY")
            snapshot, failure = RootProofIssuerCompositionGate._capture_snapshot(port)
            if (
                failure is not None
                or snapshot is None
                or not RootProofIssuerCompositionGate._statically_implements_role_port(port, role)
                or RootProofIssuerCompositionGate._credential_evidence_failures(snapshot)
            ):
                raise ProductionLocalIssuanceAuthorityError("INEXACT_CREDENTIAL_AUTHORITY_EVIDENCE")
            if role is not ProviderRole.ENTITLEMENT_REGISTRY:
                if capabilities.signing is not None or capabilities.compare_and_swap:
                    raise ProductionLocalIssuanceAuthorityError(
                        "PUBLIC_REGISTRY_CAPABILITY_MISMATCH"
                    )
                for credential in snapshot.credentials:
                    raw = port.public_key(credential.credential_identity)
                    if public_key_material_identity(raw) != credential.key_material_identity:
                        raise ProductionLocalIssuanceAuthorityError(
                            "CORRUPT_PUBLIC_KEY_MATERIAL_IDENTITY"
                        )
            evidence.extend(snapshot.credentials)
        if len({port._schema for port, _ in roles}) != 3 or len(set(runtime_roles)) != 3:
            raise ProductionLocalIssuanceAuthorityError(
                "SEPARATE_AUTHORITY_SCHEMAS_AND_ROLES_REQUIRED"
            )
        if RootProofIssuerCompositionGate._alias_failures(evidence):
            raise ProductionLocalIssuanceAuthorityError("FORBIDDEN_CREDENTIAL_ALIAS")

    def resolve_initial_binding(self, context_raw: bytes) -> _ProviderResolution:
        """Resolve the pre-account claimant exclusively from entitlement provenance."""

        from bot_core.licensing.cha_root_proof_attempt_reservation import _ProviderResolution

        context = parse_canonical(context_raw)
        subject = self._entitlement_subject
        if (
            context.get("environment") != subject.environment
            or context.get("pdsa_trust_domain") != subject.trust_domain
            or context.get("reservation_relation") != "EXACT_OPERATION_ACCOUNT"
        ):
            raise ProductionLocalIssuanceAuthorityError("INITIAL_BINDING_AUTHORITY_SCOPE_MISMATCH")
        result = self.entitlement_registry.authoritative_state(subject)
        if (
            type(result) is not RegistryReadResult
            or result.outcome is not RegistryReadOutcome.FOUND
            or type(result.state) is not AuthoritativeEntitlementState
        ):
            raise ProductionLocalIssuanceAuthorityError("AUTHORITATIVE_ENTITLEMENT_REQUIRED")
        validate_exact_snapshot(result)
        state = result.state
        if (
            state.subject != subject
            or state.identity.environment != subject.environment
            or state.identity.trust_domain != subject.trust_domain
            or state.identity.product_scope != context.get("product_scope")
            or state.lifecycle is not EntitlementLifecycle.ACTIVE
            or type(state.binding) is not UnboundBinding
        ):
            raise ProductionLocalIssuanceAuthorityError("ACTIVE_UNBOUND_ENTITLEMENT_REQUIRED")
        if state.provenance.provisioning_principal_id in (
            context.get("account_id"),
            context.get("logical_operation_id"),
        ):
            raise ProductionLocalIssuanceAuthorityError("PREACCOUNT_CLAIMANT_PRINCIPAL_REQUIRED")
        return _ProviderResolution(
            subject, REQUESTER_PRINCIPAL, state.provenance.provisioning_principal_id
        )

    def validate_resolved_credentials(
        self,
        requester: object,
        claimant: object,
        requester_evidence: tuple[CredentialRoleIdentity, ...],
        claimant_evidence: tuple[CredentialRoleIdentity, ...],
    ) -> None:
        """Bind selected ACTIVE identities to exact version, lifecycle and material.

        Population-wide qualification remains in requalify(); these identities
        alone belong to the resolved operation's retained authorization evidence.
        """

        from bot_core.licensing.cha_root_proof_attempt_reservation import (
            _ClaimantIdentityV1,
            _RequesterCredentialV1,
        )

        if (
            type(requester) is not _RequesterCredentialV1
            or type(claimant) is not _ClaimantIdentityV1
        ):
            raise ProductionLocalIssuanceAuthorityError("EXACT_ACTIVE_CREDENTIAL_RECORDS_REQUIRED")
        resolutions = (
            (
                self.requester_registry,
                requester.requester_principal_id,
                requester.requester_key_id,
                requester.requester_key_version,
                requester.registry_revision,
                requester_evidence,
            ),
            (
                self.claimant_registry,
                claimant.provisioning_principal_id,
                claimant.claimant_key_id,
                claimant.claimant_key_version,
                claimant.registry_revision,
                claimant_evidence,
            ),
        )
        for port, principal, key_id, key_version, revision, evidence in resolutions:
            current = port._current(principal)
            namespace = port.identity.provider_namespace
            expected = CredentialRoleIdentity(
                current.semantic_role,
                current.key_id,
                namespace,
                f"{current.key_id}:v{current.key_version}",
                f"{namespace}:public-lifecycle:{current.lifecycle_generation}:"
                f"{current.registry_revision}:{current.lifecycle.value}",
                public_key_material_identity(current.public_key),
            )
            matches = tuple(item for item in evidence if item.credential_identity == key_id)
            if (
                current.principal_id != principal
                or current.key_id != key_id
                or current.key_version != key_version
                or current.registry_revision != revision
                or current.lifecycle.value != "ACTIVE"
                or matches != (expected,)
                or port.public_key(key_id) != current.public_key
            ):
                raise ProductionLocalIssuanceAuthorityError("ACTIVE_CREDENTIAL_EVIDENCE_CHANGED")


@dataclass(frozen=True, slots=True)
class _RuntimeConfiguration:
    entitlement_connection: PostgreSQLConnectionConfig
    requester_connection: PostgreSQLConnectionConfig
    claimant_connection: PostgreSQLConnectionConfig
    trust_domain: str
    entitlement_schema: str
    entitlement_lookup_handle: str


@lru_cache(maxsize=1)
def _compose_runtime_authority(
    config: _RuntimeConfiguration,
) -> PostgreSQLRootProofIssuanceAuthority:
    """Preserve adapter identity while every authorization rechecks live evidence."""

    return PostgreSQLRootProofIssuanceAuthority(
        PostgreSQLEntitlementRegistryProvider(
            config.entitlement_connection,
            schema=config.entitlement_schema,
            environment="PRODUCTION",
            trust_domain=config.trust_domain,
        ),
        PostgreSQLRequesterCredentialRegistryProvider(
            config.requester_connection,
            schema=REQUESTER_SCHEMA,
            environment="PRODUCTION",
            trust_domain=config.trust_domain,
        ),
        PostgreSQLClaimantIdentityRegistryProvider(
            config.claimant_connection,
            schema=CLAIMANT_SCHEMA,
            environment="PRODUCTION",
            trust_domain=config.trust_domain,
        ),
        RegistrySubject(config.entitlement_lookup_handle, "PRODUCTION", config.trust_domain),
    )


def _configured_root_proof_issuance_authority() -> PostgreSQLRootProofIssuanceAuthority:
    """Deployment-internal seam; no provider, DSN or credential is a Stage 9 input."""

    values = tuple(os.environ.get(name, "") for name in _CONFIG_ENVIRONMENT)
    if any(not value.strip() for value in values):
        raise ProductionLocalIssuanceAuthorityError("MISSING_PRODUCTION_AUTHORITY_CONFIGURATION")
    config = _RuntimeConfiguration(
        PostgreSQLConnectionConfig(values[0]),
        PostgreSQLConnectionConfig(values[1]),
        PostgreSQLConnectionConfig(values[2]),
        values[3],
        values[4],
        values[5],
    )
    try:
        with _COMPOSITION_LOCK:
            authority = _compose_runtime_authority(config)
        authority.requalify()
        for port, reviewed in (
            (
                authority.requester_registry,
                (
                    REQUESTER_SCHEMA,
                    REQUESTER_OWNER_ROLE,
                    REQUESTER_RUNTIME_ROLE,
                    REQUESTER_ADMIN_ROLE,
                ),
            ),
            (
                authority.claimant_registry,
                (CLAIMANT_SCHEMA, CLAIMANT_OWNER_ROLE, CLAIMANT_RUNTIME_ROLE, CLAIMANT_ADMIN_ROLE),
            ),
        ):
            with port._connect() as conn, conn.transaction():
                conn.execute("SET LOCAL search_path = pg_catalog")
                selected = conn.execute(
                    sql.SQL(
                        "SELECT config->>'schema',config->>'schema_owner_role',"
                        "config->>'runtime_role',config->>'admin_role' FROM {}.metadata"
                    ).format(sql.Identifier(port.schema))
                ).fetchone()
            if selected != reviewed:
                raise ProductionLocalIssuanceAuthorityError("REVIEWED_RUNTIME_IDENTIFIERS_REQUIRED")
        return authority
    except Exception:
        raise ProductionLocalIssuanceAuthorityError("PRODUCTION_AUTHORITY_UNAVAILABLE") from None
