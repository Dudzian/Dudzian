"""Read-only genuine public authorization against the qualified V2 inventory.

This explicit test-deployment adapter does not replace the production selector.
It reuses the original immutable entitlement and public-credential readers, but
qualifies every connection against the reviewed post-migration catalog. Its
entitlement port has no BIND capability; generic issuer readiness remains closed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import psycopg
from psycopg import sql

from bot_core.entitlement_registry_contract import (
    AuthoritativeEntitlementState,
    BindRequest,
    BindResult,
    ContractValidationError,
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
    CredentialGeneration,
    CredentialResolutionError,
    PostgreSQLClaimantIdentityRegistryProvider,
    PostgreSQLRequesterCredentialRegistryProvider,
    _PostgreSQLCredentialRegistry,
    _text,
)
from bot_core.postgresql_root_proof_issuance_authority import (
    REQUESTER_PRINCIPAL,
    ProductionLocalIssuanceAuthorityError,
)
from bot_core.root_proof_issuer_substrate import (
    CredentialRoleIdentity,
    ProviderCapabilities,
    ProviderIdentity,
    ProviderRole,
    RootProofIssuerCompositionGate,
    SecurityProfile,
    SecurityProfileIdentity,
    public_key_material_identity,
)

if TYPE_CHECKING:
    from bot_core.licensing.cha_root_proof_attempt_reservation import _ProviderResolution
    from bot_core.postgresql_root_proof_preparation_readonly import (
        PostgreSQLRootProofPreparationV2Deployment,
    )

_ReadResult = TypeVar("_ReadResult")


def _validate_deployment(deployment: PostgreSQLRootProofPreparationV2Deployment) -> None:
    # The migration module also imports retained-attempt validation. Resolve
    # qualification only on use to keep that dependency cycle out of imports.
    from bot_core.postgresql_root_proof_preparation_readonly import (
        validate_root_proof_preparation_v2_deployment,
    )

    validate_root_proof_preparation_v2_deployment(deployment)


class _QualifiedV2PublicConnection:
    """Reuse reviewed data validation, with a distinct live catalog contract."""

    _component: str
    _connection: PostgreSQLConnectionConfig
    _deployment: PostgreSQLRootProofPreparationV2Deployment
    _schema: str
    _environment: str
    _trust_domain: str

    def __init__(
        self,
        connection: PostgreSQLConnectionConfig,
        *,
        deployment: PostgreSQLRootProofPreparationV2Deployment,
        schema: str,
        environment: str,
        trust_domain: str,
    ) -> None:
        _validate_deployment(deployment)
        if type(connection) is not PostgreSQLConnectionConfig:
            raise TypeError("exact PostgreSQL connection configuration required")
        expected_schema = getattr(deployment.provisioning, self._component + "_schema")
        if (
            schema != expected_schema
            or environment != "PRODUCTION"
            or trust_domain != deployment.provisioning.trust_domain
        ):
            raise ProductionLocalIssuanceAuthorityError("PINNED_PUBLIC_AUTHORITY_SCOPE_REQUIRED")
        self._deployment = deployment
        self._connection = connection
        self._schema = schema
        self._environment = environment
        self._trust_domain = trust_domain
        self._qualify()

    def _connect(self) -> psycopg.Connection[Any]:
        conn = psycopg.connect(self._connection.dsn, autocommit=True)
        try:
            conn.execute("SET search_path=pg_catalog")
            return conn
        except Exception:
            conn.close()
            raise

    def _qualify(self) -> None:
        with self._connect() as conn, conn.transaction():
            self._qualify_connection(conn)

    @contextmanager
    def _qualified_read(self) -> Iterator[psycopg.Connection[Any]]:
        """Qualify the actual read transaction; never reuse an earlier check."""
        with self._connect() as conn, conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            self._qualify_connection(conn)
            yield conn

    def _qualify_connection(self, conn: psycopg.Connection[Any]) -> None:
        from bot_core.postgresql_root_proof_preparation_readonly import (
            qualify_root_proof_preparation_v2,
        )

        component = {
            PostgreSQLEntitlementRegistryProviderV2: "entitlement",
            PostgreSQLRequesterCredentialRegistryProviderV2: "requester",
            PostgreSQLClaimantIdentityRegistryProviderV2: "claimant",
        }.get(type(self))
        if (
            component is None
            or self._component != component
            or self._schema != getattr(self._deployment.provisioning, component + "_schema")
            or self._environment != "PRODUCTION"
            or self._trust_domain != self._deployment.provisioning.trust_domain
        ):
            raise ProductionLocalIssuanceAuthorityError("PINNED_PUBLIC_AUTHORITY_SCOPE_REQUIRED")
        qualify_root_proof_preparation_v2(conn, self._deployment, component=component)


class _QualifiedV2PublicCredentials(_QualifiedV2PublicConnection, _PostgreSQLCredentialRegistry):
    """Original record validation with one qualification per data connection."""

    def retained_history(self, principal_id: str) -> tuple[CredentialGeneration, ...]:
        principal = _text(principal_id)
        with self._qualified_read() as conn:
            return self._load_history(conn, principal)

    def _historical(self, identity: str) -> CredentialGeneration:
        _text(identity)
        with self._qualified_read() as conn:
            rows = conn.execute(
                sql.SQL(
                    "SELECT credential_id,principal_id FROM {}.credentials "
                    "WHERE credential_id=%s OR key_id=%s"
                ).format(sql.Identifier(self._schema)),
                (identity, identity),
            ).fetchall()
            if len(rows) != 1:
                raise CredentialResolutionError("exact retained credential identity unavailable")
            evidence = self._load_history(conn, rows[0][1])
            matches = [item for item in evidence if item.credential_id == rows[0][0]]
            if not matches:
                raise CredentialResolutionError("credential lacks retained lifecycle evidence")
            return matches[-1]

    def credential_identities(self) -> tuple[CredentialRoleIdentity, ...]:
        with self._qualified_read() as conn:
            principals = conn.execute(
                sql.SQL(
                    "SELECT principal_id FROM {}.current_credentials ORDER BY principal_id"
                ).format(sql.Identifier(self._schema))
            ).fetchall()
            credentials: list[CredentialRoleIdentity] = []
            for (principal,) in principals:
                latest = {
                    record.credential_id: record for record in self._load_history(conn, principal)
                }
                for record in latest.values():
                    credentials.append(
                        CredentialRoleIdentity(
                            self._semantic_role,
                            record.key_id,
                            self.identity.provider_namespace,
                            f"{record.key_id}:v{record.key_version}",
                            f"{self.identity.provider_namespace}:public-lifecycle:{record.lifecycle_generation}:"
                            f"{record.registry_revision}:{record.lifecycle.value}",
                            record.key_material_identity,
                        )
                    )
            return tuple(credentials)


class PostgreSQLEntitlementRegistryProviderV2(
    _QualifiedV2PublicConnection, PostgreSQLEntitlementRegistryProvider
):
    """Immutable entitlement reads; preparation cannot perform final BIND."""

    _component = "entitlement"

    def __init_subclass__(cls) -> None:
        raise TypeError("V2 public entitlement authority cannot be subclassed")

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(True, authoritative_reads=True, durable_state=True)

    def compare_and_swap_bind(self, request: BindRequest) -> BindResult:
        raise ProductionLocalIssuanceAuthorityError("V2_PREPARATION_HAS_NO_BIND_AUTHORITY")

    def _read(
        self,
        subject: RegistrySubject,
        operation: Callable[[psycopg.Connection[Any], RegistrySubject], _ReadResult],
        unavailable: Callable[[], _ReadResult],
        corrupt: Callable[[], _ReadResult],
    ) -> _ReadResult:
        try:
            with self._qualified_read() as conn:
                return operation(conn, subject)
        except (ContractValidationError, ValueError, TypeError):
            return corrupt()
        except psycopg.Error:
            return unavailable()


class PostgreSQLRequesterCredentialRegistryProviderV2(
    _QualifiedV2PublicCredentials, PostgreSQLRequesterCredentialRegistryProvider
):
    """Original requester history and keys under the reviewed V2 catalog."""

    _component = "requester"

    def __init_subclass__(cls) -> None:
        raise TypeError("V2 requester authority cannot be subclassed")


class PostgreSQLClaimantIdentityRegistryProviderV2(
    _QualifiedV2PublicCredentials, PostgreSQLClaimantIdentityRegistryProvider
):
    """Original claimant history and keys under the reviewed V2 catalog."""

    _component = "claimant"

    def __init_subclass__(cls) -> None:
        raise TypeError("V2 claimant authority cannot be subclassed")


@dataclass(frozen=True, slots=True, init=False)
class PostgreSQLRootProofIssuanceAuthorityV2:
    """Exact public readers for genuine INITIAL_BINDING preparation only."""

    entitlement_registry: PostgreSQLEntitlementRegistryProviderV2
    requester_registry: PostgreSQLRequesterCredentialRegistryProviderV2
    claimant_registry: PostgreSQLClaimantIdentityRegistryProviderV2
    _entitlement_subject: RegistrySubject
    deployment: PostgreSQLRootProofPreparationV2Deployment

    def __init__(
        self,
        entitlement_connection: PostgreSQLConnectionConfig,
        requester_connection: PostgreSQLConnectionConfig,
        claimant_connection: PostgreSQLConnectionConfig,
        *,
        deployment: PostgreSQLRootProofPreparationV2Deployment,
        entitlement_subject: RegistrySubject,
    ) -> None:
        _validate_deployment(deployment)
        if (
            type(entitlement_subject) is not RegistrySubject
            or entitlement_subject.environment != "PRODUCTION"
            or entitlement_subject.trust_domain != deployment.provisioning.trust_domain
        ):
            raise ProductionLocalIssuanceAuthorityError("EXACT_PINNED_ENTITLEMENT_SUBJECT_REQUIRED")
        validate_exact_snapshot(entitlement_subject)
        object.__setattr__(self, "deployment", deployment)
        object.__setattr__(self, "_entitlement_subject", entitlement_subject)
        for name, connection, implementation in (
            ("entitlement", entitlement_connection, PostgreSQLEntitlementRegistryProviderV2),
            ("requester", requester_connection, PostgreSQLRequesterCredentialRegistryProviderV2),
            ("claimant", claimant_connection, PostgreSQLClaimantIdentityRegistryProviderV2),
        ):
            port = implementation(
                connection,
                deployment=deployment,
                schema=getattr(deployment.provisioning, name + "_schema"),
                environment="PRODUCTION",
                trust_domain=deployment.provisioning.trust_domain,
            )
            object.__setattr__(self, name + "_registry", port)
        self.requalify()

    def __init_subclass__(cls) -> None:
        raise TypeError("V2 public issuance authority cannot be subclassed")

    def requalify(self) -> None:
        """Recheck exact readers, each direct login and the complete live pin."""
        _validate_deployment(self.deployment)
        subject = self._entitlement_subject
        if type(self) is not PostgreSQLRootProofIssuanceAuthorityV2:
            raise ProductionLocalIssuanceAuthorityError("EXACT_GENUINE_V2_AUTHORITY_REQUIRED")
        if type(subject) is not RegistrySubject:
            raise ProductionLocalIssuanceAuthorityError("EXACT_ENTITLEMENT_SUBJECT_REQUIRED")
        validate_exact_snapshot(subject)
        security = SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, subject.trust_domain)
        roles: tuple[
            tuple[
                PostgreSQLEntitlementRegistryProviderV2
                | PostgreSQLRequesterCredentialRegistryProviderV2
                | PostgreSQLClaimantIdentityRegistryProviderV2,
                type,
                ProviderRole,
            ],
            ...,
        ] = (
            (
                self.entitlement_registry,
                PostgreSQLEntitlementRegistryProviderV2,
                ProviderRole.ENTITLEMENT_REGISTRY,
            ),
            (
                self.requester_registry,
                PostgreSQLRequesterCredentialRegistryProviderV2,
                ProviderRole.REQUESTER_CREDENTIAL_REGISTRY,
            ),
            (
                self.claimant_registry,
                PostgreSQLClaimantIdentityRegistryProviderV2,
                ProviderRole.CLAIMANT_IDENTITY_REGISTRY,
            ),
        )
        credentials: list[CredentialRoleIdentity] = []
        for port, implementation, role in roles:
            if (
                type(port) is not implementation
                or port._deployment is not self.deployment
                or port._environment != "PRODUCTION"
                or port._trust_domain != subject.trust_domain
                or subject.environment != "PRODUCTION"
                or subject.trust_domain != self.deployment.provisioning.trust_domain
            ):
                raise ProductionLocalIssuanceAuthorityError(
                    "EXACT_PINNED_V2_PUBLIC_READER_REQUIRED"
                )
            port._qualify()
            if (
                type(port.identity) is not ProviderIdentity
                or port.identity.role is not role
                or port.identity.security != security
                or type(port.capabilities) is not ProviderCapabilities
                or port.capabilities
                != ProviderCapabilities(True, authoritative_reads=True, durable_state=True)
            ):
                raise ProductionLocalIssuanceAuthorityError(
                    "READ_ONLY_V2_PUBLIC_CAPABILITIES_REQUIRED"
                )
            snapshot, failure = RootProofIssuerCompositionGate._capture_snapshot(port)
            if (
                failure is not None
                or snapshot is None
                or not RootProofIssuerCompositionGate._statically_implements_role_port(port, role)
                or RootProofIssuerCompositionGate._credential_evidence_failures(snapshot)
            ):
                raise ProductionLocalIssuanceAuthorityError("INEXACT_CREDENTIAL_AUTHORITY_EVIDENCE")
            if isinstance(
                port,
                (
                    PostgreSQLRequesterCredentialRegistryProviderV2,
                    PostgreSQLClaimantIdentityRegistryProviderV2,
                ),
            ):
                for credential in snapshot.credentials:
                    raw = port.public_key(credential.credential_identity)
                    if public_key_material_identity(raw) != credential.key_material_identity:
                        raise ProductionLocalIssuanceAuthorityError(
                            "CORRUPT_PUBLIC_KEY_MATERIAL_IDENTITY"
                        )
            credentials.extend(snapshot.credentials)
        if RootProofIssuerCompositionGate._alias_failures(credentials):
            raise ProductionLocalIssuanceAuthorityError("FORBIDDEN_CREDENTIAL_ALIAS")

    def resolve_initial_binding(self, context_raw: bytes) -> _ProviderResolution:
        """Select the claimant exclusively from immutable entitlement provenance."""
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
        """Recheck the original ACTIVE version, generation and public material."""
        from bot_core.licensing.cha_root_proof_attempt_reservation import (
            _ClaimantIdentityV1,
            _RequesterCredentialV1,
        )

        if (
            type(requester) is not _RequesterCredentialV1
            or type(claimant) is not _ClaimantIdentityV1
        ):
            raise ProductionLocalIssuanceAuthorityError("EXACT_ACTIVE_CREDENTIAL_RECORDS_REQUIRED")
        resolutions: tuple[
            tuple[
                PostgreSQLRequesterCredentialRegistryProviderV2
                | PostgreSQLClaimantIdentityRegistryProviderV2,
                str,
                str,
                int,
                int,
                tuple[CredentialRoleIdentity, ...],
            ],
            ...,
        ] = (
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
            if (
                current.principal_id != principal
                or current.key_id != key_id
                or current.key_version != key_version
                or current.registry_revision != revision
                or current.lifecycle.value != "ACTIVE"
                or tuple(item for item in evidence if item.credential_identity == key_id)
                != (expected,)
                or port.public_key(key_id) != current.public_key
            ):
                raise ProductionLocalIssuanceAuthorityError("ACTIVE_CREDENTIAL_EVIDENCE_CHANGED")
