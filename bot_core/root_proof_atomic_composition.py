"""Read-only, non-authorizing qualification of a deployment-pinned issuer graph.

Database identity comes from PostgreSQL control/catalog data, never from DSNs.
The existing authorities own their connections and commits and their SQL guards
require different direct-login principals. There is no reviewed atomic owner or
same-connection participant implementation. Consequently this module has no
READY result, issuance operation, coordinator callback, or capability factory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

import psycopg
from psycopg import sql

from bot_core import postgresql_entitlement_registry as entitlement_schema
from bot_core.authenticated_issuer_history import HistoryStreamIdentity
from bot_core.entitlement_registry_contract import RegistryReadOutcome, RegistrySubject
from bot_core.local_signing_custody import (
    LocalHistoryAttestationSigningProvider,
    LocalRootProofSigningProvider,
    SigningKeyLifecycle,
)
from bot_core.postgresql_authenticated_issuer_history import PostgreSQLAuthenticatedIssuerHistory
from bot_core.postgresql_checkpoint_authority import PostgreSQLCheckpointAuthority
from bot_core.postgresql_entitlement_registry import (
    PostgreSQLConnectionConfig,
    PostgreSQLEntitlementRegistryProvider,
)
from bot_core.postgresql_preaccount_credentials import (
    PostgreSQLClaimantIdentityRegistryProvider,
    PostgreSQLRequesterCredentialRegistryProvider,
)
from bot_core.postgresql_root_proof_issuance_authority import PostgreSQLRootProofIssuanceAuthority
from bot_core.postgresql_root_proof_retention import (
    PostgreSQLRootProofRetention,
    RootProofRetentionScope,
    _validate_snapshot,
)
from bot_core.postgresql_root_proof_signing_intent import PostgreSQLRootProofSigningIntent
from bot_core.root_proof_issuer_substrate import (
    CredentialRoleIdentity,
    ProviderIdentity,
    ProviderRole,
    RootProofIssuerCompositionGate,
    SecurityProfile,
    SecurityProfileIdentity,
)

_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]{0,62}\Z")
_AUTHORITY_NAMES = (
    "entitlement",
    "requester",
    "claimant",
    "signing_intent",
    "retention",
    "history",
    "checkpoint",
)
_IMPLEMENTATION_BLOCKERS = (
    "ATOMIC_COORDINATOR_NOT_IMPLEMENTED",
    "SAME_CONNECTION_PARTICIPATION_NOT_IMPLEMENTED",
    "INDEPENDENT_CONNECTION_SAME_PHYSICAL_TRANSACTION_UNPROVEN",
    "QUALIFIED_COORDINATOR_GRANT_MISSING",
)
# PostgreSQL 16 denies these server-file/admin functions to PUBLIC by default.
# Check effective privileges too: a deployment-wide PUBLIC grant can expose
# them without a grant naming this owner. Ordinary PUBLIC built-ins are not
# treated as excess authority solely because they are executable.
_RESTRICTED_CATALOG_FUNCTIONS = (
    "pg_read_file",
    "pg_read_binary_file",
    "pg_ls_dir",
    "pg_stat_file",
    "lo_import",
    "lo_export",
    "pg_reload_conf",
    "pg_promote",
    "pg_backup_start",
    "pg_backup_stop",
    "pg_switch_wal",
    "pg_wal_replay_pause",
    "pg_wal_replay_resume",
    "pg_rotate_logfile",
)


def _identifier(value: object) -> bool:
    return type(value) is str and _IDENTIFIER.fullmatch(value) is not None


@dataclass(frozen=True, slots=True)
class AtomicRootProofAuthorityPin:
    """Internal deployment selection; none of these fields proves authority."""

    connection: PostgreSQLConnectionConfig
    schema: str
    login_role: str

    def __post_init__(self) -> None:
        if (
            type(self.connection) is not PostgreSQLConnectionConfig
            or not _identifier(self.schema)
            or not _identifier(self.login_role)
        ):
            raise TypeError("EXACT_DEPLOYMENT_AUTHORITY_PIN_REQUIRED")


@dataclass(frozen=True, slots=True)
class AtomicRootProofDeployment:
    """Internal frozen deployment data, inaccessible to public orchestration.

    No ports, capability flags, identity claims, callbacks, or grant-writing
    options are accepted. The selected authorities are inspected independently.
    """

    scope: RootProofRetentionScope
    entitlement: AtomicRootProofAuthorityPin
    requester: AtomicRootProofAuthorityPin
    claimant: AtomicRootProofAuthorityPin
    signing_intent: AtomicRootProofAuthorityPin
    retention: AtomicRootProofAuthorityPin
    history: AtomicRootProofAuthorityPin
    checkpoint: AtomicRootProofAuthorityPin
    transaction_owner: PostgreSQLConnectionConfig
    transaction_owner_role: str

    def __post_init__(self) -> None:
        if (
            type(self.scope) is not RootProofRetentionScope
            or any(
                type(getattr(self, name)) is not AtomicRootProofAuthorityPin
                for name in _AUTHORITY_NAMES
            )
            or type(self.transaction_owner) is not PostgreSQLConnectionConfig
            or not _identifier(self.transaction_owner_role)
        ):
            raise TypeError("EXACT_INTERNAL_ATOMIC_DEPLOYMENT_REQUIRED")


@dataclass(frozen=True, slots=True)
class PostgreSQLPhysicalDatabaseIdentity:
    """Observed control/catalog lineage, also copied by replicas and backups.

    Equality cannot establish that independent connections reach one writable
    physical database or participate in one physical transaction.
    """

    system_identifier: str
    database_oid: int
    database_name: str
    database_owner_oid: int


@dataclass(frozen=True, slots=True)
class AtomicRootProofAuthorityObservation:
    """Live schema/ACL/scope observations, not semantic issuance authority."""

    authority: str
    physical_identity: PostgreSQLPhysicalDatabaseIdentity | None
    login_role: str | None
    role_oid: int | None
    schema_owner_oid: int | None
    qualified: bool
    credentials: tuple[CredentialRoleIdentity, ...] = ()


@dataclass(frozen=True, slots=True)
class AtomicRootProofCompositionQualification:
    """Detached diagnostics, never an authorization or cached readiness token."""

    failures: tuple[str, ...]
    authorities: tuple[AtomicRootProofAuthorityObservation, ...] = ()
    state: Literal["NOT_READY"] = "NOT_READY"
    authorizes_operations: Literal[False] = False

    def __post_init__(self) -> None:
        if self.state != "NOT_READY" or self.authorizes_operations is not False:
            raise TypeError("ATOMIC_QUALIFICATION_NEVER_AUTHORIZES_OPERATIONS")
        if type(self.failures) is not tuple or not self.failures:
            raise TypeError("ATOMIC_NOT_READY_REQUIRES_BLOCKERS")


def _physical_identity(conn: psycopg.Connection[Any]) -> PostgreSQLPhysicalDatabaseIdentity:
    if not 160000 <= conn.info.server_version < 170000:
        raise ValueError("POSTGRESQL_16_REQUIRED")
    row = conn.execute(
        "SELECT c.system_identifier::text,d.oid::bigint,d.datname::text,d.datdba::bigint "
        "FROM pg_catalog.pg_control_system() c CROSS JOIN pg_catalog.pg_database d "
        "WHERE d.datname=pg_catalog.current_database() AND NOT d.datistemplate "
        "AND d.datallowconn"
    ).fetchall()
    if (
        len(row) != 1
        or len(row[0]) != 4
        or type(row[0][0]) is not str
        or re.fullmatch(r"[1-9][0-9]{0,19}", row[0][0]) is None
        or int(row[0][0]) > 18_446_744_073_709_551_615
        or type(row[0][1]) is not int
        or not 0 < row[0][1] <= 4_294_967_295
        or type(row[0][2]) is not str
        or not row[0][2]
        or type(row[0][3]) is not int
        or not 0 < row[0][3] <= 4_294_967_295
    ):
        raise ValueError("PHYSICAL_DATABASE_CATALOG_EVIDENCE_REQUIRED")
    return PostgreSQLPhysicalDatabaseIdentity(*row[0])


def _qualify_entitlement(
    conn: psycopg.Connection[Any], port: PostgreSQLEntitlementRegistryProvider
) -> None:
    """Run the reviewed registry validators on this read-only connection.

    Its existing _qualify opens an independent transaction without READ ONLY;
    use its same source validators without borrowing its connection lifecycle.
    """
    rows = conn.execute(
        sql.SQL(
            "SELECT schema_identity,schema_version,profile,trust_domain,storage_family,"
            "schema_owner_role,schema_owner_role_oid,runtime_role,runtime_role_oid,"
            "admin_role,admin_role_oid,function_definition_sha256 FROM {}.metadata"
        ).format(sql.Identifier(port._schema))
    ).fetchall()
    if (
        len(rows) != 1
        or len(rows[0]) != 12
        or rows[0][:5]
        != (
            entitlement_schema.SCHEMA_IDENTITY,
            entitlement_schema.SCHEMA_VERSION,
            entitlement_schema.PROFILE,
            port._trust_domain,
            entitlement_schema.STORAGE_FAMILY,
        )
    ):
        raise ValueError("REGISTRY_SECURITY_METADATA_MISMATCH")
    metadata = rows[0]
    roles = {
        "schema_owner_role": (metadata[5], metadata[6]),
        "runtime_role": (metadata[7], metadata[8]),
        "admin_role": (metadata[9], metadata[10]),
    }
    manifest = metadata[11]
    if type(manifest) is not dict or set(manifest) != set(entitlement_schema._FUNCTION_SIGNATURES):
        raise ValueError("REGISTRY_FUNCTION_MANIFEST_MISMATCH")
    port._qualify_role(conn, roles)
    if conn.execute(
        "SELECT pg_catalog.current_setting('fsync'),"
        "pg_catalog.current_setting('synchronous_commit')"
    ).fetchone() != ("on", "on"):
        raise ValueError("REGISTRY_DURABILITY_MISMATCH")
    port._qualify_relations(conn, roles)
    port._qualify_columns(conn)
    port._qualify_constraints(conn)
    port._qualify_relation_acls(conn, roles)
    port._qualify_functions(conn, roles, manifest)


def _port_selection_failures(
    deployment: AtomicRootProofDeployment,
    issuance: PostgreSQLRootProofIssuanceAuthority,
    signing: PostgreSQLRootProofSigningIntent,
    retention: PostgreSQLRootProofRetention,
    history: PostgreSQLAuthenticatedIssuerHistory,
    checkpoint: PostgreSQLCheckpointAuthority,
) -> tuple[dict[str, Any], list[str]]:
    ports = {
        "entitlement": issuance.entitlement_registry,
        "requester": issuance.requester_registry,
        "claimant": issuance.claimant_registry,
        "signing_intent": signing,
        "retention": retention,
        "history": history,
        "checkpoint": checkpoint,
    }
    failures: list[str] = []
    if any(
        type(ports[name]) is not expected
        for name, expected in (
            ("entitlement", PostgreSQLEntitlementRegistryProvider),
            ("requester", PostgreSQLRequesterCredentialRegistryProvider),
            ("claimant", PostgreSQLClaimantIdentityRegistryProvider),
        )
    ):
        return ports, ["EXACT_GENUINE_ISSUANCE_PORTS_REQUIRED"]
    scope = deployment.scope
    subject = issuance._entitlement_subject
    if type(subject) is not RegistrySubject or (subject.environment, subject.trust_domain) != (
        scope.environment,
        scope.trust_domain,
    ):
        failures.append("ISSUANCE_AUTHORITY_SCOPE_MISMATCH")
    signing_prefix = "_PostgreSQLRootProofSigningIntent__"
    retention_prefix = "_PostgreSQLRootProofRetention__"
    if getattr(signing, signing_prefix + "authority") is not issuance:
        failures.append("SIGNING_INTENT_ISSUANCE_AUTHORITY_SUBSTITUTION")
    signer = getattr(signing, signing_prefix + "signer")
    if (
        type(signer) is not LocalRootProofSigningProvider
        or getattr(retention, retention_prefix + "verification_authority") is not signer
    ):
        failures.append("PINNED_CUSTODY_AUTHORITY_SUBSTITUTION")
    else:
        try:
            snapshot = signer.credential_snapshot()
            _validate_snapshot(snapshot, scope)
            if snapshot.lifecycle_state is not SigningKeyLifecycle.ACTIVE:
                failures.append("PINNED_CUSTODY_ACTIVE_LIFECYCLE_REQUIRED")
        except Exception:
            failures.append("PINNED_CUSTODY_IDENTITY_UNAVAILABLE")
    if checkpoint._history is not history:
        failures.append("CHECKPOINT_HISTORY_AUTHORITY_SUBSTITUTION")
    for name, port in ports.items():
        pin = getattr(deployment, name)
        prefix = signing_prefix if name == "signing_intent" else retention_prefix
        if name in ("signing_intent", "retention"):
            connection, schema = (
                getattr(port, prefix + "connection"),
                getattr(port, prefix + "schema"),
            )
            if getattr(port, prefix + "scope") != scope:
                failures.append(f"AUTHORITY_{name.upper()}_SCOPE_MISMATCH")
        else:
            connection, schema = port._connection, port._schema
        if (
            type(connection) is not PostgreSQLConnectionConfig
            or connection != pin.connection
            or schema != pin.schema
        ):
            failures.append(f"AUTHORITY_{name.upper()}_DEPLOYMENT_SELECTION_MISMATCH")
        if name in ("entitlement", "requester", "claimant") and (
            port._environment,
            port._trust_domain,
        ) != (scope.environment, scope.trust_domain):
            failures.append(f"AUTHORITY_{name.upper()}_SCOPE_MISMATCH")
        if name in ("history", "checkpoint"):
            stream = port._stream
            if type(stream) is not HistoryStreamIdentity or (
                stream.security_profile,
                stream.environment,
                stream.trust_domain,
                stream.product_scope,
                stream.issuer_authority_identity,
            ) != (
                scope.security_profile.value,
                scope.environment,
                scope.trust_domain,
                scope.product_scope,
                scope.issuer_id,
            ):
                failures.append(f"AUTHORITY_{name.upper()}_SCOPE_MISMATCH")
    if history._stream != checkpoint._stream:
        failures.append("EXACT_HISTORY_CHECKPOINT_STREAM_REQUIRED")
    if len({getattr(deployment, name).schema for name in _AUTHORITY_NAMES}) != 7:
        failures.append("DISTINCT_AUTHORITY_SCHEMAS_REQUIRED")
    if len({getattr(deployment, name).login_role for name in _AUTHORITY_NAMES}) != 7:
        failures.append("DISTINCT_AUTHORITY_LOGIN_ROLES_REQUIRED")
    if deployment.transaction_owner_role in {
        getattr(deployment, name).login_role for name in _AUTHORITY_NAMES
    }:
        failures.append("DISTINCT_TRANSACTION_OWNER_REQUIRED")
    return ports, failures


def _qualification_failure_detail(error: Exception) -> str:
    # Only stable categories leave this boundary. Driver messages can carry DSNs.
    text = str(error).lower()
    cause = error.__cause__
    if cause is not None:
        text += " " + str(cause).lower()
    if "membership" in text:
        return "ROLE_MEMBERSHIP_FORBIDDEN"
    if any(word in text for word in ("principal", "login", "role identity", "role attributes")):
        return "ROLE_OR_PRINCIPAL_REJECTED"
    if any(word in text for word in ("acl", "privilege", "owner", "grant")):
        return "ACL_OR_OWNERSHIP_REJECTED"
    return "SCHEMA_QUALIFICATION_FAILED"


def _probe_authority(
    name: str,
    pin: AtomicRootProofAuthorityPin,
    port: Any,
    issuance: PostgreSQLRootProofIssuanceAuthority,
    scope: RootProofRetentionScope,
) -> tuple[AtomicRootProofAuthorityObservation, list[str]]:
    physical = None
    login = None
    role_oid = None
    owner_oid = None
    qualified = False
    credentials: tuple[CredentialRoleIdentity, ...] = ()
    failures: list[str] = []
    label = name.upper()
    try:
        with psycopg.connect(pin.connection.dsn, autocommit=True) as conn, conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            conn.execute("SET LOCAL search_path=pg_catalog")
            role = conn.execute(
                "SELECT session_user::text,current_user::text,r.oid::bigint "
                "FROM pg_catalog.pg_roles r WHERE r.rolname=current_user"
            ).fetchone()
            if role is None or role[:2] != (pin.login_role, pin.login_role):
                failures.append(f"AUTHORITY_{label}_DIRECT_LOGIN_MISMATCH")
            else:
                login, role_oid = role[0], role[2]
            namespace = conn.execute(
                "SELECT nspowner::bigint FROM pg_catalog.pg_namespace WHERE nspname=%s",
                (pin.schema,),
            ).fetchone()
            owner_oid = namespace[0] if namespace is not None else None
            try:
                with conn.transaction():
                    physical = _physical_identity(conn)
            except Exception:
                failures.append(f"PHYSICAL_IDENTITY_{label}_UNAVAILABLE")
            try:
                with conn.transaction():
                    if not 160000 <= conn.info.server_version < 170000:
                        raise ValueError("POSTGRESQL_16_REQUIRED")
                    if conn.execute("SELECT pg_catalog.pg_is_in_recovery()").fetchone() != (False,):
                        raise ValueError("WRITABLE_PRIMARY_REQUIRED")
                    if conn.execute(
                        "SELECT pg_catalog.current_setting('fsync'),"
                        "pg_catalog.current_setting('synchronous_commit'),"
                        "pg_catalog.current_setting('full_page_writes')"
                    ).fetchone() != ("on", "on", "on"):
                        raise ValueError("DURABILITY_REQUIRED")
                    if name == "entitlement":
                        _qualify_entitlement(conn, port)
                        retained = port._validated_history(conn, issuance._entitlement_subject)
                        if retained.outcome is not RegistryReadOutcome.FOUND or not retained.states:
                            raise ValueError("AUTHORITATIVE_ENTITLEMENT_REQUIRED")
                        if retained.states[-1].identity.product_scope != scope.product_scope:
                            raise ValueError("AUTHORITATIVE_ENTITLEMENT_PRODUCT_SCOPE_MISMATCH")
                    elif name in ("requester", "claimant"):
                        port._qualify_connection(conn)
                        principals = conn.execute(
                            sql.SQL(
                                "SELECT principal_id FROM {}.current_credentials UNION "
                                "SELECT principal_id FROM {}.history UNION "
                                "SELECT principal_id FROM {}.credentials"
                            ).format(*[sql.Identifier(pin.schema)] * 3)
                        ).fetchall()
                        retained_credentials = []
                        for (principal,) in principals:
                            lineage = port._load_history(conn, principal)
                            if not lineage:
                                raise ValueError("PUBLIC_CREDENTIAL_RETAINED_HISTORY_REQUIRED")
                            latest = {record.credential_id: record for record in lineage}
                            for record in latest.values():
                                retained_credentials.append(
                                    CredentialRoleIdentity(
                                        port._semantic_role,
                                        record.key_id,
                                        f"postgresql:{pin.schema}",
                                        f"{record.key_id}:v{record.key_version}",
                                        f"postgresql:{pin.schema}:public-lifecycle:"
                                        f"{record.lifecycle_generation}:{record.registry_revision}:"
                                        f"{record.lifecycle.value}",
                                        record.key_material_identity,
                                    )
                                )
                        if not retained_credentials:
                            raise ValueError("PUBLIC_CREDENTIAL_IDENTITY_EVIDENCE_REQUIRED")
                        credentials = tuple(retained_credentials)
                    else:
                        port._qualify(conn)
                    if name in ("entitlement", "requester", "claimant", "history", "checkpoint"):
                        identity = port.identity
                        expected_role = {
                            "entitlement": ProviderRole.ENTITLEMENT_REGISTRY,
                            "requester": ProviderRole.REQUESTER_CREDENTIAL_REGISTRY,
                            "claimant": ProviderRole.CLAIMANT_IDENTITY_REGISTRY,
                            "history": ProviderRole.ISSUER_AUTHENTICATED_HISTORY,
                            "checkpoint": ProviderRole.CHECKPOINT_AUTHORITY,
                        }[name]
                        if (
                            type(identity) is not ProviderIdentity
                            or identity.role is not expected_role
                            or identity.security
                            != SecurityProfileIdentity(scope.security_profile, scope.trust_domain)
                        ):
                            raise ValueError("EXACT_PROVIDER_ROLE_SECURITY_IDENTITY_REQUIRED")
                    qualified = login is not None
            except Exception as error:
                failures.extend(
                    (
                        f"AUTHORITY_{label}_QUALIFICATION_FAILED",
                        f"AUTHORITY_{label}_{_qualification_failure_detail(error)}",
                    )
                )
                if "WRITABLE_PRIMARY_REQUIRED" in str(error):
                    failures.append(f"AUTHORITY_{label}_WRITABLE_PRIMARY_REQUIRED")
                if "DURABILITY_REQUIRED" in str(error):
                    failures.append(f"AUTHORITY_{label}_DURABILITY_REQUIRED")
                if "PRODUCT_SCOPE_MISMATCH" in str(error):
                    failures.append("AUTHORITATIVE_ENTITLEMENT_PRODUCT_SCOPE_MISMATCH")
    except Exception:
        failures.append(f"AUTHORITY_{label}_PROBE_UNAVAILABLE")
    return AtomicRootProofAuthorityObservation(
        name, physical, login, role_oid, owner_oid, qualified, credentials
    ), failures


def _probe_transaction_owner(
    deployment: AtomicRootProofDeployment,
) -> tuple[AtomicRootProofAuthorityObservation, list[str]]:
    physical = None
    login = None
    role_oid = None
    failures: list[str] = []
    try:
        with (
            psycopg.connect(deployment.transaction_owner.dsn, autocommit=True) as conn,
            conn.transaction(),
        ):
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            conn.execute("SET LOCAL search_path=pg_catalog")
            if conn.execute(
                "SELECT pg_catalog.current_setting('fsync'),"
                "pg_catalog.current_setting('synchronous_commit'),"
                "pg_catalog.current_setting('full_page_writes'),pg_catalog.pg_is_in_recovery()"
            ).fetchone() != ("on", "on", "on", False):
                failures.append("TRANSACTION_OWNER_DURABLE_WRITABLE_PRIMARY_REQUIRED")
            role = conn.execute(
                "SELECT session_user::text,current_user::text,r.oid::bigint,r.rolcanlogin,"
                "r.rolinherit,r.rolsuper,r.rolcreaterole,r.rolcreatedb,"
                "r.rolreplication,r.rolbypassrls "
                "FROM pg_catalog.pg_roles r WHERE r.rolname=current_user"
            ).fetchone()
            if role is None or role[:2] != (
                deployment.transaction_owner_role,
                deployment.transaction_owner_role,
            ):
                failures.append("TRANSACTION_OWNER_DIRECT_LOGIN_MISMATCH")
            else:
                login, role_oid = role[0], role[2]
                if role[3:] != (True, False, False, False, False, False, False):
                    failures.append("TRANSACTION_OWNER_FORBIDDEN_ROLE_ATTRIBUTES")
                if (
                    conn.execute(
                        "SELECT 1 FROM pg_catalog.pg_auth_members "
                        "WHERE roleid=%s OR member=%s LIMIT 1",
                        (role_oid, role_oid),
                    ).fetchone()
                    is not None
                ):
                    failures.append("TRANSACTION_OWNER_ROLE_MEMBERSHIP_FORBIDDEN")
                if (
                    conn.execute(
                        "SELECT 1 FROM pg_catalog.pg_namespace WHERE nspowner=%s UNION ALL "
                        "SELECT 1 FROM pg_catalog.pg_class WHERE relowner=%s UNION ALL "
                        "SELECT 1 FROM pg_catalog.pg_proc WHERE proowner=%s LIMIT 1",
                        (role_oid, role_oid, role_oid),
                    ).fetchone()
                    is not None
                ):
                    failures.append("TRANSACTION_OWNER_OBJECT_OWNERSHIP_FORBIDDEN")
                if (
                    conn.execute(
                        "SELECT 1 FROM pg_catalog.pg_namespace n WHERE n.nspname !~ '^pg_' "
                        "AND n.nspname <> 'information_schema' "
                        "AND pg_catalog.has_schema_privilege(current_user,n.oid,'CREATE') "
                        "UNION ALL SELECT 1 WHERE pg_catalog.has_database_privilege("
                        "current_user,pg_catalog.current_database(),'CREATE') LIMIT 1"
                    ).fetchone()
                    is not None
                ):
                    failures.append("TRANSACTION_OWNER_CREATE_PRIVILEGE_FORBIDDEN")
                # PostgreSQL 16 exposes PUBLIC UPDATE on this exact built-in
                # view for normal session GUC changes, with PostgreSQL checking
                # which settings the login may change. It is not an authority
                # state table. Durability is checked independently above;
                # explicit extra grants are still rejected by the ACL inventory.
                if (
                    conn.execute(
                        "SELECT 1 FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n "
                        "ON n.oid=c.relnamespace WHERE c.relkind IN ('r','p','v','f') "
                        "AND c.oid <> 'pg_catalog.pg_settings'::regclass "
                        "AND (pg_catalog.has_table_privilege(current_user,c.oid,"
                        "'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER') OR "
                        "pg_catalog.has_any_column_privilege(current_user,c.oid,"
                        "'INSERT,UPDATE,REFERENCES')) LIMIT 1"
                    ).fetchone()
                    is not None
                ):
                    failures.append("TRANSACTION_OWNER_DIRECT_DML_FORBIDDEN")
                if (
                    conn.execute(
                        "SELECT 1 FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n "
                        "ON n.oid=c.relnamespace WHERE c.relkind='S' "
                        "AND pg_catalog.has_sequence_privilege("
                        "current_user,c.oid,'USAGE,UPDATE') LIMIT 1"
                    ).fetchone()
                    is not None
                ):
                    failures.append("TRANSACTION_OWNER_SEQUENCE_WRITE_FORBIDDEN")
                if (
                    conn.execute(
                        "SELECT 1 FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n "
                        "ON n.oid=p.pronamespace WHERE n.nspname !~ '^pg_' "
                        "AND n.nspname <> 'information_schema' "
                        "AND pg_catalog.has_function_privilege("
                        "current_user,p.oid,'EXECUTE') LIMIT 1"
                    ).fetchone()
                    is not None
                ):
                    failures.append("TRANSACTION_OWNER_UNREVIEWED_FUNCTION_EXECUTE_FORBIDDEN")
                if (
                    conn.execute(
                        "SELECT 1 FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n "
                        "ON n.oid=p.pronamespace WHERE n.nspname='pg_catalog' "
                        "AND p.proname=ANY(%s) AND pg_catalog.has_function_privilege("
                        "current_user,p.oid,'EXECUTE') LIMIT 1",
                        (list(_RESTRICTED_CATALOG_FUNCTIONS),),
                    ).fetchone()
                    is not None
                ):
                    failures.append("TRANSACTION_OWNER_RESTRICTED_CATALOG_EXECUTE_FORBIDDEN")
                # Observation of the old operations, not a request to grant
                # them to the owner. Their direct-login guards still conflict.
                accessible = conn.execute(
                    "SELECT n.nspname,p.proname FROM pg_catalog.pg_proc p "
                    "JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace "
                    "WHERE n.nspname=ANY(%s) AND p.proname=ANY(%s) "
                    "AND pg_catalog.has_function_privilege(current_user,p.oid,'EXECUTE') "
                    "AND pg_catalog.has_schema_privilege(current_user,n.oid,'USAGE')",
                    (
                        [
                            getattr(deployment, name).schema
                            for name in (
                                "entitlement",
                                "signing_intent",
                                "retention",
                                "history",
                                "checkpoint",
                            )
                        ],
                        [
                            "append_bind",
                            "reserve_intent",
                            "retain_prepared",
                            "append_record",
                            "commit_checkpoint",
                        ],
                    ),
                ).fetchall()
                if len(accessible) != 5:
                    failures.append("OLD_AUTHORITY_FUNCTIONS_INACCESSIBLE")
                for catalog, acl in (
                    ("pg_namespace", "nspacl"),
                    ("pg_class", "relacl"),
                    ("pg_type", "typacl"),
                    ("pg_parameter_acl", "paracl"),
                    ("pg_default_acl", "defaclacl"),
                    ("pg_attribute", "attacl"),
                    ("pg_tablespace", "spcacl"),
                    ("pg_language", "lanacl"),
                    ("pg_foreign_server", "srvacl"),
                    ("pg_foreign_data_wrapper", "fdwacl"),
                    ("pg_largeobject_metadata", "lomacl"),
                ):
                    if (
                        conn.execute(
                            sql.SQL(
                                "SELECT 1 FROM pg_catalog.{} o CROSS JOIN LATERAL "
                                "pg_catalog.aclexplode(o.{}) a WHERE a.grantee=%s LIMIT 1"
                            ).format(sql.Identifier(catalog), sql.Identifier(acl)),
                            (role_oid,),
                        ).fetchone()
                        is not None
                    ):
                        failures.append("TRANSACTION_OWNER_EXTRA_OBJECT_GRANT_FORBIDDEN")
                if (
                    conn.execute(
                        "SELECT 1 FROM pg_catalog.pg_proc p CROSS JOIN LATERAL "
                        "pg_catalog.aclexplode(p.proacl) a WHERE a.grantee=%s AND "
                        "(p.oid <> 'pg_catalog.pg_control_system()'::regprocedure "
                        "OR a.privilege_type <> 'EXECUTE' OR a.is_grantable) LIMIT 1",
                        (role_oid,),
                    ).fetchone()
                    is not None
                ):
                    failures.append("TRANSACTION_OWNER_EXTRA_FUNCTION_GRANT_FORBIDDEN")
                if (
                    conn.execute(
                        "SELECT 1 FROM pg_catalog.pg_database d CROSS JOIN LATERAL "
                        "pg_catalog.aclexplode(d.datacl) a WHERE a.grantee=%s AND "
                        "(d.datname <> pg_catalog.current_database() OR "
                        "a.privilege_type <> 'CONNECT' OR a.is_grantable) LIMIT 1",
                        (role_oid,),
                    ).fetchone()
                    is not None
                ):
                    failures.append("TRANSACTION_OWNER_EXTRA_DATABASE_GRANT_FORBIDDEN")
                for catalog, ownership in (
                    ("pg_database", "datdba"),
                    ("pg_tablespace", "spcowner"),
                    ("pg_type", "typowner"),
                    ("pg_collation", "collowner"),
                    ("pg_conversion", "conowner"),
                    ("pg_foreign_server", "srvowner"),
                    ("pg_foreign_data_wrapper", "fdwowner"),
                    ("pg_largeobject_metadata", "lomowner"),
                    ("pg_ts_dict", "dictowner"),
                    ("pg_ts_config", "cfgowner"),
                ):
                    if (
                        conn.execute(
                            sql.SQL("SELECT 1 FROM pg_catalog.{} WHERE {}=%s LIMIT 1").format(
                                sql.Identifier(catalog), sql.Identifier(ownership)
                            ),
                            (role_oid,),
                        ).fetchone()
                        is not None
                    ):
                        failures.append("TRANSACTION_OWNER_OBJECT_OWNERSHIP_FORBIDDEN")
            try:
                with conn.transaction():
                    physical = _physical_identity(conn)
            except Exception:
                failures.append("PHYSICAL_IDENTITY_TRANSACTION_OWNER_UNAVAILABLE")
    except Exception:
        failures.append("TRANSACTION_OWNER_PROBE_UNAVAILABLE")
    qualified = login is not None and not any(
        failure.startswith("TRANSACTION_OWNER_") for failure in failures
    )
    return AtomicRootProofAuthorityObservation(
        "transaction_owner", physical, login, role_oid, None, qualified
    ), failures


def qualify_atomic_root_proof_composition(
    deployment: AtomicRootProofDeployment,
    *,
    issuance_authority: PostgreSQLRootProofIssuanceAuthority,
    signing_intent: PostgreSQLRootProofSigningIntent,
    retention: PostgreSQLRootProofRetention,
    history: PostgreSQLAuthenticatedIssuerHistory,
    checkpoint: PostgreSQLCheckpointAuthority,
) -> AtomicRootProofCompositionQualification:
    """Inspect current live deployment evidence without mutating any authority.

    Physical co-location and narrow ACLs are necessary diagnostics. They cannot
    turn independent commits, incompatible login guards or an absent reviewed
    coordinator into one crash-atomic transaction. Each call reopens probes.
    """
    failures = list(_IMPLEMENTATION_BLOCKERS)
    if type(deployment) is not AtomicRootProofDeployment or any(
        type(port) is not expected
        for port, expected in (
            (issuance_authority, PostgreSQLRootProofIssuanceAuthority),
            (signing_intent, PostgreSQLRootProofSigningIntent),
            (retention, PostgreSQLRootProofRetention),
            (history, PostgreSQLAuthenticatedIssuerHistory),
            (checkpoint, PostgreSQLCheckpointAuthority),
        )
    ):
        return AtomicRootProofCompositionQualification(
            tuple(failures + ["EXACT_INTERNAL_COMPOSITION_REQUIRED"])
        )
    try:
        deployment.__post_init__()
        # Validate runtime fields as well as exact dataclass identities; forged
        # or subsequently altered objects do not substitute trusted selection.
        RootProofRetentionScope.__post_init__(deployment.scope)
        if deployment.scope.security_profile is not SecurityProfile.PRODUCTION_LOCAL:
            raise ValueError("PRODUCTION_LOCAL_REQUIRED")
        for name in _AUTHORITY_NAMES:
            getattr(deployment, name).__post_init__()
        ports, selection_failures = _port_selection_failures(
            deployment, issuance_authority, signing_intent, retention, history, checkpoint
        )
    except Exception:
        return AtomicRootProofCompositionQualification(
            tuple(failures + ["DEPLOYMENT_SELECTION_UNAVAILABLE"])
        )
    if selection_failures:
        return AtomicRootProofCompositionQualification(tuple(failures + selection_failures))
    observations = []
    for name in _AUTHORITY_NAMES:
        observation, issues = _probe_authority(
            name, getattr(deployment, name), ports[name], issuance_authority, deployment.scope
        )
        observations.append(observation)
        failures.extend(issues)
    owner, issues = _probe_transaction_owner(deployment)
    observations.append(owner)
    failures.extend(issues)
    public_credentials = tuple(
        credential for item in observations for credential in item.credentials
    )
    try:
        signer_prefix = "_PostgreSQLRootProofSigningIntent__"
        root_signer = getattr(signing_intent, signer_prefix + "signer")
        root_snapshot = root_signer.credential_snapshot()
        _validate_snapshot(root_snapshot, deployment.scope)
        if root_snapshot.lifecycle_state is not SigningKeyLifecycle.ACTIVE:
            failures.append("PINNED_CUSTODY_ACTIVE_LIFECYCLE_REQUIRED")
        history_credentials = []
        if type(checkpoint._authorities) is not tuple or not checkpoint._authorities:
            failures.append("HISTORY_SIGNER_CREDENTIAL_EVIDENCE_UNQUALIFIED")
        else:
            for signer in checkpoint._authorities:
                if type(signer) is not LocalHistoryAttestationSigningProvider:
                    failures.append("HISTORY_SIGNER_CREDENTIAL_EVIDENCE_UNQUALIFIED")
                    continue
                captured, failure = RootProofIssuerCompositionGate._capture_snapshot(signer)
                if (
                    failure is not None
                    or captured is None
                    or captured.identity.security
                    != SecurityProfileIdentity(
                        deployment.scope.security_profile, deployment.scope.trust_domain
                    )
                    or RootProofIssuerCompositionGate._credential_evidence_failures(captured)
                ):
                    failures.append("HISTORY_SIGNER_CREDENTIAL_EVIDENCE_UNQUALIFIED")
                    continue
                history_credentials.extend(captured.credentials)
        if RootProofIssuerCompositionGate._alias_failures(
            public_credentials + (root_snapshot.credential_identity,) + tuple(history_credentials)
        ):
            failures.append("FORBIDDEN_PUBLIC_OR_ISSUER_CREDENTIAL_ALIAS")
    except Exception:
        failures.append("ISSUER_AGGREGATE_CREDENTIAL_EVIDENCE_UNAVAILABLE")
    known = [item.physical_identity for item in observations if item.physical_identity is not None]
    if len({item.system_identifier for item in known}) > 1:
        failures.append("CROSS_CLUSTER_AUTHORITIES")
    if len({(item.system_identifier, item.database_oid) for item in known}) > 1:
        failures.append("CROSS_DATABASE_AUTHORITIES")
    if len(known) != len(observations):
        failures.append("COMPLETE_PHYSICAL_DATABASE_IDENTITY_REQUIRED")
    if any(
        item.qualified
        for item in observations
        if item.authority in ("signing_intent", "retention", "history", "checkpoint")
    ):
        failures.append("AUTHORITY_FUNCTION_DIRECT_LOGIN_CONFLICT")
    return AtomicRootProofCompositionQualification(
        tuple(dict.fromkeys(failures)), tuple(observations)
    )
