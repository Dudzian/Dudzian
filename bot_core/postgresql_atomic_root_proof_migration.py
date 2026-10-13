"""Deliberate administrative installation of the isolated atomic authority V1.

There is no startup hook, environment switch, production fallback, or implicit
schema repair.  Legacy authorities are qualified under their own credentials
before the single administrative migration transaction.  Existing application
records and signed bytes are checked to remain byte-for-byte unchanged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict

from bot_core.postgresql_atomic_root_proof_qualification import (
    _ADMINISTRATIVE_ORIGIN,
    AtomicRootProofQualificationError,
    PostgreSQLAtomicRootProofPhysicalIdentity,
    PostgreSQLAtomicRootProofTestDeployment,
    _catalog_fingerprint,
    _physical_identity,
    _qualify_closed_principal,
    _qualify_roles,
    _schemas,
    qualify_atomic_root_proof_authority,
)
from bot_core.postgresql_atomic_root_proof_schema import (
    FUNC_ADMIN,
    FUNC_RUNTIME,
    PostgreSQLAtomicRootProofProvisioning,
    install_atomic_root_proof_authority,
    reviewed_function_inventory,
)
from bot_core.postgresql_entitlement_registry import (
    PostgreSQLConnectionConfig,
    PostgreSQLEntitlementRegistryProvider,
    PostgreSQLRegistryProvisioning,
    _reviewed_function_sources,
)
from bot_core.postgresql_issuer_history_schema import (
    PostgreSQLIssuerHistoryProvisioning,
    _function_sources as _history_sources,
    qualify_history_schema,
)
from bot_core.postgresql_preaccount_credentials import (
    PostgreSQLClaimantIdentityRegistryProvider,
    PostgreSQLCredentialRegistryProvisioning,
    PostgreSQLRequesterCredentialRegistryProvider,
    _apply_source,
)
from bot_core.postgresql_root_proof_retention_schema import (
    PostgreSQLRootProofRetentionProvisioning,
    _function_sources as _retention_sources,
    qualify_retention_schema,
)
from bot_core.postgresql_root_proof_signing_intent_schema import (
    PostgreSQLRootProofSigningIntentProvisioning,
    _function_sources as _intent_sources,
    qualify_signing_intent_schema,
)
from bot_core.root_proof_issuer_substrate import CredentialSemanticRole


@dataclass(frozen=True, slots=True)
class PostgreSQLAtomicRootProofLegacyConnections:
    """Separate genuine direct-login credentials required before migration."""

    entitlement_runtime: PostgreSQLConnectionConfig
    intent_writer: PostgreSQLConnectionConfig
    retention_writer: PostgreSQLConnectionConfig
    history_runtime: PostgreSQLConnectionConfig
    requester_runtime: PostgreSQLConnectionConfig
    claimant_runtime: PostgreSQLConnectionConfig

    def __post_init__(self) -> None:
        for value in (
            self.entitlement_runtime,
            self.intent_writer,
            self.retention_writer,
            self.history_runtime,
            self.requester_runtime,
            self.claimant_runtime,
        ):
            if type(value) is not PostgreSQLConnectionConfig:
                raise TypeError("exact independent legacy connection credentials required")


def _legacy_qualify(
    bootstrap: psycopg.Connection[Any],
    config: PostgreSQLAtomicRootProofProvisioning,
    connections: PostgreSQLAtomicRootProofLegacyConnections,
) -> None:
    PostgreSQLEntitlementRegistryProvider(
        connections.entitlement_runtime,
        schema=config.entitlement_schema,
        environment="PRODUCTION",
        trust_domain=config.trust_domain,
    )
    for provider, connection, schema in (
        (
            PostgreSQLRequesterCredentialRegistryProvider,
            connections.requester_runtime,
            config.requester_schema,
        ),
        (
            PostgreSQLClaimantIdentityRegistryProvider,
            connections.claimant_runtime,
            config.claimant_schema,
        ),
    ):
        provider(
            connection, schema=schema, environment="PRODUCTION", trust_domain=config.trust_domain
        )
    expected_identity = _physical_identity(bootstrap, config.instance_id)
    for connection, schema, qualifier in (
        (connections.entitlement_runtime, config.entitlement_schema, None),
        (connections.intent_writer, config.intent_schema, qualify_signing_intent_schema),
        (connections.retention_writer, config.retention_schema, qualify_retention_schema),
        (connections.history_runtime, config.history_schema, None),
        (connections.requester_runtime, config.requester_schema, None),
        (connections.claimant_runtime, config.claimant_schema, None),
    ):
        with psycopg.connect(connection.dsn, autocommit=True) as conn, conn.transaction():
            if _physical_identity(conn, config.instance_id) != expected_identity:
                raise AtomicRootProofQualificationError(
                    "legacy authority uses a different physical service"
                )
            if qualifier is not None:
                qualifier(conn, schema, config.trust_domain)
            elif schema == config.history_schema:
                qualify_history_schema(
                    conn, schema, expected_role="runtime_role", trust_domain=config.trust_domain
                )


def _component_roles(
    conn: psycopg.Connection[Any], config: PostgreSQLAtomicRootProofProvisioning
) -> tuple[tuple[tuple[str, int], ...], tuple[str, ...]]:
    roles: dict[str, int] = {}
    owners = [config.schema_owner_role]
    for schema in _schemas(config)[1:]:
        row = conn.execute(
            sql.SQL("SELECT pg_catalog.to_jsonb(m) FROM {}.metadata m WHERE singleton").format(
                sql.Identifier(schema)
            )
        ).fetchall()
        if len(row) != 1 or type(row[0][0]) is not dict:
            raise AtomicRootProofQualificationError("legacy metadata cardinality mismatch")
        metadata = row[0][0]
        if schema in (config.requester_schema, config.claimant_schema):
            metadata = metadata["config"]
        owners.append(metadata["schema_owner_role"])
        for key in ("schema_owner_role", "runtime_role", "issuer_writer_role", "admin_role"):
            if key in metadata:
                name, stored_oid = metadata[key], metadata[key + "_oid"]
                # PostgreSQL to_jsonb renders typed OID columns as JSON text;
                # preaccount config JSON retains integer OIDs. Normalize both
                # authoritative encodings before adapting an oid[] parameter.
                if type(stored_oid) not in (int, str):
                    raise AtomicRootProofQualificationError("invalid stored authority role OID")
                oid = int(stored_oid)
                if not 0 < oid < 2**32:
                    raise AtomicRootProofQualificationError("invalid stored authority role OID")
                if name in roles or oid in roles.values():
                    raise AtomicRootProofQualificationError("legacy semantic authority roles alias")
                roles[name] = oid
    for name in (config.schema_owner_role, config.issuer_writer_role, config.test_admin_role):
        if name in roles:
            raise AtomicRootProofQualificationError(
                "coordinator aliases a legacy semantic principal"
            )
        row = conn.execute(
            "SELECT oid FROM pg_catalog.pg_roles WHERE rolname=%s", (name,)
        ).fetchone()
        if row is None:
            raise AtomicRootProofQualificationError("atomic principal missing")
        roles[name] = row[0]
    return tuple(sorted(roles.items())), tuple(sorted(owners))


def _application_bytes_fingerprint(
    conn: psycopg.Connection[Any], config: PostgreSQLAtomicRootProofProvisioning
) -> str:
    """Migration conservation check; BYTEA is represented losslessly by JSONB."""
    records = []
    for schema, tables in (
        (config.entitlement_schema, ("history", "lineages")),
        (config.intent_schema, ("intents",)),
        (config.retention_schema, ("prepared",)),
        (config.history_schema, ("streams", "records", "attestations")),
        (config.requester_schema, ("credentials", "history", "current_credentials")),
        (config.claimant_schema, ("credentials", "history", "current_credentials")),
    ):
        for table in tables:
            rows = conn.execute(
                sql.SQL("SELECT pg_catalog.to_jsonb(t)::text FROM {}.{} t ORDER BY 1").format(
                    sql.Identifier(schema), sql.Identifier(table)
                )
            ).fetchall()
            records.append((schema, table, rows))
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


def _verify_reviewed_helpers(
    conn: psycopg.Connection[Any],
    config: PostgreSQLAtomicRootProofProvisioning,
    inventory: dict[tuple[str, str], tuple[str, str, str, str]],
) -> None:
    for (schema, signature), (owner, source, _declaration, _returns) in inventory.items():
        row = conn.execute(
            "SELECT p.proowner,r.rolname,p.prosrc,p.prosecdef,p.proconfig,l.lanname "
            "FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles r ON r.oid=p.proowner "
            "JOIN pg_catalog.pg_language l ON l.oid=p.prolang "
            "WHERE p.oid=pg_catalog.to_regprocedure(%s)",
            (schema + "." + signature,),
        ).fetchone()
        if row is None or row[1:] != (owner, source, True, ["search_path=pg_catalog"], "plpgsql"):
            raise AtomicRootProofQualificationError(
                "installed helper differs from reviewed static source"
            )
        expected = {(row[0], "EXECUTE", False)}
        if schema != config.schema and not signature.endswith("constraint()"):
            caller = config.schema_owner_role
        elif signature in FUNC_RUNTIME:
            caller = config.issuer_writer_role
        elif signature in FUNC_ADMIN:
            caller = config.test_admin_role
        elif signature.startswith("assert_bound_link("):
            caller = conn.execute(
                sql.SQL("SELECT schema_owner_role FROM {}.metadata WHERE singleton").format(
                    sql.Identifier(config.entitlement_schema)
                )
            ).fetchone()[0]
        elif signature.startswith("assert_history_link("):
            caller = conn.execute(
                sql.SQL("SELECT schema_owner_role FROM {}.metadata WHERE singleton").format(
                    sql.Identifier(config.history_schema)
                )
            ).fetchone()[0]
        else:
            caller = None
        if caller is not None:
            expected.add(
                (
                    conn.execute(
                        "SELECT oid FROM pg_catalog.pg_roles WHERE rolname=%s", (caller,)
                    ).fetchone()[0],
                    "EXECUTE",
                    False,
                )
            )
        if schema == config.schema and signature == "qualify_catalog()":
            expected.add(
                (
                    conn.execute(
                        "SELECT oid FROM pg_catalog.pg_roles WHERE rolname=%s",
                        (config.test_admin_role,),
                    ).fetchone()[0],
                    "EXECUTE",
                    False,
                )
            )
        acl = set(
            conn.execute(
                "SELECT a.grantee,a.privilege_type,a.is_grantable FROM pg_catalog.pg_proc p "
                "CROSS JOIN LATERAL pg_catalog.aclexplode(p.proacl) a "
                "WHERE p.oid=pg_catalog.to_regprocedure(%s)",
                (schema + "." + signature,),
            ).fetchall()
        )
        if acl != expected:
            raise AtomicRootProofQualificationError(
                "installed helper ACL differs from V1 allowlist"
            )


def _verify_principal_boundary(
    conn: psycopg.Connection[Any], config: PostgreSQLAtomicRootProofProvisioning
) -> None:
    for principal in (config.issuer_writer_role, config.test_admin_role):
        _qualify_closed_principal(conn, principal)
        if conn.execute(
            "SELECT pg_catalog.has_database_privilege(%s,pg_catalog.current_database(),'CREATE,TEMPORARY')",
            (principal,),
        ).fetchone() != (False,):
            raise AtomicRootProofQualificationError("new authority has database DDL privileges")
        if (
            conn.execute(
                "SELECT 1 FROM pg_catalog.pg_namespace "
                "WHERE pg_catalog.has_schema_privilege(%s,oid,'CREATE') LIMIT 1",
                (principal,),
            ).fetchone()
            is not None
        ):
            raise AtomicRootProofQualificationError("new authority has schema DDL privileges")
        if (
            conn.execute(
                "SELECT 1 FROM pg_catalog.pg_class WHERE relkind IN ('r','p','v','m','f') AND "
                "pg_catalog.has_table_privilege(%s,oid,'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER') "
                "LIMIT 1",
                (principal,),
            ).fetchone()
            is not None
        ):
            raise AtomicRootProofQualificationError("new authority has direct table DML privileges")
        if (
            conn.execute(
                "SELECT 1 FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace "
                "WHERE p.prosecdef AND n.nspname<>ALL(%s) AND "
                "pg_catalog.has_schema_privilege(%s,n.oid,'USAGE') AND "
                "pg_catalog.has_function_privilege(%s,p.oid,'EXECUTE') LIMIT 1",
                (_schemas(config), principal, principal),
            ).fetchone()
            is not None
        ):
            raise AtomicRootProofQualificationError("new authority can execute an external definer")


def _verify_original_sources(
    conn: psycopg.Connection[Any], config: PostgreSQLAtomicRootProofProvisioning
) -> dict[str, set[str]]:
    """Inventory V2 preserves every V1 definition while adding fixed helpers."""
    inventories = {}
    for schema, kind in (
        (config.entitlement_schema, "entitlement"),
        (config.intent_schema, "intent"),
        (config.retention_schema, "retention"),
        (config.history_schema, "history"),
        (config.requester_schema, "requester"),
        (config.claimant_schema, "claimant"),
    ):
        metadata = conn.execute(
            sql.SQL("SELECT pg_catalog.to_jsonb(m) FROM {}.metadata m WHERE singleton").format(
                sql.Identifier(schema)
            )
        ).fetchone()[0]
        if kind in {"requester", "claimant"}:
            metadata = metadata["config"]
        owner = metadata["schema_owner_role"]
        if kind == "entitlement":
            sources = _reviewed_function_sources(
                PostgreSQLRegistryProvisioning(
                    schema,
                    owner,
                    metadata["runtime_role"],
                    metadata["admin_role"],
                    config.trust_domain,
                )
            )
        elif kind == "intent":
            sources = _intent_sources(
                PostgreSQLRootProofSigningIntentProvisioning(
                    schema, owner, metadata["issuer_writer_role"], config.trust_domain
                )
            )
        elif kind == "retention":
            sources = _retention_sources(
                PostgreSQLRootProofRetentionProvisioning(
                    schema, owner, metadata["runtime_role"], config.trust_domain
                )
            )
        elif kind == "history":
            sources = _history_sources(
                PostgreSQLIssuerHistoryProvisioning(
                    schema,
                    owner,
                    metadata["runtime_role"],
                    metadata["admin_role"],
                    config.trust_domain,
                )
            )
        else:
            credential_config = PostgreSQLCredentialRegistryProvisioning(
                **{
                    name: metadata[name]
                    for name in (
                        "schema",
                        "schema_owner_role",
                        "runtime_role",
                        "admin_role",
                        "environment",
                        "trust_domain",
                    )
                }
            )
            peer = PostgreSQLCredentialRegistryProvisioning(**metadata["peer"])
            semantic_role = (
                CredentialSemanticRole.ROOT_PROOF_REQUESTER
                if kind == "requester"
                else CredentialSemanticRole.ROOT_PROOF_CLAIMANT
            )
            sources = {"apply(jsonb)": _apply_source(credential_config, peer, semantic_role)}
        for signature, source in sources.items():
            row = conn.execute(
                "SELECT r.rolname,p.prosrc,p.prosecdef,p.proconfig FROM pg_catalog.pg_proc p "
                "JOIN pg_catalog.pg_roles r ON r.oid=p.proowner "
                "WHERE p.oid=pg_catalog.to_regprocedure(%s)",
                (schema + "." + signature,),
            ).fetchone()
            if row != (owner, source, True, ["search_path=pg_catalog"]):
                raise AtomicRootProofQualificationError("legacy source changed across V2 migration")
        inventories[schema] = set(sources)
    return inventories


def _verify_database_manifest(
    conn: psycopg.Connection[Any], config: PostgreSQLAtomicRootProofProvisioning
) -> None:
    result = conn.execute(
        sql.SQL(
            "SELECT component_inventory_version,snapshot={}.catalog_snapshot() "
            "FROM {}.catalog_manifest WHERE singleton"
        ).format(sql.Identifier(config.schema), sql.Identifier(config.schema))
    ).fetchall()
    if result != [(2, True)]:
        raise AtomicRootProofQualificationError(
            "immutable V2 catalog manifest differs from live catalog"
        )


def _require_unissued_legacy_components(
    conn: psycopg.Connection[Any], config: PostgreSQLAtomicRootProofProvisioning
) -> None:
    """Refuse an unsafe synthetic backfill of pre-coordinator issuance.

    No old BOUND/history decision is reclassified or rewritten.  Such a database
    needs a separately reviewed conservation migration proving its original
    complete proof linkage; V1 provides no way to invent that evidence.
    """
    bound = conn.execute(
        sql.SQL(
            "SELECT 1 FROM {}.history WHERE state #>> '{{binding,kind}}'='BOUND' LIMIT 1"
        ).format(sql.Identifier(config.entitlement_schema))
    ).fetchone()
    issued = conn.execute(
        sql.SQL("SELECT 1 FROM {}.records LIMIT 1").format(sql.Identifier(config.history_schema))
    ).fetchone()
    attested = conn.execute(
        sql.SQL("SELECT 1 FROM {}.attestations LIMIT 1").format(
            sql.Identifier(config.history_schema)
        )
    ).fetchone()
    if bound is not None or issued is not None or attested is not None:
        raise AtomicRootProofQualificationError(
            "pre-existing issuance needs separately qualified original proof/history linkage"
        )


def requalify_existing_atomic_root_proof_v1(
    bootstrap: PostgreSQLConnectionConfig,
    writer_connection: PostgreSQLConnectionConfig,
    config: PostgreSQLAtomicRootProofProvisioning,
    *,
    expected_identity: PostgreSQLAtomicRootProofPhysicalIdentity,
) -> PostgreSQLAtomicRootProofTestDeployment:
    """Explicit read-only administrator reconstruction after process restart.

    This operation performs no CREATE, ALTER, repair, signing, or approval.  It
    must be invoked by the isolated-test administrator; detached caller pins do
    not reconstruct an authority.  It verifies compiled sources and the original
    immutable database manifest before issuing a new process-local deployment.
    """
    if (
        type(bootstrap) is not PostgreSQLConnectionConfig
        or type(writer_connection) is not PostgreSQLConnectionConfig
        or type(config) is not PostgreSQLAtomicRootProofProvisioning
        or type(expected_identity) is not PostgreSQLAtomicRootProofPhysicalIdentity
    ):
        raise TypeError("exact explicit administrator reconstruction inputs required")
    if conninfo_to_dict(writer_connection.dsn).get("user") != config.issuer_writer_role:
        raise AtomicRootProofQualificationError("explicit isolated writer credential required")
    try:
        with psycopg.connect(bootstrap.dsn, autocommit=True) as conn, conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE READ ONLY")
            conn.execute("SET LOCAL search_path=pg_catalog")
            if _physical_identity(conn, config.instance_id) != expected_identity:
                raise AtomicRootProofQualificationError(
                    "reconstruction used a different physical service"
                )
            inventory = reviewed_function_inventory(conn, config)
            _verify_reviewed_helpers(conn, config, inventory)
            expected = _verify_original_sources(conn, config)
            expected[config.schema] = set()
            for schema, signature in inventory:
                expected.setdefault(schema, set()).add(signature)
            for schema, signatures in expected.items():
                observed = conn.execute(
                    "SELECT p.proname || '(' || pg_catalog.replace("
                    "pg_catalog.oidvectortypes(p.proargtypes),', ', ',') || ')' "
                    "FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace "
                    "WHERE n.nspname=%s",
                    (schema,),
                ).fetchall()
                if {row[0] for row in observed} != signatures:
                    raise AtomicRootProofQualificationError(
                        "V2 function inventory differs from reviewed code"
                    )
            _verify_database_manifest(conn, config)
            _verify_principal_boundary(conn, config)
            role_oids, owners = _component_roles(conn, config)
            deployment = PostgreSQLAtomicRootProofTestDeployment(
                writer_connection,
                config,
                _physical_identity(conn, config.instance_id),
                _catalog_fingerprint(conn, config, role_oids),
                role_oids,
                owners,
                _ADMINISTRATIVE_ORIGIN,
            )
            _qualify_roles(conn, deployment)
        with (
            psycopg.connect(writer_connection.dsn, autocommit=True) as writer,
            writer.transaction(),
        ):
            qualify_atomic_root_proof_authority(writer, deployment)
        return deployment
    except AtomicRootProofQualificationError:
        raise
    except (psycopg.Error, ValueError, TypeError, KeyError, IndexError) as exc:
        raise AtomicRootProofQualificationError(
            "existing atomic authority reconstruction refused"
        ) from exc


def administer_atomic_root_proof_v1(
    bootstrap: PostgreSQLConnectionConfig,
    writer_connection: PostgreSQLConnectionConfig,
    config: PostgreSQLAtomicRootProofProvisioning,
    *,
    legacy_connections: PostgreSQLAtomicRootProofLegacyConnections,
) -> PostgreSQLAtomicRootProofTestDeployment:
    """Explicit, non-destructive migration, qualified before and after install.

    The caller supplies separate isolated-test credentials.  No bootstrap
    credential is retained.  There is deliberately no normal application
    startup caller.  PostgreSQL authority can still roll back the whole database;
    this test-only migration does not claim production custody isolation.
    """
    if (
        type(bootstrap) is not PostgreSQLConnectionConfig
        or type(writer_connection) is not PostgreSQLConnectionConfig
        or type(config) is not PostgreSQLAtomicRootProofProvisioning
        or type(legacy_connections) is not PostgreSQLAtomicRootProofLegacyConnections
    ):
        raise TypeError("exact explicit administrative migration inputs required")
    if conninfo_to_dict(writer_connection.dsn).get("user") != config.issuer_writer_role:
        raise AtomicRootProofQualificationError("explicit isolated writer credential required")
    try:
        with psycopg.connect(bootstrap.dsn, autocommit=True) as conn:
            if conn.info.server_version // 10000 != 16 or conn.execute(
                "SELECT pg_catalog.current_setting('fsync'),"
                "pg_catalog.current_setting('synchronous_commit'),"
                "pg_catalog.current_setting('full_page_writes'),pg_catalog.pg_is_in_recovery()"
            ).fetchone() != ("on", "on", "on", False):
                raise AtomicRootProofQualificationError("durable PostgreSQL 16 primary required")
            _legacy_qualify(conn, config, legacy_connections)
            with conn.transaction():
                conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
                conn.execute("SET LOCAL search_path=pg_catalog")
                # Prevent concurrent owner-owned writes while proving that the
                # migration preserves the entire pre-existing application set.
                for schema in _schemas(config)[1:]:
                    relations = conn.execute(
                        "SELECT c.relname FROM pg_catalog.pg_class c "
                        "JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
                        "WHERE n.nspname=%s AND c.relkind='r' ORDER BY c.relname",
                        (schema,),
                    ).fetchall()
                    for (table,) in relations:
                        conn.execute(
                            sql.SQL("LOCK TABLE {}.{} IN ACCESS EXCLUSIVE MODE").format(
                                sql.Identifier(schema), sql.Identifier(table)
                            )
                        )
                before = _application_bytes_fingerprint(conn, config)
                _require_unissued_legacy_components(conn, config)
                inventory = install_atomic_root_proof_authority(conn, config)
                role_oids, owners = _component_roles(conn, config)
                _verify_reviewed_helpers(conn, config, inventory)
                _verify_original_sources(conn, config)
                _verify_database_manifest(conn, config)
                _verify_principal_boundary(conn, config)
                if _application_bytes_fingerprint(conn, config) != before:
                    raise AtomicRootProofQualificationError(
                        "migration changed retained application bytes"
                    )
                deployment = PostgreSQLAtomicRootProofTestDeployment(
                    writer_connection,
                    config,
                    _physical_identity(conn, config.instance_id),
                    _catalog_fingerprint(conn, config, role_oids),
                    role_oids,
                    owners,
                    _ADMINISTRATIVE_ORIGIN,
                )
                _qualify_roles(conn, deployment)
            # A new authenticated connection proves the real writer, rather than
            # treating an administrative SET ROLE as a credential qualification.
            with (
                psycopg.connect(writer_connection.dsn, autocommit=True) as writer,
                writer.transaction(),
            ):
                qualify_atomic_root_proof_authority(writer, deployment)
            return deployment
    except AtomicRootProofQualificationError:
        raise
    except (psycopg.Error, ValueError, TypeError, KeyError, IndexError) as exc:
        raise AtomicRootProofQualificationError("explicit atomic migration refused") from exc


__all__ = [
    "PostgreSQLAtomicRootProofLegacyConnections",
    "administer_atomic_root_proof_v1",
    "requalify_existing_atomic_root_proof_v1",
]
