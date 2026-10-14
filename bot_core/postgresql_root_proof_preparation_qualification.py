"""Privileged Stage 9 preparation migration and administrative reconstruction.

Public V2 readers depend only on the separate read-only qualification contract.
This module alone retains lifecycle administrator and bootstrap authority.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from bot_core.postgresql_atomic_root_proof_migration import (
    PostgreSQLAtomicRootProofLegacyConnections,
    _application_bytes_fingerprint,
    _verify_database_manifest,
    _verify_principal_boundary,
)
from bot_core.postgresql_atomic_root_proof_qualification import (
    AtomicRootProofQualificationError,
    _catalog_fingerprint,
    _physical_identity,
    _schemas,
)
from bot_core.postgresql_atomic_root_proof_schema import PostgreSQLAtomicRootProofProvisioning
from bot_core.postgresql_custody_lifecycle_qualification import (
    PostgreSQLCustodyLifecycleTestDeployment,
    _bootstrap_principal,
    _lifecycle_records_fingerprint,
    _postqualify,
    _verify_v2_functions,
    validate_custody_lifecycle_deployment,
)
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core.postgresql_root_proof_preparation_readonly import (
    MIGRATION_ID,
    _BOUNDARY,
    _COMPONENTS,
    _IDENTIFIER,
    _ORIGIN,
    PostgreSQLRootProofPreparationV2Deployment,
    RootProofPreparationV2QualificationError,
    _fingerprint,
    _principal_boundary,
    _qualify_preparation_roles,
    _sources,
    _verify_preparation_sources,
    lock_root_proof_preparation_signer_v2,
    qualify_root_proof_preparation_v2,
    validate_root_proof_preparation_v2_deployment,
)


def _component_principals(
    conn: psycopg.Connection[Any], config: PostgreSQLAtomicRootProofProvisioning
) -> tuple[tuple[str, str], ...]:
    result = []
    for component, schema in zip(_COMPONENTS, _schemas(config)[1:], strict=True):
        row = conn.execute(
            sql.SQL("SELECT pg_catalog.to_jsonb(m) FROM {}.metadata m WHERE singleton").format(
                sql.Identifier(schema)
            )
        ).fetchone()
        if row is None:
            raise RootProofPreparationV2QualificationError("component metadata unavailable")
        metadata = row[0].get("config", row[0])
        result.append(
            (component, metadata["issuer_writer_role" if component == "intent" else "runtime_role"])
        )
    return tuple(result)


def _connection_components(
    connections: PostgreSQLAtomicRootProofLegacyConnections,
) -> tuple[tuple[str, PostgreSQLConnectionConfig], ...]:
    return tuple(
        zip(
            _COMPONENTS,
            (
                connections.entitlement_runtime,
                connections.intent_writer,
                connections.retention_writer,
                connections.history_runtime,
                connections.requester_runtime,
                connections.claimant_runtime,
            ),
            strict=True,
        )
    )


def _verify_component_connections(
    deployment: PostgreSQLRootProofPreparationV2Deployment,
    connections: PostgreSQLAtomicRootProofLegacyConnections,
) -> None:
    """Validate every direct credential and endpoint before administrative mutation."""
    expected = deployment.identity
    for component, connection in _connection_components(connections):
        with psycopg.connect(connection.dsn, autocommit=True) as conn:
            principal = dict(deployment.component_roles)[component]
            facts = conn.execute(
                "SELECT session_user::text,current_user::text,d.oid,d.datname,"
                "pg_catalog.current_setting('cluster_name'),pg_catalog.inet_server_addr()::text,"
                "pg_catalog.inet_server_port() FROM pg_catalog.pg_database d "
                "WHERE d.datname=pg_catalog.current_database()"
            ).fetchone()
            if facts != (
                principal,
                principal,
                expected.database_oid,
                expected.database_name,
                expected.cluster_name,
                expected.server_address,
                expected.server_port,
            ) or (conn.info.host, conn.info.port, conn.info.server_version // 10000) != (
                expected.connection_host,
                expected.connection_port,
                16,
            ):
                raise RootProofPreparationV2QualificationError(
                    "exact direct component credential and service required"
                )


def requalify_root_proof_preparation_v2(
    bootstrap: PostgreSQLConnectionConfig,
    lifecycle_deployment: PostgreSQLCustodyLifecycleTestDeployment,
    *,
    preparation_schema: str,
    connections: PostgreSQLAtomicRootProofLegacyConnections,
) -> PostgreSQLRootProofPreparationV2Deployment:
    """Read-only explicit reconstruction after restart, including unknown writes.

    This does not repair, create or overwrite any catalog object. Both the
    original immutable atomic manifest and compiled preparation source/ACL
    inventory must qualify before the process-local public pin is reconstructed.
    """
    validate_custody_lifecycle_deployment(lifecycle_deployment)
    if (
        type(bootstrap) is not PostgreSQLConnectionConfig
        or type(connections) is not PostgreSQLAtomicRootProofLegacyConnections
    ):
        raise TypeError("exact explicit administrative and component credentials required")
    config = lifecycle_deployment.atomic_deployment.provisioning
    if (
        type(preparation_schema) is not str
        or _IDENTIFIER.fullmatch(preparation_schema) is None
        or preparation_schema in _schemas(config)
    ):
        raise ValueError("separate safe preparation schema required")
    _postqualify(lifecycle_deployment)
    try:
        with psycopg.connect(bootstrap.dsn, autocommit=True) as conn, conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE READ ONLY")
            conn.execute("SET LOCAL search_path=pg_catalog")
            _bootstrap_principal(conn, config)
            atomic = lifecycle_deployment.atomic_deployment
            if _physical_identity(conn, config.instance_id) != atomic.identity:
                raise RootProofPreparationV2QualificationError(
                    "reconstruction physical authority mismatch"
                )
            _verify_v2_functions(conn, config)
            _verify_database_manifest(conn, config)
            metadata = conn.execute(
                sql.SQL(
                    "SELECT migration_id,boundary,predecessor_catalog_sha256 FROM {}.metadata WHERE singleton"
                ).format(sql.Identifier(preparation_schema))
            ).fetchall()
            if len(metadata) != 1 or metadata[0][:2] != (MIGRATION_ID, _BOUNDARY):
                raise RootProofPreparationV2QualificationError(
                    "exact durable preparation migration required"
                )
            deployment = PostgreSQLRootProofPreparationV2Deployment(
                config,
                atomic.identity,
                preparation_schema,
                atomic.role_oids,
                atomic.owner_roles,
                _component_principals(conn, config),
                "0" * 64,
                metadata[0][2],
                _ORIGIN,
            )
            validate_root_proof_preparation_v2_deployment(deployment)
            _verify_preparation_sources(conn, deployment)
            _qualify_preparation_roles(conn, deployment)
            for _, principal in deployment.component_roles:
                _principal_boundary(conn, config, principal, preparation_schema)
            _verify_component_connections(deployment, connections)
            deployment = replace(deployment, catalog_sha256=_fingerprint(conn, deployment))
        for component, connection in _connection_components(connections):
            with psycopg.connect(connection.dsn, autocommit=True) as conn, conn.transaction():
                qualify_root_proof_preparation_v2(conn, deployment, component=component)
        return deployment
    except RootProofPreparationV2QualificationError:
        raise
    except (
        psycopg.Error,
        AtomicRootProofQualificationError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
    ) as exc:
        raise RootProofPreparationV2QualificationError(
            "read-only preparation reconstruction refused"
        ) from exc


def migrate_root_proof_preparation_v2(
    bootstrap: PostgreSQLConnectionConfig,
    lifecycle_deployment: PostgreSQLCustodyLifecycleTestDeployment,
    *,
    preparation_schema: str,
    connections: PostgreSQLAtomicRootProofLegacyConnections,
) -> tuple[PostgreSQLRootProofPreparationV2Deployment, PostgreSQLCustodyLifecycleTestDeployment]:
    """Explicit ACL V2 installation after lifecycle V2 and signer enrollment V3.

    Only two read-only helper functions become available to existing component
    logins. The migration returns the explicitly updated atomic/lifecycle pin
    for the separate commit composition and a credential-free preparation pin.
    """
    validate_custody_lifecycle_deployment(lifecycle_deployment)
    if (
        type(bootstrap) is not PostgreSQLConnectionConfig
        or type(connections) is not PostgreSQLAtomicRootProofLegacyConnections
    ):
        raise TypeError("exact explicit administrative and component credentials required")
    config = lifecycle_deployment.atomic_deployment.provisioning
    if (
        type(preparation_schema) is not str
        or _IDENTIFIER.fullmatch(preparation_schema) is None
        or preparation_schema in _schemas(config)
    ):
        raise ValueError("separate safe preparation schema required")
    _postqualify(lifecycle_deployment)
    try:
        with psycopg.connect(bootstrap.dsn, autocommit=True) as conn, conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
            conn.execute("SET LOCAL search_path=pg_catalog")
            _bootstrap_principal(conn, config)
            if (
                _physical_identity(conn, config.instance_id)
                != lifecycle_deployment.atomic_deployment.identity
            ):
                raise RootProofPreparationV2QualificationError(
                    "migration physical authority mismatch"
                )
            conn.execute(
                sql.SQL("LOCK TABLE {}.decisions IN ACCESS EXCLUSIVE MODE").format(
                    sql.Identifier(config.schema)
                )
            )
            conn.execute(
                sql.SQL("LOCK TABLE {}.fences IN ACCESS EXCLUSIVE MODE").format(
                    sql.Identifier(config.schema)
                )
            )
            _verify_v2_functions(conn, config)
            _verify_database_manifest(conn, config)
            application_before = _application_bytes_fingerprint(conn, config)
            lifecycle_before = _lifecycle_records_fingerprint(conn, config)
            component_roles = _component_principals(conn, config)
            atomic = lifecycle_deployment.atomic_deployment
            deployment = PostgreSQLRootProofPreparationV2Deployment(
                config,
                atomic.identity,
                preparation_schema,
                atomic.role_oids,
                atomic.owner_roles,
                component_roles,
                "0" * 64,
                atomic.catalog_sha256,
                _ORIGIN,
            )
            validate_root_proof_preparation_v2_deployment(deployment)
            _verify_component_connections(deployment, connections)
            conn.execute(
                sql.SQL("CREATE SCHEMA {} AUTHORIZATION {}").format(
                    sql.Identifier(preparation_schema), sql.Identifier(config.schema_owner_role)
                )
            )
            conn.execute(
                sql.SQL("REVOKE ALL ON SCHEMA {} FROM PUBLIC").format(
                    sql.Identifier(preparation_schema)
                )
            )
            conn.execute(
                sql.SQL(
                    "CREATE TABLE {}.metadata(singleton boolean PRIMARY KEY DEFAULT true CHECK(singleton),"
                    "migration_id text NOT NULL,boundary text NOT NULL,predecessor_catalog_sha256 text NOT NULL)"
                ).format(sql.Identifier(preparation_schema))
            )
            conn.execute(
                sql.SQL("ALTER TABLE {}.metadata OWNER TO {}").format(
                    sql.Identifier(preparation_schema), sql.Identifier(config.schema_owner_role)
                )
            )
            conn.execute(
                sql.SQL("REVOKE ALL ON {}.metadata FROM PUBLIC").format(
                    sql.Identifier(preparation_schema)
                )
            )
            conn.execute(
                sql.SQL("INSERT INTO {}.metadata VALUES(true,%s,%s,%s)").format(
                    sql.Identifier(preparation_schema)
                ),
                (MIGRATION_ID, _BOUNDARY, atomic.catalog_sha256),
            )
            for name, event, level in (
                ("immutable_row", "UPDATE OR DELETE", "ROW"),
                ("immutable_truncate", "TRUNCATE", "STATEMENT"),
            ):
                conn.execute(
                    sql.SQL(
                        "CREATE TRIGGER {} BEFORE {} ON {}.metadata FOR EACH {} EXECUTE FUNCTION {}.reject_mutation()"
                    ).format(
                        sql.Identifier(name),
                        sql.SQL(event),
                        sql.Identifier(preparation_schema),
                        sql.SQL(level),
                        sql.Identifier(config.schema),
                    )
                )
                conn.execute(
                    sql.SQL("ALTER TABLE {}.metadata ENABLE ALWAYS TRIGGER {}").format(
                        sql.Identifier(preparation_schema), sql.Identifier(name)
                    )
                )
            for signature, (declaration, returns, source) in _sources(
                config, preparation_schema, deployment.component_roles
            ).items():
                conn.execute(
                    sql.SQL(
                        "CREATE FUNCTION {}.{}({}) RETURNS {} LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS {}"
                    ).format(
                        sql.Identifier(preparation_schema),
                        sql.Identifier(signature.split("(", 1)[0]),
                        sql.SQL(declaration),
                        sql.SQL(returns),
                        sql.Literal(source),
                    )
                )
                conn.execute(
                    sql.SQL("ALTER FUNCTION {}.{} OWNER TO {}").format(
                        sql.Identifier(preparation_schema),
                        sql.SQL(signature),
                        sql.Identifier(config.schema_owner_role),
                    )
                )
                conn.execute(
                    sql.SQL("REVOKE ALL ON FUNCTION {}.{} FROM PUBLIC").format(
                        sql.Identifier(preparation_schema), sql.SQL(signature)
                    )
                )
                for component, role in deployment.component_roles:
                    if signature == "qualify_preparation_v2()" or component in {
                        "intent",
                        "retention",
                    }:
                        conn.execute(
                            sql.SQL("GRANT EXECUTE ON FUNCTION {}.{} TO {}").format(
                                sql.Identifier(preparation_schema),
                                sql.SQL(signature),
                                sql.Identifier(role),
                            )
                        )
            for _, role in deployment.component_roles:
                conn.execute(
                    sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
                        sql.Identifier(preparation_schema), sql.Identifier(role)
                    )
                )
                _principal_boundary(conn, config, role, preparation_schema)
            _verify_preparation_sources(conn, deployment)
            # New external schema/function ACLs are explicitly reviewed here;
            # preserve the old manifest in the process-local predecessor pin.
            conn.execute(
                sql.SQL("ALTER TABLE {}.catalog_manifest DISABLE TRIGGER immutable_row").format(
                    sql.Identifier(config.schema)
                )
            )
            conn.execute("SAVEPOINT preparation_manifest_capture")
            conn.execute(
                sql.SQL(
                    "ALTER TABLE {}.catalog_manifest ENABLE ALWAYS TRIGGER immutable_row"
                ).format(sql.Identifier(config.schema))
            )
            snapshot = conn.execute(
                sql.SQL("SELECT {}.catalog_snapshot()").format(sql.Identifier(config.schema))
            ).fetchone()
            if snapshot is None:
                raise RootProofPreparationV2QualificationError("preparation manifest unavailable")
            conn.execute("ROLLBACK TO SAVEPOINT preparation_manifest_capture")
            conn.execute("RELEASE SAVEPOINT preparation_manifest_capture")
            conn.execute(
                sql.SQL("UPDATE {}.catalog_manifest SET snapshot=%s WHERE singleton").format(
                    sql.Identifier(config.schema)
                ),
                (Jsonb(snapshot[0]),),
            )
            conn.execute(
                sql.SQL(
                    "ALTER TABLE {}.catalog_manifest ENABLE ALWAYS TRIGGER immutable_row"
                ).format(sql.Identifier(config.schema))
            )
            _verify_database_manifest(conn, config)
            _verify_principal_boundary(conn, config)
            if (
                _application_bytes_fingerprint(conn, config) != application_before
                or _lifecycle_records_fingerprint(conn, config) != lifecycle_before
            ):
                raise RootProofPreparationV2QualificationError(
                    "preparation migration changed protected bytes"
                )
            catalog = _catalog_fingerprint(conn, config, atomic.role_oids)
            updated_atomic = replace(atomic, catalog_sha256=catalog)
            updated_lifecycle = replace(
                lifecycle_deployment, atomic_deployment=updated_atomic, catalog_sha256=catalog
            )
            deployment = replace(deployment, catalog_sha256=_fingerprint(conn, deployment))
        _postqualify(updated_lifecycle)
        for component, connection in zip(
            _COMPONENTS,
            (
                connections.entitlement_runtime,
                connections.intent_writer,
                connections.retention_writer,
                connections.history_runtime,
                connections.requester_runtime,
                connections.claimant_runtime,
            ),
            strict=True,
        ):
            with (
                psycopg.connect(connection.dsn, autocommit=True) as component_conn,
                component_conn.transaction(),
            ):
                qualify_root_proof_preparation_v2(component_conn, deployment, component=component)
        return deployment, updated_lifecycle
    except RootProofPreparationV2QualificationError:
        raise
    except (
        psycopg.Error,
        AtomicRootProofQualificationError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
    ) as exc:
        raise RootProofPreparationV2QualificationError(
            "explicit preparation ACL V2 migration refused"
        ) from exc


__all__ = [
    "MIGRATION_ID",
    "PostgreSQLRootProofPreparationV2Deployment",
    "RootProofPreparationV2QualificationError",
    "lock_root_proof_preparation_signer_v2",
    "migrate_root_proof_preparation_v2",
    "qualify_root_proof_preparation_v2",
    "requalify_root_proof_preparation_v2",
    "validate_root_proof_preparation_v2_deployment",
]
