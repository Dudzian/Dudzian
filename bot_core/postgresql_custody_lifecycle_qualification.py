"""Explicit migration and read-only qualification of custody lifecycle V2.

This is always an isolated, NOT_READY deployment. Reconstruction after process
restart requires the separate bootstrap authority, never a caller supplied pin.
The bootstrap credential is not retained in the administrative deployment.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict

from bot_core.postgresql_atomic_root_proof_migration import (
    _application_bytes_fingerprint,
    _component_roles,
    _verify_database_manifest,
    _verify_original_sources,
    _verify_principal_boundary,
)
from bot_core.postgresql_atomic_root_proof_qualification import (
    _ADMINISTRATIVE_ORIGIN,
    AtomicRootProofQualificationError,
    PostgreSQLAtomicRootProofPhysicalIdentity,
    PostgreSQLAtomicRootProofTestDeployment,
    _catalog_fingerprint,
    _physical_identity,
    _qualify_roles,
    qualify_atomic_root_proof_authority,
    validate_atomic_root_proof_deployment,
)
from bot_core.postgresql_atomic_root_proof_schema import (
    FUNC_ADMIN as V1_FUNC_ADMIN,
    FUNC_RUNTIME,
    PostgreSQLAtomicRootProofProvisioning,
    reviewed_function_inventory,
)
from bot_core.postgresql_custody_lifecycle_schema import (
    BOUNDARY,
    FUNC_ADMIN,
    MIGRATION_ID,
    SCHEMA_IDENTITY,
    SCHEMA_VERSION,
    SQL_CONTRACT_VERSION,
    install_custody_lifecycle_authority,
    install_custody_signer_enrollment_v3,
    reviewed_lifecycle_enrollment_function_sources,
    reviewed_lifecycle_function_sources,
)
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig

ATOMIC_ISSUER_READY = "NOT_READY"
PRODUCTION_LIFECYCLE_ISOLATION = "BLOCKED"
_LIFECYCLE_ADMINISTRATIVE_ORIGIN = object()
_CLOSED_LEGACY = frozenset(
    {"provision_fence(text,text,bigint)", "transition_fence(text,bigint,text,bigint)"}
)


class CustodyLifecycleQualificationError(AtomicRootProofQualificationError):
    """Live V2 authority, administrator or compiled catalog differs."""


@dataclass(frozen=True, slots=True)
class PostgreSQLCustodyLifecycleTestDeployment:
    """Trusted administrator credential; never pass to an issuer runtime."""

    atomic_deployment: PostgreSQLAtomicRootProofTestDeployment
    admin_connection: PostgreSQLConnectionConfig
    evidence_key: bytes = field(repr=False)
    catalog_sha256: str
    _origin: object = field(repr=False, compare=False, default=None)
    sql_contract_version: int = 2

    def __repr__(self) -> str:
        return (
            "PostgreSQLCustodyLifecycleTestDeployment("
            f"schema={self.atomic_deployment.provisioning.schema!r}, "
            f"sql_contract_version={self.sql_contract_version!r}, "
            "boundary='ISOLATED_TEST_ONLY', ATOMIC_ISSUER_READY='NOT_READY')"
        )


def validate_custody_lifecycle_deployment(
    deployment: object, *, allow_legacy_sql_contract: bool = False
) -> None:
    if (
        type(deployment) is not PostgreSQLCustodyLifecycleTestDeployment
        or deployment._origin is not _LIFECYCLE_ADMINISTRATIVE_ORIGIN
        or type(deployment.evidence_key) is not bytes
        or len(deployment.evidence_key) != 32
        or deployment.catalog_sha256 != deployment.atomic_deployment.catalog_sha256
        or type(deployment.sql_contract_version) is not int
        or deployment.sql_contract_version not in {2, SQL_CONTRACT_VERSION}
    ):
        raise CustodyLifecycleQualificationError(
            "explicit trusted V2 administrator deployment required"
        )
    validate_atomic_root_proof_deployment(deployment.atomic_deployment)
    if deployment.sql_contract_version != SQL_CONTRACT_VERSION and not allow_legacy_sql_contract:
        raise CustodyLifecycleQualificationError("explicit signer enrollment SQL upgrade required")


def _verify_v2_functions(
    conn: psycopg.Connection[Any],
    config: PostgreSQLAtomicRootProofProvisioning,
    *,
    sql_contract_version: int = SQL_CONTRACT_VERSION,
) -> None:
    """Review original V1 definitions and the exact versioned V2 ACL change."""
    inventory = reviewed_function_inventory(conn, config)
    sources = reviewed_lifecycle_function_sources(config)
    if sql_contract_version == SQL_CONTRACT_VERSION:
        sources.update(reviewed_lifecycle_enrollment_function_sources(config))
    for signature, source in sources.items():
        inventory[(config.schema, signature)] = (config.schema_owner_role, source, "", "")
    expected_functions = _verify_original_sources(conn, config)
    expected_functions[config.schema] = set()
    for (schema, signature), (owner, source, _declaration, _returns) in inventory.items():
        expected_functions.setdefault(schema, set()).add(signature)
        row = conn.execute(
            "SELECT p.proowner,r.rolname,p.prosrc,p.prosecdef,p.proconfig,l.lanname "
            "FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles r ON r.oid=p.proowner "
            "JOIN pg_catalog.pg_language l ON l.oid=p.prolang "
            "WHERE p.oid=pg_catalog.to_regprocedure(%s)",
            (schema + "." + signature,),
        ).fetchone()
        if row is None or row[1:] != (owner, source, True, ["search_path=pg_catalog"], "plpgsql"):
            raise CustodyLifecycleQualificationError(
                "V1/V2 function differs from reviewed static source"
            )
        expected = {(row[0], "EXECUTE", False)}
        callers: list[str] = []
        if schema != config.schema and not signature.endswith("constraint()"):
            callers.append(config.schema_owner_role)
        elif signature in FUNC_RUNTIME:
            callers.append(config.issuer_writer_role)
        elif signature in (V1_FUNC_ADMIN - _CLOSED_LEGACY) | FUNC_ADMIN:
            callers.append(config.test_admin_role)
        elif signature.startswith("assert_bound_link(") or signature.startswith(
            "assert_history_link("
        ):
            component = (
                config.entitlement_schema
                if signature.startswith("assert_bound_link(")
                else config.history_schema
            )
            owner_row = conn.execute(
                sql.SQL("SELECT schema_owner_role FROM {}.metadata WHERE singleton").format(
                    sql.Identifier(component)
                )
            ).fetchone()
            if owner_row is None:
                raise CustodyLifecycleQualificationError("component owner metadata missing")
            callers.append(owner_row[0])
        if schema == config.schema and signature == "qualify_catalog()":
            callers.append(config.test_admin_role)
        if schema == config.schema and signature == "qualify_lifecycle_enrollment_v3()":
            callers.extend((config.test_admin_role, config.issuer_writer_role))
        for caller in callers:
            role = conn.execute(
                "SELECT oid FROM pg_catalog.pg_roles WHERE rolname=%s", (caller,)
            ).fetchone()
            if role is None:
                raise CustodyLifecycleQualificationError("required authority role missing")
            expected.add((role[0], "EXECUTE", False))
        actual = set(
            conn.execute(
                "SELECT a.grantee,a.privilege_type,a.is_grantable FROM pg_catalog.pg_proc p "
                "CROSS JOIN LATERAL pg_catalog.aclexplode(p.proacl) a "
                "WHERE p.oid=pg_catalog.to_regprocedure(%s)",
                (schema + "." + signature,),
            ).fetchall()
        )
        if actual != expected:
            raise CustodyLifecycleQualificationError("exact versioned function ACL differs")
    for schema, signatures in expected_functions.items():
        rows = conn.execute(
            "SELECT p.proname || '(' || pg_catalog.replace(pg_catalog.oidvectortypes(p.proargtypes),', ', ',') || ')' "
            "FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace WHERE n.nspname=%s",
            (schema,),
        ).fetchall()
        if {row[0] for row in rows} != signatures:
            raise CustodyLifecycleQualificationError("exact V2 function inventory differs")


def _read_secret(
    conn: psycopg.Connection[Any], config: PostgreSQLAtomicRootProofProvisioning
) -> bytes:
    rows = conn.execute(
        sql.SQL(
            "SELECT authentication_key FROM {}.lifecycle_authority_secret WHERE singleton"
        ).format(sql.Identifier(config.schema))
    ).fetchall()
    if len(rows) != 1 or type(rows[0][0]) is not bytes or len(rows[0][0]) != 32:
        raise CustodyLifecycleQualificationError(
            "V2 administrator authentication credential unavailable"
        )
    key = rows[0][0]
    metadata = conn.execute(
        sql.SQL(
            "SELECT schema_identity,schema_version,migration_id,authentication_key_sha256,boundary FROM {}.lifecycle_metadata WHERE singleton"
        ).format(sql.Identifier(config.schema))
    ).fetchall()
    if metadata != [
        (SCHEMA_IDENTITY, SCHEMA_VERSION, MIGRATION_ID, hashlib.sha256(key).digest(), BOUNDARY)
    ]:
        raise CustodyLifecycleQualificationError(
            "exact V2 migration identity or secret pin differs"
        )
    return key


def _bootstrap_principal(
    conn: psycopg.Connection[Any], config: PostgreSQLAtomicRootProofProvisioning
) -> None:
    principal = conn.execute(
        "SELECT session_user::text,current_user::text,(SELECT rolsuper FROM pg_catalog.pg_roles WHERE rolname=session_user::text)"
    ).fetchone()
    if (
        principal is None
        or principal[0] != principal[1]
        or not principal[2]
        or principal[0]
        in {config.schema_owner_role, config.issuer_writer_role, config.test_admin_role}
    ):
        raise CustodyLifecycleQualificationError("separate direct bootstrap administrator required")


def qualify_custody_lifecycle_authority(
    conn: psycopg.Connection[Any],
    deployment: PostgreSQLCustodyLifecycleTestDeployment,
    *,
    expected_role: str = "admin",
    allow_legacy_sql_contract: bool = False,
) -> None:
    """Fresh authenticated login proves exact physical service, OIDs and catalog."""
    validate_custody_lifecycle_deployment(
        deployment, allow_legacy_sql_contract=allow_legacy_sql_contract
    )
    if expected_role not in {"admin", "issuer_writer"}:
        raise CustodyLifecycleQualificationError("explicit V2 semantic principal required")
    try:
        qualify_atomic_root_proof_authority(
            conn,
            deployment.atomic_deployment,
            expected_role="test_admin" if expected_role == "admin" else "issuer_writer",
        )
        config = deployment.atomic_deployment.provisioning
        roles = dict(deployment.atomic_deployment.role_oids)
        metadata = conn.execute(
            sql.SQL(
                "SELECT schema_identity,schema_version,migration_id,admin_role_oid,owner_role_oid,"
                "authentication_key_sha256,boundary FROM {}.lifecycle_metadata WHERE singleton"
            ).format(sql.Identifier(config.schema))
        ).fetchall()
        if metadata != [
            (
                SCHEMA_IDENTITY,
                SCHEMA_VERSION,
                MIGRATION_ID,
                roles[config.test_admin_role],
                roles[config.schema_owner_role],
                hashlib.sha256(deployment.evidence_key).digest(),
                BOUNDARY,
            )
        ]:
            raise CustodyLifecycleQualificationError("exact live V2 metadata pin differs")
        if expected_role == "admin" and conn.execute(
            sql.SQL("SELECT {}.qualify_lifecycle_catalog()").format(
                sql.Identifier(deployment.atomic_deployment.provisioning.schema)
            )
        ).fetchone() != (True,):
            raise CustodyLifecycleQualificationError("V2 immutable manifest refused")
        if deployment.sql_contract_version == SQL_CONTRACT_VERSION and conn.execute(
            sql.SQL("SELECT {}.qualify_lifecycle_enrollment_v3()").format(
                sql.Identifier(config.schema)
            )
        ).fetchone() != (True,):
            raise CustodyLifecycleQualificationError("signer enrollment SQL contract refused")
    except CustodyLifecycleQualificationError:
        raise
    except (psycopg.Error, AtomicRootProofQualificationError, ValueError, TypeError) as exc:
        raise CustodyLifecycleQualificationError(
            "live V2 lifecycle qualification unavailable"
        ) from exc


def _postqualify(
    deployment: PostgreSQLCustodyLifecycleTestDeployment, *, allow_legacy_sql_contract: bool = False
) -> None:
    for connection, role in (
        (deployment.admin_connection, "admin"),
        (deployment.atomic_deployment.writer_connection, "issuer_writer"),
    ):
        with psycopg.connect(connection.dsn, autocommit=True) as conn, conn.transaction():
            qualify_custody_lifecycle_authority(
                conn,
                deployment,
                expected_role=role,
                allow_legacy_sql_contract=allow_legacy_sql_contract,
            )


def migrate_custody_lifecycle_v2(
    bootstrap: PostgreSQLConnectionConfig,
    atomic_deployment: PostgreSQLAtomicRootProofTestDeployment,
    admin_connection: PostgreSQLConnectionConfig,
    credential_bindings: tuple[dict[str, Any], ...],
    *,
    sql_contract_version: int = SQL_CONTRACT_VERSION,
) -> PostgreSQLCustodyLifecycleTestDeployment:
    """Explicit V1→V2 migration; preserves all application bytes and V1 sources."""
    validate_atomic_root_proof_deployment(atomic_deployment)
    if type(sql_contract_version) is not int or sql_contract_version not in {
        2,
        SQL_CONTRACT_VERSION,
    }:
        raise ValueError("explicit reviewed signer SQL contract version required")
    config = atomic_deployment.provisioning
    if (
        type(bootstrap) is not PostgreSQLConnectionConfig
        or type(admin_connection) is not PostgreSQLConnectionConfig
    ):
        raise TypeError("separate exact administrator credentials required")
    if conninfo_to_dict(admin_connection.dsn).get("user") != config.test_admin_role:
        raise CustodyLifecycleQualificationError("exact direct lifecycle admin credential required")
    if (
        type(credential_bindings) is not tuple
        or not credential_bindings
        or any(type(binding) is not dict for binding in credential_bindings)
    ):
        raise TypeError("exact genuine custody enrollment bindings required")
    expected_database_identity = asdict(atomic_deployment.identity)
    expected_database_identity["system_identifier"] = str(
        atomic_deployment.identity.system_identifier
    )
    if any(
        binding.get("database_identity") != expected_database_identity
        for binding in credential_bindings
    ):
        raise CustodyLifecycleQualificationError("enrollment belongs to another physical authority")
    for connection, role in (
        (atomic_deployment.writer_connection, "issuer_writer"),
        (admin_connection, "test_admin"),
    ):
        with psycopg.connect(connection.dsn, autocommit=True) as conn, conn.transaction():
            qualify_atomic_root_proof_authority(conn, atomic_deployment, expected_role=role)
    try:
        with psycopg.connect(bootstrap.dsn, autocommit=True) as conn, conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
            conn.execute("SET LOCAL search_path=pg_catalog")
            _bootstrap_principal(conn, config)
            if _physical_identity(conn, config.instance_id) != atomic_deployment.identity:
                raise CustodyLifecycleQualificationError("migration physical service differs")
            if sql_contract_version == SQL_CONTRACT_VERSION:
                # The issuer reads decisions before locking fences. Acquire
                # this first, before the preserved V2 installer locks fences.
                conn.execute(
                    sql.SQL("LOCK TABLE {}.decisions IN ACCESS EXCLUSIVE MODE").format(
                        sql.Identifier(config.schema)
                    )
                )
            before = _application_bytes_fingerprint(conn, config)
            key = secrets.token_bytes(32)
            install_custody_lifecycle_authority(conn, config, credential_bindings, key)
            _verify_v2_functions(conn, config, sql_contract_version=2)
            _verify_database_manifest(conn, config)
            if sql_contract_version == SQL_CONTRACT_VERSION:
                predecessor = _catalog_fingerprint(conn, config, atomic_deployment.role_oids)
                install_custody_signer_enrollment_v3(conn, config, predecessor)
                _verify_v2_functions(conn, config)
                _verify_database_manifest(conn, config)
            _verify_principal_boundary(conn, config)
            if _application_bytes_fingerprint(conn, config) != before:
                raise CustodyLifecycleQualificationError(
                    "V2 migration changed retained application bytes"
                )
            catalog = _catalog_fingerprint(conn, config, atomic_deployment.role_oids)
            atomic = replace(atomic_deployment, catalog_sha256=catalog)
            _qualify_roles(conn, atomic)
            _read_secret(conn, config)
            deployment = PostgreSQLCustodyLifecycleTestDeployment(
                atomic,
                admin_connection,
                key,
                catalog,
                _LIFECYCLE_ADMINISTRATIVE_ORIGIN,
                sql_contract_version,
            )
        _postqualify(deployment, allow_legacy_sql_contract=sql_contract_version == 2)
        return deployment
    except CustodyLifecycleQualificationError:
        raise
    except (
        psycopg.Error,
        AtomicRootProofQualificationError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
    ) as exc:
        raise CustodyLifecycleQualificationError("explicit V2 migration refused") from exc


def _lifecycle_records_fingerprint(
    conn: psycopg.Connection[Any],
    config: PostgreSQLAtomicRootProofProvisioning,
) -> str:
    """Conserve protected records and authentication key without exposing them."""
    facts = []
    for table in (
        "metadata",
        "fences",
        "approved_commits",
        "decisions",
        "lifecycle_metadata",
        "lifecycle_authority_secret",
        "lifecycle_credentials",
        "lifecycle_operations",
        "lifecycle_completions",
    ):
        rows = conn.execute(
            sql.SQL("SELECT pg_catalog.to_jsonb(t)::text FROM {}.{} t ORDER BY 1").format(
                sql.Identifier(config.schema), sql.Identifier(table)
            )
        ).fetchall()
        facts.append((table, rows))
    return hashlib.sha256(json.dumps(facts, separators=(",", ":")).encode()).hexdigest()


def upgrade_custody_lifecycle_signer_enrollment_v3(
    bootstrap: PostgreSQLConnectionConfig,
    deployment: PostgreSQLCustodyLifecycleTestDeployment,
) -> PostgreSQLCustodyLifecycleTestDeployment:
    """Explicit V2→SQL-contract-3 upgrade, preserving every old authority row.

    Historical V2 reconstruction is accepted solely for this reviewed upgrade.
    Its old immutable catalog must qualify before any DDL occurs; unknown or
    modified catalogs are refused rather than repaired.
    """
    validate_custody_lifecycle_deployment(deployment, allow_legacy_sql_contract=True)
    if type(bootstrap) is not PostgreSQLConnectionConfig:
        raise TypeError("exact separate bootstrap credential required")
    if deployment.sql_contract_version == SQL_CONTRACT_VERSION:
        _postqualify(deployment)
        return deployment
    _postqualify(deployment, allow_legacy_sql_contract=True)
    atomic = deployment.atomic_deployment
    config = atomic.provisioning
    try:
        with psycopg.connect(bootstrap.dsn, autocommit=True) as conn, conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
            conn.execute("SET LOCAL search_path=pg_catalog")
            _bootstrap_principal(conn, config)
            if _physical_identity(conn, config.instance_id) != atomic.identity:
                raise CustodyLifecycleQualificationError("upgrade physical authority differs")
            # Respect issuer read(decisions)→lock(fences), including its table
            # locks, before taking conservation snapshots or installing DDL.
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
            _verify_v2_functions(conn, config, sql_contract_version=2)
            _verify_database_manifest(conn, config)
            if _catalog_fingerprint(conn, config, atomic.role_oids) != deployment.catalog_sha256:
                raise CustodyLifecycleQualificationError("exact predecessor V2 catalog changed")
            records_before = _lifecycle_records_fingerprint(conn, config)
            application_before = _application_bytes_fingerprint(conn, config)
            predecessor = conn.execute(
                sql.SQL("SELECT snapshot FROM {}.catalog_manifest WHERE singleton").format(
                    sql.Identifier(config.schema)
                )
            ).fetchone()
            if predecessor is None:
                raise CustodyLifecycleQualificationError("predecessor manifest unavailable")
            install_custody_signer_enrollment_v3(conn, config, deployment.catalog_sha256)
            _verify_v2_functions(conn, config)
            _verify_database_manifest(conn, config)
            _verify_principal_boundary(conn, config)
            archive = conn.execute(
                sql.SQL(
                    "SELECT predecessor_snapshot,predecessor_catalog_sha256 FROM {}.lifecycle_sql_contract WHERE singleton"
                ).format(sql.Identifier(config.schema))
            ).fetchall()
            if archive != [(predecessor[0], deployment.catalog_sha256)]:
                raise CustodyLifecycleQualificationError("original V2 manifest was not conserved")
            if (
                _lifecycle_records_fingerprint(conn, config) != records_before
                or _application_bytes_fingerprint(conn, config) != application_before
                or _read_secret(conn, config) != deployment.evidence_key
            ):
                raise CustodyLifecycleQualificationError(
                    "signer enrollment upgrade changed protected records"
                )
            catalog = _catalog_fingerprint(conn, config, atomic.role_oids)
            revised_atomic = replace(atomic, catalog_sha256=catalog)
            _qualify_roles(conn, revised_atomic)
            revised = replace(
                deployment,
                atomic_deployment=revised_atomic,
                catalog_sha256=catalog,
                sql_contract_version=SQL_CONTRACT_VERSION,
            )
        _postqualify(revised)
        return revised
    except CustodyLifecycleQualificationError:
        raise
    except (
        psycopg.Error,
        AtomicRootProofQualificationError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
    ) as exc:
        raise CustodyLifecycleQualificationError(
            "explicit signer enrollment upgrade refused"
        ) from exc


def requalify_existing_custody_lifecycle_v2(
    bootstrap: PostgreSQLConnectionConfig,
    atomic_config: PostgreSQLAtomicRootProofProvisioning,
    writer_connection: PostgreSQLConnectionConfig,
    admin_connection: PostgreSQLConnectionConfig,
    *,
    expected_identity: PostgreSQLAtomicRootProofPhysicalIdentity,
    expected_sql_contract_version: int = SQL_CONTRACT_VERSION,
) -> PostgreSQLCustodyLifecycleTestDeployment:
    """Read-only reconstruction of authentic V2 after administrator restart."""
    if type(expected_sql_contract_version) is not int or expected_sql_contract_version not in {
        2,
        SQL_CONTRACT_VERSION,
    }:
        raise ValueError("explicit reviewed reconstruction SQL contract version required")
    if (
        type(atomic_config) is not PostgreSQLAtomicRootProofProvisioning
        or type(expected_identity) is not PostgreSQLAtomicRootProofPhysicalIdentity
    ):
        raise TypeError("exact V2 reconstruction configuration required")
    if any(
        type(value) is not PostgreSQLConnectionConfig
        for value in (bootstrap, writer_connection, admin_connection)
    ):
        raise TypeError("separate exact reconstruction credentials required")
    if (
        conninfo_to_dict(writer_connection.dsn).get("user") != atomic_config.issuer_writer_role
        or conninfo_to_dict(admin_connection.dsn).get("user") != atomic_config.test_admin_role
    ):
        raise CustodyLifecycleQualificationError("exact direct authority credentials required")
    try:
        with psycopg.connect(bootstrap.dsn, autocommit=True) as conn, conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE READ ONLY")
            conn.execute("SET LOCAL search_path=pg_catalog")
            _bootstrap_principal(conn, atomic_config)
            if _physical_identity(conn, atomic_config.instance_id) != expected_identity:
                raise CustodyLifecycleQualificationError("reconstruction physical service differs")
            _verify_v2_functions(
                conn, atomic_config, sql_contract_version=expected_sql_contract_version
            )
            _verify_database_manifest(conn, atomic_config)
            _verify_principal_boundary(conn, atomic_config)
            key = _read_secret(conn, atomic_config)
            roles, owners = _component_roles(conn, atomic_config)
            catalog = _catalog_fingerprint(conn, atomic_config, roles)
            atomic = PostgreSQLAtomicRootProofTestDeployment(
                writer_connection,
                atomic_config,
                expected_identity,
                catalog,
                roles,
                owners,
                _ADMINISTRATIVE_ORIGIN,
            )
            _qualify_roles(conn, atomic)
            deployment = PostgreSQLCustodyLifecycleTestDeployment(
                atomic,
                admin_connection,
                key,
                catalog,
                _LIFECYCLE_ADMINISTRATIVE_ORIGIN,
                expected_sql_contract_version,
            )
        _postqualify(deployment, allow_legacy_sql_contract=expected_sql_contract_version == 2)
        return deployment
    except CustodyLifecycleQualificationError:
        raise
    except (
        psycopg.Error,
        AtomicRootProofQualificationError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
    ) as exc:
        raise CustodyLifecycleQualificationError("existing V2 reconstruction refused") from exc


__all__ = [
    "ATOMIC_ISSUER_READY",
    "PRODUCTION_LIFECYCLE_ISOLATION",
    "CustodyLifecycleQualificationError",
    "PostgreSQLCustodyLifecycleTestDeployment",
    "validate_custody_lifecycle_deployment",
    "qualify_custody_lifecycle_authority",
    "migrate_custody_lifecycle_v2",
    "requalify_existing_custody_lifecycle_v2",
    "upgrade_custody_lifecycle_signer_enrollment_v3",
]
