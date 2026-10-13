"""Exact, read-only qualification of the isolated atomic-commit V1 migration.

This contract deliberately does not replace any legacy schema qualifier.  A
trusted, explicit administrative migration pins the complete new catalog.  The
writer must prove that same catalog, physical database and direct principal on
each connection.  This is an internal test deployment, never a production gate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import psycopg
from psycopg import sql

from bot_core.postgresql_atomic_root_proof_schema import (
    MIGRATION_ID,
    SCHEMA_IDENTITY,
    SCHEMA_VERSION,
    PostgreSQLAtomicRootProofProvisioning,
)
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig


class AtomicRootProofQualificationError(RuntimeError):
    """Live authority differs from the reviewed, administratively pinned V1."""


_ADMINISTRATIVE_ORIGIN = object()


@dataclass(frozen=True, slots=True)
class PostgreSQLAtomicRootProofPhysicalIdentity:
    """Restart-stable database and service coordinates, not DSN equality."""

    system_identifier: int
    database_oid: int
    database_name: str
    cluster_name: str
    connection_host: str
    connection_port: int
    server_address: str | None
    server_port: int | None
    instance_id: str

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value <= 0
            for value in (
                self.system_identifier,
                self.database_oid,
            )
        ):
            raise ValueError("positive exact cluster/database identity required")
        if (
            any(
                type(value) is not str or not value
                for value in (
                    self.database_name,
                    self.connection_host,
                    self.instance_id,
                )
            )
            or type(self.cluster_name) is not str
        ):
            raise ValueError("exact physical database/service coordinates required")
        if type(self.connection_port) is not int or not 1 <= self.connection_port <= 65535:
            raise ValueError("exact PostgreSQL service port required")
        if self.server_address is not None and type(self.server_address) is not str:
            raise ValueError("exact PostgreSQL server address required")
        if self.server_port is not None and (
            type(self.server_port) is not int or not 1 <= self.server_port <= 65535
        ):
            raise ValueError("exact PostgreSQL server port required")


@dataclass(frozen=True, slots=True)
class PostgreSQLAtomicRootProofTestDeployment:
    """Administrative test-only pin; contains no bootstrap/admin password."""

    writer_connection: PostgreSQLConnectionConfig
    provisioning: PostgreSQLAtomicRootProofProvisioning
    identity: PostgreSQLAtomicRootProofPhysicalIdentity
    catalog_sha256: str
    role_oids: tuple[tuple[str, int], ...]
    owner_roles: tuple[str, ...]
    _origin: object = field(repr=False, compare=False, default=None)

    def __repr__(self) -> str:
        return (
            "PostgreSQLAtomicRootProofTestDeployment("
            f"schema={self.provisioning.schema!r}, boundary='ISOLATED_TEST_ONLY')"
        )


def _schemas(config: PostgreSQLAtomicRootProofProvisioning) -> list[str]:
    return [
        config.schema,
        config.entitlement_schema,
        config.intent_schema,
        config.retention_schema,
        config.history_schema,
        config.requester_schema,
        config.claimant_schema,
    ]


def _physical_identity(
    conn: psycopg.Connection[Any], instance_id: str
) -> PostgreSQLAtomicRootProofPhysicalIdentity:
    row = conn.execute(
        "SELECT c.system_identifier,d.oid,d.datname,"
        "pg_catalog.current_setting('cluster_name'),"
        "pg_catalog.inet_server_addr()::text,pg_catalog.inet_server_port() "
        "FROM pg_catalog.pg_control_system() c,pg_catalog.pg_database d "
        "WHERE d.datname=pg_catalog.current_database()"
    ).fetchone()
    if row is None or len(row) != 6:
        raise AtomicRootProofQualificationError("physical database identity unavailable")
    return PostgreSQLAtomicRootProofPhysicalIdentity(
        row[0],
        row[1],
        row[2],
        row[3],
        conn.info.host,
        conn.info.port,
        row[4],
        row[5],
        instance_id,
    )


def read_atomic_root_proof_physical_identity(
    conn: psycopg.Connection[Any], instance_id: str
) -> PostgreSQLAtomicRootProofPhysicalIdentity:
    """Read real restart-stable coordinates; missing catalog access refuses."""
    try:
        return _physical_identity(conn, instance_id)
    except (psycopg.Error, ValueError, TypeError, IndexError) as exc:
        raise AtomicRootProofQualificationError("physical authority identity unavailable") from exc


def _catalog_fingerprint(
    conn: psycopg.Connection[Any],
    config: PostgreSQLAtomicRootProofProvisioning,
    role_oids: tuple[tuple[str, int], ...],
) -> str:
    """Pin definitions, ownership, OIDs, all ACLs and auxiliary behavior.

    Catalog rows are read through pg_catalog, so no broad source-table grants
    are needed for the coordinator writer.  Mutable application records are
    excluded: committing a decision must not invalidate its qualification.
    """
    schemas = _schemas(config)
    oids = [oid for _, oid in role_oids]
    queries: dict[str, tuple[str, tuple[object, ...]]] = {
        "roles": (
            "SELECT oid,rolname,rolcanlogin,rolinherit,rolsuper,rolcreaterole,"
            "rolcreatedb,rolreplication,rolbypassrls,rolconnlimit,rolconfig "
            "FROM pg_catalog.pg_roles WHERE oid=ANY(%s) ORDER BY oid",
            (oids,),
        ),
        "memberships": (
            "SELECT roleid,member,grantor,admin_option,inherit_option,set_option "
            "FROM pg_catalog.pg_auth_members WHERE roleid=ANY(%s) OR member=ANY(%s) "
            "ORDER BY roleid,member,grantor",
            (oids, oids),
        ),
        "schemas": (
            "SELECT oid,nspname,nspowner,nspacl::text FROM pg_catalog.pg_namespace "
            "WHERE nspname=ANY(%s) ORDER BY nspname",
            (schemas,),
        ),
        "relations": (
            "SELECT c.oid,n.nspname,c.relname,c.relkind,c.relpersistence,c.relowner,"
            "c.relacl::text,c.relispartition,c.relrowsecurity,c.relforcerowsecurity,"
            "c.relreplident,c.reloptions,c.reltablespace,c.relam "
            "FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n "
            "ON n.oid=c.relnamespace WHERE n.nspname=ANY(%s) ORDER BY n.nspname,c.relname",
            (schemas,),
        ),
        "columns": (
            "SELECT n.nspname,c.relname,a.attnum,a.attname,a.atttypid,a.atttypmod,"
            "a.attnotnull,a.attidentity,a.attgenerated,a.attisdropped,a.attcollation,"
            "a.attacl::text,pg_catalog.pg_get_expr(d.adbin,d.adrelid) "
            "FROM pg_catalog.pg_attribute a JOIN pg_catalog.pg_class c ON c.oid=a.attrelid "
            "JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
            "LEFT JOIN pg_catalog.pg_attrdef d ON d.adrelid=a.attrelid AND d.adnum=a.attnum "
            "WHERE n.nspname=ANY(%s) AND a.attnum>0 ORDER BY n.nspname,c.relname,a.attnum",
            (schemas,),
        ),
        "constraints": (
            "SELECT n.nspname,c.conname,c.contype,c.conrelid,c.confrelid,c.conindid,"
            "c.conkey,c.confkey,c.condeferrable,c.condeferred,c.convalidated,"
            "pg_catalog.pg_get_constraintdef(c.oid,true) FROM pg_catalog.pg_constraint c "
            "JOIN pg_catalog.pg_namespace n ON n.oid=c.connamespace "
            "WHERE n.nspname=ANY(%s) ORDER BY n.nspname,c.conrelid,c.conname",
            (schemas,),
        ),
        "indexes": (
            "SELECT n.nspname,c.relname,i.indrelid,i.indisunique,i.indisprimary,"
            "i.indisvalid,i.indisready,i.indislive,i.indkey::text,i.indclass::text,"
            "pg_catalog.pg_get_indexdef(c.oid) FROM pg_catalog.pg_index i "
            "JOIN pg_catalog.pg_class c ON c.oid=i.indexrelid "
            "JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=ANY(%s) ORDER BY n.nspname,c.relname",
            (schemas,),
        ),
        "functions": (
            "SELECT n.nspname,p.oid,p.proname,p.proowner,p.prolang,p.prosecdef,"
            "p.proconfig,p.prosrc,p.probin,p.proacl::text,p.provolatile,p.proisstrict,"
            "p.proleakproof,p.proparallel,p.prokind,p.prorettype,p.proretset,"
            "p.proargtypes::text,p.proallargtypes,p.proargmodes,p.proargnames,"
            "p.provariadic,p.pronargdefaults,p.prosupport,pg_catalog.pg_get_functiondef(p.oid) "
            "FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace "
            "WHERE n.nspname=ANY(%s) ORDER BY n.nspname,p.oid",
            (schemas,),
        ),
        "triggers": (
            "SELECT n.nspname,c.relname,t.tgname,t.tgfoid,t.tgtype,t.tgenabled,"
            "t.tgisinternal,t.tgconstraint,t.tgdeferrable,t.tginitdeferred,"
            "pg_catalog.pg_get_triggerdef(t.oid,true) "
            "FROM pg_catalog.pg_trigger t JOIN pg_catalog.pg_class c ON c.oid=t.tgrelid "
            "JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=ANY(%s) ORDER BY n.nspname,c.relname,t.tgname",
            (schemas,),
        ),
        "rules": (
            "SELECT n.nspname,c.relname,r.rulename,pg_catalog.pg_get_ruledef(r.oid,true) "
            "FROM pg_catalog.pg_rewrite r JOIN pg_catalog.pg_class c ON c.oid=r.ev_class "
            "JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=ANY(%s) ORDER BY n.nspname,c.relname,r.rulename",
            (schemas,),
        ),
        "policies": (
            "SELECT n.nspname,c.relname,p.polname,p.polcmd,p.polpermissive,p.polroles,"
            "pg_catalog.pg_get_expr(p.polqual,p.polrelid),"
            "pg_catalog.pg_get_expr(p.polwithcheck,p.polrelid) "
            "FROM pg_catalog.pg_policy p JOIN pg_catalog.pg_class c ON c.oid=p.polrelid "
            "JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=ANY(%s) ORDER BY n.nspname,c.relname,p.polname",
            (schemas,),
        ),
        "defaults": (
            "SELECT defaclrole,defaclnamespace,defaclobjtype,defaclacl::text "
            "FROM pg_catalog.pg_default_acl WHERE defaclrole=ANY(%s) OR "
            "defaclnamespace IN (SELECT oid FROM pg_catalog.pg_namespace "
            "WHERE nspname=ANY(%s)) ORDER BY defaclrole,defaclnamespace,defaclobjtype",
            (oids, schemas),
        ),
        "schema_external_acls": (
            "SELECT n.oid,n.nspname,n.nspowner,a.grantor,a.grantee,a.privilege_type,"
            "a.is_grantable FROM pg_catalog.pg_namespace n "
            "CROSS JOIN LATERAL pg_catalog.aclexplode(n.nspacl) a "
            "WHERE a.grantee=ANY(%s) ORDER BY n.oid,a.grantee,a.privilege_type",
            (oids,),
        ),
        "relation_external_acls": (
            "SELECT c.oid,c.relowner,a.grantor,a.grantee,a.privilege_type,a.is_grantable "
            "FROM pg_catalog.pg_class c CROSS JOIN LATERAL pg_catalog.aclexplode(c.relacl) a "
            "WHERE a.grantee=ANY(%s) ORDER BY c.oid,a.grantee,a.privilege_type",
            (oids,),
        ),
        "column_external_acls": (
            "SELECT c.oid,a.attnum,g.grantor,g.grantee,g.privilege_type,g.is_grantable "
            "FROM pg_catalog.pg_attribute a JOIN pg_catalog.pg_class c ON c.oid=a.attrelid "
            "CROSS JOIN LATERAL pg_catalog.aclexplode(a.attacl) g "
            "WHERE g.grantee=ANY(%s) OR g.grantee=0 "
            "ORDER BY c.oid,a.attnum,g.grantee,g.privilege_type",
            (oids,),
        ),
        "function_external_acls": (
            "SELECT p.oid,p.proowner,a.grantor,a.grantee,a.privilege_type,a.is_grantable "
            "FROM pg_catalog.pg_proc p CROSS JOIN LATERAL pg_catalog.aclexplode(p.proacl) a "
            "WHERE a.grantee=ANY(%s) ORDER BY p.oid,a.grantee,a.privilege_type",
            (oids,),
        ),
        "database_acls": (
            "SELECT oid,datname,datdba,datacl::text,datallowconn FROM pg_catalog.pg_database "
            "ORDER BY oid",
            (),
        ),
        "role_settings": (
            "SELECT setdatabase,setrole,setconfig FROM pg_catalog.pg_db_role_setting "
            "WHERE setrole=ANY(%s) ORDER BY setdatabase,setrole",
            (oids,),
        ),
        "public_external_definers": (
            "SELECT n.nspname,p.oid,p.proowner,p.prosrc,p.proconfig,p.proacl::text "
            "FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace "
            "WHERE p.prosecdef AND n.nspname<>ALL(%s) AND "
            "pg_catalog.has_schema_privilege(%s,n.oid,'USAGE') AND "
            "pg_catalog.has_function_privilege(%s,p.oid,'EXECUTE') ORDER BY p.oid",
            (schemas, config.issuer_writer_role, config.issuer_writer_role),
        ),
    }
    facts = {
        name: conn.execute(query, params).fetchall() for name, (query, params) in queries.items()
    }
    for catalog, acl in (
        ("pg_type", "typacl"),
        ("pg_tablespace", "spcacl"),
        ("pg_language", "lanacl"),
        ("pg_foreign_server", "srvacl"),
        ("pg_foreign_data_wrapper", "fdwacl"),
        ("pg_largeobject_metadata", "lomacl"),
        ("pg_parameter_acl", "paracl"),
    ):
        facts[catalog + "_service_acl"] = conn.execute(
            sql.SQL(
                "SELECT o.oid,a.grantor,a.grantee,a.privilege_type,a.is_grantable "
                "FROM pg_catalog.{} o CROSS JOIN LATERAL pg_catalog.aclexplode(o.{}) a "
                "WHERE a.grantee=ANY(%s) OR a.grantee=0 "
                "ORDER BY o.oid,a.grantee,a.privilege_type"
            ).format(sql.Identifier(catalog), sql.Identifier(acl)),
            (oids,),
        ).fetchall()
    encoded = json.dumps(facts, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(b"CryptoHunter/AtomicRootProofCatalog/V1\0" + encoded).hexdigest()


def _qualify_roles(
    conn: psycopg.Connection[Any], deployment: PostgreSQLAtomicRootProofTestDeployment
) -> None:
    oids = [oid for _, oid in deployment.role_oids]
    if len(oids) != len(set(oids)):
        raise AtomicRootProofQualificationError("authority role identities alias")
    for name, oid in deployment.role_oids:
        role = conn.execute(
            "SELECT oid,rolcanlogin,rolinherit,rolsuper,rolcreaterole,rolcreatedb,"
            "rolreplication,rolbypassrls FROM pg_catalog.pg_roles WHERE rolname=%s",
            (name,),
        ).fetchone()
        if role != (
            oid,
            name not in deployment.owner_roles,
            False,
            False,
            False,
            False,
            False,
            False,
        ):
            raise AtomicRootProofQualificationError("authority role OID or attributes drift")
    if (
        conn.execute(
            "SELECT 1 FROM pg_catalog.pg_auth_members WHERE roleid=ANY(%s) OR member=ANY(%s) LIMIT 1",
            (oids, oids),
        ).fetchone()
        is not None
    ):
        raise AtomicRootProofQualificationError("authority role membership forbidden")


def _qualify_closed_principal(conn: psycopg.Connection[Any], principal: str) -> None:
    """Neither login owns objects or acquires hidden column/service authority."""
    oid = conn.execute(
        "SELECT oid FROM pg_catalog.pg_roles WHERE rolname=%s", (principal,)
    ).fetchone()
    if oid is None:
        raise AtomicRootProofQualificationError("authority principal missing")
    principal_oid = oid[0]
    if (
        conn.execute(
            "SELECT 1 FROM pg_catalog.pg_class WHERE relkind IN ('r','p','v','m','f') AND "
            "pg_catalog.has_any_column_privilege(%s,oid,'INSERT,UPDATE,REFERENCES') LIMIT 1",
            (principal,),
        ).fetchone()
        is not None
    ):
        raise AtomicRootProofQualificationError("authority direct column DML forbidden")
    for catalog, acl in (
        ("pg_type", "typacl"),
        ("pg_tablespace", "spcacl"),
        ("pg_language", "lanacl"),
        ("pg_foreign_server", "srvacl"),
        ("pg_foreign_data_wrapper", "fdwacl"),
        ("pg_largeobject_metadata", "lomacl"),
        ("pg_parameter_acl", "paracl"),
    ):
        if (
            conn.execute(
                sql.SQL(
                    "SELECT 1 FROM pg_catalog.{} o CROSS JOIN LATERAL pg_catalog.aclexplode(o.{}) a "
                    "WHERE a.grantee=%s LIMIT 1"
                ).format(sql.Identifier(catalog), sql.Identifier(acl)),
                (principal_oid,),
            ).fetchone()
            is not None
        ):
            raise AtomicRootProofQualificationError("authority external service grant forbidden")
    for query in (
        "SELECT 1 FROM pg_catalog.pg_default_acl d CROSS JOIN LATERAL "
        "pg_catalog.aclexplode(d.defaclacl) a WHERE a.grantee=%s LIMIT 1",
        "SELECT 1 FROM pg_catalog.pg_database d CROSS JOIN LATERAL "
        "pg_catalog.aclexplode(d.datacl) a WHERE a.grantee=%s AND "
        "a.privilege_type<>'CONNECT' LIMIT 1",
    ):
        if conn.execute(query, (principal_oid,)).fetchone() is not None:
            raise AtomicRootProofQualificationError("authority external/default grant forbidden")
    for catalog, owner in (
        ("pg_class", "relowner"),
        ("pg_proc", "proowner"),
        ("pg_namespace", "nspowner"),
        ("pg_type", "typowner"),
        ("pg_database", "datdba"),
        ("pg_tablespace", "spcowner"),
        ("pg_language", "lanowner"),
        ("pg_foreign_server", "srvowner"),
        ("pg_foreign_data_wrapper", "fdwowner"),
        ("pg_largeobject_metadata", "lomowner"),
    ):
        if (
            conn.execute(
                sql.SQL("SELECT 1 FROM pg_catalog.{} WHERE {}=%s LIMIT 1").format(
                    sql.Identifier(catalog), sql.Identifier(owner)
                ),
                (principal_oid,),
            ).fetchone()
            is not None
        ):
            raise AtomicRootProofQualificationError("authority object ownership forbidden")


def qualify_atomic_root_proof_authority(
    conn: psycopg.Connection[Any],
    deployment: PostgreSQLAtomicRootProofTestDeployment,
    *,
    expected_role: str = "issuer_writer",
) -> None:
    """Fail closed on drift; no function enables the public issuance runtime."""
    validate_atomic_root_proof_deployment(deployment)
    if expected_role not in {"issuer_writer", "test_admin"}:
        raise AtomicRootProofQualificationError("explicit administrative test deployment required")
    try:
        config = deployment.provisioning
        conn.execute("SET LOCAL search_path=pg_catalog")
        principal = (
            config.issuer_writer_role
            if expected_role == "issuer_writer"
            else config.test_admin_role
        )
        if conn.execute("SELECT session_user::text,current_user::text").fetchone() != (
            principal,
            principal,
        ):
            raise AtomicRootProofQualificationError(
                "exact direct-login authority principal required"
            )
        if conn.info.server_version // 10000 != 16:
            raise AtomicRootProofQualificationError("exact PostgreSQL 16 required")
        durable = conn.execute(
            "SELECT pg_catalog.current_setting('fsync'),"
            "pg_catalog.current_setting('synchronous_commit'),"
            "pg_catalog.current_setting('full_page_writes'),pg_catalog.pg_is_in_recovery()"
        ).fetchone()
        if durable != ("on", "on", "on", False):
            raise AtomicRootProofQualificationError("durable PostgreSQL primary required")
        if _physical_identity(conn, config.instance_id) != deployment.identity:
            raise AtomicRootProofQualificationError(
                "physical database or service identity mismatch"
            )
        metadata = conn.execute(
            sql.SQL(
                "SELECT schema_identity,schema_version,migration_id,instance_id,trust_domain,"
                "schema_owner_role,schema_owner_role_oid,issuer_writer_role,issuer_writer_role_oid,"
                "test_admin_role,test_admin_role_oid,system_identifier,db_oid,boundary "
                "FROM {}.metadata WHERE singleton"
            ).format(sql.Identifier(config.schema))
        ).fetchall()
        roles = dict(deployment.role_oids)
        expected = (
            SCHEMA_IDENTITY,
            SCHEMA_VERSION,
            MIGRATION_ID,
            config.instance_id,
            config.trust_domain,
            config.schema_owner_role,
            roles[config.schema_owner_role],
            config.issuer_writer_role,
            roles[config.issuer_writer_role],
            config.test_admin_role,
            roles[config.test_admin_role],
            str(deployment.identity.system_identifier),
            deployment.identity.database_oid,
            "ISOLATED_TEST_ONLY",
        )
        if metadata != [expected]:
            raise AtomicRootProofQualificationError("atomic migration identity/version mismatch")
        _qualify_roles(conn, deployment)
        _qualify_closed_principal(conn, principal)
        if _catalog_fingerprint(conn, config, deployment.role_oids) != deployment.catalog_sha256:
            raise AtomicRootProofQualificationError("exact atomic V1 catalog or ACL drift")
        if conn.execute(
            "SELECT pg_catalog.has_database_privilege(current_user,"
            "pg_catalog.current_database(),'CREATE,TEMPORARY')"
        ).fetchone() != (False,):
            raise AtomicRootProofQualificationError("authority database DDL privilege forbidden")
        if (
            conn.execute(
                "SELECT 1 FROM pg_catalog.pg_namespace WHERE "
                "pg_catalog.has_schema_privilege(current_user,oid,'CREATE') LIMIT 1"
            ).fetchone()
            is not None
        ):
            raise AtomicRootProofQualificationError("authority schema DDL privilege forbidden")
        if (
            conn.execute(
                "SELECT 1 FROM pg_catalog.pg_class WHERE relkind IN ('r','p','v','m','f') AND "
                "pg_catalog.has_table_privilege(current_user,oid,"
                "'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER') LIMIT 1"
            ).fetchone()
            is not None
        ):
            raise AtomicRootProofQualificationError("authority direct table DML forbidden")
        if (
            conn.execute(
                "SELECT 1 FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n "
                "ON n.oid=p.pronamespace WHERE p.prosecdef AND n.nspname<>ALL(%s) AND "
                "pg_catalog.has_schema_privilege(current_user,n.oid,'USAGE') AND "
                "pg_catalog.has_function_privilege(current_user,p.oid,'EXECUTE') LIMIT 1",
                (_schemas(config),),
            ).fetchone()
            is not None
        ):
            raise AtomicRootProofQualificationError("external security-definer authority forbidden")
        # The static owner API also pins protected source metadata rows.  Its
        # immutable database manifest is checked by direct SQL commits as well.
        if conn.execute(
            sql.SQL("SELECT {}.qualify_catalog()").format(sql.Identifier(config.schema))
        ).fetchone() != (True,):
            raise AtomicRootProofQualificationError(
                "atomic database manifest qualification refused"
            )
    except AtomicRootProofQualificationError:
        raise
    except (psycopg.Error, ValueError, TypeError, KeyError, IndexError) as exc:
        raise AtomicRootProofQualificationError(
            "atomic authority live qualification unavailable"
        ) from exc


def validate_atomic_root_proof_deployment(deployment: object) -> None:
    """Refuse detached caller evidence before opening any authority connection."""
    if (
        type(deployment) is not PostgreSQLAtomicRootProofTestDeployment
        or deployment._origin is not _ADMINISTRATIVE_ORIGIN
    ):
        raise AtomicRootProofQualificationError("explicit administrative test deployment required")


__all__ = [
    "AtomicRootProofQualificationError",
    "PostgreSQLAtomicRootProofPhysicalIdentity",
    "PostgreSQLAtomicRootProofTestDeployment",
    "qualify_atomic_root_proof_authority",
    "read_atomic_root_proof_physical_identity",
    "validate_atomic_root_proof_deployment",
]
