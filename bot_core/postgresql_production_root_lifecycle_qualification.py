"""Read-only qualification of the separate production lifecycle authority.

The expected catalog is an installation pin held outside PostgreSQL. Runtime
never learns its expected identity or catalog from the connection it checks.
This module does not authorize issuance and does not adapt a TEST_ONLY authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

import psycopg
from psycopg import sql
from psycopg.pq import TransactionStatus

from bot_core.postgresql_atomic_root_proof_qualification import (
    PostgreSQLAtomicRootProofPhysicalIdentity,
    read_atomic_root_proof_physical_identity,
)
from bot_core.postgresql_production_root_lifecycle_schema import (
    ADMINISTRATOR_FUNCTIONS,
    ISSUER_FUNCTIONS,
    PRODUCTION_LIFECYCLE_FUNCTION_DECLARATIONS,
    RELATIONS,
    SCHEMA_VERSION,
    PostgreSQLProductionRootLifecycleProvisioning,
    _reviewed_v1_function_sources,
    reviewed_production_lifecycle_function_sources,
)


class ProductionRootLifecycleQualificationError(RuntimeError):
    """The live authority does not match the trusted installation."""


@dataclass(frozen=True, slots=True)
class PostgreSQLProductionRootLifecyclePin:
    """Installation-owned expectations, with no administrator credentials.

    Constructing this value is not a provisioning ceremony. The service accepts
    it only from its fixed, independently protected installation configuration.
    The physical identity reader is shared with the old test qualifier; none of
    its test deployment objects, capabilities, catalogs or migrations are used.
    """

    provisioning: PostgreSQLProductionRootLifecycleProvisioning
    identity: PostgreSQLAtomicRootProofPhysicalIdentity
    catalog_sha256: str
    role_oids: tuple[tuple[str, int], ...]
    profile: str = field(default="PRODUCTION_LOCAL", init=False)

    def __post_init__(self) -> None:
        if type(self.provisioning) is not PostgreSQLProductionRootLifecycleProvisioning:
            raise TypeError("exact independent production provisioning required")
        if type(self.identity) is not PostgreSQLAtomicRootProofPhysicalIdentity:
            raise TypeError("exact physical installation identity required")
        if self.identity.instance_id != self.provisioning.installation_id:
            raise ValueError("physical identity belongs to another installation")
        if (
            type(self.catalog_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", self.catalog_sha256) is None
        ):
            raise ValueError("trusted installation catalog digest required")
        names = _role_names(self.provisioning)
        if (
            type(self.role_oids) is not tuple
            or len(self.role_oids) != 3
            or any(
                type(pair) is not tuple
                or len(pair) != 2
                or type(pair[0]) is not str
                or type(pair[1]) is not int
                or pair[1] <= 0
                for pair in self.role_oids
            )
            or {name for name, _ in self.role_oids} != set(names)
            or len({oid for _, oid in self.role_oids}) != 3
        ):
            raise ValueError("three distinct installation role OIDs required")


def _role_names(config: PostgreSQLProductionRootLifecycleProvisioning) -> tuple[str, ...]:
    return (config.schema_owner_role, config.issuer_role, config.administrator_role)


def _check_transaction(conn: psycopg.Connection[Any], isolation: str = "serializable") -> None:
    if conn.info.transaction_status is not TransactionStatus.INTRANS or conn.execute(
        "SELECT pg_catalog.current_setting('transaction_isolation')"
    ).fetchone() != (isolation,):
        raise ProductionRootLifecycleQualificationError(
            f"qualification must belong to the operation's {isolation.upper()} transaction"
        )


def _check_physical(
    conn: psycopg.Connection[Any],
    config: PostgreSQLProductionRootLifecycleProvisioning,
    expected: PostgreSQLAtomicRootProofPhysicalIdentity,
) -> None:
    if conn.info.server_version // 10000 != 16:
        raise ProductionRootLifecycleQualificationError("PostgreSQL 16 required")
    if conn.execute(
        "SELECT pg_catalog.current_setting('fsync'),"
        "pg_catalog.current_setting('synchronous_commit'),"
        "pg_catalog.current_setting('full_page_writes'),pg_catalog.pg_is_in_recovery()"
    ).fetchone() != ("on", "on", "on", False):
        raise ProductionRootLifecycleQualificationError("durable PostgreSQL primary required")
    if read_atomic_root_proof_physical_identity(conn, config.installation_id) != expected:
        raise ProductionRootLifecycleQualificationError("physical installation identity differs")


def _check_roles(
    conn: psycopg.Connection[Any],
    config: PostgreSQLProductionRootLifecycleProvisioning,
    role_oids: tuple[tuple[str, int], ...],
) -> None:
    for name, oid in role_oids:
        row = conn.execute(
            "SELECT oid,rolcanlogin,rolinherit,rolsuper,rolcreaterole,rolcreatedb,"
            "rolreplication,rolbypassrls FROM pg_catalog.pg_roles WHERE rolname=%s",
            (name,),
        ).fetchone()
        if row != (oid, name != config.schema_owner_role, False, False, False, False, False, False):
            raise ProductionRootLifecycleQualificationError("role OID or minimal attributes differ")
    oids = [oid for _, oid in role_oids]
    if (
        conn.execute(
            "SELECT 1 FROM pg_catalog.pg_auth_members WHERE roleid=ANY(%s) OR member=ANY(%s) LIMIT 1",
            (oids, oids),
        ).fetchone()
        is not None
    ):
        raise ProductionRootLifecycleQualificationError("authority role memberships forbidden")


def _check_closed_login(conn: psycopg.Connection[Any], principal: str, schema: str) -> None:
    """Check effective privileges, including PUBLIC, views and column grants."""
    forbidden = (
        (
            "SELECT 1 WHERE pg_catalog.has_database_privilege(%s,"
            "pg_catalog.current_database(),'CREATE,TEMPORARY')",
            (principal,),
        ),
        (
            "SELECT 1 FROM pg_catalog.pg_namespace WHERE "
            "pg_catalog.has_schema_privilege(%s,oid,'CREATE') LIMIT 1",
            (principal,),
        ),
        (
            "SELECT 1 FROM pg_catalog.pg_class WHERE relkind IN ('r','p','v','m','f','S') "
            "AND oid<>'pg_catalog.pg_settings'::pg_catalog.regclass AND "
            "pg_catalog.has_table_privilege(%s,oid,"
            "'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER') LIMIT 1",
            (principal,),
        ),
        (
            "SELECT 1 FROM pg_catalog.pg_class WHERE relkind IN ('r','p','v','m','f') "
            "AND oid<>'pg_catalog.pg_settings'::pg_catalog.regclass AND "
            "pg_catalog.has_any_column_privilege(%s,oid,'INSERT,UPDATE,REFERENCES') LIMIT 1",
            (principal,),
        ),
        (
            "SELECT 1 FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n "
            "ON n.oid=p.pronamespace WHERE p.prosecdef AND n.nspname<>%s AND "
            "pg_catalog.has_schema_privilege(%s,n.oid,'USAGE') AND "
            "pg_catalog.has_function_privilege(%s,p.oid,'EXECUTE') LIMIT 1",
            (schema, principal, principal),
        ),
        (
            "SELECT 1 FROM pg_catalog.pg_default_acl d CROSS JOIN LATERAL "
            "pg_catalog.aclexplode(d.defaclacl) a WHERE a.grantee="
            "(SELECT oid FROM pg_catalog.pg_roles WHERE rolname=%s) LIMIT 1",
            (principal,),
        ),
    )
    for query, params in forbidden:
        if conn.execute(query, params).fetchone() is not None:
            raise ProductionRootLifecycleQualificationError(
                "authority login has an alternate write or privilege path"
            )
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
                sql.SQL(
                    "SELECT 1 FROM pg_catalog.{} WHERE {}="
                    "(SELECT oid FROM pg_catalog.pg_roles WHERE rolname=%s) LIMIT 1"
                ).format(sql.Identifier(catalog), sql.Identifier(owner)),
                (principal,),
            ).fetchone()
            is not None
        ):
            raise ProductionRootLifecycleQualificationError("authority login owns an object")
    for catalog, acl in _SERVICE_ACLS:
        if (
            conn.execute(
                sql.SQL(
                    "SELECT 1 FROM pg_catalog.{} o CROSS JOIN LATERAL "
                    "pg_catalog.aclexplode(o.{}) a WHERE a.grantee="
                    "(SELECT oid FROM pg_catalog.pg_roles WHERE rolname=%s) LIMIT 1"
                ).format(sql.Identifier(catalog), sql.Identifier(acl)),
                (principal,),
            ).fetchone()
            is not None
        ):
            raise ProductionRootLifecycleQualificationError("external service grant forbidden")


_SERVICE_ACLS = (
    ("pg_type", "typacl"),
    ("pg_tablespace", "spcacl"),
    ("pg_language", "lanacl"),
    ("pg_foreign_server", "srvacl"),
    ("pg_foreign_data_wrapper", "fdwacl"),
    ("pg_largeobject_metadata", "lomacl"),
    ("pg_parameter_acl", "paracl"),
)


def _check_reviewed_schema(
    conn: psycopg.Connection[Any],
    config: PostgreSQLProductionRootLifecycleProvisioning,
    role_oids: tuple[tuple[str, int], ...],
    *,
    legacy_v1: bool = False,
) -> None:
    oids = dict(role_oids)
    owner = oids[config.schema_owner_role]
    schema = conn.execute(
        "SELECT oid,nspowner FROM pg_catalog.pg_namespace WHERE nspname=%s", (config.schema,)
    ).fetchone()
    if schema is None or schema[1] != owner:
        raise ProductionRootLifecycleQualificationError("production schema ownership differs")
    grants = conn.execute(
        "SELECT a.grantor,a.grantee,a.privilege_type,a.is_grantable "
        "FROM pg_catalog.pg_namespace n CROSS JOIN LATERAL "
        "pg_catalog.aclexplode(n.nspacl) a WHERE n.oid=%s",
        (schema[0],),
    ).fetchall()
    expected_schema = {
        (owner, owner, "USAGE", False),
        (owner, owner, "CREATE", False),
        (owner, oids[config.issuer_role], "USAGE", False),
        (owner, oids[config.administrator_role], "USAGE", False),
    }
    if set(grants) != expected_schema or len(grants) != len(expected_schema):
        raise ProductionRootLifecycleQualificationError("production schema ACL differs")
    sources = (
        _reviewed_v1_function_sources(config)
        if legacy_v1
        else reviewed_production_lifecycle_function_sources(config)
    )
    rows = conn.execute(
        "SELECT p.oid,p.proname||'('||pg_catalog.oidvectortypes(p.proargtypes)||')',"
        "p.proowner,p.prosecdef,l.lanname,p.proconfig,p.prosrc,"
        "pg_catalog.format_type(p.prorettype,NULL),p.prokind,p.provolatile,p.proisstrict,"
        "p.proleakproof,p.proparallel,p.proretset,p.pronargdefaults,p.provariadic,"
        "p.prosupport,p.probin,p.proargnames,p.proargmodes,p.proallargtypes,p.proargdefaults "
        "FROM pg_catalog.pg_proc p "
        "JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace "
        "JOIN pg_catalog.pg_language l ON l.oid=p.prolang WHERE n.nspname=%s",
        (config.schema,),
    ).fetchall()
    # oidvectortypes inserts spaces after commas, while reviewed declarations
    # use the PostgreSQL regprocedure spelling without insignificant spaces.
    if len(rows) != len(sources) or {row[1].replace(", ", ",") for row in rows} != set(sources):
        raise ProductionRootLifecycleQualificationError("reviewed function inventory differs")
    for row in rows:
        signature = row[1].replace(", ", ",")
        declaration, returns = PRODUCTION_LIFECYCLE_FUNCTION_DECLARATIONS[signature]
        argument_names = (
            [argument.strip().split()[0] for argument in declaration.split(",")]
            if declaration
            else None
        )
        if row[2:] != (
            owner,
            True,
            "plpgsql",
            ["search_path=pg_catalog"],
            sources[signature],
            returns,
            "f",
            "v",
            False,
            False,
            "u",
            False,
            0,
            0,
            "-",
            None,
            argument_names,
            None,
            None,
            None,
        ):
            raise ProductionRootLifecycleQualificationError(
                "reviewed function source or security attributes differ"
            )
        expected = {(owner, owner, "EXECUTE", False)}
        if signature in ISSUER_FUNCTIONS:
            expected.add((owner, oids[config.issuer_role], "EXECUTE", False))
        if signature in ADMINISTRATOR_FUNCTIONS or (
            legacy_v1 and signature == "complete_close(text,text,bytea)"
        ):
            expected.add((owner, oids[config.administrator_role], "EXECUTE", False))
        acl = conn.execute(
            "SELECT a.grantor,a.grantee,a.privilege_type,a.is_grantable "
            "FROM pg_catalog.pg_proc p CROSS JOIN LATERAL "
            "pg_catalog.aclexplode(p.proacl) a WHERE p.oid=%s",
            (row[0],),
        ).fetchall()
        if len(acl) != len(expected) or set(acl) != expected:
            raise ProductionRootLifecycleQualificationError("reviewed function ACL differs")
    tables = conn.execute(
        "SELECT c.oid,c.relname,c.relowner,c.relkind FROM pg_catalog.pg_class c "
        "JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
        "WHERE n.nspname=%s AND c.relkind<>'i'",
        (config.schema,),
    ).fetchall()
    if {row[1] for row in tables} != {
        "metadata",
        "catalog_manifest",
        "fences",
        "slots",
        "operations",
        "completions",
        "observations",
    } or any(row[2:] != (owner, "r") for row in tables):
        raise ProductionRootLifecycleQualificationError("production relation inventory differs")
    for oid, name, _, _ in tables:
        acl = conn.execute(
            "SELECT a.grantor,a.grantee,a.privilege_type,a.is_grantable "
            "FROM pg_catalog.pg_class c CROSS JOIN LATERAL "
            "pg_catalog.aclexplode(COALESCE(c.relacl,pg_catalog.acldefault('r',c.relowner))) a "
            "WHERE c.oid=%s",
            (oid,),
        ).fetchall()
        expected = {
            (owner, owner, privilege, False)
            for privilege in (
                "INSERT",
                "SELECT",
                "UPDATE",
                "DELETE",
                "TRUNCATE",
                "REFERENCES",
                "TRIGGER",
            )
        }
        if name in {"metadata", "catalog_manifest"}:
            expected.update(
                {
                    (owner, oids[role], "SELECT", False)
                    for role in (config.issuer_role, config.administrator_role)
                }
            )
        if len(acl) != len(expected) or set(acl) != expected:
            raise ProductionRootLifecycleQualificationError("production relation ACL differs")
    if (
        conn.execute(
            "SELECT 1 FROM pg_catalog.pg_attribute a JOIN pg_catalog.pg_class c ON c.oid=a.attrelid "
            "WHERE c.relnamespace=%s AND a.attacl IS NOT NULL LIMIT 1",
            (schema[0],),
        ).fetchone()
        is not None
    ):
        raise ProductionRootLifecycleQualificationError("production column grants forbidden")
    conn.execute(sql.SQL("SELECT {}.assert_catalog()").format(sql.Identifier(config.schema)))


def _catalog_fingerprint(
    conn: psycopg.Connection[Any],
    config: PostgreSQLProductionRootLifecycleProvisioning,
    role_oids: tuple[tuple[str, int], ...],
    *,
    migration_offline: bool = False,
) -> str:
    """Pin immutable catalog behavior and global privilege paths, never records."""
    oids = [oid for _, oid in role_oids]
    schema = config.schema
    queries: dict[str, tuple[str, tuple[object, ...]]] = {
        "roles": (
            "SELECT oid,rolname,rolcanlogin,rolinherit,rolsuper,rolcreaterole,rolcreatedb,"
            "rolreplication,rolbypassrls,rolconnlimit,rolvaliduntil::text,rolconfig "
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
            "WHERE nspname=%s ORDER BY oid",
            (schema,),
        ),
        "relations": (
            "SELECT c.oid,c.relname,c.relkind,c.relpersistence,c.relowner,c.relacl::text,"
            "c.relispartition,c.relrowsecurity,c.relforcerowsecurity,c.relreplident,"
            "c.reloptions,c.reltablespace,c.relam FROM pg_catalog.pg_class c "
            "JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=%s ORDER BY c.oid",
            (schema,),
        ),
        "columns": (
            "SELECT c.oid,a.attnum,a.attname,a.atttypid,a.atttypmod,a.attnotnull,"
            "a.attidentity,a.attgenerated,a.attisdropped,a.attcollation,a.attacl::text,"
            "pg_catalog.pg_get_expr(d.adbin,d.adrelid) FROM pg_catalog.pg_attribute a "
            "JOIN pg_catalog.pg_class c ON c.oid=a.attrelid "
            "JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
            "LEFT JOIN pg_catalog.pg_attrdef d ON d.adrelid=a.attrelid AND d.adnum=a.attnum "
            "WHERE n.nspname=%s AND a.attnum>0 ORDER BY c.oid,a.attnum",
            (schema,),
        ),
        "constraints": (
            "SELECT c.oid,c.conname,c.contype,c.conrelid,c.confrelid,c.conindid,c.conkey,"
            "c.confkey,c.condeferrable,c.condeferred,c.convalidated,"
            "pg_catalog.pg_get_constraintdef(c.oid,true) FROM pg_catalog.pg_constraint c "
            "JOIN pg_catalog.pg_namespace n ON n.oid=c.connamespace "
            "WHERE n.nspname=%s ORDER BY c.oid",
            (schema,),
        ),
        "indexes": (
            "SELECT c.oid,i.indrelid,i.indisunique,i.indisprimary,i.indisvalid,i.indisready,"
            "i.indislive,i.indkey::text,i.indclass::text,pg_catalog.pg_get_indexdef(c.oid) "
            "FROM pg_catalog.pg_index i JOIN pg_catalog.pg_class c ON c.oid=i.indexrelid "
            "JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=%s ORDER BY c.oid",
            (schema,),
        ),
        "functions": (
            "SELECT p.oid,p.proname,p.proowner,p.prolang,p.prosecdef,p.proconfig,p.prosrc,"
            "p.probin,p.proacl::text,p.provolatile,p.proisstrict,p.proleakproof,p.proparallel,"
            "p.prokind,p.prorettype,p.proretset,p.proargtypes::text,p.proallargtypes,"
            "p.proargmodes,p.proargnames,p.provariadic,p.pronargdefaults,p.prosupport,"
            "pg_catalog.pg_get_functiondef(p.oid) FROM pg_catalog.pg_proc p "
            "JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace "
            "WHERE n.nspname=%s ORDER BY p.oid",
            (schema,),
        ),
        "languages": (
            "SELECT l.oid,l.lanname,l.lanowner,l.lanispl,l.lanpltrusted,l.lanplcallfoid,"
            "l.laninline,l.lanvalidator,l.lanacl::text FROM pg_catalog.pg_language l "
            "WHERE l.oid IN (SELECT p.prolang FROM pg_catalog.pg_proc p "
            "JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace WHERE n.nspname=%s) "
            "ORDER BY l.oid",
            (schema,),
        ),
        "triggers": (
            "SELECT t.oid,t.tgrelid,t.tgname,t.tgfoid,t.tgtype,t.tgenabled,t.tgisinternal,"
            "t.tgconstraint,t.tgdeferrable,t.tginitdeferred,pg_catalog.pg_get_triggerdef(t.oid,true) "
            "FROM pg_catalog.pg_trigger t JOIN pg_catalog.pg_class c ON c.oid=t.tgrelid "
            "JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=%s ORDER BY t.oid",
            (schema,),
        ),
        "rules": (
            "SELECT r.oid,r.ev_class,r.rulename,pg_catalog.pg_get_ruledef(r.oid,true) "
            "FROM pg_catalog.pg_rewrite r JOIN pg_catalog.pg_class c ON c.oid=r.ev_class "
            "JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=%s ORDER BY r.oid",
            (schema,),
        ),
        "policies": (
            "SELECT p.oid,p.polrelid,p.polname,p.polcmd,p.polpermissive,p.polroles,"
            "pg_catalog.pg_get_expr(p.polqual,p.polrelid),"
            "pg_catalog.pg_get_expr(p.polwithcheck,p.polrelid) FROM pg_catalog.pg_policy p "
            "JOIN pg_catalog.pg_class c ON c.oid=p.polrelid "
            "JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=%s ORDER BY p.oid",
            (schema,),
        ),
        "types": (
            "SELECT t.oid,t.typname,t.typowner,t.typtype,t.typbasetype,t.typrelid,"
            "t.typinput,t.typoutput,t.typreceive,t.typsend,t.typmodin,t.typmodout,"
            "t.typcollation,t.typacl::text FROM pg_catalog.pg_type t "
            "JOIN pg_catalog.pg_namespace n ON n.oid=t.typnamespace "
            "WHERE n.nspname=%s ORDER BY t.oid",
            (schema,),
        ),
        "defaults": (
            "SELECT defaclrole,defaclnamespace,defaclobjtype,defaclacl::text "
            "FROM pg_catalog.pg_default_acl WHERE defaclrole=ANY(%s) OR defaclnamespace="
            "(SELECT oid FROM pg_catalog.pg_namespace WHERE nspname=%s) "
            "ORDER BY defaclrole,defaclnamespace,defaclobjtype",
            (oids, schema),
        ),
        "role_settings": (
            "SELECT setdatabase,setrole,setconfig FROM pg_catalog.pg_db_role_setting "
            "WHERE setrole=ANY(%s) OR setrole=0 ORDER BY setdatabase,setrole",
            (oids,),
        ),
        "databases": (
            "SELECT oid,datname,datdba,datallowconn,datacl::text FROM pg_catalog.pg_database "
            "ORDER BY oid",
            (),
        ),
        # PostgreSQL exposes PUBLIC UPDATE on this built-in session-settings
        # view. It grants no authority beyond set_config(), which every login
        # already has. Pin its definition and auxiliary behavior for this one
        # precise exception to the global DML prohibition.
        "session_settings_view": (
            "SELECT c.oid,c.relowner,c.relkind,c.relacl::text,c.reloptions,"
            "pg_catalog.pg_get_viewdef(c.oid,true) FROM pg_catalog.pg_class c "
            "WHERE c.oid='pg_catalog.pg_settings'::pg_catalog.regclass",
            (),
        ),
        "session_settings_rules": (
            "SELECT oid,rulename,ev_type,ev_enabled,is_instead,"
            "pg_catalog.pg_get_ruledef(oid,true) FROM pg_catalog.pg_rewrite "
            "WHERE ev_class='pg_catalog.pg_settings'::pg_catalog.regclass ORDER BY oid",
            (),
        ),
        "session_settings_triggers": (
            "SELECT oid,tgfoid,tgenabled,pg_catalog.pg_get_triggerdef(oid,true) "
            "FROM pg_catalog.pg_trigger WHERE tgrelid='pg_catalog.pg_settings'::pg_catalog.regclass "
            "ORDER BY oid",
            (),
        ),
        "external_definers": (
            "SELECT p.oid,p.proowner,p.prolang,p.prosrc,p.probin,p.proconfig,p.proacl::text,"
            "pg_catalog.pg_get_functiondef(p.oid) FROM pg_catalog.pg_proc p "
            "JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace "
            "WHERE n.nspname<>%s AND p.prosecdef AND (p.proowner=ANY(%s) OR "
            "(pg_catalog.has_schema_privilege(%s,n.oid,'USAGE') AND "
            "pg_catalog.has_function_privilege(%s,p.oid,'EXECUTE')) OR "
            "(pg_catalog.has_schema_privilege(%s,n.oid,'USAGE') AND "
            "pg_catalog.has_function_privilege(%s,p.oid,'EXECUTE'))) ORDER BY p.oid",
            (
                schema,
                oids,
                config.issuer_role,
                config.issuer_role,
                config.administrator_role,
                config.administrator_role,
            ),
        ),
    }
    facts = {
        name: conn.execute(query, params).fetchall() for name, (query, params) in queries.items()
    }
    if migration_offline:
        # The controlled migration requires a separately committed connection
        # gate. Only this reviewed delta from the retained online V1 catalog is
        # normalized; runtime and installation pinning always hash actual facts.
        database_oid = conn.execute(
            "SELECT oid FROM pg_catalog.pg_database WHERE datname=pg_catalog.current_database()"
        ).fetchone()[0]
        facts["databases"] = [
            (*row[:3], True, *row[4:]) if row[0] == database_oid and row[3] is False else row
            for row in facts["databases"]
        ]
    # Include PUBLIC ACLs as well as direct grants: changing a public view or
    # helper grant must invalidate an otherwise unchanged protected schema.
    for name, catalog, acl, owner in (
        ("schema_acls", "pg_namespace", "nspacl", "nspowner"),
        ("relation_acls", "pg_class", "relacl", "relowner"),
        ("function_acls", "pg_proc", "proacl", "proowner"),
        *((catalog + "_acls", catalog, acl, None) for catalog, acl in _SERVICE_ACLS),
    ):
        facts[name] = conn.execute(
            sql.SQL(
                "SELECT o.oid,a.grantor,a.grantee,a.privilege_type,a.is_grantable "
                "FROM pg_catalog.{} o CROSS JOIN LATERAL pg_catalog.aclexplode(o.{}) a "
                "WHERE a.grantee=ANY(%s) OR a.grantee=0 "
                "ORDER BY o.oid,a.grantor,a.grantee,a.privilege_type"
            ).format(sql.Identifier(catalog), sql.Identifier(acl)),
            (oids,),
        ).fetchall()
        if owner is not None:
            facts[name + "_ownership"] = conn.execute(
                sql.SQL("SELECT oid,{} FROM pg_catalog.{} WHERE {}=ANY(%s) ORDER BY oid").format(
                    sql.Identifier(owner), sql.Identifier(catalog), sql.Identifier(owner)
                ),
                (oids,),
            ).fetchall()
    facts["column_acls"] = conn.execute(
        "SELECT a.attrelid,a.attnum,g.grantor,g.grantee,g.privilege_type,g.is_grantable "
        "FROM pg_catalog.pg_attribute a CROSS JOIN LATERAL pg_catalog.aclexplode(a.attacl) g "
        "WHERE g.grantee=ANY(%s) OR g.grantee=0 "
        "ORDER BY a.attrelid,a.attnum,g.grantor,g.grantee,g.privilege_type",
        (oids,),
    ).fetchall()
    encoded = json.dumps(facts, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(b"CryptoHunter/ProductionRootLifecycleCatalog/V1\0" + encoded).hexdigest()


def pin_production_root_lifecycle_catalog(
    conn: psycopg.Connection[Any],
    provisioning: PostgreSQLProductionRootLifecycleProvisioning,
    *,
    expected_identity: PostgreSQLAtomicRootProofPhysicalIdentity,
) -> PostgreSQLProductionRootLifecyclePin:
    """Explicit installation ceremony; never called by a runtime connection.

    A separate direct bootstrap superuser pins the reviewed installation. The
    returned value must be durably installed under independent OS protection.
    This function neither creates roles nor grants privileges nor repairs drift.
    """
    if (
        type(provisioning) is not PostgreSQLProductionRootLifecycleProvisioning
        or type(expected_identity) is not PostgreSQLAtomicRootProofPhysicalIdentity
    ):
        raise TypeError("exact production provisioning and preselected physical identity required")
    if expected_identity.instance_id != provisioning.installation_id:
        raise ProductionRootLifecycleQualificationError("installation identity differs")
    try:
        _check_transaction(conn)
        principal = conn.execute(
            "SELECT session_user::text,current_user::text,"
            "(SELECT rolsuper FROM pg_catalog.pg_roles WHERE rolname=session_user::text)"
        ).fetchone()
        if (
            principal is None
            or principal[0] != principal[1]
            or principal[2] is not True
            or principal[0] in _role_names(provisioning)
        ):
            raise ProductionRootLifecycleQualificationError(
                "separate direct bootstrap administrator required"
            )
        _check_physical(conn, provisioning, expected_identity)
        roles = tuple(
            conn.execute(
                "SELECT rolname,oid FROM pg_catalog.pg_roles WHERE rolname=ANY(%s) ORDER BY rolname",
                (list(_role_names(provisioning)),),
            ).fetchall()
        )
        if {name for name, _ in roles} != set(_role_names(provisioning)):
            raise ProductionRootLifecycleQualificationError("production roles missing")
        _check_roles(conn, provisioning, roles)
        for name in (provisioning.issuer_role, provisioning.administrator_role):
            _check_closed_login(conn, name, provisioning.schema)
        _check_reviewed_schema(conn, provisioning, roles)
        return PostgreSQLProductionRootLifecyclePin(
            provisioning, expected_identity, _catalog_fingerprint(conn, provisioning, roles), roles
        )
    except ProductionRootLifecycleQualificationError:
        raise
    except (psycopg.Error, RuntimeError, ValueError, TypeError, KeyError, IndexError) as exc:
        raise ProductionRootLifecycleQualificationError(
            "production installation qualification unavailable"
        ) from exc


def migrate_production_root_lifecycle_completion_block_v2(
    conn: psycopg.Connection[Any], legacy_pin: PostgreSQLProductionRootLifecyclePin
) -> None:
    """Explicit bootstrap-only replacement of the unsafe V1 completion port.

    The caller owns and commits one SERIALIZABLE transaction. Its externally
    retained V1 pin must match the reviewed old catalog and physical instance;
    this function never derives a replacement pin. After COMMIT, a separate
    installation ceremony must pin V2 and replace the protected installation.
    Existing evidence remains unchanged and legacy completions are reported as
    unverified. Migration cannot authenticate custody or authorize emission.

    Before opening this transaction, external maintenance control must commit
    ALLOW_CONNECTIONS=false and drain every other database connection. This
    bootstrap connection must already exist. The function neither disconnects
    sessions nor changes the connection gate. After migration COMMIT, external
    control restores connections before explicitly pinning the V2 catalog;
    runtime services retain their failing old pin until separately provisioned.
    """
    if type(legacy_pin) is not PostgreSQLProductionRootLifecyclePin:
        raise ProductionRootLifecycleQualificationError("trusted legacy installation pin required")
    _check_transaction(conn)
    config = legacy_pin.provisioning
    principal = conn.execute(
        "SELECT session_user::text,current_user::text,"
        "(SELECT rolsuper FROM pg_catalog.pg_roles WHERE rolname=session_user::text)"
    ).fetchone()
    if (
        principal is None
        or principal[0] != principal[1]
        or principal[2] is not True
        or principal[0] in _role_names(config)
    ):
        raise ProductionRootLifecycleQualificationError(
            "separate direct bootstrap administrator required for migration"
        )
    gate = conn.execute(
        "SELECT oid,datallowconn,xmin::text,pg_catalog.pg_current_xact_id()::text "
        "FROM pg_catalog.pg_database WHERE datname=pg_catalog.current_database()"
    ).fetchone()
    if gate is None or gate[1] is not False:
        raise ProductionRootLifecycleQualificationError(
            "separately committed closed database connection gate required for migration"
        )
    # Expand the catalog tuple's 32-bit xid against the current xid8 epoch.
    # Current/future xid values (including caller-owned subtransactions) fail
    # closed. An ambiguous gate spanning xid wrap requires a fresh ceremony.
    current_xid = int(gate[3])
    gate_xid = (current_xid & ~0xFFFFFFFF) | int(gate[2])
    if int(gate[2]) >= 3 and (
        not 0 < current_xid - gate_xid < 2**31
        or conn.execute("SELECT pg_catalog.pg_xact_status(%s::xid8)", (str(gate_xid),)).fetchone()
        != ("committed",)
    ):
        raise ProductionRootLifecycleQualificationError(
            "database connection gate must commit before migration transaction"
        )
    conn.execute("SELECT pg_catalog.pg_stat_clear_snapshot()")
    if (
        conn.execute(
            "SELECT 1 FROM pg_catalog.pg_stat_activity "
            "WHERE datid=%s AND pid<>pg_catalog.pg_backend_pid() LIMIT 1",
            (gate[0],),
        ).fetchone()
        is not None
    ):
        raise ProductionRootLifecycleQualificationError(
            "every other database connection must be drained before migration"
        )
    # No old function body may remain suspended before its first catalog read.
    # Table locks alone cannot guarantee that, hence the offline checks above.
    schema = sql.Identifier(config.schema)
    conn.execute(
        sql.SQL("LOCK TABLE {} IN ACCESS EXCLUSIVE MODE").format(
            sql.SQL(",").join(
                sql.SQL("{}.{}").format(schema, sql.Identifier(table))
                for table in sorted(RELATIONS)
            )
        )
    )
    _check_physical(conn, config, legacy_pin.identity)
    _check_roles(conn, config, legacy_pin.role_oids)
    for name in (config.issuer_role, config.administrator_role):
        _check_closed_login(conn, name, config.schema)
    _check_reviewed_schema(conn, config, legacy_pin.role_oids, legacy_v1=True)
    if (
        _catalog_fingerprint(conn, config, legacy_pin.role_oids, migration_offline=True)
        != legacy_pin.catalog_sha256
    ):
        raise ProductionRootLifecycleQualificationError("externally pinned legacy catalog differs")

    sources = reviewed_production_lifecycle_function_sources(config)
    for signature in (
        "assert_catalog()",
        "begin_close(jsonb)",
        "read_operation(text,text,text)",
        "complete_close(text,text,bytea)",
    ):
        declaration, returns = PRODUCTION_LIFECYCLE_FUNCTION_DECLARATIONS[signature]
        conn.execute(
            sql.SQL(
                "CREATE OR REPLACE FUNCTION {}.{}({}) RETURNS {} LANGUAGE plpgsql "
                "SECURITY DEFINER SET search_path=pg_catalog AS {}"
            ).format(
                schema,
                sql.Identifier(signature.split("(")[0]),
                sql.SQL(declaration),
                sql.SQL(returns),
                sql.Literal(sources[signature]),
            )
        )
    conn.execute(
        sql.SQL(
            "REVOKE ALL ON FUNCTION {}.complete_close(text,text,bytea) FROM PUBLIC,{},{}"
        ).format(
            schema, sql.Identifier(config.issuer_role), sql.Identifier(config.administrator_role)
        )
    )
    conn.execute(sql.SQL("ALTER TABLE {}.metadata DISABLE TRIGGER immutable_row").format(schema))
    conn.execute(
        sql.SQL("UPDATE {}.metadata SET schema_version=%s").format(schema), (SCHEMA_VERSION,)
    )
    conn.execute(
        sql.SQL("ALTER TABLE {}.metadata ENABLE ALWAYS TRIGGER immutable_row").format(schema)
    )
    # Restore every trigger before taking the replacement catalog snapshot.
    # INSERT remains permitted to the trusted owner, unlike UPDATE/DELETE.
    conn.execute(
        sql.SQL("ALTER TABLE {}.catalog_manifest DISABLE TRIGGER immutable_row").format(schema)
    )
    conn.execute(sql.SQL("DELETE FROM {}.catalog_manifest").format(schema))
    conn.execute(
        sql.SQL("ALTER TABLE {}.catalog_manifest ENABLE ALWAYS TRIGGER immutable_row").format(
            schema
        )
    )
    conn.execute(
        sql.SQL("INSERT INTO {}.catalog_manifest SELECT true,{}.catalog_snapshot()").format(
            schema, schema
        )
    )
    _check_reviewed_schema(conn, config, legacy_pin.role_oids)


def qualify_production_root_lifecycle_authority(
    conn: psycopg.Connection[Any],
    pin: PostgreSQLProductionRootLifecyclePin,
    *,
    expected_role: str = "issuer",
) -> None:
    """Qualify the operation's real transaction against external installation pins."""
    _qualify_authority(conn, pin, expected_role=expected_role, isolation="serializable")


def qualify_production_root_lifecycle_resolution(
    conn: psycopg.Connection[Any], pin: PostgreSQLProductionRootLifecyclePin
) -> None:
    """Qualify administrator synchronization under READ COMMITTED.

    This is exclusively the durable-operation resolution boundary. It does not
    authorize an issuer operation, custody mutation, or an ambiguous retry.
    """
    _qualify_authority(conn, pin, expected_role="administrator", isolation="read committed")


def _qualify_authority(
    conn: psycopg.Connection[Any],
    pin: PostgreSQLProductionRootLifecyclePin,
    *,
    expected_role: str,
    isolation: str,
) -> None:
    if type(pin) is not PostgreSQLProductionRootLifecyclePin:
        raise ProductionRootLifecycleQualificationError(
            "trusted production installation pin required"
        )
    if expected_role not in {"issuer", "administrator"}:
        raise ProductionRootLifecycleQualificationError(
            "explicit production semantic principal required"
        )
    try:
        _check_transaction(conn, isolation)
        config = pin.provisioning
        principal = config.issuer_role if expected_role == "issuer" else config.administrator_role
        if conn.execute("SELECT session_user::text,current_user::text").fetchone() != (
            principal,
            principal,
        ):
            raise ProductionRootLifecycleQualificationError(
                "exact direct-login transaction owner required"
            )
        _check_physical(conn, config, pin.identity)
        _check_roles(conn, config, pin.role_oids)
        for name in (config.issuer_role, config.administrator_role):
            _check_closed_login(conn, name, config.schema)
        _check_reviewed_schema(conn, config, pin.role_oids)
        if _catalog_fingerprint(conn, config, pin.role_oids) != pin.catalog_sha256:
            raise ProductionRootLifecycleQualificationError(
                "installation catalog or privilege paths changed"
            )
    except ProductionRootLifecycleQualificationError:
        raise
    except (psycopg.Error, RuntimeError, ValueError, TypeError, KeyError, IndexError) as exc:
        raise ProductionRootLifecycleQualificationError(
            "live production qualification unavailable"
        ) from exc


__all__ = [
    "PostgreSQLProductionRootLifecyclePin",
    "ProductionRootLifecycleQualificationError",
    "pin_production_root_lifecycle_catalog",
    "migrate_production_root_lifecycle_completion_block_v2",
    "qualify_production_root_lifecycle_authority",
    "qualify_production_root_lifecycle_resolution",
]
