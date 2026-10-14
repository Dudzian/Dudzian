"""Live read-only V2 preparation qualification shared by public readers.

This module cannot reach lifecycle administrator, migration, or bootstrap code.
Public readers still check the complete pinned catalog/ACL and physical identity
on each actual PostgreSQL transaction.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

import psycopg
from psycopg import sql

from bot_core.local_signing_custody import SigningCredentialSnapshot, SigningKeyLifecycle
from bot_core.postgresql_atomic_root_proof_qualification import (
    AtomicRootProofQualificationError,
    PostgreSQLAtomicRootProofPhysicalIdentity,
    _catalog_fingerprint,
    _qualify_closed_principal,
    _schemas,
)
from bot_core.postgresql_atomic_root_proof_schema import PostgreSQLAtomicRootProofProvisioning
from bot_core.postgresql_custody_lifecycle_contract_constants import (
    ENROLLMENT_MIGRATION_ID,
    SQL_CONTRACT_VERSION,
)
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core.postgresql_root_proof_retention import _canonical, _snapshot_data

MIGRATION_ID = "ROOT_PROOF_RESERVED_STAGED_PREPARATION_ACL_V2"
_BOUNDARY = "ISOLATED_TEST_ONLY"
_ORIGIN = object()
_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]{0,62}\Z")
_COMPONENTS = ("entitlement", "intent", "retention", "history", "requester", "claimant")
_FENCE_DOMAIN = b"CryptoHunter/AtomicTestFence/v1\0"


class RootProofPreparationV2QualificationError(AtomicRootProofQualificationError):
    """Reviewed preparation inventory, direct principal or signer fence differs."""


@dataclass(frozen=True, slots=True)
class PostgreSQLRootProofPreparationV2Deployment:
    """Public administrator-reviewed pin; contains no connection credentials."""

    provisioning: PostgreSQLAtomicRootProofProvisioning
    identity: PostgreSQLAtomicRootProofPhysicalIdentity
    preparation_schema: str
    role_oids: tuple[tuple[str, int], ...]
    owner_roles: tuple[str, ...]
    component_roles: tuple[tuple[str, str], ...]
    catalog_sha256: str
    predecessor_catalog_sha256: str
    _origin: object = field(repr=False, compare=False, default=None)


def validate_root_proof_preparation_v2_deployment(deployment: object) -> None:
    if (
        type(deployment) is not PostgreSQLRootProofPreparationV2Deployment
        or deployment._origin is not _ORIGIN
        or type(deployment.provisioning) is not PostgreSQLAtomicRootProofProvisioning
        or type(deployment.identity) is not PostgreSQLAtomicRootProofPhysicalIdentity
        or _IDENTIFIER.fullmatch(deployment.preparation_schema) is None
        or deployment.preparation_schema in _schemas(deployment.provisioning)
        or tuple(name for name, _ in deployment.component_roles) != _COMPONENTS
        or len({role for _, role in deployment.component_roles}) != len(_COMPONENTS)
        or re.fullmatch(r"[0-9a-f]{64}", deployment.catalog_sha256) is None
        or re.fullmatch(r"[0-9a-f]{64}", deployment.predecessor_catalog_sha256) is None
    ):
        raise RootProofPreparationV2QualificationError(
            "explicit reviewed preparation V2 deployment required"
        )


def _sources(
    config: PostgreSQLAtomicRootProofProvisioning,
    preparation_schema: str,
    component_roles: tuple[tuple[str, str], ...],
) -> dict[str, tuple[str, str, str]]:
    roles = dict(component_roles)
    permitted = ",".join("'" + value + "'" for value in roles.values())
    owner = config.schema_owner_role
    atomic = '"' + config.schema + '"'
    preparation = '"' + preparation_schema + '"'
    guard = (
        f"IF session_user::text NOT IN ({permitted}) OR current_user::text <> '{owner}' "
        "THEN RAISE EXCEPTION 'exact preparation principal required' USING ERRCODE='42501'; END IF;"
    )
    return {
        "qualify_preparation_v2()": (
            "",
            "jsonb",
            f"""
DECLARE result jsonb;
BEGIN
 {guard}
 IF pg_catalog.current_setting('server_version_num')::integer < 160000
 OR pg_catalog.current_setting('server_version_num')::integer >= 170000
 OR pg_catalog.current_setting('fsync') <> 'on'
 OR pg_catalog.current_setting('synchronous_commit') <> 'on'
 OR pg_catalog.current_setting('full_page_writes') <> 'on'
 OR pg_catalog.pg_is_in_recovery() THEN
  RAISE EXCEPTION 'durable PostgreSQL 16 primary required' USING ERRCODE='42501'; END IF;
 PERFORM {atomic}.assert_lifecycle_catalog();
 IF NOT EXISTS(SELECT 1 FROM {preparation}.metadata WHERE singleton
  AND migration_id='{MIGRATION_ID}' AND boundary='{_BOUNDARY}') THEN
  RAISE EXCEPTION 'explicit preparation ACL V2 required' USING ERRCODE='42501'; END IF;
 IF NOT EXISTS(SELECT 1 FROM {atomic}.lifecycle_sql_contract WHERE singleton
  AND sql_contract_version={SQL_CONTRACT_VERSION} AND migration_id='{ENROLLMENT_MIGRATION_ID}'
  AND predecessor_sql_contract_version=2
  AND predecessor_snapshot_sha256=pg_catalog.sha256(pg_catalog.convert_to(predecessor_snapshot::text,'UTF8')))
 THEN RAISE EXCEPTION 'signer enrollment SQL V3 required' USING ERRCODE='42501'; END IF;
 SELECT pg_catalog.jsonb_build_array(c.system_identifier,d.oid::bigint,d.datname,
  pg_catalog.current_setting('cluster_name'),pg_catalog.inet_server_addr()::text,
  pg_catalog.inet_server_port()) INTO STRICT result
 FROM pg_catalog.pg_control_system() c,pg_catalog.pg_database d
 WHERE d.datname=pg_catalog.current_database();
 RETURN result || pg_catalog.jsonb_build_array((SELECT pg_catalog.to_jsonb(m) FROM {preparation}.metadata m WHERE singleton));
END
""",
        ),
        "lock_active_signer_v2(bytea)": (
            "p_snapshot bytea",
            "void",
            f"""
DECLARE snapshot jsonb; binding jsonb; signer_fence_key text; fence {atomic}.fences%ROWTYPE;
BEGIN
 IF session_user::text NOT IN ('{roles["intent"]}','{roles["retention"]}')
 OR current_user::text <> '{owner}' THEN
  RAISE EXCEPTION 'exact preparation writer required' USING ERRCODE='42501'; END IF;
 PERFORM {preparation}.qualify_preparation_v2();
 snapshot:=pg_catalog.convert_from(p_snapshot,'UTF8')::jsonb;
 signer_fence_key:='ROOT_PROOF_SIGNING:' || pg_catalog.encode(pg_catalog.sha256(
  pg_catalog.decode('{_FENCE_DOMAIN.hex()}','hex') || p_snapshot),'hex');
 SELECT * INTO fence FROM {atomic}.fences WHERE identity=signer_fence_key FOR SHARE;
 IF NOT FOUND OR fence.semantic_role IS DISTINCT FROM 'ROOT_PROOF_SIGNING'
 OR fence.state IS DISTINCT FROM 'ACTIVE'
 OR pg_catalog.to_jsonb(fence.generation) IS DISTINCT FROM snapshot->'lifecycle_generation'
 THEN RAISE EXCEPTION 'original ACTIVE signer fence required' USING ERRCODE='23514'; END IF;
 SELECT credential_binding INTO binding FROM {atomic}.lifecycle_credentials
 WHERE lifecycle_credentials.fence_key=signer_fence_key;
 IF binding IS NULL OR binding->>'semantic_role' IS DISTINCT FROM 'ROOT_PROOF_SIGNING'
 OR binding->>'fence_key' IS DISTINCT FROM signer_fence_key
 OR binding->'credential_identity' IS DISTINCT FROM snapshot->'credential_identity'
 OR binding->'public_key_hex' IS DISTINCT FROM snapshot->'public_key_hex'
 OR binding->'key_version' IS DISTINCT FROM snapshot->'key_version'
 OR binding->'lifecycle_generation' IS DISTINCT FROM snapshot->'lifecycle_generation'
 OR snapshot->>'lifecycle_state' IS DISTINCT FROM 'ACTIVE'
 OR snapshot->'credential_identity'->>'semantic_role' IS DISTINCT FROM 'ROOT_PROOF_ISSUER_SIGNING'
 OR snapshot->>'key_material_identity' IS DISTINCT FROM binding->'credential_identity'->>'key_material_identity'
 OR snapshot->'provider_identity' IS DISTINCT FROM pg_catalog.jsonb_build_object(
  'role','ROOT_PROOF_SIGNING','provider_namespace',binding->'credential_identity'->'provider_namespace',
  'security',pg_catalog.jsonb_build_object('profile',binding->'custody_scope'->'security_profile',
   'trust_domain',binding->'custody_scope'->'trust_domain'))
 THEN RAISE EXCEPTION 'original signer requires exact enrollment' USING ERRCODE='23514'; END IF;
END
""",
        ),
    }


def _function_oid(conn: psycopg.Connection[Any], schema: str, signature: str) -> int:
    """Read a catalog identity without requiring USAGE on the target schema."""
    rows = conn.execute(
        "SELECT p.oid FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n "
        "ON n.oid=p.pronamespace WHERE n.nspname=%s AND p.proname || '(' || "
        "pg_catalog.replace(pg_catalog.oidvectortypes(p.proargtypes),', ', ',') || ')'=%s",
        (schema, signature),
    ).fetchall()
    if len(rows) != 1 or type(rows[0][0]) is not int or rows[0][0] <= 0:
        raise RootProofPreparationV2QualificationError("exact reviewed function OID required")
    return rows[0][0]


def _verify_preparation_sources(
    conn: psycopg.Connection[Any], deployment: PostgreSQLRootProofPreparationV2Deployment
) -> None:
    schema = deployment.preparation_schema
    owner = deployment.provisioning.schema_owner_role
    roles = dict(deployment.role_oids)
    namespace = conn.execute(
        "SELECT nspowner FROM pg_catalog.pg_namespace WHERE nspname=%s", (schema,)
    ).fetchone()
    if namespace != (roles[owner],):
        raise RootProofPreparationV2QualificationError("exact preparation schema owner required")
    schema_acl = set(
        conn.execute(
            "SELECT a.grantor,a.grantee,a.privilege_type,a.is_grantable FROM pg_catalog.pg_namespace n "
            "CROSS JOIN LATERAL pg_catalog.aclexplode(n.nspacl) a WHERE n.nspname=%s",
            (schema,),
        ).fetchall()
    )
    expected_schema_acl = {
        (roles[owner], roles[owner], privilege, False) for privilege in ("CREATE", "USAGE")
    } | {
        (roles[owner], roles[principal], "USAGE", False)
        for _, principal in deployment.component_roles
    }
    if schema_acl != expected_schema_acl:
        raise RootProofPreparationV2QualificationError("exact preparation schema ACL required")
    relations = conn.execute(
        "SELECT c.relname,c.relkind,c.relpersistence,c.relowner,c.relrowsecurity,c.relforcerowsecurity,c.reloptions "
        "FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
        "WHERE n.nspname=%s ORDER BY c.relname",
        (schema,),
    ).fetchall()
    if relations != [
        ("metadata", "r", "p", roles[owner], False, False, None),
        ("metadata_pkey", "i", "p", roles[owner], False, False, None),
    ]:
        raise RootProofPreparationV2QualificationError(
            "exact preparation relation inventory required"
        )
    columns = conn.execute(
        "SELECT a.attname,pg_catalog.format_type(a.atttypid,a.atttypmod),a.attnotnull,a.attidentity,a.attgenerated "
        "FROM pg_catalog.pg_attribute a WHERE a.attrelid=pg_catalog.to_regclass(%s) AND a.attnum>0 ORDER BY a.attnum",
        (schema + ".metadata",),
    ).fetchall()
    if columns != [
        ("singleton", "boolean", True, "", ""),
        ("migration_id", "text", True, "", ""),
        ("boundary", "text", True, "", ""),
        ("predecessor_catalog_sha256", "text", True, "", ""),
    ]:
        raise RootProofPreparationV2QualificationError(
            "exact preparation metadata columns required"
        )
    defaults = conn.execute(
        "SELECT a.attname,pg_catalog.pg_get_expr(d.adbin,d.adrelid),a.attacl::text,a.attisdropped "
        "FROM pg_catalog.pg_attribute a LEFT JOIN pg_catalog.pg_attrdef d "
        "ON d.adrelid=a.attrelid AND d.adnum=a.attnum "
        "WHERE a.attrelid=pg_catalog.to_regclass(%s) AND a.attnum>0 ORDER BY a.attnum",
        (schema + ".metadata",),
    ).fetchall()
    if defaults != [
        ("singleton", "true", None, False),
        ("migration_id", None, None, False),
        ("boundary", None, None, False),
        ("predecessor_catalog_sha256", None, None, False),
    ]:
        raise RootProofPreparationV2QualificationError(
            "exact preparation defaults and column ACL required"
        )
    constraints = conn.execute(
        "SELECT c.conname,c.contype,c.condeferrable,c.condeferred,c.convalidated,"
        "pg_catalog.pg_get_constraintdef(c.oid,true) FROM pg_catalog.pg_constraint c "
        "WHERE c.connamespace=(SELECT oid FROM pg_catalog.pg_namespace WHERE nspname=%s) ORDER BY c.conname",
        (schema,),
    ).fetchall()
    if constraints != [
        ("metadata_pkey", "p", False, False, True, "PRIMARY KEY (singleton)"),
        ("metadata_singleton_check", "c", False, False, True, "CHECK (singleton)"),
    ]:
        raise RootProofPreparationV2QualificationError("exact preparation constraints required")
    index = conn.execute(
        "SELECT i.indisunique,i.indisprimary,i.indisvalid,i.indisready,i.indislive,i.indkey::text,"
        "pg_catalog.pg_get_expr(i.indexprs,i.indrelid),pg_catalog.pg_get_expr(i.indpred,i.indrelid),"
        "i.indisreplident,i.indnullsnotdistinct FROM pg_catalog.pg_index i "
        "WHERE i.indrelid=pg_catalog.to_regclass(%s)",
        (schema + ".metadata",),
    ).fetchall()
    if index != [(True, True, True, True, True, "1", None, None, False, False)]:
        raise RootProofPreparationV2QualificationError("exact preparation unique index required")
    for catalog, column in (("pg_rewrite", "ev_class"), ("pg_policy", "polrelid")):
        if (
            conn.execute(
                sql.SQL(
                    "SELECT 1 FROM pg_catalog.{} WHERE {}=pg_catalog.to_regclass(%s) LIMIT 1"
                ).format(sql.Identifier(catalog), sql.Identifier(column)),
                (schema + ".metadata",),
            ).fetchone()
            is not None
        ):
            raise RootProofPreparationV2QualificationError(
                "preparation rewrite or policy forbidden"
            )
    types = conn.execute(
        "SELECT t.typname,t.typtype,t.typowner,t.typrelid=pg_catalog.to_regclass(%s),t.typacl::text "
        "FROM pg_catalog.pg_type t JOIN pg_catalog.pg_namespace n ON n.oid=t.typnamespace "
        "WHERE n.nspname=%s ORDER BY t.typname",
        (schema + ".metadata", schema),
    ).fetchall()
    if types != [
        ("_metadata", "b", roles[owner], False, None),
        ("metadata", "c", roles[owner], True, None),
    ]:
        raise RootProofPreparationV2QualificationError("exact preparation type inventory required")
    table_acl = set(
        conn.execute(
            "SELECT a.grantor,a.grantee,a.privilege_type,a.is_grantable FROM pg_catalog.pg_class c "
            "CROSS JOIN LATERAL pg_catalog.aclexplode(c.relacl) a WHERE c.oid=pg_catalog.to_regclass(%s)",
            (schema + ".metadata",),
        ).fetchall()
    )
    if table_acl != {
        (roles[owner], roles[owner], privilege, False)
        for privilege in (
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "TRUNCATE",
            "REFERENCES",
            "TRIGGER",
        )
    }:
        raise RootProofPreparationV2QualificationError("preparation metadata ACL differs")
    triggers = conn.execute(
        "SELECT t.tgname,t.tgtype,t.tgenabled,t.tgisinternal,t.tgfoid=%s::oid,"
        "t.tgattr::text,t.tgqual::text,t.tgnargs,t.tgargs,t.tgconstraint,"
        "t.tgdeferrable,t.tginitdeferred,t.tgoldtable,t.tgnewtable "
        "FROM pg_catalog.pg_trigger t WHERE t.tgrelid=pg_catalog.to_regclass(%s) ORDER BY t.tgname",
        (
            _function_oid(conn, deployment.provisioning.schema, "reject_mutation()"),
            schema + ".metadata",
        ),
    ).fetchall()
    if triggers != [
        ("immutable_row", 27, "A", False, True, "", None, 0, b"", 0, False, False, None, None),
        ("immutable_truncate", 34, "A", False, True, "", None, 0, b"", 0, False, False, None, None),
    ]:
        raise RootProofPreparationV2QualificationError(
            "immutable preparation metadata triggers required"
        )
    sources = _sources(deployment.provisioning, schema, deployment.component_roles)
    observed = conn.execute(
        "SELECT p.proname || '(' || pg_catalog.replace(pg_catalog.oidvectortypes(p.proargtypes),"
        "', ', ',') || ')' FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n "
        "ON n.oid=p.pronamespace WHERE n.nspname=%s",
        (schema,),
    ).fetchall()
    if {row[0] for row in observed} != set(sources):
        raise RootProofPreparationV2QualificationError(
            "exact preparation function inventory required"
        )
    for signature, (_declaration, _returns, source) in sources.items():
        function_oid = _function_oid(conn, schema, signature)
        row = conn.execute(
            "SELECT p.proowner,p.prosrc,p.prosecdef,p.proconfig,l.lanname,p.provolatile,p.proisstrict,"
            "p.proleakproof,p.proparallel,p.prokind,p.proretset,p.pronargdefaults,p.prosupport::oid,"
            "p.provariadic,p.proallargtypes,p.proargmodes,p.procost,p.prorows,"
            "pg_catalog.pg_get_function_arguments(p.oid),pg_catalog.pg_get_function_result(p.oid),"
            "p.probin,p.prosqlbody IS NULL "
            "FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_language l ON l.oid=p.prolang "
            "WHERE p.oid=%s::oid",
            (function_oid,),
        ).fetchone()
        if row != (
            roles[owner],
            source,
            True,
            ["search_path=pg_catalog"],
            "plpgsql",
            "v",
            False,
            False,
            "u",
            "f",
            False,
            0,
            0,
            0,
            None,
            None,
            100.0,
            0.0,
            _declaration,
            _returns,
            None,
            True,
        ):
            raise RootProofPreparationV2QualificationError(
                "reviewed preparation function source differs"
            )
        callers = (
            dict(deployment.component_roles).values()
            if signature == "qualify_preparation_v2()"
            else (dict(deployment.component_roles)[name] for name in ("intent", "retention"))
        )
        expected = {(roles[owner], "EXECUTE", False)} | {
            (roles[caller], "EXECUTE", False) for caller in callers
        }
        actual = set(
            conn.execute(
                "SELECT a.grantee,a.privilege_type,a.is_grantable FROM pg_catalog.pg_proc p "
                "CROSS JOIN LATERAL pg_catalog.aclexplode(p.proacl) a "
                "WHERE p.oid=%s::oid",
                (function_oid,),
            ).fetchall()
        )
        if actual != expected:
            raise RootProofPreparationV2QualificationError(
                "exact preparation function ACL required"
            )


def _fingerprint(
    conn: psycopg.Connection[Any], deployment: PostgreSQLRootProofPreparationV2Deployment
) -> str:
    # The private catalog reader can only add schemas to the original seven.
    # This single pass covers every component, both coordinator/preparation
    # schemas and global role ACLs without duplicating the component queries.
    catalog = _catalog_fingerprint(
        conn,
        deployment.provisioning,
        deployment.role_oids,
        additional_schemas=(deployment.preparation_schema,),
    )
    return hashlib.sha256(
        b"CryptoHunter/RootProofPreparationCatalog/V2\0" + catalog.encode("ascii")
    ).hexdigest()


def _qualify_preparation_roles(
    conn: psycopg.Connection[Any], deployment: PostgreSQLRootProofPreparationV2Deployment
) -> None:
    oids = [oid for _, oid in deployment.role_oids]
    if len(oids) != len(set(oids)):
        raise RootProofPreparationV2QualificationError("authority role identities alias")
    for name, oid in deployment.role_oids:
        row = conn.execute(
            "SELECT oid,rolcanlogin,rolinherit,rolsuper,rolcreaterole,rolcreatedb,"
            "rolreplication,rolbypassrls FROM pg_catalog.pg_roles WHERE rolname=%s",
            (name,),
        ).fetchone()
        if row != (
            oid,
            name not in deployment.owner_roles,
            False,
            False,
            False,
            False,
            False,
            False,
        ):
            raise RootProofPreparationV2QualificationError(
                "exact authority role OID and attributes required"
            )
    if (
        conn.execute(
            "SELECT 1 FROM pg_catalog.pg_auth_members WHERE roleid=ANY(%s) OR member=ANY(%s) LIMIT 1",
            (oids, oids),
        ).fetchone()
        is not None
    ):
        raise RootProofPreparationV2QualificationError("authority role membership forbidden")


def _principal_boundary(
    conn: psycopg.Connection[Any],
    config: PostgreSQLAtomicRootProofProvisioning,
    principal: str,
    preparation_schema: str,
) -> None:
    _qualify_closed_principal(conn, principal)
    if conn.execute(
        "SELECT pg_catalog.has_database_privilege(%s,pg_catalog.current_database(),'CREATE,TEMPORARY')",
        (principal,),
    ).fetchone() != (False,):
        raise RootProofPreparationV2QualificationError("preparation database DDL forbidden")
    if (
        conn.execute(
            "SELECT 1 FROM pg_catalog.pg_namespace WHERE pg_catalog.has_schema_privilege(%s,oid,'CREATE') LIMIT 1",
            (principal,),
        ).fetchone()
        is not None
    ):
        raise RootProofPreparationV2QualificationError("preparation schema DDL forbidden")
    if (
        conn.execute(
            "SELECT 1 FROM pg_catalog.pg_class WHERE relkind IN ('r','p','v','m','f') AND "
            "pg_catalog.has_table_privilege(%s,oid,'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER') LIMIT 1",
            (principal,),
        ).fetchone()
        is not None
    ):
        raise RootProofPreparationV2QualificationError("preparation direct table DML forbidden")
    if (
        conn.execute(
            "SELECT 1 FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n "
            "ON n.oid=p.pronamespace WHERE p.prosecdef AND n.nspname<>ALL(%s) AND "
            "pg_catalog.has_schema_privilege(%s,n.oid,'USAGE') AND "
            "pg_catalog.has_function_privilege(%s,p.oid,'EXECUTE') LIMIT 1",
            (_schemas(config) + [preparation_schema], principal, principal),
        ).fetchone()
        is not None
    ):
        raise RootProofPreparationV2QualificationError(
            "external preparation security-definer forbidden"
        )
    for schema, signature in (
        (config.schema, "commit_root_proof(bytea)"),
        (config.schema, "approve_commit(jsonb)"),
        (config.entitlement_schema, "append_bind(text,text,text,bigint,json)"),
        (config.entitlement_schema, "atomic_bind(text,text,text,bigint,json)"),
        (config.history_schema, "append_record(bytea,text,json,bytea)"),
        (config.history_schema, "retain_head(bytea,json,bytea,bytea)"),
        (config.history_schema, "atomic_append_record(bytea,text,json,bytea)"),
        (config.history_schema, "atomic_retain_head(bytea,json,bytea,bytea)"),
    ):
        if conn.execute(
            "SELECT pg_catalog.has_function_privilege(%s,%s::oid,'EXECUTE')",
            (principal, _function_oid(conn, schema, signature)),
        ).fetchone() != (False,):
            raise RootProofPreparationV2QualificationError(
                "preparation final issuance privilege forbidden"
            )


def qualify_root_proof_preparation_v2(
    conn: psycopg.Connection[Any],
    deployment: PostgreSQLRootProofPreparationV2Deployment,
    *,
    component: str,
) -> None:
    """Require the original component login and every reviewed live catalog pin."""
    validate_root_proof_preparation_v2_deployment(deployment)
    if component not in _COMPONENTS:
        raise RootProofPreparationV2QualificationError("exact preparation component required")
    try:
        conn.execute("SET LOCAL search_path=pg_catalog")
        principal = dict(deployment.component_roles)[component]
        if conn.execute("SELECT session_user::text,current_user::text").fetchone() != (
            principal,
            principal,
        ):
            raise RootProofPreparationV2QualificationError(
                "exact direct preparation principal required"
            )
        if conn.info.server_version // 10000 != 16:
            raise RootProofPreparationV2QualificationError("exact PostgreSQL 16 required")
        _qualify_preparation_roles(conn, deployment)
        _principal_boundary(conn, deployment.provisioning, principal, deployment.preparation_schema)
        _verify_preparation_sources(conn, deployment)
        if _fingerprint(conn, deployment) != deployment.catalog_sha256:
            raise RootProofPreparationV2QualificationError("live preparation catalog or ACL drift")
        row = conn.execute(
            sql.SQL("SELECT {}.qualify_preparation_v2()").format(
                sql.Identifier(deployment.preparation_schema)
            )
        ).fetchone()
        if row is None or type(row[0]) is not list or len(row[0]) != 7:
            raise RootProofPreparationV2QualificationError(
                "preparation physical identity unavailable"
            )
        raw = row[0]
        if raw[6] != {
            "singleton": True,
            "migration_id": MIGRATION_ID,
            "boundary": _BOUNDARY,
            "predecessor_catalog_sha256": deployment.predecessor_catalog_sha256,
        }:
            raise RootProofPreparationV2QualificationError(
                "exact durable preparation migration required"
            )
        physical = PostgreSQLAtomicRootProofPhysicalIdentity(
            raw[0],
            raw[1],
            raw[2],
            raw[3],
            conn.info.host,
            conn.info.port,
            raw[4],
            raw[5],
            deployment.provisioning.instance_id,
        )
        if physical != deployment.identity:
            raise RootProofPreparationV2QualificationError(
                "physical preparation authority mismatch"
            )
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
            "live preparation qualification refused"
        ) from exc


def lock_root_proof_preparation_signer_v2(
    conn: psycopg.Connection[Any],
    deployment: PostgreSQLRootProofPreparationV2Deployment,
    snapshot: SigningCredentialSnapshot,
) -> None:
    """Hold an exact enrolled ACTIVE fence until the preparation transaction ends."""
    validate_root_proof_preparation_v2_deployment(deployment)
    if (
        type(snapshot) is not SigningCredentialSnapshot
        or snapshot.lifecycle_state is not SigningKeyLifecycle.ACTIVE
    ):
        raise RootProofPreparationV2QualificationError("original ACTIVE signer snapshot required")
    if conn.info.transaction_status is not psycopg.pq.TransactionStatus.INTRANS:
        raise RootProofPreparationV2QualificationError("explicit signer fence transaction required")
    try:
        conn.execute(
            sql.SQL("SELECT {}.lock_active_signer_v2(%s)").format(
                sql.Identifier(deployment.preparation_schema)
            ),
            (_canonical(_snapshot_data(snapshot)),),
        )
    except (psycopg.Error, ValueError, TypeError) as exc:
        raise RootProofPreparationV2QualificationError(
            "exact enrolled ACTIVE signer fence refused"
        ) from exc

