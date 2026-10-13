"""Explicit V1 migration for an isolated PostgreSQL atomic issuer test authority.

This module never runs during application startup.  The only evidence ingress is
the separate test verifier principal; the commit login cannot supply evidence.
The protocol is deliberately unavailable to the public RootProofIssuer.  Its
database fences are test authorities, not proof of production custody fencing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg import sql

from bot_core.postgresql_atomic_root_proof_preaccount_sql import (
    PREACCOUNT_HELPER_DECLARATION,
    PREACCOUNT_HELPER_SIGNATURE,
    reviewed_preaccount_helper,
)
from bot_core.postgresql_entitlement_registry import (
    PostgreSQLRegistryProvisioning,
    _reviewed_function_sources,
)
from bot_core.postgresql_issuer_history_schema import (
    PostgreSQLIssuerHistoryProvisioning,
    _function_sources as _history_sources,
    _principal_guard as _history_guard,
)

SCHEMA_IDENTITY = "cryptohunter.atomic_root_proof_commit.postgresql"
SCHEMA_VERSION = 1
MIGRATION_ID = "ATOMIC_ROOT_PROOF_AUTHORITY_V1_COMPONENT_INVENTORY_V2"
BOUNDARY = "ISOLATED_TEST_ONLY"
PRODUCTION_LIFECYCLE_FENCE_AVAILABLE = False
_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]{0,62}\Z")


@dataclass(frozen=True, slots=True)
class PostgreSQLAtomicRootProofProvisioning:
    schema: str
    schema_owner_role: str
    issuer_writer_role: str
    test_admin_role: str
    entitlement_schema: str
    intent_schema: str
    retention_schema: str
    history_schema: str
    trust_domain: str
    instance_id: str
    requester_schema: str
    claimant_schema: str

    def __post_init__(self) -> None:
        for field in (
            "schema",
            "schema_owner_role",
            "issuer_writer_role",
            "test_admin_role",
            "entitlement_schema",
            "intent_schema",
            "retention_schema",
            "history_schema",
            "requester_schema",
            "claimant_schema",
        ):
            if _IDENTIFIER.fullmatch(getattr(self, field)) is None:
                raise ValueError("safe lowercase PostgreSQL identifiers required")
        if len({self.schema_owner_role, self.issuer_writer_role, self.test_admin_role}) != 3:
            raise ValueError("atomic owner, writer and test verifier must be distinct")
        if (
            len(
                {
                    self.schema,
                    self.entitlement_schema,
                    self.intent_schema,
                    self.retention_schema,
                    self.history_schema,
                    self.requester_schema,
                    self.claimant_schema,
                }
            )
            != 7
        ):
            raise ValueError("atomic and component schemas must be distinct")
        if any(type(v) is not str or not v.strip() for v in (self.trust_domain, self.instance_id)):
            raise ValueError("exact trust domain and isolated test instance required")


FUNC_RUNTIME = frozenset(
    (
        "commit_root_proof(bytea)",
        "read_decision(bytea,bytea)",
        "read_approved_identity(bytea)",
        "qualify_catalog()",
    )
)
FUNC_ADMIN = frozenset(
    (
        "approve_commit(jsonb)",
        "provision_fence(text,text,bigint)",
        "transition_fence(text,bigint,text,bigint)",
    )
)
FUNCTION_DECLARATIONS = {
    "catalog_snapshot()": ("", "jsonb"),
    "assert_catalog()": ("", "void"),
    "qualify_catalog()": ("", "boolean"),
    "reject_mutation()": ("", "trigger"),
    "validate_decision()": ("", "trigger"),
    "approve_commit(jsonb)": ("p_package jsonb", "void"),
    "provision_fence(text,text,bigint)": (
        "p_identity text,p_role text,p_generation bigint",
        "void",
    ),
    "transition_fence(text,bigint,text,bigint)": (
        "p_identity text,p_expected bigint,p_state text,p_generation bigint",
        "void",
    ),
    "commit_root_proof(bytea)": ("p_attempt bytea", "jsonb"),
    "read_decision(bytea,bytea)": ("p_scope bytea,p_attempt bytea", "jsonb"),
    "read_approved_identity(bytea)": ("p_attempt bytea", "jsonb"),
    "assert_bound_link(text,text,text,bigint,jsonb)": (
        "p_handle text,p_environment text,p_trust text,p_revision bigint,p_state jsonb",
        "boolean",
    ),
    "assert_history_link(bytea,bigint,bytea,bytea,bytea)": (
        "p_stream bytea,p_sequence bigint,p_record bytea,p_head bytea,p_signature bytea",
        "boolean",
    ),
}

PACKAGE_KEYS: tuple[str, ...] = (
    "account_hex",
    "attempt_hex",
    "bound_state",
    "entitlement_generation",
    "entitlement_hex",
    "environment",
    "expected_history_digest",
    "expected_revision",
    "history_fence_generation",
    "history_fence_key",
    "history_head",
    "history_head_hex",
    "history_record",
    "history_record_hex",
    "history_signature_hex",
    "lookup_handle",
    "operation_hex",
    "payload_hex",
    "reserved_envelope_hex",
    "root_fence_generation",
    "root_fence_key",
    "root_proof_hex",
    "scope_hex",
    "signature_hex",
    "staged_envelope_hex",
    "stream_hex",
    "trust_domain",
    "trust_fence_generation",
    "trust_fence_key",
    "requester_credential",
    "claimant_credential",
)
PACKAGE_KEYS = tuple(sorted(PACKAGE_KEYS))


def _q(value: str) -> str:
    return '"' + value + '"'


def _guard(
    config: PostgreSQLAtomicRootProofProvisioning, owner: str, *, admin: bool = False
) -> str:
    login = config.test_admin_role if admin else config.issuer_writer_role
    return (
        f"IF session_user <> '{login}' OR current_user <> '{owner}' THEN "
        "RAISE EXCEPTION 'wrong atomic authority principal' USING ERRCODE='42501'; END IF; "
        "IF pg_catalog.current_setting('server_version_num')::integer < 160000 "
        "OR pg_catalog.current_setting('server_version_num')::integer >= 170000 "
        "OR pg_catalog.current_setting('fsync') <> 'on' "
        "OR pg_catalog.current_setting('synchronous_commit') <> 'on' "
        "OR pg_catalog.current_setting('full_page_writes') <> 'on' THEN "
        "RAISE EXCEPTION 'atomic authority durability mismatch' USING ERRCODE='42501'; END IF; "
    )


def _metadata(conn: psycopg.Connection[Any], schema: str) -> dict[str, Any]:
    row = conn.execute(
        sql.SQL("SELECT pg_catalog.to_jsonb(m) FROM {}.metadata m WHERE singleton").format(
            sql.Identifier(schema)
        )
    ).fetchone()
    if row is None:
        raise ValueError("component metadata missing")
    result = dict(row[0])
    if "config" in result:
        result.update(result["config"])
    return result


def _catalog_snapshot_source(config: PostgreSQLAtomicRootProofProvisioning) -> str:
    """Stable security catalog shape, excluding statistics and process identity."""
    names = ",".join(
        "'" + name + "'"
        for name in (
            config.schema,
            config.entitlement_schema,
            config.intent_schema,
            config.retention_schema,
            config.history_schema,
            config.requester_schema,
            config.claimant_schema,
        )
    )
    roles = ",".join(
        "'" + name + "'"
        for name in (
            config.schema_owner_role,
            config.issuer_writer_role,
            config.test_admin_role,
        )
    )
    component_metadata = ",".join(
        f"{_q(schema)}.atomic_metadata()"
        for schema in (
            config.entitlement_schema,
            config.intent_schema,
            config.retention_schema,
            config.history_schema,
            config.requester_schema,
            config.claimant_schema,
        )
    )
    return f"""
DECLARE result jsonb;
BEGIN
 WITH namespaces AS (SELECT oid FROM pg_catalog.pg_namespace WHERE nspname IN ({names})),
 selected_roles AS (SELECT oid FROM pg_catalog.pg_roles WHERE rolname IN ({roles})
  OR oid IN (SELECT nspowner FROM pg_catalog.pg_namespace WHERE nspname IN ({names}))
  OR oid IN (SELECT DISTINCT a.grantee FROM pg_catalog.pg_class c
   CROSS JOIN LATERAL pg_catalog.aclexplode(c.relacl) a WHERE c.relnamespace IN (SELECT oid FROM namespaces))
  OR oid IN (SELECT DISTINCT a.grantee FROM pg_catalog.pg_proc p
   CROSS JOIN LATERAL pg_catalog.aclexplode(p.proacl) a WHERE p.pronamespace IN (SELECT oid FROM namespaces)))
 SELECT pg_catalog.jsonb_build_object(
 'metadata',pg_catalog.jsonb_build_array((SELECT pg_catalog.to_jsonb(m) FROM {_q(config.schema)}.metadata m
 WHERE singleton),{component_metadata}),
 'physical',(SELECT pg_catalog.jsonb_build_array(system_identifier::text,
 (SELECT oid FROM pg_catalog.pg_database WHERE datname=pg_catalog.current_database()))
 FROM pg_catalog.pg_control_system()),
 'schemas',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(n) ORDER BY n.oid)
 FROM pg_catalog.pg_namespace n WHERE n.oid IN (SELECT oid FROM namespaces)),
 'types',(SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.to_jsonb(t) ORDER BY t.oid),'[]'::jsonb)
 FROM pg_catalog.pg_type t WHERE t.typnamespace IN (SELECT oid FROM namespaces)),
 'roles',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r) ORDER BY r.oid)
 FROM pg_catalog.pg_roles r WHERE r.oid IN (SELECT oid FROM selected_roles)),
 'memberships',(SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.to_jsonb(m)
 ORDER BY m.roleid,m.member,m.grantor),'[]'::jsonb) FROM pg_catalog.pg_auth_members m
 WHERE m.roleid IN (SELECT oid FROM selected_roles) OR m.member IN (SELECT oid FROM selected_roles)),
 'functions',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(p) ORDER BY p.oid)
 FROM pg_catalog.pg_proc p WHERE p.pronamespace IN (SELECT oid FROM namespaces)),
 'relations',(SELECT pg_catalog.jsonb_agg(pg_catalog.jsonb_build_array(c.oid,c.relname,c.relnamespace,
 c.reltype,c.reloftype,c.relowner,c.relam,c.relpersistence,c.relkind,c.relnatts,c.relchecks,c.relhasrules,
 c.relhastriggers,c.relrowsecurity,c.relforcerowsecurity,c.relispopulated,c.relispartition,c.relacl,c.reloptions)
 ORDER BY c.oid) FROM pg_catalog.pg_class c WHERE c.relnamespace IN (SELECT oid FROM namespaces)),
 'attributes',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(a) ORDER BY a.attrelid,a.attnum)
 FROM pg_catalog.pg_attribute a JOIN pg_catalog.pg_class c ON c.oid=a.attrelid
 WHERE c.relnamespace IN (SELECT oid FROM namespaces)),
 'constraints',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(c) ORDER BY c.oid)
 FROM pg_catalog.pg_constraint c WHERE c.connamespace IN (SELECT oid FROM namespaces)),
 'defaults',(SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.to_jsonb(d) ORDER BY d.oid),'[]'::jsonb)
 FROM pg_catalog.pg_attrdef d JOIN pg_catalog.pg_class c ON c.oid=d.adrelid
 WHERE c.relnamespace IN (SELECT oid FROM namespaces)),
 'indexes',(SELECT pg_catalog.jsonb_agg(pg_catalog.jsonb_build_array(x.indexrelid,x.indrelid,
 x.indnatts,x.indnkeyatts,x.indisunique,x.indnullsnotdistinct,x.indisprimary,x.indisexclusion,
 x.indimmediate,x.indisclustered,x.indisvalid,x.indisready,x.indislive,x.indisreplident,x.indkey,
 x.indcollation,x.indclass,x.indoption,pg_catalog.pg_get_indexdef(x.indexrelid)) ORDER BY x.indexrelid)
 FROM pg_catalog.pg_index x JOIN pg_catalog.pg_class c ON c.oid=x.indrelid
 WHERE c.relnamespace IN (SELECT oid FROM namespaces)),
 'triggers',(SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.jsonb_build_array(t.oid,t.tgrelid,
 t.tgname,t.tgfoid,t.tgtype,t.tgenabled,t.tgisinternal,t.tgconstraint,t.tgdeferrable,t.tginitdeferred,
 pg_catalog.pg_get_triggerdef(t.oid)) ORDER BY t.oid),'[]'::jsonb)
 FROM pg_catalog.pg_trigger t JOIN pg_catalog.pg_class c ON c.oid=t.tgrelid
 WHERE c.relnamespace IN (SELECT oid FROM namespaces)),
 'rules',(SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.to_jsonb(w) ORDER BY w.oid),'[]'::jsonb)
 FROM pg_catalog.pg_rewrite w JOIN pg_catalog.pg_class c ON c.oid=w.ev_class
 WHERE c.relnamespace IN (SELECT oid FROM namespaces)),
 'policies',(SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.to_jsonb(p) ORDER BY p.oid),'[]'::jsonb)
 FROM pg_catalog.pg_policy p JOIN pg_catalog.pg_class c ON c.oid=p.polrelid
 WHERE c.relnamespace IN (SELECT oid FROM namespaces)),
 'default_acls',(SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.to_jsonb(a) ORDER BY a.oid),'[]'::jsonb)
 FROM pg_catalog.pg_default_acl a WHERE a.defaclnamespace IN (SELECT oid FROM namespaces)
 OR a.defaclrole IN (SELECT oid FROM selected_roles)
 OR EXISTS(SELECT 1 FROM pg_catalog.aclexplode(a.defaclacl) x
 WHERE x.grantee=0 OR x.grantee IN (SELECT oid FROM selected_roles))),
 'database_acl',(SELECT pg_catalog.jsonb_build_array(d.oid,d.datdba,d.datacl) FROM pg_catalog.pg_database d
 WHERE d.datname=pg_catalog.current_database()),
 'external_schema_acl',(SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.jsonb_build_array(n.oid,n.nspowner,
 a.grantee,a.grantor,a.privilege_type,a.is_grantable) ORDER BY n.oid,a.grantee,a.privilege_type),'[]'::jsonb)
 FROM pg_catalog.pg_namespace n CROSS JOIN LATERAL pg_catalog.aclexplode(n.nspacl) a
 WHERE n.oid NOT IN (SELECT oid FROM namespaces) AND (a.grantee=0 OR a.grantee IN (SELECT oid FROM selected_roles))),
 'external_relation_acl',(SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.jsonb_build_array(c.oid,c.relowner,
 a.grantee,a.grantor,a.privilege_type,a.is_grantable) ORDER BY c.oid,a.grantee,a.privilege_type),'[]'::jsonb)
 FROM pg_catalog.pg_class c CROSS JOIN LATERAL pg_catalog.aclexplode(c.relacl) a
 WHERE c.relnamespace NOT IN (SELECT oid FROM namespaces) AND (a.grantee=0 OR a.grantee IN (SELECT oid FROM selected_roles))),
 'external_column_acl',(SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.jsonb_build_array(c.oid,t.attnum,
 a.grantee,a.grantor,a.privilege_type,a.is_grantable) ORDER BY c.oid,t.attnum,a.grantee,a.privilege_type),'[]'::jsonb)
 FROM pg_catalog.pg_attribute t JOIN pg_catalog.pg_class c ON c.oid=t.attrelid
 CROSS JOIN LATERAL pg_catalog.aclexplode(t.attacl) a
 WHERE c.relnamespace NOT IN (SELECT oid FROM namespaces) AND (a.grantee=0 OR a.grantee IN (SELECT oid FROM selected_roles))),
 'external_function_acl',(SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.jsonb_build_array(p.oid,p.proowner,
 a.grantee,a.grantor,a.privilege_type,a.is_grantable) ORDER BY p.oid,a.grantee,a.privilege_type),'[]'::jsonb)
 FROM pg_catalog.pg_proc p CROSS JOIN LATERAL pg_catalog.aclexplode(p.proacl) a
 WHERE p.pronamespace NOT IN (SELECT oid FROM namespaces) AND (a.grantee=0 OR a.grantee IN (SELECT oid FROM selected_roles))),
 'external_type_acl',(SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.jsonb_build_array(t.oid,t.typowner,
 a.grantee,a.grantor,a.privilege_type,a.is_grantable) ORDER BY t.oid,a.grantee,a.privilege_type),'[]'::jsonb)
 FROM pg_catalog.pg_type t CROSS JOIN LATERAL pg_catalog.aclexplode(t.typacl) a
 WHERE t.typnamespace NOT IN (SELECT oid FROM namespaces) AND (a.grantee=0 OR a.grantee IN (SELECT oid FROM selected_roles))),
 'external_public_definers',(SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.to_jsonb(p) ORDER BY p.oid),'[]'::jsonb)
 FROM pg_catalog.pg_proc p WHERE p.pronamespace NOT IN (SELECT oid FROM namespaces) AND p.prosecdef
 AND pg_catalog.has_function_privilege('{config.issuer_writer_role}',p.oid,'EXECUTE')),
 'service_acl',pg_catalog.jsonb_build_object(
 'tablespaces',(SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.to_jsonb(t) ORDER BY t.oid),'[]'::jsonb)
 FROM pg_catalog.pg_tablespace t WHERE t.spcowner IN (SELECT oid FROM selected_roles)
 OR EXISTS(SELECT 1 FROM pg_catalog.aclexplode(t.spcacl) a WHERE a.grantee=0 OR a.grantee IN (SELECT oid FROM selected_roles))),
 'languages',(SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.to_jsonb(t) ORDER BY t.oid),'[]'::jsonb)
 FROM pg_catalog.pg_language t WHERE t.lanowner IN (SELECT oid FROM selected_roles)
 OR EXISTS(SELECT 1 FROM pg_catalog.aclexplode(t.lanacl) a WHERE a.grantee=0 OR a.grantee IN (SELECT oid FROM selected_roles))),
 'foreign_servers',(SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.to_jsonb(t) ORDER BY t.oid),'[]'::jsonb)
 FROM pg_catalog.pg_foreign_server t WHERE t.srvowner IN (SELECT oid FROM selected_roles)
 OR EXISTS(SELECT 1 FROM pg_catalog.aclexplode(t.srvacl) a WHERE a.grantee=0 OR a.grantee IN (SELECT oid FROM selected_roles))),
 'foreign_wrappers',(SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.to_jsonb(t) ORDER BY t.oid),'[]'::jsonb)
 FROM pg_catalog.pg_foreign_data_wrapper t WHERE t.fdwowner IN (SELECT oid FROM selected_roles)
 OR EXISTS(SELECT 1 FROM pg_catalog.aclexplode(t.fdwacl) a WHERE a.grantee=0 OR a.grantee IN (SELECT oid FROM selected_roles))),
 'large_objects',(SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.to_jsonb(t) ORDER BY t.oid),'[]'::jsonb)
 FROM pg_catalog.pg_largeobject_metadata t WHERE t.lomowner IN (SELECT oid FROM selected_roles)
 OR EXISTS(SELECT 1 FROM pg_catalog.aclexplode(t.lomacl) a WHERE a.grantee=0 OR a.grantee IN (SELECT oid FROM selected_roles))),
 'parameters',(SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.to_jsonb(t) ORDER BY t.oid),'[]'::jsonb)
 FROM pg_catalog.pg_parameter_acl t WHERE EXISTS(SELECT 1 FROM pg_catalog.aclexplode(t.paracl) a
 WHERE a.grantee=0 OR a.grantee IN (SELECT oid FROM selected_roles))),
 'owned',(SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.jsonb_build_array(d.classid,d.objid,d.objsubid,
 d.refobjid,d.deptype) ORDER BY d.classid,d.objid,d.objsubid),'[]'::jsonb)
 FROM pg_catalog.pg_shdepend d WHERE d.refclassid='pg_catalog.pg_authid'::regclass
 AND d.refobjid IN (SELECT oid FROM pg_catalog.pg_roles WHERE rolname IN
 ('{config.issuer_writer_role}','{config.test_admin_role}')) AND d.deptype='o'))) INTO result;
 RETURN result;
END
"""


def _component_inventory(
    conn: psycopg.Connection[Any],
    config: PostgreSQLAtomicRootProofProvisioning,
) -> dict[tuple[str, str], tuple[str, str, str, str]]:
    """Reviewed owner helpers; no input-supplied names are evaluated by SQL."""
    result: dict[tuple[str, str], tuple[str, str, str, str]] = {}
    registry = _metadata(conn, config.entitlement_schema)
    owner = registry["schema_owner_role"]
    registry_config = PostgreSQLRegistryProvisioning(
        config.entitlement_schema,
        owner,
        registry["runtime_role"],
        registry["admin_role"],
        config.trust_domain,
    )
    bind = _reviewed_function_sources(registry_config)["append_bind(text,text,text,bigint,json)"]
    # The source is the reviewed CAS transition, not an invocation of an exposed
    # standalone writer.  Inventory V2 owns this additional principal guard.
    bind = bind.replace("BEGIN", "BEGIN\n " + _guard(config, owner), 1)
    result[(config.entitlement_schema, "atomic_bind(text,text,text,bigint,json)")] = (
        owner,
        bind,
        "p_handle text,p_environment text,p_trust text,p_expected bigint,p_state json",
        "void",
    )
    for schema, kind in ((config.intent_schema, "intent"), (config.retention_schema, "retention")):
        meta = _metadata(conn, schema)
        owner = meta["schema_owner_role"]
        table = "intents" if kind == "intent" else "prepared"
        signature = "atomic_read_" + kind + "(bytea,bytea)"
        source = f"""
DECLARE r {_q(schema)}.{table}%ROWTYPE;
BEGIN
 {_guard(config, owner)}
 SELECT * INTO r FROM {_q(schema)}.{table} WHERE scope_bytes=p_scope AND attempt_key=p_attempt;
 IF NOT FOUND THEN RAISE EXCEPTION 'missing exact retained {kind}' USING ERRCODE='23514'; END IF;
 RETURN pg_catalog.to_jsonb(r);
END
"""
        result[(schema, signature)] = (owner, source, "p_scope bytea,p_attempt bytea", "jsonb")
    meta = _metadata(conn, config.history_schema)
    owner = meta["schema_owner_role"]
    history_config = PostgreSQLIssuerHistoryProvisioning(
        config.history_schema,
        owner,
        meta["runtime_role"],
        meta["admin_role"],
        config.trust_domain,
    )
    sources = _history_sources(history_config)
    old_guard = _history_guard(login_role=meta["runtime_role"], schema_owner_role=owner)
    for old, new, declaration in (
        (
            "append_record(bytea,text,json,bytea)",
            "atomic_append_record(bytea,text,json,bytea)",
            "p_stream bytea,p_expected text,p_record json,p_canonical bytea",
        ),
        (
            "retain_head(bytea,json,bytea,bytea)",
            "atomic_retain_head(bytea,json,bytea,bytea)",
            "p_stream bytea,p_head json,p_canonical bytea,p_signature bytea",
        ),
    ):
        result[(config.history_schema, new)] = (
            owner,
            sources[old].replace(old_guard, _guard(config, owner)),
            declaration,
            "boolean",
        )
    # Deferred constraints inspect the marker through EXECUTE, never cross-owner DML.
    s = _q(config.schema)
    e = _q(config.entitlement_schema)
    source = f"""
BEGIN
 IF NEW.state #>> '{{binding,kind}}'='BOUND' THEN
  IF NEW.state->>'lifecycle'='ACTIVE' THEN
   IF NOT {s}.assert_bound_link(NEW.lookup_handle,NEW.environment,NEW.trust_domain,NEW.revision,NEW.state)
   THEN RAISE EXCEPTION 'BOUND requires exact atomic decision linkage' USING ERRCODE='23514'; END IF;
  ELSIF NOT EXISTS(SELECT 1 FROM {e}.history previous_state WHERE previous_state.lookup_handle=NEW.lookup_handle
   AND previous_state.environment=NEW.environment AND previous_state.trust_domain=NEW.trust_domain
   AND previous_state.revision=NEW.predecessor_revision AND previous_state.state->'binding'=NEW.state->'binding'
   AND previous_state.state->'identity'=NEW.state->'identity' AND previous_state.state->'subject'=NEW.state->'subject'
   AND previous_state.state->'provenance'=NEW.state->'provenance') THEN
   RAISE EXCEPTION 'lifecycle transition must preserve original BOUND' USING ERRCODE='23514';
  END IF;
 END IF;
 RETURN NEW;
END
"""
    result[(config.entitlement_schema, "atomic_bound_constraint()")] = (
        registry["schema_owner_role"],
        source,
        "",
        "trigger",
    )
    result[(config.entitlement_schema, "atomic_confirm_bound(text,text,text,bigint,jsonb)")] = (
        registry["schema_owner_role"],
        f"""
BEGIN
 {_guard(config, registry["schema_owner_role"])}
 RETURN EXISTS(SELECT 1 FROM {e}.history z JOIN {e}.lineages l
 USING(lookup_handle,environment,trust_domain) WHERE z.lookup_handle=p_handle
 AND z.environment=p_environment AND z.trust_domain=p_trust AND z.revision=p_revision
 AND z.state=p_state AND l.current_revision>=p_revision);
END
""",
        "p_handle text,p_environment text,p_trust text,p_revision bigint,p_state jsonb",
        "boolean",
    )
    source = f"""
BEGIN
 IF TG_TABLE_NAME='records' THEN
  IF NOT {s}.assert_history_link(NEW.stream_identity,NEW.sequence,NEW.canonical_record,NULL,NULL)
  THEN RAISE EXCEPTION 'history record requires atomic decision' USING ERRCODE='23514'; END IF;
 ELSE
  IF NOT {s}.assert_history_link(NEW.stream_identity,NEW.sequence,NULL,NEW.canonical_head,NEW.signature)
  THEN RAISE EXCEPTION 'history head requires atomic decision' USING ERRCODE='23514'; END IF;
 END IF;
 RETURN NEW;
END
"""
    result[(config.history_schema, "atomic_history_constraint()")] = (owner, source, "", "trigger")
    result[(config.history_schema, "atomic_confirm_history(bytea,bigint,bytea,bytea,bytea)")] = (
        owner,
        f"""
BEGIN
 {_guard(config, owner)}
 PERFORM {_q(config.history_schema)}.verify_stream(p_stream);
 RETURN EXISTS(SELECT 1 FROM {_q(config.history_schema)}.records z
 JOIN {_q(config.history_schema)}.attestations a USING(stream_identity,sequence)
 WHERE z.stream_identity=p_stream AND z.sequence=p_sequence AND z.canonical_record=p_record
 AND a.canonical_head=p_head AND a.signature=p_signature);
END
""",
        "p_stream bytea,p_sequence bigint,p_record bytea,p_head bytea,p_signature bytea",
        "boolean",
    )
    for schema, peer, semantic_role in (
        (config.requester_schema, config.claimant_schema, "ROOT_PROOF_REQUESTER"),
        (config.claimant_schema, config.requester_schema, "ROOT_PROOF_CLAIMANT"),
    ):
        owner = _metadata(conn, schema)["schema_owner_role"]
        result[(schema, PREACCOUNT_HELPER_SIGNATURE)] = (
            owner,
            reviewed_preaccount_helper(
                owner=owner,
                writer=config.issuer_writer_role,
                schema=schema,
                peer_schema=peer,
                semantic_role=semantic_role,
                trust_domain=config.trust_domain,
            ),
            PREACCOUNT_HELPER_DECLARATION,
            "void",
        )
    # Preserve all original canonical validators and qualification inventories.
    return result


def _function_sources(config: PostgreSQLAtomicRootProofProvisioning) -> dict[str, str]:
    s = _q(config.schema)
    e = _q(config.entitlement_schema)
    i = _q(config.intent_schema)
    r = _q(config.retention_schema)
    h = _q(config.history_schema)
    guard = _guard(config, config.schema_owner_role)
    admin_guard = _guard(config, config.schema_owner_role, admin=True)
    qualification_guard = guard.replace(
        f"session_user <> '{config.issuer_writer_role}'",
        f"session_user NOT IN ('{config.issuer_writer_role}','{config.test_admin_role}')",
    )
    fields = "ARRAY[" + ",".join("'" + key + "'" for key in PACKAGE_KEYS) + "]::text[]"
    credential_calls = "\n".join(
        f" PERFORM {_q(schema)}.atomic_lock_credential(p->'{key}');"
        for schema, key in sorted(
            (
                (config.requester_schema, "requester_credential"),
                (config.claimant_schema, "claimant_credential"),
            )
        )
    )
    credential_namespace = "cryptohunter.preaccount.credentials:" + ":".join(
        sorted(
            (
                config.requester_schema,
                config.claimant_schema,
            )
        )
    )
    sources = {
        "catalog_snapshot()": _catalog_snapshot_source(config),
        "assert_catalog()": f"""
DECLARE expected jsonb;
BEGIN
 SELECT snapshot INTO STRICT expected FROM {s}.catalog_manifest WHERE singleton;
 IF {s}.catalog_snapshot() IS DISTINCT FROM expected THEN
  RAISE EXCEPTION 'atomic catalog, role or ACL qualification drift' USING ERRCODE='42501'; END IF;
END
""",
        "qualify_catalog()": f"""
BEGIN
 {qualification_guard}
 PERFORM {s}.assert_catalog();
 RETURN true;
END
""",
        "reject_mutation()": "BEGIN RAISE EXCEPTION 'immutable atomic evidence' USING ERRCODE='42501'; END",
        "provision_fence(text,text,bigint)": f"""
BEGIN
 {admin_guard}
 IF p_identity IS NULL OR p_identity='' OR p_generation IS NULL OR p_generation<1
 OR p_role NOT IN ('ROOT_PROOF_SIGNING','HISTORY_ATTESTATION_SIGNING','TRUST_ACTIVATION')
 THEN RAISE EXCEPTION 'malformed test fence' USING ERRCODE='22023'; END IF;
 INSERT INTO {s}.fences VALUES(p_identity,p_role,p_generation,'PENDING_CLOSED',false);
END
""",
        "transition_fence(text,bigint,text,bigint)": f"""
DECLARE f {s}.fences%ROWTYPE;
BEGIN
 {admin_guard}
 SELECT * INTO f FROM {s}.fences WHERE identity=p_identity FOR UPDATE;
 IF NOT FOUND OR f.generation IS DISTINCT FROM p_expected OR f.state='REVOKED'
 OR p_state IS NULL OR p_generation IS NULL OR p_generation<f.generation
 OR p_state NOT IN ('PENDING_CLOSED','ACTIVE','VERIFY_ONLY','REVOKED')
 THEN RAISE EXCEPTION 'invalid lifecycle fence transition' USING ERRCODE='23514'; END IF;
 IF p_state='ACTIVE' AND (f.state<>'PENDING_CLOSED'
 OR (f.ever_active AND p_generation<=f.generation)) THEN
  RAISE EXCEPTION 'activation requires closed rotation with new generation' USING ERRCODE='23514'; END IF;
 IF p_state IN ('VERIFY_ONLY','REVOKED','PENDING_CLOSED') AND p_generation<>f.generation THEN
  RAISE EXCEPTION 'closing transition generation differs' USING ERRCODE='23514'; END IF;
 UPDATE {s}.fences SET state=p_state,generation=p_generation,
 ever_active=(ever_active OR p_state='ACTIVE') WHERE identity=p_identity;
END
""",
        "approve_commit(jsonb)": f"""
DECLARE prior jsonb; key text; attempt bytea;
BEGIN
 {admin_guard}
 IF p_package IS NULL OR pg_catalog.jsonb_typeof(p_package)<>'object'
 OR (SELECT pg_catalog.array_agg(k ORDER BY k) FROM pg_catalog.jsonb_object_keys(p_package) k)
 IS DISTINCT FROM {fields}
 THEN RAISE EXCEPTION 'exact qualified test package required' USING ERRCODE='22023'; END IF;
 FOREACH key IN ARRAY ARRAY['scope_hex','attempt_hex','operation_hex','entitlement_hex',
 'root_proof_hex','account_hex','reserved_envelope_hex','staged_envelope_hex','payload_hex',
 'signature_hex','stream_hex','history_record_hex','history_head_hex','history_signature_hex'] LOOP
  IF pg_catalog.jsonb_typeof(p_package->key)<>'string'
  OR (p_package->>key) !~ '^([0-9a-f][0-9a-f])+$' THEN
   RAISE EXCEPTION 'exact retained test bytes required' USING ERRCODE='22023'; END IF;
 END LOOP;
 IF p_package->>'trust_domain' IS DISTINCT FROM '{config.trust_domain.replace("'", "''")}'
 OR p_package->>'environment'<>'PRODUCTION'
 OR pg_catalog.octet_length(pg_catalog.decode(p_package->>'signature_hex','hex'))<>64
 OR pg_catalog.octet_length(pg_catalog.decode(p_package->>'history_signature_hex','hex'))<>64
 THEN RAISE EXCEPTION 'test package scope/signature mismatch' USING ERRCODE='22023'; END IF;
 attempt := pg_catalog.decode(p_package->>'attempt_hex','hex');
 INSERT INTO {s}.approved_commits VALUES(attempt,p_package) ON CONFLICT DO NOTHING;
 SELECT package INTO STRICT prior FROM {s}.approved_commits WHERE attempt_key=attempt;
 IF prior IS DISTINCT FROM p_package THEN
  RAISE EXCEPTION 'conflicting qualified test evidence' USING ERRCODE='23505'; END IF;
END
""",
        "read_decision(bytea,bytea)": f"""
DECLARE d {s}.decisions%ROWTYPE;
BEGIN
 {guard}
 PERFORM {s}.assert_catalog();
 SELECT * INTO d FROM {s}.decisions WHERE attempt_key=p_attempt;
 IF NOT FOUND THEN RETURN NULL; END IF;
 IF d.scope_bytes IS DISTINCT FROM p_scope THEN
  RAISE EXCEPTION 'decision scope differs' USING ERRCODE='23514'; END IF;
 RETURN d.package || pg_catalog.jsonb_build_object('decision_id',d.decision_id,
 'history_sequence',d.history_sequence,'registry_revision',d.registry_revision);
END
""",
        "read_approved_identity(bytea)": f"""
DECLARE p jsonb;
BEGIN
 {guard}
 PERFORM {s}.assert_catalog();
 SELECT package INTO p FROM {s}.approved_commits WHERE attempt_key=p_attempt;
 IF NOT FOUND THEN RAISE EXCEPTION 'qualified approval missing' USING ERRCODE='42501'; END IF;
 RETURN pg_catalog.jsonb_build_object('scope_hex',p->>'scope_hex','attempt_hex',p->>'attempt_hex',
 'operation_hex',p->>'operation_hex','entitlement_hex',p->>'entitlement_hex',
 'root_proof_hex',p->>'root_proof_hex','account_hex',p->>'account_hex');
END
""",
        "assert_bound_link(text,text,text,bigint,jsonb)": f"""
BEGIN
 PERFORM {s}.assert_catalog();
 RETURN EXISTS(SELECT 1 FROM {s}.decisions d WHERE d.lookup_handle=p_handle
 AND d.environment=p_environment AND d.trust_domain=p_trust AND d.registry_revision=p_revision
 AND d.bound_state=p_state);
END
""",
        "assert_history_link(bytea,bigint,bytea,bytea,bytea)": f"""
BEGIN
 PERFORM {s}.assert_catalog();
 RETURN EXISTS(SELECT 1 FROM {s}.decisions d WHERE d.stream_identity=p_stream
 AND d.history_sequence=p_sequence AND (p_record IS NULL OR d.canonical_record=p_record)
 AND (p_head IS NULL OR d.canonical_head=p_head)
 AND (p_signature IS NULL OR d.history_signature=p_signature));
END
""",
        "validate_decision()": f"""
DECLARE p jsonb; intent jsonb; staged jsonb;
BEGIN
 {guard}
 PERFORM {s}.assert_catalog();
 SELECT package INTO p FROM {s}.approved_commits WHERE attempt_key=NEW.attempt_key;
 IF NOT FOUND OR NEW.package IS DISTINCT FROM p
 OR NEW.scope_bytes IS DISTINCT FROM pg_catalog.decode(p->>'scope_hex','hex')
 OR NEW.attempt_key IS DISTINCT FROM pg_catalog.decode(p->>'attempt_hex','hex')
 OR NEW.operation_key IS DISTINCT FROM pg_catalog.decode(p->>'operation_hex','hex')
 OR NEW.entitlement_key IS DISTINCT FROM pg_catalog.decode(p->>'entitlement_hex','hex')
 OR NEW.root_proof_key IS DISTINCT FROM pg_catalog.decode(p->>'root_proof_hex','hex')
 OR NEW.account_key IS DISTINCT FROM pg_catalog.decode(p->>'account_hex','hex')
 OR NEW.lookup_handle IS DISTINCT FROM p->>'lookup_handle'
 OR NEW.environment IS DISTINCT FROM p->>'environment' OR NEW.trust_domain IS DISTINCT FROM p->>'trust_domain'
 OR NEW.entitlement_generation IS DISTINCT FROM (p->>'entitlement_generation')::bigint
 OR NEW.registry_revision IS DISTINCT FROM (p->>'expected_revision')::bigint+1
 OR NEW.reserved_envelope IS DISTINCT FROM pg_catalog.decode(p->>'reserved_envelope_hex','hex')
 OR NEW.staged_envelope IS DISTINCT FROM pg_catalog.decode(p->>'staged_envelope_hex','hex')
 OR NEW.canonical_payload IS DISTINCT FROM pg_catalog.decode(p->>'payload_hex','hex')
 OR NEW.signature IS DISTINCT FROM pg_catalog.decode(p->>'signature_hex','hex')
 OR NEW.stream_identity IS DISTINCT FROM pg_catalog.decode(p->>'stream_hex','hex')
 OR NEW.history_sequence IS DISTINCT FROM (p->'history_record'->>'sequence')::bigint
 OR NEW.canonical_record IS DISTINCT FROM pg_catalog.decode(p->>'history_record_hex','hex')
 OR NEW.canonical_head IS DISTINCT FROM pg_catalog.decode(p->>'history_head_hex','hex')
 OR NEW.history_signature IS DISTINCT FROM pg_catalog.decode(p->>'history_signature_hex','hex')
 OR NEW.bound_state IS DISTINCT FROM p->'bound_state'
 OR NEW.decision_id IS DISTINCT FROM 'adc_' || pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
 'CryptoHunter/AtomicRootProofDecision/v1','UTF8') || NEW.attempt_key || NEW.operation_key || NEW.entitlement_key),'hex')
 THEN RAISE EXCEPTION 'decision must retain exact approved identity and original bytes' USING ERRCODE='23514'; END IF;
 intent:={i}.atomic_read_intent(NEW.scope_bytes,NEW.attempt_key);
 staged:={r}.atomic_read_retention(NEW.scope_bytes,NEW.attempt_key);
 IF pg_catalog.decode(pg_catalog.substring(intent->>'canonical_envelope',3),'hex') IS DISTINCT FROM NEW.reserved_envelope
 OR pg_catalog.decode(pg_catalog.substring(staged->>'canonical_envelope',3),'hex') IS DISTINCT FROM NEW.staged_envelope
 OR pg_catalog.decode(pg_catalog.substring(intent->>'canonical_payload',3),'hex') IS DISTINCT FROM NEW.canonical_payload
 OR pg_catalog.decode(pg_catalog.substring(staged->>'canonical_payload',3),'hex') IS DISTINCT FROM NEW.canonical_payload
 OR pg_catalog.decode(pg_catalog.substring(staged->>'signature',3),'hex') IS DISTINCT FROM NEW.signature
 OR NOT {e}.atomic_confirm_bound(NEW.lookup_handle,NEW.environment,NEW.trust_domain,NEW.registry_revision,NEW.bound_state)
 OR NOT {h}.atomic_confirm_history(NEW.stream_identity,NEW.history_sequence,NEW.canonical_record,NEW.canonical_head,NEW.history_signature)
 THEN RAISE EXCEPTION 'decision requires complete exact atomic write set' USING ERRCODE='23514'; END IF;
 RETURN NEW;
END
""",
        "commit_root_proof(bytea)": f"""
DECLARE p jsonb; existing {s}.decisions%ROWTYPE; intent jsonb; staged jsonb;
 fence {s}.fences%ROWTYPE; fence_key text; role_name text; expected_generation bigint;
 scope bytea; operation bytea; entitlement bytea; root_id bytea; account bytea;
 payload bytea; signature bytea; stream bytea; record_bytes bytea; head_bytes bytea;
 head_signature bytea; reserved_bytes bytea; staged_bytes bytea; field_name text;
 event_identity bytea; event_payload bytea; root_snapshot bytea; authorization_bytes bytea;
 credential_identity bytea; retained_request bytea;
BEGIN
 {guard}
 PERFORM {s}.assert_catalog();
 IF pg_catalog.current_setting('transaction_isolation')<>'serializable'
 OR pg_catalog.current_setting('transaction_read_only')<>'off' THEN
  RAISE EXCEPTION 'one writable SERIALIZABLE transaction required' USING ERRCODE='25001'; END IF;
 SELECT package INTO p FROM {s}.approved_commits WHERE attempt_key=p_attempt;
 IF NOT FOUND THEN RAISE EXCEPTION 'qualified test verifier approval required' USING ERRCODE='42501'; END IF;
 SELECT * INTO existing FROM {s}.decisions WHERE attempt_key=p_attempt;
 IF FOUND THEN
  IF existing.package IS DISTINCT FROM p THEN
   RAISE EXCEPTION 'immutable decision package differs' USING ERRCODE='23514'; END IF;
  RETURN existing.package || pg_catalog.jsonb_build_object('decision_id',existing.decision_id,
  'history_sequence',existing.history_sequence,'registry_revision',existing.registry_revision);
 END IF;
 scope:=pg_catalog.decode(p->>'scope_hex','hex'); operation:=pg_catalog.decode(p->>'operation_hex','hex');
 entitlement:=pg_catalog.decode(p->>'entitlement_hex','hex'); root_id:=pg_catalog.decode(p->>'root_proof_hex','hex');
 account:=pg_catalog.decode(p->>'account_hex','hex'); payload:=pg_catalog.decode(p->>'payload_hex','hex');
 signature:=pg_catalog.decode(p->>'signature_hex','hex'); stream:=pg_catalog.decode(p->>'stream_hex','hex');
 reserved_bytes:=pg_catalog.decode(p->>'reserved_envelope_hex','hex');
 staged_bytes:=pg_catalog.decode(p->>'staged_envelope_hex','hex');
 record_bytes:=pg_catalog.decode(p->>'history_record_hex','hex');
 head_bytes:=pg_catalog.decode(p->>'history_head_hex','hex');
 head_signature:=pg_catalog.decode(p->>'history_signature_hex','hex');
 -- All lifecycle and administration use the same sorted row lock order.
 IF (SELECT pg_catalog.count(DISTINCT k) FROM pg_catalog.unnest(ARRAY[
 p->>'root_fence_key',p->>'history_fence_key',p->>'trust_fence_key']) k)<>3 THEN
  RAISE EXCEPTION 'separate qualified signer/trust fences required' USING ERRCODE='23514'; END IF;
 FOR fence_key IN SELECT k FROM pg_catalog.unnest(ARRAY[p->>'root_fence_key',
 p->>'history_fence_key',p->>'trust_fence_key']) k ORDER BY k LOOP
  SELECT * INTO fence FROM {s}.fences WHERE identity=fence_key FOR UPDATE;
  IF fence_key=p->>'root_fence_key' THEN
   role_name:='ROOT_PROOF_SIGNING'; expected_generation:=(p->>'root_fence_generation')::bigint;
  ELSIF fence_key=p->>'history_fence_key' THEN
   role_name:='HISTORY_ATTESTATION_SIGNING'; expected_generation:=(p->>'history_fence_generation')::bigint;
  ELSE role_name:='TRUST_ACTIVATION'; expected_generation:=(p->>'trust_fence_generation')::bigint; END IF;
  IF NOT FOUND OR fence.semantic_role IS DISTINCT FROM role_name OR fence.state<>'ACTIVE'
  OR fence.generation IS DISTINCT FROM expected_generation THEN
   RAISE EXCEPTION 'DB-authoritative lifecycle fence closed' USING ERRCODE='23514'; END IF;
 END LOOP;
 PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended('{credential_namespace}',0));
 {credential_calls}
 intent:={i}.atomic_read_intent(scope,p_attempt); staged:={r}.atomic_read_retention(scope,p_attempt);
 FOREACH field_name IN ARRAY ARRAY['scope_bytes','attempt_key','operation_key','entitlement_key',
 'root_proof_key','account_key','canonical_payload'] LOOP
  IF intent->field_name IS DISTINCT FROM staged->field_name THEN
   RAISE EXCEPTION 'RESERVED/STAGED identity or bytes differ' USING ERRCODE='23514'; END IF;
 END LOOP;
 IF intent->>'status'<>'RESERVED' OR staged->>'status'<>'STAGED'
 OR pg_catalog.decode(pg_catalog.substring(intent->>'scope_bytes',3),'hex') IS DISTINCT FROM scope
 OR pg_catalog.decode(pg_catalog.substring(intent->>'attempt_key',3),'hex') IS DISTINCT FROM p_attempt
 OR pg_catalog.decode(pg_catalog.substring(intent->>'operation_key',3),'hex') IS DISTINCT FROM operation
 OR pg_catalog.decode(pg_catalog.substring(intent->>'entitlement_key',3),'hex') IS DISTINCT FROM entitlement
 OR pg_catalog.decode(pg_catalog.substring(intent->>'root_proof_key',3),'hex') IS DISTINCT FROM root_id
 OR pg_catalog.decode(pg_catalog.substring(intent->>'account_key',3),'hex') IS DISTINCT FROM account
 OR pg_catalog.decode(pg_catalog.substring(intent->>'canonical_payload',3),'hex') IS DISTINCT FROM payload
 OR pg_catalog.decode(pg_catalog.substring(intent->>'canonical_envelope',3),'hex') IS DISTINCT FROM reserved_bytes
 OR pg_catalog.decode(pg_catalog.substring(staged->>'canonical_envelope',3),'hex') IS DISTINCT FROM staged_bytes
 OR pg_catalog.decode(pg_catalog.substring(staged->>'signature',3),'hex') IS DISTINCT FROM signature THEN
  RAISE EXCEPTION 'approval differs from immutable original proof bytes' USING ERRCODE='23514'; END IF;
 -- Revalidate identity using the reviewed canonical BYTEA parser in the source owner.
 FOREACH field_name IN ARRAY ARRAY['account_id','bootstrap_entitlement_id','entitlement_generation',
 'issuance_attempt_id','logical_operation_id','root_proof_id','scope','authorization','retained_request',
 'credential_snapshot'] LOOP
  IF {i}.atomic_intent_member(reserved_bytes,field_name) IS DISTINCT FROM
  {r}.atomic_retention_member(staged_bytes,field_name) THEN
   RAISE EXCEPTION 'exact retained envelope relation differs' USING ERRCODE='23514'; END IF;
 END LOOP;
 root_snapshot:={i}.atomic_intent_member(reserved_bytes,'credential_snapshot');
 authorization_bytes:={i}.atomic_intent_member(reserved_bytes,'authorization');
 credential_identity:={i}.atomic_intent_member(root_snapshot,'credential_identity');
 retained_request:={i}.atomic_intent_member(reserved_bytes,'retained_request');
 IF {i}.atomic_intent_member(authorization_bytes,'requester_principal_id') IS DISTINCT FROM
 pg_catalog.convert_to(pg_catalog.to_json(p->'requester_credential'->>'principal_id')::text,'UTF8')
 OR {i}.atomic_intent_member(authorization_bytes,'requester_key_id') IS DISTINCT FROM
 pg_catalog.convert_to(pg_catalog.to_json(p->'requester_credential'->>'key_id')::text,'UTF8')
 OR {i}.atomic_intent_member(authorization_bytes,'requester_key_version') IS DISTINCT FROM
 pg_catalog.convert_to(p->'requester_credential'->>'key_version','UTF8')
 OR {i}.atomic_intent_member(authorization_bytes,'provisioning_principal_id') IS DISTINCT FROM
 pg_catalog.convert_to(pg_catalog.to_json(p->'claimant_credential'->>'principal_id')::text,'UTF8')
 OR {i}.atomic_intent_member(authorization_bytes,'claimant_key_id') IS DISTINCT FROM
 pg_catalog.convert_to(pg_catalog.to_json(p->'claimant_credential'->>'key_id')::text,'UTF8')
 OR {i}.atomic_intent_member(authorization_bytes,'claimant_key_version') IS DISTINCT FROM
 pg_catalog.convert_to(p->'claimant_credential'->>'key_version','UTF8') THEN
  RAISE EXCEPTION 'locked request credentials differ from original authorization' USING ERRCODE='23514'; END IF;
 IF {i}.atomic_intent_member(root_snapshot,'lifecycle_generation') IS DISTINCT FROM
 pg_catalog.convert_to(p->>'root_fence_generation','UTF8')
 OR {i}.atomic_intent_member(root_snapshot,'lifecycle_state') IS DISTINCT FROM
 pg_catalog.convert_to('"ACTIVE"','UTF8')
 OR {i}.atomic_intent_member(root_snapshot,'key_version') IS DISTINCT FROM
 {i}.atomic_intent_member(payload,'issuer_key_version')
 OR {i}.atomic_intent_member(credential_identity,'credential_identity') IS DISTINCT FROM
 {i}.atomic_intent_member(payload,'issuer_key_id') THEN
  RAISE EXCEPTION 'root signer generation differs from locked fence' USING ERRCODE='23514'; END IF;
 IF (p->'bound_state')#>>'{{binding,root_proof_id}}' IS DISTINCT FROM
 pg_catalog.convert_from(root_id,'UTF8')::jsonb#>>'{{}}'
 OR (p->'bound_state')#>>'{{binding,account_id}}' IS DISTINCT FROM
 pg_catalog.convert_from(account,'UTF8')::jsonb#>>'{{}}'
 OR ((p->'bound_state')#>>'{{identity,entitlement_generation}}')::bigint IS DISTINCT FROM
 (p->>'entitlement_generation')::bigint
 OR (p->'bound_state')#>>'{{identity,bootstrap_entitlement_id}}' IS DISTINCT FROM
 pg_catalog.convert_from({i}.atomic_intent_member(payload,'bootstrap_entitlement_id'),'UTF8')::jsonb#>>'{{}}'
 OR (p->'bound_state')#>>'{{binding,logical_operation_id}}' IS DISTINCT FROM
 pg_catalog.convert_from(operation,'UTF8')::jsonb#>>'{{}}' THEN
  RAISE EXCEPTION 'BOUND identity differs from exact retained proof' USING ERRCODE='23514'; END IF;
 FOREACH field_name IN ARRAY ARRAY['requester_principal_id','requester_key_id','requester_key_version',
 'provisioning_principal_id','claimant_key_id','claimant_key_version'] LOOP
  IF p->'bound_state'->'binding'->field_name IS DISTINCT FROM
  pg_catalog.convert_from({i}.atomic_intent_member(authorization_bytes,field_name),'UTF8')::jsonb THEN
   RAISE EXCEPTION 'BOUND request credentials differ from retained authorization' USING ERRCODE='23514'; END IF;
 END LOOP;
 IF p->'bound_state'->'binding'->'issuance_attempt_id' IS DISTINCT FROM
 pg_catalog.convert_from(p_attempt,'UTF8')::jsonb
 OR p->'bound_state'->'binding'->'issuer_signing_credential_id' IS DISTINCT FROM
 pg_catalog.convert_from({i}.atomic_intent_member(payload,'issuer_key_id'),'UTF8')::jsonb
 OR p->'bound_state'->'binding'->'issuer_signing_key_version' IS DISTINCT FROM
 pg_catalog.convert_from({i}.atomic_intent_member(payload,'issuer_key_version'),'UTF8')::jsonb
 OR p->'bound_state'->'binding'->'canonical_genesis_request_fingerprint_sha256' IS DISTINCT FROM
 pg_catalog.convert_from({i}.atomic_intent_member(payload,'canonical_genesis_request_fingerprint_sha256'),'UTF8')::jsonb
 OR p->'bound_state'->'binding'->'signed_request_canonical_bytes_reference' IS DISTINCT FROM
 pg_catalog.convert_from({i}.atomic_intent_member(retained_request,'reference'),'UTF8')::jsonb
 OR p->'bound_state'->'binding'->'signed_request_payload_digest_sha256' IS DISTINCT FROM
 pg_catalog.convert_from({i}.atomic_intent_member(retained_request,'digest'),'UTF8')::jsonb THEN
  RAISE EXCEPTION 'BOUND request, issuer credential or fingerprint differs' USING ERRCODE='23514'; END IF;
 event_identity:=pg_catalog.decode(p->'history_record'->>'canonical_event_identity_hex','hex');
 IF {h}.atomic_history_member(event_identity,'issuance_attempt_id') IS DISTINCT FROM p_attempt
 OR {h}.atomic_history_member(event_identity,'logical_operation_id') IS DISTINCT FROM operation
 OR {h}.atomic_history_member(event_identity,'root_proof_id') IS DISTINCT FROM root_id THEN
  RAISE EXCEPTION 'issuer history event identity differs' USING ERRCODE='23514'; END IF;
 -- Preserve the existing canonical issuance observation and bind it to the
 -- exact protected proof and authoritative registry state.
 event_payload:=pg_catalog.decode(p->'history_record'->>'canonical_payload_hex','hex');
 IF pg_catalog.convert_from({h}.atomic_history_member(event_payload,'entitlement'),'UTF8')::jsonb
 IS DISTINCT FROM p->'bound_state'
 OR {h}.atomic_history_member(event_payload,'authorization') IS DISTINCT FROM
 {i}.atomic_intent_member(reserved_bytes,'authorization')
 OR {h}.atomic_history_member(event_payload,'observation_kind') IS DISTINCT FROM
 pg_catalog.convert_to('"ROOT_PROOF_ISSUANCE_BINDING_OBSERVATION"','UTF8') THEN
  RAISE EXCEPTION 'history event differs from exact issuance binding' USING ERRCODE='23514'; END IF;
 PERFORM {e}.atomic_bind(p->>'lookup_handle',p->>'environment',p->>'trust_domain',
 (p->>'expected_revision')::bigint,(p->'bound_state')::json);
 PERFORM {h}.atomic_append_record(stream,p->>'expected_history_digest',(p->'history_record')::json,record_bytes);
 PERFORM {h}.atomic_retain_head(stream,(p->'history_head')::json,head_bytes,head_signature);
 INSERT INTO {s}.decisions(decision_id,scope_bytes,attempt_key,operation_key,entitlement_key,root_proof_key,
 account_key,lookup_handle,environment,trust_domain,entitlement_generation,registry_revision,
 reserved_envelope,staged_envelope,canonical_payload,signature,stream_identity,history_sequence,
 canonical_record,canonical_head,history_signature,bound_state,package)
 VALUES('adc_' || pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
 'CryptoHunter/AtomicRootProofDecision/v1','UTF8') || p_attempt || operation || entitlement),'hex'),
 scope,p_attempt,operation,entitlement,root_id,account,p->>'lookup_handle',p->>'environment',
 p->>'trust_domain',(p->>'entitlement_generation')::bigint,(p->>'expected_revision')::bigint+1,
 reserved_bytes,staged_bytes,payload,signature,stream,(p->'history_record'->>'sequence')::bigint,
 record_bytes,head_bytes,head_signature,p->'bound_state',p);
 SELECT * INTO STRICT existing FROM {s}.decisions WHERE attempt_key=p_attempt;
 RETURN existing.package || pg_catalog.jsonb_build_object('decision_id',existing.decision_id,
 'history_sequence',existing.history_sequence,'registry_revision',existing.registry_revision);
END
""",
    }
    return sources


def reviewed_function_inventory(
    conn: psycopg.Connection[Any],
    config: PostgreSQLAtomicRootProofProvisioning,
) -> dict[tuple[str, str], tuple[str, str, str, str]]:
    result = _component_inventory(conn, config)
    for schema in (
        config.entitlement_schema,
        config.intent_schema,
        config.retention_schema,
        config.history_schema,
        config.requester_schema,
        config.claimant_schema,
    ):
        owner = _metadata(conn, schema)["schema_owner_role"]
        result[(schema, "atomic_metadata()")] = (
            owner,
            f"BEGIN IF current_user<>'{owner}' THEN RAISE EXCEPTION 'wrong metadata owner' "
            f"USING ERRCODE='42501'; END IF; RETURN (SELECT pg_catalog.to_jsonb(m) FROM {_q(schema)}.metadata m "
            "WHERE singleton); END",
            "",
            "jsonb",
        )
    # Canonical parser facades confer EXECUTE only, never access to source tables.
    for schema, name in (
        (config.intent_schema, "atomic_intent_member"),
        (config.retention_schema, "atomic_retention_member"),
        (config.history_schema, "atomic_history_member"),
    ):
        owner = _metadata(conn, schema)["schema_owner_role"]
        result[(schema, name + "(bytea,text)")] = (
            owner,
            f"BEGIN {_guard(config, owner)} RETURN {_q(schema)}.canonical_object_member(p_bytes,p_key); END",
            "p_bytes bytea,p_key text",
            "bytea",
        )
    for signature, source in _function_sources(config).items():
        declaration, returns = FUNCTION_DECLARATIONS[signature]
        result[(config.schema, signature)] = (
            config.schema_owner_role,
            source,
            declaration,
            returns,
        )
    return result


def _create_tables(
    conn: psycopg.Connection[Any], config: PostgreSQLAtomicRootProofProvisioning
) -> None:
    conn.execute(
        sql.SQL("""
CREATE TABLE {s}.metadata (
 singleton boolean PRIMARY KEY DEFAULT true CHECK(singleton),
 schema_identity text NOT NULL,schema_version integer NOT NULL,migration_id text NOT NULL,
 instance_id text NOT NULL,trust_domain text NOT NULL,schema_owner_role text NOT NULL,
 schema_owner_role_oid oid NOT NULL,issuer_writer_role text NOT NULL,issuer_writer_role_oid oid NOT NULL,
 test_admin_role text NOT NULL,test_admin_role_oid oid NOT NULL,system_identifier text NOT NULL,
 db_oid oid NOT NULL,boundary text NOT NULL CHECK(boundary='ISOLATED_TEST_ONLY')
);
CREATE TABLE {s}.approved_commits(attempt_key bytea PRIMARY KEY,package jsonb NOT NULL);
CREATE TABLE {s}.catalog_manifest(singleton boolean PRIMARY KEY DEFAULT true CHECK(singleton),
 component_inventory_version integer NOT NULL CHECK(component_inventory_version=2),snapshot jsonb NOT NULL);
CREATE TABLE {s}.fences (
 identity text PRIMARY KEY,semantic_role text NOT NULL CHECK(semantic_role IN
 ('ROOT_PROOF_SIGNING','HISTORY_ATTESTATION_SIGNING','TRUST_ACTIVATION')),
 generation bigint NOT NULL CHECK(generation>=1),state text NOT NULL CHECK(state IN
 ('PENDING_CLOSED','ACTIVE','VERIFY_ONLY','REVOKED')),ever_active boolean NOT NULL
);
CREATE TABLE {s}.decisions (
 decision_id text NOT NULL UNIQUE,
 scope_bytes bytea NOT NULL,attempt_key bytea PRIMARY KEY,operation_key bytea NOT NULL UNIQUE,
 entitlement_key bytea NOT NULL UNIQUE,root_proof_key bytea NOT NULL UNIQUE,account_key bytea NOT NULL,
 lookup_handle text NOT NULL,environment text NOT NULL,trust_domain text NOT NULL,
 entitlement_generation bigint NOT NULL CHECK(entitlement_generation>=1),
 registry_revision bigint NOT NULL CHECK(registry_revision>=2),reserved_envelope bytea NOT NULL,
 staged_envelope bytea NOT NULL,canonical_payload bytea NOT NULL,
 signature bytea NOT NULL CHECK(octet_length(signature)=64),stream_identity bytea NOT NULL,
 history_sequence bigint NOT NULL CHECK(history_sequence>=1),canonical_record bytea NOT NULL,
 canonical_head bytea NOT NULL,history_signature bytea NOT NULL CHECK(octet_length(history_signature)=64),
 bound_state jsonb NOT NULL,package jsonb NOT NULL,
 UNIQUE(lookup_handle,environment,trust_domain,entitlement_generation),
 UNIQUE(stream_identity,history_sequence),
 FOREIGN KEY(attempt_key) REFERENCES {s}.approved_commits(attempt_key)
);
REVOKE ALL ON ALL TABLES IN SCHEMA {s} FROM PUBLIC;
""").format(s=sql.Identifier(config.schema))
    )


def install_atomic_root_proof_authority(
    conn: psycopg.Connection[Any],
    config: PostgreSQLAtomicRootProofProvisioning,
) -> dict[tuple[str, str], tuple[str, str, str, str]]:
    """Install after explicit legacy prequalification in an admin transaction.

    The caller must postqualify V2 before COMMIT.  A failed postqualification
    rolls back both installation and all revoked standalone grants.  Existing
    identifiers, rows and immutable proof bytes are never rewritten.
    """
    if type(config) is not PostgreSQLAtomicRootProofProvisioning:
        raise TypeError("exact isolated atomic provisioning required")
    if conn.info.transaction_status != psycopg.pq.TransactionStatus.INTRANS:
        raise ValueError("explicit administrative migration transaction required")
    if not 160000 <= conn.info.server_version < 170000:
        raise ValueError("PostgreSQL 16 required")
    if (
        conn.execute(
            sql.SQL(
                "SELECT 1 FROM {}.history WHERE state#>>'{{binding,kind}}'='BOUND' LIMIT 1"
            ).format(sql.Identifier(config.entitlement_schema))
        ).fetchone()
        is not None
    ):
        raise ValueError("existing BOUND requires a separately qualified exact-byte migration")
    for table in ("records", "attestations"):
        if (
            conn.execute(
                sql.SQL("SELECT 1 FROM {}.{} LIMIT 1").format(
                    sql.Identifier(config.history_schema), sql.Identifier(table)
                )
            ).fetchone()
            is not None
        ):
            raise ValueError(
                "existing history requires a separately qualified exact-byte migration"
            )
    conn.execute(
        sql.SQL("REVOKE TEMPORARY ON DATABASE {} FROM PUBLIC").format(
            sql.Identifier(conn.info.dbname)
        )
    )
    # PostgreSQL grants PUBLIC UPDATE on this view by default. The isolated
    # authority permits no direct DML, including catalog views.
    conn.execute("REVOKE UPDATE ON TABLE pg_catalog.pg_settings FROM PUBLIC")
    for schema in (
        config.entitlement_schema,
        config.intent_schema,
        config.retention_schema,
        config.history_schema,
        config.requester_schema,
        config.claimant_schema,
    ):
        owner = _metadata(conn, schema)["schema_owner_role"]
        conn.execute(sql.SQL("ALTER ROLE {} NOLOGIN").format(sql.Identifier(owner)))
    for role, login in (
        (config.schema_owner_role, "NOLOGIN"),
        (config.issuer_writer_role, "LOGIN"),
        (config.test_admin_role, "LOGIN"),
    ):
        conn.execute(
            sql.SQL(
                "CREATE ROLE {} {} NOSUPERUSER NOCREATEDB NOCREATEROLE "
                "NOINHERIT NOREPLICATION NOBYPASSRLS"
            ).format(sql.Identifier(role), sql.SQL(login))
        )
    conn.execute(
        sql.SQL("CREATE SCHEMA {} AUTHORIZATION {}").format(
            sql.Identifier(config.schema), sql.Identifier(config.schema_owner_role)
        )
    )
    conn.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(config.schema_owner_role)))
    conn.execute(
        sql.SQL("REVOKE ALL ON SCHEMA {} FROM PUBLIC").format(sql.Identifier(config.schema))
    )
    _create_tables(conn, config)
    conn.execute("RESET ROLE")
    inventory = reviewed_function_inventory(conn, config)
    for (schema, signature), (owner, source, declaration, returns) in inventory.items():
        conn.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(owner)))
        conn.execute(
            sql.SQL(
                "CREATE FUNCTION {}.{}({}) RETURNS {} LANGUAGE plpgsql "
                "SECURITY DEFINER SET search_path=pg_catalog AS {}"
            ).format(
                sql.Identifier(schema),
                sql.Identifier(signature.split("(", 1)[0]),
                sql.SQL(declaration),
                sql.SQL(returns),
                sql.Literal(source),
            )
        )
        conn.execute(
            sql.SQL("REVOKE ALL ON FUNCTION {}.{} FROM PUBLIC").format(
                sql.Identifier(schema), sql.SQL(signature)
            )
        )
        if schema != config.schema and not signature.endswith("constraint()"):
            conn.execute(
                sql.SQL("GRANT EXECUTE ON FUNCTION {}.{} TO {}").format(
                    sql.Identifier(schema),
                    sql.SQL(signature),
                    sql.Identifier(config.schema_owner_role),
                )
            )
        elif schema == config.schema and signature in FUNC_RUNTIME | FUNC_ADMIN:
            role = (
                config.issuer_writer_role if signature in FUNC_RUNTIME else config.test_admin_role
            )
            conn.execute(
                sql.SQL("GRANT EXECUTE ON FUNCTION {}.{} TO {}").format(
                    sql.Identifier(schema), sql.SQL(signature), sql.Identifier(role)
                )
            )
            if signature == "qualify_catalog()":
                conn.execute(
                    sql.SQL("GRANT EXECUTE ON FUNCTION {}.{} TO {}").format(
                        sql.Identifier(schema),
                        sql.SQL(signature),
                        sql.Identifier(config.test_admin_role),
                    )
                )
        conn.execute("RESET ROLE")
    owners = {
        config.entitlement_schema: _metadata(conn, config.entitlement_schema)["schema_owner_role"],
        config.intent_schema: _metadata(conn, config.intent_schema)["schema_owner_role"],
        config.retention_schema: _metadata(conn, config.retention_schema)["schema_owner_role"],
        config.history_schema: _metadata(conn, config.history_schema)["schema_owner_role"],
        config.requester_schema: _metadata(conn, config.requester_schema)["schema_owner_role"],
        config.claimant_schema: _metadata(conn, config.claimant_schema)["schema_owner_role"],
    }
    for schema in owners:
        conn.execute(
            sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
                sql.Identifier(schema), sql.Identifier(config.schema_owner_role)
            )
        )
    # Close every pre-existing standalone issuance grant.  Owner privileges stay
    # with their original NOLOGIN authorities, not the commit login.
    for schema, signatures in (
        (config.entitlement_schema, ("append_bind(text,text,text,bigint,json)",)),
        (
            config.history_schema,
            ("append_record(bytea,text,json,bytea)", "retain_head(bytea,json,bytea,bytea)"),
        ),
    ):
        for signature in signatures:
            fn = conn.execute(
                "SELECT p.oid FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n "
                "ON n.oid=p.pronamespace WHERE n.nspname=%s AND "
                "p.proname=%s AND pg_catalog.oidvectortypes(p.proargtypes)=%s",
                (
                    schema,
                    signature.split("(", 1)[0],
                    signature.split("(", 1)[1][:-1].replace(",", ", "),
                ),
            ).fetchone()
            if fn is None:
                raise ValueError("standalone component function missing")
            grants = conn.execute(
                "SELECT r.rolname FROM pg_catalog.pg_proc p CROSS JOIN LATERAL "
                "pg_catalog.aclexplode(p.proacl) a JOIN pg_catalog.pg_roles r "
                "ON r.oid=a.grantee WHERE p.oid=%s AND a.grantee<>p.proowner",
                (fn[0],),
            ).fetchall()
            for (role,) in grants:
                conn.execute(
                    sql.SQL("REVOKE ALL ON FUNCTION {}.{} FROM {}").format(
                        sql.Identifier(schema), sql.SQL(signature), sql.Identifier(role)
                    )
                )
    registry_owner = owners[config.entitlement_schema]
    history_owner = owners[config.history_schema]
    for signature, role in (
        ("assert_bound_link(text,text,text,bigint,jsonb)", registry_owner),
        ("assert_history_link(bytea,bigint,bytea,bytea,bytea)", history_owner),
    ):
        conn.execute(
            sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
                sql.Identifier(config.schema), sql.Identifier(role)
            )
        )
        conn.execute(
            sql.SQL("GRANT EXECUTE ON FUNCTION {}.{} TO {}").format(
                sql.Identifier(config.schema), sql.SQL(signature), sql.Identifier(role)
            )
        )
    for schema, table, name, fn in (
        (config.entitlement_schema, "history", "atomic_bound_link", "atomic_bound_constraint"),
        (config.history_schema, "records", "atomic_record_link", "atomic_history_constraint"),
        (config.history_schema, "attestations", "atomic_head_link", "atomic_history_constraint"),
        (config.schema, "decisions", "atomic_decision_link", "validate_decision"),
    ):
        conn.execute(
            sql.SQL(
                "CREATE CONSTRAINT TRIGGER {} AFTER INSERT ON {}.{} "
                "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION {}.{}()"
            ).format(
                sql.Identifier(name),
                sql.Identifier(schema),
                sql.Identifier(table),
                sql.Identifier(schema),
                sql.Identifier(fn),
            )
        )
        conn.execute(
            sql.SQL("ALTER TABLE {}.{} ENABLE ALWAYS TRIGGER {}").format(
                sql.Identifier(schema), sql.Identifier(table), sql.Identifier(name)
            )
        )
    for table in ("metadata", "approved_commits", "decisions", "catalog_manifest"):
        for name, events, level in (
            ("immutable_row", "UPDATE OR DELETE", "ROW"),
            ("immutable_truncate", "TRUNCATE", "STATEMENT"),
        ):
            conn.execute(
                sql.SQL(
                    "CREATE TRIGGER {} BEFORE {} ON {}.{} FOR EACH {} "
                    "EXECUTE FUNCTION {}.reject_mutation()"
                ).format(
                    sql.Identifier(name),
                    sql.SQL(events),
                    sql.Identifier(config.schema),
                    sql.Identifier(table),
                    sql.SQL(level),
                    sql.Identifier(config.schema),
                )
            )
            conn.execute(
                sql.SQL("ALTER TABLE {}.{} ENABLE ALWAYS TRIGGER {}").format(
                    sql.Identifier(config.schema), sql.Identifier(table), sql.Identifier(name)
                )
            )
    for role in (config.issuer_writer_role, config.test_admin_role):
        conn.execute(
            sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
                sql.Identifier(config.schema), sql.Identifier(role)
            )
        )
        conn.execute(
            sql.SQL("GRANT SELECT ON {}.metadata TO {}").format(
                sql.Identifier(config.schema), sql.Identifier(role)
            )
        )
    # OIDs and physical identity are pinned separately from restart-sensitive PID.
    role_oids = []
    for role in (config.schema_owner_role, config.issuer_writer_role, config.test_admin_role):
        role_row = conn.execute(
            "SELECT oid FROM pg_catalog.pg_roles WHERE rolname=%s", (role,)
        ).fetchone()
        if role_row is None:
            raise ValueError("atomic principal disappeared during migration")
        role_oids.append(role_row[0])
    physical = conn.execute(
        "SELECT system_identifier::text,(SELECT oid FROM pg_catalog.pg_database "
        "WHERE datname=pg_catalog.current_database()) FROM pg_catalog.pg_control_system()"
    ).fetchone()
    if physical is None:
        raise ValueError("atomic physical database identity unavailable")
    conn.execute(
        sql.SQL(
            "INSERT INTO {}.metadata VALUES(true,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        ).format(sql.Identifier(config.schema)),
        (
            SCHEMA_IDENTITY,
            SCHEMA_VERSION,
            MIGRATION_ID,
            config.instance_id,
            config.trust_domain,
            config.schema_owner_role,
            role_oids[0],
            config.issuer_writer_role,
            role_oids[1],
            config.test_admin_role,
            role_oids[2],
            physical[0],
            physical[1],
            BOUNDARY,
        ),
    )
    conn.execute(
        sql.SQL("INSERT INTO {}.catalog_manifest VALUES(true,2,{}.catalog_snapshot())").format(
            sql.Identifier(config.schema), sql.Identifier(config.schema)
        )
    )
    return inventory


__all__ = [
    "BOUNDARY",
    "SCHEMA_IDENTITY",
    "SCHEMA_VERSION",
    "MIGRATION_ID",
    "PACKAGE_KEYS",
    "FUNC_RUNTIME",
    "FUNC_ADMIN",
    "FUNCTION_DECLARATIONS",
    "PRODUCTION_LIFECYCLE_FENCE_AVAILABLE",
    "PostgreSQLAtomicRootProofProvisioning",
    "install_atomic_root_proof_authority",
    "reviewed_function_inventory",
]
