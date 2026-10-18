"""Production lifecycle SQL, deliberately installed with every fence closed.

This is an independent production authority, not a promoted test deployment.
Only public credential facts enter PostgreSQL. The separately authenticated
administrator validates custody completion; SQL preserves its exact receipt.
No function activates a credential or performs Root Proof issuance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

SCHEMA_IDENTITY = "cryptohunter.production_root_lifecycle.postgresql"
SCHEMA_VERSION = 1
BOUNDARY = "PRODUCTION_LIFECYCLE_CLOSED"
PRODUCTION_LIFECYCLE_ISOLATION = "BLOCKED"
SEMANTIC_ROLES = frozenset(
    {"ROOT_PROOF_SIGNING", "HISTORY_ATTESTATION_SIGNING", "TRUST_ACTIVATION"}
)
RELATIONS = frozenset(
    {"metadata", "catalog_manifest", "fences", "slots", "operations", "completions", "observations"}
)
_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]{0,62}\Z")
_OPERATION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_MATERIAL = re.compile(r"sha256:[0-9a-f]{64}\Z")
_BINDING_KEYS = frozenset(
    {
        "semantic_role",
        "authority_identity",
        "identity",
        "public_material_identity",
        "key_version",
        "lifecycle_generation",
    }
)


@dataclass(frozen=True, slots=True)
class PostgreSQLProductionRootLifecycleProvisioning:
    """Installation-owned identifiers; never construct this from an issuance request."""

    schema: str
    schema_owner_role: str
    issuer_role: str
    administrator_role: str
    installation_id: str
    trust_domain: str

    def __post_init__(self) -> None:
        for value in (
            self.schema,
            self.schema_owner_role,
            self.issuer_role,
            self.administrator_role,
        ):
            if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
                raise ValueError("safe exact production authority identifiers required")
        if len({self.schema_owner_role, self.issuer_role, self.administrator_role}) != 3:
            raise ValueError("production owner, issuer and administrator must be distinct")
        for value in (self.installation_id, self.trust_domain):
            if type(value) is not str or not value.strip() or "\0" in value or len(value) > 512:
                raise ValueError("exact installed authority namespace required")


PRODUCTION_LIFECYCLE_FUNCTION_DECLARATIONS = {
    "catalog_snapshot()": ("", "jsonb"),
    "assert_catalog()": ("", "void"),
    "reject_mutation()": ("", "trigger"),
    "guard_fence_update()": ("", "trigger"),
    "guard_slot_update()": ("", "trigger"),
    "begin_close(jsonb)": ("p_request jsonb", "jsonb"),
    "read_operation(text,text,text)": (
        "p_role text,p_operation text,p_request_sha256 text",
        "jsonb",
    ),
    "complete_close(text,text,bytea)": (
        "p_operation text,p_request_sha256 text,p_receipt bytea",
        "jsonb",
    ),
    "enroll_closed_successor(jsonb)": ("p_request jsonb", "jsonb"),
    "assert_active_fences(jsonb)": ("p_expected jsonb", "jsonb"),
    "read_historical_fence(text,text,bigint)": (
        "p_role text,p_identity text,p_generation bigint",
        "jsonb",
    ),
    "observe_signer_custody(text,text)": ("p_role text,p_identity text", "jsonb"),
}
ISSUER_FUNCTIONS = frozenset(
    {
        "assert_catalog()",
        "assert_active_fences(jsonb)",
        "read_historical_fence(text,text,bigint)",
        "observe_signer_custody(text,text)",
    }
)
ADMINISTRATOR_FUNCTIONS = frozenset(
    {
        "assert_catalog()",
        "begin_close(jsonb)",
        "read_operation(text,text,text)",
        "complete_close(text,text,bytea)",
        "enroll_closed_successor(jsonb)",
        "read_historical_fence(text,text,bigint)",
    }
)


def _q(value: str) -> str:
    return '"' + value + '"'


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _principal_guard(config: PostgreSQLProductionRootLifecycleProvisioning, role: str) -> str:
    s = _q(config.schema)
    name = config.issuer_role if role == "issuer" else config.administrator_role
    column = "issuer_role_oid" if role == "issuer" else "administrator_role_oid"
    return f"""
 PERFORM {s}.assert_catalog();
 IF session_user::text IS DISTINCT FROM {_literal(name)}
 OR current_user::text IS DISTINCT FROM {_literal(config.schema_owner_role)}
 OR (SELECT oid FROM pg_catalog.pg_roles WHERE rolname=session_user::text)
 IS DISTINCT FROM (SELECT {column} FROM {s}.metadata WHERE singleton)
 OR (SELECT usename::text FROM pg_catalog.pg_stat_activity WHERE pid=pg_catalog.pg_backend_pid())
 IS DISTINCT FROM session_user::text THEN
  RAISE EXCEPTION 'exact direct production {role} login required' USING ERRCODE='42501';
 END IF;
"""


def reviewed_production_lifecycle_function_sources(
    config: PostgreSQLProductionRootLifecycleProvisioning,
) -> dict[str, str]:
    """Reviewed fixed SQL; the configuration supplies identifiers and installed pins."""
    if type(config) is not PostgreSQLProductionRootLifecycleProvisioning:
        raise TypeError("exact production lifecycle provisioning required")
    s = _q(config.schema)
    admin = _principal_guard(config, "administrator")
    issuer = _principal_guard(config, "issuer")
    binding_keys = "ARRAY['authority_identity','identity','key_version','lifecycle_generation','public_material_identity','semantic_role']::text[]"
    return {
        "catalog_snapshot()": f"""
BEGIN
 RETURN pg_catalog.jsonb_build_object(
  'schema',(SELECT pg_catalog.to_jsonb(n) FROM pg_catalog.pg_namespace n WHERE nspname={_literal(config.schema)}),
  'functions',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(p) ORDER BY p.oid)
    FROM pg_catalog.pg_proc p WHERE pronamespace={_literal(config.schema)}::regnamespace),
  'relations',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(c)-ARRAY[
    'relpages','reltuples','relallvisible','relfrozenxid','relminmxid']::text[] ORDER BY c.oid)
    FROM pg_catalog.pg_class c WHERE relnamespace={_literal(config.schema)}::regnamespace),
  'columns',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(a)-'attcacheoff' ORDER BY a.attrelid,a.attnum)
    FROM pg_catalog.pg_attribute a JOIN pg_catalog.pg_class c ON c.oid=a.attrelid
    WHERE c.relnamespace={_literal(config.schema)}::regnamespace AND a.attnum>0),
  'constraints',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(c) ORDER BY c.oid)
    FROM pg_catalog.pg_constraint c WHERE connamespace={_literal(config.schema)}::regnamespace),
  'defaults',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(d) ORDER BY d.oid)
    FROM pg_catalog.pg_attrdef d JOIN pg_catalog.pg_class c ON c.oid=d.adrelid
    WHERE c.relnamespace={_literal(config.schema)}::regnamespace),
  'indexes',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(i) ORDER BY i.indexrelid)
    FROM pg_catalog.pg_index i JOIN pg_catalog.pg_class c ON c.oid=i.indrelid
    WHERE c.relnamespace={_literal(config.schema)}::regnamespace),
  'triggers',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(t) ORDER BY t.oid)
    FROM pg_catalog.pg_trigger t JOIN pg_catalog.pg_class c ON c.oid=t.tgrelid
    WHERE c.relnamespace={_literal(config.schema)}::regnamespace),
  'rules',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r) ORDER BY r.oid)
    FROM pg_catalog.pg_rewrite r JOIN pg_catalog.pg_class c ON c.oid=r.ev_class
    WHERE c.relnamespace={_literal(config.schema)}::regnamespace),
  'policies',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(p) ORDER BY p.oid)
    FROM pg_catalog.pg_policy p JOIN pg_catalog.pg_class c ON c.oid=p.polrelid
    WHERE c.relnamespace={_literal(config.schema)}::regnamespace));
END
""",
        "assert_catalog()": f"""
DECLARE m {s}.metadata%ROWTYPE; names oid[];
BEGIN
 SELECT * INTO STRICT m FROM {s}.metadata WHERE singleton;
 IF m.schema_identity IS DISTINCT FROM '{SCHEMA_IDENTITY}' OR m.schema_version<>{SCHEMA_VERSION}
 OR m.boundary IS DISTINCT FROM '{BOUNDARY}' OR m.profile IS DISTINCT FROM 'PRODUCTION_LOCAL'
 OR m.installation_id IS DISTINCT FROM {_literal(config.installation_id)}
 OR m.trust_domain IS DISTINCT FROM {_literal(config.trust_domain)}
 OR m.system_identifier IS DISTINCT FROM (SELECT system_identifier FROM pg_catalog.pg_control_system())
 OR m.database_oid IS DISTINCT FROM (SELECT oid FROM pg_catalog.pg_database WHERE datname=pg_catalog.current_database())
 OR pg_catalog.current_setting('server_version_num')::integer NOT BETWEEN 160000 AND 169999
 OR pg_catalog.current_setting('fsync')<>'on' OR pg_catalog.current_setting('synchronous_commit')<>'on'
 OR pg_catalog.current_setting('full_page_writes')<>'on' OR pg_catalog.pg_is_in_recovery()
 THEN RAISE EXCEPTION 'durable exact production lifecycle authority required' USING ERRCODE='42501'; END IF;
 names:=ARRAY[m.owner_role_oid,m.issuer_role_oid,m.administrator_role_oid];
 IF (SELECT count(*) FROM pg_catalog.pg_roles WHERE oid=ANY(names))<>3
 OR NOT EXISTS(SELECT 1 FROM pg_catalog.pg_roles WHERE oid=m.owner_role_oid
  AND rolname={_literal(config.schema_owner_role)} AND NOT rolcanlogin AND NOT rolinherit
  AND NOT rolsuper AND NOT rolcreaterole AND NOT rolcreatedb AND NOT rolreplication AND NOT rolbypassrls)
 OR NOT EXISTS(SELECT 1 FROM pg_catalog.pg_roles WHERE oid=m.issuer_role_oid
  AND rolname={_literal(config.issuer_role)} AND rolcanlogin AND NOT rolinherit
  AND NOT rolsuper AND NOT rolcreaterole AND NOT rolcreatedb AND NOT rolreplication AND NOT rolbypassrls)
 OR NOT EXISTS(SELECT 1 FROM pg_catalog.pg_roles WHERE oid=m.administrator_role_oid
  AND rolname={_literal(config.administrator_role)} AND rolcanlogin AND NOT rolinherit
  AND NOT rolsuper AND NOT rolcreaterole AND NOT rolcreatedb AND NOT rolreplication AND NOT rolbypassrls)
 OR EXISTS(SELECT 1 FROM pg_catalog.pg_auth_members WHERE roleid=ANY(names) OR member=ANY(names))
 OR pg_catalog.has_database_privilege(m.issuer_role_oid,pg_catalog.current_database(),'CREATE')
 OR pg_catalog.has_database_privilege(m.administrator_role_oid,pg_catalog.current_database(),'CREATE')
 OR pg_catalog.has_database_privilege(m.issuer_role_oid,pg_catalog.current_database(),'TEMPORARY')
 OR pg_catalog.has_database_privilege(m.administrator_role_oid,pg_catalog.current_database(),'TEMPORARY')
 OR (SELECT snapshot FROM {s}.catalog_manifest WHERE singleton) IS DISTINCT FROM {s}.catalog_snapshot()
 THEN RAISE EXCEPTION 'production principal or reviewed catalog drift' USING ERRCODE='42501'; END IF;
END
""",
        "reject_mutation()": "BEGIN RAISE EXCEPTION 'immutable lifecycle evidence' USING ERRCODE='42501'; END",
        "guard_fence_update()": f"""
BEGIN
 IF TG_OP<>'UPDATE' OR (NEW.semantic_role,NEW.identity,NEW.authority_identity,NEW.public_material_identity,NEW.key_version)
 IS DISTINCT FROM (OLD.semantic_role,OLD.identity,OLD.authority_identity,OLD.public_material_identity,OLD.key_version)
 THEN RAISE EXCEPTION 'immutable lifecycle credential' USING ERRCODE='42501'; END IF;
 IF OLD.state IN ('ACTIVE','VERIFY_ONLY') AND NEW.state='PENDING_CLOSED'
 AND NEW.generation=OLD.generation AND EXISTS(SELECT 1 FROM {s}.operations o
  WHERE o.operation_id=NEW.pending_operation_id AND o.semantic_role=OLD.semantic_role
  AND o.identity=OLD.identity AND o.request->>'predecessor_state'=OLD.state
  AND (o.request->>'generation')::bigint=OLD.generation)
 THEN RETURN NEW; END IF;
 IF OLD.state='PENDING_CLOSED' AND NEW.state IN ('VERIFY_ONLY','REVOKED')
 AND NEW.generation=OLD.generation+1 AND NEW.pending_operation_id IS NULL
 AND EXISTS(SELECT 1 FROM {s}.operations o JOIN {s}.completions c USING(operation_id)
  WHERE o.operation_id=OLD.pending_operation_id AND o.request->>'target_state'=NEW.state)
 THEN RETURN NEW; END IF;
 RAISE EXCEPTION 'closed or revoked lifecycle cannot reopen' USING ERRCODE='42501';
END
""",
        "guard_slot_update()": f"""
BEGIN
 IF TG_OP='UPDATE' AND NEW.semantic_role=OLD.semantic_role AND NEW.identity<>OLD.identity
 AND EXISTS(SELECT 1 FROM {s}.fences predecessor JOIN {s}.fences successor USING(semantic_role)
  JOIN {s}.operations o ON o.operation_id=successor.pending_operation_id
  WHERE predecessor.semantic_role=OLD.semantic_role AND predecessor.identity=OLD.identity
  AND successor.identity=NEW.identity AND predecessor.state IN ('VERIFY_ONLY','REVOKED')
  AND successor.state='PENDING_CLOSED' AND successor.generation>predecessor.generation
  AND o.request->>'predecessor_identity'=OLD.identity AND o.request->'successor'->>'identity'=NEW.identity)
 THEN RETURN NEW; END IF;
 RAISE EXCEPTION 'immutable installed lifecycle slot or unclosed successor' USING ERRCODE='42501';
END
""",
        "begin_close(jsonb)": f"""
DECLARE f {s}.fences%ROWTYPE; prior {s}.operations%ROWTYPE; digest text;
BEGIN
 {admin}
 IF p_request IS NULL OR pg_catalog.jsonb_typeof(p_request)<>'object'
 OR (SELECT pg_catalog.array_agg(k ORDER BY k) FROM pg_catalog.jsonb_object_keys(p_request) k)
 IS DISTINCT FROM ARRAY['generation','identity','operation_id','predecessor_state','semantic_role','target_state']::text[]
 OR pg_catalog.jsonb_typeof(p_request->'operation_id') IS DISTINCT FROM 'string'
 OR p_request->>'operation_id' !~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{{0,127}}$'
 OR pg_catalog.jsonb_typeof(p_request->'identity') IS DISTINCT FROM 'string'
 OR p_request->>'identity'=''
 OR pg_catalog.jsonb_typeof(p_request->'semantic_role') IS DISTINCT FROM 'string'
 OR pg_catalog.jsonb_typeof(p_request->'predecessor_state') IS DISTINCT FROM 'string'
 OR pg_catalog.jsonb_typeof(p_request->'target_state') IS DISTINCT FROM 'string'
 OR p_request->>'predecessor_state' NOT IN ('ACTIVE','VERIFY_ONLY')
 OR p_request->>'target_state' NOT IN ('VERIFY_ONLY','REVOKED')
 OR (p_request->>'predecessor_state'='VERIFY_ONLY' AND p_request->>'target_state'<>'REVOKED')
 OR pg_catalog.jsonb_typeof(p_request->'generation') IS DISTINCT FROM 'number'
 OR p_request->>'generation' !~ '^[1-9][0-9]*$'
 THEN RAISE EXCEPTION 'exact one-way lifecycle close required' USING ERRCODE='22023'; END IF;
 PERFORM 1 FROM {s}.slots WHERE semantic_role=p_request->>'semantic_role' FOR UPDATE;
 IF NOT FOUND THEN RAISE EXCEPTION 'installed lifecycle slot missing' USING ERRCODE='23514'; END IF;
 SELECT * INTO STRICT f FROM {s}.fences WHERE semantic_role=p_request->>'semantic_role'
 AND identity=p_request->>'identity' FOR UPDATE;
 SELECT * INTO prior FROM {s}.operations WHERE operation_id=p_request->>'operation_id';
 IF FOUND THEN
  IF prior.request IS DISTINCT FROM p_request THEN
   RAISE EXCEPTION 'lifecycle operation identity conflict' USING ERRCODE='23505'; END IF;
  RETURN pg_catalog.jsonb_build_object('operation_id',prior.operation_id,'request_sha256',prior.request_sha256,
   'phase',CASE WHEN EXISTS(SELECT 1 FROM {s}.completions WHERE operation_id=prior.operation_id)
    THEN 'COMPLETED' ELSE 'PENDING_CLOSED' END);
 END IF;
 IF f.state IS DISTINCT FROM p_request->>'predecessor_state'
 OR f.generation IS DISTINCT FROM (p_request->>'generation')::bigint
 OR f.generation>=9223372036854775807 OR f.pending_operation_id IS NOT NULL
 OR (SELECT identity FROM {s}.slots WHERE semantic_role=f.semantic_role) IS DISTINCT FROM f.identity
 THEN RAISE EXCEPTION 'stale or closed lifecycle predecessor' USING ERRCODE='23514'; END IF;
 digest:=pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(p_request::text,'UTF8')),'hex');
 INSERT INTO {s}.operations(operation_id,semantic_role,identity,request,request_sha256)
 VALUES(p_request->>'operation_id',f.semantic_role,f.identity,p_request,digest);
 INSERT INTO {s}.observations VALUES(f.semantic_role,f.identity,f.generation,f.state)
 ON CONFLICT DO NOTHING;
 UPDATE {s}.fences SET state='PENDING_CLOSED',pending_operation_id=p_request->>'operation_id'
 WHERE semantic_role=f.semantic_role AND identity=f.identity;
 RETURN pg_catalog.jsonb_build_object('operation_id',p_request->>'operation_id','request_sha256',digest,'phase','PENDING_CLOSED');
END
""",
        "read_operation(text,text,text)": f"""
DECLARE op {s}.operations%ROWTYPE; completed bytea;
BEGIN
 {admin}
 -- Lock first, then observe in a fresh READ COMMITTED statement. An unknown
 -- prior COMMIT can still be running in a different PostgreSQL backend.
 IF pg_catalog.current_setting('transaction_isolation')<>'read committed' THEN
  RAISE EXCEPTION 'fresh lock-synchronized resolution required' USING ERRCODE='25001'; END IF;
 PERFORM 1 FROM {s}.slots WHERE semantic_role=p_role FOR UPDATE;
 IF NOT FOUND THEN RAISE EXCEPTION 'installed lifecycle slot missing' USING ERRCODE='23514'; END IF;
 SELECT * INTO op FROM {s}.operations WHERE operation_id=p_operation;
 IF NOT FOUND THEN RETURN NULL; END IF;
 IF op.close_xid=pg_catalog.pg_current_xact_id() THEN
  RAISE EXCEPTION 'committed lifecycle close observation required' USING ERRCODE='25001'; END IF;
 IF op.semantic_role IS DISTINCT FROM p_role OR op.request_sha256 IS DISTINCT FROM p_request_sha256 THEN
  RAISE EXCEPTION 'exact administrative operation differs' USING ERRCODE='23514'; END IF;
 SELECT receipt_sha256 INTO completed FROM {s}.completions WHERE operation_id=p_operation;
 RETURN pg_catalog.jsonb_build_object('operation_id',op.operation_id,'request',op.request,
  'request_sha256',op.request_sha256,'phase',CASE WHEN completed IS NULL THEN 'PENDING_CLOSED' ELSE 'COMPLETED' END,
  'receipt_sha256',pg_catalog.encode(completed,'hex'));
END
""",
        "complete_close(text,text,bytea)": f"""
DECLARE op {s}.operations%ROWTYPE; f {s}.fences%ROWTYPE; receipt jsonb; existing bytea;
BEGIN
 {admin}
 SELECT * INTO STRICT op FROM {s}.operations WHERE operation_id=p_operation;
 IF op.close_xid=pg_catalog.pg_current_xact_id() THEN
  RAISE EXCEPTION 'lifecycle close must commit before completion' USING ERRCODE='25001'; END IF;
 PERFORM 1 FROM {s}.slots WHERE semantic_role=op.semantic_role FOR UPDATE;
 SELECT * INTO STRICT f FROM {s}.fences WHERE semantic_role=op.semantic_role AND identity=op.identity FOR UPDATE;
 IF op.request_sha256 IS DISTINCT FROM p_request_sha256 OR p_receipt IS NULL
 OR pg_catalog.octet_length(p_receipt)>262144 THEN
  RAISE EXCEPTION 'exact close and completion receipt required' USING ERRCODE='23514'; END IF;
 receipt:=pg_catalog.convert_from(p_receipt,'UTF8')::jsonb;
 IF pg_catalog.jsonb_typeof(receipt)<>'object'
 OR (SELECT pg_catalog.array_agg(k ORDER BY k) FROM pg_catalog.jsonb_object_keys(receipt) k)
 IS DISTINCT FROM ARRAY['custody_receipt_sha256','generation','identity','key_version','operation_id','public_material_identity','request_sha256','semantic_role','state']::text[]
 OR pg_catalog.jsonb_typeof(receipt->'custody_receipt_sha256') IS DISTINCT FROM 'string'
 OR receipt->>'custody_receipt_sha256' !~ '^[0-9a-f]{{64}}$'
 OR receipt-'custody_receipt_sha256' IS DISTINCT FROM pg_catalog.jsonb_build_object(
  'operation_id',op.operation_id,'request_sha256',op.request_sha256,'semantic_role',f.semantic_role,
  'identity',f.identity,'public_material_identity',f.public_material_identity,'key_version',f.key_version,
  'generation',(op.request->>'generation')::bigint+1,'state',op.request->>'target_state')
 THEN RAISE EXCEPTION 'completion differs from exact public custody facts' USING ERRCODE='23514'; END IF;
 SELECT receipt_bytes INTO existing FROM {s}.completions WHERE operation_id=p_operation;
 IF FOUND THEN
  IF existing IS DISTINCT FROM p_receipt THEN
   RAISE EXCEPTION 'immutable completion receipt conflict' USING ERRCODE='23505'; END IF;
 ELSE
  IF f.state<>'PENDING_CLOSED' OR f.pending_operation_id IS DISTINCT FROM p_operation
  OR f.generation IS DISTINCT FROM (op.request->>'generation')::bigint THEN
   RAISE EXCEPTION 'exact pending close required' USING ERRCODE='23514'; END IF;
  INSERT INTO {s}.completions VALUES(p_operation,p_receipt,pg_catalog.sha256(p_receipt));
  UPDATE {s}.fences SET state=op.request->>'target_state',generation=generation+1,pending_operation_id=NULL
  WHERE semantic_role=f.semantic_role AND identity=f.identity;
  INSERT INTO {s}.observations VALUES(f.semantic_role,f.identity,f.generation+1,op.request->>'target_state');
 END IF;
 RETURN pg_catalog.jsonb_build_object('operation_id',p_operation,'request_sha256',p_request_sha256,
  'phase','COMPLETED','state',op.request->>'target_state','receipt_sha256',pg_catalog.encode(pg_catalog.sha256(p_receipt),'hex'));
END
""",
        "enroll_closed_successor(jsonb)": f"""
DECLARE f {s}.fences%ROWTYPE; successor jsonb; op text; prior jsonb; digest text;
BEGIN
 {admin}
 IF p_request IS NULL OR pg_catalog.jsonb_typeof(p_request)<>'object'
 OR (SELECT pg_catalog.array_agg(k ORDER BY k) FROM pg_catalog.jsonb_object_keys(p_request) k)
 IS DISTINCT FROM ARRAY['expected_generation','operation_id','predecessor_identity','successor']::text[]
 OR pg_catalog.jsonb_typeof(p_request->'operation_id') IS DISTINCT FROM 'string'
 OR p_request->>'operation_id' !~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{{0,127}}$'
 OR pg_catalog.jsonb_typeof(p_request->'expected_generation') IS DISTINCT FROM 'number'
 OR p_request->>'expected_generation' !~ '^[1-9][0-9]*$'
 OR pg_catalog.jsonb_typeof(p_request->'predecessor_identity') IS DISTINCT FROM 'string'
 OR p_request->>'predecessor_identity'=''
 THEN RAISE EXCEPTION 'exact closed successor enrollment required' USING ERRCODE='22023'; END IF;
 successor:=p_request->'successor'; op:=p_request->>'operation_id';
 IF pg_catalog.jsonb_typeof(successor) IS DISTINCT FROM 'object'
 OR (SELECT pg_catalog.array_agg(k ORDER BY k) FROM pg_catalog.jsonb_object_keys(successor) k) IS DISTINCT FROM {binding_keys}
 OR pg_catalog.jsonb_typeof(successor->'identity') IS DISTINCT FROM 'string' OR successor->>'identity'=''
 OR pg_catalog.jsonb_typeof(successor->'semantic_role') IS DISTINCT FROM 'string'
 OR pg_catalog.jsonb_typeof(successor->'authority_identity') IS DISTINCT FROM 'string'
 OR pg_catalog.jsonb_typeof(successor->'public_material_identity') IS DISTINCT FROM 'string'
 OR successor->>'public_material_identity' !~ '^sha256:[0-9a-f]{{64}}$'
 OR pg_catalog.jsonb_typeof(successor->'lifecycle_generation') IS DISTINCT FROM 'number'
 OR successor->>'lifecycle_generation' !~ '^[1-9][0-9]*$'
 OR pg_catalog.jsonb_typeof(successor->'key_version') IS DISTINCT FROM 'number'
 OR successor->>'key_version' !~ '^[1-9][0-9]*$'
 THEN RAISE EXCEPTION 'exact public successor binding required' USING ERRCODE='22023'; END IF;
 PERFORM 1 FROM {s}.slots WHERE semantic_role=successor->>'semantic_role' FOR UPDATE;
 IF NOT FOUND THEN RAISE EXCEPTION 'installed lifecycle slot missing' USING ERRCODE='23514'; END IF;
 SELECT request,request_sha256 INTO prior,digest FROM {s}.operations WHERE operation_id=op;
 IF FOUND THEN
  IF prior IS DISTINCT FROM p_request THEN RAISE EXCEPTION 'enrollment operation conflicts' USING ERRCODE='23505'; END IF;
  RETURN pg_catalog.jsonb_build_object('operation_id',op,'request_sha256',digest,'phase','PENDING_CLOSED');
 END IF;
 SELECT * INTO STRICT f FROM {s}.fences WHERE semantic_role=successor->>'semantic_role'
 AND identity=p_request->>'predecessor_identity' FOR UPDATE;
 IF f.state NOT IN ('VERIFY_ONLY','REVOKED') OR f.pending_operation_id IS NOT NULL
 OR f.generation IS DISTINCT FROM (p_request->>'expected_generation')::bigint
 OR f.authority_identity IS DISTINCT FROM successor->>'authority_identity'
 OR f.identity=successor->>'identity' OR f.public_material_identity=successor->>'public_material_identity'
 OR f.generation>=9223372036854775807 OR (successor->>'lifecycle_generation')::bigint<>f.generation+1
 OR (successor->>'key_version')::bigint<=f.key_version
 OR (SELECT identity FROM {s}.slots WHERE semantic_role=f.semantic_role) IS DISTINCT FROM f.identity
 THEN RAISE EXCEPTION 'rotation requires retired predecessor and distinct successor' USING ERRCODE='23514'; END IF;
 digest:=pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(p_request::text,'UTF8')),'hex');
 INSERT INTO {s}.operations(operation_id,semantic_role,identity,request,request_sha256)
 VALUES(op,f.semantic_role,successor->>'identity',p_request,digest);
 INSERT INTO {s}.fences VALUES(f.semantic_role,successor->>'identity',f.authority_identity,
  successor->>'public_material_identity',(successor->>'key_version')::bigint,
  (successor->>'lifecycle_generation')::bigint,'PENDING_CLOSED',op);
 UPDATE {s}.slots SET identity=successor->>'identity' WHERE semantic_role=f.semantic_role;
 RETURN pg_catalog.jsonb_build_object('operation_id',op,'request_sha256',digest,'phase','PENDING_CLOSED');
END
""",
        "assert_active_fences(jsonb)": f"""
DECLARE role_name text; f {s}.fences%ROWTYPE; expected jsonb;
BEGIN
 {issuer}
 IF pg_catalog.current_setting('transaction_isolation')<>'serializable'
 OR pg_catalog.current_setting('transaction_read_only')<>'off' THEN
  RAISE EXCEPTION 'issuer backend requires writable SERIALIZABLE transaction' USING ERRCODE='25001'; END IF;
 IF p_expected IS NULL OR pg_catalog.jsonb_typeof(p_expected)<>'array'
 OR pg_catalog.jsonb_array_length(p_expected)<>3
 OR (SELECT pg_catalog.array_agg(e->>'semantic_role' ORDER BY e->>'semantic_role')
  FROM pg_catalog.jsonb_array_elements(p_expected) e)
 IS DISTINCT FROM ARRAY['HISTORY_ATTESTATION_SIGNING','ROOT_PROOF_SIGNING','TRUST_ACTIVATION']::text[]
 THEN RAISE EXCEPTION 'all three exact independent lifecycle fences required' USING ERRCODE='23514'; END IF;
 -- Every administrator locks the slot before its fence. No requester/claimant
 -- lock may be taken before this call in the enclosing atomic transaction.
 PERFORM 1 FROM {s}.slots ORDER BY semantic_role FOR UPDATE;
 IF (SELECT pg_catalog.array_agg(semantic_role ORDER BY semantic_role) FROM {s}.slots)
 IS DISTINCT FROM ARRAY['HISTORY_ATTESTATION_SIGNING','ROOT_PROOF_SIGNING','TRUST_ACTIVATION']::text[] THEN
  RAISE EXCEPTION 'complete installed lifecycle slots required' USING ERRCODE='23514'; END IF;
 FOR role_name IN SELECT semantic_role FROM {s}.slots ORDER BY semantic_role LOOP
  SELECT f0.* INTO STRICT f FROM {s}.fences f0 JOIN {s}.slots z USING(semantic_role,identity)
  WHERE f0.semantic_role=role_name FOR UPDATE OF f0;
  SELECT e INTO STRICT expected FROM pg_catalog.jsonb_array_elements(p_expected) e WHERE e->>'semantic_role'=role_name;
  IF (SELECT pg_catalog.array_agg(k ORDER BY k) FROM pg_catalog.jsonb_object_keys(expected) k) IS DISTINCT FROM {binding_keys}
  OR expected IS DISTINCT FROM pg_catalog.jsonb_build_object('semantic_role',f.semantic_role,
   'authority_identity',f.authority_identity,'identity',f.identity,'public_material_identity',f.public_material_identity,
   'key_version',f.key_version,'lifecycle_generation',f.generation)
  OR f.state<>'ACTIVE' OR f.pending_operation_id IS NOT NULL THEN
   RAISE EXCEPTION 'current production lifecycle fence closed or mismatched' USING ERRCODE='23514'; END IF;
 END LOOP;
 RETURN pg_catalog.jsonb_build_object('backend_pid',pg_catalog.pg_backend_pid(),'transaction_id',pg_catalog.pg_current_xact_id()::text,
  'session_user',session_user::text,'fences',p_expected);
END
""",
        "read_historical_fence(text,text,bigint)": f"""
DECLARE f {s}.fences%ROWTYPE; retained text;
BEGIN
 PERFORM {s}.assert_catalog();
 IF session_user::text NOT IN ({_literal(config.issuer_role)},{_literal(config.administrator_role)}) THEN
  RAISE EXCEPTION 'direct production reader required' USING ERRCODE='42501'; END IF;
 PERFORM 1 FROM {s}.slots WHERE semantic_role=p_role FOR SHARE;
 SELECT * INTO f FROM {s}.fences WHERE semantic_role=p_role AND identity=p_identity FOR SHARE;
 IF NOT FOUND OR f.state NOT IN ('ACTIVE','VERIFY_ONLY') THEN RETURN NULL; END IF;
 SELECT state INTO retained FROM {s}.observations WHERE semantic_role=p_role AND identity=p_identity AND generation=p_generation;
 IF retained NOT IN ('ACTIVE','VERIFY_ONLY') OR retained IS NULL THEN RETURN NULL; END IF;
 RETURN pg_catalog.jsonb_build_object('semantic_role',f.semantic_role,'identity',f.identity,
  'public_material_identity',f.public_material_identity,'key_version',f.key_version,
  'historical_generation',p_generation,'historical_state',retained,'current_generation',f.generation,
  'current_state',f.state,'new_operations_authorized',false);
END
""",
        "observe_signer_custody(text,text)": f"""
DECLARE f {s}.fences%ROWTYPE; current_identity text;
BEGIN
 {issuer}
 IF pg_catalog.current_setting('transaction_isolation')<>'serializable'
 OR pg_catalog.current_setting('transaction_read_only')<>'off' THEN
  RAISE EXCEPTION 'signer observation requires writable SERIALIZABLE transaction' USING ERRCODE='25001'; END IF;
 IF p_role IS NULL OR p_role NOT IN ('ROOT_PROOF_SIGNING','HISTORY_ATTESTATION_SIGNING')
 OR p_identity IS NULL OR p_identity='' THEN
  RAISE EXCEPTION 'exact signing role and public identity required' USING ERRCODE='22023'; END IF;
 -- Public diagnostic facts only. Match the administrator's slot-before-fence
 -- lock order, including when the requested fence is a retired predecessor.
 SELECT identity INTO current_identity FROM {s}.slots WHERE semantic_role=p_role FOR SHARE;
 IF NOT FOUND THEN RETURN NULL; END IF;
 SELECT * INTO f FROM {s}.fences WHERE semantic_role=p_role AND identity=p_identity FOR SHARE;
 IF NOT FOUND THEN RETURN NULL; END IF;
 RETURN pg_catalog.jsonb_build_object('semantic_role',f.semantic_role,
  'authority_identity',f.authority_identity,'identity',f.identity,
  'public_material_identity',f.public_material_identity,'key_version',f.key_version,
  'lifecycle_generation',f.generation,'state',f.state,'pending_operation_id',f.pending_operation_id,
  'is_current',current_identity=f.identity,'new_operations_authorized',false);
END
""",
    }


def _validate_binding(binding: object) -> None:
    if type(binding) is not dict or set(binding) != _BINDING_KEYS:
        raise ValueError("exact public lifecycle binding fields required")
    if binding["semantic_role"] not in SEMANTIC_ROLES:
        raise ValueError("installed signing or trust activation role required")
    for key in ("authority_identity", "identity"):
        if type(binding[key]) is not str or not binding[key].strip() or len(binding[key]) > 4096:
            raise ValueError("exact public authority and credential identities required")
    if (
        type(binding["public_material_identity"]) is not str
        or _MATERIAL.fullmatch(binding["public_material_identity"]) is None
    ):
        raise ValueError("public material fingerprint required")
    for key in ("key_version", "lifecycle_generation"):
        if type(binding[key]) is not int or not 1 <= binding[key] < 2**63 - 1:
            raise ValueError("positive exact lifecycle generation and key version required")


def install_production_root_lifecycle_authority(
    conn: psycopg.Connection[Any],
    config: PostgreSQLProductionRootLifecycleProvisioning,
    credential_bindings: tuple[dict[str, Any], ...],
    *,
    bootstrap_operation_id: str,
) -> None:
    """Explicit trusted bootstrap installs closed public facts, never keys/activation.

    Must run in a caller-owned SERIALIZABLE transaction. The caller commits and
    establishes a separately retained qualification pin after installation.
    """
    if type(config) is not PostgreSQLProductionRootLifecycleProvisioning:
        raise TypeError("exact production provisioning required")
    if conn.info.transaction_status != psycopg.pq.TransactionStatus.INTRANS:
        raise ValueError("explicit bootstrap transaction required")
    if (
        type(bootstrap_operation_id) is not str
        or _OPERATION.fullmatch(bootstrap_operation_id) is None
    ):
        raise ValueError("immutable bootstrap operation identity required")
    if type(credential_bindings) is not tuple or len(credential_bindings) != 3:
        raise ValueError("all three public production bindings required")
    for binding in credential_bindings:
        _validate_binding(binding)
    if {b["semantic_role"] for b in credential_bindings} != SEMANTIC_ROLES or any(
        len({b[key] for b in credential_bindings}) != 3
        for key in ("authority_identity", "identity", "public_material_identity")
    ):
        raise ValueError("root, history and trust identities/material must be independent")
    principal = conn.execute(
        "SELECT session_user::text,current_user::text,(SELECT rolsuper FROM pg_catalog.pg_roles WHERE rolname=session_user::text),"
        "current_setting('transaction_isolation'),current_setting('server_version_num')::integer"
    ).fetchone()
    if (
        principal is None
        or principal[0] != principal[1]
        or not principal[2]
        or principal[3] != "serializable"
        or not 160000 <= principal[4] <= 169999
    ):
        raise ValueError("separate direct PostgreSQL 16 bootstrap superuser required")
    if principal[0] in {config.schema_owner_role, config.issuer_role, config.administrator_role}:
        raise ValueError("bootstrap principal must be distinct from production authorities")
    s = sql.Identifier(config.schema)
    for name, login in (
        (config.schema_owner_role, False),
        (config.issuer_role, True),
        (config.administrator_role, True),
    ):
        conn.execute(
            sql.SQL(
                "CREATE ROLE {} {} NOINHERIT NOSUPERUSER NOCREATEROLE NOCREATEDB NOREPLICATION NOBYPASSRLS"
            ).format(sql.Identifier(name), sql.SQL("LOGIN" if login else "NOLOGIN"))
        )
    for name in (config.issuer_role, config.administrator_role):
        if conn.execute(
            "SELECT has_database_privilege(%s,current_database(),'CREATE'),"
            "has_database_privilege(%s,current_database(),'TEMPORARY')",
            (name, name),
        ).fetchone() != (False, False):
            raise ValueError("production database CREATE/TEMPORARY must already be closed")
    conn.execute(
        sql.SQL("CREATE SCHEMA {} AUTHORIZATION {}").format(
            s, sql.Identifier(config.schema_owner_role)
        )
    )
    conn.execute(sql.SQL("REVOKE ALL ON SCHEMA {} FROM PUBLIC").format(s))
    conn.execute(
        sql.SQL("GRANT USAGE ON SCHEMA {} TO {},{}").format(
            s, sql.Identifier(config.issuer_role), sql.Identifier(config.administrator_role)
        )
    )
    conn.execute(
        sql.SQL("""
CREATE TABLE {s}.metadata(singleton boolean PRIMARY KEY CHECK(singleton),schema_identity text NOT NULL,
 schema_version integer NOT NULL,profile text NOT NULL,boundary text NOT NULL,installation_id text NOT NULL,
 trust_domain text NOT NULL,system_identifier bigint NOT NULL,database_oid oid NOT NULL,
 owner_role_oid oid NOT NULL,issuer_role_oid oid NOT NULL,administrator_role_oid oid NOT NULL);
CREATE TABLE {s}.catalog_manifest(singleton boolean PRIMARY KEY CHECK(singleton),snapshot jsonb NOT NULL);
CREATE TABLE {s}.fences(semantic_role text NOT NULL CHECK(semantic_role IN
 ('ROOT_PROOF_SIGNING','HISTORY_ATTESTATION_SIGNING','TRUST_ACTIVATION')),identity text NOT NULL UNIQUE,
 authority_identity text NOT NULL,public_material_identity text NOT NULL UNIQUE CHECK(public_material_identity ~ '^sha256:[0-9a-f]{{64}}$'),
 key_version bigint NOT NULL CHECK(key_version>0),generation bigint NOT NULL CHECK(generation>0),
 state text NOT NULL CHECK(state IN ('ACTIVE','VERIFY_ONLY','REVOKED','PENDING_CLOSED')),
 pending_operation_id text,PRIMARY KEY(semantic_role,identity));
CREATE TABLE {s}.slots(semantic_role text PRIMARY KEY,identity text NOT NULL,
 FOREIGN KEY(semantic_role,identity) REFERENCES {s}.fences(semantic_role,identity));
CREATE TABLE {s}.operations(operation_id text PRIMARY KEY,semantic_role text NOT NULL,identity text NOT NULL,
 request jsonb NOT NULL,request_sha256 text NOT NULL CHECK(request_sha256 ~ '^[0-9a-f]{{64}}$'),
 close_xid xid8 NOT NULL DEFAULT pg_catalog.pg_current_xact_id());
CREATE TABLE {s}.completions(operation_id text PRIMARY KEY REFERENCES {s}.operations(operation_id),
 receipt_bytes bytea NOT NULL,receipt_sha256 bytea NOT NULL CHECK(octet_length(receipt_sha256)=32));
CREATE TABLE {s}.observations(semantic_role text NOT NULL,identity text NOT NULL,generation bigint NOT NULL,
 state text NOT NULL CHECK(state IN ('ACTIVE','VERIFY_ONLY','REVOKED')),PRIMARY KEY(semantic_role,identity,generation),
 FOREIGN KEY(semantic_role,identity) REFERENCES {s}.fences(semantic_role,identity));
""").format(s=s)
    )
    conn.execute(
        sql.SQL(
            "INSERT INTO {}.metadata SELECT true,%s,%s,'PRODUCTION_LOCAL',%s,%s,%s,"
            "(SELECT system_identifier FROM pg_catalog.pg_control_system()),(SELECT oid FROM pg_catalog.pg_database WHERE datname=current_database()),"
            "(SELECT oid FROM pg_catalog.pg_roles WHERE rolname=%s),(SELECT oid FROM pg_catalog.pg_roles WHERE rolname=%s),"
            "(SELECT oid FROM pg_catalog.pg_roles WHERE rolname=%s)"
        ).format(s),
        (
            SCHEMA_IDENTITY,
            SCHEMA_VERSION,
            BOUNDARY,
            config.installation_id,
            config.trust_domain,
            config.schema_owner_role,
            config.issuer_role,
            config.administrator_role,
        ),
    )
    for binding in credential_bindings:
        operation_id = bootstrap_operation_id + ":" + binding["semantic_role"]
        if len(operation_id) > 128:
            raise ValueError("bootstrap operation identity exceeds lifecycle limit")
        request = {"operation_id": operation_id, "kind": "BOOTSTRAP_CLOSED", "binding": binding}
        conn.execute(
            sql.SQL(
                "INSERT INTO {}.operations(operation_id,semantic_role,identity,request,request_sha256) "
                "SELECT %s,%s,%s,%s,encode(sha256(convert_to(%s::jsonb::text,'UTF8')),'hex')"
            ).format(s),
            (
                operation_id,
                binding["semantic_role"],
                binding["identity"],
                Jsonb(request),
                Jsonb(request),
            ),
        )
        conn.execute(
            sql.SQL("INSERT INTO {}.fences VALUES(%s,%s,%s,%s,%s,%s,'PENDING_CLOSED',%s)").format(
                s
            ),
            (
                binding["semantic_role"],
                binding["identity"],
                binding["authority_identity"],
                binding["public_material_identity"],
                binding["key_version"],
                binding["lifecycle_generation"],
                operation_id,
            ),
        )
        conn.execute(
            sql.SQL("INSERT INTO {}.slots VALUES(%s,%s)").format(s),
            (binding["semantic_role"], binding["identity"]),
        )
    sources = reviewed_production_lifecycle_function_sources(config)
    for signature, source in sources.items():
        declaration, returns = PRODUCTION_LIFECYCLE_FUNCTION_DECLARATIONS[signature]
        conn.execute(
            sql.SQL(
                "CREATE FUNCTION {}.{}({}) RETURNS {} LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS {}"
            ).format(
                s,
                sql.Identifier(signature.split("(")[0]),
                sql.SQL(declaration),
                sql.SQL(returns),
                sql.Literal(source),
            )
        )
        conn.execute(
            sql.SQL("ALTER FUNCTION {}.{} OWNER TO {}").format(
                s, sql.SQL(signature), sql.Identifier(config.schema_owner_role)
            )
        )
        conn.execute(
            sql.SQL("REVOKE ALL ON FUNCTION {}.{} FROM PUBLIC").format(s, sql.SQL(signature))
        )
        for role, signatures in (
            (config.issuer_role, ISSUER_FUNCTIONS),
            (config.administrator_role, ADMINISTRATOR_FUNCTIONS),
        ):
            if signature in signatures:
                conn.execute(
                    sql.SQL("GRANT EXECUTE ON FUNCTION {}.{} TO {}").format(
                        s, sql.SQL(signature), sql.Identifier(role)
                    )
                )
    for table in sorted(RELATIONS):
        conn.execute(
            sql.SQL("ALTER TABLE {}.{} OWNER TO {}").format(
                s, sql.Identifier(table), sql.Identifier(config.schema_owner_role)
            )
        )
        conn.execute(
            sql.SQL("REVOKE ALL ON {}.{} FROM PUBLIC,{},{}").format(
                s,
                sql.Identifier(table),
                sql.Identifier(config.issuer_role),
                sql.Identifier(config.administrator_role),
            )
        )
        if table in {"metadata", "catalog_manifest"}:
            conn.execute(
                sql.SQL("GRANT SELECT ON {}.{} TO {},{}").format(
                    s,
                    sql.Identifier(table),
                    sql.Identifier(config.issuer_role),
                    sql.Identifier(config.administrator_role),
                )
            )
        if table not in {"fences", "slots"}:
            conn.execute(
                sql.SQL(
                    "CREATE TRIGGER immutable_row BEFORE UPDATE OR DELETE ON {}.{} FOR EACH ROW EXECUTE FUNCTION {}.reject_mutation()"
                ).format(s, sql.Identifier(table), s)
            )
            conn.execute(
                sql.SQL("ALTER TABLE {}.{} ENABLE ALWAYS TRIGGER immutable_row").format(
                    s, sql.Identifier(table)
                )
            )
        conn.execute(
            sql.SQL(
                "CREATE TRIGGER immutable_truncate BEFORE TRUNCATE ON {}.{} FOR EACH STATEMENT EXECUTE FUNCTION {}.reject_mutation()"
            ).format(s, sql.Identifier(table), s)
        )
        conn.execute(
            sql.SQL("ALTER TABLE {}.{} ENABLE ALWAYS TRIGGER immutable_truncate").format(
                s, sql.Identifier(table)
            )
        )
    conn.execute(
        sql.SQL(
            "CREATE TRIGGER lifecycle_update BEFORE UPDATE OR DELETE ON {}.fences FOR EACH ROW EXECUTE FUNCTION {}.guard_fence_update()"
        ).format(s, s)
    )
    conn.execute(sql.SQL("ALTER TABLE {}.fences ENABLE ALWAYS TRIGGER lifecycle_update").format(s))
    conn.execute(
        sql.SQL(
            "CREATE TRIGGER lifecycle_slot_update BEFORE UPDATE OR DELETE ON {}.slots "
            "FOR EACH ROW EXECUTE FUNCTION {}.guard_slot_update()"
        ).format(s, s)
    )
    conn.execute(
        sql.SQL("ALTER TABLE {}.slots ENABLE ALWAYS TRIGGER lifecycle_slot_update").format(s)
    )
    # pg_control_system discloses public physical identity only. Its EXECUTE
    # grant is narrow and required by the owner-owned live guard.
    conn.execute(
        sql.SQL("GRANT EXECUTE ON FUNCTION pg_catalog.pg_control_system() TO {}").format(
            sql.Identifier(config.schema_owner_role)
        )
    )
    conn.execute(
        sql.SQL("INSERT INTO {}.catalog_manifest SELECT true,{}.catalog_snapshot()").format(s, s)
    )
