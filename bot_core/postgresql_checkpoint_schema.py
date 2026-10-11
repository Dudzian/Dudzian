"""Instalacja offline i kwalifikacja trwałego lokalnego checkpointu V2.

Checkpoint i cała historia przyjęć zachowują dokładne kanoniczne bajty.
Zaufany runtime weryfikuje kryptografię, historię i custody przed CAS; uprawnienie
EXECUTE nie jest dowodem autoryzacji kryptograficznej. PostgreSQL i lokalny host
nie zapewniają niezależnej ochrony przed ich skoordynowanym rollbackiem.
W bazie nie są przechowywane prywatne klucze ani capabilities wywołującego.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg import sql

from bot_core.authenticated_issuer_history import (
    ACCEPTANCE_DOMAIN,
    ATTESTATION_DOMAIN,
    HistoryStreamIdentity,
    canonical_json_bytes,
)
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core.postgresql_issuer_history_canonical_sql import (
    CANONICAL_FUNCTION_RETURNS,
    canonical_function_sources,
)

SCHEMA_IDENTITY = "cryptohunter.issuer_history_checkpoint.postgresql"
SCHEMA_VERSION = 1
STORAGE_FAMILY = "POSTGRESQL_ISSUER_HISTORY_CHECKPOINT_V2"
PROFILE = "PRODUCTION_LOCAL"
_SAFE_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]{0,62}\Z")
_TABLES = ("metadata", "checkpoints", "entries")
_FUNCTION_CALLERS = {
    **dict.fromkeys(CANONICAL_FUNCTION_RETURNS),
    "validate_identifier(bytea)": None,
    "validate_stream(bytea)": None,
    "validate_credential(bytea)": None,
    "validate_entry(bytea,bytea,text,bigint,text,bigint,bytea,bytea,bytea,bytea)": None,
    "verify_checkpoint(bytea,bytea)": None,
    "provision_stream(bytea,bytea)": "admin_role",
    "read_checkpoint(bytea,bytea,boolean)": "runtime_role",
    "commit_checkpoint(bytea,bytea,bigint,bytea,bytea,bytea,bytea)": "runtime_role",
}


class CheckpointSchemaQualificationError(RuntimeError):
    """Baza nie spełnia przejrzanej definicji autorytetu checkpointu."""


@dataclass(frozen=True, slots=True)
class PostgreSQLCheckpointProvisioning:
    schema: str
    schema_owner_role: str
    runtime_role: str
    admin_role: str
    trust_domain: str

    def __post_init__(self) -> None:
        for name in ("schema", "schema_owner_role", "runtime_role", "admin_role"):
            _identifier(getattr(self, name))
        if len({self.schema_owner_role, self.runtime_role, self.admin_role}) != 3:
            raise ValueError("checkpoint owner, runtime and admin roles must be distinct")
        if type(self.trust_domain) is not str or not self.trust_domain:
            raise ValueError("trust_domain must be an exact non-empty str")


def _identifier(value: str) -> str:
    if type(value) is not str or _SAFE_IDENTIFIER.fullmatch(value) is None:
        raise ValueError("identifier must be a safe lowercase PostgreSQL identifier")
    return value


def _scalar(value: str) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _source_fingerprint(signature: str, source: str) -> str:
    return hashlib.sha256(
        b"CryptoHunter/PostgreSQLCheckpointFunction/v1\0"
        + signature.encode("ascii")
        + b"\0"
        + source.encode("utf-8")
    ).hexdigest()


def _principal_guard(*, login_role: str, schema_owner_role: str) -> str:
    return (
        f"IF session_user <> '{login_role}' OR current_user <> "
        f"'{schema_owner_role}' THEN RAISE EXCEPTION 'wrong checkpoint principal' "
        "USING ERRCODE='42501'; END IF;"
        " IF pg_catalog.current_setting('server_version_num')::integer < 160000 "
        "OR pg_catalog.current_setting('fsync') <> 'on' "
        "OR pg_catalog.current_setting('synchronous_commit') <> 'on' THEN "
        "RAISE EXCEPTION 'checkpoint durability settings mismatch' USING ERRCODE='42501'; END IF;"
    )


def _function_sources(config: PostgreSQLCheckpointProvisioning) -> dict[str, str]:
    s = '"' + config.schema + '"'
    runtime_guard = _principal_guard(
        login_role=config.runtime_role, schema_owner_role=config.schema_owner_role
    )
    admin_guard = _principal_guard(
        login_role=config.admin_role, schema_owner_role=config.schema_owner_role
    )
    sources = {
        "validate_identifier(bytea)": """
DECLARE item record;
BEGIN
 PERFORM pg_catalog.convert_from(p_id,'UTF8');
 SELECT * INTO STRICT item FROM {s}.canonical_string(p_id,0);
 IF item.next_position IS DISTINCT FROM pg_catalog.octet_length(p_id)
 OR pg_catalog.octet_length(item.decoded_key)=0 THEN
  RAISE EXCEPTION 'malformed checkpoint identity' USING ERRCODE='22023'; END IF;
END
""",
        "validate_stream(bytea)": """
DECLARE field_index integer; members bytea[]; member bytea;
BEGIN
 members := {s}.canonical_object_members(p_stream,ARRAY['environment','issuer_authority_identity',
  'product_scope','security_epoch','security_profile','stream_id','trust_domain']);
 FOREACH field_index IN ARRAY ARRAY[1,2,3,5,6,7] LOOP
  member := members[field_index];
  IF pg_catalog.get_byte(member,0) <> 34 OR pg_catalog.octet_length(member)=2 THEN
   RAISE EXCEPTION 'malformed checkpoint stream field' USING ERRCODE='22023'; END IF;
 END LOOP;
 IF members[5] IS DISTINCT FROM pg_catalog.convert_to('\"PRODUCTION_LOCAL\"','UTF8')
 OR pg_catalog.convert_from(members[4],'UTF8') !~ '^[1-9][0-9]*$'
 OR members[7] IS DISTINCT FROM
    (SELECT trust_domain FROM {s}.metadata WHERE singleton)
 THEN RAISE EXCEPTION 'checkpoint stream security scope mismatch' USING ERRCODE='22023'; END IF;
 RETURN members;
END
""",
        "validate_credential(bytea)": """
DECLARE members bytea[]; field_index integer; member bytea;
BEGIN
 members := {s}.canonical_object_members(p_credential,ARRAY['credential_identity',
  'custody_lifecycle_namespace','key_handle_or_version','key_material_identity',
  'provider_namespace','semantic_role']);
 IF members[6] IS DISTINCT FROM pg_catalog.convert_to('\"HISTORY_ATTESTATION_SIGNING\"','UTF8')
 THEN RAISE EXCEPTION 'wrong retained checkpoint credential role' USING ERRCODE='22023'; END IF;
 FOREACH field_index IN ARRAY ARRAY[1,2,5] LOOP
  member := members[field_index];
  IF pg_catalog.get_byte(member,0) <> 34 OR pg_catalog.octet_length(member)=2 THEN
   RAISE EXCEPTION 'malformed retained checkpoint credential' USING ERRCODE='22023'; END IF;
 END LOOP;
 FOREACH field_index IN ARRAY ARRAY[3,4] LOOP
  member := members[field_index];
  IF member IS DISTINCT FROM pg_catalog.convert_to('null','UTF8') AND
   (pg_catalog.get_byte(member,0) <> 34 OR pg_catalog.octet_length(member)=2) THEN
   RAISE EXCEPTION 'malformed retained checkpoint credential' USING ERRCODE='22023'; END IF;
 END LOOP;
END
""",
        "validate_entry(bytea,bytea,text,bigint,text,bigint,bytea,bytea,bytea,bytea)": """
DECLARE cp bytea[]; ev bytea[]; hd bytea[]; material bytea := pg_catalog.convert_to('{','UTF8');
 fields text[] := ARRAY['acceptance_digest','accepted_authenticated_head_digest',
  'accepted_history_sequence','checkpoint_authority_identity','checkpoint_revision',
  'history_signing_credential_identity','lifecycle_generation_at_acceptance',
  'lifecycle_state_at_acceptance','predecessor_acceptance_digest','stream'];
 field_index integer; sequence bigint; body bytea; digest text;
BEGIN
 cp := {s}.canonical_object_members(p_checkpoint,ARRAY['authenticated_head_digest',
  'checkpoint_id','checkpoint_revision','history_sequence','history_signing_credential_identity',
  'stream']);
 ev := {s}.canonical_object_members(p_acceptance,fields);
 IF pg_catalog.substring(p_head,1,{attestation_length}) IS DISTINCT FROM
  pg_catalog.decode('{attestation_domain}','hex')
 OR pg_catalog.octet_length(p_signature) IS DISTINCT FROM 64 THEN
  RAISE EXCEPTION 'malformed retained checkpoint head' USING ERRCODE='22023'; END IF;
 body := pg_catalog.substring(p_head,{attestation_offset});
 hd := {s}.canonical_object_members(body,
  ARRAY['record_digest','sequence','signing_credential_identity','stream']);
 PERFORM {s}.validate_credential(cp[5]);
 IF pg_catalog.convert_from(cp[4],'UTF8') !~ '^[1-9][0-9]*$'
 OR pg_catalog.convert_from(ev[7],'UTF8') !~ '^[1-9][0-9]*$'
 OR pg_catalog.convert_from(cp[1],'UTF8') !~ '^"sha256:[0-9a-f]{64}"$'
 THEN RAISE EXCEPTION 'malformed retained checkpoint numbers or digest' USING ERRCODE='22023';
 END IF;
 sequence := pg_catalog.convert_from(cp[4],'UTF8')::bigint;
 IF sequence <= p_previous_sequence
 OR cp[2] IS DISTINCT FROM p_id
 OR cp[3] IS DISTINCT FROM pg_catalog.convert_to(p_revision::text,'UTF8')
 OR cp[6] IS DISTINCT FROM p_stream
 OR ev[2] IS DISTINCT FROM cp[1]
 OR ev[3] IS DISTINCT FROM cp[4]
 OR ev[4] IS DISTINCT FROM pg_catalog.convert_to(pg_catalog.to_json(p_authority)::text,'UTF8')
 OR ev[5] IS DISTINCT FROM cp[3]
 OR ev[6] IS DISTINCT FROM cp[5]
 OR ev[8] IS DISTINCT FROM pg_catalog.convert_to('"ACTIVE"','UTF8')
 OR ev[9] IS DISTINCT FROM pg_catalog.convert_to(pg_catalog.to_json(p_predecessor)::text,'UTF8')
 OR ev[10] IS DISTINCT FROM p_stream
 OR hd[1] IS DISTINCT FROM cp[1] OR hd[2] IS DISTINCT FROM cp[4]
 OR hd[3] IS DISTINCT FROM cp[5] OR hd[4] IS DISTINCT FROM p_stream THEN
  RAISE EXCEPTION 'retained checkpoint evidence relation corrupt' USING ERRCODE='22023'; END IF;
 FOR field_index IN 2..10 LOOP
  IF field_index>2 THEN material := material || pg_catalog.convert_to(',','UTF8'); END IF;
  material := material || pg_catalog.convert_to(pg_catalog.to_json(fields[field_index])::text
    || ':','UTF8') || ev[field_index];
 END LOOP;
 material := material || pg_catalog.convert_to('}','UTF8');
 digest := 'sha256:' || pg_catalog.encode(pg_catalog.sha256(
  pg_catalog.decode('{acceptance_domain}','hex') || material),'hex');
 IF ev[1] IS DISTINCT FROM pg_catalog.convert_to(pg_catalog.to_json(digest)::text,'UTF8') THEN
  RAISE EXCEPTION 'retained checkpoint acceptance digest corrupt' USING ERRCODE='22023'; END IF;
END
""",
        "verify_checkpoint(bytea,bytea)": """
DECLARE st {s}.checkpoints%ROWTYPE; e {s}.entries%ROWTYPE;
 revision bigint := 0; previous_sequence bigint := 0; predecessor text := 'NO_PREDECESSOR';
BEGIN
 PERFORM {s}.validate_identifier(p_id);
 PERFORM {s}.validate_stream(p_stream);
 SELECT * INTO st FROM {s}.checkpoints WHERE checkpoint_id=p_id;
 IF NOT FOUND THEN RAISE EXCEPTION 'unprovisioned checkpoint' USING ERRCODE='42501'; END IF;
 IF st.stream_identity IS DISTINCT FROM p_stream
 OR st.authority_identity !~ '{authority_pattern}'
 THEN RAISE EXCEPTION 'checkpoint authority identity corrupt' USING ERRCODE='22023'; END IF;
 FOR e IN SELECT stored.* FROM {s}.entries stored WHERE stored.checkpoint_id=p_id
 ORDER BY stored.revision LOOP
  revision := revision+1;
  IF e.revision IS DISTINCT FROM revision THEN
   RAISE EXCEPTION 'checkpoint revision chain discontinuity' USING ERRCODE='22023'; END IF;
  PERFORM {s}.validate_entry(p_id,p_stream,st.authority_identity,revision,predecessor,
   previous_sequence,e.checkpoint,e.acceptance,e.head,e.signature);
  predecessor := pg_catalog.convert_from(
   {s}.canonical_object_member(e.acceptance,'acceptance_digest'),'UTF8')::jsonb #>> '{}';
  previous_sequence := pg_catalog.convert_from(
   {s}.canonical_object_member(e.checkpoint,'history_sequence'),'UTF8')::bigint;
 END LOOP;
 IF st.revision IS DISTINCT FROM revision THEN
  RAISE EXCEPTION 'checkpoint revision projection corrupt' USING ERRCODE='22023'; END IF;
END
""",
        "provision_stream(bytea,bytea)": """
BEGIN
 {admin_guard}
 PERFORM {s}.validate_identifier(p_id);
 PERFORM {s}.validate_stream(p_stream);
 INSERT INTO {s}.checkpoints(checkpoint_id,stream_identity,authority_identity,revision)
 VALUES(p_id,p_stream,'postgresql-checkpoint-authority:' || pg_catalog.gen_random_uuid()::text,0);
END
""",
        "read_checkpoint(bytea,bytea,boolean)": """
DECLARE result jsonb;
BEGIN
 {runtime_guard}
 IF p_lock IS NULL THEN RAISE EXCEPTION 'missing checkpoint lock mode' USING ERRCODE='22023';
 END IF;
 IF p_lock THEN
  PERFORM 1 FROM {s}.checkpoints WHERE checkpoint_id=p_id FOR UPDATE;
 END IF;
 PERFORM {s}.verify_checkpoint(p_id,p_stream);
 SELECT pg_catalog.jsonb_build_object('checkpoint_id',pg_catalog.encode(st.checkpoint_id,'hex'),
  'stream',pg_catalog.encode(st.stream_identity,'hex'),'authority_identity',st.authority_identity,
  'revision',st.revision,'entries',COALESCE((SELECT pg_catalog.jsonb_agg(
   pg_catalog.jsonb_build_object('revision',e.revision,'checkpoint',
    pg_catalog.encode(e.checkpoint,'hex'),'acceptance',pg_catalog.encode(e.acceptance,'hex'),
    'head',pg_catalog.encode(e.head,'hex'),'signature',pg_catalog.encode(e.signature,'hex'))
   ORDER BY e.revision) FROM {s}.entries e WHERE e.checkpoint_id=p_id),'[]'::jsonb))
 INTO result FROM {s}.checkpoints st WHERE st.checkpoint_id=p_id;
 RETURN result;
END
""",
        "commit_checkpoint(bytea,bytea,bigint,bytea,bytea,bytea,bytea)": """
DECLARE st {s}.checkpoints%ROWTYPE; prior {s}.entries%ROWTYPE;
 predecessor text := 'NO_PREDECESSOR'; previous_sequence bigint := 0;
BEGIN
 {runtime_guard}
 SELECT * INTO st FROM {s}.checkpoints WHERE checkpoint_id=p_id FOR UPDATE;
 IF NOT FOUND THEN RAISE EXCEPTION 'unprovisioned checkpoint' USING ERRCODE='42501'; END IF;
 PERFORM {s}.verify_checkpoint(p_id,p_stream);
 IF p_expected IS DISTINCT FROM st.revision THEN
  RAISE EXCEPTION 'checkpoint CAS conflict' USING ERRCODE='40001'; END IF;
 IF st.revision>0 THEN
  SELECT * INTO STRICT prior FROM {s}.entries WHERE checkpoint_id=p_id AND revision=st.revision;
  IF prior.checkpoint IS NOT DISTINCT FROM p_checkpoint
  AND prior.acceptance IS NOT DISTINCT FROM p_acceptance
  AND prior.head IS NOT DISTINCT FROM p_head
  AND prior.signature IS NOT DISTINCT FROM p_signature THEN RETURN true; END IF;
  predecessor := pg_catalog.convert_from(
   {s}.canonical_object_member(prior.acceptance,'acceptance_digest'),'UTF8')::jsonb #>> '{}';
  previous_sequence := pg_catalog.convert_from(
   {s}.canonical_object_member(prior.checkpoint,'history_sequence'),'UTF8')::bigint;
 END IF;
 PERFORM {s}.validate_entry(p_id,p_stream,st.authority_identity,st.revision+1,predecessor,
  previous_sequence,p_checkpoint,p_acceptance,p_head,p_signature);
 INSERT INTO {s}.entries(checkpoint_id,revision,checkpoint,acceptance,head,signature)
 VALUES(p_id,st.revision+1,p_checkpoint,p_acceptance,p_head,p_signature);
 UPDATE {s}.checkpoints SET revision=revision+1 WHERE checkpoint_id=p_id;
 PERFORM {s}.verify_checkpoint(p_id,p_stream);
 RETURN false;
END
""",
    }
    replacements = {
        "{s}": s,
        "{runtime_guard}": runtime_guard,
        "{admin_guard}": admin_guard,
        "{attestation_domain}": ATTESTATION_DOMAIN.hex(),
        "{acceptance_domain}": ACCEPTANCE_DOMAIN.hex(),
        "{authority_pattern}": (
            "^postgresql-checkpoint-authority:[0-9a-f]{8}-[0-9a-f]{4}-"
            "[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        ),
        "{attestation_length}": str(len(ATTESTATION_DOMAIN)),
        "{attestation_offset}": str(len(ATTESTATION_DOMAIN) + 1),
    }
    for signature, source in sources.items():
        for placeholder, value in replacements.items():
            source = source.replace(placeholder, value)
        sources[signature] = source
    return {**canonical_function_sources(config.schema), **sources}


def _create_tables(conn: psycopg.Connection[Any], config: PostgreSQLCheckpointProvisioning) -> None:
    conn.execute(
        sql.SQL("""
CREATE TABLE {s}.metadata (
 singleton boolean PRIMARY KEY DEFAULT true CHECK(singleton),
 schema_identity text NOT NULL,schema_version integer NOT NULL,profile text NOT NULL,
 trust_domain bytea NOT NULL,storage_family text NOT NULL,
 schema_owner_role text NOT NULL,schema_owner_role_oid oid NOT NULL,
 runtime_role text NOT NULL,runtime_role_oid oid NOT NULL,
 admin_role text NOT NULL,admin_role_oid oid NOT NULL,
 function_definition_sha256 jsonb NOT NULL,physical_definition_sha256 text NOT NULL
);
CREATE TABLE {s}.checkpoints (
 checkpoint_id bytea PRIMARY KEY,stream_identity bytea NOT NULL,
 authority_identity text NOT NULL UNIQUE,revision bigint NOT NULL CHECK(revision>=0)
);
CREATE TABLE {s}.entries (
 checkpoint_id bytea NOT NULL REFERENCES {s}.checkpoints(checkpoint_id),
 revision bigint NOT NULL CHECK(revision>=1),checkpoint bytea NOT NULL,
 acceptance bytea NOT NULL,head bytea NOT NULL,
 signature bytea NOT NULL CHECK(octet_length(signature)=64),
 PRIMARY KEY(checkpoint_id,revision)
);
REVOKE ALL ON ALL TABLES IN SCHEMA {s} FROM PUBLIC;
""").format(s=sql.Identifier(config.schema))
    )


_FUNCTION_DECLARATIONS = {
    "canonical_object_members(bytea,text[])": ("p_bytes bytea,p_fields text[]", "bytea[]"),
    "canonical_object_member(bytea,text)": ("p_bytes bytea,p_key text", "bytea"),
    "canonical_string(bytea,integer)": (
        "p_bytes bytea,p_position integer",
        CANONICAL_FUNCTION_RETURNS["canonical_string(bytea,integer)"],
    ),
    "canonical_value(bytea,integer)": ("p_bytes bytea,p_position integer", "integer"),
    "validate_canonical_json(bytea)": ("p_bytes bytea", "void"),
    "validate_identifier(bytea)": ("p_id bytea", "void"),
    "validate_stream(bytea)": ("p_stream bytea", "bytea[]"),
    "validate_credential(bytea)": ("p_credential bytea", "void"),
    "validate_entry(bytea,bytea,text,bigint,text,bigint,bytea,bytea,bytea,bytea)": (
        "p_id bytea,p_stream bytea,p_authority text,p_revision bigint,p_predecessor text,"
        "p_previous_sequence bigint,p_checkpoint bytea,p_acceptance bytea,p_head bytea,"
        "p_signature bytea",
        "void",
    ),
    "verify_checkpoint(bytea,bytea)": ("p_id bytea,p_stream bytea", "void"),
    "provision_stream(bytea,bytea)": ("p_id bytea,p_stream bytea", "void"),
    "read_checkpoint(bytea,bytea,boolean)": ("p_id bytea,p_stream bytea,p_lock boolean", "jsonb"),
    "commit_checkpoint(bytea,bytea,bigint,bytea,bytea,bytea,bytea)": (
        "p_id bytea,p_stream bytea,p_expected bigint,p_checkpoint bytea,p_acceptance bytea,"
        "p_head bytea,p_signature bytea",
        "boolean",
    ),
}


def _physical_fingerprint(conn: psycopg.Connection[Any], schema: str) -> str:
    """Wylicza dodatkowy fingerprint fizycznej definicji instalacji."""
    scope = "(SELECT oid FROM pg_catalog.pg_namespace WHERE nspname=%s)"
    queries = (
        "SELECT c.relname,a.attnum,a.attname,pg_catalog.format_type(a.atttypid,a.atttypmod),"
        "a.attnotnull,pg_catalog.pg_get_expr(d.adbin,d.adrelid),a.attgenerated,a.attidentity "
        "FROM pg_catalog.pg_class c JOIN pg_catalog.pg_attribute a ON a.attrelid=c.oid "
        "LEFT JOIN pg_catalog.pg_attrdef d ON d.adrelid=c.oid AND d.adnum=a.attnum "
        f"WHERE c.relnamespace={scope} AND c.relkind='r' AND a.attnum>0 AND NOT a.attisdropped "
        "ORDER BY c.relname,a.attnum",
        "SELECT c.relname,co.contype,pg_catalog.pg_get_constraintdef(co.oid),co.condeferrable,"
        "co.condeferred,co.convalidated FROM pg_catalog.pg_constraint co "
        f"JOIN pg_catalog.pg_class c ON c.oid=co.conrelid WHERE co.connamespace={scope} "
        "ORDER BY c.relname,co.contype,pg_catalog.pg_get_constraintdef(co.oid)",
        "SELECT t.relname,i.indisunique,i.indisprimary,i.indisvalid,i.indisready,"
        "pg_catalog.pg_get_indexdef(i.indexrelid) FROM pg_catalog.pg_index i "
        f"JOIN pg_catalog.pg_class t ON t.oid=i.indrelid WHERE t.relnamespace={scope} "
        "ORDER BY t.relname,pg_catalog.pg_get_indexdef(i.indexrelid)",
    )
    physical = [conn.execute(query, (schema,)).fetchall() for query in queries]
    # Nazwy referencji i indeksów pochodzą wyłącznie z instalatora.
    payload = json.dumps(physical, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b"CryptoHunter/PostgreSQLCheckpointPhysical/v1\0" + payload).hexdigest()


_REVIEWED_COLUMNS = {
    "metadata": (
        ("singleton", "boolean"),
        ("schema_identity", "text"),
        ("schema_version", "integer"),
        ("profile", "text"),
        ("trust_domain", "bytea"),
        ("storage_family", "text"),
        ("schema_owner_role", "text"),
        ("schema_owner_role_oid", "oid"),
        ("runtime_role", "text"),
        ("runtime_role_oid", "oid"),
        ("admin_role", "text"),
        ("admin_role_oid", "oid"),
        ("function_definition_sha256", "jsonb"),
        ("physical_definition_sha256", "text"),
    ),
    "checkpoints": (
        ("checkpoint_id", "bytea"),
        ("stream_identity", "bytea"),
        ("authority_identity", "text"),
        ("revision", "bigint"),
    ),
    "entries": (
        ("checkpoint_id", "bytea"),
        ("revision", "bigint"),
        ("checkpoint", "bytea"),
        ("acceptance", "bytea"),
        ("head", "bytea"),
        ("signature", "bytea"),
    ),
}


def _qualify_physical_definition(
    conn: psycopg.Connection[Any], schema: str, owner_oid: int
) -> None:
    """Wiąże DDL z lokalnym kodem niezależnie od zmienialnej metadata."""
    quoted_schema = conn.execute("SELECT pg_catalog.quote_ident(%s)", (schema,)).fetchone()[0]
    columns = conn.execute(
        "SELECT c.relname,a.attnum,a.attname,pg_catalog.format_type(a.atttypid,a.atttypmod),"
        "a.attnotnull,pg_catalog.pg_get_expr(d.adbin,d.adrelid),a.attgenerated,a.attidentity,"
        "co.collname,cn.nspname,co.collisdeterministic,a.attstorage,a.attcompression,"
        "a.attislocal,a.attinhcount,a.attstattarget,a.attoptions FROM pg_catalog.pg_class c "
        "JOIN pg_catalog.pg_attribute a ON a.attrelid=c.oid "
        "JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
        "LEFT JOIN pg_catalog.pg_attrdef d ON d.adrelid=c.oid AND d.adnum=a.attnum "
        "LEFT JOIN pg_catalog.pg_collation co ON co.oid=a.attcollation "
        "LEFT JOIN pg_catalog.pg_namespace cn ON cn.oid=co.collnamespace "
        "WHERE n.nspname=%s AND c.relkind='r' AND a.attnum>0 AND NOT a.attisdropped "
        "ORDER BY c.relname,a.attnum",
        (schema,),
    ).fetchall()
    expected_columns = [
        (
            table,
            index,
            name,
            data_type,
            True,
            "true" if (table, name) == ("metadata", "singleton") else None,
            "",
            "",
            "default" if data_type == "text" else None,
            "pg_catalog" if data_type == "text" else None,
            True if data_type == "text" else None,
            "x" if data_type in ("text", "bytea", "jsonb") else "p",
            "",
            True,
            0,
            -1,
            None,
        )
        for table in sorted(_REVIEWED_COLUMNS)
        for index, (name, data_type) in enumerate(_REVIEWED_COLUMNS[table], 1)
    ]
    if columns != expected_columns:
        raise CheckpointSchemaQualificationError(
            "checkpoint columns differ from reviewed local definition"
        )
    constraints = conn.execute(
        "SELECT c.relname,co.contype,pg_catalog.pg_get_constraintdef(co.oid),co.condeferrable,"
        "co.condeferred,co.convalidated FROM pg_catalog.pg_constraint co "
        "JOIN pg_catalog.pg_class c ON c.oid=co.conrelid "
        "JOIN pg_catalog.pg_namespace n ON n.oid=co.connamespace WHERE n.nspname=%s",
        (schema,),
    ).fetchall()
    definitions = {
        ("metadata", "p", "PRIMARY KEY (singleton)"),
        ("metadata", "c", "CHECK (singleton)"),
        ("checkpoints", "p", "PRIMARY KEY (checkpoint_id)"),
        ("checkpoints", "u", "UNIQUE (authority_identity)"),
        ("checkpoints", "c", "CHECK ((revision >= 0))"),
        ("entries", "p", "PRIMARY KEY (checkpoint_id, revision)"),
        (
            "entries",
            "f",
            f"FOREIGN KEY (checkpoint_id) REFERENCES {quoted_schema}.checkpoints(checkpoint_id)",
        ),
        ("entries", "c", "CHECK ((revision >= 1))"),
        ("entries", "c", "CHECK ((octet_length(signature) = 64))"),
    }
    expected_constraints = {(*definition, False, False, True) for definition in definitions}
    if len(constraints) != len(expected_constraints) or set(constraints) != expected_constraints:
        raise CheckpointSchemaQualificationError(
            "checkpoint constraints differ from reviewed local definition"
        )
    indexes = conn.execute(
        "SELECT t.relname,ARRAY(SELECT a.attname FROM pg_catalog.unnest(i.indkey::smallint[]) "
        "WITH ORDINALITY k(attnum,pos) JOIN pg_catalog.pg_attribute a ON a.attrelid=t.oid "
        "AND a.attnum=k.attnum ORDER BY k.pos),i.indisunique,i.indisprimary,i.indisvalid,"
        "i.indisready,i.indisexclusion,i.indimmediate,"
        "pg_catalog.pg_get_expr(i.indexprs,i.indrelid),"
        "pg_catalog.pg_get_expr(i.indpred,i.indrelid),am.amname,idx.relowner,"
        "i.indnkeyatts,i.indnatts,i.indislive,i.indisclustered,i.indnullsnotdistinct,"
        "pg_catalog.pg_get_indexdef(i.indexrelid) "
        "FROM pg_catalog.pg_index i JOIN pg_catalog.pg_class t ON t.oid=i.indrelid "
        "JOIN pg_catalog.pg_class idx ON idx.oid=i.indexrelid "
        "JOIN pg_catalog.pg_am am ON am.oid=idx.relam "
        "JOIN pg_catalog.pg_namespace n ON n.oid=t.relnamespace WHERE n.nspname=%s",
        (schema,),
    ).fetchall()
    expected_shapes = {
        ("metadata", ("singleton",), True),
        ("checkpoints", ("checkpoint_id",), True),
        ("checkpoints", ("authority_identity",), False),
        ("entries", ("checkpoint_id", "revision"), True),
    }
    expected_indexes = {
        (
            table,
            keys,
            True,
            primary,
            True,
            True,
            False,
            True,
            None,
            None,
            "btree",
            owner_oid,
            len(keys),
            len(keys),
            True,
            False,
            False,
            "btree (" + ", ".join(keys) + ")",
        )
        for table, keys, primary in expected_shapes
    }
    observed_indexes = {
        (r[0], tuple(r[1]), *r[2:-1], r[-1].split(" USING ", 1)[-1]) for r in indexes
    }
    if len(indexes) != len(expected_indexes) or observed_indexes != expected_indexes:
        raise CheckpointSchemaQualificationError(
            "checkpoint indexes differ from reviewed local definition"
        )


def _qualify_auxiliary_inventory(
    conn: psycopg.Connection[Any], schema_oid: int, owner_oid: int, relation_oids: list[int]
) -> None:
    """Odrzuca nieprzejrzane obiekty, ustawienia fizyczne i zachowania pomocnicze."""
    inventory = conn.execute(
        "SELECT relname,relkind,relpersistence,relowner,reloptions,reltablespace "
        "FROM pg_catalog.pg_class WHERE relnamespace=%s ORDER BY relname",
        (schema_oid,),
    ).fetchall()
    reviewed_relations = {
        **dict.fromkeys(_TABLES, "r"),
        "metadata_pkey": "i",
        "checkpoints_pkey": "i",
        "checkpoints_authority_identity_key": "i",
        "entries_pkey": "i",
    }
    if inventory != [
        (name, kind, "p", owner_oid, None, 0) for name, kind in sorted(reviewed_relations.items())
    ]:
        raise CheckpointSchemaQualificationError("checkpoint object inventory mismatch")
    types = conn.execute(
        "SELECT t.typname,t.typtype,t.typowner,c.relname,element.typname FROM pg_catalog.pg_type t "
        "LEFT JOIN pg_catalog.pg_class c ON c.oid=t.typrelid "
        "LEFT JOIN pg_catalog.pg_type element ON element.oid=t.typelem "
        "WHERE t.typnamespace=%s ORDER BY t.typname",
        (schema_oid,),
    ).fetchall()
    reviewed_types: list[tuple[str, str, int, str | None, str | None]] = [
        (name, "c", owner_oid, name, None) for name in _TABLES
    ]
    reviewed_types += [("_" + name, "b", owner_oid, None, name) for name in _TABLES]
    if types != sorted(reviewed_types):
        raise CheckpointSchemaQualificationError("checkpoint type inventory mismatch")
    for catalog, namespace_column in (
        ("pg_collation", "collnamespace"),
        ("pg_conversion", "connamespace"),
        ("pg_operator", "oprnamespace"),
        ("pg_opclass", "opcnamespace"),
        ("pg_opfamily", "opfnamespace"),
        ("pg_ts_config", "cfgnamespace"),
        ("pg_ts_dict", "dictnamespace"),
        ("pg_ts_parser", "prsnamespace"),
        ("pg_ts_template", "tmplnamespace"),
        ("pg_statistic_ext", "stxnamespace"),
    ):
        if (
            conn.execute(
                sql.SQL("SELECT 1 FROM pg_catalog.{} WHERE {}=%s LIMIT 1").format(
                    sql.Identifier(catalog), sql.Identifier(namespace_column)
                ),
                (schema_oid,),
            ).fetchone()
            is not None
        ):
            raise CheckpointSchemaQualificationError("unreviewed checkpoint namespace object")
    if (
        conn.execute(
            "SELECT 1 FROM pg_catalog.pg_default_acl WHERE defaclnamespace=%s LIMIT 1",
            (schema_oid,),
        ).fetchone()
        is not None
    ):
        raise CheckpointSchemaQualificationError("unreviewed checkpoint default ACL")
    behavior = conn.execute(
        "SELECT c.relname,c.relreplident,am.amname,c.relpartbound,c.relhasrules "
        "FROM pg_catalog.pg_class c JOIN pg_catalog.pg_am am ON am.oid=c.relam "
        "WHERE c.oid=ANY(%s) ORDER BY c.relname",
        (relation_oids,),
    ).fetchall()
    if behavior != [(name, "d", "heap", None, False) for name in sorted(_TABLES)]:
        raise CheckpointSchemaQualificationError("checkpoint table behavior mismatch")
    if (
        conn.execute(
            "SELECT 1 FROM pg_catalog.pg_attribute WHERE attrelid=ANY(%s) "
            "AND (attisdropped OR (attnum>0 AND attcompression <> '')) LIMIT 1",
            (relation_oids,),
        ).fetchone()
        is not None
    ):
        raise CheckpointSchemaQualificationError("unreviewed checkpoint attribute behavior")
    if conn.execute(
        "SELECT pg_catalog.count(*) FROM pg_catalog.pg_trigger WHERE tgrelid=ANY(%s)",
        (relation_oids,),
    ).fetchone() != (4,):
        raise CheckpointSchemaQualificationError("checkpoint trigger cardinality mismatch")
    for query in (
        "SELECT 1 FROM pg_catalog.pg_namespace n CROSS JOIN LATERAL "
        "pg_catalog.aclexplode(n.nspacl) a WHERE n.oid=%s AND a.grantor<>%s LIMIT 1",
        "SELECT 1 FROM pg_catalog.pg_class c CROSS JOIN LATERAL "
        "pg_catalog.aclexplode(c.relacl) a WHERE c.relnamespace=%s "
        "AND a.grantor<>%s LIMIT 1",
        "SELECT 1 FROM pg_catalog.pg_proc p CROSS JOIN LATERAL "
        "pg_catalog.aclexplode(p.proacl) a WHERE p.pronamespace=%s "
        "AND a.grantor<>%s LIMIT 1",
    ):
        if conn.execute(query, (schema_oid, owner_oid)).fetchone() is not None:
            raise CheckpointSchemaQualificationError("checkpoint ACL grantor mismatch")
    triggers = conn.execute(
        "SELECT relation.relname,p.proname,n.nspname,t.tgtype,t.tgenabled,t.tgisinternal,"
        "t.tgdeferrable,t.tginitdeferred,t.tgnargs,t.tgargs,t.tgqual,t.tgoldtable,t.tgnewtable,"
        "target.relname,co.contype,t.tgparentid "
        "FROM pg_catalog.pg_trigger t JOIN pg_catalog.pg_class relation ON relation.oid=t.tgrelid "
        "JOIN pg_catalog.pg_class target ON target.oid=t.tgconstrrelid "
        "JOIN pg_catalog.pg_constraint co ON co.oid=t.tgconstraint "
        "JOIN pg_catalog.pg_proc p ON p.oid=t.tgfoid "
        "JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace WHERE t.tgrelid=ANY(%s)",
        (relation_oids,),
    ).fetchall()
    reviewed_triggers = {
        (
            relation,
            function,
            "pg_catalog",
            trigger_type,
            "O",
            True,
            False,
            False,
            0,
            b"",
            None,
            None,
            None,
            target,
            "f",
            0,
        )
        for relation, function, trigger_type, target in (
            ("entries", "RI_FKey_check_ins", 5, "checkpoints"),
            ("entries", "RI_FKey_check_upd", 17, "checkpoints"),
            ("checkpoints", "RI_FKey_noaction_del", 9, "entries"),
            ("checkpoints", "RI_FKey_noaction_upd", 17, "entries"),
        )
    }
    if len(triggers) != len(reviewed_triggers) or set(triggers) != reviewed_triggers:
        raise CheckpointSchemaQualificationError("checkpoint constraint trigger inventory mismatch")


def provision_postgresql_checkpoint(
    bootstrap: PostgreSQLConnectionConfig,
    config: PostgreSQLCheckpointProvisioning,
) -> None:
    """Atomowo instaluje nowy schemat, role i przejrzane funkcje SQL.

    Operator konfiguruje uwierzytelnianie LOGIN osobno. CREATE odrzuca ponowną
    instalację i aktualizację istniejącego schematu. Nie zapisuje kluczy.
    """
    if (
        type(bootstrap) is not PostgreSQLConnectionConfig
        or type(config) is not PostgreSQLCheckpointProvisioning
    ):
        raise TypeError("exact connection and provisioning configuration required")
    with psycopg.connect(bootstrap.dsn, autocommit=True) as conn, conn.transaction():
        if conn.info.server_version < 160000 or conn.execute(
            "SELECT pg_catalog.current_setting('fsync'),"
            "pg_catalog.current_setting('synchronous_commit')"
        ).fetchone() != ("on", "on"):
            raise CheckpointSchemaQualificationError("checkpoint installation durability mismatch")
        for field, login in (
            ("schema_owner_role", "NOLOGIN"),
            ("runtime_role", "LOGIN"),
            ("admin_role", "LOGIN"),
        ):
            conn.execute(
                sql.SQL(
                    "CREATE ROLE {} {} NOSUPERUSER NOCREATEDB NOCREATEROLE "
                    "NOINHERIT NOREPLICATION NOBYPASSRLS"
                ).format(sql.Identifier(getattr(config, field)), sql.SQL(login))
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
        sources = _function_sources(config)
        for signature, source in sources.items():
            name = signature.split("(", 1)[0]
            declaration, returns = _FUNCTION_DECLARATIONS[signature]
            conn.execute(
                sql.SQL(
                    "CREATE FUNCTION {}.{}({}) RETURNS {} LANGUAGE plpgsql "
                    "SECURITY DEFINER SET search_path=pg_catalog AS {}"
                ).format(
                    sql.Identifier(config.schema),
                    sql.Identifier(name),
                    sql.SQL(declaration),
                    sql.SQL(returns),
                    sql.Literal(source),
                )
            )
            conn.execute(
                sql.SQL("REVOKE ALL ON FUNCTION {}.{} FROM PUBLIC").format(
                    sql.Identifier(config.schema), sql.SQL(signature)
                )
            )
            role_field = _FUNCTION_CALLERS[signature]
            if role_field is not None:
                conn.execute(
                    sql.SQL("GRANT EXECUTE ON FUNCTION {}.{} TO {}").format(
                        sql.Identifier(config.schema),
                        sql.SQL(signature),
                        sql.Identifier(getattr(config, role_field)),
                    )
                )
        conn.execute(
            sql.SQL("GRANT USAGE ON SCHEMA {} TO {},{}").format(
                sql.Identifier(config.schema),
                sql.Identifier(config.runtime_role),
                sql.Identifier(config.admin_role),
            )
        )
        conn.execute(
            sql.SQL("GRANT SELECT ON {}.metadata TO {},{}").format(
                sql.Identifier(config.schema),
                sql.Identifier(config.runtime_role),
                sql.Identifier(config.admin_role),
            )
        )
        roles = [
            conn.execute(
                "SELECT oid FROM pg_catalog.pg_roles WHERE rolname=%s", (role,)
            ).fetchone()[0]
            for role in (config.schema_owner_role, config.runtime_role, config.admin_role)
        ]
        manifest = {
            signature: _source_fingerprint(signature, source)
            for signature, source in sources.items()
        }
        conn.execute(
            sql.SQL(
                "INSERT INTO {}.metadata VALUES(true,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            ).format(sql.Identifier(config.schema)),
            (
                SCHEMA_IDENTITY,
                SCHEMA_VERSION,
                PROFILE,
                _scalar(config.trust_domain),
                STORAGE_FAMILY,
                config.schema_owner_role,
                roles[0],
                config.runtime_role,
                roles[1],
                config.admin_role,
                roles[2],
                psycopg.types.json.Jsonb(manifest),
                _physical_fingerprint(conn, config.schema),
            ),
        )


def provision_postgresql_checkpoint_stream(
    connection: PostgreSQLConnectionConfig,
    *,
    schema: str,
    checkpoint_id: str,
    stream: HistoryStreamIdentity,
) -> None:
    """Nadaje trwałą tożsamość checkpointowi z osobnego bezpośredniego LOGIN admina."""
    _identifier(schema)
    if type(connection) is not PostgreSQLConnectionConfig:
        raise TypeError("exact PostgreSQLConnectionConfig required")
    if type(checkpoint_id) is not str or not checkpoint_id:
        raise ValueError("checkpoint_id must be an exact non-empty str")
    if type(stream) is not HistoryStreamIdentity:
        raise TypeError("stream must be exact HistoryStreamIdentity")
    with psycopg.connect(connection.dsn, autocommit=True) as conn, conn.transaction():
        qualify_checkpoint_schema(
            conn, schema, expected_role="admin_role", trust_domain=stream.trust_domain
        )
        conn.execute(
            sql.SQL("SELECT {}.provision_stream(%s,%s)").format(sql.Identifier(schema)),
            (_scalar(checkpoint_id), canonical_json_bytes(stream.material())),
        )


def qualify_checkpoint_schema(
    conn: psycopg.Connection[Any],
    schema: str,
    *,
    expected_role: str,
    trust_domain: str,
) -> None:
    """Sprawdza LOGIN, trwałość, lokalną definicję DDL, źródła i dokładne ACL.

    Każda operacja autorytetu wywołuje kwalifikację wewnątrz swojej transakcji.
    Zapytania katalogowe używają pg_catalog niezależnie od search_path klienta.
    """
    _identifier(schema)
    if expected_role not in ("runtime_role", "admin_role"):
        raise ValueError("invalid expected checkpoint role")
    try:
        _qualify_checkpoint_schema(conn, schema, expected_role, trust_domain)
    except CheckpointSchemaQualificationError:
        raise
    except (psycopg.Error, ValueError, TypeError, KeyError, IndexError) as exc:
        raise CheckpointSchemaQualificationError(
            "PostgreSQL checkpoint qualification failed"
        ) from exc


def _qualify_checkpoint_schema(
    conn: psycopg.Connection[Any],
    schema: str,
    expected_role: str,
    trust_domain: str,
) -> None:
    conn.execute("SET LOCAL search_path=pg_catalog")
    row = conn.execute(
        sql.SQL(
            "SELECT schema_identity,schema_version,profile,trust_domain,storage_family,"
            "schema_owner_role,schema_owner_role_oid,runtime_role,runtime_role_oid,admin_role,"
            "admin_role_oid,function_definition_sha256,physical_definition_sha256 FROM {}.metadata"
        ).format(sql.Identifier(schema))
    ).fetchall()
    if len(row) != 1 or len(row[0]) != 13:
        raise CheckpointSchemaQualificationError("checkpoint metadata cardinality mismatch")
    metadata = row[0]
    if metadata[:5] != (
        SCHEMA_IDENTITY,
        SCHEMA_VERSION,
        PROFILE,
        _scalar(trust_domain),
        STORAGE_FAMILY,
    ):
        raise CheckpointSchemaQualificationError("checkpoint security metadata mismatch")
    config = PostgreSQLCheckpointProvisioning(
        schema, metadata[5], metadata[7], metadata[9], trust_domain
    )
    roles = {
        "schema_owner_role": (metadata[5], metadata[6]),
        "runtime_role": (metadata[7], metadata[8]),
        "admin_role": (metadata[9], metadata[10]),
    }
    role_oids = [roles[field][1] for field in ("schema_owner_role", "runtime_role", "admin_role")]
    for field, (name, oid) in roles.items():
        attrs = conn.execute(
            "SELECT oid,rolcanlogin,rolinherit,rolsuper,rolcreaterole,rolcreatedb,"
            "rolreplication,rolbypassrls "
            "FROM pg_catalog.pg_roles WHERE rolname=%s",
            (name,),
        ).fetchone()
        if attrs != (oid, field != "schema_owner_role", False, False, False, False, False, False):
            raise CheckpointSchemaQualificationError(
                "checkpoint role identity or attributes mismatch"
            )
    principal = conn.execute("SELECT session_user::text,current_user::text").fetchone()
    if principal != (roles[expected_role][0], roles[expected_role][0]):
        raise CheckpointSchemaQualificationError(
            "checkpoint requires its exact direct-login principal"
        )
    if (
        conn.execute(
            "SELECT 1 FROM pg_catalog.pg_auth_members "
            "WHERE roleid=ANY(%s) OR member=ANY(%s) LIMIT 1",
            (role_oids, role_oids),
        ).fetchone()
        is not None
    ):
        raise CheckpointSchemaQualificationError("checkpoint authority role has a membership grant")
    if conn.info.server_version < 160000 or conn.execute(
        "SELECT pg_catalog.current_setting('fsync'),"
        "pg_catalog.current_setting('synchronous_commit')"
    ).fetchone() != ("on", "on"):
        raise CheckpointSchemaQualificationError("checkpoint PostgreSQL durability mismatch")
    owner_oid = roles["schema_owner_role"][1]
    namespace = conn.execute(
        "SELECT oid,nspowner FROM pg_catalog.pg_namespace WHERE nspname=%s", (schema,)
    ).fetchone()
    if namespace is None or namespace[1] != owner_oid:
        raise CheckpointSchemaQualificationError("checkpoint schema owner mismatch")
    schema_acl = set(
        conn.execute(
            "SELECT a.grantee,a.privilege_type,a.is_grantable FROM pg_catalog.pg_namespace n "
            "CROSS JOIN LATERAL pg_catalog.aclexplode(n.nspacl) a WHERE n.nspname=%s",
            (schema,),
        ).fetchall()
    )
    if schema_acl != {
        (owner_oid, "USAGE", False),
        (owner_oid, "CREATE", False),
        (roles["runtime_role"][1], "USAGE", False),
        (roles["admin_role"][1], "USAGE", False),
    }:
        raise CheckpointSchemaQualificationError("checkpoint schema ACL differs from allowlist")
    relations = conn.execute(
        "SELECT oid,relname,relkind,relpersistence,relispartition,relrowsecurity,"
        "relforcerowsecurity,relowner FROM pg_catalog.pg_class "
        "WHERE relnamespace=%s AND relkind NOT IN ('i','S','t') ORDER BY relname",
        (namespace[0],),
    ).fetchall()
    if [r[1:] for r in relations] != [
        (name, "r", "p", False, False, False, owner_oid) for name in sorted(_TABLES)
    ]:
        raise CheckpointSchemaQualificationError("checkpoint relations differ from reviewed schema")
    relation_oids = [r[0] for r in relations]
    _qualify_auxiliary_inventory(conn, namespace[0], owner_oid, relation_oids)
    for query in (
        "SELECT 1 FROM pg_catalog.pg_inherits WHERE inhrelid=ANY(%s) OR inhparent=ANY(%s) LIMIT 1",
        "SELECT 1 FROM pg_catalog.pg_trigger WHERE tgrelid=ANY(%s) AND NOT tgisinternal LIMIT 1",
        "SELECT 1 FROM pg_catalog.pg_rewrite WHERE ev_class=ANY(%s) LIMIT 1",
        "SELECT 1 FROM pg_catalog.pg_policy WHERE polrelid=ANY(%s) LIMIT 1",
        "SELECT 1 FROM pg_catalog.pg_attribute "
        "WHERE attrelid=ANY(%s) AND attacl IS NOT NULL LIMIT 1",
    ):
        params = (relation_oids, relation_oids) if "inhparent" in query else (relation_oids,)
        if conn.execute(query, params).fetchone() is not None:
            raise CheckpointSchemaQualificationError(
                "checkpoint relation has unreviewed auxiliary behavior or column ACL"
            )
    owner_privileges = {"SELECT", "INSERT", "UPDATE", "DELETE", "TRUNCATE", "REFERENCES", "TRIGGER"}
    expected_acl = {
        (table, owner_oid, privilege, False) for table in _TABLES for privilege in owner_privileges
    }
    expected_acl |= {
        ("metadata", roles[field][1], "SELECT", False) for field in ("runtime_role", "admin_role")
    }
    observed_acl = set(
        conn.execute(
            "SELECT c.relname,a.grantee,a.privilege_type,a.is_grantable FROM pg_catalog.pg_class c "
            "CROSS JOIN LATERAL pg_catalog.aclexplode(c.relacl) a "
            "WHERE c.relnamespace=%s AND c.relkind='r'",
            (namespace[0],),
        ).fetchall()
    )
    if observed_acl != expected_acl:
        raise CheckpointSchemaQualificationError("checkpoint table ACL differs from allowlist")
    _qualify_physical_definition(conn, schema, owner_oid)
    if _physical_fingerprint(conn, schema) != metadata[12]:
        raise CheckpointSchemaQualificationError(
            "checkpoint physical definition fingerprint mismatch"
        )
    sources = _function_sources(config)
    manifest = {
        signature: _source_fingerprint(signature, source) for signature, source in sources.items()
    }
    if metadata[11] != manifest:
        raise CheckpointSchemaQualificationError("checkpoint function manifest mismatch")
    function_rows = conn.execute(
        "SELECT p.oid,p.proname || '(' || "
        "pg_catalog.replace(pg_catalog.oidvectortypes(p.proargtypes),', ', ',') || ')',"
        "p.proowner,p.prosecdef,p.proconfig,p.prosrc,"
        "pg_catalog.format_type(p.prorettype,NULL),l.lanname,p.provolatile,p.proisstrict,p.proleakproof,"
        "p.proparallel,p.prokind,p.proargmodes FROM pg_catalog.pg_proc p "
        "JOIN pg_catalog.pg_language l ON l.oid=p.prolang WHERE p.pronamespace=%s",
        (namespace[0],),
    ).fetchall()
    if len(function_rows) != len(sources):
        raise CheckpointSchemaQualificationError("checkpoint function inventory mismatch")
    for function in function_rows:
        (
            oid,
            qualified,
            function_owner,
            definer,
            settings,
            source,
            returns,
            language,
            volatility,
            strict,
            leakproof,
            parallel,
            kind,
            modes,
        ) = function
        signature = qualified
        expected_returns = _FUNCTION_DECLARATIONS.get(signature, ("", ""))[1]
        expected_modes = None
        if signature == "canonical_string(bytea,integer)":
            expected_returns = "record"
            expected_modes = ["i", "i", "t", "t"]
        if signature not in sources or (
            function_owner,
            definer,
            settings,
            source,
            returns,
            language,
            volatility,
            strict,
            leakproof,
            parallel,
            kind,
            modes,
        ) != (
            owner_oid,
            True,
            ["search_path=pg_catalog"],
            sources.get(signature),
            expected_returns,
            "plpgsql",
            "v",
            False,
            False,
            "u",
            "f",
            expected_modes,
        ):
            raise CheckpointSchemaQualificationError(
                "checkpoint function source or security metadata mismatch"
            )
        declaration = _FUNCTION_DECLARATIONS[signature][0]
        expected_names = [argument.strip().split(" ", 1)[0] for argument in declaration.split(",")]
        if signature == "canonical_string(bytea,integer)":
            expected_names += ["next_position", "decoded_key"]
        invocation = conn.execute(
            "SELECT p.proargnames,p.pronargdefaults,p.proargdefaults,p.provariadic,"
            "p.procost,p.prorows,p.prosupport,p.probin,p.prosqlbody,p.proallargtypes,"
            "p.proretset,pg_catalog.pg_get_function_result(p.oid) "
            "FROM pg_catalog.pg_proc p WHERE p.oid=%s",
            (oid,),
        ).fetchone()
        if invocation != (
            expected_names,
            0,
            None,
            0,
            100.0,
            1000.0 if signature == "canonical_string(bytea,integer)" else 0.0,
            "-",
            None,
            None,
            [17, 23, 23, 17] if signature == "canonical_string(bytea,integer)" else None,
            signature == "canonical_string(bytea,integer)",
            "TABLE(next_position integer, decoded_key bytea)"
            if signature == "canonical_string(bytea,integer)"
            else expected_returns,
        ):
            raise CheckpointSchemaQualificationError(
                "checkpoint function invocation metadata mismatch"
            )
        acl = set(
            conn.execute(
                "SELECT a.grantee,a.privilege_type,a.is_grantable FROM pg_catalog.pg_proc p "
                "CROSS JOIN LATERAL pg_catalog.aclexplode(p.proacl) a WHERE p.oid=%s",
                (oid,),
            ).fetchall()
        )
        expected = {(owner_oid, "EXECUTE", False)}
        caller = _FUNCTION_CALLERS[signature]
        if caller is not None:
            expected.add((roles[caller][1], "EXECUTE", False))
        if acl != expected:
            raise CheckpointSchemaQualificationError(
                "checkpoint function ACL differs from allowlist"
            )


__all__ = [
    "CheckpointSchemaQualificationError",
    "PostgreSQLConnectionConfig",
    "PostgreSQLCheckpointProvisioning",
    "SCHEMA_IDENTITY",
    "SCHEMA_VERSION",
    "STORAGE_FAMILY",
    "provision_postgresql_checkpoint",
    "provision_postgresql_checkpoint_stream",
    "qualify_checkpoint_schema",
]
