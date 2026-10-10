"""Offline installation and live qualification of durable issuer history.

The runtime stores history only.  Its database privileges confer neither
entitlement binding nor Root Proof issuance authority.  Full identity strings
are retained as canonical UTF-8 bytes, including JSON escaped U+0000.
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
    ATTESTATION_DOMAIN,
    RECORD_DOMAIN,
    HistoryStreamIdentity,
    canonical_json_bytes,
)
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core.postgresql_issuer_history_canonical_sql import (
    CANONICAL_FUNCTION_RETURNS,
    canonical_function_sources,
)

SCHEMA_IDENTITY = "cryptohunter.issuer_authenticated_history.postgresql"
SCHEMA_VERSION = 1
STORAGE_FAMILY = "POSTGRESQL_ISSUER_AUTHENTICATED_HISTORY_V1"
PROFILE = "PRODUCTION_LOCAL"
_SAFE_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]{0,62}\Z")
_EVENT_DOMAIN = b"CryptoHunter/M0.5/IssuerHistoryEvent/v1\0"
_TABLES = ("metadata", "streams", "records", "attestations")
_FUNCTION_SIGNATURES = (
    *CANONICAL_FUNCTION_RETURNS,
    "validate_members(bytea,text[])",
    "validate_stream(bytea)",
    "verify_stream(bytea)",
    "provision_stream(bytea)",
    "read_stream(bytea,boolean)",
    "append_record(bytea,text,json,bytea)",
    "retain_head(bytea,json,bytea,bytea)",
)
_FUNCTION_CALLERS = {
    **dict.fromkeys(CANONICAL_FUNCTION_RETURNS),
    "validate_members(bytea,text[])": None,
    "validate_stream(bytea)": None,
    "verify_stream(bytea)": None,
    "provision_stream(bytea)": "admin_role",
    "read_stream(bytea,boolean)": "runtime_role",
    "append_record(bytea,text,json,bytea)": "runtime_role",
    "retain_head(bytea,json,bytea,bytea)": "runtime_role",
}


class HistorySchemaQualificationError(RuntimeError):
    """The configured database is not the reviewed history authority."""


@dataclass(frozen=True, slots=True)
class PostgreSQLIssuerHistoryProvisioning:
    schema: str
    schema_owner_role: str
    runtime_role: str
    admin_role: str
    trust_domain: str

    def __post_init__(self) -> None:
        for name in ("schema", "schema_owner_role", "runtime_role", "admin_role"):
            _identifier(getattr(self, name))
        if len({self.schema_owner_role, self.runtime_role, self.admin_role}) != 3:
            raise ValueError("history owner, runtime and admin roles must be distinct")
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
        b"CryptoHunter/PostgreSQLIssuerHistoryFunction/v1\0"
        + signature.encode("ascii")
        + b"\0"
        + source.encode("utf-8")
    ).hexdigest()


def _function_sources(config: PostgreSQLIssuerHistoryProvisioning) -> dict[str, str]:
    s = '"' + config.schema + '"'
    runtime_guard = (
        f"IF session_user <> '{config.runtime_role}' OR current_user <> "
        f"'{config.schema_owner_role}' THEN RAISE EXCEPTION 'wrong history principal' "
        "USING ERRCODE='42501'; END IF;"
        " IF pg_catalog.current_setting('server_version_num')::integer < 160000 "
        "OR pg_catalog.current_setting('fsync') <> 'on' "
        "OR pg_catalog.current_setting('synchronous_commit') <> 'on' THEN "
        "RAISE EXCEPTION 'history durability settings mismatch' USING ERRCODE='42501'; END IF;"
    )
    admin_guard = runtime_guard.replace(config.runtime_role, config.admin_role)
    record_domain = RECORD_DOMAIN.hex()
    event_domain = _EVENT_DOMAIN.hex()
    attestation_domain = ATTESTATION_DOMAIN.hex()
    sources = {
        "validate_members(bytea,text[])": """
DECLARE field_name text; member bytea; rebuilt bytea := pg_catalog.convert_to('{','UTF8');
BEGIN
 PERFORM {s}.validate_canonical_json(p_bytes);
 FOREACH field_name IN ARRAY p_fields LOOP
  member := {s}.canonical_object_member(p_bytes,field_name);
  IF member IS NULL THEN RAISE EXCEPTION 'missing canonical identity member' USING
  ERRCODE='22023'; END IF;
  IF pg_catalog.octet_length(rebuilt)>1 THEN rebuilt := rebuilt || pg_catalog.convert_to(',',
  'UTF8'); END IF;
  rebuilt := rebuilt || pg_catalog.convert_to(pg_catalog.to_json(field_name)::text || ':',
  'UTF8') || member;
 END LOOP;
 rebuilt := rebuilt || pg_catalog.convert_to('}','UTF8');
 IF p_bytes IS DISTINCT FROM rebuilt THEN
  RAISE EXCEPTION 'canonical identity member inventory differs' USING ERRCODE='22023';
 END IF;
END
""",
        "validate_stream(bytea)": """
DECLARE key_name text; member bytea;
BEGIN
 IF p_stream IS NULL THEN RAISE EXCEPTION 'missing stream' USING ERRCODE='22023'; END IF;
 PERFORM {s}.validate_members(p_stream,ARRAY['environment','issuer_authority_identity',
  'product_scope','security_epoch','security_profile','stream_id','trust_domain']);
 FOREACH key_name IN ARRAY ARRAY['environment','issuer_authority_identity','product_scope',
  'security_profile','stream_id','trust_domain'] LOOP
  member := {s}.canonical_object_member(p_stream,key_name);
  IF pg_catalog.get_byte(member,0) <> 34 OR pg_catalog.octet_length(member)=2 THEN
   RAISE EXCEPTION 'malformed stream field' USING ERRCODE='22023';
  END IF;
 END LOOP;
 IF {s}.canonical_object_member(p_stream,'security_profile') IS DISTINCT FROM
    pg_catalog.convert_to('\"PRODUCTION_LOCAL\"','UTF8')
 OR pg_catalog.convert_from({s}.canonical_object_member(p_stream,'security_epoch'),'UTF8') !~
  '^[1-9][0-9]*$'
 OR {s}.canonical_object_member(p_stream,'trust_domain') IS DISTINCT FROM
    (SELECT trust_domain FROM {s}.metadata WHERE singleton)
 THEN RAISE EXCEPTION 'stream security scope mismatch' USING ERRCODE='22023'; END IF;
END
""",
        "verify_stream(bytea)": """
DECLARE st {s}.streams%ROWTYPE; r {s}.records%ROWTYPE; h {s}.attestations%ROWTYPE;
 seq bigint := 0; predecessor text := 'NO_PREDECESSOR';
 digest text; head_body bytea; field_name text; payload_bytes bytea; rebuilt bytea;
BEGIN
 PERFORM {s}.validate_stream(p_stream);
 SELECT * INTO st FROM {s}.streams WHERE stream_identity=p_stream;
 IF NOT FOUND THEN RAISE EXCEPTION 'unprovisioned history stream' USING ERRCODE='42501'; END IF;
 IF st.stream_id IS DISTINCT FROM {s}.canonical_object_member(p_stream,'stream_id')
 OR st.issuer_authority_identity IS DISTINCT FROM {s}.canonical_object_member(p_stream,
  'issuer_authority_identity')
 OR st.security_profile IS DISTINCT FROM {s}.canonical_object_member(p_stream,'security_profile')
 OR st.environment IS DISTINCT FROM {s}.canonical_object_member(p_stream,'environment')
 OR st.trust_domain IS DISTINCT FROM {s}.canonical_object_member(p_stream,'trust_domain')
 OR st.product_scope IS DISTINCT FROM {s}.canonical_object_member(p_stream,'product_scope')
 OR st.security_epoch IS DISTINCT FROM
  pg_catalog.convert_from({s}.canonical_object_member(p_stream,'security_epoch'),'UTF8')
 THEN RAISE EXCEPTION 'physical stream identity corrupt' USING ERRCODE='22023'; END IF;
 FOR r IN SELECT * FROM {s}.records WHERE stream_identity=p_stream ORDER BY sequence LOOP
  seq := seq + 1;
  PERFORM {s}.validate_canonical_json(r.canonical_record);
  PERFORM {s}.validate_canonical_json(r.event_identity);
  PERFORM {s}.validate_members(r.event_identity,ARRAY['issuance_attempt_id',
  'logical_operation_id','root_proof_id']);
  FOREACH field_name IN ARRAY ARRAY['issuance_attempt_id','logical_operation_id',
  'root_proof_id'] LOOP
   payload_bytes := {s}.canonical_object_member(r.event_identity,field_name);
   IF pg_catalog.get_byte(payload_bytes,0) <> 34 OR pg_catalog.octet_length(payload_bytes)=2 THEN
    RAISE EXCEPTION 'retained event identity corrupt' USING ERRCODE='22023';
   END IF;
  END LOOP;
  payload_bytes := {s}.canonical_object_member(r.canonical_record,'canonical_event_payload');
  rebuilt := pg_catalog.convert_to('{"canonical_event_payload":','UTF8') || payload_bytes
   || pg_catalog.convert_to(',"event_digest":','UTF8')
   || pg_catalog.convert_to(pg_catalog.to_json(r.event_digest)::text,'UTF8')
   || pg_catalog.convert_to(',"event_identity":','UTF8') || r.event_identity
   || pg_catalog.convert_to(',"predecessor_authenticated_digest":','UTF8')
   || pg_catalog.convert_to(pg_catalog.to_json(predecessor)::text,'UTF8')
   || pg_catalog.convert_to(',"sequence":' || seq::text || ',"stream":','UTF8')
   || p_stream || pg_catalog.convert_to('}','UTF8');
  IF payload_bytes IS NULL OR pg_catalog.get_byte(payload_bytes,0) <> 123
  OR r.canonical_record IS DISTINCT FROM rebuilt
  OR r.sequence IS DISTINCT FROM seq
  OR r.predecessor_authenticated_digest IS DISTINCT FROM predecessor
  OR r.logical_operation_id IS DISTINCT FROM {s}.canonical_object_member(r.event_identity,
  'logical_operation_id')
  OR r.issuance_attempt_id IS DISTINCT FROM {s}.canonical_object_member(r.event_identity,
  'issuance_attempt_id')
  OR r.root_proof_id IS DISTINCT FROM {s}.canonical_object_member(r.event_identity,'root_proof_id')
  OR r.event_digest IS DISTINCT FROM 'sha256:' || pg_catalog.encode(pg_catalog.sha256(
     pg_catalog.decode('{event_domain}','hex') || payload_bytes),'hex')
  OR r.authenticated_digest IS DISTINCT FROM 'sha256:' || pg_catalog.encode(pg_catalog.sha256(
     pg_catalog.decode('{record_domain}','hex') || r.canonical_record),'hex')
  THEN RAISE EXCEPTION 'retained issuer history corrupt' USING ERRCODE='22023'; END IF;
  predecessor := r.authenticated_digest;
 END LOOP;
 IF st.head_sequence IS DISTINCT FROM seq OR st.head_digest IS DISTINCT FROM predecessor
 THEN RAISE EXCEPTION 'history head projection corrupt' USING ERRCODE='22023'; END IF;
 FOR h IN SELECT * FROM {s}.attestations WHERE stream_identity=p_stream ORDER BY sequence LOOP
  IF pg_catalog.substring(h.canonical_head,1,{attestation_length}) IS DISTINCT FROM
  pg_catalog.decode('{attestation_domain}','hex')
  OR pg_catalog.octet_length(h.signature) <> 64 THEN
   RAISE EXCEPTION 'retained attestation envelope corrupt' USING ERRCODE='22023';
  END IF;
  head_body := pg_catalog.substring(h.canonical_head,{attestation_offset});
  PERFORM {s}.validate_members(head_body,
   ARRAY['record_digest','sequence','signing_credential_identity','stream']);
  PERFORM {s}.validate_members(h.credential_identity,
   ARRAY['credential_identity','custody_lifecycle_namespace','key_handle_or_version',
    'key_material_identity','provider_namespace','semantic_role']);
  SELECT authenticated_digest INTO digest FROM {s}.records WHERE stream_identity=p_stream AND
  sequence=h.sequence;
  IF NOT FOUND OR digest IS DISTINCT FROM h.record_digest
  OR {s}.canonical_object_member(head_body,'stream') IS DISTINCT FROM p_stream
  OR {s}.canonical_object_member(head_body,'sequence') IS DISTINCT FROM
  pg_catalog.convert_to(h.sequence::text,'UTF8')
  OR {s}.canonical_object_member(head_body,'record_digest') IS DISTINCT FROM
  pg_catalog.convert_to(pg_catalog.to_json(h.record_digest)::text,'UTF8')
  OR {s}.canonical_object_member(head_body,'signing_credential_identity') IS DISTINCT FROM
  h.credential_identity
  OR {s}.canonical_object_member(h.credential_identity,'semantic_role') IS DISTINCT FROM
  pg_catalog.convert_to('\"HISTORY_ATTESTATION_SIGNING\"','UTF8')
  THEN RAISE EXCEPTION 'retained attestation relation corrupt' USING ERRCODE='22023'; END IF;
  FOREACH field_name IN ARRAY ARRAY['credential_identity','custody_lifecycle_namespace',
  'provider_namespace'] LOOP
   payload_bytes := {s}.canonical_object_member(h.credential_identity,field_name);
   IF pg_catalog.get_byte(payload_bytes,0) <> 34 OR pg_catalog.octet_length(payload_bytes)=2 THEN
    RAISE EXCEPTION 'retained attestation credential corrupt' USING ERRCODE='22023';
   END IF;
  END LOOP;
  FOREACH field_name IN ARRAY ARRAY['key_handle_or_version','key_material_identity'] LOOP
   payload_bytes := {s}.canonical_object_member(h.credential_identity,field_name);
   IF (pg_catalog.get_byte(payload_bytes,0) <> 34 AND payload_bytes <>
  pg_catalog.convert_to('null','UTF8'))
   OR pg_catalog.octet_length(payload_bytes)=2 THEN
    RAISE EXCEPTION 'retained attestation credential corrupt' USING ERRCODE='22023';
   END IF;
  END LOOP;
 END LOOP;
END
""",
        "provision_stream(bytea)": """
BEGIN
 {admin_guard}
 PERFORM {s}.validate_stream(p_stream);
 INSERT INTO {s}.streams(stream_identity,stream_id,issuer_authority_identity,
  security_profile,environment,trust_domain,product_scope,security_epoch,head_sequence,head_digest)
 VALUES(p_stream,{s}.canonical_object_member(p_stream,'stream_id'),
  {s}.canonical_object_member(p_stream,'issuer_authority_identity'),
  {s}.canonical_object_member(p_stream,'security_profile'),
  {s}.canonical_object_member(p_stream,'environment'),
  {s}.canonical_object_member(p_stream,'trust_domain'),
  {s}.canonical_object_member(p_stream,'product_scope'),
  pg_catalog.convert_from({s}.canonical_object_member(p_stream,'security_epoch'),'UTF8'),0,
  'NO_PREDECESSOR');
END
""",
        "read_stream(bytea,boolean)": """
DECLARE result jsonb;
BEGIN
 {runtime_guard}
 IF p_lock IS NULL THEN RAISE EXCEPTION 'missing lock mode' USING ERRCODE='22023'; END IF;
 IF p_lock THEN
  PERFORM 1 FROM {s}.streams WHERE stream_identity=p_stream FOR UPDATE;
 END IF;
 PERFORM {s}.verify_stream(p_stream);
 SELECT pg_catalog.jsonb_build_object('canonical_identity',
  pg_catalog.encode(st.stream_identity,'hex'),
  'head_sequence',st.head_sequence,'head_digest',st.head_digest,
  'records',COALESCE((SELECT pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object(
    'sequence',r.sequence,'predecessor_authenticated_digest',r.predecessor_authenticated_digest,
    'event_identity_hex',pg_catalog.encode(r.event_identity,'hex'),'event_digest',r.event_digest,
    'authenticated_digest',r.authenticated_digest,'canonical_record',
  pg_catalog.encode(r.canonical_record,'hex'))
    ORDER BY r.sequence) FROM {s}.records r WHERE r.stream_identity=p_stream),'[]'::jsonb),
  'attestations',COALESCE((SELECT pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object(
    'sequence',h.sequence,'record_digest',h.record_digest,
    'canonical_credential_hex',pg_catalog.encode(h.credential_identity,'hex'),
    'canonical_head',pg_catalog.encode(h.canonical_head,'hex'),'signature',
  pg_catalog.encode(h.signature,'hex'))
    ORDER BY h.sequence,h.credential_identity) FROM {s}.attestations h WHERE
  h.stream_identity=p_stream),'[]'::jsonb))
 INTO result FROM {s}.streams st WHERE st.stream_identity=p_stream;
 RETURN result;
END
""",
        "append_record(bytea,text,json,bytea)": """
DECLARE st {s}.streams%ROWTYPE; prior {s}.records%ROWTYPE;
 physical jsonb; ev bytea; payload_bytes bytea;
BEGIN
 {runtime_guard}
 SELECT * INTO st FROM {s}.streams WHERE stream_identity=p_stream FOR UPDATE;
 IF NOT FOUND THEN RAISE EXCEPTION 'unprovisioned history stream' USING ERRCODE='42501'; END IF;
 PERFORM {s}.verify_stream(p_stream);
 physical := p_record::jsonb;
 IF p_record IS NULL OR p_canonical IS NULL
 OR pg_catalog.jsonb_typeof(physical) IS DISTINCT FROM 'object'
 OR (SELECT pg_catalog.array_agg(k ORDER BY k) FROM pg_catalog.json_object_keys(p_record) k)
   IS DISTINCT FROM ARRAY['authenticated_digest','canonical_event_identity_hex',
  'canonical_payload_hex','event_digest','predecessor_authenticated_digest','sequence']::text[]
 OR pg_catalog.jsonb_typeof(physical->'canonical_event_identity_hex') IS DISTINCT FROM 'string'
 OR pg_catalog.jsonb_typeof(physical->'canonical_payload_hex') IS DISTINCT FROM 'string'
 OR (physical->>'canonical_event_identity_hex') !~ '^([0-9a-f][0-9a-f])+$'
 OR (physical->>'canonical_payload_hex') !~ '^([0-9a-f][0-9a-f])+$'
 THEN RAISE EXCEPTION 'malformed append envelope' USING ERRCODE='22023'; END IF;
 ev := pg_catalog.decode(physical->>'canonical_event_identity_hex','hex');
 PERFORM {s}.validate_canonical_json(ev);
 payload_bytes := pg_catalog.decode(physical->>'canonical_payload_hex','hex');
 PERFORM {s}.validate_canonical_json(payload_bytes);
 PERFORM {s}.validate_canonical_json(p_canonical);
 IF physical->>'event_digest' IS DISTINCT FROM 'sha256:' || pg_catalog.encode(pg_catalog.sha256(
   pg_catalog.decode('{event_domain}','hex') || payload_bytes),'hex')
 THEN RAISE EXCEPTION 'event digest differs from exact payload' USING ERRCODE='22023'; END IF;
 SELECT * INTO prior FROM {s}.records WHERE stream_identity=p_stream AND event_identity=ev;
 IF FOUND THEN
  IF prior.event_digest IS DISTINCT FROM physical->>'event_digest'
 OR {s}.canonical_object_member(prior.canonical_record,'canonical_event_payload')
   IS DISTINCT FROM payload_bytes
  THEN RAISE EXCEPTION 'conflicting idempotency replay' USING ERRCODE='22023'; END IF;
  RETURN true;
 END IF;
 IF p_expected IS DISTINCT FROM st.head_digest THEN
  RAISE EXCEPTION 'stale predecessor CAS' USING ERRCODE='40001';
 END IF;
 IF (physical->'sequence')::text IS DISTINCT FROM (st.head_sequence + 1)::text
 OR physical->>'predecessor_authenticated_digest' IS DISTINCT FROM st.head_digest
 OR physical->>'authenticated_digest' IS DISTINCT FROM 'sha256:' ||
  pg_catalog.encode(pg_catalog.sha256(
   pg_catalog.decode('{record_domain}','hex') || p_canonical),'hex')
 OR {s}.canonical_object_member(p_canonical,'event_digest') IS DISTINCT FROM
    pg_catalog.convert_to(pg_catalog.to_json(physical->>'event_digest')::text,'UTF8')
 OR {s}.canonical_object_member(p_canonical,'canonical_event_payload') IS DISTINCT FROM
  payload_bytes
 THEN RAISE EXCEPTION 'append record envelope differs' USING ERRCODE='22023'; END IF;
 INSERT INTO {s}.records(stream_identity,sequence,predecessor_authenticated_digest,event_identity,
  logical_operation_id,issuance_attempt_id,root_proof_id,event_digest,authenticated_digest,
  canonical_record)
 VALUES(p_stream,st.head_sequence+1,st.head_digest,ev,
  {s}.canonical_object_member(ev,'logical_operation_id'),
  {s}.canonical_object_member(ev,'issuance_attempt_id'),
  {s}.canonical_object_member(ev,'root_proof_id'),
  physical->>'event_digest',physical->>'authenticated_digest',p_canonical);
 UPDATE {s}.streams SET head_sequence=head_sequence+1,
  head_digest=physical->>'authenticated_digest' WHERE stream_identity=p_stream;
 PERFORM {s}.verify_stream(p_stream);
 RETURN false;
END
""",
        "retain_head(bytea,json,bytea,bytea)": """
DECLARE physical jsonb; head_body bytea; credential bytea; prior {s}.attestations%ROWTYPE;
 digest text;
BEGIN
 {runtime_guard}
 PERFORM 1 FROM {s}.streams WHERE stream_identity=p_stream FOR UPDATE;
 PERFORM {s}.verify_stream(p_stream);
 physical := p_head::jsonb;
 IF p_head IS NULL OR p_canonical IS NULL OR p_signature IS NULL
 OR pg_catalog.jsonb_typeof(physical) IS DISTINCT FROM 'object'
 OR (SELECT pg_catalog.array_agg(k ORDER BY k) FROM pg_catalog.json_object_keys(p_head) k)
  IS DISTINCT FROM ARRAY['canonical_credential_hex','record_digest','sequence']::text[]
 OR pg_catalog.jsonb_typeof(physical->'canonical_credential_hex') IS DISTINCT FROM 'string'
 OR physical->>'canonical_credential_hex' !~ '^([0-9a-f][0-9a-f])+$'
 OR pg_catalog.octet_length(p_signature) <> 64
 OR pg_catalog.substring(p_canonical,1,{attestation_length}) IS DISTINCT FROM
  pg_catalog.decode('{attestation_domain}','hex')
 THEN RAISE EXCEPTION 'malformed attested head envelope' USING ERRCODE='22023'; END IF;
 credential := pg_catalog.decode(physical->>'canonical_credential_hex','hex');
 head_body := pg_catalog.substring(p_canonical,{attestation_offset});
 PERFORM {s}.validate_canonical_json(credential);
 PERFORM {s}.validate_canonical_json(head_body);
 SELECT authenticated_digest INTO digest FROM {s}.records
  WHERE stream_identity=p_stream AND sequence=(physical->>'sequence')::bigint;
 IF NOT FOUND OR digest IS DISTINCT FROM physical->>'record_digest'
 OR {s}.canonical_object_member(head_body,'stream') IS DISTINCT FROM p_stream
 OR {s}.canonical_object_member(head_body,'sequence') IS DISTINCT FROM
  pg_catalog.convert_to((physical->'sequence')::text,'UTF8')
 OR {s}.canonical_object_member(head_body,'record_digest') IS DISTINCT FROM
  pg_catalog.convert_to(pg_catalog.to_json(digest)::text,'UTF8')
 OR {s}.canonical_object_member(head_body,'signing_credential_identity') IS DISTINCT FROM credential
 THEN RAISE EXCEPTION 'attested head does not name retained record' USING ERRCODE='22023'; END IF;
 SELECT * INTO prior FROM {s}.attestations WHERE stream_identity=p_stream
  AND sequence=(physical->>'sequence')::bigint;
 IF FOUND THEN
  IF prior.credential_identity IS DISTINCT FROM credential
  OR prior.record_digest IS DISTINCT FROM digest OR prior.canonical_head IS DISTINCT FROM
  p_canonical
  OR prior.signature IS DISTINCT FROM p_signature THEN
   RAISE EXCEPTION 'conflicting attested head replay' USING ERRCODE='22023';
  END IF;
  RETURN true;
 END IF;
 INSERT INTO {s}.attestations(stream_identity,sequence,record_digest,credential_identity,
  canonical_head,signature)
 VALUES(p_stream,(physical->>'sequence')::bigint,digest,credential,p_canonical,p_signature);
 PERFORM {s}.verify_stream(p_stream);
 RETURN false;
END
""",
    }
    replacements = {
        "{s}": s,
        "{runtime_guard}": runtime_guard,
        "{admin_guard}": admin_guard,
        "{record_domain}": record_domain,
        "{event_domain}": event_domain,
        "{attestation_domain}": attestation_domain,
        "{attestation_length}": str(len(ATTESTATION_DOMAIN)),
        "{attestation_offset}": str(len(ATTESTATION_DOMAIN) + 1),
    }
    for signature, source in sources.items():
        for placeholder, value in replacements.items():
            source = source.replace(placeholder, value)
        sources[signature] = source
    sources = {**canonical_function_sources(config.schema), **sources}
    return sources


def _create_tables(
    conn: psycopg.Connection[Any], config: PostgreSQLIssuerHistoryProvisioning
) -> None:
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
CREATE TABLE {s}.streams (
 stream_identity bytea PRIMARY KEY,stream_id bytea NOT NULL,
 issuer_authority_identity bytea NOT NULL,security_profile bytea NOT NULL,
 environment bytea NOT NULL,trust_domain bytea NOT NULL,product_scope bytea NOT NULL,
 security_epoch text NOT NULL CHECK(security_epoch ~ '^[1-9][0-9]*$'),
 head_sequence bigint NOT NULL CHECK(head_sequence>=0),head_digest text NOT NULL
);
CREATE TABLE {s}.records (
 stream_identity bytea NOT NULL REFERENCES {s}.streams(stream_identity),
 sequence bigint NOT NULL CHECK(sequence>=1),predecessor_authenticated_digest text NOT NULL,
 event_identity bytea NOT NULL,logical_operation_id bytea NOT NULL,
 issuance_attempt_id bytea NOT NULL,root_proof_id bytea NOT NULL,
 event_digest text NOT NULL CHECK(event_digest ~ '^sha256:[0-9a-f]{{64}}$'),
 authenticated_digest text NOT NULL CHECK(authenticated_digest ~ '^sha256:[0-9a-f]{{64}}$'),
 canonical_record bytea NOT NULL,
 PRIMARY KEY(stream_identity,sequence),UNIQUE(stream_identity,event_identity),
 UNIQUE(stream_identity,logical_operation_id,issuance_attempt_id,root_proof_id)
);
CREATE TABLE {s}.attestations (
 stream_identity bytea NOT NULL,sequence bigint NOT NULL,
 record_digest text NOT NULL,credential_identity bytea NOT NULL,
 canonical_head bytea NOT NULL,signature bytea NOT NULL CHECK(octet_length(signature)=64),
 PRIMARY KEY(stream_identity,sequence),
 FOREIGN KEY(stream_identity,sequence) REFERENCES {s}.records(stream_identity,sequence)
);
REVOKE ALL ON ALL TABLES IN SCHEMA {s} FROM PUBLIC;
""").format(s=sql.Identifier(config.schema))
    )


_FUNCTION_DECLARATIONS = {
    "validate_members(bytea,text[])": ("p_bytes bytea,p_fields text[]", "void"),
    "canonical_object_member(bytea,text)": ("p_bytes bytea,p_key text", "bytea"),
    "canonical_string(bytea,integer)": (
        "p_bytes bytea,p_position integer",
        CANONICAL_FUNCTION_RETURNS["canonical_string(bytea,integer)"],
    ),
    "canonical_value(bytea,integer)": ("p_bytes bytea,p_position integer", "integer"),
    "validate_canonical_json(bytea)": ("p_bytes bytea", "void"),
    "validate_stream(bytea)": ("p_stream bytea", "void"),
    "verify_stream(bytea)": ("p_stream bytea", "void"),
    "provision_stream(bytea)": ("p_stream bytea", "void"),
    "read_stream(bytea,boolean)": ("p_stream bytea,p_lock boolean", "jsonb"),
    "append_record(bytea,text,json,bytea)": (
        "p_stream bytea,p_expected text,p_record json,p_canonical bytea",
        "boolean",
    ),
    "retain_head(bytea,json,bytea,bytea)": (
        "p_stream bytea,p_head json,p_canonical bytea,p_signature bytea",
        "boolean",
    ),
}


def _physical_fingerprint(conn: psycopg.Connection[Any], schema: str) -> str:
    """Bind reviewed installed DDL, excluding role OIDs and schema-dependent names."""
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
    # Names in references and index definitions are generated by this installer.
    payload = json.dumps(physical, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(
        b"CryptoHunter/PostgreSQLIssuerHistoryPhysical/v1\0" + payload
    ).hexdigest()


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
    "streams": (
        ("stream_identity", "bytea"),
        ("stream_id", "bytea"),
        ("issuer_authority_identity", "bytea"),
        ("security_profile", "bytea"),
        ("environment", "bytea"),
        ("trust_domain", "bytea"),
        ("product_scope", "bytea"),
        ("security_epoch", "text"),
        ("head_sequence", "bigint"),
        ("head_digest", "text"),
    ),
    "records": (
        ("stream_identity", "bytea"),
        ("sequence", "bigint"),
        ("predecessor_authenticated_digest", "text"),
        ("event_identity", "bytea"),
        ("logical_operation_id", "bytea"),
        ("issuance_attempt_id", "bytea"),
        ("root_proof_id", "bytea"),
        ("event_digest", "text"),
        ("authenticated_digest", "text"),
        ("canonical_record", "bytea"),
    ),
    "attestations": (
        ("stream_identity", "bytea"),
        ("sequence", "bigint"),
        ("record_digest", "text"),
        ("credential_identity", "bytea"),
        ("canonical_head", "bytea"),
        ("signature", "bytea"),
    ),
}


def _qualify_physical_definition(
    conn: psycopg.Connection[Any], schema: str, owner_oid: int
) -> None:
    """Anchor DDL to reviewed local code independently of mutable metadata."""
    quoted_schema = conn.execute("SELECT pg_catalog.quote_ident(%s)", (schema,)).fetchone()[0]
    columns = conn.execute(
        "SELECT c.relname,a.attnum,a.attname,pg_catalog.format_type(a.atttypid,a.atttypmod),"
        "a.attnotnull,pg_catalog.pg_get_expr(d.adbin,d.adrelid),a.attgenerated,a.attidentity,"
        "co.collname,cn.nspname,co.collisdeterministic FROM pg_catalog.pg_class c "
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
        )
        for table in sorted(_REVIEWED_COLUMNS)
        for index, (name, data_type) in enumerate(_REVIEWED_COLUMNS[table], 1)
    ]
    if columns != expected_columns:
        raise HistorySchemaQualificationError(
            "history columns differ from reviewed local definition"
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
        ("streams", "p", "PRIMARY KEY (stream_identity)"),
        ("streams", "c", "CHECK ((security_epoch ~ '^[1-9][0-9]*$'::text))"),
        ("streams", "c", "CHECK ((head_sequence >= 0))"),
        ("records", "p", "PRIMARY KEY (stream_identity, sequence)"),
        ("records", "u", "UNIQUE (stream_identity, event_identity)"),
        (
            "records",
            "u",
            "UNIQUE (stream_identity, logical_operation_id, issuance_attempt_id, root_proof_id)",
        ),
        (
            "records",
            "f",
            f"FOREIGN KEY (stream_identity) REFERENCES {quoted_schema}.streams(stream_identity)",
        ),
        ("records", "c", "CHECK ((sequence >= 1))"),
        ("records", "c", "CHECK ((event_digest ~ '^sha256:[0-9a-f]{64}$'::text))"),
        ("records", "c", "CHECK ((authenticated_digest ~ '^sha256:[0-9a-f]{64}$'::text))"),
        ("attestations", "p", "PRIMARY KEY (stream_identity, sequence)"),
        (
            "attestations",
            "f",
            "FOREIGN KEY (stream_identity, sequence) REFERENCES "
            f"{quoted_schema}.records(stream_identity, sequence)",
        ),
        ("attestations", "c", "CHECK ((octet_length(signature) = 64))"),
    }
    expected_constraints = {(*definition, False, False, True) for definition in definitions}
    if len(constraints) != len(expected_constraints) or set(constraints) != expected_constraints:
        raise HistorySchemaQualificationError(
            "history constraints differ from reviewed local definition"
        )
    indexes = conn.execute(
        "SELECT t.relname,ARRAY(SELECT a.attname FROM pg_catalog.unnest(i.indkey::smallint[]) "
        "WITH ORDINALITY k(attnum,pos) JOIN pg_catalog.pg_attribute a ON a.attrelid=t.oid "
        "AND a.attnum=k.attnum ORDER BY k.pos),i.indisunique,i.indisprimary,i.indisvalid,"
        "i.indisready,i.indisexclusion,i.indimmediate,"
        "pg_catalog.pg_get_expr(i.indexprs,i.indrelid),"
        "pg_catalog.pg_get_expr(i.indpred,i.indrelid),am.amname,idx.relowner,"
        "i.indnkeyatts,i.indnatts,pg_catalog.pg_get_indexdef(i.indexrelid) "
        "FROM pg_catalog.pg_index i JOIN pg_catalog.pg_class t ON t.oid=i.indrelid "
        "JOIN pg_catalog.pg_class idx ON idx.oid=i.indexrelid "
        "JOIN pg_catalog.pg_am am ON am.oid=idx.relam "
        "JOIN pg_catalog.pg_namespace n ON n.oid=t.relnamespace WHERE n.nspname=%s",
        (schema,),
    ).fetchall()
    expected_shapes = {
        ("metadata", ("singleton",), True),
        ("streams", ("stream_identity",), True),
        ("records", ("stream_identity", "sequence"), True),
        ("records", ("stream_identity", "event_identity"), False),
        (
            "records",
            ("stream_identity", "logical_operation_id", "issuance_attempt_id", "root_proof_id"),
            False,
        ),
        ("attestations", ("stream_identity", "sequence"), True),
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
            "btree (" + ", ".join(keys) + ")",
        )
        for table, keys, primary in expected_shapes
    }
    observed_indexes = {
        (r[0], tuple(r[1]), *r[2:-1], r[-1].split(" USING ", 1)[-1]) for r in indexes
    }
    if len(indexes) != len(expected_indexes) or observed_indexes != expected_indexes:
        raise HistorySchemaQualificationError(
            "history indexes differ from reviewed local definition"
        )


def provision_postgresql_issuer_history(
    bootstrap: PostgreSQLConnectionConfig,
    config: PostgreSQLIssuerHistoryProvisioning,
) -> None:
    """Install a fresh schema atomically; no credentials or signing keys are stored.

    Role login authentication is configured separately by the database operator.
    Reinstallation and in-place upgrades are deliberately rejected by CREATE.
    """
    if (
        type(bootstrap) is not PostgreSQLConnectionConfig
        or type(config) is not PostgreSQLIssuerHistoryProvisioning
    ):
        raise TypeError("exact connection and provisioning configuration required")
    with psycopg.connect(bootstrap.dsn, autocommit=True) as conn, conn.transaction():
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


def provision_postgresql_issuer_history_stream(
    connection: PostgreSQLConnectionConfig,
    *,
    schema: str,
    stream: HistoryStreamIdentity,
) -> None:
    """Authorize one exact stream using a separate direct admin login."""
    _identifier(schema)
    if type(stream) is not HistoryStreamIdentity:
        raise TypeError("stream must be exact HistoryStreamIdentity")
    with psycopg.connect(connection.dsn, autocommit=True) as conn, conn.transaction():
        qualify_history_schema(
            conn, schema, expected_role="admin_role", trust_domain=stream.trust_domain
        )
        conn.execute(
            sql.SQL("SELECT {}.provision_stream(%s)").format(sql.Identifier(schema)),
            (canonical_json_bytes(stream.material()),),
        )


def qualify_history_schema(
    conn: psycopg.Connection[Any],
    schema: str,
    *,
    expected_role: str,
    trust_domain: str,
) -> None:
    """Qualify the direct principal, durability, reviewed DDL, source and all ACLs.

    This is performed inside every authority operation's transaction.  Catalog
    function names and queries use pg_catalog irrespective of caller search_path.
    """
    _identifier(schema)
    if expected_role not in ("runtime_role", "admin_role"):
        raise ValueError("invalid expected history role")
    try:
        _qualify_history_schema(conn, schema, expected_role, trust_domain)
    except HistorySchemaQualificationError:
        raise
    except (psycopg.Error, ValueError, TypeError, KeyError, IndexError) as exc:
        raise HistorySchemaQualificationError(
            "PostgreSQL issuer history qualification failed"
        ) from exc


def _qualify_history_schema(
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
        raise HistorySchemaQualificationError("history metadata cardinality mismatch")
    metadata = row[0]
    if metadata[:5] != (
        SCHEMA_IDENTITY,
        SCHEMA_VERSION,
        PROFILE,
        _scalar(trust_domain),
        STORAGE_FAMILY,
    ):
        raise HistorySchemaQualificationError("history security metadata mismatch")
    config = PostgreSQLIssuerHistoryProvisioning(
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
            raise HistorySchemaQualificationError("history role identity or attributes mismatch")
    principal = conn.execute("SELECT session_user::text,current_user::text").fetchone()
    if principal != (roles[expected_role][0], roles[expected_role][0]):
        raise HistorySchemaQualificationError("history requires its exact direct-login principal")
    if (
        conn.execute(
            "SELECT 1 FROM pg_catalog.pg_auth_members "
            "WHERE roleid=ANY(%s) OR member=ANY(%s) LIMIT 1",
            (role_oids, role_oids),
        ).fetchone()
        is not None
    ):
        raise HistorySchemaQualificationError("history authority role has a membership grant")
    if conn.info.server_version < 160000 or conn.execute(
        "SELECT pg_catalog.current_setting('fsync'),"
        "pg_catalog.current_setting('synchronous_commit')"
    ).fetchone() != ("on", "on"):
        raise HistorySchemaQualificationError("history PostgreSQL durability mismatch")
    owner_oid = roles["schema_owner_role"][1]
    namespace = conn.execute(
        "SELECT oid,nspowner FROM pg_catalog.pg_namespace WHERE nspname=%s", (schema,)
    ).fetchone()
    if namespace is None or namespace[1] != owner_oid:
        raise HistorySchemaQualificationError("history schema owner mismatch")
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
        raise HistorySchemaQualificationError("history schema ACL differs from allowlist")
    relations = conn.execute(
        "SELECT oid,relname,relkind,relpersistence,relispartition,relrowsecurity,"
        "relforcerowsecurity,relowner FROM pg_catalog.pg_class "
        "WHERE relnamespace=%s AND relkind NOT IN ('i','S','t') ORDER BY relname",
        (namespace[0],),
    ).fetchall()
    if [r[1:] for r in relations] != [
        (name, "r", "p", False, False, False, owner_oid) for name in sorted(_TABLES)
    ]:
        raise HistorySchemaQualificationError("history relations differ from reviewed schema")
    relation_oids = [r[0] for r in relations]
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
            raise HistorySchemaQualificationError(
                "history relation has unreviewed auxiliary behavior or column ACL"
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
        raise HistorySchemaQualificationError("history table ACL differs from allowlist")
    _qualify_physical_definition(conn, schema, owner_oid)
    if _physical_fingerprint(conn, schema) != metadata[12]:
        raise HistorySchemaQualificationError("history physical definition fingerprint mismatch")
    sources = _function_sources(config)
    manifest = {
        signature: _source_fingerprint(signature, source) for signature, source in sources.items()
    }
    if metadata[11] != manifest:
        raise HistorySchemaQualificationError("history function manifest mismatch")
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
        raise HistorySchemaQualificationError("history function inventory mismatch")
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
            raise HistorySchemaQualificationError(
                "history function source or security metadata mismatch"
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
            raise HistorySchemaQualificationError("history function ACL differs from allowlist")


__all__ = [
    "HistorySchemaQualificationError",
    "PostgreSQLConnectionConfig",
    "PostgreSQLIssuerHistoryProvisioning",
    "SCHEMA_IDENTITY",
    "SCHEMA_VERSION",
    "STORAGE_FAMILY",
    "provision_postgresql_issuer_history",
    "provision_postgresql_issuer_history_stream",
    "qualify_history_schema",
]
