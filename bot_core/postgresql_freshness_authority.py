"""Low-level PostgreSQL FreshnessAuthority authority boundary.

This module intentionally is not a high-level AccountGenesis adapter.  It
provisions and qualifies the database objects which own freshness decisions.
Callers can only mutate them through narrowly granted SECURITY DEFINER
functions; in particular a JSON value supplied by a runtime process is never
treated as proof of verification or key liveness.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import base64
from typing import Any

import psycopg
from psycopg import sql

SCHEMA_IDENTITY = "cryptohunter.freshness_authority.postgresql"
SCHEMA_VERSION = 1
PROFILE = "PRODUCTION_LOCAL"
PROPOSER_ROLE = "ACCOUNT_GENESIS_FRESHNESS_PROPOSER_SIGNING_V1"
FINALIZATION_ROLE = "ACCOUNT_GENESIS_FRESHNESS_AUTHORITY_FINALIZATION_SIGNING_V1"
_SAFE = re.compile(r"[a-z_][a-z0-9_]{0,62}\Z")
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_REVIEWED_PHYSICAL_FINGERPRINT = "a5f737dcf2fd49f14ed8d3728e2803ad406310faa12a984a5cbecb99a538e8ad"
DOCUMENT_DOMAIN = b"cryptohunter.account-genesis.freshness-document-digest.v1\x00"
RECEIPT_DOMAIN = b"cryptohunter.account-genesis.freshness-finalization-receipt-authentication.v1\x00"
COMPLETE_HEAD_DOMAIN = b"cryptohunter.account-genesis.complete-semantic-head-set-digest.v1\x00"
PREPARATION_FIELDS = frozenset({
    "schema_version", "security_profile", "environment", "trust_domain",
    "authority_id", "operation_type", "expected_predecessor_generation",
    "expected_predecessor_document_digest",
    "expected_predecessor_complete_semantic_head_digest",
    "proposed_document_digest", "original_decision_identity",
    "proposer_identity", "proposer_credential_role_identity",
    "proposer_key_version",
    "proposer_lifecycle_generation_observed_for_key_binding_only",
    "proposer_public_key_material_identity", "proposer_authentication_digest",
    "finalization_credential_role_identity", "finalization_key_version",
    "finalization_lifecycle_generation_observed_for_key_binding_only",
    "finalization_public_key_material_identity",
    "authoritative_document_authentication_digest", "receipt_id",
    "receipt_canonical_digest", "finalization_request_id", "preparation_id",
    "verifier_authority_identity", "verifier_authority_version",
})
RECEIPT_FIELDS = frozenset({
    "schema_version", "environment", "trust_domain", "authority_id",
    "exact_predecessor_generation", "exact_predecessor_document_digest",
    "accepted_generation", "accepted_document_digest",
    "complete_semantic_head_digest", "finalization_request_id", "receipt_id",
    "freshness_authority_key_id", "freshness_authority_key_version",
    "authentication_tag_or_signature",
})
DOCUMENT_PAYLOAD_FIELDS = frozenset({
    "schema_version", "environment", "trust_domain", "authority_id",
    "generation", "predecessor_generation", "predecessor_document_digest",
    "complete_semantic_head_set", "freshness_authority_key_id",
    "freshness_authority_key_version", "finalization_request_id",
})
DOCUMENT_FIELDS = frozenset({"payload", "document_digest", "authentication_tag_or_signature"})


class FreshnessAuthorityQualificationError(RuntimeError):
    """The database is not the exact reviewed authority schema."""


@dataclass(frozen=True, slots=True)
class PostgreSQLConnectionConfig:
    dsn: str

    def __post_init__(self) -> None:
        if type(self.dsn) is not str or not self.dsn.strip():
            raise TypeError("dsn must be an exact non-empty str")

    def __repr__(self) -> str:
        return "PostgreSQLConnectionConfig(dsn=<redacted>)"


@dataclass(frozen=True, slots=True)
class PostgreSQLFreshnessAuthorityProvisioning:
    schema: str = "freshness_authority"
    schema_owner_role: str = "freshness_schema_owner"
    function_owner_role: str = "freshness_function_owner"
    admin_role: str = "freshness_admin"
    verifier_role: str = "freshness_crypto_verifier"
    runtime_role: str = "freshness_runtime"
    reader_role: str = "freshness_reader"

    def __post_init__(self) -> None:
        names = (self.schema, self.schema_owner_role, self.function_owner_role,
                 self.admin_role, self.verifier_role, self.runtime_role,
                 self.reader_role)
        if any(type(x) is not str or _SAFE.fullmatch(x) is None for x in names):
            raise ValueError("all names must be safe lowercase PostgreSQL identifiers")
        if len(set(names[1:])) != 6:
            raise ValueError("all authority roles must be distinct")


def canonical_json_bytes(value: object) -> bytes:
    """Restricted RFC 8785 JCS (the frozen schema forbids floating point)."""
    def check(v: object) -> None:
        if v is None or type(v) in (str, bool):
            if type(v) is str:
                v.encode("utf-8", "strict")
            return
        if type(v) is int:
            if not -(2**53 - 1) <= v <= 2**53 - 1:
                raise ValueError("integer outside JSON safe integer range")
            return
        if type(v) is list:
            for x in v: check(x)
            return
        if type(v) is dict:
            if any(type(k) is not str for k in v):
                raise ValueError("JSON object keys must be strings")
            for k, x in v.items(): k.encode("utf-8", "strict"); check(x)
            return
        raise ValueError("floats and non-JSON values are forbidden")
    check(value)
    def jcs_order(item: tuple[str, object]) -> bytes:
        return item[0].encode("utf-16-be")

    def ordered(v: object) -> object:
        if type(v) is dict:
            return {key: ordered(item) for key, item in sorted(v.items(), key=jcs_order)}
        if type(v) is list:
            return [ordered(item) for item in v]
        return v

    return json.dumps(ordered(value), ensure_ascii=False, sort_keys=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def parse_canonical_json(raw: bytes | str) -> object:
    """Parse closed cryptographic JSON, rejecting duplicates before mapping."""
    if type(raw) not in (bytes, str):
        raise TypeError("canonical JSON input must be exact bytes or str")
    if type(raw) is bytes:
        raw = raw.decode("utf-8", "strict")
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        out: dict[str, object] = {}
        for key, value in items:
            if key in out:
                raise ValueError("duplicate JSON object key")
            out[key] = value
        return out
    value = json.loads(raw, object_pairs_hook=pairs,
                       parse_float=lambda _: (_ for _ in ()).throw(ValueError("floats forbidden")),
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite forbidden")))
    canonical = canonical_json_bytes(value)
    if raw.encode("utf-8") != canonical:
        raise ValueError("input is not the exact canonical JCS representation")
    return value


def _canonical(value: object) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + canonical_json_bytes(value)).hexdigest()


def normalize_complete_semantic_head_set(value: object) -> list[object]:
    """Normalize the sole frozen set-like field by canonical UTF-8 bytes."""
    if type(value) is not list:
        raise TypeError("complete_semantic_head_set must be an exact list")
    decorated = sorted(((canonical_json_bytes(item), item) for item in value),
                       key=lambda pair: pair[0])
    if any(left[0] == right[0] for left, right in zip(decorated, decorated[1:])):
        raise ValueError("duplicate complete semantic head")
    return [item for _, item in decorated]


def complete_semantic_head_digest(value: object) -> str:
    return _digest(COMPLETE_HEAD_DOMAIN, normalize_complete_semantic_head_set(value))


def _valid_signature(value: object) -> bool:
    if type(value) is not str or "=" in value or not value:
        return False
    try:
        decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except Exception:
        return False
    return len(decoded) == 64 and base64.urlsafe_b64encode(decoded).rstrip(b"=").decode() == value


def _physical_fingerprint(conn: psycopg.Connection[Any], schema: str) -> str:
    """Fingerprint every reviewed physical column, constraint and index fact."""
    facts = conn.execute("""
      SELECT 'column',c.relname,a.attnum::text,a.attname,
             pg_catalog.format_type(a.atttypid,a.atttypmod),a.attnotnull::text,
             a.attidentity::text,a.attgenerated::text,
             coalesce(pg_catalog.pg_get_expr(d.adbin,d.adrelid),'')
      FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
      JOIN pg_catalog.pg_attribute a ON a.attrelid=c.oid AND a.attnum>0 AND NOT a.attisdropped
      LEFT JOIN pg_catalog.pg_attrdef d ON d.adrelid=c.oid AND d.adnum=a.attnum
      WHERE n.nspname=%s AND c.relkind IN ('r','S')
      UNION ALL
      SELECT 'constraint',c.relname,con.contype::text,con.conname,
             pg_catalog.pg_get_constraintdef(con.oid,true),'','','',''
      FROM pg_catalog.pg_constraint con JOIN pg_catalog.pg_class c ON c.oid=con.conrelid
      JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname=%s
      UNION ALL
      SELECT 'index',t.relname,i.indisprimary::text,i.indisunique::text,ic.relname,
             pg_catalog.pg_get_indexdef(ic.oid),'','',''
      FROM pg_catalog.pg_index i JOIN pg_catalog.pg_class t ON t.oid=i.indrelid
      JOIN pg_catalog.pg_class ic ON ic.oid=i.indexrelid
      JOIN pg_catalog.pg_namespace n ON n.oid=t.relnamespace WHERE n.nspname=%s
      ORDER BY 1,2,3,4,5
    """, (schema, schema, schema)).fetchall()
    return hashlib.sha256(json.dumps(facts, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def _sources(schema: str) -> dict[str, str]:
    q = '"' + schema.replace('"', '""') + '"'
    guard = "IF session_user <> %s OR current_user <> 'freshness_function_owner' THEN RAISE EXCEPTION 'wrong authenticated principal' USING ERRCODE='42501'; END IF;"
    prep_keys = ",".join("'" + x + "'" for x in sorted(PREPARATION_FIELDS))
    receipt_keys = ",".join("'" + x + "'" for x in sorted(RECEIPT_FIELDS))
    payload_keys = ",".join("'" + x + "'" for x in sorted(DOCUMENT_PAYLOAD_FIELDS))
    document_keys = ",".join("'" + x + "'" for x in sorted(DOCUMENT_FIELDS))
    def string_matrix(var: str, fields: set[str]) -> str:
        return " OR ".join(
            f"pg_catalog.jsonb_typeof({var}->'{field}') IS DISTINCT FROM 'string'"
            for field in sorted(fields)
        )

    def integer_matrix(var: str, fields: dict[str, int]) -> str:
        return " OR ".join(
            f"pg_catalog.jsonb_typeof({var}->'{field}') IS DISTINCT FROM 'number' OR "
            f"({var}->'{field}')::text !~ '^(0|[1-9][0-9]*)$' OR "
            f"(({var}->'{field}')::text)::numeric NOT BETWEEN {minimum} AND 9007199254740991"
            for field, minimum in sorted(fields.items())
        )

    prep_integers = {
        "schema_version": 1,
        "expected_predecessor_generation": 0,
        "proposer_key_version": 1,
        "proposer_lifecycle_generation_observed_for_key_binding_only": 1,
        "finalization_key_version": 1,
        "finalization_lifecycle_generation_observed_for_key_binding_only": 1,
        "verifier_authority_version": 1,
    }
    prep_strings = set(PREPARATION_FIELDS) - set(prep_integers)
    payload_integers = {
        "schema_version": 1, "generation": 1, "predecessor_generation": 0,
        "freshness_authority_key_version": 1,
    }
    payload_strings = set(DOCUMENT_PAYLOAD_FIELDS) - set(payload_integers) - {"complete_semantic_head_set"}
    receipt_integers = {
        "schema_version": 1, "exact_predecessor_generation": 0,
        "accepted_generation": 1, "freshness_authority_key_version": 1,
    }
    receipt_strings = set(RECEIPT_FIELDS) - set(receipt_integers)
    digest_fields = {
        "expected_predecessor_document_digest",
        "expected_predecessor_complete_semantic_head_digest",
        "proposed_document_digest", "proposer_public_key_material_identity",
        "proposer_authentication_digest", "finalization_public_key_material_identity",
        "authoritative_document_authentication_digest", "receipt_canonical_digest",
    }
    prep_digest_check = " OR ".join(
        f"p->>'{field}' !~ '^[0-9a-f]{{64}}$'" for field in sorted(digest_fields)
    )
    return {
      "jcs_sort_key(text)": r"""
DECLARE i integer; cp integer; shifted integer; result text:='';
BEGIN
 FOR i IN 1..pg_catalog.char_length($1) LOOP
   cp:=pg_catalog.ascii(pg_catalog.substr($1,i,1));
   IF cp<=65535 THEN result:=result||pg_catalog.lpad(pg_catalog.to_hex(cp),4,'0');
   ELSE shifted:=cp-65536; result:=result||pg_catalog.lpad(pg_catalog.to_hex(55296+(shifted/1024)),4,'0')||pg_catalog.lpad(pg_catalog.to_hex(56320+(shifted%1024)),4,'0'); END IF;
 END LOOP;
 RETURN result;
END""",
      "canonical_jsonb(jsonb)": r"""
DECLARE k text; item jsonb; result text; kind text := pg_catalog.jsonb_typeof($1);
BEGIN
 IF kind='object' THEN
   result:='{';
   FOR k,item IN SELECT key,value FROM pg_catalog.jsonb_each($1) ORDER BY freshness_authority.jcs_sort_key(key) COLLATE "C" LOOP
     IF result<>'{' THEN result:=result||','; END IF;
     result:=result||pg_catalog.to_jsonb(k)::text||':'||pg_catalog.convert_from(freshness_authority.canonical_jsonb(item),'UTF8');
   END LOOP; RETURN pg_catalog.convert_to(result||'}','UTF8');
 ELSIF kind='array' THEN
   result:='[';
   FOR item IN SELECT value FROM pg_catalog.jsonb_array_elements($1) WITH ORDINALITY ORDER BY ordinality LOOP
     IF result<>'[' THEN result:=result||','; END IF;
     result:=result||pg_catalog.convert_from(freshness_authority.canonical_jsonb(item),'UTF8');
   END LOOP; RETURN pg_catalog.convert_to(result||']','UTF8');
 ELSIF kind='number' THEN
   IF ($1::text !~ '^-?(0|[1-9][0-9]*)$') OR (($1::text)::numeric NOT BETWEEN -9007199254740991 AND 9007199254740991) THEN RAISE EXCEPTION 'non-safe integer' USING ERRCODE='22023'; END IF;
   RETURN pg_catalog.convert_to($1::text,'UTF8');
 ELSIF kind IN ('string','boolean','null') THEN RETURN pg_catalog.convert_to($1::text,'UTF8');
 END IF;
 RAISE EXCEPTION 'invalid JSON value' USING ERRCODE='22023';
END""",
      "canonical_complete_head_set(jsonb)": r"""
DECLARE item jsonb; previous bytea; current bytea; result text:='[';
BEGIN
 IF pg_catalog.jsonb_typeof($1)<>'array' THEN RAISE EXCEPTION 'complete semantic head set must be array' USING ERRCODE='22023'; END IF;
 FOR item IN SELECT value FROM pg_catalog.jsonb_array_elements($1) WITH ORDINALITY ORDER BY ordinality LOOP
   current:=freshness_authority.canonical_jsonb(item);
   IF previous IS NOT NULL AND current<=previous THEN RAISE EXCEPTION 'complete semantic head set not uniquely sorted' USING ERRCODE='22023'; END IF;
   IF result<>'[' THEN result:=result||','; END IF; result:=result||pg_catalog.convert_from(current,'UTF8'); previous:=current;
 END LOOP;
 RETURN pg_catalog.convert_to(result||']','UTF8');
END""",
      "provision_authority(text,text,text,text,text)": f"""
BEGIN
 {guard % "'freshness_admin'"}
 IF $1<>'PRODUCTION' OR $2='' OR $3='' OR $4!~'^[0-9a-f]{{64}}$' OR $5!~'^[0-9a-f]{{64}}$' THEN RAISE EXCEPTION 'invalid authority scope' USING ERRCODE='22023'; END IF;
 INSERT INTO {q}.authority_lineages(environment,trust_domain,authority_id,current_generation,current_document_digest,current_complete_head_digest) VALUES($1,$2,$3,0,$4,$5);
 INSERT INTO {q}.authority_generation_heads(environment,trust_domain,authority_id,generation,document_digest,complete_head_digest) VALUES($1,$2,$3,0,$4,$5);
END""",
      "provision_credential(jsonb)": f"""
DECLARE p jsonb:=$1; material bytea; material_id text; bound_role text;
BEGIN
 {guard % "'freshness_admin'"}
 IF pg_catalog.jsonb_typeof(p)<>'object' OR (SELECT count(*) FROM pg_catalog.jsonb_object_keys(p))<>10 OR NOT p ?& ARRAY['security_profile','environment','trust_domain','authority_id','credential_id','semantic_identity','semantic_role','key_id','key_version','public_key_hex'] THEN RAISE EXCEPTION 'credential schema' USING ERRCODE='22023'; END IF;
 IF pg_catalog.jsonb_typeof(p->'security_profile') IS DISTINCT FROM 'string' OR pg_catalog.jsonb_typeof(p->'environment') IS DISTINCT FROM 'string' OR pg_catalog.jsonb_typeof(p->'trust_domain') IS DISTINCT FROM 'string' OR pg_catalog.jsonb_typeof(p->'authority_id') IS DISTINCT FROM 'string' OR pg_catalog.jsonb_typeof(p->'credential_id') IS DISTINCT FROM 'string' OR pg_catalog.jsonb_typeof(p->'semantic_identity') IS DISTINCT FROM 'string' OR pg_catalog.jsonb_typeof(p->'semantic_role') IS DISTINCT FROM 'string' OR pg_catalog.jsonb_typeof(p->'key_id') IS DISTINCT FROM 'string' OR pg_catalog.jsonb_typeof(p->'key_version') IS DISTINCT FROM 'number' OR pg_catalog.jsonb_typeof(p->'public_key_hex') IS DISTINCT FROM 'string' THEN RAISE EXCEPTION 'credential field types' USING ERRCODE='22023'; END IF;
 IF p->>'semantic_role' NOT IN ('{PROPOSER_ROLE}','{FINALIZATION_ROLE}') OR p->>'environment'<>'PRODUCTION' OR p->>'security_profile'<>'PRODUCTION_LOCAL' OR p->>'trust_domain'='' OR p->>'authority_id'='' OR p->>'credential_id'='' OR p->>'semantic_identity'='' OR p->>'key_id'='' OR (p->'key_version')::text!~'^[1-9][0-9]*$' OR ((p->'key_version')::text)::numeric>9007199254740991 OR p->>'public_key_hex' !~ '^[0-9a-f]{{64}}$' THEN RAISE EXCEPTION 'credential semantics' USING ERRCODE='22023'; END IF;
 material:=pg_catalog.decode(p->>'public_key_hex','hex'); IF pg_catalog.length(material)<>32 THEN RAISE EXCEPTION 'public key must be 32 bytes' USING ERRCODE='22023'; END IF;
 material_id:=pg_catalog.encode(pg_catalog.sha256(material),'hex');
 INSERT INTO {q}.key_material_role_bindings(environment,trust_domain,authority_id,public_key_material_identity,semantic_role,public_key) VALUES(p->>'environment',p->>'trust_domain',p->>'authority_id',material_id,p->>'semantic_role',material) ON CONFLICT DO NOTHING;
 SELECT semantic_role INTO bound_role FROM {q}.key_material_role_bindings WHERE environment=p->>'environment' AND trust_domain=p->>'trust_domain' AND authority_id=p->>'authority_id' AND public_key_material_identity=material_id FOR UPDATE;
 IF bound_role IS DISTINCT FROM p->>'semantic_role' THEN RAISE EXCEPTION 'cross-role key alias' USING ERRCODE='23514'; END IF;
 INSERT INTO {q}.credentials(environment,trust_domain,authority_id,credential_id,semantic_identity,semantic_role,key_id,key_version,public_key,public_key_material_identity,lifecycle_generation,lifecycle_state) VALUES(p->>'environment',p->>'trust_domain',p->>'authority_id',p->>'credential_id',p->>'semantic_identity',p->>'semantic_role',p->>'key_id',(p->>'key_version')::bigint,material,material_id,1,'ACTIVE');
 INSERT INTO {q}.key_lifecycle_history(environment,trust_domain,authority_id,credential_id,lifecycle_generation,state,record) VALUES(p->>'environment',p->>'trust_domain',p->>'authority_id',p->>'credential_id',1,'ACTIVE',p);
END""",
      "transition_credential(text,text,text,text,bigint,text,jsonb)": f"""
DECLARE c {q}.credentials%ROWTYPE; latest bigint;
BEGIN
 {guard % "'freshness_admin'"}
 SELECT * INTO c FROM {q}.credentials WHERE environment=$1 AND trust_domain=$2 AND authority_id=$3 AND credential_id=$4 FOR UPDATE;
 SELECT lifecycle_generation INTO latest FROM {q}.key_lifecycle_history WHERE environment=$1 AND trust_domain=$2 AND authority_id=$3 AND credential_id=$4 ORDER BY lifecycle_generation DESC LIMIT 1 FOR UPDATE;
 IF NOT FOUND OR c.lifecycle_generation<>$5 OR latest<>$5 THEN RAISE EXCEPTION 'stale lifecycle' USING ERRCODE='40001'; END IF;
 IF NOT ((c.lifecycle_state='ACTIVE' AND $6 IN ('VERIFY_ONLY','REVOKED')) OR (c.lifecycle_state='VERIFY_ONLY' AND $6='REVOKED')) THEN RAISE EXCEPTION 'forbidden lifecycle transition' USING ERRCODE='22023'; END IF;
 INSERT INTO {q}.key_lifecycle_history VALUES($1,$2,$3,$4,$5+1,$6,$7); UPDATE {q}.credentials SET lifecycle_generation=$5+1,lifecycle_state=$6 WHERE environment=$1 AND trust_domain=$2 AND authority_id=$3 AND credential_id=$4;
END""",
      "resolve_verification_credential(text,text,text,text,text,bigint)": f"""
BEGIN
 {guard % "'freshness_crypto_verifier'"}
 IF $1<>'PRODUCTION' OR $2='' OR $3='' OR $4 NOT IN ('{PROPOSER_ROLE}','{FINALIZATION_ROLE}') OR $5='' OR $6<1 THEN RAISE EXCEPTION 'invalid credential selector' USING ERRCODE='22023'; END IF;
 RETURN QUERY SELECT c.credential_id,c.semantic_identity,c.lifecycle_generation,c.public_key
 FROM {q}.credentials c WHERE c.environment=$1 AND c.trust_domain=$2 AND c.authority_id=$3
 AND c.semantic_role=$4 AND c.key_id=$5 AND c.key_version=$6;
 IF NOT FOUND THEN RAISE EXCEPTION 'unknown retained credential' USING ERRCODE='28000'; END IF;
END""",
      "resolve_predecessor_head(text,text,text,bigint,text)": f"""
DECLARE result text;
BEGIN
 {guard % "'freshness_crypto_verifier'"}
 IF $1<>'PRODUCTION' OR $2='' OR $3='' OR $4<0 OR $5!~'^[0-9a-f]{{64}}$' THEN RAISE EXCEPTION 'invalid predecessor selector' USING ERRCODE='22023'; END IF;
 SELECT complete_head_digest INTO result FROM {q}.authority_generation_heads
 WHERE environment=$1 AND trust_domain=$2 AND authority_id=$3
 AND generation=$4 AND document_digest=$5;
 IF NOT FOUND THEN RAISE EXCEPTION 'unknown exact predecessor' USING ERRCODE='28000'; END IF;
 RETURN result;
END""",
      "prepare_verified_freshness_candidate(bytea,bytea,bytea)": f"""
DECLARE p jsonb; d jsonb; r jsonb; pc {q}.credentials%ROWTYPE; fc {q}.credentials%ROWTYPE; existing {q}.prepared_verifications%ROWTYPE; payload jsonb; doc_digest text; head_digest text; receipt_digest text; sig bytea; doc_sig bytea; canonical_sig text; canonical_doc_sig text; selector_credential text;
BEGIN
 {guard % "'freshness_crypto_verifier'"}
 BEGIN p:=pg_catalog.convert_from($1,'UTF8')::jsonb; d:=pg_catalog.convert_from($2,'UTF8')::jsonb; r:=pg_catalog.convert_from($3,'UTF8')::jsonb; EXCEPTION WHEN others THEN RAISE EXCEPTION 'invalid canonical JSON bytes' USING ERRCODE='22023'; END;
 IF $1 IS DISTINCT FROM {q}.canonical_jsonb(p) OR $2 IS DISTINCT FROM {q}.canonical_jsonb(d) OR $3 IS DISTINCT FROM {q}.canonical_jsonb(r) THEN RAISE EXCEPTION 'noncanonical JSON representation' USING ERRCODE='22023'; END IF;
 IF pg_catalog.jsonb_typeof(p)<>'object' OR (SELECT count(*) FROM pg_catalog.jsonb_object_keys(p))<>{len(PREPARATION_FIELDS)} OR NOT p ?& ARRAY[{prep_keys}] THEN RAISE EXCEPTION 'preparation schema' USING ERRCODE='22023'; END IF;
 IF pg_catalog.jsonb_typeof(d)<>'object' OR (SELECT count(*) FROM pg_catalog.jsonb_object_keys(d))<>{len(DOCUMENT_FIELDS)} OR NOT d ?& ARRAY[{document_keys}] OR pg_catalog.jsonb_typeof(d->'payload')<>'object' THEN RAISE EXCEPTION 'document schema' USING ERRCODE='22023'; END IF;
 payload:=d->'payload'; IF (SELECT count(*) FROM pg_catalog.jsonb_object_keys(payload))<>{len(DOCUMENT_PAYLOAD_FIELDS)} OR NOT payload ?& ARRAY[{payload_keys}] THEN RAISE EXCEPTION 'document payload schema' USING ERRCODE='22023'; END IF;
 IF pg_catalog.jsonb_typeof(r)<>'object' OR (SELECT count(*) FROM pg_catalog.jsonb_object_keys(r))<>{len(RECEIPT_FIELDS)} OR NOT r ?& ARRAY[{receipt_keys}] THEN RAISE EXCEPTION 'receipt schema' USING ERRCODE='22023'; END IF;
 IF {integer_matrix('p', prep_integers)} OR {string_matrix('p', prep_strings)} OR {prep_digest_check} THEN RAISE EXCEPTION 'preparation type or lexical contract' USING ERRCODE='22023'; END IF;
 IF {integer_matrix('payload', payload_integers)} OR {string_matrix('payload', payload_strings)} OR pg_catalog.jsonb_typeof(payload->'complete_semantic_head_set') IS DISTINCT FROM 'array' OR payload->>'predecessor_document_digest' !~ '^[0-9a-f]{{64}}$' THEN RAISE EXCEPTION 'document payload type or lexical contract' USING ERRCODE='22023'; END IF;
 IF pg_catalog.jsonb_typeof(d->'document_digest') IS DISTINCT FROM 'string' OR pg_catalog.jsonb_typeof(d->'authentication_tag_or_signature') IS DISTINCT FROM 'string' OR d->>'document_digest' !~ '^[0-9a-f]{{64}}$' THEN RAISE EXCEPTION 'document envelope type or lexical contract' USING ERRCODE='22023'; END IF;
 IF {integer_matrix('r', receipt_integers)} OR {string_matrix('r', receipt_strings)} OR r->>'exact_predecessor_document_digest' !~ '^[0-9a-f]{{64}}$' OR r->>'accepted_document_digest' !~ '^[0-9a-f]{{64}}$' OR r->>'complete_semantic_head_digest' !~ '^[0-9a-f]{{64}}$' THEN RAISE EXCEPTION 'receipt type or lexical contract' USING ERRCODE='22023'; END IF;
 IF p->>'schema_version' IS DISTINCT FROM '1' OR p->>'security_profile' IS DISTINCT FROM 'PRODUCTION_LOCAL' OR p->>'environment' IS DISTINCT FROM 'PRODUCTION' OR p->>'operation_type' IS DISTINCT FROM 'FULL_AUTHORITATIVE_DOCUMENT' THEN RAISE EXCEPTION 'preparation constants' USING ERRCODE='22023'; END IF;
 PERFORM {q}.canonical_complete_head_set(payload->'complete_semantic_head_set');
 doc_digest:=pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to('cryptohunter.account-genesis.freshness-document-digest.v1','UTF8')||pg_catalog.decode('00','hex')||{q}.canonical_jsonb(payload)),'hex');
 head_digest:=pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to('cryptohunter.account-genesis.complete-semantic-head-set-digest.v1','UTF8')||pg_catalog.decode('00','hex')||{q}.canonical_complete_head_set(payload->'complete_semantic_head_set')),'hex');
 receipt_digest:=pg_catalog.encode(pg_catalog.sha256({q}.canonical_jsonb(r)),'hex');
 BEGIN sig:=pg_catalog.decode(pg_catalog.translate(r->>'authentication_tag_or_signature','-_','+/')||pg_catalog.repeat('=',(4-pg_catalog.length(r->>'authentication_tag_or_signature')%4)%4),'base64'); EXCEPTION WHEN others THEN RAISE EXCEPTION 'invalid signature encoding' USING ERRCODE='22023'; END;
 BEGIN doc_sig:=pg_catalog.decode(pg_catalog.translate(d->>'authentication_tag_or_signature','-_','+/')||pg_catalog.repeat('=',(4-pg_catalog.length(d->>'authentication_tag_or_signature')%4)%4),'base64'); EXCEPTION WHEN others THEN RAISE EXCEPTION 'invalid document signature encoding' USING ERRCODE='22023'; END;
 canonical_sig:=pg_catalog.rtrim(pg_catalog.translate(pg_catalog.replace(pg_catalog.encode(sig,'base64'),pg_catalog.chr(10),''),'+/','-_'),'=');
 canonical_doc_sig:=pg_catalog.rtrim(pg_catalog.translate(pg_catalog.replace(pg_catalog.encode(doc_sig,'base64'),pg_catalog.chr(10),''),'+/','-_'),'=');
 IF r->>'authentication_tag_or_signature'!~'^[A-Za-z0-9_-]{{86}}$' OR d->>'authentication_tag_or_signature'!~'^[A-Za-z0-9_-]{{86}}$' OR pg_catalog.length(sig)<>64 OR pg_catalog.length(doc_sig)<>64 OR canonical_sig<>r->>'authentication_tag_or_signature' OR canonical_doc_sig<>d->>'authentication_tag_or_signature' THEN RAISE EXCEPTION 'noncanonical signature' USING ERRCODE='22023'; END IF;
 IF doc_digest IS DISTINCT FROM d->>'document_digest' OR doc_digest IS DISTINCT FROM p->>'proposed_document_digest' OR receipt_digest IS DISTINCT FROM p->>'receipt_canonical_digest' THEN RAISE EXCEPTION 'digest mismatch' USING ERRCODE='22023'; END IF;
 IF payload->>'schema_version' IS DISTINCT FROM '1' OR payload->>'environment' IS DISTINCT FROM p->>'environment' OR payload->>'trust_domain' IS DISTINCT FROM p->>'trust_domain' OR payload->>'authority_id' IS DISTINCT FROM p->>'authority_id' OR payload->>'generation' IS DISTINCT FROM ((p->>'expected_predecessor_generation')::bigint+1)::text OR payload->>'predecessor_generation' IS DISTINCT FROM p->>'expected_predecessor_generation' OR payload->>'predecessor_document_digest' IS DISTINCT FROM p->>'expected_predecessor_document_digest' OR payload->>'finalization_request_id' IS DISTINCT FROM p->>'finalization_request_id' THEN RAISE EXCEPTION 'document binding mismatch' USING ERRCODE='22023'; END IF;
 IF r->>'schema_version' IS DISTINCT FROM '1' OR r->>'environment' IS DISTINCT FROM p->>'environment' OR r->>'trust_domain' IS DISTINCT FROM p->>'trust_domain' OR r->>'authority_id' IS DISTINCT FROM p->>'authority_id' OR r->>'exact_predecessor_generation' IS DISTINCT FROM p->>'expected_predecessor_generation' OR r->>'exact_predecessor_document_digest' IS DISTINCT FROM p->>'expected_predecessor_document_digest' OR r->>'accepted_generation' IS DISTINCT FROM payload->>'generation' OR r->>'accepted_document_digest' IS DISTINCT FROM doc_digest OR r->>'complete_semantic_head_digest' IS DISTINCT FROM head_digest OR r->>'finalization_request_id' IS DISTINCT FROM p->>'finalization_request_id' OR r->>'receipt_id' IS DISTINCT FROM p->>'receipt_id' OR r->>'freshness_authority_key_version' IS DISTINCT FROM p->>'finalization_key_version' OR payload->>'freshness_authority_key_id' IS DISTINCT FROM r->>'freshness_authority_key_id' OR payload->>'freshness_authority_key_version' IS DISTINCT FROM r->>'freshness_authority_key_version' THEN RAISE EXCEPTION 'receipt binding mismatch' USING ERRCODE='22023'; END IF;
 SELECT * INTO pc FROM {q}.credentials WHERE environment=p->>'environment' AND trust_domain=p->>'trust_domain' AND authority_id=p->>'authority_id' AND credential_id=p->>'proposer_credential_role_identity'; SELECT * INTO fc FROM {q}.credentials WHERE environment=p->>'environment' AND trust_domain=p->>'trust_domain' AND authority_id=p->>'authority_id' AND credential_id=p->>'finalization_credential_role_identity';
 IF pc.semantic_role<>'{PROPOSER_ROLE}' OR pc.semantic_identity<>p->>'proposer_identity' OR pc.key_version<>(p->>'proposer_key_version')::bigint OR pc.public_key_material_identity<>p->>'proposer_public_key_material_identity' OR fc.semantic_role<>'{FINALIZATION_ROLE}' OR fc.key_version<>(p->>'finalization_key_version')::bigint OR fc.key_id<>r->>'freshness_authority_key_id' OR fc.public_key_material_identity<>p->>'finalization_public_key_material_identity' THEN RAISE EXCEPTION 'invalid authentication identity' USING ERRCODE='28000'; END IF;
 SELECT credential_id INTO selector_credential FROM {q}.credentials WHERE environment=p->>'environment' AND trust_domain=p->>'trust_domain' AND authority_id=p->>'authority_id' AND semantic_role='{FINALIZATION_ROLE}' AND key_id=r->>'freshness_authority_key_id' AND key_version=(r->>'freshness_authority_key_version')::bigint;
 IF selector_credential IS DISTINCT FROM fc.credential_id THEN RAISE EXCEPTION 'ambiguous finalization selector' USING ERRCODE='28000'; END IF;
 BEGIN
   INSERT INTO {q}.prepared_verifications(preparation_id,binding,canonical_document,canonical_document_bytes,canonical_receipt,canonical_receipt_bytes,proposer_credential_id,proposer_lifecycle_generation,finalization_credential_id,finalization_lifecycle_generation) VALUES(p->>'preparation_id',p,d,{q}.canonical_jsonb(d),r,{q}.canonical_jsonb(r),pc.credential_id,(p->>'proposer_lifecycle_generation_observed_for_key_binding_only')::bigint,fc.credential_id,(p->>'finalization_lifecycle_generation_observed_for_key_binding_only')::bigint);
 EXCEPTION WHEN unique_violation THEN
   SELECT * INTO existing FROM {q}.prepared_verifications WHERE preparation_id=p->>'preparation_id' FOR UPDATE;
   IF NOT FOUND OR existing.binding IS DISTINCT FROM p OR existing.canonical_document IS DISTINCT FROM d OR existing.canonical_document_bytes IS DISTINCT FROM {q}.canonical_jsonb(d) OR existing.canonical_receipt IS DISTINCT FROM r OR existing.canonical_receipt_bytes IS DISTINCT FROM {q}.canonical_jsonb(r) OR existing.proposer_credential_id IS DISTINCT FROM pc.credential_id OR existing.proposer_lifecycle_generation IS DISTINCT FROM (p->>'proposer_lifecycle_generation_observed_for_key_binding_only')::bigint OR existing.finalization_credential_id IS DISTINCT FROM fc.credential_id OR existing.finalization_lifecycle_generation IS DISTINCT FROM (p->>'finalization_lifecycle_generation_observed_for_key_binding_only')::bigint THEN RAISE EXCEPTION 'preparation identity conflict' USING ERRCODE='23505'; END IF;
 END;
 RETURN p->>'preparation_id';
END""",
      "compare_and_advance(text,jsonb,bytea,bytea)": f"""
DECLARE d jsonb; r jsonb; prep {q}.prepared_verifications%ROWTYPE; line {q}.authority_lineages%ROWTYPE; pc {q}.credentials%ROWTYPE; fc {q}.credentials%ROWTYPE; ph {q}.key_lifecycle_history%ROWTYPE; fh {q}.key_lifecycle_history%ROWTYPE; existing {q}.decisions%ROWTYPE; retained_doc {q}.authoritative_documents%ROWTYPE; retained_receipt {q}.finalization_receipts%ROWTYPE; seq bigint; accepted_head_digest text;
BEGIN
 {guard % "'freshness_runtime'"}
 BEGIN d:=pg_catalog.convert_from($3,'UTF8')::jsonb; r:=pg_catalog.convert_from($4,'UTF8')::jsonb; EXCEPTION WHEN others THEN RAISE EXCEPTION 'invalid canonical JSON bytes' USING ERRCODE='22023'; END;
 IF $3<>{q}.canonical_jsonb(d) OR $4<>{q}.canonical_jsonb(r) THEN RAISE EXCEPTION 'noncanonical JSON representation' USING ERRCODE='22023'; END IF;
 IF pg_catalog.current_setting('transaction_isolation')<>'serializable' THEN RAISE EXCEPTION 'SERIALIZABLE required' USING ERRCODE='25001'; END IF;
 SELECT * INTO existing FROM {q}.decisions d0 WHERE d0.original_decision_identity=$2->>'original_decision_identity' AND d0.finalization_request_id=$2->>'finalization_request_id'; IF FOUND THEN
   SELECT * INTO retained_doc FROM {q}.authoritative_documents ad WHERE ad.environment=existing.environment AND ad.trust_domain=existing.trust_domain AND ad.authority_id=existing.authority_id AND ad.generation=existing.accepted_generation;
   SELECT * INTO retained_receipt FROM {q}.finalization_receipts fr WHERE fr.decision_sequence=existing.decision_sequence;
   IF existing.candidate_binding=$2 AND retained_doc.canonical_document IS NOT DISTINCT FROM d AND retained_doc.canonical_document_bytes IS NOT DISTINCT FROM {q}.canonical_jsonb(d) AND retained_receipt.canonical_receipt IS NOT DISTINCT FROM r AND retained_receipt.canonical_receipt_bytes IS NOT DISTINCT FROM {q}.canonical_jsonb(r) THEN RETURN QUERY SELECT 'ALREADY_ACCEPTED_EXACT'::text,existing.decision_sequence,retained_receipt.canonical_receipt; RETURN; ELSE RAISE EXCEPTION 'replay conflict' USING ERRCODE='23505'; END IF;
 END IF;
 SELECT * INTO prep FROM {q}.prepared_verifications WHERE preparation_id=$1 FOR UPDATE; IF NOT FOUND OR prep.consumed_at IS NOT NULL OR prep.binding IS DISTINCT FROM $2 OR prep.canonical_document IS DISTINCT FROM d OR prep.canonical_receipt IS DISTINCT FROM r OR prep.canonical_document_bytes IS DISTINCT FROM {q}.canonical_jsonb(d) OR prep.canonical_receipt_bytes IS DISTINCT FROM {q}.canonical_jsonb(r) THEN RAISE EXCEPTION 'substitution or consumed preparation' USING ERRCODE='28000'; END IF;
 SELECT * INTO line FROM {q}.authority_lineages WHERE environment=$2->>'environment' AND trust_domain=$2->>'trust_domain' AND authority_id=$2->>'authority_id' FOR UPDATE; IF NOT FOUND OR line.current_generation<>(($2->>'expected_predecessor_generation')::bigint) OR line.current_document_digest<>$2->>'expected_predecessor_document_digest' OR line.current_complete_head_digest<>$2->>'expected_predecessor_complete_semantic_head_digest' THEN RAISE EXCEPTION 'predecessor mismatch' USING ERRCODE='40001'; END IF;
 SELECT * INTO pc FROM {q}.credentials WHERE environment=line.environment AND trust_domain=line.trust_domain AND authority_id=line.authority_id AND credential_id=prep.proposer_credential_id FOR UPDATE; SELECT * INTO fc FROM {q}.credentials WHERE environment=line.environment AND trust_domain=line.trust_domain AND authority_id=line.authority_id AND credential_id=prep.finalization_credential_id FOR UPDATE;
 SELECT * INTO ph FROM {q}.key_lifecycle_history WHERE environment=pc.environment AND trust_domain=pc.trust_domain AND authority_id=pc.authority_id AND credential_id=pc.credential_id ORDER BY lifecycle_generation DESC LIMIT 1 FOR UPDATE; SELECT * INTO fh FROM {q}.key_lifecycle_history WHERE environment=fc.environment AND trust_domain=fc.trust_domain AND authority_id=fc.authority_id AND credential_id=fc.credential_id ORDER BY lifecycle_generation DESC LIMIT 1 FOR UPDATE;
 IF pc.lifecycle_state<>'ACTIVE' OR fc.lifecycle_state<>'ACTIVE' OR pc.lifecycle_generation<>ph.lifecycle_generation OR pc.lifecycle_state<>ph.state OR fc.lifecycle_generation<>fh.lifecycle_generation OR fc.lifecycle_state<>fh.state OR ph.lifecycle_generation<>prep.proposer_lifecycle_generation OR fh.lifecycle_generation<>prep.finalization_lifecycle_generation THEN RAISE EXCEPTION 'lifecycle divergence or inactive' USING ERRCODE='28000'; END IF;
 IF pc.semantic_role<>'{PROPOSER_ROLE}' OR fc.semantic_role<>'{FINALIZATION_ROLE}' OR pc.public_key=fc.public_key THEN RAISE EXCEPTION 'credential role separation failure' USING ERRCODE='28000'; END IF;
 accepted_head_digest:=pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to('cryptohunter.account-genesis.complete-semantic-head-set-digest.v1','UTF8')||pg_catalog.decode('00','hex')||{q}.canonical_complete_head_set(d->'payload'->'complete_semantic_head_set')),'hex');
 IF accepted_head_digest<>r->>'complete_semantic_head_digest' THEN RAISE EXCEPTION 'accepted complete head mismatch' USING ERRCODE='22023'; END IF;
 INSERT INTO {q}.authoritative_documents(environment,trust_domain,authority_id,generation,predecessor_generation,document_digest,complete_head_digest,canonical_document,canonical_document_bytes) VALUES(line.environment,line.trust_domain,line.authority_id,line.current_generation+1,line.current_generation,$2->>'proposed_document_digest',accepted_head_digest,d,prep.canonical_document_bytes);
 INSERT INTO {q}.authority_generation_heads(environment,trust_domain,authority_id,generation,document_digest,complete_head_digest) VALUES(line.environment,line.trust_domain,line.authority_id,line.current_generation+1,$2->>'proposed_document_digest',accepted_head_digest);
 INSERT INTO {q}.decisions(original_decision_identity,finalization_request_id,environment,trust_domain,authority_id,predecessor_generation,accepted_generation,candidate_binding,lifecycle_reference) VALUES($2->>'original_decision_identity',$2->>'finalization_request_id',line.environment,line.trust_domain,line.authority_id,line.current_generation,line.current_generation+1,$2,pg_catalog.jsonb_build_object('proposer',ph.lifecycle_generation,'finalization',fh.lifecycle_generation)) RETURNING decisions.decision_sequence INTO seq;
 INSERT INTO {q}.finalization_receipts(receipt_id,decision_sequence,canonical_receipt,canonical_receipt_bytes,authentication_signature) VALUES(r->>'receipt_id',seq,r,prep.canonical_receipt_bytes,r->>'authentication_tag_or_signature'); UPDATE {q}.authority_lineages SET current_generation=current_generation+1,current_document_digest=$2->>'proposed_document_digest',current_complete_head_digest=accepted_head_digest,cas_revision=cas_revision+1 WHERE environment=line.environment AND trust_domain=line.trust_domain AND authority_id=line.authority_id; UPDATE {q}.prepared_verifications SET consumed_at=pg_catalog.clock_timestamp(),decision_sequence=seq WHERE preparation_id=$1;
 RETURN QUERY SELECT 'CAS_ACCEPTED'::text,seq,r;
END""",
    }

def provision_postgresql_freshness_authority(connection: PostgreSQLConnectionConfig,
                                              config: PostgreSQLFreshnessAuthorityProvisioning = PostgreSQLFreshnessAuthorityProvisioning()) -> None:
    """Provision the authority.  The connection must be an offline superuser."""
    ids = {x: sql.Identifier(x) for x in (config.schema, config.schema_owner_role,
            config.function_owner_role, config.admin_role, config.verifier_role,
            config.runtime_role, config.reader_role)}
    s, so, fo, adm, ver, run, read = (ids[x] for x in (config.schema,
        config.schema_owner_role, config.function_owner_role, config.admin_role,
        config.verifier_role, config.runtime_role, config.reader_role))
    with psycopg.connect(connection.dsn, autocommit=True) as conn:
      if conn.info.server_version < 160000: raise RuntimeError("PostgreSQL >=16 required")
      if conn.execute("SELECT current_setting('fsync'),current_setting('synchronous_commit')").fetchone() != ("on", "on"): raise RuntimeError("fsync and synchronous_commit must be on")
      for name, login in ((config.schema_owner_role,False),(config.function_owner_role,False),(config.admin_role,False),(config.verifier_role,True),(config.runtime_role,True),(config.reader_role,False)):
        conn.execute(sql.SQL("CREATE ROLE {} {} NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS").format(sql.Identifier(name),sql.SQL("LOGIN" if login else "NOLOGIN")))
      conn.execute(sql.SQL("CREATE SCHEMA {} AUTHORIZATION {}; REVOKE ALL ON SCHEMA {} FROM PUBLIC").format(s,so,s))
      ddl = sql.SQL("""
CREATE TABLE {s}.metadata(singleton boolean PRIMARY KEY DEFAULT true CHECK(singleton),schema_identity text NOT NULL,schema_version integer NOT NULL,function_manifest jsonb NOT NULL);
CREATE TABLE {s}.key_material_role_bindings(environment text NOT NULL,trust_domain text NOT NULL,authority_id text NOT NULL,public_key_material_identity text NOT NULL CHECK(public_key_material_identity~'^[0-9a-f]{{64}}$'),semantic_role text NOT NULL CHECK(semantic_role IN ({pr},{fr})),public_key bytea NOT NULL CHECK(octet_length(public_key)=32),PRIMARY KEY(environment,trust_domain,authority_id,public_key_material_identity),UNIQUE(environment,trust_domain,authority_id,public_key_material_identity,semantic_role));
CREATE TABLE {s}.credentials(environment text NOT NULL,trust_domain text NOT NULL,authority_id text NOT NULL,credential_id text NOT NULL,semantic_identity text NOT NULL,semantic_role text NOT NULL CHECK(semantic_role IN ({pr},{fr})),key_id text NOT NULL,key_version bigint NOT NULL CHECK(key_version>=1),public_key bytea NOT NULL CHECK(octet_length(public_key)=32),public_key_material_identity text NOT NULL CHECK(public_key_material_identity~'^[0-9a-f]{{64}}$'),lifecycle_generation bigint NOT NULL CHECK(lifecycle_generation>=1),lifecycle_state text NOT NULL CHECK(lifecycle_state IN ('ACTIVE','VERIFY_ONLY','REVOKED')),PRIMARY KEY(environment,trust_domain,authority_id,credential_id),UNIQUE(environment,trust_domain,authority_id,semantic_role,public_key_material_identity),CONSTRAINT credentials_authenticated_selector_key UNIQUE(environment,trust_domain,authority_id,semantic_role,key_id,key_version),FOREIGN KEY(environment,trust_domain,authority_id,public_key_material_identity,semantic_role) REFERENCES {s}.key_material_role_bindings(environment,trust_domain,authority_id,public_key_material_identity,semantic_role));
CREATE UNIQUE INDEX credentials_one_active_role_idx ON {s}.credentials(environment,trust_domain,authority_id,semantic_role) WHERE lifecycle_state='ACTIVE';
CREATE TABLE {s}.key_lifecycle_history(environment text NOT NULL,trust_domain text NOT NULL,authority_id text NOT NULL,credential_id text NOT NULL,lifecycle_generation bigint NOT NULL,state text NOT NULL CHECK(state IN ('ACTIVE','VERIFY_ONLY','REVOKED')),record jsonb NOT NULL,PRIMARY KEY(environment,trust_domain,authority_id,credential_id,lifecycle_generation),FOREIGN KEY(environment,trust_domain,authority_id,credential_id) REFERENCES {s}.credentials DEFERRABLE INITIALLY DEFERRED);
CREATE TABLE {s}.authority_lineages(environment text NOT NULL,trust_domain text NOT NULL,authority_id text NOT NULL,current_generation bigint NOT NULL CHECK(current_generation>=0),current_document_digest text NOT NULL,current_complete_head_digest text NOT NULL,cas_revision bigint NOT NULL DEFAULT 0,PRIMARY KEY(environment,trust_domain,authority_id));
CREATE TABLE {s}.authority_generation_heads(environment text NOT NULL,trust_domain text NOT NULL,authority_id text NOT NULL,generation bigint NOT NULL CHECK(generation>=0),document_digest text NOT NULL CHECK(document_digest~'^[0-9a-f]{{64}}$'),complete_head_digest text NOT NULL CHECK(complete_head_digest~'^[0-9a-f]{{64}}$'),PRIMARY KEY(environment,trust_domain,authority_id,generation),UNIQUE(environment,trust_domain,authority_id,document_digest));
CREATE TABLE {s}.prepared_verifications(preparation_id text PRIMARY KEY,binding jsonb NOT NULL,canonical_document jsonb NOT NULL,canonical_document_bytes bytea NOT NULL,canonical_receipt jsonb NOT NULL,canonical_receipt_bytes bytea NOT NULL,proposer_credential_id text NOT NULL,proposer_lifecycle_generation bigint NOT NULL,finalization_credential_id text NOT NULL,finalization_lifecycle_generation bigint NOT NULL,created_at timestamptz NOT NULL DEFAULT clock_timestamp(),consumed_at timestamptz,decision_sequence bigint);
CREATE TABLE {s}.authoritative_documents(environment text NOT NULL,trust_domain text NOT NULL,authority_id text NOT NULL,generation bigint NOT NULL,predecessor_generation bigint NOT NULL,document_digest text NOT NULL CHECK(document_digest~'^[0-9a-f]{{64}}$'),complete_head_digest text NOT NULL CHECK(complete_head_digest~'^[0-9a-f]{{64}}$'),canonical_document jsonb NOT NULL,canonical_document_bytes bytea NOT NULL,PRIMARY KEY(environment,trust_domain,authority_id,generation),UNIQUE(environment,trust_domain,authority_id,predecessor_generation));
CREATE TABLE {s}.decisions(decision_sequence bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,original_decision_identity text NOT NULL,finalization_request_id text NOT NULL,environment text NOT NULL,trust_domain text NOT NULL,authority_id text NOT NULL,predecessor_generation bigint NOT NULL,accepted_generation bigint NOT NULL,candidate_binding jsonb NOT NULL,lifecycle_reference jsonb NOT NULL,UNIQUE(original_decision_identity,finalization_request_id));
CREATE TABLE {s}.finalization_receipts(receipt_id text PRIMARY KEY,decision_sequence bigint NOT NULL UNIQUE REFERENCES {s}.decisions(decision_sequence),canonical_receipt jsonb NOT NULL,canonical_receipt_bytes bytea NOT NULL,authentication_signature text NOT NULL);
""").format(s=s,pr=sql.Literal(PROPOSER_ROLE),fr=sql.Literal(FINALIZATION_ROLE))
      conn.execute(ddl)
      sources = _sources(config.schema)
      signatures = {
       "jcs_sort_key(text)":"(text) RETURNS text", "canonical_jsonb(jsonb)":"(jsonb) RETURNS bytea",
       "canonical_complete_head_set(jsonb)":"(jsonb) RETURNS bytea", "provision_authority(text,text,text,text,text)":"(text,text,text,text,text) RETURNS void",
       "provision_credential(jsonb)":"(jsonb) RETURNS void", "transition_credential(text,text,text,text,bigint,text,jsonb)":"(text,text,text,text,bigint,text,jsonb) RETURNS void",
       "resolve_verification_credential(text,text,text,text,text,bigint)":"(text,text,text,text,text,bigint) RETURNS TABLE(credential_id text,semantic_identity text,lifecycle_generation bigint,public_key bytea)",
       "resolve_predecessor_head(text,text,text,bigint,text)":"(text,text,text,bigint,text) RETURNS text",
       "prepare_verified_freshness_candidate(bytea,bytea,bytea)":"(bytea,bytea,bytea) RETURNS text", "compare_and_advance(text,jsonb,bytea,bytea)":"(text,jsonb,bytea,bytea) RETURNS TABLE(outcome text,decision_sequence bigint,receipt jsonb)"}
      for sig, source in sources.items():
        name=sig.split("(",1)[0]
        conn.execute(sql.SQL("CREATE FUNCTION {}.{} {} LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS {}") .format(s,sql.Identifier(name),sql.SQL(signatures[sig]),sql.Literal(source)))
        conn.execute(sql.SQL("ALTER FUNCTION {}.{} OWNER TO {}; REVOKE ALL ON FUNCTION {}.{} FROM PUBLIC").format(s,sql.SQL(sig),fo,s,sql.SQL(sig)))
      conn.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {},{},{},{},{}; GRANT SELECT ON ALL TABLES IN SCHEMA {} TO {}; GRANT SELECT,INSERT,UPDATE,DELETE ON ALL TABLES IN SCHEMA {} TO {}; GRANT USAGE,SELECT ON ALL SEQUENCES IN SCHEMA {} TO {}; GRANT EXECUTE ON FUNCTION {}.provision_authority(text,text,text,text,text),{}.provision_credential(jsonb),{}.transition_credential(text,text,text,text,bigint,text,jsonb) TO {}; GRANT EXECUTE ON FUNCTION {}.resolve_verification_credential(text,text,text,text,text,bigint),{}.resolve_predecessor_head(text,text,text,bigint,text),{}.prepare_verified_freshness_candidate(bytea,bytea,bytea) TO {}; GRANT EXECUTE ON FUNCTION {}.compare_and_advance(text,jsonb,bytea,bytea) TO {}").format(s,ver,run,read,fo,adm,s,read,s,fo,s,fo,s,s,s,adm,s,s,s,ver,s,run))
      manifest={k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()}
      conn.execute(sql.SQL("INSERT INTO {}.metadata VALUES(true,%s,%s,%s)").format(s),(SCHEMA_IDENTITY,SCHEMA_VERSION,json.dumps(manifest)))
      for table in ("metadata", "key_material_role_bindings", "credentials", "key_lifecycle_history",
                    "authority_lineages", "authority_generation_heads", "prepared_verifications",
                    "authoritative_documents", "decisions",
                    "finalization_receipts"):
        conn.execute(sql.SQL("ALTER TABLE {}.{} OWNER TO {}").format(s, sql.Identifier(table), so))


def qualify_postgresql_freshness_authority(connection: PostgreSQLConnectionConfig,
                                            config: PostgreSQLFreshnessAuthorityProvisioning = PostgreSQLFreshnessAuthorityProvisioning()) -> None:
    """Fail closed unless catalog state and reviewed function text are exact."""
    with psycopg.connect(connection.dsn) as conn:
      problems=[]
      if conn.info.server_version < 160000: problems.append("server version")
      if conn.execute("SELECT current_setting('fsync'),current_setting('synchronous_commit')").fetchone() != ("on","on"): problems.append("durability")
      roles=(config.schema_owner_role,config.function_owner_role,config.admin_role,config.verifier_role,config.runtime_role,config.reader_role)
      rows=conn.execute("SELECT rolname,rolsuper,rolinherit,rolcreaterole,rolcreatedb,rolcanlogin,rolreplication,rolbypassrls FROM pg_catalog.pg_roles WHERE rolname=ANY(%s)",(list(roles),)).fetchall()
      if len(rows)!=6 or any(r[1] or r[2] or r[3] or r[4] or r[6] or r[7] for r in rows): problems.append("role attributes")
      login={r[0]:r[5] for r in rows}
      expected_login={config.schema_owner_role:False,config.function_owner_role:False,
                      config.admin_role:False,config.verifier_role:True,
                      config.runtime_role:True,config.reader_role:False}
      if login != expected_login: problems.append("role LOGIN")
      if conn.execute("SELECT count(*) FROM pg_catalog.pg_auth_members m JOIN pg_catalog.pg_roles a ON a.oid=m.roleid JOIN pg_catalog.pg_roles b ON b.oid=m.member WHERE a.rolname=ANY(%s) OR b.rolname=ANY(%s)",(list(roles),list(roles))).fetchone()[0]: problems.append("role membership")
      expected=_sources(config.schema)
      got=conn.execute("SELECT p.proname||'('||pg_catalog.pg_get_function_identity_arguments(p.oid)||')',p.prosrc,p.prosecdef,p.proconfig,r.rolname,coalesce(p.proacl::text,'') FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace JOIN pg_catalog.pg_roles r ON r.oid=p.proowner WHERE n.nspname=%s",(config.schema,)).fetchall()
      if len(got)!=len(expected): problems.append("function set")
      actual={r[0].replace(", ", ","):r for r in got}
      for sig,src in expected.items():
        r=actual.get(sig)
        public_execute = r is not None and (
            r[5].startswith("{=X/") or ",=X/" in r[5]
        )
        if not r or r[1]!=src or not r[2] or r[3]!=['search_path=pg_catalog'] or r[4]!=config.function_owner_role or public_execute: problems.append("function "+sig)
      function_callers={
          "jcs_sort_key(text)": set(),
          "canonical_jsonb(jsonb)": set(),
          "canonical_complete_head_set(jsonb)": set(),
          "provision_authority(text,text,text,text,text)": {config.admin_role},
          "provision_credential(jsonb)": {config.admin_role},
          "transition_credential(text,text,text,text,bigint,text,jsonb)": {config.admin_role},
          "resolve_verification_credential(text,text,text,text,text,bigint)": {config.verifier_role},
          "resolve_predecessor_head(text,text,text,bigint,text)": {config.verifier_role},
          "prepare_verified_freshness_candidate(bytea,bytea,bytea)": {config.verifier_role},
          "compare_and_advance(text,jsonb,bytea,bytea)": {config.runtime_role},
      }
      facl=conn.execute("""SELECT p.proname||'('||replace(pg_catalog.pg_get_function_identity_arguments(p.oid),', ', ',')||')',CASE WHEN x.grantee=0 THEN 'PUBLIC' ELSE g.rolname END,x.privilege_type,x.is_grantable FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace CROSS JOIN LATERAL pg_catalog.aclexplode(coalesce(p.proacl,pg_catalog.acldefault('f',p.proowner))) x LEFT JOIN pg_catalog.pg_roles g ON g.oid=x.grantee WHERE n.nspname=%s""",(config.schema,)).fetchall()
      expected_facl={(sig,config.function_owner_role,"EXECUTE",False) for sig in expected}
      expected_facl|={(sig,role,"EXECUTE",False) for sig,rs in function_callers.items() for role in rs}
      if set(facl)!=expected_facl: problems.append("function ACL")
      meta=conn.execute(sql.SQL("SELECT schema_identity,schema_version,function_manifest FROM {}.metadata").format(sql.Identifier(config.schema))).fetchone()
      immutable={k:hashlib.sha256(v.encode()).hexdigest() for k,v in expected.items()}
      if not meta or tuple(meta[:2])!=(SCHEMA_IDENTITY,SCHEMA_VERSION) or meta[2]!=immutable: problems.append("manifest")
      expected_relations={"metadata","key_material_role_bindings","credentials",
          "key_lifecycle_history","authority_lineages","authority_generation_heads","prepared_verifications",
          "authoritative_documents","decisions","finalization_receipts",
          "decisions_decision_sequence_seq"}
      rels=conn.execute("SELECT c.relname,c.relkind,c.relpersistence,r.rolname,c.relrowsecurity,c.relforcerowsecurity FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace JOIN pg_catalog.pg_roles r ON r.oid=c.relowner WHERE n.nspname=%s AND c.relkind IN ('r','S')",(config.schema,)).fetchall()
      if {r[0] for r in rels}!=expected_relations or any(r[1] not in ('r','S') or r[2]!='p' or r[3]!=config.schema_owner_role or r[4] or r[5] for r in rels): problems.append("relation set/shape")
      unexpected_objects=conn.execute("SELECT count(*) FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname=%s AND c.relkind NOT IN ('r','S','i')",(config.schema,)).fetchone()[0]
      orphan_indexes=conn.execute("SELECT count(*) FROM pg_catalog.pg_class i JOIN pg_catalog.pg_namespace n ON n.oid=i.relnamespace LEFT JOIN pg_catalog.pg_index x ON x.indexrelid=i.oid LEFT JOIN pg_catalog.pg_class t ON t.oid=x.indrelid WHERE n.nspname=%s AND i.relkind='i' AND (t.oid IS NULL OR t.relnamespace<>n.oid)",(config.schema,)).fetchone()[0]
      if unexpected_objects or orphan_indexes: problems.append("unexpected schema object")
      schema_acl=conn.execute("""SELECT CASE WHEN x.grantee=0 THEN 'PUBLIC' ELSE r.rolname END,x.privilege_type,x.is_grantable FROM pg_catalog.pg_namespace n CROSS JOIN LATERAL pg_catalog.aclexplode(coalesce(n.nspacl,pg_catalog.acldefault('n',n.nspowner))) x LEFT JOIN pg_catalog.pg_roles r ON r.oid=x.grantee WHERE n.nspname=%s""",(config.schema,)).fetchall()
      expected_schema={(config.schema_owner_role,p,False) for p in ('USAGE','CREATE')}|{(r,'USAGE',False) for r in (config.function_owner_role,config.admin_role,config.verifier_role,config.runtime_role,config.reader_role)}
      if set(schema_acl)!=expected_schema: problems.append("schema ACL")
      table_acl=conn.execute("""SELECT c.relname,CASE WHEN x.grantee=0 THEN 'PUBLIC' ELSE r.rolname END,x.privilege_type,x.is_grantable FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace CROSS JOIN LATERAL pg_catalog.aclexplode(coalesce(c.relacl,pg_catalog.acldefault(CASE c.relkind WHEN 'S' THEN 's'::"char" ELSE 'r'::"char" END,c.relowner))) x LEFT JOIN pg_catalog.pg_roles r ON r.oid=x.grantee WHERE n.nspname=%s AND c.relkind IN ('r','S')""",(config.schema,)).fetchall()
      tables=expected_relations-{"decisions_decision_sequence_seq"}
      owner_table_privs={'SELECT','INSERT','UPDATE','DELETE','TRUNCATE','REFERENCES','TRIGGER'}
      expected_tacl={(t,config.schema_owner_role,p,False) for t in tables for p in owner_table_privs}|{(t,config.function_owner_role,p,False) for t in tables for p in ('SELECT','INSERT','UPDATE','DELETE')}|{(t,config.reader_role,'SELECT',False) for t in tables}
      expected_tacl|={("decisions_decision_sequence_seq",config.schema_owner_role,p,False) for p in ('USAGE','SELECT','UPDATE')}|{("decisions_decision_sequence_seq",config.function_owner_role,p,False) for p in ('USAGE','SELECT')}
      if set(table_acl)!=expected_tacl: problems.append("relation ACL")
      rules=conn.execute("SELECT count(*) FROM pg_catalog.pg_rewrite w JOIN pg_catalog.pg_class c ON c.oid=w.ev_class JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname=%s AND w.rulename<>'_RETURN'",(config.schema,)).fetchone()[0]
      inheritance=conn.execute("SELECT count(*) FROM pg_catalog.pg_inherits i JOIN pg_catalog.pg_class c ON c.oid=i.inhrelid JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname=%s",(config.schema,)).fetchone()[0]
      if rules or inheritance: problems.append("rules/inheritance")
      aliases=conn.execute(sql.SQL("SELECT count(*) FROM {}.credentials c JOIN {}.credentials d USING(environment,trust_domain,authority_id,public_key) WHERE c.semantic_role<>d.semantic_role").format(sql.Identifier(config.schema),sql.Identifier(config.schema))).fetchone()[0]
      binding_mismatch=conn.execute(sql.SQL("SELECT count(*) FROM {}.key_material_role_bindings WHERE public_key_material_identity<>encode(sha256(public_key),'hex')").format(sql.Identifier(config.schema))).fetchone()[0]
      credential_mismatch=conn.execute(sql.SQL("""SELECT count(*) FROM {s}.credentials c LEFT JOIN {s}.key_material_role_bindings b USING(environment,trust_domain,authority_id,public_key_material_identity,semantic_role) WHERE c.public_key_material_identity<>pg_catalog.encode(pg_catalog.sha256(c.public_key),'hex') OR b.public_key IS NULL OR b.public_key<>c.public_key""").format(s=sql.Identifier(config.schema))).fetchone()[0]
      selector_duplicates=conn.execute(sql.SQL("SELECT count(*) FROM (SELECT 1 FROM {}.credentials GROUP BY environment,trust_domain,authority_id,semantic_role,key_id,key_version HAVING count(*)<>1) q").format(sql.Identifier(config.schema))).fetchone()[0]
      active_duplicates=conn.execute(sql.SQL("SELECT count(*) FROM (SELECT 1 FROM {}.credentials WHERE lifecycle_state='ACTIVE' GROUP BY environment,trust_domain,authority_id,semantic_role HAVING count(*)>1) q").format(sql.Identifier(config.schema))).fetchone()[0]
      if aliases or binding_mismatch or credential_mismatch or selector_duplicates or active_duplicates: problems.append("key material binding/cardinality")
      evidence_mismatch=conn.execute(sql.SQL("""SELECT
        (SELECT count(*) FROM {s}.prepared_verifications p WHERE p.canonical_document_bytes<>{s}.canonical_jsonb(p.canonical_document) OR p.canonical_receipt_bytes<>{s}.canonical_jsonb(p.canonical_receipt) OR p.binding->>'preparation_id'<>p.preparation_id) +
        (SELECT count(*) FROM {s}.authoritative_documents d WHERE d.canonical_document_bytes<>{s}.canonical_jsonb(d.canonical_document) OR d.document_digest<>d.canonical_document->>'document_digest') +
        (SELECT count(*) FROM {s}.authority_lineages l LEFT JOIN {s}.authority_generation_heads h ON h.environment=l.environment AND h.trust_domain=l.trust_domain AND h.authority_id=l.authority_id AND h.generation=l.current_generation WHERE h.generation IS NULL OR h.document_digest IS DISTINCT FROM l.current_document_digest OR h.complete_head_digest IS DISTINCT FROM l.current_complete_head_digest) +
        (SELECT count(*) FROM {s}.authority_generation_heads h LEFT JOIN {s}.authoritative_documents d ON d.environment=h.environment AND d.trust_domain=h.trust_domain AND d.authority_id=h.authority_id AND d.generation=h.generation WHERE h.generation>0 AND (d.generation IS NULL OR d.document_digest IS DISTINCT FROM h.document_digest OR d.complete_head_digest IS DISTINCT FROM h.complete_head_digest)) +
        (SELECT count(*) FROM {s}.finalization_receipts r WHERE r.canonical_receipt_bytes<>{s}.canonical_jsonb(r.canonical_receipt) OR r.receipt_id<>r.canonical_receipt->>'receipt_id' OR r.authentication_signature<>r.canonical_receipt->>'authentication_tag_or_signature')
      """).format(s=sql.Identifier(config.schema))).fetchone()[0]
      if evidence_mismatch: problems.append("retained evidence integrity")
      if config == PostgreSQLFreshnessAuthorityProvisioning() and _physical_fingerprint(conn, config.schema) != _REVIEWED_PHYSICAL_FINGERPRINT:
          problems.append("physical schema fingerprint")
      # No relation may be non-permanent, have RLS/policies, triggers, rules, or a wrong owner.
      bad=conn.execute("SELECT count(*) FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace JOIN pg_catalog.pg_roles r ON r.oid=c.relowner WHERE n.nspname=%s AND c.relkind IN ('r','S') AND (c.relpersistence<>'p' OR c.relrowsecurity OR c.relforcerowsecurity OR r.rolname<>%s)",(config.schema,config.schema_owner_role)).fetchone()[0]
      extras=conn.execute("SELECT (SELECT count(*) FROM pg_catalog.pg_trigger t JOIN pg_catalog.pg_class c ON c.oid=t.tgrelid JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname=%s AND NOT t.tgisinternal)+(SELECT count(*) FROM pg_catalog.pg_policy p JOIN pg_catalog.pg_class c ON c.oid=p.polrelid JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname=%s)",(config.schema,config.schema)).fetchone()[0]
      if bad or extras: problems.append("relation security")
      if problems: raise FreshnessAuthorityQualificationError("qualification failed: "+", ".join(problems))
