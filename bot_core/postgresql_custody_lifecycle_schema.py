"""Explicit V2 overlay for durable, authenticated custody retirement.

V1 remains an isolated test authority. This migration retires its arbitrary
fence administration grants, preserves its reviewed function definitions and
adds no activation or key provisioning API. Completion authentication proves
that the trusted administrator validated custody; SQL does not read a local
filesystem. The authentication key is separate from every signing credential.
"""

from __future__ import annotations

import hashlib
from typing import Any, cast

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from bot_core.authenticated_issuer_history import (
    ATTESTATION_DOMAIN,
    _credential_material,
    canonical_json_bytes as history_canonical_json_bytes,
)
from bot_core.local_signing_custody import SigningCredentialSnapshot, SigningKeyLifecycle
from bot_core.postgresql_atomic_root_proof_schema import (
    PostgreSQLAtomicRootProofProvisioning,
)
from bot_core.postgresql_root_proof_retention import _canonical, _snapshot_data
from bot_core.root_proof_issuer_substrate import (
    CredentialRoleIdentity,
    CredentialSemanticRole,
    ProviderIdentity,
    ProviderRole,
    SecurityProfile,
    SecurityProfileIdentity,
    public_key_material_identity,
)

SCHEMA_IDENTITY = "cryptohunter.custody_lifecycle.postgresql"
SCHEMA_VERSION = 2
MIGRATION_ID = "DURABLE_CUSTODY_LIFECYCLE_FENCING_V2"
BOUNDARY = "ISOLATED_TEST_ONLY"
COMPLETION_MAC_DOMAIN = b"CryptoHunter/CustodyLifecycleCompletion/V2\0"
_FENCE_DOMAIN = b"CryptoHunter/AtomicTestFence/v1\0"
REQUEST_KEYS = tuple(
    sorted(
        (
            "protocol_version",
            "operation_id",
            "fence_key",
            "semantic_role",
            "credential_identity",
            "public_key_hex",
            "key_version",
            "lifecycle_generation",
            "predecessor_state",
            "target_state",
            "custody_scope",
            "database_identity",
        )
    )
)
BINDING_KEYS = tuple(
    key
    for key in REQUEST_KEYS
    if key not in {"protocol_version", "operation_id", "predecessor_state", "target_state"}
)
FUNC_ADMIN = frozenset(
    {
        "begin_lifecycle_close(jsonb)",
        "read_lifecycle_operation(jsonb)",
        "complete_lifecycle_close(text,bytea,bytea)",
        "qualify_lifecycle_catalog()",
    }
)
FUNCTION_DECLARATIONS = {
    "lifecycle_hmac(bytea,bytea)": ("p_key bytea,p_message bytea", "bytea"),
    "assert_lifecycle_catalog()": ("", "void"),
    "qualify_lifecycle_catalog()": ("", "boolean"),
    "validate_lifecycle_request(jsonb)": ("p_request jsonb", "void"),
    "lifecycle_operation_result(text)": ("p_operation text", "jsonb"),
    "begin_lifecycle_close(jsonb)": ("p_request jsonb", "jsonb"),
    "read_lifecycle_operation(jsonb)": ("p_request jsonb", "jsonb"),
    "complete_lifecycle_close(text,bytea,bytea)": (
        "p_operation text,p_evidence bytea,p_authentication bytea",
        "jsonb",
    ),
    "guard_lifecycle_fence()": ("", "trigger"),
    "guard_lifecycle_decision_fences()": ("", "trigger"),
}


def expected_custody_fence_key(
    binding: dict[str, Any], *, predecessor_state: str = "ACTIVE"
) -> str:
    """Require the same canonical fence derivation used by the V1 issuer."""
    raw_identity = dict(binding["credential_identity"])
    raw_identity["semantic_role"] = CredentialSemanticRole(raw_identity["semantic_role"])
    identity = CredentialRoleIdentity(**raw_identity)
    public_key = bytes.fromhex(binding["public_key_hex"])
    role = ProviderRole(binding["semantic_role"])
    if role not in {ProviderRole.ROOT_PROOF_SIGNING, ProviderRole.HISTORY_ATTESTATION_SIGNING}:
        raise ValueError("root/history signing custody role required")
    expected_semantic_role = (
        CredentialSemanticRole.ROOT_PROOF_ISSUER_SIGNING
        if role is ProviderRole.ROOT_PROOF_SIGNING
        else CredentialSemanticRole.HISTORY_ATTESTATION_SIGNING
    )
    if (
        identity.semantic_role is not expected_semantic_role
        or len(public_key) != 32
        or public_key.hex() != binding["public_key_hex"]
        or identity.key_material_identity != public_key_material_identity(public_key)
    ):
        raise ValueError("custody semantic role and public identity differ")
    if role is ProviderRole.HISTORY_ATTESTATION_SIGNING:
        material = history_canonical_json_bytes(_credential_material(identity))
    else:
        scope = binding["custody_scope"]
        security = SecurityProfileIdentity(
            SecurityProfile(scope["security_profile"]), scope["trust_domain"]
        )
        snapshot = SigningCredentialSnapshot(
            ProviderIdentity(role, security, identity.provider_namespace),
            identity,
            binding["key_version"],
            binding["lifecycle_generation"],
            SigningKeyLifecycle(predecessor_state),
            public_key,
            public_key_material_identity(public_key),
        )
        material = _canonical(_snapshot_data(snapshot))
    return cast(str, role.value) + ":" + hashlib.sha256(_FENCE_DOMAIN + material).hexdigest()


def _q(value: str) -> str:
    return '"' + value + '"'


def _guard(config: PostgreSQLAtomicRootProofProvisioning) -> str:
    s = _q(config.schema)
    return f"""
 IF session_user::text IS DISTINCT FROM '{config.test_admin_role}'
 OR current_user::text IS DISTINCT FROM '{config.schema_owner_role}'
 OR (SELECT oid FROM pg_catalog.pg_roles WHERE rolname=session_user::text)
 IS DISTINCT FROM (SELECT admin_role_oid FROM {s}.lifecycle_metadata WHERE singleton)
 OR (SELECT oid FROM pg_catalog.pg_roles WHERE rolname=current_user::text)
 IS DISTINCT FROM (SELECT owner_role_oid FROM {s}.lifecycle_metadata WHERE singleton)
 THEN RAISE EXCEPTION 'exact lifecycle administrator required' USING ERRCODE='42501'; END IF;
 IF pg_catalog.current_setting('server_version_num')::integer NOT BETWEEN 160000 AND 169999
 OR pg_catalog.current_setting('fsync')<>'on'
 OR pg_catalog.current_setting('synchronous_commit')<>'on'
 OR pg_catalog.current_setting('full_page_writes')<>'on' OR pg_catalog.pg_is_in_recovery()
 THEN RAISE EXCEPTION 'durable lifecycle authority required' USING ERRCODE='42501'; END IF;
 PERFORM {s}.assert_lifecycle_catalog();
"""


def reviewed_lifecycle_function_sources(
    config: PostgreSQLAtomicRootProofProvisioning,
) -> dict[str, str]:
    """Static reviewed SQL; identifiers come only from validated provisioning."""
    s = _q(config.schema)
    guard = _guard(config)
    keys = "ARRAY[" + ",".join("'" + key + "'" for key in REQUEST_KEYS) + "]::text[]"
    binding_keys = "ARRAY[" + ",".join("'" + key + "'" for key in BINDING_KEYS) + "]::text[]"
    return {
        "lifecycle_hmac(bytea,bytea)": """
DECLARE ipad bytea; opad bytea; normalized bytea; k integer;
BEGIN
 IF p_key IS NULL OR p_message IS NULL THEN
  RAISE EXCEPTION 'authentication material missing' USING ERRCODE='22023'; END IF;
 normalized:=CASE WHEN pg_catalog.octet_length(p_key)>64
  THEN pg_catalog.sha256(p_key) ELSE p_key END;
 normalized:=normalized || pg_catalog.decode(pg_catalog.repeat('00',64-pg_catalog.octet_length(normalized)),'hex');
 ipad:=normalized; opad:=normalized;
 FOR k IN 0..63 LOOP
  ipad:=pg_catalog.set_byte(ipad,k,pg_catalog.get_byte(normalized,k) # 54);
  opad:=pg_catalog.set_byte(opad,k,pg_catalog.get_byte(normalized,k) # 92);
 END LOOP;
 RETURN pg_catalog.sha256(opad || pg_catalog.sha256(ipad || p_message));
END
""",
        "assert_lifecycle_catalog()": f"""
BEGIN
 PERFORM {s}.assert_catalog();
 IF (SELECT schema_identity FROM {s}.lifecycle_metadata WHERE singleton)
 IS DISTINCT FROM '{SCHEMA_IDENTITY}'
 OR (SELECT schema_version FROM {s}.lifecycle_metadata WHERE singleton) IS DISTINCT FROM 2
 OR (SELECT migration_id FROM {s}.lifecycle_metadata WHERE singleton) IS DISTINCT FROM '{MIGRATION_ID}'
 OR (SELECT boundary FROM {s}.lifecycle_metadata WHERE singleton) IS DISTINCT FROM '{BOUNDARY}'
 OR (SELECT admin_role_oid FROM {s}.lifecycle_metadata WHERE singleton)
 IS DISTINCT FROM (SELECT test_admin_role_oid FROM {s}.metadata WHERE singleton)
 OR (SELECT owner_role_oid FROM {s}.lifecycle_metadata WHERE singleton)
 IS DISTINCT FROM (SELECT schema_owner_role_oid FROM {s}.metadata WHERE singleton)
 OR (SELECT authentication_key_sha256 FROM {s}.lifecycle_metadata WHERE singleton)
 IS DISTINCT FROM (SELECT pg_catalog.sha256(authentication_key) FROM {s}.lifecycle_authority_secret WHERE singleton)
 THEN RAISE EXCEPTION 'lifecycle V2 identity or credential drift' USING ERRCODE='42501'; END IF;
END
""",
        "qualify_lifecycle_catalog()": f"BEGIN {guard} RETURN true; END",
        "validate_lifecycle_request(jsonb)": f"""
DECLARE binding jsonb; stored jsonb;
BEGIN
 IF p_request IS NULL OR pg_catalog.jsonb_typeof(p_request)<>'object'
 OR (SELECT pg_catalog.array_agg(k ORDER BY k) FROM pg_catalog.jsonb_object_keys(p_request) k)
 IS DISTINCT FROM {keys}
 OR p_request->'protocol_version' IS DISTINCT FROM '2'::jsonb
 OR pg_catalog.jsonb_typeof(p_request->'operation_id') IS DISTINCT FROM 'string'
 OR pg_catalog.length(p_request->>'operation_id') NOT BETWEEN 1 AND 128
 OR p_request->>'operation_id' !~ '^[A-Za-z0-9][A-Za-z0-9_.:-]*$'
 OR pg_catalog.jsonb_typeof(p_request->'lifecycle_generation') IS DISTINCT FROM 'number'
 OR (p_request->>'lifecycle_generation') !~ '^[1-9][0-9]*$'
 OR pg_catalog.jsonb_typeof(p_request->'predecessor_state') IS DISTINCT FROM 'string'
 OR pg_catalog.jsonb_typeof(p_request->'target_state') IS DISTINCT FROM 'string'
 OR p_request->>'predecessor_state' NOT IN ('ACTIVE','VERIFY_ONLY')
 OR p_request->>'target_state' NOT IN ('VERIFY_ONLY','REVOKED')
 OR (p_request->>'predecessor_state'='VERIFY_ONLY' AND p_request->>'target_state'<>'REVOKED')
 THEN RAISE EXCEPTION 'exact one-way lifecycle request required' USING ERRCODE='22023'; END IF;
 SELECT credential_binding INTO stored FROM {s}.lifecycle_credentials
 WHERE fence_key=p_request->>'fence_key';
 SELECT pg_catalog.jsonb_object_agg(k,p_request->k) INTO binding
 FROM pg_catalog.unnest({binding_keys}) k;
 IF stored IS NULL OR binding-'lifecycle_generation' IS DISTINCT FROM stored-'lifecycle_generation'
 OR (p_request->>'lifecycle_generation')::bigint < (stored->>'lifecycle_generation')::bigint
 THEN RAISE EXCEPTION 'exact enrolled custody identity required' USING ERRCODE='23514'; END IF;
END
""",
        "lifecycle_operation_result(text)": f"""
DECLARE op jsonb; evidence bytea;
BEGIN
 SELECT request INTO op FROM {s}.lifecycle_operations WHERE operation_id=p_operation;
 IF NOT FOUND THEN RETURN NULL; END IF;
 SELECT canonical_evidence INTO evidence FROM {s}.lifecycle_completions WHERE operation_id=p_operation;
 RETURN pg_catalog.jsonb_build_object('operation',op,
  'phase',CASE WHEN evidence IS NULL THEN 'PENDING_CLOSED' ELSE 'COMPLETED' END,
  'completion_evidence_hex',CASE WHEN evidence IS NULL THEN NULL ELSE pg_catalog.encode(evidence,'hex') END,
  'terminal_state',CASE WHEN evidence IS NULL THEN NULL ELSE op->>'target_state' END);
END
""",
        "begin_lifecycle_close(jsonb)": f"""
DECLARE fence {s}.fences%ROWTYPE; prior jsonb;
BEGIN
 {guard}
 PERFORM {s}.validate_lifecycle_request(p_request);
 SELECT * INTO fence FROM {s}.fences WHERE identity=p_request->>'fence_key' FOR UPDATE;
 IF NOT FOUND THEN RAISE EXCEPTION 'lifecycle fence missing' USING ERRCODE='23514'; END IF;
 SELECT request INTO prior FROM {s}.lifecycle_operations WHERE operation_id=p_request->>'operation_id';
 IF FOUND THEN
  IF prior IS DISTINCT FROM p_request THEN
   RAISE EXCEPTION 'administrative operation identity conflict' USING ERRCODE='23505'; END IF;
  RETURN {s}.lifecycle_operation_result(p_request->>'operation_id');
 END IF;
 IF fence.semantic_role IS DISTINCT FROM p_request->>'semantic_role'
 OR fence.generation IS DISTINCT FROM (p_request->>'lifecycle_generation')::bigint
 OR fence.state IS DISTINCT FROM p_request->>'predecessor_state'
 OR EXISTS(SELECT 1 FROM {s}.lifecycle_operations o WHERE o.fence_key=fence.identity
  AND NOT EXISTS(SELECT 1 FROM {s}.lifecycle_completions c WHERE c.operation_id=o.operation_id))
 THEN RAISE EXCEPTION 'stale or conflicting lifecycle predecessor' USING ERRCODE='23514'; END IF;
 INSERT INTO {s}.lifecycle_operations VALUES(p_request->>'operation_id',fence.identity,p_request);
 UPDATE {s}.fences SET state='PENDING_CLOSED' WHERE identity=fence.identity;
 RETURN {s}.lifecycle_operation_result(p_request->>'operation_id');
END
""",
        "read_lifecycle_operation(jsonb)": f"""
DECLARE fence {s}.fences%ROWTYPE; prior jsonb;
BEGIN
 {guard}
 PERFORM {s}.validate_lifecycle_request(p_request);
 SELECT * INTO fence FROM {s}.fences WHERE identity=p_request->>'fence_key' FOR UPDATE;
 IF NOT FOUND THEN RAISE EXCEPTION 'lifecycle fence missing' USING ERRCODE='23514'; END IF;
 -- This fresh READ COMMITTED statement follows the blocking fence lock. An
 -- absent intent cannot race a surviving backend's unresolved close COMMIT.
 SELECT request INTO prior FROM {s}.lifecycle_operations WHERE operation_id=p_request->>'operation_id';
 IF FOUND AND prior IS DISTINCT FROM p_request THEN
  RAISE EXCEPTION 'administrative operation identity conflict' USING ERRCODE='23505'; END IF;
 RETURN {s}.lifecycle_operation_result(p_request->>'operation_id');
END
""",
        "complete_lifecycle_close(text,bytea,bytea)": f"""
DECLARE op jsonb; evidence jsonb; observed jsonb; existing bytea;
 fence {s}.fences%ROWTYPE; secret bytea; fence_identity text;
BEGIN
 {guard}
 SELECT fence_key INTO fence_identity FROM {s}.lifecycle_operations WHERE operation_id=p_operation;
 IF NOT FOUND THEN RAISE EXCEPTION 'durable close required' USING ERRCODE='23514'; END IF;
 SELECT * INTO fence FROM {s}.fences WHERE identity=fence_identity FOR UPDATE;
 SELECT request INTO STRICT op FROM {s}.lifecycle_operations WHERE operation_id=p_operation;
 SELECT authentication_key INTO STRICT secret FROM {s}.lifecycle_authority_secret WHERE singleton;
 IF p_evidence IS NULL OR p_authentication IS NULL OR pg_catalog.octet_length(p_authentication)<>32
 OR p_authentication IS DISTINCT FROM {s}.lifecycle_hmac(secret,
  pg_catalog.decode('{COMPLETION_MAC_DOMAIN.hex()}','hex') || p_evidence)
 THEN RAISE EXCEPTION 'authenticated genuine custody validation required' USING ERRCODE='42501'; END IF;
 evidence:=pg_catalog.convert_from(p_evidence,'UTF8')::jsonb;
 observed:=evidence->'observed';
 IF (SELECT pg_catalog.array_agg(k ORDER BY k) FROM pg_catalog.jsonb_object_keys(evidence) k)
 IS DISTINCT FROM ARRAY['local_evidence_sha256','observed','operation']::text[]
 OR evidence->'operation' IS DISTINCT FROM op
 OR pg_catalog.jsonb_typeof(evidence->'local_evidence_sha256') IS DISTINCT FROM 'string'
 OR evidence->>'local_evidence_sha256' !~ '^[0-9a-f]{{64}}$'
 OR observed IS DISTINCT FROM pg_catalog.jsonb_build_object(
  'credential_identity',op->'credential_identity','public_key_hex',op->'public_key_hex',
  'key_version',op->'key_version','custody_scope',op->'custody_scope',
  'lifecycle_state',op->>'target_state','lifecycle_generation',(op->>'lifecycle_generation')::bigint+1)
 THEN RAISE EXCEPTION 'exact custody completion evidence required' USING ERRCODE='23514'; END IF;
 SELECT canonical_evidence INTO existing FROM {s}.lifecycle_completions WHERE operation_id=p_operation;
 IF FOUND THEN
  IF existing IS DISTINCT FROM p_evidence THEN
   RAISE EXCEPTION 'completion replay conflicts' USING ERRCODE='23505'; END IF;
  RETURN {s}.lifecycle_operation_result(p_operation);
 END IF;
 IF fence.state IS DISTINCT FROM 'PENDING_CLOSED'
 OR fence.semantic_role IS DISTINCT FROM op->>'semantic_role'
 OR fence.generation IS DISTINCT FROM (op->>'lifecycle_generation')::bigint
 THEN RAISE EXCEPTION 'completion predecessor differs' USING ERRCODE='23514'; END IF;
 INSERT INTO {s}.lifecycle_completions VALUES(p_operation,p_evidence,p_authentication);
 UPDATE {s}.fences SET state=op->>'target_state',generation=fence.generation+1 WHERE identity=fence.identity;
 RETURN {s}.lifecycle_operation_result(p_operation);
END
""",
        "guard_lifecycle_fence()": f"""
BEGIN
 IF TG_OP<>'UPDATE' THEN
  RAISE EXCEPTION 'V2 fence provisioning or deletion BLOCKED' USING ERRCODE='42501'; END IF;
 IF NEW.identity IS DISTINCT FROM OLD.identity OR NEW.semantic_role IS DISTINCT FROM OLD.semantic_role
 OR NEW.ever_active IS DISTINCT FROM OLD.ever_active THEN
  RAISE EXCEPTION 'immutable custody fence identity' USING ERRCODE='42501'; END IF;
 IF OLD.state IN ('ACTIVE','VERIFY_ONLY') AND NEW.state='PENDING_CLOSED'
 AND NEW.generation=OLD.generation AND EXISTS(
  SELECT 1 FROM {s}.lifecycle_operations o WHERE o.fence_key=OLD.identity
  AND o.request->>'predecessor_state'=OLD.state
  AND (o.request->>'lifecycle_generation')::bigint=OLD.generation
  AND NOT EXISTS(SELECT 1 FROM {s}.lifecycle_completions c WHERE c.operation_id=o.operation_id))
 THEN RETURN NEW; END IF;
 IF OLD.state='PENDING_CLOSED' AND NEW.state IN ('VERIFY_ONLY','REVOKED')
 AND NEW.generation=OLD.generation+1 AND EXISTS(
  SELECT 1 FROM {s}.lifecycle_operations o JOIN {s}.lifecycle_completions c USING(operation_id)
  WHERE o.fence_key=OLD.identity AND o.request->>'target_state'=NEW.state
  AND (o.request->>'lifecycle_generation')::bigint=OLD.generation)
 THEN RETURN NEW; END IF;
 RAISE EXCEPTION 'legacy lifecycle fence transition BLOCKED' USING ERRCODE='42501';
END
""",
        "guard_lifecycle_decision_fences()": f"""
DECLARE root_snapshot bytea; history_identity bytea; expected_root text; expected_history text;
BEGIN
 PERFORM {s}.assert_lifecycle_catalog();
 root_snapshot:={_q(config.intent_schema)}.atomic_intent_member(NEW.reserved_envelope,'credential_snapshot');
 IF pg_catalog.substring(NEW.canonical_head,1,{len(ATTESTATION_DOMAIN)})
 IS DISTINCT FROM pg_catalog.decode('{ATTESTATION_DOMAIN.hex()}','hex') THEN
  RAISE EXCEPTION 'exact original history attestation required' USING ERRCODE='23514'; END IF;
 history_identity:={_q(config.history_schema)}.atomic_history_member(
  pg_catalog.substring(NEW.canonical_head,{len(ATTESTATION_DOMAIN) + 1}),'signing_credential_identity');
 expected_root:='ROOT_PROOF_SIGNING:' || pg_catalog.encode(pg_catalog.sha256(
  pg_catalog.decode('{_FENCE_DOMAIN.hex()}','hex') || root_snapshot),'hex');
 expected_history:='HISTORY_ATTESTATION_SIGNING:' || pg_catalog.encode(pg_catalog.sha256(
  pg_catalog.decode('{_FENCE_DOMAIN.hex()}','hex') || history_identity),'hex');
 IF NEW.package->>'root_fence_key' IS DISTINCT FROM expected_root
 OR NEW.package->>'history_fence_key' IS DISTINCT FROM expected_history THEN
  RAISE EXCEPTION 'actual retained signer custody fence cannot be aliased' USING ERRCODE='23514'; END IF;
 RETURN NEW;
END
""",
    }


def install_custody_lifecycle_authority(
    conn: psycopg.Connection[Any],
    config: PostgreSQLAtomicRootProofProvisioning,
    credential_bindings: tuple[dict[str, Any], ...],
    evidence_key: bytes,
) -> dict[str, str]:
    """Install only inside an explicit, prequalified administrative transaction."""
    if type(config) is not PostgreSQLAtomicRootProofProvisioning:
        raise TypeError("exact atomic provisioning required")
    if conn.info.transaction_status != psycopg.pq.TransactionStatus.INTRANS:
        raise ValueError("explicit V2 migration transaction required")
    if type(evidence_key) is not bytes or len(evidence_key) != 32:
        raise ValueError("separate 256-bit administrator authentication key required")
    if type(credential_bindings) is not tuple or not credential_bindings:
        raise ValueError("explicit genuine custody credential enrollment required")
    # Migration serializes with in-flight commits and all old fence operations.
    conn.execute(
        sql.SQL("LOCK TABLE {}.fences IN ACCESS EXCLUSIVE MODE").format(
            sql.Identifier(config.schema)
        )
    )
    for binding in credential_bindings:
        if type(binding) is not dict or tuple(sorted(binding)) != BINDING_KEYS:
            raise ValueError("exact custody binding fields required")
        if binding["semantic_role"] not in {"ROOT_PROOF_SIGNING", "HISTORY_ATTESTATION_SIGNING"}:
            raise ValueError("only root and history signing custody can be enrolled")
        generation = binding["lifecycle_generation"]
        if type(generation) is not int or not 0 < generation < 2**63 - 1:
            raise ValueError("positive exact custody generation required")
        row = conn.execute(
            sql.SQL(
                "SELECT semantic_role,generation,state FROM {}.fences WHERE identity=%s"
            ).format(sql.Identifier(config.schema)),
            (binding["fence_key"],),
        ).fetchone()
        if row not in (
            (binding["semantic_role"], generation, "ACTIVE"),
            (binding["semantic_role"], generation, "VERIFY_ONLY"),
        ):
            raise ValueError("existing exact test fence and custody predecessor required")
        if any(
            type(binding[k]) is not dict
            for k in ("credential_identity", "custody_scope", "database_identity")
        ):
            raise ValueError("exact identity and scope objects required")
        if any(
            type(binding[k]) is not str or not binding[k] for k in ("fence_key", "public_key_hex")
        ):
            raise ValueError("exact custody public material required")
        if type(binding["key_version"]) is not int or binding["key_version"] < 1:
            raise ValueError("positive exact custody key version required")
        if row is None or binding["fence_key"] != expected_custody_fence_key(
            binding, predecessor_state=row[2]
        ):
            raise ValueError("custody binding uses a different canonical V1 issuer fence")
    s = sql.Identifier(config.schema)
    conn.execute(
        sql.SQL("""
CREATE TABLE {s}.lifecycle_metadata(
 singleton boolean PRIMARY KEY DEFAULT true CHECK(singleton),schema_identity text NOT NULL,
 schema_version integer NOT NULL CHECK(schema_version=2),migration_id text NOT NULL,
 admin_role_oid oid NOT NULL,owner_role_oid oid NOT NULL,
 authentication_key_sha256 bytea NOT NULL CHECK(octet_length(authentication_key_sha256)=32),
 boundary text NOT NULL CHECK(boundary='ISOLATED_TEST_ONLY'));
CREATE TABLE {s}.lifecycle_authority_secret(singleton boolean PRIMARY KEY DEFAULT true CHECK(singleton),
 authentication_key bytea NOT NULL CHECK(octet_length(authentication_key)=32));
CREATE TABLE {s}.lifecycle_credentials(fence_key text PRIMARY KEY REFERENCES {s}.fences(identity),
 credential_binding jsonb NOT NULL);
CREATE UNIQUE INDEX lifecycle_exact_credential_identity ON {s}.lifecycle_credentials
 ((credential_binding->'credential_identity'));
CREATE UNIQUE INDEX lifecycle_exact_public_material ON {s}.lifecycle_credentials
 ((credential_binding->'public_key_hex'));
CREATE TABLE {s}.lifecycle_operations(operation_id text PRIMARY KEY,fence_key text NOT NULL
 REFERENCES {s}.lifecycle_credentials(fence_key),request jsonb NOT NULL);
CREATE TABLE {s}.lifecycle_completions(operation_id text PRIMARY KEY REFERENCES {s}.lifecycle_operations(operation_id),
 canonical_evidence bytea NOT NULL,authentication bytea NOT NULL CHECK(octet_length(authentication)=32));
""").format(s=s)
    )
    for table in (
        "lifecycle_metadata",
        "lifecycle_authority_secret",
        "lifecycle_credentials",
        "lifecycle_operations",
        "lifecycle_completions",
    ):
        conn.execute(
            sql.SQL("ALTER TABLE {}.{} OWNER TO {}").format(
                s, sql.Identifier(table), sql.Identifier(config.schema_owner_role)
            )
        )
        conn.execute(
            sql.SQL("REVOKE ALL ON {}.{} FROM PUBLIC,{},{}").format(
                s,
                sql.Identifier(table),
                sql.Identifier(config.issuer_writer_role),
                sql.Identifier(config.test_admin_role),
            )
        )
        for trigger, event, level in (
            ("lifecycle_immutable_row", "UPDATE OR DELETE", "ROW"),
            ("lifecycle_immutable_truncate", "TRUNCATE", "STATEMENT"),
        ):
            conn.execute(
                sql.SQL(
                    "CREATE TRIGGER {} BEFORE {} ON {}.{} FOR EACH {} EXECUTE FUNCTION {}.reject_mutation()"
                ).format(
                    sql.Identifier(trigger),
                    sql.SQL(event),
                    s,
                    sql.Identifier(table),
                    sql.SQL(level),
                    s,
                )
            )
            conn.execute(
                sql.SQL("ALTER TABLE {}.{} ENABLE ALWAYS TRIGGER {}").format(
                    s, sql.Identifier(table), sql.Identifier(trigger)
                )
            )
    conn.execute(
        sql.SQL(
            "INSERT INTO {}.lifecycle_metadata SELECT true,%s,2,%s,test_admin_role_oid,schema_owner_role_oid,%s,%s FROM {}.metadata WHERE singleton"
        ).format(s, s),
        (SCHEMA_IDENTITY, MIGRATION_ID, hashlib.sha256(evidence_key).digest(), BOUNDARY),
    )
    conn.execute(
        sql.SQL("GRANT SELECT ON {}.lifecycle_metadata TO {},{}").format(
            s, sql.Identifier(config.issuer_writer_role), sql.Identifier(config.test_admin_role)
        )
    )
    conn.execute(
        sql.SQL("INSERT INTO {}.lifecycle_authority_secret VALUES(true,%s)").format(s),
        (evidence_key,),
    )
    for binding in credential_bindings:
        conn.execute(
            sql.SQL("INSERT INTO {}.lifecycle_credentials VALUES(%s,%s)").format(s),
            (binding["fence_key"], Jsonb(binding)),
        )
    inventory = reviewed_lifecycle_function_sources(config)
    for signature, source in inventory.items():
        declaration, returns = FUNCTION_DECLARATIONS[signature]
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
        if signature in FUNC_ADMIN:
            conn.execute(
                sql.SQL("GRANT EXECUTE ON FUNCTION {}.{} TO {}").format(
                    s, sql.SQL(signature), sql.Identifier(config.test_admin_role)
                )
            )
    for signature in (
        "provision_fence(text,text,bigint)",
        "transition_fence(text,bigint,text,bigint)",
    ):
        conn.execute(
            sql.SQL("REVOKE ALL ON FUNCTION {}.{} FROM {}").format(
                s, sql.SQL(signature), sql.Identifier(config.test_admin_role)
            )
        )
    conn.execute(
        sql.SQL(
            "CREATE TRIGGER lifecycle_fence_guard BEFORE INSERT OR UPDATE OR DELETE ON {}.fences FOR EACH ROW EXECUTE FUNCTION {}.guard_lifecycle_fence()"
        ).format(s, s)
    )
    conn.execute(
        sql.SQL("ALTER TABLE {}.fences ENABLE ALWAYS TRIGGER lifecycle_fence_guard").format(s)
    )
    conn.execute(
        sql.SQL(
            "CREATE TRIGGER lifecycle_fence_truncate BEFORE TRUNCATE ON {}.fences FOR EACH STATEMENT EXECUTE FUNCTION {}.reject_mutation()"
        ).format(s, s)
    )
    conn.execute(
        sql.SQL("ALTER TABLE {}.fences ENABLE ALWAYS TRIGGER lifecycle_fence_truncate").format(s)
    )
    conn.execute(
        sql.SQL(
            "CREATE TRIGGER lifecycle_decision_credential_fences BEFORE INSERT ON {}.decisions "
            "FOR EACH ROW EXECUTE FUNCTION {}.guard_lifecycle_decision_fences()"
        ).format(s, s)
    )
    conn.execute(
        sql.SQL(
            "ALTER TABLE {}.decisions ENABLE ALWAYS TRIGGER lifecycle_decision_credential_fences"
        ).format(s)
    )
    # The immutable V1 manifest is superseded explicitly by this versioned
    # administrative migration, never by a startup repair or runtime function.
    conn.execute(sql.SQL("ALTER TABLE {}.catalog_manifest DISABLE TRIGGER immutable_row").format(s))
    conn.execute("SAVEPOINT lifecycle_manifest_capture")
    conn.execute(
        sql.SQL("ALTER TABLE {}.catalog_manifest ENABLE ALWAYS TRIGGER immutable_row").format(s)
    )
    final_snapshot = conn.execute(sql.SQL("SELECT {}.catalog_snapshot()").format(s)).fetchone()
    if final_snapshot is None:
        raise ValueError("catalog snapshot unavailable")
    conn.execute("ROLLBACK TO SAVEPOINT lifecycle_manifest_capture")
    conn.execute("RELEASE SAVEPOINT lifecycle_manifest_capture")
    conn.execute(
        sql.SQL("UPDATE {}.catalog_manifest SET snapshot=%s WHERE singleton").format(s),
        (Jsonb(final_snapshot[0]),),
    )
    conn.execute(
        sql.SQL("ALTER TABLE {}.catalog_manifest ENABLE ALWAYS TRIGGER immutable_row").format(s)
    )
    return inventory


__all__ = [
    "BOUNDARY",
    "SCHEMA_IDENTITY",
    "SCHEMA_VERSION",
    "MIGRATION_ID",
    "COMPLETION_MAC_DOMAIN",
    "REQUEST_KEYS",
    "BINDING_KEYS",
    "FUNC_ADMIN",
    "FUNCTION_DECLARATIONS",
    "expected_custody_fence_key",
    "reviewed_lifecycle_function_sources",
    "install_custody_lifecycle_authority",
]
