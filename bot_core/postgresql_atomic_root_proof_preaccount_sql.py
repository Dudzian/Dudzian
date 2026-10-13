"""Reviewed final credential fences owned by the pre-account semantic authorities.

Only administrative installation chooses the fixed names embedded in a helper.
Its runtime input is an exact retained credential snapshot, never SQL names or
queries. The coordinator alone receives EXECUTE; the helper performs no DML.
"""

from __future__ import annotations

import re

from psycopg import sql

from bot_core.postgresql_preaccount_credentials import (
    CLAIMANT_CREDENTIAL_ROLE,
    REQUESTER_CREDENTIAL_ROLE,
    REQUESTER_PRINCIPAL,
)
from bot_core.root_proof_issuer_substrate import (
    _PUBLIC_KEY_FINGERPRINT_DOMAIN,
    CredentialSemanticRole,
)

_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]{0,62}\Z")
PREACCOUNT_HELPER_SIGNATURE = "atomic_lock_credential(jsonb)"
PREACCOUNT_HELPER_DECLARATION = "p_expected jsonb"
PREACCOUNT_CREDENTIAL_FIELDS = (
    "credential_id",
    "credential_role",
    "environment",
    "key_id",
    "key_material_identity",
    "key_version",
    "lifecycle",
    "lifecycle_generation",
    "principal_id",
    "public_key_hex",
    "registry_revision",
    "semantic_role",
    "trust_domain",
)


def reviewed_preaccount_helper(
    *,
    owner: str,
    writer: str,
    schema: str,
    peer_schema: str,
    semantic_role: str,
    trust_domain: str,
) -> str:
    """Generate one fixed owner helper with the existing administration lock."""
    for name in (owner, writer, schema, peer_schema):
        if type(name) is not str or _IDENTIFIER.fullmatch(name) is None:
            raise ValueError("fixed lowercase pre-account authority identifiers required")
    if schema == peer_schema or owner == writer:
        raise ValueError("independent pre-account schema owner and issuer writer required")
    if type(trust_domain) is not str or not trust_domain.strip():
        raise ValueError("fixed pre-account trust domain required")
    roles = {
        CredentialSemanticRole.ROOT_PROOF_REQUESTER.value: REQUESTER_CREDENTIAL_ROLE,
        CredentialSemanticRole.ROOT_PROOF_CLAIMANT.value: CLAIMANT_CREDENTIAL_ROLE,
    }
    if type(semantic_role) is not str or semantic_role not in roles:
        raise ValueError("fixed requester or claimant semantic role required")
    lock_namespace = "cryptohunter.preaccount.credentials:" + ":".join(
        sorted((schema, peer_schema))
    )
    source = sql.SQL("""
DECLARE actual jsonb; key text;
BEGIN
 IF session_user <> {writer} OR current_user <> {owner}
 OR (SELECT oid FROM pg_catalog.pg_roles WHERE rolname=current_user)
 IS DISTINCT FROM (SELECT (config->>'schema_owner_role_oid')::oid FROM {s}.metadata WHERE singleton)
 THEN RAISE EXCEPTION 'wrong final pre-account authority principal' USING ERRCODE='42501'; END IF;
 IF pg_catalog.current_setting('server_version_num')::integer < 160000
 OR pg_catalog.current_setting('server_version_num')::integer >= 170000
 OR pg_catalog.current_setting('transaction_isolation') <> 'serializable'
 OR pg_catalog.current_setting('transaction_read_only') <> 'off'
 OR pg_catalog.current_setting('fsync') <> 'on'
 OR pg_catalog.current_setting('synchronous_commit') <> 'on'
 OR pg_catalog.current_setting('full_page_writes') <> 'on'
 THEN RAISE EXCEPTION 'durable final credential transaction required' USING ERRCODE='25001'; END IF;
 IF (SELECT config->>'semantic_role' FROM {s}.metadata WHERE singleton)
 IS DISTINCT FROM {semantic_role}
 OR (SELECT config->>'schema' FROM {s}.metadata WHERE singleton) IS DISTINCT FROM {schema}
 OR (SELECT config#>>'{{peer,schema}}' FROM {s}.metadata WHERE singleton)
 IS DISTINCT FROM {peer_schema}
 THEN RAISE EXCEPTION 'pre-account authority metadata mismatch' USING ERRCODE='42501'; END IF;
 IF p_expected IS NULL OR pg_catalog.jsonb_typeof(p_expected) IS DISTINCT FROM 'object'
 OR (SELECT pg_catalog.array_agg(k ORDER BY k) FROM pg_catalog.jsonb_object_keys(p_expected) k)
 IS DISTINCT FROM ARRAY[{fields}]::text[]
 THEN RAISE EXCEPTION 'exact retained credential snapshot required' USING ERRCODE='22023'; END IF;
 FOREACH key IN ARRAY ARRAY['credential_id','credential_role','environment','key_id',
 'key_material_identity','lifecycle','principal_id','public_key_hex','semantic_role','trust_domain'] LOOP
  IF pg_catalog.jsonb_typeof(p_expected->key) IS DISTINCT FROM 'string'
  OR pg_catalog.btrim(p_expected->>key)='' THEN
   RAISE EXCEPTION 'invalid retained credential identity' USING ERRCODE='22023'; END IF;
 END LOOP;
 FOREACH key IN ARRAY ARRAY['key_version','lifecycle_generation','registry_revision'] LOOP
  IF pg_catalog.jsonb_typeof(p_expected->key) IS DISTINCT FROM 'number'
  OR (p_expected->>key) !~ '^[1-9][0-9]*$'
  OR (p_expected->>key)::numeric > 9007199254740991 THEN
   RAISE EXCEPTION 'invalid retained credential generation' USING ERRCODE='22023'; END IF;
 END LOOP;
 IF p_expected->>'environment' IS DISTINCT FROM 'PRODUCTION'
 OR p_expected->>'trust_domain' IS DISTINCT FROM {trust_domain}
 OR p_expected->>'semantic_role' IS DISTINCT FROM {semantic_role}
 OR p_expected->>'credential_role' IS DISTINCT FROM {credential_role}
 OR p_expected->>'lifecycle' IS DISTINCT FROM 'ACTIVE'
 OR (p_expected->>'public_key_hex') !~ '^[0-9a-f]{{64}}$'
 OR ({requester} AND p_expected->>'principal_id' IS DISTINCT FROM {requester_principal})
 THEN RAISE EXCEPTION 'new BIND requires exact ACTIVE pre-account credential' USING ERRCODE='23514'; END IF;
 -- Exactly the namespace and lock order used by legacy admin apply(jsonb).
 -- The backend retains this advisory lock and current-row lock through COMMIT.
 PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended({lock_namespace},0));
 SELECT pg_catalog.jsonb_build_object(
 'credential_id',k.credential_id,'principal_id',k.principal_id,'semantic_role',k.semantic_role,
 'credential_role',k.credential_role,'key_id',k.key_id,'key_version',k.key_version,
 'public_key_hex',pg_catalog.encode(k.public_key,'hex'),'key_material_identity',k.key_material_identity,
 'environment',k.environment,'trust_domain',k.trust_domain,
 'lifecycle_generation',h.lifecycle_generation,'lifecycle',h.lifecycle,'registry_revision',c.revision)
 INTO actual FROM {s}.current_credentials c JOIN {s}.history h
 ON (h.principal_id,h.revision)=(c.principal_id,c.revision)
 JOIN {s}.credentials k ON k.credential_id=h.credential_id AND k.principal_id=h.principal_id
 WHERE c.principal_id=p_expected->>'principal_id' FOR UPDATE OF c;
 IF NOT FOUND OR actual IS DISTINCT FROM p_expected THEN
  RAISE EXCEPTION 'current pre-account lifecycle or identity differs' USING ERRCODE='23514'; END IF;
 IF actual->>'key_material_identity' IS DISTINCT FROM 'sha256:' || pg_catalog.encode(
 pg_catalog.sha256(pg_catalog.decode({fingerprint_domain_hex},'hex') ||
 pg_catalog.decode(actual->>'public_key_hex','hex')),'hex') THEN
  RAISE EXCEPTION 'pre-account public key material identity corrupt' USING ERRCODE='XX001'; END IF;
END
""").format(
        s=sql.Identifier(schema),
        owner=sql.Literal(owner),
        writer=sql.Literal(writer),
        schema=sql.Literal(schema),
        peer_schema=sql.Literal(peer_schema),
        semantic_role=sql.Literal(semantic_role),
        credential_role=sql.Literal(roles[semantic_role]),
        trust_domain=sql.Literal(trust_domain),
        requester=sql.Literal(semantic_role == CredentialSemanticRole.ROOT_PROOF_REQUESTER.value),
        requester_principal=sql.Literal(REQUESTER_PRINCIPAL),
        lock_namespace=sql.Literal(lock_namespace),
        fields=sql.SQL(",".join("'" + field + "'" for field in PREACCOUNT_CREDENTIAL_FIELDS)),
        fingerprint_domain_hex=sql.Literal(_PUBLIC_KEY_FINGERPRINT_DOMAIN.hex()),
    )
    return str(source.as_string())


__all__ = [
    "PREACCOUNT_HELPER_SIGNATURE",
    "PREACCOUNT_HELPER_DECLARATION",
    "PREACCOUNT_CREDENTIAL_FIELDS",
    "reviewed_preaccount_helper",
]
