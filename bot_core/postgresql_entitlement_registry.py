"""PostgreSQL-backed PRODUCTION_LOCAL entitlement registry.

Schema creation is deliberately an offline operation.  Runtime and admin
providers only qualify and use a pre-provisioned schema with distinct
PostgreSQL principals.  PostgreSQL durability does not constitute an
independent checkpoint domain and cannot detect a privileged rollback of the
entire database to an earlier valid prefix.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import re
from typing import Any, Callable, TypeVar

import psycopg
from psycopg import sql
from psycopg.errors import CheckViolation, SerializationFailure, UniqueViolation

from bot_core.entitlement_registry_contract import (
    AdminOutcome,
    AdminResult,
    AuthoritativeEntitlementState,
    BindOutcome,
    BindRequest,
    BindResolutionKind,
    BindResult,
    BindingKind,
    BoundBinding,
    ContractValidationError,
    EntitlementIdentity,
    EntitlementLifecycle,
    EntitlementProvenance,
    HistoricalStateResult,
    ProvisionEntitlementRequest,
    RegistryReadOutcome,
    RegistryReadResult,
    RegistrySubject,
    RetainedHistoryResult,
    RevokeEntitlementRequest,
    SupersedeEntitlementRequest,
    UnboundBinding,
    admin_predecessor_for,
    initial_state_for,
    resolve_bind_request,
    revoked_state_for,
    supersession_states_for,
    validate_complete_lineage,
    validate_exact_snapshot,
)
from bot_core.root_proof_issuer_substrate import (
    ProviderCapabilities,
    ProviderIdentity,
    ProviderRole,
    SecurityProfile,
    SecurityProfileIdentity,
)


SCHEMA_IDENTITY = "cryptohunter.entitlement_registry.postgresql"
SCHEMA_VERSION = 5
STORAGE_FAMILY = "POSTGRESQL_APPEND_ONLY_V5"
PROFILE = "PRODUCTION_LOCAL"
_FUNCTION_DEFINITION_FINGERPRINT_DOMAIN = (
    b"CRYPT0HUNTER_ENTITLEMENT_REGISTRY_PG_FUNCTION_SOURCE_V5\x00"
)
_FUNCTION_SIGNATURES = (
    "validate_state(json)",
    "append_bind(text,text,text,bigint,json)",
    "provision(text,text,text,text,text,json)",
    "revoke(text,text,text,bigint,json)",
    "supersede(text,text,text,bigint,json,json)",
)
_SAFE_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]{0,62}\Z")
_T = TypeVar("_T")


class RegistryQualificationError(RuntimeError):
    """The configured database is not the exact reviewed registry schema."""


@dataclass(frozen=True, slots=True)
class PostgreSQLConnectionConfig:
    """Secret-bearing connection input whose repr never exposes its DSN."""

    dsn: str

    def __post_init__(self) -> None:
        if type(self.dsn) is not str or not self.dsn.strip():
            raise TypeError("dsn must be an exact non-empty str")

    def __repr__(self) -> str:
        return "PostgreSQLConnectionConfig(dsn=<redacted>)"


@dataclass(frozen=True, slots=True)
class PostgreSQLRegistryProvisioning:
    schema: str
    schema_owner_role: str
    runtime_role: str
    admin_role: str
    trust_domain: str

    def __post_init__(self) -> None:
        for name in ("schema", "schema_owner_role", "runtime_role", "admin_role"):
            value = getattr(self, name)
            if type(value) is not str or _SAFE_IDENTIFIER.fullmatch(value) is None:
                raise ValueError(f"{name} must be a safe lowercase PostgreSQL identifier")
        if len({self.schema_owner_role, self.runtime_role, self.admin_role}) != 3:
            raise ValueError("schema owner, runtime, and admin roles must be distinct")
        if type(self.trust_domain) is not str or not self.trust_domain.strip():
            raise ValueError("trust_domain must be an exact non-empty str")


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def _state_json(state: AuthoritativeEntitlementState) -> dict[str, Any]:
    trusted = validate_exact_snapshot(state)
    if not isinstance(trusted, AuthoritativeEntitlementState):
        raise ContractValidationError("state must be exact AuthoritativeEntitlementState")
    return _json_value(asdict(trusted))


def _state_from_json(payload: object) -> AuthoritativeEntitlementState:
    if type(payload) is not dict:
        raise ContractValidationError("stored state must be an exact JSON object")
    subject_data = payload.get("subject")
    identity_data = payload.get("identity")
    provenance_data = payload.get("provenance")
    binding_data = payload.get("binding")
    if not all(
        type(item) is dict for item in (subject_data, identity_data, provenance_data, binding_data)
    ):
        raise ContractValidationError("stored state has malformed nested records")
    assert isinstance(subject_data, dict)
    assert isinstance(identity_data, dict)
    assert isinstance(provenance_data, dict)
    assert isinstance(binding_data, dict)
    subject = RegistrySubject(**subject_data)
    identity = EntitlementIdentity(**identity_data)
    provenance = EntitlementProvenance(**provenance_data)
    kind = binding_data.get("kind")
    binding: UnboundBinding | BoundBinding
    if kind == BindingKind.UNBOUND.value:
        binding = UnboundBinding(BindingKind.UNBOUND)
    elif kind == BindingKind.BOUND.value:
        binding = BoundBinding(**{**binding_data, "kind": BindingKind.BOUND})
    else:
        raise ContractValidationError("stored binding kind is invalid")
    return AuthoritativeEntitlementState(
        subject=subject,
        identity=identity,
        provenance=provenance,
        lifecycle=EntitlementLifecycle(payload.get("lifecycle")),
        binding=binding,
        authoritative_state_revision=payload.get("authoritative_state_revision"),
        predecessor_revision=payload.get("predecessor_revision"),
    )


def _provisioning_sql(config: PostgreSQLRegistryProvisioning) -> sql.Composed:
    s = sql.Identifier(config.schema)
    owner = sql.Identifier(config.schema_owner_role)
    runtime = sql.Identifier(config.runtime_role)
    admin = sql.Identifier(config.admin_role)
    return sql.SQL(
        """
CREATE SCHEMA {s} AUTHORIZATION {owner};
SET LOCAL ROLE {owner};
CREATE TABLE {s}.metadata (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    schema_identity text NOT NULL,
    schema_version integer NOT NULL,
    profile text NOT NULL,
    trust_domain text NOT NULL,
    storage_family text NOT NULL,
    schema_owner_role text NOT NULL,
    schema_owner_role_oid oid NOT NULL,
    runtime_role text NOT NULL,
    runtime_role_oid oid NOT NULL,
    admin_role text NOT NULL,
    admin_role_oid oid NOT NULL,
    function_definition_sha256 jsonb NOT NULL
);
INSERT INTO {s}.metadata(
    schema_identity, schema_version, profile, trust_domain, storage_family,
    schema_owner_role, schema_owner_role_oid, runtime_role, runtime_role_oid,
    admin_role, admin_role_oid, function_definition_sha256
)
VALUES (
    {schema_identity}, {schema_version}, {profile}, {trust_domain}, {storage_family},
    {owner_name}, (SELECT oid FROM pg_catalog.pg_roles WHERE rolname={owner_name}),
    {runtime_name}, (SELECT oid FROM pg_catalog.pg_roles WHERE rolname={runtime_name}),
    {admin_name}, (SELECT oid FROM pg_catalog.pg_roles WHERE rolname={admin_name}),
    '{{}}'::jsonb
);
CREATE TABLE {s}.lineages (
    lookup_handle text NOT NULL,
    environment text NOT NULL,
    trust_domain text NOT NULL,
    product_scope text NOT NULL,
    bootstrap_entitlement_id text NOT NULL,
    current_revision bigint NOT NULL CHECK (current_revision >= 1),
    PRIMARY KEY (lookup_handle, environment, trust_domain),
    UNIQUE (environment, trust_domain, product_scope, bootstrap_entitlement_id)
);
CREATE TABLE {s}.history (
    lookup_handle text NOT NULL,
    environment text NOT NULL,
    trust_domain text NOT NULL,
    revision bigint NOT NULL CHECK (revision >= 1),
    predecessor_revision bigint,
    state jsonb NOT NULL,
    state_integrity text NOT NULL,
    PRIMARY KEY (lookup_handle, environment, trust_domain, revision),
    FOREIGN KEY (lookup_handle, environment, trust_domain)
      REFERENCES {s}.lineages(lookup_handle, environment, trust_domain)
      DEFERRABLE INITIALLY DEFERRED
);
COMMENT ON COLUMN {s}.history.state_integrity IS
  'Local accidental-corruption checksum only; not protection from schema owner or host admin';
CREATE FUNCTION {s}.validate_state(p_state json) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $fn$
DECLARE
  text_field text;
  integer_field text;
  identifier_field text;
  digest_field text;
  binding_value jsonb;
  state_value jsonb := p_state::jsonb;
  python_strip_whitespace text := pg_catalog.concat(
    pg_catalog.chr(9),pg_catalog.chr(10),pg_catalog.chr(11),
    pg_catalog.chr(12),pg_catalog.chr(13),pg_catalog.chr(28),
    pg_catalog.chr(29),pg_catalog.chr(30),pg_catalog.chr(31),
    pg_catalog.chr(32),pg_catalog.chr(133),pg_catalog.chr(160),
    pg_catalog.chr(5760),pg_catalog.chr(8192),pg_catalog.chr(8193),
    pg_catalog.chr(8194),pg_catalog.chr(8195),pg_catalog.chr(8196),
    pg_catalog.chr(8197),pg_catalog.chr(8198),pg_catalog.chr(8199),
    pg_catalog.chr(8200),pg_catalog.chr(8201),pg_catalog.chr(8202),
    pg_catalog.chr(8232),pg_catalog.chr(8233),pg_catalog.chr(8239),
    pg_catalog.chr(8287),pg_catalog.chr(12288)
  );
BEGIN
  IF p_state IS NULL OR pg_catalog.jsonb_typeof(state_value) IS DISTINCT FROM 'object'
     OR (SELECT pg_catalog.array_agg(key ORDER BY key) FROM pg_catalog.jsonb_object_keys(state_value) key)
        IS DISTINCT FROM ARRAY['authoritative_state_revision','binding','identity','lifecycle','predecessor_revision','provenance','subject']::text[]
     OR pg_catalog.jsonb_typeof(state_value->'subject') IS DISTINCT FROM 'object'
     OR (SELECT pg_catalog.array_agg(key ORDER BY key) FROM pg_catalog.jsonb_object_keys(state_value->'subject') key)
        IS DISTINCT FROM ARRAY['environment','lookup_handle','trust_domain']::text[]
     OR pg_catalog.jsonb_typeof(state_value->'identity') IS DISTINCT FROM 'object'
     OR (SELECT pg_catalog.array_agg(key ORDER BY key) FROM pg_catalog.jsonb_object_keys(state_value->'identity') key)
        IS DISTINCT FROM ARRAY['bootstrap_entitlement_id','entitlement_generation','environment','intended_action','product_scope','trust_domain']::text[]
     OR pg_catalog.jsonb_typeof(state_value->'provenance') IS DISTINCT FROM 'object'
     OR (SELECT pg_catalog.array_agg(key ORDER BY key) FROM pg_catalog.jsonb_object_keys(state_value->'provenance') key)
        IS DISTINCT FROM ARRAY['authenticated_creation_digest_sha256','authenticated_creation_reference','claimant_key_id','claimant_key_version','creation_authority_identity','provisioning_principal_id']::text[]
     OR pg_catalog.jsonb_typeof(state_value->'binding') IS DISTINCT FROM 'object'
     OR pg_catalog.jsonb_typeof(state_value->'lifecycle') IS DISTINCT FROM 'string'
     OR state_value->>'lifecycle' NOT IN ('ACTIVE','REVOKED','SUPERSEDED')
     OR pg_catalog.jsonb_typeof(state_value->'authoritative_state_revision') IS DISTINCT FROM 'number'
     OR (state_value->>'authoritative_state_revision') !~ '^[1-9][0-9]*$'
     OR NOT (
       state_value->'predecessor_revision' = 'null'::jsonb
       OR (
         pg_catalog.jsonb_typeof(state_value->'predecessor_revision') = 'number'
         AND (state_value->>'predecessor_revision') ~ '^[1-9][0-9]*$'
         AND (state_value->>'predecessor_revision')::bigint
             < (state_value->>'authoritative_state_revision')::bigint
       )
     ) THEN
    RAISE EXCEPTION 'malformed authoritative state envelope' USING ERRCODE='22023';
  END IF;

  IF (p_state #> '{{authoritative_state_revision}}')::text !~ '^[1-9][0-9]*$'
     OR (
       (p_state #> '{{predecessor_revision}}')::text IS DISTINCT FROM 'null'
       AND (p_state #> '{{predecessor_revision}}')::text !~ '^[1-9][0-9]*$'
     )
     OR (p_state #> '{{identity,entitlement_generation}}')::text !~ '^[1-9][0-9]*$'
     OR (p_state #> '{{provenance,claimant_key_version}}')::text !~ '^[1-9][0-9]*$' THEN
    RAISE EXCEPTION 'non-integral JSON numeric lexical form' USING ERRCODE='22023';
  END IF;

  FOREACH text_field IN ARRAY ARRAY[
    state_value#>>'{{subject,lookup_handle}}',state_value#>>'{{subject,environment}}',
    state_value#>>'{{subject,trust_domain}}',state_value#>>'{{identity,environment}}',
    state_value#>>'{{identity,trust_domain}}',state_value#>>'{{identity,product_scope}}',
    state_value#>>'{{provenance,provisioning_principal_id}}',
    state_value#>>'{{provenance,claimant_key_id}}',
    state_value#>>'{{provenance,creation_authority_identity}}',
    state_value#>>'{{provenance,authenticated_creation_reference}}'
  ] LOOP
    IF text_field IS NULL
       OR pg_catalog.translate(text_field,python_strip_whitespace,'') = '' THEN
      RAISE EXCEPTION 'malformed required state text' USING ERRCODE='22023';
    END IF;
  END LOOP;
  IF state_value#>>'{{subject,environment}}' IS DISTINCT FROM state_value#>>'{{identity,environment}}'
     OR state_value#>>'{{subject,trust_domain}}' IS DISTINCT FROM state_value#>>'{{identity,trust_domain}}'
     OR state_value#>>'{{identity,intended_action}}' IS DISTINCT FROM 'ACCOUNT_GENESIS_BOOTSTRAP'
     OR pg_catalog.jsonb_typeof(state_value#>'{{identity,entitlement_generation}}') IS DISTINCT FROM 'number'
     OR (state_value#>>'{{identity,entitlement_generation}}') !~ '^[1-9][0-9]*$'
     OR pg_catalog.jsonb_typeof(state_value#>'{{provenance,claimant_key_version}}') IS DISTINCT FROM 'number'
     OR (state_value#>>'{{provenance,claimant_key_version}}') !~ '^[1-9][0-9]*$'
     OR (state_value#>>'{{identity,bootstrap_entitlement_id}}') !~ '^ent_[0-9a-f]{{8}}-[0-9a-f]{{4}}-7[0-9a-f]{{3}}-[89ab][0-9a-f]{{3}}-[0-9a-f]{{12}}$'
     OR (state_value#>>'{{provenance,authenticated_creation_digest_sha256}}') !~ '^[0-9a-f]{{64}}$' THEN
    RAISE EXCEPTION 'malformed state identity or provenance' USING ERRCODE='22023';
  END IF;
  IF pg_catalog.jsonb_typeof(state_value#>'{{subject,lookup_handle}}') IS DISTINCT FROM 'string'
     OR pg_catalog.jsonb_typeof(state_value#>'{{subject,environment}}') IS DISTINCT FROM 'string'
     OR pg_catalog.jsonb_typeof(state_value#>'{{subject,trust_domain}}') IS DISTINCT FROM 'string'
     OR pg_catalog.jsonb_typeof(state_value#>'{{identity,bootstrap_entitlement_id}}') IS DISTINCT FROM 'string'
     OR pg_catalog.jsonb_typeof(state_value#>'{{identity,environment}}') IS DISTINCT FROM 'string'
     OR pg_catalog.jsonb_typeof(state_value#>'{{identity,trust_domain}}') IS DISTINCT FROM 'string'
     OR pg_catalog.jsonb_typeof(state_value#>'{{identity,product_scope}}') IS DISTINCT FROM 'string'
     OR pg_catalog.jsonb_typeof(state_value#>'{{identity,intended_action}}') IS DISTINCT FROM 'string'
     OR pg_catalog.jsonb_typeof(state_value#>'{{provenance,provisioning_principal_id}}') IS DISTINCT FROM 'string'
     OR pg_catalog.jsonb_typeof(state_value#>'{{provenance,claimant_key_id}}') IS DISTINCT FROM 'string'
     OR pg_catalog.jsonb_typeof(state_value#>'{{provenance,creation_authority_identity}}') IS DISTINCT FROM 'string'
     OR pg_catalog.jsonb_typeof(state_value#>'{{provenance,authenticated_creation_reference}}') IS DISTINCT FROM 'string'
     OR pg_catalog.jsonb_typeof(state_value#>'{{provenance,authenticated_creation_digest_sha256}}') IS DISTINCT FROM 'string' THEN
    RAISE EXCEPTION 'state text has wrong JSON type' USING ERRCODE='22023';
  END IF;

  binding_value := state_value->'binding';
  IF binding_value->>'kind' = 'UNBOUND' THEN
    IF (SELECT pg_catalog.array_agg(key ORDER BY key) FROM pg_catalog.jsonb_object_keys(binding_value) key)
       IS DISTINCT FROM ARRAY['kind']::text[] THEN
      RAISE EXCEPTION 'malformed UNBOUND binding' USING ERRCODE='22023';
    END IF;
  ELSIF binding_value->>'kind' = 'BOUND' THEN
    IF (SELECT pg_catalog.array_agg(key ORDER BY key) FROM pg_catalog.jsonb_object_keys(binding_value) key)
       IS DISTINCT FROM ARRAY['account_id','canonical_genesis_request_fingerprint_sha256','claimant_key_id','claimant_key_version','entitlement_generation','issuance_attempt_id','issuer_signing_credential_id','issuer_signing_key_version','kind','logical_operation_id','provisioning_principal_id','requester_key_id','requester_key_version','requester_principal_id','root_proof_id','signed_request_canonical_bytes_reference','signed_request_payload_digest_sha256']::text[] THEN
      RAISE EXCEPTION 'malformed BOUND binding keys' USING ERRCODE='22023';
    END IF;
    FOREACH text_field IN ARRAY ARRAY[
      binding_value->>'requester_principal_id',binding_value->>'requester_key_id',
      binding_value->>'provisioning_principal_id',binding_value->>'claimant_key_id',
      binding_value->>'signed_request_canonical_bytes_reference',
      binding_value->>'issuer_signing_credential_id'
    ] LOOP
      IF text_field IS NULL
         OR pg_catalog.translate(text_field,python_strip_whitespace,'') = '' THEN
        RAISE EXCEPTION 'malformed BOUND text' USING ERRCODE='22023';
      END IF;
    END LOOP;
    FOREACH integer_field IN ARRAY ARRAY[
      'entitlement_generation','requester_key_version',
      'claimant_key_version','issuer_signing_key_version'
    ] LOOP
      IF (p_state #> ARRAY['binding',integer_field])::text !~ '^[1-9][0-9]*$' THEN
        RAISE EXCEPTION 'non-integral BOUND JSON numeric lexical form' USING ERRCODE='22023';
      END IF;
      IF pg_catalog.jsonb_typeof(binding_value->integer_field) IS DISTINCT FROM 'number'
         OR binding_value->>integer_field !~ '^[1-9][0-9]*$' THEN
        RAISE EXCEPTION 'malformed BOUND integer' USING ERRCODE='22023';
      END IF;
    END LOOP;
    FOREACH identifier_field IN ARRAY ARRAY[
      binding_value->>'logical_operation_id',binding_value->>'account_id',
      binding_value->>'issuance_attempt_id',binding_value->>'root_proof_id'
    ] LOOP
      IF identifier_field IS NULL OR identifier_field !~ '^(ago|acct|rpa|rpf)_[0-9a-f]{{8}}-[0-9a-f]{{4}}-7[0-9a-f]{{3}}-[89ab][0-9a-f]{{3}}-[0-9a-f]{{12}}$' THEN
        RAISE EXCEPTION 'malformed BOUND identifier' USING ERRCODE='22023';
      END IF;
    END LOOP;
    IF binding_value->>'logical_operation_id' !~ '^ago_[0-9a-f]{{8}}-[0-9a-f]{{4}}-7[0-9a-f]{{3}}-[89ab][0-9a-f]{{3}}-[0-9a-f]{{12}}$'
       OR binding_value->>'account_id' !~ '^acct_[0-9a-f]{{8}}-[0-9a-f]{{4}}-7[0-9a-f]{{3}}-[89ab][0-9a-f]{{3}}-[0-9a-f]{{12}}$'
       OR binding_value->>'issuance_attempt_id' !~ '^rpa_[0-9a-f]{{8}}-[0-9a-f]{{4}}-7[0-9a-f]{{3}}-[89ab][0-9a-f]{{3}}-[0-9a-f]{{12}}$'
       OR binding_value->>'root_proof_id' !~ '^rpf_[0-9a-f]{{8}}-[0-9a-f]{{4}}-7[0-9a-f]{{3}}-[89ab][0-9a-f]{{3}}-[0-9a-f]{{12}}$' THEN
      RAISE EXCEPTION 'BOUND identifier prefix mismatch' USING ERRCODE='22023';
    END IF;
    FOREACH digest_field IN ARRAY ARRAY[
      binding_value->>'canonical_genesis_request_fingerprint_sha256',
      binding_value->>'signed_request_payload_digest_sha256'
    ] LOOP
      IF digest_field IS NULL OR digest_field !~ '^[0-9a-f]{{64}}$' THEN
        RAISE EXCEPTION 'malformed BOUND digest' USING ERRCODE='22023';
      END IF;
    END LOOP;
    IF pg_catalog.jsonb_typeof(binding_value->'logical_operation_id') IS DISTINCT FROM 'string'
       OR pg_catalog.jsonb_typeof(binding_value->'account_id') IS DISTINCT FROM 'string'
       OR pg_catalog.jsonb_typeof(binding_value->'issuance_attempt_id') IS DISTINCT FROM 'string'
       OR pg_catalog.jsonb_typeof(binding_value->'root_proof_id') IS DISTINCT FROM 'string'
       OR pg_catalog.jsonb_typeof(binding_value->'canonical_genesis_request_fingerprint_sha256') IS DISTINCT FROM 'string'
       OR pg_catalog.jsonb_typeof(binding_value->'signed_request_payload_digest_sha256') IS DISTINCT FROM 'string'
       OR pg_catalog.jsonb_typeof(binding_value->'requester_principal_id') IS DISTINCT FROM 'string'
       OR pg_catalog.jsonb_typeof(binding_value->'requester_key_id') IS DISTINCT FROM 'string'
       OR pg_catalog.jsonb_typeof(binding_value->'provisioning_principal_id') IS DISTINCT FROM 'string'
       OR pg_catalog.jsonb_typeof(binding_value->'claimant_key_id') IS DISTINCT FROM 'string'
       OR pg_catalog.jsonb_typeof(binding_value->'signed_request_canonical_bytes_reference') IS DISTINCT FROM 'string'
       OR pg_catalog.jsonb_typeof(binding_value->'issuer_signing_credential_id') IS DISTINCT FROM 'string' THEN
      RAISE EXCEPTION 'BOUND text has wrong JSON type' USING ERRCODE='22023';
    END IF;
    IF binding_value->>'entitlement_generation' IS DISTINCT FROM state_value#>>'{{identity,entitlement_generation}}'
       OR binding_value->>'provisioning_principal_id' IS DISTINCT FROM state_value#>>'{{provenance,provisioning_principal_id}}'
       OR binding_value->>'claimant_key_id' IS DISTINCT FROM state_value#>>'{{provenance,claimant_key_id}}'
       OR binding_value->>'claimant_key_version' IS DISTINCT FROM state_value#>>'{{provenance,claimant_key_version}}' THEN
      RAISE EXCEPTION 'BOUND authority anchor mismatch' USING ERRCODE='22023';
    END IF;
  ELSE
    RAISE EXCEPTION 'malformed binding kind' USING ERRCODE='22023';
  END IF;
END $fn$;
CREATE FUNCTION {s}.provision(p_handle text, p_environment text, p_trust text,
    p_product text, p_entitlement text, p_state json) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $fn$
BEGIN
  PERFORM {s}.validate_state(p_state);
  IF (p_state::jsonb) #>> '{{subject,lookup_handle}}' IS DISTINCT FROM p_handle
     OR (p_state::jsonb) #>> '{{subject,environment}}' IS DISTINCT FROM p_environment
     OR (p_state::jsonb) #>> '{{subject,trust_domain}}' IS DISTINCT FROM p_trust
     OR (p_state::jsonb) #>> '{{identity,product_scope}}' IS DISTINCT FROM p_product
     OR (p_state::jsonb) #>> '{{identity,bootstrap_entitlement_id}}' IS DISTINCT FROM p_entitlement
     OR ((p_state::jsonb) #>> '{{identity,entitlement_generation}}')::bigint IS DISTINCT FROM 1
     OR (p_state::jsonb) ->> 'lifecycle' IS DISTINCT FROM 'ACTIVE'
     OR (p_state::jsonb) #>> '{{binding,kind}}' IS DISTINCT FROM 'UNBOUND'
     OR ((p_state::jsonb) ->> 'authoritative_state_revision')::bigint IS DISTINCT FROM 1
     OR (p_state::jsonb) -> 'predecessor_revision' IS DISTINCT FROM 'null'::jsonb THEN
    RAISE EXCEPTION 'invalid genesis' USING ERRCODE = '23514';
  END IF;
  INSERT INTO {s}.lineages VALUES
    (p_handle, p_environment, p_trust, p_product, p_entitlement, 1);
  INSERT INTO {s}.history VALUES
    (p_handle, p_environment, p_trust, 1, NULL, p_state::jsonb, md5(((p_state::jsonb)::text)));
END $fn$;
CREATE FUNCTION {s}.append_bind(p_handle text, p_environment text, p_trust text,
    p_expected bigint, p_state json) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $fn$
DECLARE actual bigint; old_state jsonb;
BEGIN
  PERFORM {s}.validate_state(p_state);
  SELECT current_revision INTO actual FROM {s}.lineages
    WHERE lookup_handle=p_handle AND environment=p_environment AND trust_domain=p_trust
    FOR UPDATE;
  IF actual IS NULL OR actual <> p_expected THEN RAISE EXCEPTION 'cas conflict' USING ERRCODE='23514'; END IF;
  SELECT state INTO old_state FROM {s}.history WHERE lookup_handle=p_handle
    AND environment=p_environment AND trust_domain=p_trust AND revision=actual;
  IF old_state IS NULL THEN RAISE EXCEPTION 'missing current authority state' USING ERRCODE='22000'; END IF;
  PERFORM {s}.validate_state(old_state::json);
  IF old_state ->> 'lifecycle' IS DISTINCT FROM 'ACTIVE'
     OR old_state #>> '{{binding,kind}}' IS DISTINCT FROM 'UNBOUND'
     OR (p_state::jsonb) ->> 'lifecycle' IS DISTINCT FROM 'ACTIVE'
     OR (p_state::jsonb) #>> '{{binding,kind}}' IS DISTINCT FROM 'BOUND'
     OR (p_state::jsonb) -> 'subject' IS DISTINCT FROM old_state -> 'subject'
     OR (p_state::jsonb) -> 'identity' IS DISTINCT FROM old_state -> 'identity'
     OR (p_state::jsonb) -> 'provenance' IS DISTINCT FROM old_state -> 'provenance'
     OR ((p_state::jsonb) ->> 'authoritative_state_revision')::bigint IS DISTINCT FROM actual + 1
     OR ((p_state::jsonb) ->> 'predecessor_revision')::bigint IS DISTINCT FROM actual
     OR EXISTS (SELECT 1 FROM {s}.history h WHERE h.lookup_handle=p_handle
       AND h.environment=p_environment AND h.trust_domain=p_trust
       AND h.state #>> '{{binding,kind}}' = 'BOUND') THEN
    RAISE EXCEPTION 'invalid bind transition' USING ERRCODE='23514';
  END IF;
  INSERT INTO {s}.history VALUES (p_handle,p_environment,p_trust,actual+1,actual,p_state::jsonb,md5(((p_state::jsonb)::text)));
  UPDATE {s}.lineages SET current_revision=actual+1 WHERE lookup_handle=p_handle
    AND environment=p_environment AND trust_domain=p_trust;
END $fn$;
CREATE FUNCTION {s}.revoke(p_handle text, p_environment text, p_trust text,
    p_expected bigint, p_state json) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $fn$
DECLARE actual bigint; old_state jsonb;
BEGIN
  PERFORM {s}.validate_state(p_state);
  SELECT current_revision INTO actual FROM {s}.lineages WHERE lookup_handle=p_handle
    AND environment=p_environment AND trust_domain=p_trust FOR UPDATE;
  IF actual IS NULL OR actual <> p_expected THEN RAISE EXCEPTION 'cas conflict' USING ERRCODE='23514'; END IF;
  SELECT state INTO old_state FROM {s}.history WHERE lookup_handle=p_handle
    AND environment=p_environment AND trust_domain=p_trust AND revision=actual;
  IF old_state IS NULL THEN RAISE EXCEPTION 'missing current authority state' USING ERRCODE='22000'; END IF;
  PERFORM {s}.validate_state(old_state::json);
  IF old_state ->> 'lifecycle' IS DISTINCT FROM 'ACTIVE'
     OR (p_state::jsonb) ->> 'lifecycle' IS DISTINCT FROM 'REVOKED'
     OR (p_state::jsonb) -> 'subject' IS DISTINCT FROM old_state -> 'subject'
     OR (p_state::jsonb) -> 'binding' IS DISTINCT FROM old_state -> 'binding'
     OR (p_state::jsonb) -> 'identity' IS DISTINCT FROM old_state -> 'identity'
     OR (p_state::jsonb) -> 'provenance' IS DISTINCT FROM old_state -> 'provenance'
     OR ((p_state::jsonb) ->> 'authoritative_state_revision')::bigint IS DISTINCT FROM actual+1
     OR ((p_state::jsonb) ->> 'predecessor_revision')::bigint IS DISTINCT FROM actual THEN
    RAISE EXCEPTION 'invalid revoke transition' USING ERRCODE='23514';
  END IF;
  INSERT INTO {s}.history VALUES (p_handle,p_environment,p_trust,actual+1,actual,p_state::jsonb,md5(((p_state::jsonb)::text)));
  UPDATE {s}.lineages SET current_revision=actual+1 WHERE lookup_handle=p_handle
    AND environment=p_environment AND trust_domain=p_trust;
END $fn$;
CREATE FUNCTION {s}.supersede(p_handle text, p_environment text, p_trust text,
    p_expected bigint, p_retired json, p_successor json) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $fn$
DECLARE actual bigint; old_state jsonb;
BEGIN
  PERFORM {s}.validate_state(p_retired);
  PERFORM {s}.validate_state(p_successor);
  SELECT current_revision INTO actual FROM {s}.lineages WHERE lookup_handle=p_handle
    AND environment=p_environment AND trust_domain=p_trust FOR UPDATE;
  IF actual IS NULL OR actual <> p_expected THEN RAISE EXCEPTION 'cas conflict' USING ERRCODE='23514'; END IF;
  SELECT state INTO old_state FROM {s}.history WHERE lookup_handle=p_handle
    AND environment=p_environment AND trust_domain=p_trust AND revision=actual;
  IF old_state IS NULL THEN RAISE EXCEPTION 'missing current authority state' USING ERRCODE='22000'; END IF;
  PERFORM {s}.validate_state(old_state::json);
  IF old_state ->> 'lifecycle' IS DISTINCT FROM 'ACTIVE'
     OR (p_retired::jsonb) ->> 'lifecycle' IS DISTINCT FROM 'SUPERSEDED'
     OR (p_retired::jsonb) -> 'subject' IS DISTINCT FROM old_state -> 'subject'
     OR (p_retired::jsonb) -> 'binding' IS DISTINCT FROM old_state -> 'binding'
     OR (p_retired::jsonb) -> 'identity' IS DISTINCT FROM old_state -> 'identity'
     OR (p_retired::jsonb) -> 'provenance' IS DISTINCT FROM old_state -> 'provenance'
     OR (p_successor::jsonb) ->> 'lifecycle' IS DISTINCT FROM 'ACTIVE'
     OR (p_successor::jsonb) #>> '{{binding,kind}}' IS DISTINCT FROM 'UNBOUND'
     OR (p_successor::jsonb) -> 'subject' IS DISTINCT FROM old_state -> 'subject'
     OR (p_successor::jsonb) #>> '{{identity,bootstrap_entitlement_id}}' IS DISTINCT FROM old_state #>> '{{identity,bootstrap_entitlement_id}}'
     OR (p_successor::jsonb) #>> '{{identity,environment}}' IS DISTINCT FROM old_state #>> '{{identity,environment}}'
     OR (p_successor::jsonb) #>> '{{identity,trust_domain}}' IS DISTINCT FROM old_state #>> '{{identity,trust_domain}}'
     OR (p_successor::jsonb) #>> '{{identity,product_scope}}' IS DISTINCT FROM old_state #>> '{{identity,product_scope}}'
     OR (p_successor::jsonb) #>> '{{identity,intended_action}}' IS DISTINCT FROM old_state #>> '{{identity,intended_action}}'
     OR ((p_retired::jsonb) ->> 'authoritative_state_revision')::bigint IS DISTINCT FROM actual+1
     OR ((p_retired::jsonb) ->> 'predecessor_revision')::bigint IS DISTINCT FROM actual
     OR ((p_successor::jsonb) ->> 'authoritative_state_revision')::bigint IS DISTINCT FROM actual+2
     OR ((p_successor::jsonb) ->> 'predecessor_revision')::bigint IS DISTINCT FROM actual+1
     OR ((p_successor::jsonb) #>> '{{identity,entitlement_generation}}')::bigint
        IS DISTINCT FROM (old_state #>> '{{identity,entitlement_generation}}')::bigint + 1 THEN
    RAISE EXCEPTION 'invalid supersession' USING ERRCODE='23514';
  END IF;
  INSERT INTO {s}.history VALUES (p_handle,p_environment,p_trust,actual+1,actual,p_retired::jsonb,md5(((p_retired::jsonb)::text)));
  INSERT INTO {s}.history VALUES (p_handle,p_environment,p_trust,actual+2,actual+1,p_successor::jsonb,md5(((p_successor::jsonb)::text)));
  UPDATE {s}.lineages SET current_revision=actual+2 WHERE lookup_handle=p_handle
    AND environment=p_environment AND trust_domain=p_trust;
END $fn$;
REVOKE ALL ON SCHEMA {s} FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA {s} FROM PUBLIC;
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA {s} FROM PUBLIC;
GRANT USAGE ON SCHEMA {s} TO {runtime}, {admin};
GRANT SELECT ON {s}.metadata, {s}.lineages, {s}.history TO {runtime}, {admin};
GRANT EXECUTE ON FUNCTION {s}.append_bind(text,text,text,bigint,json) TO {runtime};
GRANT EXECUTE ON FUNCTION {s}.provision(text,text,text,text,text,json) TO {admin};
GRANT EXECUTE ON FUNCTION {s}.revoke(text,text,text,bigint,json) TO {admin};
GRANT EXECUTE ON FUNCTION {s}.supersede(text,text,text,bigint,json,json) TO {admin};
RESET ROLE;
"""
    ).format(
        s=s,
        owner=owner,
        runtime=runtime,
        admin=admin,
        schema_identity=sql.Literal(SCHEMA_IDENTITY),
        schema_version=sql.Literal(SCHEMA_VERSION),
        profile=sql.Literal(PROFILE),
        trust_domain=sql.Literal(config.trust_domain),
        storage_family=sql.Literal(STORAGE_FAMILY),
        owner_name=sql.Literal(config.schema_owner_role),
        runtime_name=sql.Literal(config.runtime_role),
        admin_name=sql.Literal(config.admin_role),
    )


def _reviewed_function_sources(
    config: PostgreSQLRegistryProvisioning,
) -> dict[str, str]:
    """Derive executable bodies from the reviewed source, never from the database."""

    rendered = _provisioning_sql(config).as_string()
    matches = re.findall(
        r"CREATE FUNCTION\s+[^.]+\.(\w+)\([^)]*\).*?AS \$fn\$(.*?)\$fn\$;",
        rendered,
        flags=re.DOTALL,
    )
    by_name = {name: source for name, source in matches}
    expected_names = {signature.partition("(")[0] for signature in _FUNCTION_SIGNATURES}
    if set(by_name) != expected_names or len(matches) != len(expected_names):
        raise RegistryQualificationError("reviewed function source template is ambiguous")
    return {signature: by_name[signature.partition("(")[0]] for signature in _FUNCTION_SIGNATURES}


def _function_source_fingerprint(signature: str, source: str) -> str:
    if signature not in _FUNCTION_SIGNATURES or type(source) is not str or not source:
        raise RegistryQualificationError("function source is unavailable")
    return hashlib.sha256(
        _FUNCTION_DEFINITION_FINGERPRINT_DOMAIN
        + signature.encode("ascii")
        + b"\x00"
        + source.encode("utf-8")
    ).hexdigest()


def _reviewed_function_manifest(
    config: PostgreSQLRegistryProvisioning,
) -> dict[str, str]:
    return {
        signature: _function_source_fingerprint(signature, source)
        for signature, source in _reviewed_function_sources(config).items()
    }


def provision_postgresql_entitlement_registry(
    bootstrap: PostgreSQLConnectionConfig,
    config: PostgreSQLRegistryProvisioning,
) -> None:
    """Create roles and the reviewed schema; must be run with bootstrap authority."""

    statements = sql.SQL(
        "CREATE ROLE {owner} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;"
        "CREATE ROLE {runtime} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;"
        "CREATE ROLE {admin} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;"
    ).format(
        owner=sql.Identifier(config.schema_owner_role),
        runtime=sql.Identifier(config.runtime_role),
        admin=sql.Identifier(config.admin_role),
    )
    with psycopg.connect(bootstrap.dsn, autocommit=True) as conn:
        with conn.transaction():
            conn.execute(statements)
            conn.execute(_provisioning_sql(config))
            expected_sources = _reviewed_function_sources(config)
            manifest = _reviewed_function_manifest(config)
            for signature in _FUNCTION_SIGNATURES:
                qualified = f"{config.schema}.{signature}"
                row = conn.execute(
                    "SELECT prosrc FROM pg_catalog.pg_proc WHERE oid=to_regprocedure(%s)",
                    (qualified,),
                ).fetchone()
                if row != (expected_sources[signature],):
                    raise RegistryQualificationError(
                        "provisioned function differs from reviewed code"
                    )
            conn.execute(
                sql.SQL("UPDATE {}.metadata SET function_definition_sha256=%s").format(
                    sql.Identifier(config.schema)
                ),
                (psycopg.types.json.Jsonb(manifest),),
            )


class _PostgreSQLRegistryBase:
    _required_functions: tuple[str, ...] = ()
    _expected_role_field = ""

    def __init__(
        self,
        connection: PostgreSQLConnectionConfig,
        *,
        schema: str,
        environment: str,
        trust_domain: str,
    ) -> None:
        if type(schema) is not str or _SAFE_IDENTIFIER.fullmatch(schema) is None:
            raise ValueError("schema must be a safe lowercase PostgreSQL identifier")
        if type(environment) is not str or not environment.strip():
            raise ValueError("environment must be an exact non-empty str")
        if type(trust_domain) is not str or not trust_domain.strip():
            raise ValueError("trust_domain must be an exact non-empty str")
        self._connection = connection
        self._schema = schema
        self._environment = environment
        self._trust_domain = trust_domain
        self._qualify()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(connection=<redacted>, schema={self._schema!r}, "
            f"environment={self._environment!r}, trust_domain={self._trust_domain!r})"
        )

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(self._connection.dsn, autocommit=True)

    def _qualify(self) -> None:
        try:
            with self._connect() as conn, conn.transaction():
                metadata = conn.execute(
                    sql.SQL(
                        "SELECT schema_identity,schema_version,profile,trust_domain,"
                        "storage_family,schema_owner_role,schema_owner_role_oid,"
                        "runtime_role,runtime_role_oid,admin_role,admin_role_oid,"
                        "function_definition_sha256 FROM {}.metadata"
                    ).format(sql.Identifier(self._schema))
                ).fetchone()
                if metadata is None or len(metadata) != 12:
                    raise RegistryQualificationError("registry metadata is missing")
                if metadata[:5] != (
                    SCHEMA_IDENTITY,
                    SCHEMA_VERSION,
                    PROFILE,
                    self._trust_domain,
                    STORAGE_FAMILY,
                ):
                    raise RegistryQualificationError("registry metadata mismatch")
                roles = {
                    "schema_owner_role": (metadata[5], metadata[6]),
                    "runtime_role": (metadata[7], metadata[8]),
                    "admin_role": (metadata[9], metadata[10]),
                }
                manifest = metadata[11]
                if type(manifest) is not dict or set(manifest) != set(_FUNCTION_SIGNATURES):
                    raise RegistryQualificationError("function manifest mismatch")
                self._qualify_role(conn, roles)
                durability = conn.execute(
                    "SELECT current_setting('fsync'), current_setting('synchronous_commit')"
                ).fetchone()
                if durability != ("on", "on"):
                    raise RegistryQualificationError("registry durability mismatch")
                if conn.info.server_version < 160000:
                    raise RegistryQualificationError("PostgreSQL 16 or newer is required")
                for relation in ("metadata", "lineages", "history"):
                    exists = conn.execute(
                        "SELECT to_regclass(%s)", (f"{self._schema}.{relation}",)
                    ).fetchone()
                    if exists is None or exists[0] is None:
                        raise RegistryQualificationError("required registry relation is missing")
                self._qualify_relations(conn, roles)
                self._qualify_columns(conn)
                self._qualify_constraints(conn)
                self._qualify_relation_acls(conn, roles)
                self._qualify_functions(conn, roles, manifest)
        except RegistryQualificationError:
            raise
        except psycopg.Error as exc:
            raise RegistryQualificationError("PostgreSQL registry qualification failed") from exc

    def _qualify_role(
        self,
        conn: psycopg.Connection[Any],
        roles: dict[str, tuple[object, object]],
    ) -> None:
        if len({roles[name][0] for name in roles}) != 3:
            raise RegistryQualificationError("persisted registry roles are not distinct")
        for role_name, role_oid in roles.values():
            catalog = conn.execute(
                "SELECT oid FROM pg_catalog.pg_roles WHERE rolname=%s", (role_name,)
            ).fetchone()
            if catalog != (role_oid,):
                raise RegistryQualificationError("persisted role name/OID identity mismatch")
        identity = conn.execute(
            "SELECT session_user,current_user,s.oid,c.oid,c.rolcanlogin,c.rolinherit,"
            "c.rolsuper,c.rolcreaterole,c.rolcreatedb,c.rolreplication,c.rolbypassrls "
            "FROM pg_catalog.pg_roles s,pg_catalog.pg_roles c "
            "WHERE s.rolname=session_user AND c.rolname=current_user"
        ).fetchone()
        expected_name, expected_oid = roles[self._expected_role_field]
        if identity is None or identity[:4] != (
            expected_name,
            expected_name,
            expected_oid,
            expected_oid,
        ):
            raise RegistryQualificationError(
                "session/current identity is not the persisted reviewed direct-login role"
            )
        if identity[4:] != (True, False, False, False, False, False, False):
            raise RegistryQualificationError("registry role has forbidden role attributes")
        owner_name, owner_oid = roles["schema_owner_role"]
        namespace_owner = conn.execute(
            "SELECT nspowner FROM pg_catalog.pg_namespace WHERE nspname=%s",
            (self._schema,),
        ).fetchone()
        if namespace_owner != (owner_oid,) or identity[1] == owner_name:
            raise RegistryQualificationError("registry schema owner identity mismatch")
        memberships = conn.execute(
            "SELECT roleid,member,admin_option,inherit_option,set_option "
            "FROM pg_catalog.pg_auth_members "
            "WHERE roleid IN (%s,%s,%s) OR member IN (%s,%s,%s)",
            (
                roles["schema_owner_role"][1],
                roles["runtime_role"][1],
                roles["admin_role"][1],
                roles["schema_owner_role"][1],
                roles["runtime_role"][1],
                roles["admin_role"][1],
            ),
        ).fetchall()
        if memberships:
            raise RegistryQualificationError("registry authority role has a membership grant")
        schema_privileges = conn.execute(
            "SELECT has_schema_privilege(current_user,%s,'USAGE'), "
            "has_schema_privilege(current_user,%s,'CREATE')",
            (self._schema, self._schema),
        ).fetchone()
        if schema_privileges != (True, False):
            raise RegistryQualificationError("registry schema privileges mismatch")
        forbidden = "INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER"
        for relation in ("metadata", "lineages", "history"):
            qualified = f"{self._schema}.{relation}"
            privileges = conn.execute(
                "SELECT has_table_privilege(current_user,%s,'SELECT'), "
                "has_table_privilege(current_user,%s,%s)",
                (qualified, qualified, forbidden),
            ).fetchone()
            if privileges != (True, False):
                raise RegistryQualificationError("registry table privileges mismatch")

    def _qualify_relations(
        self,
        conn: psycopg.Connection[Any],
        roles: dict[str, tuple[object, object]],
    ) -> None:
        owner_oid = roles["schema_owner_role"][1]
        reviewed = conn.execute(
            "SELECT c.oid,c.relname,c.relkind,c.relpersistence,c.relispartition,"
            "c.relrowsecurity,c.relforcerowsecurity,c.relowner "
            "FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n "
            "ON n.oid=c.relnamespace WHERE n.nspname=%s "
            "AND c.relkind IN ('r','p','v','m','f') ORDER BY c.relname",
            (self._schema,),
        ).fetchall()
        if [(row[1], *row[2:]) for row in reviewed] != [
            ("history", "r", "p", False, False, False, owner_oid),
            ("lineages", "r", "p", False, False, False, owner_oid),
            ("metadata", "r", "p", False, False, False, owner_oid),
        ]:
            raise RegistryQualificationError("registry relation identity mismatch")
        relation_oids = tuple(row[0] for row in reviewed)
        if len(relation_oids) != 3:
            raise RegistryQualificationError("reviewed registry relations are missing")
        inheritance = conn.execute(
            "SELECT 1 FROM pg_catalog.pg_inherits "
            "WHERE inhrelid IN (%s,%s,%s) OR inhparent IN (%s,%s,%s) LIMIT 1",
            (*relation_oids, *relation_oids),
        ).fetchone()
        if inheritance is not None:
            raise RegistryQualificationError("registry inheritance/partition edge exists")
        policies = conn.execute(
            "SELECT 1 FROM pg_catalog.pg_policy WHERE polrelid IN (%s,%s,%s) LIMIT 1",
            relation_oids,
        ).fetchone()
        if policies is not None:
            raise RegistryQualificationError("registry row-security policy exists")
        user_trigger = conn.execute(
            "SELECT 1 FROM pg_catalog.pg_trigger WHERE tgrelid IN (%s,%s,%s) "
            "AND NOT tgisinternal LIMIT 1",
            relation_oids,
        ).fetchone()
        if user_trigger is not None:
            raise RegistryQualificationError("registry user trigger exists")
        rewrite_rule = conn.execute(
            "SELECT 1 FROM pg_catalog.pg_rewrite WHERE ev_class IN (%s,%s,%s) LIMIT 1",
            relation_oids,
        ).fetchone()
        if rewrite_rule is not None:
            raise RegistryQualificationError("registry rewrite rule exists")

    def _qualify_relation_acls(
        self,
        conn: psycopg.Connection[Any],
        roles: dict[str, tuple[object, object]],
    ) -> None:
        owner_oid = roles["schema_owner_role"][1]
        runtime_oid = roles["runtime_role"][1]
        admin_oid = roles["admin_role"][1]
        schema_acl = set(
            conn.execute(
                "SELECT a.grantee,a.privilege_type,a.is_grantable "
                "FROM pg_catalog.pg_namespace n "
                "CROSS JOIN LATERAL aclexplode(n.nspacl) a WHERE n.nspname=%s",
                (self._schema,),
            ).fetchall()
        )
        if schema_acl != {
            (owner_oid, "CREATE", False),
            (owner_oid, "USAGE", False),
            (runtime_oid, "USAGE", False),
            (admin_oid, "USAGE", False),
        }:
            raise RegistryQualificationError("registry schema ACL differs from allowlist")
        owner_table_privileges = {
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "TRUNCATE",
            "REFERENCES",
            "TRIGGER",
        }
        expected_table_acl = {
            (relation, owner_oid, privilege, False)
            for relation in ("metadata", "lineages", "history")
            for privilege in owner_table_privileges
        } | {
            (relation, grantee, "SELECT", False)
            for relation in ("metadata", "lineages", "history")
            for grantee in (runtime_oid, admin_oid)
        }
        table_acl = set(
            conn.execute(
                "SELECT c.relname,a.grantee,a.privilege_type,a.is_grantable "
                "FROM pg_catalog.pg_class c "
                "JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
                "CROSS JOIN LATERAL aclexplode(c.relacl) a "
                "WHERE n.nspname=%s AND c.relkind IN ('r','p')",
                (self._schema,),
            ).fetchall()
        )
        if table_acl != expected_table_acl:
            raise RegistryQualificationError("registry table ACL differs from allowlist")
        column_acls = conn.execute(
            "SELECT c.relname,a.attname,a.attacl FROM pg_catalog.pg_attribute a "
            "JOIN pg_catalog.pg_class c ON c.oid=a.attrelid "
            "JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=%s AND c.relkind IN ('r','p') AND a.attnum>0 "
            "AND NOT a.attisdropped AND a.attacl IS NOT NULL",
            (self._schema,),
        ).fetchall()
        if column_acls:
            raise RegistryQualificationError("registry column ACL must be empty")
        for role_name, _role_oid in (roles["runtime_role"], roles["admin_role"]):
            for relation in ("metadata", "lineages", "history"):
                any_forbidden = conn.execute(
                    "SELECT has_any_column_privilege(%s,%s,'INSERT,UPDATE,REFERENCES')",
                    (role_name, f"{self._schema}.{relation}"),
                ).fetchone()
                if any_forbidden != (False,):
                    raise RegistryQualificationError("registry role has forbidden column privilege")

    def _qualify_constraints(self, conn: psycopg.Connection[Any]) -> None:
        rows = conn.execute(
            "SELECT c.conrelid::regclass::text,c.contype,pg_get_constraintdef(c.oid),"
            "c.condeferrable,c.condeferred,c.confupdtype,c.confdeltype,c.confmatchtype "
            "FROM pg_catalog.pg_constraint c WHERE c.connamespace="
            "(SELECT oid FROM pg_catalog.pg_namespace WHERE nspname=%s)",
            (self._schema,),
        ).fetchall()
        observed = set(rows)
        expected = {
            (
                f"{self._schema}.metadata",
                "p",
                "PRIMARY KEY (singleton)",
                False,
                False,
                " ",
                " ",
                " ",
            ),
            (f"{self._schema}.metadata", "c", "CHECK (singleton)", False, False, " ", " ", " "),
            (
                f"{self._schema}.lineages",
                "p",
                "PRIMARY KEY (lookup_handle, environment, trust_domain)",
                False,
                False,
                " ",
                " ",
                " ",
            ),
            (
                f"{self._schema}.lineages",
                "u",
                "UNIQUE (environment, trust_domain, product_scope, bootstrap_entitlement_id)",
                False,
                False,
                " ",
                " ",
                " ",
            ),
            (
                f"{self._schema}.lineages",
                "c",
                "CHECK ((current_revision >= 1))",
                False,
                False,
                " ",
                " ",
                " ",
            ),
            (
                f"{self._schema}.history",
                "p",
                "PRIMARY KEY (lookup_handle, environment, trust_domain, revision)",
                False,
                False,
                " ",
                " ",
                " ",
            ),
            (
                f"{self._schema}.history",
                "c",
                "CHECK ((revision >= 1))",
                False,
                False,
                " ",
                " ",
                " ",
            ),
            (
                f"{self._schema}.history",
                "f",
                f"FOREIGN KEY (lookup_handle, environment, trust_domain) REFERENCES {self._schema}.lineages(lookup_handle, environment, trust_domain) DEFERRABLE INITIALLY DEFERRED",
                True,
                True,
                "a",
                "a",
                "s",
            ),
        }
        if observed != expected:
            raise RegistryQualificationError("registry constraints differ from reviewed schema")

    def _qualify_columns(self, conn: psycopg.Connection[Any]) -> None:
        observed = conn.execute(
            "SELECT c.relname,a.attnum,a.attname,pg_catalog.format_type(a.atttypid,a.atttypmod),"
            "a.attnotnull,pg_catalog.pg_get_expr(d.adbin,d.adrelid),a.attgenerated,a.attidentity "
            "FROM pg_catalog.pg_attribute a JOIN pg_catalog.pg_class c ON c.oid=a.attrelid "
            "JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
            "LEFT JOIN pg_catalog.pg_attrdef d ON d.adrelid=a.attrelid AND d.adnum=a.attnum "
            "WHERE n.nspname=%s AND c.relkind IN ('r','p') AND a.attnum>0 "
            "AND NOT a.attisdropped ORDER BY c.relname,a.attnum",
            (self._schema,),
        ).fetchall()
        definitions = {
            "metadata": (
                ("singleton", "boolean", True, "true"),
                ("schema_identity", "text", True, None),
                ("schema_version", "integer", True, None),
                ("profile", "text", True, None),
                ("trust_domain", "text", True, None),
                ("storage_family", "text", True, None),
                ("schema_owner_role", "text", True, None),
                ("schema_owner_role_oid", "oid", True, None),
                ("runtime_role", "text", True, None),
                ("runtime_role_oid", "oid", True, None),
                ("admin_role", "text", True, None),
                ("admin_role_oid", "oid", True, None),
                ("function_definition_sha256", "jsonb", True, None),
            ),
            "lineages": (
                ("lookup_handle", "text", True, None),
                ("environment", "text", True, None),
                ("trust_domain", "text", True, None),
                ("product_scope", "text", True, None),
                ("bootstrap_entitlement_id", "text", True, None),
                ("current_revision", "bigint", True, None),
            ),
            "history": (
                ("lookup_handle", "text", True, None),
                ("environment", "text", True, None),
                ("trust_domain", "text", True, None),
                ("revision", "bigint", True, None),
                ("predecessor_revision", "bigint", False, None),
                ("state", "jsonb", True, None),
                ("state_integrity", "text", True, None),
            ),
        }
        expected = [
            (relation, position, name, data_type, not_null, default, "", "")
            for relation in sorted(definitions)
            for position, (name, data_type, not_null, default) in enumerate(
                definitions[relation], 1
            )
        ]
        if observed != expected:
            raise RegistryQualificationError("registry physical column shape mismatch")

    def _qualify_functions(
        self,
        conn: psycopg.Connection[Any],
        roles: dict[str, tuple[object, object]],
        manifest: dict[str, object],
    ) -> None:
        owner_oid = roles["schema_owner_role"][1]
        runtime_name = roles["runtime_role"][0]
        admin_name = roles["admin_role"][0]
        reviewed_config = PostgreSQLRegistryProvisioning(
            schema=self._schema,
            schema_owner_role=str(roles["schema_owner_role"][0]),
            runtime_role=str(runtime_name),
            admin_role=str(admin_name),
            trust_domain=self._trust_domain,
        )
        expected_sources = _reviewed_function_sources(reviewed_config)
        expected_manifest = _reviewed_function_manifest(reviewed_config)
        if manifest != expected_manifest:
            raise RegistryQualificationError(
                "persisted function manifest differs from reviewed code"
            )
        observed_signatures = conn.execute(
            "SELECT p.oid::regprocedure::text FROM pg_catalog.pg_proc p "
            "JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace "
            "WHERE n.nspname=%s",
            (self._schema,),
        ).fetchall()
        if {item[0] for item in observed_signatures} != {
            f"{self._schema}.{signature}" for signature in _FUNCTION_SIGNATURES
        }:
            raise RegistryQualificationError("registry function signatures mismatch")
        for signature in _FUNCTION_SIGNATURES:
            qualified = f"{self._schema}.{signature}"
            row = conn.execute(
                "SELECT p.oid,p.proowner,p.prosecdef,p.proconfig,p.prosrc,"
                "p.prorettype::regtype::text,l.lanname,p.provolatile,p.proisstrict,"
                "p.proleakproof,p.proparallel,p.prokind,p.proargmodes "
                "FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_language l "
                "ON l.oid=p.prolang WHERE p.oid=to_regprocedure(%s)",
                (qualified,),
            ).fetchone()
            if row is None:
                raise RegistryQualificationError("required registry function is missing")
            (
                function_oid,
                function_owner,
                security_definer,
                configuration,
                definition,
                return_type,
                language,
                volatility,
                strict,
                leakproof,
                parallel,
                function_kind,
                argument_modes,
            ) = row
            if (
                function_owner != owner_oid
                or security_definer is not True
                or configuration != ["search_path=pg_catalog"]
                or definition != expected_sources[signature]
                or _function_source_fingerprint(signature, definition)
                != expected_manifest[signature]
                or (return_type, language, volatility, strict, leakproof, parallel)
                != ("void", "plpgsql", "v", False, False, "u")
                or function_kind != "f"
                or argument_modes is not None
            ):
                raise RegistryQualificationError("registry function security metadata mismatch")
            expected_grantee_oid = (
                roles["runtime_role"][1]
                if signature.startswith("append_bind(")
                else (None if signature.startswith("validate_state(") else roles["admin_role"][1])
            )
            function_acl = set(
                conn.execute(
                    "SELECT a.grantee,a.privilege_type,a.is_grantable "
                    "FROM pg_catalog.pg_proc p "
                    "CROSS JOIN LATERAL aclexplode(p.proacl) a WHERE p.oid=%s",
                    (function_oid,),
                ).fetchall()
            )
            expected_function_acl = {(owner_oid, "EXECUTE", False)}
            if expected_grantee_oid is not None:
                expected_function_acl.add((expected_grantee_oid, "EXECUTE", False))
            if function_acl != expected_function_acl:
                raise RegistryQualificationError("registry function ACL differs from allowlist")
            runtime_execute = conn.execute(
                "SELECT has_function_privilege(%s,%s,'EXECUTE')",
                (runtime_name, function_oid),
            ).fetchone()
            admin_execute = conn.execute(
                "SELECT has_function_privilege(%s,%s,'EXECUTE')",
                (admin_name, function_oid),
            ).fetchone()
            expected_runtime = signature.startswith("append_bind(")
            expected_admin = not expected_runtime and not signature.startswith("validate_state(")
            if runtime_execute != (expected_runtime,) or admin_execute != (expected_admin,):
                raise RegistryQualificationError("registry function EXECUTE matrix mismatch")

    def _trusted_subject(self, value: object) -> RegistrySubject:
        trusted = validate_exact_snapshot(value)
        if not isinstance(trusted, RegistrySubject):
            raise ContractValidationError("subject must be exact RegistrySubject")
        if trusted.environment != self._environment or trusted.trust_domain != self._trust_domain:
            raise ContractValidationError("subject does not match provider security scope")
        return trusted

    def _load_history(
        self, conn: psycopg.Connection[Any], subject: RegistrySubject
    ) -> tuple[AuthoritativeEntitlementState, ...] | None:
        table = sql.Identifier(self._schema, "history")
        rows = conn.execute(
            sql.SQL(
                "SELECT lookup_handle,environment,trust_domain,revision,"
                "predecessor_revision,state,state_integrity=md5(state::text) FROM {} "
                "WHERE lookup_handle=%s AND environment=%s AND trust_domain=%s "
                "ORDER BY revision"
            ).format(table),
            (subject.lookup_handle, subject.environment, subject.trust_domain),
        ).fetchall()
        if not rows:
            return None
        states: list[AuthoritativeEntitlementState] = []
        for expected_revision, row in enumerate(rows, 1):
            handle, environment, trust_domain, revision, predecessor, payload, checksum_ok = row
            if checksum_ok is not True:
                raise ContractValidationError("local stored-state corruption checksum differs")
            state = _state_from_json(payload)
            expected_predecessor = None if expected_revision == 1 else expected_revision - 1
            if (
                (handle, environment, trust_domain)
                != (
                    state.subject.lookup_handle,
                    state.subject.environment,
                    state.subject.trust_domain,
                )
                or revision != expected_revision
                or revision != state.authoritative_state_revision
                or predecessor != expected_predecessor
                or predecessor != state.predecessor_revision
            ):
                raise ContractValidationError("physical history envelope contradicts state")
            states.append(state)
        return tuple(states)

    def _validated_history(
        self, conn: psycopg.Connection[Any], subject: RegistrySubject
    ) -> RetainedHistoryResult:
        states = self._load_history(conn, subject)
        lineage = sql.Identifier(self._schema, "lineages")
        head = conn.execute(
            sql.SQL(
                "SELECT current_revision,product_scope,bootstrap_entitlement_id FROM {} WHERE lookup_handle=%s AND environment=%s AND trust_domain=%s"
            ).format(lineage),
            (subject.lookup_handle, subject.environment, subject.trust_domain),
        ).fetchone()
        if states is None and head is None:
            return RetainedHistoryResult(RegistryReadOutcome.NOT_FOUND, subject, (), None, None)
        if states is None or head is None:
            raise ContractValidationError("lineage head/history mismatch")
        validate_complete_lineage(subject, states)
        tail = states[-1]
        if head[0] != tail.authoritative_state_revision or head[0] != len(states):
            raise ContractValidationError("current head does not equal retained history tail")
        if (
            head[1] != tail.identity.product_scope
            or head[2] != tail.identity.bootstrap_entitlement_id
        ):
            raise ContractValidationError("reverse identity projection differs from history")
        return RetainedHistoryResult(RegistryReadOutcome.FOUND, subject, states, head[0], 1)

    def _read(
        self,
        subject: RegistrySubject,
        operation: Callable[[psycopg.Connection[Any], RegistrySubject], _T],
        unavailable: Callable[[], _T],
        corrupt: Callable[[], _T],
    ) -> _T:
        try:
            with self._connect() as conn, conn.transaction():
                conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                return operation(conn, subject)
        except (ContractValidationError, ValueError, TypeError):
            return corrupt()
        except psycopg.Error:
            return unavailable()

    def _serializable(
        self,
        callback: Callable[[psycopg.Connection[Any]], _T],
        retryable: Callable[[], _T],
        unavailable: Callable[[], _T],
        conflict: Callable[[], _T],
        corrupt: Callable[[], _T],
    ) -> _T:
        try:
            with self._connect() as conn, conn.transaction():
                conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
                return callback(conn)
        except SerializationFailure:
            return retryable()
        except (UniqueViolation, CheckViolation):
            return conflict()
        except (ContractValidationError, ValueError, TypeError):
            return corrupt()
        except psycopg.Error:
            return unavailable()


class PostgreSQLEntitlementRegistryProvider(_PostgreSQLRegistryBase):
    """Least-privilege runtime authority surface backed only by PostgreSQL."""

    _required_functions = ("append_bind(text,text,text,bigint,json)",)
    _expected_role_field = "runtime_role"

    @property
    def identity(self) -> ProviderIdentity:
        return ProviderIdentity(
            ProviderRole.ENTITLEMENT_REGISTRY,
            SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, self._trust_domain),
            f"postgresql:{self._schema}",
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            True, authoritative_reads=True, durable_state=True, compare_and_swap=True
        )

    def credential_identities(self) -> tuple[()]:
        return ()

    def retained_history(self, subject: RegistrySubject) -> RetainedHistoryResult:
        trusted = self._trusted_subject(subject)
        return self._read(
            trusted,
            self._validated_history,
            lambda: RetainedHistoryResult(RegistryReadOutcome.UNAVAILABLE, trusted, (), None, None),
            lambda: RetainedHistoryResult(RegistryReadOutcome.CORRUPT, trusted, (), None, None),
        )

    def authoritative_state(self, subject: RegistrySubject) -> RegistryReadResult:
        trusted = self._trusted_subject(subject)

        def operation(
            conn: psycopg.Connection[Any], requested: RegistrySubject
        ) -> RegistryReadResult:
            history = self._validated_history(conn, requested)
            if history.outcome is RegistryReadOutcome.NOT_FOUND:
                return RegistryReadResult(RegistryReadOutcome.NOT_FOUND, None)
            return RegistryReadResult(RegistryReadOutcome.FOUND, history.states[-1])

        return self._read(
            trusted,
            operation,
            lambda: RegistryReadResult(RegistryReadOutcome.UNAVAILABLE, None),
            lambda: RegistryReadResult(RegistryReadOutcome.CORRUPT, None),
        )

    def state_at_revision(
        self, subject: RegistrySubject, authoritative_state_revision: int
    ) -> HistoricalStateResult:
        trusted = self._trusted_subject(subject)
        if type(authoritative_state_revision) is not int or authoritative_state_revision < 1:
            raise ContractValidationError("revision must be a positive exact int")
        revision = authoritative_state_revision

        def result(outcome: RegistryReadOutcome) -> HistoricalStateResult:
            return HistoricalStateResult(
                trusted, RegistryReadResult(outcome, None), revision, None, None
            )

        def operation(
            conn: psycopg.Connection[Any], requested: RegistrySubject
        ) -> HistoricalStateResult:
            history = self._validated_history(conn, requested)
            if history.outcome is RegistryReadOutcome.NOT_FOUND:
                return result(RegistryReadOutcome.NOT_FOUND)
            assert history.current_authoritative_state_revision is not None
            if revision > history.current_authoritative_state_revision:
                return result(RegistryReadOutcome.NOT_FOUND)
            state = next(
                (item for item in history.states if item.authoritative_state_revision == revision),
                None,
            )
            if state is None:
                return result(RegistryReadOutcome.CORRUPT)
            return HistoricalStateResult(
                trusted,
                RegistryReadResult(RegistryReadOutcome.FOUND, state),
                revision,
                history.current_authoritative_state_revision,
                1,
            )

        return self._read(
            trusted,
            operation,
            lambda: result(RegistryReadOutcome.UNAVAILABLE),
            lambda: result(RegistryReadOutcome.CORRUPT),
        )

    def compare_and_swap_bind(self, request: BindRequest) -> BindResult:
        trusted = validate_exact_snapshot(request)
        if not isinstance(trusted, BindRequest):
            raise ContractValidationError("request must be exact BindRequest")
        self._trusted_subject(trusted.subject)

        def operation(conn: psycopg.Connection[Any]) -> BindResult:
            history = self._validated_history(conn, trusted.subject)
            resolution = resolve_bind_request(history, trusted)
            outcomes = {
                BindResolutionKind.EXACT_REPLAY: BindOutcome.EXACT_REPLAY,
                BindResolutionKind.CONFLICT_BOUND_TO_DIFFERENT_TUPLE: BindOutcome.CONFLICT_BOUND_TO_DIFFERENT_TUPLE,
                BindResolutionKind.STALE_PREDECESSOR: BindOutcome.STALE_PREDECESSOR,
                BindResolutionKind.NOT_FOUND: BindOutcome.NOT_FOUND,
                BindResolutionKind.INACTIVE_REVOKED: BindOutcome.INACTIVE_REVOKED,
                BindResolutionKind.INACTIVE_SUPERSEDED: BindOutcome.INACTIVE_SUPERSEDED,
            }
            if resolution.kind is not BindResolutionKind.NEW_BIND_ELIGIBLE:
                return BindResult(outcomes[resolution.kind], resolution.historical_bound_state)
            current = history.states[-1]
            committed = AuthoritativeEntitlementState(
                current.subject,
                current.identity,
                current.provenance,
                current.lifecycle,
                trusted.attempted_binding,
                current.authoritative_state_revision + 1,
                current.authoritative_state_revision,
            )
            conn.execute(
                sql.SQL("SELECT {}.append_bind(%s,%s,%s,%s,%s)").format(
                    sql.Identifier(self._schema)
                ),
                (
                    *_subject_params(trusted.subject),
                    current.authoritative_state_revision,
                    psycopg.types.json.Json(_state_json(committed)),
                ),
            )
            return BindResult(BindOutcome.NEW_BIND_COMMITTED, committed)

        return self._serializable(
            operation,
            lambda: BindResult(BindOutcome.RETRYABLE_SERIALIZATION_FAILURE, None),
            lambda: BindResult(BindOutcome.UNAVAILABLE, None),
            lambda: BindResult(BindOutcome.STALE_PREDECESSOR, None),
            lambda: BindResult(BindOutcome.CORRUPT, None),
        )


class PostgreSQLEntitlementProvisioningAdminProvider(_PostgreSQLRegistryBase):
    """Reviewed provisioning/revocation/supersession authority surface."""

    _required_functions = (
        "provision(text,text,text,text,text,json)",
        "revoke(text,text,text,bigint,json)",
        "supersede(text,text,text,bigint,json,json)",
    )
    _expected_role_field = "admin_role"

    def _admin_write(
        self, callback: Callable[[psycopg.Connection[Any]], AdminResult]
    ) -> AdminResult:
        return self._serializable(
            callback,
            lambda: AdminResult(AdminOutcome.RETRYABLE_SERIALIZATION_FAILURE, None),
            lambda: AdminResult(AdminOutcome.UNAVAILABLE, None),
            lambda: AdminResult(AdminOutcome.CONFLICT, None),
            lambda: AdminResult(AdminOutcome.CORRUPT, None),
        )

    def provision_entitlement(self, request: ProvisionEntitlementRequest) -> AdminResult:
        trusted = validate_exact_snapshot(request)
        if not isinstance(trusted, ProvisionEntitlementRequest):
            raise ContractValidationError("request must be exact ProvisionEntitlementRequest")
        self._trusted_subject(trusted.subject)
        state = initial_state_for(trusted)

        def operation(conn: psycopg.Connection[Any]) -> AdminResult:
            conn.execute(
                sql.SQL("SELECT {}.provision(%s,%s,%s,%s,%s,%s)").format(
                    sql.Identifier(self._schema)
                ),
                (
                    *_subject_params(trusted.subject),
                    trusted.identity.product_scope,
                    trusted.identity.bootstrap_entitlement_id,
                    psycopg.types.json.Json(_state_json(state)),
                ),
            )
            return AdminResult(AdminOutcome.COMMITTED, state)

        return self._admin_write(operation)

    def revoke_entitlement(self, request: RevokeEntitlementRequest) -> AdminResult:
        trusted = validate_exact_snapshot(request)
        if not isinstance(trusted, RevokeEntitlementRequest):
            raise ContractValidationError("request must be exact RevokeEntitlementRequest")
        subject = self._trusted_subject(trusted.expected.subject)

        def operation(conn: psycopg.Connection[Any]) -> AdminResult:
            history = self._validated_history(conn, subject)
            if history.outcome is RegistryReadOutcome.NOT_FOUND:
                return AdminResult(AdminOutcome.NOT_FOUND, None)
            current = history.states[-1]
            if (
                admin_predecessor_for(current) != trusted.expected
                or current.lifecycle is not EntitlementLifecycle.ACTIVE
            ):
                return AdminResult(AdminOutcome.CONFLICT, None)
            state = revoked_state_for(current, trusted)
            conn.execute(
                sql.SQL("SELECT {}.revoke(%s,%s,%s,%s,%s)").format(sql.Identifier(self._schema)),
                (
                    *_subject_params(subject),
                    current.authoritative_state_revision,
                    psycopg.types.json.Json(_state_json(state)),
                ),
            )
            return AdminResult(AdminOutcome.COMMITTED, state)

        return self._admin_write(operation)

    def supersede_entitlement(self, request: SupersedeEntitlementRequest) -> AdminResult:
        trusted = validate_exact_snapshot(request)
        if not isinstance(trusted, SupersedeEntitlementRequest):
            raise ContractValidationError("request must be exact SupersedeEntitlementRequest")
        subject = self._trusted_subject(trusted.expected.subject)

        def operation(conn: psycopg.Connection[Any]) -> AdminResult:
            history = self._validated_history(conn, subject)
            if history.outcome is RegistryReadOutcome.NOT_FOUND:
                return AdminResult(AdminOutcome.NOT_FOUND, None)
            current = history.states[-1]
            if admin_predecessor_for(current) != trusted.expected:
                return AdminResult(AdminOutcome.CONFLICT, None)
            retired, successor = supersession_states_for(current, trusted)
            conn.execute(
                sql.SQL("SELECT {}.supersede(%s,%s,%s,%s,%s,%s)").format(
                    sql.Identifier(self._schema)
                ),
                (
                    *_subject_params(subject),
                    current.authoritative_state_revision,
                    psycopg.types.json.Json(_state_json(retired)),
                    psycopg.types.json.Json(_state_json(successor)),
                ),
            )
            return AdminResult(AdminOutcome.COMMITTED, successor)

        return self._admin_write(operation)


def _subject_params(subject: RegistrySubject) -> tuple[str, str, str]:
    return subject.lookup_handle, subject.environment, subject.trust_domain


__all__ = [
    "PostgreSQLConnectionConfig",
    "PostgreSQLEntitlementProvisioningAdminProvider",
    "PostgreSQLEntitlementRegistryProvider",
    "PostgreSQLRegistryProvisioning",
    "RegistryQualificationError",
    "SCHEMA_IDENTITY",
    "SCHEMA_VERSION",
    "STORAGE_FAMILY",
    "provision_postgresql_entitlement_registry",
]
