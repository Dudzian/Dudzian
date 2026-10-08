"""Durable verifier-side pre-account public credentials, never signing custody.

The paired offline installer creates two independent PostgreSQL authorities.
Only their narrowly reviewed admin functions can mutate credentials. Retained
history is append-only through these APIs; a host/schema owner is necessarily
trusted and PostgreSQL alone cannot detect rollback to an earlier valid prefix.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import psycopg
from psycopg import sql
from psycopg.errors import CheckViolation, SerializationFailure, UniqueViolation
from psycopg.types.json import Jsonb

from bot_core.postgresql_entitlement_registry import (
    PostgreSQLConnectionConfig,
    RegistryQualificationError,
)
from bot_core.root_proof_issuer_substrate import (
    _PUBLIC_KEY_FINGERPRINT_DOMAIN,
    CredentialRoleIdentity,
    CredentialSemanticRole,
    ProviderCapabilities,
    ProviderIdentity,
    ProviderRole,
    SecurityProfile,
    SecurityProfileIdentity,
    public_key_material_identity,
)

SCHEMA_VERSION = 1
STORAGE_FAMILY = "POSTGRESQL_PREACCOUNT_APPEND_ONLY_V1"
REQUESTER_SCHEMA_IDENTITY = "cryptohunter.requester_credentials.postgresql"
CLAIMANT_SCHEMA_IDENTITY = "cryptohunter.claimant_identities.postgresql"
REQUESTER_SCHEMA = "cryptohunter_requester_credentials"
CLAIMANT_SCHEMA = "cryptohunter_claimant_identities"
REQUESTER_OWNER_ROLE = "cryptohunter_requester_owner"
REQUESTER_RUNTIME_ROLE = "cryptohunter_requester_runtime"
REQUESTER_ADMIN_ROLE = "cryptohunter_requester_admin"
CLAIMANT_OWNER_ROLE = "cryptohunter_claimant_owner"
CLAIMANT_RUNTIME_ROLE = "cryptohunter_claimant_runtime"
CLAIMANT_ADMIN_ROLE = "cryptohunter_claimant_admin"
REQUESTER_PRINCIPAL = "CryptoHunterAccountAuthority"
REQUESTER_CREDENTIAL_ROLE = "ACCOUNT_GENESIS_ROOT_PROOF_ISSUANCE_REQUESTER_V1"
CLAIMANT_CREDENTIAL_ROLE = "ACCOUNT_GENESIS_ROOT_PROOF_CLAIMANT_V1"
_SAFE_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]{0,62}\Z")
_MAX_INTEGER = 9_007_199_254_740_991
_TABLES = ("metadata", "credentials", "history", "current_credentials")
_FUNCTION_SIGNATURE = "apply(jsonb)"
_OWNER_PRIVILEGES = {"SELECT", "INSERT", "UPDATE", "DELETE", "TRUNCATE", "REFERENCES", "TRIGGER"}


class CredentialLifecycle(str, Enum):
    ACTIVE = "ACTIVE"
    VERIFY_ONLY = "VERIFY_ONLY"
    REVOKED = "REVOKED"


class CredentialConflictError(RuntimeError):
    """The requested mutation is inconsistent with current authority evidence."""


class CredentialResolutionError(RuntimeError):
    """Missing, inactive or corrupt credential evidence fails closed."""


def _text(value: object) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError("credential identity must be exact nonempty text")
    return value


def _positive(value: object) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_INTEGER:
        raise ValueError("credential version/generation must be a positive interoperable integer")
    return value


@dataclass(frozen=True, slots=True)
class PostgreSQLCredentialRegistryProvisioning:
    schema: str
    schema_owner_role: str
    runtime_role: str
    admin_role: str
    environment: str
    trust_domain: str

    def __post_init__(self) -> None:
        for value in (self.schema, self.schema_owner_role, self.runtime_role, self.admin_role):
            if type(value) is not str or _SAFE_IDENTIFIER.fullmatch(value) is None:
                raise ValueError("reviewed identifiers must be safe lowercase PostgreSQL names")
        if len({self.schema_owner_role, self.runtime_role, self.admin_role}) != 3:
            raise ValueError("schema owner, runtime and provisioning roles must be distinct")
        if self.environment != "PRODUCTION" or type(self.environment) is not str:
            raise ValueError("pre-account production authority requires protocol PRODUCTION")
        _text(self.trust_domain)


@dataclass(frozen=True, slots=True)
class PostgreSQLPreaccountRegistryProvisioning:
    requester: PostgreSQLCredentialRegistryProvisioning
    claimant: PostgreSQLCredentialRegistryProvisioning

    def __post_init__(self) -> None:
        if any(
            type(item) is not PostgreSQLCredentialRegistryProvisioning
            for item in (self.requester, self.claimant)
        ):
            raise TypeError("exact paired registry provisioning configuration required")
        if self.requester.schema == self.claimant.schema:
            raise ValueError("requester and claimant schemas must be distinct")
        roles = [
            getattr(item, name)
            for item in (self.requester, self.claimant)
            for name in ("schema_owner_role", "runtime_role", "admin_role")
        ]
        if len(set(roles)) != 6:
            raise ValueError("all six pre-account authority roles must be distinct")
        if (self.requester.environment, self.requester.trust_domain) != (
            self.claimant.environment,
            self.claimant.trust_domain,
        ):
            raise ValueError("paired registries must share exact protocol/trust namespace")


@dataclass(frozen=True, slots=True)
class CredentialGeneration:
    credential_id: str
    principal_id: str
    semantic_role: CredentialSemanticRole
    credential_role: str
    key_id: str
    key_version: int
    public_key: bytes
    key_material_identity: str
    environment: str
    trust_domain: str
    lifecycle_generation: int
    lifecycle: CredentialLifecycle
    registry_revision: int

    def __post_init__(self) -> None:
        for value in (
            self.credential_id,
            self.principal_id,
            self.credential_role,
            self.key_id,
            self.key_material_identity,
            self.environment,
            self.trust_domain,
        ):
            _text(value)
        for number in (self.key_version, self.lifecycle_generation, self.registry_revision):
            _positive(number)
        if type(self.semantic_role) is not CredentialSemanticRole:
            raise TypeError("exact credential semantic role required")
        if type(self.lifecycle) is not CredentialLifecycle:
            raise TypeError("exact credential lifecycle required")
        if self.environment != "PRODUCTION":
            raise ValueError("cross-environment credential evidence rejected")
        if public_key_material_identity(self.public_key) != self.key_material_identity:
            raise CredentialResolutionError("stored public-key material identity is corrupt")


def _identity(role: CredentialSemanticRole) -> str:
    return (
        REQUESTER_SCHEMA_IDENTITY
        if role is CredentialSemanticRole.ROOT_PROOF_REQUESTER
        else (CLAIMANT_SCHEMA_IDENTITY)
    )


def _apply_source(
    config: PostgreSQLCredentialRegistryProvisioning,
    peer: PostgreSQLCredentialRegistryProvisioning,
    role: CredentialSemanticRole,
) -> str:
    """Generate reviewed SQL from fixed statements and safely quoted identifiers."""
    role_name = (
        REQUESTER_CREDENTIAL_ROLE
        if role is CredentialSemanticRole.ROOT_PROOF_REQUESTER
        else (CLAIMANT_CREDENTIAL_ROLE)
    )
    source = sql.SQL("""
DECLARE
  operation text;
  principal text;
  previous_revision bigint;
  previous_generation bigint;
  previous_credential text;
  previous_lifecycle text;
  old_key_version bigint;
  projected_credential text;
  targeted_credential text;
  existing {s}.credentials%ROWTYPE;
  stored_key bytea;
  requested_revision bigint;
  successor_revision bigint;
  python_strip_whitespace text := pg_catalog.concat(
    pg_catalog.chr(9),pg_catalog.chr(10),pg_catalog.chr(11),pg_catalog.chr(12),
    pg_catalog.chr(13),pg_catalog.chr(28),pg_catalog.chr(29),pg_catalog.chr(30),
    pg_catalog.chr(31),pg_catalog.chr(32),pg_catalog.chr(133),pg_catalog.chr(160),
    pg_catalog.chr(5760),pg_catalog.chr(8192),pg_catalog.chr(8193),pg_catalog.chr(8194),
    pg_catalog.chr(8195),pg_catalog.chr(8196),pg_catalog.chr(8197),pg_catalog.chr(8198),
    pg_catalog.chr(8199),pg_catalog.chr(8200),pg_catalog.chr(8201),pg_catalog.chr(8202),
    pg_catalog.chr(8232),pg_catalog.chr(8233),pg_catalog.chr(8239),pg_catalog.chr(8287),
    pg_catalog.chr(12288)
  );
BEGIN
  IF pg_catalog.current_setting('transaction_isolation') <> 'serializable'
     OR pg_catalog.current_setting('fsync') <> 'on'
     OR pg_catalog.current_setting('synchronous_commit') <> 'on' THEN
    RAISE EXCEPTION 'durable serializable authority transaction required' USING ERRCODE='23514';
  END IF;
  IF session_user <> {admin_name}
     OR (SELECT oid FROM pg_catalog.pg_roles WHERE rolname=session_user)
        <> (SELECT (config->>'admin_role_oid')::oid FROM {s}.metadata) THEN
    RAISE EXCEPTION 'direct provisioning login required' USING ERRCODE='42501';
  END IF;
  IF pg_catalog.jsonb_typeof(p) IS DISTINCT FROM 'object' OR
     (SELECT pg_catalog.array_agg(key ORDER BY key) FROM pg_catalog.jsonb_object_keys(p) key)
     IS DISTINCT FROM ARRAY['credential_id','expected_revision','key_id','key_material_identity',
       'key_version','lifecycle','operation','principal_id','public_key_hex']::text[] THEN
    RAISE EXCEPTION 'invalid credential request shape' USING ERRCODE='23514';
  END IF;
  operation := p->>'operation';
  principal := p->>'principal_id';
  IF operation NOT IN ('PROVISION','ROTATE','TRANSITION') OR operation IS NULL
     OR pg_catalog.jsonb_typeof(p->'operation') <> 'string'
     OR pg_catalog.jsonb_typeof(p->'principal_id') <> 'string'
     OR principal IS NULL OR pg_catalog.btrim(principal,python_strip_whitespace)='' THEN
    RAISE EXCEPTION 'invalid credential operation/principal' USING ERRCODE='23514';
  END IF;
  IF {requester} AND principal <> {requester_principal} THEN
    RAISE EXCEPTION 'requester principal mismatch' USING ERRCODE='23514';
  END IF;
  -- One database-wide transactional lock plus SERIALIZABLE conflict detection.
  -- Raw peer keys are checked even if a privileged stored fingerprint is corrupt.
  PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended({lock_namespace},0));
  SELECT c.revision,h.lifecycle_generation,h.credential_id,h.lifecycle,k.key_version
    INTO previous_revision,previous_generation,previous_credential,previous_lifecycle,old_key_version
    FROM {s}.current_credentials c JOIN {s}.history h
      ON (h.principal_id,h.revision)=(c.principal_id,c.revision)
    JOIN {s}.credentials k ON k.credential_id=h.credential_id
    WHERE c.principal_id=principal FOR UPDATE OF c;
  projected_credential := previous_credential;
  IF operation='TRANSITION' THEN
    IF (p->'credential_id' <> 'null'::jsonb AND
        (pg_catalog.jsonb_typeof(p->'credential_id') <> 'string' OR
         pg_catalog.btrim(p->>'credential_id',python_strip_whitespace)=''))
       OR p->'key_id' <> 'null'::jsonb
       OR p->'key_version' <> 'null'::jsonb OR p->'public_key_hex' <> 'null'::jsonb
       OR p->'key_material_identity' <> 'null'::jsonb
       OR pg_catalog.jsonb_typeof(p->'lifecycle') <> 'string'
       OR (p->>'lifecycle') NOT IN ('VERIFY_ONLY','REVOKED') THEN
      RAISE EXCEPTION 'invalid lifecycle transition' USING ERRCODE='23514';
    END IF;
    targeted_credential := COALESCE(p->>'credential_id',previous_credential);
    SELECT h.revision,h.lifecycle_generation,h.credential_id,h.lifecycle,k.key_version
      INTO previous_revision,previous_generation,previous_credential,previous_lifecycle,old_key_version
      FROM {s}.history h JOIN {s}.credentials k ON k.credential_id=h.credential_id
      WHERE h.principal_id=principal AND h.credential_id=targeted_credential
      ORDER BY h.revision DESC LIMIT 1;
  ELSE
    IF p->'lifecycle' <> 'null'::jsonb
       OR pg_catalog.jsonb_typeof(p->'credential_id') <> 'string'
       OR pg_catalog.btrim(p->>'credential_id',python_strip_whitespace)=''
       OR pg_catalog.jsonb_typeof(p->'key_id') <> 'string'
       OR pg_catalog.btrim(p->>'key_id',python_strip_whitespace)=''
       OR pg_catalog.jsonb_typeof(p->'key_version') <> 'number'
       OR (p->>'key_version') !~ '^[1-9][0-9]{{0,15}}$'
       OR (p->>'key_version')::numeric > 9007199254740991
       OR pg_catalog.jsonb_typeof(p->'public_key_hex') <> 'string'
       OR (p->>'public_key_hex') !~ '^[0-9a-f]{{64}}$'
       OR pg_catalog.jsonb_typeof(p->'key_material_identity') <> 'string'
       OR (p->>'key_material_identity') !~ '^sha256:[0-9a-f]{{64}}$' THEN
      RAISE EXCEPTION 'invalid canonical public credential' USING ERRCODE='23514';
    END IF;
    stored_key := pg_catalog.decode(p->>'public_key_hex','hex');
    IF p->>'key_material_identity' <> 'sha256:' || pg_catalog.encode(pg_catalog.sha256(
         pg_catalog.decode({fingerprint_domain_hex},'hex') || stored_key),'hex') THEN
      RAISE EXCEPTION 'public key material identity mismatch' USING ERRCODE='23514';
    END IF;
    IF operation='PROVISION' AND p->'expected_revision' <> 'null'::jsonb THEN
      RAISE EXCEPTION 'provisioning cannot choose predecessor' USING ERRCODE='23514';
    END IF;
    SELECT * INTO existing FROM {s}.credentials WHERE credential_id=p->>'credential_id';
    IF FOUND THEN
      IF (existing.principal_id,existing.key_id,existing.key_version,existing.public_key,
          existing.key_material_identity,existing.environment,existing.trust_domain,
          existing.semantic_role,existing.credential_role)
         IS DISTINCT FROM (principal,p->>'key_id',(p->>'key_version')::bigint,stored_key,
          p->>'key_material_identity',{environment},{trust_domain},{semantic_role},{credential_role}) THEN
        RAISE EXCEPTION 'same credential identity conflicts' USING ERRCODE='23505';
      END IF;
      IF operation='PROVISION' THEN
        RETURN (SELECT pg_catalog.to_jsonb(h) FROM {s}.history h
          WHERE h.credential_id=existing.credential_id ORDER BY h.revision LIMIT 1);
      END IF;
      IF previous_credential=existing.credential_id AND previous_lifecycle='ACTIVE'
         AND EXISTS(SELECT 1 FROM {s}.history retired WHERE retired.principal_id=principal
           AND retired.revision=previous_revision-1 AND retired.lifecycle='VERIFY_ONLY'
           AND (SELECT pg_catalog.max(predecessor.revision) FROM {s}.history predecessor
             WHERE predecessor.credential_id=retired.credential_id AND predecessor.revision<retired.revision)
             =(p->>'expected_revision')::bigint) THEN
        RETURN (SELECT pg_catalog.to_jsonb(h) FROM {s}.history h
          WHERE h.principal_id=principal AND h.revision=previous_revision);
      END IF;
      RAISE EXCEPTION 'rotation retry no longer exact' USING ERRCODE='23505';
    END IF;
    IF EXISTS(SELECT 1 FROM {s}.credentials
      WHERE key_id=p->>'credential_id' OR credential_id=p->>'key_id' OR key_id=p->>'key_id') THEN
      RAISE EXCEPTION 'ambiguous credential/key identity' USING ERRCODE='23505';
    END IF;
    IF EXISTS(SELECT 1 FROM {peer}.credentials WHERE public_key=stored_key) THEN
      RAISE EXCEPTION 'requester/claimant key material alias' USING ERRCODE='23505';
    END IF;
  END IF;
  IF operation='PROVISION' THEN
    IF p->'expected_revision' <> 'null'::jsonb OR previous_revision IS NOT NULL THEN
      RAISE EXCEPTION 'principal already provisioned' USING ERRCODE='23505';
    END IF;
    successor_revision := 1;
  ELSE
    IF pg_catalog.jsonb_typeof(p->'expected_revision') <> 'number'
       OR (p->>'expected_revision') !~ '^[1-9][0-9]{{0,15}}$'
       OR (p->>'expected_revision')::numeric > 9007199254740989 THEN
      RAISE EXCEPTION 'invalid expected revision' USING ERRCODE='23514';
    END IF;
    requested_revision := (p->>'expected_revision')::bigint;
    IF operation='TRANSITION' AND previous_lifecycle=p->>'lifecycle'
       AND (SELECT pg_catalog.max(predecessor.revision) FROM {s}.history predecessor
         WHERE predecessor.credential_id=previous_credential AND predecessor.revision<previous_revision)
         =requested_revision THEN
      RETURN (SELECT pg_catalog.to_jsonb(h) FROM {s}.history h
        WHERE h.principal_id=principal AND h.revision=previous_revision);
    END IF;
    IF previous_revision IS DISTINCT FROM requested_revision OR previous_lifecycle='REVOKED' THEN
      RAISE EXCEPTION 'stale or revoked lifecycle generation' USING ERRCODE='23505';
    END IF;
    SELECT pg_catalog.max(revision)+1 INTO successor_revision FROM {s}.history WHERE principal_id=principal;
    IF operation='TRANSITION' THEN
      IF previous_lifecycle=p->>'lifecycle' THEN
        RAISE EXCEPTION 'lifecycle must advance' USING ERRCODE='23505';
      END IF;
      INSERT INTO {s}.history VALUES(principal,successor_revision,previous_credential,
        successor_revision,p->>'lifecycle');
      IF previous_credential=projected_credential THEN
        UPDATE {s}.current_credentials SET revision=successor_revision WHERE principal_id=principal;
      END IF;
      RETURN (SELECT pg_catalog.to_jsonb(h) FROM {s}.history h
        WHERE h.principal_id=principal AND h.revision=successor_revision);
    END IF;
    IF (p->>'key_version')::bigint <= old_key_version THEN
      RAISE EXCEPTION 'key version must increase independently of lifecycle generation'
        USING ERRCODE='23505';
    END IF;
    INSERT INTO {s}.history VALUES(principal,successor_revision,previous_credential,
      successor_revision,'VERIFY_ONLY');
    successor_revision := successor_revision+1;
  END IF;
  INSERT INTO {s}.credentials VALUES(p->>'credential_id',principal,{semantic_role},
    {credential_role},p->>'key_id',(p->>'key_version')::bigint,stored_key,
    p->>'key_material_identity',{environment},{trust_domain});
  INSERT INTO {s}.history VALUES(principal,successor_revision,p->>'credential_id',
    successor_revision,'ACTIVE');
  INSERT INTO {s}.current_credentials VALUES(principal,successor_revision)
    ON CONFLICT (principal_id) DO UPDATE SET revision=EXCLUDED.revision;
  RETURN (SELECT pg_catalog.to_jsonb(h) FROM {s}.history h
    WHERE h.principal_id=principal AND h.revision=successor_revision);
END
""").format(
        s=sql.Identifier(config.schema),
        peer=sql.Identifier(peer.schema),
        admin_name=sql.Literal(config.admin_role),
        requester=sql.Literal(role is CredentialSemanticRole.ROOT_PROOF_REQUESTER),
        requester_principal=sql.Literal(REQUESTER_PRINCIPAL),
        lock_namespace=sql.Literal(
            "cryptohunter.preaccount.credentials:" + ":".join(sorted((config.schema, peer.schema)))
        ),
        environment=sql.Literal(config.environment),
        trust_domain=sql.Literal(config.trust_domain),
        semantic_role=sql.Literal(role.value),
        credential_role=sql.Literal(role_name),
        fingerprint_domain_hex=sql.Literal(_PUBLIC_KEY_FINGERPRINT_DOMAIN.hex()),
    )
    return str(source.as_string())


def _config_payload(
    config: PostgreSQLCredentialRegistryProvisioning,
    peer: PostgreSQLCredentialRegistryProvisioning,
    role: CredentialSemanticRole,
    oids: dict[str, int],
) -> dict[str, Any]:
    return {
        **asdict(config),
        "peer": asdict(peer),
        "semantic_role": role.value,
        **{
            name + "_oid": oids[getattr(config, name)]
            for name in ("schema_owner_role", "runtime_role", "admin_role")
        },
        "peer_owner_oid": oids[peer.schema_owner_role],
    }


def _ddl(
    config: PostgreSQLCredentialRegistryProvisioning,
    peer: PostgreSQLCredentialRegistryProvisioning,
    role: CredentialSemanticRole,
) -> sql.Composed:
    return sql.SQL("""
CREATE SCHEMA {s} AUTHORIZATION {owner};
SET LOCAL ROLE {owner};
CREATE TABLE {s}.metadata (
  singleton boolean PRIMARY KEY DEFAULT true CHECK(singleton),
  schema_identity text NOT NULL, schema_version integer NOT NULL,
  profile text NOT NULL, storage_family text NOT NULL,
  config jsonb NOT NULL, function_sha256 text NOT NULL
);
CREATE TABLE {s}.credentials (
  credential_id text PRIMARY KEY, principal_id text NOT NULL,
  semantic_role text NOT NULL, credential_role text NOT NULL,
  key_id text NOT NULL UNIQUE,
  key_version bigint NOT NULL CHECK(key_version BETWEEN 1 AND 9007199254740991),
  public_key bytea NOT NULL CHECK(octet_length(public_key)=32),
  key_material_identity text NOT NULL CHECK(key_material_identity ~ '^sha256:[0-9a-f]{{64}}$'),
  environment text NOT NULL CHECK(environment='PRODUCTION'), trust_domain text NOT NULL,
  UNIQUE(principal_id,key_version), UNIQUE(principal_id,credential_id)
);
CREATE TABLE {s}.history (
  principal_id text NOT NULL, revision bigint NOT NULL CHECK(revision BETWEEN 1 AND 9007199254740991),
  credential_id text NOT NULL,
  lifecycle_generation bigint NOT NULL CHECK(lifecycle_generation BETWEEN 1 AND 9007199254740991),
  lifecycle text NOT NULL CHECK(lifecycle IN ('ACTIVE','VERIFY_ONLY','REVOKED')),
  PRIMARY KEY(principal_id,revision), UNIQUE(principal_id,lifecycle_generation),
  FOREIGN KEY(principal_id,credential_id) REFERENCES {s}.credentials(principal_id,credential_id)
    DEFERRABLE INITIALLY DEFERRED
);
CREATE TABLE {s}.current_credentials (
  principal_id text PRIMARY KEY, revision bigint NOT NULL,
  FOREIGN KEY(principal_id,revision) REFERENCES {s}.history(principal_id,revision)
    DEFERRABLE INITIALLY DEFERRED
);
CREATE FUNCTION {s}.apply(p jsonb) RETURNS jsonb LANGUAGE plpgsql
SECURITY DEFINER SET search_path=pg_catalog AS $credential${source}$credential$;
REVOKE ALL ON SCHEMA {s} FROM PUBLIC;
GRANT USAGE ON SCHEMA {s} TO {runtime},{admin};
REVOKE ALL ON ALL TABLES IN SCHEMA {s} FROM PUBLIC,{runtime},{admin};
GRANT SELECT ON ALL TABLES IN SCHEMA {s} TO {runtime},{admin};
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA {s} FROM PUBLIC,{runtime},{admin};
GRANT EXECUTE ON FUNCTION {s}.apply(jsonb) TO {admin};
RESET ROLE;
""").format(
        s=sql.Identifier(config.schema),
        owner=sql.Identifier(config.schema_owner_role),
        runtime=sql.Identifier(config.runtime_role),
        admin=sql.Identifier(config.admin_role),
        source=sql.SQL(_apply_source(config, peer, role)),
    )


def provision_postgresql_preaccount_registries(
    bootstrap: PostgreSQLConnectionConfig,
    config: PostgreSQLPreaccountRegistryProvisioning,
) -> None:
    """Offline atomic schema/role installation; never called by runtime composition."""
    if (
        type(bootstrap) is not PostgreSQLConnectionConfig
        or type(config) is not PostgreSQLPreaccountRegistryProvisioning
    ):
        raise TypeError("exact bootstrap and paired provisioning configuration required")
    pairs = (
        (config.requester, config.claimant, CredentialSemanticRole.ROOT_PROOF_REQUESTER),
        (config.claimant, config.requester, CredentialSemanticRole.ROOT_PROOF_CLAIMANT),
    )
    with psycopg.connect(bootstrap.dsn, autocommit=True) as conn, conn.transaction():
        conn.execute("SET LOCAL search_path=pg_catalog")
        for item, _, _ in pairs:
            for role_name in (item.schema_owner_role, item.runtime_role, item.admin_role):
                conn.execute(
                    sql.SQL(
                        "CREATE ROLE {} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE "
                        "NOINHERIT NOREPLICATION NOBYPASSRLS"
                    ).format(sql.Identifier(role_name))
                )
        oids: dict[str, int] = dict(
            conn.execute("SELECT rolname,oid FROM pg_catalog.pg_roles").fetchall()
        )
        for item, peer, role in pairs:
            conn.execute(_ddl(item, peer, role))
            conn.execute(
                sql.SQL(
                    "INSERT INTO {}.metadata(schema_identity,schema_version,profile,"
                    "storage_family,config,function_sha256) VALUES(%s,%s,%s,%s,%s,%s)"
                ).format(sql.Identifier(item.schema)),
                (
                    _identity(role),
                    SCHEMA_VERSION,
                    SecurityProfile.PRODUCTION_LOCAL.value,
                    STORAGE_FAMILY,
                    Jsonb(_config_payload(item, peer, role, oids)),
                    hashlib.sha256(_apply_source(item, peer, role).encode()).hexdigest(),
                ),
            )
        for item, peer, _ in pairs:
            conn.execute(
                sql.SQL(
                    "GRANT USAGE ON SCHEMA {} TO {}; GRANT SELECT ON {}.credentials TO {}"
                ).format(
                    sql.Identifier(item.schema),
                    sql.Identifier(peer.schema_owner_role),
                    sql.Identifier(item.schema),
                    sql.Identifier(peer.schema_owner_role),
                )
            )


_COLUMNS = {
    "metadata": (
        ("singleton", "boolean", "true"),
        ("schema_identity", "text", None),
        ("schema_version", "integer", None),
        ("profile", "text", None),
        ("storage_family", "text", None),
        ("config", "jsonb", None),
        ("function_sha256", "text", None),
    ),
    "credentials": (
        ("credential_id", "text", None),
        ("principal_id", "text", None),
        ("semantic_role", "text", None),
        ("credential_role", "text", None),
        ("key_id", "text", None),
        ("key_version", "bigint", None),
        ("public_key", "bytea", None),
        ("key_material_identity", "text", None),
        ("environment", "text", None),
        ("trust_domain", "text", None),
    ),
    "history": (
        ("principal_id", "text", None),
        ("revision", "bigint", None),
        ("credential_id", "text", None),
        ("lifecycle_generation", "bigint", None),
        ("lifecycle", "text", None),
    ),
    "current_credentials": (("principal_id", "text", None), ("revision", "bigint", None)),
}


def _constraints(schema: str) -> set[tuple[str, str, str]]:
    maximum = "'9007199254740991'"
    return {
        ("metadata", "p", "PRIMARY KEY (singleton)"),
        ("metadata", "c", "CHECK (singleton)"),
        ("credentials", "p", "PRIMARY KEY (credential_id)"),
        ("credentials", "u", "UNIQUE (key_id)"),
        ("credentials", "u", "UNIQUE (principal_id, key_version)"),
        ("credentials", "u", "UNIQUE (principal_id, credential_id)"),
        (
            "credentials",
            "c",
            f"CHECK (((key_version >= 1) AND (key_version <= {maximum}::bigint)))",
        ),
        ("credentials", "c", "CHECK ((octet_length(public_key) = 32))"),
        ("credentials", "c", "CHECK ((key_material_identity ~ '^sha256:[0-9a-f]{64}$'::text))"),
        ("credentials", "c", "CHECK ((environment = 'PRODUCTION'::text))"),
        ("history", "p", "PRIMARY KEY (principal_id, revision)"),
        ("history", "u", "UNIQUE (principal_id, lifecycle_generation)"),
        ("history", "c", f"CHECK (((revision >= 1) AND (revision <= {maximum}::bigint)))"),
        (
            "history",
            "c",
            f"CHECK (((lifecycle_generation >= 1) AND (lifecycle_generation <= {maximum}::bigint)))",
        ),
        (
            "history",
            "c",
            "CHECK ((lifecycle = ANY (ARRAY['ACTIVE'::text, 'VERIFY_ONLY'::text, 'REVOKED'::text])))",
        ),
        (
            "history",
            "f",
            f"FOREIGN KEY (principal_id, credential_id) REFERENCES {schema}.credentials(principal_id, credential_id) DEFERRABLE INITIALLY DEFERRED",
        ),
        ("current_credentials", "p", "PRIMARY KEY (principal_id)"),
        (
            "current_credentials",
            "f",
            f"FOREIGN KEY (principal_id, revision) REFERENCES {schema}.history(principal_id, revision) DEFERRABLE INITIALLY DEFERRED",
        ),
    }


class _PostgreSQLCredentialAuthority:
    _semantic_role: CredentialSemanticRole
    _admin = False

    def __init__(
        self,
        connection: PostgreSQLConnectionConfig,
        *,
        schema: str,
        environment: str,
        trust_domain: str,
    ) -> None:
        if type(connection) is not PostgreSQLConnectionConfig:
            raise TypeError("exact PostgreSQL connection configuration required")
        if type(schema) is not str or _SAFE_IDENTIFIER.fullmatch(schema) is None:
            raise ValueError("safe reviewed schema identifier required")
        if environment != "PRODUCTION" or type(environment) is not str:
            raise ValueError("protocol environment must be PRODUCTION")
        _text(trust_domain)
        self._connection = connection
        self._schema = schema
        self._environment = environment
        self._trust_domain = trust_domain
        self._qualify()

    @property
    def schema(self) -> str:
        return self._schema

    @property
    def environment(self) -> str:
        return self._environment

    @property
    def trust_domain(self) -> str:
        return self._trust_domain

    def __repr__(self) -> str:
        return f"{type(self).__name__}(connection=<redacted>, schema={self._schema!r})"

    def _connect(self) -> psycopg.Connection[Any]:
        conn = psycopg.connect(self._connection.dsn, autocommit=True)
        # Connection options cannot redirect any qualification or read helper.
        conn.execute("SET search_path=pg_catalog")
        return conn

    def _qualify(self) -> None:
        try:
            with self._connect() as conn, conn.transaction():
                self._qualify_connection(conn)
        except RegistryQualificationError:
            raise
        except (psycopg.Error, TypeError, ValueError, KeyError) as exc:
            raise RegistryQualificationError(
                "pre-account PostgreSQL authority qualification failed"
            ) from exc

    def _qualify_connection(self, conn: psycopg.Connection[Any]) -> None:
        metadata = conn.execute(
            sql.SQL(
                "SELECT schema_identity,schema_version,profile,"
                "storage_family,config,function_sha256 FROM {}.metadata"
            ).format(sql.Identifier(self._schema))
        ).fetchall()
        if len(metadata) != 1 or metadata[0][:4] != (
            _identity(self._semantic_role),
            SCHEMA_VERSION,
            SecurityProfile.PRODUCTION_LOCAL.value,
            STORAGE_FAMILY,
        ):
            raise RegistryQualificationError("unknown pre-account schema/version/profile")
        payload = metadata[0][4]
        if type(payload) is not dict or set(payload) != {
            "schema",
            "schema_owner_role",
            "runtime_role",
            "admin_role",
            "environment",
            "trust_domain",
            "peer",
            "semantic_role",
            "schema_owner_role_oid",
            "runtime_role_oid",
            "admin_role_oid",
            "peer_owner_oid",
        }:
            raise RegistryQualificationError("invalid authority metadata shape")
        config = PostgreSQLCredentialRegistryProvisioning(
            **{
                name: payload[name]
                for name in (
                    "schema",
                    "schema_owner_role",
                    "runtime_role",
                    "admin_role",
                    "environment",
                    "trust_domain",
                )
            }
        )
        peer = PostgreSQLCredentialRegistryProvisioning(**payload["peer"])
        pair = PostgreSQLPreaccountRegistryProvisioning(config, peer)
        if (config.schema, config.environment, config.trust_domain, payload["semantic_role"]) != (
            self._schema,
            self._environment,
            self._trust_domain,
            self._semantic_role.value,
        ):
            raise RegistryQualificationError("authority environment/trust/role mismatch")
        if conn.info.server_version < 160000 or conn.execute(
            "SELECT current_setting('fsync'),current_setting('synchronous_commit')"
        ).fetchone() != ("on", "on"):
            raise RegistryQualificationError("PostgreSQL durable production profile required")
        names = [
            getattr(item, name)
            for item in (pair.requester, pair.claimant)
            for name in ("schema_owner_role", "runtime_role", "admin_role")
        ]
        roles = conn.execute(
            "SELECT rolname,oid,rolcanlogin,rolinherit,rolsuper,rolcreaterole,"
            "rolcreatedb,rolreplication,rolbypassrls FROM pg_catalog.pg_roles "
            "WHERE rolname=ANY(%s)",
            (names,),
        ).fetchall()
        if len(roles) != 6 or any(
            row[2:] != (True, False, False, False, False, False, False) for row in roles
        ):
            raise RegistryQualificationError("authority role attributes mismatch")
        oids = {row[0]: row[1] for row in roles}
        if payload != _config_payload(config, peer, self._semantic_role, oids):
            raise RegistryQualificationError("persisted authority role OID mismatch")
        expected_login = config.admin_role if self._admin else config.runtime_role
        if conn.execute("SELECT session_user,current_user").fetchone() != (
            expected_login,
            expected_login,
        ):
            raise RegistryQualificationError("direct exact authority login required")
        for principal in (config.runtime_role, config.admin_role):
            if conn.execute(
                "SELECT has_database_privilege(%s,current_database(),'CREATE')", (principal,)
            ).fetchone() != (False,):
                raise RegistryQualificationError("runtime/provisioning database DDL forbidden")
        memberships = conn.execute(
            "SELECT 1 FROM pg_catalog.pg_auth_members WHERE roleid=ANY(%s) "
            "OR member=ANY(%s) LIMIT 1",
            (list(oids.values()), list(oids.values())),
        ).fetchone()
        if memberships is not None:
            raise RegistryQualificationError("authority role membership forbidden")
        self._qualify_catalog(conn, config, peer, oids)
        source = _apply_source(config, peer, self._semantic_role)
        if metadata[0][5] != hashlib.sha256(source.encode()).hexdigest():
            raise RegistryQualificationError("reviewed function manifest mismatch")
        functions = conn.execute(
            "SELECT p.oid,p.oid::regprocedure::text,p.proowner,p.prosecdef,"
            "p.proconfig,p.prosrc,p.prorettype::regtype::text,l.lanname,p.provolatile,"
            "p.proisstrict,p.proleakproof,p.proparallel,p.prokind,p.proargmodes "
            "FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace "
            "JOIN pg_catalog.pg_language l ON l.oid=p.prolang WHERE n.nspname=%s",
            (self._schema,),
        ).fetchall()
        if len(functions) != 1 or functions[0][1:] != (
            f"{self._schema}.{_FUNCTION_SIGNATURE}",
            oids[config.schema_owner_role],
            True,
            ["search_path=pg_catalog"],
            source,
            "jsonb",
            "plpgsql",
            "v",
            False,
            False,
            "u",
            "f",
            None,
        ):
            raise RegistryQualificationError("admin function implementation/security differs")
        function_acl = set(
            conn.execute(
                "SELECT a.grantee,a.privilege_type,a.is_grantable "
                "FROM pg_catalog.pg_proc p CROSS JOIN LATERAL aclexplode(p.proacl) a WHERE p.oid=%s",
                (functions[0][0],),
            ).fetchall()
        )
        if function_acl != {
            (oids[config.schema_owner_role], "EXECUTE", False),
            (oids[config.admin_role], "EXECUTE", False),
        }:
            raise RegistryQualificationError("admin function ACL differs")

    def _qualify_catalog(
        self,
        conn: psycopg.Connection[Any],
        config: PostgreSQLCredentialRegistryProvisioning,
        peer: PostgreSQLCredentialRegistryProvisioning,
        oids: dict[str, int],
    ) -> None:
        owner = oids[config.schema_owner_role]
        runtime = oids[config.runtime_role]
        admin = oids[config.admin_role]
        peer_owner = oids[peer.schema_owner_role]
        if conn.execute(
            "SELECT nspowner FROM pg_catalog.pg_namespace WHERE nspname=%s", (self._schema,)
        ).fetchone() != (owner,):
            raise RegistryQualificationError("schema owner mismatch")
        schema_acl = set(
            conn.execute(
                "SELECT a.grantee,a.privilege_type,a.is_grantable "
                "FROM pg_catalog.pg_namespace n CROSS JOIN LATERAL aclexplode(n.nspacl) a "
                "WHERE n.nspname=%s",
                (self._schema,),
            ).fetchall()
        )
        if schema_acl != {
            (owner, "CREATE", False),
            (owner, "USAGE", False),
            (runtime, "USAGE", False),
            (admin, "USAGE", False),
            (peer_owner, "USAGE", False),
        }:
            raise RegistryQualificationError("schema ACL differs from exact allowlist")
        relations = conn.execute(
            "SELECT c.oid,c.relname,c.relkind,c.relpersistence,c.relispartition,"
            "c.relrowsecurity,c.relforcerowsecurity,c.relowner FROM pg_catalog.pg_class c "
            "JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=%s AND c.relkind IN ('r','p','v','m','f','S') ORDER BY c.relname",
            (self._schema,),
        ).fetchall()
        if [row[1:] for row in relations] != [
            (name, "r", "p", False, False, False, owner) for name in sorted(_TABLES)
        ]:
            raise RegistryQualificationError("physical authority relations differ")
        relation_oids = [row[0] for row in relations]
        for query in (
            "SELECT 1 FROM pg_catalog.pg_trigger WHERE tgrelid=ANY(%s) AND NOT tgisinternal LIMIT 1",
            "SELECT 1 FROM pg_catalog.pg_rewrite WHERE ev_class=ANY(%s) LIMIT 1",
            "SELECT 1 FROM pg_catalog.pg_policy WHERE polrelid=ANY(%s) LIMIT 1",
        ):
            if conn.execute(query, (relation_oids,)).fetchone() is not None:
                raise RegistryQualificationError("authority trigger/rule/policy forbidden")
        if (
            conn.execute(
                "SELECT 1 FROM pg_catalog.pg_inherits WHERE inhrelid=ANY(%s) "
                "OR inhparent=ANY(%s) LIMIT 1",
                (relation_oids, relation_oids),
            ).fetchone()
            is not None
        ):
            raise RegistryQualificationError("authority inheritance forbidden")
        columns = conn.execute(
            "SELECT c.relname,a.attnum,a.attname,format_type(a.atttypid,a.atttypmod),"
            "a.attnotnull,pg_get_expr(d.adbin,d.adrelid),a.attgenerated,a.attidentity,a.attacl "
            "FROM pg_catalog.pg_attribute a JOIN pg_catalog.pg_class c ON c.oid=a.attrelid "
            "JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace LEFT JOIN pg_catalog.pg_attrdef d "
            "ON d.adrelid=a.attrelid AND d.adnum=a.attnum WHERE n.nspname=%s AND a.attnum>0 "
            "AND c.relkind IN ('r','p') AND NOT a.attisdropped ORDER BY c.relname,a.attnum",
            (self._schema,),
        ).fetchall()
        expected_columns = [
            (name, i, column, kind, True, default, "", "", None)
            for name in sorted(_COLUMNS)
            for i, (column, kind, default) in enumerate(_COLUMNS[name], 1)
        ]
        if columns != expected_columns:
            raise RegistryQualificationError("physical authority columns/ACL differ")
        constraints = conn.execute(
            "SELECT r.relname,c.contype,pg_get_constraintdef(c.oid),"
            "c.convalidated,c.condeferrable,c.condeferred,c.confupdtype,c.confdeltype,c.confmatchtype "
            "FROM pg_catalog.pg_constraint c JOIN pg_catalog.pg_class r ON r.oid=c.conrelid "
            "JOIN pg_catalog.pg_namespace n ON n.oid=r.relnamespace WHERE n.nspname=%s",
            (self._schema,),
        ).fetchall()
        if {row[:3] for row in constraints} != _constraints(self._schema) or len(
            constraints
        ) != len(_constraints(self._schema)):
            raise RegistryQualificationError("reviewed authority constraints differ")
        if any(
            row[3:]
            != (
                (True, True, True, "a", "a", "s")
                if row[1] == "f"
                else (True, False, False, " ", " ", " ")
            )
            for row in constraints
        ):
            raise RegistryQualificationError("authority constraint security attributes differ")
        expected_acl = {
            (name, owner, privilege, False) for name in _TABLES for privilege in _OWNER_PRIVILEGES
        }
        expected_acl |= {
            (name, grantee, "SELECT", False) for name in _TABLES for grantee in (runtime, admin)
        }
        expected_acl.add(("credentials", peer_owner, "SELECT", False))
        table_acl = set(
            conn.execute(
                "SELECT c.relname,a.grantee,a.privilege_type,a.is_grantable "
                "FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
                "CROSS JOIN LATERAL aclexplode(c.relacl) a WHERE n.nspname=%s",
                (self._schema,),
            ).fetchall()
        )
        if table_acl != expected_acl:
            raise RegistryQualificationError("table ACL differs from exact allowlist")

    def _load_history(
        self, conn: psycopg.Connection[Any], principal_id: str
    ) -> tuple[CredentialGeneration, ...]:
        rows = conn.execute(
            sql.SQL(
                "SELECT k.credential_id,k.principal_id,k.semantic_role,k.credential_role,"
                "k.key_id,k.key_version,k.public_key,k.key_material_identity,k.environment,k.trust_domain,"
                "h.lifecycle_generation,h.lifecycle,h.revision FROM {}.history h JOIN {}.credentials k "
                "ON (k.principal_id,k.credential_id)=(h.principal_id,h.credential_id) "
                "WHERE h.principal_id=%s ORDER BY h.revision"
            ).format(sql.Identifier(self._schema), sql.Identifier(self._schema)),
            (principal_id,),
        ).fetchall()
        evidence = tuple(
            CredentialGeneration(
                credential_id=row[0],
                principal_id=row[1],
                semantic_role=CredentialSemanticRole(row[2]),
                credential_role=row[3],
                key_id=row[4],
                key_version=row[5],
                public_key=row[6],
                key_material_identity=row[7],
                environment=row[8],
                trust_domain=row[9],
                lifecycle_generation=row[10],
                lifecycle=CredentialLifecycle(row[11]),
                registry_revision=row[12],
            )
            for row in rows
        )
        pointer = conn.execute(
            sql.SQL("SELECT revision FROM {}.current_credentials WHERE principal_id=%s").format(
                sql.Identifier(self._schema)
            ),
            (principal_id,),
        ).fetchone()
        if not evidence:
            if pointer is not None:
                raise CredentialResolutionError("current credential lacks retained history")
            return ()
        if pointer is None or type(pointer[0]) is not int or not 1 <= pointer[0] <= len(evidence):
            raise CredentialResolutionError("current credential lacks an exact retained generation")
        expected_role = (
            REQUESTER_CREDENTIAL_ROLE
            if self._semantic_role is CredentialSemanticRole.ROOT_PROOF_REQUESTER
            else CLAIMANT_CREDENTIAL_ROLE
        )
        latest: dict[str, CredentialGeneration] = {}
        previous: CredentialGeneration | None = None
        last_key_version = 0
        for revision, record in enumerate(evidence, 1):
            if (
                record.principal_id,
                record.semantic_role,
                record.credential_role,
                record.environment,
                record.trust_domain,
                record.registry_revision,
                record.lifecycle_generation,
            ) != (
                principal_id,
                self._semantic_role,
                expected_role,
                self._environment,
                self._trust_domain,
                revision,
                revision,
            ):
                raise CredentialResolutionError("credential history namespace/generation corrupt")
            if previous is None and record.lifecycle is not CredentialLifecycle.ACTIVE:
                raise CredentialResolutionError("first credential was not ACTIVE")
            predecessor = latest.get(record.credential_id)
            if predecessor is not None:
                if (
                    predecessor.lifecycle is CredentialLifecycle.REVOKED
                    or record.lifecycle is CredentialLifecycle.ACTIVE
                ):
                    raise CredentialResolutionError("credential lifecycle rolled backward")
            else:
                if (
                    record.lifecycle is not CredentialLifecycle.ACTIVE
                    or record.key_version <= last_key_version
                    or (
                        previous is not None
                        and (
                            previous.lifecycle is not CredentialLifecycle.VERIFY_ONLY
                            or previous.registry_revision != revision - 1
                        )
                    )
                ):
                    raise CredentialResolutionError("credential rotation lineage corrupt")
                last_key_version = record.key_version
            latest[record.credential_id] = record
            previous = record
        active = [
            record for record in latest.values() if record.lifecycle is CredentialLifecycle.ACTIVE
        ]
        projected = evidence[pointer[0] - 1]
        if (
            latest.get(projected.credential_id) != projected
            or projected.key_version != last_key_version
        ):
            raise CredentialResolutionError(
                "current projection is not exact latest credential generation"
            )
        if len(active) > 1 or (active and active[0] != projected):
            raise CredentialResolutionError("conflicting ACTIVE credential winners")
        return evidence

    def retained_history(self, principal_id: str) -> tuple[CredentialGeneration, ...]:
        principal = _text(principal_id)
        self._qualify()
        with self._connect() as conn, conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            return self._load_history(conn, principal)

    def _current(self, principal_id: str) -> CredentialGeneration:
        evidence = self.retained_history(principal_id)
        latest = {record.credential_id: record for record in evidence}
        active = [
            record for record in latest.values() if record.lifecycle is CredentialLifecycle.ACTIVE
        ]
        if len(active) != 1:
            raise CredentialResolutionError("exact ACTIVE current credential required")
        return active[0]

    def historical_generation(self, credential_id: str, generation: int) -> CredentialGeneration:
        _positive(generation)
        record = self._historical(credential_id)
        evidence = self.retained_history(record.principal_id)
        matches = [
            item
            for item in evidence
            if item.credential_id == record.credential_id
            and item.lifecycle_generation == generation
        ]
        if len(matches) != 1:
            raise CredentialResolutionError("exact historical credential generation unavailable")
        return matches[0]

    def _historical(self, identity: str) -> CredentialGeneration:
        _text(identity)
        self._qualify()
        with self._connect() as conn, conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            rows = conn.execute(
                sql.SQL(
                    "SELECT credential_id,principal_id FROM {}.credentials "
                    "WHERE credential_id=%s OR key_id=%s"
                ).format(sql.Identifier(self._schema)),
                (identity, identity),
            ).fetchall()
            if len(rows) != 1:
                raise CredentialResolutionError("exact retained credential identity unavailable")
            evidence = self._load_history(conn, rows[0][1])
            matches = [item for item in evidence if item.credential_id == rows[0][0]]
            if not matches:
                raise CredentialResolutionError("credential lacks retained lifecycle evidence")
            return matches[-1]


class _PostgreSQLCredentialRegistry(_PostgreSQLCredentialAuthority):
    @property
    def identity(self) -> ProviderIdentity:
        role = (
            ProviderRole.REQUESTER_CREDENTIAL_REGISTRY
            if self._semantic_role is CredentialSemanticRole.ROOT_PROOF_REQUESTER
            else ProviderRole.CLAIMANT_IDENTITY_REGISTRY
        )
        return ProviderIdentity(
            role,
            SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, self._trust_domain),
            f"postgresql:{self._schema}",
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(True, authoritative_reads=True, durable_state=True)

    def public_key(self, credential_identity: str) -> bytes:
        record = self._historical(credential_identity)
        if public_key_material_identity(record.public_key) != record.key_material_identity:
            raise CredentialResolutionError("stored public-key material identity is corrupt")
        return record.public_key

    def credential_identities(self) -> tuple[CredentialRoleIdentity, ...]:
        self._qualify()
        with self._connect() as conn, conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            principals = conn.execute(
                sql.SQL(
                    "SELECT principal_id FROM {}.current_credentials ORDER BY principal_id"
                ).format(sql.Identifier(self._schema))
            ).fetchall()
            credentials: list[CredentialRoleIdentity] = []
            for (principal,) in principals:
                latest = {
                    record.credential_id: record for record in self._load_history(conn, principal)
                }
                for record in latest.values():
                    credentials.append(
                        CredentialRoleIdentity(
                            self._semantic_role,
                            record.key_id,
                            self.identity.provider_namespace,
                            f"{record.key_id}:v{record.key_version}",
                            f"{self.identity.provider_namespace}:public-lifecycle:{record.lifecycle_generation}:"
                            f"{record.registry_revision}:{record.lifecycle.value}",
                            record.key_material_identity,
                        )
                    )
            return tuple(credentials)


class PostgreSQLRequesterCredentialRegistryProvider(_PostgreSQLCredentialRegistry):
    _semantic_role = CredentialSemanticRole.ROOT_PROOF_REQUESTER

    def active_requester_credential(self, requester_id: str) -> object:
        if requester_id != REQUESTER_PRINCIPAL or type(requester_id) is not str:
            raise CredentialResolutionError("wrong requester principal")
        record = self._current(requester_id)
        from bot_core.licensing.cha_root_proof_attempt_reservation import _RequesterCredentialV1

        return _RequesterCredentialV1(
            record.principal_id,
            record.credential_role,
            record.key_id,
            record.key_version,
            record.lifecycle.value,
            record.registry_revision,
        )

    def historical_requester_credential(self, credential_id: str) -> CredentialGeneration:
        return self._historical(credential_id)


class PostgreSQLClaimantIdentityRegistryProvider(_PostgreSQLCredentialRegistry):
    _semantic_role = CredentialSemanticRole.ROOT_PROOF_CLAIMANT

    def resolve_claimant(self, provisioning_principal_id: str) -> object:
        record = self._current(provisioning_principal_id)
        from bot_core.licensing.cha_root_proof_attempt_reservation import _ClaimantIdentityV1

        return _ClaimantIdentityV1(
            record.principal_id,
            record.key_id,
            record.key_version,
            record.lifecycle.value,
            record.registry_revision,
        )

    def historical_claimant(self, claimant_id: str, generation: int) -> CredentialGeneration:
        _positive(generation)
        matches = [
            record
            for record in self.retained_history(claimant_id)
            if record.lifecycle_generation == generation
        ]
        if len(matches) != 1:
            raise CredentialResolutionError("exact historical claimant generation unavailable")
        return matches[0]


class _PostgreSQLCredentialProvisioningAdmin(_PostgreSQLCredentialAuthority):
    _admin = True

    def _mutate(self, payload: dict[str, Any]) -> CredentialGeneration:
        self._qualify()
        for attempt in range(5):
            try:
                with self._connect() as conn, conn.transaction():
                    conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
                    row = conn.execute(
                        sql.SQL("SELECT {}.apply(%s)").format(sql.Identifier(self._schema)),
                        (Jsonb(payload),),
                    ).fetchone()
                    if row is None or type(row[0]) is not dict:
                        raise CredentialResolutionError("invalid provisioning authority result")
                    evidence = self._load_history(conn, payload["principal_id"])
                    matches = [
                        record
                        for record in evidence
                        if record.registry_revision == row[0]["revision"]
                    ]
                    if len(matches) != 1:
                        raise CredentialResolutionError("mutation did not retain exact history")
                    return matches[0]
            except SerializationFailure:
                if attempt == 4:
                    raise CredentialConflictError(
                        "concurrent credential mutation requires retry"
                    ) from None
            except (UniqueViolation, CheckViolation) as exc:
                raise CredentialConflictError(
                    "credential mutation conflicts with authority evidence"
                ) from exc
        raise CredentialConflictError("credential mutation unavailable")

    def _provision_payload(
        self,
        operation: str,
        principal_id: str,
        credential_id: str,
        key_id: str,
        key_version: int,
        public_key: bytes,
        expected_revision: int | None,
    ) -> dict[str, Any]:
        for value in (principal_id, credential_id, key_id):
            _text(value)
        _positive(key_version)
        material = public_key_material_identity(public_key)
        if expected_revision is not None:
            _positive(expected_revision)
        return {
            "operation": operation,
            "principal_id": principal_id,
            "credential_id": credential_id,
            "key_id": key_id,
            "key_version": key_version,
            "public_key_hex": public_key.hex(),
            "key_material_identity": material,
            "expected_revision": expected_revision,
            "lifecycle": None,
        }

    def provision_credential(
        self,
        *,
        principal_id: str,
        credential_id: str,
        key_id: str,
        key_version: int,
        public_key: bytes,
    ) -> CredentialGeneration:
        return self._mutate(
            self._provision_payload(
                "PROVISION", principal_id, credential_id, key_id, key_version, public_key, None
            )
        )

    def rotate_credential(
        self,
        *,
        principal_id: str,
        credential_id: str,
        key_id: str,
        key_version: int,
        public_key: bytes,
        expected_revision: int,
    ) -> CredentialGeneration:
        return self._mutate(
            self._provision_payload(
                "ROTATE",
                principal_id,
                credential_id,
                key_id,
                key_version,
                public_key,
                expected_revision,
            )
        )

    def transition_lifecycle(
        self,
        *,
        principal_id: str,
        lifecycle: CredentialLifecycle,
        expected_revision: int,
        credential_id: str | None = None,
    ) -> CredentialGeneration:
        _text(principal_id)
        _positive(expected_revision)
        if type(lifecycle) is not CredentialLifecycle or lifecycle is CredentialLifecycle.ACTIVE:
            raise ValueError("only forward VERIFY_ONLY/REVOKED transitions are permitted")
        if credential_id is not None:
            _text(credential_id)
        return self._mutate(
            {
                "operation": "TRANSITION",
                "principal_id": principal_id,
                "credential_id": credential_id,
                "key_id": None,
                "key_version": None,
                "public_key_hex": None,
                "key_material_identity": None,
                "expected_revision": expected_revision,
                "lifecycle": lifecycle.value,
            }
        )


class PostgreSQLRequesterCredentialProvisioningAdminProvider(
    _PostgreSQLCredentialProvisioningAdmin
):
    _semantic_role = CredentialSemanticRole.ROOT_PROOF_REQUESTER


class PostgreSQLClaimantIdentityProvisioningAdminProvider(_PostgreSQLCredentialProvisioningAdmin):
    _semantic_role = CredentialSemanticRole.ROOT_PROOF_CLAIMANT
