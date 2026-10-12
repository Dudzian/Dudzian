"""Instalacja offline i kwalifikacja trwałego magazynu podpisanych bajtów.

Magazyn zachowuje wyłącznie stan STAGED: zapis nie zatwierdza wydania Root
Proof, nie wiąże entitlement i nie dopisuje historii autorytatywnego issuera.
BYTEA zachowuje oryginalne bajty, także tożsamości z escaped U+0000. Runtime
weryfikuje podpis i zaufaną tożsamość poza SQL. Administrator PostgreSQL może
cofnąć całą bazę; ten magazyn nie stanowi niezależnej domeny checkpointu.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core.postgresql_issuer_history_canonical_sql import (
    CANONICAL_FUNCTION_RETURNS,
    canonical_function_sources,
)

SCHEMA_IDENTITY = "cryptohunter.root_proof_retention.postgresql"
SCHEMA_VERSION = 1
STORAGE_FAMILY = "POSTGRESQL_STAGED_ROOT_PROOF_RETENTION_V1"
PROFILE = "PRODUCTION_LOCAL"
_SAFE_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]{0,62}\Z")
_TABLES = ("metadata", "prepared")
_READ = "read_prepared(bytea,bytea)"
_RETAIN = "retain_prepared(bytea,bytea,bytea,bytea,bytea,bytea,bytea,bytea)"
_RUNTIME_FUNCTIONS = frozenset((_READ, _RETAIN))
_DECLARATIONS = {
    "canonical_string(bytea,integer)": (
        "p_bytes bytea,p_position integer",
        CANONICAL_FUNCTION_RETURNS["canonical_string(bytea,integer)"],
    ),
    "canonical_value(bytea,integer)": ("p_bytes bytea,p_position integer", "integer"),
    "validate_canonical_json(bytea)": ("p_bytes bytea", "void"),
    "canonical_object_member(bytea,text)": ("p_bytes bytea,p_key text", "bytea"),
    "canonical_object_members(bytea,text[])": ("p_bytes bytea,p_fields text[]", "bytea[]"),
    "validate_scope(bytea)": ("p_scope bytea", "void"),
    "reject_mutation()": ("", "trigger"),
    _READ: (
        "p_scope bytea,p_attempt bytea",
        "TABLE(envelope bytea,payload bytea,signature bytea)",
    ),
    _RETAIN: (
        "p_scope bytea,p_attempt bytea,p_operation bytea,p_entitlement bytea,"
        "p_root_proof bytea,p_envelope bytea,p_payload bytea,p_signature bytea",
        "TABLE(envelope bytea,payload bytea,signature bytea)",
    ),
}
_COLUMNS = {
    "metadata": (
        ("singleton", "boolean"),
        ("schema_identity", "text"),
        ("schema_version", "integer"),
        ("schema_name", "text"),
        ("profile", "text"),
        ("trust_domain", "bytea"),
        ("storage_family", "text"),
        ("schema_owner_role", "text"),
        ("schema_owner_role_oid", "oid"),
        ("runtime_role", "text"),
        ("runtime_role_oid", "oid"),
        ("function_definition_sha256", "jsonb"),
    ),
    "prepared": (
        ("scope_bytes", "bytea"),
        ("attempt_key", "bytea"),
        ("operation_key", "bytea"),
        ("entitlement_key", "bytea"),
        ("root_proof_key", "bytea"),
        ("account_key", "bytea"),
        ("canonical_envelope", "bytea"),
        ("canonical_payload", "bytea"),
        ("signature", "bytea"),
        ("status", "text"),
    ),
}
_INDEXES = {
    ("metadata", "metadata_pkey", ("singleton",), True),
    ("prepared", "prepared_pkey", ("scope_bytes", "attempt_key"), True),
    ("prepared", "prepared_attempt_key_key", ("attempt_key",), False),
    ("prepared", "prepared_root_proof_key_key", ("root_proof_key",), False),
    ("prepared", "prepared_scope_bytes_operation_key_key", ("scope_bytes", "operation_key"), False),
    (
        "prepared",
        "prepared_scope_bytes_entitlement_key_key",
        ("scope_bytes", "entitlement_key"),
        False,
    ),
    ("prepared", "prepared_scope_bytes_account_key_key", ("scope_bytes", "account_key"), False),
}


class RetentionSchemaQualificationError(RuntimeError):
    """Schemat lub principal nie spełnia przejrzanej granicy retencji STAGED."""


@dataclass(frozen=True, slots=True)
class PostgreSQLRootProofRetentionProvisioning:
    """Dokładna konfiguracja jednorazowej instalacji, bez danych klucza."""

    schema: str
    schema_owner_role: str
    runtime_role: str
    trust_domain: str

    def __post_init__(self) -> None:
        for name in ("schema", "schema_owner_role", "runtime_role"):
            _identifier(getattr(self, name))
        if self.schema_owner_role == self.runtime_role:
            raise ValueError("retention owner and runtime must be distinct")
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
        b"CryptoHunter/PostgreSQLRootProofRetentionFunction/v1\0"
        + signature.encode("ascii")
        + b"\0"
        + source.encode("utf-8")
    ).hexdigest()


def _function_sources(config: PostgreSQLRootProofRetentionProvisioning) -> dict[str, str]:
    s = '"' + config.schema + '"'
    guard = (
        f"IF session_user <> '{config.runtime_role}' OR current_user <> "
        f"'{config.schema_owner_role}' THEN RAISE EXCEPTION 'wrong retention principal' "
        "USING ERRCODE='42501'; END IF; "
        "IF pg_catalog.current_setting('server_version_num')::integer < 160000 "
        "OR pg_catalog.current_setting('fsync') <> 'on' "
        "OR pg_catalog.current_setting('synchronous_commit') <> 'on' "
        "OR pg_catalog.current_setting('full_page_writes') <> 'on' THEN "
        "RAISE EXCEPTION 'retention durability mismatch' USING ERRCODE='42501'; END IF;"
    )
    sources = {
        "validate_scope(bytea)": """
DECLARE members bytea[]; member bytea; decoded record;
BEGIN
 members := {s}.canonical_object_members(p_scope,
 ARRAY['environment','issuer_id','product_scope','security_profile','trust_domain']);
 FOREACH member IN ARRAY members LOOP
  SELECT * INTO STRICT decoded FROM {s}.canonical_string(member,0);
  IF decoded.next_position IS DISTINCT FROM pg_catalog.octet_length(member)
  OR pg_catalog.octet_length(decoded.decoded_key)=0 THEN
   RAISE EXCEPTION 'malformed retention scope' USING ERRCODE='22023'; END IF;
 END LOOP;
 IF members[1] IS DISTINCT FROM pg_catalog.convert_to('"PRODUCTION"','UTF8')
 OR members[4] IS DISTINCT FROM pg_catalog.convert_to('"PRODUCTION_LOCAL"','UTF8')
 OR members[5] IS DISTINCT FROM (SELECT trust_domain FROM {s}.metadata WHERE singleton)
 THEN RAISE EXCEPTION 'retention scope mismatch' USING ERRCODE='22023'; END IF;
END
""",
        "reject_mutation()": """
BEGIN
 RAISE EXCEPTION 'retained STAGED bytes are immutable' USING ERRCODE='42501';
END
""",
        _READ: """
DECLARE decoded record;
BEGIN
 {guard}
 PERFORM {s}.validate_scope(p_scope);
 PERFORM pg_catalog.convert_from(p_attempt,'UTF8');
 SELECT * INTO STRICT decoded FROM {s}.canonical_string(p_attempt,0);
 IF decoded.next_position IS DISTINCT FROM pg_catalog.octet_length(p_attempt)
 OR pg_catalog.octet_length(decoded.decoded_key)=0 THEN
  RAISE EXCEPTION 'malformed retention attempt' USING ERRCODE='22023'; END IF;
 RETURN QUERY SELECT r.canonical_envelope,r.canonical_payload,r.signature
 FROM {s}.prepared r WHERE r.scope_bytes=p_scope AND r.attempt_key=p_attempt;
END
""",
        _RETAIN: """
DECLARE prior {s}.prepared%ROWTYPE; key_bytes bytea; decoded record;
 entitlement_members bytea[]; account bytea; fields text[];
 field_name text; envelope_member bytea; payload_member bytea;
BEGIN
 {guard}
 PERFORM {s}.validate_scope(p_scope);
 FOREACH key_bytes IN ARRAY ARRAY[p_attempt,p_operation,p_root_proof] LOOP
  PERFORM pg_catalog.convert_from(key_bytes,'UTF8');
  SELECT * INTO STRICT decoded FROM {s}.canonical_string(key_bytes,0);
  IF decoded.next_position IS DISTINCT FROM pg_catalog.octet_length(key_bytes)
  OR pg_catalog.octet_length(decoded.decoded_key)=0 THEN
   RAISE EXCEPTION 'malformed retention key' USING ERRCODE='22023'; END IF;
 END LOOP;
 entitlement_members := {s}.canonical_object_members(p_entitlement,
 ARRAY['bootstrap_entitlement_id','entitlement_generation']);
 SELECT * INTO STRICT decoded FROM {s}.canonical_string(entitlement_members[1],0);
 IF pg_catalog.octet_length(decoded.decoded_key)=0
 OR pg_catalog.convert_from(entitlement_members[2],'UTF8') !~ '^[1-9][0-9]*$'
 THEN RAISE EXCEPTION 'malformed retention entitlement' USING ERRCODE='22023'; END IF;
 PERFORM {s}.validate_canonical_json(p_envelope);
 PERFORM {s}.validate_canonical_json(p_payload);
 account := {s}.canonical_object_member(p_envelope,'account_id');
 SELECT * INTO STRICT decoded FROM {s}.canonical_string(account,0);
 IF pg_catalog.octet_length(decoded.decoded_key)=0
 OR p_signature IS NULL OR pg_catalog.octet_length(p_signature)<>64
 OR {s}.canonical_object_member(p_envelope,'state') IS DISTINCT FROM
 pg_catalog.convert_to('"STAGED"','UTF8')
 OR {s}.canonical_object_member(p_envelope,'scope') IS DISTINCT FROM p_scope
 OR {s}.canonical_object_member(p_envelope,'issuance_attempt_id') IS DISTINCT FROM p_attempt
 OR {s}.canonical_object_member(p_envelope,'logical_operation_id') IS DISTINCT FROM p_operation
 OR {s}.canonical_object_member(p_envelope,'root_proof_id') IS DISTINCT FROM p_root_proof
 OR {s}.canonical_object_member(p_envelope,'bootstrap_entitlement_id')
 IS DISTINCT FROM entitlement_members[1]
 OR {s}.canonical_object_member(p_envelope,'entitlement_generation')
 IS DISTINCT FROM entitlement_members[2]
 THEN RAISE EXCEPTION 'retention envelope identity mismatch' USING ERRCODE='22023'; END IF;
 fields := ARRAY['account_id','bootstrap_entitlement_id','entitlement_generation',
 'issuance_attempt_id','logical_operation_id','root_proof_id'];
 FOREACH field_name IN ARRAY fields LOOP
  envelope_member := {s}.canonical_object_member(p_envelope,field_name);
  payload_member := {s}.canonical_object_member(p_payload,field_name);
  IF envelope_member IS NULL OR payload_member IS DISTINCT FROM envelope_member THEN
   RAISE EXCEPTION 'retention payload identity mismatch' USING ERRCODE='22023'; END IF;
 END LOOP;
 FOREACH field_name IN ARRAY ARRAY['environment','issuer_id','product_scope','trust_domain'] LOOP
  IF {s}.canonical_object_member(p_payload,field_name) IS DISTINCT FROM
  {s}.canonical_object_member(p_scope,field_name) THEN
   RAISE EXCEPTION 'retention payload scope mismatch' USING ERRCODE='22023'; END IF;
 END LOOP;
 INSERT INTO {s}.prepared(scope_bytes,attempt_key,operation_key,entitlement_key,
 root_proof_key,account_key,canonical_envelope,canonical_payload,signature,status)
 VALUES(p_scope,p_attempt,p_operation,p_entitlement,p_root_proof,account,
 p_envelope,p_payload,p_signature,'STAGED') ON CONFLICT DO NOTHING;
 SELECT * INTO prior FROM {s}.prepared
 WHERE scope_bytes=p_scope AND attempt_key=p_attempt;
 IF NOT FOUND THEN
  RAISE EXCEPTION 'conflicting retained identity' USING ERRCODE='23505'; END IF;
 IF prior.operation_key IS DISTINCT FROM p_operation
 OR prior.entitlement_key IS DISTINCT FROM p_entitlement
 OR prior.root_proof_key IS DISTINCT FROM p_root_proof
 OR prior.account_key IS DISTINCT FROM account
 OR prior.canonical_envelope IS DISTINCT FROM p_envelope
 OR prior.canonical_payload IS DISTINCT FROM p_payload
 OR prior.signature IS DISTINCT FROM p_signature OR prior.status<>'STAGED' THEN
  RAISE EXCEPTION 'conflicting retained bytes' USING ERRCODE='23505'; END IF;
 RETURN QUERY SELECT prior.canonical_envelope,prior.canonical_payload,prior.signature;
END
""",
    }
    return {
        **canonical_function_sources(config.schema),
        **{
            signature: source.replace("{s}", s).replace("{guard}", guard)
            for signature, source in sources.items()
        },
    }


def _create_tables(conn: psycopg.Connection[Any], schema: str) -> None:
    conn.execute(
        sql.SQL("""
CREATE TABLE {s}.metadata (
 singleton boolean PRIMARY KEY DEFAULT true CHECK(singleton),
 schema_identity text NOT NULL,schema_version integer NOT NULL,schema_name text NOT NULL,
 profile text NOT NULL,trust_domain bytea NOT NULL,storage_family text NOT NULL,
 schema_owner_role text NOT NULL,schema_owner_role_oid oid NOT NULL,
 runtime_role text NOT NULL,runtime_role_oid oid NOT NULL,
 function_definition_sha256 jsonb NOT NULL
);
CREATE TABLE {s}.prepared (
 scope_bytes bytea NOT NULL,attempt_key bytea NOT NULL,operation_key bytea NOT NULL,
 entitlement_key bytea NOT NULL,root_proof_key bytea NOT NULL,account_key bytea NOT NULL,
 canonical_envelope bytea NOT NULL,canonical_payload bytea NOT NULL,
 signature bytea NOT NULL CHECK(octet_length(signature)=64),
 status text NOT NULL CHECK(status='STAGED'),
 PRIMARY KEY(scope_bytes,attempt_key),UNIQUE(attempt_key),UNIQUE(root_proof_key),
 UNIQUE(scope_bytes,operation_key),UNIQUE(scope_bytes,entitlement_key),
 UNIQUE(scope_bytes,account_key)
);
REVOKE ALL ON ALL TABLES IN SCHEMA {s} FROM PUBLIC;
""").format(s=sql.Identifier(schema))
    )


def _durability(conn: psycopg.Connection[Any]) -> None:
    if conn.info.server_version < 160000 or conn.execute(
        "SELECT pg_catalog.current_setting('fsync'),"
        "pg_catalog.current_setting('synchronous_commit'),"
        "pg_catalog.current_setting('full_page_writes')"
    ).fetchone() != ("on", "on", "on"):
        raise RetentionSchemaQualificationError("retention PostgreSQL durability mismatch")


def provision_postgresql_root_proof_retention(
    connection: PostgreSQLConnectionConfig,
    config: PostgreSQLRootProofRetentionProvisioning,
) -> None:
    """Atomowo tworzy świeży schemat i dwie role; istniejące obiekty odrzuca.

    Operator oddzielnie ustawia uwierzytelnianie LOGIN. Instalator nie modyfikuje
    grantów istniejących komponentów i nigdy nie jest uruchamiany przez runtime.
    """
    if (
        type(connection) is not PostgreSQLConnectionConfig
        or type(config) is not PostgreSQLRootProofRetentionProvisioning
    ):
        raise TypeError("exact connection and provisioning configuration required")
    with psycopg.connect(connection.dsn, autocommit=True) as conn, conn.transaction():
        _durability(conn)
        for role, login in ((config.schema_owner_role, "NOLOGIN"), (config.runtime_role, "LOGIN")):
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
        _create_tables(conn, config.schema)
        sources = _function_sources(config)
        for signature, source in sources.items():
            declaration, returns = _DECLARATIONS[signature]
            conn.execute(
                sql.SQL(
                    "CREATE FUNCTION {}.{}({}) RETURNS {} LANGUAGE plpgsql "
                    "SECURITY DEFINER SET search_path=pg_catalog AS {}"
                ).format(
                    sql.Identifier(config.schema),
                    sql.Identifier(signature.split("(", 1)[0]),
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
            if signature in _RUNTIME_FUNCTIONS:
                conn.execute(
                    sql.SQL("GRANT EXECUTE ON FUNCTION {}.{} TO {}").format(
                        sql.Identifier(config.schema),
                        sql.SQL(signature),
                        sql.Identifier(config.runtime_role),
                    )
                )
        for table in _TABLES:
            for trigger, events, level in (
                ("immutable_row", "UPDATE OR DELETE", "ROW"),
                ("immutable_truncate", "TRUNCATE", "STATEMENT"),
            ):
                conn.execute(
                    sql.SQL(
                        "CREATE TRIGGER {} BEFORE {} ON {}.{} FOR EACH {} "
                        "EXECUTE FUNCTION {}.reject_mutation()"
                    ).format(
                        sql.Identifier(trigger),
                        sql.SQL(events),
                        sql.Identifier(config.schema),
                        sql.Identifier(table),
                        sql.SQL(level),
                        sql.Identifier(config.schema),
                    )
                )
                conn.execute(
                    sql.SQL("ALTER TABLE {}.{} ENABLE ALWAYS TRIGGER {}").format(
                        sql.Identifier(config.schema),
                        sql.Identifier(table),
                        sql.Identifier(trigger),
                    )
                )
        conn.execute(
            sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
                sql.Identifier(config.schema), sql.Identifier(config.runtime_role)
            )
        )
        conn.execute(
            sql.SQL("GRANT SELECT ON {}.metadata TO {}").format(
                sql.Identifier(config.schema), sql.Identifier(config.runtime_role)
            )
        )
        role_oids = []
        for role in (config.schema_owner_role, config.runtime_role):
            role_row = conn.execute(
                "SELECT oid FROM pg_catalog.pg_roles WHERE rolname=%s", (role,)
            ).fetchone()
            if role_row is None:
                raise RetentionSchemaQualificationError("new retention role disappeared")
            role_oids.append(role_row[0])
        manifest = {
            signature: _source_fingerprint(signature, src) for signature, src in sources.items()
        }
        conn.execute(
            sql.SQL("INSERT INTO {}.metadata VALUES(true,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)").format(
                sql.Identifier(config.schema)
            ),
            (
                SCHEMA_IDENTITY,
                SCHEMA_VERSION,
                config.schema,
                PROFILE,
                _scalar(config.trust_domain),
                STORAGE_FAMILY,
                config.schema_owner_role,
                role_oids[0],
                config.runtime_role,
                role_oids[1],
                Jsonb(manifest),
            ),
        )


def qualify_retention_schema(conn: psycopg.Connection[Any], schema: str, trust_domain: str) -> None:
    """W bieżącej transakcji sprawdza direct-login, trwałość, DDL, źródła i ACL.

    Odczyt metadanych nie jest samodzielnym dowodem kwalifikacji: definicje
    katalogowe i funkcje są porównywane z przejrzanym kodem lokalnym.
    """
    _identifier(schema)
    if type(trust_domain) is not str or not trust_domain:
        raise ValueError("trust_domain must be an exact non-empty str")
    try:
        _qualify(conn, schema, trust_domain)
    except RetentionSchemaQualificationError:
        raise
    except (psycopg.Error, ValueError, TypeError, KeyError, IndexError) as exc:
        raise RetentionSchemaQualificationError(
            "PostgreSQL retention qualification failed"
        ) from exc


def _qualify(conn: psycopg.Connection[Any], schema: str, trust_domain: str) -> None:
    conn.execute("SET LOCAL search_path=pg_catalog")
    _durability(conn)
    rows = conn.execute(
        sql.SQL(
            "SELECT schema_identity,schema_version,schema_name,profile,trust_domain,storage_family,"
            "schema_owner_role,schema_owner_role_oid,runtime_role,runtime_role_oid,"
            "function_definition_sha256 FROM {}.metadata"
        ).format(sql.Identifier(schema))
    ).fetchall()
    if len(rows) != 1 or rows[0][:6] != (
        SCHEMA_IDENTITY,
        SCHEMA_VERSION,
        schema,
        PROFILE,
        _scalar(trust_domain),
        STORAGE_FAMILY,
    ):
        raise RetentionSchemaQualificationError("retention security metadata mismatch")
    metadata = rows[0]
    config = PostgreSQLRootProofRetentionProvisioning(
        schema, metadata[6], metadata[8], trust_domain
    )
    owner_oid, runtime_oid = metadata[7], metadata[9]
    for name, oid, login in (
        (config.schema_owner_role, owner_oid, False),
        (config.runtime_role, runtime_oid, True),
    ):
        attrs = conn.execute(
            "SELECT oid,rolcanlogin,rolinherit,rolsuper,rolcreaterole,rolcreatedb,rolreplication,"
            "rolbypassrls FROM pg_catalog.pg_roles WHERE rolname=%s",
            (name,),
        ).fetchone()
        if attrs != (oid, login, False, False, False, False, False, False):
            raise RetentionSchemaQualificationError(
                "retention role identity or attributes mismatch"
            )
    if conn.execute("SELECT session_user::text,current_user::text").fetchone() != (
        config.runtime_role,
        config.runtime_role,
    ):
        raise RetentionSchemaQualificationError("retention requires exact direct-login runtime")
    if (
        conn.execute(
            "SELECT 1 FROM pg_catalog.pg_auth_members WHERE roleid=ANY(%s) OR member=ANY(%s) LIMIT 1",
            ([owner_oid, runtime_oid], [owner_oid, runtime_oid]),
        ).fetchone()
        is not None
    ):
        raise RetentionSchemaQualificationError("retention role has a membership grant")
    namespace = conn.execute(
        "SELECT oid,nspowner FROM pg_catalog.pg_namespace WHERE nspname=%s", (schema,)
    ).fetchone()
    if namespace is None or namespace[1] != owner_oid:
        raise RetentionSchemaQualificationError("retention namespace owner mismatch")
    schema_oid = namespace[0]
    _acl(
        conn,
        "pg_namespace",
        "nspacl",
        "oid",
        schema_oid,
        {
            (owner_oid, owner_oid, "CREATE", False),
            (owner_oid, owner_oid, "USAGE", False),
            (owner_oid, runtime_oid, "USAGE", False),
        },
    )
    inventory = conn.execute(
        "SELECT relname,relkind,relpersistence,relowner,reloptions,reltablespace "
        "FROM pg_catalog.pg_class WHERE relnamespace=%s ORDER BY relname",
        (schema_oid,),
    ).fetchall()
    expected_inventory = [(name, "r", "p", owner_oid, None, 0) for name in _TABLES]
    expected_inventory += [(name, "i", "p", owner_oid, None, 0) for _, name, _, _ in _INDEXES]
    if inventory != sorted(expected_inventory):
        raise RetentionSchemaQualificationError("retention relation inventory mismatch")
    relations = conn.execute(
        "SELECT c.oid,relname,relispartition,relrowsecurity,relforcerowsecurity,relreplident,"
        "relpartbound,relhasrules,am.amname FROM pg_catalog.pg_class c "
        "JOIN pg_catalog.pg_am am ON am.oid=c.relam WHERE relnamespace=%s AND relkind='r' "
        "ORDER BY relname",
        (schema_oid,),
    ).fetchall()
    if [r[1:] for r in relations] != [
        (t, False, False, False, "d", None, False, "heap") for t in _TABLES
    ]:
        raise RetentionSchemaQualificationError("retention table behavior mismatch")
    relation_oids = [r[0] for r in relations]
    for oid, table, *_ in relations:
        privileges = {"SELECT", "INSERT", "UPDATE", "DELETE", "TRUNCATE", "REFERENCES", "TRIGGER"}
        expected = {(owner_oid, owner_oid, privilege, False) for privilege in privileges}
        if table == "metadata":
            expected.add((owner_oid, runtime_oid, "SELECT", False))
        _acl(conn, "pg_class", "relacl", "oid", oid, expected)
    _qualify_tables(conn, schema_oid, owner_oid, relation_oids)
    _qualify_functions(conn, schema_oid, owner_oid, runtime_oid, config, metadata[10])
    _qualify_auxiliary(conn, schema_oid, owner_oid, runtime_oid, relation_oids)


def _acl(
    conn: psycopg.Connection[Any],
    catalog: str,
    column: str,
    identity: str,
    oid: int,
    expected: set[tuple[int, int, str, bool]],
) -> None:
    rows = conn.execute(
        sql.SQL(
            "SELECT a.grantor,a.grantee,a.privilege_type,a.is_grantable "
            "FROM pg_catalog.{} o CROSS JOIN LATERAL pg_catalog.aclexplode(o.{}) a WHERE o.{}=%s"
        ).format(sql.Identifier(catalog), sql.Identifier(column), sql.Identifier(identity)),
        (oid,),
    ).fetchall()
    if len(rows) != len(expected) or set(rows) != expected:
        raise RetentionSchemaQualificationError("retention ACL differs from reviewed allowlist")


def _qualify_tables(
    conn: psycopg.Connection[Any], schema_oid: int, owner_oid: int, relation_oids: list[int]
) -> None:
    columns = conn.execute(
        "SELECT c.relname,a.attnum,a.attname,pg_catalog.format_type(a.atttypid,a.atttypmod),"
        "a.attnotnull,pg_catalog.pg_get_expr(d.adbin,d.adrelid),a.attgenerated,a.attidentity,"
        "co.collname,cn.nspname,co.collisdeterministic,a.attisdropped,a.attcompression "
        "FROM pg_catalog.pg_class c JOIN pg_catalog.pg_attribute a ON a.attrelid=c.oid "
        "LEFT JOIN pg_catalog.pg_attrdef d ON d.adrelid=c.oid AND d.adnum=a.attnum "
        "LEFT JOIN pg_catalog.pg_collation co ON co.oid=a.attcollation "
        "LEFT JOIN pg_catalog.pg_namespace cn ON cn.oid=co.collnamespace "
        "WHERE c.relnamespace=%s AND c.relkind='r' AND a.attnum>0 ORDER BY c.relname,a.attnum",
        (schema_oid,),
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
            False,
            "",
        )
        for table in _TABLES
        for index, (name, data_type) in enumerate(_COLUMNS[table], 1)
    ]
    if columns != expected_columns:
        raise RetentionSchemaQualificationError("retention columns differ from reviewed definition")
    constraints = conn.execute(
        "SELECT c.relname,co.contype,pg_catalog.pg_get_constraintdef(co.oid),"
        "co.condeferrable,co.condeferred,co.convalidated FROM pg_catalog.pg_constraint co "
        "JOIN pg_catalog.pg_class c ON c.oid=co.conrelid WHERE co.connamespace=%s",
        (schema_oid,),
    ).fetchall()
    definitions = {
        (
            table,
            "p" if primary else "u",
            ("PRIMARY KEY" if primary else "UNIQUE") + " (" + ", ".join(keys) + ")",
        )
        for table, _, keys, primary in _INDEXES
    }
    definitions |= {
        ("metadata", "c", "CHECK (singleton)"),
        ("prepared", "c", "CHECK ((octet_length(signature) = 64))"),
        ("prepared", "c", "CHECK ((status = 'STAGED'::text))"),
    }
    expected_constraints = {(*definition, False, False, True) for definition in definitions}
    if len(constraints) != len(expected_constraints) or set(constraints) != expected_constraints:
        raise RetentionSchemaQualificationError(
            "retention constraints differ from reviewed definition"
        )
    indexes = conn.execute(
        "SELECT t.relname,idx.relname,ARRAY(SELECT a.attname FROM "
        "pg_catalog.unnest(i.indkey::smallint[]) WITH ORDINALITY k(attnum,pos) "
        "JOIN pg_catalog.pg_attribute a ON a.attrelid=t.oid AND a.attnum=k.attnum ORDER BY k.pos),"
        "i.indisprimary,i.indisunique,i.indisvalid,i.indisready,i.indisexclusion,i.indimmediate,"
        "pg_catalog.pg_get_expr(i.indexprs,i.indrelid),pg_catalog.pg_get_expr(i.indpred,i.indrelid),"
        "am.amname,idx.relowner,i.indnkeyatts,i.indnatts,pg_catalog.pg_get_indexdef(i.indexrelid) "
        "FROM pg_catalog.pg_index i JOIN pg_catalog.pg_class t ON t.oid=i.indrelid "
        "JOIN pg_catalog.pg_class idx ON idx.oid=i.indexrelid "
        "JOIN pg_catalog.pg_am am ON am.oid=idx.relam WHERE t.relnamespace=%s",
        (schema_oid,),
    ).fetchall()
    expected_indexes = {
        (
            table,
            name,
            keys,
            primary,
            True,
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
        for table, name, keys, primary in _INDEXES
    }
    observed_indexes = {
        (r[0], r[1], tuple(r[2]), *r[3:-1], r[-1].split(" USING ", 1)[-1]) for r in indexes
    }
    if len(indexes) != len(expected_indexes) or observed_indexes != expected_indexes:
        raise RetentionSchemaQualificationError("retention indexes differ from reviewed definition")
    triggers = conn.execute(
        "SELECT c.relname,t.tgname,p.proname,n.nspname,t.tgtype,t.tgenabled,t.tgisinternal,"
        "t.tgdeferrable,t.tginitdeferred,t.tgnargs,t.tgargs,t.tgqual,t.tgoldtable,t.tgnewtable,"
        "t.tgparentid,t.tgconstraint,t.tgconstrrelid FROM pg_catalog.pg_trigger t "
        "JOIN pg_catalog.pg_class c ON c.oid=t.tgrelid JOIN pg_catalog.pg_proc p ON p.oid=t.tgfoid "
        "JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace WHERE t.tgrelid=ANY(%s)",
        (relation_oids,),
    ).fetchall()
    schema_row = conn.execute(
        "SELECT nspname FROM pg_catalog.pg_namespace WHERE oid=%s", (schema_oid,)
    ).fetchone()
    if schema_row is None:
        raise RetentionSchemaQualificationError("retention namespace disappeared")
    schema = schema_row[0]
    expected_triggers = {
        (
            table,
            trigger,
            "reject_mutation",
            schema,
            trigger_type,
            "A",
            False,
            False,
            False,
            0,
            b"",
            None,
            None,
            None,
            0,
            0,
            0,
        )
        for table in _TABLES
        for trigger, trigger_type in (("immutable_row", 27), ("immutable_truncate", 34))
    }
    if len(triggers) != len(expected_triggers) or set(triggers) != expected_triggers:
        raise RetentionSchemaQualificationError("retention immutable trigger inventory mismatch")


def _qualify_functions(
    conn: psycopg.Connection[Any],
    schema_oid: int,
    owner_oid: int,
    runtime_oid: int,
    config: PostgreSQLRootProofRetentionProvisioning,
    manifest: object,
) -> None:
    sources = _function_sources(config)
    if manifest != {
        signature: _source_fingerprint(signature, src) for signature, src in sources.items()
    }:
        raise RetentionSchemaQualificationError("retention function manifest mismatch")
    functions = conn.execute(
        "SELECT p.oid,p.proname || '(' || "
        "pg_catalog.replace(pg_catalog.oidvectortypes(p.proargtypes),', ', ',') || ')',"
        "p.proowner,p.prosecdef,p.proconfig,p.prosrc,pg_catalog.format_type(p.prorettype,NULL),"
        "l.lanname,p.provolatile,p.proisstrict,p.proleakproof,p.proparallel,p.prokind,p.proargmodes,"
        "p.proretset,p.pronargdefaults,p.prosupport,p.proargnames,"
        "CASE WHEN p.proallargtypes IS NULL THEN NULL ELSE ARRAY(SELECT "
        "pg_catalog.format_type(arg,NULL) FROM pg_catalog.unnest(p.proallargtypes) arg) END,"
        "p.probin,p.procost,p.prorows,p.prosqlbody IS NULL FROM pg_catalog.pg_proc p "
        "JOIN pg_catalog.pg_language l ON l.oid=p.prolang WHERE p.pronamespace=%s",
        (schema_oid,),
    ).fetchall()
    if len(functions) != len(sources):
        raise RetentionSchemaQualificationError("retention function inventory mismatch")
    for oid, signature, *definition in functions:
        if signature not in sources:
            raise RetentionSchemaQualificationError("unreviewed retention function")
        returns = _DECLARATIONS[signature][1]
        declaration = _DECLARATIONS[signature][0]
        argument_names = (
            [argument.split(" ", 1)[0] for argument in declaration.split(",")]
            if declaration
            else None
        )
        all_argument_types = None
        modes = None
        set_result = returns.startswith("TABLE(")
        if signature == "canonical_string(bytea,integer)":
            modes = ["i", "i", "t", "t"]
            argument_names = ["p_bytes", "p_position", "next_position", "decoded_key"]
            all_argument_types = ["bytea", "integer", "integer", "bytea"]
        elif signature in _RUNTIME_FUNCTIONS:
            input_count = 2 if signature == _READ else 8
            modes = ["i"] * input_count + ["t"] * 3
            argument_names = (argument_names or []) + ["envelope", "payload", "signature"]
            all_argument_types = ["bytea"] * (input_count + 3)
        if set_result:
            returns = "record"
        if tuple(definition) != (
            owner_oid,
            True,
            ["search_path=pg_catalog"],
            sources[signature],
            returns,
            "plpgsql",
            "v",
            False,
            False,
            "u",
            "f",
            modes,
            set_result,
            0,
            "-",
            argument_names,
            all_argument_types,
            None,
            100.0,
            1000.0 if set_result else 0.0,
            True,
        ):
            raise RetentionSchemaQualificationError(
                "retention function source or security metadata mismatch"
            )
        expected = {(owner_oid, owner_oid, "EXECUTE", False)}
        if signature in _RUNTIME_FUNCTIONS:
            expected.add((owner_oid, runtime_oid, "EXECUTE", False))
        _acl(conn, "pg_proc", "proacl", "oid", oid, expected)


def _qualify_auxiliary(
    conn: psycopg.Connection[Any],
    schema_oid: int,
    owner_oid: int,
    runtime_oid: int,
    relation_oids: list[int],
) -> None:
    types = conn.execute(
        "SELECT t.typname,t.typtype,t.typowner,c.relname,e.typname,t.typacl IS NULL "
        "FROM pg_catalog.pg_type t "
        "LEFT JOIN pg_catalog.pg_class c ON c.oid=t.typrelid "
        "LEFT JOIN pg_catalog.pg_type e ON e.oid=t.typelem WHERE t.typnamespace=%s ORDER BY t.typname",
        (schema_oid,),
    ).fetchall()
    expected_types: list[tuple[str, str, int, str | None, str | None, bool]] = [
        (name, "c", owner_oid, name, None, True) for name in _TABLES
    ]
    expected_types += [("_" + name, "b", owner_oid, None, name, True) for name in _TABLES]
    if types != sorted(expected_types):
        raise RetentionSchemaQualificationError("retention type inventory mismatch")
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
            raise RetentionSchemaQualificationError("unreviewed retention namespace object")
    for query, params in (
        (
            "SELECT 1 FROM pg_catalog.pg_inherits WHERE inhrelid=ANY(%s) OR inhparent=ANY(%s) LIMIT 1",
            (relation_oids, relation_oids),
        ),
        ("SELECT 1 FROM pg_catalog.pg_rewrite WHERE ev_class=ANY(%s) LIMIT 1", (relation_oids,)),
        ("SELECT 1 FROM pg_catalog.pg_policy WHERE polrelid=ANY(%s) LIMIT 1", (relation_oids,)),
        (
            "SELECT 1 FROM pg_catalog.pg_attribute WHERE attrelid=ANY(%s) AND attacl IS NOT NULL LIMIT 1",
            (relation_oids,),
        ),
        (
            "SELECT 1 FROM pg_catalog.pg_default_acl WHERE defaclnamespace=%s OR defaclrole=ANY(%s) "
            "LIMIT 1",
            (schema_oid, [owner_oid, runtime_oid]),
        ),
    ):
        if conn.execute(query, params).fetchone() is not None:
            raise RetentionSchemaQualificationError("unreviewed retention auxiliary behavior")
    for catalog, acl, namespace in (
        ("pg_namespace", "nspacl", "oid"),
        ("pg_class", "relacl", "relnamespace"),
        ("pg_proc", "proacl", "pronamespace"),
        ("pg_type", "typacl", "typnamespace"),
    ):
        if (
            conn.execute(
                sql.SQL(
                    "SELECT 1 FROM pg_catalog.{} o CROSS JOIN LATERAL pg_catalog.aclexplode(o.{}) a "
                    "WHERE a.grantee=%s AND o.{}<>%s LIMIT 1"
                ).format(sql.Identifier(catalog), sql.Identifier(acl), sql.Identifier(namespace)),
                (runtime_oid, schema_oid),
            ).fetchone()
            is not None
        ):
            raise RetentionSchemaQualificationError("retention runtime has an extra object grant")
    for query in (
        "SELECT 1 FROM pg_catalog.pg_attribute a CROSS JOIN LATERAL "
        "pg_catalog.aclexplode(a.attacl) g WHERE g.grantee=%s LIMIT 1",
        "SELECT 1 FROM pg_catalog.pg_default_acl d CROSS JOIN LATERAL "
        "pg_catalog.aclexplode(d.defaclacl) g WHERE g.grantee=%s LIMIT 1",
        "SELECT 1 FROM pg_catalog.pg_database d CROSS JOIN LATERAL "
        "pg_catalog.aclexplode(d.datacl) g WHERE g.grantee=%s "
        "AND g.privilege_type<>'CONNECT' LIMIT 1",
    ):
        if conn.execute(query, (runtime_oid,)).fetchone() is not None:
            raise RetentionSchemaQualificationError("retention runtime has an extra privilege")
    for catalog, column in (
        ("pg_tablespace", "spcacl"),
        ("pg_language", "lanacl"),
        ("pg_foreign_server", "srvacl"),
        ("pg_foreign_data_wrapper", "fdwacl"),
        ("pg_largeobject_metadata", "lomacl"),
        ("pg_parameter_acl", "paracl"),
    ):
        if (
            conn.execute(
                sql.SQL(
                    "SELECT 1 FROM pg_catalog.{} o CROSS JOIN LATERAL pg_catalog.aclexplode(o.{}) a "
                    "WHERE a.grantee=%s LIMIT 1"
                ).format(sql.Identifier(catalog), sql.Identifier(column)),
                (runtime_oid,),
            ).fetchone()
            is not None
        ):
            raise RetentionSchemaQualificationError("retention runtime has an extra service grant")
    for query in (
        "SELECT 1 FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n "
        "ON n.oid=p.pronamespace WHERE n.oid<>%s AND n.nspname<>'pg_catalog' "
        "AND n.nspname<>'information_schema' "
        "AND pg_catalog.has_function_privilege(%s,p.oid,'EXECUTE') LIMIT 1",
        "SELECT 1 FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n "
        "ON n.oid=c.relnamespace WHERE n.oid<>%s AND n.nspname<>'pg_catalog' "
        "AND n.nspname<>'information_schema' AND c.relkind IN ('r','p','v','m','f') "
        "AND (pg_catalog.has_table_privilege(%s,c.oid,"
        "'SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER') "
        "OR pg_catalog.has_any_column_privilege(%s,c.oid,'SELECT,INSERT,UPDATE,REFERENCES')) LIMIT 1",
        "SELECT 1 FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n "
        "ON n.oid=c.relnamespace WHERE n.oid<>%s AND n.nspname<>'pg_catalog' "
        "AND n.nspname<>'information_schema' AND CASE WHEN c.relkind='S' "
        "THEN pg_catalog.has_sequence_privilege(%s,c.oid,'SELECT,UPDATE,USAGE') ELSE false END LIMIT 1",
    ):
        params = (
            (schema_oid, runtime_oid, runtime_oid)
            if "has_any_column" in query
            else (schema_oid, runtime_oid)
        )
        if conn.execute(query, params).fetchone() is not None:
            raise RetentionSchemaQualificationError(
                "retention runtime has extra effective authority"
            )
    if conn.execute(
        "SELECT 1 FROM pg_catalog.pg_namespace WHERE pg_catalog.has_schema_privilege(%s,oid,'CREATE') "
        "LIMIT 1",
        (runtime_oid,),
    ).fetchone() is not None or conn.execute(
        "SELECT pg_catalog.has_database_privilege(%s,pg_catalog.current_database(),'CREATE')",
        (runtime_oid,),
    ).fetchone() != (False,):
        raise RetentionSchemaQualificationError("retention runtime can create database objects")
    for catalog, owner in (
        ("pg_class", "relowner"),
        ("pg_proc", "proowner"),
        ("pg_namespace", "nspowner"),
        ("pg_type", "typowner"),
        ("pg_database", "datdba"),
        ("pg_tablespace", "spcowner"),
        ("pg_language", "lanowner"),
        ("pg_foreign_server", "srvowner"),
        ("pg_foreign_data_wrapper", "fdwowner"),
        ("pg_largeobject_metadata", "lomowner"),
    ):
        if (
            conn.execute(
                sql.SQL("SELECT 1 FROM pg_catalog.{} WHERE {}=%s LIMIT 1").format(
                    sql.Identifier(catalog), sql.Identifier(owner)
                ),
                (runtime_oid,),
            ).fetchone()
            is not None
        ):
            raise RetentionSchemaQualificationError("retention runtime owns database objects")


__all__ = [
    "PostgreSQLRootProofRetentionProvisioning",
    "RetentionSchemaQualificationError",
    "SCHEMA_IDENTITY",
    "SCHEMA_VERSION",
    "STORAGE_FAMILY",
    "provision_postgresql_root_proof_retention",
    "qualify_retention_schema",
]
