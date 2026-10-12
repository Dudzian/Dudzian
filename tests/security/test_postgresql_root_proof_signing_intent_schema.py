"""Prawdziwy PostgreSQL: rezerwacja danych i izolacja issuer_writer.

Syntetyczne bajty służą wyłącznie kontroli SQL. Nie są autentyczną lineage ani
zweryfikowaną autoryzacją; bezpośredni SQL writer nie ustanawia takiego dowodu.
"""

from __future__ import annotations

import os
import hashlib
import uuid
from types import SimpleNamespace

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.errors import (
    CheckViolation,
    DataCorrupted,
    InsufficientPrivilege,
    InvalidParameterValue,
    UniqueViolation,
)

from bot_core import postgresql_root_proof_signing_intent_schema as implementation
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig

pytestmark = pytest.mark.external_postgresql
BASE_DSN = os.environ.get(
    "CRYPTOHUNTER_TEST_POSTGRES_DSN",
    os.environ.get(
        "DUDZIAN_TEST_POSTGRES_DSN",
        "host=127.0.0.1 port=55432 dbname=postgres user=postgres connect_timeout=5",
    ),
)


def _config():
    suffix = uuid.uuid4().hex[:12]
    return implementation.PostgreSQLRootProofSigningIntentProvisioning(
        "rpi_" + suffix, "rpio_" + suffix, "rpiw_" + suffix, "td-intent-sql-test"
    )


@pytest.fixture
def intent_schema():
    config = _config()
    implementation.provision_postgresql_root_proof_signing_intent(
        PostgreSQLConnectionConfig(BASE_DSN), config
    )
    try:
        yield SimpleNamespace(
            config=config,
            writer_dsn=make_conninfo(BASE_DSN, user=config.issuer_writer_role),
        )
    finally:
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            # DEFAULT ACL musi zniknąć przed usunięciem testowego owner.
            conn.execute(
                sql.SQL("DROP OWNED BY {} CASCADE").format(sql.Identifier(config.schema_owner_role))
            )
            conn.execute(
                sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(config.issuer_writer_role))
            )
            conn.execute(
                sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(config.schema_owner_role))
            )


def _candidate(*, identity="a", **changes):
    values = {
        "issuance_attempt_id": "rpa_018f3e70-7b5d-7c21-8b9a-0123456789a" + identity,
        "root_proof_id": "rpf_018f3e70-7b5e-7c21-8b9a-0123456789a" + identity,
        "logical_operation_id": "operation-" + identity,
        "bootstrap_entitlement_id": "entitlement-" + identity,
        "entitlement_generation": 1,
        "account_id": "account-" + identity,
    }
    values.update(changes)
    scope = {
        "security_profile": "PRODUCTION_LOCAL",
        "environment": "PRODUCTION",
        "trust_domain": values.pop("trust_domain", "td-intent-sql-test"),
        "issuer_id": values.pop("issuer_id", "issuer-test"),
        "product_scope": values.pop("product_scope", "product-test"),
    }
    envelope = {**values, "state": "RESERVED", "scope": scope}
    payload = {
        **values,
        **{
            name: scope[name]
            for name in ("environment", "trust_domain", "issuer_id", "product_scope")
        },
    }
    return scope, envelope, payload


def _reserve(conn, config, candidate, *, raw_payload=None):
    scope, envelope, payload = candidate
    args = (
        canonical_json_bytes(scope),
        canonical_json_bytes(envelope["issuance_attempt_id"]),
        canonical_json_bytes(envelope["logical_operation_id"]),
        canonical_json_bytes(
            {
                "bootstrap_entitlement_id": envelope["bootstrap_entitlement_id"],
                "entitlement_generation": envelope["entitlement_generation"],
            }
        ),
        canonical_json_bytes(envelope["root_proof_id"]),
        canonical_json_bytes(envelope),
        canonical_json_bytes(payload) if raw_payload is None else raw_payload,
    )
    return conn.execute(
        sql.SQL("SELECT envelope,payload FROM {}.reserve_intent(%s,%s,%s,%s,%s,%s,%s)").format(
            sql.Identifier(config.schema)
        ),
        args,
    ).fetchall()


def _qualify(value, *, dsn=None):
    with psycopg.connect(value.writer_dsn if dsn is None else dsn) as conn:
        implementation.qualify_signing_intent_schema(
            conn, value.config.schema, value.config.trust_domain
        )


def test_sql_exact_retry_retains_original_bytes_after_reconnect(intent_schema):
    value = intent_schema
    candidate = _candidate()
    _qualify(value)
    with psycopg.connect(value.writer_dsn) as conn:
        original = _reserve(conn, value.config, candidate)
    with psycopg.connect(value.writer_dsn) as conn:
        assert _reserve(conn, value.config, candidate) == original
        rows = conn.execute(
            sql.SQL("SELECT envelope,payload FROM {}.read_intent(%s,%s)").format(
                sql.Identifier(value.config.schema)
            ),
            (
                canonical_json_bytes(candidate[0]),
                canonical_json_bytes(candidate[1]["issuance_attempt_id"]),
            ),
        ).fetchall()
    assert (
        rows
        == original
        == [(canonical_json_bytes(candidate[1]), canonical_json_bytes(candidate[2]))]
    )


def test_read_resolves_global_attempt_before_caller_scope_conflict(intent_schema):
    value = intent_schema
    candidate = _candidate()
    with psycopg.connect(value.writer_dsn) as conn:
        original = _reserve(conn, value.config, candidate)
    changed_scope = {**candidate[0], "issuer_id": "other-issuer", "product_scope": "other-product"}
    with psycopg.connect(value.writer_dsn) as conn:
        rows = conn.execute(
            sql.SQL("SELECT envelope,payload FROM {}.read_intent(%s,%s)").format(
                sql.Identifier(value.config.schema)
            ),
            (
                canonical_json_bytes(changed_scope),
                canonical_json_bytes(candidate[1]["issuance_attempt_id"]),
            ),
        ).fetchall()
    # Scope callera nie zmienia istniejącego globalnego attempt w brak rekordu.
    # Python porównuje zachowany scope i zgłasza CONFLICT.
    assert rows == original


@pytest.mark.parametrize(
    "tamper",
    [
        "root_proof_id",
        "issuance_attempt_id",
        "logical_operation_id",
        "bootstrap_entitlement_id",
        "entitlement_generation",
        "account_id",
        "scope",
        "malformed_payload",
    ],
)
def test_read_rejects_coherently_changed_bytes_without_matching_indexed_identity(
    intent_schema, tamper
):
    value = intent_schema
    scope, envelope, payload = _candidate()
    with psycopg.connect(value.writer_dsn) as conn:
        _reserve(conn, value.config, (scope, envelope, payload))
    if tamper == "scope":
        envelope["scope"] = {**scope, "issuer_id": "other-issuer"}
        payload["issuer_id"] = "other-issuer"
    elif tamper != "malformed_payload":
        changed = 2 if tamper == "entitlement_generation" else _candidate(identity="b")[1][tamper]
        envelope[tamper] = payload[tamper] = changed
    raw_payload = b"not JSON" if tamper == "malformed_payload" else canonical_json_bytes(payload)
    envelope["canonical_payload_sha256"] = hashlib.sha256(raw_payload).hexdigest()
    with psycopg.connect(BASE_DSN) as conn:
        conn.execute(
            sql.SQL("ALTER TABLE {}.intents DISABLE TRIGGER immutable_row").format(
                sql.Identifier(value.config.schema)
            )
        )
        conn.execute(
            sql.SQL("UPDATE {}.intents SET canonical_envelope=%s,canonical_payload=%s").format(
                sql.Identifier(value.config.schema)
            ),
            (canonical_json_bytes(envelope), raw_payload),
        )
        conn.execute(
            sql.SQL("ALTER TABLE {}.intents ENABLE ALWAYS TRIGGER immutable_row").format(
                sql.Identifier(value.config.schema)
            )
        )
    _qualify(value)
    with psycopg.connect(value.writer_dsn) as conn, pytest.raises(DataCorrupted):
        conn.execute(
            sql.SQL("SELECT envelope,payload FROM {}.read_intent(%s,%s)").format(
                sql.Identifier(value.config.schema)
            ),
            (
                canonical_json_bytes(scope),
                canonical_json_bytes(_candidate()[1]["issuance_attempt_id"]),
            ),
        ).fetchall()


@pytest.mark.parametrize(
    "fence",
    ["issuance_attempt_id", "root_proof_id", "logical_operation_id", "entitlement", "account_id"],
)
@pytest.mark.parametrize("changed_scope", [False, True])
def test_global_fences_reject_other_identity_even_with_changed_scope(
    intent_schema, fence, changed_scope
):
    value = intent_schema
    first = _candidate()
    change = (
        {"bootstrap_entitlement_id": first[1]["bootstrap_entitlement_id"]}
        if fence == "entitlement"
        else {fence: first[1][fence]}
    )
    if changed_scope:
        change["issuer_id"] = "other-issuer"
        change["product_scope"] = "other-product"
    second = _candidate(identity="b", **change)
    with psycopg.connect(value.writer_dsn) as conn:
        _reserve(conn, value.config, first)
    with psycopg.connect(value.writer_dsn) as conn, pytest.raises(UniqueViolation):
        _reserve(conn, value.config, second)
    with psycopg.connect(BASE_DSN) as conn:
        assert conn.execute(
            sql.SQL("SELECT count(*) FROM {}.intents").format(sql.Identifier(value.config.schema))
        ).fetchone() == (1,)


@pytest.mark.parametrize("malformation", ["trust_domain", "payload_identity", "state", "canonical"])
def test_sql_rejects_malformed_or_other_trust_domain_without_fencing(intent_schema, malformation):
    value = intent_schema
    scope, envelope, payload = _candidate()
    if malformation == "trust_domain":
        scope["trust_domain"] = payload["trust_domain"] = "other-domain"
    elif malformation == "payload_identity":
        payload["account_id"] = "substituted-account"
    elif malformation == "state":
        envelope["state"] = "STAGED"
    raw_payload = canonical_json_bytes(payload) + b" " if malformation == "canonical" else None
    with (
        psycopg.connect(value.writer_dsn) as conn,
        pytest.raises((InvalidParameterValue, CheckViolation)),
    ):
        _reserve(conn, value.config, (scope, envelope, payload), raw_payload=raw_payload)
    with psycopg.connect(value.writer_dsn) as conn:
        assert _reserve(conn, value.config, _candidate())


@pytest.mark.parametrize("operation", ["UPDATE", "DELETE", "TRUNCATE", "INSERT", "SELECT"])
def test_writer_has_no_direct_table_authority(intent_schema, operation):
    value = intent_schema
    statements = {
        "UPDATE": "UPDATE {}.intents SET status='RESERVED'",
        "DELETE": "DELETE FROM {}.intents",
        "TRUNCATE": "TRUNCATE {}.intents",
        "INSERT": "INSERT INTO {}.intents DEFAULT VALUES",
        "SELECT": "SELECT * FROM {}.intents",
    }
    with psycopg.connect(value.writer_dsn) as conn, pytest.raises(InsufficientPrivilege):
        conn.execute(sql.SQL(statements[operation]).format(sql.Identifier(value.config.schema)))


def test_untrusted_direct_login_cannot_use_sql_or_assume_writer(intent_schema):
    value = intent_schema
    caller = "rpi_caller_" + uuid.uuid4().hex[:12]
    with psycopg.connect(BASE_DSN, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE ROLE {} LOGIN NOINHERIT").format(sql.Identifier(caller)))
        try:
            caller_dsn = make_conninfo(BASE_DSN, user=caller)
            with psycopg.connect(caller_dsn) as conn, pytest.raises(InsufficientPrivilege):
                _reserve(conn, value.config, _candidate())
            with psycopg.connect(caller_dsn) as conn, pytest.raises(InsufficientPrivilege):
                conn.execute(
                    sql.SQL("SET ROLE {}").format(sql.Identifier(value.config.issuer_writer_role))
                )
        finally:
            admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(caller)))


def test_sql_writer_storage_does_not_attest_authorization(intent_schema):
    """Direct SQL writer może utrwalić niezweryfikowane dane: to jawna granica."""
    value = intent_schema
    scope, envelope, payload = _candidate()
    envelope["caller_asserted_verified"] = True
    with psycopg.connect(value.writer_dsn) as conn:
        assert _reserve(conn, value.config, (scope, envelope, payload)) == [
            (canonical_json_bytes(envelope), canonical_json_bytes(payload))
        ]
    with psycopg.connect(BASE_DSN) as conn:
        assert conn.execute(
            sql.SQL("SELECT status FROM {}.intents").format(sql.Identifier(value.config.schema))
        ).fetchone() == ("RESERVED",)
    # Kwalifikacja SQL nie jest weryfikacją lineage/request ani capability.
    _qualify(value)


@pytest.mark.parametrize("operation", ["UPDATE", "DELETE", "TRUNCATE"])
def test_immutable_triggers_reject_privileged_row_repair(intent_schema, operation):
    value = intent_schema
    with psycopg.connect(value.writer_dsn) as conn:
        original = _reserve(conn, value.config, _candidate())
    statements = {
        "UPDATE": "UPDATE {}.intents SET canonical_payload=decode('00','hex')",
        "DELETE": "DELETE FROM {}.intents",
        "TRUNCATE": "TRUNCATE {}.intents",
    }
    with psycopg.connect(BASE_DSN) as conn, pytest.raises(InsufficientPrivilege):
        conn.execute(sql.SQL(statements[operation]).format(sql.Identifier(value.config.schema)))
    with psycopg.connect(value.writer_dsn) as conn:
        assert _reserve(conn, value.config, _candidate()) == original


@pytest.mark.parametrize(
    "tamper", ["table_acl", "function_path", "function_acl", "trigger", "constraint", "membership"]
)
def test_qualification_rejects_unreviewed_security_changes(intent_schema, tamper):
    value = intent_schema
    _qualify(value)
    schema = sql.Identifier(value.config.schema)
    writer = sql.Identifier(value.config.issuer_writer_role)
    owner = sql.Identifier(value.config.schema_owner_role)
    statements = {
        "table_acl": sql.SQL("GRANT SELECT ON {}.intents TO {}").format(schema, writer),
        "function_path": sql.SQL(
            "ALTER FUNCTION {}.read_intent(bytea,bytea) SET search_path=public"
        ).format(schema),
        "function_acl": sql.SQL(
            "GRANT EXECUTE ON FUNCTION {}.read_intent(bytea,bytea) TO PUBLIC"
        ).format(schema),
        "trigger": sql.SQL("ALTER TABLE {}.intents DISABLE TRIGGER immutable_row").format(schema),
        "constraint": sql.SQL(
            "ALTER TABLE {}.intents DROP CONSTRAINT intents_account_key_key"
        ).format(schema),
        "membership": sql.SQL("GRANT {} TO {}").format(owner, writer),
    }
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        conn.execute(statements[tamper])
    with pytest.raises(implementation.SigningIntentSchemaQualificationError):
        _qualify(value)


def test_effective_public_grant_outside_store_is_rejected(intent_schema):
    value = intent_schema
    outside = sql.Identifier("rpi_outside_" + uuid.uuid4().hex[:12])
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE TABLE public.{}(value integer)").format(outside))
        try:
            conn.execute(sql.SQL("GRANT SELECT ON public.{} TO PUBLIC").format(outside))
            with pytest.raises(implementation.SigningIntentSchemaQualificationError):
                _qualify(value)
        finally:
            conn.execute(sql.SQL("DROP TABLE public.{}").format(outside))


def test_session_durability_downgrade_is_rejected(intent_schema):
    with pytest.raises(implementation.SigningIntentSchemaQualificationError, match="durability"):
        _qualify(
            intent_schema,
            dsn=make_conninfo(intent_schema.writer_dsn, options="-c synchronous_commit=off"),
        )


def test_installer_failure_rolls_back_roles_schema_and_ddl(monkeypatch):
    config = _config()
    original = implementation._create_tables

    def fail_after_ddl(conn, schema):
        original(conn, schema)
        raise RuntimeError("injected installer failure")

    monkeypatch.setattr(implementation, "_create_tables", fail_after_ddl)
    with pytest.raises(RuntimeError, match="injected installer failure"):
        implementation.provision_postgresql_root_proof_signing_intent(
            PostgreSQLConnectionConfig(BASE_DSN), config
        )
    with psycopg.connect(BASE_DSN) as conn:
        assert (
            conn.execute(
                "SELECT 1 FROM pg_catalog.pg_namespace WHERE nspname=%s", (config.schema,)
            ).fetchone()
            is None
        )
        assert (
            conn.execute(
                "SELECT 1 FROM pg_catalog.pg_roles WHERE rolname=ANY(%s)",
                ([config.schema_owner_role, config.issuer_writer_role],),
            ).fetchone()
            is None
        )


def test_fsync_off_cluster_rejects_installation_before_creating_objects():
    unsafe_dsn = os.environ.get(
        "CRYPTOHUNTER_TEST_POSTGRES_UNSAFE_DSN",
        "host=127.0.0.1 port=55433 dbname=postgres user=postgres connect_timeout=2",
    )
    config = _config()
    with psycopg.connect(unsafe_dsn) as conn:
        assert conn.execute("SELECT current_setting('fsync')").fetchone() == ("off",)
    with pytest.raises(implementation.SigningIntentSchemaQualificationError, match="durability"):
        implementation.provision_postgresql_root_proof_signing_intent(
            PostgreSQLConnectionConfig(unsafe_dsn), config
        )
    with psycopg.connect(unsafe_dsn) as conn:
        assert (
            conn.execute(
                "SELECT 1 FROM pg_catalog.pg_namespace WHERE nspname=%s", (config.schema,)
            ).fetchone()
            is None
        )
        assert (
            conn.execute(
                "SELECT 1 FROM pg_catalog.pg_roles WHERE rolname=ANY(%s)",
                ([config.schema_owner_role, config.issuer_writer_role],),
            ).fetchone()
            is None
        )
