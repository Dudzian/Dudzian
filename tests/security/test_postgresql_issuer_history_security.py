"""Independent negative tests of the real PostgreSQL history authority."""

from __future__ import annotations

import hashlib
import os
import sys
import uuid
from dataclasses import dataclass

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.errors import InsufficientPrivilege
from psycopg.types.json import Json, Jsonb

from bot_core import postgresql_issuer_history_schema as history_schema
from bot_core.authenticated_issuer_history import (
    NO_PREDECESSOR,
    RECORD_DOMAIN,
    HistoryContractError,
    HistoryEventIdentity,
    HistoryStreamIdentity,
    ReferenceAuthenticatedHistory,
    canonical_json_bytes,
)
from bot_core.postgresql_authenticated_issuer_history import (
    HistoryStorageUnavailable,
    PostgreSQLAuthenticatedIssuerHistory,
)
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig

pytestmark = pytest.mark.external_postgresql
BASE_DSN = os.environ.get(
    "DUDZIAN_TEST_POSTGRES_DSN",
    "host=127.0.0.1 port=55432 dbname=postgres user=postgres connect_timeout=5",
)
EVENT_DOMAIN = b"CryptoHunter/M0.5/IssuerHistoryEvent/v1\0"


@dataclass(frozen=True)
class Boundary:
    setup: history_schema.PostgreSQLIssuerHistoryProvisioning
    stream: HistoryStreamIdentity

    def connection(self, role: str | None = None) -> PostgreSQLConnectionConfig:
        return PostgreSQLConnectionConfig(
            make_conninfo(BASE_DSN, user=role or self.setup.runtime_role)
        )

    def runtime(self, role: str | None = None) -> PostgreSQLAuthenticatedIssuerHistory:
        return PostgreSQLAuthenticatedIssuerHistory(
            self.connection(role), schema=self.setup.schema, stream=self.stream
        )


@pytest.fixture
def boundary():
    suffix = uuid.uuid4().hex[:10]
    setup = history_schema.PostgreSQLIssuerHistoryProvisioning(
        f"ihsec_{suffix}",
        f"ihseco_{suffix}",
        f"ihsecr_{suffix}",
        f"ihseca_{suffix}",
        "security-test",
    )
    stream = HistoryStreamIdentity(
        "isolated-security-history",
        "isolated-issuer",
        "PRODUCTION_LOCAL",
        "isolated-test-environment",
        setup.trust_domain,
        "isolated-test-product",
        1,
    )
    boundary = Boundary(setup, stream)
    history_schema.provision_postgresql_issuer_history(PostgreSQLConnectionConfig(BASE_DSN), setup)
    try:
        history_schema.provision_postgresql_issuer_history_stream(
            boundary.connection(setup.admin_role), schema=setup.schema, stream=stream
        )
        yield boundary
    finally:
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(setup.schema))
            )
            for role in (setup.runtime_role, setup.admin_role, setup.schema_owner_role):
                conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))


def _raw_append(boundary: Boundary, payload: bytes, *, role: str | None = None) -> None:
    """Compute matching digests so lexical failures cannot hide behind hashing."""
    event = canonical_json_bytes(
        HistoryEventIdentity("isolated-op", "isolated-attempt", "isolated-proof").material()
    )
    stream = canonical_json_bytes(boundary.stream.material())
    event_digest = "sha256:" + hashlib.sha256(EVENT_DOMAIN + payload).hexdigest()
    raw = (
        b'{"canonical_event_payload":'
        + payload
        + b',"event_digest":"'
        + event_digest.encode("ascii")
        + b'","event_identity":'
        + event
        + b',"predecessor_authenticated_digest":"NO_PREDECESSOR","sequence":1,"stream":'
        + stream
        + b"}"
    )
    envelope = {
        "sequence": 1,
        "predecessor_authenticated_digest": NO_PREDECESSOR,
        "canonical_event_identity_hex": event.hex(),
        "event_digest": event_digest,
        "authenticated_digest": "sha256:" + hashlib.sha256(RECORD_DOMAIN + raw).hexdigest(),
        "canonical_payload_hex": payload.hex(),
    }
    with psycopg.connect(boundary.connection(role).dsn, autocommit=True) as conn:
        conn.execute(
            sql.SQL("SELECT {}.append_record(%s,%s,%s,%s)").format(
                sql.Identifier(boundary.setup.schema)
            ),
            (stream, NO_PREDECESSOR, Json(envelope), raw),
        )


@pytest.mark.parametrize(
    "payload",
    [
        b'{"z":0,"a":0}',
        b'{"a":0,"a":0}',
        b'{"":0}',
        b'{"a": 1}',
        b'{"a":1.0}',
        b'{"a":1e2}',
        b'{"a":-0}',
        b'{"a":01}',
        b'{"a":"\\/"}',
        b'{"a":"\\u0041"}',
        b'{"a":"\\u000a"}',
        b'{"a":"\\u001F"}',
        b'{"a":"\x00"}',
        b'{"a":"\xff"}',
        b'{"a":[1,]}',
        b'{"a":truex}',
        b"{}\n",
        b"[]",
    ],
)
def test_raw_runtime_append_rejects_noncanonical_bytes_with_matching_digests(boundary, payload):
    with pytest.raises(psycopg.Error):
        _raw_append(boundary, payload)
    assert boundary.runtime().current_record() is None


def test_raw_runtime_append_preserves_entire_frozen_canonical_domain(boundary):
    controls = {chr(number): chr(number) for number in range(32)}
    payload = {
        "controls": controls,
        "numbers": [0, -1, 10**500, -(10**500)],
        "scalars": [True, False, None, "", '\\"/', "ą😀\u2028\x7f"],
        "unicode-order": {"\x00": 1, "a": 2, "é": 3, "\uffff": 4, "😀": 5},
    }
    raw = canonical_json_bytes(payload)
    _raw_append(boundary, raw)
    record = boundary.runtime().current_record()
    assert record is not None
    assert record.event_digest == "sha256:" + hashlib.sha256(EVENT_DOMAIN + raw).hexdigest()
    boundary.runtime().verify()


def test_deep_canonical_payload_matches_reference_with_runtime_recursion_budget(boundary):
    original_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(max(original_limit, 5000))
        nested: object = None
        for _ in range(700):
            nested = [nested]
        payload = {"nested": nested}
        event = HistoryEventIdentity("isolated-op", "isolated-attempt", "isolated-proof")
        expected, _ = ReferenceAuthenticatedHistory(boundary.stream).append(
            expected_digest=NO_PREDECESSOR, event_identity=event, payload=payload
        )
        _raw_append(boundary, canonical_json_bytes(payload))
        actual = boundary.runtime().current_record()
        assert actual is not None
        assert actual.authenticated_digest == expected.authenticated_digest
        boundary.runtime().verify()
    finally:
        sys.setrecursionlimit(original_limit)


def test_runtime_and_admin_cannot_bypass_separate_authority_ports(boundary):
    setup = boundary.setup
    stream = canonical_json_bytes(boundary.stream.material())
    with psycopg.connect(boundary.connection().dsn, autocommit=True) as conn:
        for table in ("streams", "records", "attestations"):
            with pytest.raises(InsufficientPrivilege):
                conn.execute(
                    sql.SQL("SELECT * FROM {}.{}").format(
                        sql.Identifier(setup.schema), sql.Identifier(table)
                    )
                )
        for table in ("metadata", "streams", "records", "attestations"):
            with pytest.raises(InsufficientPrivilege):
                conn.execute(
                    sql.SQL("DELETE FROM {}.{}").format(
                        sql.Identifier(setup.schema), sql.Identifier(table)
                    )
                )
        with pytest.raises(InsufficientPrivilege):
            conn.execute(
                sql.SQL("SELECT {}.provision_stream(%s)").format(sql.Identifier(setup.schema)),
                (stream,),
            )
        with pytest.raises(InsufficientPrivilege):
            conn.execute(
                sql.SQL("SELECT {}.validate_canonical_json(%s)").format(
                    sql.Identifier(setup.schema)
                ),
                (b"{}",),
            )
        with pytest.raises(InsufficientPrivilege):
            conn.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(setup.schema_owner_role)))
    with pytest.raises(InsufficientPrivilege):
        _raw_append(boundary, b"{}", role=setup.admin_role)
    for role in (setup.admin_role, "postgres"):
        with pytest.raises(HistoryStorageUnavailable):
            boundary.runtime(role)


def test_session_search_path_cannot_spoof_database_durability(boundary):
    shadow = "ihshadow_" + uuid.uuid4().hex[:10]
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(shadow)))
        conn.execute(
            sql.SQL(
                "CREATE FUNCTION {}.current_setting(text) RETURNS text "
                "LANGUAGE SQL AS 'SELECT ''on''::text'"
            ).format(sql.Identifier(shadow))
        )
        conn.execute(
            sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
                sql.Identifier(shadow), sql.Identifier(boundary.setup.runtime_role)
            )
        )
    try:
        connection = PostgreSQLConnectionConfig(
            make_conninfo(
                boundary.connection().dsn,
                options=f"-c search_path={shadow},pg_catalog -c synchronous_commit=off",
            )
        )
        with psycopg.connect(connection.dsn) as conn:
            assert conn.execute("SELECT current_setting('synchronous_commit')").fetchone() == (
                "on",
            )
            assert conn.execute(
                "SELECT pg_catalog.current_setting('synchronous_commit')"
            ).fetchone() == ("off",)
        with pytest.raises(HistoryStorageUnavailable):
            PostgreSQLAuthenticatedIssuerHistory(
                connection, schema=boundary.setup.schema, stream=boundary.stream
            )
    finally:
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(shadow)))


@pytest.mark.parametrize(
    "mutation",
    [
        "public_execute",
        "table_grant",
        "column_grant",
        "membership",
        "role_super",
        "unlogged",
        "row_security",
    ],
)
def test_catalog_privilege_or_durability_drift_fails_live_authority_operations(boundary, mutation):
    runtime = boundary.runtime()
    s = sql.Identifier(boundary.setup.schema)
    r = sql.Identifier(boundary.setup.runtime_role)
    a = sql.Identifier(boundary.setup.admin_role)
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        statements = {
            "public_execute": sql.SQL(
                "GRANT EXECUTE ON FUNCTION {}.append_record(bytea,text,json,bytea) TO PUBLIC"
            ).format(s),
            "table_grant": sql.SQL("GRANT SELECT ON {}.records TO {}").format(s, r),
            "column_grant": sql.SQL("GRANT SELECT (schema_version) ON {}.metadata TO {}").format(
                s, r
            ),
            "membership": sql.SQL("GRANT {} TO {}").format(a, r),
            "role_super": sql.SQL("ALTER ROLE {} SUPERUSER").format(r),
            "unlogged": sql.SQL("ALTER TABLE {}.attestations SET UNLOGGED").format(s),
            "row_security": sql.SQL("ALTER TABLE {}.records ENABLE ROW LEVEL SECURITY").format(s),
        }
        conn.execute(statements[mutation])
    with pytest.raises(HistoryStorageUnavailable):
        runtime.current_record()
    with pytest.raises(HistoryStorageUnavailable):
        boundary.runtime()
    if mutation == "membership":
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            conn.execute(sql.SQL("REVOKE {} FROM {}").format(a, r))


def test_coordinated_validator_source_and_manifest_tamper_fails_qualification(boundary):
    setup = boundary.setup
    signature = "validate_canonical_json(bytea)"
    replacement = "BEGIN RETURN; END"
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        conn.execute(
            sql.SQL(
                "CREATE OR REPLACE FUNCTION {}.validate_canonical_json(p_bytes bytea) RETURNS void "
                "LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS {}"
            ).format(sql.Identifier(setup.schema), sql.Literal(replacement))
        )
        manifest = conn.execute(
            sql.SQL("SELECT function_definition_sha256 FROM {}.metadata").format(
                sql.Identifier(setup.schema)
            )
        ).fetchone()[0]
        manifest[signature] = history_schema._source_fingerprint(signature, replacement)
        conn.execute(
            sql.SQL("UPDATE {}.metadata SET function_definition_sha256=%s").format(
                sql.Identifier(setup.schema)
            ),
            (Jsonb(manifest),),
        )
    with pytest.raises(HistoryStorageUnavailable):
        boundary.runtime()


def test_coordinated_constraint_and_physical_fingerprint_tamper_fails_qualification(boundary):
    setup = boundary.setup
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        constraint = conn.execute(
            "SELECT co.conname FROM pg_catalog.pg_constraint co JOIN pg_catalog.pg_class c "
            "ON c.oid=co.conrelid JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=%s AND c.relname='records' AND co.contype='u' LIMIT 1",
            (setup.schema,),
        ).fetchone()
        assert constraint is not None
        conn.execute(
            sql.SQL("ALTER TABLE {}.records DROP CONSTRAINT {}").format(
                sql.Identifier(setup.schema), sql.Identifier(constraint[0])
            )
        )
        fingerprint = history_schema._physical_fingerprint(conn, setup.schema)
        conn.execute(
            sql.SQL("UPDATE {}.metadata SET physical_definition_sha256=%s").format(
                sql.Identifier(setup.schema)
            ),
            (fingerprint,),
        )
    with pytest.raises(HistoryStorageUnavailable):
        boundary.runtime()


def test_non_tail_corruption_blocks_read_append_and_exact_replay(boundary):
    runtime = boundary.runtime()
    event = HistoryEventIdentity("corrupt-op", "corrupt-attempt", "corrupt-proof")
    first, _ = runtime.append(
        expected_digest=NO_PREDECESSOR, event_identity=event, payload={"fixture": 1}
    )
    runtime.append(
        expected_digest=first.authenticated_digest,
        event_identity=HistoryEventIdentity("second-op", "second-attempt", "second-proof"),
        payload={"fixture": 2},
    )
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        conn.execute(
            sql.SQL("UPDATE {}.records SET canonical_record=%s WHERE sequence=1").format(
                sql.Identifier(boundary.setup.schema)
            ),
            (b"{}",),
        )
    for operation in (
        runtime.current_record,
        runtime.verify,
        lambda: runtime.record_at(1000),
        lambda: runtime.append(
            expected_digest=NO_PREDECESSOR, event_identity=event, payload={"fixture": 1}
        ),
    ):
        with pytest.raises(HistoryContractError):
            operation()
    with pytest.raises(psycopg.Error):
        _raw_append(boundary, b"{}")
