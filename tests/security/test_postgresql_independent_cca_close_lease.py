"""S9-RPI-12: real PostgreSQL 16 synthetic CCA read inside a held transaction.

Every successful outcome remains NOT_QUALIFIED and PRODUCTION_NO_GO.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import psycopg
import pytest
from psycopg import sql
from psycopg.errors import LockNotAvailable
from psycopg.types.json import Jsonb

from bot_core.test_only_independent_cca_close_lease import (
    SyntheticCloseExpectation,
    TestOnlyCloseLeaseError,
    hold_synthetic_committed_close,
)
from tests.architecture.test_stage9_atomic_root_proof_commit_boundary import (
    _repository_dependencies,
)
from tests.security.test_postgresql_independent_cca_reader_inspection import (
    _synthetic_test_pin,
)
from tests.security.test_postgresql_independent_cca_reader_smoke import (
    isolated_cca_read,
)

pytestmark = pytest.mark.external_postgresql


def _expected_fixture(value) -> SyntheticCloseExpectation:
    """Out-of-band TEST fixture source; never use this loader in production."""
    with psycopg.connect(value.dsn, autocommit=True) as bootstrap:
        request_text, digest, close_xid = bootstrap.execute(
            sql.SQL(
                "SELECT request::text,request_sha256,close_xid FROM {}.operations "
                "WHERE operation_id='rpi10-close-1'"
            ).format(sql.Identifier(value.schema))
        ).fetchone()
    with psycopg.connect(value.login("source"), autocommit=True) as source:
        with source.transaction():
            source_xid = source.execute("SELECT pg_catalog.pg_current_xact_id()::text").fetchone()[
                0
            ]
    assert source_xid != close_xid
    return SyntheticCloseExpectation(
        operation_id="rpi10-close-1",
        semantic_role="ROOT_PROOF_SIGNING",
        identity="isolated-root-identity",
        generation=1,
        predecessor_state="ACTIVE",
        target_state="VERIFY_ONLY",
        request_bytes=request_text.encode("utf-8"),
        request_sha256=digest,
        close_xid=close_xid,
        source_observation_xid=source_xid,
    )


def test_inspection_and_original_comparison_share_held_reader_transaction(isolated_cca_read):
    value = isolated_cca_read
    pin = _synthetic_test_pin(value)
    expected = _expected_fixture(value)
    with (
        psycopg.connect(value.login("reader"), autocommit=True) as reader,
        psycopg.connect(value.dsn, autocommit=True) as administrator,
    ):
        with reader.transaction():
            reader.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
            with hold_synthetic_committed_close(
                reader, catalog_pin=pin, expected=expected
            ) as evidence:
                assert evidence.status == "NOT_QUALIFIED"
                assert evidence.production_authorization == "BLOCKED"
                assert evidence.close_lease_qualified is False
                assert evidence.request_bytes == expected.request_bytes
                assert evidence.close_xid == expected.close_xid
                assert evidence.observation_xid not in {
                    expected.close_xid,
                    expected.source_observation_xid,
                }
                with pytest.raises(LockNotAvailable):
                    with administrator.transaction():
                        administrator.execute("SET LOCAL lock_timeout='150ms'")
                        administrator.execute(
                            sql.SQL(
                                "UPDATE {}.slots SET installed_marker='during-held-lease'"
                            ).format(sql.Identifier(value.schema))
                        )
            # Exiting the test-only context does not release caller-owned PG locks.
            with pytest.raises(LockNotAvailable):
                with administrator.transaction():
                    administrator.execute("SET LOCAL lock_timeout='150ms'")
                    administrator.execute(
                        sql.SQL(
                            "UPDATE {}.slots SET installed_marker='before-transaction-end'"
                        ).format(sql.Identifier(value.schema))
                    )
        with administrator.transaction():
            administrator.execute("SET LOCAL lock_timeout='1s'")
            administrator.execute(
                sql.SQL("UPDATE {}.slots SET installed_marker='after-release'").format(
                    sql.Identifier(value.schema)
                )
            )


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("request_sha256", "0" * 64),
        ("request_bytes", b"{}"),
        ("close_xid", "99999999999999999999"),
        ("identity", "other-credential"),
        ("generation", 2),
    ],
)
def test_source_mismatch_fails_before_yield(isolated_cca_read, field, replacement):
    pin = _synthetic_test_pin(isolated_cca_read)
    expected = replace(_expected_fixture(isolated_cca_read), **{field: replacement})
    with psycopg.connect(isolated_cca_read.login("reader"), autocommit=True) as conn:
        with pytest.raises(TestOnlyCloseLeaseError):
            with conn.transaction():
                conn.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
                with hold_synthetic_committed_close(conn, catalog_pin=pin, expected=expected):
                    pytest.fail("mismatched synthetic original must never be yielded")


@pytest.mark.parametrize("generation", [True, 1.0], ids=["boolean", "float"])
def test_non_integer_json_generation_with_matching_digest_denied(isolated_cca_read, generation):
    value = isolated_cca_read
    pin = _synthetic_test_pin(value)
    with psycopg.connect(value.dsn, autocommit=True) as bootstrap:
        with bootstrap.transaction():
            bootstrap.execute(
                sql.SQL("SELECT * FROM {}.slots FOR UPDATE").format(sql.Identifier(value.schema))
            )
            bootstrap.execute(
                sql.SQL(
                    "UPDATE {}.operations SET request=pg_catalog.jsonb_set("
                    "request,ARRAY['generation'],%s),close_xid=pg_catalog.pg_current_xact_id()::text "
                    "WHERE operation_id='rpi10-close-1'"
                ).format(sql.Identifier(value.schema)),
                (Jsonb(generation),),
            )
            bootstrap.execute(
                sql.SQL(
                    "UPDATE {}.operations SET request_sha256="
                    "pg_catalog.encode(pg_catalog.sha256("
                    "pg_catalog.convert_to(request::text,'UTF8')),'hex') "
                    "WHERE operation_id='rpi10-close-1'"
                ).format(sql.Identifier(value.schema))
            )
    expected = _expected_fixture(value)
    assert type(expected.generation) is int and expected.generation == 1
    retained_generation = json.loads(expected.request_bytes)["generation"]
    assert type(retained_generation) is type(generation)
    assert retained_generation == expected.generation
    assert hashlib.sha256(expected.request_bytes).hexdigest() == expected.request_sha256
    with psycopg.connect(value.login("reader"), autocommit=True) as conn:
        with pytest.raises(TestOnlyCloseLeaseError, match="^close request differs from expected"):
            with conn.transaction():
                conn.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
                with hold_synthetic_committed_close(conn, catalog_pin=pin, expected=expected):
                    pytest.fail("non-integer JSON generation must never yield evidence")


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("predecessor_state", "VERIFY_ONLY"),
        ("target_state", "REVOKED"),
        ("semantic_role", "HISTORY_ATTESTATION_SIGNING"),
    ],
)
def test_invalid_synthetic_expectations_rejected_at_construction(
    isolated_cca_read, field, replacement
):
    with pytest.raises(ValueError):
        replace(_expected_fixture(isolated_cca_read), **{field: replacement})


def test_source_observation_cannot_alias_cca_reader_xid(isolated_cca_read):
    value = isolated_cca_read
    pin = _synthetic_test_pin(value)
    expected = _expected_fixture(value)
    with psycopg.connect(value.login("reader"), autocommit=True) as conn:
        with pytest.raises(TestOnlyCloseLeaseError, match="independent transaction"):
            with conn.transaction():
                conn.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
                own_xid = conn.execute("SELECT pg_catalog.pg_current_xact_id()::text").fetchone()[0]
                assert own_xid not in {expected.close_xid, expected.source_observation_xid}
                with hold_synthetic_committed_close(
                    conn,
                    catalog_pin=pin,
                    expected=replace(expected, source_observation_xid=own_xid),
                ):
                    pytest.fail("CCA observation alias must fail before yielding")


def test_unknown_close_fails_closed_without_evidence(isolated_cca_read):
    value = isolated_cca_read
    pin = _synthetic_test_pin(value)
    expected = replace(_expected_fixture(value), operation_id="unknown-operation")
    with psycopg.connect(value.login("reader"), autocommit=True) as conn:
        with pytest.raises(TestOnlyCloseLeaseError, match="read denied"):
            with conn.transaction():
                conn.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
                with hold_synthetic_committed_close(conn, catalog_pin=pin, expected=expected):
                    pytest.fail("unknown operation must not yield positive evidence")


def test_catalog_drift_denies_before_synthetic_close_read(isolated_cca_read):
    value = isolated_cca_read
    pin = replace(
        _synthetic_test_pin(value),
        function_definition_sha256="f" * 64,
    )
    expected = _expected_fixture(value)
    with psycopg.connect(value.login("reader"), autocommit=True) as conn:
        with pytest.raises(TestOnlyCloseLeaseError, match="read denied"):
            with conn.transaction():
                conn.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
                with hold_synthetic_committed_close(conn, catalog_pin=pin, expected=expected):
                    pytest.fail("catalog drift must deny before yielding")


def test_production_cca_verifier_has_no_test_only_close_lease_integration():
    pending = ["bot_core.independent_cca_decision"]
    visited = set()
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        assert module != "bot_core.test_only_independent_cca_close_lease", (
            "production CCA verifier reaches the test-only close lease"
        )
        pending.extend(_repository_dependencies(module) - visited)
