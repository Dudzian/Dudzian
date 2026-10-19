"""S9-RPI-13: PostgreSQL 16 TEST-only cooperative CCA catalog DDL serialization.

The advisory gate is voluntary, not a PostgreSQL permission boundary.
These tests never install the production CCA, SQL catalog or private keys.
"""

from __future__ import annotations

import psycopg
import pytest
from psycopg import sql

from bot_core import test_only_independent_cca_close_lease as held
from bot_core.postgresql_independent_cca_reader_inspection import (
    IndependentCCAReaderCatalogError,
    inspect_independent_cca_reader_catalog,
)
from bot_core.test_only_cca_catalog_ddl_gate import (
    TestOnlyCCACatalogGateError,
    acquire_synthetic_reviewed_ddl_gate,
)
from bot_core.test_only_independent_cca_close_lease import (
    TestOnlyCloseLeaseError,
    hold_synthetic_committed_close,
)
from tests.architecture.test_stage9_atomic_root_proof_commit_boundary import (
    _repository_dependencies,
)
from tests.security.test_postgresql_independent_cca_close_lease import (
    _expected_fixture,
)
from tests.security.test_postgresql_independent_cca_reader_inspection import (
    _synthetic_test_pin,
)
from tests.security.test_postgresql_independent_cca_reader_smoke import (
    isolated_cca_read,
)

pytestmark = pytest.mark.external_postgresql


def _grant_public(bootstrap, value) -> None:
    with bootstrap.transaction():
        bootstrap.execute("SET LOCAL lock_timeout='1s'")
        bootstrap.execute("SET LOCAL statement_timeout='5s'")
        bootstrap.execute(
            sql.SQL("GRANT EXECUTE ON FUNCTION {}.read_exact_close(text) TO PUBLIC").format(
                sql.Identifier(value.schema)
            )
        )


def _revoke_public(bootstrap, value) -> None:
    with bootstrap.transaction():
        bootstrap.execute("SET LOCAL lock_timeout='1s'")
        bootstrap.execute("SET LOCAL statement_timeout='5s'")
        bootstrap.execute(
            sql.SQL("REVOKE EXECUTE ON FUNCTION {}.read_exact_close(text) FROM PUBLIC").format(
                sql.Identifier(value.schema)
            )
        )


def test_reviewed_ddl_writer_waits_for_reader_transaction_commit(isolated_cca_read):
    value = isolated_cca_read
    pin, expected = _synthetic_test_pin(value), _expected_fixture(value)
    with (
        psycopg.connect(value.login("reader"), autocommit=True) as reader,
        psycopg.connect(value.dsn, autocommit=True) as writer,
    ):
        with reader.transaction():
            reader.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
            with hold_synthetic_committed_close(
                reader, catalog_pin=pin, expected=expected
            ) as evidence:
                assert evidence.status == "NOT_QUALIFIED"
                with pytest.raises(TestOnlyCCACatalogGateError, match="exclusive"):
                    with writer.transaction():
                        writer.execute("SET LOCAL lock_timeout='150ms'")
                        acquire_synthetic_reviewed_ddl_gate(writer, pin)
                        pytest.fail("reviewed DDL must not pass the held reader gate")
            # Exiting Python's context did not end the SQL transaction.
            with pytest.raises(TestOnlyCCACatalogGateError, match="exclusive"):
                with writer.transaction():
                    writer.execute("SET LOCAL lock_timeout='150ms'")
                    acquire_synthetic_reviewed_ddl_gate(writer, pin)
        # After the reader commits, a participating reviewed DDL writer proceeds.
        with writer.transaction():
            writer.execute("SET LOCAL lock_timeout='1s'")
            acquire_synthetic_reviewed_ddl_gate(writer, pin)
            _grant_public(writer, value)
        with pytest.raises(IndependentCCAReaderCatalogError, match="EXECUTE"):
            with reader.transaction():
                reader.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
                inspect_independent_cca_reader_catalog(reader, pin)
        _revoke_public(writer, value)


def test_exclusive_reviewed_ddl_gate_blocks_late_reader(isolated_cca_read):
    value = isolated_cca_read
    pin, expected = _synthetic_test_pin(value), _expected_fixture(value)
    with (
        psycopg.connect(value.dsn, autocommit=True) as writer,
        psycopg.connect(value.login("reader"), autocommit=True) as reader,
    ):
        with writer.transaction():
            acquire_synthetic_reviewed_ddl_gate(writer, pin)
            with pytest.raises(TestOnlyCloseLeaseError, match="read denied"):
                with reader.transaction():
                    reader.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
                    reader.execute("SET LOCAL lock_timeout='150ms'")
                    with hold_synthetic_committed_close(reader, catalog_pin=pin, expected=expected):
                        pytest.fail("reader must wait for reviewed migration gate")
        with reader.transaction():
            reader.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
            with hold_synthetic_committed_close(
                reader, catalog_pin=pin, expected=expected
            ) as evidence:
                assert evidence.production_authorization == "BLOCKED"


def test_uncooperative_catalog_grant_between_inspection_and_read_is_denied(
    isolated_cca_read, monkeypatch
):
    value = isolated_cca_read
    pin, expected = _synthetic_test_pin(value), _expected_fixture(value)
    original = held.inspect_independent_cca_reader_catalog
    calls = []
    grant_committed = False
    with psycopg.connect(value.dsn, autocommit=True) as uncooperative:

        def inspect_then_drift(conn, catalog_pin):
            nonlocal grant_committed
            result = original(conn, catalog_pin)
            calls.append(result)
            if len(calls) == 1:
                # Does NOT take the reviewed exclusive advisory gate. Demonstrates
                # why a second catalog check is useful but not a full DDL fence.
                _grant_public(uncooperative, value)
                assert uncooperative.info.transaction_status == psycopg.pq.TransactionStatus.IDLE
                grant_committed = True
            return result

        monkeypatch.setattr(held, "inspect_independent_cca_reader_catalog", inspect_then_drift)
        try:
            with psycopg.connect(value.login("reader"), autocommit=True) as reader:
                with pytest.raises(TestOnlyCloseLeaseError, match="read denied") as denied:
                    with reader.transaction():
                        reader.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
                        with hold_synthetic_committed_close(
                            reader, catalog_pin=pin, expected=expected
                        ):
                            pytest.fail("drift after inspection must not yield")
            assert grant_committed, "administrator GRANT must commit before reinspection"
            assert len(calls) == 1
            assert isinstance(denied.value.__cause__, IndependentCCAReaderCatalogError)
            assert "EXECUTE" in str(denied.value.__cause__)
        finally:
            _revoke_public(uncooperative, value)


def test_uncooperative_catalog_grant_during_consumer_is_detected_on_exit(
    isolated_cca_read,
):
    value = isolated_cca_read
    pin, expected = _synthetic_test_pin(value), _expected_fixture(value)
    grant_committed = False
    with (
        psycopg.connect(value.dsn, autocommit=True) as uncooperative,
        psycopg.connect(value.login("reader"), autocommit=True) as reader,
    ):
        try:
            with pytest.raises(TestOnlyCloseLeaseError, match="catalog drifted") as denied:
                with reader.transaction():
                    reader.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
                    with hold_synthetic_committed_close(
                        reader, catalog_pin=pin, expected=expected
                    ) as evidence:
                        assert evidence.close_lease_qualified is False
                        _grant_public(uncooperative, value)
                        assert (
                            uncooperative.info.transaction_status
                            == psycopg.pq.TransactionStatus.IDLE
                        )
                        grant_committed = True
            assert grant_committed, "administrator GRANT must commit before reinspection"
            assert isinstance(denied.value.__cause__, IndependentCCAReaderCatalogError)
            assert "EXECUTE" in str(denied.value.__cause__)
        finally:
            _revoke_public(uncooperative, value)


def test_production_cca_import_graph_excludes_synthetic_catalog_gate():
    pending = ["bot_core.independent_cca_decision"]
    visited = set()
    forbidden = {
        "bot_core.test_only_cca_catalog_ddl_gate",
        "bot_core.test_only_independent_cca_close_lease",
    }
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        assert module not in forbidden, "production CCA imports synthetic DDL gate"
        pending.extend(_repository_dependencies(module) - visited)
