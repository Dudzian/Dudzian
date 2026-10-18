"""A transaction token cannot substitute identity or survive an early COMMIT."""

from __future__ import annotations

from dataclasses import replace

import psycopg
import pytest

from bot_core import postgresql_atomic_root_proof_transaction as transaction
from tests.security.test_postgresql_checkpoint_authority import BASE_DSN


@pytest.mark.parametrize("connection", [None, object(), {"transaction_isolation": "serializable"}])
def test_caller_claims_do_not_establish_a_live_transaction_owner(connection):
    with pytest.raises(transaction.AtomicRootProofTransactionError, match="OPEN_FINAL_TRANSACTION"):
        transaction.capture_atomic_root_proof_transaction_owner(
            connection, expected_login="issuer_writer"
        )


@pytest.fixture
def live_connection():
    try:
        connection = psycopg.connect(BASE_DSN, autocommit=True)
    except psycopg.Error:
        pytest.skip("isolated PostgreSQL is unavailable")
    try:
        if connection.info.server_version // 10000 != 16:
            pytest.skip("isolated PostgreSQL 16 required")
        yield connection
    finally:
        connection.close()


def _begin(connection, mode="SERIALIZABLE"):
    connection.execute("BEGIN ISOLATION LEVEL " + mode)
    return connection.execute("SELECT session_user::text").fetchone()[0]


@pytest.mark.external_postgresql
def test_observation_keeps_the_same_transaction_open_and_does_not_commit(live_connection):
    connection = live_connection
    login = _begin(connection)
    connection.execute("CREATE TEMPORARY TABLE root_proof_owner_probe(value integer)")
    owner = transaction.capture_atomic_root_proof_transaction_owner(
        connection, expected_login=login
    )
    connection.execute("INSERT INTO root_proof_owner_probe VALUES(1)")
    transaction.validate_atomic_root_proof_transaction_owner(connection, owner)
    assert connection.info.transaction_status is psycopg.pq.TransactionStatus.INTRANS
    assert "authorization=False" in repr(owner)
    connection.rollback()
    assert connection.execute(
        "SELECT pg_catalog.to_regclass('pg_temp.root_proof_owner_probe')"
    ).fetchone() == (None,)


@pytest.mark.external_postgresql
@pytest.mark.parametrize("mode", ["READ COMMITTED", "REPEATABLE READ", "SERIALIZABLE READ ONLY"])
def test_final_owner_refuses_read_only_and_weaker_isolation(live_connection, mode):
    login = _begin(live_connection, mode)
    with pytest.raises(transaction.AtomicRootProofTransactionError, match="WRITABLE_SERIALIZABLE"):
        transaction.capture_atomic_root_proof_transaction_owner(
            live_connection, expected_login=login
        )


@pytest.mark.external_postgresql
def test_early_commit_then_new_transaction_cannot_reuse_old_owner(live_connection):
    connection = live_connection
    login = _begin(connection)
    owner = transaction.capture_atomic_root_proof_transaction_owner(
        connection, expected_login=login
    )
    connection.commit()
    with pytest.raises(transaction.AtomicRootProofTransactionError, match="OPEN_FINAL_TRANSACTION"):
        transaction.validate_atomic_root_proof_transaction_owner(connection, owner)
    _begin(connection)
    with pytest.raises(transaction.AtomicRootProofTransactionError, match="OWNER_CHANGED"):
        transaction.validate_atomic_root_proof_transaction_owner(connection, owner)


@pytest.mark.external_postgresql
def test_same_database_and_login_on_another_backend_cannot_replace_connection(live_connection):
    login = _begin(live_connection)
    owner = transaction.capture_atomic_root_proof_transaction_owner(
        live_connection, expected_login=login
    )
    with psycopg.connect(BASE_DSN, autocommit=True) as other:
        _begin(other)
        with pytest.raises(transaction.AtomicRootProofTransactionError, match="EXACT_OBSERVED"):
            transaction.validate_atomic_root_proof_transaction_owner(other, owner)


@pytest.mark.external_postgresql
def test_wrong_direct_login_and_caller_constructed_tokens_fail_closed(live_connection):
    login = _begin(live_connection)
    with pytest.raises(transaction.AtomicRootProofTransactionError, match="WRITABLE_SERIALIZABLE"):
        transaction.capture_atomic_root_proof_transaction_owner(
            live_connection, expected_login=login + "_other"
        )
    owner = transaction.capture_atomic_root_proof_transaction_owner(
        live_connection, expected_login=login
    )
    forged = replace(owner, _origin=None)
    with pytest.raises(transaction.AtomicRootProofTransactionError, match="EXACT_OBSERVED"):
        transaction.validate_atomic_root_proof_transaction_owner(live_connection, forged)


@pytest.mark.external_postgresql
def test_role_switch_during_final_transaction_fails_closed(live_connection):
    connection = live_connection
    login = _begin(connection)
    owner = transaction.capture_atomic_root_proof_transaction_owner(
        connection, expected_login=login
    )
    if not connection.execute(
        "SELECT rolsuper FROM pg_catalog.pg_roles WHERE rolname=session_user"
    ).fetchone()[0]:
        pytest.skip("isolated administrative fixture required for role substitution refusal")
    # Deliberate adversarial substitution: this never grants a production path.
    connection.execute("SET LOCAL ROLE pg_read_all_data")
    with pytest.raises(transaction.AtomicRootProofTransactionError, match="WRITABLE_SERIALIZABLE"):
        transaction.validate_atomic_root_proof_transaction_owner(connection, owner)
