"""Real PostgreSQL transaction boundaries for atomic authority qualification."""

from __future__ import annotations

from dataclasses import replace

import psycopg
import pytest

from bot_core import postgresql_atomic_root_proof_commit as runtime
from bot_core import postgresql_atomic_root_proof_qualification as qualification
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from tests.security.test_atomic_root_proof_commit_contract import _deployment
from tests.security.test_postgresql_checkpoint_authority import BASE_DSN

pytestmark = pytest.mark.external_postgresql


@pytest.fixture
def live_connection():
    try:
        conn = psycopg.connect(BASE_DSN, autocommit=True)
    except psycopg.Error:
        pytest.skip("isolated PostgreSQL is unavailable")
    try:
        if conn.info.server_version // 10000 != 16:
            pytest.skip("isolated PostgreSQL 16 required")
        yield conn
    finally:
        conn.close()


def test_autocommit_qualification_refuses_before_ineffective_set_local(live_connection):
    notices = []
    live_connection.add_notice_handler(lambda notice: notices.append(notice.message_primary))
    with pytest.raises(
        qualification.AtomicRootProofQualificationError, match="explicit qualification transaction"
    ):
        qualification.qualify_atomic_root_proof_authority(
            live_connection, _deployment(administrative=True)
        )
    assert live_connection.info.transaction_status is psycopg.pq.TransactionStatus.IDLE
    assert notices == []


def test_default_connection_begins_implicit_qualification_transaction(live_connection):
    conn = live_connection
    original = conn.execute("SHOW search_path").fetchone()
    conn.autocommit = False
    notices = []
    conn.add_notice_handler(lambda notice: notices.append(notice.message_primary))
    with pytest.raises(qualification.AtomicRootProofQualificationError, match="direct-login"):
        qualification.qualify_atomic_root_proof_authority(conn, _deployment(administrative=True))
    assert conn.info.transaction_status is psycopg.pq.TransactionStatus.INTRANS
    assert conn.execute("SHOW search_path").fetchone() == ("pg_catalog",)
    conn.rollback()
    assert conn.info.transaction_status is psycopg.pq.TransactionStatus.IDLE
    conn.autocommit = True
    assert conn.execute("SHOW search_path").fetchone() == original
    assert notices == []


@pytest.mark.parametrize("autocommit", [True, False])
@pytest.mark.parametrize("rollback_savepoint", [True, False])
def test_local_path_is_effective_and_caller_transaction_stays_owned(
    live_connection, autocommit, rollback_savepoint
):
    conn = live_connection
    conn.autocommit = autocommit
    notices = []
    conn.add_notice_handler(lambda notice: notices.append(notice.message_primary))
    with conn.transaction():
        original = conn.execute("SHOW search_path").fetchone()
        xid = conn.execute("SELECT pg_catalog.pg_current_xact_id()::text").fetchone()
        with conn.transaction(force_rollback=rollback_savepoint):
            # This deliberately uninstalled authority must still refuse the
            # wrong direct login. SET LOCAL must first take effect in this tx.
            with pytest.raises(
                qualification.AtomicRootProofQualificationError, match="direct-login"
            ):
                qualification.qualify_atomic_root_proof_authority(
                    conn, _deployment(administrative=True)
                )
            assert conn.execute("SHOW search_path").fetchone() == ("pg_catalog",)
            assert conn.info.transaction_status is psycopg.pq.TransactionStatus.INTRANS
            assert conn.execute("SELECT pg_catalog.pg_current_xact_id()::text").fetchone() == xid
        assert conn.execute("SHOW search_path").fetchone() == (
            original if rollback_savepoint else ("pg_catalog",)
        )
        assert conn.execute("SELECT pg_catalog.pg_current_xact_id()::text").fetchone() == xid
    assert conn.info.transaction_status is psycopg.pq.TransactionStatus.IDLE
    # A new transaction sees the original session path, without leakage.
    with conn.transaction():
        assert conn.execute("SHOW search_path").fetchone() == original
    assert notices == []


def test_constructor_qualifies_inside_an_explicit_transaction(live_connection, monkeypatch):
    notices = []
    observations = []
    original_qualify = qualification.qualify_atomic_root_proof_authority
    original_connect = psycopg.connect

    def connect(*args, **kwargs):
        conn = original_connect(*args, **kwargs)
        conn.add_notice_handler(lambda notice: notices.append(notice.message_primary))
        return conn

    def qualify(conn, deployment):
        observations.append(conn.info.transaction_status)
        return original_qualify(conn, deployment)

    monkeypatch.setattr(runtime.psycopg, "connect", connect)
    monkeypatch.setattr(runtime, "qualify_atomic_root_proof_authority", qualify)
    deployment = replace(
        _deployment(administrative=True), writer_connection=PostgreSQLConnectionConfig(BASE_DSN)
    )
    with pytest.raises(qualification.AtomicRootProofQualificationError, match="direct-login"):
        runtime.PostgreSQLAtomicRootProofCommit(deployment)
    assert observations == [psycopg.pq.TransactionStatus.INTRANS]
    assert notices == []
