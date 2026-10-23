"""SQLite membership handles must close without depending on garbage collection."""

import sqlite3

import pytest

from bot_core.instruments import source_producer_membership as membership
from bot_core.instruments.core_time import ProductionCoreClock
from bot_core.instruments.testing_core_time import TestSQLiteMembershipCarrier


@pytest.fixture
def recorded_connections(monkeypatch):
    connect = sqlite3.connect
    opened = []

    def record(*args, **kwargs):
        connection = connect(*args, **kwargs)
        # Retain strong references: passing this regression must never depend
        # on CPython reference counting or the cyclic collector running first.
        opened.append(connection)
        return connection

    monkeypatch.setattr(membership.sqlite3, "connect", record)
    yield opened
    for connection in opened:
        connection.close()


def assert_closed(connections):
    assert connections
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connection.execute("SELECT 1")


def test_initialize_read_and_reopen_close_connections(tmp_path, recorded_connections):
    path = tmp_path / "memberships.jsonl"
    carrier = membership.SQLiteMembershipCarrier(path)
    assert_closed(recorded_connections)
    assert carrier.read() == ()
    assert_closed(recorded_connections)
    restored = membership.SQLiteMembershipCarrier.open_existing(path)
    assert restored.read() == ()
    assert_closed(recorded_connections)
    # Windows also requires these handles to be closed before temp cleanup.
    path.unlink()
    assert not path.exists()


def test_append_and_existing_decisions_close_connections(
    tmp_path, recorded_connections, monkeypatch
):
    monkeypatch.setattr(ProductionCoreClock, "now_utc", lambda self: "2026-09-14T00:00:00Z")
    carrier = membership.SQLiteMembershipCarrier(tmp_path / "memberships.sqlite3")
    authority = membership.SourceProducerMembershipAuthority(carrier)
    grant = authority.admit_release_grant("core_release_1_45_binance_spot")
    assert grant is not None
    assert_closed(recorded_connections)
    assert authority.admit_release_grant("core_release_1_45_binance_spot") == grant
    assert_closed(recorded_connections)
    assert carrier.read() == (grant.to_mapping(),)
    assert_closed(recorded_connections)


def test_denied_and_failed_append_close_and_rollback(tmp_path, recorded_connections):
    carrier = membership.SQLiteMembershipCarrier(tmp_path / "memberships.sqlite3")

    def denied(_records, _proposed):
        return "DENY", None

    def fail(_records, _proposed):
        raise ValueError("synthetic admission failure")

    assert carrier._transact_append(lambda: {}, denied) is None
    assert_closed(recorded_connections)
    with pytest.raises(ValueError, match="synthetic admission failure"):
        carrier._transact_append(lambda: {}, fail)
    assert_closed(recorded_connections)
    assert carrier.read() == ()
    assert_closed(recorded_connections)


def test_wrong_domain_initialization_and_reopen_close_on_error(tmp_path, recorded_connections):
    path = tmp_path / "test-domain.sqlite3"
    TestSQLiteMembershipCarrier(path)
    assert_closed(recorded_connections)
    for operation in (
        membership.SQLiteMembershipCarrier,
        membership.SQLiteMembershipCarrier.open_existing,
    ):
        with pytest.raises(ValueError, match="MEMBERSHIP_AUTHORITY_DOMAIN_MISMATCH"):
            operation(path)
        assert_closed(recorded_connections)


def test_connection_setup_closes_if_pragma_fails(tmp_path, monkeypatch):
    connect = sqlite3.connect
    opened = []

    class RejectingPragma(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql == "PRAGMA journal_mode = WAL":
                raise sqlite3.OperationalError("synthetic setup failure")
            return super().execute(sql, parameters)

    def record(*args, **kwargs):
        connection = connect(*args, factory=RejectingPragma, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(membership.sqlite3, "connect", record)
    try:
        with pytest.raises(sqlite3.OperationalError, match="synthetic setup failure"):
            membership.SQLiteMembershipCarrier(tmp_path / "memberships.sqlite3")
        assert_closed(opened)
    finally:
        for connection in opened:
            connection.close()
