"""Unit fault injection for the disposable TEST migration connection helper."""

from __future__ import annotations

from types import SimpleNamespace

import psycopg
import pytest

from tests.security import test_postgresql_production_root_lifecycle_qualification as fixture


class _Result:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class _TestDatabase:
    database = "isolated_test_database"
    owner_pid = 101

    def __init__(self, *, replacement_workers=0, termination=True, exits_before_signal=False):
        self.allow_connections = True
        self.sessions = {101: self.database, 102: self.database, 201: "another_database"}
        self.replacement_workers = replacement_workers
        self.next_worker_pid = 103
        self.termination = termination
        self.exits_before_signal = exits_before_signal
        self.calls = []
        self.drain_count = 0
        self.snapshot = None
        self.owner_closed = False
        self.fail_gate_close = False
        self.fail_gate_restore = False

    def connect(self, dsn, *, autocommit):
        assert autocommit is True
        return _Connection(self, owner=dsn == "test-dsn")


class _Connection:
    def __init__(self, state, *, owner):
        self.state = state
        self.owner = owner
        self.info = SimpleNamespace(backend_pid=state.owner_pid if owner else 301)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.owner:
            self.state.owner_closed = True

    def execute(self, statement, parameters=None):
        query = statement if isinstance(statement, str) else statement.as_string(None)
        state = self.state
        state.calls.append((query, parameters))
        if query.startswith("ALTER DATABASE"):
            assert f'"{state.database}"' in query
            if query.endswith("=true") and state.fail_gate_restore:
                raise psycopg.OperationalError("restoring gate failed")
            state.allow_connections = query.endswith("=true")
            if query.endswith("=false") and state.fail_gate_close:
                raise psycopg.OperationalError("lost committed gate acknowledgement")
            return _Result([])
        if "set_config" in query:
            assert 0 < int(parameters[0][:-2]) <= 5000
            return _Result([(parameters[0],)])
        if "pg_stat_clear_snapshot" in query:
            state.snapshot = None
            return _Result([(None,)])
        if "pg_terminate_backend" in query:
            assert query.endswith("WHERE datname=%s AND pid<>%s")
            timeout_ms, database, owner_pid = parameters
            assert 0 < timeout_ms <= 5000
            assert database == state.database
            assert owner_pid == state.owner_pid
            state.drain_count += 1
            targets = [
                pid
                for pid, database in state.sessions.items()
                if database == state.database and pid != state.owner_pid
            ]
            # Model activity cached by the drain statement itself. A fresh
            # verification must discard it, even on a transaction wrapper.
            state.snapshot = [(pid,) for pid in targets]
            if isinstance(state.termination, Exception):
                raise state.termination
            if state.termination is True or state.exits_before_signal:
                for pid in targets:
                    del state.sessions[pid]
                if state.replacement_workers:
                    state.sessions[state.next_worker_pid] = state.database
                    state.next_worker_pid += 1
                    state.replacement_workers -= 1
            return _Result([(pid, state.termination) for pid in targets])
        if "pg_stat_activity" in query:
            assert query.endswith("WHERE datname=%s AND pid<>%s")
            assert parameters == (state.database, state.owner_pid)
            if state.snapshot is None:
                state.snapshot = [
                    (pid,)
                    for pid, database in state.sessions.items()
                    if database == state.database and pid != state.owner_pid
                ]
            return _Result(state.snapshot)
        raise AssertionError(f"unexpected TEST helper query: {query}")


def _boundary(monkeypatch, **kwargs):
    state = _TestDatabase(**kwargs)
    monkeypatch.setattr(fixture.psycopg, "connect", state.connect)
    return state, SimpleNamespace(dsn="test-dsn", database=state.database)


def test_drain_removes_only_test_sessions_and_refreshes_activity(monkeypatch):
    state, boundary = _boundary(monkeypatch)
    with fixture._offline_migration_connection(boundary) as owner:
        assert owner.info.backend_pid == state.owner_pid
        assert state.sessions == {101: state.database, 201: "another_database"}
        assert state.allow_connections is False
        assert state.owner_closed is False
    assert state.owner_closed is True
    assert state.allow_connections is True
    assert state.drain_count == 1


def test_late_test_worker_is_drained_before_yield(monkeypatch):
    state, boundary = _boundary(monkeypatch, replacement_workers=1)
    with fixture._offline_migration_connection(boundary):
        assert state.sessions == {101: state.database, 201: "another_database"}
        assert state.drain_count == 2
    assert state.allow_connections is True


def test_false_termination_requires_fresh_proof_sampled_session_exited(monkeypatch):
    state, boundary = _boundary(monkeypatch, termination=False, exits_before_signal=True)
    with fixture._offline_migration_connection(boundary):
        assert state.sessions == {101: state.database, 201: "another_database"}
    assert state.drain_count == 1
    assert state.allow_connections is True


def test_departed_sampled_pid_does_not_hide_a_different_late_worker(monkeypatch):
    state, boundary = _boundary(
        monkeypatch, termination=False, exits_before_signal=True, replacement_workers=1
    )
    with fixture._offline_migration_connection(boundary):
        assert state.sessions == {101: state.database, 201: "another_database"}
        assert state.drain_count == 2
    assert state.allow_connections is True


@pytest.mark.parametrize(
    "termination",
    [False, None, psycopg.errors.InsufficientPrivilege(), psycopg.errors.QueryCanceled()],
)
def test_termination_failure_refuses_and_restores_gate(monkeypatch, termination):
    state, boundary = _boundary(monkeypatch, termination=termination)
    error = type(termination) if isinstance(termination, Exception) else RuntimeError
    with pytest.raises(error):
        with fixture._offline_migration_connection(boundary):
            pytest.fail("a failed drain must never yield a migration connection")
    assert state.sessions == {101: state.database, 102: state.database, 201: "another_database"}
    assert state.allow_connections is True
    assert state.owner_closed is True
    assert state.drain_count == 1


def test_persistent_test_sessions_refuse_after_bounded_passes(monkeypatch):
    state, boundary = _boundary(monkeypatch, replacement_workers=10)
    with pytest.raises(RuntimeError, match="still has active sessions"):
        with fixture._offline_migration_connection(boundary):
            pytest.fail("remaining sessions must never be ignored")
    assert state.drain_count == 3
    assert state.sessions == {101: state.database, 105: state.database, 201: "another_database"}
    assert state.allow_connections is True


def test_drain_deadline_refuses_without_another_termination(monkeypatch):
    state, boundary = _boundary(monkeypatch, replacement_workers=10)
    clock = iter((0.0, 0.0, 5.0))
    monkeypatch.setattr(fixture, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    with pytest.raises(RuntimeError, match="deadline"):
        with fixture._offline_migration_connection(boundary):
            pytest.fail("expired maintenance budget must refuse")
    assert state.drain_count == 1
    assert state.allow_connections is True


@pytest.mark.parametrize("clock_values", [(0.0, 0.0, 5.0), (0.0, 0.0, 0.0, 5.0)])
def test_expired_budget_never_yields_even_after_last_session_exits(monkeypatch, clock_values):
    state, boundary = _boundary(monkeypatch)
    clock = iter(clock_values)
    monkeypatch.setattr(fixture, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    with pytest.raises(RuntimeError):
        with fixture._offline_migration_connection(boundary):
            pytest.fail("an empty result after the deadline must not yield")
    assert state.sessions == {101: state.database, 201: "another_database"}
    assert state.drain_count == 1
    assert state.allow_connections is True


def test_body_failure_restores_gate_and_closes_owner(monkeypatch):
    state, boundary = _boundary(monkeypatch)
    with pytest.raises(ValueError, match="migration failed"):
        with fixture._offline_migration_connection(boundary):
            raise ValueError("migration failed")
    assert state.allow_connections is True
    assert state.owner_closed is True


def test_lost_gate_close_acknowledgement_still_restores_connections(monkeypatch):
    state, boundary = _boundary(monkeypatch)
    state.fail_gate_close = True
    with pytest.raises(psycopg.OperationalError, match="lost committed gate"):
        with fixture._offline_migration_connection(boundary):
            pytest.fail("a failed gate operation must never yield")
    assert state.allow_connections is True
    assert state.owner_closed is True
    assert state.drain_count == 0


def test_gate_restore_error_is_not_hidden(monkeypatch):
    state, boundary = _boundary(monkeypatch)
    state.fail_gate_restore = True
    with pytest.raises(psycopg.OperationalError, match="restoring gate failed"):
        with fixture._offline_migration_connection(boundary):
            assert state.allow_connections is False
    assert state.allow_connections is False
    assert state.owner_closed is True


def test_negative_migration_precondition_can_retain_existing_sessions(monkeypatch):
    state, boundary = _boundary(monkeypatch)
    with fixture._offline_migration_connection(boundary, drain_existing=False):
        assert state.sessions[102] == state.database
        assert state.allow_connections is False
    assert state.drain_count == 0
    assert state.allow_connections is True
