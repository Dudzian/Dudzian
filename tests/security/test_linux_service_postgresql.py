"""Negatywne granice adaptera; atrapy nie kwalifikują usług produktu."""

from __future__ import annotations

import copy
import os
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import psycopg
import pytest
from psycopg.pq import TransactionStatus

from bot_core import linux_service_postgresql as adapter
from deployment.linux_physical_postgresql import (
    PostgreSQLProbeError,
    PostgreSQLQualificationBlocked,
    require_service_qualification,
    validate_service_connection_observation,
)


@pytest.fixture
def service(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("PG"):
            monkeypatch.delenv(name)
    monkeypatch.setattr(adapter.sys, "platform", "linux")
    monkeypatch.setattr(adapter.os, "getresuid", lambda: (1001,) * 3, raising=False)
    monkeypatch.setattr(adapter.os, "getresgid", lambda: (1002,) * 3, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "pwd",
        SimpleNamespace(getpwuid=lambda _: SimpleNamespace(pw_name="os_freshness_crypto_verifier")),
    )
    process = {
        "pid": 1234,
        "uid": 1001,
        "gid": 1002,
        "groups": [1002],
        "start_time_ticks": 999,
        "exe_sha256": "e" * 64,
        "cgroup": "/system.slice/verifier.service",
    }
    monkeypatch.setattr(adapter, "observe_current_process", lambda: copy.deepcopy(process))

    class Connection:
        def __init__(self):
            self.closed = False
            self.autocommit = False
            self.info = SimpleNamespace(transaction_status=TransactionStatus.IDLE, backend_pid=4321)
            self.identity = (
                "peer:os_freshness_crypto_verifier",
                "freshness_crypto_verifier",
                "freshness_crypto_verifier",
                4321,
                "freshness_gate",
                None,
                None,
                "on",
                False,
            )
            self.backend = (
                4321,
                "freshness_crypto_verifier",
                "freshness_gate",
                None,
                -1,
                "client backend",
            )
            self.calls = []
            self.denial = psycopg.errors.InsufficientPrivilege()
            self.rolled_back = False
            self.result = None
            self.failure = None

        def execute(self, sql):
            self.calls.append(sql)
            if self.failure is not None:
                raise self.failure
            if sql in (*adapter.DENIAL_SQL, adapter.OPPOSITE_ROLE_SQL["verifier"]):
                if self.denial:
                    raise self.denial
                self.result = (None,)
            elif sql == adapter.IDENTITY_SQL:
                self.result = self.identity
            elif sql == adapter.BACKEND_SQL:
                self.result = self.backend
            elif sql == adapter.ALLOWED_SQL:
                self.result = (1,)
            return self

        def fetchone(self):
            return self.result

        def transaction(self):
            return nullcontext()

        def rollback(self):
            self.rolled_back = True

    monkeypatch.setattr(adapter.psycopg, "Connection", Connection)
    connection = Connection()
    arguments = dict(
        challenge_id="a" * 64,
        installation_id="b" * 64,
        role="verifier",
        expected_uid=1001,
        expected_gid=1002,
        expected_os_principal="os_freshness_crypto_verifier",
        expected_database_role="freshness_crypto_verifier",
        expected_executable_sha256="e" * 64,
        expected_cgroup=process["cgroup"],
    )
    return SimpleNamespace(connection=connection, arguments=arguments, process=process)


def observe(service):
    return adapter.observe_service_connection(service.connection, **service.arguments)


def replay(service, report):
    validate_service_connection_observation(
        report,
        challenge_id=service.arguments["challenge_id"],
        installation_id=service.arguments["installation_id"],
        role="verifier",
        expected_process=service.process,
        expected_os_principal=service.arguments["expected_os_principal"],
        expected_database_role=service.arguments["expected_database_role"],
    )


def test_service_owned_connection_has_raw_peer_backend_and_real_denial_checks(service):
    report = observe(service)
    assert service.connection.rolled_back
    assert report["scope"] == "SERVICE_OWNED_CONNECTION_OBSERVATION"
    assert report["process"] == service.process
    assert len(report["denials"]) == 4
    assert service.connection.calls[0] == "SET TRANSACTION READ ONLY"
    replay(service, report)
    with pytest.raises(PostgreSQLQualificationBlocked):
        require_service_qualification(report)


@pytest.mark.parametrize(
    "field,value",
    [
        (0, None),
        (0, "trust:os_freshness_crypto_verifier"),
        (0, "peer:root"),
        (1, "postgres"),
        (2, "postgres"),
        (3, 999),
        (3, True),
        (4, "another_database"),
        (5, "127.0.0.1"),
        (6, "127.0.0.1"),
        (7, "off"),
        (8, True),
    ],
)
def test_wrong_peer_mapping_role_or_backend_refuses(service, field, value):
    row = list(service.connection.identity)
    row[field] = value
    service.connection.identity = tuple(row)
    with pytest.raises(adapter.ServicePostgreSQLObservationError, match="AUTHENTICATED_PEER"):
        observe(service)
    assert service.connection.rolled_back


@pytest.mark.parametrize(
    "error",
    [
        None,
        psycopg.errors.ReadOnlySqlTransaction(),
        psycopg.errors.QueryCanceled(),
        psycopg.errors.UndefinedTable(),
        psycopg.OperationalError(),
    ],
)
def test_only_sqlstate_42501_counts_as_denial(service, error):
    service.connection.denial = error
    with pytest.raises(adapter.ServicePostgreSQLObservationError):
        observe(service)
    assert service.connection.rolled_back


@pytest.mark.parametrize(
    "field,value",
    [
        ("expected_uid", 0),
        ("expected_gid", True),
        ("expected_executable_sha256", "z" * 64),
        ("challenge_id", "old"),
        ("installation_id", "old"),
        ("role", "administrator"),
        ("expected_cgroup", "relative"),
    ],
)
def test_invalid_construction_policy_never_executes_sql(service, field, value):
    service.arguments[field] = value
    with pytest.raises(adapter.ServicePostgreSQLObservationError):
        observe(service)
    assert not service.connection.calls


def test_existing_business_transaction_is_never_rolled_back(service):
    service.connection.info.transaction_status = TransactionStatus.INTRANS
    with pytest.raises(adapter.ServicePostgreSQLObservationError, match="IDLE_SERVICE"):
        observe(service)
    assert not service.connection.rolled_back
    assert not service.connection.calls


@pytest.mark.parametrize("name", ["PGHOST", "PGPASSWORD", "PGSERVICE", "PGOPTIONS"])
def test_ambient_libpq_settings_refuse_before_sql(service, monkeypatch, name):
    monkeypatch.setenv(name, "niejawna-konfiguracja-testowa")
    with pytest.raises(adapter.ServicePostgreSQLObservationError, match="AMBIENT_LIBPQ"):
        observe(service)
    assert not service.connection.calls


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_foreign_platform_refuses_before_native_accounts_or_sql(service, monkeypatch, platform):
    monkeypatch.setattr(adapter.sys, "platform", platform)
    with pytest.raises(adapter.ServicePostgreSQLObservationError, match="NATIVE_LINUX"):
        observe(service)
    assert not service.connection.calls


@pytest.mark.parametrize(
    "field,value",
    [("uid", 0), ("gid", 0), ("exe_sha256", "f" * 64), ("cgroup", "/foreign.service")],
)
def test_initial_process_identity_program_and_cgroup_must_match_policy(service, field, value):
    service.process[field] = value
    with pytest.raises(adapter.ServicePostgreSQLObservationError, match="SERVICE_PROCESS_MISMATCH"):
        observe(service)
    assert not service.connection.calls


def test_process_reuse_program_or_cgroup_swap_refuses(service, monkeypatch):
    changed = {**service.process, "start_time_ticks": service.process["start_time_ticks"] + 1}
    values = iter((service.process, changed))
    monkeypatch.setattr(adapter, "observe_current_process", lambda: next(values))
    with pytest.raises(adapter.ServicePostgreSQLObservationError, match="SERVICE_PROCESS_CHANGED"):
        observe(service)


@pytest.mark.parametrize(
    "field,value",
    [
        ("challenge_id", "c" * 64),
        ("installation_id", "c" * 64),
        ("role", "runtime"),
        ("backend_row", []),
        ("identity_row", []),
        ("denials", []),
        ("allowed", {"sql": adapter.ALLOWED_SQL, "row": [True]}),
        ("transaction", "COMMITTED"),
    ],
)
def test_independent_replay_refuses_evidence_swaps(service, field, value):
    report = observe(service)
    report[field] = value
    with pytest.raises(PostgreSQLProbeError):
        replay(service, report)


def test_independent_replay_uses_external_process_not_json_claim(service):
    report = observe(service)
    report["process"]["start_time_ticks"] += 1
    with pytest.raises(PostgreSQLProbeError):
        replay(service, report)


def test_driver_errors_are_redacted_and_rollback_is_required(service):
    service.connection.failure = psycopg.OperationalError("TAJNY_KOMUNIKAT_TESTOWY")
    with pytest.raises(adapter.ServicePostgreSQLObservationError) as caught:
        observe(service)
    assert "TAJNY" not in str(caught.value)
    assert service.connection.rolled_back


@pytest.mark.skipif(sys.platform != "linux", reason="Rzeczywista obserwacja procfs wymaga Linux")
def test_current_process_identity_is_real():
    result = adapter.observe_current_process()
    assert result["pid"] == os.getpid()
    assert result["uid"] == os.getuid()
    assert result["gid"] == os.getgid()
    assert result["start_time_ticks"] > 0
    assert result["cgroup"].startswith("/")
