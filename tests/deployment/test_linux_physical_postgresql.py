"""Odczytowe sondy nie są dowodem połączenia właściwego procesu produktu."""

from __future__ import annotations

from contextlib import nullcontext
import copy
import os
import sys
from types import SimpleNamespace

import psycopg
import pytest

from deployment import linux_physical_postgresql as pg


@pytest.mark.skipif(sys.platform != "linux", reason="Odczyt rzeczywistego /proc wymaga Linux")
def test_live_process_read_observes_current_pid_and_program_without_mutation():
    observation = pg._process()
    assert observation["pid"] == os.getpid()
    assert observation["uid"] == os.getuid()
    assert observation["gid"] == os.getgid()
    assert observation["start_time_ticks"] > 0
    assert len(observation["exe_sha256"]) == 64


@pytest.fixture
def probe(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("PG"):
            monkeypatch.delenv(name)
    monkeypatch.setattr(pg.sys, "platform", "linux")
    monkeypatch.setattr(pg.os, "getresuid", lambda: (1001, 1001, 1001), raising=False)
    monkeypatch.setattr(pg.os, "getresgid", lambda: (1002, 1002, 1002), raising=False)
    monkeypatch.setattr(pg.os, "getgroups", lambda: [1002, 1003], raising=False)
    monkeypatch.setattr(pg.os, "getpid", lambda: 1234)
    accounts = SimpleNamespace(
        getpwuid=lambda uid: SimpleNamespace(pw_uid=uid, pw_name="ch_runtime")
    )
    monkeypatch.setitem(sys.modules, "pwd", accounts)
    process = {
        "pid": 1234,
        "start_time_ticks": 123456,
        "uid": 1001,
        "gid": 1002,
        "groups": [1002, 1003],
        "no_new_privs": 1,
        "exe_sha256": "e" * 64,
        "capabilities": {"CapEff": "0000000000000000"},
    }
    monkeypatch.setattr(pg, "_process", lambda: copy.deepcopy(process))
    identity = ("peer:ch_runtime", "db_runtime", "db_runtime", 4321, None, None, "on", False)

    class Connection:
        def __init__(self):
            self.identity = identity
            self.denial_error = psycopg.errors.InsufficientPrivilege("odmowa testowa")
            self.calls = []
            self.rollback_count = 0
            self.closed = False
            self.transactions = 0

        def execute(self, sql):
            self.calls.append(sql)
            if sql == pg.DENIAL_SQL and self.denial_error is not None:
                raise self.denial_error
            return self

        def fetchone(self):
            return self.identity

        def transaction(self):
            self.transactions += 1
            return nullcontext()

        def rollback(self):
            self.rollback_count += 1

        def close(self):
            self.closed = True

    connection = Connection()
    connect_calls = []

    def connect(**kwargs):
        connect_calls.append(kwargs)
        return connection

    monkeypatch.setattr(psycopg, "connect", connect)
    arguments = {
        "config": {"socket": "/run/postgresql", "port": 5432, "database": "cryptohunter"},
        "role": "runtime",
        "expected_uid": 1001,
        "expected_gid": 1002,
        "expected_groups": [1002, 1003],
        "expected_os_principal": "ch_runtime",
        "expected_database_role": "db_runtime",
        "expected_executable_sha256": "e" * 64,
        "challenge_id": "a" * 64,
    }
    return SimpleNamespace(
        arguments=arguments,
        connection=connection,
        connect_calls=connect_calls,
        process=process,
        accounts=accounts,
    )


def test_real_peer_and_denied_select_are_only_diagnostic(probe):
    report = pg.collect_role_diagnostic(**probe.arguments)
    assert report["status"] == "BLOCKED"
    assert report["scope"] == "ROLE_PROBE_DIAGNOSTIC_ONLY"
    assert report["role"] == "runtime"
    assert report["challenge_id"] == "a" * 64
    assert report["process"] == probe.process
    assert report["connection"]["system_user"] == "peer:ch_runtime"
    assert report["connection"]["backend_pid"] == 4321
    assert report["denial"] == {
        "operation": "READ_AUTHID_CATALOG_WITHOUT_ROWS",
        "sqlstate": "42501",
    }
    assert probe.connection.calls == [pg.IDENTITY_SQL, pg.DENIAL_SQL]
    assert probe.connection.transactions == 1
    assert probe.connection.rollback_count == 1
    assert probe.connection.closed
    with pytest.raises(pg.PostgreSQLQualificationBlocked, match="BLOCKED"):
        pg.require_service_qualification(report)


def test_verifier_role_stays_diagnostic(probe):
    probe.arguments["role"] = "verifier"
    assert pg.collect_role_diagnostic(**probe.arguments)["status"] == "BLOCKED"


def test_connector_has_no_password_and_forces_bounded_local_readonly(probe):
    pg.collect_role_diagnostic(**probe.arguments)
    options = probe.connect_calls[0]
    assert options["host"] == "/run/postgresql"
    assert options["password"] is None
    assert options["passfile"] == "/dev/null"
    assert options["sslmode"] == "disable"
    assert options["connect_timeout"] == 5
    assert "default_transaction_read_only=on" in options["options"]
    assert "statement_timeout=5000" in options["options"]
    assert "lock_timeout=1000" in options["options"]
    assert "idle_in_transaction_session_timeout=5000" in options["options"]
    assert all(
        not sql.lstrip().startswith(("INSERT", "UPDATE", "DELETE", "SET ROLE"))
        for sql in probe.connection.calls
    )


@pytest.mark.parametrize(
    "environment",
    ["PGPASSWORD", "PGPASSFILE", "PGSERVICE", "PGSERVICEFILE", "PGOPTIONS", "PGHOSTADDR"],
)
def test_ambient_libpq_settings_refuse_before_connection(probe, monkeypatch, environment):
    monkeypatch.setenv(environment, "niejawna-konfiguracja-testowa")
    with pytest.raises(pg.PostgreSQLProbeError, match="Niejawna konfiguracja"):
        pg.collect_role_diagnostic(**probe.arguments)
    assert not probe.connect_calls


@pytest.mark.parametrize(
    "mutation",
    [
        lambda args: args.update(role="administrator"),
        lambda args: args.update(expected_uid=0),
        lambda args: args.update(expected_uid=True),
        lambda args: args.update(expected_gid=0),
        lambda args: args.update(expected_gid=False),
        lambda args: args.update(expected_groups=[True]),
        lambda args: args.update(expected_groups=[1002, 1002]),
        lambda args: args.update(expected_os_principal="runtime'; DROP DATABASE x; --"),
        lambda args: args.update(expected_database_role="runtime admin"),
        lambda args: args.update(expected_executable_sha256=""),
        lambda args: args.update(challenge_id="old"),
        lambda args: args["config"].update(socket="127.0.0.1"),
        lambda args: args["config"].update(socket="/run/../socket"),
        lambda args: args["config"].update(socket="/run/postgresql/"),
        lambda args: args["config"].update(port=True),
        lambda args: args["config"].update(port=0),
        lambda args: args["config"].update(database="another database"),
        lambda args: args["config"].update(password=None),
    ],
)
def test_malformed_scope_cannot_start_connector(probe, mutation):
    mutation(probe.arguments)
    with pytest.raises(pg.PostgreSQLProbeError):
        pg.collect_role_diagnostic(**probe.arguments)
    assert not probe.connect_calls


def test_other_platform_refuses_before_touching_os_identity(probe, monkeypatch):
    monkeypatch.setattr(pg.sys, "platform", "win32")
    with pytest.raises(pg.PostgreSQLProbeError, match="natywnego Linux"):
        pg.collect_role_diagnostic(**probe.arguments)
    assert not probe.connect_calls


@pytest.mark.parametrize("uids", [(0, 0, 0), (1001, 0, 1001), (1001, 1001, 0)])
def test_root_or_saved_root_identity_refuses(probe, monkeypatch, uids):
    monkeypatch.setattr(pg.os, "getresuid", lambda: uids)
    with pytest.raises(pg.PostgreSQLProbeError, match="nieadministracyjną"):
        pg.collect_role_diagnostic(**probe.arguments)
    assert not probe.connect_calls


@pytest.mark.parametrize("gids", [(0, 0, 0), (1002, 0, 1002), (1002, 1002, 0)])
def test_effective_or_saved_group_mismatch_refuses(probe, monkeypatch, gids):
    monkeypatch.setattr(pg.os, "getresgid", lambda: gids)
    with pytest.raises(pg.PostgreSQLProbeError, match="nieadministracyjną"):
        pg.collect_role_diagnostic(**probe.arguments)
    assert not probe.connect_calls


def test_additional_group_is_effective_privilege_drift(probe, monkeypatch):
    monkeypatch.setattr(pg.os, "getgroups", lambda: [1002, 1003, 0])
    with pytest.raises(pg.PostgreSQLProbeError, match="nieadministracyjną"):
        pg.collect_role_diagnostic(**probe.arguments)


def test_os_principal_is_independent_of_json_claim(probe, monkeypatch):
    monkeypatch.setattr(
        probe.accounts, "getpwuid", lambda uid: SimpleNamespace(pw_uid=uid, pw_name="other")
    )
    with pytest.raises(pg.PostgreSQLProbeError, match="Nazwa konta"):
        pg.collect_role_diagnostic(**probe.arguments)
    assert not probe.connect_calls


def test_missing_os_principal_refuses(probe, monkeypatch):
    def absent(uid):
        raise KeyError(uid)

    monkeypatch.setattr(probe.accounts, "getpwuid", absent)
    with pytest.raises(pg.PostgreSQLProbeError, match="konta systemowego"):
        pg.collect_role_diagnostic(**probe.arguments)


@pytest.mark.parametrize(
    "field,value",
    [("pid", 9999), ("uid", 0), ("gid", 0), ("groups", [0]), ("exe_sha256", "f" * 64)],
)
def test_process_identity_and_program_are_pinned_before_connect(probe, field, value):
    probe.process[field] = value
    with pytest.raises(pg.PostgreSQLProbeError):
        pg.collect_role_diagnostic(**probe.arguments)
    assert not probe.connect_calls


@pytest.mark.parametrize(
    "field,value",
    [
        (0, "password:ch_runtime"),
        (0, "peer:root"),
        (1, "postgres"),
        (2, "postgres"),
        (3, True),
        (3, 0),
        (4, "127.0.0.1"),
        (5, "127.0.0.1"),
        (6, "off"),
        (7, True),
        (7, 0),
    ],
)
def test_server_must_confirm_exact_peer_role_local_readonly_transport(probe, field, value):
    identity = list(probe.connection.identity)
    identity[field] = value
    probe.connection.identity = tuple(identity)
    with pytest.raises(pg.PostgreSQLProbeError, match="Serwer nie potwierdził"):
        pg.collect_role_diagnostic(**probe.arguments)
    assert probe.connection.calls == [pg.IDENTITY_SQL]
    assert probe.connection.closed


@pytest.mark.parametrize("identity", [None, (), ("peer:ch_runtime",), ["peer:ch_runtime"] * 8])
def test_incomplete_server_identity_refuses(probe, identity):
    probe.connection.identity = identity
    with pytest.raises(pg.PostgreSQLProbeError, match="Serwer nie potwierdził"):
        pg.collect_role_diagnostic(**probe.arguments)


def test_successful_unauthorized_select_refuses(probe):
    probe.connection.denial_error = None
    with pytest.raises(pg.PostgreSQLProbeError, match="Nieautoryzowany odczyt"):
        pg.collect_role_diagnostic(**probe.arguments)
    assert probe.connection.closed


@pytest.mark.parametrize(
    "error",
    [
        psycopg.errors.UndefinedTable("test"),
        psycopg.errors.ReadOnlySqlTransaction("test"),
        psycopg.OperationalError("test"),
    ],
)
def test_other_sql_error_is_not_permission_denial(probe, error):
    probe.connection.denial_error = error
    with pytest.raises(pg.PostgreSQLProbeError, match="nie wynika z uprawnień"):
        pg.collect_role_diagnostic(**probe.arguments)
    assert probe.connection.closed


def test_pid_reuse_or_executable_change_during_probe_refuses(probe, monkeypatch):
    before = copy.deepcopy(probe.process)
    after = {**before, "start_time_ticks": before["start_time_ticks"] + 1}
    observations = iter((before, after))
    monkeypatch.setattr(pg, "_process", lambda: next(observations))
    with pytest.raises(pg.PostgreSQLProbeError, match="zmienił się"):
        pg.collect_role_diagnostic(**probe.arguments)


def test_connection_error_does_not_expose_driver_message(probe, monkeypatch):
    def failed(**kwargs):
        raise psycopg.OperationalError("TAJNY_KOMUNIKAT_TESTOWY")

    monkeypatch.setattr(psycopg, "connect", failed)
    with pytest.raises(pg.PostgreSQLProbeError) as caught:
        pg.collect_role_diagnostic(**probe.arguments)
    assert "TAJNY_KOMUNIKAT_TESTOWY" not in str(caught.value)
    assert caught.value.__suppress_context__


def test_blocked_report_factory_is_not_shared_mutable_state():
    changed = pg.blocked_service_qualification()
    changed["blockers"].clear()
    assert pg.blocked_service_qualification()["blockers"] == list(pg.BLOCKERS)


def test_strict_blocked_report_validation_does_not_authorize():
    report = pg.blocked_service_qualification()
    pg.validate_blocked_service_qualification(report)
    with pytest.raises(pg.PostgreSQLQualificationBlocked):
        pg.require_service_qualification(report)


@pytest.mark.parametrize(
    "value", [None, {}, [], {"status": "PASS"}, {"service_bound": True, "signature": "signed"}]
)
def test_manual_claims_cannot_supply_missing_service_adapter(value):
    with pytest.raises(pg.PostgreSQLProbeError):
        pg.validate_blocked_service_qualification(value)
    with pytest.raises(pg.PostgreSQLQualificationBlocked):
        pg.require_service_qualification(value)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report.update(schema_version=True),
        lambda report: report.update(status="PASS"),
        lambda report: report.update(adapter_id="self-declared-adapter"),
        lambda report: report["blockers"].pop(),
        lambda report: report.update(service_bound=True),
    ],
)
def test_altered_blocked_report_refuses(mutation):
    report = pg.blocked_service_qualification()
    mutation(report)
    with pytest.raises(pg.PostgreSQLProbeError):
        pg.validate_blocked_service_qualification(report)
    with pytest.raises(pg.PostgreSQLQualificationBlocked):
        pg.require_service_qualification(report)
