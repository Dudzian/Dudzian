"""Hermetic service capture tests; these do not qualify native Windows IPC."""

from __future__ import annotations

import copy
from datetime import datetime
import json
from pathlib import PureWindowsPath
import sys
import threading
from types import SimpleNamespace
from typing import Any

import pytest
import psycopg

from deployment import windows_stage9_postgresql_passive as passive
from deployment import windows_stage9_service_sql as service_sql
from deployment.windows_installer.contract import CONTRACT


class DatabaseError(psycopg.Error):
    def __init__(self, state: str | None) -> None:
        super().__init__("secret server message and credential path")
        self.sqlstate = state


class Connection:
    def __init__(self, binding: dict[str, Any]) -> None:
        self.info = SimpleNamespace(backend_pid=321)
        self.closed = False
        self.queries: list[str] = []
        self.responses: dict[str, Any] = {
            "identity": [
                (
                    "freshness_gate",
                    binding["role"],
                    binding["role"],
                    "127.0.0.1",
                    CONTRACT.postgresql_port,
                    321,
                    datetime.fromisoformat("2026-10-16T00:00:01+00:00"),
                    170011,
                    "127.0.0.1",
                    CONTRACT.postgresql_port,
                    True,
                    True,
                    True,
                )
            ],
            "ssl": [
                (
                    True,
                    "TLSv1.3",
                    "TLS_AES_256_GCM_SHA384",
                    256,
                    f"/CN={binding['certificate_identity']}",
                )
            ],
            "data": [(r"C:\ProgramData\CryptoHunter\PostgreSQL\Data",)],
            "control": [("7448802115599552300",)],
            "hba": [],
            "roles": [],
            "membership": [(0,)],
        }

    def cursor(self) -> Any:
        connection = self

        class Cursor:
            def __enter__(self) -> Any:
                return self

            def __exit__(self, *_args: Any) -> None:
                return None

            def execute(self, query: str) -> None:
                assert query in passive._ALLOWED_SQL
                connection.queries.append(query)
                self.key = service_sql.QUERY_NAMES[query]
                if isinstance(connection.responses[self.key], Exception):
                    raise connection.responses[self.key]

            def fetchmany(self, limit: int) -> Any:
                assert 0 < limit <= 65
                return connection.responses[self.key][:limit]

        return Cursor()

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], Connection, dict[str, Any]]:
    name = CONTRACT.backend_service
    binding = {
        "name": name,
        "identity": name,
        "role": "freshness_runtime",
        "certificate_identity": name,
        "process_id": 123,
        "process_path": rf"C:\Program Files\CryptoHunter\{name}.exe",
        "process_started": "2026-10-16T00:00:00+00:00",
        "token_sid": "S-1-5-80-111",
        "paths": {
            "program_files": r"C:\Program Files\CryptoHunter",
            "program_data": r"C:\ProgramData\CryptoHunter",
        },
    }
    connection = Connection(binding)
    request = {
        "version": 1,
        "nonce": "a" * 64,
        "collection_id": "b" * 64,
        "issued": 100.0,
        "expires": 130.0,
    }
    monkeypatch.setattr(service_sql.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(service_sql, "_own_native_binding", lambda _name: copy.deepcopy(binding))

    def connect(selected: dict[str, Any]) -> Any:
        assert selected["identity"] == name
        assert selected["paths"] == binding["paths"]
        if selected["role"] != binding["role"]:
            raise DatabaseError("28000")
        return connection

    monkeypatch.setattr(passive, "_connect", connect)
    return binding, connection, request


def test_capture_keeps_own_native_session_open_until_ack_and_reads_only_fixed_sql(
    capture: tuple[dict[str, Any], Connection, dict[str, Any]],
) -> None:
    binding, connection, request = capture
    with service_sql.collect_service_sql(binding["name"], request, threading.Event()) as payload:
        assert connection.closed is False
        assert payload["nonce"] == request["nonce"]
        assert payload["collection_id"] == request["collection_id"]
        assert payload["process"] == binding
        assert payload["backend_pid"] == 321
        assert {
            name: len(batches) for name, batches in payload["queries"].items()
        } == service_sql.QUERY_COUNTS
        assert set(payload["post_probe"]) == {"identity", "ssl", "control", "data"}
        assert all(
            result == {"status": "DENIED", "sqlstate": "28000"}
            for result in payload["probes"].values()
        )
        assert "secret" not in json.dumps(payload)
        assert payload["queries"]["identity"][0]["rows"][0][6].endswith("+00:00")
    assert connection.closed is True
    assert set(connection.queries) == passive._ALLOWED_SQL


@pytest.mark.parametrize(
    "change",
    [
        {"version": True},
        {"sql": "SELECT 1"},
        {"nonce": "A" * 64},
        {"issued": float("nan")},
        {"issued": True},
        {"issued": 10**1000},
        {"issued": -1},
        {"issued": 101},
        {"expires": float("inf")},
        {"expires": 100},
        {"expires": 131},
        {"collection_id": "short"},
    ],
)
def test_invalid_request_never_opens_credentials(
    capture: tuple[dict[str, Any], Connection, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    change: dict[str, Any],
) -> None:
    binding, connection, request = capture
    monkeypatch.setattr(passive, "_connect", lambda _binding: pytest.fail("credential access"))
    with pytest.raises(passive.PassivePostgreSQLError):
        with service_sql.collect_service_sql(
            binding["name"], {**request, **change}, threading.Event()
        ):
            pytest.fail("invalid request yielded evidence")
    assert connection.closed is False


@pytest.mark.parametrize(
    "state,expected",
    [
        ("28000", "DENIED"),
        ("28P01", "DENIED"),
        ("57014", "BLOCKED"),
        ("42501", "BLOCKED"),
        (None, "BLOCKED"),
        ("28000 credentials", "BLOCKED"),
    ],
)
def test_only_explicit_fresh_authentication_denial_counts(
    capture: tuple[dict[str, Any], Connection, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    state: str | None,
    expected: str,
) -> None:
    binding, connection, request = capture
    roles: list[str] = []

    def connect(selected: dict[str, Any]) -> Any:
        assert selected["identity"] == binding["identity"]
        roles.append(selected["role"])
        if selected["role"] == binding["role"]:
            return connection
        raise DatabaseError(state)

    monkeypatch.setattr(passive, "_connect", connect)
    with service_sql.collect_service_sql(binding["name"], request, threading.Event()) as payload:
        assert set(roles) == {"freshness_runtime", "freshness_crypto_verifier", "postgres"}
        assert all(result["status"] == expected for result in payload["probes"].values())
        assert "secret" not in json.dumps(payload)


def test_local_exception_cannot_assert_server_authentication_denial(
    capture: tuple[dict[str, Any], Connection, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding, connection, request = capture

    class LocalError(Exception):
        sqlstate = "28000"

    def connect(selected: dict[str, Any]) -> Any:
        if selected["role"] == binding["role"]:
            return connection
        raise LocalError("secret credential diagnostic")

    monkeypatch.setattr(passive, "_connect", connect)
    with service_sql.collect_service_sql(binding["name"], request, threading.Event()) as payload:
        assert all(
            result == {"status": "BLOCKED", "sqlstate": None}
            for result in payload["probes"].values()
        )


def test_unexpected_forbidden_acceptance_is_retained_and_closed(
    capture: tuple[dict[str, Any], Connection, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding, connection, request = capture
    forbidden = [Connection(binding), Connection(binding)]
    candidates = iter(forbidden)
    monkeypatch.setattr(
        passive,
        "_connect",
        lambda selected: connection if selected["role"] == binding["role"] else next(candidates),
    )
    with service_sql.collect_service_sql(binding["name"], request, threading.Event()) as payload:
        assert all(result["status"] == "ACCEPTED" for result in payload["probes"].values())
        assert all(item.closed for item in forbidden)


def test_permission_errors_are_sanitized_batches_without_catalog_promotion(
    capture: tuple[dict[str, Any], Connection, dict[str, Any]],
) -> None:
    binding, connection, request = capture
    connection.responses["control"] = DatabaseError("42501")
    with service_sql.collect_service_sql(binding["name"], request, threading.Event()) as payload:
        assert payload["queries"]["control"] == [{"sqlstate": "42501"}] * 3
        assert payload["post_probe"]["control"] == {"sqlstate": "42501"}
        assert "secret" not in json.dumps(payload)


@pytest.mark.parametrize("termination", ("stop", "deadline", "disconnect"))
def test_stop_timeout_disconnect_close_the_held_sql_connection(
    capture: tuple[dict[str, Any], Connection, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    termination: str,
) -> None:
    binding, connection, request = capture
    stop = threading.Event()
    with pytest.raises((passive.PassivePostgreSQLError, ConnectionError)):
        with service_sql.collect_service_sql(binding["name"], request, stop):
            if termination == "stop":
                stop.set()
            elif termination == "deadline":
                monkeypatch.setattr(service_sql.time, "monotonic", lambda: 131.0)
            else:
                raise ConnectionError("peer disconnected")
    assert connection.closed is True


def test_changed_native_service_instance_cannot_yield_evidence(
    capture: tuple[dict[str, Any], Connection, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding, connection, request = capture
    calls = iter([binding, {**binding, "process_started": "2026-10-17T00:00:00+00:00"}])
    monkeypatch.setattr(service_sql, "_own_native_binding", lambda _name: next(calls))
    with pytest.raises(passive.PassivePostgreSQLError) as failure:
        with service_sql.collect_service_sql(binding["name"], request, threading.Event()):
            pytest.fail("changed instance yielded evidence")
    assert failure.value.status == "FAIL"
    assert connection.closed is True


@pytest.mark.parametrize(
    "mode", ("valid", "admin", "impersonation", "thread-denied", "wrong-pid", "wrong-image")
)
def test_native_binding_checks_only_own_process_token_and_scm_entry(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    name, pid = CONTRACT.backend_service, 123
    image = rf"C:\Program Files\CryptoHunter\{name}.exe"
    events: list[Any] = []

    class Handle:
        def Close(self) -> None:
            events.append("token-close")

    def thread_token(*_args: Any) -> Any:
        if mode == "impersonation":
            return Handle()
        error = OSError("unavailable")
        error.winerror = 5 if mode == "thread-denied" else 1008
        raise error

    api = SimpleNamespace(GetCurrentThread=lambda: "thread", GetCurrentProcess=lambda: "process")
    security = SimpleNamespace(
        OpenThreadToken=thread_token,
        OpenProcessToken=lambda *_args: Handle(),
        GetTokenInformation=lambda *_args: ("S-1-5-18" if mode == "admin" else "S-1-5-80-111", 0),
        ConvertSidToStringSid=lambda value: value,
        LookupAccountName=lambda _host, account: ("S-1-5-80-111", None, None),
    )
    process = SimpleNamespace(
        QueryFullProcessImageName=lambda *_args: image
        + (".wrong" if mode == "wrong-image" else ""),
        GetProcessTimes=lambda *_args: {
            "CreationTime": datetime.fromisoformat("2026-10-16T00:00:00+00:00")
        },
    )

    def open_service(_manager: Any, selected: str, access: int) -> str:
        assert selected == name
        assert access == 5
        events.append(selected)
        return name

    scm = SimpleNamespace(
        SC_MANAGER_CONNECT=1,
        SERVICE_QUERY_CONFIG=1,
        SERVICE_QUERY_STATUS=4,
        OpenSCManager=lambda *_args: "manager",
        OpenService=open_service,
        QueryServiceConfig=lambda _service: (
            16,
            None,
            None,
            f'"{image}"',
            None,
            None,
            None,
            rf"NT SERVICE\{name}",
        ),
        QueryServiceStatusEx=lambda _service: {
            "CurrentState": 4,
            "ProcessId": 124 if mode == "wrong-pid" else pid,
        },
        CloseServiceHandle=lambda value: events.append(("close", value)),
    )
    for module_name, module in (
        ("win32api", api),
        ("win32security", security),
        ("win32process", process),
        ("win32service", scm),
    ):
        monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(service_sql, "os", SimpleNamespace(name="nt", getpid=lambda: pid))
    monkeypatch.setattr(service_sql.sys, "executable", image)
    monkeypatch.setattr(passive, "_native_x64", lambda: None)
    monkeypatch.setattr(
        passive,
        "_known_folder",
        lambda index: r"C:\Program Files\CryptoHunter"
        if index == 0x26
        else r"C:\ProgramData\CryptoHunter",
    )
    for forbidden in ("_hardware_uuid", "_native_listeners", "_process_details"):
        monkeypatch.setattr(
            passive, forbidden, lambda *_args: pytest.fail("host-wide service inspection")
        )
    if mode == "valid":
        binding = service_sql._own_native_binding(name)
        assert binding["name"] == name
        assert binding["process_id"] == pid
        assert binding["token_sid"] == "S-1-5-80-111"
        assert events[-2:] == [("close", name), ("close", "manager")]
    else:
        with pytest.raises(passive.PassivePostgreSQLError):
            service_sql._own_native_binding(name)


def test_receipt_validator_rejects_stopped_service_without_exporting_its_error() -> None:
    observation = {name: {} for name in passive._BINDING_FIELDS}
    receipt = object.__new__(passive._LivePostgreSQLReceipt)

    def invalid() -> None:
        raise ValueError("secret native error")

    passive._RECEIPTS[receipt] = {
        "observation_id": id(observation),
        "pid": passive.os.getpid(),
        "created": passive.time.monotonic(),
        "fingerprint": passive._binding_fingerprint(observation),
        "validator": invalid,
    }
    result = passive.summarize_passive_postgresql(receipt, observation)
    assert result["status"] == "UNVERIFIED"
    assert "secret" not in json.dumps(result)


def test_instance_witness_is_structured_not_parsed_from_diagnostic_text(
    capture: tuple[dict[str, Any], Connection, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding, connection, _request = capture
    listener = {
        "process_id": 320,
        "process_path": r"C:\Program Files\CryptoHunter\PostgreSQL\bin\postgres.exe",
        "process_started": "2026-10-16T00:00:00+00:00",
    }
    monkeypatch.setattr(
        passive,
        "_process_details",
        lambda _pid: {
            **listener,
            "process_id": 321,
            "parent_process_id": 320,
            "process_started": "2026-10-16T00:00:01+00:00",
        },
    )
    witness: dict[str, Any] = {}
    passive._collect_sql(connection, {**binding, "listeners": [listener]}, witness=witness)
    assert witness == {
        "postmaster_started": "2026-10-16T00:00:01+00:00",
        "system_identifier": "7448802115599552300",
        "data_directory": str(
            PureWindowsPath(binding["paths"]["program_data"]) / "PostgreSQL" / "Data"
        ),
    }
