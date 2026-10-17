"""Simulated adapter security tests; these do not qualify native Windows SQL."""

from __future__ import annotations

import ast
import copy
import ctypes
import gc
import inspect
import json
import socket
import struct
import sys
import uuid
import weakref
from datetime import datetime
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from typing import Any

import pytest

from deployment import windows_stage9_postgresql_passive as passive
from deployment.windows_installer.contract import CONTRACT


def observation() -> dict[str, Any]:
    root = PureWindowsPath(r"C:\Program Files\CryptoHunter")
    services = []
    for index, name in enumerate(
        (CONTRACT.postgresql_service, CONTRACT.backend_service, CONTRACT.verifier_service), 1
    ):
        image = root / ("PostgreSQL" if index == 1 else "") / f"{name}.exe"
        services.append(
            {
                "name": name,
                "account": rf"NT SERVICE\{name}",
                "command_line": f'"{image}"',
                "state": "Running",
                "process_id": index * 10,
                "second_process_id": index * 10,
                "second_state": 4,
                "process_path": str(image),
                "process_started": "2026-10-16T00:00:00+00:00",
            }
        )
    return {
        "host_identity": {
            "hostname": "approved-host",
            "machine_guid": "0a16b42b-ded1-4eac-938d-bd03ea794ad0",
            "hardware_uuid": "3bc7c43f-af07-4bc3-a377-a5bde9550479",
        },
        "paths": {
            "program_files": str(root),
            "program_data": r"C:\ProgramData\CryptoHunter",
        },
        "services": services,
        "listeners": [
            {
                "address": CONTRACT.postgresql_host,
                "port": CONTRACT.postgresql_port,
                "process_id": 11,
                "parent_process_id": 10,
                "process_path": str(root / "PostgreSQL/bin/postgres.exe"),
                "process_started": "2026-10-16T00:00:00+00:00",
            }
        ],
    }


COMPONENTS = (
    "effective_configuration",
    "parsed_hba",
    "service_authentication",
    "role_catalog",
    "active_hba",
    "cross_role_authentication",
    "instance_identity",
)


@pytest.fixture
def native(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    facts = observation()
    facts["services"][1]["process_id"] = passive.os.getpid()
    facts["services"][1]["second_process_id"] = passive.os.getpid()
    services = {
        item["name"]: {**copy.deepcopy(item), "state": 4, "service_type": 16}
        for item in facts["services"]
    }
    own = services[CONTRACT.backend_service]
    state = {
        "paths": copy.deepcopy(facts["paths"]),
        "host_identity": copy.deepcopy(facts["host_identity"]),
        "services": services,
        "listeners": copy.deepcopy(facts["listeners"]),
        "service_sids": {
            CONTRACT.backend_service: "S-1-5-80-111",
            CONTRACT.verifier_service: "S-1-5-80-222",
        },
        "token_sid": "S-1-5-80-111",
        "thread_impersonation": False,
        "current_pid": passive.os.getpid(),
        "current": copy.deepcopy(own),
    }
    monkeypatch.setattr(passive.sys, "executable", own["process_path"])
    binding = passive._validate_native_state(state, facts)
    return facts, state, binding


class ScriptedConnection:
    """Private SQL-policy simulation; never accepted by the production connector."""

    def __init__(self, binding: dict[str, Any]) -> None:
        self.info = SimpleNamespace(backend_pid=12)
        self.closed = False
        self.queries: list[str] = []
        self.fetch_bounds: list[int] = []
        role = binding["role"]
        self.responses: dict[str, Any] = {
            passive._IDENTITY_SQL: [
                (
                    "freshness_gate",
                    role,
                    role,
                    CONTRACT.postgresql_host,
                    CONTRACT.postgresql_port,
                    12,
                    datetime.fromisoformat("2026-10-16T00:00:01+00:00"),
                    170011,
                    CONTRACT.postgresql_host,
                    CONTRACT.postgresql_port,
                    True,
                    True,
                    True,
                )
            ],
            passive._SSL_SQL: [
                (True, "TLSv1.3", "TLS_AES_256_GCM_SHA384", 256, "CN=CryptoHunterBackend")
            ],
            passive._DATA_SQL: [(r"C:\ProgramData\CryptoHunter\PostgreSQL\Data",)],
            passive._CONTROL_SQL: [("7448802115599552300",)],
            passive._HBA_SQL: [
                (
                    1,
                    "hostssl",
                    ["freshness_gate"],
                    ["freshness_crypto_verifier"],
                    "127.0.0.1",
                    "255.255.255.255",
                    "cert",
                    ["map=stage8_cert", "clientcert=verify-full"],
                    None,
                ),
                (
                    2,
                    "hostssl",
                    ["freshness_gate"],
                    ["freshness_runtime"],
                    "127.0.0.1",
                    "255.255.255.255",
                    "cert",
                    ["map=stage8_cert", "clientcert=verify-full"],
                    None,
                ),
                (
                    3,
                    "host",
                    ["freshness_gate"],
                    ["all"],
                    "127.0.0.1",
                    "255.255.255.255",
                    "reject",
                    None,
                    None,
                ),
                (4, "host", ["all"], ["all"], "127.0.0.1", "255.255.255.255", "reject", None, None),
                (5, "host", ["all"], ["all"], "0.0.0.0", "0.0.0.0", "reject", None, None),
                (6, "host", ["all"], ["all"], "::", "::", "reject", None, None),
            ],
            passive._ROLES_SQL: sorted(
                (name, False, False, False, False, True, False, False)
                for name in ("freshness_runtime", "freshness_crypto_verifier")
            ),
            passive._MEMBERSHIP_SQL: [(0,)],
        }

    def cursor(self) -> ScriptedCursor:
        return ScriptedCursor(self)

    def close(self) -> None:
        self.closed = True


class ScriptedCursor:
    def __init__(self, connection: ScriptedConnection) -> None:
        self.connection = connection
        self.sql = ""

    def __enter__(self) -> ScriptedCursor:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def execute(self, sql: str) -> None:
        self.sql = sql
        self.connection.queries.append(sql)
        value = self.connection.responses[sql]
        if isinstance(value, Exception):
            raise value

    def fetchmany(self, bound: int) -> list[tuple[Any, ...]]:
        self.connection.fetch_bounds.append(bound)
        value = self.connection.responses[self.sql]
        return value() if callable(value) else value[:bound]


@pytest.fixture
def sql(
    native: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], dict[str, Any], ScriptedConnection]:
    facts, state, binding = native
    monkeypatch.setattr(passive, "_read_native_state", lambda: copy.deepcopy(state))
    monkeypatch.setattr(
        passive,
        "_process_details",
        lambda _pid: {
            "process_id": 12,
            "parent_process_id": 11,
            "process_path": facts["listeners"][0]["process_path"],
            "process_started": "2026-10-16T00:00:02+00:00",
        },
    )
    return facts, binding, ScriptedConnection(binding)


@pytest.mark.parametrize(
    "receipt",
    (
        None,
        object(),
        {"status": "PASS", "source": "native", "components": {}},
        {"effective_sql": {"session_roles": ["freshness_runtime", "freshness_crypto_verifier"]}},
        json.dumps(
            {"status": "PASS", "components": {name: {"status": "PASS"} for name in COMPONENTS}}
        ),
    ),
    ids=("missing", "arbitrary-object", "asserted-native", "saved-sql", "saved-json"),
)
def test_caller_supplied_sql_and_saved_reports_are_never_authority(receipt: object) -> None:
    facts = observation()
    summary = passive.summarize_passive_postgresql(receipt, facts)
    assert summary["status"] != "PASS"
    for component in COMPONENTS:
        assert summary["components"][component]["status"] != "PASS"
        with pytest.raises(passive.PassivePostgreSQLError):
            passive.require_passive_postgresql_component(receipt, facts, component)


def test_public_collector_accepts_no_identity_session_dsn_or_callback() -> None:
    assert tuple(inspect.signature(passive.collect_passive_postgresql).parameters) == (
        "observation",
    )


def test_json_safe_summary_does_not_make_an_authoritative_receipt() -> None:
    facts = observation()
    summary = passive.summarize_passive_postgresql(None, facts)
    saved = json.loads(json.dumps(summary))
    copied = copy.deepcopy(summary)
    for replay in (saved, copied):
        with pytest.raises(passive.PassivePostgreSQLError):
            passive.require_passive_postgresql_component(replay, facts, "effective_configuration")


@pytest.mark.parametrize("error", (PermissionError, TimeoutError, RuntimeError))
def test_unavailable_native_authority_blocks_before_connect_or_key_access(
    monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    secret = "private-key-material-password-do-not-report"
    connection_attempts = []

    def unavailable(_facts: dict[str, Any]) -> None:
        raise error(secret)

    def forbidden_connect(_binding: dict[str, Any]) -> None:
        connection_attempts.append(True)
        pytest.fail("unverified native identity must not open a certificate session")

    monkeypatch.setattr(passive, "_native_binding", unavailable)
    monkeypatch.setattr(passive, "_connect", forbidden_connect)
    facts = observation()
    receipt = passive.collect_passive_postgresql(facts)
    summary = passive.summarize_passive_postgresql(receipt, facts)

    assert connection_attempts == []
    assert summary["status"] != "PASS"
    assert secret not in json.dumps(summary)
    assert secret not in repr(receipt)
    for component in COMPONENTS:
        assert summary["components"][component]["status"] != "PASS"


@pytest.mark.parametrize(
    "change",
    (
        "administrator-token",
        "system-token-with-service-group",
        "other-service-token",
        "thread-impersonation",
        "unknown-thread-token",
        "wrong-current-pid",
        "wrong-current-image",
        "wrong-current-lifetime",
        "wrong-scm-account",
        "wrong-scm-image",
        "wrong-scm-lifetime",
        "wrong-host",
        "public-listener",
        "wrong-postmaster-parent",
        "stopped-clean-base",
    ),
)
def test_unapproved_native_identity_never_reaches_own_private_key(
    native: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    facts, state, _binding = native
    own = state["services"][CONTRACT.backend_service]
    if change == "administrator-token":
        state["token_sid"] = "S-1-5-21-1000"
    elif change == "system-token-with-service-group":
        state["token_sid"] = "S-1-5-18"
        state["token_groups"] = [state["service_sids"][CONTRACT.backend_service]]
    elif change == "other-service-token":
        state["token_sid"] = state["service_sids"][CONTRACT.verifier_service]
    elif change == "thread-impersonation":
        state["thread_impersonation"] = True
    elif change == "unknown-thread-token":
        state.pop("thread_impersonation")
    elif change == "wrong-current-pid":
        state["current_pid"] += 1
    elif change == "wrong-current-image":
        state["current"]["process_path"] = r"C:\Windows\System32\python.exe"
    elif change == "wrong-current-lifetime":
        state["current"]["process_started"] = "2026-10-16T00:00:02+00:00"
    elif change == "wrong-scm-account":
        own["account"] = "LocalSystem"
    elif change == "wrong-scm-image":
        own["process_path"] = r"C:\Temp\CryptoHunterBackend.exe"
    elif change == "wrong-scm-lifetime":
        own["process_started"] = "2026-10-16T00:00:02+00:00"
    elif change == "wrong-host":
        state["host_identity"]["machine_guid"] = "other-machine"
    elif change == "public-listener":
        state["listeners"][0]["address"] = "0.0.0.0"
    elif change == "wrong-postmaster-parent":
        state["listeners"][0]["parent_process_id"] = 999
    elif change == "stopped-clean-base":
        own["state"] = 1
        own["process_id"] = 0
        facts["services"][1].update(
            state="Stopped", process_id=0, second_process_id=0, second_state=1
        )
    monkeypatch.setattr(passive, "_read_native_state", lambda: state)

    def forbidden_connect(_binding: dict[str, Any]) -> None:
        pytest.fail("native identity mismatch reached the private-key connection boundary")

    monkeypatch.setattr(passive, "_connect", forbidden_connect)
    with pytest.raises(passive.PassivePostgreSQLError):
        passive._validate_native_state(state, facts)
    receipt = passive.collect_passive_postgresql(facts)
    assert passive.summarize_passive_postgresql(receipt, facts)["status"] != "PASS"


def test_simulated_valid_current_identity_leaves_missing_authentication_explicitly_blocked(
    sql: tuple[dict[str, Any], dict[str, Any], ScriptedConnection],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facts, _binding, connection = sql
    monkeypatch.setattr(passive, "_connect", lambda _binding: connection)
    receipt = passive.collect_passive_postgresql(facts)
    summary = passive.summarize_passive_postgresql(receipt, facts)

    assert summary["status"] == "BLOCKED"
    assert summary["identity"] == CONTRACT.backend_service
    assert connection.closed
    for name in ("effective_configuration", "instance_identity", "parsed_hba", "role_catalog"):
        passive.require_passive_postgresql_component(receipt, facts, name)
    for name in ("active_hba", "cross_role_authentication"):
        assert summary["components"][name]["status"] == "BLOCKED"
        with pytest.raises(passive.PassivePostgreSQLError):
            passive.require_passive_postgresql_component(receipt, facts, name)
    saved = json.loads(json.dumps(summary))
    assert passive.summarize_passive_postgresql(saved, facts)["status"] == "UNVERIFIED"


@pytest.mark.parametrize("method", (copy.copy, copy.deepcopy))
def test_real_receipt_cannot_be_copied_or_serialized(
    sql: tuple[dict[str, Any], dict[str, Any], ScriptedConnection],
    monkeypatch: pytest.MonkeyPatch,
    method: Any,
) -> None:
    facts, _binding, connection = sql
    monkeypatch.setattr(passive, "_connect", lambda _binding: connection)
    receipt = passive.collect_passive_postgresql(facts)
    with pytest.raises(TypeError):
        method(receipt)
    with pytest.raises(TypeError):
        json.dumps(receipt)


@pytest.mark.parametrize(
    "change", ("copied-observation", "host", "service-pid", "listener-pid", "expired", "process")
)
def test_live_receipt_rejects_host_process_instance_and_age_replay(
    sql: tuple[dict[str, Any], dict[str, Any], ScriptedConnection],
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    facts, _binding, connection = sql
    monkeypatch.setattr(passive, "_connect", lambda _binding: connection)
    receipt = passive.collect_passive_postgresql(facts)
    passive.require_passive_postgresql_component(receipt, facts, "effective_configuration")
    if change == "copied-observation":
        facts = copy.deepcopy(facts)
    elif change == "host":
        facts["host_identity"]["hostname"] = "another-host"
    elif change == "service-pid":
        facts["services"][0]["process_id"] += 1
    elif change == "listener-pid":
        facts["listeners"][0]["process_id"] += 1
    elif change == "expired":
        now = passive.time.monotonic()
        monkeypatch.setattr(passive.time, "monotonic", lambda: now + 61)
    elif change == "process":
        pid = passive.os.getpid()
        monkeypatch.setattr(passive.os, "getpid", lambda: pid + 1)
    assert passive.summarize_passive_postgresql(receipt, facts)["status"] == "UNVERIFIED"
    with pytest.raises(passive.PassivePostgreSQLError):
        passive.require_passive_postgresql_component(receipt, facts, "effective_configuration")


@pytest.mark.parametrize(
    "change",
    (
        "wrong-database",
        "administrator-session",
        "role-switch",
        "public-endpoint",
        "wrong-version",
        "writeable-transaction",
        "ssl-disabled",
        "legacy-tls",
        "weak-cipher",
        "other-client-certificate",
        "missing-session",
    ),
)
def test_untrusted_sql_session_tls_and_incomplete_sessions_fail_closed(
    sql: tuple[dict[str, Any], dict[str, Any], ScriptedConnection],
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    facts, _binding, connection = sql
    identity = list(connection.responses[passive._IDENTITY_SQL][0])
    tls = list(connection.responses[passive._SSL_SQL][0])
    if change == "wrong-database":
        identity[0] = "postgres"
    elif change == "administrator-session":
        identity[1:3] = ["postgres", "postgres"]
    elif change == "role-switch":
        identity[2] = "freshness_crypto_verifier"
    elif change == "public-endpoint":
        identity[3] = "192.0.2.20"
    elif change == "wrong-version":
        identity[7] = 170010
    elif change == "writeable-transaction":
        identity[11] = False
    elif change == "ssl-disabled":
        tls[0] = False
    elif change == "legacy-tls":
        tls[1] = "TLSv1"
    elif change == "weak-cipher":
        tls[3] = 64
    elif change == "other-client-certificate":
        tls[4] = "CN=CryptoHunterFreshnessVerifier"
    connection.responses[passive._IDENTITY_SQL] = (
        [] if change == "missing-session" else [tuple(identity)]
    )
    connection.responses[passive._SSL_SQL] = [tuple(tls)]
    monkeypatch.setattr(passive, "_connect", lambda _binding: connection)
    receipt = passive.collect_passive_postgresql(facts)

    assert connection.closed
    assert passive.summarize_passive_postgresql(receipt, facts)["status"] in ("FAIL", "BLOCKED")
    with pytest.raises(passive.PassivePostgreSQLError):
        passive.require_passive_postgresql_component(receipt, facts, "effective_configuration")


@pytest.mark.parametrize(
    "change",
    ("parse-error", "trust-rule", "missing-rule", "extra-rule", "wrong-map", "wrong-order"),
)
def test_malformed_or_unapproved_hba_never_qualifies(
    sql: tuple[dict[str, Any], dict[str, Any], ScriptedConnection], change: str
) -> None:
    _facts, binding, connection = sql
    rows = connection.responses[passive._HBA_SQL]
    first = list(rows[0])
    if change == "parse-error":
        first[8] = "invalid HBA private-error-text"
    elif change == "trust-rule":
        first[6] = "trust"
    elif change == "wrong-map":
        first[7] = ["map=other", "clientcert=verify-full"]
    rows[0] = tuple(first)
    if change == "missing-rule":
        rows.pop()
    elif change == "extra-rule":
        rows.append(rows[-1])
    elif change == "wrong-order":
        rows[0], rows[1] = rows[1], rows[0]

    components = passive._collect_sql(connection, binding)
    assert components["parsed_hba"]["status"] == "FAIL"
    assert "private-error-text" not in json.dumps(components)
    assert components["active_hba"]["status"] == "BLOCKED"


@pytest.mark.parametrize(
    "change", ("superuser", "inheritance", "create-role", "missing-role", "membership", "oversized")
)
def test_malformed_or_privileged_role_catalog_never_proves_separation(
    sql: tuple[dict[str, Any], dict[str, Any], ScriptedConnection], change: str
) -> None:
    _facts, binding, connection = sql
    roles = connection.responses[passive._ROLES_SQL]
    first = list(roles[0])
    if change in ("superuser", "inheritance", "create-role"):
        first[{"superuser": 1, "inheritance": 2, "create-role": 3}[change]] = True
    roles[0] = tuple(first)
    if change == "missing-role":
        roles.pop()
    elif change == "membership":
        connection.responses[passive._MEMBERSHIP_SQL] = [(1,)]
    elif change == "oversized":
        roles.append(roles[0])
    components = passive._collect_sql(connection, binding)
    assert components["role_catalog"]["status"] == "FAIL"
    assert components["cross_role_authentication"]["status"] == "BLOCKED"


@pytest.mark.parametrize("catalog", ("_HBA_SQL", "_ROLES_SQL", "_CONTROL_SQL"))
@pytest.mark.parametrize("error", (PermissionError, TimeoutError))
def test_catalog_errors_preserve_only_independent_proofs_and_sanitize_diagnostics(
    sql: tuple[dict[str, Any], dict[str, Any], ScriptedConnection],
    catalog: str,
    error: type[Exception],
) -> None:
    _facts, binding, connection = sql
    secret = "password=private-key-material-unredacted-DSN"
    connection.responses[getattr(passive, catalog)] = error(secret)
    components = passive._collect_sql(connection, binding)
    assert components["effective_configuration"]["status"] == "PASS"
    assert components["service_authentication"]["status"] == "PASS"
    name = {
        "_HBA_SQL": "parsed_hba",
        "_ROLES_SQL": "role_catalog",
        "_CONTROL_SQL": "instance_identity",
    }[catalog]
    assert components[name]["status"] == "BLOCKED"
    assert secret not in json.dumps(components)
    assert components["active_hba"]["status"] == "BLOCKED"
    assert components["cross_role_authentication"]["status"] == "BLOCKED"


def test_changed_postgresql_system_identifier_is_a_failed_instance_proof(
    sql: tuple[dict[str, Any], dict[str, Any], ScriptedConnection],
) -> None:
    _facts, binding, connection = sql
    replies = iter(([("7448802115599552300",)], [("7448802115599552301",)]))
    connection.responses[passive._CONTROL_SQL] = lambda: next(replies)
    components = passive._collect_sql(connection, binding)
    assert components["instance_identity"]["status"] == "FAIL"


def test_native_instance_change_during_collection_invalidates_all_live_proofs(
    sql: tuple[dict[str, Any], dict[str, Any], ScriptedConnection],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facts, binding, connection = sql
    changed = copy.deepcopy(binding)
    changed["listeners"][0]["process_id"] += 1
    reads = iter((binding, changed))
    monkeypatch.setattr(passive, "_native_binding", lambda _facts: next(reads))
    monkeypatch.setattr(passive, "_connect", lambda _binding: connection)
    receipt = passive.collect_passive_postgresql(facts)
    summary = passive.summarize_passive_postgresql(receipt, facts)
    assert connection.closed
    assert summary["status"] == "FAIL"
    assert all(value["status"] == "FAIL" for value in summary["components"].values())


@pytest.mark.parametrize("error", (PermissionError, TimeoutError))
def test_connection_errors_are_sanitized_without_inventing_an_authentication_denial(
    native: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    error: type[Exception],
) -> None:
    facts, state, _binding = native
    secret = "password=connection-secret-private-client-key"
    monkeypatch.setattr(passive, "_read_native_state", lambda: state)

    def failed_connect(_binding: dict[str, Any]) -> None:
        raise error(secret)

    monkeypatch.setattr(passive, "_connect", failed_connect)
    receipt = passive.collect_passive_postgresql(facts)
    summary = passive.summarize_passive_postgresql(receipt, facts)
    assert summary["status"] == "BLOCKED"
    assert secret not in json.dumps(summary)
    assert summary["components"]["cross_role_authentication"]["status"] == "BLOCKED"


@pytest.mark.parametrize(
    "query",
    (
        "CREATE ROLE attacker",
        "ALTER ROLE freshness_runtime SUPERUSER",
        "GRANT postgres TO freshness_runtime",
        "REVOKE ALL ON DATABASE freshness_gate FROM PUBLIC",
        "COPY (SELECT 1) TO PROGRAM 'command'",
        "SELECT pg_reload_conf()",
        "SELECT 1; ALTER ROLE freshness_runtime SUPERUSER",
    ),
)
def test_queries_outside_fixed_select_allowlist_never_reach_driver(
    sql: tuple[dict[str, Any], dict[str, Any], ScriptedConnection], query: str
) -> None:
    _facts, _binding, connection = sql
    with pytest.raises(passive.PassivePostgreSQLError):
        passive._rows(connection, query, 1)
    assert connection.queries == []


@pytest.mark.parametrize("identity", (CONTRACT.backend_service, CONTRACT.verifier_service))
def test_fixed_direct_connection_uses_only_own_key_bounded_readonly_startup_and_real_driver(
    native: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    identity: str,
) -> None:
    import psycopg

    _facts, _state, binding = native
    binding["identity"] = identity
    binding["role"] = (
        "freshness_runtime" if identity == CONTRACT.backend_service else "freshness_crypto_verifier"
    )
    credential_paths = []
    captured = {}

    def metadata_only(path: Any) -> str:
        credential_paths.append(str(path))
        return str(path)

    def fake_driver(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(passive, "_safe_credential_path", metadata_only)
    monkeypatch.setattr(psycopg, "connect", fake_driver)
    with pytest.raises(passive.PassivePostgreSQLError) as failure:
        passive._connect(binding)
    assert failure.value.status == "FAIL"
    assert captured["host"] == CONTRACT.postgresql_host
    assert captured["port"] == CONTRACT.postgresql_port
    assert captured["dbname"] == "freshness_gate"
    assert captured["user"] == binding["role"]
    assert captured["sslmode"] == "verify-full"
    assert captured["autocommit"] is True
    assert 0 < captured["connect_timeout"] <= 5
    assert "default_transaction_read_only=on" in captured["options"]
    assert "statement_timeout=5000" in captured["options"]
    assert "lock_timeout=1000" in captured["options"]
    assert "transaction_timeout=15000" in captured["options"]
    own = "runtime" if identity == CONTRACT.backend_service else "verifier"
    other = "verifier" if own == "runtime" else "runtime"
    assert len(credential_paths) == 3
    assert sum("client.key" in path for path in credential_paths) == 1
    assert own in captured["sslkey"]
    assert all(other not in path for path in credential_paths)


def test_all_successful_simulated_reads_are_allowlisted_and_bounded(
    sql: tuple[dict[str, Any], dict[str, Any], ScriptedConnection],
) -> None:
    _facts, binding, connection = sql
    passive._collect_sql(connection, binding)
    assert connection.queries
    assert set(connection.queries) <= passive._ALLOWED_SQL
    assert all(query.lstrip().upper().startswith("SELECT ") for query in connection.queries)
    assert max(connection.fetch_bounds) <= 65


def test_adapter_has_no_service_process_token_or_file_mutation_operations() -> None:
    tree = ast.parse(inspect.getsource(passive))
    forbidden_calls = {
        "StartService",
        "ControlService",
        "StopService",
        "CreateService",
        "DeleteService",
        "ChangeServiceConfig",
        "CreateProcessAsUser",
        "CreateProcessWithTokenW",
        "LogonUser",
        "ImpersonateLoggedOnUser",
        "SetThreadToken",
        "AdjustTokenPrivileges",
        "Popen",
        "run",
        "system",
        "write_bytes",
        "write_text",
        "read_bytes",
        "read_text",
    }
    called_names = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert not called_names.intersection(forbidden_calls)


class NativeFunction:
    """A simulated DLL entry point with ctypes signature attributes."""

    def __init__(self, callback: Any) -> None:
        self.callback = callback

    def __call__(self, *args: Any) -> Any:
        return self.callback(*args)


def install_dlls(monkeypatch: pytest.MonkeyPatch, **libraries: Any) -> None:
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(**libraries), raising=False)


@pytest.mark.parametrize(
    "mode", ("success", "impersonation", "unknown-thread", "administrator", "scm-denied")
)
def test_native_state_reads_real_token_fields_and_closes_readonly_handles(
    native: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    facts, state, _binding = native
    events = []

    class Handle:
        def __init__(self, name: str) -> None:
            self.name = name

        def Close(self) -> None:
            events.append(("close", self.name))

        def __enter__(self) -> Handle:
            return self

        def __exit__(self, *_args: Any) -> None:
            self.Close()

    def thread_token(*_args: Any) -> Handle:
        if mode == "impersonation":
            return Handle("thread")
        error = OSError("token query unavailable")
        error.winerror = 5 if mode == "unknown-thread" else 1008
        raise error

    def process_token(_handle: Any, access: int) -> Handle:
        assert access == 8
        events.append(("open", "token"))
        return Handle("token")

    def open_service(_manager: Any, name: str, access: int) -> str:
        assert access == 5
        events.append(("open", name))
        return name

    def service_configuration(name: str) -> tuple[Any, ...]:
        if mode == "scm-denied":
            raise PermissionError("SCM query denied")
        row = state["services"][name]
        return (16, None, None, row["command_line"], None, None, None, row["account"])

    api = SimpleNamespace(
        GetCurrentThread=lambda: "thread",
        GetCurrentProcess=lambda: "process",
        GetComputerName=lambda: facts["host_identity"]["hostname"],
    )
    security = SimpleNamespace(
        OpenThreadToken=thread_token,
        OpenProcessToken=process_token,
        GetTokenInformation=lambda _token, _kind: (
            "S-1-5-18" if mode == "administrator" else state["token_sid"],
            0,
        ),
        ConvertSidToStringSid=lambda sid: sid,
        LookupAccountName=lambda _host, name: (
            state["service_sids"][name.split("\\")[1]],
            None,
            None,
        ),
    )
    scm = SimpleNamespace(
        SC_MANAGER_CONNECT=1,
        SERVICE_QUERY_CONFIG=1,
        SERVICE_QUERY_STATUS=4,
        OpenSCManager=lambda *_args: "manager",
        OpenService=open_service,
        QueryServiceConfig=service_configuration,
        QueryServiceStatusEx=lambda name: {
            "ProcessId": state["services"][name]["process_id"],
            "CurrentState": 4,
        },
        CloseServiceHandle=lambda name: events.append(("close", name)),
    )
    registry = SimpleNamespace(
        HKEY_LOCAL_MACHINE=1,
        KEY_READ=0x20019,
        KEY_WOW64_64KEY=0x100,
        OpenKey=lambda *_args: Handle("registry"),
        QueryValueEx=lambda _key, _name: (facts["host_identity"]["machine_guid"], None),
    )
    for name, module in (
        ("win32api", api),
        ("win32security", security),
        ("win32service", scm),
        ("winreg", registry),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    pid = passive.os.getpid()
    monkeypatch.setattr(passive, "os", SimpleNamespace(name="nt", getpid=lambda: pid))
    monkeypatch.setattr(passive, "_native_x64", lambda: None)
    monkeypatch.setattr(passive, "_hardware_uuid", lambda: facts["host_identity"]["hardware_uuid"])
    monkeypatch.setattr(
        passive,
        "_known_folder",
        lambda index: facts["paths"]["program_files" if index == 0x26 else "program_data"],
    )
    monkeypatch.setattr(passive, "_native_listeners", lambda: copy.deepcopy(state["listeners"]))
    processes = {row["process_id"]: row for row in state["services"].values()}
    monkeypatch.setattr(
        passive, "_process_details", lambda process_id: copy.deepcopy(processes[process_id])
    )
    if mode == "success":
        collected = passive._read_native_state()
        assert collected["token_sid"] == state["token_sid"]
        assert set(collected["services"]) == set(state["services"])
        assert ("close", "token") in events
        assert ("close", "registry") in events
        assert events[-1] == ("close", "manager")
        assert all(("close", name) in events for name in state["services"])
    else:
        with pytest.raises((passive.PassivePostgreSQLError, PermissionError)):
            passive._read_native_state()
        if mode == "impersonation":
            assert events == [("close", "thread")]
        elif mode == "unknown-thread":
            assert events == []
        elif mode == "administrator":
            assert events == [("open", "token"), ("close", "token")]
        elif mode == "scm-denied":
            assert events[-2:] == [("close", CONTRACT.postgresql_service), ("close", "manager")]


@pytest.mark.parametrize(
    "machine,host,available",
    ((0, 0x8664, True), (0x14C, 0x8664, True), (0, 0xAA64, True), (0, 0x8664, False)),
)
def test_native_architecture_query_rejects_emulation_or_unavailable_identity(
    monkeypatch: pytest.MonkeyPatch, machine: int, host: int, available: bool
) -> None:
    def query(_process: Any, process_machine: Any, host_machine: Any) -> bool:
        process_machine._obj.value = machine
        host_machine._obj.value = host
        return available

    install_dlls(monkeypatch, kernel32=SimpleNamespace(IsWow64Process2=NativeFunction(query)))
    if available and machine == 0 and host == 0x8664:
        passive._native_x64()
    else:
        with pytest.raises(passive.PassivePostgreSQLError):
            passive._native_x64()


@pytest.mark.parametrize(
    "mode",
    ("valid", "unavailable", "oversized", "changed", "old-version", "zero-uuid", "malformed"),
)
def test_smbios_identity_parser_is_bounded_and_rejects_unusable_firmware(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    expected = uuid.UUID("3bc7c43f-af07-4bc3-a377-a5bde9550479")
    structure = bytearray(27)
    structure[0:2] = bytes((1, 25))
    structure[8:24] = bytes(16) if mode == "zero-uuid" else expected.bytes_le
    if mode == "malformed":
        structure[1] = 255
    header = bytes((0, 2, 5 if mode == "old-version" else 6, 0)) + struct.pack("<I", len(structure))
    data = header + structure

    def firmware(_provider: Any, _table: Any, buffer: Any, _size: int) -> int:
        if buffer is None:
            return (
                0
                if mode == "unavailable"
                else 1024 * 1024 + 1
                if mode == "oversized"
                else len(data)
            )
        ctypes.memmove(buffer, data, len(data))
        return len(data) - 1 if mode == "changed" else len(data)

    install_dlls(
        monkeypatch, kernel32=SimpleNamespace(GetSystemFirmwareTable=NativeFunction(firmware))
    )
    if mode == "valid":
        assert passive._hardware_uuid() == str(expected).upper()
    else:
        with pytest.raises(passive.PassivePostgreSQLError):
            passive._hardware_uuid()


@pytest.mark.parametrize("mode", ("valid", "changed", "oversized", "bad-count"))
def test_tcp_listener_tables_are_bounded_and_owner_pids_are_observed(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    ipv4 = bytearray(24)
    ipv4[4:8] = socket.inet_pton(socket.AF_INET, "127.0.0.1")
    ipv4[8:10] = CONTRACT.postgresql_port.to_bytes(2, "big")
    struct.pack_into("<I", ipv4, 20, 11)
    ipv6 = bytearray(56)
    ipv6[0:16] = socket.inet_pton(socket.AF_INET6, "::1")
    ipv6[20:22] = CONTRACT.postgresql_port.to_bytes(2, "big")
    struct.pack_into("<I", ipv6, 52, 12)
    tables = {
        socket.AF_INET: struct.pack("<I", 2 if mode == "bad-count" else 1) + ipv4,
        socket.AF_INET6: struct.pack("<I", 1) + ipv6,
    }

    def table(buffer: Any, size: Any, _order: Any, family: int, _class: Any, _reserved: Any) -> int:
        data = tables[family]
        if buffer is None:
            size._obj.value = 1024 * 1024 + 1 if mode == "oversized" else len(data)
            return 122
        ctypes.memmove(buffer, bytes(data), len(data))
        return 122 if mode == "changed" else 0

    install_dlls(monkeypatch, iphlpapi=SimpleNamespace(GetExtendedTcpTable=NativeFunction(table)))
    monkeypatch.setattr(
        passive, "_process_details", lambda pid: {"process_id": pid, "parent_process_id": 10}
    )
    if mode == "valid":
        rows = passive._native_listeners()
        assert [(row["address"], row["process_id"]) for row in rows] == [
            ("127.0.0.1", 11),
            ("::1", 12),
        ]
    else:
        with pytest.raises(passive.PassivePostgreSQLError):
            passive._native_listeners()


@pytest.mark.parametrize("mode", ("found", "absent", "invalid-snapshot"))
def test_process_snapshot_bounded_search_closes_handle(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    closed = []

    def first(_handle: Any, entry: Any) -> bool:
        entry._obj.pid = 12
        entry._obj.parent = 11
        return True

    kernel = SimpleNamespace(
        CreateToolhelp32Snapshot=NativeFunction(
            lambda *_args: None if mode == "invalid-snapshot" else 100
        ),
        Process32FirstW=NativeFunction(first),
        Process32NextW=NativeFunction(lambda *_args: False),
        CloseHandle=NativeFunction(lambda handle: closed.append(handle)),
    )
    install_dlls(monkeypatch, kernel32=kernel)
    if mode == "found":
        assert passive._process_parent(12) == 11
    else:
        with pytest.raises(passive.PassivePostgreSQLError):
            passive._process_parent(99)
    assert closed == ([] if mode == "invalid-snapshot" else [100])


@pytest.mark.parametrize("fail", (False, True))
def test_process_details_close_native_handle_on_success_and_query_failure(
    monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    closed = []
    handle = SimpleNamespace(Close=lambda: closed.append(True))

    def image(_handle: Any, _flags: Any) -> str:
        if fail:
            raise PermissionError("native image access denied")
        return r"C:\Program Files\CryptoHunter\PostgreSQL\bin\postgres.exe"

    monkeypatch.setitem(
        sys.modules, "win32api", SimpleNamespace(OpenProcess=lambda access, inherit, pid: handle)
    )
    monkeypatch.setitem(
        sys.modules,
        "win32process",
        SimpleNamespace(
            QueryFullProcessImageName=image,
            GetProcessTimes=lambda _handle: {
                "CreationTime": datetime.fromisoformat("2026-10-16T00:00:00+00:00")
            },
        ),
    )
    monkeypatch.setattr(passive, "_process_parent", lambda _pid: 11)
    if fail:
        with pytest.raises(PermissionError):
            passive._process_details(12)
    else:
        details = passive._process_details(12)
        assert details["parent_process_id"] == 11
        assert details["process_id"] == 12
    assert closed == [True]


def test_credential_metadata_checks_reject_symlink_without_reading_key(
    tmp_path: Path,
) -> None:
    credential = tmp_path / "own.key"
    credential.write_bytes(b"unit-test-key-placeholder")
    assert passive._safe_credential_path(credential) == str(credential)
    redirected = tmp_path / "redirected.key"
    redirected.symlink_to(credential)
    with pytest.raises(passive.PassivePostgreSQLError) as failure:
        passive._safe_credential_path(redirected)
    assert failure.value.status == "FAIL"
    with pytest.raises(passive.PassivePostgreSQLError):
        passive._safe_credential_path(tmp_path)


def test_own_private_key_hardlink_is_rejected_by_metadata_without_reading_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = tmp_path / "original.key"
    original.write_bytes(b"unit-test-key-placeholder")
    alias = tmp_path / "client.key"
    alias.hardlink_to(original)

    def forbidden_content_read(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("credential metadata qualification must not read key contents")

    with monkeypatch.context() as boundary:
        boundary.setattr(Path, "open", forbidden_content_read)
        with pytest.raises(passive.PassivePostgreSQLError) as failure:
            passive._safe_credential_path(alias)
    assert failure.value.status == "FAIL"


@pytest.mark.parametrize("mode", ("bounded", "old-interval", "historical-timeout"))
def test_client_socket_deadline_is_enforced_or_blocks_before_credential_access(
    native: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    import psycopg

    _facts, _state, binding = native
    credential_access = []
    waits = []

    class DriverConnection:
        def wait(
            self, _generator: Any, interval: float = 0.1, timeout: float | None = None
        ) -> None:
            waits.append(timeout)

        def close(self) -> None:
            return None

    if mode == "old-interval":

        def old_wait(self: Any, _generator: Any, interval: float = 0.1) -> None:
            return None

        monkeypatch.setattr(DriverConnection, "wait", old_wait)
    elif mode == "historical-timeout":

        def polling_wait(self: Any, _generator: Any, timeout: float = 0.1) -> None:
            return None

        monkeypatch.setattr(DriverConnection, "wait", polling_wait)
    monkeypatch.setattr(psycopg, "Connection", DriverConnection)
    monkeypatch.setattr(psycopg, "connect", lambda **_kwargs: DriverConnection())
    monkeypatch.setattr(
        passive,
        "_safe_credential_path",
        lambda path: credential_access.append(str(path)) or str(path),
    )
    if mode == "bounded":
        connection = passive._connect(binding)
        connection.wait(object())
        assert waits == [5.0]
        assert len(credential_access) == 3
    else:
        with pytest.raises(passive.PassivePostgreSQLError) as failure:
            passive._connect(binding)
        assert failure.value.status == "BLOCKED"
        assert credential_access == []


def test_receipt_registry_does_not_retain_a_self_referencing_observation(
    sql: tuple[dict[str, Any], dict[str, Any], ScriptedConnection],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original, _binding, connection = sql
    monkeypatch.setattr(passive, "_connect", lambda _binding: connection)

    def issue_receipt() -> Any:
        facts = copy.deepcopy(original)
        receipt = passive.collect_passive_postgresql(facts)
        facts["receipt"] = receipt
        return weakref.ref(receipt)

    witness = issue_receipt()
    gc.collect()
    assert witness() is None


@pytest.mark.parametrize("later_error", ("sql-reread", "native-reread", "cleanup"))
def test_known_catalog_failure_is_not_downgraded_by_later_unavailability(
    sql: tuple[dict[str, Any], dict[str, Any], ScriptedConnection],
    monkeypatch: pytest.MonkeyPatch,
    later_error: str,
) -> None:
    facts, binding, connection = sql
    roles = connection.responses[passive._ROLES_SQL]
    roles[0] = (roles[0][0], True, *roles[0][2:])
    monkeypatch.setattr(passive, "_connect", lambda _binding: connection)
    if later_error == "sql-reread":
        identity = connection.responses[passive._IDENTITY_SQL]
        calls = 0

        def session_reread() -> Any:
            nonlocal calls
            calls += 1
            if calls > 1:
                raise TimeoutError("sensitive final SQL error")
            return identity

        connection.responses[passive._IDENTITY_SQL] = session_reread
    elif later_error == "native-reread":
        native_calls = 0

        def native_reread(_facts: Any) -> Any:
            nonlocal native_calls
            native_calls += 1
            if native_calls > 1:
                raise PermissionError("sensitive native reread error")
            return binding

        monkeypatch.setattr(passive, "_native_binding", native_reread)
    elif later_error == "cleanup":

        def failed_close() -> None:
            raise RuntimeError("sensitive close error")

        monkeypatch.setattr(connection, "close", failed_close)
    receipt = passive.collect_passive_postgresql(facts)
    summary = passive.summarize_passive_postgresql(receipt, facts)
    assert summary["status"] == "FAIL"
    assert "sensitive" not in json.dumps(summary)
