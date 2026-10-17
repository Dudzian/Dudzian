"""Passive PostgreSQL reads under the current, independently verified service token.

There is deliberately no session, identity, connection factory, or report-input
API. An administrator cannot borrow a stopped service's identity. Parsed HBA
files are not evidence of the rules loaded by the postmaster. Reports are only
diagnostics; authority is a short-lived, process-local opaque receipt.
"""

from __future__ import annotations

import copy
import ctypes
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path, PureWindowsPath
import socket
import stat
import struct
import sys
import time
import uuid
import weakref
from datetime import datetime
from functools import partial
from typing import Any

from deployment.windows_installer.contract import CONTRACT
from deployment.windows_postgresql_auth_contract import (
    RUNTIME_IDENTITY,
    RUNTIME_ROLE,
    VERIFIER_IDENTITY,
    VERIFIER_ROLE,
    qualify_hba_rows,
)

COMPONENTS = (
    "effective_configuration",
    "instance_identity",
    "parsed_hba",
    "service_authentication",
    "role_catalog",
    "active_hba",
    "cross_role_authentication",
)
_BINDING_FIELDS = ("host_identity", "paths", "services", "listeners")
_MAX_AGE = 60.0
_DATABASE = "freshness_gate"


class PassivePostgreSQLError(ValueError):
    """A read is unavailable or contradicts the frozen contract."""

    def __init__(self, message: str, *, status: str = "BLOCKED") -> None:
        super().__init__(message)
        self.status = status


class _LivePostgreSQLReceipt:
    __slots__ = ("__weakref__",)

    def __new__(cls) -> _LivePostgreSQLReceipt:
        raise TypeError("live SQL receipts are issued only by the passive collector")

    def __reduce__(self) -> Any:
        raise TypeError("live SQL receipts cannot be serialized")


_RECEIPTS: weakref.WeakKeyDictionary[_LivePostgreSQLReceipt, dict[str, Any]] = (
    weakref.WeakKeyDictionary()
)


def _binding_fingerprint(observation: dict[str, Any]) -> str:
    try:
        selected = {name: observation[name] for name in _BINDING_FIELDS}
        encoded = json.dumps(selected, sort_keys=True, allow_nan=False, separators=(",", ":"))
    except (KeyError, TypeError, ValueError) as exc:
        raise PassivePostgreSQLError(
            "host comparison binding is unavailable", status="UNVERIFIED"
        ) from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _result(status: str, detail: str) -> dict[str, str]:
    return {"status": status, "detail": detail}


def _blocked_components(detail: str, status: str = "BLOCKED") -> dict[str, dict[str, str]]:
    return {name: _result(status, detail) for name in COMPONENTS}


def _overall(components: dict[str, dict[str, str]]) -> str:
    statuses = {value["status"] for value in components.values()}
    return next(
        (status for status in ("FAIL", "BLOCKED", "UNVERIFIED") if status in statuses), "PASS"
    )


def _timestamp(value: Any) -> float:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise PassivePostgreSQLError(
                "process/server creation time has no timezone", status="FAIL"
            )
        return value.timestamp()
    if isinstance(value, str):
        return _timestamp(datetime.fromisoformat(value.replace("Z", "+00:00")))
    raise PassivePostgreSQLError("process/server creation time is unavailable")


def _same_path(first: Any, second: Any) -> bool:
    return (
        isinstance(first, str)
        and isinstance(second, str)
        and (PureWindowsPath(first) == PureWindowsPath(second))
    )


def _known_folder(index: int) -> str:
    buffer = ctypes.create_unicode_buffer(32768)
    native = importlib.import_module("ctypes")
    if native.windll.shell32.SHGetFolderPathW(None, index, None, 0, buffer) != 0:
        raise PassivePostgreSQLError("native known folder is unavailable")
    return str(PureWindowsPath(buffer.value) / "CryptoHunter")


def _native_x64() -> None:
    native = importlib.import_module("ctypes")
    machine, host = ctypes.c_ushort(), ctypes.c_ushort()
    query = native.windll.kernel32.IsWow64Process2
    query.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ushort),
        ctypes.POINTER(ctypes.c_ushort),
    ]
    if not query(ctypes.c_void_p(-1), ctypes.byref(machine), ctypes.byref(host)):
        raise PassivePostgreSQLError("native architecture query is unavailable")
    if machine.value != 0 or host.value != 0x8664:
        raise PassivePostgreSQLError("native AMD64 Windows process is required")


def _hardware_uuid() -> str:
    """Read the SMBIOS 2.6+ UUID without an unbounded WMI enumeration."""
    native = importlib.import_module("ctypes")
    query = native.windll.kernel32.GetSystemFirmwareTable
    size = query(0x52534D42, 0, None, 0)  # RSMB provider
    if not 8 <= size <= 1024 * 1024:
        raise PassivePostgreSQLError("native SMBIOS identity is unavailable")
    buffer = ctypes.create_string_buffer(size)
    if query(0x52534D42, 0, buffer, size) != size:
        raise PassivePostgreSQLError("native SMBIOS identity changed or is unavailable")
    data = buffer.raw
    if (data[1], data[2]) < (2, 6) or struct.unpack_from("<I", data, 4)[0] != size - 8:
        raise PassivePostgreSQLError("supported native SMBIOS identity is unavailable")
    offset = 8
    while offset + 4 <= size:
        kind, length = data[offset : offset + 2]
        if length < 4 or offset + length > size:
            break
        if kind == 1 and length >= 25:
            value = data[offset + 8 : offset + 24]
            if value in (bytes(16), bytes([255]) * 16):
                break
            return str(uuid.UUID(bytes_le=value)).upper()
        terminator = data.find(b"\x00\x00", offset + length)
        if terminator < 0:
            break
        offset = terminator + 2
    raise PassivePostgreSQLError("native SMBIOS hardware UUID is unavailable")


def _process_parent(pid: int) -> int:
    native = importlib.import_module("ctypes")
    kernel = native.windll.kernel32

    class ProcessEntry(ctypes.Structure):
        _fields_ = [
            ("size", ctypes.c_ulong),
            ("usage", ctypes.c_ulong),
            ("pid", ctypes.c_ulong),
            ("heap", ctypes.c_size_t),
            ("module", ctypes.c_ulong),
            ("threads", ctypes.c_ulong),
            ("parent", ctypes.c_ulong),
            ("priority", ctypes.c_long),
            ("flags", ctypes.c_ulong),
            ("image", ctypes.c_wchar * 260),
        ]

    kernel.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel.Process32FirstW.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessEntry)]
    kernel.Process32NextW.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessEntry)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot in (None, ctypes.c_void_p(-1).value):
        raise PassivePostgreSQLError("native process snapshot is unavailable")
    try:
        entry = ProcessEntry()
        entry.size = ctypes.sizeof(entry)
        present = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        for _ in range(32768):
            if not present:
                break
            if entry.pid == pid:
                return int(entry.parent)
            present = kernel.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    raise PassivePostgreSQLError("native process parent is unavailable")


def _process_details(pid: int) -> dict[str, Any]:
    api = importlib.import_module("win32api")
    process = importlib.import_module("win32process")
    handle = api.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    try:
        image = process.QueryFullProcessImageName(handle, 0)
        created = process.GetProcessTimes(handle)["CreationTime"]
        return {
            "process_id": pid,
            "process_path": image,
            "process_started": created.isoformat(),
            "parent_process_id": _process_parent(pid),
        }
    finally:
        handle.Close()


def _native_listeners() -> list[dict[str, Any]]:
    """Read bounded IPv4 and IPv6 listener-owner tables directly from IP Helper."""
    rows: list[dict[str, Any]] = []
    for family, row_size in ((socket.AF_INET, 24), (socket.AF_INET6, 56)):
        size = ctypes.c_ulong(0)
        native = importlib.import_module("ctypes")
        table = native.windll.iphlpapi.GetExtendedTcpTable
        code = table(None, ctypes.byref(size), False, family, 3, 0)
        if code not in (0, 122) or not 4 <= size.value <= 1024 * 1024:
            raise PassivePostgreSQLError("native TCP owner table is unavailable")
        buffer = ctypes.create_string_buffer(size.value)
        if table(buffer, ctypes.byref(size), False, family, 3, 0) != 0:
            raise PassivePostgreSQLError("native TCP owner table changed or is unavailable")
        data = buffer.raw[: size.value]
        count = struct.unpack_from("<I", data)[0]
        if 4 + count * row_size > len(data):
            raise PassivePostgreSQLError("native TCP owner table is malformed", status="FAIL")
        for index in range(count):
            row = data[4 + index * row_size : 4 + (index + 1) * row_size]
            address_offset, port_offset, pid_offset = (
                (4, 8, 20) if family == socket.AF_INET else (0, 20, 52)
            )
            port = int.from_bytes(row[port_offset : port_offset + 2], "big")
            if port == CONTRACT.postgresql_port:
                width = 4 if family == socket.AF_INET else 16
                address = socket.inet_ntop(family, row[address_offset : address_offset + width])
                pid = struct.unpack_from("<I", row, pid_offset)[0]
                rows.append({"address": address, "port": port, **_process_details(pid)})
    return rows


def _read_native_state() -> dict[str, Any]:
    if os.name != "nt" or struct.calcsize("P") != 8:
        raise PassivePostgreSQLError("native Windows x64 service-process access is required")
    _native_x64()
    api = importlib.import_module("win32api")
    security = importlib.import_module("win32security")
    scm = importlib.import_module("win32service")
    # Do not remove or replace an effective thread token: its presence blocks us.
    try:
        thread = security.OpenThreadToken(api.GetCurrentThread(), 8, True)
    except Exception as exc:
        if getattr(exc, "winerror", None) != 1008:
            raise PassivePostgreSQLError("thread-token absence cannot be established") from exc
    else:
        thread.Close()
        raise PassivePostgreSQLError("thread impersonation is not an approved service identity")
    token = security.OpenProcessToken(api.GetCurrentProcess(), 8)
    try:
        user = security.GetTokenInformation(token, 1)
        token_sid = security.ConvertSidToStringSid(user[0] if isinstance(user, tuple) else user)
    finally:
        token.Close()
    candidates = {}
    for name in (CONTRACT.backend_service, CONTRACT.verifier_service):
        sid = security.LookupAccountName(None, rf"NT SERVICE\{name}")[0]
        candidates[name] = security.ConvertSidToStringSid(sid)
    if token_sid not in candidates.values():
        raise PassivePostgreSQLError("current process token is not either canonical service SID")
    winreg = importlib.import_module("winreg")

    with winreg.OpenKey(
        winreg.HKEY_LOCAL_MACHINE,
        r"SOFTWARE\Microsoft\Cryptography",
        0,
        winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
    ) as key:
        machine_guid = winreg.QueryValueEx(key, "MachineGuid")[0]
    state: dict[str, Any] = {
        "paths": {"program_files": _known_folder(0x26), "program_data": _known_folder(0x23)},
        "host_identity": {
            "hostname": api.GetComputerName(),
            "machine_guid": machine_guid,
            "hardware_uuid": _hardware_uuid(),
        },
        "token_sid": token_sid,
        "service_sids": candidates,
        "thread_impersonation": False,
        "current_pid": os.getpid(),
        "current": _process_details(os.getpid()),
        "services": {},
    }
    manager = scm.OpenSCManager(None, None, scm.SC_MANAGER_CONNECT)
    try:
        for name in (
            CONTRACT.postgresql_service,
            CONTRACT.backend_service,
            CONTRACT.verifier_service,
        ):
            service = scm.OpenService(
                manager, name, scm.SERVICE_QUERY_CONFIG | scm.SERVICE_QUERY_STATUS
            )
            try:
                config = scm.QueryServiceConfig(service)
                status = scm.QueryServiceStatusEx(service)
                pid = status["ProcessId"]
                state["services"][name] = {
                    "name": name,
                    "account": config[7],
                    "command_line": config[3],
                    "service_type": config[0],
                    "state": status["CurrentState"],
                    "process_id": pid,
                    **(_process_details(pid) if pid else {}),
                }
            finally:
                scm.CloseServiceHandle(service)
    finally:
        scm.CloseServiceHandle(manager)
    state["listeners"] = _native_listeners()
    return state


def _validate_native_state(state: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    """Compare native reads with the host observation; caller JSON supplies no identity."""
    if state.get("thread_impersonation") is not False:
        raise PassivePostgreSQLError("effective thread identity is unavailable or impersonated")
    sids = state["service_sids"]
    matches = [
        name
        for name in (CONTRACT.backend_service, CONTRACT.verifier_service)
        if state.get("token_sid") == sids.get(name) and isinstance(sids.get(name), str)
    ]
    if len(matches) != 1:
        raise PassivePostgreSQLError("current token is not a canonical service identity")
    name = matches[0]
    root = PureWindowsPath(state["paths"]["program_files"])
    if state["host_identity"] != observation.get("host_identity"):
        raise PassivePostgreSQLError("native host identity differs from inspection", status="FAIL")
    for key in ("program_files", "program_data"):
        if not _same_path(state["paths"][key], observation.get("paths", {}).get(key)):
            raise PassivePostgreSQLError("native canonical product path differs", status="FAIL")
    observed = {row["name"]: row for row in observation["services"]}
    if len(observation["services"]) != 3 or set(observed) != set(state["services"]):
        raise PassivePostgreSQLError("native SCM service set differs", status="FAIL")
    for service_name, service in state["services"].items():
        expected = str(
            root
            / ("PostgreSQL" if service_name == CONTRACT.postgresql_service else "")
            / f"{service_name}.exe"
        )
        row = observed[service_name]
        if (
            service["account"] != rf"NT SERVICE\{service_name}"
            or service["service_type"] != 16
            or service["command_line"]
            not in (f'"{expected}"', expected if " " not in expected else None)
            or row.get("command_line") != service["command_line"]
            or row.get("account") != service["account"]
            or row.get("process_id") != service["process_id"]
            or row.get("second_process_id") != service["process_id"]
            or row.get("second_state") != service["state"]
            or row.get("state") != ("Running" if service["state"] == 4 else "Stopped")
        ):
            raise PassivePostgreSQLError(
                "SCM process/configuration differs from inspection", status="FAIL"
            )
        if service["process_id"]:
            if (
                not _same_path(service["process_path"], expected)
                or not _same_path(row.get("process_path"), expected)
                or abs(
                    _timestamp(service["process_started"]) - _timestamp(row.get("process_started"))
                )
                > 0.001
            ):
                raise PassivePostgreSQLError(
                    "SCM process image/creation time differs", status="FAIL"
                )
    current = state["current"]
    own = state["services"][name]
    if (
        own["state"] != 4
        or own["process_id"] != state["current_pid"]
        or state["current_pid"] != os.getpid()
    ):
        raise PassivePostgreSQLError("current process is not the running SCM service process")
    if (
        not _same_path(current["process_path"], str(root / f"{name}.exe"))
        or not _same_path(sys.executable, current["process_path"])
        or current["process_started"] != own["process_started"]
    ):
        raise PassivePostgreSQLError(
            "current service executable/creation time differs", status="FAIL"
        )
    listeners = state["listeners"]
    expected_postgres = str(root / "PostgreSQL" / "bin" / "postgres.exe")
    pg_service = state["services"][CONTRACT.postgresql_service]
    if (
        pg_service["state"] != 4
        or len(listeners) != 1
        or listeners[0]["address"] != CONTRACT.postgresql_host
        or listeners[0]["port"] != CONTRACT.postgresql_port
        or not _same_path(listeners[0]["process_path"], expected_postgres)
        or listeners[0]["parent_process_id"] != pg_service["process_id"]
    ):
        raise PassivePostgreSQLError("native private PostgreSQL listener differs", status="FAIL")
    comparison = observation.get("listeners")
    if (
        not isinstance(comparison, list)
        or len(comparison) != 1
        or any(
            comparison[0].get(key) != listeners[0][key]
            for key in ("address", "port", "process_id", "parent_process_id")
        )
        or not _same_path(comparison[0].get("process_path"), expected_postgres)
    ):
        raise PassivePostgreSQLError("native listener differs from host observation", status="FAIL")
    if _timestamp(listeners[0]["process_started"]) < _timestamp(pg_service["process_started"]):
        raise PassivePostgreSQLError("postmaster predates its SCM parent", status="FAIL")
    return {
        **state,
        "identity": name,
        "role": RUNTIME_ROLE if name == CONTRACT.backend_service else VERIFIER_ROLE,
        "certificate_identity": RUNTIME_IDENTITY
        if name == CONTRACT.backend_service
        else VERIFIER_IDENTITY,
    }


def _native_binding(observation: dict[str, Any]) -> dict[str, Any]:
    return _validate_native_state(_read_native_state(), observation)


def _safe_credential_path(path: Path) -> str:
    # Inspect metadata only; libpq alone opens the current identity's key.
    target = path.lstat()
    if not stat.S_ISREG(target.st_mode) or target.st_nlink != 1:
        raise PassivePostgreSQLError(
            "credential file is not a single-link regular file", status="FAIL"
        )
    for item in (path, *path.parents):
        metadata = item.lstat()
        if item.is_symlink() or getattr(metadata, "st_file_attributes", 0) & 0x400:
            raise PassivePostgreSQLError("credential path is redirected", status="FAIL")
    return str(path)


def _connect(binding: dict[str, Any]) -> Any:
    psycopg = importlib.import_module("psycopg")
    if not {"interval", "timeout"} <= inspect.signature(psycopg.Connection.wait).parameters.keys():
        raise PassivePostgreSQLError("installed psycopg cannot bound client socket waits")
    security = Path(binding["paths"]["program_data"]) / "Security"
    own = security / ("runtime" if binding["identity"] == CONTRACT.backend_service else "verifier")
    connection = psycopg.connect(
        host=CONTRACT.postgresql_host,
        hostaddr=CONTRACT.postgresql_host,
        port=CONTRACT.postgresql_port,
        dbname=_DATABASE,
        user=binding["role"],
        sslmode="verify-full",
        sslrootcert=_safe_credential_path(security / "ca.crt"),
        sslcert=_safe_credential_path(own / "client.crt"),
        sslkey=_safe_credential_path(own / "client.key"),
        connect_timeout=5,
        autocommit=True,
        gssencmode="disable",
        application_name="CryptoHunterStage9Passive",
        options="-c statement_timeout=5000 -c lock_timeout=1000 "
        "-c idle_in_transaction_session_timeout=5000 -c transaction_timeout=15000 "
        "-c default_transaction_read_only=on",
    )
    if type(connection) is not psycopg.Connection:
        # Never call methods on a supplied non-driver object.
        raise PassivePostgreSQLError("connection is not a native psycopg connection", status="FAIL")
    try:
        # A server GUC cannot bound a suspended backend. This fresh, private
        # connection also imposes a driver socket-wait deadline on each query.
        connection.wait = partial(connection.wait, timeout=5.0)
    except Exception:
        connection.close()
        raise PassivePostgreSQLError("native driver client-wait bound is unavailable") from None
    return connection


_IDENTITY_SQL = """SELECT current_database(), session_user::text, current_user::text,
inet_server_addr()::text, inet_server_port(), pg_backend_pid(), pg_postmaster_start_time(),
current_setting('server_version_num')::int, current_setting('listen_addresses'),
current_setting('port')::int, current_setting('ssl')::boolean,
current_setting('transaction_read_only')::boolean,
current_setting('default_transaction_read_only')::boolean"""
_SSL_SQL = "SELECT ssl, version, cipher, bits, client_dn FROM pg_catalog.pg_stat_ssl WHERE pid=pg_backend_pid()"
_ROLES_SQL = """SELECT rolname, rolsuper, rolinherit, rolcreaterole, rolcreatedb,
rolcanlogin, rolreplication, rolbypassrls FROM pg_catalog.pg_roles
WHERE rolname IN ('freshness_runtime','freshness_crypto_verifier') ORDER BY rolname"""
_MEMBERSHIP_SQL = """SELECT count(*) FROM pg_catalog.pg_auth_members m
JOIN pg_catalog.pg_roles a ON a.oid=m.roleid JOIN pg_catalog.pg_roles b ON b.oid=m.member
WHERE a.rolname IN ('freshness_runtime','freshness_crypto_verifier')
OR b.rolname IN ('freshness_runtime','freshness_crypto_verifier')"""
_HBA_SQL = """SELECT line_number, type, database, user_name, address, netmask,
auth_method, options, error FROM pg_catalog.pg_hba_file_rules ORDER BY rule_number LIMIT 65"""
_DATA_SQL = "SELECT current_setting('data_directory')"
_CONTROL_SQL = "SELECT system_identifier::text FROM pg_catalog.pg_control_system()"
_ALLOWED_SQL = frozenset(
    (_IDENTITY_SQL, _SSL_SQL, _ROLES_SQL, _MEMBERSHIP_SQL, _HBA_SQL, _DATA_SQL, _CONTROL_SQL)
)


def _rows(connection: Any, sql: str, limit: int) -> list[tuple[Any, ...]]:
    if sql not in _ALLOWED_SQL or not sql.lstrip().upper().startswith("SELECT "):
        raise PassivePostgreSQLError("query is outside the passive allowlist", status="FAIL")
    with connection.cursor() as cursor:
        cursor.execute(sql)
        values = cursor.fetchmany(limit + 1)
    if len(values) > limit or any(type(value) is not tuple for value in values):
        raise PassivePostgreSQLError(
            "SQL result is malformed or exceeds the read bound", status="FAIL"
        )
    return [tuple(value) for value in values]


def _one(connection: Any, sql: str, width: int) -> tuple[Any, ...]:
    rows = _rows(connection, sql, 1)
    if len(rows) != 1 or len(rows[0]) != width:
        raise PassivePostgreSQLError("required SQL row is unavailable", status="BLOCKED")
    return rows[0]


def _query_error(exc: Exception) -> dict[str, str]:
    if isinstance(exc, PassivePostgreSQLError):
        return _result(exc.status, str(exc))
    # Error text, DSN and diagnostic fields can include credentials. Never export them.
    if getattr(exc, "sqlstate", None) == "42501":
        return _result("BLOCKED", "approved service role lacks permission for this catalog read")
    return _result("BLOCKED", "bounded SQL read failed; no authentication denial is inferred")


def _collect_sql(connection: Any, binding: dict[str, Any]) -> dict[str, dict[str, str]]:
    components = _blocked_components("required approved SQL evidence is unavailable")
    components["active_hba"] = _result(
        "BLOCKED", "parsed pg_hba_file_rules does not prove the HBA loaded by the postmaster"
    )
    components["cross_role_authentication"] = _result(
        "BLOCKED",
        "no cross-role authentication denial matrix was collected; the second service identity is unavailable",
    )
    identity = _one(connection, _IDENTITY_SQL, 13)
    role = binding["role"]
    if any(type(identity[index]) is not int for index in (4, 5, 7, 9)) or identity[:6] != (
        _DATABASE,
        role,
        role,
        CONTRACT.postgresql_host,
        CONTRACT.postgresql_port,
        connection.info.backend_pid,
    ):
        raise PassivePostgreSQLError("SQL endpoint/session identity differs", status="FAIL")
    backend = _process_details(identity[5])
    listener = binding["listeners"][0]
    started = _timestamp(identity[6])
    if (
        backend["parent_process_id"] != listener["process_id"]
        or not _same_path(backend["process_path"], listener["process_path"])
        or _timestamp(backend["process_started"]) < _timestamp(listener["process_started"])
        or not 0 <= started - _timestamp(listener["process_started"]) <= 60
    ):
        raise PassivePostgreSQLError(
            "SQL backend/postmaster is not bound to the native listener", status="FAIL"
        )
    if any(identity[index] is not True for index in (10, 11, 12)) or identity[7:] != (
        170011,
        CONTRACT.postgresql_host,
        CONTRACT.postgresql_port,
        True,
        True,
        True,
    ):
        raise PassivePostgreSQLError(
            "effective PostgreSQL endpoint/version/read-only settings differ", status="FAIL"
        )
    ssl = _one(connection, _SSL_SQL, 5)
    expected_dn = f"CN={binding['certificate_identity']}"
    if (
        ssl[0] is not True
        or ssl[1] not in ("TLSv1.2", "TLSv1.3")
        or not isinstance(ssl[2], str)
        or not ssl[2]
        or type(ssl[3]) is not int
        or ssl[3] < 128
        or ssl[4] != expected_dn
    ):
        raise PassivePostgreSQLError("actual session TLS/client certificate differs", status="FAIL")
    components["effective_configuration"] = _result(
        "PASS",
        f"live native-bound {CONTRACT.postgresql_host}:{CONTRACT.postgresql_port}, "
        f"PostgreSQL 17.11, backend PID {identity[5]}, postmaster started {identity[6]}, "
        f"{ssl[1]} {ssl[2]} ({ssl[3]} bits), client DN {ssl[4]}",
    )
    components["service_authentication"] = _result(
        "PASS",
        f"only current identity {binding['identity']} authenticated as {role}; second identity unavailable",
    )
    try:
        directory = _one(connection, _DATA_SQL, 1)[0]
        system_id = _one(connection, _CONTROL_SQL, 1)[0]
        expected_data = str(
            PureWindowsPath(binding["paths"]["program_data"]) / "PostgreSQL" / "Data"
        )
        if (
            not _same_path(directory, expected_data)
            or not isinstance(system_id, str)
            or not system_id.isascii()
            or not system_id.isdecimal()
            or not 0 < int(system_id) <= 2**64 - 1
        ):
            raise PassivePostgreSQLError(
                "live PostgreSQL data/system identity differs", status="FAIL"
            )
        if _one(connection, _CONTROL_SQL, 1) != (system_id,):
            raise PassivePostgreSQLError(
                "PostgreSQL system identity changed during reads", status="FAIL"
            )
        components["instance_identity"] = _result(
            "PASS",
            f"canonical data directory {directory}; stable control-system identifier {system_id}",
        )
    except Exception as exc:
        components["instance_identity"] = _query_error(exc)
    try:
        rows = _rows(connection, _HBA_SQL, 64)
        qualify_hba_rows(rows, _DATABASE)
        components["parsed_hba"] = _result(
            "PASS",
            "parsed on-disk HBA rules match the frozen contract; active HBA remains unproved",
        )
    except ValueError as exc:
        if isinstance(exc, PassivePostgreSQLError):
            components["parsed_hba"] = _query_error(exc)
        else:
            components["parsed_hba"] = _result(
                "FAIL", "parsed HBA rows differ from the frozen contract"
            )
    except Exception as exc:
        components["parsed_hba"] = _query_error(exc)
    try:
        roles = _rows(connection, _ROLES_SQL, 2)
        expected = sorted(
            (name, False, False, False, False, True, False, False)
            for name in (RUNTIME_ROLE, VERIFIER_ROLE)
        )
        membership = _one(connection, _MEMBERSHIP_SQL, 1)
        if (
            roles != expected
            or any(type(flag) is not bool for row in roles for flag in row[1:])
            or type(membership[0]) is not int
            or membership != (0,)
        ):
            raise PassivePostgreSQLError("role flags/membership separation differs", status="FAIL")
        components["role_catalog"] = _result(
            "PASS",
            "both login roles have frozen restricted flags and no incoming/outgoing role memberships; authentication matrix remains separate",
        )
    except Exception as exc:
        components["role_catalog"] = _query_error(exc)
    try:
        if _one(connection, _IDENTITY_SQL, 13) != identity or _one(connection, _SSL_SQL, 5) != ssl:
            raise PassivePostgreSQLError(
                "SQL server/session identity changed during reads", status="FAIL"
            )
        if components["instance_identity"]["status"] == "PASS" and (
            _one(connection, _CONTROL_SQL, 1) != (system_id,)
            or _one(connection, _DATA_SQL, 1) != (directory,)
        ):
            raise PassivePostgreSQLError(
                "PostgreSQL data/system identity changed during reads", status="FAIL"
            )
    except Exception as exc:
        outcome = _query_error(exc)
        if any(value["status"] == "FAIL" for value in components.values()):
            raise PassivePostgreSQLError(
                "SQL binding reread failed after contradictory catalog evidence", status="FAIL"
            ) from None
        raise PassivePostgreSQLError(outcome["detail"], status=outcome["status"]) from None
    return components


def collect_passive_postgresql(observation: dict[str, Any]) -> _LivePostgreSQLReceipt:
    """Read the current legitimate service session, or issue an explicit blocked receipt."""
    receipt = object.__new__(_LivePostgreSQLReceipt)
    fingerprint = None
    binding = None
    components = _blocked_components("approved native service SQL authority is unavailable")
    connection = None
    try:
        fingerprint = _binding_fingerprint(observation)
        binding = _native_binding(observation)
        connection = _connect(binding)
        components = _collect_sql(connection, binding)
        if _native_binding(observation) != binding:
            raise PassivePostgreSQLError(
                "native service/host/listener identity changed during reads", status="FAIL"
            )
        if _binding_fingerprint(observation) != fingerprint:
            raise PassivePostgreSQLError(
                "host observation changed during SQL collection", status="FAIL"
            )
    except Exception as exc:
        outcome = _query_error(exc)
        status = (
            "FAIL"
            if any(value["status"] == "FAIL" for value in components.values())
            else outcome["status"]
        )
        components = _blocked_components(outcome["detail"], status)
    finally:
        if connection is not None:
            try:
                # autocommit plus startup read-only settings avoid an open transaction.
                connection.close()
            except Exception:
                status = (
                    "FAIL"
                    if any(value["status"] == "FAIL" for value in components.values())
                    else "BLOCKED"
                )
                components = _blocked_components("native SQL connection cleanup failed", status)
    _RECEIPTS[receipt] = {
        "observation_id": id(observation),
        "fingerprint": fingerprint,
        "created": time.monotonic(),
        "pid": os.getpid(),
        "identity": binding["identity"] if binding else None,
        "role": binding["role"] if binding else None,
        "components": components,
    }
    return receipt


def summarize_passive_postgresql(receipt: Any, observation: dict[str, Any]) -> dict[str, Any]:
    """Export diagnostics; the returned JSON never carries receipt authority."""
    record = _RECEIPTS.get(receipt) if type(receipt) is _LivePostgreSQLReceipt else None
    valid = (
        record is not None
        and record["observation_id"] == id(observation)
        and record["pid"] == os.getpid()
    )
    if valid and record is not None:
        try:
            valid = 0 <= time.monotonic() - record["created"] <= _MAX_AGE and record[
                "fingerprint"
            ] == _binding_fingerprint(observation)
        except PassivePostgreSQLError:
            valid = False
    if not valid or record is None:
        return {
            "source": "unverified",
            "status": "UNVERIFIED",
            "identity": None,
            "role": None,
            "components": _blocked_components(
                "saved, forged, stale or differently bound SQL evidence has no live authority",
                "UNVERIFIED",
            ),
        }
    components = copy.deepcopy(record["components"])
    return {
        "source": "live_passive_postgresql",
        "status": _overall(components),
        "binding_sha256": record["fingerprint"],
        "identity": record["identity"],
        "role": record["role"],
        "components": components,
    }


def require_passive_postgresql_component(
    receipt: Any, observation: dict[str, Any], component: str
) -> None:
    if component not in COMPONENTS:
        raise PassivePostgreSQLError("unknown passive PostgreSQL component", status="FAIL")
    result = summarize_passive_postgresql(receipt, observation)["components"][component]
    if result["status"] != "PASS":
        raise PassivePostgreSQLError(result["detail"], status=result["status"])
