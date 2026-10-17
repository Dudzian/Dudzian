"""Fixed SQL capture inside an already running canonical Windows service.

The transport owns authentication. This module never accepts an observation,
SQL text, credential location or connection factory from a requester. Native
checks concern only this process and its own SCM entry; host-wide inspection
belongs to the administrator client. The yielded connection remains open until
the transport has received its bounded acknowledgement.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import importlib
import math
import os
from pathlib import PureWindowsPath
import re
import struct
import sys
import threading
import time
from typing import Any, Iterator

from deployment import windows_stage9_postgresql_passive as passive
from deployment.windows_installer.contract import CONTRACT
from deployment.windows_postgresql_auth_contract import (
    RUNTIME_IDENTITY,
    RUNTIME_ROLE,
    VERIFIER_IDENTITY,
    VERIFIER_ROLE,
)

MAX_REQUEST_AGE = 30.0
_HEX = re.compile(r"[0-9a-f]{64}")
_SQLSTATE = re.compile(r"[0-9A-Z]{5}")
_SERVICES = (CONTRACT.backend_service, CONTRACT.verifier_service)
QUERY_SQL = {
    "identity": passive._IDENTITY_SQL,
    "ssl": passive._SSL_SQL,
    "data": passive._DATA_SQL,
    "control": passive._CONTROL_SQL,
    "hba": passive._HBA_SQL,
    "roles": passive._ROLES_SQL,
    "membership": passive._MEMBERSHIP_SQL,
}
QUERY_NAMES = {sql: name for name, sql in QUERY_SQL.items()}
QUERY_COUNTS = {
    "identity": 2,
    "ssl": 2,
    "data": 2,
    "control": 3,
    "hba": 1,
    "roles": 1,
    "membership": 1,
}
_LIMITS = {"identity": 1, "ssl": 1, "data": 1, "control": 1, "hba": 64, "roles": 2, "membership": 1}
_PLAN = (
    "identity",
    "ssl",
    "data",
    "control",
    "control",
    "hba",
    "roles",
    "membership",
    "identity",
    "ssl",
    "control",
    "data",
)


def _blocked(detail: str) -> passive.PassivePostgreSQLError:
    return passive.PassivePostgreSQLError(detail)


def validate_request(request: Any, stop: threading.Event) -> float:
    """Check the only accepted wire request, using the native monotonic clock."""
    if type(request) is not dict or set(request) != {
        "version",
        "nonce",
        "collection_id",
        "issued",
        "expires",
    }:
        raise _blocked("service SQL request schema is unavailable")
    if type(request["version"]) is not int or request["version"] != 1:
        raise _blocked("service SQL protocol version is unavailable")
    for field in ("nonce", "collection_id"):
        if type(request[field]) is not str or _HEX.fullmatch(request[field]) is None:
            raise _blocked("service SQL challenge is malformed")
    for field in ("issued", "expires"):
        try:
            valid_time = (
                type(request[field]) in (int, float)
                and math.isfinite(request[field])
                and request[field] >= 0
            )
        except OverflowError:
            valid_time = False
        if not valid_time:
            raise _blocked("service SQL deadline is malformed")
    issued, expires = request["issued"], request["expires"]
    now = time.monotonic()
    if not issued <= now < expires or not 0 < expires - issued <= MAX_REQUEST_AGE:
        raise _blocked("service SQL challenge is stale or has an invalid lifetime")
    _check_deadline(float(expires), stop)
    return float(expires)


def _check_deadline(deadline: float, stop: threading.Event) -> None:
    if stop.is_set() or time.monotonic() >= deadline:
        raise _blocked("service SQL collection stopped or exceeded its deadline")


def _own_native_binding(name: str) -> dict[str, Any]:
    """Read this process and this service only, without impersonation."""
    if name not in _SERVICES or os.name != "nt" or struct.calcsize("P") != 8:
        raise _blocked("canonical native Windows x64 service process is required")
    passive._native_x64()
    api = importlib.import_module("win32api")
    security = importlib.import_module("win32security")
    process = importlib.import_module("win32process")
    scm = importlib.import_module("win32service")
    try:
        thread = security.OpenThreadToken(api.GetCurrentThread(), 8, True)
    except Exception as exc:
        if getattr(exc, "winerror", None) != 1008:
            raise _blocked("service thread-token absence cannot be established") from None
    else:
        thread.Close()
        raise _blocked("service SQL collection forbids thread impersonation")
    own_process = api.GetCurrentProcess()
    token = security.OpenProcessToken(own_process, 8)
    try:
        user = security.GetTokenInformation(token, 1)
        sid = security.ConvertSidToStringSid(user[0] if isinstance(user, tuple) else user)
    finally:
        token.Close()
    expected_sid = security.ConvertSidToStringSid(
        security.LookupAccountName(None, rf"NT SERVICE\{name}")[0]
    )
    if sid != expected_sid:
        raise _blocked("service SQL process TokenUser is not its canonical service SID")
    paths = {
        "program_files": passive._known_folder(0x26),
        "program_data": passive._known_folder(0x23),
    }
    expected = str(PureWindowsPath(paths["program_files"]) / f"{name}.exe")
    image = process.QueryFullProcessImageName(own_process, 0)
    created = process.GetProcessTimes(own_process)["CreationTime"].isoformat()
    passive._timestamp(created)
    if not passive._same_path(image, expected) or not passive._same_path(sys.executable, expected):
        raise passive.PassivePostgreSQLError("canonical service executable differs", status="FAIL")
    manager = scm.OpenSCManager(None, None, scm.SC_MANAGER_CONNECT)
    service = None
    try:
        service = scm.OpenService(
            manager, name, scm.SERVICE_QUERY_CONFIG | scm.SERVICE_QUERY_STATUS
        )
        config = scm.QueryServiceConfig(service)
        status = scm.QueryServiceStatusEx(service)
        if (
            config[0] != 16
            or config[7] != rf"NT SERVICE\{name}"
            or config[3] not in (f'"{expected}"', expected if " " not in expected else None)
            or status["CurrentState"] != 4
            or status["ProcessId"] != os.getpid()
        ):
            raise _blocked("current process is not its canonical running SCM service instance")
    finally:
        if service is not None:
            scm.CloseServiceHandle(service)
        scm.CloseServiceHandle(manager)
    return {
        "name": name,
        "identity": name,
        "role": RUNTIME_ROLE if name == CONTRACT.backend_service else VERIFIER_ROLE,
        "certificate_identity": RUNTIME_IDENTITY
        if name == CONTRACT.backend_service
        else VERIFIER_IDENTITY,
        "process_id": os.getpid(),
        "process_path": image,
        "process_started": created,
        "token_sid": sid,
        "paths": paths,
    }


def _wire(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        raise passive.PassivePostgreSQLError("SQL result nesting exceeds its bound", status="FAIL")
    if value is None or type(value) in (bool, int):
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise passive.PassivePostgreSQLError("SQL timestamp has no timezone", status="FAIL")
        return value.isoformat()
    if type(value) is str and len(value.encode("utf-8")) <= 8192:
        return value
    if type(value) in (tuple, list) and len(value) <= 65:
        return [_wire(item, depth + 1) for item in value]
    raise passive.PassivePostgreSQLError(
        "SQL result value exceeds the wire contract", status="FAIL"
    )


def _sqlstate(exc: Exception) -> str | None:
    psycopg = importlib.import_module("psycopg")
    if not isinstance(exc, psycopg.Error):
        return None
    value = getattr(exc, "sqlstate", None)
    return value if type(value) is str and _SQLSTATE.fullmatch(value) else None


def _capture(connection: Any, name: str, deadline: float, stop: threading.Event) -> dict[str, Any]:
    _check_deadline(deadline, stop)
    try:
        rows = passive._rows(connection, QUERY_SQL[name], _LIMITS[name])
    except passive.PassivePostgreSQLError:
        raise
    except Exception as exc:
        result = {"sqlstate": _sqlstate(exc)}
    else:
        result = {"rows": _wire(rows)}
    _check_deadline(deadline, stop)
    return result


def _probe(
    binding: dict[str, Any], role: str, deadline: float, stop: threading.Event
) -> dict[str, Any]:
    _check_deadline(deadline, stop)
    try:
        # Identity selects the owning credential directory; only the fixed
        # forbidden PostgreSQL role changes. No other private key is inspected.
        connection = passive._connect({**binding, "role": role})
    except Exception as exc:
        state = _sqlstate(exc)
        result = {
            "status": "DENIED" if state in {"28000", "28P01"} else "BLOCKED",
            "sqlstate": state,
        }
    else:
        try:
            connection.close()
        except Exception:
            # Acceptance is already contradictory even when cleanup fails.
            pass
        result = {"status": "ACCEPTED", "sqlstate": None}
    _check_deadline(deadline, stop)
    return result


@contextmanager
def collect_service_sql(name: str, request: Any, stop: threading.Event) -> Iterator[dict[str, Any]]:
    """Capture fixed facts under the real owning service token, holding SQL open."""
    deadline = validate_request(request, stop)
    binding = _own_native_binding(name)
    connection = None
    try:
        _check_deadline(deadline, stop)
        connection = passive._connect(binding)
        _check_deadline(deadline, stop)
        backend_pid = connection.info.backend_pid
        if type(backend_pid) is not int or backend_pid <= 0:
            raise passive.PassivePostgreSQLError(
                "native SQL backend PID is malformed", status="FAIL"
            )
        queries: dict[str, list[dict[str, Any]]] = {key: [] for key in QUERY_SQL}
        for query in _PLAN:
            queries[query].append(_capture(connection, query, deadline, stop))
        other = VERIFIER_ROLE if name == CONTRACT.backend_service else RUNTIME_ROLE
        probes = {role: _probe(binding, role, deadline, stop) for role in (other, "postgres")}
        post_probe = {
            query: _capture(connection, query, deadline, stop)
            for query in ("identity", "ssl", "control", "data")
        }
        if _own_native_binding(name) != binding:
            raise passive.PassivePostgreSQLError(
                "service process changed during SQL capture", status="FAIL"
            )
        _check_deadline(deadline, stop)
        yield {
            "version": 1,
            "nonce": request["nonce"],
            "collection_id": request["collection_id"],
            "service": name,
            "process": binding,
            "backend_pid": backend_pid,
            "queries": queries,
            "probes": probes,
            "post_probe": post_probe,
        }
        _check_deadline(deadline, stop)
        if _own_native_binding(name) != binding:
            raise passive.PassivePostgreSQLError(
                "service process changed before SQL acknowledgement", status="FAIL"
            )
    finally:
        if connection is not None:
            active_error = sys.exc_info()[0] is not None
            try:
                connection.close()
            except Exception:
                if not active_error:
                    raise _blocked("native service SQL connection cleanup failed") from None
