"""Stałe, odczytowe obserwacje połączenia należącego do procesu usługi.

To API nie otwiera połączenia, nie przyjmuje SQL i nie zmienia tożsamości.
Wywołuje je właściwa aplikacja na własnym połączeniu. Surowy wynik nie jest
kwalifikacją instalacji: obserwator musi niezależnie przypiąć proces, wydanie,
wyzwanie oraz konfigurację HBA/ident. Brak runtime nadal blokuje pełny kontrakt.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import sys
from typing import Any

import psycopg
from psycopg.pq import TransactionStatus

ADAPTER_ID = "cryptohunter.linux.postgresql.service-owned-readonly.v1"
IDENTITY_SQL = (
    "SELECT system_user::text,session_user::text,current_user::text,pg_backend_pid(),"
    "current_database(),inet_client_addr(),inet_server_addr(),"
    "current_setting('transaction_read_only'),"
    "coalesce((SELECT ssl FROM pg_catalog.pg_stat_ssl WHERE pid=pg_backend_pid()),false)"
)
BACKEND_SQL = (
    "SELECT pid,usename::text,datname::text,client_addr,client_port,backend_type "
    "FROM pg_catalog.pg_stat_activity WHERE pid=pg_backend_pid()"
)
ALLOWED_SQL = "SELECT 1::integer"
DENIAL_SQL = (
    "SELECT 1 FROM pg_catalog.pg_authid LIMIT 0",
    "SELECT 1 FROM freshness_authority.credentials LIMIT 0",
    "SELECT 1 FROM freshness_authority.prepared_verifications LIMIT 0",
)
OPPOSITE_ROLE_SQL = {
    "runtime": (
        "SELECT freshness_authority.prepare_verified_freshness_candidate"
        "(NULL::bytea,NULL::bytea,NULL::bytea)"
    ),
    "verifier": (
        "SELECT freshness_authority.compare_and_advance"
        "(NULL::text,NULL::jsonb,NULL::bytea,NULL::bytea)"
    ),
}


class ServicePostgreSQLObservationError(ValueError):
    """Połączenia nie udało się bezpiecznie powiązać z procesem i peer."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ServicePostgreSQLObservationError(code)


def _read(directory: int, name: str) -> bytes:
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
    try:
        raw = os.read(descriptor, 65537)
        _require(0 < len(raw) <= 65536, "BOUNDED_PROCESS_OBSERVATION_REQUIRED")
        return raw
    finally:
        os.close(descriptor)


def observe_current_process() -> dict[str, Any]:
    """Odczytaj własny okres życia, wszystkie UID/GID, program i cgroup v2."""
    if sys.platform != "linux":
        raise ServicePostgreSQLObservationError("NATIVE_LINUX_REQUIRED")
    directory = os.open("/proc/self", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    executable = None
    try:
        before = _read(directory, "stat")
        pid = int(before.split(b" ", 1)[0])
        ticks = int(before[before.rfind(b")") + 2 :].split()[19])
        fields = dict(line.split(b":", 1) for line in _read(directory, "status").splitlines())
        uids = list(map(int, fields[b"Uid"].split()))
        gids = list(map(int, fields[b"Gid"].split()))
        groups = sorted(map(int, fields[b"Groups"].split()))
        cgroup = _read(directory, "cgroup").decode("utf-8", errors="strict").splitlines()
        _require(
            len(cgroup) == 1 and cgroup[0].startswith("0::/") and ticks > 0,
            "EXACT_PROCESS_CGROUP_V2_REQUIRED",
        )
        executable = os.open("exe", os.O_RDONLY | os.O_CLOEXEC, dir_fd=directory)
        first = os.fstat(executable)
        _require(stat.S_ISREG(first.st_mode), "PROCESS_EXECUTABLE_REQUIRED")
        hasher = hashlib.sha256()
        count = 0
        while chunk := os.read(executable, 65536):
            count += len(chunk)
            _require(count <= 64 * 1024 * 1024, "BOUNDED_EXECUTABLE_REQUIRED")
            hasher.update(chunk)
        last = os.fstat(executable)
        linked = os.stat("exe", dir_fd=directory)
        after = _read(directory, "stat")
        _require(
            (first.st_dev, first.st_ino, first.st_size, first.st_mtime_ns, first.st_ctime_ns)
            == (last.st_dev, last.st_ino, last.st_size, last.st_mtime_ns, last.st_ctime_ns)
            == (
                linked.st_dev,
                linked.st_ino,
                linked.st_size,
                linked.st_mtime_ns,
                linked.st_ctime_ns,
            )
            and int(after[after.rfind(b")") + 2 :].split()[19]) == ticks
            and pid == os.getpid()
            and len(uids) == len(gids) == 4
            and len(set(uids)) == len(set(gids)) == 1,
            "PROCESS_CHANGED_DURING_OBSERVATION",
        )
        return {
            "pid": pid,
            "uid": uids[0],
            "gid": gids[0],
            "groups": groups,
            "start_time_ticks": ticks,
            "exe_sha256": hasher.hexdigest(),
            "cgroup": cgroup[0][3:],
        }
    except (OSError, KeyError, IndexError, UnicodeError, ValueError) as exc:
        if isinstance(exc, ServicePostgreSQLObservationError):
            raise
        raise ServicePostgreSQLObservationError("PROCESS_OBSERVATION_UNAVAILABLE") from None
    finally:
        if executable is not None:
            os.close(executable)
        os.close(directory)


def observe_service_connection(
    connection: psycopg.Connection[Any],
    *,
    challenge_id: str,
    installation_id: str,
    role: str,
    expected_uid: int,
    expected_gid: int,
    expected_os_principal: str,
    expected_database_role: str,
    expected_executable_sha256: str,
    expected_cgroup: str,
) -> dict[str, Any]:
    """Obserwuj własne, bezczynne połączenie; zawsze wycofaj swoją transakcję.

    Parametry pochodzą z konstrukcji usługi i chronionej instalacji, nigdy z
    żądania IPC. LIMIT 0 sprawdza rzeczywiste ACL bez odczytu cudzych wierszy.
    NULL w przeciwnej funkcji nie daje klientowi żadnych parametrów biznesowych;
    READ ONLY dodatkowo uniemożliwia trwałe zapisy nawet przy wadliwym ACL.
    """
    if sys.platform != "linux":
        raise ServicePostgreSQLObservationError("NATIVE_LINUX_REQUIRED")
    _require(
        all(
            type(v) is str and re.fullmatch(r"[0-9a-f]{64}", v)
            for v in (challenge_id, installation_id, expected_executable_sha256)
        )
        and role in OPPOSITE_ROLE_SQL
        and all(
            type(v) is str and re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", v)
            for v in (expected_os_principal, expected_database_role)
        )
        and type(expected_uid) is int
        and expected_uid > 0
        and type(expected_gid) is int
        and expected_gid > 0
        and type(expected_cgroup) is str
        and expected_cgroup.startswith("/")
        and "\n" not in expected_cgroup,
        "EXACT_SERVICE_POLICY_REQUIRED",
    )
    _require(
        not any(name.startswith("PG") for name in os.environ),
        "AMBIENT_LIBPQ_CONFIGURATION_FORBIDDEN",
    )
    _require(isinstance(connection, psycopg.Connection), "OWNED_PSYCOPG_CONNECTION_REQUIRED")
    _require(
        not connection.closed
        and not connection.autocommit
        and connection.info.transaction_status == TransactionStatus.IDLE,
        "IDLE_SERVICE_CONNECTION_REQUIRED",
    )
    import pwd

    try:
        principal = pwd.getpwuid(expected_uid)
    except KeyError:
        raise ServicePostgreSQLObservationError("SYSTEM_PRINCIPAL_UNAVAILABLE") from None
    _require(principal.pw_name == expected_os_principal, "SYSTEM_PRINCIPAL_MISMATCH")
    before = observe_current_process()
    _require(
        before["uid"] == expected_uid
        and before["gid"] == expected_gid
        and before["exe_sha256"] == expected_executable_sha256
        and before["cgroup"] == expected_cgroup
        and set(os.getresuid()) == {expected_uid}
        and set(os.getresgid()) == {expected_gid},
        "SERVICE_PROCESS_MISMATCH",
    )
    started = False
    try:
        started = True
        connection.execute("SET TRANSACTION READ ONLY")
        connection.execute("SET LOCAL statement_timeout=5000")
        connection.execute("SET LOCAL lock_timeout=1000")
        connection.execute("SET LOCAL idle_in_transaction_session_timeout=5000")
        identity = connection.execute(IDENTITY_SQL).fetchone()
        _require(
            isinstance(identity, tuple)
            and len(identity) == 9
            and identity[0] == "peer:" + expected_os_principal
            and identity[1] == identity[2] == expected_database_role
            and type(identity[3]) is int
            and identity[3] > 0
            and identity[3] == connection.info.backend_pid
            and identity[4] == "freshness_gate"
            and identity[5:] == (None, None, "on", False),
            "AUTHENTICATED_PEER_MAPPING_REQUIRED",
        )
        backend = connection.execute(BACKEND_SQL).fetchone()
        _require(
            backend
            == (identity[3], expected_database_role, "freshness_gate", None, -1, "client backend"),
            "SERVICE_BACKEND_MISMATCH",
        )
        _require(connection.execute(ALLOWED_SQL).fetchone() == (1,), "ALLOWED_READ_FAILED")
        denials = []
        for sql in (*DENIAL_SQL, OPPOSITE_ROLE_SQL[role]):
            denied = False
            try:
                with connection.transaction():
                    connection.execute(sql)
            except psycopg.Error as exc:
                _require(exc.sqlstate == "42501", "EFFECTIVE_PRIVILEGE_DENIAL_REQUIRED")
                denied = True
            _require(denied, "FORBIDDEN_OPERATION_SUCCEEDED")
            denials.append({"sql": sql, "sqlstate": "42501"})
        _require(observe_current_process() == before, "SERVICE_PROCESS_CHANGED")
        _require(
            connection.execute(IDENTITY_SQL).fetchone() == identity, "SERVICE_CONNECTION_CHANGED"
        )
        return {
            "schema_version": 1,
            "adapter_id": ADAPTER_ID,
            "scope": "SERVICE_OWNED_CONNECTION_OBSERVATION",
            "challenge_id": challenge_id,
            "installation_id": installation_id,
            "role": role,
            "process": before,
            "identity_sql": IDENTITY_SQL,
            "identity_row": list(identity),
            "backend_sql": BACKEND_SQL,
            "backend_row": list(backend),
            "allowed": {"sql": ALLOWED_SQL, "row": [1]},
            "denials": denials,
            "transaction": "READ_ONLY_ROLLED_BACK",
        }
    except ServicePostgreSQLObservationError:
        raise
    except (psycopg.Error, OSError, TypeError, ValueError):
        raise ServicePostgreSQLObservationError("SERVICE_POSTGRESQL_OBSERVATION_FAILED") from None
    finally:
        if started:
            try:
                connection.rollback()
            except psycopg.Error:
                raise ServicePostgreSQLObservationError(
                    "SERVICE_TRANSACTION_ROLLBACK_FAILED"
                ) from None
