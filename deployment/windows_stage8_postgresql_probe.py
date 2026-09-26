"""Live, fail-closed Windows PostgreSQL substrate and SSPI qualification.

This is an acceptance probe, not an installer.  It owns an isolated temporary
cluster and temporary SCM helper services and never opens or changes a system
PGDATA or a pre-installed PostgreSQL service.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any

from deployment.host_identity import canonical_host_os
from deployment.platform_evidence import WINDOWS_STAGE8_ITEMS

SUBSTRATE, PRINCIPAL_AUTH = WINDOWS_STAGE8_ITEMS
MINIMUM_MAJOR = 16
COMMAND_TIMEOUT = 30
START_TIMEOUT = 30
SERVICE_TIMEOUT = 20
STDERR_LIMIT = 8192
DATABASE = "stage8_freshness"
RUNTIME_SERVICE = "CryptoHunterBackend"
VERIFIER_SERVICE = "CryptoHunterFreshnessVerifier"
RUNTIME_ACCOUNT = rf"NT SERVICE\{RUNTIME_SERVICE}"
VERIFIER_ACCOUNT = rf"NT SERVICE\{VERIFIER_SERVICE}"
RUNTIME_ROLE = "freshness_runtime"
VERIFIER_ROLE = "freshness_crypto_verifier"
MAP_NAME = "stage8_sspi"
REQUIRED_BINARIES = ("initdb.exe", "postgres.exe", "pg_ctl.exe", "psql.exe")
FORBIDDEN_ONLINE_METHODS = frozenset({"trust", "password", "md5", "scram-sha-256"})
AUTHENTICATED_RE = re.compile(r'connection authenticated: identity="([^"]+)" method=sspi')
DIAGNOSTIC_LABELS = (
    "POSTGRESQL_DISCOVERY",
    "POSTGRESQL_CLUSTER_INIT",
    "POSTGRESQL_START",
    "POSTGRESQL_DURABILITY",
    "POSTGRESQL_SCHEMA_PROVISION",
    "POSTGRESQL_SCHEMA_QUALIFICATION",
    "WINDOWS_SSPI_RUNTIME_IDENTITY",
    "WINDOWS_SSPI_VERIFIER_IDENTITY",
    "WINDOWS_SSPI_RUNTIME_CONNECT",
    "WINDOWS_SSPI_VERIFIER_CONNECT",
    "WINDOWS_SSPI_CROSS_ROLE_DENIAL",
    "WINDOWS_SSPI_OUTSIDER_DENIAL",
    "WINDOWS_SSPI_HBA_QUALIFICATION",
    "WINDOWS_SSPI_IDENT_QUALIFICATION",
    "STAGE8_CLEANUP",
)


class Stage8PostgreSQLProbeError(RuntimeError):
    def __init__(self, label: str, item: str, detail: str) -> None:
        self.label, self.item, self.detail = label, item, detail
        super().__init__(f"[{label}] {detail}")


@dataclass(frozen=True)
class PostgreSQLBinaries:
    bindir: Path
    major: int

    def executable(self, name: str) -> Path:
        return self.bindir / name


def _run(command: list[str], *, timeout: int = COMMAND_TIMEOUT) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise Stage8PostgreSQLProbeError("POSTGRESQL_DISCOVERY", SUBSTRATE, "bounded command timed out") from exc


def discover_postgresql(environment: dict[str, str] | None = None) -> PostgreSQLBinaries:
    values = os.environ if environment is None else environment
    candidates: list[Path] = []
    if values.get("PGBIN"):
        candidates.append(Path(values["PGBIN"]))
    pg_config = shutil.which("pg_config.exe") or shutil.which("pg_config")
    if pg_config:
        result = _run([pg_config, "--bindir"])
        if result.returncode == 0 and result.stdout.strip():
            candidates.append(Path(result.stdout.strip()))
    for candidate in candidates:
        missing = [name for name in REQUIRED_BINARIES if not (candidate / name).is_file()]
        if missing:
            continue
        version = _run([str(candidate / "postgres.exe"), "--version"])
        match = re.search(r"(\d+)(?:\.\d+)?", version.stdout + version.stderr)
        if version.returncode or not match:
            continue
        major = int(match.group(1))
        if major < MINIMUM_MAJOR:
            raise Stage8PostgreSQLProbeError("POSTGRESQL_DISCOVERY", SUBSTRATE, f"PostgreSQL {major} is below required major 16")
        return PostgreSQLBinaries(candidate.resolve(), major)
    raise Stage8PostgreSQLProbeError("POSTGRESQL_DISCOVERY", SUBSTRATE, "required PostgreSQL binaries were not found via PGBIN or pg_config")


def final_hba_lines() -> tuple[str, ...]:
    options = f"sspi map={MAP_NAME} include_realm=1"
    return (
        f"host {DATABASE} {VERIFIER_ROLE} 127.0.0.1/32 {options}",
        f"host {DATABASE} {RUNTIME_ROLE} 127.0.0.1/32 {options}",
        f"host {DATABASE} all 127.0.0.1/32 reject",
        "host all all 127.0.0.1/32 reject",
        "host all all 0.0.0.0/0 reject",
        "host all all ::0/0 reject",
    )


def ident_lines(runtime_principal: str, verifier_principal: str) -> tuple[str, ...]:
    if not runtime_principal or not verifier_principal or runtime_principal.casefold() == verifier_principal.casefold():
        raise Stage8PostgreSQLProbeError("WINDOWS_SSPI_IDENT_QUALIFICATION", PRINCIPAL_AUTH, "virtual service accounts did not yield two distinct SSPI principals")
    if any(re.search(r"[\s/]|\.\*|\^|\$", value) for value in (runtime_principal, verifier_principal)):
        raise Stage8PostgreSQLProbeError("WINDOWS_SSPI_IDENT_QUALIFICATION", PRINCIPAL_AUTH, "unsafe principal syntax cannot be represented by an exact pg_ident map")
    return (
        f"{MAP_NAME} {runtime_principal} {RUNTIME_ROLE}",
        f"{MAP_NAME} {verifier_principal} {VERIFIER_ROLE}",
    )


def qualify_hba_rows(rows: list[tuple[Any, ...]]) -> None:
    expected = [
        (DATABASE, VERIFIER_ROLE, "sspi"),
        (DATABASE, RUNTIME_ROLE, "sspi"),
        (DATABASE, "all", "reject"),
        ("all", "all", "reject"),
        ("all", "all", "reject"),
        ("all", "all", "reject"),
    ]
    if len(rows) != len(expected):
        raise Stage8PostgreSQLProbeError("WINDOWS_SSPI_HBA_QUALIFICATION", PRINCIPAL_AUTH, "unexpected effective HBA row count")
    previous = -1
    for row, wanted in zip(rows, expected, strict=True):
        line, kind, databases, users, address, netmask, method, options, error = row
        if error is not None or line <= previous or kind != "host":
            raise Stage8PostgreSQLProbeError("WINDOWS_SSPI_HBA_QUALIFICATION", PRINCIPAL_AUTH, "invalid HBA parse result or ordering")
        previous = line
        database, user, auth = wanted
        if databases != [database] or users != [user] or method != auth:
            raise Stage8PostgreSQLProbeError("WINDOWS_SSPI_HBA_QUALIFICATION", PRINCIPAL_AUTH, "effective HBA does not match the exact reviewed sequence")
        if auth == "sspi" and (address != "127.0.0.1" or netmask != "255.255.255.255" or sorted(options or []) != ["include_realm=1", f"map={MAP_NAME}"]):
            raise Stage8PostgreSQLProbeError("WINDOWS_SSPI_HBA_QUALIFICATION", PRINCIPAL_AUTH, "online HBA is not exact loopback SSPI")
        if user in {RUNTIME_ROLE, VERIFIER_ROLE} and method in FORBIDDEN_ONLINE_METHODS:
            raise Stage8PostgreSQLProbeError("WINDOWS_SSPI_HBA_QUALIFICATION", PRINCIPAL_AUTH, "forbidden online authentication method")


def _qualify_effective_hba_offline(postgres: Path, data: Path) -> None:
    """Read the server's parser output without adding an administrative HBA rule."""
    query = """COPY (
SELECT json_build_array(line_number,type,database,user_name,address,netmask,auth_method,options,error)
FROM pg_catalog.pg_hba_file_rules ORDER BY line_number
) TO STDOUT;
"""
    try:
        completed = subprocess.run(
            [str(postgres), "--single", "-D", str(data), DATABASE],
            input=query,
            check=False,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise Stage8PostgreSQLProbeError("WINDOWS_SSPI_HBA_QUALIFICATION", PRINCIPAL_AUTH, "offline HBA parser timed out") from exc
    rows: list[tuple[Any, ...]] = []
    for line in completed.stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and len(value) == 9:
            rows.append(tuple(value))
    if completed.returncode or not rows:
        raise Stage8PostgreSQLProbeError("WINDOWS_SSPI_HBA_QUALIFICATION", PRINCIPAL_AUTH, (completed.stderr or completed.stdout)[-STDERR_LIMIT:])
    qualify_hba_rows(rows)


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _admin_dsn(port: int, database: str = "postgres") -> str:
    return f"host=127.0.0.1 port={port} dbname={database} user=stage8_bootstrap connect_timeout=5"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _wait_invocation_message(
    path: Path,
    invocation_token: str,
    *,
    expected_pid: int | None = None,
    timeout: float = SERVICE_TIMEOUT,
) -> dict[str, Any]:
    """Wait for a message bound to this invocation, ignoring stale contents."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = _read_json(path)
        if value is not None and secrets.compare_digest(
            str(value.get("invocation_token", "")), invocation_token
        ):
            pid = value.get("pid")
            if type(pid) is int and (expected_pid is None or pid == expected_pid):
                return value
        time.sleep(0.05)
    raise TimeoutError(f"deadline expired waiting for fresh invocation message {path.name}")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


def _sc(*args: str, timeout: int = COMMAND_TIMEOUT) -> subprocess.CompletedProcess[str]:
    return _run(["sc.exe", *args], timeout=timeout)


def _service_command(service: str, request: Path) -> str:
    helper = Path(__file__).with_name("windows_stage8_postgresql_service.py").resolve()
    return f'"{sys.executable}" "{helper}" --service-name {service} --request "{request}"'


def _service_is_absent(result: subprocess.CompletedProcess[str]) -> bool:
    return result.returncode != 0 and "1060" in (result.stdout + result.stderr)


def _create_service(name: str, account: str, request: Path) -> None:
    existing = _sc("query", name)
    if not _service_is_absent(existing):
        raise RuntimeError(f"refusing to replace or adopt pre-existing service {name}")
    result = _sc(
        "create",
        name,
        "binPath=",
        _service_command(name, request),
        "start=",
        "demand",
        "obj=",
        account,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout)[-STDERR_LIMIT:]
        raise RuntimeError(f"SCM create failed for {name}: {detail}")


def _service_observation(name: str) -> tuple[str, int]:
    result = _sc("queryex", name)
    if result.returncode:
        raise RuntimeError(f"SCM query failed for owned service {name}")
    state_match = re.search(r"STATE\s*:\s*\d+\s+(\w+)", result.stdout, re.I)
    pid_match = re.search(r"PID\s*:\s*(\d+)", result.stdout, re.I)
    if not state_match or not pid_match:
        raise RuntimeError(f"SCM returned an unparseable state for {name}")
    return state_match.group(1).upper(), int(pid_match.group(1))


def _wait_service_state(name: str, expected: str, timeout: int = SERVICE_TIMEOUT) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state, pid = _service_observation(name)
        if state == expected:
            return pid
        time.sleep(0.05)
    raise TimeoutError(f"deadline expired waiting for service {name} state {expected}")


def _wait_service_absent(name: str, timeout: int = SERVICE_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = _sc("query", name)
        if _service_is_absent(result):
            return
        # 1072 is ERROR_SERVICE_MARKED_FOR_DELETE, not absence.
        time.sleep(0.05)
    raise TimeoutError(f"deadline expired waiting for deletion of owned service {name}")


def _token_identity(pid: int) -> tuple[str, str]:
    import win32api
    import win32con
    import win32security

    process = win32api.OpenProcess(win32con.PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    try:
        token = win32security.OpenProcessToken(process, win32con.TOKEN_QUERY)
        try:
            sid = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
            name, domain, _ = win32security.LookupAccountSid(None, sid)
            return win32security.ConvertSidToStringSid(sid), f"{domain}\\{name}"
        finally:
            token.Close()
    finally:
        process.Close()


def _assert_expected_token(pid: int, expected: str) -> tuple[str, str]:
    import win32security

    sid, account = _token_identity(pid)
    expected_sid, _, _ = win32security.LookupAccountName(None, expected)
    if sid != win32security.ConvertSidToStringSid(expected_sid) or account.casefold() != expected.casefold():
        raise RuntimeError(f"service token mismatch: pid={pid}, sid={sid}, account={account}")
    return sid, account


def _open_process_exit_witness(pid: int) -> Any:
    """Hold a handle to the exact helper process, avoiding PID-reuse ambiguity."""
    import win32api
    import win32con

    return win32api.OpenProcess(win32con.SYNCHRONIZE, False, pid)


def _wait_process_exit(process: Any, timeout: float = SERVICE_TIMEOUT) -> None:
    import win32event

    if win32event.WaitForSingleObject(process, int(timeout * 1000)) != win32event.WAIT_OBJECT_0:
        raise TimeoutError("deadline expired waiting for helper process exit")


def _close_process_exit_witness(process: Any) -> None:
    process.Close()


def _service_failure_label(service: str, *, identity: bool) -> str:
    if service == RUNTIME_SERVICE:
        return "WINDOWS_SSPI_RUNTIME_IDENTITY" if identity else "WINDOWS_SSPI_RUNTIME_CONNECT"
    return "WINDOWS_SSPI_VERIFIER_IDENTITY" if identity else "WINDOWS_SSPI_VERIFIER_CONNECT"


def _prepare_services(
    root: Path, owned_services: set[str]
) -> dict[str, Path]:
    requests: dict[str, Path] = {}
    for service, account in (
        (RUNTIME_SERVICE, RUNTIME_ACCOUNT),
        (VERIFIER_SERVICE, VERIFIER_ACCOUNT),
    ):
        label = _service_failure_label(service, identity=True)
        request = root / f"{service}.current-request.json"
        try:
            _create_service(service, account, request)
        except Exception as exc:
            raise Stage8PostgreSQLProbeError(label, PRINCIPAL_AUTH, str(exc)) from exc
        owned_services.add(service)
        requests[service] = request
        try:
            acl = _run(
                ["icacls.exe", str(root), "/grant", f"{account}:(OI)(CI)F"],
                timeout=COMMAND_TIMEOUT,
            )
        except Exception as exc:
            raise Stage8PostgreSQLProbeError(label, PRINCIPAL_AUTH, str(exc)) from exc
        if acl.returncode:
            raise Stage8PostgreSQLProbeError(
                label,
                PRINCIPAL_AUTH,
                "could not grant helper service access to probe-owned scratch: "
                f"{(acl.stderr or acl.stdout)[-512:]}",
            )
    return requests


def _invoke_service(
    root: Path,
    request: Path,
    service: str,
    account: str,
    port: int,
    role: str,
    database: str = DATABASE,
    *,
    failure_label: str | None = None,
) -> dict[str, Any]:
    label = failure_label or _service_failure_label(service, identity=False)
    invocation_token = secrets.token_hex(32)
    invocation = root / "invocations" / invocation_token
    invocation.mkdir(parents=True)
    ready, go, result = (invocation / name for name in ("ready.json", "go.json", "result.json"))
    _atomic_json(
        request,
        {
            "invocation_token": invocation_token,
            "port": port,
            "database": database,
            "role": role,
            "ready": str(ready),
            "go": str(go),
            "result": str(result),
        },
    )
    witness: Any | None = None
    try:
        started = _sc("start", service)
        if started.returncode:
            raise RuntimeError((started.stderr or started.stdout)[-STDERR_LIMIT:])
        ready_payload = _wait_invocation_message(ready, invocation_token)
        if ready_payload.get("ready") is not True:
            raise RuntimeError("fresh READY message did not assert readiness")
        scm_pid = _wait_service_state(service, "RUNNING")
        if ready_payload["pid"] != scm_pid:
            raise RuntimeError(
                f"READY PID does not match SCM PID: ready={ready_payload['pid']}, scm={scm_pid}"
            )
        witness = _open_process_exit_witness(scm_pid)
        # The independently observed process token is authority. GO is published
        # only after this check succeeds, so the child cannot connect beforehand.
        sid, observed = _assert_expected_token(scm_pid, account)
        _atomic_json(go, {"invocation_token": invocation_token, "pid": scm_pid, "go": True})
        payload = _wait_invocation_message(result, invocation_token, expected_pid=scm_pid)
        _wait_service_state(service, "STOPPED")
        _wait_process_exit(witness)
        payload.update({"service": service, "pid": scm_pid, "sid": sid, "account": observed})
        return payload
    except Exception as exc:
        raise Stage8PostgreSQLProbeError(label, PRINCIPAL_AUTH, str(exc)) from exc
    finally:
        if witness is not None:
            _close_process_exit_witness(witness)


def _extract_new_principal(log: Path, offset: int, label: str) -> str:
    text = log.read_text(encoding="utf-8", errors="replace")[offset:]
    found = AUTHENTICATED_RE.findall(text)
    if not found:
        raise Stage8PostgreSQLProbeError(
            label, PRINCIPAL_AUTH, "PostgreSQL did not log an SSPI authenticated identity"
        )
    return found[-1]


def _cleanup_service(name: str) -> list[str]:
    errors: list[str] = []
    witness: Any | None = None
    try:
        state, pid = _service_observation(name)
        if pid > 0:
            witness = _open_process_exit_witness(pid)
        if state != "STOPPED":
            stopped = _sc("stop", name, timeout=10)
            if stopped.returncode:
                errors.append(
                    f"service {name} stop failed: {(stopped.stderr or stopped.stdout)[-512:]}"
                )
                return errors
            else:
                _wait_service_state(name, "STOPPED")
        if witness is not None:
            _wait_process_exit(witness)
        deleted = _sc("delete", name, timeout=10)
        if deleted.returncode:
            errors.append(
                f"service {name} delete failed: {(deleted.stderr or deleted.stdout)[-512:]}"
            )
        else:
            _wait_service_absent(name)
    except Exception as exc:
        errors.append(f"service {name}: {exc}")
    finally:
        if witness is not None:
            _close_process_exit_witness(witness)
    return errors


def _finish_cleanup(primary: BaseException | None, cleanup_errors: list[str]) -> None:
    if not cleanup_errors:
        return
    detail = "; ".join(cleanup_errors)
    if primary is not None:
        print(f"[STAGE8_CLEANUP] secondary failure: {detail}", file=sys.stderr)
        return
    raise Stage8PostgreSQLProbeError("STAGE8_CLEANUP", SUBSTRATE, detail)


def _matrix_label(service: str, role: str, expected: bool) -> str:
    if expected and service == RUNTIME_SERVICE and role == RUNTIME_ROLE:
        return "WINDOWS_SSPI_RUNTIME_CONNECT"
    if expected and service == VERIFIER_SERVICE and role == VERIFIER_ROLE:
        return "WINDOWS_SSPI_VERIFIER_CONNECT"
    if (service, role) in {
        (RUNTIME_SERVICE, VERIFIER_ROLE),
        (VERIFIER_SERVICE, RUNTIME_ROLE),
    }:
        return "WINDOWS_SSPI_CROSS_ROLE_DENIAL"
    return "WINDOWS_SSPI_OUTSIDER_DENIAL"


def run_probe(scratch_parent: Path) -> dict[str, str]:
    if canonical_host_os() != "Windows" or os.name != "nt":
        raise Stage8PostgreSQLProbeError("POSTGRESQL_DISCOVERY", SUBSTRATE, "live Stage-8 requires a real Windows host")
    binaries = discover_postgresql()
    primary: BaseException | None = None
    cleanup_errors: list[str] = []
    cluster_started = False
    root: Path | None = None
    owned_services: set[str] = set()
    try:
        root = Path(tempfile.mkdtemp(prefix="CryptoHunter-Stage8-", dir=scratch_parent))
        data, log = root / "cluster", root / "postgresql.log"
        init = _run([str(binaries.executable("initdb.exe")), "-D", str(data), "-U", "stage8_bootstrap", "--auth-host=trust", "--auth-local=trust"])
        if init.returncode:
            raise Stage8PostgreSQLProbeError("POSTGRESQL_CLUSTER_INIT", SUBSTRATE, (init.stderr or init.stdout)[-STDERR_LIMIT:])
        port = _port()
        with (data / "postgresql.conf").open("a", encoding="utf-8") as stream:
            stream.write("\nlisten_addresses='127.0.0.1'\nfsync=on\nsynchronous_commit=on\nlog_connections=on\nlog_disconnections=on\nlogging_collector=off\n")
        started = _run([str(binaries.executable("pg_ctl.exe")), "-D", str(data), "-l", str(log), "-o", f"-p {port}", "-w", "start"], timeout=START_TIMEOUT)
        if started.returncode:
            raise Stage8PostgreSQLProbeError("POSTGRESQL_START", SUBSTRATE, (started.stderr or started.stdout)[-STDERR_LIMIT:])
        cluster_started = True
        import psycopg
        from bot_core.postgresql_freshness_authority import (
            PostgreSQLConnectionConfig,
            provision_postgresql_freshness_authority,
            qualify_postgresql_freshness_authority,
        )
        with psycopg.connect(_admin_dsn(port), autocommit=True) as connection:
            settings = connection.execute("SELECT current_setting('fsync'),current_setting('synchronous_commit'),current_setting('listen_addresses'),current_setting('port'),current_setting('server_version_num')").fetchone()
            if settings[:3] != ("on", "on", "127.0.0.1") or int(settings[4]) < 160000:
                raise Stage8PostgreSQLProbeError("POSTGRESQL_DURABILITY", SUBSTRATE, f"unsafe settings: {settings}")
            connection.execute(f'CREATE DATABASE "{DATABASE}"')
        authority = PostgreSQLConnectionConfig(_admin_dsn(port, DATABASE))
        try:
            provision_postgresql_freshness_authority(authority)
        except Exception as exc:
            raise Stage8PostgreSQLProbeError("POSTGRESQL_SCHEMA_PROVISION", SUBSTRATE, f"{type(exc).__name__}: {exc}") from exc
        try:
            qualify_postgresql_freshness_authority(authority)
        except Exception as exc:
            raise Stage8PostgreSQLProbeError("POSTGRESQL_SCHEMA_QUALIFICATION", SUBSTRATE, f"{type(exc).__name__}: {exc}") from exc

        # Temporary diagnostic-only mapping discovers what SSPI actually authenticates.
        (data / "pg_ident.conf").write_text(f"{MAP_NAME} /^(.*)$/ stage8_bootstrap\n", encoding="utf-8")
        (data / "pg_hba.conf").write_text(f"host {DATABASE} stage8_bootstrap 127.0.0.1/32 sspi map={MAP_NAME} include_realm=1\nhost all all 0.0.0.0/0 reject\nhost all all ::0/0 reject\n", encoding="utf-8")
        _run([str(binaries.executable("pg_ctl.exe")), "-D", str(data), "reload"])
        service_requests = _prepare_services(root, owned_services)
        principals: list[str] = []
        for service, account in ((RUNTIME_SERVICE, RUNTIME_ACCOUNT), (VERIFIER_SERVICE, VERIFIER_ACCOUNT)):
            offset = log.stat().st_size
            identity_label = _service_failure_label(service, identity=True)
            payload = _invoke_service(
                root,
                service_requests[service],
                service,
                account,
                port,
                "stage8_bootstrap",
                failure_label=identity_label,
            )
            if not payload.get("ok"):
                raise Stage8PostgreSQLProbeError("WINDOWS_SSPI_RUNTIME_IDENTITY" if service == RUNTIME_SERVICE else "WINDOWS_SSPI_VERIFIER_IDENTITY", PRINCIPAL_AUTH, f"SSPI diagnostic connection failed: {payload}")
            principals.append(_extract_new_principal(log, offset, identity_label))

        (data / "pg_ident.conf").write_text("\n".join(ident_lines(*principals)) + "\n", encoding="utf-8")
        (data / "pg_hba.conf").write_text("\n".join(final_hba_lines()) + "\n", encoding="utf-8")
        reload_result = _run([str(binaries.executable("pg_ctl.exe")), "-D", str(data), "reload"])
        if reload_result.returncode:
            raise Stage8PostgreSQLProbeError("WINDOWS_SSPI_HBA_QUALIFICATION", PRINCIPAL_AUTH, "final configuration reload failed")
        # Stop, inspect pg_hba_file_rules through a bounded single-user backend, then
        # start again with precisely the same final files. No administrative network
        # authentication rule is introduced for this read-only qualification.
        stopped = _run([str(binaries.executable("pg_ctl.exe")), "-D", str(data), "-m", "fast", "-w", "stop"], timeout=START_TIMEOUT)
        if stopped.returncode:
            raise Stage8PostgreSQLProbeError("WINDOWS_SSPI_HBA_QUALIFICATION", PRINCIPAL_AUTH, "could not stop for offline HBA qualification")
        cluster_started = False
        _qualify_effective_hba_offline(binaries.executable("postgres.exe"), data)
        restarted = _run([str(binaries.executable("pg_ctl.exe")), "-D", str(data), "-l", str(log), "-o", f"-p {port}", "-w", "start"], timeout=START_TIMEOUT)
        if restarted.returncode:
            raise Stage8PostgreSQLProbeError("POSTGRESQL_START", SUBSTRATE, "final configured cluster restart failed")
        cluster_started = True
        matrix = (
            (RUNTIME_SERVICE, RUNTIME_ACCOUNT, RUNTIME_ROLE, True, DATABASE),
            (VERIFIER_SERVICE, VERIFIER_ACCOUNT, VERIFIER_ROLE, True, DATABASE),
            (RUNTIME_SERVICE, RUNTIME_ACCOUNT, VERIFIER_ROLE, False, DATABASE),
            (VERIFIER_SERVICE, VERIFIER_ACCOUNT, RUNTIME_ROLE, False, DATABASE),
            (RUNTIME_SERVICE, RUNTIME_ACCOUNT, "stage8_bootstrap", False, DATABASE),
        )
        for service, account, role, expected, database in matrix:
            payload = _invoke_service(
                root,
                service_requests[service],
                service,
                account,
                port,
                role,
                database,
                failure_label=_matrix_label(service, role, expected),
            )
            if bool(payload.get("ok")) != expected or (expected and payload.get("session_user") != role):
                raise Stage8PostgreSQLProbeError(_matrix_label(service, role, expected), PRINCIPAL_AUTH, f"connection matrix mismatch: service={service}, requested_role={role}, result={payload}")
        for role in (RUNTIME_ROLE, VERIFIER_ROLE):
            try:
                psycopg.connect(host="127.0.0.1", port=port, dbname=DATABASE, user=role, connect_timeout=5)
            except psycopg.Error:
                pass
            else:
                raise Stage8PostgreSQLProbeError("WINDOWS_SSPI_OUTSIDER_DENIAL", PRINCIPAL_AUTH, f"interactive caller reached {role}")
        # Re-qualify schema state before final shutdown. The exact qualifier itself
        # already covers roles, memberships, owners, ACLs, functions and raw DML.
        # Final HBA is parsed by the server; use a temporary offline local socket is
        # impossible on Windows, therefore validate its exact text and reject syntax.
        if tuple((data / "pg_hba.conf").read_text(encoding="utf-8").splitlines()) != final_hba_lines():
            raise Stage8PostgreSQLProbeError("WINDOWS_SSPI_HBA_QUALIFICATION", PRINCIPAL_AUTH, "final HBA changed after reload")
        if tuple((data / "pg_ident.conf").read_text(encoding="utf-8").splitlines()) != ident_lines(*principals):
            raise Stage8PostgreSQLProbeError("WINDOWS_SSPI_IDENT_QUALIFICATION", PRINCIPAL_AUTH, "final ident changed after reload")
        return {SUBSTRATE: "PASS", PRINCIPAL_AUTH: "PASS", "details": f"PostgreSQL {binaries.major}; distinct SSPI principals; isolated port {port}"}
    except BaseException as exc:
        primary = exc
        raise
    finally:
        for name in tuple(owned_services):
            try:
                cleanup_errors.extend(_cleanup_service(name))
            except Exception as exc:
                cleanup_errors.append(f"service cleanup {name}: {exc}")
        if cluster_started and root is not None:
            try:
                stopped = _run([str(binaries.executable("pg_ctl.exe")), "-D", str(root / "cluster"), "-m", "fast", "-w", "stop"], timeout=START_TIMEOUT)
                if stopped.returncode:
                    cleanup_errors.append("PostgreSQL did not stop cleanly")
            except Exception as exc:
                cleanup_errors.append(f"PostgreSQL cleanup: {exc}")
        if root is not None:
            try:
                shutil.rmtree(root)
            except OSError as exc:
                cleanup_errors.append(f"scratch cleanup: {exc}")
        _finish_cleanup(primary, cleanup_errors)
