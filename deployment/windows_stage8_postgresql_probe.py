"""Live, fail-closed Windows PostgreSQL substrate and SSPI qualification.

This is an acceptance probe, not an installer.  It owns an isolated temporary
cluster and temporary SCM helper services and never opens or changes a system
PGDATA or a pre-installed PostgreSQL service.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import argparse
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
AUTHORITY_STATEMENT_TIMEOUT_MS = 60_000
AUTHORITY_LOCK_TIMEOUT_MS = 5_000
STDERR_LIMIT = 8192
# postmaster.pid records whole Unix seconds while Win32 FILETIME has sub-second
# precision; three seconds also covers the short spawn-to-pidfile-write interval.
POSTMASTER_START_TOLERANCE_SECONDS = 3.0
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

PHASES = (
    "POSTGRESQL_DISCOVERY",
    "POSTGRESQL_CLUSTER_INIT",
    "POSTGRESQL_START",
    "POSTGRESQL_DURABILITY",
    "POSTGRESQL_SCHEMA_PROVISION",
    "POSTGRESQL_SCHEMA_QUALIFICATION",
    "HELPER_SERVICE_PREPARE",
    "SSPI_RUNTIME_DISCOVERY",
    "SSPI_VERIFIER_DISCOVERY",
    "FINAL_IDENT_WRITE",
    "FINAL_HBA_WRITE",
    "FINAL_HBA_OFFLINE_QUALIFICATION",
    "FINAL_CLUSTER_RESTART",
    "MATRIX_RUNTIME",
    "MATRIX_VERIFIER",
    "MATRIX_RUNTIME_TO_VERIFIER_DENIAL",
    "MATRIX_VERIFIER_TO_RUNTIME_DENIAL",
    "MATRIX_WRONG_ROLE_DENIAL",
    "INTERACTIVE_OUTSIDER_DENIAL",
    "CLEANUP_SERVICES",
    "CLEANUP_POSTGRESQL",
    "CLEANUP_SCRATCH",
)

_status_file: Path | None = None
_status: dict[str, Any] = {}


def _write_status(**updates: Any) -> None:
    """Publish non-secret ownership and progress state for the supervising process."""
    _status.update(updates)
    if _status_file is not None:
        _atomic_json(_status_file, _status)


@contextmanager
def _phase(name: str):
    started = time.monotonic()
    _write_status(last_phase=name, phase_started=started)
    print(f"[STAGE8_PHASE] START {name}", flush=True)
    try:
        yield
    except BaseException:
        elapsed = time.monotonic() - started
        print(f"[STAGE8_PHASE] FAIL {name} elapsed={elapsed:.3f}s", flush=True)
        raise
    else:
        elapsed = time.monotonic() - started
        print(f"[STAGE8_PHASE] PASS {name} elapsed={elapsed:.3f}s", flush=True)


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


@dataclass(frozen=True)
class PgCtlResult:
    """Bounded result from the special no-pipe pg_ctl process boundary."""

    returncode: int | None
    elapsed: float
    timed_out: bool
    stdout_tail: str
    stderr_tail: str


@dataclass(frozen=True)
class PostmasterPidFile:
    pid: int
    data: Path
    start_time: float


@dataclass(frozen=True)
class PostmasterWitness:
    pid: int
    process: Any
    start_time: float
    executable: Path


@dataclass(frozen=True)
class ProcessIdentitySnapshot:
    pid: int
    process: Any
    creation_time: float
    executable: Path


def _tail(path: Path, limit: int = STDERR_LIMIT) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            return stream.read(limit).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _run_pg_ctl(
    pg_ctl: Path,
    data: Path,
    diagnostic_root: Path,
    operation: str,
    arguments: list[str],
    *,
    timeout: int = START_TIMEOUT,
) -> PgCtlResult:
    """Run pg_ctl with real files so a postgres child cannot retain a Python pipe."""
    token = f"{operation}-{time.time_ns()}"
    stdout_path = diagnostic_root / f"pg_ctl-{token}.stdout.log"
    stderr_path = diagnostic_root / f"pg_ctl-{token}.stderr.log"
    command = [str(pg_ctl), "-D", str(data), *arguments]
    started = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    timed_out = False
    returncode: int | None = None
    # Binary file handles are deliberately passed directly to Popen. In particular,
    # neither stream is PIPE and there is no communicate()/pipe-EOF dependency.
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    returncode = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    returncode = None
    return PgCtlResult(
        returncode=returncode,
        elapsed=time.monotonic() - started,
        timed_out=timed_out,
        stdout_tail=_tail(stdout_path),
        stderr_tail=_tail(stderr_path),
    )


def _run(command: list[str], *, timeout: int = COMMAND_TIMEOUT) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_DISCOVERY", SUBSTRATE, "bounded command timed out"
        ) from exc


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
            raise Stage8PostgreSQLProbeError(
                "POSTGRESQL_DISCOVERY", SUBSTRATE, f"PostgreSQL {major} is below required major 16"
            )
        return PostgreSQLBinaries(candidate.resolve(), major)
    raise Stage8PostgreSQLProbeError(
        "POSTGRESQL_DISCOVERY",
        SUBSTRATE,
        "required PostgreSQL binaries were not found via PGBIN or pg_config",
    )


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
    if (
        not runtime_principal
        or not verifier_principal
        or runtime_principal.casefold() == verifier_principal.casefold()
    ):
        raise Stage8PostgreSQLProbeError(
            "WINDOWS_SSPI_IDENT_QUALIFICATION",
            PRINCIPAL_AUTH,
            "virtual service accounts did not yield two distinct SSPI principals",
        )
    if any(
        re.search(r"[\s/]|\.\*|\^|\$", value) for value in (runtime_principal, verifier_principal)
    ):
        raise Stage8PostgreSQLProbeError(
            "WINDOWS_SSPI_IDENT_QUALIFICATION",
            PRINCIPAL_AUTH,
            "unsafe principal syntax cannot be represented by an exact pg_ident map",
        )
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
        raise Stage8PostgreSQLProbeError(
            "WINDOWS_SSPI_HBA_QUALIFICATION", PRINCIPAL_AUTH, "unexpected effective HBA row count"
        )
    previous = -1
    for row, wanted in zip(rows, expected, strict=True):
        line, kind, databases, users, address, netmask, method, options, error = row
        if error is not None or line <= previous or kind != "host":
            raise Stage8PostgreSQLProbeError(
                "WINDOWS_SSPI_HBA_QUALIFICATION",
                PRINCIPAL_AUTH,
                "invalid HBA parse result or ordering",
            )
        previous = line
        database, user, auth = wanted
        if databases != [database] or users != [user] or method != auth:
            raise Stage8PostgreSQLProbeError(
                "WINDOWS_SSPI_HBA_QUALIFICATION",
                PRINCIPAL_AUTH,
                "effective HBA does not match the exact reviewed sequence",
            )
        if auth == "sspi" and (
            address != "127.0.0.1"
            or netmask != "255.255.255.255"
            or sorted(options or []) != ["include_realm=1", f"map={MAP_NAME}"]
        ):
            raise Stage8PostgreSQLProbeError(
                "WINDOWS_SSPI_HBA_QUALIFICATION",
                PRINCIPAL_AUTH,
                "online HBA is not exact loopback SSPI",
            )
        if user in {RUNTIME_ROLE, VERIFIER_ROLE} and method in FORBIDDEN_ONLINE_METHODS:
            raise Stage8PostgreSQLProbeError(
                "WINDOWS_SSPI_HBA_QUALIFICATION",
                PRINCIPAL_AUTH,
                "forbidden online authentication method",
            )


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
        raise Stage8PostgreSQLProbeError(
            "WINDOWS_SSPI_HBA_QUALIFICATION", PRINCIPAL_AUTH, "offline HBA parser timed out"
        ) from exc
    rows: list[tuple[Any, ...]] = []
    for line in completed.stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and len(value) == 9:
            rows.append(tuple(value))
    if completed.returncode or not rows:
        raise Stage8PostgreSQLProbeError(
            "WINDOWS_SSPI_HBA_QUALIFICATION",
            PRINCIPAL_AUTH,
            (completed.stderr or completed.stdout)[-STDERR_LIMIT:],
        )
    qualify_hba_rows(rows)


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _postgresql_pid(data: Path) -> int | None:
    try:
        return int((data / "postmaster.pid").read_text(encoding="utf-8").splitlines()[0])
    except (OSError, ValueError, IndexError):
        return None


def _postmaster_pidfile(data: Path) -> PostmasterPidFile | None:
    try:
        lines = (data / "postmaster.pid").read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_IDENTITY", SUBSTRATE, f"could not read owned postmaster.pid: {exc}"
        ) from exc
    if len(lines) < 3:
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_IDENTITY", SUBSTRATE, "owned postmaster.pid is truncated"
        )
    try:
        pid = int(lines[0])
        start_time = float(lines[2])
    except ValueError as exc:
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_IDENTITY", SUBSTRATE, "owned postmaster.pid identity is malformed"
        ) from exc
    if pid <= 0 or not lines[1].strip() or start_time <= 0:
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_IDENTITY", SUBSTRATE, "owned postmaster.pid identity is invalid"
        )
    pid_data = Path(lines[1])
    try:
        matches = os.path.normcase(str(pid_data.resolve())) == os.path.normcase(str(data.resolve()))
    except OSError as exc:
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_IDENTITY", SUBSTRATE, f"could not canonicalize owned PGDATA: {exc}"
        ) from exc
    if not matches:
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_IDENTITY",
            SUBSTRATE,
            f"postmaster.pid data directory does not match exact owned PGDATA: {pid_data}",
        )
    return PostmasterPidFile(pid, pid_data, start_time)


def _filetime_seconds(value: Any) -> float:
    timestamp = getattr(value, "timestamp", None)
    if callable(timestamp):
        return float(timestamp())
    return float(value)


def _query_process_image_path(process_handle: Any) -> Path:
    """Read a process image through minimal-rights QueryFullProcessImageNameW."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    query = kernel32.QueryFullProcessImageNameW
    query.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    )
    query.restype = wintypes.BOOL
    # The Win32 extended path limit is 32,767 UTF-16 code units.  Supplying
    # that bound avoids truncating a reviewed executable identity.
    capacity = 32_768
    buffer = ctypes.create_unicode_buffer(capacity)
    length = wintypes.DWORD(capacity)
    try:
        native_handle = wintypes.HANDLE(int(process_handle))
    except (TypeError, ValueError, OverflowError) as exc:
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_IDENTITY", SUBSTRATE, "could not convert the Win32 process handle"
        ) from exc
    if not query(native_handle, 0, buffer, ctypes.byref(length)):
        error = ctypes.get_last_error()
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_IDENTITY",
            SUBSTRATE,
            f"QueryFullProcessImageNameW failed: winerror={error}",
        )
    if length.value <= 0 or length.value >= capacity:
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_IDENTITY", SUBSTRATE, "invalid process image path length"
        )
    return Path(buffer.value)


def _process_identity_snapshot(pid: int) -> ProcessIdentitySnapshot | None:
    """Open and describe one live Win32 process object; caller owns the handle."""
    import pywintypes
    import win32api
    import win32con
    import win32event
    import win32process

    access = win32con.SYNCHRONIZE | win32con.PROCESS_QUERY_LIMITED_INFORMATION
    try:
        process = win32api.OpenProcess(access, False, pid)
    except pywintypes.error as exc:
        if exc.winerror == 87:
            return None
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_IDENTITY",
            SUBSTRATE,
            f"process identity query failed for pid={pid}: winerror={exc.winerror}",
        ) from exc
    try:
        state = win32event.WaitForSingleObject(process, 0)
        if state == win32event.WAIT_OBJECT_0:
            process.Close()
            return None
        if state != win32event.WAIT_TIMEOUT:
            raise RuntimeError(f"unexpected wait result {state}")
        image = _query_process_image_path(process)
        creation = _filetime_seconds(win32process.GetProcessTimes(process)["CreationTime"])
        return ProcessIdentitySnapshot(pid, process, creation, image)
    except Exception as exc:
        process.Close()
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_IDENTITY",
            SUBSTRATE,
            f"process identity query was ambiguous for pid={pid}: {exc}",
        ) from exc


def _postmaster_witness(data: Path, postgres: Path) -> PostmasterWitness | None:
    """Bind pidfile identity to one live Win32 process object without mutation."""
    identity = _postmaster_pidfile(data)
    if identity is None:
        return None
    snapshot = _process_identity_snapshot(identity.pid)
    if snapshot is None:
        return None
    try:
        image_matches = os.path.normcase(str(snapshot.executable.resolve())) == os.path.normcase(
            str(postgres.resolve())
        )
        time_matches = (
            abs(snapshot.creation_time - identity.start_time) <= POSTMASTER_START_TOLERANCE_SECONDS
        )
        if not image_matches or not time_matches:
            snapshot.process.Close()
            mismatch = "executable" if not image_matches else "creation time"
            raise Stage8PostgreSQLProbeError(
                "POSTGRESQL_PROCESS_IDENTITY",
                SUBSTRATE,
                f"live PID does not match owned postmaster {mismatch}: pid={identity.pid}",
            )
        return PostmasterWitness(
            identity.pid, snapshot.process, identity.start_time, snapshot.executable
        )
    except Stage8PostgreSQLProbeError:
        raise
    except Exception as exc:
        snapshot.process.Close()
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_IDENTITY",
            SUBSTRATE,
            f"postmaster identity query was ambiguous for pid={identity.pid}: {exc}",
        ) from exc


def _close_postmaster_witness(witness: PostmasterWitness) -> None:
    witness.process.Close()


def _require_exact_owned_data(data: Path, root: Path) -> None:
    try:
        matches = data.resolve() == (root / "cluster").resolve()
    except OSError as exc:
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_IDENTITY",
            SUBSTRATE,
            f"could not canonicalize Stage-8 PGDATA: {exc}",
        ) from exc
    if not matches:
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_IDENTITY",
            SUBSTRATE,
            "PGDATA is not the exact Stage-8-owned cluster",
        )


def _process_exists(pid: int | None) -> bool:
    """Query process liveness on Windows without signalling or mutating it."""
    if pid is None or pid <= 0:
        return False
    import pywintypes
    import win32api
    import win32con
    import win32event

    try:
        process = win32api.OpenProcess(win32con.SYNCHRONIZE, False, pid)
    except pywintypes.error as exc:
        # ERROR_INVALID_PARAMETER is returned for a PID with no process object.
        if exc.winerror == 87:
            return False
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_STATE",
            SUBSTRATE,
            f"non-destructive process-state query failed for pid={pid}: winerror={exc.winerror}",
        ) from exc
    try:
        state = win32event.WaitForSingleObject(process, 0)
        if state == win32event.WAIT_TIMEOUT:
            return True
        if state == win32event.WAIT_OBJECT_0:
            return False
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_STATE",
            SUBSTRATE,
            f"unexpected process-state result for pid={pid}: wait_result={state}",
        )
    finally:
        process.Close()


def _cluster_running(pg_ctl: Path, data: Path, root: Path) -> tuple[bool, int | None]:
    """Require exact pidfile, process identity and exact-PGDATA pg_ctl status."""
    _require_exact_owned_data(data, root)
    witness = _postmaster_witness(data, pg_ctl.with_name("postgres.exe"))
    if witness is None:
        return False, _postgresql_pid(data)
    try:
        status = _run_pg_ctl(pg_ctl, data, root, "status", ["status"], timeout=10)
        return not status.timed_out and status.returncode == 0, witness.pid
    finally:
        _close_postmaster_witness(witness)


def _pg_ctl_failure_detail(
    operation: str, result: PgCtlResult, data: Path, running: bool, pid: int | None, log: Path
) -> str:
    return (
        f"pg_ctl operation={operation} returncode={result.returncode} "
        f"timed_out={result.timed_out} elapsed={result.elapsed:.3f}s cluster={data} "
        f"postmaster_pid={pid} cluster_running={running}; "
        f"stdout_tail={result.stdout_tail[-STDERR_LIMIT:]!r}; "
        f"stderr_tail={result.stderr_tail[-STDERR_LIMIT:]!r}; "
        f"postgresql_log_tail={_tail(log)!r}"
    )


def _stop_owned_cluster(
    pg_ctl: Path,
    data: Path,
    root: Path,
    log: Path,
    *,
    mode: str = "fast",
    timeout: int = START_TIMEOUT,
) -> str | None:
    _require_exact_owned_data(data, root)
    witness = _postmaster_witness(data, pg_ctl.with_name("postgres.exe"))
    if witness is None:
        return None
    pid = witness.pid
    witness_error: str | None = None
    try:
        result = _run_pg_ctl(
            pg_ctl, data, root, "stop", ["-m", mode, "-w", "stop"], timeout=timeout
        )
        status = _run_pg_ctl(pg_ctl, data, root, "status", ["status"], timeout=10)
        running = not status.timed_out and status.returncode == 0
        try:
            _wait_process_exit(witness.process, timeout=timeout)
        except Exception as exc:
            witness_error = str(exc)
        status_stopped = not status.timed_out and status.returncode == 3
        if (
            result.timed_out
            or result.returncode != 0
            or not status_stopped
            or witness_error is not None
        ):
            detail = _pg_ctl_failure_detail("stop", result, data, running, pid, log)
            detail += (
                f"; status_returncode={status.returncode}; status_timed_out={status.timed_out}"
            )
            return detail if witness_error is None else f"{detail}; process_exit={witness_error}"
        return None
    finally:
        _close_postmaster_witness(witness)


def _stop_if_owned_cluster_running(
    pg_ctl: Path,
    data: Path,
    root: Path,
    log: Path,
    *,
    cluster_started: bool,
    mode: str = "fast",
    timeout: int = START_TIMEOUT,
) -> str | None:
    """Stop based on live cluster state, not solely on the possibly stale marker."""
    # The stale cache marker is deliberately not authority to signal anything.
    # _stop_owned_cluster obtains one exact identity witness before invoking pg_ctl.
    return _stop_owned_cluster(pg_ctl, data, root, log, mode=mode, timeout=timeout)


def _admin_dsn(port: int, database: str = "postgres") -> str:
    options = (
        f"-c statement_timeout={AUTHORITY_STATEMENT_TIMEOUT_MS} "
        f"-c lock_timeout={AUTHORITY_LOCK_TIMEOUT_MS}"
    )
    return (
        f"host=127.0.0.1 port={port} dbname={database} user=stage8_bootstrap "
        f"connect_timeout=5 options='{options}'"
    )


def _qualify_session_timeout_settings(rows: list[tuple[str, str, str]]) -> None:
    timeout_settings = {name: (int(setting), unit) for name, setting, unit in rows}
    expected = {
        "statement_timeout": (AUTHORITY_STATEMENT_TIMEOUT_MS, "ms"),
        "lock_timeout": (AUTHORITY_LOCK_TIMEOUT_MS, "ms"),
    }
    if timeout_settings != expected:
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_DURABILITY",
            SUBSTRATE,
            f"unsafe acceptance SQL timeout settings: {timeout_settings}",
        )


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
    if (
        sid != win32security.ConvertSidToStringSid(expected_sid)
        or account.casefold() != expected.casefold()
    ):
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


def _prepare_services(root: Path, owned_services: set[str]) -> dict[str, Path]:
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
        _write_status(owned_services=sorted(owned_services))
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
        print(f"[STAGE8_CLEANUP] secondary failure: {detail}", file=sys.stderr, flush=True)
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


def run_probe(scratch_parent: Path, *, status_file: Path | None = None) -> dict[str, str]:
    global _status_file, _status
    _status_file = status_file
    _status = {
        "last_phase": None,
        "cluster_started": False,
        "owned_services": [],
        "root": None,
    }
    if canonical_host_os() != "Windows" or os.name != "nt":
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_DISCOVERY", SUBSTRATE, "live Stage-8 requires a real Windows host"
        )
    with _phase("POSTGRESQL_DISCOVERY"):
        binaries = discover_postgresql()
        _write_status(pgbin=str(binaries.bindir))
    primary: BaseException | None = None
    cleanup_errors: list[str] = []
    cluster_started = False
    root: Path | None = None
    owned_services: set[str] = set()
    try:
        root = Path(tempfile.mkdtemp(prefix="CryptoHunter-Stage8-", dir=scratch_parent))
        _write_status(root=str(root))
        data, log = root / "cluster", root / "postgresql.log"
        with _phase("POSTGRESQL_CLUSTER_INIT"):
            init = _run(
                [
                    str(binaries.executable("initdb.exe")),
                    "-D",
                    str(data),
                    "-U",
                    "stage8_bootstrap",
                    "--auth-host=trust",
                    "--auth-local=trust",
                ]
            )
            if init.returncode:
                raise Stage8PostgreSQLProbeError(
                    "POSTGRESQL_CLUSTER_INIT",
                    SUBSTRATE,
                    (init.stderr or init.stdout)[-STDERR_LIMIT:],
                )
        port = _port()
        with (data / "postgresql.conf").open("a", encoding="utf-8") as stream:
            stream.write(
                "\nlisten_addresses='127.0.0.1'\nfsync=on\nsynchronous_commit=on\nlog_connections=on\nlog_disconnections=on\nlogging_collector=off\n"
            )
        with _phase("POSTGRESQL_START"):
            started = _run_pg_ctl(
                binaries.executable("pg_ctl.exe"),
                data,
                root,
                "start",
                [
                    "-l",
                    str(log),
                    "-o",
                    f"-p {port}",
                    "-w",
                    "start",
                ],
                timeout=START_TIMEOUT,
            )
            running, pid = _cluster_running(binaries.executable("pg_ctl.exe"), data, root)
            if started.timed_out or started.returncode != 0 or not running or pid is None:
                raise Stage8PostgreSQLProbeError(
                    "POSTGRESQL_START",
                    SUBSTRATE,
                    _pg_ctl_failure_detail("start", started, data, running, pid, log),
                )
            cluster_started = True
            _write_status(cluster_started=True, postgresql_pid=pid)
        import psycopg
        from bot_core.postgresql_freshness_authority import (
            PostgreSQLConnectionConfig,
            provision_postgresql_freshness_authority,
            qualify_postgresql_freshness_authority,
        )

        with _phase("POSTGRESQL_DURABILITY"):
            with psycopg.connect(_admin_dsn(port), autocommit=True) as connection:
                settings = connection.execute(
                    "SELECT current_setting('fsync'),current_setting('synchronous_commit'),current_setting('listen_addresses'),current_setting('port'),current_setting('server_version_num')"
                ).fetchone()
                if settings[:3] != ("on", "on", "127.0.0.1") or int(settings[4]) < 160000:
                    raise Stage8PostgreSQLProbeError(
                        "POSTGRESQL_DURABILITY", SUBSTRATE, f"unsafe settings: {settings}"
                    )
                _qualify_session_timeout_settings(
                    connection.execute(
                        "SELECT name, setting, unit FROM pg_catalog.pg_settings "
                        "WHERE name IN ('statement_timeout', 'lock_timeout')"
                    ).fetchall()
                )
                connection.execute(f'CREATE DATABASE "{DATABASE}"')
        authority = PostgreSQLConnectionConfig(_admin_dsn(port, DATABASE))
        with _phase("POSTGRESQL_SCHEMA_PROVISION"):
            try:
                provision_postgresql_freshness_authority(authority)
            except Exception as exc:
                raise Stage8PostgreSQLProbeError(
                    "POSTGRESQL_SCHEMA_PROVISION", SUBSTRATE, f"{type(exc).__name__}: {exc}"
                ) from exc
        with _phase("POSTGRESQL_SCHEMA_QUALIFICATION"):
            try:
                qualify_postgresql_freshness_authority(authority)
            except Exception as exc:
                raise Stage8PostgreSQLProbeError(
                    "POSTGRESQL_SCHEMA_QUALIFICATION", SUBSTRATE, f"{type(exc).__name__}: {exc}"
                ) from exc

        # Temporary diagnostic-only mapping discovers what SSPI actually authenticates.
        (data / "pg_ident.conf").write_text(
            f"{MAP_NAME} /^(.*)$/ stage8_bootstrap\n", encoding="utf-8"
        )
        (data / "pg_hba.conf").write_text(
            f"host {DATABASE} stage8_bootstrap 127.0.0.1/32 sspi map={MAP_NAME} include_realm=1\nhost all all 0.0.0.0/0 reject\nhost all all ::0/0 reject\n",
            encoding="utf-8",
        )
        _run_pg_ctl(binaries.executable("pg_ctl.exe"), data, root, "reload", ["reload"])
        with _phase("HELPER_SERVICE_PREPARE"):
            service_requests = _prepare_services(root, owned_services)
            _write_status(owned_services=sorted(owned_services))
        principals: list[str] = []
        for discovery_phase, service, account in (
            ("SSPI_RUNTIME_DISCOVERY", RUNTIME_SERVICE, RUNTIME_ACCOUNT),
            ("SSPI_VERIFIER_DISCOVERY", VERIFIER_SERVICE, VERIFIER_ACCOUNT),
        ):
            with _phase(discovery_phase):
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
                    raise Stage8PostgreSQLProbeError(
                        "WINDOWS_SSPI_RUNTIME_IDENTITY"
                        if service == RUNTIME_SERVICE
                        else "WINDOWS_SSPI_VERIFIER_IDENTITY",
                        PRINCIPAL_AUTH,
                        f"SSPI diagnostic connection failed: {payload}",
                    )
                principals.append(_extract_new_principal(log, offset, identity_label))

        with _phase("FINAL_IDENT_WRITE"):
            (data / "pg_ident.conf").write_text(
                "\n".join(ident_lines(*principals)) + "\n", encoding="utf-8"
            )
        with _phase("FINAL_HBA_WRITE"):
            (data / "pg_hba.conf").write_text("\n".join(final_hba_lines()) + "\n", encoding="utf-8")
        reload_result = _run_pg_ctl(
            binaries.executable("pg_ctl.exe"), data, root, "reload", ["reload"]
        )
        if reload_result.timed_out or reload_result.returncode != 0:
            raise Stage8PostgreSQLProbeError(
                "WINDOWS_SSPI_HBA_QUALIFICATION",
                PRINCIPAL_AUTH,
                "final configuration reload failed",
            )
        # Stop, inspect pg_hba_file_rules through a bounded single-user backend, then
        # start again with precisely the same final files. No administrative network
        # authentication rule is introduced for this read-only qualification.
        with _phase("FINAL_HBA_OFFLINE_QUALIFICATION"):
            stop_error = _stop_owned_cluster(binaries.executable("pg_ctl.exe"), data, root, log)
            if stop_error:
                raise Stage8PostgreSQLProbeError(
                    "WINDOWS_SSPI_HBA_QUALIFICATION",
                    PRINCIPAL_AUTH,
                    f"could not stop for offline HBA qualification: {stop_error}",
                )
            cluster_started = False
            _write_status(cluster_started=False, postgresql_pid=None)
            _qualify_effective_hba_offline(binaries.executable("postgres.exe"), data)
        with _phase("FINAL_CLUSTER_RESTART"):
            restarted = _run_pg_ctl(
                binaries.executable("pg_ctl.exe"),
                data,
                root,
                "restart",
                [
                    "-l",
                    str(log),
                    "-o",
                    f"-p {port}",
                    "-w",
                    "start",
                ],
                timeout=START_TIMEOUT,
            )
            running, pid = _cluster_running(binaries.executable("pg_ctl.exe"), data, root)
            if restarted.timed_out or restarted.returncode != 0 or not running or pid is None:
                raise Stage8PostgreSQLProbeError(
                    "POSTGRESQL_START",
                    SUBSTRATE,
                    _pg_ctl_failure_detail("restart", restarted, data, running, pid, log),
                )
            cluster_started = True
            _write_status(cluster_started=True, postgresql_pid=pid)
        matrix = (
            (RUNTIME_SERVICE, RUNTIME_ACCOUNT, RUNTIME_ROLE, True, DATABASE),
            (VERIFIER_SERVICE, VERIFIER_ACCOUNT, VERIFIER_ROLE, True, DATABASE),
            (RUNTIME_SERVICE, RUNTIME_ACCOUNT, VERIFIER_ROLE, False, DATABASE),
            (VERIFIER_SERVICE, VERIFIER_ACCOUNT, RUNTIME_ROLE, False, DATABASE),
            (RUNTIME_SERVICE, RUNTIME_ACCOUNT, "stage8_bootstrap", False, DATABASE),
        )
        matrix_phases = (
            "MATRIX_RUNTIME",
            "MATRIX_VERIFIER",
            "MATRIX_RUNTIME_TO_VERIFIER_DENIAL",
            "MATRIX_VERIFIER_TO_RUNTIME_DENIAL",
            "MATRIX_WRONG_ROLE_DENIAL",
        )
        for matrix_phase, (service, account, role, expected, database) in zip(
            matrix_phases, matrix, strict=True
        ):
            with _phase(matrix_phase):
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
                if bool(payload.get("ok")) != expected or (
                    expected and payload.get("session_user") != role
                ):
                    raise Stage8PostgreSQLProbeError(
                        _matrix_label(service, role, expected),
                        PRINCIPAL_AUTH,
                        f"connection matrix mismatch: service={service}, requested_role={role}, result={payload}",
                    )
        with _phase("INTERACTIVE_OUTSIDER_DENIAL"):
            for role in (RUNTIME_ROLE, VERIFIER_ROLE):
                try:
                    psycopg.connect(
                        host="127.0.0.1", port=port, dbname=DATABASE, user=role, connect_timeout=5
                    )
                except psycopg.Error:
                    pass
                else:
                    raise Stage8PostgreSQLProbeError(
                        "WINDOWS_SSPI_OUTSIDER_DENIAL",
                        PRINCIPAL_AUTH,
                        f"interactive caller reached {role}",
                    )
        # Re-qualify schema state before final shutdown. The exact qualifier itself
        # already covers roles, memberships, owners, ACLs, functions and raw DML.
        # Final HBA is parsed by the server; use a temporary offline local socket is
        # impossible on Windows, therefore validate its exact text and reject syntax.
        if (
            tuple((data / "pg_hba.conf").read_text(encoding="utf-8").splitlines())
            != final_hba_lines()
        ):
            raise Stage8PostgreSQLProbeError(
                "WINDOWS_SSPI_HBA_QUALIFICATION", PRINCIPAL_AUTH, "final HBA changed after reload"
            )
        if tuple((data / "pg_ident.conf").read_text(encoding="utf-8").splitlines()) != ident_lines(
            *principals
        ):
            raise Stage8PostgreSQLProbeError(
                "WINDOWS_SSPI_IDENT_QUALIFICATION",
                PRINCIPAL_AUTH,
                "final ident changed after reload",
            )
        return {
            SUBSTRATE: "PASS",
            PRINCIPAL_AUTH: "PASS",
            "details": f"PostgreSQL {binaries.major}; distinct SSPI principals; isolated port {port}",
        }
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            with _phase("CLEANUP_SERVICES"):
                for name in tuple(owned_services):
                    try:
                        cleanup_errors.extend(_cleanup_service(name))
                    except Exception as exc:
                        cleanup_errors.append(f"service cleanup {name}: {exc}")
                if cleanup_errors:
                    raise RuntimeError("; ".join(cleanup_errors))
        except RuntimeError:
            pass
        postgresql_cleanup_ok = True
        if root is not None:
            try:
                with _phase("CLEANUP_POSTGRESQL"):
                    data, log = root / "cluster", root / "postgresql.log"
                    stop_error = _stop_if_owned_cluster_running(
                        binaries.executable("pg_ctl.exe"),
                        data,
                        root,
                        log,
                        cluster_started=cluster_started,
                    )
                    if stop_error:
                        raise RuntimeError(stop_error)
                    _write_status(cluster_started=False, postgresql_pid=None)
            except Exception as exc:
                postgresql_cleanup_ok = False
                cleanup_errors.append(f"PostgreSQL cleanup: {exc}")
        else:
            with _phase("CLEANUP_POSTGRESQL"):
                pass
        if root is not None and postgresql_cleanup_ok:
            try:
                with _phase("CLEANUP_SCRATCH"):
                    shutil.rmtree(root)
                    _write_status(root=None)
            except OSError as exc:
                cleanup_errors.append(f"scratch cleanup: {exc}")
        else:
            with _phase("CLEANUP_SCRATCH"):
                pass
        _finish_cleanup(primary, cleanup_errors)


def emergency_cleanup(status: dict[str, Any], scratch_parent: Path) -> list[str]:
    """Bounded cleanup after the supervisor terminates a stuck Stage-8 worker."""
    errors: list[str] = []
    owned = status.get("owned_services", [])
    if isinstance(owned, list):
        for name in owned:
            if name not in {RUNTIME_SERVICE, VERIFIER_SERVICE}:
                errors.append(f"refused unexpected service ownership marker: {name}")
                continue
            errors.extend(_cleanup_service(name))
    root_value = status.get("root")
    root = Path(root_value) if isinstance(root_value, str) else None
    try:
        safe_root = (
            root is not None
            and root.name.startswith("CryptoHunter-Stage8-")
            and root.resolve().parent == scratch_parent.resolve()
        )
    except OSError:
        safe_root = False
    if root is not None and not safe_root:
        errors.append("refused scratch cleanup outside Stage-8-owned root")
        return errors
    postgresql_cleanup_ok = True
    if safe_root:
        pgbin = status.get("pgbin")
        if isinstance(pgbin, str):
            try:
                pg_ctl = Path(pgbin) / "pg_ctl.exe"
                data, log = root / "cluster", root / "postgresql.log"
                stop_error = _stop_if_owned_cluster_running(
                    pg_ctl,
                    data,
                    root,
                    log,
                    cluster_started=status.get("cluster_started") is True,
                    mode="immediate",
                    timeout=10,
                )
                if stop_error:
                    postgresql_cleanup_ok = False
                    errors.append(f"emergency PostgreSQL stop failed: {stop_error}")
            except Exception as exc:
                postgresql_cleanup_ok = False
                errors.append(f"emergency PostgreSQL stop: {exc}")
        elif status.get("cluster_started") is True or _postgresql_pid(root / "cluster") is not None:
            postgresql_cleanup_ok = False
            errors.append("emergency PostgreSQL stop lacks reviewed binary path")
    if safe_root and postgresql_cleanup_ok:
        try:
            shutil.rmtree(root)
        except OSError as exc:
            errors.append(f"emergency scratch cleanup: {exc}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Internal reviewed Windows Stage-8 worker")
    parser.add_argument("--scratch-parent", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_probe(args.scratch_parent, status_file=args.status_file)
    except Stage8PostgreSQLProbeError as exc:
        _write_status(failed_item=exc.item, error=str(exc))
        print(json.dumps({"item": exc.item, "error": str(exc)}), file=sys.stderr, flush=True)
        return 1
    _write_status(
        completed=True,
        results={item: result[item] for item in WINDOWS_STAGE8_ITEMS},
    )
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
