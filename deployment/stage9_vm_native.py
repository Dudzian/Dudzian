"""Rzeczywiste obserwacje Stage 9 wywoływane dopiero po bramce jednorazowej VM.

Ten moduł nie przygotowuje kont, klastra, polityki ani autorytetów. Wyłącznie
jawne ``observe_systemd(start=True)`` przeładowuje managera i startuje właściwą
usługę; nie tworzy zastępczego procesu. Częściowa diagnostyka IPC zawsze
zachowuje blokadę pełnej kwalifikacji produktu.
"""

from __future__ import annotations

import os
import secrets
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

UNIT = "cryptohunter-freshness-verifier.service"
UNIT_PATH = Path("/etc/systemd/system") / UNIT
SOCKET_PATH = "/run/cryptohunter/freshness/verifier.sock"
SAFE_ENV = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C", "LC_ALL": "C"}
_STATE_FIELDS = (
    "LoadState",
    "ActiveState",
    "SubState",
    "Result",
    "ExecMainCode",
    "ExecMainStatus",
    "MainPID",
    "ExecMainPID",
    "FragmentPath",
    "DropInPaths",
)
_STATE_VALUES = frozenset(
    {
        "loaded",
        "not-found",
        "masked",
        "error",
        "bad-setting",
        "active",
        "inactive",
        "failed",
        "activating",
        "deactivating",
        "running",
        "dead",
        "exited",
        "start-pre",
        "start",
        "auto-restart",
        "stop",
        "stop-sigterm",
        "final-sigterm",
        "success",
        "resources",
        "timeout",
        "exit-code",
        "signal",
        "core-dump",
        "watchdog",
        "start-limit-hit",
        "protocol",
        "dependency",
        "oom-kill",
        "condition",
        "assert",
    }
)


class NativeObservationError(ValueError):
    """Stała przyczyna bez stdout, stderr, DSN lub treści wyjątku produktu."""

    def __init__(self, reason: str, *, status: str = "FAIL") -> None:
        super().__init__(reason)
        self.reason = reason
        self.status = status


def _require(condition: bool, reason: str, *, status: str = "FAIL") -> None:
    if not condition:
        raise NativeObservationError(reason, status=status)


def _command(arguments: list[str]) -> str:
    _require(
        bool(arguments) and arguments[0] in {"/usr/bin/systemctl", "/usr/bin/systemd-analyze"},
        "NATIVE_COMMAND_NOT_APPROVED",
    )
    try:
        result = subprocess.run(
            arguments,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
            cwd="/",
            env=SAFE_ENV,
            text=True,
            encoding="utf-8",
            errors="strict",
            check=False,
        )
    except (OSError, UnicodeError, subprocess.SubprocessError) as exc:
        raise NativeObservationError("NATIVE_COMMAND_UNAVAILABLE", status="BLOCKED") from exc
    _require(result.returncode == 0, "NATIVE_COMMAND_REFUSED", status="BLOCKED")
    _require(len(result.stdout) <= 65536, "NATIVE_COMMAND_REPLY_TOO_LARGE")
    return result.stdout


def _state() -> dict[str, str]:
    """Wąska diagnostyka stanu; nie archiwizujemy Environment ani journald."""
    raw = _command(
        ["/usr/bin/systemctl", "show", UNIT, "--no-pager", "--property=" + ",".join(_STATE_FIELDS)]
    )
    result: dict[str, str] = {}
    for line in raw.splitlines():
        key, separator, value = line.partition("=")
        _require(
            bool(separator) and key in _STATE_FIELDS and key not in result, "UNIT_STATE_INVALID"
        )
        if key == "FragmentPath":
            _require(value in {"", str(UNIT_PATH)}, "UNIT_FRAGMENT_CHANGED")
        elif key == "DropInPaths":
            _require(value == "", "UNIT_DROPIN_NOT_APPROVED")
        elif key in {"ExecMainCode", "ExecMainStatus", "MainPID", "ExecMainPID"}:
            _require(
                value.isascii() and value.isdecimal() and len(value) <= 20, "UNIT_STATE_INVALID"
            )
        else:
            _require(value in _STATE_VALUES, "UNIT_STATE_INVALID")
        result[key] = value
    _require(set(result) == set(_STATE_FIELDS), "UNIT_STATE_INCOMPLETE")
    return result


def _protected_unit() -> Any:
    """Weryfikacja podpisanego runtime i całej jednostki przed daemon-reload."""
    from deployment.freshness_runtime import validate_runtime_unit, verify_installed_runtime
    from deployment.production_local_installer_host import UNIT_QUERY, _query

    runtime = verify_installed_runtime()
    for path in (*reversed(UNIT_PATH.parents), UNIT_PATH):
        info = path.lstat()
        _require(
            not stat.S_ISLNK(info.st_mode)
            and info.st_uid == info.st_gid == 0
            and not info.st_mode & 0o022,
            "UNIT_PATH_NOT_PROTECTED",
        )
        if path == UNIT_PATH:
            _require(
                stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size <= 65536,
                "UNIT_FILE_INVALID",
            )
        else:
            _require(stat.S_ISDIR(info.st_mode), "UNIT_PATH_NOT_PROTECTED")
    content = UNIT_PATH.read_bytes()
    validate_runtime_unit(content, str(runtime.executable))
    for directory in (
        "/etc/systemd/system",
        "/run/systemd/system",
        "/usr/lib/systemd/system",
        "/usr/local/lib/systemd/system",
    ):
        for suffix in (
            UNIT + ".d",
            "cryptohunter-freshness-.service.d",
            "cryptohunter-.service.d",
            "service.d",
        ):
            _require(not os.path.lexists(Path(directory) / suffix), "UNIT_DROPIN_NOT_APPROVED")
    raw_origin = _query(UNIT_QUERY)
    if raw_origin is None:
        raise NativeObservationError("UNIT_ORIGIN_UNAVAILABLE")
    origin: dict[str, str] = {}
    for line in raw_origin.splitlines():
        key, separator, value = line.partition("=")
        _require(
            bool(separator)
            and key in {"LoadState", "FragmentPath", "DropInPaths"}
            and key not in origin,
            "UNIT_ORIGIN_INVALID",
        )
        origin[key] = value
    _require(
        origin == {"LoadState": "loaded", "FragmentPath": str(UNIT_PATH), "DropInPaths": ""}
        or origin == {"LoadState": "not-found", "FragmentPath": "", "DropInPaths": ""},
        "UNIT_NOT_LOADED",
    )
    return runtime


def _service_observation() -> dict[str, Any]:
    """Istniejący kwalifikator plus spójny, przypięty PIDFD inwentarz dowodów."""
    if sys.platform == "win32":
        raise NativeObservationError("NATIVE_LINUX_REQUIRED", status="BLOCKED")
    import grp
    import pwd as posix_accounts

    from bot_core import freshness_service_identity as identity
    from deployment.freshness_runtime import verify_installed_runtime
    from deployment.linux_physical_native import process_observation

    runtime = verify_installed_runtime()
    uid = posix_accounts.getpwnam("os_freshness_crypto_verifier").pw_uid
    gid = grp.getgrnam("freshness_verifier_ipc").gr_gid
    identity.qualify_freshness_service_identity(uid=uid, gid=gid)
    arguments = ("show", UNIT, "--no-pager", "--property=" + ",".join(identity.SYSTEMD_PROPERTIES))
    properties = identity._properties(identity._systemctl(*arguments))
    pid = int(properties["MainPID"])
    descriptor = os.pidfd_open(pid, 0)
    try:
        identity._check_pidfd(descriptor, pid)
        process = identity.capture_process_identity(pid)
        program = identity._program_observation(pid, runtime)
        command = identity._read_exec_start(pid, runtime)
        bootstrap = identity._read_bootstrap_start()
        kernel = process_observation(pid)
        _require(
            process["uid"] == uid
            and process["gid"] == gid
            and process["cgroup"] == properties["ControlGroup"] == identity.CGROUP
            and program["argv"] == identity.expected_runtime_argv(runtime)
            and process["exe_sha256"] == program["exe_sha256"] == runtime.base_executable_sha256
            and kernel["pid"] == pid
            and kernel["uid"] == uid
            and kernel["gid"] == gid
            and kernel["start_time_ticks"] == process["start_time_ticks"]
            and kernel["exe_sha256"] == process["exe_sha256"]
            and kernel["no_new_privs"] == 1
            and set(kernel["capabilities"]) == {"CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"}
            and all(value == "0000000000000000" for value in kernel["capabilities"].values()),
            "SERVICE_PROCESS_BINDING_CHANGED",
        )
        identity.qualify_freshness_service_identity(uid=uid, gid=gid)
        _require(
            properties == identity._properties(identity._systemctl(*arguments))
            and process == identity.capture_process_identity(pid)
            and program == identity._program_observation(pid, runtime)
            and command == identity._read_exec_start(pid, runtime)
            and bootstrap == identity._read_bootstrap_start()
            and kernel == process_observation(pid)
            and runtime == verify_installed_runtime(),
            "SERVICE_CHANGED_DURING_OBSERVATION",
        )
        identity._check_pidfd(descriptor, pid)
        return {
            "properties": properties,
            "process": process,
            "exec_start": command,
            "exec_start_pre": bootstrap,
            "proc_exe": str(runtime.base_executable),
            "program": program,
            "kernel": kernel,
            "pidfd": True,
            "runtime_manifest_sha256": runtime.manifest_sha256,
        }
    finally:
        os.close(descriptor)


def _refusal(exc: Exception, fallback: str) -> dict[str, Any]:
    if isinstance(exc, NativeObservationError):
        return {"status": exc.status, "reason": exc.reason}
    return {"status": "FAIL", "reason": fallback}


def observe_systemd(start: bool = False) -> dict[str, Any]:
    """Sprawdź prawdziwy manager; start wymaga uprzedniej zewnętrznej bramki VM."""
    try:
        _require(
            sys.platform == "linux" and os.geteuid() == 0, "NATIVE_ROOT_REQUIRED", status="BLOCKED"
        )
        _require(type(start) is bool, "SYSTEMD_START_SELECTION_INVALID")
        _protected_unit()
        _command(["/usr/bin/systemd-analyze", "verify", "--man=no", str(UNIT_PATH)])
        if start:
            _command(["/usr/bin/systemctl", "daemon-reload"])
            # Ponowna kontrola fragmentu/drop-inów obejmuje efekty daemon-reload.
            _protected_unit()
            _command(["/usr/bin/systemctl", "start", UNIT])
        state = _state()
        _require(
            state["ActiveState"] == "active"
            and state["SubState"] == "running"
            and state["MainPID"] == state["ExecMainPID"]
            and int(state["MainPID"]) > 1,
            "SERVICE_NOT_RUNNING",
            status="BLOCKED",
        )
        evidence = _service_observation()
        _require(state == _state(), "SERVICE_STATE_CHANGED")
        return {"status": "PASS", "reason": "LIVE_SYSTEMD_IDENTITY_VERIFIED", "evidence": evidence}
    except Exception as exc:
        result = _refusal(exc, "SYSTEMD_IDENTITY_REFUSED")
        try:
            result["state"] = _state()
        except Exception:
            pass
        return result


def observe_postgresql_ipc() -> dict[str, Any]:
    """Zachowaj rzeczywistą diagnostykę verifiera bez produkcyjnego PASS IPC."""
    try:
        _require(
            sys.platform == "linux" and os.geteuid() == 0, "NATIVE_ROOT_REQUIRED", status="BLOCKED"
        )
        from bot_core.linux_service_ipc import load_installed_qualification_policy
        from deployment.linux_physical_ipc import (
            ExpectedPeerProcess,
            collect_verifier_observation,
            validate_verifier_observation,
        )

        installation_id = load_installed_qualification_policy()
        _require(
            installation_id is not None, "QUALIFICATION_POLICY_NOT_INSTALLED", status="BLOCKED"
        )
        service = _service_observation()
        process = service["process"]
        expected = ExpectedPeerProcess(
            pid=process["pid"],
            uid=process["uid"],
            gid=process["gid"],
            start_time_ticks=process["start_time_ticks"],
            executable_sha256=process["exe_sha256"],
        )
        context = {
            "expected": expected,
            "expected_cgroup": process["cgroup"],
            "challenge_id": secrets.token_hex(32),
            "installation_id": installation_id,
        }
        observation = collect_verifier_observation(SOCKET_PATH, **context)
        validate_verifier_observation(observation, **context)
        _require(service == _service_observation(), "SERVICE_CHANGED_DURING_IPC")
        postgres = observation["response"]["postgresql"]
        return {
            "status": "BLOCKED",
            "reason": "FULL_RUNTIME_IPC_QUALIFICATION_UNAVAILABLE",
            "diagnostic_exchange": "PASS",
            "postgresql_service_connection": "PASS" if postgres is not None else "BLOCKED",
            "production_qualification": "BLOCKED",
            "evidence": observation,
        }
    except Exception as exc:
        return _refusal(exc, "POSTGRESQL_IPC_OBSERVATION_REFUSED")
