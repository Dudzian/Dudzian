"""Windows Job Object boundary used by the SCM acceptance service."""

from __future__ import annotations

import json
import ntpath
import os
from pathlib import Path
import subprocess
import time
from typing import Any

JOB_PREFIX = r"Global\CryptoHunterBackend.ProcessTree."
MARKER_NAME = "process-tree.json"
GATE_NAME = "process-tree.gate"
SCHEMA_VERSION = 1
POLL_INTERVAL = 0.05
TREE_TIMEOUT = 20.0
CHILD_EXIT_TIMEOUT = 5.0
# Win32 DETACHED_PROCESS.  Keep this private Stage-5 constant import-safe on
# non-Windows hosts, where subprocess does not expose Windows creation flags.
_DETACHED_PROCESS = 0x00000008
TRANSIENT_NAMES = (
    MARKER_NAME,
    GATE_NAME,
    "process-tree.tmp",
    "process-tree.json.tmp",
)


class ProcessTreeError(RuntimeError):
    """The child Job Object contract could not be established."""


def reviewed_python_executable(record_path: Path) -> str:
    """Load the sole reviewed child interpreter authority from Stage-2 ownership."""
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProcessTreeError("reviewed Python ownership record is unavailable") from exc
    required = {
        "ownership_phase": "SERVICE_PROVEN",
        "strict_create_result": "CREATED",
        "service_name": "CryptoHunterBackend",
        "service_identity": r"NT SERVICE\CryptoHunterBackend",
    }
    if not isinstance(record, dict) or any(record.get(key) != value for key, value in required.items()):
        raise ProcessTreeError("reviewed Python ownership authority mismatch")
    executable = record.get("python_executable")
    if (not isinstance(executable, str) or not executable or not ntpath.isabs(executable)
            or not ntpath.splitdrive(executable)[0] or "/" in executable):
        raise ProcessTreeError("reviewed Python path is not a fully qualified native path")
    if not Path(executable).is_file():
        raise ProcessTreeError("reviewed Python executable is not an existing regular file")
    return executable


def job_name(service_pid: int) -> str:
    if not isinstance(service_pid, int) or isinstance(service_pid, bool) or service_pid <= 0:
        raise ProcessTreeError("service_pid must be positive")
    return f"{JOB_PREFIX}{service_pid}"


def _pids(info: Any) -> set[int]:
    if isinstance(info, dict):
        values = info.get("ProcessIdList", info.get("ProcessIds", []))
    else:
        values = info
    return {int(value) for value in values}


def wait_for_exact_pids(job: Any, expected: set[int], *, win32job: Any,
                        timeout: float = TREE_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    observed: set[int] = set()
    while time.monotonic() < deadline:
        observed = _pids(win32job.QueryInformationJobObject(
            job, win32job.JobObjectBasicProcessIdList))
        if observed == expected:
            return
        time.sleep(POLL_INTERVAL)
    raise ProcessTreeError(f"job PID set mismatch expected={sorted(expected)} observed={sorted(observed)}")


class ChildJob:
    """Own a kill-on-close job and its deliberately gated qualification child."""

    def __init__(self, runtime: Path, *, python_executable: str, win32api: Any, win32con: Any,
                 win32job: Any) -> None:
        if not isinstance(python_executable, str) or not python_executable:
            raise ProcessTreeError("reviewed python_executable is required")
        self.runtime = runtime
        self.python_executable = python_executable
        self.win32api, self.win32con, self.win32job = win32api, win32con, win32job
        self.service_pid = os.getpid()
        self.name = job_name(self.service_pid)
        self.handle: Any = None
        self.child: subprocess.Popen[str] | None = None

    def start(self) -> dict[str, Any]:
        gate = self.runtime / GATE_NAME
        marker = self.runtime / MARKER_NAME
        gate.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)
        self.handle = self.win32job.CreateJobObject(None, self.name)
        # CreateJobObject may return an existing named object: never adopt it.
        if self.win32api.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
            self.close()
            raise ProcessTreeError("named Job Object already exists")
        extended = self.win32job.QueryInformationJobObject(
            self.handle, self.win32job.JobObjectExtendedLimitInformation)
        extended["BasicLimitInformation"]["LimitFlags"] = (
            self.win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
        self.win32job.SetInformationJobObject(
            self.handle, self.win32job.JobObjectExtendedLimitInformation, extended)
        confirmed = self.win32job.QueryInformationJobObject(
            self.handle, self.win32job.JobObjectExtendedLimitInformation)
        if confirmed["BasicLimitInformation"]["LimitFlags"] != (
                self.win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE):
            self.close()
            raise ProcessTreeError("KILL_ON_JOB_CLOSE query-back mismatch")
        helper = Path(__file__).with_name("windows_process_tree_child.py")
        assignment_proven = False
        try:
            self.child = subprocess.Popen(
                [self.python_executable, str(helper), str(gate)],
                text=True,
                creationflags=_DETACHED_PROCESS,
            )
            process = self.win32api.OpenProcess(
                self.win32con.PROCESS_SET_QUOTA | self.win32con.PROCESS_TERMINATE |
                self.win32con.PROCESS_QUERY_LIMITED_INFORMATION, False, self.child.pid)
            try:
                self.win32job.AssignProcessToJobObject(self.handle, process)
            finally:
                process.Close()
            wait_for_exact_pids(self.handle, {self.child.pid}, win32job=self.win32job)
            assignment_proven = True
            # Assignment and its query-back intentionally precede gate release.
            self._release_gate(gate)
            grandchild_pid = self._wait_for_grandchild(gate)
            wait_for_exact_pids(self.handle, {self.child.pid, grandchild_pid},
                                win32job=self.win32job)
            value = {"schema_version": SCHEMA_VERSION, "service_pid": self.service_pid,
                     "child_pid": self.child.pid, "grandchild_pid": grandchild_pid,
                     "job_name": self.name}
            temporary = marker.with_name(marker.name + ".tmp")
            temporary.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")
            temporary.replace(marker)
            return value
        except Exception as exc:
            self._rollback_start(assignment_proven, exc)
            raise

    @staticmethod
    def _release_gate(gate: Path) -> None:
        gate.write_text("assigned", encoding="ascii")

    def _wait_for_grandchild(self, gate: Path) -> int:
        deadline = time.monotonic() + TREE_TIMEOUT
        while time.monotonic() < deadline:
            try:
                value = int(gate.read_text(encoding="ascii"))
                if value > 0:
                    return value
            except (OSError, ValueError):
                pass
            time.sleep(POLL_INTERVAL)
        raise ProcessTreeError("grandchild creation deadline expired")

    def close(self) -> None:
        if self.handle is not None:
            self.handle.Close()
            self.handle = None

    def _rollback_start(self, assignment_proven: bool, failure: Exception) -> None:
        child = self.child
        self.close()  # kill-on-close is primary once membership was proven
        emergency_kill = False
        try:
            if child is not None:
                if not assignment_proven:
                    try:
                        child.terminate()
                    except Exception:
                        emergency_kill = True
                        child.kill()
                try:
                    child.wait(timeout=CHILD_EXIT_TIMEOUT)
                except subprocess.TimeoutExpired:
                    emergency_kill = True
                    child.kill()
                    child.wait(timeout=CHILD_EXIT_TIMEOUT)
        finally:
            for name in TRANSIENT_NAMES:
                (self.runtime / name).unlink(missing_ok=True)
        if assignment_proven and emergency_kill:
            raise ProcessTreeError(
                "kill-on-close did not stop the child within the bounded deadline") from failure
