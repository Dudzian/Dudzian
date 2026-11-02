"""SCM host for the exact bundled PostgreSQL process tree."""

from __future__ import annotations
import os
import json
from pathlib import Path
import stat
import subprocess
import threading
import time
from typing import Any
from deployment.windows_installer.contract import CONTRACT
from deployment.windows_installer.service_base import (
    build_smoke_requested,
    machine_root,
    run_service,
)

SERVICE_NAME = CONTRACT.postgresql_service
READY_TIMEOUT = 60.0
STARTUP_DIAGNOSTIC = ".CryptoHunter.postgresql-service-startup.json"
STARTUP_STAGES = (
    "SERVICE_ENTRY",
    "SCM_DISPATCHER",
    "QUALIFY_PATHS",
    "CREATE_JOB_OBJECT",
    "CONFIGURE_JOB_OBJECT",
    "CREATE_SUSPENDED_PROCESS",
    "ASSIGN_JOB",
    "VERIFY_JOB_MEMBERSHIP",
    "RESUME_PROCESS",
    "POSTGRES_PROCESS_ALIVE",
    "WAIT_READY",
    "REPORT_RUNNING",
    "MONITOR_PROCESS",
    "STOP_PROCESS",
)


class PostgreSQLReadinessTimeout(RuntimeError):
    """The live postmaster did not become ready within the fixed deadline."""


def startup_diagnostic_path() -> Path:
    """The installer-owned file is outside its rollback-owned directory tree."""
    return Path(os.environ["ProgramData"]) / STARTUP_DIAGNOSTIC


def _first_exception_type(exc: BaseException) -> str:
    seen: set[int] = set()
    current = exc
    while id(current) not in seen:
        seen.add(id(current))
        earlier = current.__cause__ or current.__context__
        if earlier is None:
            break
        current = earlier
    return type(current).__name__


def try_write_startup_diagnostic(
    stage: str, exc: BaseException | None = None, child_exit_code: int | None = None
) -> bool:
    """Best-effort write to the install-time file; never create or affect startup."""
    try:
        if stage not in STARTUP_STAGES:
            return False
        path = startup_diagnostic_path()
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or (
            getattr(metadata, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        ):
            return False
        value: dict[str, object] = {
            "service": SERVICE_NAME,
            "stage": stage,
            "first_exception": _first_exception_type(exc) if exc is not None else "NONE",
        }
        if child_exit_code is not None:
            value["child_exit_code"] = int(child_exit_code)
        # r+ cannot create a missing target. The service SID needs no rights on
        # the ProgramData root and normal post-commit starts become a no-op.
        with path.open("r+", encoding="utf-8") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or (
                getattr(opened, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
            ):
                return False
            stream.seek(0)
            json.dump(value, stream, sort_keys=True)
            stream.truncate()
            stream.flush()
            os.fsync(stream.fileno())
        return True
    except Exception:
        return False


def _install_root() -> Path:
    return Path(os.environ["ProgramFiles"]) / "CryptoHunter"


def create_suspended_in_job(
    command: list[str],
    cwd: Path,
    *,
    win32api: Any,
    win32con: Any,
    win32event: Any,
    win32job: Any,
    win32process: Any,
    stage: Any = lambda _value: None,
) -> tuple[Any, Any, Any, int]:
    """Create suspended, assign and query-back before any child can spawn."""
    job = process = thread = None
    try:
        stage("CREATE_JOB_OBJECT")
        job = win32job.CreateJobObject(None, "")
        stage("CONFIGURE_JOB_OBJECT")
        limits = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
        limits["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, limits)
        stage("CREATE_SUSPENDED_PROCESS")
        startup = win32process.STARTUPINFO()
        process, thread, pid, _ = win32process.CreateProcess(
            None,
            subprocess.list2cmdline(command),
            None,
            None,
            False,
            win32con.CREATE_SUSPENDED,
            None,
            str(cwd),
            startup,
        )
        stage("ASSIGN_JOB")
        win32job.AssignProcessToJobObject(job, process)
        stage("VERIFY_JOB_MEMBERSHIP")
        observed = win32job.QueryInformationJobObject(job, win32job.JobObjectBasicProcessIdList)
        try:
            pids = {int(value) for value in observed}
        except (TypeError, ValueError) as exc:
            raise RuntimeError("unexpected JobObjectBasicProcessIdList result") from exc
        if pids != {pid}:
            raise RuntimeError("PostgreSQL root Job assignment query-back failed")
        stage("RESUME_PROCESS")
        win32process.ResumeThread(thread)
    except BaseException as original:
        # Job close cannot kill a child whose assignment failed. Terminate the
        # exact retained handle even before assignment, then observe its exit
        # before releasing the process handle. Cleanup failures must neither
        # hide the original error nor prevent the other resources being freed.
        exit_confirmed = process is None
        if process is not None:
            try:
                win32api.TerminateProcess(process, 1)
            except BaseException as cleanup:
                original.add_note(
                    f"PostgreSQL TerminateProcess cleanup failed: {type(cleanup).__name__}"
                )
        if job is not None:
            try:
                job.Close()
            except BaseException as cleanup:
                original.add_note(f"PostgreSQL job cleanup failed: {type(cleanup).__name__}")
        if process is not None:
            try:
                observed = win32event.WaitForSingleObject(process, 10_000)
                if observed != win32event.WAIT_OBJECT_0:
                    raise RuntimeError(
                        f"PostgreSQL cleanup exit unconfirmed: wait_result={observed}"
                    )
                exit_confirmed = True
            except BaseException as cleanup:
                original.add_note(f"PostgreSQL exit confirmation failed: {type(cleanup).__name__}")
        if not exit_confirmed:
            # Preserve ownership for the caller if Windows cannot confirm
            # death. Dropping the last process handle would make a suspended,
            # unassigned child unrecoverable without unsafe PID rediscovery.
            setattr(original, "postgresql_unconfirmed_process_handle", process)
            original.add_note("PostgreSQL process handle retained: exit not confirmed")
        for label, handle in (("thread", thread), ("process", process)):
            if handle is not None and (label != "process" or exit_confirmed):
                try:
                    handle.Close()
                except BaseException as cleanup:
                    original.add_note(
                        f"PostgreSQL {label} cleanup failed: {type(cleanup).__name__}"
                    )
        raise
    return job, process, thread, pid


def wait_ready(
    pg_isready: Path,
    process: Any,
    stop: threading.Event,
    *,
    timeout: float = READY_TIMEOUT,
    clock: Any = time.monotonic,
    run: Any = subprocess.run,
    win32event: Any,
    win32process: Any,
) -> bool:
    """Return True when ready, False when cancelled; observed failures raise."""
    deadline = clock() + timeout

    def require_alive() -> None:
        observed = win32event.WaitForSingleObject(process, 0)
        if observed == win32event.WAIT_OBJECT_0:
            code = win32process.GetExitCodeProcess(process)
            raise RuntimeError(f"PostgreSQL exited before readiness: exit_code={code}")
        if observed != win32event.WAIT_TIMEOUT:
            raise RuntimeError(f"unexpected PostgreSQL wait result: {observed}")

    while not stop.is_set():
        if clock() >= deadline:
            break
        require_alive()
        if stop.is_set():
            return False
        result = run(
            [
                str(pg_isready),
                "-h",
                CONTRACT.postgresql_host,
                "-p",
                str(CONTRACT.postgresql_port),
                "-d",
                "freshness_gate",
            ],
            capture_output=True,
            timeout=5,
        )
        # The original postmaster may have exited while the probe ran. An
        # observed process/API failure must not be hidden by a concurrent stop.
        require_alive()
        if stop.is_set():
            return False
        if clock() >= deadline:
            break
        if result.returncode == 0:
            return True
        stop.wait(0.1)
    if stop.is_set():
        return False
    raise PostgreSQLReadinessTimeout("PostgreSQL local readiness deadline expired")


def require_clean_exit(
    process: Any, *, timeout_ms: int, win32event: Any, win32process: Any
) -> None:
    observed = win32event.WaitForSingleObject(process, timeout_ms)
    if observed != win32event.WAIT_OBJECT_0:
        raise RuntimeError(f"PostgreSQL did not stop cleanly: wait_result={observed}")
    code = win32process.GetExitCodeProcess(process)
    if code != 0:
        raise RuntimeError(f"PostgreSQL stopped with exit_code={code}")


def monitor_process(
    process: Any, stop: threading.Event, *, win32event: Any, win32process: Any
) -> None:
    """Observe the original postmaster handle until an operator requests stop."""
    while not stop.is_set():
        observed = win32event.WaitForSingleObject(process, 100)
        if observed == win32event.WAIT_OBJECT_0:
            code = win32process.GetExitCodeProcess(process)
            # A spontaneous clean exit is also a service failure: nobody asked
            # this worker to stop, and it can no longer truthfully be RUNNING.
            raise RuntimeError(f"PostgreSQL exited while running: exit_code={code}")
        if observed != win32event.WAIT_TIMEOUT:
            raise RuntimeError(f"unexpected PostgreSQL wait result: {observed}")


def _run(stop: threading.Event, report: object) -> None:
    import win32api, win32con, win32event, win32job, win32process, win32service  # type: ignore[import-not-found]

    if stop.is_set():
        return
    stage = "QUALIFY_PATHS"
    try_write_startup_diagnostic(stage)
    postgres = _install_root() / "PostgreSQL" / "bin" / "postgres.exe"
    pg_ctl = postgres.with_name("pg_ctl.exe")
    pg_isready = postgres.with_name("pg_isready.exe")
    pgdata = machine_root() / "PostgreSQL" / "Data"
    try:
        for path in (postgres, pg_ctl, pg_isready, pgdata / "PG_VERSION"):
            if not path.exists():
                raise RuntimeError(f"qualified private PostgreSQL path absent: {path.name}")
    except BaseException as exc:
        try_write_startup_diagnostic(stage, exc)
        raise
    job = process = thread = None
    failure: BaseException | None = None
    try:

        def set_stage(value: str) -> None:
            nonlocal stage
            stage = value
            try_write_startup_diagnostic(stage)

        job, process, thread, _ = create_suspended_in_job(
            [str(postgres), "-D", str(pgdata)],
            postgres.parent,
            win32api=win32api,
            win32con=win32con,
            win32event=win32event,
            win32job=win32job,
            win32process=win32process,
            stage=set_stage,
        )
        set_stage("POSTGRES_PROCESS_ALIVE")
        if win32event.WaitForSingleObject(process, 0) != win32event.WAIT_TIMEOUT:
            raise RuntimeError("PostgreSQL exited immediately after resume")
        set_stage("WAIT_READY")
        ready = wait_ready(
            pg_isready, process, stop, win32event=win32event, win32process=win32process
        )
        if ready and not stop.is_set():
            set_stage("REPORT_RUNNING")
            report(win32service.SERVICE_RUNNING)  # type: ignore[operator]
            set_stage("MONITOR_PROCESS")
            monitor_process(process, stop, win32event=win32event, win32process=win32process)
        set_stage("STOP_PROCESS")
        subprocess.run(
            [str(pg_ctl), "stop", "-D", str(pgdata), "-m", "fast", "-w", "-t", "30"],
            check=True,
            timeout=35,
        )
        require_clean_exit(
            process, timeout_ms=10_000, win32event=win32event, win32process=win32process
        )
    except BaseException as exc:
        failure = exc
        exit_code = None
        try:
            if (
                process is not None
                and win32event.WaitForSingleObject(process, 0) == win32event.WAIT_OBJECT_0
            ):
                exit_code = win32process.GetExitCodeProcess(process)
        except BaseException:
            exit_code = None
        try_write_startup_diagnostic(stage, exc, exit_code)
        raise
    finally:
        cleanup_failure: BaseException | None = None
        for label, handle in (("thread", thread), ("process", process), ("job", job)):
            if handle is not None:
                try:
                    handle.Close()
                except BaseException as cleanup:
                    if failure is not None:
                        failure.add_note(
                            f"PostgreSQL {label} cleanup failed: {type(cleanup).__name__}"
                        )
                    elif cleanup_failure is None:
                        cleanup_failure = cleanup
                    else:
                        cleanup_failure.add_note(
                            f"PostgreSQL {label} cleanup failed: {type(cleanup).__name__}"
                        )
        if failure is None and cleanup_failure is not None:
            try_write_startup_diagnostic(stage, cleanup_failure)
            raise cleanup_failure


def main() -> None:
    if build_smoke_requested():
        return
    try_write_startup_diagnostic("SERVICE_ENTRY")
    try_write_startup_diagnostic("SCM_DISPATCHER")
    worker_failed = False

    def worker(stop: threading.Event, report: object) -> None:
        nonlocal worker_failed
        try:
            _run(stop, report)
        except BaseException:
            worker_failed = True
            raise

    try:
        run_service(SERVICE_NAME, worker)
    except BaseException as exc:
        if not worker_failed:
            try_write_startup_diagnostic("SCM_DISPATCHER", exc)
        raise


if __name__ == "__main__":
    main()
