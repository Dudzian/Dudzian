"""SCM host for the exact bundled PostgreSQL process tree."""

from __future__ import annotations
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any
from deployment.windows_installer.contract import CONTRACT
from deployment.windows_installer.service_base import machine_root, run_service

SERVICE_NAME = "CryptoHunterPostgreSQL"
READY_TIMEOUT = 60.0


def _install_root() -> Path:
    return Path(os.environ["ProgramFiles"]) / "CryptoHunter"


def create_suspended_in_job(
    command: list[str], cwd: Path, *, win32api: Any, win32con: Any, win32job: Any, win32process: Any
) -> tuple[Any, Any, Any, int]:
    """Create suspended, assign and query-back before any child can spawn."""
    job = win32job.CreateJobObject(None, None)
    limits = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
    limits["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, limits)
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
    try:
        win32job.AssignProcessToJobObject(job, process)
        observed = win32job.QueryInformationJobObject(job, win32job.JobObjectBasicProcessIdList)
        pids = {
            int(value) for value in observed.get("ProcessIdList", observed.get("ProcessIds", []))
        }
        if pids != {pid}:
            raise RuntimeError("PostgreSQL root Job assignment query-back failed")
        win32process.ResumeThread(thread)
    except BaseException:
        job.Close()
        process.Close()
        thread.Close()
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
) -> None:
    deadline = clock() + timeout
    while clock() < deadline and not stop.is_set():
        observed = win32event.WaitForSingleObject(process, 0)
        if observed == win32event.WAIT_OBJECT_0:
            code = win32process.GetExitCodeProcess(process)
            raise RuntimeError(f"PostgreSQL exited before readiness: exit_code={code}")
        if observed != win32event.WAIT_TIMEOUT:
            raise RuntimeError(f"unexpected PostgreSQL wait result: {observed}")
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
        if result.returncode == 0:
            return
        stop.wait(0.1)
    raise RuntimeError("PostgreSQL local readiness deadline expired")


def require_clean_exit(
    process: Any, *, timeout_ms: int, win32event: Any, win32process: Any
) -> None:
    observed = win32event.WaitForSingleObject(process, timeout_ms)
    if observed != win32event.WAIT_OBJECT_0:
        raise RuntimeError(f"PostgreSQL did not stop cleanly: wait_result={observed}")
    code = win32process.GetExitCodeProcess(process)
    if code != 0:
        raise RuntimeError(f"PostgreSQL stopped with exit_code={code}")


def _run(stop: threading.Event, report: object) -> None:
    import win32api, win32con, win32event, win32job, win32process, win32service  # type: ignore[import-not-found]

    postgres = _install_root() / "PostgreSQL" / "bin" / "postgres.exe"
    pg_ctl = postgres.with_name("pg_ctl.exe")
    pg_isready = postgres.with_name("pg_isready.exe")
    pgdata = machine_root() / "PostgreSQL" / "Data"
    for path in (postgres, pg_ctl, pg_isready, pgdata / "PG_VERSION"):
        if not path.exists():
            raise RuntimeError(f"qualified private PostgreSQL path absent: {path.name}")
    job, process, thread, _ = create_suspended_in_job(
        [str(postgres), "-D", str(pgdata)],
        postgres.parent,
        win32api=win32api,
        win32con=win32con,
        win32job=win32job,
        win32process=win32process,
    )
    try:
        wait_ready(pg_isready, process, stop, win32event=win32event, win32process=win32process)
        report(win32service.SERVICE_RUNNING)  # type: ignore[operator]
        stop.wait()
        subprocess.run(
            [str(pg_ctl), "stop", "-D", str(pgdata), "-m", "fast", "-w", "-t", "30"],
            check=True,
            timeout=35,
        )
        require_clean_exit(
            process, timeout_ms=10_000, win32event=win32event, win32process=win32process
        )
    finally:
        thread.Close()
        process.Close()
        job.Close()


def main() -> None:
    run_service(SERVICE_NAME, _run)


if __name__ == "__main__":
    main()
