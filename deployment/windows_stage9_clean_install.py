"""Canonical Stage-9 authority: build and prove one fresh MSI lifecycle."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import time
from typing import Callable
from deployment.platform_evidence import WINDOWS_STAGE9_ITEMS
from deployment.windows_installer.build import main as build_main
from deployment.windows_installer.contract import CONTRACT
from deployment.windows_installer.provision import read_safe_failure, safe_failure_path
from deployment.windows_installer.postgresql_service import (
    SERVICE_NAME as POSTGRESQL_SERVICE_NAME,
    STARTUP_DIAGNOSTIC,
    STARTUP_STAGES,
    startup_diagnostic_path,
)

PROBE_ID = "cryptohunter.windows.clean-install.v1"
PROOFS = (
    "files",
    "services",
    "dacl",
    "postgresql",
    "authority_absent",
    "production_enrollment_fail_closed",
    "uninstall",
    "acceptance_cleanup",
)


class CleanInstallError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _service_exists(name: str) -> bool:
    return subprocess.run(["sc.exe", "query", name], capture_output=True).returncode == 0


def fresh_preconditions() -> None:
    pf = Path(os.environ["ProgramFiles"]) / "CryptoHunter"
    pd = Path(os.environ["ProgramData"]) / "CryptoHunter"
    conflicts = [
        name
        for name in (
            CONTRACT.backend_service,
            CONTRACT.verifier_service,
            CONTRACT.postgresql_service,
        )
        if _service_exists(name)
    ]
    if pf.exists() or pd.exists() or conflicts:
        raise CleanInstallError(
            f"clean-install conflict: services={conflicts}, program_files={pf.exists()}, program_data={pd.exists()}"
        )


def _msiexec(arguments: list[str], log: Path, timeout: int = 600) -> int:
    process = subprocess.run(
        ["msiexec.exe", *arguments, "/qn", "/norestart", "/l*v", str(log)], timeout=timeout
    )
    if process.returncode != 0:
        raise CleanInstallError(f"msiexec failed with {process.returncode}; log={log}")
    return process.returncode


def _capture_provision_failure(logs: Path) -> None:
    """Copy the strict machine diagnostic before reporting the MSI failure."""
    try:
        diagnostic = read_safe_failure()
    except KeyError:
        # Non-Windows unit-test hosts do not define the canonical machine root.
        return
    if diagnostic is None:
        return
    target = logs / "provision-failure.json"
    target.write_text(json.dumps(diagnostic, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PROVISION_FAILURE_STAGE = {diagnostic['stage']}", flush=True)
    print(
        f"PROVISION_FAILURE_EXCEPTION = {diagnostic['first_exception']}", flush=True
    )


def _read_service_startup_diagnostic() -> dict[str, object] | None:
    try:
        value = json.loads(startup_diagnostic_path().read_text(encoding="utf-8"))
    except (KeyError, FileNotFoundError, OSError, ValueError):
        return None
    required = {"service", "stage", "first_exception"}
    if (
        not isinstance(value, dict)
        or not required <= set(value) <= required | {"child_exit_code"}
        or value.get("service") != POSTGRESQL_SERVICE_NAME
        or value.get("stage") not in STARTUP_STAGES
        or not isinstance(value.get("first_exception"), str)
        or (
            "child_exit_code" in value
            and not isinstance(value["child_exit_code"], int)
        )
    ):
        return None
    return value


def _collect_scm_events() -> list[dict[str, object]]:
    """Collect structured SCM fields, not unrestricted event messages."""
    script = rf"""
$start = (Get-Date).AddMinutes(-15)
$events = Get-WinEvent -FilterHashtable @{{LogName='System'; ProviderName='Service Control Manager'; StartTime=$start}} -ErrorAction Stop
$safe = foreach ($event in $events) {{
  $properties = @($event.Properties | ForEach-Object {{ [string]$_.Value }})
  if (($properties -contains '{POSTGRESQL_SERVICE_NAME}') -or ($properties -contains '{CONTRACT.postgresql_display_name}')) {{
    [pscustomobject]@{{time=$event.TimeCreated.ToUniversalTime().ToString('o'); id=$event.Id; properties=$properties}}
  }}
}}
@($safe) | ConvertTo-Json -Compress -Depth 4
"""
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        value = json.loads(result.stdout or "[]")
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    return [
        item
        for item in value
        if isinstance(item, dict) and event_belongs_to_postgresql(item)
    ]


def event_belongs_to_postgresql(event: dict[str, object]) -> bool:
    """Match exact structured identities; never inspect localized messages."""
    properties = event.get("properties", [])
    return isinstance(properties, list) and any(
        value in {CONTRACT.postgresql_service, CONTRACT.postgresql_display_name}
        for value in properties
    )


def classify_service_start_failure(
    diagnostic: dict[str, object] | None, events: list[dict[str, object]]
) -> str:
    """Return a category only when its defining evidence is present."""
    if diagnostic is not None:
        stage = diagnostic["stage"]
        failed = diagnostic["first_exception"] != "NONE"
        if failed and stage == "SCM_DISPATCHER":
            return "SERVICE_HOST_START_FAILURE"
        if failed and stage in {"CREATE_JOB_OBJECT", "CONFIGURE_JOB_OBJECT"}:
            return "JOB_OBJECT_SETUP_FAILURE"
        if failed and stage == "CREATE_SUSPENDED_PROCESS":
            return "POSTGRES_CREATEPROCESS_FAILURE"
        if failed and stage == "ASSIGN_JOB":
            return "JOB_ASSIGNMENT_FAILURE"
        if failed and stage == "VERIFY_JOB_MEMBERSHIP":
            return "JOB_MEMBERSHIP_VERIFICATION_FAILURE"
        if failed and "child_exit_code" in diagnostic and stage in {
            "POSTGRES_PROCESS_ALIVE",
            "WAIT_READY",
        }:
            return "POSTGRES_EARLY_EXIT"
        if (
            failed
            and stage == "WAIT_READY"
            and diagnostic["first_exception"] == "PostgreSQLReadinessTimeout"
        ):
            return "POSTGRES_READINESS_TIMEOUT"
        if failed and stage == "REPORT_RUNNING":
            return "SCM_REPORTING_FAILURE"
    for event in events:
        if not event_belongs_to_postgresql(event):
            continue
        if event.get("id") == 7038:
            return "SERVICE_LOGON_FAILURE"
        properties = event.get("properties", [])
        if event.get("id") == 7000 and isinstance(properties, list) and any(
            value in {"%%2", "%%193"} for value in properties
        ):
            return "SERVICE_HOST_START_FAILURE"
    # Event 7000 alone is deliberately not enough: it is the generic SCM
    # wrapper for several distinct failures, including child/readiness faults.
    return "UNCLASSIFIED"


def _capture_service_start_failure(logs: Path) -> None:
    logs.mkdir(parents=True, exist_ok=True)
    diagnostic = _read_service_startup_diagnostic()
    events = _collect_scm_events()
    evidence = {
        "category": classify_service_start_failure(diagnostic, events),
        "service_diagnostic": diagnostic,
        "scm_events": events,
    }
    (logs / "postgresql-service-startup-evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if diagnostic is not None:
        (logs / STARTUP_DIAGNOSTIC).write_text(
            json.dumps(diagnostic, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"POSTGRES_SERVICE_START_FAILURE = {evidence['category']}", flush=True)


def _install_and_prove(msi: Path, manifest: Path, install_log: Path) -> tuple[int, dict[str, str]]:
    try:
        install_code = _msiexec(["/i", str(msi)], install_log)
    except Exception:
        _capture_provision_failure(install_log.parent)
        _capture_service_start_failure(install_log.parent)
        print("MSI_INSTALL = FAIL", flush=True)
        print(f"MSI_INSTALL_LOG = {install_log}", flush=True)
        raise
    else:
        print("MSI_INSTALL = PASS", flush=True)

    try:
        proofs = _proof_install(manifest)
    except Exception:
        print("CLEAN_INSTALL_PROBE = FAIL", flush=True)
        raise
    else:
        print("CLEAN_INSTALL_PROBE = PASS", flush=True)
        return install_code, proofs


def prove_files(root: Path, machine: Path, manifest: dict[str, object]) -> None:
    expected_exes = {
        "CryptoHunterBackend.exe": root / "CryptoHunterBackend.exe",
        "CryptoHunterFreshnessVerifier.exe": root / "CryptoHunterFreshnessVerifier.exe",
        "CryptoHunterPostgreSQL.exe": root / "PostgreSQL" / "CryptoHunterPostgreSQL.exe",
        "CryptoHunterProvision.exe": root / "CryptoHunterProvision.exe",
    }
    for name, path in expected_exes.items():
        if not path.is_file():
            raise CleanInstallError(f"production file absent: {name}")
        expected = manifest["production_executables"][name]  # type: ignore[index]
        if _sha(path) != expected:
            raise CleanInstallError(f"installed executable hash differs: {name}")
    installed_files = manifest.get("installed_files")
    if not isinstance(installed_files, dict) or not installed_files:
        raise CleanInstallError("installed file hash manifest absent")
    observed = {
        path.relative_to(root).as_posix(): _sha(path)
        for path in root.rglob("*")
        if path.is_file()
    }
    if observed != installed_files:
        raise CleanInstallError("installed file set or SHA-256 differs")
    pg = root / "PostgreSQL"
    for relative in (
        "bin/postgres.exe",
        "bin/pg_ctl.exe",
        "bin/initdb.exe",
        "bin/psql.exe",
        "lib",
        "share",
    ):
        if not (pg / relative).exists():
            raise CleanInstallError(f"bundled PostgreSQL path absent: {relative}")
    forbidden = {
        "Config",
        "State",
        "Logs",
        "Runtime",
        "Updates",
        "Security",
        "Data",
        ".stage9-install-transaction.json",
    }
    if forbidden & {item.name for item in root.iterdir()}:
        raise CleanInstallError("mutable state found in Program Files")
    for name in ("Config", "State", "Logs", "Runtime", "Updates", "PostgreSQL", "Security"):
        if not (machine / name).is_dir():
            raise CleanInstallError(f"machine directory absent: {name}")
    if any(
        path.exists()
        for name in ("windows_test_service.py", "windows_stage8_postgresql_service.py")
        for path in root.rglob(name)
    ):
        raise CleanInstallError("acceptance harness installed")


def _service_facts(name: str) -> tuple[object, object, str]:
    import win32service  # type: ignore[import-not-found]

    manager = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_CONNECT)
    service = win32service.OpenService(
        manager, name, win32service.SERVICE_QUERY_CONFIG | win32service.SERVICE_QUERY_STATUS
    )
    try:
        return (
            win32service.QueryServiceConfig(service),
            win32service.QueryServiceStatus(service),
            name,
        )
    finally:
        win32service.CloseServiceHandle(service)
        win32service.CloseServiceHandle(manager)


def prove_services(root: Path) -> None:
    import win32security, win32service  # type: ignore[import-not-found]

    expected = {
        CONTRACT.postgresql_service: (
            root / "PostgreSQL" / "CryptoHunterPostgreSQL.exe",
            win32service.SERVICE_AUTO_START,
            win32service.SERVICE_RUNNING,
        ),
        CONTRACT.backend_service: (
            root / "CryptoHunterBackend.exe",
            win32service.SERVICE_DEMAND_START,
            win32service.SERVICE_STOPPED,
        ),
        CONTRACT.verifier_service: (
            root / "CryptoHunterFreshnessVerifier.exe",
            win32service.SERVICE_DEMAND_START,
            win32service.SERVICE_STOPPED,
        ),
    }
    for name, (binary, start, state) in expected.items():
        config, status, _ = _service_facts(name)
        observed = Path(str(config[3]).strip('"')).resolve()
        if (
            observed != binary.resolve()
            or config[1] != start
            or config[7] != rf"NT SERVICE\{name}"
            or status[1] != state
        ):
            raise CleanInstallError(f"exact SCM facts differ: {name}")
        win32security.LookupAccountName(None, rf"NT SERVICE\{name}")
        if name == CONTRACT.backend_service and CONTRACT.postgresql_service not in config[6]:
            raise CleanInstallError("backend PostgreSQL dependency absent")


def _qualifier(root: Path, machine: Path, command: str) -> None:
    subprocess.run(
        [
            str(root / "CryptoHunterProvision.exe"),
            command,
            "--program-files",
            str(root),
            "--program-data",
            str(machine),
        ],
        check=True,
        timeout=120,
    )


def prove_dacl(root: Path, machine: Path) -> None:
    _qualifier(root, machine, "qualify-dacl")


def prove_postgresql(root: Path, machine: Path) -> None:
    _qualifier(root, machine, "qualify-postgresql")


def prove_mtls_matrix(root: Path, machine: Path) -> None:
    _qualifier(root, machine, "qualify-mtls")


def prove_backend(root: Path, machine: Path) -> None:
    _qualifier(root, machine, "qualify-backend")


def prove_logging(root: Path, machine: Path) -> None:
    _qualifier(root, machine, "qualify-logging")


def prove_authority_absent(machine: Path) -> None:
    if (machine / "State" / "corehost.sqlite").exists():
        raise CleanInstallError("machine installation created product authority")


def prove_production_enrollment_fail_closed(root: Path, machine: Path) -> None:
    result = subprocess.run(
        [
            str(root / "CryptoHunterProvision.exe"),
            "enroll",
            "--program-files",
            str(root),
            "--program-data",
            str(machine),
        ],
        capture_output=True,
        timeout=120,
    )
    stderr = result.stderr.decode(errors="replace")
    expected = (
        "provisioning failed: operation=enroll "
        "first_exception=WindowsProvisioningAdapterUnavailable"
    )
    if (
        result.returncode == 0
        or (machine / "State" / "corehost.sqlite").exists()
        or expected not in stderr.splitlines()
    ):
        raise CleanInstallError("production enrollment did not fail closed for expected cause")


def _proof_install(manifest_path: Path) -> dict[str, str]:
    root = Path(os.environ["ProgramFiles"]) / "CryptoHunter"
    machine = Path(os.environ["ProgramData"]) / "CryptoHunter"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    proofs: dict[str, str] = {}
    prove_files(root, machine, manifest)
    proofs["files"] = "PASS"
    prove_services(root)
    proofs["services"] = "PASS"
    prove_dacl(root, machine)
    proofs["dacl"] = "PASS"
    prove_postgresql(root, machine)
    proofs["postgresql"] = "PASS"
    prove_authority_absent(machine)
    proofs["authority_absent"] = "PASS"
    prove_production_enrollment_fail_closed(root, machine)
    proofs["production_enrollment_fail_closed"] = "PASS"
    return proofs


def acceptance_owned_purge(machine: Path) -> None:
    record_path = machine / ".stage9-install-ownership.json"
    value = json.loads(record_path.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != 1
        or value.get("state") != "COMMITTED"
        or value.get("program_data") != str(machine.resolve())
        or value.get("service_names")
        != [CONTRACT.postgresql_service, CONTRACT.backend_service, CONTRACT.verifier_service]
    ):
        raise CleanInstallError("persistent install ownership record differs")
    expected = {
        "Config",
        "State",
        "Logs",
        "Runtime",
        "Updates",
        "PostgreSQL",
        "Security",
        record_path.name,
    }
    if {item.name for item in machine.iterdir()} != expected:
        raise CleanInstallError("acceptance cleanup found unowned top-level content")
    for path in (machine, *machine.rglob("*")):
        if getattr(path.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise CleanInstallError("acceptance cleanup rejects reparse points")
        if machine.resolve() not in path.resolve().parents and path.resolve() != machine.resolve():
            raise CleanInstallError("acceptance cleanup path escaped machine root")
    shutil.rmtree(machine)
    # A successful commit normally removed this already.  Acceptance may clean
    # a valid residue only after the run had its opportunity to collect evidence.
    diagnostic = read_safe_failure()
    if diagnostic is not None:
        safe_failure_path().unlink(missing_ok=True)


def run(args: argparse.Namespace) -> None:
    if os.name != "nt" or platform.machine().upper() not in {"AMD64", "X86_64"}:
        raise CleanInstallError("native Windows x64 required")
    if WINDOWS_STAGE9_ITEMS != ("WINDOWS_CLEAN_INSTALL",):
        raise CleanInstallError("Stage-9 item contract changed")
    print("EXE_BUILD = NOT_RUN", flush=True)
    print("WIX_COMPILE = NOT_RUN", flush=True)
    print("MSI_CREATED = NO", flush=True)
    print("MSI_INSTALL = NOT_RUN", flush=True)
    print("CLEAN_INSTALL_PROBE = NOT_RUN", flush=True)
    print("POST_ENROLLMENT_LIVE_QUALIFICATION = REQUIRED", flush=True)
    fresh_preconditions()
    build_main(
        [
            "--version",
            args.version,
            "--output",
            str(args.output),
            *(
                ["--postgresql-archive", str(args.postgresql_archive)]
                if args.postgresql_archive
                else []
            ),
        ]
    )
    msi = args.output / f"CryptoHunter-{args.version}-windows-x64.msi"
    manifest = args.output / "installer-manifest.json"
    install_code, proofs = _install_and_prove(msi, manifest, args.logs / "install.log")
    uninstall_code = _msiexec(["/x", str(msi)], args.logs / "uninstall.log")
    if (
        any(
            _service_exists(n)
            for n in (
                CONTRACT.backend_service,
                CONTRACT.verifier_service,
                CONTRACT.postgresql_service,
            )
        )
        or (Path(os.environ["ProgramFiles"]) / "CryptoHunter").exists()
    ):
        raise CleanInstallError("uninstall left product resources")
    proofs["uninstall"] = "PASS"
    machine = Path(os.environ["ProgramData"]) / "CryptoHunter"
    # CI-only purge after exact clean-install ownership was proven by this run.
    acceptance_owned_purge(machine)
    proofs["acceptance_cleanup"] = "PASS"
    receipt = {
        "schema_version": 1,
        "source_revision": os.environ["GITHUB_SHA"],
        "ci_provider": os.environ["GITHUB_SERVER_URL"],
        "ci_run_id": os.environ["GITHUB_RUN_ID"],
        "runner_os": os.environ["RUNNER_OS"],
        "runner_arch": os.environ["RUNNER_ARCH"],
        "probe_id": PROBE_ID,
        "product_version": args.version,
        "msi_sha256": _sha(msi),
        "manifest_sha256": _sha(manifest),
        "install_exit_code": install_code,
        "uninstall_exit_code": uninstall_code,
        "proofs": proofs,
        "post_enrollment_live_qualification": "REQUIRED",
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--version", required=True)
    p.add_argument("--postgresql-archive", type=Path)
    p.add_argument("--output", type=Path, default=Path("dist/windows"))
    p.add_argument("--logs", type=Path, default=Path("dist/windows/logs"))
    p.add_argument("--receipt", type=Path, required=True)
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    args.logs.mkdir(parents=True, exist_ok=True)
    try:
        run(args)
    except Exception as exc:
        print(f"Stage-9 probe failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
