"""Canonical, fail-closed entrypoint for the frozen live Windows acceptance slice."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Callable

from deployment.core_test_plan import MANIFEST, execute_plan
from deployment.host_identity import canonical_host_os
from deployment.platform_evidence import (
    SCM_ITEMS,
    WINDOWS_LIVE_ITEMS,
    WINDOWS_STAGE6_ITEMS,
    WINDOWS_STAGE7_ITEMS,
    WINDOWS_STAGE8_ITEMS,
    evidence_document,
)
from deployment.windows_stage6_probe import Stage6ProbeError, run_probe
from deployment.windows_stage7_network_probe import (
    Stage7NetworkProbeError,
    run_probe as run_network_probe,
)
from deployment import windows_stage8_postgresql_probe as stage8_probe
from deployment.windows_stage8_postgresql_probe import Stage8PostgreSQLProbeError

GITHUB_PROVIDER = "https://github.com"
LOCAL_PROVIDER = "LOCAL_REVIEWED_WINDOWS_EXECUTION"
RESULT_NAMES = ("WINDOWS_CORE_PLAN", *WINDOWS_LIVE_ITEMS)
STAGE8_DEADLINE_SECONDS = 15 * 60


@contextmanager
def _acceptance_phase(name: str):
    started = time.monotonic()
    print(f"[WINDOWS_ACCEPTANCE_PHASE] START {name}", flush=True)
    try:
        yield
    except BaseException:
        print(
            f"[WINDOWS_ACCEPTANCE_PHASE] FAIL {name} elapsed={time.monotonic() - started:.3f}s",
            flush=True,
        )
        raise
    else:
        print(
            f"[WINDOWS_ACCEPTANCE_PHASE] PASS {name} elapsed={time.monotonic() - started:.3f}s",
            flush=True,
        )


class WindowsAcceptanceError(RuntimeError):
    """Controlled acceptance failure carrying the result which failed."""

    def __init__(self, diagnostic: str, result: str | None = None) -> None:
        super().__init__(diagnostic)
        self.diagnostic = diagnostic
        self.result = result


class AcceptanceBoundary:
    """Reviewed live boundary, replaceable only by unit tests (never evidence qualification)."""

    def host_os(self) -> str:
        return canonical_host_os()

    def is_elevated(self) -> bool:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())

    def dependency_error(self, staging_parent: Path) -> str | None:
        if sys.version_info < (3, 11):
            return "Python 3.11 or newer is required"
        if importlib.util.find_spec("win32serviceutil") is None:
            return "pywin32 is required"
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            return f"project dependency check failed: {completed.stdout.strip()}"
        if not os.environ.get("ProgramData"):
            return "ProgramData is unavailable"
        if shutil.which("powershell") is None or shutil.which("sc.exe") is None:
            return "Windows SCM tooling is unavailable"
        if not staging_parent.is_dir():
            return "test staging location is not an existing directory"
        try:
            with tempfile.NamedTemporaryFile(dir=staging_parent, delete=True):
                pass
        except OSError as exc:
            return f"test staging location is not writable: {exc}"
        return None

    def staging_parent(self) -> Path:
        """Return the existing native root outside the protected Stage-4 layout."""
        program_data = os.environ.get("ProgramData")
        if not program_data:
            raise RuntimeError("ProgramData is unavailable")
        return Path(program_data)

    def run_core(self, revision: str, provider: str, run_id: str, output: Path) -> None:
        execute_plan(
            manifest_path=MANIFEST,
            runner_os="Windows",
            source_revision=revision,
            ci_run_id=run_id,
            ci_provider=provider,
            output=output,
        )

    def run_scm(self) -> dict[str, str]:
        probe = Path(__file__).with_name("windows_scm_probe.ps1")
        run_token = secrets.token_hex(32)
        command = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(probe),
            "-PythonExecutable",
            sys.executable,
            "-WindowsAcceptanceRunToken",
            run_token,
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired as exc:
            cleanup_confirmed = False
            try:
                cleanup = subprocess.run(
                    [*command, "-CleanupOnly"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                cleanup_confirmed = (
                    json.loads(cleanup.stdout.strip().splitlines()[-1]).get("cleanup") == "PASS"
                )
            except (subprocess.TimeoutExpired, IndexError, json.JSONDecodeError):
                pass
            diagnostic = "[TIMEOUT] SCM probe timed out"
            if not cleanup_confirmed:
                diagnostic += "; [CLEANUP] timeout cleanup was not confirmed"
            raise WindowsAcceptanceError(diagnostic, "WINDOWS_CLEANUP") from exc
        if completed.returncode:
            diagnostic = completed.stderr.strip() or completed.stdout.strip() or "SCM probe failed"
            raise WindowsAcceptanceError(diagnostic, _failed_scm_result(diagnostic))
        try:
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise WindowsAcceptanceError("SCM probe returned invalid result", SCM_ITEMS[0]) from exc
        if payload.get("cleanup") != "PASS":
            raise WindowsAcceptanceError("SCM probe cleanup was not confirmed", "WINDOWS_CLEANUP")
        for item in SCM_ITEMS:
            if payload.get(item) != "PASS":
                raise WindowsAcceptanceError(f"SCM result was not PASS: {item}", item)
        if payload.get("WINDOWS_SAFE_OS_SHUTDOWN") != "PASS":
            raise WindowsAcceptanceError(
                "SCM Stage-7 shutdown result was not PASS", "WINDOWS_SAFE_OS_SHUTDOWN"
            )
        return payload

    def run_stage6(self, scratch_parent: Path) -> dict[str, str]:
        return run_probe(scratch_parent)

    def run_stage7_network(self, scratch_parent: Path) -> dict[str, str]:
        return run_network_probe(scratch_parent)

    def run_stage8(self, scratch_parent: Path) -> dict[str, str]:
        status_file = scratch_parent / f".stage8-supervisor-{secrets.token_hex(16)}.json"
        command = [
            sys.executable,
            "-u",
            "-m",
            "deployment.windows_stage8_postgresql_probe",
            "--scratch-parent",
            str(scratch_parent),
            "--status-file",
            str(status_file),
        ]
        started = time.monotonic()
        process = subprocess.Popen(command)
        final_status: dict[str, object] = {}
        try:
            returncode = process.wait(timeout=STAGE8_DEADLINE_SECONDS)
        except subprocess.TimeoutExpired as exc:
            secondary_errors: list[str] = []
            worker_exit_confirmed = False
            try:
                process.terminate()
            except (OSError, PermissionError) as termination_exc:
                secondary_errors.append(f"[STAGE8_TERMINATION] terminate failed: {termination_exc}")
            try:
                process.wait(timeout=10)
                worker_exit_confirmed = True
            except subprocess.TimeoutExpired:
                secondary_errors.append("[STAGE8_TERMINATION] worker did not exit after terminate")
            except OSError as termination_exc:
                secondary_errors.append(
                    f"[STAGE8_TERMINATION] wait after terminate failed: {termination_exc}"
                )
            if not worker_exit_confirmed:
                try:
                    process.kill()
                except (OSError, PermissionError) as termination_exc:
                    secondary_errors.append(f"[STAGE8_TERMINATION] kill failed: {termination_exc}")
                try:
                    process.wait(timeout=10)
                    worker_exit_confirmed = True
                except subprocess.TimeoutExpired:
                    secondary_errors.append(
                        "[STAGE8_TERMINATION] worker process exit not confirmed after kill"
                    )
                except OSError as termination_exc:
                    secondary_errors.append(
                        f"[STAGE8_TERMINATION] wait after kill failed: {termination_exc}"
                    )
            status = stage8_probe._read_json(status_file) or {}
            try:
                _capture_stage8_timeout_observations(status)
            except Exception as diagnostic_exc:
                secondary_errors.append(
                    f"[STAGE8_DIAGNOSTIC] capture failed: {type(diagnostic_exc).__name__}: "
                    f"{diagnostic_exc}"
                )
            try:
                cleanup_errors = stage8_probe.emergency_cleanup(status, scratch_parent)
            except Exception as cleanup_exc:
                cleanup_errors = [
                    f"[STAGE8_CLEANUP] emergency cleanup raised: {type(cleanup_exc).__name__}: "
                    f"{cleanup_exc}"
                ]
            secondary_errors.extend(cleanup_errors)
            diagnostic = _stage8_timeout_diagnostic(
                status, time.monotonic() - started, secondary_errors
            )
            last_phase = str(status.get("last_phase", "UNKNOWN"))
            item = (
                "WINDOWS_LOCAL_PRINCIPAL_AUTHENTICATION"
                if _stage8_principal_phase(last_phase)
                else "WINDOWS_POSTGRESQL_SUBSTRATE"
            )
            raise Stage8PostgreSQLProbeError("STAGE8_TIMEOUT", item, diagnostic) from exc
        else:
            final_status = stage8_probe._read_json(status_file) or {}
        finally:
            status_file.unlink(missing_ok=True)
        if returncode:
            item = str(final_status.get("failed_item", "WINDOWS_POSTGRESQL_SUBSTRATE"))
            raise Stage8PostgreSQLProbeError(
                "STAGE8_WORKER",
                item,
                str(final_status.get("error", "Stage-8 worker failed")),
            )
        results = final_status.get("results")
        expected = {item: "PASS" for item in WINDOWS_STAGE8_ITEMS}
        if final_status.get("completed") is not True or results != expected:
            raise Stage8PostgreSQLProbeError(
                "STAGE8_WORKER_RESULT_INVALID",
                "WINDOWS_POSTGRESQL_SUBSTRATE",
                "worker exited successfully without exact terminal Stage-8 PASS proof",
            )
        return expected


def _stage8_principal_phase(phase: str) -> bool:
    return phase.startswith(("SSPI_", "FINAL_", "MATRIX_", "INTERACTIVE_", "HELPER_"))


def _stage8_timeout_diagnostic(
    status: dict[str, object], elapsed: float, cleanup_errors: list[str]
) -> str:
    last_phase = str(status.get("last_phase", "UNKNOWN"))
    services = status.get("owned_services", [])
    observations = status.get("timeout_service_observations", [])
    if not isinstance(observations, list):
        observations = []
    if not observations and isinstance(services, list):
        for name in services:
            if name in {stage8_probe.RUNTIME_SERVICE, stage8_probe.VERIFIER_SERVICE}:
                try:
                    state, pid = stage8_probe._service_observation(str(name))
                    observations.append(f"{name}:state={state},pid={pid}")
                except Exception as exc:
                    observations.append(f"{name}:state-unavailable={exc}")
    log_tail = str(status.get("timeout_log_tail", ""))
    root = status.get("root")
    if isinstance(root, str):
        try:
            log_tail = (Path(root) / "postgresql.log").read_text(
                encoding="utf-8", errors="replace"
            )[-stage8_probe.STDERR_LIMIT :]
        except OSError:
            pass
    cleanup = "; ".join(cleanup_errors) if cleanup_errors else "PASS"
    if cleanup_errors:
        print(f"[STAGE8_CLEANUP] secondary failure: {cleanup}", file=sys.stderr, flush=True)
    return (
        f"STAGE8_TIMEOUT last_phase={last_phase} elapsed={elapsed:.3f}s "
        f"postgresql_cluster_started={'yes' if status.get('cluster_started') is True else 'no'} "
        f"helper_services_created={'yes' if services else 'no'} "
        f"service_observations={observations!r} postgresql_log_tail={log_tail!r} "
        f"cleanup={cleanup}"
    )


def _capture_stage8_timeout_observations(status: dict[str, object]) -> None:
    """Capture volatile diagnostics before emergency cleanup removes owned resources."""
    observations: list[str] = []
    services = status.get("owned_services", [])
    if isinstance(services, list):
        for name in services:
            if name in {stage8_probe.RUNTIME_SERVICE, stage8_probe.VERIFIER_SERVICE}:
                try:
                    state, pid = stage8_probe._service_observation(str(name))
                    observations.append(f"{name}:state={state},pid={pid}")
                except Exception as exc:
                    observations.append(f"{name}:state-unavailable={exc}")
    status["timeout_service_observations"] = observations
    root = status.get("root")
    if isinstance(root, str):
        try:
            status["timeout_log_tail"] = (Path(root) / "postgresql.log").read_text(
                encoding="utf-8", errors="replace"
            )[-stage8_probe.STDERR_LIMIT :]
        except OSError:
            pass


def publish_artifacts(
    staged_core: Path,
    core_output: Path,
    staged_evidence: Path,
    evidence_output: Path,
    *,
    replace: Callable[[Path, Path], None] | None = None,
) -> None:
    """Publish a new pair, rolling both paths back if either replacement fails."""
    if core_output.exists() or evidence_output.exists():
        raise WindowsAcceptanceError("WINDOWS_ACCEPTANCE_OUTPUT_ALREADY_EXISTS")
    move = replace or (lambda source, target: source.replace(target))
    try:
        move(staged_core, core_output)
        move(staged_evidence, evidence_output)
    except OSError as exc:
        cleanup_errors = []
        for output in (core_output, evidence_output):
            try:
                output.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                cleanup_errors.append(str(cleanup_exc))
        diagnostic = f"WINDOWS_ACCEPTANCE_ARTIFACT_PUBLICATION_FAILED: {exc}"
        if cleanup_errors:
            diagnostic += f"; rollback failed: {'; '.join(cleanup_errors)}"
        raise WindowsAcceptanceError(diagnostic) from exc


def _failed_scm_result(message: str) -> str:
    labels = {
        "INSTALL": "WINDOWS_SERVICE_INSTALLATION",
        "START": "WINDOWS_SERVICE_START",
        "STOP": "WINDOWS_GRACEFUL_STOP",
        "RESTART": "WINDOWS_MANUAL_RESTART",
        "AUTOSTART_QUERY": "WINDOWS_AUTOSTART",
        "CRASH_RESTART": "WINDOWS_AUTOMATIC_CRASH_RESTART",
        "POST_RECOVERY_GRACEFUL_STOP": "WINDOWS_AUTOMATIC_CRASH_RESTART",
        "PROCESS_TREE_START": "WINDOWS_PROCESS_TREE",
        "PROCESS_TREE_MANUAL_RESTART": "WINDOWS_PROCESS_TREE",
        "PROCESS_TREE_CRASH_RESTART": "WINDOWS_PROCESS_TREE",
        "NO_ORPHAN_GRACEFUL_STOP": "WINDOWS_NO_ORPHAN_CHILDREN",
        "NO_ORPHAN_CRASH_RESTART": "WINDOWS_NO_ORPHAN_CHILDREN",
        "NO_ORPHAN_FINAL_STOP": "WINDOWS_NO_ORPHAN_CHILDREN",
        "STAGE5_LOGS_PROVISION": "WINDOWS_PERSISTENT_LOGGING",
        "STAGE5_LOGS_QUALIFICATION": "WINDOWS_PERSISTENT_LOGGING",
        "PERSISTENT_LOGGING_START": "WINDOWS_PERSISTENT_LOGGING",
        "PERSISTENT_LOGGING_MANUAL_RESTART": "WINDOWS_PERSISTENT_LOGGING",
        "PERSISTENT_LOGGING_CRASH_RESTART": "WINDOWS_PERSISTENT_LOGGING",
        "PERSISTENT_LOGGING_FINAL_STOP": "WINDOWS_PERSISTENT_LOGGING",
        "NATIVE_PATHS_DACL": "WINDOWS_ACL_QUALIFICATION",
        "READ_ONLY_DACL_QUALIFICATION": "WINDOWS_ACL_QUALIFICATION",
        "CLEANUP": "WINDOWS_CLEANUP",
        "SAFE_OS_SHUTDOWN_ACCEPTANCE_QUERY": "WINDOWS_SAFE_OS_SHUTDOWN",
        "SAFE_OS_SHUTDOWN_CONTROL": "WINDOWS_SAFE_OS_SHUTDOWN",
        "SAFE_OS_SHUTDOWN_STOP": "WINDOWS_SAFE_OS_SHUTDOWN",
        "SAFE_OS_SHUTDOWN_PROCESS_TREE": "WINDOWS_SAFE_OS_SHUTDOWN",
        "SAFE_OS_SHUTDOWN_LOGGING": "WINDOWS_SAFE_OS_SHUTDOWN",
    }
    upper = message.upper()
    return next((result for label, result in labels.items() if f"[{label}]" in upper), SCM_ITEMS[0])


def _revision(explicit: str | None) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    git_revision = completed.stdout.strip() if completed.returncode == 0 else ""
    value = explicit or git_revision
    if not value or value.lower() in {"unknown", "latest", "working-tree"}:
        raise WindowsAcceptanceError("SOURCE_REVISION_REQUIRED")
    if git_revision:
        if value != git_revision:
            raise WindowsAcceptanceError("SOURCE_REVISION_DOES_NOT_MATCH_CHECKOUT")
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        )
        if status.returncode or status.stdout.strip():
            raise WindowsAcceptanceError("SOURCE_REVISION_HAS_UNCOMMITTED_CHANGES")
    return value


def _identity(mode: str, revision: str | None) -> tuple[str, str, str]:
    if mode == "github":
        source_revision = revision or os.environ.get("GITHUB_SHA", "")
        if not source_revision or source_revision.lower() in {"unknown", "latest", "working-tree"}:
            raise WindowsAcceptanceError("SOURCE_REVISION_REQUIRED")
        if os.environ.get("GITHUB_SERVER_URL") != GITHUB_PROVIDER or not os.environ.get(
            "GITHUB_RUN_ID"
        ):
            raise WindowsAcceptanceError("GITHUB_RELEASE_IDENTITY_REQUIRED")
        if source_revision != os.environ.get("GITHUB_SHA"):
            raise WindowsAcceptanceError("SOURCE_REVISION_DOES_NOT_MATCH_GITHUB_SHA")
        return source_revision, GITHUB_PROVIDER, os.environ["GITHUB_RUN_ID"]
    source_revision = _revision(revision)
    return source_revision, LOCAL_PROVIDER, LOCAL_PROVIDER


def run_acceptance(
    *,
    mode: str,
    revision: str | None,
    core_output: Path,
    evidence_output: Path,
    boundary: AcceptanceBoundary | None = None,
    publisher: Callable[[Path, Path, Path, Path], None] = publish_artifacts,
) -> None:
    """Run the sole reviewed Windows core + SCM sequence and atomically publish evidence."""
    live = boundary or AcceptanceBoundary()
    if live.host_os() != "Windows":
        raise WindowsAcceptanceError("WINDOWS_ACCEPTANCE_RUNTIME=UNVERIFIED_ENVIRONMENT_LIMITATION")
    if not live.is_elevated():
        raise WindowsAcceptanceError("WINDOWS_ACCEPTANCE_REQUIRES_ELEVATION")
    try:
        staging_parent = live.staging_parent()
    except Exception as exc:
        raise WindowsAcceptanceError(f"WINDOWS_NATIVE_PATH_QUALIFICATION_FAILED: {exc}") from exc
    dependency_error = live.dependency_error(staging_parent)
    if dependency_error:
        raise WindowsAcceptanceError(f"WINDOWS_ACCEPTANCE_DEPENDENCY_MISSING: {dependency_error}")
    source_revision, provider, run_id = _identity(mode, revision)
    if core_output.exists() or evidence_output.exists():
        raise WindowsAcceptanceError("WINDOWS_ACCEPTANCE_OUTPUT_ALREADY_EXISTS")
    core_output.parent.mkdir(parents=True, exist_ok=True)
    evidence_output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=core_output.parent) as temporary:
        core_temp = Path(temporary) / "core.json"
        try:
            with _acceptance_phase("CORE"):
                live.run_core(source_revision, provider, run_id, core_temp)
        except Exception as exc:
            raise WindowsAcceptanceError(str(exc), "WINDOWS_CORE_PLAN") from exc
        with _acceptance_phase("SCM_STAGE0_5_AND_STAGE7_SHUTDOWN"):
            scm = live.run_scm()
        try:
            with _acceptance_phase("STAGE6"):
                stage6 = live.run_stage6(staging_parent)
        except Stage6ProbeError as exc:
            raise WindowsAcceptanceError(str(exc), exc.item) from exc
        except Exception as exc:
            raise WindowsAcceptanceError(
                f"[SQLITE_POST_CRASH_VERIFY] {type(exc).__name__}: {exc}",
                "WINDOWS_SQLITE_CRASH_INTEGRITY",
            ) from exc
        for item in WINDOWS_STAGE6_ITEMS:
            if stage6.get(item) != "PASS":
                raise WindowsAcceptanceError(f"Stage-6 result was not PASS: {item}", item)
        try:
            with _acceptance_phase("STAGE7_NETWORK"):
                stage7_network = live.run_stage7_network(staging_parent)
        except Stage7NetworkProbeError as exc:
            raise WindowsAcceptanceError(str(exc), exc.item) from exc
        except Exception as exc:
            raise WindowsAcceptanceError(
                f"[NETWORK_RECOVERY_VERIFY] {type(exc).__name__}: {exc}",
                "WINDOWS_NETWORK_RECOVERY",
            ) from exc
        if stage7_network.get("WINDOWS_NETWORK_RECOVERY") != "PASS":
            raise WindowsAcceptanceError(
                "Stage-7 network result was not PASS", "WINDOWS_NETWORK_RECOVERY"
            )
        stage7 = {
            **stage7_network,
            "WINDOWS_SAFE_OS_SHUTDOWN": scm.get("WINDOWS_SAFE_OS_SHUTDOWN", "FAIL"),
        }
        for item in WINDOWS_STAGE7_ITEMS:
            if stage7.get(item) != "PASS":
                raise WindowsAcceptanceError(f"Stage-7 result was not PASS: {item}", item)
        try:
            with _acceptance_phase("STAGE8"):
                stage8 = live.run_stage8(staging_parent)
        except Stage8PostgreSQLProbeError as exc:
            raise WindowsAcceptanceError(str(exc), exc.item) from exc
        except Exception as exc:
            raise WindowsAcceptanceError(
                f"[POSTGRESQL_DISCOVERY] {type(exc).__name__}: {exc}",
                "WINDOWS_POSTGRESQL_SUBSTRATE",
            ) from exc
        for item in WINDOWS_STAGE8_ITEMS:
            if stage8.get(item) != "PASS":
                raise WindowsAcceptanceError(f"Stage-8 result was not PASS: {item}", item)
        evidence_started = time.monotonic()
        print("[WINDOWS_ACCEPTANCE_PHASE] START EVIDENCE_BUILD", flush=True)
        try:
            results = [
                {
                    "item": item,
                    "status": "PASS",
                    "evidence_class": "LIVE_WINDOWS_INTEGRATION",
                    "test_or_probe": "deployment.windows_acceptance/windows_scm_probe.ps1",
                    "details": scm.get("details", "reviewed SCM lifecycle and cleanup verified"),
                }
                for item in SCM_ITEMS
            ]
            results.extend(
                {
                    "item": item,
                    "status": "PASS",
                    "evidence_class": "LIVE_WINDOWS_INTEGRATION",
                    "test_or_probe": "deployment.windows_stage6_probe",
                    "details": "live Windows forced-termination probe passed",
                }
                for item in WINDOWS_STAGE6_ITEMS
            )
            results.extend(
                (
                    {
                        "item": "WINDOWS_NETWORK_RECOVERY",
                        "status": "PASS",
                        "evidence_class": "LIVE_WINDOWS_INTEGRATION",
                        "test_or_probe": "deployment.windows_stage7_network_probe",
                        "details": "one production request recovered from a real loopback TCP reset",
                    },
                    {
                        "item": "WINDOWS_SAFE_OS_SHUTDOWN",
                        "status": "PASS",
                        "evidence_class": "LIVE_WINDOWS_INTEGRATION",
                        "test_or_probe": "deployment.windows_scm_probe.ps1 / SvcShutdown acceptance",
                        "details": scm.get(
                            "details", "live accepted-controls and shutdown cleanup verified"
                        ),
                    },
                )
            )
            results.extend(
                {
                    "item": item,
                    "status": "PASS",
                    "evidence_class": "LIVE_WINDOWS_INTEGRATION",
                    "test_or_probe": "deployment.windows_stage8_postgresql_probe",
                    "details": stage8.get(
                        "details",
                        "isolated PostgreSQL and distinct SSPI service principals qualified",
                    ),
                }
                for item in WINDOWS_STAGE8_ITEMS
            )
            evidence_temp = Path(temporary) / "scm.json"
            evidence_temp.write_text(
                json.dumps(
                    evidence_document(
                        platform_name="WINDOWS",
                        source_revision=source_revision,
                        ci_provider=provider,
                        ci_run_id=run_id,
                        runner_os="Windows",
                        runner_arch=os.environ.get("PROCESSOR_ARCHITECTURE", "unknown"),
                        results=results,
                    ),
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                "[WINDOWS_ACCEPTANCE_PHASE] PASS EVIDENCE_BUILD "
                f"elapsed={time.monotonic() - evidence_started:.3f}s",
                flush=True,
            )
        except BaseException:
            print(
                "[WINDOWS_ACCEPTANCE_PHASE] FAIL EVIDENCE_BUILD "
                f"elapsed={time.monotonic() - evidence_started:.3f}s",
                flush=True,
            )
            raise
        with _acceptance_phase("ARTIFACT_PUBLICATION"):
            publisher(core_temp, core_output, evidence_temp, evidence_output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Canonical live Windows acceptance")
    parser.add_argument("--mode", required=True, choices=("github", "local"))
    parser.add_argument("--source-revision")
    parser.add_argument("--core-output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run_acceptance(
            mode=args.mode,
            revision=args.source_revision,
            core_output=args.core_output,
            evidence_output=args.evidence_output,
        )
    except WindowsAcceptanceError as exc:
        if exc.result:
            print(f"{exc.result}=FAIL")
        print(exc.diagnostic, file=sys.stderr)
        print("WINDOWS_PRODUCTION_READY=NOT_READY")
        return 1
    for result in RESULT_NAMES:
        print(f"{result}=PASS")
    print("WINDOWS_PRODUCTION_READY=NOT_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
