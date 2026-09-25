"""Canonical, fail-closed entrypoint for the frozen live Windows acceptance slice."""

from __future__ import annotations

import argparse
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
from typing import Callable

from deployment.core_test_plan import MANIFEST, execute_plan
from deployment.host_identity import canonical_host_os
from deployment.platform_evidence import SCM_ITEMS, evidence_document
from deployment.platforms.windows import resolve_paths

GITHUB_PROVIDER = "https://github.com"
LOCAL_PROVIDER = "LOCAL_REVIEWED_WINDOWS_EXECUTION"
RESULT_NAMES = ("WINDOWS_CORE_PLAN", *SCM_ITEMS)


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
            [sys.executable, "-m", "pip", "check"], check=False,
            capture_output=True, text=True,
        )
        if completed.returncode:
            return f"project dependency check failed: {completed.stdout.strip()}"
        if not os.environ.get("ProgramData"):
            return "ProgramData is unavailable"
        if shutil.which("powershell") is None or shutil.which("sc.exe") is None:
            return "Windows SCM tooling is unavailable"
        try:
            staging_parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=staging_parent, delete=True):
                pass
        except OSError as exc:
            return f"test staging location is not writable: {exc}"
        return None

    def staging_parent(self) -> Path:
        """Qualify native known-folder paths before creating any live-test state."""
        return resolve_paths().runtime / "AcceptanceStaging"

    def run_core(self, revision: str, provider: str, run_id: str, output: Path) -> None:
        execute_plan(
            manifest_path=MANIFEST, runner_os="Windows", source_revision=revision,
            ci_run_id=run_id, ci_provider=provider, output=output,
        )

    def run_scm(self) -> dict[str, str]:
        probe = Path(__file__).with_name("windows_scm_probe.ps1")
        run_token = secrets.token_hex(32)
        command = [
            "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(probe),
            "-PythonExecutable", sys.executable,
            "-WindowsAcceptanceRunToken", run_token,
        ]
        try:
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True, timeout=180,
            )
        except subprocess.TimeoutExpired as exc:
            cleanup_confirmed = False
            try:
                cleanup = subprocess.run(
                    [*command, "-CleanupOnly"], check=False, capture_output=True,
                    text=True, timeout=60,
                )
                cleanup_confirmed = json.loads(cleanup.stdout.strip().splitlines()[-1]).get(
                    "cleanup"
                ) == "PASS"
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
        return payload


def publish_artifacts(
    staged_core: Path, core_output: Path, staged_evidence: Path, evidence_output: Path,
    *, replace: Callable[[Path, Path], None] | None = None,
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
        "CLEANUP": "WINDOWS_CLEANUP",
    }
    upper = message.upper()
    return next((result for label, result in labels.items() if f"[{label}]" in upper), SCM_ITEMS[0])


def _revision(explicit: str | None) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"], check=False,
        capture_output=True, text=True,
    )
    git_revision = completed.stdout.strip() if completed.returncode == 0 else ""
    value = explicit or git_revision
    if not value or value.lower() in {"unknown", "latest", "working-tree"}:
        raise WindowsAcceptanceError("SOURCE_REVISION_REQUIRED")
    if git_revision:
        if value != git_revision:
            raise WindowsAcceptanceError("SOURCE_REVISION_DOES_NOT_MATCH_CHECKOUT")
        status = subprocess.run(
            ["git", "status", "--porcelain"], check=False, capture_output=True, text=True,
        )
        if status.returncode or status.stdout.strip():
            raise WindowsAcceptanceError("SOURCE_REVISION_HAS_UNCOMMITTED_CHANGES")
    return value


def _identity(mode: str, revision: str | None) -> tuple[str, str, str]:
    if mode == "github":
        source_revision = revision or os.environ.get("GITHUB_SHA", "")
        if not source_revision or source_revision.lower() in {"unknown", "latest", "working-tree"}:
            raise WindowsAcceptanceError("SOURCE_REVISION_REQUIRED")
        if os.environ.get("GITHUB_SERVER_URL") != GITHUB_PROVIDER or not os.environ.get("GITHUB_RUN_ID"):
            raise WindowsAcceptanceError("GITHUB_RELEASE_IDENTITY_REQUIRED")
        if source_revision != os.environ.get("GITHUB_SHA"):
            raise WindowsAcceptanceError("SOURCE_REVISION_DOES_NOT_MATCH_GITHUB_SHA")
        return source_revision, GITHUB_PROVIDER, os.environ["GITHUB_RUN_ID"]
    source_revision = _revision(revision)
    return source_revision, LOCAL_PROVIDER, LOCAL_PROVIDER


def run_acceptance(
    *, mode: str, revision: str | None, core_output: Path, evidence_output: Path,
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
            live.run_core(source_revision, provider, run_id, core_temp)
        except Exception as exc:
            raise WindowsAcceptanceError(str(exc), "WINDOWS_CORE_PLAN") from exc
        scm = live.run_scm()
        results = [
            {"item": item, "status": "PASS", "evidence_class": "LIVE_WINDOWS_INTEGRATION",
             "test_or_probe": "deployment.windows_acceptance/windows_scm_probe.ps1",
             "details": scm.get("details", "reviewed SCM lifecycle and cleanup verified")}
            for item in SCM_ITEMS
        ]
        evidence_temp = Path(temporary) / "scm.json"
        evidence_temp.write_text(json.dumps(evidence_document(
            platform_name="WINDOWS", source_revision=source_revision, ci_provider=provider,
            ci_run_id=run_id, runner_os="Windows", runner_arch=os.environ.get(
                "PROCESSOR_ARCHITECTURE", "unknown"
            ), results=results,
        ), indent=2) + "\n", encoding="utf-8")
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
            mode=args.mode, revision=args.source_revision, core_output=args.core_output,
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
