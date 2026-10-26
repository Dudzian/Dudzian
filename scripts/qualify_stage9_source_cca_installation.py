"""Explicit, offline TEST ONLY operator; never a production admission command.

Only --execute can modify the host. Do not invoke it in CI or Codex. The
unprivileged modes and orchestration tests provide no physical qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
from typing import Any, Callable

from deployment.stage9_source_cca_test_package import (
    HarnessRefusal,
    build_package,
    plan,
    validate_package,
)
from deployment.stage9_source_cca_test_runtime import PhysicalRun, inspect_cleanup, preflight
from deployment.stage9_trusted_installation_identity_gate import inspect_installation

SCHEMA = "CryptoHunter.SourceCCAPhysicalTestReportV1"
PRODUCTION_BLOCKERS = [
    "INDEPENDENT_PRODUCTION_MANIFEST_APPROVAL_MISSING",
    "HOST_ATTESTATION_NOT_PERFORMED",
    "SOURCE_CCA_PRODUCTION_EXECUTION_AND_DURABILITY_NOT_QUALIFIED",
    "STAGE9_IN_PROGRESS_PRODUCTION_NO_GO",
]


def _bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _commit(*, clean: bool = False) -> str:
    """Bind a run to reviewed committed code, without executing repository hooks."""
    repository = Path(__file__).resolve().parents[1]
    try:
        answer = subprocess.run(
            (
                "/usr/bin/git",
                "-c",
                "core.fsmonitor=false",
                "-C",
                str(repository),
                "rev-parse",
                "HEAD",
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        if not re.fullmatch(r"[0-9a-f]{40}", answer):
            raise HarnessRefusal("COMMIT_UNAVAILABLE")
        if clean:
            changes = subprocess.run(
                (
                    "/usr/bin/git",
                    "-c",
                    "core.fsmonitor=false",
                    "-C",
                    str(repository),
                    "status",
                    "--porcelain",
                ),
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
            if changes:
                raise HarnessRefusal("CLEAN_REVIEWED_CHECKOUT_REQUIRED")
        return answer
    except (OSError, subprocess.SubprocessError) as exc:
        raise HarnessRefusal("COMMIT_UNAVAILABLE") from exc


def _report(mode: str, run_id: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "version": 1,
        "commit": None,
        "run_id": run_id,
        "test_only": "TEST_ONLY",
        "stage9": "IN_PROGRESS / PRODUCTION_NO_GO",
        "status": "BLOCKED",
        "qualification": "NOT_QUALIFIED",
        "production_emission": "BLOCKED",
        "mode": mode,
        "physical_result": "NOT_EXECUTED",
        "physical_test_status": "PHYSICAL_TEST_PENDING",
        "preflight": None,
        "checks": [],
        "gate": None,
        "refusals": [],
        "cleanup": {"clean": False, "state": "NOT_EXECUTED"},
        "production_blockers": PRODUCTION_BLOCKERS.copy(),
    }


class _ReportDestination:
    """Pin the directory chain and reserve evidence files before host mutation."""

    def __init__(self, path: Path):
        self.directories: list[tuple[int | None, str, int, tuple[int, ...]]] = []
        self.outputs: list[tuple[str, int, tuple[int, ...]]] = []
        path = Path(os.path.abspath(path))
        self.parent = -1
        try:
            parent: int | None = None
            for name in ("/", *path.parts[1:-1]):
                before = os.stat(name, dir_fd=parent, follow_symlinks=False)
                if not stat.S_ISDIR(before.st_mode):
                    raise HarnessRefusal("REPORT_PATH_UNSAFE")
                descriptor = os.open(
                    name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent
                )
                self.directories.append((parent, name, descriptor, self._identity(before)))
                if self._identity(os.fstat(descriptor)) != self._identity(before):
                    raise HarnessRefusal("REPORT_PATH_UNSAFE")
                parent = descriptor
            assert parent is not None
            self.parent = parent
            for name in (path.name, path.name + ".sha256"):
                descriptor = os.open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                    0o600,
                    dir_fd=self.parent,
                )
                self.outputs.append((name, descriptor, self._identity(os.fstat(descriptor))))
            self._recheck()
        except BaseException as exc:
            self.close()
            if isinstance(exc, OSError):
                raise HarnessRefusal("REPORT_WRITE_BLOCKED") from None
            raise

    @staticmethod
    def _identity(info: os.stat_result) -> tuple[int, ...]:
        return info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid

    def _recheck(self) -> None:
        for parent, name, descriptor, expected in self.directories:
            if (
                self._identity(os.fstat(descriptor)) != expected
                or self._identity(os.stat(name, dir_fd=parent, follow_symlinks=False)) != expected
            ):
                raise HarnessRefusal("REPORT_PATH_UNSAFE")
        for name, descriptor, expected in self.outputs:
            current = os.fstat(descriptor)
            if (
                self._identity(current) != expected
                or current.st_nlink != 1
                or self._identity(os.stat(name, dir_fd=self.parent, follow_symlinks=False))
                != expected
            ):
                raise HarnessRefusal("REPORT_PATH_UNSAFE")

    def write(self, report: dict[str, Any]) -> None:
        self._recheck()
        raw = _bytes(report)
        checksum = (hashlib.sha256(raw).hexdigest() + "\n").encode("ascii")
        for (_, descriptor, _), payload in zip(self.outputs, (raw, checksum), strict=True):
            with os.fdopen(os.dup(descriptor), "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        self._recheck()
        os.fsync(self.parent)

    def close(self) -> None:
        for _, descriptor, _ in reversed(self.outputs):
            os.close(descriptor)
        self.outputs.clear()
        for _, _, descriptor, _ in reversed(self.directories):
            os.close(descriptor)
        self.directories.clear()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    destination = _ReportDestination(path)
    try:
        destination.write(report)
    finally:
        destination.close()


def _matched(result: dict[str, Any]) -> bool:
    return (
        result.get("installation_checks") == "MATCHED"
        and set(result.get("role_checks", {})) == {"root_source", "history_source", "cca"}
        and all(value.get("checks") == "MATCHED" for value in result["role_checks"].values())
        and result.get("status") == "BLOCKED"
        and result.get("qualification") == "NOT_QUALIFIED"
        and result.get("production_emission") == "BLOCKED"
        and result.get("observed_failures") == []
    )


def _operator_code_locations() -> tuple[set[Path], set[Path]]:
    """Inventory unloaded code too; sys.modules is not an import closure."""
    repository = Path(__file__).resolve().parents[1]
    required = {
        repository,
        repository / ".git",
        Path(__file__),
        Path(sys.executable).resolve(strict=True),
        repository / "deployment/__init__.py",
        repository / "scripts/__init__.py",
        repository / "deployment/stage9_source_cca_test_controls.py",
    }
    searches = {Path(entry or os.getcwd()) for entry in sys.path}
    for module in tuple(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if isinstance(source, str) and not source.startswith("<"):
            required.add(Path(source))
        # Namespace/package imports can search locations outside sys.path.
        for location in getattr(module, "__path__", ()):
            searches.add(Path(location))
    return required, searches


def _trusted_operator() -> None:
    """Fresh protection checks over present and future import locations.

    This process and its startup imports must already be independently trusted.
    Metadata checks cannot authenticate Python code that has already executed,
    existing bytecode, import hooks, or the checking process itself. The external
    pre-interpreter verification prerequisite is never discharged here.
    """
    from scripts.qualify_stage41b_signer_isolation import _check_protected

    try:
        required, searches = _operator_code_locations()
        if any(".." in path.parts for path in required | searches):
            raise HarnessRefusal("PROTECTED_REVIEWED_OPERATOR_REQUIRED")
        required = {path.absolute() for path in required}
        searches = {path.absolute() for path in searches}
        visited: set[Path] = set()

        def check_ancestors(location: Path) -> None:
            for directory in reversed(location.parents):
                _check_protected(directory, directory=True)

        for location in required | searches:
            try:
                location.lstat()
            except FileNotFoundError:
                if location in required:
                    raise
                # A stock Python search path may include an absent stdlib zip.
                # Protect every existing ancestor; an absent user-writable
                # import root must not be populated after this check.
                for directory in reversed(location.parents):
                    try:
                        directory.lstat()
                    except FileNotFoundError:
                        continue
                    _check_protected(directory, directory=True)
                continue
            check_ancestors(location)
            pending = [location]
            while pending:
                current = pending.pop()
                if current in visited:
                    continue
                # lstat + _check_protected reject links before any descent.
                # Check every file, including unloaded .py, .pyc and .pth.
                is_directory = stat.S_ISDIR(current.lstat().st_mode)
                _check_protected(current, directory=is_directory)
                visited.add(current)
                if is_directory:
                    pending.extend(current.iterdir())
    except Exception:
        raise HarnessRefusal("PROTECTED_REVIEWED_OPERATOR_REQUIRED") from None


def _prepare_privileged_controls() -> Callable[..., Any]:
    """Load known late code before resource creation, after tree protection.

    The second check is consistency after a trusted import, not authentication
    of import-time code. Trusted Python startup remains an external condition.
    """
    try:
        _trusted_operator()
        from deployment import stage9_source_cca_test_controls as controls

        expected = (
            Path(__file__).resolve().parents[1] / "deployment/stage9_source_cca_test_controls.py"
        )
        if Path(controls.__file__) != expected:
            raise HarnessRefusal("PROTECTED_REVIEWED_OPERATOR_REQUIRED")
        _trusted_operator()
        return controls.run_controls
    except Exception:
        raise HarnessRefusal("PROTECTED_REVIEWED_OPERATOR_REQUIRED") from None


def _authorize(args: argparse.Namespace) -> None:
    if args.authorize_test_only != args.run_id or args.physical_host != "Dell-Latitude-5490":
        raise HarnessRefusal("EXPLICIT_PHYSICAL_TEST_AUTHORIZATION_REQUIRED")
    if not args.package_sha256 or not re.fullmatch(r"[0-9a-f]{64}", args.package_sha256):
        raise HarnessRefusal("REVIEWED_PACKAGE_SHA256_REQUIRED")
    if any(
        os.environ.get(name)
        for name in (
            "CI",
            "GITHUB_ACTIONS",
            "CODEX_THREAD_ID",
            "CODEX_CLOUD",
            "CODEX_SANDBOX",
            "CODEX_CI",
        )
    ):
        raise HarnessRefusal("CI_OR_CODEX_EXECUTION_FORBIDDEN")
    if sys.platform != "linux" or os.geteuid() != 0:
        raise HarnessRefusal("PHYSICAL_ROOT_EXECUTION_REQUIRED")
    _trusted_operator()


def _interrupted(_number: int, _frame: Any) -> None:
    raise HarnessRefusal("OPERATOR_INTERRUPTED")


def _require_administrative_inspection() -> None:
    """Explicit read-only inspection never escalates the invoking account."""
    if sys.platform != "linux":
        raise HarnessRefusal("NATIVE_LINUX_REQUIRED")
    if os.geteuid() != 0:
        raise HarnessRefusal("ADMINISTRATIVE_INSPECTION_REQUIRES_ROOT")


def _execute(args: argparse.Namespace, report: dict[str, Any]) -> None:
    if sys.platform != "linux":
        raise HarnessRefusal("PHYSICAL_ROOT_EXECUTION_REQUIRED")
    _authorize(args)
    run_controls = _prepare_privileged_controls()
    package = validate_package(args.package, args.package_sha256)
    if package["plan"] != plan(args.run_id, args.uid_base):
        raise HarnessRefusal("PACKAGE_RESOURCE_PLAN_MISMATCH")
    if package["commit"] != report["commit"]:
        raise HarnessRefusal("PACKAGE_COMMIT_MISMATCH")
    report["package_sha256"] = package["package_sha256"]
    report["preflight"] = preflight(package["plan"], administrative=True)
    if not report["preflight"]["ready"]:
        raise HarnessRefusal("PREFLIGHT_BLOCKED")
    # Recheck metadata even when contents/commit are unchanged. No resource may
    # be created if code protection changed during package/preflight reads.
    _trusted_operator()
    previous = {
        number: signal.getsignal(number)
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    run = PhysicalRun(package)
    try:
        for number in previous:
            signal.signal(number, _interrupted)
        # __enter__ also cleans partial installation. The final cleanup below
        # covers failures before entry and independently checks its outcome.
        with run:
            run.start()
            report["physical_result"] = "STARTED_TEST_ONLY"
            report["gate"] = inspect_installation(
                str(run.manifest_path), approval_pin_path=str(run.pin_path)
            )
            if not _matched(report["gate"]):
                raise HarnessRefusal("PHYSICAL_INSTALLATION_COMPARISON_REFUSED")
            _trusted_operator()
            report["checks"] = run_controls(run)
            report["physical_result"] = "TEST_INSTALLATION_MATCHED"
    finally:
        # Defer repeated interruptions until the cleanup state is published.
        from scripts.qualify_stage41b_signer_isolation import _defer_interrupts

        with _defer_interrupts():
            try:
                cleanup = run.cleanup()
            except (Exception, KeyboardInterrupt):
                cleanup = {"clean": False, "blockers": ["CLEANUP_OPERATION_FAILED"]}
            try:
                independent = inspect_cleanup(package["plan"], administrative=True)
            except (Exception, KeyboardInterrupt):
                independent = {"clean": False, "blockers": ["CLEANUP_INSPECTION_FAILED"]}
            report["cleanup"] = {
                "execution": cleanup,
                "independent": independent,
                "clean": cleanup.get("clean") is True
                and independent.get("clean") is True
                and independent.get("full_confirmation") is True,
            }
            for number, handler in previous.items():
                signal.signal(number, handler)
        if not report["cleanup"]["clean"]:
            report["physical_result"] = "BLOCKED"
            raise HarnessRefusal("CLEANUP_BLOCKED")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("check", "prepare", "execute", "inspect-cleanup", "inspect-cleanup-admin"):
        modes.add_argument("--" + name, action="store_true")
    parser.add_argument(
        "--run-id", required=True, help="exactly 12 lowercase hexadecimal characters"
    )
    parser.add_argument("--uid-base", type=int, default=62000)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--package-sha256")
    parser.add_argument("--authorize-test-only")
    parser.add_argument("--physical-host")
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mode = next(
        name
        for name in ("check", "prepare", "execute", "inspect_cleanup", "inspect_cleanup_admin")
        if getattr(args, name)
    )
    report = _report(mode, args.run_id)
    exit_code = 2
    destination: _ReportDestination | None = None
    try:
        resources = plan(args.run_id, args.uid_base)
        report["commit"] = _commit(clean=args.prepare or args.execute)
        if args.execute:
            if args.package is None or args.report is None:
                raise HarnessRefusal("PACKAGE_AND_REPORT_REQUIRED")
            _authorize(args)
        if args.inspect_cleanup_admin:
            _require_administrative_inspection()
        if args.report is not None:
            destination = _ReportDestination(args.report)
        if args.execute:
            _execute(args, report)
        elif args.prepare:
            if args.package is None:
                raise HarnessRefusal("PACKAGE_REQUIRED")
            package = build_package(args.package, args.run_id, args.uid_base, report["commit"])
            report["package"] = {
                "package_sha256": package["package_sha256"],
                "inventory": "inventory.json",
                "files_sha256": {
                    name: hashlib.sha256(raw).hexdigest() for name, raw in package["files"].items()
                },
            }
            exit_code = 0
        elif args.check:
            report["preflight"] = preflight(resources, administrative=False)
            exit_code = 0 if report["preflight"]["ready"] else 2
        else:
            report["cleanup"] = inspect_cleanup(
                resources, administrative=args.inspect_cleanup_admin
            )
            if args.inspect_cleanup_admin:
                complete = (
                    report["cleanup"].get("clean") is True
                    and report["cleanup"].get("full_confirmation") is True
                )
            else:
                # A successful rootless inspection is scoped evidence, never
                # a substitute for the administrative absence confirmation.
                complete = report["cleanup"].get("scope_clear") is True
            exit_code = 0 if complete else 2
    except HarnessRefusal as exc:
        report["refusals"].append(exc.code)
    except (Exception, KeyboardInterrupt):
        # Never serialize native exception text or arbitrary backend values.
        report["refusals"].append("OPERATOR_OPERATION_BLOCKED")
    report["evidence_sha256"] = {
        field: hashlib.sha256(_bytes(report[field])).hexdigest()
        for field in ("preflight", "checks", "gate", "cleanup")
    }
    if destination is not None:
        try:
            destination.write(report)
        except (OSError, HarnessRefusal):
            report["refusals"].append("REPORT_WRITE_BLOCKED")
            exit_code = 2
        finally:
            destination.close()
    print(_bytes(report).decode("utf-8"), end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
