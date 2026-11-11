"""Odczytowa inspekcja lub jawna próba na niezależnie zatwierdzonej VM TEST_ONLY.

Raport trafia wyłącznie na stdout. Harness nie tworzy kont, klastra, kluczy,
bindingu ani konfiguracji diagnostycznej; nie naprawia zastanych artefaktów.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import runpy
import stat
import subprocess
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any, cast

DEFAULT_APPROVAL = Path("/etc/cryptohunter/stage9-vm-trial-approval.json")
PHASES = ("runtime-verify", "runtime-publish", "preflight", "install", "systemd", "postgresql-ipc")
NEGATIVE_SCENARIOS = {
    "manifest": "podpis przed interpretacją manifestu",
    "signature": "niezależny autorytet Ed25519",
    "runtime-file": "pełny inwentarz runtime",
    "missing-package": "zamknięcie pakietów runtime",
    "extra-package": "zamknięcie pakietów runtime",
    "interpreter": "podpisany interpreter i zewnętrzny CPython",
    "symlink": "chronione ścieżki i dokładny inwentarz",
    "systemd-dropin": "efektywna konfiguracja systemd",
    "execstart": "dokładna jednostka i argumenty procesu",
    "uid-gid": "tożsamość NSS i działającego procesu",
    "permissions": "uprawnienia chronionych ścieżek",
    "process-replacement": "PIDFD, inode, hash i ponowna obserwacja",
    "postgresql-cluster": "tożsamość klastra i konfiguracja PostgreSQL",
    "postgresql-disconnect": "utrata rzeczywistego połączenia adaptera",
    "endpoint-replacement": "tożsamość endpointu IPC i PostgreSQL",
    "missing-authority": "zewnętrzny autorytet podpisu",
    "interrupted-publication": "publikacja bez nadpisania i ręczny recovery",
    "staging-journal": "odmowa przy częściowej publikacji",
    "foreign-deployment": "ochrona istniejących danych",
}
SNAPSHOT_PATHS = (
    Path("/etc/passwd"),
    Path("/etc/group"),
    Path("/etc/postgresql"),
    Path("/etc/cryptohunter"),
    Path("/etc/systemd/system"),
    Path("/usr/lib/cryptohunter"),
    Path("/var/lib/cryptohunter-installer"),
    Path("/opt/cryptohunter/freshness-runtime"),
)
STAGING_PARENT = Path("/usr/lib")
STAGING_PREFIX = ".cryptohunter-stage-"
COMMAND_ENV = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C", "LC_ALL": "C"}


class TrialError(RuntimeError):
    """Wyłącznie stały kod; bez treści plików i błędów poleceń hosta."""

    def __init__(self, reason: str, *, status: str = "FAIL") -> None:
        super().__init__(reason)
        self.status = status


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise TrialError(code)


def _protected(path: Path, *, directory: bool = False) -> None:
    _require(path.is_absolute() and ".." not in path.parts, "INPUT_PATH_INVALID")
    for component in (*reversed(path.parents), path):
        info = component.lstat()
        last = component == path
        _require(
            info.st_uid == 0
            and not info.st_mode & 0o022
            and not stat.S_ISLNK(info.st_mode)
            and (
                stat.S_ISDIR(info.st_mode) if not last or directory else stat.S_ISREG(info.st_mode)
            )
            and (not last or directory or info.st_nlink == 1),
            "INPUT_NOT_PROTECTED",
        )


def _read(path: Path, limit: int) -> bytes:
    _protected(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "INPUT_NOT_REGULAR")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        _require(len(raw) <= limit, "INPUT_TOO_LARGE")
        _require(_stable(path.lstat()) == _stable(info), "INPUT_CHANGED_DURING_READ")
        return raw
    finally:
        os.close(fd)


def _stable(info: os.stat_result) -> tuple[int, ...]:
    # Sam odczyt może zmienić atime; zapis i podmiana muszą być widoczne.
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _json(raw: bytes) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, "JSON_DUPLICATE_KEY")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=unique)
    except (ValueError, UnicodeError) as exc:
        raise TrialError("JSON_INVALID") from exc
    _require(type(value) is dict, "JSON_OBJECT_REQUIRED")
    return cast(dict[str, Any], value)


def load_inputs(path: Path) -> dict[str, dict[str, str]]:
    value = _json(_read(path, 65536))
    _require(set(value) == {"runtime", "application"}, "INPUT_SCHEMA_INVALID")
    for section, names in (
        ("runtime", {"runtime", "manifest", "signature", "trusted_key"}),
        ("application", {"bundle", "manifest", "signature", "trusted_key"}),
    ):
        record = value[section]
        _require(type(record) is dict and set(record) == names, "INPUT_SCHEMA_INVALID")
        for name, item in record.items():
            _require(type(item) is str and str(Path(item)) == item, "INPUT_PATH_INVALID")
            _protected(Path(item), directory=name in {"runtime", "bundle"})
    runtime, application = value["runtime"], value["application"]
    keys = [_read(Path(item["trusted_key"]), 32) for item in (runtime, application)]
    _require(all(len(key) == 32 for key in keys) and keys[0] != keys[1], "AUTHORITIES_NOT_SEPARATE")
    for key in (Path(runtime["trusted_key"]), Path(application["trusted_key"])):
        _require(
            not any(
                key.is_relative_to(Path(root))
                for root in (runtime["runtime"], application["bundle"])
            ),
            "AUTHORITY_INSIDE_PAYLOAD",
        )
    return value


def _helper(name: str) -> Path:
    path = Path(__file__).absolute().with_name(name)
    _protected(path)
    return path


def _environment_helpers(execute: bool) -> dict[str, Any]:
    path = Path(__file__).absolute().with_name("stage9_vm_environment.py")
    if execute:
        _protected(Path(__file__).absolute())
        _protected(path)
    # Nie odczytujemy sąsiedniego bytecode ani modułów z cwd/PYTHONPATH.
    sys.pycache_prefix = "/dev/null"
    return runpy.run_path(str(path))


def _command(arguments: list[str], *, timeout: int = 180) -> tuple[int, bytes]:
    try:
        result = subprocess.run(
            arguments,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd="/",
            env=COMMAND_ENV,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TrialError("COMMAND_UNAVAILABLE_OR_TIMEOUT", status="BLOCKED") from exc
    _require(len(result.stdout) <= 1024 * 1024, "COMMAND_REPLY_TOO_LARGE")
    return result.returncode, result.stdout


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        _require(stat.S_ISREG(info.st_mode), "SNAPSHOT_SPECIAL_FILE")
        _require(info.st_size <= 512 * 1024 * 1024, "SNAPSHOT_FILE_TOO_LARGE")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            for chunk in iter(lambda: stream.read(65536), b""):
                digest.update(chunk)
        _require(
            _stable(os.fstat(fd)) == _stable(info) and _stable(path.lstat()) == _stable(info),
            "SNAPSHOT_CHANGED_DURING_READ",
        )
    finally:
        os.close(fd)
    return digest.hexdigest()


def snapshot(paths: tuple[Path, ...] = SNAPSHOT_PATHS) -> str:
    """Hash treści/UID/GID/trybów/linków kontrolowanych drzew, bez atime i PGDATA.

    Obejmuje też staging instalatora obok katalogu kodu. Nie jest pełnym
    snapshotem hosta: pomija PGDATA, stan usług i zewnętrzne pliki runtime.
    Treści ani osobne hashe konfiguracji nie opuszczają tej funkcji.
    """
    digest = hashlib.sha256()
    count = 0

    def visit(path: Path) -> None:
        nonlocal count
        count += 1
        _require(count <= 100000, "SNAPSHOT_TOO_MANY_FILES")
        try:
            info = path.lstat()
        except FileNotFoundError:
            digest.update(json.dumps([str(path), "absent"]).encode())
            return
        record: list[Any] = [str(path), info.st_mode, info.st_uid, info.st_gid, info.st_nlink]
        if stat.S_ISREG(info.st_mode):
            record.append(_digest(path))
        elif stat.S_ISLNK(info.st_mode):
            record.append(os.readlink(path))
        elif not stat.S_ISDIR(info.st_mode):
            raise TrialError("SNAPSHOT_SPECIAL_FILE")
        digest.update(json.dumps(record, separators=(",", ":")).encode())
        if stat.S_ISDIR(info.st_mode):
            for child in sorted(path.iterdir()):
                visit(child)

    for path in paths:
        visit(path)
    _require(stat.S_ISDIR(STAGING_PARENT.lstat().st_mode), "SNAPSHOT_STAGING_PARENT_INVALID")
    digest.update(json.dumps([str(STAGING_PARENT), STAGING_PREFIX]).encode())
    for child in sorted(STAGING_PARENT.iterdir()):
        if child.name.startswith(STAGING_PREFIX):
            visit(child)
    return digest.hexdigest()


def _installer(
    inputs: dict[str, dict[str, str]], revision: str, *, install: bool
) -> dict[str, Any]:
    app = inputs["application"]
    arguments = [
        "/usr/bin/python3",
        "-I",
        "-s",
        "-B",
        str(_helper("production_local_installer.py")),
        "--install" if install else "--preflight",
    ]
    for name in ("bundle", "manifest", "signature", "trusted_key"):
        arguments.extend(["--" + name.replace("_", "-"), app[name]])
    arguments.extend(["--expected-revision", revision])
    code, raw = _command(arguments)
    reply = _json(raw)
    if code != 0:
        _require(
            reply.get("result") == "REFUSED" and reply.get("code") == code,
            "INSTALLER_REPLY_INVALID",
        )
        return {"status": "FAIL", "reason": "INSTALLER_REFUSED", "exit_code": code}
    _require(
        reply.get("result") == ("INSTALLED" if install else "READY")
        and reply.get("scope") == "cryptohunter.freshness.substrate.v2"
        and reply.get("production_qualification") == "WITHHELD"
        and reply.get("state") in ({"installed"} if install else {"fresh", "installed"}),
        "INSTALLER_REPLY_INVALID",
    )
    return {
        "status": "PASS",
        "reason": "INSTALLER_V2_VERIFIED",
        "state": reply["state"],
        "manifest_sha256": reply["manifest_sha256"],
    }


def checked_preflight(inputs: dict[str, dict[str, str]], revision: str) -> dict[str, Any]:
    before = snapshot()
    try:
        result = _installer(inputs, revision, install=False)
    finally:
        after = snapshot()
        _require(before == after, "PREFLIGHT_MUTATED_HOST")
    return {**result, "before_sha256": before, "after_sha256": after, "readonly": True}


def _runtime(
    inputs: dict[str, dict[str, str]], revision: str, environment_id: str, *, publish: bool
) -> dict[str, Any]:
    runtime = inputs["runtime"]
    arguments = [
        "/usr/bin/python3",
        "-I",
        "-s",
        "-B",
        str(_helper("freshness_runtime.py")),
        "publish" if publish else "verify",
    ]
    for name in ("runtime", "manifest", "signature", "trusted_key"):
        arguments.extend(["--" + name.replace("_", "-"), runtime[name]])
    arguments.extend(["--environment-id", environment_id, "--revision", revision])
    code, raw = _command(arguments, timeout=600)
    if code != 0:
        return {"status": "FAIL", "reason": "SIGNED_RUNTIME_REFUSED", "exit_code": code}
    result = _json(raw)
    _require(
        set(result) == {"environment_id", "manifest_sha256"}
        and result["environment_id"] == environment_id
        and re.fullmatch(r"[0-9a-f]{64}", str(result["manifest_sha256"])) is not None,
        "RUNTIME_REPLY_INVALID",
    )
    return {"status": "PASS", "reason": "SIGNED_RUNTIME_VERIFIED", **result}


def _native_module() -> Any:
    # Zaufany checkout jest odrębny od bundle aplikacji. Nie wykonujemy __init__.
    root = Path(__file__).absolute().parent.parent
    for package_name in ("deployment", "bot_core"):
        package_root = root / package_name
        _protected(package_root, directory=True)
        for path in package_root.glob("*.py"):
            _protected(path)
        _require(package_name not in sys.modules, "PACKAGE_PRELOADED")
        package = ModuleType(package_name)
        package.__package__ = package_name
        package.__path__ = [str(package_root)]
        package.__spec__ = ModuleSpec(package_name, loader=None, is_package=True)
        package.__spec__.submodule_search_locations = package.__path__
        sys.modules[package_name] = package
    return importlib.import_module("deployment.stage9_vm_native")


def _installed_executable(revision: str, environment_id: str) -> str:
    # Najpierw systemowy, chroniony bootstrap; nigdy import z ocenianego venv.
    code, _raw = _command(
        [
            "/usr/bin/python3",
            "-I",
            "-s",
            "-B",
            str(_helper("freshness_runtime_bootstrap.py")),
            "verify-installed",
        ],
        timeout=600,
    )
    _require(code == 0, "INSTALLED_RUNTIME_NOT_VERIFIED")
    binding = _json(_read(Path("/etc/cryptohunter/freshness-runtime.json"), 65536))
    executable = f"/opt/cryptohunter/freshness-runtime/releases/{environment_id}/bin/python3.12"
    _require(
        binding.get("source_revision") == revision
        and binding.get("environment_id") == environment_id
        and binding.get("executable") == executable,
        "INSTALLED_RUNTIME_BINDING_MISMATCH",
    )
    return executable


def _native_trial(args: argparse.Namespace, inputs: dict[str, dict[str, str]]) -> dict[str, Any]:
    executable = _installed_executable(args.revision, args.environment_id)
    if args.native_worker:
        _require(sys.executable == executable, "NATIVE_WORKER_INTERPRETER_NOT_VERIFIED")
        return execute_phase(args.phase, inputs, args.revision, args.environment_id)
    # Walidacja odpowiedzi PG importuje psycopg. Zapewnia go wyłącznie wcześniej
    # zweryfikowany runtime, bez modyfikacji systemowych site-packages.
    arguments = [
        executable,
        "-I",
        "-s",
        "-B",
        str(Path(__file__).absolute()),
        "--mode",
        "EXECUTE-VM-TRIAL",
        "--approval",
        str(args.approval),
        "--revision",
        args.revision,
        "--environment-id",
        args.environment_id,
        "--phase",
        args.phase,
        "--inputs",
        str(args.inputs),
        "--native-worker",
    ]
    code, raw = _command(arguments, timeout=600)
    report = _json(raw)
    _require(
        report.get("purpose") == "TEST_ONLY"
        and report.get("revision") == args.revision
        and report.get("environment_id") == args.environment_id
        and report.get("production_qualification") == "BLOCKED"
        and type(report.get("checks")) is dict,
        "NATIVE_WORKER_REPLY_INVALID",
    )
    result = report["checks"].get(args.phase)
    _require(
        type(result) is dict
        and result.get("status") in {"PASS", "FAIL", "BLOCKED"}
        and (code == 0) == (result["status"] == "PASS"),
        "NATIVE_WORKER_REPLY_INVALID",
    )
    return cast(dict[str, Any], result)


def new_report(mode: str, revision: str | None, environment_id: str | None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "purpose": "TEST_ONLY",
        "mode": mode,
        "revision": revision,
        "environment_id": environment_id,
        "native_trial": "PENDING",
        "production_qualification": "BLOCKED",
        "linux_physical_production_attestation": False,
        "stage9_production_ready": False,
        "checks": {phase: {"status": "NOT_EXECUTED"} for phase in PHASES},
        "negative_checks": {
            name: {"status": "NOT_EXECUTED", "trust_boundary": boundary}
            for name, boundary in NEGATIVE_SCENARIOS.items()
        },
    }


def summarize(checks: dict[str, Any]) -> str:
    statuses = {entry.get("status") for entry in checks.values()}
    _require(statuses <= {"PASS", "FAIL", "BLOCKED", "NOT_EXECUTED"}, "REPORT_STATUS_INVALID")
    if "FAIL" in statuses:
        return "FAIL"
    if "BLOCKED" in statuses:
        return "BLOCKED"
    if not statuses or "NOT_EXECUTED" in statuses:
        return "NOT_EXECUTED"
    return "PASS"


def execute_phase(
    phase: str, inputs: dict[str, dict[str, str]], revision: str, environment_id: str
) -> dict[str, Any]:
    if phase in {"runtime-verify", "runtime-publish"}:
        before = snapshot() if phase == "runtime-verify" else None
        try:
            result = _runtime(inputs, revision, environment_id, publish=phase == "runtime-publish")
        finally:
            if before is not None:
                _require(snapshot() == before, "RUNTIME_VERIFY_MUTATED_HOST")
        if before is not None:
            result["readonly"] = True
        return result
    if phase == "preflight":
        return checked_preflight(inputs, revision)
    if phase == "install":
        preflight = checked_preflight(inputs, revision)
        if preflight["status"] != "PASS":
            return {
                "status": "BLOCKED",
                "reason": "POSITIVE_PREFLIGHT_REQUIRED",
                "preflight": preflight,
            }
        result = _installer(inputs, revision, install=True)
        if result["status"] != "PASS":
            return result
        post = checked_preflight(inputs, revision)
        if (
            post["status"] != "PASS"
            and type(post.get("exit_code")) is int
            and post["exit_code"] == 21
        ):
            # Publikacja plików nie dowodzi aktywacji konfiguracji PostgreSQL.
            # Operator może ponowić tę samą fazę po niezależnie zatwierdzonej
            # aktywacji. Nie restartujemy usług ani nie zaliczamy idempotencji.
            return {
                **result,
                "status": "BLOCKED",
                "reason": "POST_INSTALL_PREFLIGHT_REFUSED",
                "postflight": post,
                "identical_reinstall": "NOT_EXECUTED",
            }
        _require(
            post["status"] == "PASS" and post["state"] == "installed", "POST_INSTALL_NOT_VERIFIED"
        )
        before = snapshot()
        try:
            repeat = _installer(inputs, revision, install=True)
        finally:
            _require(snapshot() == before, "IDENTICAL_REINSTALL_MUTATED_HOST")
        _require(
            repeat["status"] == "PASS" and repeat["state"] == "installed",
            "IDENTICAL_REINSTALL_REFUSED",
        )
        return {**result, "postflight": post, "identical_reinstall": "PASS"}
    _require(phase in {"systemd", "postgresql-ipc"}, "PHASE_INVALID")
    preflight = checked_preflight(inputs, revision)
    if preflight["status"] != "PASS" or preflight.get("state") != "installed":
        return {
            "status": "BLOCKED",
            "reason": "VERIFIED_INSTALLATION_REQUIRED",
            "preflight": preflight,
        }
    native = _native_module()
    return cast(
        dict[str, Any],
        (
            native.observe_systemd(start=True)
            if phase == "systemd"
            else native.observe_postgresql_ipc()
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("INSPECT", "EXECUTE-VM-TRIAL"))
    parser.add_argument("--approval", type=Path, default=DEFAULT_APPROVAL)
    parser.add_argument("--revision")
    parser.add_argument("--environment-id")
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--native-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    report = new_report(args.mode, args.revision, args.environment_id)
    code = 0
    try:
        execute = args.mode == "EXECUTE-VM-TRIAL"
        helpers = _environment_helpers(execute)
        if not execute:
            _require(
                args.phase is None and args.inputs is None and not args.native_worker,
                "INSPECT_EXECUTION_ARGUMENTS_REFUSED",
            )
            report["environment"] = helpers["inspect_environment"]()
            report["result"] = "NOT_EXECUTED"
        else:
            _require(
                args.phase is not None
                and args.inputs is not None
                and args.revision is not None
                and args.environment_id is not None,
                "EXECUTION_ARGUMENTS_REQUIRED",
            )
            report["vm_approval"] = helpers["require_approved_vm"](
                args.approval,
                revision=args.revision,
                environment_id=args.environment_id,
            )
            inputs = load_inputs(args.inputs)
            report["manifest_sha256"] = {
                name: hashlib.sha256(_read(Path(record["manifest"]), 32 * 1024 * 1024)).hexdigest()
                for name, record in inputs.items()
            }
            report["authority_sha256"] = {
                name: hashlib.sha256(_read(Path(record["trusted_key"]), 32)).hexdigest()
                for name, record in inputs.items()
            }
            if args.phase in {"systemd", "postgresql-ipc"}:
                report["checks"][args.phase] = _native_trial(args, inputs)
            else:
                _require(not args.native_worker, "NATIVE_WORKER_PHASE_INVALID")
                report["checks"][args.phase] = execute_phase(
                    args.phase, inputs, args.revision, args.environment_id
                )
            report["result"] = summarize(report["checks"])
            code = 1 if report["checks"][args.phase]["status"] in {"FAIL", "BLOCKED"} else 0
    except Exception as exc:
        # Nie serializujemy obcych wyjątków, stderr, DSN ani treści plików.
        status = exc.status if isinstance(exc, TrialError) else "BLOCKED"
        report["result"] = status
        report["reason"] = str(exc) if isinstance(exc, TrialError) else "VM_TRIAL_REFUSED"
        if args.phase is not None:
            report["checks"][args.phase] = {"status": status, "reason": report["reason"]}
        code = 2
    print(json.dumps(report, sort_keys=True, ensure_ascii=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
