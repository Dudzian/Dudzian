"""Odczytowa inspekcja lub jawna próba na niezależnie zatwierdzonej VM TEST_ONLY.

Raport trafia wyłącznie na stdout. Harness nie tworzy kont, klastra, kluczy,
bindingu ani konfiguracji diagnostycznej; nie naprawia zastanych artefaktów.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
from importlib.machinery import BuiltinImporter, FileFinder, FrozenImporter, ModuleSpec, PathFinder
from pathlib import Path
from types import CodeType, FunctionType, ModuleType
from typing import Any, cast
from zipimport import zipimporter

HARNESS_INVENTORY = Path("/etc/cryptohunter/stage9-harness-inventory.json")
# Reviewed transitive graph, including delayed and TYPE_CHECKING imports.
HARNESS_MODULES = frozenset(
    {
        "deployment.stage9_vm_native",
        "deployment.freshness_runtime",
        "deployment.freshness_runtime_bootstrap",
        "deployment.production_local_installer_host",
        "deployment.linux_physical_ipc",
        "deployment.linux_physical_native",
        "deployment.linux_physical_observations",
        "deployment.linux_physical_postgresql",
        "deployment.linux_release_identity",
        "deployment.linux_physical_evidence",
        "bot_core.freshness_service_identity",
        "bot_core.linux_service_ipc",
        "bot_core.linux_service_postgresql",
        "bot_core.production_signer_ipc",
    }
)
HARNESS_HELPERS = frozenset(
    {
        "deployment/stage9_vm_trial.py",
        "deployment/stage9_vm_environment.py",
        "deployment/production_local_installer.py",
        "deployment/freshness_application.py",
    }
)
EXTERNAL_IMPORTS = frozenset(
    {
        "__future__",
        "argparse",
        "ast",
        "base64",
        "binascii",
        "collections",
        "contextlib",
        "cryptography",
        "ctypes",
        "dataclasses",
        "datetime",
        "email",
        "fcntl",
        "grp",
        "hashlib",
        "importlib",
        "json",
        "math",
        "os",
        "pathlib",
        "platform",
        "psycopg",
        "pwd",
        "re",
        "runpy",
        "secrets",
        "select",
        "signal",
        "socket",
        "sqlite3",
        "stat",
        "struct",
        "subprocess",
        "sys",
        "time",
        "types",
        "typing",
        "zipimport",
    }
)

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
    before = path.lstat()
    _protected(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "INPUT_NOT_REGULAR")
        _require(_stable(before) == _stable(info), "INPUT_CHANGED_DURING_READ")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        _require(len(raw) <= limit, "INPUT_TOO_LARGE")
        _require(
            _stable(os.fstat(fd)) == _stable(info) and _stable(path.lstat()) == _stable(info),
            "INPUT_CHANGED_DURING_READ",
        )
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
    sources = _harness_sources()
    _require(path.relative_to(path.parent.parent).as_posix() in sources, "HELPER_NOT_APPROVED")
    return path


def _environment_helpers(execute: bool) -> dict[str, Any]:
    path = Path(__file__).absolute().with_name("stage9_vm_environment.py")
    sources = _harness_sources()
    name = "_stage9_vm_environment"
    loader = _SourceSnapshotLoader(
        {name: (str(path), sources["deployment/stage9_vm_environment.py"])}
    )
    spec = importlib.util.spec_from_loader(name, loader)
    _require(spec is not None, "HELPER_NOT_APPROVED")
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module.__dict__


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


def _project(name: str) -> bool:
    return name.split(".", 1)[0] in {"deployment", "bot_core"}


def _check_graph(sources: dict[str, bytes]) -> None:
    helpers = {name.removesuffix(".py").replace("/", ".") for name in HARNESS_HELPERS}
    allowed = HARNESS_MODULES | helpers
    reached: set[str] = set()
    pending = ["deployment.stage9_vm_native", *sorted(helpers)]
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        _require(name in allowed, "HARNESS_IMPORT_NOT_APPROVED")
        reached.add(name)
        tree = ast.parse(sources[name.replace(".", "/") + ".py"])
        for node in ast.walk(tree):
            imports: list[str] = []
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                _require(node.level == 0 and node.module is not None, "HARNESS_RELATIVE_IMPORT")
                imports = [node.module or ""]
                if node.module in {"deployment", "bot_core"}:
                    imports = [node.module + "." + alias.name for alias in node.names]
            elif isinstance(node, ast.Call):
                function = node.func
                reviewed_loader_call = name == "deployment.stage9_vm_trial" and (
                    (
                        isinstance(function, ast.Attribute)
                        and function.attr == "import_module"
                        and isinstance(function.value, ast.Name)
                        and function.value.id == "importlib"
                    )
                    or (
                        isinstance(function, ast.Attribute)
                        and function.attr == "exec_module"
                        and isinstance(function.value, ast.Name)
                        and function.value.id == "loader"
                    )
                )
                _require(
                    reviewed_loader_call
                    or (
                        not (
                            isinstance(function, ast.Name)
                            and function.id in {"__import__", "eval", "exec"}
                        )
                        and not (
                            isinstance(function, ast.Attribute)
                            and function.attr in {"import_module", "exec_module", "load_module"}
                        )
                    ),
                    "HARNESS_DYNAMIC_IMPORT_REFUSED",
                )
            for dependency in imports:
                if _project(dependency):
                    _require(
                        dependency in (allowed if name in helpers else HARNESS_MODULES),
                        "HARNESS_IMPORT_NOT_APPROVED",
                    )
                    pending.append(dependency)
                else:
                    _require(
                        dependency.split(".", 1)[0] in EXTERNAL_IMPORTS,
                        "HARNESS_DEPENDENCY_NOT_APPROVED",
                    )
    _require(reached == allowed, "HARNESS_GRAPH_CHANGED")


def _harness_sources(revision: str | None = None) -> dict[str, bytes]:
    """Authenticate exact inventory before executing any helper; never discover code."""
    root = Path(__file__).absolute().parent.parent
    _protected(root, directory=True)
    document = _json(_read(HARNESS_INVENTORY, 1024 * 1024))
    _require(
        set(document) == {"schema", "source_revision", "tree_root", "objects"}
        and type(document["schema"]) is int
        and document["schema"] == 1
        and type(document["source_revision"]) is str
        and re.fullmatch(r"[0-9a-f]{40}", document["source_revision"]) is not None
        and document["tree_root"] == str(root)
        and (revision is None or document["source_revision"] == revision)
        and type(document["objects"]) is list,
        "HARNESS_INVENTORY_INVALID",
    )
    required = {name.replace(".", "/") + ".py" for name in HARNESS_MODULES} | set(HARNESS_HELPERS)
    directories = {"."} | {str(parent) for name in required for parent in Path(name).parents}
    expected: dict[str, dict[str, Any]] = {}
    for record in document["objects"]:
        _require(type(record) is dict and type(record.get("path")) is str, "HARNESS_RECORD_INVALID")
        name = record["path"]
        _require(name not in expected, "HARNESS_RECORD_DUPLICATE")
        kind = "file" if name in required else "directory" if name in directories else None
        _require(
            kind is not None
            and record.get("kind") == kind
            and set(record)
            == ({"path", "kind", "mode", "sha256"} if kind == "file" else {"path", "kind", "mode"})
            and type(record["mode"]) is int
            and record["mode"] == (0o444 if kind == "file" else 0o555),
            "HARNESS_OBJECT_NOT_APPROVED",
        )
        if kind == "file":
            _require(
                type(record["sha256"]) is str
                and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None,
                "HARNESS_DIGEST_INVALID",
            )
        expected[name] = record
    _require(set(expected) == required | directories, "HARNESS_INVENTORY_INCOMPLETE")
    observed: set[str] = set()
    sources: dict[str, bytes] = {}

    def visit(path: Path, name: str) -> None:
        _require(name in expected, "HARNESS_EXTRA_OBJECT")
        record = expected[name]
        _protected(path, directory=record["kind"] == "directory")
        info = path.lstat()
        _require(stat.S_IMODE(info.st_mode) == record["mode"], "HARNESS_MODE_CHANGED")
        observed.add(name)
        if record["kind"] == "directory":
            for child in sorted(path.iterdir()):
                visit(child, child.relative_to(root).as_posix())
        else:
            raw = _read(path, 8 * 1024 * 1024)
            _require(
                hashlib.sha256(raw).hexdigest() == record["sha256"], "HARNESS_INTEGRITY_CHANGED"
            )
            sources[name] = raw
        _require(_stable(path.lstat()) == _stable(info), "HARNESS_CHANGED_DURING_READ")

    visit(root, ".")
    _require(observed == set(expected), "HARNESS_INVENTORY_INCOMPLETE")
    _check_graph(sources)
    return sources


def _import_state() -> None:
    _require(
        bool(sys.flags.isolated and sys.flags.no_user_site and sys.dont_write_bytecode),
        "HARNESS_ISOLATED_PROCESS_REQUIRED",
    )
    _require(
        type(sys.modules) is dict and all(type(name) is str for name in sys.modules),
        "IMPORT_CACHE_NOT_APPROVED",
    )
    _require(
        type(sys.meta_path) is list
        and type(sys.path_hooks) is list
        and type(sys.path_importer_cache) is dict
        and type(sys.path) is list
        and all(type(item) is str for item in sys.path)
        and all(type(item) is str for item in sys.path_importer_cache),
        "IMPORT_CACHE_NOT_APPROVED",
    )
    _require(not any(_project(name) for name in sys.modules), "PACKAGE_PRELOADED")
    _require(
        all(
            name == "__main__" or name.split(".", 1)[0] in sys.stdlib_module_names
            for name in sys.modules
        ),
        "EXTERNAL_MODULE_PRELOADED",
    )
    for name, module in sys.modules.items():
        if name in {"typing.io", "typing.re"}:
            typing_module = sys.modules.get("typing")
            _require(
                type(typing_module) is ModuleType
                and module is typing_module.__dict__.get(name.split(".")[1]),
                "IMPORT_CACHE_NOT_APPROVED",
            )
        else:
            # Do not inspect arbitrary cached objects through __getattr__ or
            # subclass callbacks before their type has been accepted.
            _require(type(module) is ModuleType, "IMPORT_CACHE_NOT_APPROVED")
    _require(
        len(sys.meta_path) == 3
        and all(
            actual is expected
            for actual, expected in zip(
                sys.meta_path, (BuiltinImporter, FrozenImporter, PathFinder), strict=True
            )
        ),
        "IMPORT_HOOK_NOT_APPROVED",
    )
    for finder in sys.path_importer_cache.values():
        _require(
            finder is None or type(finder) is FileFinder or type(finder) is zipimporter,
            "IMPORT_CACHE_NOT_APPROVED",
        )
    safe_hook = FileFinder.path_hook(
        (importlib.machinery.ExtensionFileLoader, importlib.machinery.EXTENSION_SUFFIXES),
        (importlib.machinery.SourceFileLoader, importlib.machinery.SOURCE_SUFFIXES),
    )
    for hook in sys.path_hooks:
        _require(
            hook is zipimporter
            or (type(hook) is FunctionType and hook.__code__ is safe_hook.__code__),
            "IMPORT_HOOK_NOT_APPROVED",
        )
    # Do not call arbitrary cache invalidators or hooks. Project imports never use these.
    sys.path_importer_cache.clear()
    sys.path_hooks[:] = [safe_hook]


class _SourceSnapshotLoader(importlib.abc.SourceLoader):
    """Import only authenticated in-memory source, without filesystem or bytecode fallback.

    Callers authenticate the inventory or signed runtime before constructing a
    snapshot. The standard importlib execution lifecycle owns compilation and
    module execution; these hooks bind a module name to its exact source bytes.
    Overriding get_code is deliberate: SourceLoader's default may consult .pyc.
    """

    def __init__(self, snapshots: dict[str, tuple[str, bytes]]) -> None:
        self._snapshots = dict(snapshots)

    def get_filename(self, fullname: str) -> str:
        _require(fullname in self._snapshots, "SOURCE_MODULE_NOT_APPROVED")
        return self._snapshots[fullname][0]

    def get_data(self, path: str) -> bytes:
        for filename, source in self._snapshots.values():
            if path == filename:
                return source
        raise TrialError("SOURCE_PATH_NOT_APPROVED")

    def get_code(self, fullname: str) -> CodeType:
        filename = self.get_filename(fullname)
        return self.source_to_code(self.get_data(filename), filename)


class _HarnessLoader(_SourceSnapshotLoader):
    """Project sources use authenticated bytes; external modules use signed runtime objects."""

    def __init__(self, sources: dict[str, bytes], root: Path) -> None:
        snapshots = {
            name: (str(root / (name.replace(".", "/") + ".py")), sources[relative])
            for name in HARNESS_MODULES
            if (relative := name.replace(".", "/") + ".py") in sources
        }
        # Namespace packages have no on-disk __init__, and no package search path.
        snapshots.update(
            {name: (str(root / name / "__init__.py"), b"") for name in ("deployment", "bot_core")}
        )
        super().__init__(snapshots)
        self.sources = dict(sources)
        self.root = root
        self.runtime: dict[str, dict[str, Any]] | None = None
        # Linux CPython's host stdlib path comes from the trusted interpreter.
        # Do not let sysconfig lazily import an environment-selected data module.
        self.bootstrap_stdlib = (
            Path(sys.base_prefix)
            / sys.platlibdir
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
        )

    def bind_runtime(self, manifest: dict[str, Any]) -> None:
        self.runtime = {
            str(Path(manifest["installation_path"]) / name): record
            for name, record in manifest["files"].items()
        } | manifest["external_files"]
        # Initial Python/stdlib execution belongs to the independently trusted
        # bootstrap. Backing-file checks cannot attest a compromised interpreter.
        for name, module in tuple(sys.modules.items()):
            if name == "__main__":
                continue
            if name in {"typing.io", "typing.re"}:
                # CPython 3.11 caches these deprecated aliases as classes,
                # rather than modules with specs. Bind them to trusted typing.
                typing_module = sys.modules["typing"]
                _require(
                    type(typing_module) is ModuleType
                    and module is typing_module.__dict__[name.split(".")[1]],
                    "RUNTIME_MODULE_NOT_APPROVED",
                )
                continue
            if _project(name):
                _require(getattr(module, "__loader__", None) is self, "PACKAGE_CACHE_CHANGED")
                continue
            spec = getattr(module, "__spec__", None)
            _require(spec is not None, "RUNTIME_MODULE_NOT_APPROVED")
            assert spec is not None
            if spec.origin in {"built-in", "frozen"}:
                continue
            _require(type(spec.origin) is str, "RUNTIME_MODULE_NOT_APPROVED")
            self._runtime_bytes(Path(spec.origin))

    def _runtime_bytes(self, origin: Path) -> bytes:
        assert self.runtime is not None
        record = self.runtime.get(str(origin))
        _require(record is not None and record.get("kind") == "file", "RUNTIME_IMPORT_NOT_APPROVED")
        assert record is not None
        raw = _read(origin, 512 * 1024 * 1024)
        _require(
            hashlib.sha256(raw).hexdigest() == record["sha256"]
            and len(raw) == record["size"]
            and stat.S_IMODE(origin.lstat().st_mode) == record["mode"],
            "RUNTIME_IMPORT_INTEGRITY_CHANGED",
        )
        return raw

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> ModuleSpec | None:
        if _project(fullname):
            _require(
                fullname in HARNESS_MODULES or fullname in {"deployment", "bot_core"},
                "HARNESS_IMPORT_NOT_APPROVED",
            )
            project_spec = ModuleSpec(
                fullname,
                self,
                origin=self.get_filename(fullname),
                is_package=fullname in {"deployment", "bot_core"},
            )
            project_spec.has_location = True
            return project_spec
        if (
            BuiltinImporter.find_spec(fullname) is not None
            or FrozenImporter.find_spec(fullname) is not None
        ):
            return None
        spec = PathFinder.find_spec(fullname, path)
        # Optional imports in trusted stdlib (e.g. copy's org.python.core)
        # require Python's absence semantics. Never let another finder execute
        # a fallback; an existing unapproved object remains a hard refusal.
        if spec is None:
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        _require(type(spec.origin) is str, "RUNTIME_IMPORT_NOT_APPROVED")
        origin = Path(cast(str, spec.origin))
        if self.runtime is None:
            # Before runtime signature verification only the independently
            # protected host stdlib may execute. Read source bytes directly.
            stdlib = self.bootstrap_stdlib
            _require(
                fullname.split(".", 1)[0] in sys.stdlib_module_names
                and origin.is_relative_to(stdlib)
                and not {"site-packages", "dist-packages", "__pycache__"}
                & set(origin.relative_to(stdlib).parts),
                "RUNTIME_NOT_VERIFIED",
            )
            raw = _read(origin, 512 * 1024 * 1024)
        else:
            raw = self._runtime_bytes(origin)
        if type(spec.loader) is importlib.machinery.SourceFileLoader:
            # Bypass even authenticated bytecode; execute the checked source bytes.
            self.sources[fullname] = raw
            self._snapshots[fullname] = (str(origin), raw)
            replacement = ModuleSpec(
                fullname,
                self,
                origin=str(origin),
                is_package=spec.submodule_search_locations is not None,
            )
            replacement.submodule_search_locations = spec.submodule_search_locations
            replacement.has_location = True
            return replacement
        _require(
            type(spec.loader) is importlib.machinery.ExtensionFileLoader,
            "RUNTIME_LOADER_NOT_APPROVED",
        )
        return spec


def _native_module() -> Any:
    sources = _harness_sources()
    _import_state()
    loader = _HarnessLoader(sources, Path(__file__).absolute().parent.parent)
    sys.meta_path.insert(0, loader)
    before = set(sys.modules)
    try:
        runtime = importlib.import_module("deployment.freshness_runtime")
        verified = runtime.verify_installed_runtime()
        manifest = verified.manifest
        loader.bind_runtime(manifest)
        return importlib.import_module("deployment.stage9_vm_native")
    except BaseException:
        sys.meta_path.remove(loader)
        for name in set(sys.modules) - before:
            if _project(name):
                del sys.modules[name]
        raise


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
    parser.add_argument(
        "--mode",
        required=True,
        choices=("INSPECT", "EXECUTE-VM-TRIAL"),
        help="INSPECT: diagnostyka odczytowa z chronionego drzewa harnessu, bez instalacji runtime/aplikacji ani approval VM; EXECUTE-VM-TRIAL: osobno zatwierdzona próba VM.",
    )
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
        if not execute:
            _require(
                args.phase is None and args.inputs is None and not args.native_worker,
                "INSPECT_EXECUTION_ARGUMENTS_REFUSED",
            )
        _harness_sources(args.revision if execute else None)
        _import_state()
        helpers = _environment_helpers(execute)
        if not execute:
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
