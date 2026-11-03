"""Rootless eksport zatwierdzonych Git blobs do niepodpisanego pakietu VM v2.

Nie ustanawia zaufania ani nie podpisuje materiałów. Osobny operator sprawdza
wynik i stosuje niezależny autorytet TEST_ONLY przed przeniesieniem do VM.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

BOOTSTRAP_FILES = (
    "deployment/install-production-local.sh",
    "deployment/production_local_installer.py",
    "deployment/production_local_installer_host.py",
    "deployment/freshness_runtime.py",
    "deployment/freshness_runtime_bootstrap.py",
)
UNIT = "deployment/systemd/cryptohunter-freshness-verifier.service"
TOKEN = b"@FRESHNESS_RUNTIME_PYTHON@"
MAX_FILE = 8 * 1024 * 1024


class BundleError(RuntimeError):
    """Stała przyczyna odmowy bez treści plików i stderr Git."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise BundleError(reason)


def _path(path: Path, *, missing_leaf: bool = False) -> None:
    _require(path.is_absolute() and ".." not in path.parts, "ABSOLUTE_PATH_REQUIRED")
    components = [*reversed(path.parents), path]
    for component in components:
        if component == path and missing_leaf:
            _require(not os.path.lexists(component), "OUTPUT_ALREADY_EXISTS")
            continue
        info = component.lstat()
        _require(not stat.S_ISLNK(info.st_mode), "SYMLINK_REFUSED")
        if component != path:
            _require(stat.S_ISDIR(info.st_mode), "DIRECTORY_REQUIRED")


def _read(path: Path) -> bytes:
    _path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "REGULAR_FILE_REQUIRED")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            value = stream.read(MAX_FILE + 1)
        _require(len(value) <= MAX_FILE, "FILE_TOO_LARGE")
        return value
    finally:
        os.close(fd)


def _git(source: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["/usr/bin/git", "--no-optional-locks", "-c", "core.fsmonitor=false", *arguments],
            cwd=source,
            env={
                "PATH": "/usr/bin:/bin",
                "LANG": "C",
                "LC_ALL": "C",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_NO_REPLACE_OBJECTS": "1",
            },
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BundleError("GIT_UNAVAILABLE") from exc
    _require(result.returncode == 0 and len(result.stdout) <= MAX_FILE, "GIT_SOURCE_REFUSED")
    return result.stdout


def _relative(value: str) -> None:
    _require(type(value) is str, "INVENTORY_PATH_REFUSED")
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and str(path) == value
        and len(path.parts) >= 2
        and all(part not in {".", ".."} for part in path.parts)
        and path.parts[0] in {"bot_core", "deployment"},
        "INVENTORY_PATH_REFUSED",
    )


def _blob(source: Path, revision: str, relative: str) -> tuple[bytes, int]:
    _relative(relative)
    tree = _git(source, "ls-tree", revision, "--", relative).decode("ascii")
    fields = tree.split()
    _require(
        len(fields) == 4
        and fields[0] in {"100644", "100755"}
        and fields[1] == "blob"
        and fields[3] == relative,
        "GIT_BLOB_REQUIRED",
    )
    value = _git(source, "show", revision + ":" + relative)
    _require(_read(source / relative) == value, "WORKTREE_SOURCE_CHANGED")
    return value, 0o755 if fields[0] == "100755" else 0o644


def _assignment(content: bytes, name: str) -> ast.expr:
    matches = [
        node.value
        for node in ast.parse(content).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    _require(len(matches) == 1, "SOURCE_CONTRACT_REFUSED")
    return matches[0]


def _contract(installer: bytes, runtime: bytes) -> tuple[set[str], bytes]:
    _require(
        ast.literal_eval(_assignment(installer, "SCOPE")) == "cryptohunter.freshness.substrate.v2",
        "V2_SCOPE_REQUIRED",
    )
    node = _assignment(installer, "REQUIRED_FILES")
    _require(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "frozenset"
        and len(node.args) == 1
        and not node.keywords,
        "SOURCE_INVENTORY_REFUSED",
    )
    # The exact AST shape above is required before accessing its argument.
    if not isinstance(node, ast.Call):
        raise BundleError("SOURCE_INVENTORY_REFUSED")
    required = ast.literal_eval(node.args[0])
    _require(type(required) is set and bool(required), "SOURCE_INVENTORY_REFUSED")
    for relative in required:
        _relative(relative)
    _require(
        {UNIT, "deployment/freshness_runtime.py", "deployment/freshness_runtime_bootstrap.py"}
        <= required,
        "SIGNED_RUNTIME_INVENTORY_REQUIRED",
    )
    template = ast.literal_eval(_assignment(runtime, "UNIT_TEMPLATE"))
    _require(type(template) is bytes and template.count(TOKEN) == 1, "UNIT_TEMPLATE_REFUSED")
    return required, template


def _binding(path: Path, revision: str) -> dict[str, str]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, "DUPLICATE_BINDING_FIELD")
            result[key] = value
        return result

    document = json.loads(_read(path), object_pairs_hook=unique)
    expected = {
        "environment_id",
        "manifest_sha256",
        "source_revision",
        "executable",
        "base_executable",
        "base_executable_sha256",
    }
    _require(
        type(document) is dict
        and set(document) == expected | {"schema"}
        and type(document["schema"]) is int
        and document["schema"] == 1
        and all(type(document.get(key)) is str for key in expected),
        "BINDING_SCHEMA_REFUSED",
    )
    binding = {key: document[key] for key in expected}
    environment_id = binding["environment_id"]
    _require(
        re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,95}", environment_id) is not None
        and binding["source_revision"] == revision
        and binding["base_executable"] == "/usr/bin/python3.12"
        and binding["executable"]
        == f"/opt/cryptohunter/freshness-runtime/releases/{environment_id}/bin/python3.12"
        and all(
            re.fullmatch(r"[0-9a-f]{64}", binding[key]) is not None
            for key in ("manifest_sha256", "base_executable_sha256")
        ),
        "BINDING_VALUES_REFUSED",
    )
    return binding


def _write(output: Path, relative: str, content: bytes, mode: int) -> None:
    if sys.platform == "win32":
        raise BundleError("LINUX_BUILDER_REQUIRED")
    target = output / relative
    for directory in reversed(target.parent.relative_to(output).parents):
        candidate = output / directory
        if candidate != output and not candidate.exists():
            candidate.mkdir(mode=0o755)
            candidate.chmod(0o755)
    if not target.parent.exists():
        target.parent.mkdir(mode=0o755)
        target.parent.chmod(0o755)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(fd, "wb") as stream:
        stream.write(content)
        os.fchmod(stream.fileno(), mode)


def _output_parent_safe(output: Path) -> None:
    """Obcy UID nie może przemianować przodka nowego katalogu kandydata."""
    if sys.platform == "win32":
        raise BundleError("LINUX_BUILDER_REQUIRED")
    uid = os.geteuid()
    for directory in output.parents:
        info = directory.lstat()
        safe_temporary_root = (
            directory == Path("/tmp") and info.st_uid == 0 and bool(info.st_mode & stat.S_ISVTX)
        )
        _require(
            stat.S_ISDIR(info.st_mode)
            and info.st_uid in {0, uid}
            and (not info.st_mode & 0o022 or safe_temporary_root),
            "OUTPUT_PARENT_UNPROTECTED",
        )


def prepare(source: Path, revision: str, binding_path: Path, output: Path) -> dict[str, Any]:
    """Eksportuj nowy pakiet; nie naprawiaj ani nie usuwaj zastanych danych."""
    if sys.platform == "win32":
        raise BundleError("LINUX_BUILDER_REQUIRED")
    _require(os.geteuid() != 0, "ROOTLESS_BUILDER_REQUIRED")
    _require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "REVISION_REFUSED")
    _path(source)
    _path(output, missing_leaf=True)
    _output_parent_safe(output)
    _require(source.is_dir(), "SOURCE_DIRECTORY_REQUIRED")
    _require(
        not any(
            output.is_relative_to(Path(prefix))
            for prefix in (
                "/etc",
                "/usr",
                "/opt",
                "/var/lib",
                "/run",
                "/boot",
                "/sys",
                "/proc",
                "/dev",
            )
        ),
        "OUTPUT_SYSTEM_PATH_REFUSED",
    )
    _require(_git(source, "rev-parse", "HEAD").decode("ascii").strip() == revision, "HEAD_CHANGED")
    installer, _ = _blob(source, revision, "deployment/production_local_installer.py")
    runtime, _ = _blob(source, revision, "deployment/freshness_runtime.py")
    required, template = _contract(installer, runtime)
    binding = _binding(binding_path, revision)
    unit = template.replace(TOKEN, binding["executable"].encode("ascii"))
    inputs = {relative: _blob(source, revision, relative) for relative in required}
    _require(inputs[UNIT][0] == template, "SOURCE_UNIT_TEMPLATE_CHANGED")
    bootstrap = {relative: _blob(source, revision, relative) for relative in BOOTSTRAP_FILES}
    # All input validation precedes the first write. A later interruption leaves
    # this new candidate for review; no recursive cleanup or overwrite occurs.
    output.mkdir(mode=0o755)
    output.chmod(0o755)
    files = []
    for relative, (content, mode) in sorted(inputs.items()):
        if relative == UNIT:
            content = unit
        _write(output, "payload/" + relative, content, mode)
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
                "mode": mode,
            }
        )
    manifest = {
        "schema_version": 2,
        "scope": "cryptohunter.freshness.substrate.v2",
        "source_revision": revision,
        "files": files,
        "runtime": binding,
    }
    _write(
        output,
        "release.json",
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(),
        0o644,
    )
    sums = []
    for relative, (content, mode) in bootstrap.items():
        _write(output, "bootstrap/" + relative, content, mode)
        sums.append(hashlib.sha256(content).hexdigest() + "  " + relative + "\n")
    _write(output, "bootstrap.sha256", "".join(sums).encode("ascii"), 0o644)
    _write(output, "revision.txt", (revision + "\n").encode("ascii"), 0o644)
    return {
        "result": "UNSIGNED_TEST_ONLY_CANDIDATE",
        "source_revision": revision,
        "files": len(files),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare(args.source, args.revision, args.binding, args.output)
    except (BundleError, OSError, ValueError, SyntaxError) as exc:
        reason = str(exc) if isinstance(exc, BundleError) else "INPUT_OR_EXPORT_REFUSED"
        print(json.dumps({"result": "REFUSED", "reason": reason}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
