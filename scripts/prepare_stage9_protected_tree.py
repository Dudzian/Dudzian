"""Build data-only candidates, then copy authenticated bytes into a new protected tree.

This does not authenticate a Git commit, authorize an installation, or execute
candidate code. The operator supplies an independently reviewed commit, file set
and inventory digest. Transport archives are not accepted as installed trees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, cast

INVENTORY = Path("/etc/cryptohunter/stage9-harness-inventory.json")
MAX_FILE = 16 * 1024 * 1024
MAX_INVENTORY = 8 * 1024 * 1024
MAX_TOTAL = 256 * 1024 * 1024


class TreeError(RuntimeError):
    """A fixed refusal reason, without source file contents or command stderr."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise TreeError(reason)


def _absolute(path: Path) -> None:
    _require(path.is_absolute() and ".." not in path.parts, "ABSOLUTE_PATH_REQUIRED")
    for component in (*reversed(path.parents), path):
        if os.path.lexists(component):
            info = component.lstat()
            _require(not stat.S_ISLNK(info.st_mode), "SYMLINK_REFUSED")
            if component != path:
                _require(stat.S_ISDIR(info.st_mode), "DIRECTORY_REQUIRED")


def _relative(value: str) -> None:
    _require(
        type(value) is str
        and re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", value) is not None
        and all(part not in {".", "..", ".git"} for part in PurePosixPath(value).parts),
        "INVENTORY_PATH_REFUSED",
    )


def _stable(info: os.stat_result) -> tuple[int, ...]:
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


def _read(path: Path, limit: int = MAX_FILE) -> bytes:
    _absolute(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "REGULAR_FILE_REQUIRED")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        _require(len(raw) <= limit, "FILE_TOO_LARGE")
        _require(
            _stable(os.fstat(fd)) == _stable(info) and _stable(path.lstat()) == _stable(info),
            "FILE_CHANGED_DURING_READ",
        )
        return raw
    finally:
        os.close(fd)


def _git(source: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            [
                "/usr/bin/git",
                "--no-optional-locks",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.hooksPath=/dev/null",
                *arguments,
            ],
            cwd=source,
            env={
                "PATH": "/usr/bin:/bin",
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
        raise TreeError("GIT_UNAVAILABLE") from exc
    _require(result.returncode == 0 and len(result.stdout) <= MAX_FILE, "GIT_SOURCE_REFUSED")
    return result.stdout


def _write(path: Path, value: bytes, mode: int) -> None:
    if sys.platform != "linux":
        raise TreeError("LINUX_REQUIRED")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        _require(stream.write(value) == len(value), "WRITE_INCOMPLETE")
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), mode)


def _new(path: Path) -> None:
    _absolute(path)
    _require(path.parent.is_dir() and not os.path.lexists(path), "NEW_OUTPUT_REQUIRED")


def _object(source: Path, kind: str, object_id: str) -> bytes:
    raw = _git(source, "cat-file", kind, object_id)
    _require(
        hashlib.sha1(kind.encode() + b" " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        == object_id,
        "GIT_OBJECT_INTEGRITY_REFUSED",
    )
    return raw


def _git_blob(source: Path, tree_id: str, relative: str) -> tuple[bytes, bytes]:
    parts = PurePosixPath(relative).parts
    for position, part in enumerate(parts):
        raw = _object(source, "tree", tree_id)
        entries: dict[bytes, tuple[bytes, str]] = {}
        cursor = 0
        while cursor < len(raw):
            boundary = raw.index(b"\0", cursor)
            mode, name = raw[cursor:boundary].split(b" ", 1)
            _require(name not in entries and boundary + 21 <= len(raw), "GIT_TREE_REFUSED")
            entries[name] = (mode, raw[boundary + 1 : boundary + 21].hex())
            cursor = boundary + 21
        _require(part.encode("ascii") in entries, "EXACT_GIT_BLOB_REQUIRED")
        mode, object_id = entries[part.encode("ascii")]
        if position < len(parts) - 1:
            _require(mode == b"40000", "REGULAR_GIT_BLOB_REQUIRED")
            tree_id = object_id
        else:
            _require(mode in {b"100644", b"100755"}, "REGULAR_GIT_BLOB_REQUIRED")
            return _object(source, "blob", object_id), mode
    raise TreeError("EXACT_GIT_BLOB_REQUIRED")


def export(
    source: Path,
    revision: str,
    candidate: Path,
    tree_root: Path,
    inventory_path: Path,
    paths: tuple[str, ...],
) -> dict[str, Any]:
    """Export explicit Git blobs as data; never read or chmod checkout sources."""
    _require(sys.platform == "linux", "LINUX_REQUIRED")
    _absolute(source)
    _absolute(tree_root)
    _new(candidate)
    _new(inventory_path)
    _require(
        not inventory_path.is_relative_to(candidate)
        and not candidate.is_relative_to(source)
        and not tree_root.is_relative_to(source)
        and not inventory_path.is_relative_to(source)
        and inventory_path != candidate,
        "SEPARATE_OUTPUT_REQUIRED",
    )
    _require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "FULL_COMMIT_REQUIRED")
    _require(
        _git(source, "rev-parse", "--verify", revision + "^{commit}").strip().decode() == revision,
        "EXACT_COMMIT_REQUIRED",
    )
    commit = _object(source, "commit", revision)
    first_line = commit.split(b"\n", 1)[0]
    _require(re.fullmatch(rb"tree [0-9a-f]{40}", first_line) is not None, "GIT_COMMIT_REFUSED")
    tree_id = first_line[5:].decode("ascii")
    _require(
        bool(paths) and len(paths) <= 10000 and len(set(paths)) == len(paths),
        "EXPLICIT_FILE_SET_REQUIRED",
    )
    records: dict[str, dict[str, Any]] = {".": {"path": ".", "kind": "directory", "mode": 0o555}}
    contents: dict[str, bytes] = {}
    total_bytes = 0
    for relative in sorted(paths):
        _relative(relative)
        raw, mode = _git_blob(source, tree_id, relative)
        total_bytes += len(raw)
        _require(total_bytes <= MAX_TOTAL, "TREE_TOO_LARGE")
        contents[relative] = raw
        records[relative] = {
            "path": relative,
            "kind": "file",
            "mode": 0o555 if mode == b"100755" else 0o444,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        for parent in PurePosixPath(relative).parents:
            if str(parent) != ".":
                records[str(parent)] = {"path": str(parent), "kind": "directory", "mode": 0o555}
    document = {
        "schema": 1,
        "source_revision": revision,
        "tree_root": str(tree_root),
        "objects": [records[key] for key in sorted(records)],
    }
    raw_inventory = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    _require(len(raw_inventory) <= MAX_INVENTORY, "INVENTORY_TOO_LARGE")
    candidate.mkdir(mode=0o700)
    for relative, item in sorted(records.items()):
        if item["kind"] == "directory" and relative != ".":
            (candidate / relative).mkdir(mode=0o700)
    for relative, value in contents.items():
        _write(candidate / relative, value, records[relative]["mode"])
    # Candidate directories remain writable by their unprivileged owner. They
    # are transport data, and cannot satisfy protected verification or execution.
    _write(inventory_path, raw_inventory, 0o400)
    return {
        "status": "UNTRUSTED_CANDIDATE",
        "source_revision": revision,
        "inventory_sha256": hashlib.sha256(raw_inventory).hexdigest(),
        "objects": len(records),
    }


def _document(raw: bytes, digest: str) -> dict[str, Any]:
    _require(
        re.fullmatch(r"[0-9a-f]{64}", digest) is not None
        and hashlib.sha256(raw).hexdigest() == digest,
        "INVENTORY_PIN_REFUSED",
    )

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, "DUPLICATE_JSON_FIELD")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=unique)
    _require(
        type(value) is dict
        and set(value) == {"schema", "source_revision", "tree_root", "objects"}
        and type(value["schema"]) is int
        and value["schema"] == 1
        and type(value["source_revision"]) is str
        and re.fullmatch(r"[0-9a-f]{40}", value["source_revision"]) is not None
        and type(value["tree_root"]) is str
        and str(Path(value["tree_root"])) == value["tree_root"]
        and Path(value["tree_root"]).is_absolute()
        and ".." not in Path(value["tree_root"]).parts
        and type(value["objects"]) is list
        and 0 < len(value["objects"]) <= 10000,
        "INVENTORY_SCHEMA_REFUSED",
    )
    names: dict[str, str] = {}
    for item in value["objects"]:
        _require(type(item) is dict and type(item.get("path")) is str, "INVENTORY_SCHEMA_REFUSED")
        relative = item["path"]
        if relative != ".":
            _relative(relative)
        kind = item.get("kind")
        _require(
            type(kind) is str
            and kind in {"file", "directory"}
            and set(item)
            == ({"path", "kind", "mode", "sha256"} if kind == "file" else {"path", "kind", "mode"})
            and type(item["mode"]) is int
            and item["mode"] in ({0o444, 0o555} if kind == "file" else {0o555})
            and relative not in names,
            "INVENTORY_SCHEMA_REFUSED",
        )
        if kind == "file":
            _require(
                type(item["sha256"]) is str
                and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is not None,
                "INVENTORY_SCHEMA_REFUSED",
            )
        names[relative] = kind
    _require(names.get(".") == "directory", "ROOT_INVENTORY_REQUIRED")
    for relative in names:
        for parent in PurePosixPath(relative).parents:
            _require(names.get(str(parent)) == "directory", "DIRECTORY_INVENTORY_REQUIRED")
    return cast(dict[str, Any], value)


def _protected_ancestors(path: Path) -> None:
    _absolute(path)
    for component in reversed(path.parents):
        info = component.lstat()
        _require(
            stat.S_ISDIR(info.st_mode) and info.st_uid == 0 and not info.st_mode & 0o022,
            "PROTECTED_PARENT_REQUIRED",
        )


def verify_tree(root: Path, document: dict[str, Any], *, protected: bool) -> dict[str, bytes]:
    """Compare the complete object set, including empty directories and specials."""
    _absolute(root)
    expected = {item["path"]: item for item in document["objects"]}
    actual: set[str] = set()
    values: dict[str, bytes] = {}
    total_bytes = 0

    def visit(path: Path, relative: str) -> None:
        nonlocal total_bytes
        _require(relative in expected, "UNAPPROVED_TREE_OBJECT")
        actual.add(relative)
        item = expected[relative]
        info = path.lstat()
        is_directory = item["kind"] == "directory"
        _require(
            stat.S_ISDIR(info.st_mode) if is_directory else stat.S_ISREG(info.st_mode),
            "TREE_OBJECT_TYPE_REFUSED",
        )
        if protected:
            _require(
                info.st_uid == 0 and stat.S_IMODE(info.st_mode) == item["mode"],
                "TREE_NOT_PROTECTED",
            )
        if is_directory:
            for child in sorted(path.iterdir()):
                visit(child, child.relative_to(root).as_posix())
        else:
            _require(info.st_nlink == 1, "HARDLINK_REFUSED")
            value = _read(path)
            _require(hashlib.sha256(value).hexdigest() == item["sha256"], "TREE_INTEGRITY_REFUSED")
            total_bytes += len(value)
            _require(total_bytes <= MAX_TOTAL, "TREE_TOO_LARGE")
            values[relative] = value
        _require(_stable(path.lstat()) == _stable(info), "TREE_CHANGED_DURING_READ")

    if protected:
        _protected_ancestors(root)
        _require(str(root) == document["tree_root"], "TREE_ROOT_MISMATCH")
    visit(root, ".")
    _require(actual == set(expected), "TREE_INVENTORY_INCOMPLETE")
    return values


def install(candidate: Path, inventory_path: Path, digest: str) -> dict[str, Any]:
    """Copy a pinned candidate into a previously absent root-owned installation."""
    _require(sys.platform == "linux" and os.geteuid() == 0, "NATIVE_ROOT_INSTALL_REQUIRED")
    raw = _read(inventory_path, MAX_INVENTORY)
    document = _document(raw, digest)
    root = Path(document["tree_root"])
    _new(root)
    _new(INVENTORY)
    _protected_ancestors(root)
    _protected_ancestors(INVENTORY)
    _require(
        not root.is_relative_to(candidate)
        and not candidate.is_relative_to(root)
        and not INVENTORY.is_relative_to(root),
        "SEPARATE_INSTALLATION_REQUIRED",
    )
    values = verify_tree(candidate, document, protected=False)
    root.mkdir(mode=0o700)
    try:
        records = {item["path"]: item for item in document["objects"]}
        for relative, item in sorted(records.items()):
            if item["kind"] == "directory" and relative != ".":
                (root / relative).mkdir(mode=0o700)
        for relative, value in values.items():
            _write(root / relative, value, records[relative]["mode"])
        for relative, item in sorted(records.items(), reverse=True):
            if item["kind"] == "directory" and relative != ".":
                (root / relative).chmod(item["mode"])
        root.chmod(0o555)
        verify_tree(root, document, protected=True)
        _write(INVENTORY, raw, 0o444)
    except (TreeError, OSError, ValueError):
        # No cleanup or reuse: preserve the failed candidate behind root-only
        # access for explicit operator investigation.
        root.chmod(0o700)
        raise
    return {
        "status": "PROTECTED_TREE_BUILT",
        "source_revision": document["source_revision"],
        "inventory_sha256": digest,
    }


def verify_installed(digest: str) -> dict[str, Any]:
    """Read-only verification of the fixed protected external inventory and tree."""
    _require(sys.platform == "linux", "LINUX_REQUIRED")
    _protected_ancestors(INVENTORY)
    info = INVENTORY.lstat()
    _require(
        stat.S_ISREG(info.st_mode)
        and info.st_uid == 0
        and info.st_nlink == 1
        and stat.S_IMODE(info.st_mode) == 0o444,
        "INVENTORY_NOT_PROTECTED",
    )
    document = _document(_read(INVENTORY, MAX_INVENTORY), digest)
    verify_tree(Path(document["tree_root"]), document, protected=True)
    return {
        "status": "PROTECTED_TREE_VERIFIED",
        "source_revision": document["source_revision"],
        "inventory_sha256": digest,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest="operation", required=True)
    export_parser = operations.add_parser("export")
    export_parser.add_argument("--source", type=Path, required=True)
    export_parser.add_argument("--revision", required=True)
    export_parser.add_argument("--candidate", type=Path, required=True)
    export_parser.add_argument("--tree-root", type=Path, required=True)
    export_parser.add_argument("--inventory", type=Path, required=True)
    export_parser.add_argument("--path", action="append", required=True)
    install_parser = operations.add_parser("install")
    install_parser.add_argument("--candidate", type=Path, required=True)
    install_parser.add_argument("--inventory", type=Path, required=True)
    install_parser.add_argument("--inventory-sha256", required=True)
    verify_parser = operations.add_parser("verify")
    verify_parser.add_argument("--inventory-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.operation == "export":
            result = export(
                args.source,
                args.revision,
                args.candidate,
                args.tree_root,
                args.inventory,
                tuple(args.path),
            )
        elif args.operation == "install":
            result = install(args.candidate, args.inventory, args.inventory_sha256)
        else:
            result = verify_installed(args.inventory_sha256)
    except (TreeError, OSError, ValueError):
        print(json.dumps({"status": "FAIL", "reason": "PROTECTED_TREE_REFUSED"}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
