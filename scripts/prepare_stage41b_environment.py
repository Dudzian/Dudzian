"""Prepare only the reviewed Stage 41B test runtime; never start a service.

Build runs unprivileged. Installation is offline and must be bootstrapped with
an independently verified SHA256 of this standalone, standard-library program.
Git object proofs bind every exported source file and the wheel lock to the
operator's full reviewed commit. No Git configuration or package build code is
executed as root. This is test preparation, never production provisioning.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import subprocess
import sys
from typing import Any
from urllib.parse import urlsplit
from urllib.request import build_opener, HTTPRedirectHandler
import venv
import zipfile

PREFIX = Path("/opt/cryptohunter-stage41b")
INSTALLER = "scripts/prepare_stage41b_environment.py"
LOCK = "deployment/stage41b-wheel-lock.json"
REQUIREMENTS = "deployment/stage41b-requirements.lock"
EXACT_FILES = frozenset(
    {
        INSTALLER,
        LOCK,
        REQUIREMENTS,
        "deployment/stage41b_role_worker.py",
        "scripts/qualify_stage41b_signer_isolation.py",
    }
)
MAX_BUNDLE = 64 * 1024 * 1024
MAX_FILE = 32 * 1024 * 1024
MAX_WHEEL_CONTENT = 100 * 1024 * 1024


class PreparationBlocked(RuntimeError):
    """A fixed public reason; never native exception or credential text."""


def require_commit(value: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise PreparationBlocked("FULL_REVIEWED_COMMIT_REQUIRED")
    return value


def _json(raw: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        answer: dict[str, Any] = {}
        for key, value in items:
            if key in answer:
                raise PreparationBlocked("DUPLICATE_JSON_FIELD")
            answer[key] = value
        return answer

    def bad_constant(_value: str) -> None:
        raise PreparationBlocked("NONFINITE_JSON_FIELD")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise PreparationBlocked("NONFINITE_JSON_FIELD")
        return parsed

    return json.loads(
        raw, object_pairs_hook=pairs, parse_constant=bad_constant, parse_float=finite_float
    )


def _b64(value: object) -> bytes:
    if type(value) is not str:
        raise PreparationBlocked("EXACT_BASE64_REQUIRED")
    answer = base64.b64decode(value, validate=True)
    if base64.b64encode(answer).decode("ascii") != value:
        raise PreparationBlocked("CANONICAL_BASE64_REQUIRED")
    return answer


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_oid(kind: str, raw: bytes) -> str:
    # SHA1 here reproduces Git's object identifier, never password/key hashing.
    return hashlib.sha1(
        kind.encode("ascii") + b" " + str(len(raw)).encode("ascii") + b"\0" + raw
    ).hexdigest()


def selected_source(path: str) -> bool:
    return path in EXACT_FILES or (path.startswith("bot_core/") and path.endswith(".py"))


def _tree_entries(raw: bytes) -> list[tuple[str, str, str]]:
    entries: list[tuple[str, str, str]] = []
    offset = 0
    seen: set[str] = set()
    while offset < len(raw):
        stop = raw.index(b"\0", offset)
        mode, name_raw = raw[offset:stop].split(b" ", 1)
        name = name_raw.decode("utf-8")
        if name in {"", ".", ".."} or "/" in name or "\\" in name or name in seen:
            raise PreparationBlocked("UNSAFE_GIT_TREE_NAME")
        seen.add(name)
        object_bytes = raw[stop + 1 : stop + 21]
        if len(object_bytes) != 20:
            raise PreparationBlocked("TRUNCATED_GIT_TREE")
        entries.append((mode.decode("ascii"), name, object_bytes.hex()))
        offset = stop + 21
    return entries


def verify_proof(document: object, expected_commit: str) -> dict[str, bytes]:
    """Verify a minimal Git Merkle proof without invoking an untrusted Git."""
    require_commit(expected_commit)
    if type(document) is not dict or set(document) != {
        "schema",
        "commit",
        "commit_b64",
        "trees",
        "files",
    }:
        raise PreparationBlocked("EXACT_BUNDLE_SCHEMA_REQUIRED")
    if (
        type(document["schema"]) is not int
        or document["schema"] != 1
        or document["commit"] != expected_commit
    ):
        raise PreparationBlocked("BUNDLE_COMMIT_MISMATCH")
    raw_commit = _b64(document["commit_b64"])
    if git_oid("commit", raw_commit) != expected_commit:
        raise PreparationBlocked("GIT_COMMIT_INTEGRITY_FAILURE")
    first = raw_commit.split(b"\n", 1)[0]
    if re.fullmatch(rb"tree [0-9a-f]{40}", first) is None:
        raise PreparationBlocked("GIT_COMMIT_TREE_REQUIRED")
    trees, files = document["trees"], document["files"]
    if type(trees) is not dict or type(files) is not dict:
        raise PreparationBlocked("EXACT_GIT_PROOF_REQUIRED")
    expected_files: dict[str, str] = {}
    visited: set[str] = set()

    def walk(oid: str, prefix: str) -> None:
        if oid not in trees:
            raise PreparationBlocked("GIT_TREE_PROOF_MISSING")
        raw = _b64(trees[oid])
        if git_oid("tree", raw) != oid:
            raise PreparationBlocked("GIT_TREE_INTEGRITY_FAILURE")
        visited.add(oid)
        for mode, name, child in _tree_entries(raw):
            path = prefix + name
            if mode == "40000":
                if path in {"bot_core", "deployment", "scripts"} or path.startswith("bot_core/"):
                    walk(child, path + "/")
            elif selected_source(path):
                if mode not in {"100644", "100755"}:
                    raise PreparationBlocked("NONREGULAR_SOURCE_FORBIDDEN")
                expected_files[path] = child

    walk(first[5:].decode("ascii"), "")
    if (
        set(trees) != visited
        or set(files) != set(expected_files)
        or not EXACT_FILES.issubset(files)
    ):
        raise PreparationBlocked("INCOMPLETE_OR_EXTRA_SOURCE_PROOF")
    verified: dict[str, bytes] = {}
    for path, oid in expected_files.items():
        record = files[path]
        if (
            type(record) is not dict
            or set(record) != {"oid", "content_b64"}
            or record["oid"] != oid
        ):
            raise PreparationBlocked("GIT_BLOB_PROOF_MISMATCH")
        raw = _b64(record["content_b64"])
        if git_oid("blob", raw) != oid:
            raise PreparationBlocked("GIT_BLOB_INTEGRITY_FAILURE")
        verified[path] = raw
    return verified


def _git(source: Path, *arguments: str) -> bytes:
    command = [
        "git",
        "--no-pager",
        "--no-optional-locks",
        "-C",
        str(source),
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-c",
        "core.hooksPath=/dev/null",
        *arguments,
    ]
    result = subprocess.run(
        command, stdin=subprocess.DEVNULL, capture_output=True, timeout=120, check=False
    )
    if result.returncode:
        raise PreparationBlocked("SOURCE_GIT_VERIFICATION_FAILED")
    return result.stdout


def source_proof(source: Path, commit: str) -> dict[str, Any]:
    """Read only Git objects; never copy working-tree caches or ignored files."""
    require_commit(commit)
    if _git(source, "rev-parse", "--verify", "HEAD").strip().decode("ascii") != commit:
        raise PreparationBlocked("SOURCE_HEAD_MISMATCH")
    if _git(source, "status", "--porcelain=v1", "-z", "--untracked-files=all"):
        raise PreparationBlocked("CLEAN_SOURCE_REPOSITORY_REQUIRED")
    _git(source, "fsck", "--strict", "--no-reflogs", commit)
    raw_commit = _git(source, "cat-file", "commit", commit)
    document: dict[str, Any] = {
        "schema": 1,
        "commit": commit,
        "commit_b64": base64.b64encode(raw_commit).decode("ascii"),
        "trees": {},
        "files": {},
    }

    def walk(oid: str, prefix: str) -> None:
        raw = _git(source, "cat-file", "tree", oid)
        document["trees"][oid] = base64.b64encode(raw).decode("ascii")
        for mode, name, child in _tree_entries(raw):
            path = prefix + name
            if mode == "40000":
                if path in {"bot_core", "deployment", "scripts"} or path.startswith("bot_core/"):
                    walk(child, path + "/")
            elif selected_source(path):
                content = _git(source, "cat-file", "blob", child)
                document["files"][path] = {
                    "oid": child,
                    "content_b64": base64.b64encode(content).decode("ascii"),
                }

    walk(raw_commit.split(b"\n", 1)[0][5:].decode("ascii"), "")
    verify_proof(document, commit)
    return document


def wheel_lock(raw: bytes) -> list[dict[str, str]]:
    document = _json(raw)
    if type(document) is not dict or set(document) != {"schema", "python", "platform", "wheels"}:
        raise PreparationBlocked("EXACT_WHEEL_LOCK_REQUIRED")
    if (
        type(document["schema"]) is not int
        or document["schema"] != 1
        or document["python"] != "3.12"
        or document["platform"] != "linux-x86_64"
    ):
        raise PreparationBlocked("SUPPORTED_WHEEL_LOCK_REQUIRED")
    entries = document["wheels"]
    if type(entries) is not list or not 1 <= len(entries) <= 32:
        raise PreparationBlocked("BOUNDED_WHEEL_LOCK_REQUIRED")
    names: set[str] = set()
    filenames: set[str] = set()
    for entry in entries:
        if (
            type(entry) is not dict
            or set(entry) != {"name", "version", "filename", "sha256", "url"}
            or any(type(value) is not str for value in entry.values())
        ):
            raise PreparationBlocked("EXACT_WHEEL_RECORD_REQUIRED")
        filename = entry["filename"]
        url = urlsplit(entry["url"])
        if (
            re.fullmatch(r"[A-Za-z0-9_.-]+\.whl", filename) is None
            or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
            or url.scheme != "https"
            or url.netloc != "files.pythonhosted.org"
            or url.query
            or url.fragment
            or not url.path.startswith("/packages/")
            or url.path.rsplit("/", 1)[-1] != filename
            or entry["name"] in names
            or filename in filenames
        ):
            raise PreparationBlocked("UNCONTROLLED_WHEEL_SOURCE_FORBIDDEN")
        names.add(entry["name"])
        filenames.add(filename)
    return entries


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        raise PreparationBlocked("WHEEL_REDIRECT_FORBIDDEN")


def _snapshot(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _read_descriptor(descriptor: int, limit: int) -> bytes:
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit or before.st_nlink != 1:
            raise PreparationBlocked("BOUNDED_REGULAR_ARTIFACT_REQUIRED")
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > limit or _snapshot(before) != _snapshot(os.fstat(descriptor)):
            raise PreparationBlocked("ARTIFACT_CHANGED_DURING_READ")
        return data
    finally:
        os.close(descriptor)


def _read(path: Path, limit: int = MAX_FILE) -> bytes:
    return _read_descriptor(
        os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC), limit
    )


def _open_directory(path: Path) -> int:
    """Pin each ancestor without following symlinks in an untrusted bundle."""
    if not path.is_absolute() or ".." in path.parts:
        raise PreparationBlocked("ABSOLUTE_ARTIFACT_DIRECTORY_REQUIRED")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in path.parts[1:]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _read_at(directory: int, name: str, limit: int = MAX_FILE) -> bytes:
    if "/" in name or name in {"", ".", ".."}:
        raise PreparationBlocked("EXACT_ARTIFACT_NAME_REQUIRED")
    return _read_descriptor(
        os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=directory),
        limit,
    )


def _write_new(path: Path, raw: bytes, mode: int = 0o644) -> None:
    if sys.platform != "linux":
        raise PreparationBlocked("LINUX_ARTIFACT_WRITER_REQUIRED")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    with os.fdopen(
        os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode), "wb"
    ) as stream:
        os.fchmod(stream.fileno(), mode)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def build_bundle(source: Path, commit: str, bundle: Path) -> dict[str, Any]:
    if sys.platform != "linux" or os.geteuid() == 0:
        raise PreparationBlocked("UNPRIVILEGED_BUILD_REQUIRED")
    if bundle.resolve().is_relative_to(source.resolve()):
        raise PreparationBlocked("BUNDLE_MUST_BE_OUTSIDE_SOURCE")
    proof = source_proof(source, commit)
    files = verify_proof(proof, commit)
    lock = wheel_lock(files[LOCK])
    bundle.mkdir(mode=0o700)
    # An incomplete unprivileged build is retained, never guessed away.
    _write_new(
        bundle / "bundle.json",
        json.dumps(proof, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        0o600,
    )
    opener = build_opener(_NoRedirect())
    for record in lock:
        with opener.open(record["url"], timeout=60) as response:
            raw = response.read(MAX_FILE + 1)
        if len(raw) > MAX_FILE or _sha256(raw) != record["sha256"]:
            raise PreparationBlocked("DOWNLOADED_WHEEL_INTEGRITY_FAILURE")
        wheel_members(raw, record)
        _write_new(bundle / "wheels" / record["filename"], raw, 0o600)
    return {
        "status": "BUNDLE_READY",
        "commit": commit,
        "bundle": str(bundle),
        "installer_sha256": _sha256(files[INSTALLER]),
        "physical_executed": False,
    }


def wheel_members(raw: bytes, record: dict[str, str]) -> dict[str, bytes]:
    """Authenticate wheel content, reject executable startup/path tricks."""
    if _sha256(raw) != record["sha256"]:
        raise PreparationBlocked("WHEEL_INTEGRITY_FAILURE")
    members: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        total = 0
        names: set[str] = set()
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            kind = stat.S_IFMT(entry.external_attr >> 16)
            if (
                entry.filename in names
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in entry.filename
                or not path.parts
                or any("\0" in part for part in path.parts)
                or kind not in {0, stat.S_IFREG, stat.S_IFDIR}
                or path.suffix in {".pth", ".pyc"}
                or any(part.endswith(".data") for part in path.parts)
            ):
                raise PreparationBlocked("UNSAFE_WHEEL_MEMBER")
            names.add(entry.filename)
            if entry.is_dir():
                continue
            total += entry.file_size
            if total > MAX_WHEEL_CONTENT or entry.file_size > MAX_FILE:
                raise PreparationBlocked("WHEEL_CONTENT_LIMIT")
            members[entry.filename] = archive.read(entry)
    records = [name for name in members if name.endswith(".dist-info/RECORD")]
    if len(records) != 1:
        raise PreparationBlocked("WHEEL_RECORD_REQUIRED")
    record_path = records[0]
    rows = list(csv.reader(io.StringIO(members[record_path].decode("utf-8"))))
    seen: set[str] = set()
    for row in rows:
        if len(row) != 3 or row[0] in seen or row[0] not in members:
            raise PreparationBlocked("WHEEL_RECORD_MALFORMED")
        name, digest, size = row
        seen.add(name)
        if name == record_path:
            if digest or size:
                raise PreparationBlocked("WHEEL_RECORD_SELF_HASH_FORBIDDEN")
        else:
            expected = "sha256=" + base64.urlsafe_b64encode(
                hashlib.sha256(members[name]).digest()
            ).rstrip(b"=").decode("ascii")
            if digest != expected or size != str(len(members[name])):
                raise PreparationBlocked("WHEEL_RECORD_INTEGRITY_FAILURE")
    if seen != set(members):
        raise PreparationBlocked("WHEEL_RECORD_INCOMPLETE")
    metadata = members.get(record_path.removesuffix("RECORD") + "METADATA", b"")
    from email.parser import BytesParser

    parsed = BytesParser().parsebytes(metadata)
    normalize = lambda value: re.sub(r"[-_.]+", "-", value).lower()
    if (
        normalize(parsed.get("Name", "")) != normalize(record["name"])
        or parsed.get("Version") != record["version"]
    ):
        raise PreparationBlocked("WHEEL_DISTRIBUTION_MISMATCH")
    return members


def _protected(path: Path, *, directory: bool) -> os.stat_result:
    info = path.lstat()
    kind = stat.S_ISDIR if directory else stat.S_ISREG
    if not kind(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
        raise PreparationBlocked("ROOT_OWNED_PROTECTED_PATH_REQUIRED")
    return info


def _host_installation_guard() -> None:
    if (
        sys.platform != "linux"
        or sys.version_info[:2] != (3, 12)
        or platform.machine() != "x86_64"
        or os.geteuid() != 0
    ):
        raise PreparationBlocked("ROOT_UBUNTU_PYTHON_3_12_X86_64_REQUIRED")
    release = dict(
        line.split("=", 1)
        for line in Path("/etc/os-release").read_text().splitlines()
        if "=" in line
    )
    if (
        release.get("ID", "").strip('"') != "ubuntu"
        or release.get("VERSION_ID", "").strip('"') != "24.04"
    ):
        raise PreparationBlocked("UBUNTU_24_04_REQUIRED")
    if (
        Path(sys.executable).resolve() != Path("/usr/bin/python3.12")
        or not sys.flags.isolated
        or not sys.dont_write_bytecode
    ):
        raise PreparationBlocked("ISOLATED_SYSTEM_PYTHON_REQUIRED")
    for path in (Path("/usr/bin/python3.12"), Path(__file__).resolve()):
        _protected(path, directory=False)
        for parent in path.parents:
            _protected(parent, directory=True)


def _inventory(root: Path) -> dict[str, Any]:
    """One no-follow walk plus pinned content reads; no stat-only hash cache."""
    files: dict[str, str] = {}
    directories: list[str] = []
    for current, children, names in os.walk(root, followlinks=False):
        directory = Path(current)
        _protected(directory, directory=True)
        directories.append(directory.relative_to(root).as_posix())
        for child in children:
            _protected(directory / child, directory=True)
        for name in names:
            path = directory / name
            relative = path.relative_to(root).as_posix()
            if relative == "installation.json":
                continue
            _protected(path, directory=False)
            if path.suffix in {".pth", ".pyc"}:
                raise PreparationBlocked("UNEXPECTED_PYTHON_STARTUP_ARTIFACT")
            files[relative] = _sha256(_read(path, MAX_BUNDLE))
    return {"files": files, "directories": sorted(directories)}


def install_bundle(bundle: Path, commit: str) -> dict[str, Any]:
    """The only privileged preparation operation. No service or network call."""
    require_commit(commit)
    _host_installation_guard()
    bundle_fd = _open_directory(bundle)
    try:
        raw_proof = _read_at(bundle_fd, "bundle.json", MAX_BUNDLE)
        proof = _json(raw_proof)
        source = verify_proof(proof, commit)
        if _read(Path(__file__).resolve()) != source[INSTALLER]:
            raise PreparationBlocked("BOOTSTRAP_INSTALLER_COMMIT_MISMATCH")
        lock = wheel_lock(source[LOCK])
        wheels_fd = os.open(
            "wheels", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=bundle_fd
        )
        try:
            # Hold authenticated bytes in memory before any privileged
            # extraction. User bundle replacement cannot change the snapshots.
            wheels = {entry["filename"]: _read_at(wheels_fd, entry["filename"]) for entry in lock}
        finally:
            os.close(wheels_fd)
    finally:
        os.close(bundle_fd)
    decoded = {entry["filename"]: wheel_members(wheels[entry["filename"]], entry) for entry in lock}
    for parent in (Path("/opt"), Path("/")):
        _protected(parent, directory=True)
    if os.path.lexists(PREFIX):
        _protected(PREFIX, directory=True)
    else:
        PREFIX.mkdir(mode=0o755)
        PREFIX.chmod(0o755)
    root = PREFIX / commit
    if os.path.lexists(root):
        return verify_installation(root, commit)
    root.mkdir(mode=0o700)
    # Failures preserve this exclusive scope for manual provenance review. No
    # recursive rollback or replacement of an existing installation is allowed.
    _write_new(
        root / "preparation-origin.json",
        json.dumps({"stage": "41B_TEST_ONLY", "commit": commit}).encode("ascii"),
    )
    _write_new(root / "bundle.json", raw_proof)
    for path, content in source.items():
        _write_new(root / "source" / path, content)
    for filename, content in wheels.items():
        _write_new(root / "artifacts/wheels" / filename, content)
    venv.EnvBuilder(with_pip=False, symlinks=False).create(root / "venv")
    # CPython creates this compatibility alias even with symlinks=False. The
    # selected Ubuntu interpreter uses lib, so retain a wholly regular tree.
    alias = root / "venv/lib64"
    if os.path.lexists(alias):
        alias_info = alias.lstat()
        if (
            not stat.S_ISLNK(alias_info.st_mode)
            or alias_info.st_uid != 0
            or os.readlink(alias) != "lib"
        ):
            raise PreparationBlocked("UNEXPECTED_VENV_ALIAS")
        alias.unlink()
    site = root / "venv/lib/python3.12/site-packages"
    site.mkdir(parents=True, exist_ok=True)
    for members in decoded.values():
        for path, content in members.items():
            _write_new(site / path, content)
    # Normalize only directories in this newly created, still-private scope.
    # A restrictive operator umask must not leave test roles unable to read the
    # verified runtime. Existing installation/prefix permissions are untouched.
    for current, children, _names in os.walk(root, followlinks=False):
        directory = Path(current)
        _protected(directory, directory=True)
        for child in children:
            child_directory = directory / child
            _protected(child_directory, directory=True)
            child_directory.chmod(0o755)
    # venv writes executable copies and activation text but no package code is
    # executed while root. Check library dependencies with the prepared runtime
    # only during the separate operator-approved preflight, after verification.
    inventory = _inventory(root)
    receipt = {
        "schema": 1,
        "kind": "STAGE41B_TEST_ENVIRONMENT",
        "commit": commit,
        "python": "3.12",
        "platform": "linux-x86_64",
        **inventory,
    }
    _write_new(
        root / "installation.json", json.dumps(receipt, sort_keys=True, indent=2).encode("utf-8")
    )
    # A verified receipt is the completion marker. A scope lacking that receipt
    # remains incomplete and is retained for manual review, never overwritten.
    root.chmod(0o755)
    return verify_installation(root, commit)


def verify_installation(root: Path, expected_commit: str) -> dict[str, Any]:
    require_commit(expected_commit)
    if root != PREFIX / expected_commit:
        raise PreparationBlocked("EXACT_VERSIONED_INSTALLATION_REQUIRED")
    for path in (root, *root.parents):
        _protected(path, directory=True)
    _protected(root / "installation.json", directory=False)
    receipt = _json(_read(root / "installation.json", MAX_BUNDLE))
    if type(receipt) is not dict or set(receipt) != {
        "schema",
        "kind",
        "commit",
        "python",
        "platform",
        "files",
        "directories",
    }:
        raise PreparationBlocked("EXACT_INSTALLATION_RECEIPT_REQUIRED")
    if (
        type(receipt["schema"]) is not int
        or receipt["schema"] != 1
        or receipt["kind"] != "STAGE41B_TEST_ENVIRONMENT"
        or receipt["commit"] != expected_commit
        or receipt["python"] != "3.12"
        or receipt["platform"] != "linux-x86_64"
    ):
        raise PreparationBlocked("INSTALLATION_RECEIPT_MISMATCH")
    inventory = _inventory(root)
    if inventory["files"] != receipt["files"] or inventory["directories"] != receipt["directories"]:
        raise PreparationBlocked("INSTALLATION_CONTENT_INTEGRITY_FAILURE")
    source = verify_proof(_json(_read(root / "bundle.json", MAX_BUNDLE)), expected_commit)
    for source_path, raw in source.items():
        if inventory["files"].get("source/" + source_path) != _sha256(raw):
            raise PreparationBlocked("INSTALLED_GIT_SOURCE_MISMATCH")
    for entry in wheel_lock(source[LOCK]):
        archive = _read(root / "artifacts/wheels" / entry["filename"])
        for member_path, raw in wheel_members(archive, entry).items():
            if inventory["files"].get(
                "venv/lib/python3.12/site-packages/" + member_path
            ) != _sha256(raw):
                raise PreparationBlocked("INSTALLED_WHEEL_CONTENT_MISMATCH")
    config = _read(root / "venv/pyvenv.cfg").decode("utf-8")
    settings = dict(line.split(" = ", 1) for line in config.splitlines() if " = " in line)
    if settings.get("include-system-site-packages") != "false" or not settings.get(
        "version", ""
    ).startswith("3.12."):
        raise PreparationBlocked("ISOLATED_VENV_CONFIGURATION_REQUIRED")
    if Path(settings.get("executable", "")).resolve() != Path("/usr/bin/python3.12"):
        raise PreparationBlocked("VENV_SYSTEM_INTERPRETER_MISMATCH")
    _protected(Path("/usr/bin/python3.12"), directory=False)
    if inventory["files"].get("venv/bin/python") != _sha256(_read(Path("/usr/bin/python3.12"))):
        raise PreparationBlocked("VENV_INTERPRETER_CONTENT_MISMATCH")
    return {
        "status": "PREPARED_ENVIRONMENT_VERIFIED",
        "commit": expected_commit,
        "installation": str(root),
        "source": str(root / "source"),
        "interpreter": str(root / "venv/bin/python"),
        "physical_executed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--install", action="store_true")
    mode.add_argument("--verify-installed", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args(argv)
    commit_validated = False
    try:
        require_commit(args.commit)
        commit_validated = True
        if args.build:
            if args.source is None or args.bundle is None:
                raise PreparationBlocked("SOURCE_AND_NEW_BUNDLE_REQUIRED")
            answer = build_bundle(args.source, args.commit, args.bundle)
        elif args.install:
            if args.bundle is None:
                raise PreparationBlocked("VERIFIED_BUNDLE_REQUIRED")
            answer = install_bundle(args.bundle, args.commit)
        else:
            answer = verify_installation(PREFIX / args.commit, args.commit)
        print(json.dumps(answer, sort_keys=True, indent=2))
        return 0
    except Exception as error:
        reason = str(error) if isinstance(error, PreparationBlocked) else "PREPARATION_UNAVAILABLE"
        if re.fullmatch(r"[A-Z][A-Z0-9_]+", reason) is None:
            reason = "PREPARATION_UNAVAILABLE"
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "reason": reason,
                    "physical_executed": False,
                    "preserved_installation": str(PREFIX / args.commit)
                    if args.install and commit_validated
                    else None,
                    "preserved_bundle": str(args.bundle)
                    if args.build and args.bundle is not None
                    else None,
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
