"""Closed schemas and byte inventory for the Stage 9 Observer preflight profile."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, cast

PROPOSAL = "stage9-v5d10-observer-preflight-proposal-v1"
INPUTS = "stage9-v5d10-observer-preflight-inputs-v1"
APPROVAL = "stage9-v5d10-observer-preflight-approval-v1"
RECEIPT = "stage9-v5d10-observer-preflight-receipt-v1"
ENTRYPOINT = "scripts.stage9_observer_kvm_preflight_v5d1"
RECEIPT_NAME = "ENVIRONMENT.json"
HOST = {
    "python_full_version": "3.12.3",
    "implementation": "CPython",
    "machine": "x86_64",
    "system": "Linux",
    "soabi": "cpython-312-x86_64-linux-gnu",
    "distribution": "ubuntu",
    "distribution_version": "24.04",
    "glibc": "glibc 2.39",
}
BINDINGS = ("libvirt.py", "libvirtmod.cpython-312-x86_64-linux-gnu.so")
QEMU_IMG = "/usr/bin/qemu-img"
APPROVAL_DIGESTS = frozenset(
    {
        "proposal_sha256",
        "dynamic_review_sha256",
        "abi_request_sha256",
        "abi_expectations_sha256",
        "file_plan_sha256",
    }
)
OMISSION_FIELDS = frozenset(
    {
        "wheel_omissions",
        "wheel_shared_members",
        "wheel_header_omissions",
        "wheel_artifact_omissions",
    }
)
LIMIT = 512 * 1024 * 1024


class Refusal(ValueError):
    """The claimed preflight environment did not satisfy its contract."""


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise Refusal(reason)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def loads(data: bytes) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, "DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    def bad_constant(_: str) -> Any:
        raise Refusal("INVALID_JSON_CONSTANT")

    value = json.loads(data, object_pairs_hook=unique, parse_constant=bad_constant)
    require(type(value) is dict, "JSON_OBJECT_REQUIRED")
    return cast(dict[str, Any], value)


def sha256_value(value: Any) -> bool:
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def path_value(value: Any, *, absolute: bool) -> bool:
    if type(value) is not str or not value or "\x00" in value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return (
        path.is_absolute() == absolute
        and (not absolute or path.anchor == "/")
        and path.as_posix() == value
        and value not in {".", "/"}
        and ".." not in path.parts
    )


def _schema(doc: Any, profile: str, fields: set[str] | frozenset[str]) -> None:
    require(type(doc) is dict and set(doc) == {"profile", *fields}, "PROFILE_SCHEMA_REJECTED")
    require(doc["profile"] == profile, "PROFILE_ID_REJECTED")


def _hash_map(value: Any, *, absolute: bool) -> None:
    require(
        type(value) is dict
        and bool(value)
        and all(
            path_value(key, absolute=absolute) and sha256_value(item) for key, item in value.items()
        ),
        "PIN_MAP_REJECTED",
    )


def _host(value: Any) -> None:
    require(
        type(value) is dict
        and set(value) == {*HOST, "interpreter_sha256"}
        and all(value[key] == expected for key, expected in HOST.items())
        and sha256_value(value["interpreter_sha256"]),
        "HOST_CONTRACT_REJECTED",
    )


def _base_runtime(value: Any) -> None:
    _schema(
        value,
        "stage9-v5d4-base-runtime-inventory-v1",
        {"roots", "files", "native", "symlinks", "excluded", "scope", "installation_approved"},
    )
    _hash_map(value["files"], absolute=True)
    for field in ("roots", "native", "excluded"):
        paths = value[field]
        require(
            type(paths) is list and all(path_value(path, absolute=True) for path in paths),
            "BASE_PATHS_REJECTED",
        )
        require(paths == sorted(set(paths)), "BASE_PATHS_REJECTED")
    require(
        bool(value["roots"]) and set(value["native"]) <= set(value["files"]), "BASE_PATHS_REJECTED"
    )
    require(
        value["scope"] == "STATIC_BYTES_SUPERSET_NOT_DYNAMIC_COMPLETENESS"
        and value["installation_approved"] is False,
        "BASE_SCOPE_REJECTED",
    )
    links = value["symlinks"]
    require(type(links) is dict, "BASE_LINKS_REJECTED")
    for path, row in links.items():
        require(
            path_value(path, absolute=True)
            and type(row) is dict
            and set(row) == {"target", "resolved", "sha256"}
            and type(row["target"]) is str
            and bool(row["target"])
            and "\x00" not in row["target"]
            and path_value(row["resolved"], absolute=True)
            and sha256_value(row["sha256"])
            and value["files"].get(row["resolved"]) == row["sha256"],
            "BASE_LINKS_REJECTED",
        )


def validate_inputs(doc: Any) -> None:
    fields = {
        "repository",
        "wheelhouse",
        "origins",
        "proposal",
        "abi_request",
        "abi_expectations",
        "dynamic_review",
    }
    _schema(doc, INPUTS, fields)
    require(all(path_value(doc[key], absolute=True) for key in fields), "INPUT_PATH_REJECTED")


def validate_approval(doc: Any) -> None:
    _schema(
        doc,
        APPROVAL,
        APPROVAL_DIGESTS
        | OMISSION_FIELDS
        | {"host", "code", "wheel_native", "bindings", "base_runtime", "host_tools"},
    )
    require(all(sha256_value(doc[key]) for key in APPROVAL_DIGESTS), "APPROVAL_DIGEST_REJECTED")
    require(all(doc[key] == [] for key in OMISSION_FIELDS), "A_WHEEL_EXCEPTIONS_FORBIDDEN")
    _host(doc["host"])
    _base_runtime(doc["base_runtime"])
    _hash_map(doc["code"], absolute=False)
    _hash_map(doc["wheel_native"], absolute=False)
    _hash_map(doc["host_tools"], absolute=True)
    require(set(doc["host_tools"]) == {QEMU_IMG}, "HOST_TOOLS_REJECTED")
    bindings = doc["bindings"]
    require(type(bindings) is dict and set(bindings) == set(BINDINGS), "BINDINGS_REJECTED")
    for name, row in bindings.items():
        require(
            type(row) is dict
            and set(row) == {"path", "sha256"}
            and row["path"] == f"/usr/lib/python3/dist-packages/{name}"
            and sha256_value(row["sha256"]),
            "BINDINGS_REJECTED",
        )


def validate_plan(plan: Any) -> None:
    require(type(plan) is dict and set(plan) == {"files", "directories"}, "FILE_PLAN_REJECTED")
    files, dirs = plan["files"], plan["directories"]
    require(
        type(files) is dict and bool(files) and type(dirs) is dict and dirs.get(".") == 0o700,
        "FILE_PLAN_REJECTED",
    )
    for path, mode in dirs.items():
        require(
            (path == "." or path_value(path, absolute=False))
            and type(mode) is int
            and mode == 0o700,
            "DIRECTORY_PLAN_REJECTED",
        )
    for path, row in files.items():
        require(
            path_value(path, absolute=False)
            and path != RECEIPT_NAME
            and path not in dirs
            and type(row) is dict
            and set(row) == {"sha256", "mode"}
            and sha256_value(row["sha256"])
            and type(row["mode"]) is int
            and row["mode"] in {0o600, 0o644, 0o755},
            "FILE_PLAN_REJECTED",
        )
    require(
        all(PurePosixPath(path).parent.as_posix() in dirs for path in {*files, *dirs}),
        "FILE_PLAN_PARENT_MISSING",
    )


def plan_sha256(plan: Any) -> str:
    validate_plan(plan)
    return digest(canonical(plan))


def validate_receipt(doc: Any) -> None:
    _schema(
        doc,
        RECEIPT,
        APPROVAL_DIGESTS
        | {
            "root",
            "entrypoint",
            "approval_sha256",
            "files",
            "directories",
            "host",
            "base_runtime",
            "external_files",
            "authorization",
            "native_load",
        },
    )
    require(
        path_value(doc["root"], absolute=True) and doc["entrypoint"] == ENTRYPOINT,
        "RECEIPT_SCOPE_REJECTED",
    )
    require(
        all(sha256_value(doc[key]) for key in {*APPROVAL_DIGESTS, "approval_sha256"}),
        "RECEIPT_DIGEST_REJECTED",
    )
    require(
        doc["authorization"] == "NOT_AUTHORIZED" and doc["native_load"] == "NOT_EXECUTED",
        "RECEIPT_AUTHORIZATION_REJECTED",
    )
    require(
        doc["file_plan_sha256"]
        == plan_sha256({"files": doc["files"], "directories": doc["directories"]}),
        "RECEIPT_PLAN_REJECTED",
    )
    _host(doc["host"])
    _base_runtime(doc["base_runtime"])
    _hash_map(doc["external_files"], absolute=True)
    require(
        QEMU_IMG in doc["external_files"]
        and not any(Path(path).is_relative_to(doc["root"]) for path in doc["external_files"]),
        "RECEIPT_EXTERNAL_PINS_REJECTED",
    )


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_uid,
        info.st_gid,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def secure_read(path: Path, *, limit: int = LIMIT) -> bytes:
    no_follow, nonblock, directory_flag = (
        getattr(os, name, None) for name in ("O_NOFOLLOW", "O_NONBLOCK", "O_DIRECTORY")
    )
    opener = os.open
    if (
        not all(type(flag) is int and flag > 0 for flag in (no_follow, nonblock, directory_flag))
        or opener not in os.supports_dir_fd
    ):
        raise Refusal("SECURE_READER_UNAVAILABLE")
    assert no_follow is not None and nonblock is not None and directory_flag is not None
    path = path.absolute()
    require(path_value(str(path), absolute=True), "SECURE_READ_PATH_REJECTED")
    root_flags = os.O_RDONLY | directory_flag | no_follow
    descriptor = opener(path.anchor, root_flags)
    try:
        for component in path.parts[1:-1]:
            child = opener(component, root_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        fd = opener(path.name, os.O_RDONLY | no_follow | nonblock, dir_fd=descriptor)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            require(
                stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= limit,
                "SECURE_READ_OBJECT_REJECTED",
            )
            data = stream.read(limit + 1)
            require(
                len(data) == before.st_size
                and _identity(os.fstat(stream.fileno())) == _identity(before),
                "SECURE_READ_CHANGED",
            )
        require(_identity(path.lstat()) == _identity(before), "SECURE_READ_CHANGED")
        return data
    finally:
        os.close(descriptor)


def inventory(root: Path, *, exclude_receipt: bool = False) -> dict[str, Any]:
    root = root.absolute()
    require(not any(item.is_symlink() for item in (root, *root.parents)), "PLAN_SYMLINK_REJECTED")
    files: dict[str, dict[str, Any]] = {}
    dirs: dict[str, int] = {}
    seen: dict[Path, tuple[int, ...]] = {}
    pending = [root]
    while pending:
        path = pending.pop()
        before = path.lstat()
        name = path.relative_to(root).as_posix()
        if stat.S_ISDIR(before.st_mode):
            dirs[name] = stat.S_IMODE(before.st_mode)
            seen[path] = _identity(before)
            pending.extend(sorted(path.iterdir(), reverse=True))
        else:
            data = secure_read(path)
            require(_identity(path.lstat()) == _identity(before), "FILE_PLAN_CHANGED")
            if not (exclude_receipt and name == RECEIPT_NAME):
                files[name] = {"sha256": digest(data), "mode": stat.S_IMODE(before.st_mode)}
        require(len(files) + len(dirs) <= 100_000, "FILE_PLAN_LIMIT")
    require(
        all(_identity(path.lstat()) == before for path, before in seen.items()), "FILE_PLAN_CHANGED"
    )
    result = {"files": files, "directories": dirs}
    validate_plan(result)
    return result
