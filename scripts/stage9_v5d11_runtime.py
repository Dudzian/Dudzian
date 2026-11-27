"""Derive and verify the root-owned runtime copy of a sealed Profile A."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import stat
import sys
from typing import Any, cast

PROFILE = "stage9-v5d11-observer-profile-a-runtime-receipt-v1"
DIRECTORY_MODE = 0o550
READ_MODE = 0o440
EXECUTE_MODE = 0o550
_LIMIT = 16 * 1024 * 1024
_STAGE_NAME = "root"


class Refusal(ValueError):
    """A source binding, projection or protected path was refused."""


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise Refusal(reason)


def file_mode(materialized_mode: int) -> int:
    if materialized_mode == 0o755:
        return EXECUTE_MODE
    require(materialized_mode in {0o600, 0o644}, "RUNTIME_SOURCE_MODE_REJECTED")
    return READ_MODE


def _ids(uid: Any, gid: Any) -> None:
    require(
        all(
            type(identifier) is int and 0 < identifier < (1 << 32) - 1 for identifier in (uid, gid)
        ),
        "RUNTIME_ACCOUNT_REJECTED",
    )


def _source_paths(
    source: dict[str, Any], source_sha: str, contract: dict[str, Any]
) -> dict[str, Any]:
    return {
        "directories": {name: DIRECTORY_MODE for name in source["directories"]},
        "files": {
            **{
                name: {"sha256": row["sha256"], "mode": file_mode(row["mode"])}
                for name, row in source["files"].items()
            },
            contract["RECEIPT_NAME"]: {"sha256": source_sha, "mode": READ_MODE},
        },
    }


def _pinned_json(raw: bytes, expected: str, contract: dict[str, Any]) -> dict[str, Any]:
    require(
        contract["sha256_value"](expected) and contract["digest"](raw) == expected,
        "RUNTIME_SOURCE_PIN_MISMATCH",
    )
    return cast(dict[str, Any], contract["loads"](raw))


def derive(
    receipt_raw: bytes,
    receipt_sha256: str,
    approval_raw: bytes,
    approval_sha256: str,
    uid: int,
    gid: int,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Produce a review candidate from exact independently selected V5d10 inputs."""
    _ids(uid, gid)
    source = _pinned_json(receipt_raw, receipt_sha256, contract)
    approval = _pinned_json(approval_raw, approval_sha256, contract)
    contract["validate_receipt"](source)
    contract["validate_approval"](approval)
    require(
        source["approval_sha256"] == approval_sha256
        and source["host"] == approval["host"]
        and source["base_runtime"] == approval["base_runtime"]
        and all(source[key] == approval[key] for key in contract["APPROVAL_DIGESTS"])
        and all(
            source["external_files"].get(name) == sha
            for name, sha in approval["host_tools"].items()
        ),
        "RUNTIME_APPROVAL_BINDING_REJECTED",
    )
    interpreter = source["files"].get("venv/bin/python3.12", {})
    require(
        interpreter.get("mode") == 0o755
        and interpreter.get("sha256") == source["host"]["interpreter_sha256"],
        "RUNTIME_INTERPRETER_BINDING_REJECTED",
    )
    plan = _source_paths(source, receipt_sha256, contract)
    native: dict[str, str] = {}
    for key, sha in approval["wheel_native"].items():
        wheel, separator, member = key.partition(":")
        require(
            bool(separator)
            and wheel.endswith(".whl")
            and "/" not in wheel
            and contract["path_value"](member, absolute=False)
            and member not in native,
            "RUNTIME_CODE_BINDING_REJECTED",
        )
        native[member] = sha
    code_pins = {
        **approval["code"],
        **native,
        **{name: row["sha256"] for name, row in approval["bindings"].items()},
    }
    require(
        len(code_pins)
        == len(approval["code"]) + len(approval["wheel_native"]) + len(approval["bindings"])
        and all(
            plan["files"].get(f"packages/{name}", {}).get("sha256") == sha
            for name, sha in code_pins.items()
        ),
        "RUNTIME_CODE_BINDING_REJECTED",
    )
    return {
        "profile": PROFILE,
        "root": source["root"],
        "owner_uid": 0,
        "uid": uid,
        "gid": gid,
        "materialization_receipt_sha256": receipt_sha256,
        "materialization_approval_sha256": approval_sha256,
        "materialization_file_plan_sha256": source["file_plan_sha256"],
        "source_code_pins_sha256": contract["digest"](contract["canonical"](approval["code"])),
        "runtime_file_plan_sha256": contract["digest"](contract["canonical"](plan)),
        "authorization": "NOT_AUTHORIZED",
        **plan,
    }


def candidate_from_manifest(manifest: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    for name in ("root", "approval"):
        require(
            contract["path_value"](manifest[name], absolute=True), "RUNTIME_SOURCE_PATH_REJECTED"
        )
    return derive(
        contract["secure_read"](Path(manifest["root"]) / contract["RECEIPT_NAME"], limit=_LIMIT),
        manifest["receipt_sha256"],
        contract["secure_read"](Path(manifest["approval"]), limit=_LIMIT),
        manifest["approval_sha256"],
        manifest["uid"],
        manifest["gid"],
        contract,
    )


def _projection(
    view: Path, document: dict[str, Any], expected_sha: str, contract: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    require(
        contract["sha256_value"](expected_sha)
        and contract["digest"](contract["canonical"](document)) == expected_sha,
        "RUNTIME_RECEIPT_PIN_MISMATCH",
    )
    digest_keys = {
        "materialization_receipt_sha256",
        "materialization_approval_sha256",
        "materialization_file_plan_sha256",
        "source_code_pins_sha256",
        "runtime_file_plan_sha256",
    }
    require(
        set(document)
        == {
            "profile",
            "root",
            "owner_uid",
            "uid",
            "gid",
            "authorization",
            "files",
            "directories",
            *digest_keys,
        }
        and document["profile"] == PROFILE
        and type(document["owner_uid"]) is int
        and document["owner_uid"] == 0
        and document["authorization"] == "NOT_AUTHORIZED"
        and contract["path_value"](document["root"], absolute=True)
        and all(contract["sha256_value"](document[key]) for key in digest_keys),
        "RUNTIME_RECEIPT_REJECTED",
    )
    _ids(document["uid"], document["gid"])
    require(view.is_absolute() and ".." not in view.parts, "RUNTIME_VIEW_PATH_REJECTED")
    root = view / document["root"][1:]
    gid = document["gid"]
    for parent in (item for item in root.parents if item == view or item.is_relative_to(view)):
        info = parent.lstat()
        visibility = info.st_mode >> (3 if info.st_gid == gid else 0)
        require(
            stat.S_ISDIR(info.st_mode)
            and info.st_uid == 0
            and not info.st_mode & 0o022
            and visibility & 0o5 == 0o5,
            "RUNTIME_ANCESTOR_NOT_PROTECTED",
        )
    raw = contract["secure_read"](root / contract["RECEIPT_NAME"], limit=_LIMIT)
    source = _pinned_json(raw, document["materialization_receipt_sha256"], contract)
    contract["validate_receipt"](source)
    plan = _source_paths(source, document["materialization_receipt_sha256"], contract)
    require(
        document["root"] == source["root"]
        and document["materialization_approval_sha256"] == source["approval_sha256"]
        and document["materialization_file_plan_sha256"] == source["file_plan_sha256"]
        and document["runtime_file_plan_sha256"] == contract["digest"](contract["canonical"](plan))
        and all(document[name] == value for name, value in plan.items()),
        "RUNTIME_PROJECTION_MISMATCH",
    )
    return root, source


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


def _inventory(
    root: Path,
    dirs: dict[str, int],
    files: dict[str, dict[str, Any]],
    uid: int,
    gid: int,
    contract: dict[str, Any],
) -> dict[Path, tuple[int, ...]]:
    """Match the exact tree, and hold each entry to one identity across reads."""
    device = root.parent.lstat().st_dev
    pending = [root]
    objects: dict[Path, tuple[int, ...]] = {}
    while pending:
        path = pending.pop()
        name = path.relative_to(root).as_posix()
        before = path.lstat()
        directory = name in dirs
        row = files.get(name)
        require(directory or row is not None, "RUNTIME_UNEXPECTED_PATH")
        if directory:
            require(stat.S_ISDIR(before.st_mode), "RUNTIME_DIRECTORY_REJECTED")
            expected_mode = dirs[name]
            pending.extend(path.iterdir())
            require(len(pending) + len(objects) < len(files) + len(dirs), "RUNTIME_EXTRA_PATHS")
        else:
            assert row is not None
            require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "RUNTIME_FILE_REJECTED")
            expected_mode = row["mode"]
            raw = contract["secure_read"](path)
            require(contract["digest"](raw) == row["sha256"], "RUNTIME_FILE_PIN_MISMATCH")
        require(
            before.st_dev == device
            and before.st_uid == uid
            and before.st_gid == gid
            and stat.S_IMODE(before.st_mode) == expected_mode,
            "RUNTIME_DAC_MISMATCH",
        )
        objects[path] = _identity(before)
    require(len(objects) == len(files) + len(dirs), "RUNTIME_PATHS_MISSING")
    require(
        len({observed[:2] for observed in objects.values()}) == len(objects)
        and all(_identity(path.lstat()) == observed for path, observed in objects.items()),
        "RUNTIME_PATH_CHANGED",
    )
    return objects


def apply_tree(
    view: Path, document: dict[str, Any], expected_sha: str, contract: dict[str, Any]
) -> None:
    """Apply exact DAC only to a protected, fresh stage copy before pivoting."""
    require(
        sys.platform == "linux"
        and getattr(os, "geteuid", lambda: -1)() == 0
        and view.is_absolute()
        and view.name == _STAGE_NAME
        and all(not path.is_symlink() for path in (view, *view.parents)),
        "RUNTIME_PRIVATE_STAGE_REQUIRED",
    )
    parent = view.parent.lstat()
    require(
        stat.S_ISDIR(parent.st_mode)
        and stat.S_IMODE(parent.st_mode) == 0o700
        and parent.st_uid == 0,
        "RUNTIME_PRIVATE_STAGE_REQUIRED",
    )
    root, source = _projection(view, document, expected_sha, contract)
    original = Path(source["root"])
    require(
        not root.is_relative_to(original) and not original.is_relative_to(view),
        "RUNTIME_MATERIALIZATION_OVERLAP",
    )
    source_files = {
        **source["files"],
        contract["RECEIPT_NAME"]: {
            "sha256": document["materialization_receipt_sha256"],
            "mode": 0o600,
        },
    }
    objects = _inventory(
        root, source["directories"], source_files, document["uid"], document["gid"], contract
    )
    # A second pathname could refer to the materialized inode through a bind
    # mount. Refuse before the first ownership change, including inside subtrees.
    original_inodes = {
        _identity((original / path.relative_to(root)).lstat())[:2] for path in objects
    }
    require(
        not {observed[:2] for observed in objects.values()} & original_inodes,
        "RUNTIME_MATERIALIZATION_OVERLAP",
    )
    set_owner = getattr(os, "fchown", None)
    set_mode = getattr(os, "fchmod", None)
    require(callable(set_owner) and callable(set_mode), "RUNTIME_DAC_UNAVAILABLE")
    assert set_owner is not None and set_mode is not None
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    for path, observed in sorted(
        objects.items(), key=lambda entry: len(entry[0].parts), reverse=True
    ):
        name = path.relative_to(root).as_posix()
        mode = document["directories"].get(name, document["files"].get(name, {}).get("mode"))
        fd = os.open(path, flags)
        try:
            require(_identity(os.fstat(fd)) == observed, "RUNTIME_PATH_CHANGED")
            set_owner(fd, 0, document["gid"])
            set_mode(fd, mode)
        finally:
            os.close(fd)
    verify_tree(view, document, expected_sha, contract)


def verify_tree(
    view: Path, document: dict[str, Any], expected_sha: str, contract: dict[str, Any]
) -> None:
    root, _ = _projection(view, document, expected_sha, contract)
    _inventory(root, document["directories"], document["files"], 0, document["gid"], contract)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("review",))
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        contract = runpy.run_path(str(Path(__file__).with_name("stage9_v5d10_contract.py")))
        require(args.manifest.is_absolute(), "RUNTIME_SOURCE_PATH_REJECTED")
        manifest = contract["loads"](contract["secure_read"](args.manifest, limit=_LIMIT))
        candidate = candidate_from_manifest(manifest, contract)
        sys.stdout.buffer.write(contract["canonical"](candidate))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        reason = str(exc) if isinstance(exc, Refusal) else "RUNTIME_INPUT_REJECTED"
        print(f"REFUSED {reason}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
