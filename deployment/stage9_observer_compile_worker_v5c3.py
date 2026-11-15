"""Fixed-path compiler for a separately qualified, disposable offline VM.

This file is copied verbatim into a pinned tool image and invoked with the
exact target CPython using ``-I -S -B``. Its JSON output is GUEST_REPORTED,
never independent execution evidence. The controller must qualify isolation,
interpreter/loader/library/import bytes and immutable inputs before launching
anything. This module neither creates nor qualifies a VM.

Only py_compile compiles the supplied source; the source and generated PYC are
never imported or executed. The private compiler function is also exercised
with trusted host Python and synthetic inputs; those tests cannot qualify the
target interpreter or reconstruction.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import py_compile
import re
import stat
import sys
from typing import Any, cast

REQUEST_PROFILE = "stage9-observer-compile-request-v5c3"
REPORT_PROFILE = "stage9-observer-compile-report-v5c3"
REQUEST_PATH = Path("/stage9/input/request.json")
SOURCE_PATH = Path("/stage9/input/source.py")
OUTPUT_PATH = Path("/stage9/output/reconstructed.pyc")
TOOL_PATH = "/stage9/tool/compile_worker.py"
TARGET_INTERPRETER_SHA256 = "e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f"
TARGET_MAGIC = bytes.fromhex("cb0d0d0a")
MAX_REQUEST_BYTES = 16384
MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_OUTPUT_BYTES = 32 * 1024 * 1024
_REQUEST_KEYS = {
    "profile",
    "run_id",
    "nonce",
    "source_sha256",
    "source_size",
    "source_hash_hex",
    "expected_pyc_sha256",
    "co_filename",
    "optimization",
    "invalidation",
}


class WorkerRefusal(Exception):
    """A bounded category suitable for the untrusted guest report."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise WorkerRefusal(code)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, "REQUEST_REJECTED")
        result[key] = value
    return result


def _hex(value: object, length: int) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value) is not None


def _request(raw: bytes) -> dict[str, Any]:
    _require(type(raw) is bytes and len(raw) <= MAX_REQUEST_BYTES, "REQUEST_REJECTED")
    try:
        request = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise WorkerRefusal("REQUEST_REJECTED") from exc
    _require(type(request) is dict and set(request) == _REQUEST_KEYS, "REQUEST_REJECTED")
    _require(request["profile"] == REQUEST_PROFILE, "REQUEST_REJECTED")
    run_id = request["run_id"]
    _require(
        type(run_id) is str
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", run_id) is not None,
        "REQUEST_REJECTED",
    )
    for key in ("source_sha256", "expected_pyc_sha256", "nonce"):
        _require(_hex(request[key], 64), "REQUEST_REJECTED")
    _require(_hex(request["source_hash_hex"], 16), "REQUEST_REJECTED")
    _require(
        type(request["source_size"]) is int and 0 <= request["source_size"] <= MAX_SOURCE_BYTES,
        "REQUEST_REJECTED",
    )
    _require(
        type(request["optimization"]) is int and 0 <= request["optimization"] <= 2,
        "REQUEST_REJECTED",
    )
    _require(request["invalidation"] == "checked_hash", "REQUEST_REJECTED")
    name = request["co_filename"]
    _require(
        type(name) is str
        and 0 < len(name) <= 4096
        and name.startswith("/")
        and not name.startswith("//")
        and "\\" not in name
        and "\x00" not in name
        and PurePosixPath(name).as_posix() == name
        and ".." not in PurePosixPath(name).parts,
        "REQUEST_REJECTED",
    )
    return cast(dict[str, Any], request)


def _read_regular(path: Path, maximum: int) -> bytes:
    _require(path.is_absolute() and path.resolve(strict=True) == path, "PATH_REJECTED")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode) and metadata.st_size <= maximum, "FILE_REJECTED")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            content = stream.read(maximum + 1)
        _require(len(content) <= maximum and len(content) == metadata.st_size, "FILE_REJECTED")
        after = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_uid",
            "st_gid",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        _require(
            all(getattr(metadata, key) == getattr(after, key) for key in stable_fields),
            "FILE_CHANGED",
        )
        return content
    finally:
        os.close(descriptor)


def _compile_one(
    request_raw: bytes,
    *,
    source_path: Path,
    output_path: Path,
    expected_magic: bytes,
) -> dict[str, Any]:
    """Compile and measure bytes; the CLI alone fixes target runtime and paths.

    ``expected_magic`` is explicitly supplied so trusted-host synthetic unit
    tests do not pretend their interpreter is the target. The CLI always passes
    TARGET_MAGIC, after its runtime guard. Output on a measured mismatch remains
    available to the external controller as a failed forensic artifact.
    """
    request = _request(request_raw)
    _require(type(expected_magic) is bytes and len(expected_magic) == 4, "MAGIC_REJECTED")
    source = _read_regular(source_path, MAX_SOURCE_BYTES)
    _require(
        len(source) == request["source_size"]
        and hashlib.sha256(source).hexdigest() == request["source_sha256"],
        "SOURCE_MISMATCH",
    )
    _require(
        output_path.is_absolute()
        and output_path.parent.resolve(strict=True) == output_path.parent
        and not os.path.lexists(output_path),
        "OUTPUT_PATH_REJECTED",
    )
    py_compile.compile(
        str(source_path),
        cfile=str(output_path),
        dfile=request["co_filename"],
        doraise=True,
        optimize=request["optimization"],
        invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH,
    )
    _require(_read_regular(source_path, MAX_SOURCE_BYTES) == source, "SOURCE_CHANGED")
    output = _read_regular(output_path, MAX_OUTPUT_BYTES)
    _require(len(output) >= 16, "PYC_HEADER_REJECTED")
    measured_sha256 = hashlib.sha256(output).hexdigest()
    comparison = (
        "MATCH"
        if measured_sha256 == request["expected_pyc_sha256"]
        and output[:4] == expected_magic
        and int.from_bytes(output[4:8], "little") == 3
        and output[8:16].hex() == request["source_hash_hex"]
        else "RECONSTRUCTION_MISMATCH"
    )
    return {
        "profile": REPORT_PROFILE,
        "origin": "GUEST_REPORTED",
        "run_id": request["run_id"],
        "nonce": request["nonce"],
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "output_sha256": measured_sha256,
        "output_size": len(output),
        "magic_hex": output[:4].hex(),
        "flags": int.from_bytes(output[4:8], "little"),
        "source_hash_hex": output[8:16].hex(),
        "comparison": comparison,
    }


def _runtime_guard() -> None:
    _require(sys.argv == [TOOL_PATH], "ARGUMENTS_REJECTED")
    _require(
        sys.flags.isolated == 1 and sys.flags.no_site == 1 and sys.flags.dont_write_bytecode == 1,
        "RUNTIME_FLAGS_REJECTED",
    )
    _require(sys.implementation.name == "cpython", "INTERPRETER_REJECTED")
    _require(sys.version_info[:3] == (3, 12, 3), "INTERPRETER_REJECTED")
    _require(importlib.util.MAGIC_NUMBER == TARGET_MAGIC, "INTERPRETER_REJECTED")
    interpreter = _read_regular(Path(sys.executable).resolve(strict=True), MAX_OUTPUT_BYTES)
    _require(
        hashlib.sha256(interpreter).hexdigest() == TARGET_INTERPRETER_SHA256,
        "INTERPRETER_REJECTED",
    )
    _require(
        dict(os.environ) == {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "UTC"},
        "ENVIRONMENT_REJECTED",
    )


def main() -> int:
    """No command-line bypass for target checks or fixed input/output paths."""
    try:
        _runtime_guard()
        report = _compile_one(
            _read_regular(REQUEST_PATH, MAX_REQUEST_BYTES),
            source_path=SOURCE_PATH,
            output_path=OUTPUT_PATH,
            expected_magic=TARGET_MAGIC,
        )
        code = 0 if report["comparison"] == "MATCH" else 2
    except (KeyboardInterrupt, InterruptedError):
        report = {
            "profile": REPORT_PROFILE,
            "origin": "GUEST_REPORTED",
            "comparison": "REFUSED",
            "reason": "OPERATION_INTERRUPTED",
        }
        code = 2
    except (WorkerRefusal, OSError, ValueError, TypeError, py_compile.PyCompileError) as exc:
        report = {
            "profile": REPORT_PROFILE,
            "origin": "GUEST_REPORTED",
            "comparison": "REFUSED",
            "reason": str(exc) if isinstance(exc, WorkerRefusal) else "COMPILATION_FAILED",
        }
        code = 2
    print(json.dumps(report, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
