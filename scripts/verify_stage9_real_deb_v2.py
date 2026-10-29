"""Measure already downloaded DEBs against separately frozen, unapproved evidence.

Run as ``python -m scripts.verify_stage9_real_deb_v2 --directory PATH`` from the
repository. This orchestration never downloads, extracts or executes artifacts.
It uses the existing Ubuntu adapter and its decoder, not another DEB parser.
Exit 2 means at least one full package inventory remains unsupported/incomplete.
The separate frame result cannot authorize a package or claim V2 completion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from deployment.stage9_observer_contract import ContractRefusal
from deployment.stage9_observer_offline import Limits
from deployment.stage9_observer_ubuntu import (
    UbuntuMember,
    UbuntuPackage,
    _Region,
    _ZstdReader,
    _pread,
    verify_ubuntu_archive,
)

_INVENTORY = (
    Path(__file__).resolve().parents[1]
    / "docs/architecture/evidence/stage9_observer_3a_real_deb_inventory_v2.json"
)
_INVENTORY_SHA256 = "30f1e21955bca2a54768a0da2f897b1fba0423996d8cfb274408e8a83b0136a3"


def _blocked_status() -> dict[str, str]:
    return {
        "V2_COMPLETE": "NO",
        "ARCHIVE_TO_TARGET_METADATA_MAPPING": "NOT_YET_PROVEN",
        "stage_9": "IN_PROGRESS",
        "qualification": "BLOCKED",
        "CONTRACT_APPROVED": "NO",
        "STAGE_3A_COMPLETE": "NO",
        "PRODUCTION_QUALIFIED": "NO",
        "production_status": "PRODUCTION_NO_GO",
        "runtime_attestation": "MISSING",
        "host_pid_namespace": "UNVERIFIED",
    }


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def measure(directory: Path, selected: str | None = None) -> dict[str, Any]:
    """Run read-only measurements; frozen expectations are never rewritten."""
    if sys.platform != "linux":
        raise ContractRefusal("UNSUPPORTED_SOURCE")
    inventory_raw = _INVENTORY.read_bytes()
    if hashlib.sha256(inventory_raw).hexdigest() != _INVENTORY_SHA256:
        raise ValueError("frozen inventory digest mismatch")
    inventory = json.loads(inventory_raw)
    limits = Limits()
    deadline = time.monotonic() + limits.timeout_s
    consumed = 0

    def check() -> None:
        if time.monotonic() >= deadline:
            raise ContractRefusal("LIMIT_EXCEEDED")

    results: list[dict[str, Any]] = []
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        for declaration in inventory["packages"]:
            if selected is not None and declaration["package"] != selected:
                continue
            check()
            filename = Path(declaration["filename"]).name
            fd = os.open(
                filename,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                dir_fd=directory_fd,
            )
            try:
                before = os.fstat(fd)
                if not stat.S_ISREG(before.st_mode) or before.st_size != declaration["size"]:
                    raise ValueError("archive type/size mismatch")
                digest = hashlib.sha256()
                position = 0
                while position < before.st_size:
                    check()
                    block = _pread(fd, min(limits.chunk_bytes, before.st_size - position), position)
                    if not block:
                        raise ValueError("archive ended during full hash")
                    digest.update(block)
                    position += len(block)
                if digest.hexdigest() != declaration["sha256"]:
                    raise ValueError("archive SHA-256 mismatch")
                if _pread(fd, 8, 0) != b"!<arch>\n":
                    raise ValueError("archive magic mismatch")
                frames: list[dict[str, Any]] = []
                for record in declaration["ar"]:
                    header = _pread(fd, 60, record["offset"] - 60)
                    if header != record["header_ascii"].encode("ascii"):
                        raise ValueError("frozen ar header mismatch")
                    if "zstd" not in record:
                        if _pread(fd, record["size"], record["offset"]) != b"2.0\n":
                            raise ValueError("debian-binary mismatch")
                        continue
                    frame_limits = replace(
                        limits, max_unpacked_bytes=limits.max_unpacked_bytes - consumed
                    )
                    region = _Region(
                        fd, record["offset"], record["size"], limits.chunk_bytes, check
                    )
                    stream = _ZstdReader(
                        region, frame_limits, check, memory_bytes=limits.working_memory_bytes
                    )
                    total = 0
                    frame_digest = hashlib.sha256()
                    try:
                        while block := stream.read(limits.chunk_bytes):
                            total += len(block)
                            consumed += len(block)
                            if consumed > limits.max_unpacked_bytes or total > record["size"] * 100:
                                raise ContractRefusal("LIMIT_EXCEEDED")
                            frame_digest.update(block)
                    finally:
                        stream.close()
                    if (
                        total != record["unpacked_bytes"]
                        or frame_digest.hexdigest() != record["unpacked_sha256"]
                    ):
                        raise ValueError("frozen decoded-frame hash/size mismatch")
                    frames.append(
                        {
                            "name": record["name"],
                            "decoded_bytes": total,
                            "sha256": frame_digest.hexdigest(),
                            "result": "PASS",
                        }
                    )
                data = next(
                    record for record in declaration["ar"] if record["name"] == "data.tar.zst"
                )
                expected = tuple(
                    UbuntuMember(
                        record["name"].removeprefix("./"), "", record["sha256"], record["size"]
                    )
                    for record in data["records"]
                    if record["typeflag"] in ("0", "\0")
                )
                package = UbuntuPackage(
                    "observed-unapproved-ubuntu",
                    filename,
                    "unused-unapproved-proof",
                    declaration["package"],
                    declaration["version"],
                    declaration["architecture"],
                    expected,
                )
                refusal: str | None = None
                try:
                    verify_ubuntu_archive(
                        fd,
                        package,
                        {
                            "source_id": package.source_id,
                            "kind": "ubuntu_apt_snapshot",
                            "artifact_size": declaration["size"],
                        },
                        {},
                        limits,
                        check,
                    )
                except ContractRefusal as error:
                    refusal = str(error)
                after = os.fstat(fd)
                binding = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
                if _identity(before) != _identity(after) or _identity(after) != _identity(binding):
                    raise ValueError("archive identity changed during measurement")
                results.append(
                    {
                        "package": declaration["package"],
                        "archive_sha256": declaration["sha256"],
                        "archive_bytes": before.st_size,
                        "full_archive_sha256_and_size": "PASS",
                        "frames": frames,
                        "REAL_DEB_ZSTD_PARSED": "PASS",
                        "zstd_result_scope": "complete control/data compressed frames only; not TAR acceptance",
                        "REAL_DEB_MEMBER_INVENTORY": "UNSUPPORTED"
                        if refusal == "UNSUPPORTED_SOURCE"
                        else ("FAIL" if refusal else "PASS"),
                        "REAL_DEB_BYTE_CONSISTENCY": "INCOMPLETE" if refusal else "MATCH",
                        "adapter_refusal": refusal,
                        "SOURCE_AUTHENTICITY": "UNVERIFIED",
                    }
                )
            finally:
                os.close(fd)
    finally:
        os.close(directory_fd)
    if not results:
        raise ValueError("no declared package selected")
    return {
        "frozen_inventory_sha256": _INVENTORY_SHA256,
        "expectation_classification": inventory["classification"],
        "packages": results,
        "combined_decoded_bytes": consumed,
        **_blocked_status(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument(
        "--package",
        choices=(
            "python3.12-minimal",
            "libpython3.12-minimal",
            "libpython3.12-stdlib",
            "libpython3.12-dev",
        ),
    )
    args = parser.parse_args()
    try:
        result = measure(args.directory, args.package)
    except (OSError, ValueError, ContractRefusal) as error:
        print(
            json.dumps(
                {
                    "REAL_DEB_ZSTD_PARSED": "FAIL",
                    "REAL_DEB_MEMBER_INVENTORY": "FAIL",
                    "REAL_DEB_BYTE_CONSISTENCY": "INCOMPLETE",
                    "SOURCE_AUTHENTICITY": "UNVERIFIED",
                    **_blocked_status(),
                    "error": str(error),
                },
                indent=2,
            )
        )
        return 2
    print(json.dumps(result, indent=2))
    return (
        0
        if all(package["REAL_DEB_MEMBER_INVENTORY"] == "PASS" for package in result["packages"])
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
