"""Compare provisioned DEBs with frozen, unapproved package-only V3 evidence.

No acquisition/update mode, extraction, installation or target code execution.
Exit 0 means only the complete inert package inventory agreed. Production
qualification and source authenticity remain blocked even on that exit code.
Use a separately trusted checkout/interpreter: imports are not sandboxed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

from deployment.stage9_observer_contract import ContractRefusal
from deployment.stage9_observer_offline import Limits, _absolute_directory, _Budget, _snapshot
from deployment.stage9_observer_package_v3 import (
    PackageInventoryResultV3,
    verify_ubuntu_package_inventory_v3,
)
from deployment.stage9_observer_tar_v3 import TarResultV3

_INVENTORY = (
    Path(__file__).resolve().parents[1]
    / "docs/architecture/evidence/stage9_observer_3a_real_deb_inventory_v3.json"
)
_INVENTORY_SHA256 = "aafaae5853b06b70f704a39bc6471cced92207b9ff64055b97a870f60dc10137"


def _require(condition: bool, code: str = "PROVENANCE_INCOMPLETE") -> None:
    if not condition:
        raise ContractRefusal(code)


def _frozen_inventory() -> bytes:
    """Bound and pin the declaration before any archive is opened."""
    budget = _Budget(Limits())
    parent = _absolute_directory(_INVENTORY.parent, budget)
    descriptor: int | None = None
    rebound: int | None = None
    try:
        directory_before = _snapshot(os.fstat(parent))
        descriptor = os.open(
            _INVENTORY.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
            dir_fd=parent,
        )
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "FILE_POLICY_REJECTED")
        _require(before.st_dev == directory_before[0], "FILE_POLICY_REJECTED")
        _require(0 < before.st_size <= 2 * 1024 * 1024, "LIMIT_EXCEEDED")
        raw = bytearray()
        while len(raw) < before.st_size:
            budget.check()
            block = os.read(descriptor, min(65536, before.st_size - len(raw)))
            _require(bool(block))
            raw.extend(block)
        _require(hashlib.sha256(raw).hexdigest() == _INVENTORY_SHA256, "ARTIFACT_MISMATCH")
        _require(
            _snapshot(os.fstat(descriptor)) == _snapshot(before)
            and _snapshot(os.stat(_INVENTORY.name, dir_fd=parent, follow_symlinks=False))
            == _snapshot(before)
            and _snapshot(os.fstat(parent)) == directory_before,
            "FILE_CHANGED",
        )
        rebound = _absolute_directory(_INVENTORY.parent, budget)
        _require(_snapshot(os.fstat(rebound)) == directory_before, "FILE_CHANGED")
        return bytes(raw)
    finally:
        failed = sys.exc_info()[0] is not None
        cleanup_failed = False
        for opened in (rebound, descriptor, parent):
            if opened is not None:
                try:
                    os.close(opened)
                except OSError:
                    cleanup_failed = True
        if cleanup_failed and not failed:
            raise ContractRefusal("PROVENANCE_INCOMPLETE")


def _tar_summary(result: TarResultV3) -> dict[str, Any]:
    return {
        "header_profile": result.profile,
        "inventory": "MATCH",
        "metadata": "MATCH",
        "regular_payloads": "MATCH",
        "literal_symlink_records": "MATCH",
        "complete_eof": "PASS",
        "members": result.members,
        "regular": result.regular,
        "directory": result.directories,
        "symlink": result.symlinks,
        "unpacked_bytes": result.unpacked_bytes,
        "unpacked_sha256": result.sha256,
    }


def _report(result: PackageInventoryResultV3) -> dict[str, Any]:
    report = {
        item.name: getattr(result, item.name)
        for item in fields(result)
        if item.name not in {"packages", "reason_codes"}
    }
    report["reason_codes"] = list(result.reason_codes)
    report["independent_acceptance"] = "NOT_INDEPENDENT_ACCEPTANCE"
    report["SOURCE_AUTHENTICITY"] = result.source_authenticity
    report["packages"] = [
        {
            "package": item.package,
            "archive_sha256": item.archive_sha256,
            "archive_bytes": item.archive_bytes,
            "full_archive_sha256_and_size": "PASS",
            "control": _tar_summary(item.control),
            "data": _tar_summary(item.data),
            "REAL_DEB_MEMBER_INVENTORY": "PASS",
            "REAL_DEB_BYTE_CONSISTENCY": "MATCH",
            "SOURCE_AUTHENTICITY": "UNVERIFIED",
        }
        for item in result.packages
    ]
    return report


def measure(directory: Path) -> dict[str, Any]:
    """Measure all frozen packages in one public operation with shared budgets."""
    if sys.platform != "linux":
        raise ContractRefusal("UNSUPPORTED_SOURCE")
    if os.geteuid() == 0:
        raise ContractRefusal("FILE_POLICY_REJECTED")
    raw = _frozen_inventory()
    # absolute() retains a malicious ancestor for the descriptor walk to refuse.
    result = verify_ubuntu_package_inventory_v3(directory.absolute(), inventory_raw=raw)
    return _report(result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = measure(args.directory)
    except ContractRefusal as error:
        report = _report(PackageInventoryResultV3("INCOMPLETE", (error.code,)))
    except (OSError, ValueError, UnicodeError, MemoryError, OverflowError, RecursionError):
        report = _report(PackageInventoryResultV3("INCOMPLETE", ("PROVENANCE_INCOMPLETE",)))
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report["consistency"] == "MATCH" else 2


if __name__ == "__main__":
    raise SystemExit(main())
