"""Read-only V5d6 wheelhouse diagnostic; measured candidates never grant approval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
from typing import Any
import zipfile


EXPECTED = {
    "WHEELS_CHECKED": 69,
    "EXPECTED_SHARED_PATHS": 62,
    "EXPECTED_FOREIGN_PYC_OMISSIONS": 2,
    "EXPECTED_HEADER_OMISSIONS": 1,
}


def inspect(pyproject: Path, wheelhouse: Path, origins: dict[str, Any]) -> dict[str, Any]:
    """Retain exact refusal codes and archive digests without writing to inputs."""
    deps = runpy.run_path(str(Path(__file__).with_name("stage9_v5d4_dependencies.py")))
    headers = runpy.run_path(str(Path(__file__).with_name("stage9_v5d6_headers.py")))
    report: dict[str, Any] = {
        "profile": "stage9-v5d6-wheelhouse-static-inspection-v1",
        "status": "REFUSED",
        "expected_counts": EXPECTED,
        "counts": {
            "WHEELS_CHECKED": 0,
            "EXPECTED_SHARED_PATHS": 0,
            "EXPECTED_FOREIGN_PYC_OMISSIONS": 0,
            "EXPECTED_HEADER_OMISSIONS": 0,
        },
        "archives": [],
        "refusals": [],
        "authorization": "NOT_AUTHORIZED",
        "independent_approvals": "NOT_SUPPLIED_NOT_GENERATED",
        "STAGE_9": "IN_PROGRESS",
        "OBSERVER_3A": "BLOCKED",
        "STAGE_3B": "NOT_AUTHORIZED",
        "PRODUCTION": "NO-GO",
    }
    packages: list[dict[str, Any]] = []
    try:
        deps["require"](not wheelhouse.is_symlink() and wheelhouse.is_dir(), "WHEELHOUSE_DIRECTORY")
        paths = sorted(wheelhouse.iterdir())
        for path in paths:
            row: dict[str, Any] = {"filename": path.name, "status": "REFUSED"}
            report["archives"].append(row)
            try:
                deps["require"](path.suffix == ".whl", "NON_WHEEL_DISTRIBUTION")
                data = deps["read_regular"](path)
                row["sha256"] = deps["digest"](data)
                row["size"] = len(data)
                report["counts"]["WHEELS_CHECKED"] += 1
                package = deps["inspect_wheel_bytes"](data, path.name, origins.get(path.name))
                packages.append(package)
                row.update(
                    status="STATIC_CANDIDATE_INSPECTED",
                    distribution=package["name"],
                    version=package["version"],
                    foreign_pyc_omissions=package["omitted_members"],
                    header_omissions=package["header_omissions"],
                )
            except (ValueError, OSError, zipfile.BadZipFile, KeyError, TypeError) as exc:
                row["reason"] = str(exc)
                report["refusals"].append({"filename": path.name, "reason": str(exc)})
        report["counts"]["EXPECTED_FOREIGN_PYC_OMISSIONS"] = sum(
            len(package["omitted_members"]) for package in packages
        )
        report["counts"]["EXPECTED_HEADER_OMISSIONS"] = sum(
            len(package["header_omissions"]) for package in packages
        )
        deps["require"](not report["refusals"], "WHEELHOUSE_MEMBER_REFUSALS_RETAINED")
        report["shared_wheel_members"] = deps["shared_inventory"](packages)
        report["header_omissions"] = headers["header_inventory"](packages)
        report["counts"]["EXPECTED_SHARED_PATHS"] = len(report["shared_wheel_members"]["members"])
        deps["require"](report["counts"] == EXPECTED, "WHEELHOUSE_EXPECTED_COUNTS_MISMATCH")
        proposal = deps["propose"](pyproject, wheelhouse, origins, allow_shared_candidates=True)
        report["proposal_sha256"] = deps["digest"](deps["canonical"](proposal))
        report["status"] = "STATIC_CANDIDATES_INSPECTED_REQUIRES_INDEPENDENT_APPROVAL"
    except (ValueError, OSError, zipfile.BadZipFile, KeyError, TypeError) as exc:
        report["refusals"].append({"reason": str(exc)})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--origins", type=Path, required=True)
    args = parser.parse_args()
    deps = runpy.run_path(str(Path(__file__).with_name("stage9_v5d4_dependencies.py")))
    try:
        origins = json.loads(deps["read_regular"](args.origins).decode("utf-8"))
        deps["require"](type(origins) is dict, "ORIGIN_SCHEMA")
        result = inspect(args.pyproject, args.wheelhouse, origins)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        result = {"status": "REFUSED", "reason": str(exc), "authorization": "NOT_AUTHORIZED"}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 2 if result["status"] == "REFUSED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
