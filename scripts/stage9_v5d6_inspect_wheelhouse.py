"""Read-only V5d6 wheelhouse diagnostic; measured candidates never grant approval."""

from __future__ import annotations

import argparse
import csv
import json
import runpy
import zipfile
import zlib
from pathlib import Path
from typing import Any

EXPECTED = {
    "WHEELS_CHECKED": 69,
    "EXPECTED_SHARED_PATHS": 62,
    "EXPECTED_FOREIGN_PYC_OMISSIONS": 2,
    "EXPECTED_HEADER_OMISSIONS": 1,
}
EXPECTED_KEYS = {
    "WHEELS_CHECKED": "WHEELS_CHECKED",
    "SHARED_PATHS": "EXPECTED_SHARED_PATHS",
    "FOREIGN_PYC_OMISSIONS": "EXPECTED_FOREIGN_PYC_OMISSIONS",
    "HEADER_OMISSIONS": "EXPECTED_HEADER_OMISSIONS",
}
INSPECTION_ERRORS = (
    ValueError,
    OSError,
    zipfile.BadZipFile,
    KeyError,
    TypeError,
    csv.Error,
    zlib.error,
)


def dependency_tools() -> dict[str, Any]:
    return runpy.run_path(str(Path(__file__).with_name("stage9_v5d4_dependencies.py")))


def measurement(value: int, wheels_evaluated: int, complete: bool) -> dict[str, Any]:
    if not wheels_evaluated:
        return {"status": "NOT_TESTED", "value": None, "wheels_evaluated": 0}
    return {
        "status": "MEASURED" if complete else "PARTIAL",
        "value": value,
        "wheels_evaluated": wheels_evaluated,
    }


def empty_report() -> dict[str, Any]:
    return {
        "profile": "stage9-v5d6-wheelhouse-static-inspection-v2",
        "status": "REFUSED",
        "expected_counts": dict(EXPECTED),
        "measurements": {name: measurement(0, 0, False) for name in EXPECTED_KEYS},
        "archive_scan": {
            "entries_listed": None,
            "bytes_read": 0,
            "wheels_read": 0,
            "wheels_fully_inspected": 0,
        },
        "closure_status": "NOT_TESTED",
        "archives": [],
        "refusals": [],
        "shared_wheel_members": None,
        "shared_wheel_members_scope": None,
        "header_omissions": None,
        "authorization": "NOT_AUTHORIZED",
        "independent_approvals": "NOT_SUPPLIED_NOT_GENERATED",
        "STAGE_9": "IN_PROGRESS",
        "OBSERVER_3A": "BLOCKED",
        "STAGE_3B": "NOT_AUTHORIZED",
        "PRODUCTION": "NO-GO",
    }


def inspect_sharing(
    deps: dict[str, Any],
    report: dict[str, Any],
    maps: list[dict[str, Any]],
    complete_inputs: bool,
) -> None:
    # These validated maps may come from wheels refused by a later member check.
    # The scope and refusal stay visible, and the maps never feed the proposal.
    if not maps:
        return
    scope = "AVAILABLE_MATERIALIZATION_MAPS"
    observed_wheels = len(maps)
    try:
        shared = deps["shared_inventory"](maps)
    except INSPECTION_ERRORS as exc:
        report["refusals"].append({"check": "shared_inventory", "reason": str(exc)})
        pinned = set(deps["shared_tools"]()["WHEELS"])
        pair = [item for item in maps if item["name"] in pinned]
        if len(pair) != len(pinned) or {item["name"] for item in pair} != pinned or pair == maps:
            return
        try:
            shared = deps["shared_inventory"](pair)
        except INSPECTION_ERRORS as pair_exc:
            report["refusals"].append(
                {"check": "shared_inventory_pinned_pair", "reason": str(pair_exc)}
            )
            return
        scope = "PINNED_WHEEL_PAIR"
        observed_wheels = len(pair)
    report["shared_wheel_members"] = shared
    report["shared_wheel_members_scope"] = scope
    report["measurements"]["SHARED_PATHS"] = measurement(
        len(shared["members"]),
        observed_wheels,
        complete_inputs and observed_wheels == report["archive_scan"]["wheels_read"],
    )


def inspect(pyproject: Path, wheelhouse: Path, origins: dict[str, Any]) -> dict[str, Any]:
    """Retain completed checks and exact refusals as separate diagnostic evidence."""
    deps = dependency_tools()
    report = empty_report()
    scan = report["archive_scan"]
    maps: list[dict[str, Any]] = []
    try:
        deps["require"](not wheelhouse.is_symlink() and wheelhouse.is_dir(), "WHEELHOUSE_DIRECTORY")
        paths = sorted(wheelhouse.iterdir())
        scan["entries_listed"] = len(paths)
        for path in paths:
            observations: dict[str, Any] = {}
            row: dict[str, Any] = {
                "filename": path.name,
                "status": "REFUSED",
                "observations": observations,
            }
            report["archives"].append(row)
            try:
                deps["require"](path.suffix == ".whl", "NON_WHEEL_DISTRIBUTION")
                data = deps["read_regular"](path)
                row["sha256"] = deps["digest"](data)
                row["size"] = len(data)
                scan["bytes_read"] += len(data)
                scan["wheels_read"] += 1
                package = deps["inspect_wheel_bytes"](
                    data, path.name, origins.get(path.name), observations=observations
                )
                scan["wheels_fully_inspected"] += 1
                row.update(
                    status="STATIC_CANDIDATE_INSPECTED",
                    distribution=package["name"],
                    version=package["version"],
                )
            except INSPECTION_ERRORS as exc:
                row["reason"] = str(exc)
                report["refusals"].append({"filename": path.name, "reason": str(exc)})
            materialization_input = observations.pop("materialization_input", None)
            if materialization_input is not None:
                maps.append(materialization_input)
                row["materialization_map_status"] = "MEASURED"
            else:
                row["materialization_map_status"] = "NOT_TESTED"

        complete_inputs = (
            scan["wheels_read"] == scan["entries_listed"] == EXPECTED["WHEELS_CHECKED"]
        )
        report["measurements"]["WHEELS_CHECKED"] = measurement(
            scan["wheels_read"], scan["wheels_read"], complete_inputs
        )
        for metric, field in (
            ("FOREIGN_PYC_OMISSIONS", "foreign_pyc_omissions"),
            ("HEADER_OMISSIONS", "header_omissions"),
        ):
            inventories = [
                row["observations"][field]
                for row in report["archives"]
                if field in row["observations"]
            ]
            report["measurements"][metric] = measurement(
                sum(map(len, inventories)),
                len(inventories),
                complete_inputs and len(inventories) == scan["wheels_read"],
            )

        archive_refused = bool(report["refusals"])
        inspect_sharing(deps, report, maps, complete_inputs)
        if maps:
            try:
                report["header_omissions"] = deps["header_inventory"](maps)
            except INSPECTION_ERRORS as exc:
                report["refusals"].append({"check": "header_inventory", "reason": str(exc)})
        deps["require"](not archive_refused, "WHEELHOUSE_MEMBER_REFUSALS_RETAINED")
        if report["refusals"]:
            return report
        deps["require"](
            all(
                report["measurements"][key]["status"] == "MEASURED"
                and report["measurements"][key]["value"] == EXPECTED[expected_key]
                for key, expected_key in EXPECTED_KEYS.items()
            ),
            "WHEELHOUSE_EXPECTED_COUNTS_MISMATCH",
        )
        report["closure_status"] = "REFUSED"
        proposal = deps["propose"](pyproject, wheelhouse, origins, allow_shared_candidates=True)
        report["closure_status"] = "MEASURED"
        report["proposal_sha256"] = deps["digest"](deps["canonical"](proposal))
        report["status"] = "STATIC_CANDIDATES_INSPECTED_REQUIRES_INDEPENDENT_APPROVAL"
    except INSPECTION_ERRORS as exc:
        report["refusals"].append({"reason": str(exc)})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--origins", type=Path, required=True)
    args = parser.parse_args()
    deps = dependency_tools()
    try:
        origins = json.loads(deps["read_regular"](args.origins).decode("utf-8"))
        deps["require"](type(origins) is dict, "ORIGIN_SCHEMA")
        result = inspect(args.pyproject, args.wheelhouse, origins)
    except INSPECTION_ERRORS as exc:
        result = empty_report()
        result["reason"] = str(exc)
        result["refusals"].append({"reason": str(exc)})
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 2 if result["status"] == "REFUSED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
