"""Collect Stage 9 V5d7 evidence from the exact V5d6 wheelhouse, offline."""

from __future__ import annotations

import argparse
import io
import json
import platform
import runpy
import sysconfig
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REFERENCE_SHA256 = "9f57fc7e34adf94c7cdf734e406e8066df0fd909871011110a479d8add0a1bc2"
REFERENCE_REVISION = "7e6c21ae8e2472aaad6b6946a9c8545083f1f559"
SAFETY = {
    "STAGE_9": "IN_PROGRESS",
    "OBSERVER_3A": "BLOCKED",
    "STAGE_3B": "NOT_AUTHORIZED",
    "PRODUCTION": "NO-GO",
    "authorization": "NOT_AUTHORIZED",
    "native_load": "NOT_EXECUTED",
}


def tools(filename: str) -> dict[str, Any]:
    return runpy.run_path(str(Path(__file__).with_name(filename)))


def read_reference(raw: bytes, deps: dict[str, Any]) -> tuple[dict[str, Any], list[Any]]:
    deps["require"](deps["digest"](raw) == REFERENCE_SHA256, "V5D6_REFERENCE_SHA256")
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        official_entry = archive.extractfile("v5d6-official.json")
        inventory_entry = archive.extractfile("member-inventory.json")
        if official_entry is None or inventory_entry is None:
            raise ValueError("V5D6_REFERENCE_REPORTS_MISSING")
        with official_entry, inventory_entry:
            official = json.load(official_entry)
            inventory = json.load(inventory_entry)
    expected = {
        row["filename"]: {key: row[key] for key in ("sha256", "size")}
        for row in official["archives"]
    }
    independent = {
        row["wheel"]: {key: row[key] for key in ("sha256", "size")}
        for row in inventory["archives"]
        if row["origin_sha_matches"] is True
    }
    deps["require"](expected == independent, "V5D6_REFERENCE_ARCHIVES_DISAGREE")
    return expected, inventory["cross_wheel_collisions"]


def collect(
    reference: Path, pyproject: Path, wheelhouse: Path, origins_file: Path
) -> dict[str, Any]:
    diagnostic = tools("stage9_v5d7_inspect_wheelhouse.py")
    deps = diagnostic["dependency_tools"]()
    focused = tools("stage9_v5d7_focused.py")
    native = tools("stage9_v5d7_native.py")
    expected, reference_shared = read_reference(deps["read_regular"](reference), deps)
    origins = deps["json_data"](origins_file)
    deps["require"](type(origins) is dict, "ORIGIN_SCHEMA")

    native_members: list[dict[str, Any]] = []
    native_details: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    focused_members: list[dict[str, Any]] = []
    source_references: list[dict[str, Any]] = []
    paths: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def observe(wheel: str, member: str, content: bytes, mode: int, projected: str | None) -> None:
        member_digest = deps["digest"](content)
        row = {
            "wheel": wheel,
            "wheel_sha256": expected[wheel]["sha256"],
            "member": member,
            "sha256": member_digest,
            "size": len(content),
            "mode": mode,
            "type": "file",
        }
        if projected is not None:
            if projected in focused["SHARED_PYTHON_PATHS"]:
                row["content_b64"] = focused["focused_evidence"](content)["content_b64"]
            paths[projected].append(row)
        kind = focused["focused_kind"](wheel, member)
        if kind is not None:
            focused_members.append(
                {
                    **row,
                    **focused["focused_evidence"](content),
                    "kind": kind,
                    "projected": projected,
                }
            )
        references = focused["focus_references"](wheel, member, content)
        if references:
            source_references.append({**row, "references": references})
        if wheel in focused["METADATA_WHEELS"] and member.endswith(".dist-info/METADATA"):
            metadata.setdefault(wheel, []).append({**row, **focused["metadata_evidence"](content)})
        classification = native["analyze_native"](content, member, mode)
        if classification is not None:
            intrinsic_key = classification["format"] + ":" + member_digest
            native_details.setdefault(intrinsic_key, classification.pop("details"))
            native_members.append(
                {**row, **classification, "projected": projected, "native_details": intrinsic_key}
            )

    report = diagnostic["inspect"](
        pyproject, wheelhouse, origins, on_member=observe, expected_archives=expected
    )
    directory_paths = {
        directory
        for row in report["archives"]
        for directory in row.get("projected_directories", [])
    }
    measured_archives = {
        row["filename"]: row["sha256"]
        for row in report["archives"]
        if row.get("record_status") == "MEASURED"
    }
    complete = (
        set(measured_archives) == set(expected)
        and all(row["member_scan_status"] == "MEASURED" for row in report["archives"])
        and not any(refusal.get("check") == "member_reader" for refusal in report["refusals"])
    )
    focus_status = "MEASURED" if complete else "PARTIAL" if measured_archives else "NOT_TESTED"
    native_status = focus_status
    if native_status == "MEASURED" and any(row["status"] != "MEASURED" for row in native_members):
        native_status = "PARTIAL"
    sharing = focused["shared_evidence"](
        paths,
        reference_shared,
        measured_archives,
        directories=directory_paths,
        reference_members=deps["shared_tools"]()["MEMBERS"],
    )
    dependency_synthesis = native["summarize_dependencies"](native_members, native_details)
    needed = set(dependency_synthesis["bundled_candidates_by_needed_name"]) | set(
        dependency_synthesis["unmatched_needed_names"]
    )
    status = report["status"]
    if status == "MEASURED" and sharing["status"] == "REFUSED":
        status = "REFUSED"
    elif status == "MEASURED" and any(
        measurement != "MEASURED"
        for measurement in (focus_status, native_status, sharing["status"])
    ):
        status = "PARTIAL"
    try:
        release = platform.freedesktop_os_release()
    except OSError:
        release = {}
    return {
        **SAFETY,
        "profile": "stage9-v5d7-original-wheelhouse-evidence-v1",
        "status": status,
        "reference": {"sha256": REFERENCE_SHA256, "physical_revision": REFERENCE_REVISION},
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "soabi": sysconfig.get_config_var("SOABI"),
            "distribution": release.get("ID"),
            "distribution_version": release.get("VERSION_ID"),
        },
        "input": {
            "expected_archives": len(expected),
            "record_verified_archives": len(measured_archives),
        },
        "diagnostic": report,
        "focus": {
            "status": focus_status,
            "members": focused_members,
            "static_source_references": source_references,
            "metadata": metadata,
        },
        "pyside6_sharing": sharing,
        "native_inventory": {
            "status": native_status,
            "tools": native["native_tool_versions"](),
            "counts_by_detected_format": dict(Counter(row["format"] for row in native_members)),
            "members": native_members,
            "details": native_details,
            "dependency_synthesis": dependency_synthesis,
            "host_loader_cache": native["measure_host_loader_cache"](needed),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--pyproject", type=Path, default=Path(__file__).parent.parent / "pyproject.toml"
    )
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--origins", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = collect(args.reference, args.pyproject, args.wheelhouse, args.origins)
    except (KeyError, OSError, TypeError, ValueError, tarfile.TarError) as exc:
        report = {**SAFETY, "status": "REFUSED", "reason": str(exc)}
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if report["status"] == "REFUSED":
        return 2
    return 0 if report["status"] == "MEASURED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
