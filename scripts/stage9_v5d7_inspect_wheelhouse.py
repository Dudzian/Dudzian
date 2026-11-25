"""Collect Stage 9 wheel evidence without producing a proposal or approval."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable
import csv
from functools import partial
import io
import json
from pathlib import Path, PurePosixPath
import runpy
import stat
from typing import Any, cast
import zipfile
import zlib

REPORT_PROFILE = "stage9-v5d7-wheelhouse-static-diagnostic-v1"
EXPECTED_WHEELS = 69
EXPECTED_SHARED_PATHS = 62
EXPECTED_PYC_OMISSIONS = 2
EXPECTED_HEADER_OMISSIONS = 1
EXPECTED_ARTIFACT_OMISSIONS = 7
STATIC_ERRORS = (
    ValueError,
    OSError,
    zipfile.BadZipFile,
    KeyError,
    TypeError,
    csv.Error,
    EOFError,
    NotImplementedError,
    zlib.error,
)
ArchiveMemberReader = Callable[[str, bytes, int, str | None], None]
WheelhouseMemberReader = Callable[[str, str, bytes, int, str | None], None]


def dependency_tools() -> dict[str, Any]:
    return runpy.run_path(str(Path(__file__).with_name("stage9_v5d4_dependencies.py")))


def new_archive(filename: str) -> dict[str, Any]:
    return {
        "filename": filename,
        "sha256": None,
        "size": None,
        "distribution": None,
        "version": None,
        "record_status": "NOT_TESTED",
        "record_entries_verified": 0,
        "metadata_status": "NOT_TESTED",
        "metadata": None,
        "requires_python_status": "NOT_TESTED",
        "dependency_metadata_status": "NOT_TESTED",
        "member_scan_status": "NOT_TESTED",
        "materialization_map_status": "NOT_TESTED",
        "members": [],
        "archive_directories": [],
        "projected_directories": [],
        "foreign_pyc_omissions": None,
        "header_omissions": None,
        "artifact_omissions": None,
        "artifact_candidate_checks": [],
        "dependency_candidate": None,
        "refusals": [],
        "legacy_refusals": [],
    }


def refuse(
    wheel: dict[str, Any],
    check: str,
    reason: str,
    member: dict[str, Any] | None = None,
    *,
    legacy_only: bool = False,
) -> None:
    row = {
        "filename": wheel["filename"],
        "wheel_sha256": wheel["sha256"],
        "check": check,
        "reason": reason,
    }
    if member is not None:
        row.update({key: member[key] for key in ("member", "sha256", "size", "mode", "type")})
        for key in ("format_prefix_hex", "projected_path"):
            if key in member:
                row[key] = member[key]
    rows = wheel["legacy_refusals" if legacy_only else "refusals"]
    if row not in rows:
        rows.append(row)


def member_row(item: zipfile.ZipInfo, sha256: str | None) -> dict[str, Any]:
    mode = item.external_attr >> 16
    file_type = stat.S_IFMT(mode)
    return {
        "member": item.filename,
        "sha256": sha256,
        "size": item.file_size,
        "mode": mode,
        "type": (
            "directory"
            if item.is_dir() and file_type in {0, stat.S_IFDIR}
            else "regular"
            if not item.is_dir() and file_type in {0, stat.S_IFREG}
            else "special"
        ),
    }


def measure_archive_structure(
    deps: dict[str, Any], archive: zipfile.ZipFile, wheel: dict[str, Any]
) -> list[zipfile.ZipInfo]:
    members = deps["wheel_archive_index"](archive)
    for item in members:
        try:
            deps["check_wheel_archive_member"](archive, item)
        except STATIC_ERRORS as exc:
            refuse(wheel, "archive_structure", str(exc), member_row(item, None))
    return cast(list[zipfile.ZipInfo], members)


def measure_record(
    deps: dict[str, Any],
    archive: zipfile.ZipFile,
    files: dict[str, zipfile.ZipInfo],
    record_name: str,
    wheel: dict[str, Any],
) -> dict[str, str]:
    """Measure independent RECORD rows before allowing policy or callback reads."""
    wheel["record_status"] = "REFUSED"
    if record_name not in files:
        refuse(wheel, "record", "WHEEL_RECORD_MISSING")
        return {}
    try:
        records = deps["wheel_record_entries"](archive, record_name)
    except STATIC_ERRORS as exc:
        refuse(wheel, "record", str(exc))
        return {}
    if set(records) != set(files):
        wheel["record_missing_members"] = sorted(files.keys() - records.keys())
        wheel["record_extra_members"] = sorted(records.keys() - files.keys())
        refuse(wheel, "record", "WHEEL_RECORD_INCOMPLETE")
    hashes = {}
    for member in sorted(files.keys() & records.keys()):
        item = files[member]
        row = member_row(item, None)
        content = None
        try:
            content = archive.read(item)
            hashes[member] = deps["check_wheel_record_member"](
                member, *records[member], record_name, content
            )
        except STATIC_ERRORS as exc:
            if content is not None:
                row["sha256"] = deps["digest"](content)
            refuse(wheel, "record", str(exc), row)
    wheel["record_entries_verified"] = len(hashes)
    if len(hashes) == len(files) == len(records):
        wheel["record_status"] = "MEASURED"
    return hashes


def measure_metadata(
    deps: dict[str, Any],
    archive: zipfile.ZipFile,
    dist_info: str,
    records: dict[str, str],
    wheel: dict[str, Any],
) -> tuple[str, str] | None:
    metadata_name = dist_info + "/METADATA"
    metadata_row = member_row(archive.getinfo(metadata_name), records[metadata_name])
    wheel["metadata_status"] = "REFUSED"
    try:
        metadata, wheel_header = deps["read_wheel_metadata"](archive, dist_info)
        wheel["metadata"] = {
            **metadata_row,
            "headers": [list(item) for item in metadata.raw_items()],
        }
        candidate = deps["validate_wheel_metadata"](
            wheel["filename"], dist_info, metadata, wheel_header, v5d7=True
        )
    except STATIC_ERRORS as exc:
        refuse(wheel, "metadata", str(exc), metadata_row)
        return None
    wheel["metadata_status"] = "MEASURED"
    wheel.update(distribution=candidate["name"], version=candidate["version"])
    if candidate.get("metadata_version") == "2.5":
        refuse(
            wheel, "metadata", "WHEEL_METADATA_VERSION_UNSUPPORTED", metadata_row, legacy_only=True
        )
    wheel["requires_python_status"] = "REFUSED"
    try:
        candidate["requires_python"] = deps["check_wheel_python"](metadata)
    except STATIC_ERRORS as exc:
        refuse(wheel, "requires_python", str(exc), metadata_row)
    else:
        wheel["requires_python_status"] = "MEASURED"
    wheel["dependency_metadata_status"] = "REFUSED"
    try:
        candidate.update(deps["metadata_dependency_fields"](metadata))
    except STATIC_ERRORS as exc:
        refuse(wheel, "dependency_metadata", str(exc), metadata_row)
    else:
        wheel["dependency_metadata_status"] = "MEASURED"
        if wheel["requires_python_status"] == "MEASURED":
            wheel["dependency_candidate"] = candidate
    return candidate["name"], candidate["version"]


def measure_artifact_pins(
    tools: dict[str, Any],
    archive: zipfile.ZipFile,
    records: dict[str, str],
    wheel: dict[str, Any],
) -> None:
    """Preserve each measurable pin result before applying the whole-set policy."""
    for expected in tools["pinned_rows"](wheel["sha256"]):
        member = expected["member"]
        row = {
            "member": member,
            "sha256": None,
            "size": None,
            "mode": None,
            "type": None,
        }
        reasons = []
        if member not in records:
            reasons.append("ARTIFACT_OMISSION_SET_MISMATCH")
        else:
            row = member_row(archive.getinfo(member), records[member])
            if any(row[key] != expected[key] for key in ("size", "mode", "type")):
                reasons.append("ARTIFACT_OMISSION_MEMBER_METADATA_MISMATCH:" + member)
            if row["sha256"] != expected["sha256"]:
                reasons.append("ARTIFACT_OMISSION_MEMBER_HASH_MISMATCH:" + member)
        wheel["artifact_candidate_checks"].append(
            {**row, "expected": expected, "status": "REFUSED" if reasons else "MEASURED"}
        )
        for reason in reasons:
            refuse(wheel, "artifact_omissions", reason, row)


def measure_omissions(
    deps: dict[str, Any],
    archive: zipfile.ZipFile,
    records: dict[str, str],
    wheel: dict[str, Any],
    identity: tuple[str, str] | None,
) -> tuple[set[str], set[str]]:
    if identity is None:
        return set(), set()
    omitted: set[str] = set()
    for check, helper in (
        ("foreign_pyc_omissions", "inspect_wheel_bytecode"),
        ("header_omissions", "inspect_wheel_headers"),
    ):
        try:
            sources, omissions = deps[helper](
                archive, set(records), *identity, wheel["filename"], wheel["sha256"]
            )
        except STATIC_ERRORS as exc:
            refuse(wheel, check, str(exc))
        else:
            wheel[check] = omissions
            omitted.update(sources)
    tools = deps["artifact_tools"]()
    measure_artifact_pins(tools, archive, records, wheel)
    try:
        artifacts = tools["inspect_omissions"](
            *identity, wheel["filename"], wheel["sha256"], archive, set(records)
        )
    except STATIC_ERRORS as exc:
        if not any(
            item["check"] == "artifact_omissions" and item["reason"] == str(exc)
            for item in wheel["refusals"]
        ):
            refuse(wheel, "artifact_omissions", str(exc))
        return omitted, set()
    wheel["artifact_omissions"] = artifacts
    return omitted, {row["member"] for row in artifacts}


def measure_projection(
    deps: dict[str, Any],
    wheel: dict[str, Any],
    row: dict[str, Any],
    dist_info: str,
    omitted: set[str],
) -> str | None:
    member = row["member"]
    if member in omitted:
        return None
    try:
        target = deps["materialized_path"](member, dist_info)
        deps["check_startup"](member)
        deps["check_startup"](target)
    except STATIC_ERRORS as exc:
        refuse(wheel, "materialization_path", str(exc), row)
        return None
    return cast(str, target)


def measure_formats(
    deps: dict[str, Any],
    wheel: dict[str, Any],
    row: dict[str, Any],
    content: bytes,
    artifact_candidate: bool,
) -> None:
    member = row["member"]
    prefix = content[:4]
    if content.startswith(b"#!"):
        row["shebang"] = True
    row["native_elf"] = prefix == b"\x7fELF"
    row["execute_bits"] = bool(row["mode"] & 0o111)
    row["native_extension_candidate"] = member.endswith((".so", ".pyd", ".dll", ".dylib"))
    try:
        deps["check_executable_format"](member, prefix)
    except STATIC_ERRORS as exc:
        row["format_prefix_hex"] = prefix.hex()
        refuse(wheel, "executable_format", str(exc), row, legacy_only=artifact_candidate)
    try:
        deps["check_v5d7_coff"](member, content[:20], len(content))
    except STATIC_ERRORS as exc:
        row["coff_or_obj"] = True
        if not artifact_candidate:
            row["format_prefix_hex"] = prefix.hex()
            refuse(wheel, "coff", str(exc), row)
    if artifact_candidate:
        return
    native = row["native_elf"] or row["native_extension_candidate"] or ".so." in member
    try:
        deps["check_native_member"](member, prefix)
    except STATIC_ERRORS as exc:
        refuse(wheel, "native_soabi", str(exc), row)
    if native and ".data/" in member:
        refuse(wheel, "native_relocation", "NATIVE_DATA_RELOCATION_REQUIRES_REVIEW", row)


def measure_members(
    deps: dict[str, Any],
    archive: zipfile.ZipFile,
    members: list[zipfile.ZipInfo],
    records: dict[str, str],
    wheel: dict[str, Any],
    dist_info: str,
    identity: tuple[str, str] | None,
    on_member: ArchiveMemberReader | None,
) -> None:
    wheel["member_scan_status"] = "PARTIAL"
    omitted, artifacts = measure_omissions(deps, archive, records, wheel, identity)
    header = deps["header_tools"]()["HEADER"]
    archive_directories = []
    projected_directories: set[str] = set()
    for item in members:
        row = member_row(item, records.get(item.filename))
        if item.is_dir():
            archive_directories.append(row)
            try:
                deps["check_startup"](item.filename[:-1])
            except STATIC_ERRORS as exc:
                refuse(wheel, "directory_startup", str(exc), row)
            continue
        target = measure_projection(deps, wheel, row, dist_info, omitted)
        artifact_candidate = item.filename in artifacts
        target_parts = PurePosixPath(target).parts if target is not None else ()
        row["projected_path"] = target
        if target is not None and (
            row["sha256"] == header["sha256"]
            or header["member"] in {item.filename, target}
            or (
                len(target_parts) > 1
                and target_parts[0].endswith(".data")
                and target_parts[1] == "headers"
            )
        ):
            refuse(
                wheel,
                "header_inventory",
                "HEADER_RUNTIME_INCLUSION_FORBIDDEN",
                row,
                legacy_only=artifact_candidate,
            )
        if artifact_candidate:
            row["legacy_projected_path"] = target
            target = None
        row["projected_path"] = target
        if target is not None:
            projected_directories.update(
                parent.as_posix() for parent in PurePosixPath(target).parents if parent.parts
            )
        content = archive.read(item)
        measure_formats(deps, wheel, row, content, artifact_candidate)
        if on_member is not None:
            try:
                on_member(item.filename, content, row["mode"], target)
            except STATIC_ERRORS as exc:
                refuse(wheel, "member_reader", str(exc), row)
        wheel["members"].append(row)
    wheel["archive_directories"] = archive_directories
    wheel["projected_directories"] = sorted(projected_directories)
    wheel["member_scan_status"] = "MEASURED"
    wheel["materialization_map_status"] = (
        "PARTIAL"
        if any(
            item["check"] in {"materialization_path", "directory_startup"}
            for item in wheel["refusals"]
        )
        or any(
            wheel[field] is None
            for field in ("foreign_pyc_omissions", "header_omissions", "artifact_omissions")
        )
        else "MEASURED"
    )


def inspect_archive_data(
    deps: dict[str, Any],
    data: bytes,
    wheel: dict[str, Any],
    origin: Any,
    on_member: ArchiveMemberReader | None,
) -> None:
    """Apply archive prerequisites once, then collect independent checks."""
    filename = wheel["filename"]
    check = "origin"
    try:
        deps["check_wheel_origin"](data, filename, origin)
        check = "archive_structure"
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = measure_archive_structure(deps, archive, wheel)
            if wheel["refusals"]:
                return
            files = {item.filename: item for item in members if not item.is_dir()}
            dist_info = deps["wheel_dist_info"](list(files))
            collisions = set(files) & deps["wheel_member_directories"](members)
            check = "record"
            records = measure_record(deps, archive, files, dist_info + "/RECORD", wheel)
            for member in sorted(collisions):
                refuse(
                    wheel,
                    "archive_structure",
                    "WHEEL_MEMBER_FILE_DIRECTORY_COLLISION",
                    member_row(files[member], records.get(member)),
                )
            if wheel["record_status"] != "MEASURED" or collisions:
                return
            check = "metadata"
            identity = measure_metadata(deps, archive, dist_info, records, wheel)
            check = "member_scan"
            measure_members(deps, archive, members, records, wheel, dist_info, identity, on_member)
    except STATIC_ERRORS as exc:
        refuse(wheel, check, str(exc))


def inspect_archive(
    data: bytes,
    filename: str,
    origin: Any,
    *,
    on_member: ArchiveMemberReader | None = None,
) -> dict[str, Any]:
    """Read a wheel snapshot. The optional reader sees only complete RECORDs."""
    deps = dependency_tools()
    wheel = new_archive(filename)
    wheel.update(sha256=deps["digest"](data), size=len(data))
    inspect_archive_data(deps, data, wheel, origin, on_member)
    inspect_artifact_aliases(deps["artifact_tools"](), [wheel])
    return wheel


def inspect_artifact_aliases(tools: dict[str, Any], archives: list[dict[str, Any]]) -> None:
    """Check every projected member against omissions from the entire wheelhouse."""
    omissions = [item for wheel in archives for item in (wheel["artifact_omissions"] or [])]
    if not omissions:
        return
    inventory = {"policy": tools["POLICY_ID"], "members": omissions}
    sources = {row["member"] for row in omissions}
    for wheel in archives:
        for row in wheel["members"]:
            path = row["projected_path"]
            if path is None:
                continue
            try:
                tools["require"](
                    row["member"] not in sources, "ARTIFACT_OMISSION_RUNTIME_INCLUSION_FORBIDDEN"
                )
                tools["verify_runtime_absence"]({path: row["sha256"]}, inventory)
            except STATIC_ERRORS as exc:
                refuse(wheel, "artifact_omission_alias", str(exc), row)


def inspect_collisions(
    archives: list[dict[str, Any]],
    shared: dict[str, Any],
    *,
    check_complete_contract: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    owners: dict[str, list[dict[str, Any]]] = defaultdict(list)
    directories: set[str] = set()
    refusals = []
    for wheel in archives:
        local: set[str] = set()
        directories.update(wheel["projected_directories"])
        for member in wheel["members"]:
            path = member["projected_path"]
            if path is None:
                continue
            owner = {
                **{key: member[key] for key in ("member", "sha256", "size", "mode", "type")},
                "filename": wheel["filename"],
                "wheel_sha256": wheel["sha256"],
                "distribution": wheel["distribution"],
                "version": wheel["version"],
            }
            if path in local:
                refusals.append(
                    {"reason": "WHEEL_MATERIALIZATION_COLLISION", "path": path, "owner": owner}
                )
            local.add(path)
            owners[path].append(owner)
    for path in sorted(owners.keys() & directories):
        refusals.append(
            {
                "reason": "WHEEL_MATERIALIZATION_FILE_DIRECTORY_COLLISION",
                "path": path,
                "owners": owners[path],
            }
        )
    collisions = []
    contracts = shared["MEMBERS"]
    pinned_owners = {
        (name, row["version"], row["wheel_filename"], row["wheel_sha256"])
        for name, row in shared["WHEELS"].items()
    }
    owner_fields = ("distribution", "version", "filename", "wheel_sha256")
    if check_complete_contract:
        distributions = {wheel["distribution"] for wheel in archives}
        missing = {owner for owner in pinned_owners if owner[0] not in distributions}
        if missing:
            refusals.append(
                {
                    "reason": "SHARED_WHEEL_OWNER_SET_MISMATCH",
                    "missing_owners": [
                        dict(zip(owner_fields, row, strict=True)) for row in sorted(missing)
                    ],
                }
            )
    collision_paths = {
        path for path, rows in owners.items() if len({owner["filename"] for owner in rows}) >= 2
    }
    checked_paths = collision_paths | (set(contracts) if check_complete_contract else set())
    for path in sorted(checked_paths):
        path_owners = owners.get(path, [])
        pin = contracts.get(path)
        row = {"path": path, "owners": path_owners}
        if path not in collision_paths or pin is None:
            refusals.append({"reason": "SHARED_WHEEL_MEMBER_SET_MISMATCH", **row})
        if pin is not None:
            identity_fields = (*owner_fields, "member")
            expected = {(*owner, path) for owner in pinned_owners}
            actual_owners = {tuple(owner[key] for key in identity_fields) for owner in path_owners}
            if len(path_owners) != len(expected) or actual_owners != expected:
                refusals.append(
                    {
                        "reason": "SHARED_WHEEL_OWNER_MISMATCH",
                        "expected_owners": [
                            dict(zip(identity_fields, owner, strict=True))
                            for owner in sorted(expected)
                        ],
                        **row,
                    }
                )
            if any(
                not shared["exact"](
                    {key: owner[key] for key in ("sha256", "size", "mode", "type")}, pin
                )
                for owner in path_owners
            ):
                refusals.append({"reason": "SHARED_WHEEL_CONTENT_OR_MODE_MISMATCH", **row})
        if path in collision_paths:
            collisions.append(row)
    return collisions, refusals


def measure(value: int, observed: int, complete: bool) -> dict[str, Any]:
    return {
        "status": "NOT_TESTED" if not observed else "MEASURED" if complete else "PARTIAL",
        "value": value if observed else None,
        "wheels_evaluated": observed,
    }


def check_closure(
    deps: dict[str, Any], pyproject: Path, archives: list[dict[str, Any]], inputs_complete: bool
) -> dict[str, Any]:
    if not inputs_complete or any(wheel["dependency_candidate"] is None for wheel in archives):
        return {"status": "NOT_TESTED"}
    try:
        candidates = {
            wheel["dependency_candidate"]["name"]: wheel["dependency_candidate"]
            for wheel in archives
        }
        deps["require"](len(candidates) == len(archives), "AMBIGUOUS_PACKAGE_VERSION")
        project = deps["read_regular"](pyproject, 8 * 1024 * 1024)
        roots = deps["closure_roots"](project)
        packages, edges = deps["resolve_dependency_closure"](roots, candidates)
        return {
            "status": "MEASURED",
            "root_requirements": roots,
            "packages": packages,
            "edges": edges,
        }
    except STATIC_ERRORS as exc:
        return {"status": "REFUSED", "reason": str(exc)}


def inspect(
    pyproject: Path,
    wheelhouse: Path,
    origins: dict[str, Any],
    *,
    on_member: WheelhouseMemberReader | None = None,
    expected_archives: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Inspect the whole directory without introducing a materialization input."""
    deps = dependency_tools()
    expected_count = EXPECTED_WHEELS if expected_archives is None else len(expected_archives)
    archives: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    paths = []
    directory_listed = False
    try:
        deps["require"](not wheelhouse.is_symlink() and wheelhouse.is_dir(), "WHEELHOUSE_DIRECTORY")
        paths = sorted(wheelhouse.iterdir())
        directory_listed = True
        actual = {path.name for path in paths}
        if set(origins) != actual:
            refusals.append(
                {
                    "reason": "ORIGIN_INVENTORY",
                    "missing": sorted(actual - origins.keys()),
                    "extra": sorted(origins.keys() - actual),
                }
            )
        if expected_archives is not None and set(expected_archives) != actual:
            refusals.append(
                {
                    "reason": "EVIDENCE_ARCHIVE_INVENTORY",
                    "missing": sorted(expected_archives.keys() - actual),
                    "extra": sorted(actual - expected_archives.keys()),
                }
            )
    except STATIC_ERRORS as exc:
        refusals.append({"check": "wheelhouse", "reason": str(exc)})

    for path in paths:
        wheel = new_archive(path.name)
        archives.append(wheel)
        try:
            deps["require"](path.suffix == ".whl", "NON_WHEEL_DISTRIBUTION")
            data = deps["read_regular"](path)
            wheel.update(sha256=deps["digest"](data), size=len(data))
            if expected_archives is not None:
                pin = expected_archives.get(path.name)
                deps["require"](
                    pin == {key: wheel[key] for key in ("sha256", "size")},
                    "EVIDENCE_ARCHIVE_IDENTITY",
                )
            reader = None if on_member is None else partial(on_member, path.name)
            inspect_archive_data(deps, data, wheel, origins.get(path.name), reader)
        except STATIC_ERRORS as exc:
            refuse(wheel, "archive_read", str(exc))

    read_count = sum(wheel["sha256"] is not None for wheel in archives)
    record_count = sum(wheel["record_status"] == "MEASURED" for wheel in archives)
    map_count = sum(wheel["materialization_map_status"] == "MEASURED" for wheel in archives)
    map_coverage = sum(wheel["materialization_map_status"] != "NOT_TESTED" for wheel in archives)
    inputs_complete = bool(paths) and read_count == len(paths) == expected_count and not refusals
    inspect_artifact_aliases(deps["artifact_tools"](), archives)
    shared = deps["shared_tools"](v5d7=True)
    pinned_names = {row["wheel_filename"] for row in shared["WHEELS"].values()}
    bound_names = pinned_names & (expected_archives or {}).keys()
    observed_names = {wheel["filename"] for wheel in archives}
    sharing_applicable = bool(
        expected_count == EXPECTED_WHEELS or bound_names or pinned_names & observed_names
    )
    manifest_complete = inputs_complete and map_count == expected_count
    collisions, collision_refusals = inspect_collisions(
        archives, shared, check_complete_contract=sharing_applicable and manifest_complete
    )
    absent_bound = bound_names - observed_names
    if directory_listed and absent_bound:
        collision_refusals.append(
            {
                "reason": "SHARED_WHEEL_OWNER_SET_MISMATCH",
                "missing_owners": [
                    {
                        "distribution": name,
                        "version": pin["version"],
                        "filename": pin["wheel_filename"],
                        "wheel_sha256": pin["wheel_sha256"],
                    }
                    for name, pin in sorted(shared["WHEELS"].items())
                    if pin["wheel_filename"] in absent_bound
                ],
            }
        )
    _, legacy_collisions = inspect_collisions(archives, deps["shared_tools"]())
    refusals.extend(collision_refusals)
    refusals.extend(item for wheel in archives for item in wheel["refusals"])
    if not inputs_complete:
        refusals.append({"reason": "WHEELHOUSE_EXPECTED_INPUTS_MISMATCH"})
    closure = check_closure(deps, pyproject, archives, inputs_complete)
    if closure["status"] == "REFUSED":
        refusals.append({"check": "dependency_closure", "reason": closure["reason"]})
    metrics = {
        "WHEELS_CHECKED": measure(read_count, read_count, read_count == expected_count),
        "RECORD_VERIFIED": measure(record_count, record_count, record_count == expected_count),
        "SHARED_PATHS": measure(len(collisions), map_coverage, map_count == expected_count),
    }
    for key, field in (
        ("FOREIGN_PYC_OMISSIONS", "foreign_pyc_omissions"),
        ("HEADER_OMISSIONS", "header_omissions"),
        ("ARTIFACT_OMISSIONS", "artifact_omissions"),
    ):
        rows = [wheel[field] for wheel in archives if wheel[field] is not None]
        metrics[key] = measure(sum(map(len, rows)), len(rows), len(rows) == expected_count)
    expected_counts = {
        "WHEELS_CHECKED": expected_count,
        "SHARED_PATHS": EXPECTED_SHARED_PATHS,
        "FOREIGN_PYC_OMISSIONS": EXPECTED_PYC_OMISSIONS,
        "HEADER_OMISSIONS": EXPECTED_HEADER_OMISSIONS,
        "ARTIFACT_OMISSIONS": EXPECTED_ARTIFACT_OMISSIONS,
    }
    if expected_count == EXPECTED_WHEELS:
        for metric, expected in expected_counts.items():
            row = metrics[metric]
            if row["status"] == "MEASURED" and row["value"] != expected:
                refusals.append(
                    {
                        "check": metric,
                        "reason": "WHEELHOUSE_EXPECTED_COUNTS_MISMATCH",
                        "expected": expected,
                        "measured": row["value"],
                    }
                )
    sharing_coverage = measure(
        len(collisions), map_coverage, sharing_applicable and manifest_complete
    )["status"]
    artifact_coverage = measure(
        sum(row["projected_path"] is not None for wheel in archives for row in wheel["members"]),
        map_coverage,
        manifest_complete,
    )["status"]
    manifest_checks = {
        "sharing": {
            "coverage": sharing_coverage,
            "status": "REFUSED" if collision_refusals else sharing_coverage,
            "policy": shared["POLICY_ID"],
            "expected_paths": len(shared["MEMBERS"]),
            "expected_owners_per_path": len(shared["WHEELS"]),
        },
        "artifact_omissions": {
            "coverage": artifact_coverage,
            "status": "REFUSED"
            if any(
                item.get("check")
                in {"artifact_omissions", "artifact_omission_alias", "ARTIFACT_OMISSIONS"}
                for item in refusals
            )
            else artifact_coverage,
            "candidates": metrics["ARTIFACT_OMISSIONS"]["value"],
        },
    }
    return {
        "profile": REPORT_PROFILE,
        "status": "REFUSED" if refusals else "MEASURED",
        "expected_counts": expected_counts,
        "measurements": metrics,
        "manifest_checks": manifest_checks,
        "archives": archives,
        "archive_scan": {
            "entries_listed": len(paths),
            "wheels_read": read_count,
            "bytes_read": sum(wheel["size"] or 0 for wheel in archives),
        },
        "shared_wheel_members": collisions,
        "refusals": refusals,
        "legacy_refusals": [
            *(item for wheel in archives for item in wheel["legacy_refusals"]),
            *(item for item in legacy_collisions if item["reason"].startswith("SHARED_")),
        ],
        "dependency_closure": closure,
        "closure_status": closure["status"],
        "authorization": "NOT_AUTHORIZED",
        "independent_approvals": "NOT_SUPPLIED_NOT_GENERATED",
        "STAGE_9": "IN_PROGRESS",
        "OBSERVER_3A": "BLOCKED",
        "STAGE_3B": "NOT_AUTHORIZED",
        "PRODUCTION": "NO-GO",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--origins", type=Path, required=True)
    args = parser.parse_args()
    deps = dependency_tools()
    try:
        origins = deps["json_data"](args.origins)
        deps["require"](type(origins) is dict, "ORIGIN_SCHEMA")
        report = inspect(args.pyproject, args.wheelhouse, origins)
    except STATIC_ERRORS as exc:
        report = {
            "profile": REPORT_PROFILE,
            "status": "REFUSED",
            "refusals": [{"check": "origins", "reason": str(exc)}],
            "authorization": "NOT_AUTHORIZED",
        }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 2 if report["status"] == "REFUSED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
