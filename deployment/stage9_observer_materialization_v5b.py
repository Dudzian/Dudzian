"""Bounded offline symlink-to-regular agreement on verified immutable DEBs.

This is a separate V5b contract, not V1 ``materialize_member``. It computes no
installation tree and grants no authority. V5a's static validation rules and
literal V3 record identities are reused without changing V5a's success profile.
"""

from __future__ import annotations

import hashlib
import heapq
import zlib
from dataclasses import dataclass, field
from typing import Any, Literal

from deployment import stage9_observer_contract as contract, stage9_observer_image_v5a as v5a
from deployment.canonical_json import canonical_json_bytes
from deployment.stage9_observer_member_source_v5b import (
    CapturedMemberV5b,
    FrozenPackageV5b,
    MemberSelectionV5b,
    capture_member_sources_v5b,
)
from deployment.stage9_observer_offline import Limits, _validate_limits
from deployment.stage9_observer_package_v3 import (
    PackageInventoryResultV3,
    UbuntuPackageInventoryV3,
    _parse,
)
from deployment.stage9_observer_tar_v3 import TarMemberV3

EXPECTED_FORMAT = "stage9-observer-expected-image-v5b"
MAPPING_FORMAT = "stage9-observer-materialization-v5b"
METADATA_POLICY = "source-record-exact-v1"
TRANSFORM_KIND = "symlink_to_regular_v5b"
_DEFAULT_LIMITS = Limits()
_REF_KEYS = {"source_id", "record_id", "member"}


@dataclass(frozen=True, slots=True)
class DependencyV5b:
    node_id: str
    dependencies: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MaterializedFileV5b:
    transform_id: str
    file_id: str
    path: str
    sha256: str
    size: int
    uid: int
    gid: int
    mode: int
    link: v5a.MappedOriginV5a
    source: v5a.MappedOriginV5a


@dataclass(frozen=True, slots=True)
class MaterializationResultV5b:
    """MATCH means complete agreement only within this bounded offline profile."""

    consistency: Literal["MATCH", "MISMATCH", "INCOMPLETE"]
    reason_codes: tuple[str, ...]
    materialized_files: tuple[MaterializedFileV5b, ...] = ()
    mapped_files: tuple[v5a.MappedFileV5a, ...] = ()
    dependencies: tuple[DependencyV5b, ...] = ()
    resolution_order: tuple[str, ...] = ()
    inventory_sha256: str | None = None
    expected_image_sha256: str | None = None
    mapping_sha256: str | None = None
    snapshot_sha256: str | None = None
    package_result_sha256: str | None = None
    apt_evidence_sha256: str | None = None
    source_bytes_sha256: str | None = None
    v4_binding: Literal["NOT_PROVIDED", "CALLER_CLAIM_BOUND"] = "NOT_PROVIDED"
    unsupported_transformations: tuple[str, ...] = ()
    unmet_requirements: tuple[str, ...] = v5a._UNMET
    profile: Literal["stage9-observer-materialization-result-v5b"] = field(
        default="stage9-observer-materialization-result-v5b", init=False
    )
    scope: Literal["BOUNDED_OFFLINE_MEMBER_MATERIALIZATION_ONLY"] = field(
        default="BOUNDED_OFFLINE_MEMBER_MATERIALIZATION_ONLY", init=False
    )
    source_authenticity: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    evidence_origin: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    external_authority_acceptance: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    acceptance: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    authenticated_time: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    durable_anti_rollback: Literal["BLOCKED"] = field(default="BLOCKED", init=False)
    runtime_qualification: Literal["BLOCKED"] = field(default="BLOCKED", init=False)
    execution: Literal["MISSING"] = field(default="MISSING", init=False)
    host_state: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    snapshot_membership: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    stage_9: Literal["IN_PROGRESS"] = field(default="IN_PROGRESS", init=False)
    qualification: Literal["BLOCKED"] = field(default="BLOCKED", init=False)
    stage_3b: Literal["NOT AUTHORIZED"] = field(default="NOT AUTHORIZED", init=False)
    production: Literal["NO-GO"] = field(default="NO-GO", init=False)
    CONTRACT_APPROVED: Literal["NO"] = field(default="NO", init=False)
    STAGE_3A_COMPLETE: Literal["NO"] = field(default="NO", init=False)
    PRODUCTION_QUALIFIED: Literal["NO"] = field(default="NO", init=False)
    production_status: Literal["PRODUCTION_NO_GO"] = field(default="PRODUCTION_NO_GO", init=False)
    refused: Literal[True] = field(default=True, init=False)


@dataclass(frozen=True, slots=True)
class _Reference:
    source_id: str
    record_id: str
    record: TarMemberV3

    @property
    def key(self) -> tuple[str, str]:
        return self.source_id, self.record_id


@dataclass(frozen=True, slots=True)
class _Operation:
    transform_id: str
    file_id: str
    link: _Reference
    source: _Reference


def _require(condition: bool, code: str = "FORMAT_REJECTED") -> None:
    if not condition:
        raise v5a.ImageMappingRefusalV5a(code)


def _record_index(
    sources: dict[str, UbuntuPackageInventoryV3], budget: v5a._MappingBudget
) -> tuple[dict[tuple[str, str], TarMemberV3], dict[tuple[str, str], TarMemberV3]]:
    records: dict[tuple[str, str], TarMemberV3] = {}
    paths: dict[tuple[str, str], TarMemberV3] = {}
    for source_id, package in sources.items():
        for record in package.data.records:
            budget.compare()
            budget.reserve(2048)
            key = source_id, v5a.record_id_v5a(record)
            path = source_id, record.canonical_name
            _require(key not in records and path not in paths, "AMBIGUOUS_ALIAS")
            records[key], paths[path] = record, record
    return records, paths


def _reference(
    value: Any, records: dict[tuple[str, str], TarMemberV3], budget: v5a._MappingBudget
) -> _Reference:
    row = v5a._object(value, _REF_KEYS)
    key = v5a._id(row["source_id"]), v5a._digest(row["record_id"])
    budget.compare()
    _require(key in records, "PROVENANCE_INCOMPLETE")
    record = records[key]
    _require(type(row["member"]) is str and row["member"] == record.name, "ARTIFACT_MISMATCH")
    return _Reference(key[0], key[1], record)


def _target(
    link: _Reference,
    paths: dict[tuple[str, str], TarMemberV3],
    budget: v5a._MappingBudget,
) -> TarMemberV3:
    """Resolve components lexically and require each intermediate directory.

    Even a component later followed by '..' must be a declared directory.
    No recursion, host lookup, decoding, Unicode normalization or link following.
    """
    target = link.record.linkname
    _require(type(target) is str and bool(target), "UNSUPPORTED_TRANSFORMATION")
    _require(
        not target.startswith("/")
        and "\\" not in target
        and all(ord(char) >= 32 and not 127 <= ord(char) <= 159 for char in target),
        "UNSUPPORTED_TRANSFORMATION",
    )
    _require(len(target.encode("utf-8")) <= contract.MAX_PATH_BYTES, "LIMIT_EXCEEDED")
    parts = target.split("/")
    _require(len(parts) <= budget.limits.max_path_depth, "LIMIT_EXCEEDED")
    _require(all(part for part in parts), "UNSUPPORTED_TRANSFORMATION")
    current = link.record.canonical_name.split("/")[:-1]
    record: TarMemberV3 | None = None
    for index, part in enumerate(parts):
        budget.compare()
        if part == "..":
            _require(bool(current), "PATH_ESCAPE")
            current.pop()
        elif part != ".":
            # The image path validator checks every component using the same
            # spelling rules as the independently supplied expected image.
            v5a._path("/" + part, budget.limits)
            current.append(part)
        _require(len(current) <= budget.limits.max_path_depth, "LIMIT_EXCEEDED")
        name = "/".join(current) or "."
        budget.compare()
        record = paths.get((link.source_id, name))
        _require(record is not None, "DANGLING_LINK")
        assert record is not None
        if index != len(parts) - 1:
            _require(record.typeflag == "5", "UNSUPPORTED_TRANSFORMATION")
    _require(record is not None, "DANGLING_LINK")
    assert record is not None
    _require(record.typeflag in {"0", "\0"}, "UNSUPPORTED_TRANSFORMATION")
    return record


def _plan(
    image: dict[str, Any],
    mapping: dict[str, Any],
    files: dict[str, dict[str, Any]],
    sources: dict[str, UbuntuPackageInventoryV3],
    budget: v5a._MappingBudget,
) -> tuple[
    dict[str, list[tuple[str, str, TarMemberV3]]],
    tuple[_Operation, ...],
    dict[str, set[str]],
]:
    records, paths = _record_index(sources, budget)
    expected: dict[str, tuple[_Reference, _Reference]] = {}
    for row in v5a._rows(image["materializations"], budget.limits.max_files):
        budget.compare()
        row = v5a._object(row, {"file_id", "link", "source", "metadata_policy"})
        file_id = v5a._id(row["file_id"])
        _require(file_id in files and file_id not in expected, "CLOSURE_INCOMPLETE")
        _require(row["metadata_policy"] == METADATA_POLICY, "UNSUPPORTED_TRANSFORMATION")
        expected[file_id] = (
            _reference(row["link"], records, budget),
            _reference(row["source"], records, budget),
        )
    graph: dict[str, set[str]] = {}
    for source_id in sources:
        graph["package:" + source_id] = set()
    for source_id, record_id in records:
        graph[f"member:{source_id}:{record_id}"] = {"package:" + source_id}
    seen: set[tuple[str, str]] = set()
    operations: dict[str, _Operation] = {}
    outputs: set[str] = set()
    for row in v5a._rows(mapping["transforms"], budget.limits.max_files):
        budget.compare()
        row = v5a._object(
            row,
            {"transform_id", "kind", "link", "source", "output_file_id", "metadata_policy"},
        )
        transform_id = v5a._id(row["transform_id"])
        _require(transform_id not in operations, "CLOSURE_INCOMPLETE")
        if row["kind"] != TRANSFORM_KIND:
            kind = contract._launch_exact_text(row["kind"])
            raise v5a.ImageMappingRefusalV5a("UNSUPPORTED_TRANSFORMATION", (kind,))
        _require(row["metadata_policy"] == METADATA_POLICY, "UNSUPPORTED_TRANSFORMATION")
        file_id = v5a._id(row["output_file_id"])
        _require(file_id in expected and file_id not in outputs, "CLOSURE_INCOMPLETE")
        link = _reference(row["link"], records, budget)
        source = _reference(row["source"], records, budget)
        _require((link, source) == expected[file_id], "ARTIFACT_MISMATCH")
        _require(link.record.typeflag == "2", "UNSUPPORTED_TRANSFORMATION")
        _require(link.source_id == source.source_id, "CROSS_PACKAGE_LINK")
        _require(link.key not in seen, "CLOSURE_INCOMPLETE")
        resolved = _target(link, paths, budget)
        _require(resolved == source.record, "ARTIFACT_MISMATCH")
        _require(source.record.typeflag in {"0", "\0"}, "UNSUPPORTED_TRANSFORMATION")
        file = files[file_id]
        _require(file["type"] == "regular", "ARTIFACT_MISMATCH")
        _require(file["path"] == "/" + link.record.canonical_name, "ARTIFACT_MISMATCH")
        if any(file[key] != getattr(source.record, key) for key in ("uid", "gid", "mode")):
            raise v5a.ImageMappingRefusalV5a("ARTIFACT_MISMATCH", ("metadata_change",))
        _require(
            file["size"] == source.record.size and file["sha256"] == source.record.sha256,
            "ARTIFACT_MISMATCH",
        )
        seen.add(link.key)
        outputs.add(file_id)
        operations[transform_id] = _Operation(transform_id, file_id, link, source)
        graph["symlink:" + transform_id] = {
            f"member:{link.source_id}:{link.record_id}",
            f"member:{source.source_id}:{source.record_id}",
        }
    _require(outputs == set(expected), "CLOSURE_INCOMPLETE")
    direct: dict[str, list[tuple[str, str, TarMemberV3]]] = {}
    used_operations: set[str] = set()
    for row in v5a._rows(mapping["entries"], budget.limits.max_archive_members):
        budget.compare()
        _require(type(row) is dict)
        kind = contract._enum(row.get("kind"), {"source_member", "transform_output"})
        file_id = v5a._id(row.get("file_id"))
        _require(file_id in files, "CLOSURE_INCOMPLETE")
        if kind == "source_member":
            row = v5a._object(row, {"kind", "file_id"} | _REF_KEYS)
            ref = _reference({key: row[key] for key in _REF_KEYS}, records, budget)
            _require(ref.key not in seen and file_id not in outputs, "CLOSURE_INCOMPLETE")
            _require(ref.record.typeflag != "2", "UNSUPPORTED_TRANSFORMATION")
            seen.add(ref.key)
            direct.setdefault(file_id, []).append((ref.source_id, ref.record_id, ref.record))
            graph.setdefault("expected:" + file_id, set()).add(
                f"member:{ref.source_id}:{ref.record_id}"
            )
        else:
            row = v5a._object(row, {"kind", "transform_id", "file_id"})
            transform_id = v5a._id(row["transform_id"])
            _require(
                transform_id in operations and transform_id not in used_operations,
                "CLOSURE_INCOMPLETE",
            )
            _require(operations[transform_id].file_id == file_id, "CLOSURE_INCOMPLETE")
            used_operations.add(transform_id)
            graph["expected:" + file_id] = {"symlink:" + transform_id}
    _require(used_operations == set(operations), "CLOSURE_INCOMPLETE")
    _require(seen == set(records), "CLOSURE_INCOMPLETE")
    _require(set(direct) | outputs == set(files), "CLOSURE_INCOMPLETE")
    for dependencies in graph.values():
        budget.compare()
        budget.reserve(512 + 256 * len(dependencies))
    return direct, tuple(operations[key] for key in sorted(operations)), graph


def _order(
    graph: dict[str, set[str]], budget: v5a._MappingBudget
) -> tuple[tuple[DependencyV5b, ...], tuple[str, ...]]:
    """Deterministic iterative topological order; missing nodes and cycles refuse."""
    pending = {node: len(deps) for node, deps in graph.items()}
    children: dict[str, list[str]] = {node: [] for node in graph}
    for node, deps in graph.items():
        for dep in deps:
            budget.compare()
            _require(dep in graph, "DEPENDENCY_MISSING")
            children[dep].append(node)
    ready = [node for node, count in pending.items() if count == 0]
    heapq.heapify(ready)
    result: list[str] = []
    while ready:
        budget.compare()
        node = heapq.heappop(ready)
        result.append(node)
        for child in children[node]:
            budget.compare()
            pending[child] -= 1
            if pending[child] == 0:
                heapq.heappush(ready, child)
    _require(len(result) == len(graph), "DEPENDENCY_CYCLE")
    dependencies = tuple(DependencyV5b(node, tuple(sorted(graph[node]))) for node in sorted(graph))
    return dependencies, tuple(result)


def _origin(ref: _Reference, sources: dict[str, UbuntuPackageInventoryV3]) -> v5a.MappedOriginV5a:
    package = sources[ref.source_id]
    return v5a.MappedOriginV5a(
        ref.source_id, package.package, package.version, ref.record_id, ref.record.name
    )


def _materialize(
    operations: tuple[_Operation, ...],
    captured: tuple[CapturedMemberV5b, ...],
    files: dict[str, dict[str, Any]],
    sources: dict[str, UbuntuPackageInventoryV3],
    budget: v5a._MappingBudget,
) -> tuple[tuple[MaterializedFileV5b, ...], tuple[v5a.MappedFileV5a, ...], str]:
    by_source = {(item.source_id, item.record_id): item for item in captured}
    staged: list[MaterializedFileV5b] = []
    mapped: list[v5a.MappedFileV5a] = []
    bindings: list[dict[str, Any]] = []
    for op in operations:
        budget.compare()
        _require(op.source.key in by_source, "PROVENANCE_INCOMPLETE")
        item = by_source[op.source.key]
        package = sources[op.source.source_id]
        _require(
            item.package == package.package
            and item.member == op.source.record.name
            and item.archive_sha256 == package.sha256
            and item.archive_size == package.size,
            "ARTIFACT_MISMATCH",
        )
        digest = hashlib.sha256()
        for offset in range(0, len(item.content), budget.limits.chunk_bytes):
            budget.compare()
            digest.update(item.content[offset : offset + budget.limits.chunk_bytes])
        actual_sha = digest.hexdigest()
        file = files[op.file_id]
        _require(
            actual_sha == op.source.record.sha256 == file["sha256"]
            and len(item.content) == op.source.record.size == file["size"],
            "ARTIFACT_MISMATCH",
        )
        budget.reserve(8192)
        link, source = _origin(op.link, sources), _origin(op.source, sources)
        staged.append(
            MaterializedFileV5b(
                op.transform_id,
                op.file_id,
                file["path"],
                actual_sha,
                len(item.content),
                file["uid"],
                file["gid"],
                file["mode"],
                link,
                source,
            )
        )
        mapped.append(
            v5a.MappedFileV5a(
                op.file_id,
                file["path"],
                "regular",
                actual_sha,
                len(item.content),
                file["uid"],
                file["gid"],
                file["mode"],
                (link, source),
            )
        )
        bindings.append(
            dict(
                transform_id=op.transform_id,
                source_id=item.source_id,
                record_id=item.record_id,
                member=item.member,
                archive_sha256=item.archive_sha256,
                archive_size=item.archive_size,
                sha256=actual_sha,
                size=len(item.content),
            )
        )
    budget.check()
    binding = hashlib.sha256(
        b"stage9-v5b-source-bytes\0" + canonical_json_bytes(bindings)
    ).hexdigest()
    return tuple(staged), tuple(mapped), binding


def verify_member_materialization_v5b(
    *,
    inventory_raw: bytes,
    expected_image_raw: bytes,
    mapping_raw: bytes,
    expected_image_sha256: str,
    archives: tuple[FrozenPackageV5b, ...],
    package_result: PackageInventoryResultV3 | None = None,
    apt_evidence: v5a.AptEvidenceV5a | None = None,
    limits: Limits = _DEFAULT_LIMITS,
) -> MaterializationResultV5b:
    """Verify complete byte-identical operations; any failure erases all bindings.

    Immutable archive bytes are always remeasured. A caller-supplied V3 result
    is additionally required for continuity with V5a and compared to those
    actual measurements. Optional V4 evidence remains a bounded caller claim.
    The expected image and its out-of-band pin are never generated here.
    """
    unsupported: tuple[str, ...] = ()
    try:
        _validate_limits(limits)
        budget = v5a._MappingBudget(limits)
        v5a._digest(expected_image_sha256)
        v5a._preflight(inventory_raw, budget)
        packages = _parse(inventory_raw, limits, budget)
        _require(len(packages) <= v5a.MAX_PACKAGES, "LIMIT_EXCEEDED")
        count = sum(
            len(package.control.records) + len(package.data.records) for package in packages
        )
        _require(count <= limits.max_archive_members, "LIMIT_EXCEEDED")
        image = v5a._decode(expected_image_raw, budget)
        mapping = v5a._decode(mapping_raw, budget)
        inventory_digest = hashlib.sha256(inventory_raw).hexdigest()
        image_digest = hashlib.sha256(expected_image_raw).hexdigest()
        mapping_digest = hashlib.sha256(mapping_raw).hexdigest()
        _require(image_digest == expected_image_sha256, "ARTIFACT_MISMATCH")
        image_keys = {
            "format",
            "image_id",
            "snapshot_sha256",
            "sources",
            "files",
            "materializations",
        }
        mapping_keys = {
            "format",
            "snapshot_sha256",
            "inventory_sha256",
            "expected_image_sha256",
            "sources",
            "directory_policy",
            "entries",
            "transforms",
        }
        v5a._object(image, image_keys | ({"apt_context"} if "apt_context" in image else set()))
        v5a._object(
            mapping, mapping_keys | ({"apt_context"} if "apt_context" in mapping else set())
        )
        _require(
            image["format"] == EXPECTED_FORMAT and mapping["format"] == MAPPING_FORMAT,
            "UNSUPPORTED_SOURCE",
        )
        _require(mapping["directory_policy"] == "identical-explicit-v1", "UNSUPPORTED_SOURCE")
        _require(
            v5a._digest(mapping["inventory_sha256"]) == inventory_digest
            and v5a._digest(mapping["expected_image_sha256"]) == image_digest,
            "ARTIFACT_MISMATCH",
        )
        # Reuse the unchanged static file validator on the static fields only.
        # This internal view is never submitted as a successful V5a result.
        static = {key: value for key, value in image.items() if key != "materializations"}
        static["format"] = v5a.EXPECTED_FORMAT
        files = v5a._image(static, budget)
        sources = v5a._sources(mapping["sources"], packages, budget)
        _require(
            sources == v5a._sources(image["sources"], packages, budget), "PROVENANCE_INCOMPLETE"
        )
        snapshot = v5a._digest(mapping["snapshot_sha256"])
        _require(snapshot == image["snapshot_sha256"], "SNAPSHOT_MISMATCH")
        selection = v5a._apt_selection(image, mapping, budget)
        _require(selection is None or apt_evidence is not None, "PROVENANCE_INCOMPLETE")
        direct, operations, graph = _plan(image, mapping, files, sources, budget)
        mapped = v5a._map(direct, files, sources, budget)
        dependencies, order = _order(graph, budget)
        selections: dict[tuple[str, str], MemberSelectionV5b] = {}
        for op in operations:
            budget.compare()
            ref = op.source
            selections[ref.key] = MemberSelectionV5b(
                ref.source_id, sources[ref.source_id].package, ref.record.name, ref.record_id
            )
        actual, captured = capture_member_sources_v5b(
            packages=packages,
            archives=archives,
            selections=tuple(selections.values()),
            budget=budget,
        )
        v5a._measurements(actual, packages, budget)
        _require(
            type(package_result) is PackageInventoryResultV3
            and package_result.consistency == "MATCH"
            and package_result.reason_codes == (),
            "PROVENANCE_INCOMPLETE",
        )
        assert package_result is not None
        v5a._measurements(package_result.packages, packages, budget)
        _require(actual == package_result.packages, "ARTIFACT_MISMATCH")
        package_digest = v5a._claim_digest(package_result, budget)
        apt_digest = (
            None
            if apt_evidence is None
            else v5a._apt(apt_evidence, inventory_digest, snapshot, packages, budget, selection)
        )
        materialized, transformed, source_digest = _materialize(
            operations, captured, files, sources, budget
        )
        budget.check()
        return MaterializationResultV5b(
            "MATCH",
            (),
            materialized,
            tuple(sorted((*mapped, *transformed), key=lambda item: item.file_id)),
            dependencies,
            order,
            inventory_digest,
            image_digest,
            mapping_digest,
            snapshot,
            package_digest,
            apt_digest,
            source_digest,
            "NOT_PROVIDED" if apt_evidence is None else "CALLER_CLAIM_BOUND",
            unmet_requirements=v5a._UNMET + (("V4_NOT_PROVIDED",) if apt_evidence is None else ()),
        )
    except (v5a.ImageMappingRefusalV5a, contract.ContractRefusal) as exc:
        if isinstance(exc, v5a.ImageMappingRefusalV5a):
            unsupported = exc.unsupported
        reason = exc.code
        consistency: Literal["MISMATCH", "INCOMPLETE"] = (
            "MISMATCH"
            if reason
            in {
                "ARTIFACT_MISMATCH",
                "SNAPSHOT_MISMATCH",
                "PATH_COLLISION",
                "REPOSITORY_MISMATCH",
                "CONTEXT_BINDING_REJECTED",
            }
            else "INCOMPLETE"
        )
    except (KeyboardInterrupt, InterruptedError):
        consistency, reason = "INCOMPLETE", "OPERATION_INTERRUPTED"
    except MemoryError:
        consistency, reason = "INCOMPLETE", "LIMIT_EXCEEDED"
    except OSError:
        consistency, reason = "INCOMPLETE", "PROVENANCE_INCOMPLETE"
    except (
        ValueError,
        TypeError,
        AttributeError,
        UnicodeError,
        RecursionError,
        OverflowError,
        EOFError,
        zlib.error,
    ):
        consistency, reason = "INCOMPLETE", "FORMAT_REJECTED"
    return MaterializationResultV5b(
        consistency,
        (reason,),
        unsupported_transformations=tuple(sorted(set(unsupported))),
        unmet_requirements=v5a._UNMET + ("MATERIALIZATION_INCOMPLETE",),
    )
