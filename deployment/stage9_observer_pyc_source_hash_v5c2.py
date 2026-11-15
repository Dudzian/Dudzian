"""Independent, bounded PEP 552 comparison over V5b's actual source bytes.

V5c1 declarations keep their original meaning and format. This separate result
adds a local computation using a fixed, reviewed algorithm profile. Neither
material identity agreement nor a matching header attests compiler execution.
The opaque bytecode body is never interpreted, imported or deserialized.
"""

from __future__ import annotations

import hashlib
import zlib
from dataclasses import dataclass, field, fields
from typing import Any, Literal

from deployment import (
    stage9_observer_contract as contract,
    stage9_observer_image_v5a as v5a,
    stage9_observer_pyc_v5c1 as v5c1,
    stage9_observer_source_hash_algorithm_v5c2 as algorithm,
)
from deployment.canonical_json import canonical_json_bytes
from deployment.stage9_observer_member_source_v5b import (
    CapturedMemberV5b,
    FrozenPackageV5b,
    MemberSelectionV5b,
    capture_member_sources_v5b,
)
from deployment.stage9_observer_offline import Limits, _validate_limits
from deployment.stage9_observer_package_v3 import PackageInventoryResultV3, _parse
from deployment.stage9_observer_tar_v3 import TarMemberV3

# Independently measured archive identities already retained by the static
# build-materials/runtime-closure evidence. These are a local comparison profile,
# not a claim that this interpreter was invoked or its publisher accepted.
TARGET_INTERPRETER_SHA256 = "e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f"
TARGET_BUILD_ID = "337d65cf00021797985cc9f77c0cc334a9fbeb38"
_DEFAULT_LIMITS = Limits()
_UNMET = tuple(item for item in v5c1._UNMET if item != "SOURCE_HASH_ALGORITHM_UNVERIFIED") + (
    "ALGORITHM_AUTHORITY_ACCEPTANCE_UNVERIFIED",
    "EXACT_TARGET_INTERPRETER_REFERENCE_NOT_TESTED",
)


@dataclass(frozen=True, slots=True)
class SourceHashFileV5c2:
    file_id: str
    transform_id: str
    path: str
    sha256: str
    size: int
    uid: int
    gid: int
    mode: int
    source: v5c1.SourceBindingV5c1
    compiler_sha256: str
    header: v5c1.PycHeaderV5c1
    computed_source_hash_hex: str
    header_source_hash_hex: str
    algorithm_id: str
    algorithm_materials_sha256: str
    source_bytes: Literal["SOURCE_BYTES_VERIFIED"] = field(
        default="SOURCE_BYTES_VERIFIED", init=False
    )
    comparison: Literal["SOURCE_HASH_MATCH"] = field(default="SOURCE_HASH_MATCH", init=False)
    byte_identity: Literal["BYTE_IDENTITY_VERIFIED"] = field(
        default="BYTE_IDENTITY_VERIFIED", init=False
    )
    reconstruction: Literal["RECONSTRUCTION_UNVERIFIED"] = field(
        default="RECONSTRUCTION_UNVERIFIED", init=False
    )


@dataclass(frozen=True, slots=True)
class PycSourceHashResultV5c2:
    """Only a complete local comparison may publish any per-file evidence."""

    consistency: Literal[
        "HASH_MATCH_LOCAL",
        "HASH_MISMATCH",
        "ALGORITHM_EVIDENCE_INCOMPLETE",
        "INCOMPLETE",
        "NOT_DECLARED",
    ]
    reason_codes: tuple[str, ...]
    files: tuple[SourceHashFileV5c2, ...] = ()
    inventory_sha256: str | None = None
    expected_image_sha256: str | None = None
    mapping_sha256: str | None = None
    snapshot_sha256: str | None = None
    package_result_sha256: str | None = None
    source_bytes_sha256: str | None = None
    algorithm_id: str | None = None
    algorithm_materials: tuple[tuple[str, str], ...] = ()
    algorithm_materials_sha256: str | None = None
    unmet_requirements: tuple[str, ...] = _UNMET
    profile: Literal["stage9-observer-pyc-source-hash-result-v5c2"] = field(
        default="stage9-observer-pyc-source-hash-result-v5c2", init=False
    )
    scope: Literal["BOUNDED_OFFLINE_SOURCE_HASH_COMPARISON_ONLY"] = field(
        default="BOUNDED_OFFLINE_SOURCE_HASH_COMPARISON_ONLY", init=False
    )
    algorithm_acceptance: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    algorithm_evidence: Literal["ALGORITHM_EVIDENCE_INCOMPLETE"] = field(
        default="ALGORITHM_EVIDENCE_INCOMPLETE", init=False
    )
    exact_target_reference: Literal["NOT_TESTED"] = field(default="NOT_TESTED", init=False)
    pyc_reconstruction: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    reconstruction: Literal["RECONSTRUCTION_UNVERIFIED"] = field(
        default="RECONSTRUCTION_UNVERIFIED", init=False
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


def _materials(
    algorithm_id: str,
    materials: tuple[tuple[str, str], ...] | None,
    budget: v5a._MappingBudget,
) -> str:
    v5c1._require(
        type(algorithm_id) is str and algorithm_id == algorithm.ALGORITHM_ID,
        "UNSUPPORTED_ALGORITHM",
    )
    v5c1._require(materials is not None, "ALGORITHM_EVIDENCE_INCOMPLETE")
    v5c1._require(type(materials) is tuple, "ALGORITHM_MATERIAL_MISMATCH")
    assert materials is not None
    v5c1._require(
        len(materials) == len(algorithm.ALGORITHM_MATERIALS), "ALGORITHM_MATERIAL_MISMATCH"
    )
    for row, expected in zip(materials, algorithm.ALGORITHM_MATERIALS, strict=True):
        budget.compare()
        v5c1._require(
            type(row) is tuple
            and len(row) == 2
            and all(type(value) is str for value in row)
            and row == expected,
            "ALGORITHM_MATERIAL_MISMATCH",
        )
    raw = canonical_json_bytes(
        dict(algorithm_id=algorithm_id, materials=[list(row) for row in materials])
    )
    budget.read(len(raw))
    budget.reserve(len(raw) * 3)
    return hashlib.sha256(raw).hexdigest()


def verify_pyc_source_hash_v5c2(
    *,
    inventory_raw: bytes,
    expected_image_raw: bytes,
    mapping_raw: bytes,
    expected_image_sha256: str,
    archives: tuple[FrozenPackageV5b, ...],
    pyc_inputs: tuple[v5c1.FrozenPycV5c1, ...],
    algorithm_id: str = algorithm.ALGORITHM_ID,
    algorithm_materials: tuple[tuple[str, str], ...] | None = None,
    package_result: PackageInventoryResultV3 | None = None,
    limits: Limits = _DEFAULT_LIMITS,
) -> PycSourceHashResultV5c2:
    """Atomically compare independently computed hash with verified PYC bytes.

    The algorithm ID/material list must agree with the fixed local profile.
    Agreement is not external acceptance. V5c1's independently pinned complete
    artifact and source declarations are validated again against actual DEBs.
    No supplied result or claimed hash replaces computing from exact bytes.
    """
    try:
        _validate_limits(limits)
        for descriptor in fields(Limits):
            v5c1._require(
                getattr(limits, descriptor.name) <= getattr(_DEFAULT_LIMITS, descriptor.name),
                "LIMIT_EXCEEDED",
            )
        budget = v5a._MappingBudget(limits)
        materials_digest = _materials(algorithm_id, algorithm_materials, budget)
        v5a._digest(expected_image_sha256)
        v5a._preflight(inventory_raw, budget)
        packages = _parse(inventory_raw, limits, budget)
        v5c1._require(len(packages) <= v5a.MAX_PACKAGES, "LIMIT_EXCEEDED")
        v5c1._require(
            sum(len(p.control.records) + len(p.data.records) for p in packages)
            <= limits.max_archive_members,
            "LIMIT_EXCEEDED",
        )
        image = v5a._decode(expected_image_raw, budget)
        mapping = v5a._decode(mapping_raw, budget)
        image_digest = hashlib.sha256(expected_image_raw).hexdigest()
        inventory_digest = hashlib.sha256(inventory_raw).hexdigest()
        mapping_digest = hashlib.sha256(mapping_raw).hexdigest()
        v5c1._require(image_digest == expected_image_sha256, "ARTIFACT_MISMATCH")
        v5a._object(
            image,
            {
                "format",
                "image_id",
                "snapshot_sha256",
                "sources",
                "files",
                "bytecode",
                "import_roots",
            },
        )
        v5a._object(
            mapping,
            {
                "format",
                "snapshot_sha256",
                "inventory_sha256",
                "expected_image_sha256",
                "sources",
                "directory_policy",
                "transforms",
            },
        )
        v5c1._require(
            image["format"] == v5c1.EXPECTED_FORMAT and mapping["format"] == v5c1.MAPPING_FORMAT,
            "UNSUPPORTED_SOURCE",
        )
        v5c1._require(
            type(image["import_roots"]) is list and image["import_roots"] == [],
            "PYC_IMPORT_FORBIDDEN",
        )
        v5c1._require(mapping["directory_policy"] == "identical-explicit-v1", "UNSUPPORTED_SOURCE")
        v5c1._require(
            v5a._digest(mapping["inventory_sha256"]) == inventory_digest
            and v5a._digest(mapping["expected_image_sha256"]) == image_digest,
            "ARTIFACT_MISMATCH",
        )
        snapshot = v5a._digest(image["snapshot_sha256"])
        v5c1._require(v5a._digest(mapping["snapshot_sha256"]) == snapshot, "SNAPSHOT_MISMATCH")
        static = {key: image[key] for key in ("image_id", "snapshot_sha256", "sources", "files")}
        static["format"] = v5a.EXPECTED_FORMAT
        files = v5a._image(static, budget)
        sources = v5a._sources(image["sources"], packages, budget)
        v5c1._require(
            sources == v5a._sources(mapping["sources"], packages, budget), "PROVENANCE_INCOMPLETE"
        )
        expected_ops: dict[str, dict[str, Any]] = {}
        for value in v5a._rows(image["bytecode"], limits.max_files):
            budget.compare()
            row = v5a._object(value, v5c1._OP_KEYS)
            transform_id = v5a._id(row["transform_id"])
            v5c1._require(transform_id not in expected_ops, "CLOSURE_INCOMPLETE")
            expected_ops[transform_id] = row
        operations: dict[str, tuple[dict[str, Any], v5c1.SourceBindingV5c1, TarMemberV3, str]] = {}
        outputs: set[str] = set()
        selections: dict[tuple[str, str], MemberSelectionV5b] = {}
        for value in v5a._rows(mapping["transforms"], limits.max_files):
            budget.compare()
            row = v5a._object(value, v5c1._OP_KEYS)
            transform_id = v5a._id(row["transform_id"])
            v5c1._require(
                transform_id in expected_ops and transform_id not in operations,
                "CLOSURE_INCOMPLETE",
            )
            v5c1._require(row == expected_ops[transform_id], "ARTIFACT_MISMATCH")
            v5c1._require(
                row["kind"] == v5c1.TRANSFORM_KIND
                and row["bytecode_format"] == v5c1.BYTECODE_FORMAT,
                "UNSUPPORTED_TRANSFORMATION",
            )
            v5c1._require(row["invalidation"] == "checked_hash", "PYC_INVALIDATION_UNSUPPORTED")
            v5c1._require(row["usage"] == "inert_artifact_only", "PYC_IMPORT_FORBIDDEN")
            v5c1._hex(row["source_hash_hex"], 16)
            file_id = v5a._id(row["output_file_id"])
            v5c1._require(file_id in files and file_id not in outputs, "CLOSURE_INCOMPLETE")
            file = files[file_id]
            v5c1._require(
                file["type"] == "regular" and file["path"].endswith(".pyc"), "ARTIFACT_MISMATCH"
            )
            source, record = v5c1._source(row["source"], sources, snapshot, budget)
            compiler_digest = v5c1._compiler(row["compiler"], record, file["path"], limits)
            v5c1._require(
                row["compiler"]["build_id"] == TARGET_BUILD_ID
                and row["compiler"]["interpreter_sha256"] == TARGET_INTERPRETER_SHA256,
                "COMPILER_PROFILE_MISMATCH",
            )
            selections[(source.source_id, source.record_id)] = MemberSelectionV5b(
                source.source_id, source.package, source.member, source.record_id
            )
            operations[transform_id] = row, source, record, compiler_digest
            outputs.add(file_id)
            budget.reserve(8192)
        v5c1._require(set(operations) == set(expected_ops), "CLOSURE_INCOMPLETE")
        v5c1._require(
            outputs == {file_id for file_id, file in files.items() if file["type"] == "regular"},
            "CLOSURE_INCOMPLETE",
        )
        candidates = v5c1._inputs(pyc_inputs, files, budget)
        v5c1._require(set(candidates) == outputs, "CLOSURE_INCOMPLETE")
        actual, captured = capture_member_sources_v5b(
            packages=packages,
            archives=archives,
            selections=tuple(selections.values()),
            budget=budget,
        )
        v5a._measurements(actual, packages, budget)
        v5c1._require(
            type(package_result) is PackageInventoryResultV3
            and package_result.consistency == "MATCH"
            and package_result.reason_codes == (),
            "PROVENANCE_INCOMPLETE",
        )
        assert package_result is not None
        v5a._measurements(package_result.packages, packages, budget)
        v5c1._require(actual == package_result.packages, "ARTIFACT_MISMATCH")
        package_digest = v5a._claim_digest(package_result, budget)
        by_source: dict[tuple[str, str], CapturedMemberV5b] = {}
        for item in captured:
            budget.compare()
            key = item.source_id, item.record_id
            v5c1._require(key not in by_source, "PROVENANCE_INCOMPLETE")
            by_source[key] = item
        v5c1._require(set(by_source) == set(selections), "PROVENANCE_INCOMPLETE")
        staged: list[SourceHashFileV5c2] = []
        bindings: list[dict[str, Any]] = []
        for transform_id in sorted(operations):
            budget.compare()
            row, source, record, compiler_digest = operations[transform_id]
            captured_source = by_source[(source.source_id, source.record_id)]
            v5c1._require(
                captured_source.package == source.package
                and captured_source.member == source.member
                and captured_source.archive_sha256 == source.archive_sha256
                and captured_source.archive_size == source.archive_size,
                "ARTIFACT_MISMATCH",
            )
            source_digest = v5c1._hash(captured_source.content, budget)
            v5c1._require(
                source_digest == source.sha256 == record.sha256
                and len(captured_source.content) == source.size == record.size,
                "ARTIFACT_MISMATCH",
            )
            file = files[row["output_file_id"]]
            candidate = candidates[row["output_file_id"]]
            digest = v5c1._hash(candidate.content, budget)
            v5c1._require(
                digest == file["sha256"] and len(candidate.content) == file["size"],
                "ARTIFACT_MISMATCH",
            )
            header = v5c1._header(
                candidate.content, row["compiler"]["magic_hex"], row["source_hash_hex"]
            )
            budget.read(len(captured_source.content))
            budget.reserve(4096)
            computed = algorithm.independent_source_hash(
                captured_source.content,
                algorithm_id,
                magic=bytes.fromhex(header.magic_hex),
                check=budget.check,
                max_source_bytes=min(limits.max_file_bytes, limits.max_total_bytes),
            )
            v5c1._require(
                type(computed) is bytes and len(computed) == 8, "ALGORITHM_RESULT_REJECTED"
            )
            v5c1._require(computed.hex() == header.source_hash_hex, "SOURCE_HASH_MISMATCH")
            budget.reserve(8192)
            staged.append(
                SourceHashFileV5c2(
                    candidate.file_id,
                    transform_id,
                    candidate.path,
                    digest,
                    len(candidate.content),
                    candidate.uid,
                    candidate.gid,
                    candidate.mode,
                    source,
                    compiler_digest,
                    header,
                    computed.hex(),
                    header.source_hash_hex,
                    algorithm_id,
                    materials_digest,
                )
            )
            bindings.append(
                dict(
                    transform_id=transform_id,
                    source=row["source"],
                    compiler_sha256=compiler_digest,
                    source_sha256=source_digest,
                    pyc_sha256=digest,
                    computed_source_hash_hex=computed.hex(),
                    algorithm_id=algorithm_id,
                    algorithm_materials_sha256=materials_digest,
                )
            )
        budget.check()
        source_bytes_digest = hashlib.sha256(
            b"stage9-v5c2-source-bytes\0" + canonical_json_bytes(bindings)
        ).hexdigest()
        budget.check()
        if not staged:
            return PycSourceHashResultV5c2(
                "NOT_DECLARED", (), unmet_requirements=_UNMET + ("PYC_NOT_DECLARED",)
            )
        assert algorithm_materials is not None
        return PycSourceHashResultV5c2(
            "HASH_MATCH_LOCAL",
            (),
            tuple(staged),
            inventory_digest,
            image_digest,
            mapping_digest,
            snapshot,
            package_digest,
            source_bytes_digest,
            algorithm_id,
            algorithm_materials,
            materials_digest,
        )
    except (v5a.ImageMappingRefusalV5a, contract.ContractRefusal) as exc:
        reason = exc.code
        consistency: Literal["HASH_MISMATCH", "ALGORITHM_EVIDENCE_INCOMPLETE", "INCOMPLETE"]
        if reason == "ALGORITHM_EVIDENCE_INCOMPLETE":
            consistency = "ALGORITHM_EVIDENCE_INCOMPLETE"
        elif reason in {
            "ARTIFACT_MISMATCH",
            "SNAPSHOT_MISMATCH",
            "COMPILER_PROFILE_MISMATCH",
            "PYC_MAGIC_MISMATCH",
            "PYC_SOURCE_HASH_MISMATCH",
            "SOURCE_HASH_MISMATCH",
        }:
            consistency = "HASH_MISMATCH"
        else:
            consistency = "INCOMPLETE"
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
    return PycSourceHashResultV5c2(
        consistency, (reason,), unmet_requirements=_UNMET + ("SOURCE_HASH_EVIDENCE_INCOMPLETE",)
    )
