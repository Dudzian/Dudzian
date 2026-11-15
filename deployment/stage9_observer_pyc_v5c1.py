"""Bounded inert CPython bytecode evidence, never reconstruction or execution.

The independent expected-image pin fixes complete PYC bytes and declarations.
The unchanged V5b adapter remeasures DEBs and supplies actual Python source.
Only the 16-byte PEP 552 checked-hash header is interpreted; its remaining
payload is opaque, including malformed or unrelated marshal data. Header hash
agreement is a declaration comparison, not a CPython source-hash computation.
"""

from __future__ import annotations

import hashlib
import re
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

EXPECTED_FORMAT = "stage9-observer-expected-image-v5c1"
MAPPING_FORMAT = "stage9-observer-pyc-mapping-v5c1"
TRANSFORM_KIND = "prebuilt_pyc_v5c1"
GENERATION_POLICY = "prebuilt-pep552-checked-hash-v1"
BYTECODE_FORMAT = "cpython-pep552-v1"
# An explicitly supported *declaration* profile, never host-derived identity.
# CPython 3.12's importlib header magic is 3531, encoded little-endian + CRLF.
CPYTHON_VERSION = "3.12.3"
CPYTHON_MAGIC_HEX = "cb0d0d0a"
_DEFAULT_LIMITS = Limits()
_OP_KEYS = {
    "transform_id",
    "kind",
    "output_file_id",
    "source",
    "compiler",
    "bytecode_format",
    "invalidation",
    "source_hash_hex",
    "usage",
}
_SOURCE_KEYS = {
    "source_id",
    "package",
    "version",
    "architecture",
    "snapshot_sha256",
    "archive_sha256",
    "archive_size",
    "member",
    "record_id",
    "sha256",
    "size",
}
_COMPILER_KEYS = {
    "build_id",
    "interpreter_sha256",
    "version",
    "architecture",
    "magic_hex",
    "optimization",
    "co_filename",
    "target_path",
    "generation_policy",
}
_UNMET = v5a._UNMET + (
    "SOURCE_HASH_ALGORITHM_UNVERIFIED",
    "PYC_RECONSTRUCTION_UNVERIFIED",
    "COMPILER_EXECUTION_UNVERIFIED",
    "CO_FILENAME_PAYLOAD_UNVERIFIED",
    "EXPECTED_IMAGE_ACCEPTANCE_UNVERIFIED",
)


@dataclass(frozen=True, slots=True)
class FrozenPycV5c1:
    """Explicit immutable candidate bytes and caller-declared target metadata.

    Metadata agreement does not attest an installed filesystem's ownership.
    """

    file_id: str
    path: str
    content: bytes
    uid: int
    gid: int
    mode: int


@dataclass(frozen=True, slots=True)
class PycHeaderV5c1:
    magic_hex: str
    flags: int
    invalidation: Literal["checked_hash"]
    source_hash_hex: str


@dataclass(frozen=True, slots=True)
class SourceBindingV5c1:
    source_id: str
    package: str
    version: str
    architecture: str
    snapshot_sha256: str
    archive_sha256: str
    archive_size: int
    member: str
    record_id: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class VerifiedPycV5c1:
    file_id: str
    transform_id: str
    path: str
    sha256: str
    size: int
    uid: int
    gid: int
    mode: int
    source: SourceBindingV5c1
    compiler_sha256: str
    header: PycHeaderV5c1
    byte_identity: Literal["BYTE_IDENTITY_VERIFIED"] = field(
        default="BYTE_IDENTITY_VERIFIED", init=False
    )
    header_structure: Literal["HEADER_STRUCTURE_VERIFIED"] = field(
        default="HEADER_STRUCTURE_VERIFIED", init=False
    )
    source_binding: Literal["SOURCE_BINDING_VERIFIED"] = field(
        default="SOURCE_BINDING_VERIFIED", init=False
    )
    source_hash: Literal["SOURCE_HASH_UNVERIFIED"] = field(
        default="SOURCE_HASH_UNVERIFIED", init=False
    )
    reconstruction: Literal["RECONSTRUCTION_UNVERIFIED"] = field(
        default="RECONSTRUCTION_UNVERIFIED", init=False
    )


@dataclass(frozen=True, slots=True)
class PycResultV5c1:
    """BYTECODE_EVIDENCE_ONLY does not assert valid code or reconstruction."""

    consistency: Literal["BYTECODE_EVIDENCE_ONLY", "NOT_DECLARED", "MISMATCH", "INCOMPLETE"]
    reason_codes: tuple[str, ...]
    files: tuple[VerifiedPycV5c1, ...] = ()
    inventory_sha256: str | None = None
    expected_image_sha256: str | None = None
    mapping_sha256: str | None = None
    snapshot_sha256: str | None = None
    package_result_sha256: str | None = None
    source_bytes_sha256: str | None = None
    unmet_requirements: tuple[str, ...] = _UNMET
    profile: Literal["stage9-observer-pyc-result-v5c1"] = field(
        default="stage9-observer-pyc-result-v5c1", init=False
    )
    scope: Literal["BOUNDED_OFFLINE_PYC_EVIDENCE_ONLY"] = field(
        default="BOUNDED_OFFLINE_PYC_EVIDENCE_ONLY", init=False
    )
    pyc_reconstruction: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    reconstruction: Literal["RECONSTRUCTION_UNVERIFIED"] = field(
        default="RECONSTRUCTION_UNVERIFIED", init=False
    )
    source_hash: Literal["SOURCE_HASH_UNVERIFIED"] = field(
        default="SOURCE_HASH_UNVERIFIED", init=False
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


def _require(condition: bool, code: str = "FORMAT_REJECTED") -> None:
    if not condition:
        raise v5a.ImageMappingRefusalV5a(code)


def _hex(value: Any, length: int) -> str:
    _require(type(value) is str and len(value) == length)
    _require(re.fullmatch("[0-9a-f]+", value) is not None)
    return str(value)


def _source(
    value: Any,
    sources: dict[str, UbuntuPackageInventoryV3],
    snapshot: str,
    budget: v5a._MappingBudget,
) -> tuple[SourceBindingV5c1, TarMemberV3]:
    row = v5a._object(value, _SOURCE_KEYS)
    source_id = v5a._id(row["source_id"])
    _require(source_id in sources, "PROVENANCE_INCOMPLETE")
    package = sources[source_id]
    _require(v5a._digest(row["snapshot_sha256"]) == snapshot, "SNAPSHOT_MISMATCH")
    record_id = v5a._digest(row["record_id"])
    matches: list[TarMemberV3] = []
    for record in package.data.records:
        budget.compare()
        if v5a.record_id_v5a(record) == record_id:
            matches.append(record)
    _require(len(matches) == 1, "PROVENANCE_INCOMPLETE")
    record = matches[0]
    _require(record.typeflag in {"0", "\0"} and record.name.endswith(".py"), "UNSUPPORTED_SOURCE")
    contract._integer(row["archive_size"], maximum=budget.limits.max_file_bytes)
    contract._integer(row["size"], maximum=budget.limits.max_file_bytes)
    expected = dict(
        source_id=source_id,
        package=package.package,
        version=package.version,
        architecture=package.architecture,
        snapshot_sha256=snapshot,
        archive_sha256=package.sha256,
        archive_size=package.size,
        member=record.name,
        record_id=record_id,
        sha256=record.sha256,
        size=record.size,
    )
    _require(row == expected, "ARTIFACT_MISMATCH")
    return SourceBindingV5c1(**row), record


def _compiler(value: Any, record: TarMemberV3, path: str, limits: Limits) -> str:
    row = v5a._object(value, _COMPILER_KEYS)
    build_id = row["build_id"]
    # The bounded Ubuntu amd64 profile declares the complete 20-byte GNU ID,
    # never an abbreviated prefix. Agreement still does not attest invocation.
    _hex(build_id, 40)
    v5a._digest(row["interpreter_sha256"])
    _require(
        row["version"] == CPYTHON_VERSION
        and row["architecture"] == "amd64"
        and row["magic_hex"] == CPYTHON_MAGIC_HEX,
        "COMPILER_PROFILE_MISMATCH",
    )
    optimization = contract._integer(row["optimization"], maximum=2)
    _require(row["generation_policy"] == GENERATION_POLICY, "UNSUPPORTED_TRANSFORMATION")
    _require(
        v5a._path(row["co_filename"], limits) == "/" + record.canonical_name,
        "COMPILER_PROFILE_MISMATCH",
    )
    _require(v5a._path(row["target_path"], limits) == path, "COMPILER_PROFILE_MISMATCH")
    stem = record.canonical_name.rsplit("/", 1)[-1][:-3]
    suffix = "" if optimization == 0 else ".opt-" + str(optimization)
    _require(
        path.rsplit("/", 1)[-1] == stem + ".cpython-312" + suffix + ".pyc",
        "COMPILER_PROFILE_MISMATCH",
    )
    return hashlib.sha256(canonical_json_bytes(row)).hexdigest()


def _header(raw: bytes, magic_hex: str, source_hash_hex: str) -> PycHeaderV5c1:
    """Read a fixed header only; the opaque body is never deserialized."""
    _require(len(raw) >= 16, "PYC_HEADER_REJECTED")
    _require(raw[:4].hex() == magic_hex, "PYC_MAGIC_MISMATCH")
    flags = int.from_bytes(raw[4:8], "little")
    # Reserved bits and check-source without hash-based bit are invalid.
    _require(flags in {0, 1, 3}, "PYC_FLAGS_REJECTED")
    _require(flags == 3, "PYC_INVALIDATION_UNSUPPORTED")
    _require(raw[8:16].hex() == source_hash_hex, "PYC_SOURCE_HASH_MISMATCH")
    return PycHeaderV5c1(raw[:4].hex(), flags, "checked_hash", raw[8:16].hex())


def _inputs(
    inputs: tuple[FrozenPycV5c1, ...],
    files: dict[str, dict[str, Any]],
    budget: v5a._MappingBudget,
) -> dict[str, FrozenPycV5c1]:
    _require(type(inputs) is tuple, "FORMAT_REJECTED")
    _require(len(inputs) <= budget.limits.max_files, "LIMIT_EXCEEDED")
    result: dict[str, FrozenPycV5c1] = {}
    for item in inputs:
        budget.compare()
        _require(
            type(item) is FrozenPycV5c1 and type(item.content) is bytes, "FILE_POLICY_REJECTED"
        )
        file_id = v5a._id(item.file_id)
        _require(file_id not in result and file_id in files, "CLOSURE_INCOMPLETE")
        file = files[file_id]
        _require(file["type"] == "regular", "ARTIFACT_MISMATCH")
        _require(len(item.content) <= budget.limits.max_file_bytes, "LIMIT_EXCEEDED")
        _require(len(item.content) >= 16, "PYC_HEADER_REJECTED")
        _require(v5a._path(item.path, budget.limits) == file["path"], "ARTIFACT_MISMATCH")
        for name in ("uid", "gid", "mode"):
            contract._integer(getattr(item, name), maximum=0o777 if name == "mode" else 2**32 - 2)
            _require(getattr(item, name) == file[name], "ARTIFACT_MISMATCH")
        budget.reserve(len(item.content) + 4096)
        result[file_id] = item
    return result


def _hash(raw: bytes, budget: v5a._MappingBudget) -> str:
    digest = hashlib.sha256()
    for offset in range(0, len(raw), budget.limits.chunk_bytes):
        budget.compare()
        part = raw[offset : offset + budget.limits.chunk_bytes]
        budget.read(len(part))
        digest.update(part)
    return digest.hexdigest()


def verify_pyc_v5c1(
    *,
    inventory_raw: bytes,
    expected_image_raw: bytes,
    mapping_raw: bytes,
    expected_image_sha256: str,
    archives: tuple[FrozenPackageV5b, ...],
    pyc_inputs: tuple[FrozenPycV5c1, ...],
    package_result: PackageInventoryResultV3 | None = None,
    limits: Limits = _DEFAULT_LIMITS,
) -> PycResultV5c1:
    """Compare pinned bytecode evidence atomically, with no execution authority.

    The expected-image SHA-256 is supplied independently; neither its PYC hash
    nor source-header hash is filled from candidate bytes. Compiler descriptors
    are caller claims independently repeated by the pinned expected image.
    No proof input or successful byte comparison can promote reconstruction.
    """
    try:
        _validate_limits(limits)
        budget = v5a._MappingBudget(limits)
        v5a._digest(expected_image_sha256)
        v5a._preflight(inventory_raw, budget)
        packages = _parse(inventory_raw, limits, budget)
        _require(len(packages) <= v5a.MAX_PACKAGES, "LIMIT_EXCEEDED")
        _require(
            sum(len(p.control.records) + len(p.data.records) for p in packages)
            <= limits.max_archive_members,
            "LIMIT_EXCEEDED",
        )
        image = v5a._decode(expected_image_raw, budget)
        mapping = v5a._decode(mapping_raw, budget)
        image_digest = hashlib.sha256(expected_image_raw).hexdigest()
        inventory_digest = hashlib.sha256(inventory_raw).hexdigest()
        mapping_digest = hashlib.sha256(mapping_raw).hexdigest()
        _require(image_digest == expected_image_sha256, "ARTIFACT_MISMATCH")
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
        _require(
            image["format"] == EXPECTED_FORMAT and mapping["format"] == MAPPING_FORMAT,
            "UNSUPPORTED_SOURCE",
        )
        _require(
            type(image["import_roots"]) is list and image["import_roots"] == [],
            "PYC_IMPORT_FORBIDDEN",
        )
        _require(mapping["directory_policy"] == "identical-explicit-v1", "UNSUPPORTED_SOURCE")
        _require(
            v5a._digest(mapping["inventory_sha256"]) == inventory_digest
            and v5a._digest(mapping["expected_image_sha256"]) == image_digest,
            "ARTIFACT_MISMATCH",
        )
        snapshot = v5a._digest(image["snapshot_sha256"])
        _require(v5a._digest(mapping["snapshot_sha256"]) == snapshot, "SNAPSHOT_MISMATCH")
        static = {key: image[key] for key in ("image_id", "snapshot_sha256", "sources", "files")}
        static["format"] = v5a.EXPECTED_FORMAT
        files = v5a._image(static, budget)
        sources = v5a._sources(image["sources"], packages, budget)
        _require(
            sources == v5a._sources(mapping["sources"], packages, budget), "PROVENANCE_INCOMPLETE"
        )
        expected_ops: dict[str, dict[str, Any]] = {}
        for value in v5a._rows(image["bytecode"], limits.max_files):
            budget.compare()
            row = v5a._object(value, _OP_KEYS)
            transform_id = v5a._id(row["transform_id"])
            _require(transform_id not in expected_ops, "CLOSURE_INCOMPLETE")
            expected_ops[transform_id] = row
        operations: dict[str, tuple[dict[str, Any], SourceBindingV5c1, TarMemberV3, str]] = {}
        outputs: set[str] = set()
        selections: dict[tuple[str, str], MemberSelectionV5b] = {}
        for value in v5a._rows(mapping["transforms"], limits.max_files):
            budget.compare()
            row = v5a._object(value, _OP_KEYS)
            transform_id = v5a._id(row["transform_id"])
            _require(
                transform_id in expected_ops and transform_id not in operations,
                "CLOSURE_INCOMPLETE",
            )
            _require(row == expected_ops[transform_id], "ARTIFACT_MISMATCH")
            _require(
                row["kind"] == TRANSFORM_KIND and row["bytecode_format"] == BYTECODE_FORMAT,
                "UNSUPPORTED_TRANSFORMATION",
            )
            _require(row["invalidation"] == "checked_hash", "PYC_INVALIDATION_UNSUPPORTED")
            _require(row["usage"] == "inert_artifact_only", "PYC_IMPORT_FORBIDDEN")
            _hex(row["source_hash_hex"], 16)
            file_id = v5a._id(row["output_file_id"])
            _require(file_id in files and file_id not in outputs, "CLOSURE_INCOMPLETE")
            file = files[file_id]
            _require(
                file["type"] == "regular" and file["path"].endswith(".pyc"), "ARTIFACT_MISMATCH"
            )
            source, record = _source(row["source"], sources, snapshot, budget)
            compiler_digest = _compiler(row["compiler"], record, file["path"], limits)
            selections[(source.source_id, source.record_id)] = MemberSelectionV5b(
                source.source_id, source.package, source.member, source.record_id
            )
            operations[transform_id] = row, source, record, compiler_digest
            outputs.add(file_id)
            budget.reserve(8192)
        _require(set(operations) == set(expected_ops), "CLOSURE_INCOMPLETE")
        _require(
            outputs == {file_id for file_id, file in files.items() if file["type"] == "regular"},
            "CLOSURE_INCOMPLETE",
        )
        candidates = _inputs(pyc_inputs, files, budget)
        _require(set(candidates) == outputs, "CLOSURE_INCOMPLETE")
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
        by_source: dict[tuple[str, str], CapturedMemberV5b] = {}
        for item in captured:
            budget.compare()
            key = item.source_id, item.record_id
            _require(key not in by_source, "PROVENANCE_INCOMPLETE")
            by_source[key] = item
        _require(set(by_source) == set(selections), "PROVENANCE_INCOMPLETE")
        staged: list[VerifiedPycV5c1] = []
        bindings: list[dict[str, Any]] = []
        for transform_id in sorted(operations):
            budget.compare()
            row, source, record, compiler_digest = operations[transform_id]
            captured_source = by_source[(source.source_id, source.record_id)]
            _require(
                captured_source.package == source.package
                and captured_source.member == source.member
                and captured_source.archive_sha256 == source.archive_sha256
                and captured_source.archive_size == source.archive_size,
                "ARTIFACT_MISMATCH",
            )
            source_digest = _hash(captured_source.content, budget)
            _require(
                source_digest == source.sha256 == record.sha256
                and len(captured_source.content) == source.size == record.size,
                "ARTIFACT_MISMATCH",
            )
            file = files[row["output_file_id"]]
            candidate = candidates[row["output_file_id"]]
            digest = _hash(candidate.content, budget)
            _require(
                digest == file["sha256"] and len(candidate.content) == file["size"],
                "ARTIFACT_MISMATCH",
            )
            header = _header(
                candidate.content, row["compiler"]["magic_hex"], row["source_hash_hex"]
            )
            budget.reserve(8192)
            staged.append(
                VerifiedPycV5c1(
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
                )
            )
            bindings.append(
                dict(
                    transform_id=transform_id,
                    source=row["source"],
                    compiler_sha256=compiler_digest,
                    source_sha256=source_digest,
                    pyc_sha256=digest,
                )
            )
        budget.check()
        source_bytes_digest = hashlib.sha256(
            b"stage9-v5c1-source-bytes\0" + canonical_json_bytes(bindings)
        ).hexdigest()
        budget.check()
        return PycResultV5c1(
            "BYTECODE_EVIDENCE_ONLY" if staged else "NOT_DECLARED",
            (),
            tuple(staged),
            inventory_digest,
            image_digest,
            mapping_digest,
            snapshot,
            package_digest,
            source_bytes_digest,
            unmet_requirements=_UNMET + (() if staged else ("PYC_NOT_DECLARED",)),
        )
    except (v5a.ImageMappingRefusalV5a, contract.ContractRefusal) as exc:
        reason = exc.code
        consistency: Literal["MISMATCH", "INCOMPLETE"] = (
            "MISMATCH"
            if reason
            in {
                "ARTIFACT_MISMATCH",
                "SNAPSHOT_MISMATCH",
                "COMPILER_PROFILE_MISMATCH",
                "PYC_MAGIC_MISMATCH",
                "PYC_SOURCE_HASH_MISMATCH",
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
    return PycResultV5c1(
        consistency, (reason,), unmet_requirements=_UNMET + ("PYC_EVIDENCE_INCOMPLETE",)
    )
