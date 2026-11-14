"""Bounded, inert archive-to-image agreement, without installation or authority.

V5a has its own static declarations. It neither activates the V1 Observer nor
changes its provenance/runtime semantics. Caller-supplied measurement objects
are compared as claims; Python objects cannot authenticate their own execution.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Literal, cast

from deployment import (
    stage9_observer_apt_v4a as v4a,
    stage9_observer_apt_v4b as v4b,
    stage9_observer_apt_v4c as v4c,
    stage9_observer_checkpoint_v4c as checkpoint,
    stage9_observer_contract as contract,
)
from deployment.canonical_json import canonical_json_bytes
from deployment.stage9_observer_gpgv_v4b import GpgvEvidenceV4b
from deployment.stage9_observer_offline import Limits, _Budget, _validate_limits
from deployment.stage9_observer_package_v3 import (
    PackageInventoryResultV3,
    PackageMeasurementV3,
    UbuntuPackageInventoryV3,
    _control_identity,
    _parse,
)
from deployment.stage9_observer_tar_v3 import TarInventoryV3, TarMemberV3, TarResultV3

EXPECTED_FORMAT = "stage9-observer-expected-image-v5a"
MAPPING_FORMAT = "stage9-observer-image-mapping-v5a"
MAX_DECLARATION_BYTES = 2 * 1024 * 1024
MAX_PACKAGES = 128
MAX_DEPTH = 16
_DEFAULT_LIMITS = Limits()
_SOURCE_FIELDS = {"source_id", "package", "version", "architecture", "filename", "sha256", "size"}
_UNMET = (
    "EXTERNAL_AUTHORITY_ACCEPTANCE_UNVERIFIED",
    "AUTHENTICATED_TIME_UNVERIFIED",
    "DURABLE_ANTI_ROLLBACK_BLOCKED",
    "RUNTIME_QUALIFICATION_BLOCKED",
    "HOST_STATE_UNVERIFIED",
    "SNAPSHOT_MEMBERSHIP_UNVERIFIED",
)
_V4B_BLOCKERS = (
    "EXTERNAL_AUTHORITY_UNVERIFIED",
    "FRESHNESS_BLOCKED",
    "CHECKPOINT_BLOCKED",
    "ROLLBACK_BLOCKED",
    "TOOL_ACCEPTANCE_UNVERIFIED",
    "SYSTEM_ISOLATION_UNVERIFIED",
)


class ImageMappingRefusalV5a(ValueError):
    """V5a-only diagnostics; the closed V1 vocabulary remains unchanged."""

    def __init__(self, code: str, unsupported: tuple[str, ...] = ()) -> None:
        self.code = code
        self.unsupported = unsupported
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class AptEvidenceV5a:
    """Explicit caller association, never authenticated evidence transport."""

    inventory_sha256: str
    snapshot_sha256: str
    declaration: v4a.AptDeclarationV4a
    result: v4a.AptResultV4a
    signature_result: v4b.AptResultV4b | None = None
    acceptance_result: v4c.AptResultV4c | None = None


@dataclass(frozen=True, slots=True)
class MappedOriginV5a:
    source_id: str
    package: str
    version: str
    record_id: str
    member: str


@dataclass(frozen=True, slots=True)
class MappedFileV5a:
    file_id: str
    path: str
    type: str
    sha256: str | None
    size: int
    uid: int
    gid: int
    mode: int
    origins: tuple[MappedOriginV5a, ...]


@dataclass(frozen=True, slots=True)
class ImageMappingResultV5a:
    """MATCH concerns only the complete supported static mapping profile."""

    consistency: Literal["MATCH", "MISMATCH", "INCOMPLETE"]
    reason_codes: tuple[str, ...]
    mapped_files: tuple[MappedFileV5a, ...] = ()
    inventory_sha256: str | None = None
    expected_image_sha256: str | None = None
    mapping_sha256: str | None = None
    snapshot_sha256: str | None = None
    package_result_sha256: str | None = None
    apt_evidence_sha256: str | None = None
    v4_binding: Literal["NOT_PROVIDED", "CALLER_CLAIM_BOUND"] = "NOT_PROVIDED"
    unsupported_transformations: tuple[str, ...] = ()
    unmet_requirements: tuple[str, ...] = _UNMET
    profile: Literal["stage9-observer-image-result-v5a"] = field(
        default="stage9-observer-image-result-v5a", init=False
    )
    scope: Literal["STATIC_ARCHIVE_TO_IMAGE_ONLY"] = field(
        default="STATIC_ARCHIVE_TO_IMAGE_ONLY", init=False
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
    STAGE_3A_COMPLETE: Literal["NO"] = field(default="NO", init=False)
    PRODUCTION_QUALIFIED: Literal["NO"] = field(default="NO", init=False)
    production_status: Literal["PRODUCTION_NO_GO"] = field(default="PRODUCTION_NO_GO", init=False)
    refused: Literal[True] = field(default=True, init=False)


def _require(condition: bool, code: str = "FORMAT_REJECTED") -> None:
    if not condition:
        raise ImageMappingRefusalV5a(code)


class _MappingBudget(_Budget):
    """One shared budget, including bounded JSON scans and indexed comparisons."""

    def __init__(self, limits: Limits) -> None:
        super().__init__(limits)
        self.comparisons = 0
        self.comparison_limit = (limits.max_archive_members + limits.max_files + MAX_PACKAGES) * (
            limits.max_path_depth + 32
        )

    def compare(self) -> None:
        self.check()
        self.comparisons += 1
        _require(self.comparisons <= self.comparison_limit, "LIMIT_EXCEEDED")


def _preflight(raw: bytes, budget: _MappingBudget) -> None:
    """Null-aware lexical bounds before either JSON decoder can recurse."""
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_DECLARATION_BYTES, "LIMIT_EXCEEDED")
    budget.read(len(raw))
    budget.reserve(len(raw) * 20)
    depth = 0
    in_string = escaped = False
    for index, char in enumerate(raw):
        if index % 4096 == 0:
            budget.check()
        if in_string:
            if escaped:
                escaped = False
            elif char == 92:
                escaped = True
            elif char == 34:
                in_string = False
        elif char == 34:
            in_string = True
        elif char in (91, 123):
            depth += 1
            _require(depth <= MAX_DEPTH, "LIMIT_EXCEEDED")
        elif char in (93, 125):
            depth -= 1
            _require(depth >= 0)
    _require(depth == 0 and not in_string)
    budget.check()


def _decode(raw: bytes, budget: _MappingBudget) -> dict[str, Any]:
    _preflight(raw, budget)
    # The existing strict decoder rejects duplicate keys, null, floats,
    # oversized integers, control characters and malformed JSON.
    value = contract._decode(raw, MAX_DECLARATION_BYTES)
    _require(type(value) is dict)
    budget.check()
    return cast(dict[str, Any], value)


def _rows(value: Any, maximum: int) -> list[Any]:
    _require(type(value) is list)
    _require(len(value) <= maximum, "LIMIT_EXCEEDED")
    return cast(list[Any], value)


def _object(value: Any, keys: set[str]) -> dict[str, Any]:
    _require(type(value) is dict and set(value) == keys)
    return cast(dict[str, Any], value)


def _digest(value: Any) -> str:
    return contract._digest(value)


def _id(value: Any) -> str:
    return contract._identifier(value)


def _path(value: Any, limits: Limits) -> str:
    result = contract._path(value)
    _require(len(result.split("/")) - 1 <= limits.max_path_depth, "LIMIT_EXCEEDED")
    return result


def record_id_v5a(record: TarMemberV3) -> str:
    """Domain-separated hash of every literal V3 record field, including name.

    This helper identifies declarations; it does not measure archive bytes.
    The verifier invokes it only after the bounded V3 inventory parser.
    """
    _require(type(record) is TarMemberV3)
    values: dict[str, Any] = {}
    for item in fields(record):
        value = getattr(record, item.name)
        # Keep the standalone ID helper bounded even outside the verifier.
        # Actual inventory records have V3's much narrower per-field bounds.
        if type(value) is str:
            _require(len(value) <= 4096, "LIMIT_EXCEEDED")
        elif type(value) is int:
            contract._integer(value)
        else:
            _require(value is None)
        values[item.name] = value
    raw = canonical_json_bytes(values)
    return hashlib.sha256(b"stage9-v5a-record\0" + raw).hexdigest()


def _claim(value: Any, budget: _MappingBudget, depth: int = 0) -> Any:
    """Bounded serialization of supplied measurements, not an attestation."""
    budget.compare()
    _require(depth < MAX_DEPTH, "LIMIT_EXCEEDED")
    budget.reserve(256)
    if value is None or type(value) in (bool, int):
        if type(value) is int:
            contract._integer(value)
        return value
    if type(value) in (str, bytes):
        _require(len(value) <= 65536, "LIMIT_EXCEEDED")
        budget.reserve(len(value) * 16)
        return value.hex() if type(value) is bytes else value
    if type(value) is tuple:
        _require(len(value) <= budget.limits.max_archive_members, "LIMIT_EXCEEDED")
        budget.reserve(len(value) * 256)
        return [_claim(item, budget, depth + 1) for item in value]
    _require(
        type(value)
        in {
            AptEvidenceV5a,
            v4a.AptDeclarationV4a,
            v4a.AptResultV4a,
            v4b.AptResultV4b,
            v4c.AptResultV4c,
            checkpoint.SnapshotStateV4c,
            GpgvEvidenceV4b,
            PackageInventoryResultV3,
            PackageMeasurementV3,
            TarResultV3,
        }
    )
    _require(is_dataclass(value) and not isinstance(value, type))
    descriptors = fields(value)
    _require(len(descriptors) <= 64, "LIMIT_EXCEEDED")
    budget.reserve(sum(256 + len(item.name) * 16 for item in descriptors))
    return {item.name: _claim(getattr(value, item.name), budget, depth + 1) for item in descriptors}


def _claim_digest(value: Any, budget: _MappingBudget) -> str:
    return hashlib.sha256(canonical_json_bytes(_claim(value, budget))).hexdigest()


def _sources(
    value: Any, packages: tuple[UbuntuPackageInventoryV3, ...], budget: _MappingBudget
) -> dict[str, UbuntuPackageInventoryV3]:
    rows = _rows(value, min(MAX_PACKAGES, budget.limits.max_files))
    by_name = {package.package: package for package in packages}
    result: dict[str, UbuntuPackageInventoryV3] = {}
    seen: set[str] = set()
    for row in rows:
        budget.compare()
        source = _object(row, _SOURCE_FIELDS)
        source_id = _id(source["source_id"])
        _require(source_id not in result, "CLOSURE_INCOMPLETE")
        name = contract._launch_exact_text(source["package"])
        _require(name in by_name and name not in seen, "PROVENANCE_INCOMPLETE")
        package = by_name[name]
        expected = {
            key: getattr(package, key)
            for key in ("package", "version", "architecture", "filename", "sha256", "size")
        }
        _require(type(source["size"]) is int)
        _require(
            {key: item for key, item in source.items() if key != "source_id"} == expected,
            "ARTIFACT_MISMATCH",
        )
        result[source_id] = package
        seen.add(name)
    _require(seen == set(by_name), "PROVENANCE_INCOMPLETE")
    return result


def _image(value: dict[str, Any], budget: _MappingBudget) -> dict[str, dict[str, Any]]:
    _object(value, {"format", "image_id", "snapshot_sha256", "sources", "files"})
    _require(value["format"] == EXPECTED_FORMAT, "UNSUPPORTED_SOURCE")
    _id(value["image_id"])
    _digest(value["snapshot_sha256"])
    result: dict[str, dict[str, Any]] = {}
    paths: dict[str, str] = {}
    size = 0
    for row in _rows(value["files"], budget.limits.max_files):
        budget.compare()
        _require(type(row) is dict)
        kind = contract._enum(row.get("type"), {"regular", "directory"})
        keys = {"file_id", "path", "type", "size", "uid", "gid", "mode"}
        if kind == "regular":
            keys.add("sha256")
        file = _object(row, keys)
        file_id = _id(file["file_id"])
        path = _path(file["path"], budget.limits)
        _require(file_id not in result and path not in paths, "CLOSURE_INCOMPLETE")
        contract._integer(file["uid"], maximum=2**32 - 2)
        contract._integer(file["gid"], maximum=2**32 - 2)
        contract._integer(file["mode"], maximum=0o777)
        _require(file["mode"] & 0o022 == 0, "FILE_POLICY_REJECTED")
        contract._integer(file["size"], maximum=budget.limits.max_file_bytes)
        if kind == "regular":
            _digest(file["sha256"])
            size += file["size"]
            _require(size <= budget.limits.max_unpacked_bytes, "LIMIT_EXCEEDED")
        else:
            _require(file["size"] == 0)
        budget.reserve(2048 + len(path) * 16)
        result[file_id] = file
        paths[path] = kind
    _require(paths.get("/") == "directory", "CLOSURE_INCOMPLETE")
    for path in paths:
        budget.compare()
        if path != "/":
            parent = path.rpartition("/")[0] or "/"
            _require(paths.get(parent) == "directory", "CLOSURE_INCOMPLETE")
    return result


def _tar_measurement(actual: TarResultV3, expected: TarInventoryV3, budget: _MappingBudget) -> None:
    _require(type(actual) is TarResultV3, "PROVENANCE_INCOMPLETE")
    counts = {"regular": 0, "directories": 0, "symlinks": 0}
    payload: TarMemberV3 | None = None
    for record in expected.records:
        budget.compare()
        kind = (
            "regular"
            if record.typeflag in {"0", "\0"}
            else ("directories" if record.typeflag == "5" else "symlinks")
        )
        counts[kind] += 1
        if record.canonical_name == "control" and kind == "regular":
            payload = record
    for name, wanted in (
        ("unpacked_bytes", expected.unpacked_bytes),
        ("members", len(expected.records)),
        ("profile", expected.profile),
        ("sha256", expected.unpacked_sha256),
        *counts.items(),
    ):
        budget.compare()
        observed = getattr(actual, name)
        _require(type(observed) is type(wanted) and observed == wanted, "ARTIFACT_MISMATCH")
    if payload is None:
        _require(actual.control is None, "ARTIFACT_MISMATCH")
    else:
        _require(type(actual.control) is bytes, "PROVENANCE_INCOMPLETE")
        assert actual.control is not None
        _require(len(actual.control) <= 65536, "LIMIT_EXCEEDED")
        budget.reserve(len(actual.control) * 16)
        _require(
            len(actual.control) == payload.size
            and hashlib.sha256(actual.control).hexdigest() == payload.sha256,
            "ARTIFACT_MISMATCH",
        )


def _measurements(
    actual: tuple[PackageMeasurementV3, ...],
    packages: tuple[UbuntuPackageInventoryV3, ...],
    budget: _MappingBudget,
) -> None:
    _require(type(actual) is tuple, "PROVENANCE_INCOMPLETE")
    _require(len(actual) == len(packages), "PROVENANCE_INCOMPLETE")
    by_name = {package.package: package for package in packages}
    seen: set[str] = set()
    for item in actual:
        budget.compare()
        _require(type(item) is PackageMeasurementV3, "PROVENANCE_INCOMPLETE")
        _require(
            type(item.package) is str and item.package in by_name and item.package not in seen,
            "PROVENANCE_INCOMPLETE",
        )
        seen.add(item.package)
        expected = by_name[item.package]
        _require(
            type(item.archive_bytes) is int
            and item.archive_bytes == expected.size
            and type(item.archive_sha256) is str
            and item.archive_sha256 == expected.sha256,
            "ARTIFACT_MISMATCH",
        )
        _tar_measurement(item.control, expected.control, budget)
        _tar_measurement(item.data, expected.data, budget)
        _control_identity(item.control.control, expected, budget)


def _apt(
    evidence: AptEvidenceV5a,
    inventory_digest: str,
    snapshot: str,
    packages: tuple[UbuntuPackageInventoryV3, ...],
    budget: _MappingBudget,
) -> str:
    _require(type(evidence) is AptEvidenceV5a, "PROVENANCE_INCOMPLETE")
    _require(
        evidence.inventory_sha256 == inventory_digest and evidence.snapshot_sha256 == snapshot,
        "SNAPSHOT_MISMATCH",
    )
    _require(type(evidence.declaration) is v4a.AptDeclarationV4a, "PROVENANCE_INCOMPLETE")
    v4a._selection(evidence.declaration, budget.limits)
    _require(evidence.declaration.release_sha256 == snapshot, "SNAPSHOT_MISMATCH")
    result = evidence.result
    _require(
        type(result) is v4a.AptResultV4a
        and result.consistency == "MATCH"
        and result.reason_codes == (),
        "PROVENANCE_INCOMPLETE",
    )
    _measurements(result.packages, packages, budget)
    signature = evidence.signature_result
    if signature is not None:
        _require(type(signature) is v4b.AptResultV4b, "PROVENANCE_INCOMPLETE")
        _require(signature.release_sha256 == snapshot, "SNAPSHOT_MISMATCH")
        _require(
            signature.cryptographic_signature == "VALID"
            and signature.signer_policy == "MATCH"
            and signature.reason_codes == _V4B_BLOCKERS
            and type(signature.evidence) is GpgvEvidenceV4b
            and type(signature.v4a) is v4a.AptResultV4a,
            "PROVENANCE_INCOMPLETE",
        )
        assert signature.v4a is not None
        assert signature.evidence is not None
        signed = signature.evidence
        for fingerprint in (signed.signing_fingerprint, signed.primary_fingerprint):
            budget.compare()
            _require(
                type(fingerprint) is str and re.fullmatch(r"[0-9A-F]{40}", fingerprint) is not None,
                "PROVENANCE_INCOMPLETE",
            )
        for number, allowed in (
            (signed.public_key_algorithm, {1, 22}),
            (signed.hash_algorithm, {8, 10}),
        ):
            _require(type(number) is int and number in allowed, "PROVENANCE_INCOMPLETE")
        for timestamp in (signed.signature_timestamp, signed.signature_expiry):
            _require(
                type(timestamp) is int and 0 <= timestamp <= 99999999999, "PROVENANCE_INCOMPLETE"
            )
        _require(
            signed.signature_timestamp > 0
            and (
                signed.signature_expiry == 0 or signed.signature_expiry > signed.signature_timestamp
            ),
            "PROVENANCE_INCOMPLETE",
        )
        _require(
            signature.v4a.consistency == "MATCH" and signature.v4a.reason_codes == (),
            "PROVENANCE_INCOMPLETE",
        )
        _measurements(signature.v4a.packages, packages, budget)
    acceptance = evidence.acceptance_result
    if acceptance is not None:
        _require(
            type(acceptance) is v4c.AptResultV4c and signature is not None, "PROVENANCE_INCOMPLETE"
        )
        _require(
            acceptance.technical_validation == "VALID"
            and acceptance.reason_codes == v4c._BLOCKERS
            and type(acceptance.proposed_state) is checkpoint.SnapshotStateV4c,
            "PROVENANCE_INCOMPLETE",
        )
        assert acceptance.proposed_state is not None
        try:
            checkpoint._record(acceptance.proposed_state, "PROVENANCE_INCOMPLETE")
        except ValueError:
            raise ImageMappingRefusalV5a("PROVENANCE_INCOMPLETE") from None
        _require(not acceptance.proposed_state.revoked, "PROVENANCE_INCOMPLETE")
        _require(acceptance.proposed_state.snapshot_sha256 == snapshot, "SNAPSHOT_MISMATCH")
    return _claim_digest(evidence, budget)


def _references(
    mapping: dict[str, Any],
    sources: dict[str, UbuntuPackageInventoryV3],
    files: dict[str, dict[str, Any]],
    budget: _MappingBudget,
) -> tuple[dict[str, list[tuple[str, str, TarMemberV3]]], tuple[str, ...]]:
    records: dict[tuple[str, str], TarMemberV3] = {}
    for source_id, package in sources.items():
        for record in package.data.records:
            budget.compare()
            key = (source_id, record_id_v5a(record))
            _require(key not in records, "CLOSURE_INCOMPLETE")
            budget.reserve(1536)
            records[key] = record
    seen: set[tuple[str, str]] = set()
    direct: dict[str, list[tuple[str, str, TarMemberV3]]] = {}
    transforms: dict[str, dict[str, Any]] = {}
    unsupported: set[str] = set()

    def reference(value: Any) -> tuple[str, str, TarMemberV3]:
        ref = _object(value, {"source_id", "record_id", "member"})
        key = (_id(ref["source_id"]), _digest(ref["record_id"]))
        budget.compare()
        _require(key in records, "PROVENANCE_INCOMPLETE")
        record = records[key]
        _require(type(ref["member"]) is str and ref["member"] == record.name, "ARTIFACT_MISMATCH")
        return key[0], key[1], record

    for row in _rows(mapping["transforms"], budget.limits.max_files):
        budget.compare()
        transform = _object(row, {"transform_id", "kind", "inputs", "output_file_id"})
        transform_id = _id(transform["transform_id"])
        _require(transform_id not in transforms, "CLOSURE_INCOMPLETE")
        kind = contract._enum(transform["kind"], {"materialize_member", "pyc"})
        inputs = _rows(transform["inputs"], 1)
        _require(len(inputs) == 1, "PROVENANCE_INCOMPLETE")
        source_id, record_id, _ = reference(inputs[0])
        key = (source_id, record_id)
        _require(key not in seen, "CLOSURE_INCOMPLETE")
        seen.add(key)
        output = _id(transform["output_file_id"])
        _require(output in files and files[output]["type"] == "regular", "CLOSURE_INCOMPLETE")
        transforms[transform_id] = transform
        unsupported.add(kind)

    outputs: set[str] = set()
    for row in _rows(mapping["entries"], budget.limits.max_archive_members):
        budget.compare()
        _require(type(row) is dict)
        kind = contract._enum(row.get("kind"), {"source_member", "transform_output"})
        file_id = _id(row.get("file_id"))
        _require(file_id in files, "CLOSURE_INCOMPLETE")
        if kind == "source_member":
            _object(row, {"kind", "source_id", "record_id", "member", "file_id"})
            source_id, record_id, record = reference(
                {key: row[key] for key in ("source_id", "record_id", "member")}
            )
            key = (source_id, record_id)
            _require(key not in seen, "CLOSURE_INCOMPLETE")
            seen.add(key)
            direct.setdefault(file_id, []).append((source_id, record_id, record))
            if record.typeflag == "2":
                unsupported.add("symlink")
        else:
            _object(row, {"kind", "transform_id", "file_id"})
            transform_id = _id(row["transform_id"])
            _require(
                transform_id in transforms and transform_id not in outputs, "CLOSURE_INCOMPLETE"
            )
            _require(transforms[transform_id]["output_file_id"] == file_id, "CLOSURE_INCOMPLETE")
            outputs.add(transform_id)
    _require(outputs == set(transforms), "CLOSURE_INCOMPLETE")
    output_files = {transform["output_file_id"] for transform in transforms.values()}
    _require(
        len(output_files) == len(transforms) and not output_files.intersection(direct),
        "CLOSURE_INCOMPLETE",
    )
    _require(
        seen == set(records) and set(direct) | output_files == set(files), "CLOSURE_INCOMPLETE"
    )
    return direct, tuple(sorted(unsupported))


def _map(
    direct: dict[str, list[tuple[str, str, TarMemberV3]]],
    files: dict[str, dict[str, Any]],
    sources: dict[str, UbuntuPackageInventoryV3],
    budget: _MappingBudget,
) -> tuple[MappedFileV5a, ...]:
    result: list[MappedFileV5a] = []
    for file_id, origins in direct.items():
        budget.compare()
        file = files[file_id]
        if len(origins) > 1:
            _require(
                file["type"] == "directory"
                and all(record.typeflag == "5" for _, _, record in origins),
                "PATH_COLLISION",
            )
        budget.reserve(2048 + len(origins) * 1024)
        staged: list[MappedOriginV5a] = []
        for source_id, record_id, record in origins:
            budget.compare()
            name = record.canonical_name
            path = "/" if name == "." else "/" + name
            if path != file["path"]:
                raise ImageMappingRefusalV5a("ARTIFACT_MISMATCH", ("relocation",))
            kind = "directory" if record.typeflag == "5" else "regular"
            _require(record.typeflag != "2", "UNSUPPORTED_TRANSFORMATION")
            _require(kind == file["type"], "ARTIFACT_MISMATCH")
            if any(getattr(record, key) != file[key] for key in ("uid", "gid", "mode")):
                raise ImageMappingRefusalV5a("ARTIFACT_MISMATCH", ("metadata_change",))
            _require(record.size == file["size"], "ARTIFACT_MISMATCH")
            _require(record.sha256 == file.get("sha256"), "ARTIFACT_MISMATCH")
            package = sources[source_id]
            staged.append(
                MappedOriginV5a(source_id, package.package, package.version, record_id, record.name)
            )
        result.append(
            MappedFileV5a(
                file_id,
                file["path"],
                file["type"],
                file.get("sha256"),
                file["size"],
                file["uid"],
                file["gid"],
                file["mode"],
                tuple(staged),
            )
        )
    return tuple(result)


def verify_image_mapping_v5a(
    *,
    inventory_raw: bytes,
    expected_image_raw: bytes,
    mapping_raw: bytes,
    expected_image_sha256: str,
    package_result: PackageInventoryResultV3 | None = None,
    apt_evidence: AptEvidenceV5a | None = None,
    limits: Limits = _DEFAULT_LIMITS,
) -> ImageMappingResultV5a:
    """Compare complete declarations and explicitly associated measurement claims.

    No filesystem, target imports, network, tools or package code are accessed.
    The separate expected byte pin is a caller expectation, never acceptance.
    Every refusal, interruption or exhausted budget discards ALL mapped files
    and successful evidence bindings. Accounting is cooperative, not an RSS or
    operating-system CPU sandbox. V3 package/version restrictions remain intact.
    """
    budget: _MappingBudget | None = None
    unsupported: tuple[str, ...] = ()
    try:
        _validate_limits(limits)
        budget = _MappingBudget(limits)
        _digest(expected_image_sha256)
        _preflight(inventory_raw, budget)
        packages = _parse(inventory_raw, limits, budget)
        _require(len(packages) <= MAX_PACKAGES, "LIMIT_EXCEEDED")
        # Path count includes both archive namespaces and declared image paths.
        archive_count = sum(
            len(package.data.records) + len(package.control.records) for package in packages
        )
        _require(archive_count <= limits.max_archive_members, "LIMIT_EXCEEDED")
        image = _decode(expected_image_raw, budget)
        mapping = _decode(mapping_raw, budget)
        inventory_digest = hashlib.sha256(inventory_raw).hexdigest()
        image_digest = hashlib.sha256(expected_image_raw).hexdigest()
        mapping_digest = hashlib.sha256(mapping_raw).hexdigest()
        _require(image_digest == expected_image_sha256, "ARTIFACT_MISMATCH")
        _object(
            mapping,
            {
                "format",
                "snapshot_sha256",
                "inventory_sha256",
                "expected_image_sha256",
                "sources",
                "directory_policy",
                "entries",
                "transforms",
            },
        )
        _require(mapping["format"] == MAPPING_FORMAT, "UNSUPPORTED_SOURCE")
        _require(mapping["directory_policy"] == "identical-explicit-v1", "UNSUPPORTED_SOURCE")
        _require(
            _digest(mapping["inventory_sha256"]) == inventory_digest
            and _digest(mapping["expected_image_sha256"]) == image_digest,
            "ARTIFACT_MISMATCH",
        )
        files = _image(image, budget)
        _require(
            archive_count + len(files) <= limits.max_archive_members + limits.max_files,
            "LIMIT_EXCEEDED",
        )
        snapshot = _digest(mapping["snapshot_sha256"])
        _require(snapshot == image["snapshot_sha256"], "SNAPSHOT_MISMATCH")
        sources = _sources(mapping["sources"], packages, budget)
        expected_sources = _sources(image["sources"], packages, budget)
        _require(sources == expected_sources, "PROVENANCE_INCOMPLETE")
        direct, unsupported = _references(mapping, sources, files, budget)
        _require(not unsupported, "UNSUPPORTED_TRANSFORMATION")
        mapped = _map(direct, files, sources, budget)
        _require(package_result is not None, "PROVENANCE_INCOMPLETE")
        _require(
            type(package_result) is PackageInventoryResultV3
            and package_result.consistency == "MATCH"
            and package_result.reason_codes == (),
            "PROVENANCE_INCOMPLETE",
        )
        assert package_result is not None
        _measurements(package_result.packages, packages, budget)
        package_digest = _claim_digest(package_result, budget)
        apt_digest = (
            None
            if apt_evidence is None
            else _apt(
                apt_evidence,
                inventory_digest,
                snapshot,
                packages,
                budget,
            )
        )
        budget.check()
        return ImageMappingResultV5a(
            "MATCH",
            (),
            mapped,
            inventory_digest,
            image_digest,
            mapping_digest,
            snapshot,
            package_digest,
            apt_digest,
            "NOT_PROVIDED" if apt_evidence is None else "CALLER_CLAIM_BOUND",
            unmet_requirements=_UNMET + (("V4_NOT_PROVIDED",) if apt_evidence is None else ()),
        )
    except (ImageMappingRefusalV5a, contract.ContractRefusal) as exc:
        if isinstance(exc, ImageMappingRefusalV5a):
            unsupported = tuple(sorted(set(unsupported) | set(exc.unsupported)))
        consistency: Literal["MISMATCH", "INCOMPLETE"] = (
            "MISMATCH"
            if exc.code in {"ARTIFACT_MISMATCH", "SNAPSHOT_MISMATCH", "PATH_COLLISION"}
            else "INCOMPLETE"
        )
        reason = exc.code
    except (KeyboardInterrupt, InterruptedError):
        consistency, reason = "INCOMPLETE", "OPERATION_INTERRUPTED"
    except MemoryError:
        consistency, reason = "INCOMPLETE", "LIMIT_EXCEEDED"
    except (ValueError, UnicodeError, RecursionError, OverflowError, TypeError, AttributeError):
        consistency, reason = "INCOMPLETE", "FORMAT_REJECTED"
    return ImageMappingResultV5a(
        consistency,
        (reason,),
        unsupported_transformations=unsupported,
        unmet_requirements=_UNMET + ("STATIC_MAPPING_INCOMPLETE",),
    )
