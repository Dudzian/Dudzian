"""Controlled reconstruction collection, separate from acceptance of evidence.

The only execution boundary is an explicitly supplied, live outside-VM
collector. This module never launches a host process or interprets a PYC body.
A collector is trusted code, not a JSON report or independent VM attestation.
No concrete controller is qualified by this module; absent one, execution is
blocked. Synthetic collectors exercise mechanics and cannot verify a target run.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import time
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from typing import Any, Literal, Protocol, cast

from deployment.canonical_json import canonical_json_bytes
from deployment.stage9_observer_contract import ContractRefusal, _path
from deployment.stage9_observer_member_source_v5b import (
    FrozenPackageV5b,
    MemberSelectionV5b,
    capture_member_sources_v5b,
)
from deployment.stage9_observer_offline import Limits, _Budget, _validate_limits
from deployment.stage9_observer_package_v3 import _parse
from deployment.stage9_observer_pyc_reconstruction_v5c2 import (
    PycReconstructionPolicyV5c2,
    _policy,
)
from deployment.stage9_observer_pyc_source_hash_v5c2 import (
    TARGET_BUILD_ID,
    TARGET_INTERPRETER_SHA256,
)
from deployment.stage9_observer_runtime_identity_v5c3 import (
    CapturedRuntimeFileV5c3,
    RuntimeInventoryV5c3,
    RuntimeMeasurementV5c3,
    validate_runtime_identity_v5c3,
    validate_runtime_inventory_structure_v5c3,
)
from deployment.stage9_observer_source_hash_algorithm_v5c2 import (
    ALGORITHM_ID,
    MAGIC,
    independent_source_hash,
)

ARGUMENTS = (
    "/usr/bin/python3.12",
    "-I",
    "-S",
    "-B",
    "/stage9/tool/compile_worker.py",
)
ENVIRONMENT = (("LANG", "C.UTF-8"), ("LC_ALL", "C.UTF-8"), ("TZ", "UTC"))
INPUT_PATH = "/stage9/input/source.py"
OUTPUT_PATH = "/stage9/output/reconstructed.pyc"
RESOURCE_CONTROLS = ("cpu", "memory", "wall_time", "process_count")
ISOLATION_REQUIREMENTS = (
    "no-network",
    "no-shared-user-data",
    "no-secrets",
    "no-host-sockets",
    "no-production-access",
    "disposable-disk",
    "read-only-input",
    "separate-output",
    "resource-limits",
    "whole-vm-stop",
)
_DEFAULT_LIMITS = Limits()
_MAX_STREAM_BYTES = 65536
_MAX_SOURCE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ExecutionLimitsV5c3:
    cpu_count: int = 1
    memory_bytes: int = 512 * 1024 * 1024
    wall_time_s: int = 30
    process_count: int = 64


@dataclass(frozen=True, slots=True)
class IsolationObservationV5c3:
    """Outside-controller observations, never guest self-attestation."""

    vm_id: str
    disk_id: str
    network_interfaces: tuple[str, ...]
    shared_paths: tuple[str, ...]
    secret_names: tuple[str, ...]
    host_sockets: tuple[str, ...]
    production_access: bool
    read_only_sources: bool
    separate_output: bool
    disposable_disk: bool
    stop_available: bool
    limits: ExecutionLimitsV5c3
    enforced_limits: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ControllerObservationV5c3:
    isolation: IsolationObservationV5c3
    tool_image_id: str
    tool_image_sha256: str
    runtime_files: tuple[CapturedRuntimeFileV5c3, ...]
    complete: bool
    runtime_tree_complete: bool
    runtime_tree_immutable: bool


@dataclass(frozen=True, slots=True)
class ProcessObservationV5c3:
    """Receipt collected outside the VM, with actual output bytes.

    stdout/stderr remain opaque guest-reported data. They cannot override
    measured input/output or process termination observed by the controller.
    """

    run_id: str
    nonce: str
    arguments: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    input_sha256: str
    output: bytes | None
    exit_code: int | None
    stdout: bytes
    stderr: bytes
    elapsed_s: float
    interrupted: bool
    interruption_reason: str | None
    complete: bool
    runtime_accesses: tuple[tuple[str, str], ...]
    runtime_accesses_complete: bool
    mapped_elf_paths: tuple[str, ...]


class PreparedVMCollectorV5c3(Protocol):
    """Trusted external controller interface; implementing it is not approval.

    inspect must observe effective hypervisor state and freeze the complete
    runtime tree. execute must stage immutable source/request in read-only
    input, start only ARGUMENTS with ENVIRONMENT, enforce the supplied timeout,
    collect process status and read actual output separately from stdout.
    Each call starts with a fresh output tree. stop must terminate the whole
    disposable VM and confirm it stopped. It must never alter an existing
    Stage 9 VM. No controller supplied by serialized guest data is admissible.
    """

    collector_id: str
    origin: Literal["CONTROLLER_OBSERVED", "SYNTHETIC", "GUEST_REPORTED"]

    def inspect(self) -> ControllerObservationV5c3: ...

    def execute(
        self, request_raw: bytes, source: bytes, timeout_s: float
    ) -> ProcessObservationV5c3: ...

    def stop(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class ReconstructionRequestV5c3:
    policy: PycReconstructionPolicyV5c2
    expected_source_hash_hex: str
    target_path: str


@dataclass(frozen=True, slots=True)
class OutputMeasurementV5c3:
    sha256: str
    size: int
    magic_hex: str
    flags: int
    source_hash_hex: str
    opaque_body_sha256: str


@dataclass(frozen=True, slots=True)
class ExecutionRecordV5c3:
    repository_commit: str
    vm_id: str
    disk_id: str
    tool_image_id: str
    tool_image_sha256: str
    policy_sha256: str
    runtime_inventory_sha256: str
    interpreter_sha256: str
    interpreter_build_id: str
    runtime_measurements: tuple[RuntimeMeasurementV5c3, ...]
    runtime_accesses: tuple[tuple[str, str], ...]
    runtime_accesses_complete: bool
    mapped_elf_paths: tuple[str, ...]
    runtime_tree_complete: bool
    runtime_tree_immutable: bool
    run_id: str
    nonce: str
    input_path: str
    output_path: str
    target_path: str
    arguments: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    isolation: IsolationObservationV5c3
    started_local: str
    ended_local: str
    exit_code: int | None
    stdout: bytes
    stderr: bytes
    elapsed_s: float
    controller_elapsed_ns: int
    source_sha256: str
    source_size: int
    output: OutputMeasurementV5c3 | None
    comparison: str
    interrupted: bool
    interruption_reason: str | None
    collector_id: str
    collector_origin: str
    record_complete: bool
    profile: Literal["stage9-observer-pyc-execution-record-v5c3"] = field(
        default="stage9-observer-pyc-execution-record-v5c3", init=False
    )
    clock: Literal["LOCAL_UNAUTHENTICATED"] = field(default="LOCAL_UNAUTHENTICATED", init=False)
    guest_report_origin: Literal["GUEST_REPORTED"] = field(default="GUEST_REPORTED", init=False)
    integrity: Literal["LOCAL_UNAUTHENTICATED"] = field(default="LOCAL_UNAUTHENTICATED", init=False)


@dataclass(frozen=True, slots=True)
class ReconstructionResultV5c3:
    consistency: Literal[
        "EXECUTION_BLOCKED",
        "EXECUTION_FAILED",
        "RECONSTRUCTION_MISMATCH",
        "SYNTHETIC_MECHANISM_MATCH",
        "LOCAL_EXECUTION_OBSERVATION",
    ]
    reason_codes: tuple[str, ...]
    files: tuple[ExecutionRecordV5c3, ...] = ()
    diagnostic_records: tuple[ExecutionRecordV5c3, ...] = ()
    profile: Literal["stage9-observer-pyc-execution-result-v5c3"] = field(
        default="stage9-observer-pyc-execution-result-v5c3", init=False
    )
    independently_bound_evidence: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    accepted_reconstruction_evidence: Literal["UNVERIFIED"] = field(
        default="UNVERIFIED", init=False
    )
    reconstruction: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    external_authority_acceptance: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    qualification: Literal["BLOCKED"] = field(default="BLOCKED", init=False)
    authenticated_time: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    durable_anti_rollback: Literal["BLOCKED"] = field(default="BLOCKED", init=False)
    runtime_qualification: Literal["BLOCKED"] = field(default="BLOCKED", init=False)
    runtime_closure: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    stage_9: Literal["IN_PROGRESS"] = field(default="IN_PROGRESS", init=False)
    observer_3a: Literal["BLOCKED"] = field(default="BLOCKED", init=False)
    stage_3b: Literal["NOT AUTHORIZED"] = field(default="NOT AUTHORIZED", init=False)
    production: Literal["NO-GO"] = field(default="NO-GO", init=False)
    refused: Literal[True] = field(default=True, init=False)

    @property
    def exact_target_test(self) -> str:
        if self.consistency == "LOCAL_EXECUTION_OBSERVATION":
            return "LOCAL_OBSERVATION_UNACCEPTED"
        return "NOT_TESTED"


class ExecutionRefusalV5c3(ContractRefusal):
    """V5c3 diagnostics without expanding the unchanged V1 vocabulary."""

    def __init__(self, reason: str) -> None:
        super().__init__("REPORT_REJECTED")
        self.execution_reason = reason


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ExecutionRefusalV5c3(code)


def _digest(value: object) -> None:
    _require(
        type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "FORMAT_REJECTED"
    )


def runtime_inventory_sha256_v5c3(inventory: RuntimeInventoryV5c3) -> str:
    """A local content pin, not acceptance or origin authentication."""
    validate_runtime_inventory_structure_v5c3(inventory)
    return _model_sha256(asdict(inventory))


def _model_sha256(value: Any) -> str:
    # Canonical JSON rejects Python tuples. Convert a bounded caller model,
    # never an execution report, to ordinary JSON containers before encoding.
    plain = json.loads(json.dumps(value, allow_nan=False))
    return hashlib.sha256(canonical_json_bytes(plain)).hexdigest()


def reconstruction_requests_sha256_v5c3(requests: tuple[ReconstructionRequestV5c3, ...]) -> str:
    """Pin the entire plan before execution, including independent outputs."""
    _require(type(requests) is tuple and 0 < len(requests) <= 128, "LIMIT_EXCEEDED")
    return _model_sha256([asdict(item) for item in requests])


def serialize_reconstruction_result_v5c3(result: ReconstructionResultV5c3) -> bytes:
    """Export observations with an unauthenticated local integrity checksum.

    Streams use hexadecimal, never lossy text decoding. Durations are integral
    nanoseconds. The export cannot be imported as a live collector and its
    checksum neither authenticates origin nor supplies independent evidence.
    """
    _require(type(result) is ReconstructionResultV5c3, "FORMAT_REJECTED")
    _require(len(result.files) + len(result.diagnostic_records) <= 128, "LIMIT_EXCEEDED")
    payload = asdict(result)
    payload["exact_target_test"] = result.exact_target_test
    for collection in ("files", "diagnostic_records"):
        rows: list[dict[str, Any]] = []
        for record in getattr(result, collection):
            _require(type(record) is ExecutionRecordV5c3, "EXECUTION_RECORD_INCOMPLETE")
            row = asdict(record)
            for stream in ("stdout", "stderr"):
                content = row.pop(stream)
                _require(
                    type(content) is bytes and len(content) <= _MAX_STREAM_BYTES,
                    "PROCESS_STREAM_LIMIT_EXCEEDED",
                )
                row[stream + "_hex"] = content.hex()
            duration = row.pop("elapsed_s")
            _require(
                type(duration) in (int, float) and math.isfinite(duration) and 0 <= duration <= 60,
                "EXECUTION_RECORD_INCOMPLETE",
            )
            row["reported_elapsed_ns"] = int(duration * 1_000_000_000)
            rows.append(row)
        payload[collection] = rows
    plain = json.loads(json.dumps(payload, allow_nan=False))
    content = canonical_json_bytes(plain)
    return cast(
        bytes,
        canonical_json_bytes(
            dict(
                format="stage9-observer-pyc-execution-export-v5c3",
                payload=plain,
                payload_sha256=hashlib.sha256(content).hexdigest(),
                integrity_origin="LOCAL_UNAUTHENTICATED",
            )
        ),
    )


def _resource_limits(value: ExecutionLimitsV5c3) -> None:
    _require(type(value) is ExecutionLimitsV5c3, "LIMIT_EXCEEDED")
    for descriptor in fields(value):
        _require(type(getattr(value, descriptor.name)) is int, "LIMIT_EXCEEDED")
    _require(1 <= value.cpu_count <= 2, "LIMIT_EXCEEDED")
    _require(64 * 1024 * 1024 <= value.memory_bytes <= 1024 * 1024 * 1024, "LIMIT_EXCEEDED")
    _require(1 <= value.wall_time_s <= 60, "LIMIT_EXCEEDED")
    _require(1 <= value.process_count <= 128, "LIMIT_EXCEEDED")


def _isolation(observed: ControllerObservationV5c3, limits: ExecutionLimitsV5c3) -> None:
    _require(type(observed) is ControllerObservationV5c3, "ISOLATION_RECORD_INCOMPLETE")
    _require(observed.complete is True, "ISOLATION_RECORD_INCOMPLETE")
    _require(
        observed.runtime_tree_complete is True and observed.runtime_tree_immutable is True,
        "RUNTIME_TREE_NOT_FROZEN",
    )
    isolation = observed.isolation
    _require(type(isolation) is IsolationObservationV5c3, "ISOLATION_RECORD_INCOMPLETE")
    for identity in (isolation.vm_id, isolation.disk_id, observed.tool_image_id):
        _require(type(identity) is str and 0 < len(identity) <= 256, "ISOLATION_RECORD_INCOMPLETE")
    _digest(observed.tool_image_sha256)
    for paths, code in (
        (isolation.network_interfaces, "NETWORK_PRESENT"),
        (isolation.shared_paths, "UNEXPECTED_SHARED_PATH"),
        (isolation.secret_names, "SECRETS_PRESENT"),
        (isolation.host_sockets, "HOST_SOCKET_PRESENT"),
    ):
        _require(type(paths) is tuple and paths == (), code)
    _require(isolation.production_access is False, "PRODUCTION_ACCESS_PRESENT")
    for state in (
        isolation.read_only_sources,
        isolation.separate_output,
        isolation.disposable_disk,
        isolation.stop_available,
    ):
        _require(state is True, "ISOLATION_NOT_ENFORCED")
    _require(isolation.limits == limits, "RESOURCE_LIMIT_MISMATCH")
    _require(isolation.enforced_limits == RESOURCE_CONTROLS, "RESOURCE_LIMIT_NOT_ENFORCED")


def _source_bytes(
    inventory_raw: bytes,
    archives: tuple[FrozenPackageV5b, ...],
    requests: tuple[ReconstructionRequestV5c3, ...],
    budget: _Budget,
) -> tuple[bytes, ...]:
    packages = _parse(inventory_raw, budget.limits, budget)
    selections = tuple(
        dict.fromkeys(
            MemberSelectionV5b(
                item.policy.source.source_id,
                item.policy.source.package,
                item.policy.source.member,
                item.policy.source.record_id,
            )
            for item in requests
        )
    )
    _, captured = capture_member_sources_v5b(
        packages=packages, archives=archives, selections=selections, budget=budget
    )
    by_key = {(item.package, item.member): item for item in captured}
    by_package = {item.package: item for item in packages}
    sources: list[bytes] = []
    for request in requests:
        source = request.policy.source
        package = by_package[source.package]
        item = by_key[(source.package, source.member)]
        _require(package.version == source.version, "SOURCE_METADATA_MISMATCH")
        _require(package.architecture == source.architecture, "SOURCE_METADATA_MISMATCH")
        _require(
            item.archive_sha256 == source.archive_sha256
            and item.archive_size == source.archive_size,
            "SOURCE_ARCHIVE_MISMATCH",
        )
        _require(
            len(item.content) == source.size
            and hashlib.sha256(item.content).hexdigest() == source.sha256,
            "SOURCE_BYTES_MISMATCH",
        )
        _require(len(item.content) <= _MAX_SOURCE_BYTES, "LIMIT_EXCEEDED")
        actual_hash = independent_source_hash(item.content, ALGORITHM_ID, check=budget.check)
        _require(actual_hash.hex() == request.expected_source_hash_hex, "SOURCE_HASH_MISMATCH")
        sources.append(item.content)
    return tuple(sources)


def _output(raw: bytes | None, maximum: int) -> OutputMeasurementV5c3 | None:
    if raw is None:
        return None
    _require(type(raw) is bytes and len(raw) <= maximum, "OUTPUT_LIMIT_EXCEEDED")
    return OutputMeasurementV5c3(
        hashlib.sha256(raw).hexdigest(),
        len(raw),
        raw[:4].hex(),
        int.from_bytes(raw[4:8], "little"),
        raw[8:16].hex(),
        hashlib.sha256(raw[16:]).hexdigest(),
    )


def _process_structure(value: ProcessObservationV5c3) -> None:
    """Bound every retained receipt field before constructing a record."""
    _require(type(value) is ProcessObservationV5c3, "EXECUTION_RECORD_INCOMPLETE")
    for text, maximum in ((value.run_id, 128), (value.nonce, 64), (value.input_sha256, 64)):
        _require(type(text) is str and len(text) <= maximum, "EXECUTION_RECORD_INCOMPLETE")
    _require(
        type(value.arguments) is tuple and len(value.arguments) <= 128,
        "EXECUTION_RECORD_INCOMPLETE",
    )
    for argument in value.arguments:
        _require(type(argument) is str and len(argument) <= 4096, "EXECUTION_RECORD_INCOMPLETE")
    _require(
        type(value.environment) is tuple and len(value.environment) <= 128,
        "EXECUTION_RECORD_INCOMPLETE",
    )
    for entry in value.environment:
        _require(
            type(entry) is tuple
            and len(entry) == 2
            and all(type(part) is str and len(part) <= 4096 for part in entry),
            "EXECUTION_RECORD_INCOMPLETE",
        )
    for stream in (value.stdout, value.stderr):
        _require(
            type(stream) is bytes and len(stream) <= _MAX_STREAM_BYTES,
            "PROCESS_STREAM_LIMIT_EXCEEDED",
        )
    _require(
        value.exit_code is None
        or (type(value.exit_code) is int and -255 <= value.exit_code <= 255),
        "EXECUTION_RECORD_INCOMPLETE",
    )
    _require(
        type(value.complete) is bool and type(value.interrupted) is bool,
        "EXECUTION_RECORD_INCOMPLETE",
    )
    _require(
        value.interruption_reason is None
        or (type(value.interruption_reason) is str and len(value.interruption_reason) <= 4096),
        "EXECUTION_RECORD_INCOMPLETE",
    )
    _require(
        type(value.elapsed_s) in (int, float)
        and math.isfinite(value.elapsed_s)
        and 0 <= value.elapsed_s <= 60,
        "EXECUTION_RECORD_INCOMPLETE",
    )
    _require(type(value.runtime_accesses_complete) is bool, "RUNTIME_TRACE_INCOMPLETE")
    _require(
        type(value.runtime_accesses) is tuple and len(value.runtime_accesses) <= 4096,
        "RUNTIME_TRACE_INCOMPLETE",
    )
    for entry in value.runtime_accesses:
        _require(
            type(entry) is tuple
            and len(entry) == 2
            and all(type(part) is str and len(part) <= 4096 for part in entry),
            "RUNTIME_TRACE_INCOMPLETE",
        )
    _require(
        type(value.mapped_elf_paths) is tuple and len(value.mapped_elf_paths) <= 4096,
        "RUNTIME_TRACE_INCOMPLETE",
    )
    for path in value.mapped_elf_paths:
        _require(type(path) is str and len(path) <= 4096, "RUNTIME_TRACE_INCOMPLETE")


def _process(
    value: ProcessObservationV5c3,
    run_id: str,
    nonce: str,
    source_sha256: str,
    limits: ExecutionLimitsV5c3,
    inventory: RuntimeInventoryV5c3,
) -> None:
    _require(value.complete is True, "EXECUTION_RECORD_INCOMPLETE")
    _require(value.run_id == run_id and value.nonce == nonce, "RUN_BINDING_MISMATCH")
    _require(value.arguments == ARGUMENTS, "PROCESS_ARGUMENT_MISMATCH")
    _require(value.environment == ENVIRONMENT, "PROCESS_ENVIRONMENT_MISMATCH")
    _require(value.input_sha256 == source_sha256, "SOURCE_BYTES_MISMATCH")
    _require(value.elapsed_s < limits.wall_time_s, "EXECUTION_TIMEOUT")
    _require(
        value.interrupted is False and value.interruption_reason is None, "OPERATION_INTERRUPTED"
    )
    _require(type(value.exit_code) is int, "PROCESS_FAILED")
    _require(value.runtime_accesses_complete is True, "RUNTIME_TRACE_INCOMPLETE")
    pinned = {item.path: item for item in inventory.files}
    accesses: dict[str, str] = {}
    for path, digest in value.runtime_accesses:
        _require(
            path not in accesses and path in pinned and pinned[path].sha256 == digest,
            "UNPINNED_RUNTIME_ACCESS",
        )
        accesses[path] = digest
    mapped = set(value.mapped_elf_paths)
    _require(len(mapped) == len(value.mapped_elf_paths), "RUNTIME_TRACE_INCOMPLETE")
    for path in mapped:
        _require(path in accesses and pinned[path].build_id is not None, "UNPINNED_RUNTIME_MAPPING")
    for pin in inventory.files:
        if pin.role in {"interpreter", "loader"}:
            _require(pin.path in mapped, "RUNTIME_MAPPING_MISSING")
        if pin.role == "tool" or pin.path == "/usr/lib/python3.12/py_compile.py":
            _require(pin.path in accesses, "COMPILER_INPUT_READ_MISSING")
    for consumer, _, resolved in inventory.needed_resolutions:
        if consumer in mapped:
            _require(resolved in mapped, "RUNTIME_MAPPING_MISSING")


def reconstruct_pyc_set_v5c3(
    *,
    inventory_raw: bytes,
    archives: tuple[FrozenPackageV5b, ...],
    requests: tuple[ReconstructionRequestV5c3, ...],
    expected_inventory_sha256: str,
    expected_requests_sha256: str,
    repository_commit: str,
    expected_runtime_inventory: RuntimeInventoryV5c3 | None = None,
    expected_runtime_inventory_sha256: str | None = None,
    collector: PreparedVMCollectorV5c3 | None = None,
    resource_limits: ExecutionLimitsV5c3 = ExecutionLimitsV5c3(),
    limits: Limits = _DEFAULT_LIMITS,
) -> ReconstructionResultV5c3:
    """Capture source, control a prepared VM and atomically compare actual bytes.

    The caller must supply independently maintained policy/inventory pins and
    trusted controller code. A matching guest report is never consumed as
    proof. Runtime inventory identity is checked before every launch and after
    every run; source and output are immutable bytes measured in this process.
    A failed member or failed VM stop invalidates the complete declared set.
    """
    records: list[ExecutionRecordV5c3] = []
    controller_used = False
    execution_started = False
    reason: str | None = None
    synthetic = True
    try:
        _validate_limits(limits)
        for descriptor in fields(Limits):
            _require(
                getattr(limits, descriptor.name) <= getattr(_DEFAULT_LIMITS, descriptor.name),
                "LIMIT_EXCEEDED",
            )
        budget = _Budget(limits)
        _resource_limits(resource_limits)
        _require(
            type(repository_commit) is str
            and re.fullmatch(r"[0-9a-f]{40}", repository_commit) is not None,
            "REPOSITORY_COMMIT_MISSING",
        )
        _digest(expected_inventory_sha256)
        _require(
            type(inventory_raw) is bytes and 0 < len(inventory_raw) <= 2 * 1024 * 1024,
            "LIMIT_EXCEEDED",
        )
        budget.reserve(len(inventory_raw) * 8)
        _require(
            hashlib.sha256(inventory_raw).hexdigest() == expected_inventory_sha256,
            "INVENTORY_PIN_MISMATCH",
        )
        _require(
            type(requests) is tuple and 0 < len(requests) <= min(128, limits.max_files),
            "LIMIT_EXCEEDED",
        )
        _digest(expected_requests_sha256)
        target_paths: set[str] = set()
        for request in requests:
            _require(type(request) is ReconstructionRequestV5c3, "FORMAT_REJECTED")
            _policy(request.policy, budget)
            _require(
                request.policy.interpreter_sha256 == TARGET_INTERPRETER_SHA256,
                "INTERPRETER_MISMATCH",
            )
            _require(
                request.policy.interpreter_build_id == TARGET_BUILD_ID,
                "INTERPRETER_BUILD_ID_MISMATCH",
            )
            _require(request.policy.arguments == ARGUMENTS, "PROCESS_ARGUMENT_MISMATCH")
            _require(request.policy.environment == ENVIRONMENT, "PROCESS_ENVIRONMENT_MISMATCH")
            _require(
                request.policy.isolation_requirements == ISOLATION_REQUIREMENTS,
                "ISOLATION_POLICY_MISMATCH",
            )
            _require(
                type(request.expected_source_hash_hex) is str
                and re.fullmatch(r"[0-9a-f]{16}", request.expected_source_hash_hex) is not None,
                "SOURCE_HASH_MISMATCH",
            )
            _path(request.target_path)
            _require(
                request.target_path.endswith(".pyc") and request.target_path not in target_paths,
                "TARGET_PATH_MISMATCH",
            )
            target_paths.add(request.target_path)
        _require(
            reconstruction_requests_sha256_v5c3(requests) == expected_requests_sha256,
            "RECONSTRUCTION_PLAN_PIN_MISMATCH",
        )
        sources = _source_bytes(inventory_raw, archives, requests, budget)
        _require(collector is not None, "QUALIFIED_ISOLATION_UNAVAILABLE")
        assert collector is not None
        _require(
            collector.origin in {"CONTROLLER_OBSERVED", "SYNTHETIC"},
            "GUEST_REPORTED_ISOLATION_REJECTED",
        )
        _require(
            type(collector.collector_id) is str and 0 < len(collector.collector_id) <= 256,
            "COLLECTOR_ID_MISSING",
        )
        synthetic = collector.origin == "SYNTHETIC"
        _require(expected_runtime_inventory is not None, "RUNTIME_INVENTORY_MISSING")
        assert expected_runtime_inventory is not None
        _digest(expected_runtime_inventory_sha256)
        runtime_digest = runtime_inventory_sha256_v5c3(expected_runtime_inventory)
        _require(
            runtime_digest == expected_runtime_inventory_sha256, "RUNTIME_INVENTORY_PIN_MISMATCH"
        )
        controller_used = True
        for request, source in zip(requests, sources, strict=True):
            budget.check()
            observed = collector.inspect()
            _isolation(observed, resource_limits)
            _require(
                observed.tool_image_id == request.policy.tool_image_id
                and observed.tool_image_sha256 == request.policy.tool_image_sha256,
                "TOOL_IMAGE_MISMATCH",
            )
            _require(
                type(observed.runtime_files) is tuple
                and len(observed.runtime_files) <= limits.max_files,
                "LIMIT_EXCEEDED",
            )
            for item in observed.runtime_files:
                _require(type(item) is CapturedRuntimeFileV5c3, "RUNTIME_RECORD_INCOMPLETE")
                if item.data is not None:
                    _require(
                        type(item.data) is bytes and len(item.data) <= limits.max_file_bytes,
                        "LIMIT_EXCEEDED",
                    )
                    budget.reserve(len(item.data))
                    budget.read(len(item.data))
            runtime = validate_runtime_identity_v5c3(
                expected_runtime_inventory,
                observed.runtime_files,
                request.policy,
                environment=ENVIRONMENT,
                max_file_bytes=limits.max_file_bytes,
            )
            run_id, nonce = "v5c3-" + secrets.token_hex(16), secrets.token_hex(32)
            request_raw = canonical_json_bytes(
                dict(
                    profile="stage9-observer-compile-request-v5c3",
                    source_sha256=request.policy.source.sha256,
                    source_size=len(source),
                    source_hash_hex=request.expected_source_hash_hex,
                    expected_pyc_sha256=request.policy.expected_pyc_sha256,
                    co_filename=request.policy.co_filename,
                    optimization=request.policy.optimization,
                    invalidation="checked_hash",
                    run_id=run_id,
                    nonce=nonce,
                )
            )
            started = datetime.now(timezone.utc).isoformat()
            monotonic_started = time.monotonic()
            execution_started = True
            process = collector.execute(request_raw, source, float(resource_limits.wall_time_s))
            observed_elapsed = time.monotonic() - monotonic_started
            ended = datetime.now(timezone.utc).isoformat()
            _require(type(process) is ProcessObservationV5c3, "EXECUTION_RECORD_INCOMPLETE")
            _require(
                math.isfinite(observed_elapsed)
                and 0 <= observed_elapsed < resource_limits.wall_time_s,
                "EXECUTION_TIMEOUT",
            )
            for stream in (process.stdout, process.stderr):
                _require(
                    type(stream) is bytes and len(stream) <= _MAX_STREAM_BYTES,
                    "PROCESS_STREAM_LIMIT_EXCEEDED",
                )
            _process_structure(process)
            measurement = _output(process.output, limits.max_file_bytes)
            record = ExecutionRecordV5c3(
                repository_commit,
                observed.isolation.vm_id,
                observed.isolation.disk_id,
                observed.tool_image_id,
                observed.tool_image_sha256,
                _model_sha256(asdict(request.policy)),
                runtime_digest,
                TARGET_INTERPRETER_SHA256,
                TARGET_BUILD_ID,
                runtime,
                process.runtime_accesses,
                process.runtime_accesses_complete,
                process.mapped_elf_paths,
                observed.runtime_tree_complete,
                observed.runtime_tree_immutable,
                run_id,
                nonce,
                INPUT_PATH,
                OUTPUT_PATH,
                request.target_path,
                process.arguments,
                process.environment,
                observed.isolation,
                started,
                ended,
                process.exit_code,
                process.stdout,
                process.stderr,
                process.elapsed_s,
                int(observed_elapsed * 1_000_000_000),
                request.policy.source.sha256,
                len(source),
                measurement,
                "SET_PENDING",
                process.interrupted,
                process.interruption_reason,
                collector.collector_id,
                collector.origin,
                process.complete,
            )
            records.append(record)
            _process(
                process,
                run_id,
                nonce,
                request.policy.source.sha256,
                resource_limits,
                expected_runtime_inventory,
            )
            _require(measurement is not None, "OUTPUT_MISSING")
            assert measurement is not None
            _require(
                measurement.size >= 16
                and measurement.magic_hex == MAGIC.hex()
                and measurement.flags == 3
                and measurement.source_hash_hex == request.expected_source_hash_hex,
                "PYC_HEADER_MISMATCH",
            )
            _require(
                measurement.sha256 == request.policy.expected_pyc_sha256, "RECONSTRUCTION_MISMATCH"
            )
            _require(process.exit_code == 0, "PROCESS_FAILED")
            _require(collector.inspect() == observed, "ISOLATION_OR_RUNTIME_CHANGED")
            budget.check()
    except ContractRefusal as exc:
        reason = getattr(exc, "runtime_reason", getattr(exc, "execution_reason", exc.code))
    except (KeyboardInterrupt, InterruptedError):
        reason = "OPERATION_INTERRUPTED"
    except TimeoutError:
        reason = "EXECUTION_TIMEOUT"
    except MemoryError:
        reason = "LIMIT_EXCEEDED"
    except Exception:
        reason = "EXECUTION_RECORD_INCOMPLETE"
    finally:
        if controller_used and collector is not None:
            try:
                if collector.stop() is not True:
                    reason = "VM_STOP_UNCONFIRMED"
            except (Exception, KeyboardInterrupt):
                reason = "VM_STOP_UNCONFIRMED"
    if reason is not None:
        diagnostics = tuple(replace(record, comparison="SET_INVALIDATED") for record in records)
        consistency: Literal["EXECUTION_BLOCKED", "EXECUTION_FAILED", "RECONSTRUCTION_MISMATCH"]
        if reason in {"RECONSTRUCTION_MISMATCH", "PYC_HEADER_MISMATCH"}:
            consistency = "RECONSTRUCTION_MISMATCH"
        elif not execution_started:
            consistency = "EXECUTION_BLOCKED"
        else:
            consistency = "EXECUTION_FAILED"
        return ReconstructionResultV5c3(consistency, (reason,), diagnostic_records=diagnostics)
    return ReconstructionResultV5c3(
        "SYNTHETIC_MECHANISM_MATCH" if synthetic else "LOCAL_EXECUTION_OBSERVATION",
        (),
        files=tuple(replace(record, comparison="OUTPUT_BYTES_MATCH") for record in records),
    )
