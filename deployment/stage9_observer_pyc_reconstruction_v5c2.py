"""Inert future reconstruction declarations; never execution evidence.

An independently supplied policy is required for structural comparison. Its
acceptance and the declaration's origin remain unverified. Even a complete
matching declaration cannot establish that an interpreter ran or produced PYC.
No interpreter, source, bytecode, subprocess or deserializer is executed here.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, fields
from typing import Literal, cast

from deployment import stage9_observer_contract as contract
from deployment.stage9_observer_offline import Limits, _Budget, _validate_limits
from deployment.stage9_observer_pyc_v5c1 import SourceBindingV5c1

RECONSTRUCTION_PROFILE = "stage9-observer-pyc-reconstruction-evidence-v5c2"
POLICY_PROFILE = "stage9-observer-pyc-reconstruction-policy-v5c2"
_DEFAULT_LIMITS = Limits()
_UNMET = (
    "RECONSTRUCTION_EXECUTION_EVIDENCE_MISSING",
    "RECONSTRUCTION_EVIDENCE_ORIGIN_UNVERIFIED",
    "RECONSTRUCTION_POLICY_ACCEPTANCE_UNVERIFIED",
    "INTERPRETER_AND_LIBRARY_BYTES_UNVERIFIED",
    "ISOLATION_ENFORCEMENT_UNVERIFIED",
    "PYC_RECONSTRUCTION_UNVERIFIED",
)


@dataclass(frozen=True, slots=True)
class ReconstructionPackageV5c2:
    package: str
    version: str
    architecture: str
    archive_sha256: str
    snapshot_sha256: str


@dataclass(frozen=True, slots=True)
class ReconstructionLibraryV5c2:
    path: str
    sha256: str
    build_id: str
    package: ReconstructionPackageV5c2


@dataclass(frozen=True, slots=True)
class PycReconstructionPolicyV5c2:
    """Caller-pinned requirements, not an accepted execution authority.

    All arguments, environment entries, and isolation requirements are literal,
    ordered data. Their presence never attests enforcement of those settings.
    """

    interpreter_sha256: str
    interpreter_build_id: str
    interpreter_package: ReconstructionPackageV5c2
    required_libraries: tuple[ReconstructionLibraryV5c2, ...]
    source: SourceBindingV5c1
    parser_version: str
    encoding_configuration: str
    co_filename: str
    optimization: int
    invalidation: str
    arguments: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    isolation_requirements: tuple[str, ...]
    tool_image_id: str
    tool_image_sha256: str
    expected_image_sha256: str
    expected_pyc_sha256: str
    profile: str = POLICY_PROFILE


@dataclass(frozen=True, slots=True)
class PycReconstructionEvidenceV5c2:
    """A future run declaration; V5c2 cannot accept it as proof of a run."""

    policy: PycReconstructionPolicyV5c2
    output_pyc_sha256: str
    run_id: str
    evidence_id: str
    evidence_origin: str
    interrupted: bool
    interruption_reason: str | None
    verification_status: str = "RECONSTRUCTION_UNVERIFIED"
    profile: str = RECONSTRUCTION_PROFILE


@dataclass(frozen=True, slots=True)
class PycReconstructionContractResultV5c2:
    """STRUCTURE_VALID describes data only; it confers no authorization."""

    consistency: Literal["STRUCTURE_VALID", "NOT_DECLARED", "MISMATCH", "INCOMPLETE"]
    reason_codes: tuple[str, ...]
    declaration: PycReconstructionEvidenceV5c2 | None = None
    unmet_requirements: tuple[str, ...] = _UNMET
    profile: Literal["stage9-observer-pyc-reconstruction-result-v5c2"] = field(
        default="stage9-observer-pyc-reconstruction-result-v5c2", init=False
    )
    reconstruction: Literal["RECONSTRUCTION_UNVERIFIED"] = field(
        default="RECONSTRUCTION_UNVERIFIED", init=False
    )
    pyc_reconstruction: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    execution: Literal["MISSING"] = field(default="MISSING", init=False)
    evidence_origin: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    external_authority_acceptance: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    authenticated_time: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    durable_anti_rollback: Literal["BLOCKED"] = field(default="BLOCKED", init=False)
    runtime_qualification: Literal["BLOCKED"] = field(default="BLOCKED", init=False)
    stage_9: Literal["IN_PROGRESS"] = field(default="IN_PROGRESS", init=False)
    qualification: Literal["BLOCKED"] = field(default="BLOCKED", init=False)
    stage_3b: Literal["NOT AUTHORIZED"] = field(default="NOT AUTHORIZED", init=False)
    production: Literal["NO-GO"] = field(default="NO-GO", init=False)
    refused: Literal[True] = field(default=True, init=False)


def _require(condition: bool, code: str = "FORMAT_REJECTED") -> None:
    if not condition:
        raise contract.ContractRefusal(code)


def _text(value: object, maximum: int = 4096) -> str:
    text = contract._string(value, maximum)
    _require(bool(text))
    return cast(str, text)


def _build_id(value: str) -> None:
    _text(value, 128)
    _require(re.fullmatch(r"(?:[0-9a-f]{2}){4,64}", value) is not None)


def _package(value: ReconstructionPackageV5c2, budget: _Budget) -> None:
    budget.reserve(1024)
    _require(type(value) is ReconstructionPackageV5c2)
    contract._identifier(value.package)
    _text(value.version, 128)
    _text(value.architecture, 32)
    _require(value.architecture == "amd64")
    contract._digest(value.archive_sha256)
    contract._digest(value.snapshot_sha256)


def _tuple(value: tuple[object, ...], maximum: int, minimum: int = 0) -> None:
    _require(type(value) is tuple)
    _require(minimum <= len(value) <= maximum, "LIMIT_EXCEEDED")


def _source(value: SourceBindingV5c1, limits: Limits) -> None:
    _require(type(value) is SourceBindingV5c1)
    contract._identifier(value.source_id)
    contract._identifier(value.package)
    _text(value.version, 128)
    _text(value.architecture, 32)
    _require(value.architecture == "amd64")
    for digest in (
        value.snapshot_sha256,
        value.archive_sha256,
        value.record_id,
        value.sha256,
    ):
        contract._digest(digest)
    contract._integer(value.archive_size, minimum=1, maximum=limits.max_file_bytes)
    contract._integer(value.size, maximum=limits.max_file_bytes)
    _text(value.member)
    _require(value.member.endswith(".py"))


def _policy(value: PycReconstructionPolicyV5c2, budget: _Budget) -> None:
    budget.reserve(32768)
    _require(type(value) is PycReconstructionPolicyV5c2)
    _text(value.profile, 128)
    _require(value.profile == POLICY_PROFILE)
    contract._digest(value.interpreter_sha256)
    _build_id(value.interpreter_build_id)
    _require(len(value.interpreter_build_id) == 40)
    _package(value.interpreter_package, budget)
    _require(value.interpreter_package.package == "python3.12-minimal")
    _require(value.interpreter_package.version == "3.12.3-1ubuntu0.17")
    _tuple(value.required_libraries, min(256, budget.limits.max_files), minimum=1)
    paths: set[str] = set()
    for library in value.required_libraries:
        budget.check()
        _require(type(library) is ReconstructionLibraryV5c2)
        contract._path(library.path)
        _require(library.path not in paths)
        paths.add(library.path)
        contract._digest(library.sha256)
        _build_id(library.build_id)
        _package(library.package, budget)
        _require(
            library.package.snapshot_sha256 == value.interpreter_package.snapshot_sha256,
            "ARTIFACT_MISMATCH",
        )
    _source(value.source, budget.limits)
    _require(
        value.source.snapshot_sha256 == value.interpreter_package.snapshot_sha256,
        "ARTIFACT_MISMATCH",
    )
    _require(_text(value.parser_version, 128) == "CPython-3.12.3")
    _require(_text(value.encoding_configuration, 128) == "pep263-utf8-default-v1")
    contract._path(value.co_filename)
    contract._integer(value.optimization, maximum=2)
    _require(_text(value.invalidation, 128) == "checked_hash")
    _tuple(value.arguments, 128, minimum=1)
    for argument in value.arguments:
        budget.reserve(len(_text(argument).encode("utf-8")))
    _tuple(value.environment, 128)
    names: set[str] = set()
    for entry in value.environment:
        _tuple(entry, 2, minimum=2)
        name, setting = entry
        _text(name, 128)
        _require(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is not None)
        _require(name not in names)
        names.add(name)
        contract._string(setting, 4096)
        budget.reserve(len(name.encode("utf-8")) + len(setting.encode("utf-8")))
    _tuple(value.isolation_requirements, 128, minimum=1)
    for isolation in value.isolation_requirements:
        budget.reserve(len(_text(isolation).encode("utf-8")))
    _require(len(set(value.isolation_requirements)) == len(value.isolation_requirements))
    contract._identifier(value.tool_image_id)
    for digest in (
        value.tool_image_sha256,
        value.expected_image_sha256,
        value.expected_pyc_sha256,
    ):
        contract._digest(digest)
    budget.check()


def validate_reconstruction_evidence_v5c2(
    evidence: PycReconstructionEvidenceV5c2 | None,
    *,
    expected_policy: PycReconstructionPolicyV5c2 | None,
    source_bytes: bytes | None,
    limits: Limits = _DEFAULT_LIMITS,
) -> PycReconstructionContractResultV5c2:
    """Check bounded structure and caller pin agreement without proving a run.

    ``source_bytes`` are exact bytes supplied by the caller, whose upstream
    package verification must be performed by the unchanged V5b adapter. This
    structure-only checker never upgrades those bytes to verified provenance.
    ``expected_policy`` must be supplied separately; acceptance is unverified
    even if a caller passes the declaration's own policy as that argument.
    """
    try:
        _validate_limits(limits)
        for descriptor in fields(Limits):
            _require(
                getattr(limits, descriptor.name) <= getattr(_DEFAULT_LIMITS, descriptor.name),
                "LIMIT_EXCEEDED",
            )
        budget = _Budget(limits)
        if evidence is None:
            return PycReconstructionContractResultV5c2(
                "NOT_DECLARED", ("RECONSTRUCTION_NOT_DECLARED",)
            )
        _require(type(evidence) is PycReconstructionEvidenceV5c2)
        _text(evidence.profile, 128)
        _require(evidence.profile == RECONSTRUCTION_PROFILE)
        _require(_text(evidence.verification_status, 128) == "RECONSTRUCTION_UNVERIFIED")
        if expected_policy is None:
            raise contract.ContractRefusal("PROVENANCE_INCOMPLETE")
        _policy(expected_policy, budget)
        _policy(evidence.policy, budget)
        _require(evidence.policy == expected_policy, "ARTIFACT_MISMATCH")
        contract._digest(evidence.output_pyc_sha256)
        _require(
            evidence.output_pyc_sha256 == expected_policy.expected_pyc_sha256,
            "ARTIFACT_MISMATCH",
        )
        contract._identifier(evidence.run_id)
        contract._identifier(evidence.evidence_id)
        _text(evidence.evidence_origin, 2048)
        _require(type(evidence.interrupted) is bool)
        if evidence.interrupted:
            _text(evidence.interruption_reason)
            return PycReconstructionContractResultV5c2("INCOMPLETE", ("OPERATION_INTERRUPTED",))
        _require(evidence.interruption_reason is None)
        if type(source_bytes) is not bytes:
            raise contract.ContractRefusal("PROVENANCE_INCOMPLETE")
        _require(len(source_bytes) <= limits.max_file_bytes, "LIMIT_EXCEEDED")
        _require(len(source_bytes) == expected_policy.source.size, "ARTIFACT_MISMATCH")
        budget.reserve(len(source_bytes))
        digest = hashlib.sha256()
        for start in range(0, len(source_bytes), limits.chunk_bytes):
            budget.read(min(limits.chunk_bytes, len(source_bytes) - start))
            digest.update(memoryview(source_bytes)[start : start + limits.chunk_bytes])
        _require(digest.hexdigest() == expected_policy.source.sha256, "ARTIFACT_MISMATCH")
        budget.check()
        return PycReconstructionContractResultV5c2("STRUCTURE_VALID", (), evidence)
    except contract.ContractRefusal as exc:
        return PycReconstructionContractResultV5c2(
            "MISMATCH" if exc.code == "ARTIFACT_MISMATCH" else "INCOMPLETE", (exc.code,)
        )
    except (KeyboardInterrupt, InterruptedError):
        reason = "OPERATION_INTERRUPTED"
    except MemoryError:
        reason = "LIMIT_EXCEEDED"
    except (ValueError, TypeError, AttributeError, UnicodeError, OverflowError, RecursionError):
        reason = "FORMAT_REJECTED"
    return PycReconstructionContractResultV5c2("INCOMPLETE", (reason,))
