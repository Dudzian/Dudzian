"""Synthetic future declarations never establish execution or reconstruction."""

from __future__ import annotations

import builtins
import hashlib
from dataclasses import replace
from typing import Any

import pytest

from deployment import (
    stage9_observer_offline as offline,
    stage9_observer_pyc_reconstruction_v5c2 as reconstruction,
)
from deployment.stage9_observer_pyc_v5c1 import SourceBindingV5c1

pytestmark = [pytest.mark.security, pytest.mark.unit]
_SOURCE = b"message = 'inert fixture only'\n"


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def declaration() -> reconstruction.PycReconstructionEvidenceV5c2:
    snapshot = _sha(b"independent synthetic snapshot pin")
    package = reconstruction.ReconstructionPackageV5c2(
        "python3.12-minimal",
        "3.12.3-1ubuntu0.17",
        "amd64",
        _sha(b"synthetic interpreter package"),
        snapshot,
    )
    source = SourceBindingV5c1(
        "source-0",
        "fixture-source",
        "1.0-1",
        "amd64",
        snapshot,
        _sha(b"synthetic source archive"),
        2048,
        "./module.py",
        _sha(b"independent source record pin"),
        _sha(_SOURCE),
        len(_SOURCE),
    )
    library = reconstruction.ReconstructionLibraryV5c2(
        "/usr/lib/x86_64-linux-gnu/libc.so.6",
        _sha(b"synthetic library"),
        "12" * 20,
        replace(package, package="libc6", version="2.39-0ubuntu8"),
    )
    policy = reconstruction.PycReconstructionPolicyV5c2(
        interpreter_sha256=_sha(b"synthetic interpreter bytes"),
        interpreter_build_id="34" * 20,
        interpreter_package=package,
        required_libraries=(library,),
        source=source,
        parser_version="CPython-3.12.3",
        encoding_configuration="pep263-utf8-default-v1",
        co_filename="/module.py",
        optimization=0,
        invalidation="checked_hash",
        arguments=("/usr/bin/python3.12", "-I", "future-compiler"),
        environment=(("PYTHONHASHSEED", "0"),),
        isolation_requirements=("no-network", "read-only-input", "resource-limits"),
        tool_image_id="synthetic-noble-image",
        tool_image_sha256=_sha(b"synthetic tool image"),
        expected_image_sha256=_sha(b"separately pinned expected image"),
        expected_pyc_sha256=_sha(b"independently expected inert PYC"),
    )
    return reconstruction.PycReconstructionEvidenceV5c2(
        policy,
        policy.expected_pyc_sha256,
        "synthetic-run-1",
        "synthetic-evidence-1",
        "synthetic fixture declaration; no trusted collector",
        False,
        None,
    )


def _verify(
    declaration: reconstruction.PycReconstructionEvidenceV5c2,
    **kwargs: Any,
) -> reconstruction.PycReconstructionContractResultV5c2:
    return reconstruction.validate_reconstruction_evidence_v5c2(
        declaration,
        expected_policy=kwargs.pop("expected_policy", declaration.policy),
        source_bytes=kwargs.pop("source_bytes", _SOURCE),
        **kwargs,
    )


def _unverified(result: reconstruction.PycReconstructionContractResultV5c2) -> None:
    assert result.reconstruction == "RECONSTRUCTION_UNVERIFIED"
    assert result.pyc_reconstruction == "UNVERIFIED"
    assert result.execution == "MISSING"
    assert result.evidence_origin == "UNVERIFIED"
    assert result.external_authority_acceptance == "UNVERIFIED"
    assert result.authenticated_time == "UNVERIFIED"
    assert result.durable_anti_rollback == "BLOCKED"
    assert result.runtime_qualification == "BLOCKED"
    assert result.stage_9 == "IN_PROGRESS"
    assert result.qualification == "BLOCKED"
    assert result.stage_3b == "NOT AUTHORIZED"
    assert result.production == "NO-GO"
    assert result.refused is True


def _refused(result: reconstruction.PycReconstructionContractResultV5c2) -> None:
    assert result.consistency in {"MISMATCH", "INCOMPLETE"}
    assert result.reason_codes
    assert result.declaration is None
    _unverified(result)


def test_complete_structure_is_not_execution_evidence(declaration: Any) -> None:
    result = _verify(declaration)
    assert result.consistency == "STRUCTURE_VALID"
    assert result.declaration is declaration
    assert "RECONSTRUCTION_EXECUTION_EVIDENCE_MISSING" in result.unmet_requirements
    _unverified(result)


def test_origin_claim_never_authenticates_a_run(declaration: Any) -> None:
    claimed = replace(declaration, evidence_origin="trusted-signed-execution-proof")
    result = _verify(claimed)
    assert result.consistency == "STRUCTURE_VALID"
    _unverified(result)


@pytest.mark.parametrize("status", ["RECONSTRUCTION_VERIFIED", "VERIFIED", "PASS", ""])
def test_fictional_reconstruction_status_is_refused(declaration: Any, status: str) -> None:
    _refused(_verify(replace(declaration, verification_status=status)))


@pytest.mark.parametrize("field", ["profile", "run_id", "evidence_id", "evidence_origin"])
def test_missing_or_wrong_evidence_fields_refused(declaration: Any, field: str) -> None:
    _refused(_verify(replace(declaration, **{field: ""})))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("profile", "stage9-observer-pyc-reconstruction-policy-v5c1"),
        ("interpreter_sha256", "ab" * 32),
        ("interpreter_build_id", "ab" * 20),
        ("parser_version", "CPython-3.11.9"),
        ("encoding_configuration", "host-default"),
        ("co_filename", "/other.py"),
        ("optimization", 1),
        ("invalidation", "unchecked_hash"),
        ("arguments", ("/usr/bin/python3.12", "unexpected")),
        ("environment", (("PYTHONHASHSEED", "random"),)),
        ("isolation_requirements", ("unrestricted-host",)),
        ("tool_image_sha256", "cd" * 32),
        ("expected_image_sha256", "ef" * 32),
        ("expected_pyc_sha256", "aa" * 32),
    ],
)
def test_policy_substitution_refused(declaration: Any, field: str, replacement: Any) -> None:
    forged = replace(declaration, policy=replace(declaration.policy, **{field: replacement}))
    _refused(_verify(forged, expected_policy=declaration.policy))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("package", "python3.11-minimal"),
        ("version", "3.12.3-1ubuntu0.16"),
        ("architecture", "arm64"),
        ("archive_sha256", "ab" * 32),
        ("snapshot_sha256", "cd" * 32),
    ],
)
def test_interpreter_package_substitution_refused(
    declaration: Any, field: str, replacement: str
) -> None:
    package = replace(declaration.policy.interpreter_package, **{field: replacement})
    forged = replace(declaration, policy=replace(declaration.policy, interpreter_package=package))
    _refused(_verify(forged, expected_policy=declaration.policy))


@pytest.mark.parametrize("field", ["record_id", "snapshot_sha256", "archive_sha256", "sha256"])
def test_source_identity_substitution_refused(declaration: Any, field: str) -> None:
    source = replace(declaration.policy.source, **{field: "ab" * 32})
    forged = replace(declaration, policy=replace(declaration.policy, source=source))
    _refused(_verify(forged, expected_policy=declaration.policy))


def test_source_package_and_same_name_different_bytes_refused(declaration: Any) -> None:
    source = replace(declaration.policy.source, package="other-source-package")
    forged = replace(declaration, policy=replace(declaration.policy, source=source))
    _refused(_verify(forged, expected_policy=declaration.policy))
    _refused(_verify(declaration, source_bytes=_SOURCE[:-2] + b"X\n"))


def test_library_identity_and_missing_libraries_refused(declaration: Any) -> None:
    library = replace(declaration.policy.required_libraries[0], build_id="ab" * 20)
    forged = replace(declaration, policy=replace(declaration.policy, required_libraries=(library,)))
    _refused(_verify(forged, expected_policy=declaration.policy))
    _refused(
        _verify(replace(declaration, policy=replace(declaration.policy, required_libraries=())))
    )


def test_duplicate_environment_or_library_refused(declaration: Any) -> None:
    for changes in (
        {"environment": (("A", "one"), ("A", "two"))},
        {"required_libraries": declaration.policy.required_libraries * 2},
    ):
        _refused(_verify(replace(declaration, policy=replace(declaration.policy, **changes))))


@pytest.mark.parametrize("value", [False, -1, 3, None, float("nan")])
def test_invalid_optimization_refused(declaration: Any, value: Any) -> None:
    _refused(_verify(replace(declaration, policy=replace(declaration.policy, optimization=value))))


def test_missing_expected_policy_and_source_refused(declaration: Any) -> None:
    _refused(_verify(declaration, expected_policy=None))
    _refused(_verify(declaration, source_bytes=None))


def test_output_pin_cannot_be_self_substituted(declaration: Any) -> None:
    _refused(_verify(replace(declaration, output_pyc_sha256="ab" * 32)))


def test_missing_declaration_remains_unverified() -> None:
    result = reconstruction.validate_reconstruction_evidence_v5c2(
        None, expected_policy=None, source_bytes=None
    )
    assert result.consistency == "NOT_DECLARED"
    assert result.declaration is None
    _unverified(result)


def test_interrupted_run_is_incomplete(declaration: Any) -> None:
    interrupted = replace(declaration, interrupted=True, interruption_reason="terminated")
    result = _verify(interrupted)
    assert result.reason_codes == ("OPERATION_INTERRUPTED",)
    _refused(result)


def test_unreported_or_malformed_interruption_refused(declaration: Any) -> None:
    _refused(_verify(replace(declaration, interruption_reason="hidden termination")))
    _refused(_verify(replace(declaration, interrupted=True, interruption_reason=None)))
    _refused(_verify(replace(declaration, interrupted=1)))


def test_resource_limits_not_enlarged(declaration: Any) -> None:
    _refused(_verify(declaration, limits=replace(offline.Limits(), timeout_s=61)))
    _refused(
        _verify(declaration, limits=replace(offline.Limits(), max_file_bytes=128 * 1024 * 1024))
    )


def test_memory_limit_clears_declaration(declaration: Any) -> None:
    result = _verify(
        declaration,
        limits=replace(offline.Limits(), chunk_bytes=512, working_memory_bytes=65536),
    )
    assert result.reason_codes == ("LIMIT_EXCEEDED",)
    _refused(result)


def test_timeout_clears_declaration(declaration: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter([0.0, 61.0])
    monkeypatch.setattr(offline.time, "monotonic", lambda: next(clock))
    result = _verify(declaration)
    assert result.reason_codes == ("LIMIT_EXCEEDED",)
    _refused(result)


@pytest.mark.parametrize("failure", [MemoryError, KeyboardInterrupt, InterruptedError])
def test_computation_failure_clears_declaration(
    declaration: Any, monkeypatch: pytest.MonkeyPatch, failure: type[BaseException]
) -> None:
    def fail() -> None:
        raise failure

    monkeypatch.setattr(reconstruction.hashlib, "sha256", fail)
    _refused(_verify(declaration))


@pytest.mark.parametrize("failure", [KeyboardInterrupt, InterruptedError])
def test_interruption_after_first_source_chunk_clears_declaration(
    declaration: Any, monkeypatch: pytest.MonkeyPatch, failure: type[BaseException]
) -> None:
    source_bytes = _SOURCE * 100
    source = replace(declaration.policy.source, sha256=_sha(source_bytes), size=len(source_bytes))
    declaration = replace(declaration, policy=replace(declaration.policy, source=source))
    original_read = reconstruction._Budget.read
    completed_reads = 0

    def interrupt_second_read(budget: Any, amount: int) -> None:
        nonlocal completed_reads
        original_read(budget, amount)
        completed_reads += 1
        if completed_reads == 2:
            raise failure

    monkeypatch.setattr(reconstruction._Budget, "read", interrupt_second_read)
    result = _verify(
        declaration,
        source_bytes=source_bytes,
        limits=replace(offline.Limits(), chunk_bytes=512),
    )
    assert completed_reads == 2
    assert result.reason_codes == ("OPERATION_INTERRUPTED",)
    _refused(result)


def test_validation_has_no_execution_path(
    declaration: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("execution forbidden")

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "compile", forbidden)
        patch.setattr(builtins, "exec", forbidden)
        patch.setattr(builtins, "eval", forbidden)
        result = _verify(declaration)
    assert result.consistency == "STRUCTURE_VALID"
    _unverified(result)
