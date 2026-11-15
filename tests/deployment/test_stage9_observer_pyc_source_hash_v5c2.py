"""Independent PEP 552 comparisons over actual immutable synthetic DEBs.

Header expectations are static independent C-reference vectors. The bytecode
body is deliberately invalid data; no investigated source or interpreter runs.
"""

from __future__ import annotations

import _imp
import builtins
import copy
import hashlib
import importlib.util
import marshal
import os
import socket
import subprocess
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import pytest

from deployment import (
    stage9_observer_image_v5a as image,
    stage9_observer_offline as offline,
    stage9_observer_package_v3 as v3,
    stage9_observer_pyc_source_hash_v5c2 as pyc,
    stage9_observer_source_hash_algorithm_v5c2 as algorithm,
)
from deployment.stage9_observer_member_source_v5b import FrozenPackageV5b
from deployment.stage9_observer_offline import Limits
from tests.deployment.test_stage9_observer_image_v5a import (
    _blocked,
    _build,
    _directory,
    _file,
    _raw,
)
from tests.deployment.test_stage9_observer_pyc_v5c1 import (
    PycCase,
    _build_case as _build_v5c1_case,
)

pytestmark = [pytest.mark.security, pytest.mark.unit]
_SOURCE = b"value = 37\n"
# This static value comes from the independently compiled C reference retained
# with pyc_source_hash_v5c2 vectors, never from the function under test.
_SOURCE_HASH_HEX = "0bc3e3a8ab4ff6bb"
_INTERPRETER_SHA256 = "e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f"
_PYC_BODY = b"INERT BODY; NOT A MARSHALLED CODE OBJECT\n"
_PYC_SHA256 = "7f1e07689792df18b6cae867be8efd86cb8742819a0e7e493b64465dcbdd30d0"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass
class SourceHashCase(PycCase):
    def verify(self, **kwargs: Any) -> Any:
        mapping = copy.deepcopy(self.mapping)
        mapping["inventory_sha256"] = _sha(self.inventory_raw)
        mapping["expected_image_sha256"] = _sha(self.expected_raw)
        return pyc.verify_pyc_source_hash_v5c2(
            inventory_raw=kwargs.pop("inventory_raw", self.inventory_raw),
            expected_image_raw=kwargs.pop("expected_image_raw", self.expected_raw),
            mapping_raw=kwargs.pop("mapping_raw", _raw(mapping)),
            expected_image_sha256=kwargs.pop("expected_image_sha256", _sha(self.expected_raw)),
            archives=kwargs.pop("archives", self.archives),
            pyc_inputs=kwargs.pop("pyc_inputs", self.inputs),
            package_result=kwargs.pop("package_result", self.base.result),
            algorithm_id=kwargs.pop("algorithm_id", algorithm.ALGORITHM_ID),
            algorithm_materials=kwargs.pop("algorithm_materials", algorithm.ALGORITHM_MATERIALS),
            **kwargs,
        )


def _build_case(directory: Path, *, second: bool = False) -> SourceHashCase:
    old = _build_v5c1_case(directory, second=second)
    case = SourceHashCase(old.base, old.expected, old.mapping, old.archives, old.inputs)
    raw = bytes.fromhex("cb0d0d0a03000000" + _SOURCE_HASH_HEX) + _PYC_BODY
    case.inputs = tuple(replace(item, content=raw) for item in case.inputs)
    for row in case.expected["files"]:
        if row["type"] == "regular":
            row.update(size=57, sha256=_PYC_SHA256)
    for rows in (case.expected["bytecode"], case.mapping["transforms"]):
        for row in rows:
            row["source_hash_hex"] = _SOURCE_HASH_HEX
            row["compiler"]["interpreter_sha256"] = _INTERPRETER_SHA256
    return case


@pytest.fixture
def case(tmp_path: Path) -> SourceHashCase:
    return _build_case(tmp_path / "debs")


def _unverified(result: Any) -> None:
    _blocked(result)
    assert result.profile == "stage9-observer-pyc-source-hash-result-v5c2"
    assert result.pyc_reconstruction == "UNVERIFIED"
    assert result.reconstruction == "RECONSTRUCTION_UNVERIFIED"
    assert result.algorithm_acceptance == "UNVERIFIED"
    assert result.algorithm_evidence == "ALGORITHM_EVIDENCE_INCOMPLETE"
    assert result.exact_target_reference == "NOT_TESTED"
    assert "ALGORITHM_AUTHORITY_ACCEPTANCE_UNVERIFIED" in result.unmet_requirements
    assert "EXACT_TARGET_INTERPRETER_REFERENCE_NOT_TESTED" in result.unmet_requirements


def _refused(result: Any, code: str | None = None) -> None:
    assert result.consistency in {
        "HASH_MISMATCH",
        "INCOMPLETE",
        "ALGORITHM_EVIDENCE_INCOMPLETE",
    }, result
    assert result.reason_codes
    assert result.files == ()
    assert result.algorithm_id is None
    assert result.algorithm_materials == ()
    for field in (
        "inventory_sha256",
        "expected_image_sha256",
        "mapping_sha256",
        "snapshot_sha256",
        "package_result_sha256",
        "source_bytes_sha256",
        "algorithm_materials_sha256",
    ):
        assert getattr(result, field) is None, field
    if code is not None:
        assert code in result.reason_codes
    _unverified(result)


def test_actual_deb_bytes_independent_hash_and_pinned_pyc_compare(case: SourceHashCase) -> None:
    result = case.verify()
    assert result.consistency == "HASH_MATCH_LOCAL", result.reason_codes
    assert result.reason_codes == ()
    assert len(result.files) == 1
    output = result.files[0]
    assert output.computed_source_hash_hex == _SOURCE_HASH_HEX
    assert output.header_source_hash_hex == _SOURCE_HASH_HEX
    assert output.comparison == "SOURCE_HASH_MATCH"
    assert output.source_bytes == "SOURCE_BYTES_VERIFIED"
    assert output.byte_identity == "BYTE_IDENTITY_VERIFIED"
    assert output.source.sha256 == _sha(_SOURCE)
    assert output.source.size == len(_SOURCE)
    assert output.source.member == "./module.py"
    assert output.source.record_id == case.operation()["source"]["record_id"]
    assert output.source.snapshot_sha256 == case.expected["snapshot_sha256"]
    assert output.sha256 == _PYC_SHA256 == case.expected["files"][2]["sha256"]
    assert output.header.magic_hex == "cb0d0d0a"
    assert result.inventory_sha256 == _sha(case.inventory_raw)
    assert result.expected_image_sha256 == _sha(case.expected_raw)
    assert result.mapping_sha256 and result.source_bytes_sha256
    assert result.package_result_sha256 and result.algorithm_materials_sha256
    _unverified(result)


def test_matching_hash_with_invalid_code_body_never_proves_reconstruction(
    case: SourceHashCase,
) -> None:
    result = case.verify()
    assert result.consistency == "HASH_MATCH_LOCAL", result.reason_codes
    assert case.inputs[0].content[16:] == _PYC_BODY
    assert result.files[0].reconstruction == "RECONSTRUCTION_UNVERIFIED"
    assert "PYC_RECONSTRUCTION_UNVERIFIED" in result.unmet_requirements
    _unverified(result)


@pytest.mark.parametrize("algorithm_id", ["siphash24", "siphash13", "sha256", "", None, 1])
def test_unsupported_algorithm_cannot_substitute_for_local_profile(
    case: SourceHashCase, algorithm_id: Any
) -> None:
    _refused(case.verify(algorithm_id=algorithm_id))


@pytest.mark.parametrize("kind", ["missing", "changed-hash", "changed-id", "extra", "duplicate"])
def test_algorithm_material_identities_must_equal_local_profile(
    case: SourceHashCase, kind: str
) -> None:
    materials: Any = algorithm.ALGORITHM_MATERIALS
    if kind == "missing":
        materials = None
    elif kind == "changed-hash":
        materials = ((materials[0][0], "1" * 64),) + materials[1:]
    elif kind == "changed-id":
        materials = (("substituted-material", materials[0][1]),) + materials[1:]
    elif kind == "extra":
        materials += (("extra-material", "2" * 64),)
    else:
        materials += (materials[0],)
    _refused(case.verify(algorithm_materials=materials))


@pytest.mark.parametrize("alteration", ["header-only", "declaration-only", "both"])
def test_declared_hash_cannot_override_independent_computation(
    case: SourceHashCase, alteration: str
) -> None:
    if alteration != "declaration-only":
        raw = case.inputs[0].content[:8] + b"12345678" + _PYC_BODY
        case.inputs = (replace(case.inputs[0], content=raw),)
        case.expected["files"][2]["sha256"] = _sha(raw)
    if alteration != "header-only":
        for document in ("expected", "mapping"):
            case.operation(document)["source_hash_hex"] = b"12345678".hex()
    _refused(case.verify())


@pytest.mark.parametrize("mode", ["reversed", "wrong-key"])
def test_wrong_endian_or_keyed_hash_is_rejected_even_when_pinned(
    case: SourceHashCase, mode: str
) -> None:
    wrong = (
        bytes.fromhex(_SOURCE_HASH_HEX)[::-1]
        if mode == "reversed"
        # Independent C result for the incorrect low-16-bit-only magic key.
        else bytes.fromhex("4db70e9b9f4dc7c2")
    )
    raw = case.inputs[0].content[:8] + wrong + _PYC_BODY
    case.inputs = (replace(case.inputs[0], content=raw),)
    case.expected["files"][2]["sha256"] = _sha(raw)
    for document in ("expected", "mapping"):
        case.operation(document)["source_hash_hex"] = wrong.hex()
    result = case.verify()
    _refused(result)
    assert result.consistency == "HASH_MISMATCH"


def test_changed_source_byte_is_remeasured_from_actual_deb(case: SourceHashCase) -> None:
    archive = case.archives[0]
    assert _SOURCE in archive.content
    case.archives = (replace(archive, content=archive.content.replace(_SOURCE, b"value = 38\n")),)
    _refused(case.verify())


@pytest.mark.parametrize("flags", [0, 1, 2, 4, 7, 2**32 - 1])
def test_unsupported_invalidation_or_reserved_flags_refuse(
    case: SourceHashCase, flags: int
) -> None:
    raw = case.inputs[0].content[:4] + flags.to_bytes(4, "little") + case.inputs[0].content[8:]
    case.inputs = (replace(case.inputs[0], content=raw),)
    case.expected["files"][2]["sha256"] = _sha(raw)
    _refused(case.verify())


@pytest.mark.parametrize("invalidation", ["timestamp", "unchecked_hash"])
def test_other_invalidation_declarations_are_unsupported(
    case: SourceHashCase, invalidation: str
) -> None:
    for document in ("expected", "mapping"):
        case.operation(document)["invalidation"] = invalidation
    _refused(case.verify())


def test_changed_magic_is_not_admitted_by_self_consistent_declarations(
    case: SourceHashCase,
) -> None:
    raw = bytes.fromhex("a70d0d0a") + case.inputs[0].content[4:]
    case.inputs = (replace(case.inputs[0], content=raw),)
    case.expected["files"][2]["sha256"] = _sha(raw)
    for document in ("expected", "mapping"):
        case.operation(document)["compiler"]["magic_hex"] = "a70d0d0a"
    _refused(case.verify())


@pytest.mark.parametrize(
    "field,value",
    [
        ("build_id", "fedcba0987654321fedcba0987654321fedcba09"),
        ("interpreter_sha256", "1" * 64),
        ("version", "3.12.4"),
        ("architecture", "arm64"),
        ("co_filename", "/other.py"),
        ("generation_policy", "compile-with-host"),
    ],
)
def test_self_consistent_substituted_target_identity_refuses(
    case: SourceHashCase, field: str, value: Any
) -> None:
    for document in ("expected", "mapping"):
        case.operation(document)["compiler"][field] = value
    _refused(case.verify())


@pytest.mark.parametrize(
    "field,value",
    [
        ("package", "libpython3.12-minimal"),
        ("version", "3.12.3-1ubuntu0.18"),
        ("architecture", "arm64"),
        ("archive_sha256", "1" * 64),
        ("archive_size", 99),
        ("member", "module.py"),
        ("record_id", "2" * 64),
        ("source_id", "unknown"),
        ("sha256", "3" * 64),
        ("size", 99),
    ],
)
def test_full_source_package_and_literal_member_binding_are_required(
    case: SourceHashCase, field: str, value: Any
) -> None:
    for document in ("expected", "mapping"):
        case.operation(document)["source"][field] = value
    _refused(case.verify())


@pytest.mark.parametrize("where", ["expected", "mapping", "source"])
def test_snapshot_substitution_refuses(case: SourceHashCase, where: str) -> None:
    if where == "source":
        for document in ("expected", "mapping"):
            case.operation(document)["source"]["snapshot_sha256"] = "1" * 64
    else:
        (case.expected if where == "expected" else case.mapping)["snapshot_sha256"] = "1" * 64
    _refused(case.verify())


@pytest.mark.parametrize("where", ["expected", "mapping"])
def test_wrong_declaration_contract_version_refuses(case: SourceHashCase, where: str) -> None:
    (case.expected if where == "expected" else case.mapping)["format"] = "unversioned-pyc"
    _refused(case.verify())


@pytest.mark.parametrize(
    "where", ["input", "expected-file", "expected-op", "mapping-op", "archive"]
)
def test_extra_artifacts_or_ambiguous_inputs_refuse(case: SourceHashCase, where: str) -> None:
    if where == "input":
        case.inputs += (replace(case.inputs[0], file_id="extra"),)
    elif where == "expected-file":
        case.expected["files"].append(_file("extra", "/extra.pyc", b"extra"))
    elif where == "expected-op":
        case.expected["bytecode"].append(copy.deepcopy(case.operation()))
    elif where == "mapping-op":
        case.mapping["transforms"].append(copy.deepcopy(case.operation("mapping")))
    else:
        case.archives += case.archives
    _refused(case.verify())


@pytest.mark.parametrize(
    "field", ["source", "compiler", "source_hash_hex", "invalidation", "usage", "bytecode_format"]
)
def test_missing_required_declaration_refuses(case: SourceHashCase, field: str) -> None:
    for document in ("expected", "mapping"):
        case.operation(document).pop(field)
    _refused(case.verify())


@pytest.mark.parametrize("field", ["sha256", "size"])
def test_missing_independent_pyc_expected_is_not_filled_from_candidate(
    case: SourceHashCase, field: str
) -> None:
    case.expected["files"][2].pop(field)
    _refused(case.verify())


def test_expected_image_pin_cannot_be_replaced_with_candidate_expectation(
    case: SourceHashCase,
) -> None:
    expected_pin = _sha(case.expected_raw)
    case.expected["image_id"] = "substituted-image"
    _refused(case.verify(expected_image_sha256=expected_pin))


@pytest.mark.parametrize(
    "claim",
    [
        None,
        v3.PackageInventoryResultV3("MATCH", ()),
        v3.PackageInventoryResultV3("INCOMPLETE", ("PROVENANCE_INCOMPLETE",)),
    ],
)
def test_incomplete_v3_claim_cannot_replace_measured_deb_continuity(
    case: SourceHashCase, claim: Any
) -> None:
    _refused(case.verify(package_result=claim))


@pytest.mark.parametrize("where", ["archive", "input", "mapping-op"])
def test_missing_actual_v5b_input_or_operation_refuses(case: SourceHashCase, where: str) -> None:
    if where == "archive":
        case.archives = ()
    elif where == "input":
        case.inputs = ()
    else:
        case.mapping["transforms"] = []
    _refused(case.verify())


@pytest.mark.parametrize(
    "changes",
    [
        {"working_memory_bytes": 262144},
        {"max_total_bytes": 512},
        {"max_files": 2},
        {"max_archive_members": 2},
        {"max_file_bytes": 16},
    ],
)
def test_memory_and_input_exhaustion_clear_all_evidence(
    case: SourceHashCase, changes: dict[str, int]
) -> None:
    _refused(case.verify(limits=replace(Limits(), **changes)), "LIMIT_EXCEEDED")


@pytest.mark.parametrize("exception", [KeyboardInterrupt, InterruptedError, MemoryError])
def test_interrupted_processing_returns_no_positive_bindings(
    case: SourceHashCase, monkeypatch: pytest.MonkeyPatch, exception: type[BaseException]
) -> None:
    def interrupted(self: Any) -> None:
        raise exception

    monkeypatch.setattr(offline._Budget, "check", interrupted)
    _refused(case.verify())


def test_elapsed_deadline_refuses(case: SourceHashCase, monkeypatch: pytest.MonkeyPatch) -> None:
    original = offline._Budget.check
    checks = 0

    def exhausted(self: Any) -> None:
        nonlocal checks
        checks += 1
        if checks >= 70:
            self.started -= 61
        original(self)

    monkeypatch.setattr(offline._Budget, "check", exhausted)
    _refused(case.verify(), "LIMIT_EXCEEDED")
    assert checks >= 70


@pytest.mark.parametrize(
    "exception", [KeyboardInterrupt, InterruptedError, MemoryError, ValueError, OSError]
)
def test_failure_during_second_independent_hash_erases_first_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exception: type[BaseException]
) -> None:
    case = _build_case(tmp_path / "debs", second=True)
    original = algorithm.independent_source_hash
    computations = 0

    def fail_second(*args: Any, **kwargs: Any) -> bytes:
        nonlocal computations
        computations += 1
        if computations == 2:
            raise exception
        return original(*args, **kwargs)

    monkeypatch.setattr(algorithm, "independent_source_hash", fail_second)
    _refused(case.verify())
    assert computations == 2


def test_bad_second_artifact_cannot_leave_first_file_success(tmp_path: Path) -> None:
    case = _build_case(tmp_path / "debs", second=True)
    assert case.verify().consistency == "HASH_MATCH_LOCAL"
    case.inputs = (case.inputs[0], replace(case.inputs[1], content=case.inputs[1].content + b"!"))
    _refused(case.verify())


def test_fictitious_reconstruction_proof_is_not_an_accepted_extension(case: SourceHashCase) -> None:
    for document in ("expected", "mapping"):
        case.operation(document)["reconstruction_proof_sha256"] = _sha(b"caller claim")
    _refused(case.verify())


def test_empty_bytecode_set_does_not_claim_hash_verification(case: SourceHashCase) -> None:
    case.expected["files"] = [_directory("root", "/")]
    case.expected["bytecode"] = []
    case.mapping["transforms"] = []
    case.inputs = ()
    result = case.verify()
    assert result.consistency == "NOT_DECLARED", result.reason_codes
    assert result.files == ()
    _unverified(result)


def test_host_magic_and_hash_apis_are_not_used(
    case: SourceHashCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("V5c2 attempted host hashing or investigated-code execution")

    monkeypatch.setattr(importlib.util, "MAGIC_NUMBER", b"HOST")
    monkeypatch.setattr(importlib.util, "source_hash", forbidden)
    monkeypatch.setattr(_imp, "source_hash", forbidden)
    monkeypatch.setattr(os, "open", forbidden)
    monkeypatch.setattr(os, "readlink", forbidden)
    monkeypatch.setattr(Path, "resolve", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(marshal, "loads", forbidden)
    with monkeypatch.context() as code:
        code.setattr(builtins, "compile", forbidden)
        code.setattr(builtins, "exec", forbidden)
        code.setattr(builtins, "eval", forbidden)
        result = case.verify()
    assert result.consistency == "HASH_MATCH_LOCAL", result.reason_codes
    assert result.files[0].computed_source_hash_hex == _SOURCE_HASH_HEX
    _unverified(result)


def test_same_member_name_in_distinct_packages_never_selects_wrong_bytes(tmp_path: Path) -> None:
    case = _build_case(tmp_path / "original")
    other = _build(
        tmp_path / "two-packages",
        entries=[("./", "5", b""), ("./module.py", "0", _SOURCE)],
        second_entries=[("./", "5", b""), ("./module.py", "0", b"value = 73\n")],
    )
    case.base = other
    case.expected["sources"] = copy.deepcopy(other.expected["sources"])
    case.mapping["sources"] = copy.deepcopy(other.expected["sources"])
    case.archives = tuple(
        FrozenPackageV5b(row["filename"], (other.directory / row["filename"]).read_bytes())
        for row in other.inventory["packages"]
    )
    # Selecting first source is valid despite the identically named second member.
    assert case.verify().consistency == "HASH_MATCH_LOCAL"
    package = v3.parse_package_inventory_v3(other.inventory_raw)[1]
    record = next(row for row in package.data.records if row.name == "./module.py")
    for document in ("expected", "mapping"):
        case.operation(document)["source"].update(
            source_id="source-1",
            package=package.package,
            archive_sha256=package.sha256,
            archive_size=package.size,
            record_id=image.record_id_v5a(record),
            sha256=_sha(b"value = 73\n"),
        )
    # All source/DEB pins now correctly describe the other bytes. Its equal name
    # cannot satisfy the independently specified original source hash/header.
    result = case.verify()
    _refused(result)
    assert result.consistency == "HASH_MISMATCH"


@pytest.mark.parametrize("name", [descriptor.name for descriptor in fields(Limits)])
def test_local_limits_can_never_expand_v5c1_defaults(case: SourceHashCase, name: str) -> None:
    defaults = Limits()
    value = getattr(defaults, name)
    _refused(case.verify(limits=replace(defaults, **{name: value + 1})), "LIMIT_EXCEEDED")


@pytest.mark.parametrize("value", [None, b"", bytes(7), bytes(9), "0bc3e3a8ab4ff6bb", bytearray(8)])
def test_invalid_primitive_result_is_never_published_as_hash_match(
    case: SourceHashCase, monkeypatch: pytest.MonkeyPatch, value: Any
) -> None:
    monkeypatch.setattr(algorithm, "independent_source_hash", lambda *args, **kwargs: value)
    _refused(case.verify(), "ALGORITHM_RESULT_REJECTED")


def test_second_source_hash_mismatch_clears_first_positive_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _build_case(tmp_path / "debs", second=True)
    original = algorithm.independent_source_hash
    computations = 0

    def wrong_second(*args: Any, **kwargs: Any) -> bytes:
        nonlocal computations
        computations += 1
        if computations == 2:
            return bytes(8)
        return original(*args, **kwargs)

    monkeypatch.setattr(algorithm, "independent_source_hash", wrong_second)
    _refused(case.verify(), "SOURCE_HASH_MISMATCH")
    assert computations == 2


def test_elapsed_budget_is_checked_inside_independent_computation(
    case: SourceHashCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = algorithm.independent_source_hash

    def expire_during_hash(*args: Any, **kwargs: Any) -> bytes:
        check = kwargs["check"]
        check.__self__.started -= 61
        return original(*args, **kwargs)

    monkeypatch.setattr(algorithm, "independent_source_hash", expire_during_hash)
    _refused(case.verify(), "LIMIT_EXCEEDED")


@pytest.mark.parametrize("raw", [b"{}", b'{"format":1,"format":2}', b"[" * 50 + b"]" * 50])
def test_malformed_or_ambiguous_mapping_is_refused(case: SourceHashCase, raw: bytes) -> None:
    _refused(case.verify(mapping_raw=raw))
