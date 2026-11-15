"""Inert bytecode evidence over actual immutable synthetic DEB source bytes.

These fixtures deliberately contain no marshalled code object. Header and whole
file identity cannot establish reconstruction, even when every declaration agrees.
No test compiles, imports, deserializes or executes the investigated artifact.
"""

from __future__ import annotations

import builtins
import copy
import hashlib
import importlib.util
import marshal
import os
import socket
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from deployment import (
    stage9_observer_image_v5a as image,
    stage9_observer_offline as offline,
    stage9_observer_package_v3 as v3,
    stage9_observer_pyc_v5c1 as pyc,
)
from deployment.stage9_observer_member_source_v5b import FrozenPackageV5b
from deployment.stage9_observer_offline import Limits
from tests.deployment.test_stage9_observer_image_v5a import (
    ImageCase,
    _blocked,
    _build,
    _directory,
    _file,
    _raw,
)

pytestmark = [pytest.mark.security, pytest.mark.unit]
_SOURCE = b"value = 37\n"
_PYC = bytes.fromhex("cb0d0d0a030000003132333435363738") + (
    b"INERT BODY; NOT A MARSHALLED CODE OBJECT\n"
)
# Independently specified fixture expectation; verification never fills this
# field from the supplied artifact or from a measurement result.
_PYC_SHA256 = "01707eb957370a0a6ce8fb1d747cbd6774bdd5c047bd98a1a3ad8e95469ba70d"
_PYC_PATH = "/cache/module.cpython-312.pyc"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass
class PycCase:
    base: ImageCase
    expected: dict[str, Any]
    mapping: dict[str, Any]
    archives: tuple[FrozenPackageV5b, ...]
    inputs: tuple[pyc.FrozenPycV5c1, ...]

    @property
    def inventory(self) -> dict[str, Any]:
        return self.base.inventory

    @property
    def inventory_raw(self) -> bytes:
        return self.base.inventory_raw

    @property
    def expected_raw(self) -> bytes:
        return _raw(self.expected)

    def operation(self, document: str = "expected") -> dict[str, Any]:
        return (
            self.expected["bytecode"][0]
            if document == "expected"
            else self.mapping["transforms"][0]
        )

    def verify(self, **kwargs: Any) -> Any:
        mapping = copy.deepcopy(self.mapping)
        mapping["inventory_sha256"] = _sha(self.inventory_raw)
        mapping["expected_image_sha256"] = _sha(self.expected_raw)
        return pyc.verify_pyc_v5c1(
            inventory_raw=kwargs.pop("inventory_raw", self.inventory_raw),
            expected_image_raw=kwargs.pop("expected_image_raw", self.expected_raw),
            mapping_raw=kwargs.pop("mapping_raw", _raw(mapping)),
            expected_image_sha256=kwargs.pop("expected_image_sha256", _sha(self.expected_raw)),
            archives=kwargs.pop("archives", self.archives),
            pyc_inputs=kwargs.pop("pyc_inputs", self.inputs),
            package_result=kwargs.pop("package_result", self.base.result),
            **kwargs,
        )


def _build_case(directory: Path, *, second: bool = False) -> PycCase:
    base = _build(
        directory,
        entries=[("./", "5", b""), ("./module.py", "0", _SOURCE)],
        expected_files=[_directory("root", "/"), _file("source", "/module.py", _SOURCE)],
    )
    package = v3.parse_package_inventory_v3(base.inventory_raw)[0]
    record = next(row for row in package.data.records if row.name == "./module.py")
    source = dict(
        source_id="source-0",
        package=package.package,
        version=package.version,
        architecture=package.architecture,
        snapshot_sha256=base.expected["snapshot_sha256"],
        archive_sha256=package.sha256,
        archive_size=package.size,
        member="./module.py",
        record_id=image.record_id_v5a(record),
        sha256=_sha(_SOURCE),
        size=len(_SOURCE),
    )
    compiler = dict(
        build_id="337d65cf00021797985cc9f77c0cc334a9fbeb38",
        interpreter_sha256=_sha(b"independently pinned synthetic interpreter"),
        version="3.12.3",
        architecture="amd64",
        magic_hex="cb0d0d0a",
        optimization=0,
        co_filename="/module.py",
        target_path=_PYC_PATH,
        generation_policy="prebuilt-pep552-checked-hash-v1",
    )
    operation = dict(
        transform_id="prebuilt-module",
        kind="prebuilt_pyc_v5c1",
        output_file_id="bytecode",
        source=source,
        compiler=compiler,
        bytecode_format="cpython-pep552-v1",
        invalidation="checked_hash",
        source_hash_hex="3132333435363738",
        usage="inert_artifact_only",
    )
    expected = copy.deepcopy(base.expected)
    expected.update(
        format="stage9-observer-expected-image-v5c1",
        files=[
            _directory("root", "/"),
            _directory("cache", "/cache"),
            dict(
                file_id="bytecode",
                path=_PYC_PATH,
                type="regular",
                size=57,
                sha256=_PYC_SHA256,
                uid=0,
                gid=0,
                mode=0o644,
            ),
        ],
        bytecode=[copy.deepcopy(operation)],
        import_roots=[],
    )
    mapping = dict(
        format="stage9-observer-pyc-mapping-v5c1",
        snapshot_sha256=expected["snapshot_sha256"],
        inventory_sha256=_sha(base.inventory_raw),
        expected_image_sha256=_sha(_raw(expected)),
        sources=copy.deepcopy(expected["sources"]),
        directory_policy="identical-explicit-v1",
        transforms=[copy.deepcopy(operation)],
    )
    archives = tuple(
        FrozenPackageV5b(row["filename"], (directory / row["filename"]).read_bytes())
        for row in base.inventory["packages"]
    )
    inputs = (pyc.FrozenPycV5c1("bytecode", _PYC_PATH, _PYC, 0, 0, 0o644),)
    case = PycCase(base, expected, mapping, archives, inputs)
    if second:
        path = "/second/module.cpython-312.pyc"
        expected["files"].extend(
            [
                _directory("second", "/second"),
                dict(expected["files"][2], file_id="second-pyc", path=path),
            ]
        )
        op = copy.deepcopy(operation)
        op.update(transform_id="second-transform", output_file_id="second-pyc")
        op["compiler"]["target_path"] = path
        expected["bytecode"].append(copy.deepcopy(op))
        mapping["transforms"].append(op)
        case.inputs += (pyc.FrozenPycV5c1("second-pyc", path, _PYC, 0, 0, 0o644),)
    return case


@pytest.fixture
def case(tmp_path: Path) -> PycCase:
    return _build_case(tmp_path / "debs")


def _unverified(result: Any) -> None:
    _blocked(result)
    assert result.pyc_reconstruction == "UNVERIFIED"
    assert result.profile == "stage9-observer-pyc-result-v5c1"


def _refused(result: Any, code: str | None = None) -> None:
    assert result.consistency in {"MISMATCH", "INCOMPLETE"}, result
    assert result.reason_codes
    assert result.files == ()
    for field in (
        "inventory_sha256",
        "expected_image_sha256",
        "mapping_sha256",
        "snapshot_sha256",
        "package_result_sha256",
        "source_bytes_sha256",
    ):
        assert getattr(result, field) is None, field
    if code is not None:
        assert code in result.reason_codes
    _unverified(result)


def test_pinned_complete_bytes_checked_hash_and_actual_source_binding(case: PycCase) -> None:
    result = case.verify()
    assert result.consistency == "BYTECODE_EVIDENCE_ONLY", result.reason_codes
    assert len(result.files) == 1
    output = result.files[0]
    assert (output.file_id, output.path, output.sha256, output.size) == (
        "bytecode",
        _PYC_PATH,
        _PYC_SHA256,
        57,
    )
    assert (output.uid, output.gid, output.mode) == (0, 0, 0o644)
    assert output.byte_identity == "BYTE_IDENTITY_VERIFIED"
    assert output.header_structure == "HEADER_STRUCTURE_VERIFIED"
    assert output.source_binding == "SOURCE_BINDING_VERIFIED"
    assert output.header.magic_hex == "cb0d0d0a"
    assert output.header.flags == 3
    assert output.header.invalidation == "checked_hash"
    assert output.header.source_hash_hex == "3132333435363738"
    assert output.source.sha256 == _sha(_SOURCE)
    assert output.source.member == "./module.py"
    assert output.source.record_id == case.operation()["source"]["record_id"]
    assert result.inventory_sha256 == _sha(case.inventory_raw)
    assert result.expected_image_sha256 == _sha(case.expected_raw)
    assert result.mapping_sha256 and result.package_result_sha256 and result.source_bytes_sha256
    _unverified(result)


def test_pinned_structurally_valid_non_code_body_never_proves_reconstruction(case: PycCase) -> None:
    # The header is valid and its complete artifact hash agrees, but this body
    # was deliberately not generated from _SOURCE by any CPython interpreter.
    result = case.verify()
    assert result.consistency == "BYTECODE_EVIDENCE_ONLY", result.reason_codes
    assert result.files[0].sha256 == _PYC_SHA256
    assert result.files[0].source_hash == "SOURCE_HASH_UNVERIFIED"
    assert result.files[0].reconstruction == "RECONSTRUCTION_UNVERIFIED"
    assert "MATCH" != result.consistency
    _unverified(result)


@pytest.mark.parametrize("flags", [0, 1, 2, 4, 7, 2**32 - 1])
def test_timestamp_unchecked_hash_and_invalid_flags_are_refused(case: PycCase, flags: int) -> None:
    changed = _PYC[:4] + flags.to_bytes(4, "little") + _PYC[8:]
    case.inputs = (replace(case.inputs[0], content=changed),)
    case.expected["files"][2]["sha256"] = _sha(changed)
    _refused(case.verify())


@pytest.mark.parametrize("invalidation", ["timestamp", "unchecked_hash"])
def test_declarations_do_not_equate_other_invalidation_profiles(
    case: PycCase, invalidation: str
) -> None:
    for document in ("expected", "mapping"):
        case.operation(document)["invalidation"] = invalidation
    _refused(case.verify())


@pytest.mark.parametrize("raw", [b"", b"x" * 15, b"BAD!" + _PYC[4:]])
def test_minimum_size_and_magic_are_checked_after_independent_pinning(
    case: PycCase, raw: bytes
) -> None:
    case.inputs = (replace(case.inputs[0], content=raw),)
    case.expected["files"][2].update(size=len(raw), sha256=_sha(raw))
    _refused(case.verify())


def test_fixed_header_boundary_does_not_pretend_to_validate_opaque_body(case: PycCase) -> None:
    raw = _PYC[:16]
    case.inputs = (replace(case.inputs[0], content=raw),)
    case.expected["files"][2].update(size=16, sha256=_sha(raw))
    result = case.verify()
    assert result.consistency == "BYTECODE_EVIDENCE_ONLY", result.reason_codes
    assert result.files[0].reconstruction == "RECONSTRUCTION_UNVERIFIED"
    _unverified(result)


def test_all_eight_remaining_header_bytes_are_bound(case: PycCase) -> None:
    changed = _PYC[:8] + b"87654321" + _PYC[16:]
    case.inputs = (replace(case.inputs[0], content=changed),)
    case.expected["files"][2]["sha256"] = _sha(changed)
    _refused(case.verify())


def test_replaced_pyc_cannot_redefine_its_independent_expected_digest(case: PycCase) -> None:
    case.inputs = (replace(case.inputs[0], content=_PYC[:-1] + b"!"),)
    _refused(case.verify())


def test_expected_image_pin_cannot_be_derived_from_replacement(case: PycCase) -> None:
    expected_pin = _sha(case.expected_raw)
    case.expected["image_id"] = "substituted-image"
    _refused(case.verify(expected_image_sha256=expected_pin))


@pytest.mark.parametrize("kind", [bytearray, memoryview])
def test_pyc_input_must_be_immutable_bytes(case: PycCase, kind: Any) -> None:
    changed: Any = kind(_PYC)
    case.inputs = (replace(case.inputs[0], content=changed),)
    _refused(case.verify())


def test_source_payload_replacement_is_measured_in_actual_deb(case: PycCase) -> None:
    archive = case.archives[0]
    assert _SOURCE in archive.content
    case.archives = (replace(archive, content=archive.content.replace(_SOURCE, b"value = 73\n")),)
    _refused(case.verify())


@pytest.mark.parametrize(
    "field,value", [("sha256", "1" * 64), ("size", 999), ("record_id", "2" * 64)]
)
def test_source_hash_size_and_v5a_record_id_must_match_real_record(
    case: PycCase, field: str, value: Any
) -> None:
    for document in ("expected", "mapping"):
        case.operation(document)["source"][field] = value
    _refused(case.verify())


@pytest.mark.parametrize(
    "field,value", [("sha256", "1" * 64), ("size", 999), ("name", "./other.py")]
)
def test_changed_source_inventory_cannot_reuse_old_deb(
    case: PycCase, field: str, value: Any
) -> None:
    case.inventory["packages"][0]["data"]["records"][1][field] = value
    _refused(case.verify())


@pytest.mark.parametrize(
    "field,value",
    [
        ("build_id", "fedcba0987654321fedcba0987654321fedcba09"),
        ("interpreter_sha256", "1" * 64),
        ("version", "3.12.4"),
        ("architecture", "arm64"),
        ("magic_hex", "a70d0d0a"),
        ("optimization", 1),
        ("co_filename", "/other.py"),
        ("target_path", "/cache/other.cpython-312.pyc"),
        ("generation_policy", "compile-with-host"),
    ],
)
def test_compiler_claims_cannot_disagree_with_independent_expectation(
    case: PycCase, field: str, value: Any
) -> None:
    case.operation("mapping")["compiler"][field] = value
    _refused(case.verify())


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", "3.11.0"),
        ("co_filename", "/other.py"),
        ("architecture", "arm64"),
        ("optimization", 3),
    ],
)
def test_self_consistent_but_unsupported_or_source_inconsistent_compiler_is_refused(
    case: PycCase, field: str, value: Any
) -> None:
    for document in ("expected", "mapping"):
        case.operation(document)["compiler"][field] = value
    _refused(case.verify())


@pytest.mark.parametrize("build_id", ["12345678", "a" * 39, "a" * 41, "G" * 40, "A" * 40])
def test_full_canonical_build_id_cannot_be_abbreviated_or_malformed(
    case: PycCase, build_id: str
) -> None:
    for document in ("expected", "mapping"):
        case.operation(document)["compiler"]["build_id"] = build_id
    _refused(case.verify())


@pytest.mark.parametrize("optimization", [1, 2])
def test_explicit_supported_optimization_and_target_path_agree(
    case: PycCase, optimization: int
) -> None:
    path = f"/cache/module.cpython-312.opt-{optimization}.pyc"
    case.expected["files"][2]["path"] = path
    case.inputs = (replace(case.inputs[0], path=path),)
    for document in ("expected", "mapping"):
        case.operation(document)["compiler"].update(optimization=optimization, target_path=path)
    result = case.verify()
    assert result.consistency == "BYTECODE_EVIDENCE_ONLY", result.reason_codes
    assert result.files[0].reconstruction == "RECONSTRUCTION_UNVERIFIED"


@pytest.mark.parametrize(
    "path",
    [
        "/cache/wrong.pyc",
        "/cache/../module.cpython-312.pyc",
        "/missing/module.cpython-312.pyc",
        "relative.pyc",
        "/cache/module.py",
    ],
)
def test_disallowed_or_undeclared_pyc_location_refuses(case: PycCase, path: str) -> None:
    case.expected["files"][2]["path"] = path
    case.inputs = (replace(case.inputs[0], path=path),)
    for document in ("expected", "mapping"):
        case.operation(document)["compiler"]["target_path"] = path
    _refused(case.verify())


@pytest.mark.parametrize(
    "field,value",
    [
        ("uid", 9),
        ("gid", 9),
        ("mode", 0o600),
        ("size", 99),
        ("path", "/cache/other.pyc"),
        ("file_id", "other"),
    ],
)
def test_supplied_output_identity_and_metadata_are_measured_exactly(
    case: PycCase, field: str, value: Any
) -> None:
    if field == "size":
        case.expected["files"][2][field] = value
    else:
        case.inputs = (replace(case.inputs[0], **{field: value}),)
    _refused(case.verify())


@pytest.mark.parametrize(
    "field,value", [("uid", 9), ("gid", 9), ("mode", 0o600), ("type", "directory")]
)
def test_independent_image_metadata_cannot_be_inferred_from_pyc(
    case: PycCase, field: str, value: Any
) -> None:
    case.expected["files"][2][field] = value
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
        ("source_id", "unknown"),
    ],
)
def test_complete_source_package_provenance_is_required(
    case: PycCase, field: str, value: Any
) -> None:
    for document in ("expected", "mapping"):
        case.operation(document)["source"][field] = value
    _refused(case.verify())


@pytest.mark.parametrize("where", ["expected", "mapping", "source"])
def test_snapshots_cannot_be_mixed(case: PycCase, where: str) -> None:
    if where == "source":
        for document in ("expected", "mapping"):
            case.operation(document)["source"]["snapshot_sha256"] = "1" * 64
    else:
        (case.expected if where == "expected" else case.mapping)["snapshot_sha256"] = "1" * 64
    _refused(case.verify())


@pytest.mark.parametrize(
    "where", ["input", "expected-file", "expected-op", "mapping-op", "archive"]
)
def test_additional_or_ambiguous_evidence_is_refused(case: PycCase, where: str) -> None:
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
    "where", ["input", "expected-op", "mapping-op", "archive", "parent-directory"]
)
def test_missing_data_is_never_automatically_completed(case: PycCase, where: str) -> None:
    if where == "input":
        case.inputs = ()
    elif where == "expected-op":
        case.expected["bytecode"].clear()
    elif where == "mapping-op":
        case.mapping["transforms"].clear()
    elif where == "archive":
        case.archives = ()
    else:
        case.expected["files"].pop(1)
    _refused(case.verify())


@pytest.mark.parametrize(
    "field", ["source", "compiler", "source_hash_hex", "invalidation", "usage", "bytecode_format"]
)
def test_missing_required_provenance_or_parameters_refuse(case: PycCase, field: str) -> None:
    for document in ("expected", "mapping"):
        case.operation(document).pop(field)
    _refused(case.verify())


@pytest.mark.parametrize("field", ["sha256", "size"])
def test_missing_independent_pyc_expectations_are_never_filled_from_candidate(
    case: PycCase, field: str
) -> None:
    case.expected["files"][2].pop(field)
    _refused(case.verify())


def test_reconstruction_claim_cannot_promote_status_through_extra_input(case: PycCase) -> None:
    for document in ("expected", "mapping"):
        case.operation(document)["reconstruction_proof_sha256"] = _sha(b"caller claim")
    _refused(case.verify())


def test_ambiguous_duplicate_source_identity_refuses(case: PycCase) -> None:
    case.expected["sources"].append(copy.deepcopy(case.expected["sources"][0]))
    case.mapping["sources"].append(copy.deepcopy(case.mapping["sources"][0]))
    _refused(case.verify())


def test_import_root_never_admits_pyc_as_active_observer_code(case: PycCase) -> None:
    case.expected["import_roots"] = ["/cache"]
    _refused(case.verify())


@pytest.mark.parametrize("usage", ["active_import", "observer_entrypoint", "runtime_stdlib"])
def test_runtime_roles_are_outside_v5c1(case: PycCase, usage: str) -> None:
    for document in ("expected", "mapping"):
        case.operation(document)["usage"] = usage
    _refused(case.verify())


def test_no_declaration_does_not_claim_bytecode_qualification(case: PycCase) -> None:
    case.expected["files"] = [_directory("root", "/")]
    case.expected["bytecode"] = []
    case.mapping["transforms"] = []
    case.inputs = ()
    result = case.verify()
    assert result.consistency == "NOT_DECLARED", result.reason_codes
    assert result.files == ()
    _unverified(result)


@pytest.mark.parametrize(
    "result",
    [
        None,
        v3.PackageInventoryResultV3("MATCH", ()),
        v3.PackageInventoryResultV3("INCOMPLETE", ("PROVENANCE_INCOMPLETE",)),
    ],
)
def test_v3_claim_cannot_substitute_for_complete_continuity(case: PycCase, result: Any) -> None:
    _refused(case.verify(package_result=result))


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
def test_memory_and_input_budgets_refuse_without_partial_success(
    case: PycCase, changes: dict[str, int]
) -> None:
    _refused(case.verify(limits=replace(Limits(), **changes)), "LIMIT_EXCEEDED")


@pytest.mark.parametrize("exception", [KeyboardInterrupt, InterruptedError, MemoryError])
def test_interruption_refuses_every_partial_result(
    case: PycCase, monkeypatch: pytest.MonkeyPatch, exception: type[BaseException]
) -> None:
    def interrupted(self: Any) -> None:
        raise exception

    monkeypatch.setattr(offline._Budget, "check", interrupted)
    _refused(case.verify())


def test_elapsed_deadline_during_bounded_processing_refuses(
    case: PycCase, monkeypatch: pytest.MonkeyPatch
) -> None:
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


def test_bad_second_artifact_erases_first_verified_file(tmp_path: Path) -> None:
    case = _build_case(tmp_path / "debs", second=True)
    assert case.verify().consistency == "BYTECODE_EVIDENCE_ONLY"
    case.inputs = (case.inputs[0], replace(case.inputs[1], content=_PYC + b"extra"))
    _refused(case.verify())


@pytest.mark.parametrize(
    "exception", [KeyboardInterrupt, InterruptedError, MemoryError, ValueError]
)
def test_failure_after_first_staged_file_erases_all_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exception: type[BaseException]
) -> None:
    case = _build_case(tmp_path / "debs", second=True)
    original = pyc._header
    headers = 0

    def fail_second(*args: Any, **kwargs: Any) -> Any:
        nonlocal headers
        headers += 1
        if headers == 2:
            raise exception
        return original(*args, **kwargs)

    monkeypatch.setattr(pyc, "_header", fail_second)
    _refused(case.verify())
    assert headers == 2


def test_target_magic_never_comes_from_host_default(
    case: PycCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(importlib.util, "MAGIC_NUMBER", b"HOST")
    result = case.verify()
    assert result.consistency == "BYTECODE_EVIDENCE_ONLY", result.reason_codes
    assert result.files[0].header.magic_hex == "cb0d0d0a"
    _unverified(result)


@pytest.mark.parametrize(
    "raw",
    [
        b"{}",
        b'{"format":1,"format":2}',
        b"[" * 50 + b"]" * 50,
        b'{"format":NaN}',
        b'{"format":null}',
    ],
)
def test_malformed_declarations_refuse(case: PycCase, raw: bytes) -> None:
    _refused(case.verify(mapping_raw=raw))


def test_verifier_never_uses_host_io_network_execution_or_marshal(
    case: PycCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("V5c1 attempted host I/O, execution or code object deserialization")

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
    assert result.consistency == "BYTECODE_EVIDENCE_ONLY", result.reason_codes
    assert result.files[0].reconstruction == "RECONSTRUCTION_UNVERIFIED"
    _unverified(result)
