"""Static image comparison with separately specified expectations and inert DEBs.

Portable mapper tests deliberately pass caller-created V3 result objects. The
separate Linux integration measures the actual fixture bytes through V3/V4a.
Neither type of result object authenticates the caller or grants acceptance.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from deployment import (
    stage9_observer_apt_v4a as v4a,
    stage9_observer_apt_v4b as v4b,
    stage9_observer_apt_v4c as v4c,
    stage9_observer_image_v5a as image,
    stage9_observer_offline as offline,
    stage9_observer_package_v3 as v3,
)
from deployment.stage9_observer_checkpoint_v4c import SnapshotStateV4c
from deployment.stage9_observer_gpgv_v4b import GpgvEvidenceV4b
from deployment.stage9_observer_offline import Limits
from deployment.stage9_observer_tar_v3 import TarResultV3
from tests.deployment.test_stage9_observer_package_v3 import (
    _tar_fixture,
    build_package_case,
)

pytestmark = [pytest.mark.security, pytest.mark.unit]
_PAYLOAD = b"observer payload\n"
_VERSION = "3.12.3-1ubuntu0.17"
_SNAPSHOT = hashlib.sha256(b"independently selected snapshot").hexdigest()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _raw(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _directory(file_id: str, path: str) -> dict[str, Any]:
    return dict(file_id=file_id, path=path, type="directory", size=0, uid=0, gid=0, mode=0o755)


def _file(file_id: str, path: str, content: bytes = _PAYLOAD) -> dict[str, Any]:
    return dict(
        file_id=file_id,
        path=path,
        type="regular",
        size=len(content),
        uid=0,
        gid=0,
        mode=0o644,
        sha256=_sha(content),
    )


def _source(package: dict[str, Any], source_id: str) -> dict[str, Any]:
    return {
        "source_id": source_id,
        **{
            key: package[key]
            for key in ("package", "version", "architecture", "filename", "sha256", "size")
        },
    }


def _tar_measurement(declaration: dict[str, Any], control: bytes | None) -> TarResultV3:
    records = declaration["records"]
    return TarResultV3(
        unpacked_bytes=declaration["unpacked_bytes"],
        members=len(records),
        regular=sum(record["typeflag"] in {"0", "\0"} for record in records),
        directories=sum(record["typeflag"] == "5" for record in records),
        symlinks=sum(record["typeflag"] == "2" for record in records),
        sha256=declaration["unpacked_sha256"],
        control=control,
        profile=declaration["profile"],
    )


@dataclass
class ImageCase:
    directory: Path
    inventory: dict[str, Any]
    expected: dict[str, Any]
    mapping: dict[str, Any]
    result: v3.PackageInventoryResultV3

    @property
    def inventory_raw(self) -> bytes:
        return _raw(self.inventory)

    @property
    def expected_raw(self) -> bytes:
        return _raw(self.expected)

    def verify(self, **kwargs: Any) -> Any:
        mapping = copy.deepcopy(self.mapping)
        mapping["inventory_sha256"] = _sha(self.inventory_raw)
        mapping["expected_image_sha256"] = _sha(self.expected_raw)
        return image.verify_image_mapping_v5a(
            inventory_raw=kwargs.pop("inventory_raw", self.inventory_raw),
            expected_image_raw=kwargs.pop("expected_image_raw", self.expected_raw),
            mapping_raw=kwargs.pop("mapping_raw", _raw(mapping)),
            expected_image_sha256=kwargs.pop("expected_image_sha256", _sha(self.expected_raw)),
            package_result=kwargs.pop("package_result", self.result),
            **kwargs,
        )


def _build(
    directory: Path,
    *,
    entries: list[tuple[str, str, bytes | str]] | None = None,
    expected_files: list[dict[str, Any]] | None = None,
    second_entries: list[tuple[str, str, bytes | str]] | None = None,
) -> ImageCase:
    # The image expectation is a separate specification. It is never populated
    # by reading records, hashing extracted DEB payloads or observing a host.
    if expected_files is None:
        expected_files = [_directory("root", "/"), _file("value", "/value")]
    if entries is None:
        entries = [("./", "5", b""), ("./value", "0", _PAYLOAD)]
    packages = []
    measurements = []
    sources = []
    for index, (name, data_entries) in enumerate(
        [("python3.12-minimal", entries)]
        + ([] if second_entries is None else [("libpython3.12-minimal", second_entries)])
    ):
        fixture = build_package_case(directory, package=name)
        data_raw, data_declaration = _tar_fixture(data_entries)
        fixture.entries[2] = ("data.tar", data_raw)
        fixture.package["data"] = data_declaration
        fixture.reassemble()
        package = fixture.package
        packages.append(package)
        sources.append(_source(package, f"source-{index}"))
        control = (
            f"Package: {name}\nVersion: {_VERSION}\n"
            "Architecture: amd64\nDescription: synthetic inert V3 fixture\n"
        ).encode()
        measurements.append(
            v3.PackageMeasurementV3(
                name,
                package["sha256"],
                package["size"],
                _tar_measurement(package["control"], control),
                _tar_measurement(package["data"], None),
            )
        )
    inventory = {
        "format": v3.FORMAT,
        "classification": v3.CLASSIFICATION,
        "source_authenticity": "UNVERIFIED",
        "packages": packages,
    }
    expected = dict(
        format="stage9-observer-expected-image-v5a",
        image_id="test-image",
        snapshot_sha256=_SNAPSHOT,
        sources=copy.deepcopy(sources),
        files=copy.deepcopy(expected_files),
    )
    by_path = {item["path"]: item["file_id"] for item in expected_files}
    mapped_entries = []
    # Parsing here is used solely to address frozen source records. Expected
    # image bytes and metadata above do not depend on this parser.
    for source, package in zip(
        sources, v3.parse_package_inventory_v3(_raw(inventory)), strict=True
    ):
        for record in package.data.records:
            path = "/" if record.canonical_name == "." else "/" + record.canonical_name
            mapped_entries.append(
                dict(
                    kind="source_member",
                    source_id=source["source_id"],
                    record_id=image.record_id_v5a(record),
                    member=record.name,
                    file_id=by_path.get(path, "missing-image-entry"),
                )
            )
    mapping = dict(
        format="stage9-observer-image-mapping-v5a",
        snapshot_sha256=_SNAPSHOT,
        inventory_sha256=_sha(_raw(inventory)),
        expected_image_sha256=_sha(_raw(expected)),
        sources=copy.deepcopy(sources),
        directory_policy="identical-explicit-v1",
        entries=mapped_entries,
        transforms=[],
    )
    return ImageCase(
        directory,
        inventory,
        expected,
        mapping,
        v3.PackageInventoryResultV3("MATCH", (), tuple(measurements)),
    )


@pytest.fixture
def case(tmp_path: Path) -> ImageCase:
    return _build(tmp_path / "debs")


def _blocked(result: Any) -> None:
    assert result.external_authority_acceptance == "UNVERIFIED"
    assert result.authenticated_time == "UNVERIFIED"
    assert result.durable_anti_rollback == "BLOCKED"
    assert result.runtime_qualification == "BLOCKED"
    assert result.stage_9 == "IN_PROGRESS"
    assert result.qualification == "BLOCKED"
    assert result.stage_3b == "NOT AUTHORIZED"
    assert result.production == "NO-GO"
    assert result.snapshot_membership == "UNVERIFIED"


def _refused(result: Any, code: str | None = None) -> None:
    assert result.consistency in {"MISMATCH", "INCOMPLETE"}
    assert result.reason_codes
    assert result.mapped_files == ()
    assert result.inventory_sha256 is None
    assert result.expected_image_sha256 is None
    assert result.mapping_sha256 is None
    assert result.snapshot_sha256 is None
    assert result.package_result_sha256 is None
    assert result.apt_evidence_sha256 is None
    assert result.v4_binding == "NOT_PROVIDED"
    if code is not None:
        assert code in result.reason_codes
    _blocked(result)


def test_one_package_one_file_matches_only_static_mapping(case: ImageCase) -> None:
    result = case.verify()
    assert result.consistency == "MATCH", result.reason_codes
    assert not result.reason_codes
    assert len(result.mapped_files) == 2
    _blocked(result)


def test_complete_explicit_directory_tree(tmp_path: Path) -> None:
    fixture = _build(
        tmp_path / "debs",
        entries=[
            ("./", "5", b""),
            ("./usr/", "5", b""),
            ("./usr/lib/", "5", b""),
            ("./usr/lib/value", "0", _PAYLOAD),
            ("./empty/", "5", b""),
        ],
        expected_files=[
            _directory("root", "/"),
            _directory("usr", "/usr"),
            _directory("lib", "/usr/lib"),
            _file("value", "/usr/lib/value"),
            _directory("empty", "/empty"),
        ],
    )
    result = fixture.verify()
    assert result.consistency == "MATCH", result.reason_codes
    assert len(result.mapped_files) == 5
    _blocked(result)


def test_multiple_packages_with_disjoint_files_and_explicit_shared_root(tmp_path: Path) -> None:
    fixture = _build(
        tmp_path / "debs",
        second_entries=[("./", "5", b""), ("./second", "0", b"second payload\n")],
        expected_files=[
            _directory("root", "/"),
            _file("value", "/value"),
            _file("second", "/second", b"second payload\n"),
        ],
    )
    result = fixture.verify()
    assert result.consistency == "MATCH", result.reason_codes
    assert len(result.mapped_files) == 3
    assert len(fixture.mapping["entries"]) == 4
    _blocked(result)


def test_explicit_shared_nonroot_directory_is_not_a_regular_file_collision(tmp_path: Path) -> None:
    fixture = _build(
        tmp_path / "debs",
        entries=[("./", "5", b""), ("./usr/", "5", b""), ("./usr/value", "0", _PAYLOAD)],
        second_entries=[("./", "5", b""), ("./usr/", "5", b""), ("./usr/other", "0", b"other")],
        expected_files=[
            _directory("root", "/"),
            _directory("usr", "/usr"),
            _file("value", "/usr/value"),
            _file("other", "/usr/other", b"other"),
        ],
    )
    assert fixture.verify().consistency == "MATCH"


@pytest.mark.parametrize(
    "field,value", [("sha256", "1" * 64), ("size", 99), ("uid", 12), ("gid", 12), ("mode", 0o600)]
)
def test_regular_file_metadata_is_compared_exactly(case: ImageCase, field: str, value: Any) -> None:
    case.expected["files"][1][field] = value
    result = case.verify()
    assert result.consistency == "MISMATCH", result.reason_codes
    _refused(result)


@pytest.mark.parametrize("field,value", [("uid", 1), ("gid", 2), ("mode", 0o700)])
def test_directory_metadata_cannot_be_silently_fixed(
    case: ImageCase, field: str, value: int
) -> None:
    case.expected["files"][0][field] = value
    result = case.verify()
    _refused(result, "ARTIFACT_MISMATCH")
    assert result.unsupported_transformations == ("metadata_change",)


def test_all_local_hashes_match_but_independent_expected_content_differs(case: ImageCase) -> None:
    assert case.result.consistency == "MATCH"
    case.expected["files"][1] = _file("value", "/value", b"reviewer selected different content\n")
    _refused(case.verify(), "ARTIFACT_MISMATCH")


def test_expected_image_replacement_fails_out_of_band_pin(case: ImageCase) -> None:
    pinned = _sha(case.expected_raw)
    case.expected["image_id"] = "substituted-image"
    _refused(case.verify(expected_image_sha256=pinned))


@pytest.mark.parametrize(
    "field,value",
    [
        ("member", "value"),
        ("record_id", "1" * 64),
        ("source_id", "unknown"),
        ("file_id", "unknown"),
    ],
)
def test_mapping_requires_exact_literal_record_and_unique_known_references(
    case: ImageCase, field: str, value: str
) -> None:
    case.mapping["entries"][1][field] = value
    _refused(case.verify())


@pytest.mark.parametrize(
    "field,value",
    [
        ("package", "libpython3.12-minimal"),
        ("version", "3.12.3-1ubuntu0.18"),
        ("architecture", "arm64"),
        ("filename", "other.deb"),
        ("sha256", "1" * 64),
        ("size", 999),
    ],
)
def test_full_package_source_identity_is_required(case: ImageCase, field: str, value: Any) -> None:
    case.mapping["sources"][0][field] = value
    _refused(case.verify())


@pytest.mark.parametrize("which", ["file", "directory", "mapping-entry", "source"])
def test_missing_image_or_mapping_members_never_match(case: ImageCase, which: str) -> None:
    if which == "file":
        case.expected["files"].pop()
    elif which == "directory":
        case.expected["files"].pop(0)
    elif which == "mapping-entry":
        case.mapping["entries"].pop()
    else:
        case.mapping["sources"].clear()
    _refused(case.verify())


def test_extra_expected_file_is_not_implicitly_created(case: ImageCase) -> None:
    case.expected["files"].append(_file("hidden", "/.hidden"))
    _refused(case.verify())


def test_extra_archive_file_cannot_be_omitted(tmp_path: Path) -> None:
    fixture = _build(
        tmp_path / "debs",
        entries=[("./", "5", b""), ("./value", "0", _PAYLOAD), ("./.hidden", "0", b"hidden")],
    )
    fixture.mapping["entries"].pop()
    _refused(fixture.verify())


def test_undeclared_package_source_is_not_ignored(case: ImageCase) -> None:
    extra = copy.deepcopy(case.mapping["sources"][0])
    extra["source_id"] = "unexplained-source"
    case.mapping["sources"].append(extra)
    _refused(case.verify())


def test_duplicate_mapping_is_ambiguous_provenance(case: ImageCase) -> None:
    case.mapping["entries"].append(copy.deepcopy(case.mapping["entries"][1]))
    _refused(case.verify())


def test_two_package_regular_file_collision_is_never_merged(tmp_path: Path) -> None:
    fixture = _build(
        tmp_path / "debs", second_entries=[("./", "5", b""), ("./value", "0", _PAYLOAD)]
    )
    _refused(fixture.verify())


def test_shared_directory_requires_explicit_supported_policy(case: ImageCase) -> None:
    case.mapping["directory_policy"] = "merge-whatever"
    _refused(case.verify())


def test_conflicting_shared_directory_metadata_is_refused(tmp_path: Path) -> None:
    fixture = _build(
        tmp_path / "debs",
        second_entries=[("./", "5", b""), ("./second", "0", b"second")],
        expected_files=[
            _directory("root", "/"),
            _file("value", "/value"),
            _file("second", "/second", b"second"),
        ],
    )
    fixture.inventory["packages"][1]["data"]["records"][0]["uid"] = 3
    _refused(fixture.verify())


@pytest.mark.parametrize(
    "path", ["/../value", "/./value", "//value", "/value/", "value", "/a\\value"]
)
def test_target_path_aliases_and_traversal_are_refused(case: ImageCase, path: str) -> None:
    case.expected["files"][1]["path"] = path
    _refused(case.verify())


@pytest.mark.parametrize("name", ["../value", "/value", "./usr/../value", "./usr//value"])
def test_archive_traversal_absolute_and_alias_paths_are_refused(case: ImageCase, name: str) -> None:
    case.inventory["packages"][0]["data"]["records"][1]["name"] = name
    _refused(case.verify())


def test_direct_mapping_cannot_relocate_a_member(case: ImageCase) -> None:
    case.expected["files"][1]["path"] = "/relocated"
    result = case.verify()
    _refused(result, "ARTIFACT_MISMATCH")
    assert result.unsupported_transformations == ("relocation",)


def test_regular_file_cannot_be_parent(case: ImageCase) -> None:
    case.expected["files"].append(_file("child", "/value/child"))
    _refused(case.verify())


@pytest.mark.parametrize("target", ["/outside", "../outside", "second", "link-b"])
def test_literal_symlinks_are_never_materialized_or_followed(tmp_path: Path, target: str) -> None:
    fixture = _build(
        tmp_path / "debs",
        entries=[
            ("./", "5", b""),
            ("./value", "0", _PAYLOAD),
            ("./link-a", "2", target),
            ("./link-b", "2", "link-a"),
        ],
        expected_files=[
            _directory("root", "/"),
            _file("value", "/value"),
            _file("link-a", "/link-a"),
            _file("link-b", "/link-b"),
        ],
    )
    _refused(fixture.verify())


def test_cross_package_symlink_cannot_borrow_other_package_content(tmp_path: Path) -> None:
    fixture = _build(
        tmp_path / "debs",
        entries=[("./", "5", b""), ("./value", "0", _PAYLOAD), ("./link", "2", "second")],
        second_entries=[("./", "5", b""), ("./second", "0", _PAYLOAD)],
        expected_files=[
            _directory("root", "/"),
            _file("value", "/value"),
            _file("link", "/link"),
            _file("second", "/second"),
        ],
    )
    _refused(fixture.verify())


def test_file_beneath_archive_symlink_is_refused(case: ImageCase) -> None:
    records = case.inventory["packages"][0]["data"]["records"]
    records[0].update(name="./value/", typeflag="2", linkname="/outside")
    records[1]["name"] = "./value/child"
    _refused(case.verify())


@pytest.mark.parametrize("kind", ["materialize_member", "pyc"])
def test_transform_outputs_are_structurally_bound_but_unsupported(
    case: ImageCase, kind: str
) -> None:
    original = case.mapping["entries"][1]
    case.mapping["transforms"] = [
        dict(
            transform_id="transform-1",
            kind=kind,
            inputs=[{key: original[key] for key in ("source_id", "record_id", "member")}],
            output_file_id="value",
        )
    ]
    case.mapping["entries"][1] = dict(
        kind="transform_output", transform_id="transform-1", file_id="value"
    )
    _refused(case.verify())


def test_unexplained_transform_output_is_refused(case: ImageCase) -> None:
    case.mapping["entries"][1] = dict(
        kind="transform_output", transform_id="missing-transform", file_id="value"
    )
    _refused(case.verify())


def test_transform_with_ambiguous_output_is_refused(case: ImageCase) -> None:
    original = case.mapping["entries"][1]
    transform = dict(
        transform_id="transform-1",
        kind="materialize_member",
        inputs=[{key: original[key] for key in ("source_id", "record_id", "member")}],
        output_file_id="value",
    )
    case.mapping["transforms"] = [transform, {**transform, "transform_id": "transform-2"}]
    _refused(case.verify())


def test_snapshot_mixing_is_refused(case: ImageCase) -> None:
    case.mapping["snapshot_sha256"] = _sha(b"unrelated snapshot")
    _refused(case.verify())


def test_missing_or_negative_measurement_is_not_complete(case: ImageCase) -> None:
    _refused(case.verify(package_result=None))
    _refused(
        case.verify(package_result=v3.PackageInventoryResultV3("MISMATCH", ("ARTIFACT_MISMATCH",)))
    )


@pytest.mark.parametrize(
    "part",
    ["archive_hash", "archive_size", "data_hash", "data_count", "control_bytes", "control_profile"],
)
def test_v3_measurements_must_bind_full_frozen_record_set(case: ImageCase, part: str) -> None:
    measured = case.result.packages[0]
    if part == "archive_hash":
        measured = replace(measured, archive_sha256="1" * 64)
    elif part == "archive_size":
        measured = replace(measured, archive_bytes=measured.archive_bytes + 1)
    elif part == "data_hash":
        measured = replace(measured, data=replace(measured.data, sha256="1" * 64))
    elif part == "data_count":
        measured = replace(measured, data=replace(measured.data, members=999))
    elif part == "control_bytes":
        measured = replace(measured, control=replace(measured.control, control=b"substituted\n"))
    else:
        measured = replace(measured, control=replace(measured.control, profile="other"))
    _refused(case.verify(package_result=replace(case.result, packages=(measured,))))


def _evidence(case: ImageCase) -> Any:
    declaration = v4a.AptDeclarationV4a(
        "Release", _SNAPSHOT, 100, "Packages", _sha(b"index"), 200, "Packages"
    )
    return image.AptEvidenceV5a(
        _sha(case.inventory_raw),
        _SNAPSHOT,
        declaration,
        v4a.AptResultV4a("MATCH", (), case.result.packages),
    )


def test_corresponding_v4a_measurement_is_bound_without_authenticating_it(case: ImageCase) -> None:
    evidence = _evidence(case)
    result = case.verify(apt_evidence=evidence)
    assert result.consistency == "MATCH", result.reason_codes
    _blocked(result)
    assert evidence.result.source_authenticity == "UNVERIFIED"
    assert evidence.result.acceptance == "UNVERIFIED"


def _signature_evidence(case: ImageCase) -> Any:
    evidence = _evidence(case)
    context = v4b.TrustContextV4b(
        "TEST-context",
        1,
        (v4b.SignerRuleV4b("1" * 40, "2" * 40, 0, 1000),),
        ((22, 8),),
        "EXACT_SIGNING_PRIMARY_PAIRS",
        "TEST-repository",
        "TEST-snapshot",
        _SNAPSHOT,
        _sha(b"TEST-keyring"),
        0,
        1000,
        False,
        v4b.TimeRulesV4b(100, 300, 0),
        v4b.ExternalAuthorityReferenceV4b("TEST-authority", "TEST-reference", "TEST-delivery"),
    )
    selection = {
        "repository_id": "TEST-repository",
        "snapshot_id": "TEST-snapshot",
        "context_sha256": v4c.context_sha256_v4c(context),
    }
    case.expected["apt_context"] = copy.deepcopy(selection)
    case.mapping["apt_context"] = copy.deepcopy(selection)
    signature = v4b.AptResultV4b(
        "VALID",
        "MATCH",
        (
            "EXTERNAL_AUTHORITY_UNVERIFIED",
            "FRESHNESS_BLOCKED",
            "CHECKPOINT_BLOCKED",
            "ROLLBACK_BLOCKED",
            "TOOL_ACCEPTANCE_UNVERIFIED",
            "SYSTEM_ISOLATION_UNVERIFIED",
        ),
        GpgvEvidenceV4b("1" * 40, "2" * 40, 22, 8, 100, 0),
        evidence.result,
        _SNAPSHOT,
    )
    return replace(evidence, signature_result=signature, trust_context=context)


def _acceptance_evidence(case: ImageCase) -> Any:
    evidence = _signature_evidence(case)
    state = SnapshotStateV4c(
        "TEST-repository",
        "TEST-checkpoint-source",
        "TEST-checkpoint",
        1,
        _SNAPSHOT,
        v4c.context_sha256_v4c(evidence.trust_context),
        _sha(b"checkpoint"),
    )
    return replace(evidence, acceptance_result=v4c.AptResultV4c("VALID", v4c._BLOCKERS, state))


def test_v4b_and_v4c_claims_remain_unverified_and_unchanged_after_match(case: ImageCase) -> None:
    evidence = _signature_evidence(case)
    state = SnapshotStateV4c(
        "TEST-repository",
        "TEST-checkpoint-source",
        "TEST-checkpoint",
        1,
        _SNAPSHOT,
        v4c.context_sha256_v4c(evidence.trust_context),
        _sha(b"checkpoint"),
    )
    # Real V4c technical success retains these four blockers in reason_codes;
    # an empty tuple is not the successful V4c wire/API contract.
    acceptance = v4c.AptResultV4c(
        "VALID",
        (
            "EXTERNAL_AUTHORITY_UNVERIFIED",
            "AUTHENTICATED_TIME_UNVERIFIED",
            "DURABLE_ANTI_ROLLBACK_BLOCKED",
            "RUNTIME_QUALIFICATION_BLOCKED",
        ),
        state,
    )
    evidence = replace(evidence, acceptance_result=acceptance)
    before = copy.deepcopy(evidence)
    result = case.verify(apt_evidence=evidence)
    assert result.consistency == "MATCH", result.reason_codes
    assert evidence == before
    assert evidence.signature_result.external_authority_acceptance == "UNVERIFIED"
    assert evidence.signature_result.freshness == "BLOCKED"
    assert evidence.acceptance_result.external_authority_acceptance == "UNVERIFIED"
    assert evidence.acceptance_result.authenticated_time == "UNVERIFIED"
    assert evidence.acceptance_result.durable_anti_rollback == "BLOCKED"
    _blocked(result)


@pytest.mark.parametrize("part", ["snapshot", "result", "crypto", "signer", "missing-blockers"])
def test_unrelated_or_failed_v4b_claim_cannot_be_rebound(case: ImageCase, part: str) -> None:
    evidence = _signature_evidence(case)
    signature = evidence.signature_result
    if part == "snapshot":
        signature = replace(signature, release_sha256="1" * 64)
    elif part == "result":
        signature = replace(signature, v4a=replace(evidence.result, packages=()))
    elif part == "crypto":
        signature = replace(signature, cryptographic_signature="UNVERIFIED")
    elif part == "signer":
        signature = replace(signature, signer_policy="UNVERIFIED")
    else:
        signature = replace(signature, reason_codes=())
    _refused(case.verify(apt_evidence=replace(evidence, signature_result=signature)))


@pytest.mark.parametrize(
    "part", ["snapshot", "missing-state", "status", "missing-blockers", "missing-signature"]
)
def test_invalid_or_unrelated_v4c_claim_never_bypasses_existing_boundary(
    case: ImageCase, part: str
) -> None:
    evidence = _signature_evidence(case)
    state = SnapshotStateV4c(
        "TEST-repository",
        "TEST-source",
        "TEST-checkpoint",
        1,
        _SNAPSHOT,
        v4c.context_sha256_v4c(evidence.trust_context),
        _sha(b"checkpoint"),
    )
    acceptance = v4c.AptResultV4c("VALID", v4c._BLOCKERS, state)
    if part == "snapshot":
        acceptance = replace(acceptance, proposed_state=replace(state, snapshot_sha256="1" * 64))
    elif part == "missing-state":
        acceptance = replace(acceptance, proposed_state=None)
    elif part == "status":
        acceptance = replace(acceptance, technical_validation="REFUSED")
    elif part == "missing-blockers":
        acceptance = replace(acceptance, reason_codes=())
    else:
        evidence = replace(evidence, signature_result=None)
    _refused(case.verify(apt_evidence=replace(evidence, acceptance_result=acceptance)))


def test_v4b_context_matches_independent_selection_without_granting_acceptance(
    case: ImageCase,
) -> None:
    evidence = _signature_evidence(case)
    before = copy.deepcopy(evidence)
    result = case.verify(apt_evidence=evidence)
    assert result.consistency == "MATCH", result.reason_codes
    assert result.v4_binding == "CALLER_CLAIM_BOUND"
    assert evidence == before
    _blocked(result)


@pytest.mark.parametrize(
    "field,value",
    [
        ("repository_id", "TEST-other-repository"),
        ("context_sha256", _sha(b"other context with same snapshot")),
    ],
)
def test_same_release_cannot_rebind_v4c_to_other_repository_or_context(
    case: ImageCase, field: str, value: str
) -> None:
    evidence = _acceptance_evidence(case)
    baseline = case.verify(apt_evidence=evidence)
    assert baseline.consistency == "MATCH", baseline.reason_codes
    original_state = evidence.acceptance_result.proposed_state
    swapped = replace(original_state, **{field: value})
    acceptance = replace(evidence.acceptance_result, proposed_state=swapped)
    before = copy.deepcopy(evidence)
    result = case.verify(apt_evidence=replace(evidence, acceptance_result=acceptance))
    assert result.consistency == "MISMATCH"
    _refused(
        result,
        "REPOSITORY_MISMATCH" if field == "repository_id" else "CONTEXT_BINDING_REJECTED",
    )
    assert evidence == before


@pytest.mark.parametrize(
    "part",
    [
        "repository_id",
        "snapshot_id",
        "context_id",
        "keyring_sha256",
        "time_rules",
        "external_authority",
    ],
)
def test_substituted_v4b_context_cannot_reuse_same_release_hash(case: ImageCase, part: str) -> None:
    evidence = _signature_evidence(case)
    context = evidence.trust_context
    substitutions = {
        "repository_id": "TEST-other-repository",
        "snapshot_id": "TEST-other-snapshot",
        "context_id": "TEST-other-context",
        "keyring_sha256": _sha(b"other keyring"),
        "time_rules": replace(context.time_rules, maximum_signature_age_seconds=301),
        "external_authority": v4b.ExternalAuthorityReferenceV4b(
            "TEST-other-authority", "TEST-reference", "TEST-delivery"
        ),
    }
    changed = replace(context, **{part: substitutions[part]})
    _refused(case.verify(apt_evidence=replace(evidence, trust_context=changed)))


@pytest.mark.parametrize("part", ["repository_id", "keyring_sha256"])
def test_simultaneous_context_and_v4c_state_swap_cannot_replace_independent_selection(
    case: ImageCase, part: str
) -> None:
    evidence = _acceptance_evidence(case)
    context = replace(
        evidence.trust_context,
        **{part: "TEST-other-repository" if part == "repository_id" else _sha(b"other keyring")},
    )
    state = replace(
        evidence.acceptance_result.proposed_state,
        repository_id=context.repository_id,
        context_sha256=v4c.context_sha256_v4c(context),
    )
    changed = replace(
        evidence,
        trust_context=context,
        acceptance_result=replace(evidence.acceptance_result, proposed_state=state),
    )
    _refused(case.verify(apt_evidence=changed))


@pytest.mark.parametrize(
    "part",
    [
        "missing-context",
        "missing-expected-pin",
        "missing-mapping-pin",
        "missing-both-pins",
        "missing-evidence",
    ],
)
def test_stronger_v4_claims_require_full_context_and_both_document_pins(
    case: ImageCase, part: str
) -> None:
    evidence = _acceptance_evidence(case)
    if part == "missing-context":
        evidence = replace(evidence, trust_context=None)
    elif part == "missing-expected-pin":
        case.expected.pop("apt_context")
    elif part == "missing-mapping-pin":
        case.mapping.pop("apt_context")
    elif part == "missing-both-pins":
        case.expected.pop("apt_context")
        case.mapping.pop("apt_context")
    else:
        evidence = None
    _refused(case.verify(apt_evidence=evidence))


@pytest.mark.parametrize("document", ["expected", "mapping"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("repository_id", "TEST-other-repository"),
        ("snapshot_id", "TEST-other-snapshot"),
        ("context_sha256", "1" * 64),
    ],
)
def test_mixed_independent_and_mapping_context_selections_are_refused(
    case: ImageCase, document: str, field: str, value: str
) -> None:
    evidence = _signature_evidence(case)
    declaration = case.expected if document == "expected" else case.mapping
    declaration["apt_context"][field] = value
    _refused(case.verify(apt_evidence=evidence))


@pytest.mark.parametrize(
    "selection",
    [
        None,
        {},
        {"repository_id": "TEST-repository"},
        {"repository_id": True, "snapshot_id": "TEST-snapshot", "context_sha256": "1" * 64},
        {
            "repository_id": "TEST-repository",
            "snapshot_id": "TEST-snapshot",
            "context_sha256": "1" * 64,
            "unknown": "field",
        },
    ],
)
def test_context_selection_has_closed_fields_and_strict_types(
    case: ImageCase, selection: Any
) -> None:
    evidence = _signature_evidence(case)
    case.expected["apt_context"] = copy.deepcopy(selection)
    case.mapping["apt_context"] = copy.deepcopy(selection)
    _refused(case.verify(apt_evidence=evidence))


def test_context_pin_substitution_cannot_replace_expected_image_byte_pin(case: ImageCase) -> None:
    evidence = _acceptance_evidence(case)
    original_pin = _sha(case.expected_raw)
    context = replace(evidence.trust_context, context_id="TEST-substituted-context")
    digest = v4c.context_sha256_v4c(context)
    case.expected["apt_context"]["context_sha256"] = digest
    case.mapping["apt_context"]["context_sha256"] = digest
    state = replace(evidence.acceptance_result.proposed_state, context_sha256=digest)
    changed = replace(
        evidence,
        trust_context=context,
        acceptance_result=replace(evidence.acceptance_result, proposed_state=state),
    )
    _refused(
        case.verify(apt_evidence=changed, expected_image_sha256=original_pin), "ARTIFACT_MISMATCH"
    )


@pytest.mark.parametrize("part", ["signer", "algorithm", "revoked-signer", "signature-time"])
def test_bound_context_rechecks_signer_policy_instead_of_accepting_status_claim(
    case: ImageCase, part: str
) -> None:
    evidence = _signature_evidence(case)
    context = evidence.trust_context
    signature = evidence.signature_result
    if part == "signer":
        signature = replace(
            signature, evidence=replace(signature.evidence, signing_fingerprint="3" * 40)
        )
    elif part == "algorithm":
        signature = replace(signature, evidence=replace(signature.evidence, public_key_algorithm=1))
    elif part == "revoked-signer":
        context = replace(
            context, allowed_signers=(replace(context.allowed_signers[0], revoked=True),)
        )
        digest = v4c.context_sha256_v4c(context)
        case.expected["apt_context"]["context_sha256"] = digest
        case.mapping["apt_context"]["context_sha256"] = digest
    else:
        signature = replace(
            signature, evidence=replace(signature.evidence, signature_timestamp=999)
        )
    _refused(
        case.verify(
            apt_evidence=replace(evidence, trust_context=context, signature_result=signature)
        )
    )


def test_oversized_context_signer_tuple_is_refused_with_no_partial_binding(case: ImageCase) -> None:
    evidence = _signature_evidence(case)
    context = replace(
        evidence.trust_context, allowed_signers=evidence.trust_context.allowed_signers * 8193
    )
    _refused(case.verify(apt_evidence=replace(evidence, trust_context=context)))


def test_context_processing_uses_same_deadline_and_discards_completed_mapping(
    case: ImageCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _acceptance_evidence(case)

    def exhausted(context: Any) -> str:
        raise offline.contract.ContractRefusal("LIMIT_EXCEEDED")

    monkeypatch.setattr(v4c, "context_sha256_v4c", exhausted)
    _refused(case.verify(apt_evidence=evidence), "LIMIT_EXCEEDED")


def test_arbitrary_nested_dataclass_is_not_serialized_as_evidence(case: ImageCase) -> None:
    @dataclass
    class UnrelatedClaim:
        content: str

    evidence = _signature_evidence(case)
    unrelated: Any = UnrelatedClaim("caller-controlled")
    signature = replace(
        evidence.signature_result,
        evidence=GpgvEvidenceV4b(unrelated, "2" * 40, 22, 8, 100, 0),
    )
    _refused(case.verify(apt_evidence=replace(evidence, signature_result=signature)))


def test_oversized_nested_claim_is_bounded_before_serialization(case: ImageCase) -> None:
    evidence = _signature_evidence(case)
    signature = replace(
        evidence.signature_result,
        evidence=GpgvEvidenceV4b("x" * 65537, "2" * 40, 22, 8, 100, 0),
    )
    _refused(case.verify(apt_evidence=replace(evidence, signature_result=signature)))


@pytest.mark.parametrize(
    "field,value",
    [
        ("signing_fingerprint", ""),
        ("signing_fingerprint", "a" * 40),
        ("primary_fingerprint", ""),
        ("public_key_algorithm", False),
        ("public_key_algorithm", 17),
        ("hash_algorithm", False),
        ("hash_algorithm", 1),
        ("signature_timestamp", False),
        ("signature_timestamp", 0),
        ("signature_expiry", False),
        ("signature_expiry", 99),
    ],
)
def test_invalid_gpgv_fields_cannot_be_imported_as_valid_claims(
    case: ImageCase, field: str, value: Any
) -> None:
    evidence = _signature_evidence(case)
    signature = replace(
        evidence.signature_result,
        evidence=replace(evidence.signature_result.evidence, **{field: value}),
    )
    _refused(case.verify(apt_evidence=replace(evidence, signature_result=signature)))


def test_forged_empty_gpgv_evidence_does_not_gain_success_from_boolean_coercion(
    case: ImageCase,
) -> None:
    evidence = _signature_evidence(case)
    signature = replace(
        evidence.signature_result,
        evidence=GpgvEvidenceV4b("", "", False, False, False, False),
    )
    _refused(case.verify(apt_evidence=replace(evidence, signature_result=signature)))


@pytest.mark.parametrize("part", ["inventory", "snapshot", "release", "measurements", "status"])
def test_v4a_from_another_snapshot_or_inventory_is_refused(case: ImageCase, part: str) -> None:
    evidence = _evidence(case)
    if part == "inventory":
        evidence = replace(evidence, inventory_sha256="1" * 64)
    elif part == "snapshot":
        evidence = replace(evidence, snapshot_sha256="1" * 64)
    elif part == "release":
        evidence = replace(
            evidence, declaration=replace(evidence.declaration, release_sha256="1" * 64)
        )
    elif part == "measurements":
        evidence = replace(evidence, result=replace(evidence.result, packages=()))
    else:
        evidence = replace(evidence, result=replace(evidence.result, consistency="INCOMPLETE"))
    _refused(case.verify(apt_evidence=evidence))


@pytest.mark.parametrize(
    "changes",
    [
        {"max_archive_members": 4},
        {"max_files": 1},
        {"working_memory_bytes": 262144},
        {"max_total_bytes": 512},
        {"max_path_depth": 1},
    ],
)
def test_resource_limits_refuse_without_partial_success(
    case: ImageCase, changes: dict[str, int]
) -> None:
    if "max_path_depth" in changes:
        case.expected["files"][1]["path"] = "/deep/path"
    _refused(case.verify(limits=replace(Limits(), **changes)), "LIMIT_EXCEEDED")


def test_timeout_after_some_mapping_discards_partial_success(
    case: ImageCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = offline._Budget.check
    calls = 0

    def fail_late(self: Any) -> None:
        nonlocal calls
        calls += 1
        if calls >= 30:
            raise offline.contract.ContractRefusal("LIMIT_EXCEEDED")
        original(self)

    monkeypatch.setattr(offline._Budget, "check", fail_late)
    _refused(case.verify(), "LIMIT_EXCEEDED")


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, InterruptedError])
def test_interruption_discards_all_mapping(
    case: ImageCase, monkeypatch: pytest.MonkeyPatch, interruption: type[BaseException]
) -> None:
    def interrupt(self: Any) -> None:
        raise interruption

    monkeypatch.setattr(offline._Budget, "check", interrupt)
    _refused(case.verify())


@pytest.mark.parametrize(
    "raw",
    [
        b"{}",
        b'{"format":1,"format":2}',
        b"[" * 50 + b"]" * 50,
        b'{"format":NaN}',
        b'{"format":true}',
        b'{"format":null}',
    ],
)
def test_malformed_bounded_declarations_fail_closed(case: ImageCase, raw: bytes) -> None:
    _refused(case.verify(mapping_raw=raw))


def test_mapping_has_no_filesystem_network_install_or_runtime_operations(
    case: ImageCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Static mapping attempted host I/O or execution")

    monkeypatch.setattr(os, "open", forbidden)
    monkeypatch.setattr(os, "readlink", forbidden)
    monkeypatch.setattr(os, "symlink", forbidden)
    monkeypatch.setattr(Path, "resolve", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    result = case.verify()
    assert result.consistency == "MATCH", result.reason_codes
    _blocked(result)


def test_linux_real_v3_and_v4a_pipeline_uses_binary_portable_fixtures(case: ImageCase) -> None:
    if sys.platform != "linux":
        pytest.skip(
            "Public V3/V4a descriptor measurement supports Linux only; mapper tests remain portable"
        )
    index = b"\n".join(
        (
            f"Package: {package['package']}\nVersion: {package['version']}\nArchitecture: {package['architecture']}\nFilename: {package['filename']}\nSHA256: {package['sha256']}\nSize: {package['size']}\n"
        ).encode()
        for package in case.inventory["packages"]
    )
    release = f"SHA256:\n {_sha(index)} {len(index)} Packages\n".encode()
    declaration = v4a.AptDeclarationV4a(
        "Release", _sha(release), len(release), "Packages", _sha(index), len(index), "Packages"
    )
    # V3 intentionally sees only its declared flat DEB tree.
    measured_v3 = v3.verify_ubuntu_package_inventory_v3(
        case.directory, inventory_raw=case.inventory_raw
    )
    assert measured_v3.consistency == "MATCH", measured_v3.reason_codes
    (case.directory / "Packages").write_bytes(index)
    (case.directory / "Release").write_bytes(release)
    measured_v4a = v4a.verify_apt_provenance_v4a(
        case.directory, declaration=declaration, inventory_raw=case.inventory_raw
    )
    assert measured_v4a.consistency == "MATCH", measured_v4a.reason_codes
    case.expected["snapshot_sha256"] = _sha(release)
    case.mapping["snapshot_sha256"] = _sha(release)
    evidence = image.AptEvidenceV5a(
        _sha(case.inventory_raw), _sha(release), declaration, measured_v4a
    )
    result = case.verify(package_result=measured_v3, apt_evidence=evidence)
    assert result.consistency == "MATCH", result.reason_codes
    _blocked(result)


def test_static_declarations_can_exceed_the_runtime_document_cap(tmp_path: Path) -> None:
    names = [f"file-{index:04d}-payload" for index in range(400)]
    fixture = _build(
        tmp_path / "debs",
        entries=[("./", "5", b"")] + [("./" + name, "0", _PAYLOAD) for name in names],
        expected_files=[_directory("root", "/")] + [_file(name, "/" + name) for name in names],
    )
    assert len(fixture.expected_raw) > 65536
    result = fixture.verify()
    assert result.consistency == "MATCH", result.reason_codes
    assert len(result.mapped_files) == 401


@pytest.mark.parametrize("field,value", [("name", "x" * 4097), ("uid", True), ("uid", 2**53)])
def test_standalone_record_id_helper_bounds_unparsed_records(
    case: ImageCase, field: str, value: Any
) -> None:
    record = v3.parse_package_inventory_v3(case.inventory_raw)[0].data.records[1]
    with pytest.raises(ValueError):
        image.record_id_v5a(replace(record, **{field: value}))


@pytest.mark.parametrize("failure", [KeyboardInterrupt, MemoryError, ValueError])
def test_failure_after_complete_staging_discards_every_positive_binding(
    case: ImageCase, monkeypatch: pytest.MonkeyPatch, failure: type[BaseException]
) -> None:
    staged = False
    original = image._map

    def capture(*args: Any, **kwargs: Any) -> Any:
        nonlocal staged
        result = original(*args, **kwargs)
        assert len(result) == 2
        staged = True
        return result

    def fail(*args: Any, **kwargs: Any) -> Any:
        assert staged
        raise failure

    monkeypatch.setattr(image, "_map", capture)
    monkeypatch.setattr(image, "_claim_digest", fail)
    _refused(case.verify())
    assert staged
