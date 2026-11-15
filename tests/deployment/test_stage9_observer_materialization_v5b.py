"""Real inert DEBs and independent expected content for bounded V5b materialization.

Portable cases feed immutable DEB bytes to the actual byte adapter. The separate
Linux integration additionally invokes the public V3 and V4a descriptor readers.
No case installs a package, follows a host link, or executes archive contents.
"""

from __future__ import annotations

import copy
import hashlib
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
    stage9_observer_image_v5a as image,
    stage9_observer_materialization_v5b as materialization,
    stage9_observer_offline as offline,
    stage9_observer_package_v3 as v3,
)
from deployment.stage9_observer_member_source_v5b import FrozenPackageV5b
from deployment.stage9_observer_offline import Limits
from tests.deployment.test_stage9_observer_image_v5a import (
    ImageCase,
    _acceptance_evidence,
    _blocked,
    _build,
    _directory,
    _file,
    _raw,
    _signature_evidence,
)

pytestmark = [pytest.mark.security, pytest.mark.unit]
_PAYLOAD = b"observer payload\n"
_POLICY = "source-record-exact-v1"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass
class MaterializationCase:
    base: ImageCase
    expected: dict[str, Any]
    mapping: dict[str, Any]
    archives: tuple[FrozenPackageV5b, ...]

    @property
    def inventory(self) -> dict[str, Any]:
        return self.base.inventory

    @property
    def inventory_raw(self) -> bytes:
        return self.base.inventory_raw

    @property
    def expected_raw(self) -> bytes:
        return _raw(self.expected)

    def ref(self, member: str, source_id: str = "source-0") -> dict[str, str]:
        package = v3.parse_package_inventory_v3(self.inventory_raw)[int(source_id.split("-")[-1])]
        record = next(record for record in package.data.records if record.name == member)
        return dict(source_id=source_id, member=member, record_id=image.record_id_v5a(record))

    def verify(self, **kwargs: Any) -> Any:
        mapping = copy.deepcopy(self.mapping)
        mapping["inventory_sha256"] = _sha(self.inventory_raw)
        mapping["expected_image_sha256"] = _sha(self.expected_raw)
        return materialization.verify_member_materialization_v5b(
            inventory_raw=kwargs.pop("inventory_raw", self.inventory_raw),
            expected_image_raw=kwargs.pop("expected_image_raw", self.expected_raw),
            mapping_raw=kwargs.pop("mapping_raw", _raw(mapping)),
            expected_image_sha256=kwargs.pop("expected_image_sha256", _sha(self.expected_raw)),
            archives=kwargs.pop("archives", self.archives),
            package_result=kwargs.pop("package_result", self.base.result),
            **kwargs,
        )


def _build_case(
    directory: Path,
    *,
    entries: list[tuple[str, str, bytes | str]] | None = None,
    expected_files: list[dict[str, Any]] | None = None,
    second_entries: list[tuple[str, str, bytes | str]] | None = None,
    links: dict[str, tuple[str, str]] | None = None,
) -> MaterializationCase:
    if entries is None:
        entries = [("./", "5", b""), ("./value", "0", _PAYLOAD), ("./link", "2", "value")]
    if expected_files is None:
        # Content hashes and target metadata are selected independently of
        # archive parsing and of the materialization result.
        expected_files = [
            _directory("root", "/"),
            _file("value", "/value", _PAYLOAD),
            _file("link", "/link", _PAYLOAD),
        ]
    base = _build(
        directory,
        entries=entries,
        expected_files=expected_files,
        second_entries=second_entries,
    )
    expected = copy.deepcopy(base.expected)
    expected["format"] = "stage9-observer-expected-image-v5b"
    expected["materializations"] = []
    mapping = copy.deepcopy(base.mapping)
    mapping["format"] = "stage9-observer-materialization-v5b"
    archives = tuple(
        FrozenPackageV5b(package["filename"], (directory / package["filename"]).read_bytes())
        for package in base.inventory["packages"]
    )
    case = MaterializationCase(base, expected, mapping, archives)
    parsed = v3.parse_package_inventory_v3(case.inventory_raw)
    symlinks = {
        (f"source-{index}", record.name)
        for index, package in enumerate(parsed)
        for record in package.data.records
        if record.typeflag == "2"
    }
    direct = []
    for entry in mapping["entries"]:
        if (entry["source_id"], entry["member"]) not in symlinks:
            direct.append(entry)
            continue
        link = {key: entry[key] for key in ("source_id", "record_id", "member")}
        source_id, member = (links or {}).get(entry["member"], ("source-0", "./value"))
        source = case.ref(member, source_id)
        transform_id = "materialize-" + entry["file_id"]
        mapping["transforms"].append(
            dict(
                transform_id=transform_id,
                kind="symlink_to_regular_v5b",
                link=copy.deepcopy(link),
                source=copy.deepcopy(source),
                output_file_id=entry["file_id"],
                metadata_policy=_POLICY,
            )
        )
        expected["materializations"].append(
            dict(
                file_id=entry["file_id"],
                link=copy.deepcopy(link),
                source=copy.deepcopy(source),
                metadata_policy=_POLICY,
            )
        )
        direct.append(
            dict(kind="transform_output", transform_id=transform_id, file_id=entry["file_id"])
        )
    mapping["entries"] = direct
    return case


@pytest.fixture
def case(tmp_path: Path) -> MaterializationCase:
    return _build_case(tmp_path / "debs")


def _refused(result: Any, code: str | None = None) -> None:
    assert result.consistency in {"MISMATCH", "INCOMPLETE"}
    assert result.reason_codes
    assert result.materialized_files == ()
    assert result.mapped_files == ()
    assert result.dependencies == ()
    assert result.resolution_order == ()
    for field in (
        "inventory_sha256",
        "expected_image_sha256",
        "mapping_sha256",
        "snapshot_sha256",
        "package_result_sha256",
        "apt_evidence_sha256",
        "source_bytes_sha256",
    ):
        assert getattr(result, field) is None, field
    assert result.v4_binding == "NOT_PROVIDED"
    if code is not None:
        assert code in result.reason_codes
    _blocked(result)


def test_one_real_deb_symlink_materializes_exact_verified_source(case: MaterializationCase) -> None:
    result = case.verify()
    assert result.consistency == "MATCH", result.reason_codes
    assert result.reason_codes == ()
    assert len(result.mapped_files) == 3
    assert len(result.materialized_files) == 1
    output = result.materialized_files[0]
    assert (output.file_id, output.path, output.sha256, output.size) == (
        "link",
        "/link",
        _sha(_PAYLOAD),
        len(_PAYLOAD),
    )
    assert (output.uid, output.gid, output.mode) == (0, 0, 0o644)
    assert result.inventory_sha256 == _sha(case.inventory_raw)
    assert result.expected_image_sha256 == _sha(case.expected_raw)
    assert result.source_bytes_sha256
    assert result.dependencies
    assert result.resolution_order
    _blocked(result)


def test_two_independent_links_can_share_one_regular_source(tmp_path: Path) -> None:
    fixture = _build_case(
        tmp_path / "debs",
        entries=[
            ("./", "5", b""),
            ("./link-b", "2", "value"),
            ("./value", "0", _PAYLOAD),
            ("./link-a", "2", "value"),
        ],
        expected_files=[
            _directory("root", "/"),
            _file("value", "/value"),
            _file("a", "/link-a"),
            _file("b", "/link-b"),
        ],
    )
    result = fixture.verify()
    assert result.consistency == "MATCH", result.reason_codes
    assert len(result.materialized_files) == 2
    assert len(result.mapped_files) == 4
    assert {row.sha256 for row in result.materialized_files} == {_sha(_PAYLOAD)}
    assert {row.size for row in result.materialized_files} == {len(_PAYLOAD)}
    assert result.materialized_files[0].source == result.materialized_files[1].source
    before = copy.deepcopy(fixture.mapping)
    fixture.mapping["transforms"].reverse()
    fixture.mapping["entries"].reverse()
    reordered = fixture.verify()
    assert reordered.consistency == "MATCH", reordered.reason_codes
    assert reordered.resolution_order == result.resolution_order
    assert before["transforms"] != fixture.mapping["transforms"]
    _blocked(reordered)


def test_dependency_order_places_exact_prerequisites_before_outputs(
    case: MaterializationCase,
) -> None:
    result = case.verify()
    assert result.consistency == "MATCH", result.reason_codes
    graph = {row.node_id: set(row.dependencies) for row in result.dependencies}
    positions = {node: index for index, node in enumerate(result.resolution_order)}
    assert len(positions) == len(result.resolution_order) == len(graph)
    assert set(positions) == set(graph)
    for node, prerequisites in graph.items():
        for prerequisite in prerequisites:
            assert positions[prerequisite] < positions[node]
    output = result.materialized_files[0]
    link_node = f"member:{output.link.source_id}:{output.link.record_id}"
    source_node = f"member:{output.source.source_id}:{output.source.record_id}"
    operation_node = "symlink:" + output.transform_id
    assert graph[operation_node] == {link_node, source_node}
    assert graph["expected:" + output.file_id] == {operation_node}
    assert graph[link_node] == {"package:" + output.link.source_id}
    assert graph[source_node] == {"package:" + output.source.source_id}
    _blocked(result)


@pytest.mark.parametrize(
    "graph,code",
    [
        ({"member": {"absent"}}, "DEPENDENCY_MISSING"),
        ({"a": {"b"}, "b": {"a"}}, "DEPENDENCY_CYCLE"),
    ],
)
def test_iterative_graph_validator_refuses_missing_nodes_and_cycles(
    graph: dict[str, set[str]], code: str
) -> None:
    with pytest.raises(image.ImageMappingRefusalV5a) as caught:
        materialization._order(graph, image._MappingBudget(Limits()))
    assert caught.value.code == code


@pytest.mark.parametrize("target", ["./value", "sub/../value"])
def test_lexical_components_can_stay_inside_declared_image(tmp_path: Path, target: str) -> None:
    fixture = _build_case(
        tmp_path / "debs",
        entries=[
            ("./", "5", b""),
            ("./sub/", "5", b""),
            ("./value", "0", _PAYLOAD),
            ("./link", "2", target),
        ],
        expected_files=[
            _directory("root", "/"),
            _directory("sub", "/sub"),
            _file("value", "/value"),
            _file("link", "/link"),
        ],
    )
    result = fixture.verify()
    assert result.consistency == "MATCH", result.reason_codes
    _blocked(result)


def test_legacy_v5a_success_definition_is_unchanged(case: MaterializationCase) -> None:
    legacy = case.base.verify()
    assert legacy.consistency == "INCOMPLETE"
    assert legacy.mapped_files == ()
    assert "symlink" in legacy.unsupported_transformations
    result = case.verify()
    assert result.consistency == "MATCH", result.reason_codes
    assert result.profile != legacy.profile
    _blocked(result)


def test_independent_expected_hash_cannot_be_replaced_by_transform_hash(
    case: MaterializationCase,
) -> None:
    assert case.verify().consistency == "MATCH"
    case.expected["files"][2] = _file("link", "/link", b"independently selected different bytes\n")
    _refused(case.verify(), "ARTIFACT_MISMATCH")


def test_expected_image_out_of_band_pin_survives_replacement(case: MaterializationCase) -> None:
    pin = _sha(case.expected_raw)
    case.expected["image_id"] = "substituted expectation"
    _refused(case.verify(expected_image_sha256=pin))


@pytest.mark.parametrize(
    "target",
    [
        "/value",
        "../value",
        "../../outside",
        "missing",
        "value/child",
        "sub/../../value",
        "value/../value",
        "value//child",
        "value\\child",
    ],
)
def test_unsupported_or_escaping_literal_targets_refuse(tmp_path: Path, target: str) -> None:
    fixture = _build_case(
        tmp_path / "debs",
        entries=[("./", "5", b""), ("./value", "0", _PAYLOAD), ("./link", "2", target)],
    )
    _refused(fixture.verify())


def test_symlink_to_directory_refuses(tmp_path: Path) -> None:
    fixture = _build_case(
        tmp_path / "debs",
        entries=[
            ("./", "5", b""),
            ("./dir/", "5", b""),
            ("./value", "0", _PAYLOAD),
            ("./link", "2", "dir"),
        ],
        expected_files=[
            _directory("root", "/"),
            _directory("dir", "/dir"),
            _file("value", "/value"),
            _file("link", "/link"),
        ],
        links={"./link": ("source-0", "./dir/")},
    )
    _refused(fixture.verify())


@pytest.mark.parametrize(
    "targets", [("link-b", "value"), ("link-b", "link-a"), ("link-a", "value")]
)
def test_link_chains_and_cycles_refuse(tmp_path: Path, targets: tuple[str, str]) -> None:
    fixture = _build_case(
        tmp_path / "debs",
        entries=[
            ("./", "5", b""),
            ("./value", "0", _PAYLOAD),
            ("./link-a", "2", targets[0]),
            ("./link-b", "2", targets[1]),
        ],
        expected_files=[
            _directory("root", "/"),
            _file("value", "/value"),
            _file("a", "/link-a"),
            _file("b", "/link-b"),
        ],
    )
    _refused(fixture.verify())


def test_cross_package_source_is_refused_even_if_bytes_match(tmp_path: Path) -> None:
    fixture = _build_case(
        tmp_path / "debs",
        entries=[("./", "5", b""), ("./value", "0", _PAYLOAD), ("./link", "2", "second")],
        second_entries=[("./", "5", b""), ("./second", "0", _PAYLOAD)],
        expected_files=[
            _directory("root", "/"),
            _file("value", "/value"),
            _file("link", "/link"),
            _file("second", "/second"),
        ],
        links={"./link": ("source-1", "./second")},
    )
    _refused(fixture.verify())


@pytest.mark.parametrize(
    "field,value",
    [
        ("sha256", "1" * 64),
        ("size", 99),
        ("uid", 9),
        ("gid", 9),
        ("mode", 0o777),
        ("path", "/relocated"),
        ("type", "directory"),
    ],
)
def test_expected_output_metadata_is_exact(
    case: MaterializationCase, field: str, value: Any
) -> None:
    case.expected["files"][2][field] = value
    _refused(case.verify())


def test_metadata_change_is_an_explicit_unsupported_requirement(case: MaterializationCase) -> None:
    case.expected["files"][2]["mode"] = 0o600
    result = case.verify()
    _refused(result)
    assert "metadata_change" in result.unsupported_transformations


@pytest.mark.parametrize("where", ["mapping", "expected"])
@pytest.mark.parametrize("role", ["link", "source"])
@pytest.mark.parametrize(
    "field,value", [("source_id", "unknown"), ("record_id", "1" * 64), ("member", "value")]
)
def test_literal_record_identity_cannot_be_substituted(
    case: MaterializationCase, where: str, role: str, field: str, value: str
) -> None:
    row = (
        case.mapping["transforms"][0]
        if where == "mapping"
        else case.expected["materializations"][0]
    )
    row[role][field] = value
    _refused(case.verify())


@pytest.mark.parametrize(
    "field,value", [("sha256", "1" * 64), ("size", 99), ("uid", 4), ("name", "./substituted")]
)
def test_source_record_tampering_cannot_reuse_verified_archive(
    case: MaterializationCase, field: str, value: Any
) -> None:
    case.inventory["packages"][0]["data"]["records"][1][field] = value
    _refused(case.verify())


def test_actual_source_payload_tampering_is_measured(case: MaterializationCase) -> None:
    archive = case.archives[0]
    assert _PAYLOAD in archive.content
    changed = archive.content.replace(_PAYLOAD, b"tampered payload\n")
    case.archives = (FrozenPackageV5b(archive.filename, changed),)
    _refused(case.verify())


def test_mutable_archive_buffer_cannot_be_used_as_frozen_source(case: MaterializationCase) -> None:
    archive = case.archives[0]
    mutable: Any = bytearray(archive.content)
    _refused(case.verify(archives=(FrozenPackageV5b(archive.filename, mutable),)))


def test_archive_filename_cannot_be_rebound(case: MaterializationCase) -> None:
    archive = case.archives[0]
    _refused(case.verify(archives=(FrozenPackageV5b("other.deb", archive.content),)))


@pytest.mark.parametrize("kind", ["1", "3", "4", "6", "7"])
def test_hardlinks_devices_fifo_and_other_special_types_refuse(
    case: MaterializationCase, kind: str
) -> None:
    case.inventory["packages"][0]["data"]["records"][2]["typeflag"] = kind
    _refused(case.verify())


@pytest.mark.parametrize(
    "field,value",
    [
        ("package", "libpython3.12-minimal"),
        ("version", "3.12.3-1ubuntu0.18"),
        ("architecture", "arm64"),
        ("filename", "other.deb"),
        ("sha256", "1" * 64),
        ("size", 99),
    ],
)
def test_full_package_identity_is_required(
    case: MaterializationCase, field: str, value: Any
) -> None:
    case.mapping["sources"][0][field] = value
    _refused(case.verify())


@pytest.mark.parametrize(
    "which",
    ["archives", "transform", "entry", "expected-materialization", "source-entry", "expected-file"],
)
def test_missing_inputs_or_accounting_never_match(case: MaterializationCase, which: str) -> None:
    if which == "archives":
        case.archives = ()
    elif which == "transform":
        case.mapping["transforms"].clear()
    elif which == "entry":
        case.mapping["entries"].pop()
    elif which == "expected-materialization":
        case.expected["materializations"].clear()
    elif which == "source-entry":
        case.mapping["entries"].pop(1)
    else:
        case.expected["files"].pop()
    _refused(case.verify())


def test_extra_expected_file_is_not_implicitly_created(case: MaterializationCase) -> None:
    case.expected["files"].append(_file("hidden", "/.hidden"))
    _refused(case.verify())


def test_extra_archive_member_cannot_be_ignored(tmp_path: Path) -> None:
    fixture = _build_case(
        tmp_path / "debs",
        entries=[
            ("./", "5", b""),
            ("./value", "0", _PAYLOAD),
            ("./link", "2", "value"),
            ("./hidden", "0", b"hidden"),
        ],
    )
    fixture.mapping["entries"].pop()
    _refused(fixture.verify())


@pytest.mark.parametrize(
    "which", ["mapping-entry", "expected-materialization", "transform", "archive"]
)
def test_duplicate_accounting_and_outputs_are_refused(
    case: MaterializationCase, which: str
) -> None:
    if which == "mapping-entry":
        case.mapping["entries"].append(copy.deepcopy(case.mapping["entries"][-1]))
    elif which == "expected-materialization":
        case.expected["materializations"].append(
            copy.deepcopy(case.expected["materializations"][0])
        )
    elif which == "transform":
        row = copy.deepcopy(case.mapping["transforms"][0])
        row["transform_id"] = "other-transform"
        case.mapping["transforms"].append(row)
    else:
        case.archives += case.archives
    _refused(case.verify())


@pytest.mark.parametrize("which", ["expected", "mapping"])
def test_mixed_snapshots_are_refused(case: MaterializationCase, which: str) -> None:
    document = case.expected if which == "expected" else case.mapping
    document["snapshot_sha256"] = _sha(b"unrelated snapshot")
    _refused(case.verify())


@pytest.mark.parametrize("kind", ["materialize_member", "pyc", "dpkg", "maintainer_script"])
def test_v1_or_unsupported_transform_cannot_expand_v5b(
    case: MaterializationCase, kind: str
) -> None:
    case.mapping["transforms"][0]["kind"] = kind
    _refused(case.verify())


@pytest.mark.parametrize("document", ["mapping", "expected"])
def test_metadata_policy_must_be_explicit_and_supported(
    case: MaterializationCase, document: str
) -> None:
    row = (
        case.mapping["transforms"][0]
        if document == "mapping"
        else case.expected["materializations"][0]
    )
    row["metadata_policy"] = "fix-automatically"
    _refused(case.verify())


@pytest.mark.parametrize(
    "result",
    [
        None,
        v3.PackageInventoryResultV3("INCOMPLETE", ("PROVENANCE_INCOMPLETE",)),
        v3.PackageInventoryResultV3("MATCH", ()),
    ],
)
def test_incomplete_v3_claim_is_refused(case: MaterializationCase, result: Any) -> None:
    # No caller-created MATCH can substitute for independently supplied bytes.
    _refused(case.verify(package_result=result))


def _copy_context(case: MaterializationCase, evidence: Any) -> None:
    case.expected["apt_context"] = copy.deepcopy(case.base.expected["apt_context"])
    case.mapping["apt_context"] = copy.deepcopy(case.base.mapping["apt_context"])


def test_v4b_v4c_context_binding_retains_all_production_blockers(case: MaterializationCase) -> None:
    evidence = _acceptance_evidence(case.base)
    _copy_context(case, evidence)
    before = copy.deepcopy(evidence)
    result = case.verify(apt_evidence=evidence)
    assert result.consistency == "MATCH", result.reason_codes
    assert result.v4_binding == "CALLER_CLAIM_BOUND"
    assert result.apt_evidence_sha256
    assert evidence == before
    _blocked(result)


@pytest.mark.parametrize("part", ["snapshot", "context", "repository", "v4a", "v4c"])
def test_invalid_v4_context_clears_all_materialization(
    case: MaterializationCase, part: str
) -> None:
    evidence = _acceptance_evidence(case.base)
    _copy_context(case, evidence)
    if part == "snapshot":
        evidence = replace(evidence, snapshot_sha256="1" * 64)
    elif part == "context":
        evidence = replace(
            evidence, trust_context=replace(evidence.trust_context, context_id="other-context")
        )
    elif part == "repository":
        state = replace(evidence.acceptance_result.proposed_state, repository_id="other-repository")
        evidence = replace(
            evidence, acceptance_result=replace(evidence.acceptance_result, proposed_state=state)
        )
    elif part == "v4a":
        evidence = replace(evidence, result=replace(evidence.result, packages=()))
    else:
        evidence = replace(
            evidence, acceptance_result=replace(evidence.acceptance_result, reason_codes=())
        )
    _refused(case.verify(apt_evidence=evidence))


@pytest.mark.parametrize(
    "changes",
    [
        {"max_archive_members": 5},
        {"max_files": 2},
        {"max_file_bytes": 8},
        {"max_total_bytes": 512},
        {"working_memory_bytes": 262144},
        {"max_path_depth": 1},
    ],
)
def test_resource_limits_never_return_partial_materialization(
    case: MaterializationCase, changes: dict[str, int]
) -> None:
    if "max_path_depth" in changes:
        case.expected["files"][2]["path"] = "/deep/link"
    _refused(case.verify(limits=replace(Limits(), **changes)))


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, InterruptedError, MemoryError])
def test_interruption_discards_every_partial_binding(
    case: MaterializationCase, monkeypatch: pytest.MonkeyPatch, interruption: type[BaseException]
) -> None:
    def interrupt(self: Any) -> None:
        raise interruption

    monkeypatch.setattr(offline._Budget, "check", interrupt)
    _refused(case.verify())


def test_deadline_exhaustion_during_processing_refuses(
    case: MaterializationCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = offline._Budget.check
    calls = 0

    def exhausted(self: Any) -> None:
        nonlocal calls
        calls += 1
        if calls >= 70:
            raise offline.contract.ContractRefusal("LIMIT_EXCEEDED")
        original(self)

    monkeypatch.setattr(offline._Budget, "check", exhausted)
    _refused(case.verify(), "LIMIT_EXCEEDED")


@pytest.mark.parametrize("failure", [KeyboardInterrupt, MemoryError, ValueError])
def test_failure_after_first_of_two_materializations_erases_every_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: type[BaseException]
) -> None:
    fixture = _build_case(
        tmp_path / "debs",
        entries=[
            ("./", "5", b""),
            ("./value", "0", _PAYLOAD),
            ("./link-a", "2", "value"),
            ("./link-b", "2", "value"),
        ],
        expected_files=[
            _directory("root", "/"),
            _file("value", "/value"),
            _file("a", "/link-a"),
            _file("b", "/link-b"),
        ],
    )
    original = materialization._origin
    calls = 0

    def fail_second(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        # Each operation builds its two origins before staging the result.
        # A third call therefore proves that the first operation was staged.
        if calls == 3:
            raise failure
        return original(*args, **kwargs)

    monkeypatch.setattr(materialization, "_origin", fail_second)
    _refused(fixture.verify())
    assert calls == 3


@pytest.mark.parametrize("failure", [KeyboardInterrupt, MemoryError, ValueError])
def test_failure_after_complete_staging_erases_every_positive_result(
    case: MaterializationCase, monkeypatch: pytest.MonkeyPatch, failure: type[BaseException]
) -> None:
    original_materialize = materialization._materialize
    original_check = offline._Budget.check
    staged = False

    def capture(*args: Any, **kwargs: Any) -> Any:
        nonlocal staged
        materialized, mapped, digest = original_materialize(*args, **kwargs)
        assert len(materialized) == len(mapped) == 1
        assert digest
        staged = True
        return materialized, mapped, digest

    def fail_final_check(self: Any) -> None:
        if staged:
            raise failure
        original_check(self)

    monkeypatch.setattr(materialization, "_materialize", capture)
    monkeypatch.setattr(offline._Budget, "check", fail_final_check)
    _refused(case.verify())
    assert staged


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
def test_malformed_declarations_fail_closed(case: MaterializationCase, raw: bytes) -> None:
    _refused(case.verify(mapping_raw=raw))


def test_no_host_resolution_filesystem_network_or_execution(
    case: MaterializationCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("V5b attempted host I/O or runtime execution")

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


def test_linux_public_v3_v4a_pipeline_binds_real_deb_and_snapshot(
    case: MaterializationCase,
) -> None:
    if sys.platform != "linux":
        pytest.skip(
            "Public V3/V4a descriptor readers require Linux; immutable byte tests remain portable"
        )
    measured_v3 = v3.verify_ubuntu_package_inventory_v3(
        case.base.directory, inventory_raw=case.inventory_raw
    )
    assert measured_v3.consistency == "MATCH", measured_v3.reason_codes
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
    (case.base.directory / "Packages").write_bytes(index)
    (case.base.directory / "Release").write_bytes(release)
    measured_v4a = v4a.verify_apt_provenance_v4a(
        case.base.directory, declaration=declaration, inventory_raw=case.inventory_raw
    )
    assert measured_v4a.consistency == "MATCH", measured_v4a.reason_codes
    case.expected["snapshot_sha256"] = _sha(release)
    case.mapping["snapshot_sha256"] = _sha(release)
    evidence = image.AptEvidenceV5a(
        _sha(case.inventory_raw), _sha(release), declaration, measured_v4a
    )
    result = case.verify(package_result=measured_v3, apt_evidence=evidence)
    assert result.consistency == "MATCH", result.reason_codes
    assert result.inventory_sha256 == evidence.inventory_sha256
    assert result.snapshot_sha256 == evidence.snapshot_sha256
    assert result.snapshot_sha256 == declaration.release_sha256
    assert result.package_result_sha256
    assert result.apt_evidence_sha256
    assert result.source_bytes_sha256
    assert len(result.materialized_files) == 1
    _blocked(result)
