"""Actual byte reads, immutable snapshots and complete archive rejection for V5b."""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from deployment import stage9_observer_member_source_v5b as source
from deployment.stage9_observer_contract import ContractRefusal
from deployment.stage9_observer_image_v5a import record_id_v5a
from deployment.stage9_observer_offline import Limits, _Budget
from deployment.stage9_observer_package_v3 import parse_package_inventory_v3
from tests.deployment.test_stage9_observer_package_v3 import PackageCase, build_package_case

pytestmark = [pytest.mark.security, pytest.mark.unit]


def _selection(case: PackageCase, member: str = "./usr/value.txt") -> source.MemberSelectionV5b:
    package = parse_package_inventory_v3(case.raw_inventory)[0]
    record = next(record for record in package.data.records if record.name == member)
    return source.MemberSelectionV5b("source-0", package.package, member, record_id_v5a(record))


def _capture(case: PackageCase, **kwargs: Any) -> Any:
    packages = parse_package_inventory_v3(case.raw_inventory)
    return source.capture_member_sources_v5b(
        packages=kwargs.pop("packages", packages),
        archives=kwargs.pop(
            "archives", (source.FrozenPackageV5b(case.package["filename"], case.path.read_bytes()),)
        ),
        selections=kwargs.pop("selections", (_selection(case),)),
        budget=kwargs.pop("budget", _Budget(Limits())),
        **kwargs,
    )


@pytest.mark.parametrize("gnu", [True, False], ids=["gnu", "ustar"])
@pytest.mark.parametrize("variant", ["raw", "gz"])
def test_remeasures_complete_archive_and_preserves_literal_record_identity(
    tmp_path: Path, gnu: bool, variant: str
) -> None:
    case = build_package_case(tmp_path / "debs", gnu=gnu, variant=variant)
    measurements, captured = _capture(case)
    assert len(measurements) == len(captured) == 1
    assert measurements[0].archive_sha256 == case.package["sha256"]
    assert measurements[0].control.members == len(case.package["control"]["records"])
    assert measurements[0].data.members == len(case.package["data"]["records"])
    member = captured[0]
    assert member.member == "./usr/value.txt"
    assert member.record_id == _selection(case).record_id
    assert member.archive_sha256 == hashlib.sha256(case.path.read_bytes()).hexdigest()
    assert member.archive_size == case.path.stat().st_size
    assert type(member.content) is bytes
    assert member.content == b"inert fixture bytes\n"


def test_zstandard_remains_an_explicit_adapter_refusal(tmp_path: Path) -> None:
    case = build_package_case(tmp_path / "debs", variant="zst")
    with pytest.raises(ContractRefusal, match="UNSUPPORTED_SOURCE"):
        _capture(case)


@pytest.mark.parametrize(
    "mutation", ["member-alias", "record-id", "directory", "link", "duplicate"]
)
def test_selector_cannot_substitute_an_alias_or_nonregular_record(
    tmp_path: Path, mutation: str
) -> None:
    case = build_package_case(tmp_path / "debs")
    selection = _selection(case)
    selections = (selection,)
    if mutation == "member-alias":
        selections = (replace(selection, member="usr/value.txt"),)
    elif mutation == "record-id":
        selections = (replace(selection, record_id="1" * 64),)
    elif mutation == "directory":
        selections = (_selection(case, "./usr/"),)
    elif mutation == "link":
        selections = (_selection(case, "./absolute-link"),)
    else:
        selections = (selection, selection)
    with pytest.raises(ContractRefusal):
        _capture(case, selections=selections)


@pytest.mark.parametrize(
    "mutation", ["bytearray", "missing", "extra", "duplicate", "wrong-name", "bytes"]
)
def test_every_supplied_archive_must_be_unique_exact_immutable_bytes(
    tmp_path: Path, mutation: str
) -> None:
    case = build_package_case(tmp_path / "debs")
    archive = source.FrozenPackageV5b(case.package["filename"], case.path.read_bytes())
    archives = (archive,)
    if mutation == "bytearray":
        archives = (replace(archive, content=bytearray(archive.content)),)
    elif mutation == "missing":
        archives = ()
    elif mutation in {"extra", "duplicate"}:
        archives = (
            archive,
            replace(archive, filename="other.deb") if mutation == "extra" else archive,
        )
    elif mutation == "wrong-name":
        archives = (replace(archive, filename="other.deb"),)
    else:
        archives = (replace(archive, content=archive.content[:-1] + b"x"),)
    with pytest.raises(ContractRefusal):
        _capture(case, archives=archives)


def test_rehashed_archive_does_not_hide_changed_source_bytes(tmp_path: Path) -> None:
    case = build_package_case(tmp_path / "debs")
    original = case.path.read_bytes()
    changed = original.replace(b"inert fixture bytes\n", b"other fixture bytes\n")
    assert changed != original and len(changed) == len(original)
    case.rebind_archive(changed)
    with pytest.raises(ContractRefusal, match="ARTIFACT_MISMATCH"):
        _capture(case)


def test_changed_record_digest_is_checked_against_actual_payload(tmp_path: Path) -> None:
    case = build_package_case(tmp_path / "debs")
    record = next(
        record for record in case.package["data"]["records"] if record["name"] == "./usr/value.txt"
    )
    record["sha256"] = "1" * 64
    with pytest.raises(ContractRefusal, match="ARTIFACT_MISMATCH"):
        _capture(case)


def test_missing_declared_record_does_not_hide_extra_tar_member(tmp_path: Path) -> None:
    case = build_package_case(tmp_path / "debs")
    case.package["data"]["records"].pop()
    with pytest.raises(ContractRefusal, match="CLOSURE_INCOMPLETE"):
        _capture(case)


@pytest.mark.parametrize("limit", ["memory", "bytes", "members", "files", "deadline"])
def test_shared_budget_is_enforced_before_payload_can_escape(tmp_path: Path, limit: str) -> None:
    case = build_package_case(tmp_path / "debs")
    limits = Limits()
    selections = (_selection(case),)
    if limit == "memory":
        limits = replace(limits, working_memory_bytes=512 * 1024)
    elif limit == "bytes":
        limits = replace(limits, max_total_bytes=8)
    elif limit == "members":
        limits = replace(limits, max_archive_members=7)
    elif limit == "files":
        limits = replace(limits, max_files=1)
        selections = (*selections, _selection(case))
    budget = _Budget(limits)
    if limit == "deadline":
        budget.started -= limits.timeout_s
    with pytest.raises(ContractRefusal, match="LIMIT_EXCEEDED"):
        _capture(case, budget=budget, selections=selections)


def test_later_package_failure_returns_neither_earlier_bytes_nor_measurements(
    tmp_path: Path,
) -> None:
    case = build_package_case(tmp_path / "debs")
    second = build_package_case(case.directory, package="libpython3.12-minimal")
    case.declaration["packages"].append(second.package)
    archives = (
        source.FrozenPackageV5b(case.package["filename"], case.path.read_bytes()),
        source.FrozenPackageV5b(second.package["filename"], second.path.read_bytes()[:-1] + b"x"),
    )
    result = None
    with pytest.raises(ContractRefusal, match="ARTIFACT_MISMATCH"):
        result = _capture(case, archives=archives)
    assert result is None


def test_interruption_during_real_tar_read_has_no_returned_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = build_package_case(tmp_path / "debs")
    original = source._MemoryRegion.read
    calls = 0

    def interrupted(region: Any, size: int = -1) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 6:
            raise KeyboardInterrupt
        return original(region, size)

    monkeypatch.setattr(source._MemoryRegion, "read", interrupted)
    result = None
    with pytest.raises(KeyboardInterrupt):
        result = _capture(case)
    assert result is None


def _native_reader() -> None:
    if sys.platform != "linux":
        pytest.skip("Descriptor snapshot adapter requires native Linux")
    if os.geteuid() == 0:
        pytest.skip("Descriptor snapshot adapter refuses privileged readers")


def test_native_snapshot_survives_later_source_replacement(tmp_path: Path) -> None:
    _native_reader()
    case = build_package_case(tmp_path / "debs")
    snapshots = source.snapshot_package_sources_v5b(
        case.directory, inventory_raw=case.raw_inventory
    )
    frozen_digest = hashlib.sha256(snapshots[0].content).hexdigest()
    case.path.write_bytes(b"changed after freeze")
    measurements, captured = _capture(case, archives=snapshots)
    assert measurements[0].archive_sha256 == frozen_digest
    assert captured[0].content == b"inert fixture bytes\n"


def test_native_snapshot_detects_change_even_when_copied_content_still_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _native_reader()
    case = build_package_case(tmp_path / "debs")
    original_read = source._pread
    original_content = case.path.read_bytes()
    changed = False

    def read_and_change(descriptor: int, size: int, offset: int) -> bytes:
        nonlocal changed
        content = original_read(descriptor, size, offset)
        if not changed:
            changed = True
            case.path.write_bytes(original_content)
            info = case.path.stat()
            os.utime(case.path, ns=(info.st_atime_ns, info.st_mtime_ns + 1))
        return content

    monkeypatch.setattr(source, "_pread", read_and_change)
    with pytest.raises(ContractRefusal, match="FILE_CHANGED"):
        source.snapshot_package_sources_v5b(case.directory, inventory_raw=case.raw_inventory)
    assert changed


@pytest.mark.parametrize("kind", ["extra", "symlink", "hardlink"])
def test_native_snapshot_rejects_unaccounted_files_and_host_aliases(
    tmp_path: Path, kind: str
) -> None:
    _native_reader()
    case = build_package_case(tmp_path / "debs")
    if kind == "extra":
        (case.directory / "unexpected").write_bytes(b"unaccounted")
    elif kind == "symlink":
        original = tmp_path / "external.deb"
        case.path.rename(original)
        case.path.symlink_to(original)
    else:
        os.link(case.path, tmp_path / "archive-alias.deb")
    with pytest.raises(ContractRefusal):
        source.snapshot_package_sources_v5b(case.directory, inventory_raw=case.raw_inventory)


def test_native_snapshot_late_failure_discards_first_package(tmp_path: Path) -> None:
    _native_reader()
    case = build_package_case(tmp_path / "debs")
    second = build_package_case(case.directory, package="libpython3.12-minimal")
    case.declaration["packages"].append(second.package)
    second.path.write_bytes(second.path.read_bytes()[:-1] + b"x")
    snapshots = None
    with pytest.raises(ContractRefusal, match="ARTIFACT_MISMATCH"):
        snapshots = source.snapshot_package_sources_v5b(
            case.directory, inventory_raw=case.raw_inventory
        )
    assert snapshots is None
