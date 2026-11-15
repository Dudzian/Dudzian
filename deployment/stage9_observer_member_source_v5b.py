"""Bounded immutable DEB reads for V5b; no extraction, installation or trust.

Only raw and gzip TAR members are admitted by this adapter. V3's additional
Zstandard profile remains unchanged and is deliberately not enabled here.
Every byte returned comes from a complete immutable DEB remeasured against V3.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from deployment.stage9_observer_contract import ContractRefusal, _digest, _identifier
from deployment.stage9_observer_image_v5a import record_id_v5a
from deployment.stage9_observer_offline import Limits, _Budget, _InputReader, _validate_limits
from deployment.stage9_observer_package_v3 import (
    PackageMeasurementV3,
    UbuntuPackageInventoryV3,
    _ar_integer,
    _control_identity,
    _parse,
)
from deployment.stage9_observer_tar_v3 import TarMemberV3, TarResultV3, _header, verify_tar_v3
from deployment.stage9_observer_ubuntu import _PACKAGES, CPYTHON_PACKAGE_VERSION, _pread

_DEFAULT_LIMITS = Limits()


@dataclass(frozen=True, slots=True)
class FrozenPackageV5b:
    filename: str
    content: bytes


@dataclass(frozen=True, slots=True)
class MemberSelectionV5b:
    source_id: str
    package: str
    member: str
    record_id: str


@dataclass(frozen=True, slots=True)
class CapturedMemberV5b:
    source_id: str
    package: str
    member: str
    record_id: str
    archive_sha256: str
    archive_size: int
    content: bytes


def _require(condition: bool, code: str = "FORMAT_REJECTED") -> None:
    if not condition:
        raise ContractRefusal(code)


class _MemoryRegion(io.RawIOBase):
    """Read-only view of immutable bytes, with no archive-sized read request."""

    def __init__(self, raw: bytes, start: int, size: int, budget: _Budget) -> None:
        self.raw, self.start, self.remaining = raw, start, size
        self.position, self.budget = 0, budget

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        self.budget.check()
        amount = min(
            self.budget.limits.chunk_bytes,
            self.remaining,
            self.budget.limits.chunk_bytes if size < 0 else size,
        )
        part = self.raw[self.start + self.position : self.start + self.position + amount]
        _require(len(part) == amount, "PROVENANCE_INCOMPLETE")
        self.position += amount
        self.remaining -= amount
        return part


class _Capture:
    def __init__(self, stream: Any, maximum: int, budget: _Budget) -> None:
        self.stream, self.maximum, self.budget = stream, maximum, budget
        self.content = bytearray()

    def read(self, size: int) -> bytes:
        self.budget.check()
        _require(
            type(size) is int and 0 <= size <= self.budget.limits.chunk_bytes, "LIMIT_EXCEEDED"
        )
        part = self.stream.read(size)
        _require(type(part) is bytes and len(part) <= size, "PROVENANCE_INCOMPLETE")
        _require(len(self.content) + len(part) <= self.maximum, "LIMIT_EXCEEDED")
        self.content.extend(part)
        return cast(bytes, part)


def _selected_content(
    raw: bytes,
    package: UbuntuPackageInventoryV3,
    selected: dict[str, MemberSelectionV5b],
    budget: _Budget,
) -> tuple[CapturedMemberV5b, ...]:
    """Locate payloads only after the public V3 verifier accepts every TAR byte."""
    records = {record.name: record for record in package.data.records}
    found: list[CapturedMemberV5b] = []
    position = 0
    while position + 512 <= len(raw):
        budget.check()
        header = raw[position : position + 512]
        if header == bytes(512):
            break
        record = _header(header, package.data.profile, budget.limits)
        position += 512
        if record.name in selected:
            declaration = records[record.name]
            selection = selected[record.name]
            _require(record.typeflag in {"0", "\0"}, "UNSUPPORTED_SOURCE")
            _require(record_id_v5a(declaration) == selection.record_id, "ARTIFACT_MISMATCH")
            budget.reserve(record.size + 2048)
            content = raw[position : position + record.size]
            digest = hashlib.sha256()
            for offset in range(0, len(content), budget.limits.chunk_bytes):
                budget.check()
                digest.update(content[offset : offset + budget.limits.chunk_bytes])
            _require(
                len(content) == declaration.size and digest.hexdigest() == declaration.sha256,
                "ARTIFACT_MISMATCH",
            )
            found.append(
                CapturedMemberV5b(
                    selection.source_id,
                    package.package,
                    declaration.name,
                    selection.record_id,
                    package.sha256,
                    package.size,
                    content,
                )
            )
        position += record.size + (-record.size) % 512
    _require({item.member for item in found} == set(selected), "PROVENANCE_INCOMPLETE")
    return tuple(found)


def _archive(
    raw: bytes,
    package: UbuntuPackageInventoryV3,
    selected: dict[str, MemberSelectionV5b],
    budget: _Budget,
) -> tuple[PackageMeasurementV3, tuple[CapturedMemberV5b, ...]]:
    _require(raw[:8] == b"!<arch>\n", "UNSUPPORTED_SOURCE")
    offset = 8
    results: list[TarResultV3] = []
    captured: tuple[CapturedMemberV5b, ...] = ()
    for index in range(3):
        budget.check()
        header = raw[offset : offset + 60]
        _require(len(header) == 60, "PROVENANCE_INCOMPLETE")
        _require(header[58:] == b"`\n")
        for value in (header[16:28], header[28:34], header[34:40]):
            _ar_integer(value)
        _require(_ar_integer(header[40:48], octal=True) <= 0o177777)
        size = _ar_integer(header[48:58])
        name = header[:16].rstrip(b" ").removesuffix(b"/")
        start = offset + 60
        _require(start + size + size % 2 <= len(raw), "PROVENANCE_INCOMPLETE")
        if index == 0:
            _require(name == b"debian-binary" and size == 4, "UNSUPPORTED_SOURCE")
            _require(raw[start : start + 4] == b"2.0\n", "UNSUPPORTED_SOURCE")
        else:
            base = b"control.tar" if index == 1 else b"data.tar"
            _require(name in {base, base + b".gz"}, "UNSUPPORTED_SOURCE")
            expected = package.control if index == 1 else package.data
            remaining = replace(
                budget.limits,
                max_unpacked_bytes=budget.limits.max_unpacked_bytes - budget.unpacked_bytes,
                max_archive_members=budget.limits.max_archive_members - budget.archive_members,
            )
            _require(
                remaining.max_unpacked_bytes > 0 and remaining.max_archive_members > 0,
                "LIMIT_EXCEEDED",
            )
            region = _MemoryRegion(raw, start, size, budget)
            stream: Any = (
                gzip.GzipFile(fileobj=region, mode="rb") if name.endswith(b".gz") else region
            )
            capture: _Capture | None = None
            if index == 2 and selected:
                # bytearray growth, immutable conversion, and verification buffers.
                budget.reserve(expected.unpacked_bytes * 3)
                capture = _Capture(stream, expected.unpacked_bytes, budget)
            try:
                result = verify_tar_v3(
                    capture if capture is not None else stream,
                    expected,
                    limits=remaining,
                    check=budget.check,
                    compressed_size=size,
                )
                _require(region.remaining == 0, "PROVENANCE_INCOMPLETE")
            finally:
                stream.close()
            budget.unpacked_bytes += result.unpacked_bytes
            budget.archive_members += result.members
            if index == 1:
                _control_identity(result.control, package, budget)
            elif capture is not None:
                captured = _selected_content(bytes(capture.content), package, selected, budget)
            results.append(result)
        if size % 2:
            _require(raw[start + size : start + size + 1] == b"\n")
        offset = start + size + size % 2
    _require(offset == len(raw), "CLOSURE_INCOMPLETE")
    return (
        PackageMeasurementV3(package.package, package.sha256, len(raw), results[0], results[1]),
        captured,
    )


def capture_member_sources_v5b(
    *,
    packages: tuple[UbuntuPackageInventoryV3, ...],
    archives: tuple[FrozenPackageV5b, ...],
    selections: tuple[MemberSelectionV5b, ...],
    budget: _Budget,
) -> tuple[tuple[PackageMeasurementV3, ...], tuple[CapturedMemberV5b, ...]]:
    """Verify every immutable DEB and return selected bytes only on full success.

    Inputs and outputs are unapproved data. No MATCH object can substitute for
    bytes. Exceptions, including interruption, return no partial measurements
    or payloads; the enclosing V5b boundary converts them into a refusal.
    """
    _validate_limits(budget.limits)
    budget.check()
    _require(
        type(packages) is tuple
        and 0 < len(packages) <= min(len(_PACKAGES), budget.limits.max_files),
        "LIMIT_EXCEEDED",
    )
    _require(type(archives) is tuple and len(archives) == len(packages), "PROVENANCE_INCOMPLETE")
    _require(type(selections) is tuple, "FORMAT_REJECTED")
    _require(len(selections) <= budget.limits.max_files, "LIMIT_EXCEEDED")
    by_name: dict[str, UbuntuPackageInventoryV3] = {}
    for package in packages:
        budget.check()
        _require(type(package) is UbuntuPackageInventoryV3)
        _require(
            package.package in _PACKAGES
            and package.version == CPYTHON_PACKAGE_VERSION
            and package.architecture == "amd64",
            "UNSUPPORTED_SOURCE",
        )
        _require(package.package not in by_name, "CLOSURE_INCOMPLETE")
        _require(
            package.filename == f"{package.package}_{package.version}_{package.architecture}.deb"
        )
        _digest(package.sha256)
        _require(
            type(package.size) is int and 8 <= package.size <= budget.limits.max_file_bytes,
            "LIMIT_EXCEEDED",
        )
        by_name[package.package] = package
    available: dict[str, bytes] = {}
    for archive in archives:
        budget.check()
        _require(type(archive) is FrozenPackageV5b and type(archive.filename) is str)
        _require(0 < len(archive.filename) <= 256, "LIMIT_EXCEEDED")
        _require(type(archive.content) is bytes, "FILE_POLICY_REJECTED")
        _require(archive.filename not in available, "CLOSURE_INCOMPLETE")
        _require(0 < len(archive.content) <= budget.limits.max_file_bytes, "LIMIT_EXCEEDED")
        budget.reserve(len(archive.content) + 1024)
        available[archive.filename] = archive.content
    _require(set(available) == {package.filename for package in packages}, "PROVENANCE_INCOMPLETE")
    selected: dict[str, dict[str, MemberSelectionV5b]] = {name: {} for name in by_name}
    source_ids: dict[str, str] = {}
    package_sources: dict[str, str] = {}
    records_by_package: dict[str, dict[str, TarMemberV3]] = {}
    record_count = 0
    for package in packages:
        budget.check()
        record_count += len(package.data.records) + len(package.control.records)
        _require(record_count <= budget.limits.max_archive_members, "LIMIT_EXCEEDED")
        records: dict[str, TarMemberV3] = {}
        for record in package.data.records:
            budget.check()
            _require(type(record) is TarMemberV3 and type(record.name) is str)
            _require(record.name not in records, "CLOSURE_INCOMPLETE")
            records[record.name] = record
            budget.reserve(512)
        records_by_package[package.package] = records
    for selection in selections:
        budget.check()
        _require(type(selection) is MemberSelectionV5b)
        _identifier(selection.source_id)
        _digest(selection.record_id)
        _require(
            type(selection.package) is str and selection.package in by_name, "PROVENANCE_INCOMPLETE"
        )
        _require(type(selection.member) is str and len(selection.member) <= 256, "LIMIT_EXCEEDED")
        _require(selection.member not in selected[selection.package], "CLOSURE_INCOMPLETE")
        _require(
            source_ids.get(selection.source_id, selection.package) == selection.package,
            "PROVENANCE_INCOMPLETE",
        )
        _require(
            package_sources.get(selection.package, selection.source_id) == selection.source_id,
            "PROVENANCE_INCOMPLETE",
        )
        source_ids[selection.source_id] = selection.package
        package_sources[selection.package] = selection.source_id
        records = records_by_package[selection.package]
        _require(selection.member in records, "PROVENANCE_INCOMPLETE")
        record = records[selection.member]
        _require(record.typeflag in {"0", "\0"}, "UNSUPPORTED_SOURCE")
        _require(record_id_v5a(record) == selection.record_id, "ARTIFACT_MISMATCH")
        budget.reserve(2048 + len(selection.member) * 8)
        selected[selection.package][selection.member] = selection
    budget.reserve(budget.limits.chunk_bytes * 8 + 1048576)
    measurements: list[PackageMeasurementV3] = []
    captured: list[CapturedMemberV5b] = []
    for package in packages:
        budget.check()
        raw = available[package.filename]
        _require(len(raw) == package.size, "ARTIFACT_MISMATCH")
        digest = hashlib.sha256()
        for position in range(0, len(raw), budget.limits.chunk_bytes):
            budget.check()
            block = raw[position : position + budget.limits.chunk_bytes]
            budget.read(len(block))
            digest.update(block)
        _require(digest.hexdigest() == package.sha256, "ARTIFACT_MISMATCH")
        measurement, content = _archive(raw, package, selected[package.package], budget)
        measurements.append(measurement)
        captured.extend(content)
    budget.check()
    return tuple(measurements), tuple(captured)


def snapshot_package_sources_v5b(
    input_directory: str | Path,
    *,
    inventory_raw: bytes,
    limits: Limits = _DEFAULT_LIMITS,
    budget: _Budget | None = None,
) -> tuple[FrozenPackageV5b, ...]:
    """Freeze a closed DEB-only input tree; later verification uses these bytes.

    The descriptor/no-follow/stat model detects source changes during copying.
    Once copied, immutable bytes eliminate the verification/use filesystem race.
    This helper neither grants source authenticity nor promises stable host files.
    """
    _validate_limits(limits)
    if sys.platform != "linux":
        raise ContractRefusal("UNSUPPORTED_SOURCE")
    _require(os.geteuid() != 0, "FILE_POLICY_REJECTED")
    if budget is None:
        budget = _Budget(limits)
    _require(budget.limits == limits, "LIMIT_EXCEEDED")
    packages = _parse(inventory_raw, limits, budget)
    expected = {package.filename: "regular" for package in packages}
    reader = _InputReader(input_directory, budget)
    try:
        reader.baseline = reader.scan(expected)
        snapshots: list[FrozenPackageV5b] = []
        for package in packages:
            budget.reserve(package.size * 3 + limits.chunk_bytes)
            with reader.open_regular(package.filename) as descriptor:
                _require(os.fstat(descriptor).st_size == package.size, "ARTIFACT_MISMATCH")
                content = bytearray()
                digest = hashlib.sha256()
                while len(content) < package.size:
                    budget.check()
                    part = _pread(
                        descriptor,
                        min(limits.chunk_bytes, package.size - len(content)),
                        len(content),
                    )
                    _require(bool(part), "PROVENANCE_INCOMPLETE")
                    budget.read(len(part))
                    digest.update(part)
                    content.extend(part)
                _require(digest.hexdigest() == package.sha256, "ARTIFACT_MISMATCH")
                frozen = FrozenPackageV5b(package.filename, bytes(content))
            snapshots.append(frozen)
        reader.finish(expected)
        budget.check()
    finally:
        reader.close()
    return tuple(snapshots)
