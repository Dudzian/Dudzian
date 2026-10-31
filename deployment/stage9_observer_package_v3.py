"""Inert, versioned whole-DEB comparisons, separate from Observer image V1.

Declarations, local tools and agreeing bytes are not authenticated expectations.
Nothing here installs a package, resolves a package link or grants authority.
The public operation owns hashing, descriptor checks and shared batch budgets.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import sys
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from deployment.stage9_observer_contract import ContractRefusal
from deployment.stage9_observer_offline import Limits, _Budget, _InputReader, _validate_limits
from deployment.stage9_observer_tar_v3 import (
    TarInventoryV3,
    TarResultV3,
    parse_tar_inventory,
    retained_tar_bytes,
    verify_tar_v3,
)
from deployment.stage9_observer_ubuntu import (
    CPYTHON_PACKAGE_VERSION,
    _PACKAGES,
    _pread,
    _Region,
    _ZstdReader,
)

FORMAT = "stage9-ubuntu-package-inventory-v3"
CLASSIFICATION = "SAME_ORIGIN_OBSERVED_COMPARISON_ONLY"


@dataclass(frozen=True, slots=True)
class UbuntuPackageInventoryV3:
    package: str
    version: str
    architecture: str
    filename: str
    sha256: str
    size: int
    control: TarInventoryV3
    data: TarInventoryV3
    profile: Literal["stage9-ubuntu-package-inventory-v3"] = field(
        default="stage9-ubuntu-package-inventory-v3", init=False
    )


@dataclass(frozen=True, slots=True)
class PackageMeasurementV3:
    package: str
    archive_sha256: str
    archive_bytes: int
    control: TarResultV3
    data: TarResultV3


@dataclass(frozen=True, slots=True)
class PackageInventoryResultV3:
    """Only package declarations agree; no image or execution was measured."""

    consistency: Literal["MATCH", "MISMATCH", "INCOMPLETE"]
    reason_codes: tuple[str, ...]
    packages: tuple[PackageMeasurementV3, ...] = ()
    scope: Literal["INERT_PACKAGE_INVENTORY_ONLY"] = field(
        default="INERT_PACKAGE_INVENTORY_ONLY", init=False
    )
    classification: Literal["SAME_ORIGIN_OBSERVED_COMPARISON_ONLY"] = field(
        default="SAME_ORIGIN_OBSERVED_COMPARISON_ONLY", init=False
    )
    source_authenticity: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    acceptance: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    execution: Literal["MISSING"] = field(default="MISSING", init=False)
    ARCHIVE_TO_TARGET_METADATA_MAPPING: Literal["NOT_YET_PROVEN"] = field(
        default="NOT_YET_PROVEN", init=False
    )
    stage_9: Literal["IN_PROGRESS"] = field(default="IN_PROGRESS", init=False)
    qualification: Literal["BLOCKED"] = field(default="BLOCKED", init=False)
    CONTRACT_APPROVED: Literal["NO"] = field(default="NO", init=False)
    STAGE_3A_COMPLETE: Literal["NO"] = field(default="NO", init=False)
    PRODUCTION_QUALIFIED: Literal["NO"] = field(default="NO", init=False)
    production_status: Literal["PRODUCTION_NO_GO"] = field(default="PRODUCTION_NO_GO", init=False)
    runtime_attestation: Literal["MISSING"] = field(default="MISSING", init=False)
    host_pid_namespace: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    V2_COMPLETE: Literal["NO"] = field(default="NO", init=False)
    refused: Literal[True] = field(default=True, init=False)


def _require(condition: bool, code: str = "FORMAT_REJECTED") -> None:
    if not condition:
        raise ContractRefusal(code)


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    _require(len(pairs) <= 32)
    value: dict[str, Any] = {}
    for key, item in pairs:
        _require(key not in value)
        value[key] = item
    return value


def _no_number(value: str) -> Any:
    raise ContractRefusal("FORMAT_REJECTED")


def _parse_tar(value: Any, limits: Limits, budget: _Budget) -> TarInventoryV3:
    # Reserve the new record/map/set allowance before constructing models.
    # The decoded JSON containers are already covered by the raw allowance.
    _require(type(value) is dict and type(value.get("records")) is list)
    records = value["records"]
    _require(0 < len(records) <= limits.max_archive_members, "LIMIT_EXCEEDED")
    retained = 4096
    for record in records:
        budget.check()
        _require(type(record) is dict)
        retained += 4096
        for key, maximum in (("name", 256), ("linkname", 100)):
            text = record.get(key)
            _require(type(text) is str)
            _require(len(text) <= maximum, "LIMIT_EXCEEDED")
            retained += 8 * len(text.encode("utf-8", errors="strict"))
    budget.reserve(retained)
    result = parse_tar_inventory(value, limits, budget.check)
    _require(retained == retained_tar_bytes(result), "LIMIT_EXCEEDED")
    return result


def _parse(raw: bytes, limits: Limits, budget: _Budget) -> tuple[UbuntuPackageInventoryV3, ...]:
    _require(type(raw) is bytes)
    _require(0 < len(raw) <= 2 * 1024 * 1024, "LIMIT_EXCEEDED")
    # Bounded JSON plus retained Python containers/records. This conservative
    # allowance is cooperative accounting, not hard process memory isolation.
    budget.reserve(len(raw) * 16)
    value = json.loads(
        raw.decode("utf-8", errors="strict"),
        object_pairs_hook=_object,
        parse_float=_no_number,
        parse_constant=_no_number,
    )
    budget.check()
    _require(
        type(value) is dict
        and set(value) == {"format", "classification", "source_authenticity", "packages"}
    )
    _require(value["format"] == FORMAT, "UNSUPPORTED_SOURCE")
    _require(
        value["classification"] == CLASSIFICATION and value["source_authenticity"] == "UNVERIFIED"
    )
    declarations = value["packages"]
    _require(type(declarations) is list and 0 < len(declarations) <= len(_PACKAGES))
    _require(len(declarations) <= limits.max_files, "LIMIT_EXCEEDED")
    packages: list[UbuntuPackageInventoryV3] = []
    names: set[str] = set()
    total_size = total_unpacked = total_members = 0
    for item in declarations:
        budget.check()
        _require(
            type(item) is dict
            and set(item)
            == {
                "package",
                "version",
                "architecture",
                "filename",
                "sha256",
                "size",
                "control",
                "data",
            }
        )
        _require(
            all(
                type(item[key]) is str
                for key in ("package", "version", "architecture", "filename", "sha256")
            )
        )
        _require(item["package"] in _PACKAGES, "UNSUPPORTED_SOURCE")
        _require(item["package"] not in names, "CLOSURE_INCOMPLETE")
        names.add(item["package"])
        _require(
            item["version"] == CPYTHON_PACKAGE_VERSION and item["architecture"] == "amd64",
            "ARTIFACT_MISMATCH",
        )
        _require(
            item["filename"] == f"{item['package']}_{item['version']}_{item['architecture']}.deb"
        )
        _require(
            re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is not None and item["sha256"] != "0" * 64
        )
        _require(
            type(item["size"]) is int and 8 <= item["size"] <= limits.max_file_bytes,
            "LIMIT_EXCEEDED",
        )
        control = _parse_tar(item["control"], limits, budget)
        data = _parse_tar(item["data"], limits, budget)
        total_size += item["size"]
        total_unpacked += control.unpacked_bytes + data.unpacked_bytes
        total_members += len(control.records) + len(data.records)
        _require(
            total_size <= limits.max_total_bytes
            and total_unpacked <= limits.max_unpacked_bytes
            and total_members <= limits.max_archive_members,
            "LIMIT_EXCEEDED",
        )
        _require(
            any(
                record.canonical_name == "control" and record.typeflag in ("0", "\0")
                for record in control.records
            ),
            "PROVENANCE_INCOMPLETE",
        )
        packages.append(
            UbuntuPackageInventoryV3(
                **{
                    key: item[key]
                    for key in ("package", "version", "architecture", "filename", "sha256", "size")
                },
                control=control,
                data=data,
            )
        )
    budget.check()
    return tuple(packages)


def parse_package_inventory_v3(
    raw: bytes, limits: Limits = Limits()
) -> tuple[UbuntuPackageInventoryV3, ...]:
    """Parse unapproved package declarations; this performs no archive I/O."""
    _validate_limits(limits)
    try:
        return _parse(raw, limits, _Budget(limits))
    except (ValueError, UnicodeError, OverflowError, RecursionError):
        raise ContractRefusal("FORMAT_REJECTED") from None
    except MemoryError:
        raise ContractRefusal("LIMIT_EXCEEDED") from None


def _control_identity(
    raw: bytes | None, package: UbuntuPackageInventoryV3, budget: _Budget
) -> None:
    _require(raw is not None, "PROVENANCE_INCOMPLETE")
    assert raw is not None
    text = raw.decode("utf-8", errors="strict")
    _require("\0" not in text and "\r" not in text)
    fields: dict[str, str] = {}
    seen: set[str] = set()
    previous = ""
    for line in text.rstrip("\n").split("\n"):
        budget.check()
        if line.startswith((" ", "\t")):
            _require(bool(previous) and previous not in {"package", "version", "architecture"})
            continue
        key, colon, value = line.partition(":")
        _require(bool(colon) and re.fullmatch(r"[A-Za-z][A-Za-z0-9-]{0,127}", key) is not None)
        key = key.lower()
        _require(key not in seen and len(seen) < 128)
        seen.add(key)
        previous = key
        if key in {"package", "version", "architecture"}:
            fields[key] = value.strip()
    _require(
        fields
        == {
            "package": package.package,
            "version": package.version,
            "architecture": package.architecture,
        },
        "ARTIFACT_MISMATCH",
    )


def _ar_integer(raw: bytes, *, octal: bool = False) -> int:
    value = raw.rstrip(b" ")
    _require(re.fullmatch(rb"[0-7]+" if octal else rb"[0-9]+", value) is not None)
    return int(value, 8 if octal else 10)


def _verify_archive(
    fd: int, package: UbuntuPackageInventoryV3, limits: Limits, budget: _Budget
) -> PackageMeasurementV3:
    _require(_pread(fd, 8, 0) == b"!<arch>\n", "UNSUPPORTED_SOURCE")
    offset = 8
    results: list[TarResultV3] = []
    for index in range(3):
        budget.check()
        header = _pread(fd, 60, offset)
        _require(len(header) == 60, "PROVENANCE_INCOMPLETE")
        _require(header[58:] == b"`\n")
        _ar_integer(header[16:28])
        _ar_integer(header[28:34])
        _ar_integer(header[34:40])
        _require(_ar_integer(header[40:48], octal=True) <= 0o177777)
        size = _ar_integer(header[48:58])
        name = header[:16].rstrip(b" ").removesuffix(b"/")
        start = offset + 60
        _require(start + size + size % 2 <= package.size, "PROVENANCE_INCOMPLETE")
        if index == 0:
            _require(name == b"debian-binary" and size == 4, "UNSUPPORTED_SOURCE")
            _require(_pread(fd, 4, start) == b"2.0\n", "UNSUPPORTED_SOURCE")
        else:
            base = b"control.tar" if index == 1 else b"data.tar"
            _require(name in (base, base + b".gz", base + b".zst"), "UNSUPPORTED_SOURCE")
            remaining = replace(
                limits,
                max_unpacked_bytes=limits.max_unpacked_bytes - budget.unpacked_bytes,
                max_archive_members=limits.max_archive_members - budget.archive_members,
            )
            _require(
                remaining.max_unpacked_bytes > 0 and remaining.max_archive_members > 0,
                "LIMIT_EXCEEDED",
            )
            region = _Region(fd, start, size, limits.chunk_bytes, budget.check)
            stream: Any = region
            if name.endswith(b".zst"):
                stream = _ZstdReader(
                    region,
                    remaining,
                    budget.check,
                    memory_bytes=limits.working_memory_bytes - budget.retained,
                )
            elif name.endswith(b".gz"):
                stream = gzip.GzipFile(fileobj=region, mode="rb")
            try:
                result = verify_tar_v3(
                    stream,
                    package.control if index == 1 else package.data,
                    limits=remaining,
                    check=budget.check,
                    compressed_size=size,
                )
                _require(region.remaining == 0, "PROVENANCE_INCOMPLETE")
                if isinstance(stream, _ZstdReader):
                    _require(stream.finished, "PROVENANCE_INCOMPLETE")
            except BaseException:
                try:
                    stream.close()
                except (ContractRefusal, OSError, EOFError, zlib.error, MemoryError):
                    pass
                raise
            else:
                stream.close()
            budget.unpacked_bytes += result.unpacked_bytes
            budget.archive_members += result.members
            if index == 1:
                _control_identity(result.control, package, budget)
            results.append(result)
        if size % 2:
            _require(_pread(fd, 1, start + size) == b"\n")
        offset = start + size + size % 2
    _require(offset == package.size, "CLOSURE_INCOMPLETE")
    return PackageMeasurementV3(
        package.package, package.sha256, package.size, results[0], results[1]
    )


def verify_ubuntu_package_inventory_v3(
    input_directory: str | Path, *, inventory_raw: bytes, limits: Limits = Limits()
) -> PackageInventoryResultV3:
    """Compare a closed local batch; every failure discards partial matches."""
    reader: _InputReader | None = None
    measurements: tuple[PackageMeasurementV3, ...] = ()
    consistency: Literal["MATCH", "MISMATCH", "INCOMPLETE"] = "INCOMPLETE"
    reasons: set[str] = set()
    try:
        _validate_limits(limits)
        if sys.platform != "linux":
            raise ContractRefusal("UNSUPPORTED_SOURCE")
        _require(os.geteuid() != 0, "FILE_POLICY_REJECTED")
        budget = _Budget(limits)
        packages = _parse(inventory_raw, limits, budget)
        budget.reserve(limits.chunk_bytes * 8 + 1048576)
        expected = {package.filename: "regular" for package in packages}
        reader = _InputReader(input_directory, budget)
        reader.baseline = reader.scan(expected)
        staged: list[PackageMeasurementV3] = []
        for package in packages:
            with reader.open_regular(package.filename) as fd:
                _require(os.fstat(fd).st_size == package.size, "ARTIFACT_MISMATCH")
                digest = hashlib.sha256()
                position = 0
                while position < package.size:
                    budget.check()
                    block = _pread(fd, min(limits.chunk_bytes, package.size - position), position)
                    _require(bool(block), "PROVENANCE_INCOMPLETE")
                    budget.read(len(block))
                    digest.update(block)
                    position += len(block)
                _require(digest.hexdigest() == package.sha256, "ARTIFACT_MISMATCH")
                staged.append(_verify_archive(fd, package, limits, budget))
        reader.finish(expected)
        budget.check()
        measurements = tuple(staged)
        consistency = "MATCH"
    except ContractRefusal as exc:
        reasons.add(exc.code)
        if exc.code == "ARTIFACT_MISMATCH":
            consistency = "MISMATCH"
    except (
        EOFError,
        gzip.BadGzipFile,
        zlib.error,
        UnicodeError,
        ValueError,
        OverflowError,
        RecursionError,
    ):
        reasons.add("FORMAT_REJECTED")
    except MemoryError:
        reasons.add("LIMIT_EXCEEDED")
    except OSError:
        reasons.add("PROVENANCE_INCOMPLETE")
    finally:
        if reader is not None:
            try:
                reader.close()
            except OSError:
                reasons.add("PROVENANCE_INCOMPLETE")
                consistency = "INCOMPLETE"
    return PackageInventoryResultV3(
        consistency, tuple(sorted(reasons)), measurements if consistency == "MATCH" else ()
    )
