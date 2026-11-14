"""Bounded local Release -> Packages -> inert V3 DEB comparisons.

This deliberately narrow provisioned slice is not an APT client. Neither the
declarations nor agreeing bytes authenticate a publisher, signature or runtime.
No signature tool, network, installation, extraction or target execution runs.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from deployment.stage9_observer_contract import ContractRefusal
from deployment.stage9_observer_offline import (
    Limits,
    _Budget,
    _InputReader,
    _relative_path,
    _validate_limits,
)
from deployment.stage9_observer_package_v3 import (
    PackageMeasurementV3,
    UbuntuPackageInventoryV3,
    _parse,
    _verify_archive,
)
from deployment.stage9_observer_ubuntu import _pread

_RELEASE_BYTES = 1024 * 1024
_INDEX_BYTES = 2 * 1024 * 1024
_DEFAULT_LIMITS = Limits()
_RELEASE_FIELDS = frozenset(
    {
        "origin",
        "label",
        "suite",
        "version",
        "codename",
        "date",
        "valid-until",
        "architectures",
        "components",
        "description",
        "acquire-by-hash",
        "sha256",
    }
)
_PACKAGE_FIELDS = frozenset({"package", "version", "architecture", "filename", "sha256", "size"})


@dataclass(frozen=True, slots=True)
class AptDeclarationV4a:
    """Caller-frozen selections and byte identities, without trust authority.

    Paths are input-root relative. The Release checksum row is relative to the
    Release directory. DEBs and Packages Filename values use V3's flat names.
    """

    release_path: str
    release_sha256: str
    release_size: int
    packages_path: str
    packages_sha256: str
    packages_size: int
    packages_format: Literal["Packages", "Packages.gz"]


@dataclass(frozen=True, slots=True)
class AptResultV4a:
    consistency: Literal["MATCH", "MISMATCH", "INCOMPLETE"]
    reason_codes: tuple[str, ...]
    packages: tuple[PackageMeasurementV3, ...] = ()
    profile: Literal["stage9-offline-apt-chain-v4a"] = field(
        default="stage9-offline-apt-chain-v4a", init=False
    )
    scope: Literal["LOCAL_APT_CHAIN_ONLY"] = field(default="LOCAL_APT_CHAIN_ONLY", init=False)
    signature_verification: Literal["NOT_PERFORMED"] = field(default="NOT_PERFORMED", init=False)
    source_authenticity: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    acceptance: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    stage_9: Literal["IN_PROGRESS"] = field(default="IN_PROGRESS", init=False)
    qualification: Literal["BLOCKED"] = field(default="BLOCKED", init=False)
    STAGE_3A_COMPLETE: Literal["NO"] = field(default="NO", init=False)
    PRODUCTION_QUALIFIED: Literal["NO"] = field(default="NO", init=False)
    production_status: Literal["PRODUCTION_NO_GO"] = field(default="PRODUCTION_NO_GO", init=False)
    refused: Literal[True] = field(default=True, init=False)


def _require(condition: bool, code: str = "FORMAT_REJECTED") -> None:
    if not condition:
        raise ContractRefusal(code)


def _path(value: str, limits: Limits) -> str:
    value = _relative_path(value, limits)
    # No URL decoding, Unicode normalization, drive names or spelling aliases.
    _require(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+/-]*", value) is not None, "FILE_POLICY_REJECTED"
    )
    return value


def _digest(value: str) -> None:
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None)
    _require(value != "0" * 64)


def _size(value: str, limits: Limits) -> int:
    _require(re.fullmatch(r"[1-9][0-9]{0,9}", value) is not None)
    size = int(value)
    _require(size <= limits.max_file_bytes, "LIMIT_EXCEEDED")
    return size


def _selection(declaration: AptDeclarationV4a, limits: Limits) -> str:
    _require(type(declaration) is AptDeclarationV4a)
    release = _path(declaration.release_path, limits)
    index = _path(declaration.packages_path, limits)
    _require(release.rsplit("/", 1)[-1] == "Release", "UNSUPPORTED_SOURCE")
    _require(
        type(declaration.packages_format) is str
        and declaration.packages_format in {"Packages", "Packages.gz"},
        "UNSUPPORTED_SOURCE",
    )
    _require(index.rsplit("/", 1)[-1] == declaration.packages_format, "UNSUPPORTED_SOURCE")
    parent = release.rpartition("/")[0]
    prefix = parent + "/" if parent else ""
    _require(index.startswith(prefix), "FILE_POLICY_REJECTED")
    relative = _path(index[len(prefix) :], limits)
    for digest, size, maximum in (
        (declaration.release_sha256, declaration.release_size, _RELEASE_BYTES),
        (declaration.packages_sha256, declaration.packages_size, _INDEX_BYTES),
    ):
        _digest(digest)
        _require(
            type(size) is int and 0 < size <= min(maximum, limits.max_file_bytes), "LIMIT_EXCEEDED"
        )
    return relative


def _tree(paths: tuple[str, ...], limits: Limits, budget: _Budget) -> dict[str, str]:
    expected: dict[str, str] = {}
    for path in paths:
        budget.check()
        path = _path(path, limits)
        _require(path not in expected, "CLOSURE_INCOMPLETE")
        parts = path.split("/")
        for depth in range(1, len(parts) + 1):
            relative = "/".join(parts[:depth])
            kind = "regular" if depth == len(parts) else "directory"
            if relative in expected:
                _require(expected[relative] == kind, "CLOSURE_INCOMPLETE")
            else:
                _require(len(expected) < limits.max_files, "LIMIT_EXCEEDED")
                budget.reserve(512 + 8 * len(relative))
                expected[relative] = kind
    return expected


def _read_metadata(reader: _InputReader, path: str, size: int, digest: str) -> bytes:
    budget = reader.budget
    # Covers retained bytes, joining copies, UTF8 text, bounded lines and maps.
    budget.reserve(size * 16)
    with reader.open_regular(path) as fd:
        _require(os.fstat(fd).st_size == size, "ARTIFACT_MISMATCH")
        content = bytearray()
        hasher = hashlib.sha256()
        while len(content) < size:
            budget.check()
            block = _pread(fd, min(budget.limits.chunk_bytes, size - len(content)), len(content))
            _require(bool(block), "PROVENANCE_INCOMPLETE")
            budget.read(len(block))
            content.extend(block)
            hasher.update(block)
        _require(hasher.hexdigest() == digest, "ARTIFACT_MISMATCH")
        budget.files += 1
        return bytes(content)


def _lines(raw: bytes, budget: _Budget) -> list[str]:
    budget.check()
    text = raw.decode("utf-8", errors="strict")
    _require(text.endswith("\n"))
    _require(all(char == "\n" or 32 <= ord(char) < 127 for char in text))
    lines = text.split("\n")
    for line in lines:
        budget.check()
        _require(len(line) <= 4096, "LIMIT_EXCEEDED")
    return lines


def _field(line: str) -> tuple[str, str]:
    name, colon, value = line.partition(":")
    _require(bool(colon) and re.fullmatch(r"[A-Za-z][A-Za-z0-9-]{0,127}", name) is not None)
    _require(not value or (value.startswith(" ") and value == " " + value[1:].strip()))
    return name.lower(), value[1:] if value else ""


def _release(
    raw: bytes, relative_index: str, declaration: AptDeclarationV4a, limits: Limits, budget: _Budget
) -> None:
    seen: set[str] = set()
    previous = ""
    row: tuple[str, int, str] | None = None
    ended = False
    for line in _lines(raw, budget):
        budget.check()
        if not line:
            ended = True
            continue
        _require(not ended)
        if line.startswith(" "):
            _require(previous == "sha256" and row is None)
            match = re.fullmatch(r" +([0-9a-f]{64}) +([0-9]{1,10}) +([^ ]+)", line)
            _require(match is not None)
            assert match is not None
            digest, size, path = match.groups()
            _digest(digest)
            row = (digest, _size(size, limits), _path(path, limits))
        else:
            key, value = _field(line)
            _require(key not in seen)
            _require(key in _RELEASE_FIELDS, "UNSUPPORTED_SOURCE")
            seen.add(key)
            previous = key
            _require((key == "sha256" and not value) or (key != "sha256" and bool(value)))
    _require("sha256" in seen and row is not None, "PROVENANCE_INCOMPLETE")
    _require(
        row == (declaration.packages_sha256, declaration.packages_size, relative_index),
        "ARTIFACT_MISMATCH",
    )
    budget.archive_members += 1
    _require(budget.archive_members <= limits.max_archive_members, "LIMIT_EXCEEDED")


def _decode_index(
    raw: bytes, declaration: AptDeclarationV4a, limits: Limits, budget: _Budget
) -> bytes:
    maximum = min(_INDEX_BYTES, limits.max_unpacked_bytes - budget.unpacked_bytes)
    if declaration.packages_format == "Packages":
        _require(len(raw) <= maximum, "LIMIT_EXCEEDED")
        budget.unpacked_bytes += len(raw)
        return raw
    # Only ordinary gzip without optional fields; mtimes/XFL/OS are inert.
    _require(len(raw) >= 18 and raw[:3] == b"\x1f\x8b\x08" and raw[3] == 0, "UNSUPPORTED_SOURCE")
    budget.reserve(262144 + limits.chunk_bytes * 4)  # zlib window/context and temporary blocks.
    decoder = zlib.decompressobj(31)
    output = bytearray()
    position = 0
    pending = b""
    while not decoder.eof:
        budget.check()
        if not pending:
            _require(position < len(raw), "PROVENANCE_INCOMPLETE")
            pending = raw[position : position + limits.chunk_bytes]
            position += len(pending)
        allowance = min(limits.chunk_bytes, maximum - len(output) + 1)
        _require(allowance > 0, "LIMIT_EXCEEDED")
        # Reserve before native allocation and bytearray growth; include final
        # bytes/text/line/map copies. No unbounded decompress/flush operation.
        budget.reserve(allowance * 16)
        block = decoder.decompress(pending, allowance)
        budget.retained -= (allowance - len(block)) * 16
        _require(
            len(output) + len(block) <= maximum and len(output) + len(block) <= len(raw) * 100,
            "LIMIT_EXCEEDED",
        )
        output.extend(block)
        pending = decoder.unconsumed_tail
    _require(not decoder.unused_data and not pending and position == len(raw), "FORMAT_REJECTED")
    budget.check()
    budget.unpacked_bytes += len(output)
    return bytes(output)


def _packages(
    raw: bytes, packages: tuple[UbuntuPackageInventoryV3, ...], limits: Limits, budget: _Budget
) -> None:
    wanted = {package.package: package for package in packages}
    seen: set[str] = set()
    fields: dict[str, str] = {}

    def finish() -> None:
        _require(set(fields) == _PACKAGE_FIELDS)
        name = fields["package"]
        _require(name not in seen, "CLOSURE_INCOMPLETE")
        _require(name in wanted, "ARTIFACT_MISMATCH")
        package = wanted[name]
        _path(fields["filename"], limits)
        _digest(fields["sha256"])
        size = _size(fields["size"], limits)
        _require(
            (fields["version"], fields["architecture"], fields["filename"], fields["sha256"], size)
            == (
                package.version,
                package.architecture,
                package.filename,
                package.sha256,
                package.size,
            ),
            "ARTIFACT_MISMATCH",
        )
        budget.archive_members += 1
        _require(budget.archive_members <= limits.max_archive_members, "LIMIT_EXCEEDED")
        seen.add(name)
        fields.clear()

    for line in _lines(raw, budget):
        budget.check()
        if not line:
            if fields:
                finish()
            continue
        key, value = _field(line)
        _require(key not in fields)
        _require(key in _PACKAGE_FIELDS, "UNSUPPORTED_SOURCE")
        _require(bool(value))
        fields[key] = value
    _require(seen == set(wanted), "PROVENANCE_INCOMPLETE")


def verify_apt_provenance_v4a(
    input_directory: str | Path,
    *,
    declaration: AptDeclarationV4a,
    inventory_raw: bytes,
    limits: Limits = _DEFAULT_LIMITS,
) -> AptResultV4a:
    """Compare a complete local slice. Every failure discards all measurements.

    Uses the V3 declaration/parser semantics and the same pinned descriptor for
    whole-DEB hash and archive parsing. Shared budgets cover the entire chain.
    Cooperative accounting/deadlines and stat checks are not an OS sandbox or
    complete protection against filesystem races, blocked I/O or native calls.
    """
    reader: _InputReader | None = None
    budget: _Budget | None = None
    consistency: Literal["MATCH", "MISMATCH", "INCOMPLETE"] = "INCOMPLETE"
    reasons: set[str] = set()
    measurements: tuple[PackageMeasurementV3, ...] = ()
    try:
        _validate_limits(limits)
        if sys.platform != "linux":
            raise ContractRefusal("UNSUPPORTED_SOURCE")
        _require(os.geteuid() != 0, "FILE_POLICY_REJECTED")
        budget = _Budget(limits)
        relative_index = _selection(declaration, limits)
        packages = _parse(inventory_raw, limits, budget)
        budget.reserve(limits.chunk_bytes * 8 + 1048576)
        expected = _tree(
            (
                declaration.release_path,
                declaration.packages_path,
                *(package.filename for package in packages),
            ),
            limits,
            budget,
        )
        reader = _InputReader(input_directory, budget)
        reader.baseline = reader.scan(expected)
        release = _read_metadata(
            reader, declaration.release_path, declaration.release_size, declaration.release_sha256
        )
        _release(release, relative_index, declaration, limits, budget)
        index = _read_metadata(
            reader,
            declaration.packages_path,
            declaration.packages_size,
            declaration.packages_sha256,
        )
        decoded = _decode_index(index, declaration, limits, budget)
        _packages(decoded, packages, limits, budget)
        # Drop text/index buffers before DEB parsing. Reservations stay charged
        # conservatively across the invocation, including subsequent snapshots.
        del release, index, decoded
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
    except (EOFError, zlib.error, UnicodeError, ValueError, OverflowError, RecursionError):
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
        if budget is not None:
            try:
                budget.check()
            except ContractRefusal as exc:
                reasons.add(exc.code)
                consistency = "INCOMPLETE"
    return AptResultV4a(
        consistency, tuple(sorted(reasons)), measurements if consistency == "MATCH" else ()
    )
