"""Bounded advisory artifact measurements; never a trusted Observer or launcher.

All supplied manifests, policies, proofs and package declarations are untrusted
data. This module reads only the explicitly selected input tree. Matching bytes
does not authenticate their origin, approve expectations or attest execution.
Descriptor and metadata checks detect some changes; they are not complete TOCTOU
protection or a guarantee that measured files stay unchanged after return.
"""

from __future__ import annotations

import hashlib
import math
import os
import stat
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Literal, cast

from deployment import stage9_observer_contract as contract
from deployment.canonical_json import canonical_json_bytes
from deployment.stage9_observer_ubuntu import UbuntuMember, UbuntuPackage, verify_ubuntu_archive

__all__ = ["Limits", "OfflineResult", "UbuntuMember", "UbuntuPackage", "verify_offline"]


@dataclass(frozen=True, slots=True)
class Limits:
    """Local ceilings may tighten, never enlarge, the proposed V1 bounds.

    Memory accounting conservatively bounds retained input/metadata and buffers;
    it is not an operating-system RSS limit. Deadlines are checked around each
    bounded operation and cannot interrupt a blocked kernel filesystem request.
    """

    max_files: int = 4096
    max_file_bytes: int = 64 * 1024 * 1024
    max_total_bytes: int = 256 * 1024 * 1024
    max_archive_members: int = 8192
    max_unpacked_bytes: int = 256 * 1024 * 1024
    chunk_bytes: int = 65536
    max_path_depth: int = 32
    timeout_s: float = 60
    working_memory_bytes: int = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class OfflineResult:
    """An advisory measurement outcome; it has no authorization capability."""

    artifact_consistency: Literal["MATCH", "MISMATCH", "INCOMPLETE"]
    structure: Literal["VALID", "REJECTED"]
    reason_codes: tuple[str, ...]
    report: contract.ParsedContract | None = None
    measured_files: int = 0
    measured_bytes: int = 0
    source_authenticity: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    acceptance: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    execution: Literal["MISSING"] = field(default="MISSING", init=False)
    stage_9: Literal["IN_PROGRESS"] = field(default="IN_PROGRESS", init=False)
    qualification: Literal["BLOCKED"] = field(default="BLOCKED", init=False)
    CONTRACT_APPROVED: Literal["NO"] = field(default="NO", init=False)
    STAGE_3A_COMPLETE: Literal["NO"] = field(default="NO", init=False)
    PRODUCTION_QUALIFIED: Literal["NO"] = field(default="NO", init=False)
    production_status: Literal["PRODUCTION_NO_GO"] = field(default="PRODUCTION_NO_GO", init=False)
    runtime_attestation: Literal["MISSING"] = field(default="MISSING", init=False)
    host_pid_namespace: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    refused: Literal[True] = field(default=True, init=False)


def _require(condition: bool, code: str = "FILE_POLICY_REJECTED") -> None:
    if not condition:
        raise contract.ContractRefusal(code)


def _validate_limits(limits: Limits) -> None:
    _require(type(limits) is Limits, "LIMIT_EXCEEDED")
    for name, maximum in (
        ("max_files", contract.MAX_FILES),
        ("max_file_bytes", 536870912),
        ("max_total_bytes", 2147483648),
        ("max_archive_members", 65536),
        ("max_unpacked_bytes", 2147483648),
        ("chunk_bytes", 1048576),
        ("max_path_depth", 128),
        ("working_memory_bytes", 268435456),
    ):
        value = getattr(limits, name)
        _require(type(value) is int and 1 <= value <= maximum, "LIMIT_EXCEEDED")
    _require(
        type(limits.timeout_s) in (int, float)
        and math.isfinite(limits.timeout_s)
        and 0 < limits.timeout_s <= 60,
        "LIMIT_EXCEEDED",
    )
    _require(limits.chunk_bytes * 4 <= limits.working_memory_bytes, "LIMIT_EXCEEDED")
    _require(limits.chunk_bytes >= 512, "LIMIT_EXCEEDED")


class _Budget:
    def __init__(self, limits: Limits) -> None:
        self.limits = limits
        self.started = time.monotonic()
        self.retained = 0
        self.total_bytes = 0
        self.files = 0
        self.unpacked_bytes = 0
        self.archive_members = 0

    def check(self) -> None:
        elapsed = time.monotonic() - self.started
        _require(
            math.isfinite(elapsed) and 0 <= elapsed < self.limits.timeout_s,
            "LIMIT_EXCEEDED",
        )

    def reserve(self, amount: int) -> None:
        self.check()
        self.retained += amount
        _require(self.retained <= self.limits.working_memory_bytes, "LIMIT_EXCEEDED")

    def read(self, amount: int) -> None:
        self.check()
        self.total_bytes += amount
        _require(self.total_bytes <= self.limits.max_total_bytes, "LIMIT_EXCEEDED")


def _snapshot(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _relative_path(value: Any, limits: Limits) -> str:
    _require(type(value) is str and bool(value))
    _require(len(value) <= contract.MAX_PATH_BYTES, "LIMIT_EXCEEDED")
    _require(len(value.encode("utf-8", errors="strict")) <= contract.MAX_PATH_BYTES)
    _require(
        not value.startswith("/")
        and "\\" not in value
        and all(ord(char) >= 32 and not 127 <= ord(char) <= 159 for char in value)
    )
    parts = value.split("/")
    _require(len(parts) <= limits.max_path_depth, "LIMIT_EXCEEDED")
    _require(all(part not in {"", ".", ".."} for part in parts))
    return cast(str, value)


def _absolute_directory(path: str | Path, budget: _Budget) -> int:
    """Pin every ancestor without following a symlink, including the root."""
    _require(isinstance(path, (str, Path)))
    raw = os.fspath(path)
    _require(type(raw) is str and raw.startswith("/") and "\x00" not in raw)
    _require(len(raw) <= contract.MAX_PATH_BYTES, "LIMIT_EXCEEDED")
    _require("\\" not in raw and ".." not in raw.split("/"))
    _require(len(raw.encode("utf-8")) <= contract.MAX_PATH_BYTES, "LIMIT_EXCEEDED")
    parts = [part for part in raw.split("/") if part]
    _require(all(part != "." for part in parts))
    _require(len(parts) <= 128, "LIMIT_EXCEEDED")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in parts:
            budget.check()
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


class _InputReader:
    def __init__(self, path: str | Path, budget: _Budget) -> None:
        self.path = path
        self.budget = budget
        self.root = _absolute_directory(path, budget)
        try:
            self.root_snapshot = _snapshot(os.fstat(self.root))
        except BaseException:
            os.close(self.root)
            raise
        self.baseline: dict[str, tuple[int, ...]] = {}

    def close(self) -> None:
        os.close(self.root)

    def _directory_matches(self, descriptor: int, relative: str) -> None:
        expected = self.baseline.get(relative) if relative else self.root_snapshot
        if expected is not None:
            _require(_snapshot(os.fstat(descriptor)) == expected, "FILE_CHANGED")

    def scan(self, expected: dict[str, str]) -> dict[str, tuple[int, ...]]:
        found: dict[str, tuple[int, ...]] = {}
        regular_identities: set[tuple[int, int]] = set()

        def walk(descriptor: int, prefix: str, depth: int) -> None:
            self.budget.check()
            _require(depth <= self.budget.limits.max_path_depth, "LIMIT_EXCEEDED")
            before = _snapshot(os.fstat(descriptor))
            self._directory_matches(descriptor, prefix)
            with os.scandir(descriptor) as entries:
                for entry in entries:
                    self.budget.check()
                    relative = (prefix + "/" if prefix else "") + entry.name
                    _relative_path(relative, self.budget.limits)
                    _require(len(found) < self.budget.limits.max_files, "LIMIT_EXCEEDED")
                    _require(relative in expected, "ARTIFACT_MISMATCH")
                    info = entry.stat(follow_symlinks=False)
                    _require(info.st_dev == self.root_snapshot[0])
                    if stat.S_ISDIR(info.st_mode):
                        kind = "directory"
                    elif stat.S_ISREG(info.st_mode):
                        kind = "regular"
                        identity = (info.st_dev, info.st_ino)
                        _require(info.st_nlink == 1 and identity not in regular_identities)
                        regular_identities.add(identity)
                    else:
                        raise contract.ContractRefusal("FILE_POLICY_REJECTED")
                    _require(kind == expected[relative], "ARTIFACT_MISMATCH")
                    self.budget.reserve(512 + 8 * len(relative.encode("utf-8")))
                    found[relative] = _snapshot(info)
                    if kind == "directory":
                        child = os.open(
                            entry.name,
                            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=descriptor,
                        )
                        try:
                            _require(_snapshot(os.fstat(child)) == found[relative], "FILE_CHANGED")
                            walk(child, relative, depth + 1)
                        finally:
                            os.close(child)
                        _require(
                            _snapshot(os.stat(entry.name, dir_fd=descriptor, follow_symlinks=False))
                            == found[relative],
                            "FILE_CHANGED",
                        )
            _require(_snapshot(os.fstat(descriptor)) == before, "FILE_CHANGED")

        walk(self.root, "", 0)
        _require(set(found) == set(expected), "PROVENANCE_INCOMPLETE")
        self.budget.check()
        return found

    @contextmanager
    def open_regular(self, relative: str) -> Iterator[int]:
        relative = _relative_path(relative, self.budget.limits)
        parts = relative.split("/")
        parent = os.dup(self.root)
        descriptor: int | None = None
        try:
            prefix = ""
            self._directory_matches(parent, prefix)
            for part in parts[:-1]:
                self.budget.check()
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=parent,
                )
                os.close(parent)
                parent = child
                prefix = (prefix + "/" if prefix else "") + part
                self._directory_matches(parent, prefix)
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                dir_fd=parent,
            )
            before = os.fstat(descriptor)
            _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1)
            _require(_snapshot(before) == self.baseline[relative], "FILE_CHANGED")
            yield descriptor
            self.budget.check()
            _require(_snapshot(os.fstat(descriptor)) == _snapshot(before), "FILE_CHANGED")
            _require(
                _snapshot(os.stat(parts[-1], dir_fd=parent, follow_symlinks=False))
                == _snapshot(before),
                "FILE_CHANGED",
            )
            self._directory_matches(parent, prefix)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent)

    def measure(self, relative: str, size: int | None, digest: str) -> None:
        with self.open_regular(relative) as descriptor:
            actual_size = os.fstat(descriptor).st_size
            _require(actual_size <= self.budget.limits.max_file_bytes, "LIMIT_EXCEEDED")
            _require(size is None or actual_size == size, "ARTIFACT_MISMATCH")
            measured = 0
            hasher = hashlib.sha256()
            while True:
                self.budget.check()
                chunk = os.read(descriptor, self.budget.limits.chunk_bytes)
                self.budget.read(len(chunk))
                if not chunk:
                    break
                measured += len(chunk)
                _require(measured <= actual_size, "FILE_CHANGED")
                hasher.update(chunk)
            _require(measured == actual_size, "FILE_CHANGED")
            _require(hasher.hexdigest() == digest, "ARTIFACT_MISMATCH")
            self.budget.files += 1

    def finish(self, expected: dict[str, str]) -> None:
        current = self.scan(expected)
        _require(current == self.baseline, "FILE_CHANGED")
        rebound = _absolute_directory(self.path, self.budget)
        try:
            _require(_snapshot(os.fstat(rebound)) == self.root_snapshot, "FILE_CHANGED")
        finally:
            os.close(rebound)


def _expected_tree(
    manifest: dict[str, Any], packages: tuple[UbuntuPackage, ...], limits: Limits, budget: _Budget
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    image: dict[str, str] = {}
    transport: dict[str, str] = {}
    files: dict[str, dict[str, Any]] = {}

    def put(tree: dict[str, str], relative: str, kind: str) -> None:
        existing = tree.get(relative)
        _require(existing is None or existing == kind)
        if existing is None:
            _require(len(image) + len(transport) < limits.max_files, "LIMIT_EXCEEDED")
            budget.reserve(512 + len(relative.encode("utf-8")) * 8)
            tree[relative] = kind

    for item in manifest["files"]:
        budget.check()
        relative = _relative_path(item["path"][1:], limits)
        _require(relative not in image, "CLOSURE_INCOMPLETE")
        put(image, relative, item["type"])
        files[item["file_id"]] = item
    # Only manifest records define the image. Never fill a missing directory
    # from the filesystem or from auxiliary artifact paths. This V1 reader also
    # refuses undeclared staging ancestors above a nested image root: the
    # proposed contract does not represent them as image records.
    for relative in image:
        parent = relative.rpartition("/")[0]
        while parent:
            budget.check()
            _require(image.get(parent) == "directory", "CLOSURE_INCOMPLETE")
            parent = parent.rpartition("/")[0]
    image_root = files[manifest["runtime"]["image_root_file_id"]]["path"][1:]

    def put_transport(relative: str, kind: str) -> None:
        _require(
            relative not in image
            and relative != image_root
            and not relative.startswith(image_root + "/"),
            "FILE_POLICY_REJECTED",
        )
        put(transport, relative, kind)

    _require(type(packages) is tuple and len(packages) <= contract.MAX_SOURCES)
    source_ids: set[str] = set()
    for package in packages:
        _require(type(package) is UbuntuPackage)
        _require(
            type(package.source_id) is str
            and len(package.source_id) <= 128
            and package.source_id not in source_ids
        )
        source_ids.add(package.source_id)
        for relative in (package.archive_path, package.proof_path):
            relative = _relative_path(relative, limits)
            _require(relative not in transport, "FILE_POLICY_REJECTED")
            put_transport(relative, "regular")
            parent = relative.rpartition("/")[0]
            while parent:
                budget.check()
                put_transport(parent, "directory")
                parent = parent.rpartition("/")[0]
    # The merged map is only an I/O inventory. Transport entries never acquire
    # manifest metadata, provenance or a runtime role through this merge.
    budget.reserve(128 * (len(image) + len(transport)))
    return image | transport, files


def _metadata_matches(info: tuple[int, ...], record: dict[str, Any]) -> None:
    _require(
        info[3] == record["uid"]
        and info[4] == record["gid"]
        and stat.S_IMODE(info[2]) == record["mode"],
        "ARTIFACT_MISMATCH",
    )


def _report(
    manifest: contract.ProposedContract,
    policy: contract.ProposedContract | None,
    launch: contract.ParsedContract | None,
    consistency: Literal["MATCH", "MISMATCH", "INCOMPLETE"],
    reasons: tuple[str, ...],
) -> contract.ParsedContract | None:
    if policy is None or launch is None:
        return None
    value = {
        "schema": "OfflineVerificationResultV1",
        "version": 1,
        "synthetic": True,
        "manifest_sha256": manifest.sha256,
        "policy_sha256": policy.sha256,
        "launch_policy_sha256": launch.sha256,
        "artifact_consistency": consistency,
        "acceptance": "UNVERIFIED",
        "reason_codes": list(reasons),
        **contract._blocked_fields(),
    }
    return contract.parse_offline_result(canonical_json_bytes(value))


def verify_offline(
    input_directory: str | Path,
    *,
    manifest_raw: bytes,
    policy_raw: bytes | None = None,
    launch_policy_raw: bytes | None = None,
    acceptance_raw: bytes | None = None,
    trust_context: object = None,
    ubuntu_packages: tuple[UbuntuPackage, ...] = (),
    limits: Limits = Limits(),
) -> OfflineResult:
    """Measure staged data against declarations, refusing unsupported evidence.

    Absolute manifest image paths are interpreted beneath input_directory after
    removing exactly their leading slash. Package/proof paths are relative to
    that same tree. Image directories must have explicit manifest records;
    implicit parents are limited to auxiliary paths outside the image namespace.
    Unsupported sources/transforms prevent an overall MATCH. No acceptance or
    signature-shaped data is ever promoted to an independently trusted context.
    """
    reasons = {
        "TRUST_CONTEXT_MISSING",
        "RUNTIME_LAUNCH_UNVERIFIED",
        "HOST_SCOPE_UNVERIFIED",
    }
    structure: Literal["VALID", "REJECTED"] = "REJECTED"
    consistency: Literal["MATCH", "MISMATCH", "INCOMPLETE"] = "INCOMPLETE"
    manifest_record: contract.ProposedContract | None = None
    policy_record: contract.ProposedContract | None = None
    launch_record: contract.ParsedContract | None = None
    bindings_checked = False
    budget: _Budget | None = None
    reader: _InputReader | None = None
    try:
        _validate_limits(limits)
        budget = _Budget(limits)
        # Bound parser allocations before decoding; this is local conservative
        # accounting, not a claim of OS-enforced process memory isolation.
        _require(type(manifest_raw) is bytes, "FORMAT_REJECTED")
        documents = (manifest_raw, policy_raw, launch_policy_raw, acceptance_raw)
        for raw in documents:
            if raw is not None:
                _require(type(raw) is bytes, "FORMAT_REJECTED")
                budget.reserve(len(raw) * 64)
        budget.reserve(limits.chunk_bytes * 4)
        manifest_record = contract.parse_proposed_manifest(manifest_raw)
        budget.check()
        manifest = manifest_record.as_dict()
        if policy_raw is not None:
            policy_record = contract.parse_proposed_acceptance_policy(policy_raw)
        if launch_policy_raw is not None:
            launch_record = contract.parse_launch_policy(launch_policy_raw)
            contract._validate_manifest_launch(manifest, launch_record.as_dict())
        if policy_record is not None:
            policy = policy_record.as_dict()
            _require(policy["target"] == manifest["target"], "ACCEPTANCE_REJECTED")
            _require(policy["allowed_sources"] == manifest["sources"], "ACCEPTANCE_REJECTED")
            _require(
                policy["launch_policy_sha256"] == manifest["launch_policy_sha256"],
                "LAUNCH_POLICY_MISMATCH",
            )
        bindings_checked = policy_record is not None and launch_record is not None
        if trust_context is not None or acceptance_raw is not None:
            if acceptance_raw is not None:
                # Parsing detects malformed declarations; no cryptographic or
                # independently enrolled acceptance backend exists in this V1.
                contract.parse_acceptance(acceptance_raw)
            structure = "VALID"
            raise contract.ContractRefusal("ACCEPTANCE_REJECTED")
        structure = "VALID"
        if sys.platform != "linux":
            raise contract.ContractRefusal("UNSUPPORTED_SOURCE")
        _require(
            hasattr(os, "O_NOFOLLOW")
            and os.open in os.supports_dir_fd
            and os.scandir in os.supports_fd,
            "UNSUPPORTED_SOURCE",
        )
        _require(os.geteuid() != 0, "FILE_POLICY_REJECTED")
        expected, files = _expected_tree(manifest, ubuntu_packages, limits, budget)
        reader = _InputReader(input_directory, budget)
        reader.baseline = reader.scan(expected)
        for file in files.values():
            budget.check()
            relative = file["path"][1:]
            _metadata_matches(reader.baseline[relative], file)
            if file["type"] == "regular":
                reader.measure(relative, file["size"], file["sha256"])
        sources = {source["source_id"]: source for source in manifest["sources"]}
        provided = {package.source_id for package in ubuntu_packages}
        _require(provided <= set(sources), "ARTIFACT_MISMATCH")
        _require(provided == set(sources), "UNSUPPORTED_SOURCE")
        _require(not manifest["transforms"], "UNSUPPORTED_SOURCE")
        interpreter = files[manifest["runtime"]["interpreter_file_id"]]
        interpreter_source = interpreter["provenance"].get("source_id")
        for package in ubuntu_packages:
            source = sources[package.source_id]
            _require(source["kind"] == "ubuntu_apt_snapshot", "UNSUPPORTED_SOURCE")
            if package.source_id == interpreter_source:
                target_interpreter = manifest["target"]["interpreter"]
                _require(
                    target_interpreter["version"] == "3.12.3"
                    and target_interpreter["package_version"] == package.version,
                    "ARTIFACT_MISMATCH",
                )
            reader.measure(package.archive_path, source["artifact_size"], source["artifact_sha256"])
            reader.measure(package.proof_path, None, source["proof_record_sha256"])
            with reader.open_regular(package.archive_path) as descriptor:
                remaining_unpacked = limits.max_unpacked_bytes - budget.unpacked_bytes
                remaining_members = limits.max_archive_members - budget.archive_members
                _require(remaining_unpacked > 0 and remaining_members > 0, "LIMIT_EXCEEDED")
                adapter_limits = replace(
                    limits,
                    working_memory_bytes=limits.working_memory_bytes - budget.retained,
                    max_unpacked_bytes=remaining_unpacked,
                    max_archive_members=remaining_members,
                )
                unpacked, members = verify_ubuntu_archive(
                    descriptor, package, source, files, adapter_limits, budget.check
                )
                budget.unpacked_bytes += unpacked
                budget.archive_members += members
        reader.finish(expected)
        budget.check()
        consistency = "MATCH"
    except contract.ContractRefusal as exc:
        reasons.add(exc.code)
        if exc.code == "ARTIFACT_MISMATCH":
            consistency = "MISMATCH"
    except (OSError, UnicodeError, ValueError, OverflowError, RecursionError, MemoryError):
        # Diagnostics deliberately contain no paths, input text or native error
        # messages. An incomplete read cannot yield a successful measurement.
        reasons.add("PROVENANCE_INCOMPLETE")
    finally:
        if reader is not None:
            try:
                reader.close()
            except OSError:
                reasons.add("PROVENANCE_INCOMPLETE")
                consistency = "INCOMPLETE"
    reason_codes = tuple(sorted(reasons))
    report = (
        _report(manifest_record, policy_record, launch_record, consistency, reason_codes)
        if manifest_record is not None and bindings_checked
        else None
    )
    return OfflineResult(
        consistency,
        structure,
        reason_codes,
        report,
        budget.files if budget else 0,
        budget.total_bytes if budget else 0,
    )
