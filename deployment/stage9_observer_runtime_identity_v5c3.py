"""Bounded static identity checks over externally captured runtime bytes.

No ELF object is loaded, invoked or resolved against the host. An independently
pinned inventory supplies every resolution. Static DT_NEEDED agreement does
not establish the complete runtime import/dlopen closure or VM attestation.
"""

from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass
from typing import Literal

from deployment.stage9_observer_contract import ContractRefusal
from deployment.stage9_observer_pyc_reconstruction_v5c2 import PycReconstructionPolicyV5c2
from deployment.stage9_observer_pyc_source_hash_v5c2 import (
    TARGET_BUILD_ID,
    TARGET_INTERPRETER_SHA256,
)

MAX_RUNTIME_FILE_BYTES = 64 * 1024 * 1024
MAX_RUNTIME_BYTES = 128 * 1024 * 1024
_ROLES = {"interpreter", "library", "loader", "stdlib", "tool", "configuration"}


class RuntimeIdentityRefusalV5c3(ContractRefusal):
    """Keep V1's code vocabulary and expose a separate fixed V5c3 detail."""

    def __init__(self, reason: str) -> None:
        super().__init__("ARTIFACT_MISMATCH")
        self.runtime_reason = reason

    def __str__(self) -> str:
        return self.runtime_reason


@dataclass(frozen=True, slots=True)
class RuntimeFilePinV5c3:
    path: str
    sha256: str | None
    build_id: str | None
    role: Literal["interpreter", "library", "loader", "stdlib", "tool", "configuration"]
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CapturedRuntimeFileV5c3:
    path: str
    data: bytes | None
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RuntimeInventoryV5c3:
    files: tuple[RuntimeFilePinV5c3, ...]
    needed_resolutions: tuple[tuple[str, str, str], ...]
    loader_configuration_paths: tuple[str, ...]
    import_paths: tuple[str, ...]
    provider_configuration_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementV5c3:
    path: str
    sha256: str | None
    build_id: str | None
    needed: tuple[str, ...] = ()
    interpreter_path: str | None = None
    rpath: tuple[str, ...] = ()
    runpath: tuple[str, ...] = ()


def _require(condition: bool, reason: str = "RUNTIME_IDENTITY_REJECTED") -> None:
    if not condition:
        raise RuntimeIdentityRefusalV5c3(reason)


def _path(path: str) -> None:
    _require(type(path) is str and 1 < len(path) <= 4096)
    _require(path.startswith("/") and not any(c in path for c in "\\\x00\n\r"))
    _require(all(ord(c) >= 32 and not 0xD800 <= ord(c) <= 0xDFFF for c in path))
    _require(all(part not in {"", ".", ".."} for part in path[1:].split("/")))


def validate_runtime_inventory_structure_v5c3(expected: RuntimeInventoryV5c3) -> None:
    """Bound every field before recursive dataclass/JSON materialization."""
    _require(type(expected) is RuntimeInventoryV5c3)
    _require(type(expected.files) is tuple and 5 <= len(expected.files) <= 512)
    text_bytes = 0

    def reserve(value: str) -> None:
        nonlocal text_bytes
        text_bytes += len(value.encode("utf-8"))
        _require(text_bytes <= 1024 * 1024, "RUNTIME_MANIFEST_LIMIT_EXCEEDED")

    paths: set[str] = set()
    aliases: set[str] = set()
    for pin in expected.files:
        _require(type(pin) is RuntimeFilePinV5c3)
        _path(pin.path)
        reserve(pin.path)
        _require(pin.path not in paths)
        paths.add(pin.path)
        _require(type(pin.role) is str and pin.role in _ROLES)
        _require(
            pin.sha256 is None
            or (type(pin.sha256) is str and re.fullmatch(r"[0-9a-f]{64}", pin.sha256) is not None)
        )
        _require(pin.sha256 is not None or pin.role == "configuration")
        _require(
            pin.build_id is None
            or (
                type(pin.build_id) is str
                and re.fullmatch(r"(?:[0-9a-f]{2}){4,64}", pin.build_id) is not None
            )
        )
        _require(type(pin.aliases) is tuple and len(pin.aliases) <= 32)
        _require(not pin.aliases or pin.role == "loader")
        for alias in pin.aliases:
            _path(alias)
            reserve(alias)
            _require(alias not in aliases and alias != pin.path)
            aliases.add(alias)
    _require(not paths.intersection(aliases))
    _require(
        type(expected.needed_resolutions) is tuple and len(expected.needed_resolutions) <= 4096
    )
    edges: set[tuple[str, str]] = set()
    for resolution in expected.needed_resolutions:
        _require(type(resolution) is tuple and len(resolution) == 3)
        consumer, name, resolved = resolution
        _path(consumer)
        _path(resolved)
        _require(type(name) is str and 1 <= len(name) <= 4096)
        _require(all(32 < ord(c) < 127 and c not in "/:" for c in name))
        for value in resolution:
            reserve(value)
        _require(consumer in paths and resolved in paths)
        _require((consumer, name) not in edges)
        edges.add((consumer, name))
    config_paths: set[str] = set()
    for group in (expected.loader_configuration_paths, expected.provider_configuration_paths):
        _require(type(group) is tuple and len(group) <= 256)
        for path in group:
            _path(path)
            reserve(path)
            _require(path in paths and path not in config_paths)
            config_paths.add(path)
    _require(len(config_paths) <= 256)
    _require(type(expected.import_paths) is tuple and 1 <= len(expected.import_paths) <= 64)
    imports: set[str] = set()
    for path in expected.import_paths:
        _path(path)
        reserve(path)
        _require(path not in imports)
        imports.add(path)


def _span(raw: bytes, offset: int, size: int) -> bytes:
    _require(0 <= offset <= len(raw) and 0 <= size <= len(raw) - offset, "ELF_REJECTED")
    return raw[offset : offset + size]


def _text(raw: bytes) -> str:
    _require(0 < len(raw) <= 4096 and b"\x00" not in raw, "ELF_REJECTED")
    try:
        result = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise RuntimeIdentityRefusalV5c3("ELF_REJECTED") from exc
    _require(all(32 < ord(c) < 127 for c in result), "ELF_REJECTED")
    return result


def _build_ids(raw: bytes) -> tuple[str, ...]:
    result: list[str] = []
    offset = 0
    count = 0
    while offset < len(raw):
        count += 1
        _require(count <= 1024, "ELF_REJECTED")
        _require(len(raw) - offset >= 12, "ELF_REJECTED")
        name_size, data_size, kind = struct.unpack_from("<III", raw, offset)
        offset += 12
        name = _span(raw, offset, name_size)
        offset += (name_size + 3) & ~3
        data = _span(raw, offset, data_size)
        offset += (data_size + 3) & ~3
        _require(offset <= len(raw), "ELF_REJECTED")
        if name == b"GNU\x00" and kind == 3:
            _require(4 <= len(data) <= 64, "ELF_REJECTED")
            result.append(data.hex())
    return tuple(result)


def inspect_elf_v5c3(
    raw: bytes, *, path: str = "/captured/object", max_file_bytes: int = MAX_RUNTIME_FILE_BYTES
) -> RuntimeMeasurementV5c3:
    """Read only ELF64 little-endian amd64 headers, notes and dynamic strings."""
    _path(path)
    _require(type(max_file_bytes) is int and 64 <= max_file_bytes <= MAX_RUNTIME_FILE_BYTES)
    _require(type(raw) is bytes and 64 <= len(raw) <= max_file_bytes, "ELF_REJECTED")
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", raw)
    ident = header[0]
    _require(ident[:7] == b"\x7fELF\x02\x01\x01", "ELF_REJECTED")
    _require(header[1] in {2, 3} and header[2] == 62 and header[3] == 1, "ELF_REJECTED")
    _require(header[8] == 64 and header[9] == 56 and 1 <= header[10] <= 256, "ELF_REJECTED")
    _require(header[5] >= 64, "ELF_REJECTED")
    _span(raw, header[5], header[9] * header[10])
    segments: list[tuple[int, int, int, int]] = []
    dynamic: bytes | None = None
    interpreter: str | None = None
    builds: list[str] = []
    for index in range(header[10]):
        ph = struct.unpack_from("<IIQQQQQQ", raw, header[5] + index * 56)
        kind, _, offset, address, _, size, memory, _ = ph
        data = _span(raw, offset, size)
        if kind == 1:
            _require(size <= memory, "ELF_REJECTED")
            segments.append((address, offset, size, memory))
        elif kind == 2:
            _require(dynamic is None and size % 16 == 0 and size <= 65536, "ELF_REJECTED")
            dynamic = data
        elif kind == 3:
            _require(interpreter is None and data.endswith(b"\x00"), "ELF_REJECTED")
            interpreter = _text(data[:-1])
            _path(interpreter)
        elif kind == 4:
            builds.extend(_build_ids(data))
    _require(bool(builds) and len(set(builds)) == 1, "ELF_BUILD_ID_MISSING_OR_AMBIGUOUS")
    tags: dict[int, int] = {}
    needed_offsets: list[int] = []
    if dynamic is not None:
        terminated = False
        for offset in range(0, len(dynamic), 16):
            tag, value = struct.unpack_from("<qQ", dynamic, offset)
            if tag == 0:
                terminated = True
                break
            _require(
                tag not in {0x6FFFFEFB, 0x6FFFFEFC, 0x7FFFFFFD, 0x7FFFFFFF}, "ELF_EXTRA_LOADER"
            )
            if tag == 1:
                needed_offsets.append(value)
                _require(len(needed_offsets) <= 256, "ELF_REJECTED")
            elif tag in {5, 10, 15, 29}:
                _require(tag not in tags, "ELF_REJECTED")
                tags[tag] = value
        _require(terminated, "ELF_REJECTED")
    needed: tuple[str, ...] = ()
    rpath: tuple[str, ...] = ()
    runpath: tuple[str, ...] = ()
    if needed_offsets or 15 in tags or 29 in tags:
        _require(5 in tags and 10 in tags and 0 < tags[10] <= 1024 * 1024, "ELF_REJECTED")
        mappings = [
            (offset + tags[5] - address)
            for address, offset, size, _ in segments
            if address <= tags[5] and tags[5] - address + tags[10] <= size
        ]
        _require(len(mappings) == 1, "ELF_REJECTED")
        strings = _span(raw, mappings[0], tags[10])

        def string_at(offset: int) -> str:
            _require(0 <= offset < len(strings), "ELF_REJECTED")
            end = strings.find(b"\x00", offset, min(len(strings), offset + 4097))
            _require(end >= 0, "ELF_REJECTED")
            return _text(strings[offset:end])

        needed = tuple(string_at(offset) for offset in needed_offsets)
        _require(len(set(needed)) == len(needed), "ELF_REJECTED")
        _require(all("/" not in token and ":" not in token for token in needed), "ELF_REJECTED")
        rpath = (string_at(tags[15]),) if 15 in tags else ()
        runpath = (string_at(tags[29]),) if 29 in tags else ()
    return RuntimeMeasurementV5c3(
        path, hashlib.sha256(raw).hexdigest(), builds[0], needed, interpreter, rpath, runpath
    )


def validate_runtime_identity_v5c3(
    expected: RuntimeInventoryV5c3,
    captured: tuple[CapturedRuntimeFileV5c3, ...],
    policy: PycReconstructionPolicyV5c2,
    *,
    environment: tuple[tuple[str, str], ...] = (),
    max_file_bytes: int = MAX_RUNTIME_FILE_BYTES,
) -> tuple[RuntimeMeasurementV5c3, ...]:
    """Validate bytes against an independent local manifest, never VM origin.

    A ``None`` configuration digest pins its observed absence. Captured paths
    must already have been resolved without unapproved symlinks by the external
    controller. The caller must separately verify capture origin and enforcement.
    """
    validate_runtime_inventory_structure_v5c3(expected)
    _require(type(policy) is PycReconstructionPolicyV5c2)
    _require(type(max_file_bytes) is int and 64 <= max_file_bytes <= MAX_RUNTIME_FILE_BYTES)
    _require(type(expected.files) is tuple and 5 <= len(expected.files) <= 512)
    _require(type(captured) is tuple and len(captured) == len(expected.files))
    _require(type(environment) is tuple and len(environment) <= 128)
    names: set[str] = set()
    for entry in environment:
        _require(type(entry) is tuple and len(entry) == 2)
        name, value = entry
        _require(type(name) is str and type(value) is str and len(value) <= 4096)
        _require(name not in names and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is not None)
        names.add(name)
        _require(
            not name.startswith(("LD_", "OPENSSL_", "GLIBC_", "GCONV_", "LOCPATH"))
            and (not name.startswith("PYTHON") or (name == "PYTHONHASHSEED" and value == "0")),
            "LOADER_ENVIRONMENT_REJECTED",
        )
    pins: dict[str, RuntimeFilePinV5c3] = {}
    observed: dict[str, bytes | None] = {}
    aliases: set[str] = set()
    total_bytes = 0
    for pin, capture in zip(expected.files, captured, strict=True):
        _require(type(pin) is RuntimeFilePinV5c3 and type(capture) is CapturedRuntimeFileV5c3)
        _require(type(capture.path) is str)
        _path(pin.path)
        _require(
            pin.path not in pins
            and capture.path == pin.path
            and type(pin.role) is str
            and pin.role in _ROLES
        )
        _require(
            pin.sha256 is None
            or (type(pin.sha256) is str and re.fullmatch(r"[0-9a-f]{64}", pin.sha256) is not None)
        )
        _require(pin.sha256 is not None or pin.role == "configuration")
        _require(type(pin.aliases) is tuple and len(pin.aliases) <= 32)
        _require(capture.aliases == pin.aliases and type(capture.aliases) is tuple)
        _require(not pin.aliases or pin.role == "loader")
        for alias in pin.aliases:
            _path(alias)
            _require(alias not in aliases and alias != pin.path)
            aliases.add(alias)
        _require(capture.data is None or type(capture.data) is bytes)
        _require((capture.data is None) == (pin.sha256 is None), "RUNTIME_FILE_MISSING")
        if capture.data is not None:
            total_bytes += len(capture.data)
            _require(len(capture.data) <= max_file_bytes and total_bytes <= MAX_RUNTIME_BYTES)
            _require(
                hashlib.sha256(capture.data).hexdigest() == pin.sha256, "RUNTIME_HASH_MISMATCH"
            )
        pins[pin.path] = pin
        observed[pin.path] = capture.data
    _require(not aliases.intersection(pins))
    for role in ("interpreter", "loader", "tool"):
        _require(sum(pin.role == role for pin in pins.values()) == 1)
    _require(
        any(
            pin.role == "stdlib" and pin.path == "/usr/lib/python3.12/py_compile.py"
            for pin in pins.values()
        )
    )
    interpreter_pin = next(pin for pin in pins.values() if pin.role == "interpreter")
    loader_pin = next(pin for pin in pins.values() if pin.role == "loader")
    tool_pin = next(pin for pin in pins.values() if pin.role == "tool")
    _require(interpreter_pin.path == "/usr/bin/python3.12", "WRONG_INTERPRETER_PATH")
    _require(tool_pin.path == "/stage9/tool/compile_worker.py", "WRONG_TOOL_PATH")
    _require(
        policy.interpreter_sha256 == interpreter_pin.sha256 == TARGET_INTERPRETER_SHA256
        and policy.interpreter_build_id == interpreter_pin.build_id == TARGET_BUILD_ID,
        "WRONG_INTERPRETER",
    )
    package = policy.interpreter_package
    _require(
        package.package == "python3.12-minimal"
        and package.version == "3.12.3-1ubuntu0.17"
        and package.architecture == "amd64"
        and package.snapshot_sha256 == policy.source.snapshot_sha256,
        "RUNTIME_PACKAGE_MISMATCH",
    )
    expected_libraries = {lib.path: lib for lib in policy.required_libraries}
    _require(len(expected_libraries) == len(policy.required_libraries))
    libraries = {pin.path: pin for pin in pins.values() if pin.role in {"library", "loader"}}
    _require(libraries.keys() == expected_libraries.keys(), "RUNTIME_LIBRARY_MISSING")
    for path, pin in libraries.items():
        library = expected_libraries[path]
        _require(
            pin.sha256 == library.sha256
            and pin.build_id == library.build_id
            and library.package.architecture == "amd64"
            and library.package.snapshot_sha256 == package.snapshot_sha256,
            "RUNTIME_LIBRARY_MISMATCH",
        )
    _require(type(expected.loader_configuration_paths) is tuple)
    _require(type(expected.provider_configuration_paths) is tuple)
    config_paths = expected.loader_configuration_paths + expected.provider_configuration_paths
    for path in config_paths:
        _path(path)
    _require(len(config_paths) <= 256 and len(set(config_paths)) == len(config_paths))
    _require(
        {"/etc/ld.so.preload", "/etc/ld.so.cache", "/etc/ld.so.conf"}
        <= set(expected.loader_configuration_paths),
        "LOADER_CONFIGURATION_MISSING",
    )
    _require(
        "/etc/ssl/openssl.cnf" in expected.provider_configuration_paths,
        "PROVIDER_CONFIGURATION_MISSING",
    )
    _require(
        set(config_paths) == {pin.path for pin in pins.values() if pin.role == "configuration"}
    )
    _require(observed["/etc/ld.so.preload"] in {None, b""}, "PRELOAD_REJECTED")
    _require(type(expected.import_paths) is tuple and 1 <= len(expected.import_paths) <= 64)
    for path in expected.import_paths:
        _path(path)
    _require(len(set(expected.import_paths)) == len(expected.import_paths))
    for pin in pins.values():
        if pin.role == "stdlib":
            _require(any(pin.path.startswith(root + "/") for root in expected.import_paths))
    measurements: list[RuntimeMeasurementV5c3] = []
    for pin in pins.values():
        raw = observed[pin.path]
        native_stdlib = pin.role == "stdlib" and (
            pin.path.endswith(".so") or (raw is not None and raw.startswith(b"\x7fELF"))
        )
        if pin.role in {"interpreter", "library", "loader"} or native_stdlib:
            _require(type(raw) is bytes)
            assert raw is not None
            measured = inspect_elf_v5c3(raw, path=pin.path, max_file_bytes=max_file_bytes)
            _require(measured.build_id == pin.build_id, "RUNTIME_BUILD_ID_MISMATCH")
            _require(not measured.rpath and not measured.runpath, "RUNTIME_SEARCH_PATH_REJECTED")
        else:
            _require(pin.build_id is None)
            measured = RuntimeMeasurementV5c3(pin.path, pin.sha256, None)
        measurements.append(measured)
    by_path = {row.path: row for row in measurements}
    _require(
        by_path[interpreter_pin.path].interpreter_path in (loader_pin.path,) + loader_pin.aliases,
        "UNKNOWN_LOADER",
    )
    _require(by_path[loader_pin.path].interpreter_path is None, "UNKNOWN_LOADER")
    _require(
        type(expected.needed_resolutions) is tuple and len(expected.needed_resolutions) <= 4096
    )
    actual_edges: set[tuple[str, str]] = set()
    for resolution in expected.needed_resolutions:
        _require(type(resolution) is tuple and len(resolution) == 3)
        consumer, name, resolved = resolution
        _require(consumer in by_path and resolved in libraries, "UNKNOWN_LIBRARY")
        _require(name in by_path[consumer].needed, "UNKNOWN_LIBRARY")
        _require((consumer, name) not in actual_edges)
        actual_edges.add((consumer, name))
    required_edges = {(row.path, name) for row in measurements for name in row.needed}
    _require(actual_edges == required_edges, "RUNTIME_LIBRARY_MISSING")
    return tuple(measurements)
