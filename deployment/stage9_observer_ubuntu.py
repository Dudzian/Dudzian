"""Restricted inert DEB reader; byte agreement never authenticates Ubuntu.

Only DEB 2.0 with USTAR control/data archives (raw, gzip or bounded Zstandard)
is supported. Zstandard requires the separately installed, exact offline extra.
No extraction, maintainer script, target interpreter or external tool runs.
XZ, links, sparse/PAX/GNU records and other variants refuse in full.
Package/member expectations are caller declarations, never independent approval.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib
import io
import os
import re
import sys
import zlib
from dataclasses import dataclass, replace
from typing import Any, Callable, cast

from deployment.stage9_observer_contract import ContractRefusal

CPYTHON_PACKAGE_VERSION = "3.12.3-1ubuntu0.17"
_PACKAGES = frozenset(
    {
        "python3.12",
        "python3.12-minimal",
        "libpython3.12-minimal",
        "libpython3.12-stdlib",
        "libpython3.12-dev",
    }
)


@dataclass(frozen=True, slots=True)
class UbuntuMember:
    """One regular data member; empty file_id means no image projection."""

    member: str
    file_id: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class UbuntuPackage:
    """Explicit adapter inputs, not a new acceptance or production contract."""

    source_id: str
    archive_path: str
    proof_path: str
    package: str
    version: str
    architecture: str
    members: tuple[UbuntuMember, ...]


def _require(condition: bool, code: str = "FORMAT_REJECTED") -> None:
    if not condition:
        raise ContractRefusal(code)


def _pread(fd: int, size: int, offset: int) -> bytes:
    if sys.platform != "linux":
        raise ContractRefusal("UNSUPPORTED_SOURCE")
    return os.pread(fd, size, offset)


def _member_name(raw: str, *, directory: bool = False) -> str:
    _require(type(raw) is str)
    _require(len(raw) <= 4096, "LIMIT_EXCEEDED")
    name = raw[2:] if raw.startswith("./") else raw
    if directory:
        name = name.removesuffix("/")
        if name in ("", "."):
            return "."
    _require(bool(name) and len(name.encode("utf-8")) <= 4096, "FILE_POLICY_REJECTED")
    _require(
        "\\" not in name
        and all(ord(c) >= 32 and not 127 <= ord(c) <= 159 for c in name)
        and all(part not in ("", ".", "..") for part in name.split("/")),
        "FILE_POLICY_REJECTED",
    )
    return name


class _Region(io.RawIOBase):
    """A descriptor range; archive-controlled sizes never determine allocations."""

    def __init__(self, fd: int, start: int, size: int, chunk: int, check: Callable[[], None]):
        self.fd, self.start, self.remaining = fd, start, size
        self.position, self.chunk, self.check = 0, chunk, check

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        self.check()
        amount = min(self.chunk, self.remaining, self.chunk if size < 0 else size)
        data = _pread(self.fd, amount, self.start + self.position)
        _require(len(data) == amount, "PROVENANCE_INCOMPLETE")
        self.position += amount
        self.remaining -= amount
        self.check()
        return data


def _zstandard() -> Any:
    """Load only the reviewed decoder combination, never install or fall back.

    These version checks identify an implementation, not an accepted binary or
    trust root. Its wheel, native extension and host dependencies still require
    independent provenance acceptance outside this advisory verifier.
    """
    try:
        decoder = importlib.import_module("zstandard")
    except (ImportError, OSError):
        raise ContractRefusal("UNSUPPORTED_SOURCE") from None
    _require(
        getattr(decoder, "__version__", None) == "0.25.0"
        and getattr(decoder, "backend", None) == "cext"
        and getattr(decoder, "ZSTD_VERSION", None) == (1, 5, 7),
        "UNSUPPORTED_SOURCE",
    )
    return decoder


def _zstd_frame(region: _Region, decoder: Any, limits: Any) -> Any:
    """Preflight one complete ordinary frame without retaining its payload.

    stream_reader alone does not prove EOF on truncated input. This independent
    envelope walk proves the exact last-block/checksum boundary before native
    decoding; the native decoder must then validate every block and checksum.
    No skippable, dictionary, magicless or concatenated variant is admitted.
    """
    region.check()
    size = region.remaining
    header = _pread(region.fd, min(18, size), region.start)
    _require(len(header) == min(18, size), "PROVENANCE_INCOMPLETE")
    _require(len(header) >= 6, "PROVENANCE_INCOMPLETE")
    magic = int.from_bytes(header[:4], "little")
    _require(not 0x184D2A50 <= magic <= 0x184D2A5F, "UNSUPPORTED_SOURCE")
    _require(magic == 0xFD2FB528, "UNSUPPORTED_SOURCE")
    # Reserved/unused bits and even a zero-valued dictionary-ID field are
    # outside the deliberately narrow no-dictionary variant.
    _require(header[4] & 0x1B == 0, "UNSUPPORTED_SOURCE")
    params = decoder.get_frame_parameters(header)
    _require(params.dict_id == 0, "UNSUPPORTED_SOURCE")
    if params.content_size != decoder.CONTENTSIZE_UNKNOWN:
        _require(
            params.content_size <= limits.max_unpacked_bytes and params.content_size <= size * 100,
            "LIMIT_EXCEEDED",
        )
    offset = decoder.frame_header_size(header)
    while True:
        region.check()
        _require(offset + 3 <= size, "PROVENANCE_INCOMPLETE")
        block_raw = _pread(region.fd, 3, region.start + offset)
        _require(len(block_raw) == 3, "PROVENANCE_INCOMPLETE")
        block = int.from_bytes(block_raw, "little")
        kind, block_size = (block >> 1) & 3, block >> 3
        _require(kind != 3 and block_size <= 131072)
        offset += 3 + (1 if kind == 1 else block_size)
        _require(offset <= size, "PROVENANCE_INCOMPLETE")
        if block & 1:
            break
    offset += 4 if params.has_checksum else 0
    _require(offset <= size, "PROVENANCE_INCOMPLETE")
    _require(offset == size)
    region.check()
    return params


class _ZstdReader:
    """Bounded output and native window for exactly one preflighted frame.

    This is accounting for the pinned implementation, not an OS RSS sandbox.
    Callers must drain to EOF; close alone does not validate the frame payload.
    """

    def __init__(
        self,
        region: _Region,
        limits: Any,
        check: Callable[[], None],
        *,
        memory_bytes: int,
    ) -> None:
        self.decoder = _zstandard()
        self.region, self.limits, self.check = region, limits, check
        self.compressed_size, self.total = region.remaining, 0
        self.closed, self.finished = False, False
        try:
            self.params = _zstd_frame(region, self.decoder, limits)
            # libzstd 1.5.7 estimateDStreamSize is DCtx + window + at most
            # three 128-KiB blocks + 64 bytes. Round that workspace upward.
            # Python input/output buffers are reserved separately by caller.
            self.native_limit = (
                self.params.window_size
                + self.decoder.estimate_decompression_context_size()
                + 524288
            )
            _require(self.native_limit <= memory_bytes, "LIMIT_EXCEEDED")
            # Despite the Python docstring's KiB wording, the pinned C backend
            # forwards this value to ZSTD_DCtx_setMaxWindowSize in BYTES.
            self.context = self.decoder.ZstdDecompressor(
                max_window_size=max(1024, self.params.window_size)
            )
            _require(self.context.memory_size() <= self.native_limit, "LIMIT_EXCEEDED")
            self.stream = self.context.stream_reader(
                region,
                read_size=limits.chunk_bytes,
                read_across_frames=False,
                closefd=False,
            )
            check()
        except self.decoder.ZstdError:
            raise ContractRefusal("FORMAT_REJECTED") from None
        except MemoryError:
            raise ContractRefusal("LIMIT_EXCEEDED") from None

    def read(self, size: int) -> bytes:
        self.check()
        _require(not self.closed)
        _require(type(size) is int and 0 <= size <= self.limits.chunk_bytes, "LIMIT_EXCEEDED")
        if not size or self.finished:
            return b""
        try:
            data = self.stream.read(size)
            _require(type(data) is bytes and len(data) <= size, "PROVENANCE_INCOMPLETE")
            _require(self.context.memory_size() <= self.native_limit, "LIMIT_EXCEEDED")
        except self.decoder.ZstdError:
            raise ContractRefusal("FORMAT_REJECTED") from None
        except MemoryError:
            raise ContractRefusal("LIMIT_EXCEEDED") from None
        self.total += len(data)
        _require(
            self.total <= self.limits.max_unpacked_bytes
            and self.total <= self.compressed_size * 100,
            "LIMIT_EXCEEDED",
        )
        if not data:
            _require(self.region.remaining == 0, "PROVENANCE_INCOMPLETE")
            if self.params.content_size != self.decoder.CONTENTSIZE_UNKNOWN:
                _require(self.total == self.params.content_size, "PROVENANCE_INCOMPLETE")
            self.finished = True
        self.check()
        return cast(bytes, data)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                self.stream.close()
            finally:
                # Release the native window before constructing a decoder for
                # the next control/data range or the next package.
                del self.stream
                del self.context


class _TarReader:
    def __init__(self, stream: Any, limits: Any, check: Callable[[], None], compressed_size: int):
        self.stream, self.limits, self.check = stream, limits, check
        self.total, self.members, self.compressed_size = 0, 0, compressed_size

    def read(self, size: int) -> bytes:
        self.check()
        _require(0 <= size <= max(512, self.limits.chunk_bytes), "LIMIT_EXCEEDED")
        data = self.stream.read(size)
        _require(type(data) is bytes, "PROVENANCE_INCOMPLETE")
        self.total += len(data)
        _require(
            self.total <= self.limits.max_unpacked_bytes
            and self.total <= self.compressed_size * 100,
            "LIMIT_EXCEEDED",
        )
        self.check()
        return cast(bytes, data)

    def exact(self, size: int) -> bytes:
        # Only fixed TAR blocks/padding are collected here, never member bodies.
        _require(size <= 512, "LIMIT_EXCEEDED")
        data = bytearray()
        while len(data) < size:
            part = self.read(size - len(data))
            _require(bool(part), "PROVENANCE_INCOMPLETE")
            data.extend(part)
        return bytes(data)


def _octal(raw: bytes) -> int:
    value = raw.rstrip(b"\0 ").lstrip(b" ")
    _require(bool(value) and re.fullmatch(rb"[0-7]+", value) is not None)
    return int(value, 8)


def _text(raw: bytes) -> str:
    first, _, rest = raw.partition(b"\0")
    _require(not rest.strip(b"\0"))
    try:
        return first.decode("utf-8", errors="strict")
    except UnicodeError:
        raise ContractRefusal("FORMAT_REJECTED") from None


def _tar(
    stream: Any,
    *,
    limits: Any,
    check: Callable[[], None],
    compressed_size: int,
    expected: dict[str, UbuntuMember] | None,
    required_directories: set[str] | None = None,
) -> tuple[bytes | None, int, int]:
    reader = _TarReader(stream, limits, check, compressed_size)
    seen: set[str] = set()
    regular: set[str] = set()
    directories: set[str] = set()
    _require(len(required_directories or ()) <= limits.max_archive_members, "LIMIT_EXCEEDED")
    allowed_directories = {"."} | (required_directories or set())
    if expected is not None:
        for member in expected:
            parent = member.rpartition("/")[0]
            while parent:
                check()
                if parent not in allowed_directories:
                    _require(
                        len(allowed_directories) < limits.max_archive_members, "LIMIT_EXCEEDED"
                    )
                allowed_directories.add(parent)
                parent = parent.rpartition("/")[0]
    control: bytes | None = None
    while True:
        header = reader.exact(512)
        if header == bytes(512):
            _require(reader.exact(512) == bytes(512))
            while tail := reader.read(limits.chunk_bytes):
                _require(not tail.strip(b"\0"))
            break
        reader.members += 1
        _require(reader.members <= limits.max_archive_members, "LIMIT_EXCEEDED")
        _require(header[257:265] == b"ustar\x0000", "UNSUPPORTED_SOURCE")
        _require(_octal(header[148:156]) == sum(header[:148]) + 8 * 32 + sum(header[156:]))
        kind = header[156:157]
        _require(kind in (b"0", b"\0", b"5"), "UNSUPPORTED_SOURCE")
        _require(not header[157:257].strip(b"\0"), "UNSUPPORTED_SOURCE")
        name = _text(header[:100])
        prefix = _text(header[345:500])
        if prefix:
            name = prefix + "/" + name
        name = _member_name(name, directory=kind == b"5")
        _require(len(name.split("/")) <= limits.max_path_depth, "LIMIT_EXCEEDED")
        _require(name not in seen, "CLOSURE_INCOMPLETE")
        seen.add(name)
        size = _octal(header[124:136])
        _require(size <= limits.max_file_bytes, "LIMIT_EXCEEDED")
        if kind == b"5":
            _require(size == 0)
            if expected is not None:
                _require(name in allowed_directories, "CLOSURE_INCOMPLETE")
            directories.add(name)
            continue
        regular.add(name)
        wanted = expected.get(name) if expected is not None else None
        if expected is not None:
            _require(wanted is not None, "CLOSURE_INCOMPLETE")
            assert wanted is not None
            _require(wanted.size == size, "ARTIFACT_MISMATCH")
        is_control = expected is None and name == "control"
        if is_control:
            _require(size <= 65536, "LIMIT_EXCEEDED")
        body = bytearray()
        digest = hashlib.sha256()
        remaining = size
        while remaining:
            part = reader.read(min(remaining, limits.chunk_bytes))
            _require(bool(part), "PROVENANCE_INCOMPLETE")
            remaining -= len(part)
            digest.update(part)
            if is_control:
                body.extend(part)
        if wanted is not None:
            _require(digest.hexdigest() == wanted.sha256, "ARTIFACT_MISMATCH")
        if is_control:
            control = bytes(body)
        padding = reader.exact((-size) % 512)
        _require(not padding.strip(b"\0"))
    if expected is not None:
        _require(regular == set(expected), "CLOSURE_INCOMPLETE")
        _require((required_directories or set()) <= directories, "CLOSURE_INCOMPLETE")
    else:
        _require(control is not None, "PROVENANCE_INCOMPLETE")
    return control, reader.total, reader.members


def _control(raw: bytes, package: UbuntuPackage, check: Callable[[], None]) -> None:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError:
        raise ContractRefusal("FORMAT_REJECTED") from None
    _require("\0" not in text and "\r" not in text)
    fields: dict[str, str] = {}
    seen: set[str] = set()
    previous = ""
    for line in io.StringIO(text.rstrip("\n")):
        check()
        line = line.rstrip("\n")
        if line.startswith((" ", "\t")):
            _require(bool(previous) and previous not in {"Package", "Version", "Architecture"})
            continue
        key, colon, value = line.partition(":")
        _require(len(key) <= 128 and len(seen) < 128, "LIMIT_EXCEEDED")
        _require(bool(colon) and re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*", key) is not None)
        _require(key.lower() not in seen)
        seen.add(key.lower())
        if key in {"Package", "Version", "Architecture"}:
            fields[key] = value.strip()
        previous = key
    for key, value in (
        ("Package", package.package),
        ("Version", package.version),
        ("Architecture", package.architecture),
    ):
        _require(fields.get(key) == value, "ARTIFACT_MISMATCH")


def verify_ubuntu_archive(
    fd: int,
    package: UbuntuPackage,
    source: dict[str, Any],
    files: dict[str, dict[str, Any]],
    limits: Any,
    check: Callable[[], None],
) -> tuple[int, int]:
    """Verify the whole restricted DEB and its declared regular-member inventory.

    The caller pins and hashes the descriptor/proof and rechecks identity later.
    Opaque proof bytes are not signature, APT freshness or root authentication.
    """
    _require(type(package) is UbuntuPackage and type(package.members) is tuple)
    _require(
        all(
            type(value) is str
            for value in (
                package.source_id,
                package.archive_path,
                package.proof_path,
                package.package,
                package.version,
                package.architecture,
            )
        )
    )
    _require(package.package in _PACKAGES, "UNSUPPORTED_SOURCE")
    _require(
        package.version == CPYTHON_PACKAGE_VERSION and package.architecture == "amd64",
        "ARTIFACT_MISMATCH",
    )
    _require(source["source_id"] == package.source_id and source["kind"] == "ubuntu_apt_snapshot")
    _require(0 < len(package.members) <= limits.max_archive_members, "LIMIT_EXCEEDED")
    # Conservative retained-name/record allowance plus decoder/window buffers.
    retained_memory = (limits.max_archive_members + len(package.members)) * 2048
    buffer_memory = limits.chunk_bytes * 8
    decoder_memory = limits.working_memory_bytes - retained_memory - buffer_memory
    _require(decoder_memory >= 1048576, "LIMIT_EXCEEDED")
    expected: dict[str, UbuntuMember] = {}
    bound_ids: set[str] = set()
    for member in package.members:
        check()
        _require(type(member) is UbuntuMember)
        _require(_member_name(member.member) == member.member)
        # USTAR's prefix/name fields cannot represent a longer UTF-8 path.
        # Refuse before deriving retained ancestor names from declarations.
        _require(len(member.member.encode("utf-8")) <= 256, "UNSUPPORTED_SOURCE")
        _require(len(member.member.split("/")) <= limits.max_path_depth, "LIMIT_EXCEEDED")
        _require(member.member not in expected, "CLOSURE_INCOMPLETE")
        _require(type(member.size) is int and 0 <= member.size <= limits.max_file_bytes)
        _require(
            type(member.sha256) is str
            and re.fullmatch(r"[0-9a-f]{64}", member.sha256) is not None
            and member.sha256 != "0" * 64
        )
        _require(type(member.file_id) is str)
        if member.file_id:
            _require(
                member.file_id in files and member.file_id not in bound_ids, "CLOSURE_INCOMPLETE"
            )
            bound_ids.add(member.file_id)
            file = files[member.file_id]
            _require(
                file["type"] == "regular"
                and file["sha256"] == member.sha256
                and file["size"] == member.size
                and file["provenance"]
                == {
                    "kind": "source_member",
                    "source_id": package.source_id,
                    "member": member.member,
                },
                "ARTIFACT_MISMATCH",
            )
        expected[member.member] = member
    required_ids = {
        key
        for key, value in files.items()
        if value["type"] == "regular" and value["provenance"].get("source_id") == package.source_id
    }
    _require(required_ids == bound_ids, "CLOSURE_INCOMPLETE")
    required_directories = {
        value["provenance"]["member"]
        for value in files.values()
        if value["type"] == "directory"
        and value["provenance"].get("source_id") == package.source_id
    }
    _require(_pread(fd, 8, 0) == b"!<arch>\n", "UNSUPPORTED_SOURCE")
    offset = 8
    archive_size = source["artifact_size"]
    unpacked = 0
    members = 0
    try:
        for index in range(3):
            check()
            header = _pread(fd, 60, offset)
            _require(len(header) == 60, "PROVENANCE_INCOMPLETE")
            _require(header[58:] == b"`\n")
            name = header[:16].rstrip(b" ").removesuffix(b"/")
            size_raw = header[48:58].strip(b" ")
            _require(re.fullmatch(rb"[0-9]{1,10}", size_raw) is not None)
            size = int(size_raw)
            start = offset + 60
            _require(start + size + size % 2 <= archive_size, "PROVENANCE_INCOMPLETE")
            if index == 0:
                _require(name == b"debian-binary" and size == 4, "UNSUPPORTED_SOURCE")
                _require(_pread(fd, 4, start) == b"2.0\n", "UNSUPPORTED_SOURCE")
            else:
                base = b"control.tar" if index == 1 else b"data.tar"
                _require(name in (base, base + b".gz", base + b".zst"), "UNSUPPORTED_SOURCE")
                remaining_limits = replace(
                    limits,
                    max_unpacked_bytes=limits.max_unpacked_bytes - unpacked,
                    max_archive_members=limits.max_archive_members - members,
                )
                _require(
                    remaining_limits.max_unpacked_bytes > 0
                    and remaining_limits.max_archive_members > 0,
                    "LIMIT_EXCEEDED",
                )
                region = _Region(fd, start, size, limits.chunk_bytes, check)
                stream: Any
                if name.endswith(b".zst"):
                    stream = _ZstdReader(
                        region,
                        remaining_limits,
                        check,
                        # Preserve the original fixed allowance for control
                        # bytes, Python reader objects and temporary records,
                        # even when the caller tightens chunk_bytes to 512.
                        memory_bytes=decoder_memory - 1048576,
                    )
                elif name.endswith(b".gz"):
                    stream = gzip.GzipFile(fileobj=region, mode="rb")
                else:
                    stream = region
                try:
                    control, total, count = _tar(
                        stream,
                        limits=remaining_limits,
                        check=check,
                        compressed_size=size,
                        expected=None if index == 1 else expected,
                        required_directories=required_directories if index == 2 else None,
                    )
                    _require(region.remaining == 0, "PROVENANCE_INCOMPLETE")
                finally:
                    stream.close()
                unpacked += total
                members += count
                _require(
                    unpacked <= limits.max_unpacked_bytes and members <= limits.max_archive_members,
                    "LIMIT_EXCEEDED",
                )
                if index == 1:
                    assert control is not None
                    _control(control, package, check)
            if size % 2:
                _require(_pread(fd, 1, start + size) == b"\n")
            offset = start + size + size % 2
        _require(offset == archive_size, "CLOSURE_INCOMPLETE")
        check()
        return unpacked, members
    except (EOFError, gzip.BadGzipFile, UnicodeError, zlib.error):
        raise ContractRefusal("FORMAT_REJECTED") from None
