"""Strict package-only TAR inventories; links are never filesystem operations.

The old GNU layout is documented by GNU tar's ``struct oldgnu_header`` in
``src/tar.h`` and the GNU tar manual, "Standard" / "Old GNU Format":
https://www.gnu.org/software/tar/manual/html_node/Standard.html
https://www.gnu.org/software/tar/manual/html_node/Old-GNU-Format.html
Pinned GNU source layout (unofficial mirror of GNU tar):
https://github.com/gnu-mirror-unofficial/tar/blob/5c713540e34f51323b9f63b7459254bc845603ed/src/tar.h#L131
The admitted GNU subset has an entirely NUL extension tail. In particular its
bytes 345..511 are never interpreted as the POSIX prefix. This is a deliberately
narrow profile, not general GNU/PAX/sparse support or an image contract.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, fields, replace
from typing import Any, Callable, cast

from deployment.stage9_observer_contract import ContractRefusal
from deployment.stage9_observer_offline import Limits, _validate_limits

_GNU = "gnu-inert-v3"
_USTAR = "ustar-inert-v3"
_MAGICS = {_GNU: b"ustar  \0", _USTAR: b"ustar\x0000"}
_KINDS = frozenset({"0", "\0", "5", "2"})
_REGULAR = frozenset({"0", "\0"})
_MEMBER_KEYS = frozenset(
    {
        "name",
        "typeflag",
        "size",
        "mode",
        "uid",
        "gid",
        "mtime",
        "linkname",
        "magic_version_hex",
        "uname",
        "gname",
        "devmajor",
        "devminor",
        "sha256",
        "tail_hex",
    }
)


@dataclass(frozen=True, slots=True)
class TarMemberV3:
    """One literal archive record, including inert link data and raw tail bytes."""

    name: str
    typeflag: str
    size: int
    mode: int
    uid: int
    gid: int
    mtime: int
    linkname: str
    magic_version_hex: str
    uname: str
    gname: str
    devmajor: str
    devminor: str
    sha256: str | None
    tail_hex: str

    @property
    def canonical_name(self) -> str:
        return _name(self.name, self.typeflag, 128)


@dataclass(frozen=True, slots=True)
class TarInventoryV3:
    profile: str
    unpacked_bytes: int
    unpacked_sha256: str
    records: tuple[TarMemberV3, ...]


@dataclass(frozen=True, slots=True)
class TarResultV3:
    unpacked_bytes: int
    members: int
    regular: int
    directories: int
    symlinks: int
    sha256: str
    control: bytes | None
    profile: str


def _require(condition: bool, code: str = "FORMAT_REJECTED") -> None:
    if not condition:
        raise ContractRefusal(code)


def _string(value: Any, maximum: int, *, empty: bool = True) -> str:
    _require(type(value) is str)
    _require(len(value) <= maximum, "LIMIT_EXCEEDED")
    try:
        raw = value.encode("utf-8", errors="strict")
    except UnicodeError:
        raise ContractRefusal("FORMAT_REJECTED") from None
    _require(len(raw) <= maximum, "LIMIT_EXCEEDED")
    _require(empty or bool(value))
    _require(all(ord(char) >= 32 and not 127 <= ord(char) <= 159 for char in value))
    return cast(str, value)


def _hex(value: Any, size: int) -> bytes:
    _require(type(value) is str and len(value) == size * 2)
    _require(re.fullmatch(r"[0-9a-f]+", value) is not None)
    return bytes.fromhex(value)


def _digest(value: Any) -> str:
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None)
    _require(value != "0" * 64)
    return cast(str, value)


def _integer(value: Any, maximum: int, code: str = "FORMAT_REJECTED") -> int:
    _require(type(value) is int and value >= 0)
    _require(value <= maximum, code)
    return cast(int, value)


def _name(raw: str, kind: str, depth: int) -> str:
    _string(raw, 256, empty=False)
    _require(not raw.startswith("/") and "\\" not in raw, "FILE_POLICY_REJECTED")
    name = raw[2:] if raw.startswith("./") else raw
    if kind == "5":
        name = name.removesuffix("/")
        if name in {"", "."}:
            _require(raw in {"./", "."}, "FILE_POLICY_REJECTED")
            return "."
    parts = name.split("/")
    _require(len(parts) <= depth, "LIMIT_EXCEEDED")
    _require(all(part not in {"", ".", ".."} for part in parts), "FILE_POLICY_REJECTED")
    return name


def _octal(raw: bytes, *, empty_zero: bool = False) -> int:
    if empty_zero and raw == bytes(len(raw)):
        return 0
    # Spaces may surround digits and NULs may terminate them; embedded padding,
    # base-256, signs and non-octal bytes are never interpreted permissively.
    _require(re.fullmatch(rb" *[0-7]+[\0 ]*", raw) is not None)
    return int(raw.strip(b"\0 "), 8)


def _text(raw: bytes) -> str:
    first, _, remaining = raw.partition(b"\0")
    _require(not remaining.strip(b"\0"))
    try:
        value = first.decode("utf-8", errors="strict")
    except UnicodeError:
        raise ContractRefusal("FORMAT_REJECTED") from None
    return _string(value, len(raw))


def _tail(profile: str, raw: bytes) -> None:
    _require(len(raw) == 167)
    if profile == _GNU:
        # oldgnu_header: atime[12], ctime[12], offset[12], longnames[4],
        # unused[1], sparse[4][24], isextended[1], realsize[12], padding[17].
        # None of these optional GNU features is admitted by this profile.
        _require(raw == bytes(167), "UNSUPPORTED_SOURCE")
    else:
        _text(raw[:155])  # POSIX prefix only in the USTAR profile.
        _require(raw[155:] == bytes(12), "UNSUPPORTED_SOURCE")


def _member(value: Any, profile: str, limits: Limits) -> TarMemberV3:
    _require(type(value) is dict and set(value) == _MEMBER_KEYS)
    kind = value["typeflag"]
    _require(type(kind) is str and kind in _KINDS, "UNSUPPORTED_SOURCE")
    name = _string(value["name"], 100 if profile == _GNU else 256, empty=False)
    _name(name, kind, limits.max_path_depth)
    size = _integer(value["size"], limits.max_file_bytes, "LIMIT_EXCEEDED")
    mode = _integer(value["mode"], 0o777)
    uid = _integer(value["uid"], (1 << 31) - 1)
    gid = _integer(value["gid"], (1 << 31) - 1)
    mtime = _integer(value["mtime"], (1 << 33) - 1)
    link = _string(value["linkname"], 100)
    if kind == "2":
        _require(size == 0 and bool(link))
    elif kind == "5":
        _require(size == 0 and not link)
    else:
        _require(not link)
    magic = _hex(value["magic_version_hex"], 8)
    _require(magic == _MAGICS[profile], "UNSUPPORTED_SOURCE")
    uname, gname = _string(value["uname"], 32), _string(value["gname"], 32)
    major, minor = _hex(value["devmajor"], 8), _hex(value["devminor"], 8)
    _require(
        _octal(major, empty_zero=True) == _octal(minor, empty_zero=True) == 0, "UNSUPPORTED_SOURCE"
    )
    digest = _digest(value["sha256"]) if kind in _REGULAR else None
    if kind not in _REGULAR:
        _require(value["sha256"] is None)
    tail = _hex(value["tail_hex"], 167)
    _tail(profile, tail)
    return TarMemberV3(
        name,
        kind,
        size,
        mode,
        uid,
        gid,
        mtime,
        link,
        magic.hex(),
        uname,
        gname,
        major.hex(),
        minor.hex(),
        digest,
        tail.hex(),
    )


def retained_tar_bytes(inventory: TarInventoryV3) -> int:
    """Conservative retained declaration, lookup and seen-set allowance."""
    return 4096 + sum(_member_memory(record) for record in inventory.records)


def _member_memory(record: TarMemberV3) -> int:
    return 4096 + 8 * (len(record.name.encode("utf-8")) + len(record.linkname.encode("utf-8")))


def _validate_inventory(
    inventory: TarInventoryV3, limits: Limits, check: Callable[[], None]
) -> None:
    _require(type(inventory) is TarInventoryV3)
    _require(type(inventory.profile) is str and inventory.profile in _MAGICS, "UNSUPPORTED_SOURCE")
    _integer(inventory.unpacked_bytes, limits.max_unpacked_bytes, "LIMIT_EXCEEDED")
    _require(inventory.unpacked_bytes >= 1024 and inventory.unpacked_bytes % 512 == 0)
    _digest(inventory.unpacked_sha256)
    _require(type(inventory.records) is tuple and bool(inventory.records))
    _require(len(inventory.records) <= limits.max_archive_members, "LIMIT_EXCEEDED")
    retained = 4096 + limits.chunk_bytes * 4
    _require(
        retained + len(inventory.records) * 4096 <= limits.working_memory_bytes, "LIMIT_EXCEEDED"
    )
    records: dict[str, TarMemberV3] = {}
    for record in inventory.records:
        check()
        _require(type(record) is TarMemberV3)
        clean = _member(
            {field.name: getattr(record, field.name) for field in fields(record)},
            inventory.profile,
            limits,
        )
        retained += _member_memory(clean)
        if clean.canonical_name == "control" and clean.typeflag in _REGULAR:
            _require(clean.size <= 65536, "LIMIT_EXCEEDED")
            retained += clean.size * 2  # bytearray plus immutable return copy.
        _require(retained <= limits.working_memory_bytes, "LIMIT_EXCEEDED")
        name = clean.canonical_name
        _require(name not in records, "CLOSURE_INCOMPLETE")
        records[name] = clean
    _require("." in records and records["."].typeflag == "5", "CLOSURE_INCOMPLETE")
    for name in records:
        parent = name.rpartition("/")[0]
        while parent:
            check()
            _require(parent in records and records[parent].typeflag == "5", "CLOSURE_INCOMPLETE")
            parent = parent.rpartition("/")[0]


def parse_tar_inventory(value: Any, limits: Limits, check: Callable[[], None]) -> TarInventoryV3:
    """Validate an explicit package TAR declaration, never infer missing records."""
    _validate_limits(limits)
    _require(
        type(value) is dict
        and set(value) == {"profile", "unpacked_bytes", "unpacked_sha256", "records"}
    )
    profile = value["profile"]
    _require(type(profile) is str and profile in _MAGICS, "UNSUPPORTED_SOURCE")
    records = value["records"]
    _require(
        type(records) is list and 0 < len(records) <= limits.max_archive_members, "LIMIT_EXCEEDED"
    )
    retained = 4096 + limits.chunk_bytes * 4
    _require(retained + len(records) * 4096 <= limits.working_memory_bytes, "LIMIT_EXCEEDED")
    parsed: list[TarMemberV3] = []
    for record in records:
        check()
        member = _member(record, profile, limits)
        retained += _member_memory(member)
        _require(retained <= limits.working_memory_bytes, "LIMIT_EXCEEDED")
        parsed.append(member)
    result = TarInventoryV3(
        profile,
        _integer(value["unpacked_bytes"], limits.max_unpacked_bytes, "LIMIT_EXCEEDED"),
        _digest(value["unpacked_sha256"]),
        tuple(parsed),
    )
    _validate_inventory(result, limits, check)
    return result


class _Reader:
    def __init__(
        self, stream: Any, limits: Limits, check: Callable[[], None], compressed_size: int
    ) -> None:
        self.stream, self.limits, self.check = stream, limits, check
        self.compressed_size, self.total = compressed_size, 0
        self.digest = hashlib.sha256()

    def read(self, size: int) -> bytes:
        self.check()
        _require(0 <= size <= max(512, self.limits.chunk_bytes), "LIMIT_EXCEEDED")
        data = self.stream.read(size)
        _require(type(data) is bytes and len(data) <= size, "PROVENANCE_INCOMPLETE")
        self.total += len(data)
        _require(
            self.total <= self.limits.max_unpacked_bytes
            and self.total <= self.compressed_size * 100,
            "LIMIT_EXCEEDED",
        )
        self.digest.update(data)
        self.check()
        return cast(bytes, data)

    def exact(self, size: int) -> bytes:
        _require(0 <= size <= 512, "LIMIT_EXCEEDED")
        value = bytearray()
        while len(value) < size:
            part = self.read(size - len(value))
            _require(bool(part), "PROVENANCE_INCOMPLETE")
            value.extend(part)
        return bytes(value)


def _header(header: bytes, profile: str, limits: Limits) -> TarMemberV3:
    _require(header[257:265] == _MAGICS[profile], "UNSUPPORTED_SOURCE")
    _require(_octal(header[148:156]) == sum(header[:148]) + 8 * 32 + sum(header[156:]))
    kind = header[156:157].decode("latin-1")
    _require(kind in _KINDS, "UNSUPPORTED_SOURCE")
    _tail(profile, header[345:512])
    name = _text(header[:100])
    if profile == _USTAR:
        prefix = _text(header[345:500])
        if prefix:
            name = prefix + "/" + name
    # Validate via the same strict schema, with a temporary well-formed digest;
    # the payload digest is computed and compared before accepting the record.
    return _member(
        {
            "name": name,
            "typeflag": kind,
            "size": _octal(header[124:136]),
            "mode": _octal(header[100:108]),
            "uid": _octal(header[108:116]),
            "gid": _octal(header[116:124]),
            "mtime": _octal(header[136:148]),
            "linkname": _text(header[157:257]),
            "magic_version_hex": header[257:265].hex(),
            "uname": _text(header[265:297]),
            "gname": _text(header[297:329]),
            "devmajor": header[329:337].hex(),
            "devminor": header[337:345].hex(),
            "sha256": "1" * 64 if kind in _REGULAR else None,
            "tail_hex": header[345:512].hex(),
        },
        profile,
        limits,
    )


def verify_tar_v3(
    stream: Any,
    expected: TarInventoryV3,
    *,
    limits: Limits,
    check: Callable[[], None],
    compressed_size: int,
) -> TarResultV3:
    """Compare every record and complete TAR bytes without filesystem access."""
    _validate_limits(limits)
    _validate_inventory(expected, limits, check)
    _require(type(compressed_size) is int and compressed_size > 0, "LIMIT_EXCEEDED")
    reader = _Reader(stream, limits, check, compressed_size)
    wanted = {record.canonical_name: record for record in expected.records}
    seen: set[str] = set()
    regular = directories = symlinks = 0
    control: bytes | None = None
    while True:
        header = reader.exact(512)
        if header == bytes(512):
            _require(reader.exact(512) == bytes(512))
            while tail := reader.read(limits.chunk_bytes):
                _require(not tail.strip(b"\0"))
            _require(reader.total % 512 == 0)
            break
        _require(len(seen) < limits.max_archive_members, "LIMIT_EXCEEDED")
        actual = _header(header, expected.profile, limits)
        name = actual.canonical_name
        _require(name not in seen and name in wanted, "CLOSURE_INCOMPLETE")
        seen.add(name)
        declaration = wanted[name]
        _require(replace(actual, sha256=declaration.sha256) == declaration, "ARTIFACT_MISMATCH")
        if actual.typeflag in _REGULAR:
            regular += 1
            digest = hashlib.sha256()
            is_control = name == "control"
            if is_control:
                _require(actual.size <= 65536, "LIMIT_EXCEEDED")
            body = bytearray()
            remaining = actual.size
            while remaining:
                part = reader.read(min(remaining, limits.chunk_bytes))
                _require(bool(part), "PROVENANCE_INCOMPLETE")
                digest.update(part)
                remaining -= len(part)
                if is_control:
                    body.extend(part)
            _require(digest.hexdigest() == declaration.sha256, "ARTIFACT_MISMATCH")
            if is_control:
                control = bytes(body)
            _require(not reader.exact((-actual.size) % 512).strip(b"\0"))
        elif actual.typeflag == "5":
            directories += 1
        else:
            symlinks += 1
    _require(seen == set(wanted), "CLOSURE_INCOMPLETE")
    digest_value = reader.digest.hexdigest()
    _require(
        reader.total == expected.unpacked_bytes and digest_value == expected.unpacked_sha256,
        "ARTIFACT_MISMATCH",
    )
    check()
    return TarResultV3(
        reader.total,
        len(seen),
        regular,
        directories,
        symlinks,
        digest_value,
        control,
        expected.profile,
    )
