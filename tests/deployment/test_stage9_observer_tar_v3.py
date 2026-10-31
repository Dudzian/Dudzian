"""Adversarial inert TAR fixtures; no investigated code or link target executes."""

from __future__ import annotations

import builtins
import copy
import hashlib
import io
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from deployment.stage9_observer_contract import ContractRefusal
from deployment.stage9_observer_offline import Limits
from deployment import stage9_observer_tar_v3 as tar

pytestmark = [pytest.mark.security, pytest.mark.unit]

_CONTROL = b"Package: python3.12-minimal\nVersion: 3.12.3-1ubuntu0.17\nArchitecture: amd64\n"
_Entry = tuple[bytes, bytes, dict[str, Any]]


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _octal(value: int, width: int) -> bytes:
    return f"{value:0{width - 1}o}".encode("ascii") + b"\0"


def _checksum(header: bytes) -> bytes:
    fixed = bytearray(header)
    fixed[148:156] = b" " * 8
    fixed[148:156] = f"{sum(fixed):06o}".encode("ascii") + b"\0 "
    return bytes(fixed)


def _entry(
    name: str,
    kind: str = "0",
    body: bytes = b"payload",
    *,
    target: str = "",
    profile: str = "gnu-inert-v3",
    prefix: str = "",
    mode: int | None = None,
) -> _Entry:
    if kind in ("5", "2"):
        body = b""
    mode = (0o755 if kind == "5" else 0o777 if kind == "2" else 0o644) if mode is None else mode
    header = bytearray(512)
    name_bytes = name.encode("utf-8")
    assert len(name_bytes) <= 100
    header[: len(name_bytes)] = name_bytes
    header[100:108] = _octal(mode, 8)
    header[108:116] = _octal(0, 8)
    header[116:124] = _octal(0, 8)
    header[124:136] = _octal(len(body), 12)
    header[136:148] = _octal(1234, 12)
    header[156:157] = kind.encode("ascii")
    target_bytes = target.encode("utf-8")
    assert len(target_bytes) <= 100
    header[157 : 157 + len(target_bytes)] = target_bytes
    magic = b"ustar  \0" if profile == "gnu-inert-v3" else b"ustar\x0000"
    header[257:265] = magic
    header[265:269] = b"root"
    header[297:301] = b"root"
    if prefix:
        assert profile == "ustar-inert-v3"
        prefix_bytes = prefix.encode("utf-8")
        assert len(prefix_bytes) <= 155
        header[345 : 345 + len(prefix_bytes)] = prefix_bytes
    final = _checksum(bytes(header))
    record = {
        "name": prefix + "/" + name if prefix else name,
        "typeflag": kind,
        "size": len(body),
        "mode": mode,
        "uid": 0,
        "gid": 0,
        "mtime": 1234,
        "linkname": target,
        "magic_version_hex": magic.hex(),
        "uname": "root",
        "gname": "root",
        "devmajor": bytes(8).hex(),
        "devminor": bytes(8).hex(),
        "sha256": _digest(body) if kind in ("0", "\0") else None,
        "tail_hex": final[345:512].hex(),
    }
    return final, body, record


def _archive(entries: list[_Entry], *, trailer: bytes = bytes(1024)) -> bytes:
    return (
        b"".join(header + body + bytes((-len(body)) % 512) for header, body, _ in entries) + trailer
    )


def _declaration(
    raw: bytes, entries: list[_Entry], *, profile: str = "gnu-inert-v3"
) -> dict[str, Any]:
    return {
        "profile": profile,
        "unpacked_bytes": len(raw),
        "unpacked_sha256": _digest(raw),
        "records": [copy.deepcopy(record) for _, _, record in entries],
    }


def _simple(*, profile: str = "gnu-inert-v3") -> tuple[bytes, dict[str, Any]]:
    entries = [
        _entry("./", "5", profile=profile),
        _entry("./control", body=_CONTROL, profile=profile),
    ]
    raw = _archive(entries)
    return raw, _declaration(raw, entries, profile=profile)


def _verify(
    raw: bytes, declaration: dict[str, Any], *, limits: Limits | None = None, **kwargs: Any
) -> Any:
    limits = Limits() if limits is None else limits
    expected = tar.parse_tar_inventory(declaration, limits, lambda: None)
    return tar.verify_tar_v3(
        io.BytesIO(raw),
        expected,
        limits=limits,
        check=lambda: None,
        compressed_size=kwargs.pop("compressed_size", len(raw)),
        **kwargs,
    )


def _rebind_tar(declaration: dict[str, Any], raw: bytes) -> dict[str, Any]:
    changed = copy.deepcopy(declaration)
    changed["unpacked_bytes"] = len(raw)
    changed["unpacked_sha256"] = _digest(raw)
    return changed


def _patch(raw: bytes, offset: int, value: bytes, *, checksum: bool = True) -> bytes:
    changed = bytearray(raw)
    changed[offset : offset + len(value)] = value
    if checksum:
        start = offset // 512 * 512
        changed[start : start + 512] = _checksum(bytes(changed[start : start + 512]))
    return bytes(changed)


@pytest.mark.parametrize("profile", ["gnu-inert-v3", "ustar-inert-v3"])
def test_complete_supported_profiles_are_inert_and_capture_control(profile: str) -> None:
    entries = [
        _entry("./", "5", profile=profile),
        _entry("./control", body=_CONTROL, profile=profile),
        _entry("./empty/", "5", profile=profile),
        _entry("./usr/", "5", profile=profile),
        _entry("./usr/payload", body=b"ordinary bytes", profile=profile),
        _entry("./usr/link", "2", target="/etc/python3.12/sitecustomize.py", profile=profile),
    ]
    raw = _archive(entries)
    result = _verify(raw, _declaration(raw, entries, profile=profile))
    assert (result.members, result.regular, result.directories, result.symlinks) == (6, 2, 3, 1)
    assert result.control == _CONTROL
    assert result.unpacked_bytes == len(raw)
    assert result.sha256 == _digest(raw)


def test_ustar_prefix_is_interpreted_only_for_ustar() -> None:
    entries = [
        _entry("./", "5", profile="ustar-inert-v3"),
        _entry("./usr/", "5", profile="ustar-inert-v3"),
        _entry("file", body=b"x", profile="ustar-inert-v3", prefix="usr"),
    ]
    raw = _archive(entries)
    assert _verify(raw, _declaration(raw, entries, profile="ustar-inert-v3")).members == 3


@pytest.mark.parametrize(
    "target", ["/etc/python3.12/sitecustomize.py", "../outside", "absent", "b", "a"]
)
def test_literal_link_targets_never_reach_filesystem_or_execution_apis(
    monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    entries = [_entry("./", "5"), _entry("a", "2", target=target), _entry("b", "2", target="a")]
    raw = _archive(entries)
    expected = tar.parse_tar_inventory(_declaration(raw, entries), Limits(), lambda: None)
    calls: list[str] = []

    def denied(*args: Any, **kwargs: Any) -> Any:
        calls.append("unexpected target IO or execution")
        raise AssertionError("inert archive parser attempted filesystem IO or execution")

    with monkeypatch.context() as patch:
        for owner, names in (
            (os, ("open", "stat", "lstat", "readlink", "symlink", "system")),
            (Path, ("open", "resolve", "readlink", "symlink_to")),
            (subprocess, ("run", "Popen")),
            (builtins, ("open",)),
        ):
            for name in names:
                patch.setattr(owner, name, denied)
        result = tar.verify_tar_v3(
            io.BytesIO(raw), expected, limits=Limits(), check=lambda: None, compressed_size=len(raw)
        )
    assert result.symlinks == 2
    assert calls == []


@pytest.mark.parametrize(
    "offset,value",
    [
        (100, b"0004644\0"),
        (108, b"\x80" + bytes(7)),
        (116, b"-000001\0"),
        (124, b"00000000009\0"),
        (136, b"77777777777\0"),
        (257, b"ustar\000"),
        (265, b"root\0hidden"),
        (297, b"root\0hidden"),
        (329, b"0000001\0"),
        (337, b"0000001\0"),
        (345, b"1"),
        (357, b"1"),
        (369, b"1"),
        (381, b"1"),
        (385, b"1"),
        (386, b"1"),
        (482, b"1"),
        (483, b"1"),
        (495, b"1"),
        (511, b"1"),
    ],
)
def test_malformed_or_unexpected_headers_refuse_after_rebinding_tar_digest(
    offset: int, value: bytes
) -> None:
    raw, declaration = _simple()
    changed = _patch(raw, offset, value)
    with pytest.raises(ContractRefusal):
        _verify(changed, _rebind_tar(declaration, changed))


@pytest.mark.parametrize("kind", ["1", "3", "4", "6", "7", "L", "K", "S", "x", "g", "V"])
def test_unsupported_typeflags_refuse_even_with_valid_header_checksum(kind: str) -> None:
    raw, declaration = _simple()
    changed = _patch(raw, 512 + 156, kind.encode("ascii"))
    with pytest.raises(ContractRefusal):
        _verify(changed, _rebind_tar(declaration, changed))


def test_checksum_is_not_replaced_by_expected_whole_tar_digest() -> None:
    raw, declaration = _simple()
    changed = _patch(raw, 512 + 100, b"0000600\0", checksum=False)
    with pytest.raises(ContractRefusal):
        _verify(changed, _rebind_tar(declaration, changed))


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", 0o600),
        ("uid", 1),
        ("gid", 1),
        ("mtime", 1235),
        ("uname", "other"),
        ("gname", "other"),
        ("size", 1),
        ("sha256", "1" * 64),
    ],
)
def test_each_declared_member_metadata_and_payload_digest_is_compared(
    field: str, value: Any
) -> None:
    raw, declaration = _simple()
    declaration["records"][1][field] = value
    with pytest.raises(ContractRefusal):
        _verify(raw, declaration)


def test_regular_payload_mutation_is_detected_with_rebound_tar_digest() -> None:
    raw, declaration = _simple()
    changed = _patch(raw, 1024, b"X", checksum=False)
    with pytest.raises(ContractRefusal):
        _verify(changed, _rebind_tar(declaration, changed))


def test_literal_link_target_is_exactly_compared() -> None:
    entries = [_entry("./", "5"), _entry("link", "2", target="/etc/expected")]
    raw = _archive(entries)
    declaration = _declaration(raw, entries)
    changed = _patch(raw, 512 + 157, b"/etc/replaced")
    with pytest.raises(ContractRefusal):
        _verify(changed, _rebind_tar(declaration, changed))


@pytest.mark.parametrize(
    "kind,body,target", [("0", b"extra", ""), ("5", b"", ""), ("2", b"", "missing")]
)
@pytest.mark.parametrize("omitted_from", ["archive", "declaration"])
def test_missing_or_additional_records_of_every_type_refuse(
    kind: str, body: bytes, target: str, omitted_from: str
) -> None:
    base = [_entry("./", "5"), _entry("control", body=_CONTROL)]
    entries = base + [_entry("extra", kind, body, target=target)]
    actual = _archive(base if omitted_from == "archive" else entries)
    declared = entries if omitted_from == "archive" else base
    with pytest.raises(ContractRefusal):
        _verify(actual, _declaration(actual, declared))


@pytest.mark.parametrize("names", [["./", "./"], ["./", "."], ["file", "./file"]])
def test_duplicate_names_and_canonical_aliases_refuse(names: list[str]) -> None:
    entries = [_entry(name, "5" if name in ("./", ".") else "0") for name in names]
    if names[0] not in ("./", "."):
        entries.insert(0, _entry("./", "5"))
    raw = _archive(entries)
    with pytest.raises(ContractRefusal):
        _verify(raw, _declaration(raw, entries))


@pytest.mark.parametrize("kind,target", [("0", ""), ("2", "../outside")])
@pytest.mark.parametrize("reverse", [False, True])
def test_file_or_symlink_parent_conflicts_refuse_in_both_orders(
    kind: str, target: str, reverse: bool
) -> None:
    entries = [_entry("a", kind, target=target), _entry("a/child")]
    if reverse:
        entries.reverse()
    entries.insert(0, _entry("./", "5"))
    raw = _archive(entries)
    with pytest.raises(ContractRefusal):
        _verify(raw, _declaration(raw, entries))


@pytest.mark.parametrize("names", [["file"], ["./", "a/file"]])
def test_root_and_every_parent_directory_must_be_explicit(names: list[str]) -> None:
    entries = [_entry(name, "5" if name == "./" else "0") for name in names]
    raw = _archive(entries)
    with pytest.raises(ContractRefusal):
        _verify(raw, _declaration(raw, entries))


@pytest.mark.parametrize(
    "name",
    ["/absolute", "../outside", "a/../b", "a//b", "a/./b", "a\\b", "bad\nname", "bad\x7fname"],
)
def test_member_paths_remain_restricted_even_when_targets_are_literal(name: str) -> None:
    entries = [_entry("./", "5"), _entry(name)]
    raw = _archive(entries)
    with pytest.raises(ContractRefusal):
        _verify(raw, _declaration(raw, entries))


@pytest.mark.parametrize("cut", [1, 511, 512, 513, 1023, 1024, 1537])
def test_truncated_payload_headers_or_trailer_refuse(cut: int) -> None:
    raw, declaration = _simple()
    changed = raw[:-cut]
    with pytest.raises(ContractRefusal):
        _verify(changed, _rebind_tar(declaration, changed))


@pytest.mark.parametrize("tail", [b"\0", bytes(513), b"X" + bytes(511)])
def test_trailer_requires_zero_blocks_and_alignment(tail: bytes) -> None:
    raw, declaration = _simple()
    changed = raw + tail
    with pytest.raises(ContractRefusal):
        _verify(changed, _rebind_tar(declaration, changed))


def test_hidden_second_tar_and_nonzero_body_padding_refuse() -> None:
    raw, declaration = _simple()
    second = raw + _archive([_entry("hidden")])
    padding = _patch(raw, 1024 + len(_CONTROL), b"X", checksum=False)
    for changed in (second, padding):
        with pytest.raises(ContractRefusal):
            _verify(changed, _rebind_tar(declaration, changed))


def test_additional_complete_zero_padding_and_reordered_inventory_are_supported() -> None:
    raw, declaration = _simple()
    changed = raw + bytes(1024)
    declaration = _rebind_tar(declaration, changed)
    declaration["records"].reverse()
    assert _verify(changed, declaration).members == 2


def test_bounded_short_reads_are_supported() -> None:
    class SmallReads(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            return super().read(min(size, 7) if size >= 0 else 7)

    raw, declaration = _simple()
    limits = replace(Limits(), chunk_bytes=512)
    expected = tar.parse_tar_inventory(declaration, limits, lambda: None)
    result = tar.verify_tar_v3(
        SmallReads(raw), expected, limits=limits, check=lambda: None, compressed_size=len(raw)
    )
    assert result.sha256 == _digest(raw)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_archive_members", 1),
        ("max_file_bytes", 1),
        ("max_unpacked_bytes", 1024),
        ("working_memory_bytes", 512),
    ],
)
def test_tightened_resource_limits_refuse(field: str, value: int) -> None:
    raw, declaration = _simple()
    with pytest.raises(ContractRefusal):
        _verify(raw, declaration, limits=replace(Limits(), **{field: value}))


def test_expansion_limit_and_deadline_refuse() -> None:
    raw, declaration = _simple()
    with pytest.raises(ContractRefusal):
        _verify(raw, declaration, compressed_size=1)
    expected = tar.parse_tar_inventory(declaration, Limits(), lambda: None)

    def expired() -> None:
        raise ContractRefusal("LIMIT_EXCEEDED")

    with pytest.raises(ContractRefusal, match="LIMIT_EXCEEDED"):
        tar.verify_tar_v3(
            io.BytesIO(raw), expected, limits=Limits(), check=expired, compressed_size=len(raw)
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("profile", "gnu"),
        ("profile", None),
        ("unpacked_bytes", True),
        ("unpacked_bytes", -1),
        ("unpacked_sha256", "0" * 64),
        ("records", None),
    ],
)
def test_inventory_schema_requires_exact_types_and_profile(field: str, value: Any) -> None:
    raw, declaration = _simple()
    declaration[field] = value
    with pytest.raises(ContractRefusal):
        _verify(raw, declaration)


@pytest.mark.parametrize("field", ["profile", "unpacked_bytes", "unpacked_sha256", "records"])
def test_missing_inventory_fields_refuse(field: str) -> None:
    raw, declaration = _simple()
    del declaration[field]
    with pytest.raises(ContractRefusal):
        _verify(raw, declaration)


@pytest.mark.parametrize(
    "field,value",
    [
        ("size", True),
        ("uid", True),
        ("gid", -1),
        ("mode", 0o4644),
        ("mtime", -1),
        ("sha256", None),
        ("tail_hex", "zz"),
        ("typeflag", "L"),
        ("extra", "unreviewed"),
    ],
)
def test_record_schema_refuses_bad_types_metadata_and_unknown_fields(
    field: str, value: Any
) -> None:
    raw, declaration = _simple()
    declaration["records"][1][field] = value
    with pytest.raises(ContractRefusal):
        _verify(raw, declaration)


def test_unknown_inventory_fields_and_incomplete_record_refuse() -> None:
    raw, declaration = _simple()
    declaration["extra"] = "unknown"
    with pytest.raises(ContractRefusal):
        _verify(raw, declaration)
    del declaration["extra"]
    del declaration["records"][1]["uid"]
    with pytest.raises(ContractRefusal):
        _verify(raw, declaration)


def test_nul_regular_typeflag_and_empty_payload_are_supported() -> None:
    entries = [_entry("./", "5"), _entry("empty", "\0", b"")]
    raw = _archive(entries)
    result = _verify(raw, _declaration(raw, entries))
    assert result.regular == 1
    assert result.control is None


def test_zero_octal_device_fields_are_supported_without_ignoring_raw_encoding() -> None:
    entries = [_entry("./", "5"), _entry("file")]
    header, body, record = entries[1]
    header = _patch(header, 329, b"0000000\0")
    header = _patch(header, 337, b"0000000\0")
    record["devmajor"] = record["devminor"] = b"0000000\0".hex()
    entries[1] = header, body, record
    raw = _archive(entries)
    assert _verify(raw, _declaration(raw, entries)).regular == 1
    declaration = _declaration(raw, entries)
    declaration["records"][1]["devmajor"] = bytes(8).hex()
    with pytest.raises(ContractRefusal):
        _verify(raw, declaration)


@pytest.mark.parametrize("field,value", [("unpacked_sha256", "1" * 64), ("unpacked_bytes", 4096)])
def test_exact_tar_digest_and_size_are_enforced(field: str, value: Any) -> None:
    raw, declaration = _simple()
    assert value != declaration[field]
    declaration[field] = value
    with pytest.raises(ContractRefusal):
        _verify(raw, declaration)


def test_mixed_header_profiles_and_raw_name_alias_changes_refuse() -> None:
    raw, declaration = _simple()
    mixed = _patch(raw, 512 + 257, b"ustar\x0000")
    alias = _patch(raw, 512, b"control" + bytes(93))
    for changed in (mixed, alias):
        with pytest.raises(ContractRefusal):
            _verify(changed, _rebind_tar(declaration, changed))


def test_duplicate_observed_record_refuses_with_unique_declaration() -> None:
    entries = [_entry("./", "5"), _entry("control", body=_CONTROL)]
    raw = _archive(entries + [entries[1]])
    with pytest.raises(ContractRefusal):
        _verify(raw, _declaration(raw, entries))


@pytest.mark.parametrize("kind", ["5", "2"])
def test_nonregular_records_cannot_carry_payloads_or_regular_digest(kind: str) -> None:
    entries = [_entry("./", "5"), _entry("item", kind, target="target" if kind == "2" else "")]
    raw = _archive(entries)
    declaration = _declaration(raw, entries)
    for field, value in (("size", 1), ("sha256", _digest(b""))):
        changed = copy.deepcopy(declaration)
        changed["records"][1][field] = value
        with pytest.raises(ContractRefusal):
            _verify(raw, changed)


@pytest.mark.parametrize("target", ["", "bad\nline", "bad\0target", "x" * 101, "\ud800"])
def test_link_targets_require_bounded_valid_literal_text(target: str) -> None:
    entries = [_entry("./", "5"), _entry("link", "2", target="valid")]
    raw = _archive(entries)
    declaration = _declaration(raw, entries)
    declaration["records"][1]["linkname"] = target
    with pytest.raises(ContractRefusal):
        _verify(raw, declaration)


def test_regular_and_directory_records_cannot_carry_link_targets() -> None:
    raw, declaration = _simple()
    for index in (0, 1):
        changed = copy.deepcopy(declaration)
        changed["records"][index]["linkname"] = "unreviewed"
        with pytest.raises(ContractRefusal):
            _verify(raw, changed)


@pytest.mark.parametrize("bad", [None, "text", bytearray(b"bytes"), b"X" * 513])
def test_stream_return_contract_refuses_nonbytes_or_overread(bad: Any) -> None:
    class InvalidReads:
        def read(self, size: int = -1) -> Any:
            return bad

    raw, declaration = _simple()
    expected = tar.parse_tar_inventory(declaration, Limits(), lambda: None)
    with pytest.raises(ContractRefusal):
        tar.verify_tar_v3(
            InvalidReads(), expected, limits=Limits(), check=lambda: None, compressed_size=len(raw)
        )


def test_header_text_requires_strict_utf8() -> None:
    raw, declaration = _simple()
    changed = _patch(raw, 512 + 265, b"\xff")
    with pytest.raises(ContractRefusal):
        _verify(changed, _rebind_tar(declaration, changed))


def test_control_buffer_and_path_depth_limits_refuse_before_payload_reads() -> None:
    entries = [_entry("./", "5"), _entry("control", body=b"X" * 65537)]
    raw = _archive(entries)
    with pytest.raises(ContractRefusal):
        _verify(raw, _declaration(raw, entries))
    entries = [_entry("./", "5"), _entry("a/", "5"), _entry("a/child")]
    raw = _archive(entries)
    with pytest.raises(ContractRefusal):
        _verify(raw, _declaration(raw, entries), limits=replace(Limits(), max_path_depth=1))


def test_invalid_direct_dataclass_input_is_revalidated() -> None:
    raw, declaration = _simple()
    expected = tar.parse_tar_inventory(declaration, Limits(), lambda: None)
    for changed in (
        replace(expected, profile="unknown"),
        replace(expected, unpacked_bytes=True),
        replace(expected, records=()),
        replace(expected, records=(replace(expected.records[0], uid=True),)),
    ):
        with pytest.raises(ContractRefusal):
            tar.verify_tar_v3(
                io.BytesIO(raw),
                changed,
                limits=Limits(),
                check=lambda: None,
                compressed_size=len(raw),
            )
