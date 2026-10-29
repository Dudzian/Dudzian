"""Bounded Zstandard transport never confers source or execution authority."""

from __future__ import annotations

import copy
import gzip
import io
import os
import struct
import tarfile
import weakref
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import zstandard

from deployment import stage9_observer_offline as offline
from deployment import stage9_observer_ubuntu as ubuntu
from deployment.canonical_json import canonical_json_bytes
from tests.deployment.test_stage9_observer_offline import (
    _Case,
    _ar,
    _blocked,
    _refused,
    _tar,
    case as case,
)

pytestmark = [pytest.mark.security, pytest.mark.unit]


def _compress(raw: bytes, *, content_size: bool = True) -> bytes:
    return zstandard.ZstdCompressor(write_checksum=True, write_content_size=content_size).compress(
        raw
    )


def _archive(
    case: _Case,
    *,
    control: bytes | None = None,
    data: bytes | None = None,
    control_suffix: str = ".zst",
    data_suffix: str = ".zst",
    content_size: bool = True,
) -> tuple[bytes, bytes]:
    control_tar = _tar(
        [
            (
                "control",
                b"Package: python3.12-minimal\nVersion: 3.12.3-1ubuntu0.17\n"
                b"Architecture: amd64\nDescription: inert synthetic fixture\n",
            ),
            ("postinst", b"#!/bin/sh\nexit 99\n"),
        ]
    )
    data_tar = _tar(case.data_entries())

    def encode(raw: bytes, suffix: str) -> bytes:
        if suffix == ".zst":
            return _compress(raw, content_size=content_size)
        if suffix == ".gz":
            return gzip.compress(raw, mtime=0)
        return raw

    control_body = control if control is not None else encode(control_tar, control_suffix)
    data_body = data if data is not None else encode(data_tar, data_suffix)
    case.set_archive(
        _ar(
            [
                ("debian-binary", b"2.0\n"),
                ("control.tar" + control_suffix, control_body),
                ("data.tar" + data_suffix, data_body),
            ]
        )
    )
    return control_body, data_body


@pytest.mark.parametrize("content_size", [True, False])
@pytest.mark.parametrize("chunk", [512, 4096, 65536])
def test_complete_zstandard_streams_match_bytes_and_keep_all_authority_refused(
    case: _Case, content_size: bool, chunk: int
) -> None:
    _archive(case, content_size=content_size)
    result = case.verify(limits=offline.Limits(chunk_bytes=chunk))
    assert result.artifact_consistency == "MATCH"
    _blocked(result)


@pytest.mark.parametrize(
    ("control_suffix", "data_suffix"),
    [("", ".zst"), (".zst", ""), (".gz", ".zst"), (".zst", ".gz")],
)
def test_zstandard_composes_with_existing_raw_and_gzip_members(
    case: _Case, control_suffix: str, data_suffix: str
) -> None:
    _archive(case, control_suffix=control_suffix, data_suffix=data_suffix)
    result = case.verify()
    assert result.artifact_consistency == "MATCH"
    _blocked(result)


@pytest.mark.parametrize("member", ["control", "data"])
@pytest.mark.parametrize("cut", [0, 1, 4, 5, 8, -1, -4])
def test_truncated_zstandard_refuses_even_after_rebinding_whole_archive_hash(
    case: _Case, member: str, cut: int
) -> None:
    control, data = _archive(case)
    raw = control if member == "control" else data
    _archive(case, **{member: raw[:cut]})
    _refused(case.verify())


@pytest.mark.parametrize("member", ["control", "data"])
def test_frame_checksum_corruption_refuses(case: _Case, member: str) -> None:
    control, data = _archive(case)
    raw = control if member == "control" else data
    _archive(case, **{member: raw[:-1] + bytes([raw[-1] ^ 1])})
    _refused(case.verify(), "FORMAT_REJECTED")


@pytest.mark.parametrize("member", ["control", "data"])
def test_reserved_zstandard_header_bit_refuses(case: _Case, member: str) -> None:
    control, data = _archive(case)
    raw = bytearray(control if member == "control" else data)
    raw[4] |= 8
    _archive(case, **{member: bytes(raw)})
    _refused(case.verify())


@pytest.mark.parametrize(
    "tail",
    [b"\0", b"garbage", _compress(b""), struct.pack("<II", 0x184D2A50, 0)],
    ids=["zero", "garbage", "second-frame", "skippable-frame"],
)
@pytest.mark.parametrize("member", ["control", "data"])
def test_no_bytes_or_additional_frames_are_admitted_after_first_frame(
    case: _Case, member: str, tail: bytes
) -> None:
    control, data = _archive(case)
    raw = control if member == "control" else data
    _archive(case, **{member: raw + tail})
    _refused(case.verify())


@pytest.mark.parametrize(
    "frame",
    [
        b"bad!" + bytes(32),
        struct.pack("<II", 0x184D2A50, 3) + b"abc",
        # Single-segment frame, one-byte dictionary ID 1, content size 0.
        b"\x28\xb5\x2f\xfd\x21\x01\x00\x01\x00\x00",
        # Unknown-content-size frame with a 2 GiB window; no decoder may allocate it.
        b"\x28\xb5\x2f\xfd\x00\xa8\x01\x00\x00",
        # Single-segment frame declares a 1 TiB output before its empty last block.
        b"\x28\xb5\x2f\xfd\xe0" + (1 << 40).to_bytes(8, "little") + b"\x01\x00\x00",
    ],
    ids=["magic", "special", "external-dictionary", "large-window", "large-content-size"],
)
def test_unsupported_or_unbounded_frame_declarations_refuse(case: _Case, frame: bytes) -> None:
    _archive(case, data=frame)
    _refused(case.verify())


def test_decompression_ratio_bomb_refuses_without_whole_output_allocation(case: _Case) -> None:
    # An unknown content size ensures the streaming ratio ceiling is also checked.
    _archive(case, data=_compress(bytes(1024 * 1024), content_size=False))
    _refused(case.verify(), "LIMIT_EXCEEDED")


@pytest.mark.parametrize(
    "limits",
    [
        offline.Limits(max_unpacked_bytes=10240),
        offline.Limits(max_archive_members=2),
        offline.Limits(working_memory_bytes=262144),
        offline.Limits(max_path_depth=2),
    ],
)
def test_zstandard_retains_unpack_member_memory_and_depth_limits(
    case: _Case, limits: offline.Limits
) -> None:
    _archive(case)
    _refused(case.verify(limits=limits), "LIMIT_EXCEEDED")


def test_unpacked_budget_is_shared_between_zstandard_control_and_data(case: _Case) -> None:
    control, data = _archive(case)
    sizes = [zstandard.frame_content_size(body) for body in (control, data)]
    ceiling = sum(sizes) - 1
    assert max(sizes) < ceiling
    _refused(case.verify(limits=offline.Limits(max_unpacked_bytes=ceiling)), "LIMIT_EXCEEDED")


@pytest.mark.parametrize("limit", ["max_unpacked_bytes", "max_archive_members"])
def test_zstandard_global_budget_is_not_reset_for_another_package(case: _Case, limit: str) -> None:
    control, data = _archive(case)
    original = (case.root / case.package.archive_path).read_bytes()
    unpacked = sum(zstandard.frame_content_size(body) for body in (control, data))
    limits = (
        offline.Limits(max_unpacked_bytes=unpacked + 1)
        if limit == "max_unpacked_bytes"
        else offline.Limits(max_archive_members=len(case.manifest["files"]) + 3)
    )
    assert case.verify(limits=limits).artifact_consistency == "MATCH"
    source = copy.deepcopy(case.manifest["sources"][0])
    source["source_id"] = "second-source"
    case.manifest["sources"].append(source)
    case.file("seccomp")["provenance"]["source_id"] = "second-source"
    original_members = case.package.members
    case.package = replace(
        case.package,
        members=tuple(
            replace(member, file_id="") if member.file_id == "seccomp" else member
            for member in original_members
        ),
    )
    second = replace(
        case.package,
        source_id="second-source",
        archive_path="sources/second.deb",
        proof_path="sources/second-proof.bin",
        members=tuple(
            member if member.file_id == "seccomp" else replace(member, file_id="")
            for member in original_members
        ),
    )
    (case.root / second.archive_path).write_bytes(original)
    (case.root / second.proof_path).write_bytes((case.root / case.package.proof_path).read_bytes())
    result = offline.verify_offline(
        case.root,
        manifest_raw=canonical_json_bytes(case.manifest),
        ubuntu_packages=(case.package, second),
        limits=limits,
    )
    _refused(result, "LIMIT_EXCEEDED")


def test_zstandard_does_not_bypass_whole_archive_hash(case: _Case) -> None:
    _archive(case)
    case.manifest["sources"][0]["artifact_sha256"] = "1" * 64
    _refused(case.verify(), "ARTIFACT_MISMATCH")


def test_zstandard_does_not_bypass_regular_member_hash(case: _Case) -> None:
    entries = case.data_entries()
    index = next(index for index, (_, raw) in enumerate(entries) if raw)
    name, raw = entries[index]
    entries[index] = (name, bytes([raw[0] ^ 1]) + raw[1:])
    _archive(case, data=_compress(_tar(entries)))
    _refused(case.verify(), "ARTIFACT_MISMATCH")


@pytest.mark.parametrize(
    "kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, b"x", b"g", b"S", b"?"]
)
def test_zstandard_does_not_expand_supported_tar_record_types(case: _Case, kind: bytes) -> None:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        member = tarfile.TarInfo("unsupported")
        member.type = kind
        if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
            member.linkname = "../../outside"
        archive.addfile(member)
    _archive(case, data=_compress(output.getvalue()[:1536]))
    _refused(case.verify(), "UNSUPPORTED_SOURCE")


def test_zstandard_does_not_bypass_tar_checksum(case: _Case) -> None:
    raw = bytearray(_tar(case.data_entries()))
    raw[0] ^= 1
    _archive(case, data=_compress(bytes(raw)))
    _refused(case.verify(), "FORMAT_REJECTED")


@pytest.mark.parametrize("name", ["../outside", "/outside", "a/../../outside", "a\\outside"])
def test_zstandard_tar_paths_still_refuse_traversal(case: _Case, name: str) -> None:
    _archive(case, data=_compress(_tar(case.data_entries() + [(name, b"inert")])))
    _refused(case.verify(), "FILE_POLICY_REJECTED")


def test_zstandard_does_not_invent_required_directory_members(case: _Case) -> None:
    name = case.file("stdlib-root")["provenance"]["member"] + "/"
    entries = [item for item in case.data_entries() if item[0] != name]
    _archive(case, data=_compress(_tar(entries)))
    _refused(case.verify(), "CLOSURE_INCOMPLETE")


def test_zstandard_does_not_fill_missing_image_directory_from_transport(case: _Case) -> None:
    _archive(case)
    directory = case.file("stdlib-root")
    case.manifest["files"].remove(directory)
    _refused(case.verify(), "CLOSURE_INCOMPLETE")


def test_zstandard_archive_replacement_during_parser_read_refuses(
    case: _Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _archive(case)
    target = case.root / case.package.archive_path
    raw = target.read_bytes()
    inode = target.stat().st_ino
    original = os.pread
    replaced = False

    def replace_during_read(fd: int, amount: int, offset: int) -> bytes:
        nonlocal replaced
        result = original(fd, amount, offset)
        if result and not replaced and os.fstat(fd).st_ino == inode:
            replaced = True
            target.rename(tmp_path / "retained-archive")
            target.write_bytes(raw)
        return result

    monkeypatch.setattr(os, "pread", replace_during_read)
    _refused(case.verify(), "FILE_CHANGED")
    assert replaced


def test_zstandard_compressed_region_short_read_refuses(
    case: _Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    _archive(case)
    original = os.pread

    def short_read(fd: int, amount: int, offset: int) -> bytes:
        raw = original(fd, amount, offset)
        # ar magic/header and debian-binary remain complete; compressed reads do not.
        if amount > 60:
            return raw[:-1]
        return raw

    monkeypatch.setattr(os, "pread", short_read)
    _refused(case.verify(), "PROVENANCE_INCOMPLETE")


def test_zstandard_keeps_deadline_checks(case: _Case, monkeypatch: pytest.MonkeyPatch) -> None:
    _archive(case)
    timestamps = iter([0.0] * 60 + [61.0])
    monkeypatch.setattr(offline.time, "monotonic", lambda: next(timestamps, 61.0))
    _refused(case.verify(), "LIMIT_EXCEEDED")


def test_zstandard_does_not_expand_gnu_tar_metadata_interpretation(case: _Case) -> None:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.GNU_FORMAT) as archive:
        member = tarfile.TarInfo("unsupported-gnu-magic")
        archive.addfile(member)
    _archive(case, data=_compress(output.getvalue()[:1536]))
    _refused(case.verify(), "UNSUPPORTED_SOURCE")


@pytest.mark.parametrize(
    ("attribute", "value"),
    [("__version__", "0.24.0"), ("backend", "cffi"), ("ZSTD_VERSION", (1, 5, 6))],
)
def test_unpinned_decoder_version_or_backend_refuses(
    case: _Case, monkeypatch: pytest.MonkeyPatch, attribute: str, value: Any
) -> None:
    _archive(case)
    monkeypatch.setattr(zstandard, attribute, value)
    _refused(case.verify(), "UNSUPPORTED_SOURCE")


def test_missing_decoder_library_refuses_without_fallback(
    case: _Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    _archive(case)
    original = ubuntu.importlib.import_module

    def import_without_zstandard(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "zstandard":
            raise ModuleNotFoundError("synthetic absent decoder")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(ubuntu.importlib, "import_module", import_without_zstandard)
    _refused(case.verify(), "UNSUPPORTED_SOURCE")


@pytest.mark.parametrize("failure", [zstandard.ZstdError, OSError, MemoryError])
def test_decoder_creation_errors_refuse(
    case: _Case, monkeypatch: pytest.MonkeyPatch, failure: type[Exception]
) -> None:
    _archive(case)

    def fail_decoder(*args: Any, **kwargs: Any) -> Any:
        raise failure("synthetic native decoder failure")

    monkeypatch.setattr(zstandard, "ZstdDecompressor", fail_decoder)
    _refused(case.verify(), _native_reason(failure))


@pytest.mark.parametrize("failure", [zstandard.ZstdError, OSError, MemoryError])
def test_decoder_read_errors_refuse(
    case: _Case, monkeypatch: pytest.MonkeyPatch, failure: type[Exception]
) -> None:
    _archive(case)

    class FailingReader:
        def read(self, size: int) -> bytes:
            raise failure("synthetic native decoder read failure")

        def close(self) -> None:
            pass

    class FailingDecoder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def memory_size(self) -> int:
            return 100000

        def stream_reader(self, *args: Any, **kwargs: Any) -> FailingReader:
            return FailingReader()

    monkeypatch.setattr(zstandard, "ZstdDecompressor", FailingDecoder)
    _refused(case.verify(), _native_reason(failure))


def _native_reason(failure: type[Exception]) -> str:
    if failure is MemoryError:
        return "LIMIT_EXCEEDED"
    if failure is OSError:
        return "PROVENANCE_INCOMPLETE"
    if failure is ubuntu.ContractRefusal:
        return "ARTIFACT_MISMATCH"
    return "FORMAT_REJECTED"


def _tracking_decoder_failures(
    monkeypatch: pytest.MonkeyPatch,
    *,
    read_failure: type[Exception] | None = None,
    close_failure: type[Exception] | None = None,
    fail_close_at: int | None = None,
) -> tuple[list[weakref.ReferenceType[Any]], list[int]]:
    original = zstandard.ZstdDecompressor
    references: list[weakref.ReferenceType[Any]] = []
    closed: list[int] = []

    class Reader:
        def __init__(self, owner: Any, native: Any) -> None:
            self.owner, self.native = owner, native
            references.append(weakref.ref(self))

        def read(self, size: int) -> bytes:
            if read_failure is not None:
                message = (
                    "ARTIFACT_MISMATCH"
                    if read_failure is ubuntu.ContractRefusal
                    else "untrusted native read diagnostic"
                )
                raise read_failure(message)
            return self.native.read(size)

        def close(self) -> None:
            closed.append(1)
            if close_failure is not None and (
                fail_close_at is None or len(closed) == fail_close_at
            ):
                raise close_failure("untrusted native cleanup diagnostic")
            self.native.close()

    class Decoder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.native = original(*args, **kwargs)
            references.append(weakref.ref(self))

        def memory_size(self) -> int:
            return self.native.memory_size()

        def stream_reader(self, *args: Any, **kwargs: Any) -> Reader:
            return Reader(self, self.native.stream_reader(*args, **kwargs))

    monkeypatch.setattr(zstandard, "ZstdDecompressor", Decoder)
    return references, closed


@pytest.mark.parametrize("failure", [zstandard.ZstdError, OSError, MemoryError])
def test_native_cleanup_failure_refuses_and_releases_resources(
    case: _Case, monkeypatch: pytest.MonkeyPatch, failure: type[Exception]
) -> None:
    _archive(case)
    references, closed = _tracking_decoder_failures(monkeypatch, close_failure=failure)
    result = case.verify()
    _refused(result, _native_reason(failure))
    assert "untrusted native" not in repr(result)
    assert closed == [1]
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize(
    "read_failure", [zstandard.ZstdError, OSError, MemoryError, ubuntu.ContractRefusal]
)
@pytest.mark.parametrize("close_failure", [zstandard.ZstdError, OSError, MemoryError])
def test_read_and_cleanup_failure_keep_primary_reason(
    case: _Case,
    monkeypatch: pytest.MonkeyPatch,
    read_failure: type[Exception],
    close_failure: type[Exception],
) -> None:
    _archive(case)
    references, closed = _tracking_decoder_failures(
        monkeypatch, read_failure=read_failure, close_failure=close_failure
    )
    result = case.verify()
    _refused(result, _native_reason(read_failure))
    if read_failure is ubuntu.ContractRefusal:
        assert result.artifact_consistency == "MISMATCH"
    assert set(result.reason_codes) == {
        "TRUST_CONTEXT_MISSING",
        "RUNTIME_LAUNCH_UNVERIFIED",
        "HOST_SCOPE_UNVERIFIED",
        _native_reason(read_failure),
    }
    assert "untrusted native" not in repr(result)
    assert closed == [1]
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("failure", [zstandard.ZstdError, OSError, MemoryError])
def test_member_integrity_failure_survives_native_cleanup_failure(
    case: _Case, monkeypatch: pytest.MonkeyPatch, failure: type[Exception]
) -> None:
    entries = case.data_entries()
    index = next(index for index, (_, raw) in enumerate(entries) if raw)
    name, raw = entries[index]
    entries[index] = (name, bytes([raw[0] ^ 1]) + raw[1:])
    _archive(case, data=_compress(_tar(entries)))
    references, closed = _tracking_decoder_failures(
        monkeypatch, close_failure=failure, fail_close_at=2
    )
    result = case.verify()
    _refused(result, "ARTIFACT_MISMATCH")
    assert result.artifact_consistency == "MISMATCH"
    assert _native_reason(failure) not in result.reason_codes
    assert closed == [1, 1]
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("failure", [None, zstandard.ZstdError, OSError, MemoryError])
def test_native_reader_close_is_idempotent_even_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: type[Exception] | None
) -> None:
    raw = _compress(b"inert bounded decoder fixture")
    monkeypatch.setattr(ubuntu, "_pread", lambda fd, size, offset: raw[offset : offset + size])
    target = tmp_path / "frame.zst"
    target.write_bytes(raw)
    references, closed = _tracking_decoder_failures(monkeypatch, close_failure=failure)
    limits = offline.Limits()
    with target.open("rb") as source:
        reader = ubuntu._ZstdReader(
            ubuntu._Region(source.fileno(), 0, len(raw), limits.chunk_bytes, lambda: None),
            limits,
            lambda: None,
            memory_bytes=4 * 1024 * 1024,
        )
        if failure is None:
            reader.close()
        else:
            with pytest.raises(ubuntu.ContractRefusal, match="^" + _native_reason(failure) + "$"):
                reader.close()
        reader.close()
        assert reader.closed
        assert reader.stream is reader.context is None
    assert closed == [1]
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("failure", [zstandard.ZstdError, OSError, MemoryError])
def test_constructor_post_creation_failure_releases_stream_and_preserves_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: type[Exception]
) -> None:
    raw = _compress(b"inert bounded decoder fixture")
    monkeypatch.setattr(ubuntu, "_pread", lambda fd, size, offset: raw[offset : offset + size])
    target = tmp_path / "frame.zst"
    target.write_bytes(raw)
    references, closed = _tracking_decoder_failures(monkeypatch, close_failure=failure)
    limits = offline.Limits()

    def refuse_after_creation() -> None:
        if len(references) == 2:
            raise ubuntu.ContractRefusal("LIMIT_EXCEEDED")

    with target.open("rb") as source:
        with pytest.raises(ubuntu.ContractRefusal, match="^LIMIT_EXCEEDED$"):
            ubuntu._ZstdReader(
                ubuntu._Region(source.fileno(), 0, len(raw), limits.chunk_bytes, lambda: None),
                limits,
                refuse_after_creation,
                memory_bytes=4 * 1024 * 1024,
            )
    assert closed == [1]
    assert all(reference() is None for reference in references)


def test_archive_verifier_requires_decoder_eof_even_when_compressed_range_was_consumed(
    case: _Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    _archive(case)

    def incomplete_tar(stream: Any, **kwargs: Any) -> tuple[None, int, int]:
        assert stream.read(512)
        assert stream.region.remaining == 0
        return None, 512, 0

    monkeypatch.setattr(ubuntu, "_tar", incomplete_tar)
    _refused(case.verify(), "PROVENANCE_INCOMPLETE")


def test_zstandard_reader_never_requests_unbounded_compressed_or_output_reads(
    case: _Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    _archive(case)
    original_region_read = ubuntu._Region.read
    original_zstd_read = ubuntu._ZstdReader.read
    compressed_sizes: list[int] = []
    output_sizes: list[int] = []

    def region_read(self: Any, size: int = -1) -> bytes:
        compressed_sizes.append(size)
        assert 0 <= size <= 512
        return original_region_read(self, size)

    def zstd_read(self: Any, size: int) -> bytes:
        output_sizes.append(size)
        assert 0 <= size <= 512
        return original_zstd_read(self, size)

    monkeypatch.setattr(ubuntu._Region, "read", region_read)
    monkeypatch.setattr(ubuntu._ZstdReader, "read", zstd_read)
    result = case.verify(limits=offline.Limits(chunk_bytes=512))
    assert result.artifact_consistency == "MATCH"
    assert compressed_sizes and output_sizes
    _blocked(result)


@pytest.mark.parametrize("behavior", ["unconsumed-range", "early-eof", "excess-output"])
def test_native_reader_cannot_claim_complete_or_bounded_processing_without_evidence(
    case: _Case, monkeypatch: pytest.MonkeyPatch, behavior: str
) -> None:
    _archive(case)
    control = _tar(
        [
            (
                "control",
                b"Package: python3.12-minimal\nVersion: 3.12.3-1ubuntu0.17\n"
                b"Architecture: amd64\nDescription: inert synthetic fixture\n",
            ),
            ("postinst", b"#!/bin/sh\nexit 99\n"),
        ]
    )

    class IncompleteDecoder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def memory_size(self) -> int:
            return 100000

        def stream_reader(self, *args: Any, **kwargs: Any) -> Any:
            class Reader:
                def __init__(self) -> None:
                    self.raw = io.BytesIO(control)

                def read(self, size: int) -> bytes:
                    if behavior == "early-eof":
                        return b""
                    if behavior == "excess-output":
                        return bytes(size + 1)
                    return self.raw.read(size)

                def close(self) -> None:
                    self.raw.close()

            return Reader()

    monkeypatch.setattr(zstandard, "ZstdDecompressor", IncompleteDecoder)
    _refused(case.verify())


def test_native_context_exceeding_reserved_memory_refuses(
    case: _Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    _archive(case)

    class ExcessMemoryDecoder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def memory_size(self) -> int:
            return 1 << 30

        def stream_reader(self, *args: Any, **kwargs: Any) -> Any:
            pytest.fail("decoder over memory reservation must refuse before streaming")

    monkeypatch.setattr(zstandard, "ZstdDecompressor", ExcessMemoryDecoder)
    _refused(case.verify(), "LIMIT_EXCEEDED")


def test_zstandard_unused_frame_header_bit_refuses(case: _Case) -> None:
    control, data = _archive(case)
    raw = bytearray(data)
    raw[4] |= 0x10
    _archive(case, control=control, data=bytes(raw))
    _refused(case.verify())


@pytest.mark.parametrize("kind", [0, 1, 2, 3], ids=["raw", "rle", "compressed", "reserved"])
def test_invalid_or_oversized_zstandard_block_header_refuses(case: _Case, kind: int) -> None:
    size = 131073 if kind != 3 else 0
    header = ((size << 3) | (kind << 1) | 1).to_bytes(3, "little")
    body = b"\0" if kind == 1 else bytes(size)
    # Standard frame with unknown content size and a 128 KiB advertised window.
    frame = b"\x28\xb5\x2f\xfd\x00\x38" + header + body
    _archive(case, data=frame)
    _refused(case.verify())


def test_complete_frame_with_invalid_compressed_payload_refuses(case: _Case) -> None:
    header = ((3 << 3) | (2 << 1) | 1).to_bytes(3, "little")
    frame = b"\x28\xb5\x2f\xfd\x00\x38" + header + b"bad"
    _archive(case, data=frame)
    _refused(case.verify(), "FORMAT_REJECTED")


def test_frame_declared_size_must_agree_with_native_output(case: _Case) -> None:
    control, data = _archive(case)
    raw = bytearray(data)
    # Fixture frames are single-segment, have no dictionary field, and carry FCS.
    assert raw[4] & 0x20 and raw[4] & 3 == 0
    raw[5] ^= 1
    _archive(case, control=control, data=bytes(raw))
    _refused(case.verify(), "FORMAT_REJECTED")


def test_native_context_and_stream_release_before_next_archive_member(
    case: _Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    _archive(case)
    original = zstandard.ZstdDecompressor
    contexts: list[weakref.ReferenceType[Any]] = []

    class TrackingStream:
        def __init__(self, owner: Any, native: Any) -> None:
            self.owner, self.native = owner, native

        def read(self, size: int) -> bytes:
            return self.native.read(size)

        def close(self) -> None:
            # Closing a native stream need not release its context reference.
            self.native.close()

    class TrackingDecoder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            assert all(reference() is None for reference in contexts), (
                "previous native context is retained while creating the next decoder"
            )
            self.native = original(*args, **kwargs)
            contexts.append(weakref.ref(self))

        def memory_size(self) -> int:
            return self.native.memory_size()

        def stream_reader(self, *args: Any, **kwargs: Any) -> TrackingStream:
            return TrackingStream(self, self.native.stream_reader(*args, **kwargs))

    monkeypatch.setattr(zstandard, "ZstdDecompressor", TrackingDecoder)
    result = case.verify()
    assert result.artifact_consistency == "MATCH"
    assert len(contexts) == 2
    assert all(reference() is None for reference in contexts)
    _blocked(result)
