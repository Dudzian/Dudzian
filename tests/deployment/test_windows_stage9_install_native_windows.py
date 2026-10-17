"""Real Windows x64 handle qualification; no emulator, ACL changes or MSI execution.

The independent kernel32 probes deliberately use their own ctypes signatures.
All writable objects belong to tmp_path. These tests qualify handle semantics;
public protected-tree ACL acceptance is a separate boundary.
"""

from __future__ import annotations

import ctypes
import hashlib
import io
import os
import struct
import zipfile
from pathlib import Path
from typing import Any

import pytest

from deployment import windows_stage9_install_native as native

pytestmark = [
    pytest.mark.qualification,
    pytest.mark.security,
    pytest.mark.skipif(
        os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8,
        reason="real Windows x64 APIs are required; emulation is not qualification evidence",
    ),
]

_INVALID_HANDLE = ctypes.c_void_p(-1).value
_SHARE_ALL = 1 | 2 | 4
_SHARING_VIOLATION = 32
_ACCESS_DENIED = 5
_INVALID_HANDLE_ERROR = 6


class _Win32Probe:
    """An independent caller that attempts mutations against the adapter's pins."""

    def __init__(self) -> None:
        self.kernel = getattr(ctypes, "WinDLL")(
            "kernel32.dll", use_last_error=True, winmode=0x00000800
        )
        pointer, dword = ctypes.c_void_p, ctypes.c_uint32
        signatures: dict[str, tuple[list[Any], Any]] = {
            "CreateFileW": (
                [ctypes.c_wchar_p, dword, dword, pointer, dword, dword, pointer],
                pointer,
            ),
            "CloseHandle": ([pointer], ctypes.c_int),
            "GetFileInformationByHandle": ([pointer, pointer], ctypes.c_int),
            "GetFileInformationByHandleEx": ([pointer, ctypes.c_int, pointer, dword], ctypes.c_int),
            "GetFinalPathNameByHandleW": (
                [pointer, ctypes.c_wchar_p, dword, dword],
                dword,
            ),
            "WriteFile": ([pointer, pointer, dword, pointer, pointer], ctypes.c_int),
            "DeleteFileW": ([ctypes.c_wchar_p], ctypes.c_int),
            "RemoveDirectoryW": ([ctypes.c_wchar_p], ctypes.c_int),
            "MoveFileExW": ([ctypes.c_wchar_p, ctypes.c_wchar_p, dword], ctypes.c_int),
            "DeviceIoControl": (
                [pointer, dword, pointer, dword, pointer, dword, pointer, pointer],
                ctypes.c_int,
            ),
            "GetCurrentProcess": ([], pointer),
            "GetProcessHandleCount": ([pointer, pointer], ctypes.c_int),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.kernel, name)
            function.argtypes, function.restype = arguments, result

    @staticmethod
    def error() -> int:
        return int(getattr(ctypes, "get_last_error")())

    def open(self, path: Path, *, access: int, creation: int = 3) -> tuple[Any, int]:
        handle = self.kernel.CreateFileW(
            str(path), access, _SHARE_ALL, None, creation, 0x02200000, None
        )
        return handle, self.error()

    def close(self, handle: int) -> None:
        assert self.kernel.CloseHandle(handle), self.error()

    def assert_closed(self, handle: int) -> None:
        buffer = ctypes.create_string_buffer(52)
        assert not self.kernel.GetFileInformationByHandle(handle, buffer)
        assert self.error() == _INVALID_HANDLE_ERROR

    def handle_count(self) -> int:
        count = ctypes.c_uint32()
        assert self.kernel.GetProcessHandleCount(
            self.kernel.GetCurrentProcess(), ctypes.byref(count)
        )
        return count.value

    def mutate(self, action: str, target: Path, other: Path) -> tuple[bool, int]:
        if action in {"write", "truncate"}:
            handle, error = self.open(
                target, access=0x40000000, creation=5 if action == "truncate" else 3
            )
            if handle in {None, _INVALID_HANDLE}:
                return False, error
            try:
                data = ctypes.create_string_buffer(b"attacker")
                written = ctypes.c_uint32()
                result = self.kernel.WriteFile(handle, data, 8, ctypes.byref(written), None)
                error = self.error()
                assert result and written.value == 8, error
                return True, error
            finally:
                self.close(handle)
        if action == "delete":
            result = self.kernel.DeleteFileW(str(target))
        elif action == "remove_directory":
            result = self.kernel.RemoveDirectoryW(str(target))
        elif action == "rename":
            result = self.kernel.MoveFileExW(str(target), str(other), 0)
        elif action == "replace":
            result = self.kernel.MoveFileExW(str(other), str(target), 1)
        else:
            raise AssertionError(action)
        return bool(result), self.error()

    def create_junction(self, junction: Path, target: Path) -> None:
        junction.mkdir()
        substitute = ("\\??\\" + str(target)).encode("utf-16-le")
        display = str(target).encode("utf-16-le")
        names = substitute + b"\0\0" + display + b"\0\0"
        mount = struct.pack("<HHHH", 0, len(substitute), len(substitute) + 2, len(display))
        payload = struct.pack("<IHH", 0xA0000003, len(mount) + len(names), 0) + mount + names
        handle, error = self.open(junction, access=0x40000000)
        assert handle not in {None, _INVALID_HANDLE}, error
        try:
            buffer = ctypes.create_string_buffer(payload)
            returned = ctypes.c_uint32()
            result = self.kernel.DeviceIoControl(
                handle, 0x000900A4, buffer, len(payload), None, 0, ctypes.byref(returned), None
            )
            error = self.error()
        finally:
            self.close(handle)
        if not result:
            junction.rmdir()
            if error in {1, 50, 1314}:
                pytest.skip(f"native junction creation unavailable on this runner: Win32 {error}")
            pytest.fail(f"FSCTL_SET_REPARSE_POINT failed: Win32 {error}")


@pytest.fixture
def adapter() -> native._Native:
    # _Native itself also rejects non-AMD64 native system architectures.
    return native._Native()


@pytest.fixture
def probe() -> _Win32Probe:
    return _Win32Probe()


def _inputs(adapter: native._Native, root: Path) -> native.LockedInputs:
    # Only individual pins are exercised here: no host ancestry or ACL bypass.
    return native.LockedInputs(
        adapter,
        artifact_root=root / "artifacts",
        policy_path=root / "policy" / "host.json",
        executor_root=root / "checkout",
        staging_root=root / "staging",
    )


@pytest.mark.parametrize("directory", [False, True])
def test_real_identity_matches_independent_win32_buffers(tmp_path, adapter, probe, directory):
    target, distinct = tmp_path / "protected", tmp_path / "distinct"
    if directory:
        target.mkdir()
        distinct.mkdir()
    else:
        target.write_bytes(b"immutable native input")
        distinct.write_bytes(target.read_bytes())
    assert ctypes.sizeof(native._FileInfo) == 52
    assert ctypes.sizeof(native._BasicInfo) == 40
    assert native._BasicInfo.attributes.offset == 32
    handle = adapter.open(target)
    second = adapter.open(target)
    other = adapter.open(distinct)
    try:
        identity = adapter.identity(handle)
        info = ctypes.create_string_buffer(52)
        basic = ctypes.create_string_buffer(40)
        final = ctypes.create_unicode_buffer(32768)
        assert probe.kernel.GetFileInformationByHandle(handle, info), probe.error()
        assert probe.kernel.GetFileInformationByHandleEx(handle, 0, basic, 40), probe.error()
        length = probe.kernel.GetFinalPathNameByHandleW(handle, final, len(final), 0)
        assert 0 < length < len(final)
        fields = struct.unpack("<13I", info.raw)
        times = struct.unpack("<qqqqI4x", basic.raw)
        assert identity.file_id == (fields[7], fields[11], fields[12])
        assert identity.attributes == fields[0] == times[4]
        assert identity.links == fields[10]
        assert identity.size == fields[8] * 2**32 + fields[9]
        assert (identity.write, identity.change) == (times[2], times[3])
        assert final.value == "\\\\?\\" + identity.final_path
        assert identity.final_path.casefold() == str(target).casefold()
        native._check_identity(target, identity, directory=directory)
        assert adapter.identity(second) == identity
        assert adapter.identity(other).file_id != identity.file_id
    finally:
        for item in (other, second, handle):
            adapter.close(item)
    for item in (other, second, handle):
        probe.assert_closed(item)


@pytest.mark.parametrize("action", ["write", "truncate", "delete", "rename", "replace"])
def test_real_file_pin_denies_mutation_until_release(tmp_path, adapter, probe, action):
    target, other = tmp_path / "source", tmp_path / "other"
    content = b"source must remain byte-for-byte immutable"
    target.write_bytes(content)
    if action == "replace":
        other.write_bytes(b"replacement")
    handle = adapter.open(target)
    try:
        identity = adapter.identity(handle)
        result, error = probe.mutate(action, target, other)
        assert not result, f"{action} unexpectedly succeeded with a live protected handle"
        # MoveFileExW replacement may report access denied for a protected
        # destination. The identical operation must succeed after release.
        expected_errors = (
            {_SHARING_VIOLATION, _ACCESS_DENIED} if action == "replace" else {_SHARING_VIOLATION}
        )
        assert error in expected_errors
        assert adapter.identity(handle) == identity
        assert b"".join(adapter.chunks(handle)) == content
        assert target.read_bytes() == content
    finally:
        adapter.close(handle)
    probe.assert_closed(handle)
    # The same operation succeeds under the same ACL after release. A denial
    # caused merely by an unprivileged token would not satisfy this control.
    result, error = probe.mutate(action, target, other)
    assert result, f"post-release {action} control failed: Win32 {error}"


@pytest.mark.parametrize("action", ["rename", "remove_directory"])
def test_real_directory_pin_denies_removal_until_release(tmp_path, adapter, probe, action):
    target, other = tmp_path / "directory", tmp_path / "renamed"
    target.mkdir()
    handle = adapter.open(target)
    try:
        identity = adapter.identity(handle)
        native._check_identity(target, identity, directory=True)
        result, error = probe.mutate(action, target, other)
        assert not result
        assert error == _SHARING_VIOLATION
        assert target.is_dir()
        assert adapter.identity(handle) == identity
    finally:
        adapter.close(handle)
    result, error = probe.mutate(action, target, other)
    assert result, f"post-release directory control failed: Win32 {error}"


def test_real_pin_rejects_preexisting_writer(tmp_path, adapter, probe):
    target = tmp_path / "already-writable"
    target.write_bytes(b"original")
    writer, error = probe.open(target, access=0x40000000)
    assert writer not in {None, _INVALID_HANDLE}, error
    try:
        with pytest.raises(OSError) as caught:
            adapter.open(target)
        assert caught.value.errno == _SHARING_VIOLATION
        assert target.read_bytes() == b"original"
    finally:
        probe.close(writer)
    handle = adapter.open(target)
    adapter.close(handle)
    probe.assert_closed(handle)


def test_real_handle_chunks_hash_seek_and_zip_reader(tmp_path, adapter, probe):
    payload = b"native payload\0" * 100000
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as package:
        package.writestr("data.bin", payload)
    content = archive.getvalue()
    target = tmp_path / "archive.zip"
    target.write_bytes(content)
    handle = adapter.open(target)
    try:
        chunks = list(adapter.chunks(handle))
        assert len(chunks) > 1
        assert b"".join(chunks) == content
        assert native._handle_sha256(adapter, handle) == hashlib.sha256(content).hexdigest()
        reader = native._HandleReader(adapter, handle)
        assert reader.readable() and reader.seekable()
        assert reader.read(4) == content[:4]
        assert reader.tell() == 4
        assert reader.seek(-2, 1) == 2
        assert reader.read(3) == content[2:5]
        assert reader.seek(-8, 2) == len(content) - 8
        assert reader.read() == content[-8:]
        with pytest.raises(ValueError, match="negative"):
            reader.seek(-1)
        with pytest.raises(ValueError, match="origin"):
            reader.seek(0, 3)
        with zipfile.ZipFile(reader) as package:
            assert package.read("data.bin") == payload
        assert target.read_bytes() == content
    finally:
        adapter.close(handle)
    probe.assert_closed(handle)


def test_real_write_flush_and_truncate_existing_temporary_file(tmp_path, adapter, probe):
    target = tmp_path / "owned-output"
    target.write_bytes(b"old-content-that-must-be-truncated")
    handle = adapter.open(target, writable=True)
    try:
        adapter.write(handle, iter((b"new", b"\0", b"data")))
        assert b"".join(adapter.chunks(handle)) == b"new\0data"
        assert adapter.identity(handle).size == 8
        adapter.write(handle, iter((b"x",)))
        assert adapter.identity(handle).size == 1
        assert b"".join(adapter.chunks(handle)) == b"x"
    finally:
        adapter.close(handle)
    probe.assert_closed(handle)
    assert target.read_bytes() == b"x"


def test_real_hardlink_identity_rejected_without_handle_leaks(tmp_path, adapter, probe):
    target, alias = tmp_path / "original", tmp_path / "hardlink"
    target.write_bytes(b"hardlinked original")
    try:
        os.link(target, alias)
    except OSError as exc:
        if exc.winerror in {1, 50}:
            pytest.skip(f"runner filesystem has no native hardlink support: Win32 {exc.winerror}")
        raise
    handle, linked = adapter.open(target), adapter.open(alias)
    try:
        identity, alias_identity = adapter.identity(handle), adapter.identity(linked)
        assert identity.file_id == alias_identity.file_id
        assert identity.links == alias_identity.links == 2
        with pytest.raises(native.InstallationIntegrityError, match="hardlink"):
            native._check_identity(target, identity, directory=False)
    finally:
        adapter.close(linked)
        adapter.close(handle)
    before = probe.handle_count()
    for _ in range(16):
        inputs = _inputs(adapter, tmp_path)
        try:
            with pytest.raises(native.InstallationIntegrityError, match="hardlink"):
                inputs._pin(alias, directory=False)
            assert inputs._pins == {}
        finally:
            inputs.close()
    assert probe.handle_count() == before
    assert target.read_bytes() == alias.read_bytes() == b"hardlinked original"


@pytest.mark.parametrize("kind", ["file_symlink", "directory_symlink", "junction"])
def test_real_reparse_object_is_observed_and_rejected(tmp_path, adapter, probe, kind):
    target, link = tmp_path / "target", tmp_path / "reparse"
    directory = kind != "file_symlink"
    if directory:
        target.mkdir()
    else:
        target.write_bytes(b"original target")
    if kind == "junction":
        probe.create_junction(link, target)
    else:
        try:
            link.symlink_to(target, target_is_directory=directory)
        except OSError as exc:
            if exc.winerror == 1314:
                pytest.skip("native symlink creation requires unavailable runner privilege")
            raise
    try:
        handle = adapter.open(link)
        try:
            identity = adapter.identity(handle)
            assert identity.attributes & 0x400
            with pytest.raises(native.InstallationIntegrityError, match="reparse"):
                native._check_identity(link, identity, directory=directory)
        finally:
            adapter.close(handle)
        probe.assert_closed(handle)
        before = probe.handle_count()
        inputs = _inputs(adapter, tmp_path)
        try:
            with pytest.raises(native.InstallationIntegrityError, match="reparse"):
                inputs._walk(tmp_path)
        finally:
            inputs.close()
        assert probe.handle_count() == before
    finally:
        # Explicitly remove the link itself, never recurse through a junction.
        if directory:
            link.rmdir()
        else:
            link.unlink()
    assert target.exists()


@pytest.mark.parametrize("exception", [False, True])
def test_real_pins_release_all_handles_after_success_or_exception(
    tmp_path, adapter, probe, exception
):
    target = tmp_path / "input"
    target.write_bytes(b"protected")
    before = probe.handle_count()
    for _ in range(16):
        inputs = _inputs(adapter, tmp_path)
        handles = []
        try:
            handles.append(inputs._pin(tmp_path, directory=True).handle)
            handles.append(inputs._pin(target, directory=False).handle)
            if exception:
                with pytest.raises(RuntimeError, match="qualification exception"):
                    try:
                        raise RuntimeError("qualification exception")
                    finally:
                        inputs.close()
        finally:
            inputs.close()
        for handle in handles:
            probe.assert_closed(handle)
        assert inputs._closed
    assert probe.handle_count() == before
    target.write_bytes(b"post-release positive control")
    target.rename(tmp_path / "released")


def test_real_failed_pin_type_check_releases_opened_handle(tmp_path, adapter, probe):
    target = tmp_path / "regular-file"
    target.write_bytes(b"original")
    before = probe.handle_count()
    for _ in range(16):
        inputs = _inputs(adapter, tmp_path)
        try:
            with pytest.raises(native.InstallationIntegrityError, match="reparse"):
                inputs._pin(target, directory=True)
            assert inputs._pins == {}
        finally:
            inputs.close()
    assert probe.handle_count() == before
    target.unlink()


@pytest.mark.parametrize("action", ["delete", "rename", "replace"])
def test_real_log_pin_allows_data_writer_but_keeps_object_identity(
    tmp_path, adapter, probe, action
):
    target, other = tmp_path / "installer.log", tmp_path / "other"
    target.write_bytes(b"log")
    if action == "replace":
        other.write_bytes(b"replacement")

    class LogInputs(native.LockedInputs):
        def _inventory(self) -> set[str]:
            # Limit inventory qualification to this isolated temporary tree.
            found: set[str] = self._walk(tmp_path)
            return found

    inputs = LogInputs(
        adapter,
        artifact_root=tmp_path / "unused-artifacts",
        policy_path=tmp_path / "unused-policy" / "host.json",
        executor_root=tmp_path / "unused-checkout",
        staging_root=tmp_path / "unused-staging",
    )
    pin = inputs._pin(target, directory=False, log=True)
    try:
        assert pin.mutable
        # The replacement donor is still unpinned here, so its own sharing
        # restrictions cannot cause a false positive for the log's protection.
        result, error = probe.mutate(action, target, other)
        assert not result
        # As for immutable files, MoveFileExW may report access denied for
        # destination replacement; success is always a qualification failure.
        expected_errors = (
            {_SHARING_VIOLATION, _ACCESS_DENIED} if action == "replace" else {_SHARING_VIOLATION}
        )
        assert error in expected_errors
        assert target.read_bytes() == b"log"
        inputs._inventory()
        inputs.verify_pins()
        result, error = probe.mutate("write", target, other)
        assert result, f"log data writer rejected: Win32 {error}"
        assert adapter.identity(pin.handle).size == 8
        assert target.read_bytes() == b"attacker"
        assert adapter.identity(pin.handle) != pin.identity
        # The production mutable-pin branch must permit size/time changes
        # while retaining file identity, attributes, links and final path.
        inputs.verify_pins()
    finally:
        inputs.close()
    probe.assert_closed(pin.handle)
    result, error = probe.mutate(action, target, other)
    assert result, f"post-release log control failed: Win32 {error}"
