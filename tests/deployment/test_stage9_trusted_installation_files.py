"""Unprivileged filesystem refusal tests; synthetic ancestry is never evidence."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest

from deployment import stage9_trusted_installation_files as files


@pytest.fixture
def readable_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    if sys.platform != "linux":
        pytest.skip("Linux descriptor-relative paths and POSIX ACL controls")

    # Temp roots belong to this unprivileged runner. Only the ancestry ownership
    # policy is synthetic; open/read/link/race checks below operate on real files.
    # The production ancestry predicate is tested independently below.
    def synthetic_ancestry(info: os.stat_result) -> None:
        if not stat.S_ISDIR(info.st_mode) or (
            stat.S_IMODE(info.st_mode) & 0o022 and not info.st_mode & stat.S_ISVTX
        ):
            raise files.FileRefusal("FILE_ANCESTOR_UNSAFE")

    monkeypatch.setattr(files, "_check_directory", synthetic_ancestry)
    target = tmp_path / "expected.bin"
    target.write_bytes(b"independently expected synthetic bytes\n")
    target.chmod(0o600)
    return target


def pin(path: Path, **kwargs: object) -> files.PinnedFile:
    return files.PinnedFile(str(path), uid=os.getuid(), gid=os.getgid(), **kwargs)


def test_reads_retains_and_closes_descriptor(readable_file: Path) -> None:
    expected = readable_file.read_bytes()
    with pin(readable_file, mode=0o600) as item:
        descriptor = item.fd
        assert item.data == expected
        assert item.sha256 == hashlib.sha256(expected).hexdigest()
        assert os.fstat(descriptor).st_ino == readable_file.stat().st_ino
        assert item.identity == (os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino)
        item.recheck()
    assert item.fd == -1
    with pytest.raises(OSError):
        os.fstat(descriptor)
    with pytest.raises(files.FileRefusal, match="FILE_ARGUMENT_INVALID"):
        _ = item.identity


def test_identity_unavailable_before_open(readable_file: Path) -> None:
    item = pin(readable_file)
    with pytest.raises(files.FileRefusal, match="FILE_ARGUMENT_INVALID"):
        _ = item.identity


def test_executable_data_need_not_be_retained(readable_file: Path) -> None:
    with pin(readable_file, retain_data=False) as item:
        assert item.data == b""
        assert item.sha256 == hashlib.sha256(readable_file.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "path", ["relative", "/", "//alias/file", "/a/./file", "/a/../file", "/a//file", "/a/ "]
)
def test_rejects_noncanonical_path(path: str) -> None:
    with pytest.raises(files.FileRefusal, match="FILE_PATH_UNSAFE"):
        files.PinnedFile(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"uid": True},
        {"uid": -1},
        {"gid": True},
        {"gid": 2**32},
        {"mode": True},
        {"mode": 0o4755},
        {"max_bytes": True},
        {"max_bytes": 0},
        {"max_bytes": 64 * 1024 * 1024 + 1},
        {"retain_data": 1},
    ],
)
def test_rejects_invalid_options(kwargs: dict) -> None:
    with pytest.raises(files.FileRefusal, match="FILE_ARGUMENT_INVALID"):
        files.PinnedFile("/expected/file", **kwargs)


@pytest.mark.parametrize("mode", [0o620, 0o602, 0o4600, 0o2600, 0o1600])
def test_rejects_unsafe_file_modes(readable_file: Path, mode: int) -> None:
    readable_file.chmod(mode)
    with pytest.raises(files.FileRefusal, match="FILE_METADATA_UNSAFE"):
        with pin(readable_file):
            pass


def test_rejects_unexpected_exact_mode(readable_file: Path) -> None:
    with pytest.raises(files.FileRefusal, match="FILE_METADATA_UNSAFE"):
        with pin(readable_file, mode=0o400):
            pass


@pytest.mark.parametrize("field", ["uid", "gid"])
def test_rejects_unexpected_file_owner(readable_file: Path, field: str) -> None:
    values = {"uid": os.getuid(), "gid": os.getgid()}
    values[field] += 1
    with pytest.raises(files.FileRefusal, match="FILE_METADATA_UNSAFE"):
        with files.PinnedFile(str(readable_file), **values):
            pass


def test_rejects_leaf_symlink(readable_file: Path) -> None:
    link = readable_file.with_name("alias")
    link.symlink_to(readable_file)
    with pytest.raises(files.FileRefusal, match="FILE_METADATA_UNSAFE"):
        with pin(link):
            pass


def test_rejects_ancestor_symlink(readable_file: Path) -> None:
    link = readable_file.parent / "alias-dir"
    link.symlink_to(readable_file.parent, target_is_directory=True)
    with pytest.raises(files.FileRefusal, match="FILE_ANCESTOR_UNSAFE"):
        with pin(link / readable_file.name):
            pass


def test_rejects_hardlink(readable_file: Path) -> None:
    os.link(readable_file, readable_file.with_name("hardlink"))
    with pytest.raises(files.FileRefusal, match="FILE_METADATA_UNSAFE"):
        with pin(readable_file):
            pass


def test_rejects_writable_ancestor(readable_file: Path) -> None:
    readable_file.parent.chmod(0o777)
    with pytest.raises(files.FileRefusal, match="FILE_ANCESTOR_UNSAFE"):
        with pin(readable_file):
            pass


@pytest.mark.parametrize("size", [0, 65537])
def test_rejects_empty_or_oversized_file(readable_file: Path, size: int) -> None:
    readable_file.write_bytes(b"x" * size)
    with pytest.raises(files.FileRefusal, match="FILE_SIZE_INVALID"):
        with pin(readable_file):
            pass


def test_rejects_missing_file(readable_file: Path) -> None:
    readable_file.unlink()
    with pytest.raises(files.FileRefusal, match="FILE_UNAVAILABLE") as error:
        with pin(readable_file):
            pass
    assert str(readable_file) not in str(error.value)


@pytest.mark.parametrize("attribute", ["system.posix_acl_access", "system.posix_acl_default"])
def test_rejects_acl(readable_file: Path, monkeypatch: pytest.MonkeyPatch, attribute: str) -> None:
    monkeypatch.setattr(files.os, "listxattr", lambda descriptor: [attribute])
    with pytest.raises(files.FileRefusal, match="FILE_ACL_FORBIDDEN"):
        with pin(readable_file):
            pass


def test_rejects_unreadable_acl_inventory(
    readable_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(descriptor: int) -> list[str]:
        raise PermissionError("private path must never appear in refusal")

    monkeypatch.setattr(files.os, "listxattr", unavailable)
    with pytest.raises(files.FileRefusal, match="^FILE_UNAVAILABLE$"):
        with pin(readable_file):
            pass


def test_rejects_replacement_during_read(
    readable_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_read = os.read
    replacement = readable_file.with_name("replacement")
    replacement.write_bytes(readable_file.read_bytes())
    replacement.chmod(0o600)
    replaced = False

    def swap(descriptor: int, amount: int) -> bytes:
        nonlocal replaced
        content = real_read(descriptor, amount)
        if not replaced:
            replaced = True
            replacement.replace(readable_file)
        return content

    monkeypatch.setattr(files.os, "read", swap)
    with pytest.raises(files.FileRefusal, match="FILE_CHANGED_DURING_CHECK"):
        with pin(readable_file):
            pass


def test_rejects_replacement_after_entry(readable_file: Path) -> None:
    with pin(readable_file) as item:
        replacement = readable_file.with_name("replacement")
        replacement.write_bytes(readable_file.read_bytes())
        replacement.chmod(0o600)
        replacement.replace(readable_file)
        with pytest.raises(files.FileRefusal, match="FILE_CHANGED_DURING_CHECK"):
            item.recheck()


def test_rejects_ancestor_rebinding_after_entry(readable_file: Path) -> None:
    with pin(readable_file) as item:
        original_parent = readable_file.parent
        moved_parent = original_parent.with_name(original_parent.name + "-moved")
        original_parent.rename(moved_parent)
        original_parent.mkdir(mode=0o700)
        with pytest.raises(files.FileRefusal, match="FILE_CHANGED_DURING_CHECK"):
            item.recheck()


def test_rejects_content_mutation_after_entry(readable_file: Path) -> None:
    with pin(readable_file) as item:
        readable_file.write_bytes(b"replacement of different bytes")
        with pytest.raises(files.FileRefusal, match="FILE_CHANGED_DURING_CHECK"):
            item.recheck()


def test_rejects_descriptor_read_error(
    readable_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(descriptor: int, amount: int) -> bytes:
        raise OSError("secret details")

    monkeypatch.setattr(files.os, "read", unavailable)
    with pytest.raises(files.FileRefusal, match="^FILE_UNAVAILABLE$"):
        with pin(readable_file):
            pass


def test_rejects_file_growth_during_read(
    readable_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_read = os.read
    grown = False

    def grow(descriptor: int, amount: int) -> bytes:
        nonlocal grown
        content = original_read(descriptor, amount)
        if not grown:
            grown = True
            with readable_file.open("ab") as writer:
                writer.write(b"x" * 65536)
        return content

    monkeypatch.setattr(files.os, "read", grow)
    with pytest.raises(files.FileRefusal, match="FILE_SIZE_INVALID"):
        with pin(readable_file):
            pass


def test_rejects_object_swap_between_stat_and_open(
    readable_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_open = os.open
    replacement = readable_file.with_name("replacement")
    replacement.write_bytes(readable_file.read_bytes())
    replacement.chmod(0o600)

    def swapped(name: str, flags: int, **kwargs: object) -> int:
        if name == readable_file.name:
            replacement.replace(readable_file)
        return original_open(name, flags, **kwargs)

    monkeypatch.setattr(files.os, "open", swapped)
    with pytest.raises(files.FileRefusal, match="FILE_CHANGED_DURING_CHECK"):
        with pin(readable_file):
            pass


def test_rejects_unavailable_retained_file(readable_file: Path) -> None:
    with pin(readable_file) as item:
        readable_file.unlink()
        with pytest.raises(files.FileRefusal, match="FILE_CHANGED_DURING_CHECK"):
            item.recheck()


def test_reports_close_failure(readable_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_close = os.close

    def close_failed(descriptor: int) -> None:
        original_close(descriptor)
        raise OSError("secret details")

    with pytest.raises(files.FileRefusal, match="^FILE_UNAVAILABLE$"):
        with pin(readable_file):
            monkeypatch.setattr(files.os, "close", close_failed)


def test_rejects_read_deadline(readable_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter([1.0, 4.0])
    monkeypatch.setattr(files.time, "monotonic", lambda: next(ticks))
    with pytest.raises(files.FileRefusal, match="FILE_READ_TIMEOUT"):
        with pin(readable_file):
            pass


def test_rejects_reentry_and_closed_recheck(readable_file: Path) -> None:
    with pin(readable_file) as item:
        with pytest.raises(files.FileRefusal, match="FILE_ARGUMENT_INVALID"):
            item.__enter__()
    with pytest.raises(files.FileRefusal, match="FILE_ARGUMENT_INVALID"):
        item.recheck()


def test_rejects_non_linux(readable_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(files.sys, "platform", "win32")
    with pytest.raises(files.FileRefusal, match="NATIVE_LINUX_REQUIRED"):
        with pin(readable_file):
            pass


@pytest.mark.parametrize(
    "changes",
    [
        {"st_uid": 1000},
        {"st_gid": 1000},
        {"st_mode": stat.S_IFREG | 0o755},
        {"st_mode": stat.S_IFLNK | 0o755},
        {"st_mode": stat.S_IFDIR | 0o775},
        {"st_mode": stat.S_IFDIR | 0o757},
        {"st_mode": stat.S_IFDIR | 0o2755},
    ],
)
def test_production_ancestry_policy_rejects_unsafe_metadata(changes: dict) -> None:
    info = {"st_uid": 0, "st_gid": 0, "st_mode": stat.S_IFDIR | 0o755}
    info.update(changes)
    with pytest.raises(files.FileRefusal, match="FILE_ANCESTOR_UNSAFE"):
        files._check_directory(SimpleNamespace(**info))


def test_production_ancestry_accepts_root_owned_safe_directory() -> None:
    files._check_directory(SimpleNamespace(st_uid=0, st_gid=0, st_mode=stat.S_IFDIR | 0o755))


def test_refusal_sanitizes_unknown_codes() -> None:
    assert str(files.FileRefusal("private path details")) == "FILE_UNAVAILABLE"
