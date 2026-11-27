"""Synthetic image lease and kernel capability boundary checks."""

from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path
import stat
import struct
import sys
from types import SimpleNamespace

import pytest

from scripts import stage9_v5d11_images as images

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux inode flags")
NAME = "/var/lib/libvirt/images/probe.raw"
GROUP = 13011
DATA = b"synthetic V5d11 image\0" * 300


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def staging(tmp_path, monkeypatch):
    tree = tmp_path / "root"
    target = tree / NAME[1:]
    target.parent.mkdir(parents=True)
    source = tmp_path / "source.raw"
    source.write_bytes(DATA)
    initial_identity = source.stat().st_ino
    original_stat = os.fstat
    emulated_groups = {}
    observed_flags = {}
    flag_writes = []

    def fstat(fd):
        result = original_stat(fd)
        if result.st_ino in emulated_groups:
            return SimpleNamespace(
                **{
                    name: getattr(result, name)
                    for name in dir(result)
                    if name.startswith("st_") and name not in {"st_uid", "st_gid"}
                },
                st_uid=0,
                st_gid=emulated_groups[result.st_ino],
            )
        return result

    def fchown(fd, owner, group):
        assert owner == 0
        inode = original_stat(fd).st_ino
        assert inode != initial_identity
        emulated_groups[inode] = group

    def get_flags(fd):
        return observed_flags.get(original_stat(fd).st_ino, 0x80000)

    def set_flags(fd, flags):
        inode = original_stat(fd).st_ino
        assert inode != initial_identity
        observed_flags[inode] = flags
        flag_writes.append((inode, flags))

    monkeypatch.setattr(images, "_directory_identity", lambda _: None)
    monkeypatch.setattr(images, "_filesystem", lambda _: "ext4")
    monkeypatch.setattr(images, "_flags", get_flags)
    monkeypatch.setattr(images, "_set_flags", set_flags)
    monkeypatch.setattr(os, "fstat", fstat)
    monkeypatch.setattr(os, "fchown", fchown)
    return tree, target, source, flag_writes


def test_fresh_copy_is_pinned_and_only_owned_inode_is_changed(staging, monkeypatch):
    tree, target, source, writes = staging
    original = source.stat()
    monkeypatch.setattr(images, "CHUNK", 4096)
    source.write_bytes(DATA + bytes(8192))
    with images.ImageStaging() as lease:
        assert lease.arguments() == []
        lease.copy(tree, NAME, source, sha(source.read_bytes()), GROUP)
        observed = lease.evidence()[NAME]
        assert observed == {
            "filesystem": "ext4",
            "sha256": sha(source.read_bytes()),
            "dev": target.stat().st_dev,
            "ino": target.stat().st_ino,
            "size": source.stat().st_size,
            "mode": 0o440,
            "uid": 0,
            "gid": GROUP,
            "flags": 0x80010,
        }
        assert target.read_bytes() == source.read_bytes()
        assert stat.S_IMODE(target.stat().st_mode) == 0o440
        assert lease.arguments() == [
            "--images",
            NAME,
            f"{observed['dev']}:{observed['ino']}:{observed['size']}",
        ]
        with pytest.raises(images.Refusal, match="IMAGE_STAGING_LEASE_REJECTED"):
            lease.copy(tree, NAME, source, sha(DATA), GROUP)
    expected_inode = observed["ino"]
    assert writes == [(expected_inode, 0x80010), (expected_inode, 0x80000)]
    assert not target.exists() and source.stat().st_ino == original.st_ino
    assert source.read_bytes() == DATA + bytes(8192)


def test_copy_refuses_pin_mismatch_and_leaves_source_unchanged(staging):
    tree, target, source, writes = staging
    with (
        images.ImageStaging() as lease,
        pytest.raises(images.Refusal, match="IMAGE_SOURCE_PIN_MISMATCH"),
    ):
        lease.copy(tree, NAME, source, sha(b"other data"), GROUP)
    assert not target.exists() and writes == [] and source.read_bytes() == DATA


@pytest.mark.parametrize(
    "input_name,data",
    [(NAME, b"\x7fELF fixture"), (NAME, b"#!fixture"), (NAME, b"PK\x03\x04 fixture"), (NAME, b"")],
)
def test_nonimage_data_fails_closed(staging, input_name, data):
    tree, target, source, writes = staging
    source.write_bytes(data)
    with (
        images.ImageStaging() as lease,
        pytest.raises(images.Refusal, match="EXECUTABLE_DATA_REJECTED|IMAGE_SOURCE_REJECTED"),
    ):
        lease.copy(tree, input_name, source, sha(data), GROUP)
    assert not target.exists() and writes == []


def test_refuses_existing_destination_and_unapproved_paths(staging):
    tree, target, source, writes = staging
    target.write_bytes(b"pre-existing synthetic file")
    with images.ImageStaging() as lease:
        with pytest.raises(FileExistsError):
            lease.copy(tree, NAME, source, sha(DATA), GROUP)
        for unsafe in ("/etc/other.raw", f"{NAME}/../outside.raw", f"{NAME}*"):
            with pytest.raises(images.Refusal, match="IMAGE_STAGING_INPUT_REJECTED"):
                lease.copy(tree, unsafe, source, sha(DATA), GROUP)
    assert target.read_bytes() == b"pre-existing synthetic file" and writes == []


def test_source_symlinks_are_refused(staging):
    tree, target, source, writes = staging
    alternate = source.with_name("symlink.raw")
    alternate.symlink_to(source)
    with images.ImageStaging() as lease, pytest.raises(OSError):
        lease.copy(tree, NAME, alternate, sha(DATA), GROUP)
    assert not target.exists() and writes == []


def test_requires_ext4_before_opening_data(staging, monkeypatch):
    tree, target, source, writes = staging
    monkeypatch.setattr(images, "_filesystem", lambda _: "tmpfs")
    with (
        images.ImageStaging() as lease,
        pytest.raises(images.Refusal, match="IMAGE_STAGING_EXT4_REQUIRED"),
    ):
        lease.copy(tree, NAME, source, sha(DATA), GROUP)
    assert not target.exists() and writes == []


def test_requires_real_bit_and_removes_failed_copy(staging, monkeypatch):
    tree, target, source, writes = staging
    monkeypatch.setattr(images, "_set_flags", lambda *_: None)
    with images.ImageStaging() as lease:
        with pytest.raises(images.Refusal, match="IMAGE_STAGING_IMMUTABLE_REQUIRED"):
            lease.copy(tree, NAME, source, sha(DATA), GROUP)
        with pytest.raises(images.Refusal, match="IMAGE_STAGING_LEASE_INCOMPLETE"):
            lease.arguments()
    assert not target.exists() and writes == []


def test_never_unlinks_a_substituted_staging_path(staging):
    tree, target, source, writes = staging
    alternate = target.with_name("original-held-inode")
    with (
        pytest.raises(images.Refusal, match="IMAGE_STAGING_CLEANUP_FAILED"),
        images.ImageStaging() as lease,
    ):
        lease.copy(tree, NAME, source, sha(DATA), GROUP)
        target.rename(alternate)
        target.write_bytes(b"replacement synthetic fixture")
    assert target.read_bytes() == b"replacement synthetic fixture"
    assert [entry[1] for entry in writes] == [0x80010, 0x80000]
    assert alternate.read_bytes() == DATA


def test_real_proc_mount_tracking_and_flags_abi(tmp_path, monkeypatch):
    import fcntl

    fd = os.open("/proc", os.O_RDONLY | os.O_DIRECTORY)
    try:
        assert images._filesystem(fd) == "proc"
    finally:
        os.close(fd)
    path = tmp_path / "source"
    path.write_bytes(DATA)
    requests = []

    def ioctl(file, request, value):
        requests.append((file, request, value))
        return struct.pack("=Q", 0x80010)

    monkeypatch.setattr(fcntl, "ioctl", ioctl)
    with path.open("rb") as stream:
        assert images._flags(stream.fileno()) & images.FS_IMMUTABLE_FL
        images._set_flags(stream.fileno(), 0x80000)
    assert [call[1] for call in requests] == [0x80086601, 0x40086602]
    assert struct.unpack("=Q", requests[-1][2]) == (0x80000,)


def test_transfer_refuses_inflight_source_mutation(staging, monkeypatch):
    tree, target, source, writes = staging
    read = os.read
    first = True
    source_stat = source.stat()

    def mutate_in_fixture(fd, size):
        nonlocal first
        if first and os.fstat(fd).st_ino == source_stat.st_ino:
            first = False
            chunk = read(fd, size)
            with source.open("ab") as stream:
                stream.write(b"fixture-only change")
            return chunk
        return read(fd, size)

    monkeypatch.setattr(os, "read", mutate_in_fixture)
    with (
        images.ImageStaging() as lease,
        pytest.raises(images.Refusal, match="IMAGE_SOURCE_CHANGED"),
    ):
        lease.copy(tree, NAME, source, sha(DATA), GROUP)
    assert not target.exists() and writes == []


def test_operator_opt_in_native_ext4_fixture():
    """Operational optional test, only a new dedicated fixture can be mutated."""
    import tempfile

    root = os.environ.get("STAGE9_V5D11_EXT4_FIXTURE_ROOT")
    if not root or os.geteuid() != 0:
        pytest.skip("NOT_TESTED: protected ext4 scratch and CAP_LINUX_IMMUTABLE required")
    with tempfile.TemporaryDirectory(prefix="stage9-image-synthetic-", dir=root) as folder:
        tree = Path(folder)
        target = tree / NAME[1:]
        target.parent.mkdir(parents=True)
        source = tree / "source.raw"
        source.write_bytes(DATA)
        with images.ImageStaging() as lease:
            lease.copy(tree, NAME, source, sha(DATA), GROUP)
            with target.open("rb") as stream:
                assert images._flags(stream.fileno()) & images.FS_IMMUTABLE_FL
            with pytest.raises(PermissionError) as blocked:
                target.unlink()
            assert blocked.value.errno == errno.EPERM
        assert not target.exists() and source.read_bytes() == DATA
