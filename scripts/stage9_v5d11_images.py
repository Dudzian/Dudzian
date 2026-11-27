"""Stage private, pinned image copies with an explicit ext4 immutable inode lease."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import stat
import struct
import sys
from types import TracebackType
from typing import Any

IMAGE_PREFIX = "/var/lib/libvirt/images/"
FS_IOC_GETFLAGS = 0x80086601
FS_IOC_SETFLAGS = 0x40086602
FS_IMMUTABLE_FL = 0x10
FILE_MODE = 0o440
CHUNK = 1024 * 1024
MAX_PROC = 4 * 1024 * 1024
MAX_IMAGE_BYTES = 128 * 1024 * 1024 * 1024


class Refusal(ValueError):
    """A source, destination, filesystem or immutable inode was refused."""


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise Refusal(reason)


def _canonical(path: str) -> bool:
    return re.fullmatch(r"(?:/[A-Za-z0-9_.-]+)+", path) is not None and not any(
        part in {".", ".."} for part in path.split("/")
    )


def _open_directory(path: Path, *, protected: bool) -> int:
    require(sys.platform == "linux", "IMAGE_STAGING_LINUX_REQUIRED")
    require(
        path.is_absolute() and (path == Path("/") or _canonical(str(path))), "IMAGE_PATH_REJECTED"
    )
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    current = os.open("/", flags)
    try:
        if protected:
            _directory_identity(current)
        for part in path.parts[1:]:
            child = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = child
            if protected:
                _directory_identity(current)
        return current
    except BaseException:
        os.close(current)
        raise


def _directory_identity(fd: int) -> None:
    info = os.fstat(fd)
    require(
        stat.S_ISDIR(info.st_mode) and info.st_uid == 0 and info.st_mode & 0o022 == 0,
        "IMAGE_STAGING_DIRECTORY_UNPROTECTED",
    )


def _read_proc(path: Path) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(MAX_PROC + 1)
    require(len(data) <= MAX_PROC, "IMAGE_MOUNT_METADATA_LIMIT")
    return data


def _filesystem(fd: int) -> str:
    """Resolve the exact opened mount ID without matching paths or device names."""
    fdinfo = _read_proc(Path(f"/proc/self/fdinfo/{fd}"))
    identities = re.findall(rb"^mnt_id:\s*([0-9]+)$", fdinfo, re.MULTILINE)
    require(len(identities) == 1, "IMAGE_MOUNT_ID_REJECTED")
    mountinfo = _read_proc(Path("/proc/self/mountinfo"))
    types = []
    for line in mountinfo.splitlines():
        metadata, separator, filesystem = line.partition(b" - ")
        if separator and metadata.split(maxsplit=1)[0] == identities[0]:
            types.append(filesystem.split(maxsplit=1)[0])
    require(len(types) == 1, "IMAGE_MOUNT_ID_REJECTED")
    return types[0].decode("ascii")


def _flags(fd: int) -> int:
    if sys.platform == "linux":
        import fcntl

        result = fcntl.ioctl(fd, FS_IOC_GETFLAGS, bytes(8))
        return int(struct.unpack("=Q", result)[0])
    raise Refusal("IMAGE_STAGING_LINUX_REQUIRED")


def _set_flags(fd: int, flags: int) -> None:
    if sys.platform == "linux":
        import fcntl

        fcntl.ioctl(fd, FS_IOC_SETFLAGS, struct.pack("=Q", flags))
    else:
        raise Refusal("IMAGE_STAGING_LINUX_REQUIRED")


def _fingerprint(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_uid,
        info.st_gid,
        info.st_mode,
        info.st_nlink,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _inode(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _read_file(parent: int, name: str) -> int:
    require(sys.platform == "linux", "IMAGE_STAGING_LINUX_REQUIRED")
    return os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=parent)


def _source(path: Path) -> int:
    parent = _open_directory(path.parent, protected=False)
    try:
        return _read_file(parent, path.name)
    finally:
        os.close(parent)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        require(written > 0, "IMAGE_STAGING_WRITE_FAILED")
        view = view[written:]


def _transfer(source: int, destination: int, expected: str) -> None:
    before = os.fstat(source)
    require(
        stat.S_ISREG(before.st_mode) and 0 < before.st_size <= MAX_IMAGE_BYTES,
        "IMAGE_SOURCE_REJECTED",
    )
    sha256 = hashlib.sha256()
    null_bytes = bytes(CHUNK)
    count = 0
    while chunk := os.read(source, CHUNK):
        if not count:
            require(
                not chunk.startswith((b"\x7fELF", b"#!", b"PK\x03\x04")), "EXECUTABLE_DATA_REJECTED"
            )
        count += len(chunk)
        require(count <= before.st_size, "IMAGE_SOURCE_CHANGED")
        sha256.update(chunk)
        if chunk == null_bytes[: len(chunk)]:
            os.lseek(destination, len(chunk), os.SEEK_CUR)
        else:
            _write_all(destination, chunk)
    require(
        count == before.st_size and _fingerprint(before) == _fingerprint(os.fstat(source)),
        "IMAGE_SOURCE_CHANGED",
    )
    require(sha256.hexdigest() == expected, "IMAGE_SOURCE_PIN_MISMATCH")
    os.ftruncate(destination, count)


@dataclass
class _Copy:
    parent: int
    filename: str
    fd: int
    identity: tuple[int, int]
    record: dict[str, Any]
    ready: bool = False

    def cleanup(self) -> None:
        try:
            require(_inode(os.fstat(self.fd)) == self.identity, "IMAGE_CLEANUP_INODE_CHANGED")
            flags = _flags(self.fd)
            remaining = flags & ~FS_IMMUTABLE_FL
            if flags != remaining:
                _set_flags(self.fd, remaining)
            require(_flags(self.fd) == remaining, "IMAGE_CLEANUP_FLAGS_CHANGED")
            _remove(self.parent, self.filename, self.identity)
        finally:
            os.close(self.fd)
            os.close(self.parent)


def _remove(parent: int, name: str, identity: tuple[int, int]) -> None:
    observed = os.stat(name, dir_fd=parent, follow_symlinks=False)
    require(_inode(observed) == identity, "IMAGE_CLEANUP_PATH_CHANGED")
    os.unlink(name, dir_fd=parent)


class ImageStaging:
    """Own only inodes created exclusively in an independently protected staging tree."""

    def __init__(self) -> None:
        self._copies: dict[str, _Copy] = {}
        self._entered = False

    def __enter__(self) -> ImageStaging:
        require(not self._entered, "IMAGE_STAGING_ALREADY_ENTERED")
        self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._entered = False
        failures = []
        for copy in reversed(tuple(self._copies.values())):
            try:
                copy.cleanup()
            except (OSError, Refusal) as error:
                failures.append(error)
        self._copies.clear()
        if failures:
            raise Refusal("IMAGE_STAGING_CLEANUP_FAILED") from failures[0]

    def copy(self, tree: Path, name: str, source: Path, sha256: str, gid: int) -> None:
        if sys.platform != "linux":
            raise Refusal("IMAGE_STAGING_LINUX_REQUIRED")
        require(self._entered and name not in self._copies, "IMAGE_STAGING_LEASE_REJECTED")
        require(
            name.startswith(IMAGE_PREFIX)
            and _canonical(name)
            and _canonical(str(source))
            and re.fullmatch(r"[a-f0-9]{64}", sha256) is not None
            and type(gid) is int
            and 0 < gid < 2**32 - 1,
            "IMAGE_STAGING_INPUT_REJECTED",
        )
        target = tree / name[1:]
        directory = _open_directory(target.parent, protected=True)
        writer = reader = origin = -1
        owned = False
        identity = None
        try:
            require(_filesystem(directory) == "ext4", "IMAGE_STAGING_EXT4_REQUIRED")
            origin = _source(source)
            writer = os.open(
                target.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                mode=0o600,
                dir_fd=directory,
            )
            identity = _inode(os.fstat(writer))
            require(identity != _inode(os.fstat(origin)), "IMAGE_SOURCE_IS_DESTINATION")
            _transfer(origin, writer, sha256)
            os.fchown(writer, 0, gid)
            os.fchmod(writer, FILE_MODE)
            os.fsync(writer)
            os.close(writer)
            writer = -1
            reader = _read_file(directory, target.name)
            info = os.fstat(reader)
            require(
                _inode(info) == identity
                and stat.S_ISREG(info.st_mode)
                and info.st_nlink == 1
                and info.st_uid == 0
                and info.st_gid == gid
                and stat.S_IMODE(info.st_mode) == FILE_MODE
                and _filesystem(reader) == "ext4",
                "IMAGE_STAGING_INODE_REJECTED",
            )
            original_flags = _flags(reader)
            require(not original_flags & FS_IMMUTABLE_FL, "IMAGE_STAGING_INITIAL_FLAGS_REJECTED")
            immutable_flags = original_flags | FS_IMMUTABLE_FL
            record: dict[str, Any] = {
                "filesystem": "ext4",
                "sha256": sha256,
                "dev": info.st_dev,
                "ino": info.st_ino,
                "size": info.st_size,
                "mode": FILE_MODE,
                "uid": 0,
                "gid": gid,
                "flags": immutable_flags,
            }
            lease = _Copy(directory, target.name, reader, identity, record)
            self._copies[name] = lease
            owned = True
            _set_flags(reader, immutable_flags)
            require(_flags(reader) == immutable_flags, "IMAGE_STAGING_IMMUTABLE_REQUIRED")
            with os.fdopen(os.dup(reader), "rb") as stream:
                require(
                    hashlib.file_digest(stream, "sha256").hexdigest() == sha256,
                    "IMAGE_STAGING_DESTINATION_PIN_MISMATCH",
                )
            lease.ready = True
        finally:
            for fd in (origin, writer):
                if fd >= 0:
                    os.close(fd)
            if not owned:
                try:
                    if reader >= 0:
                        os.close(reader)
                    if identity is not None:
                        _remove(directory, target.name, identity)
                finally:
                    os.close(directory)

    def evidence(self) -> dict[str, dict[str, Any]]:
        require(all(copy.ready for copy in self._copies.values()), "IMAGE_STAGING_LEASE_INCOMPLETE")
        return {name: dict(copy.record) for name, copy in sorted(self._copies.items())}

    def arguments(self) -> list[str]:
        if not self._copies:
            return []
        arguments = ["--images"]
        for name, record in self.evidence().items():
            arguments.extend([name, f"{record['dev']}:{record['ino']}:{record['size']}"])
        return arguments
