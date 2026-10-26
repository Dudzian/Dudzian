"""Protected descriptor reads for the Stage 9 installation comparison gate.

These reads establish observed local object consistency, never independent
authorization or an atomic filesystem snapshot. No path or OS error text is
included in a refusal.
"""

from __future__ import annotations

import hashlib
import os
import stat
import sys
import time
from types import TracebackType

from deployment.stage9_native_linux_cgroup_membership_observation import _valid_path
from deployment.stage9_native_linux_unit_fragment_observation import _metadata

_MAX_BYTES = 64 * 1024 * 1024
_ACL_NAMES = frozenset({"system.posix_acl_access", "system.posix_acl_default"})
_CODES = frozenset(
    {
        "FILE_ARGUMENT_INVALID",
        "NATIVE_LINUX_REQUIRED",
        "FILE_PATH_UNSAFE",
        "FILE_ANCESTOR_UNSAFE",
        "FILE_METADATA_UNSAFE",
        "FILE_SIZE_INVALID",
        "FILE_ACL_FORBIDDEN",
        "FILE_CHANGED_DURING_CHECK",
        "FILE_READ_TIMEOUT",
        "FILE_UNAVAILABLE",
    }
)


class FileRefusal(ValueError):
    """A closed, safe diagnostic code; exception causes must not be logged."""

    def __init__(self, code: str) -> None:
        self.code = code if type(code) is str and code in _CODES else "FILE_UNAVAILABLE"
        super().__init__(self.code)


def _check_directory(info: os.stat_result) -> None:
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != 0
        or info.st_gid != 0
        or stat.S_IMODE(info.st_mode) & 0o7022
    ):
        raise FileRefusal("FILE_ANCESTOR_UNSAFE")


def _no_acl(fd: int) -> None:
    if _ACL_NAMES.intersection(os.listxattr(fd)):
        raise FileRefusal("FILE_ACL_FORBIDDEN")


class PinnedFile:
    """Keep a checked file and its path-component bindings until final recheck.

    Production callers require root ownership in the closed manifest contract.
    ``uid``/``gid`` are explicit expected metadata, never a trust-source claim.
    ``retain_data=False`` avoids keeping executable contents in memory.
    """

    def __init__(
        self,
        path: str,
        *,
        uid: int = 0,
        gid: int = 0,
        mode: int | None = None,
        max_bytes: int = 65536,
        retain_data: bool = True,
    ) -> None:
        if not _valid_path(path):
            raise FileRefusal("FILE_PATH_UNSAFE")
        if (
            type(uid) is not int
            or not 0 <= uid < 2**32 - 1
            or type(gid) is not int
            or not 0 <= gid < 2**32 - 1
            or (mode is not None and (type(mode) is not int or not 0 <= mode <= 0o777))
            or type(max_bytes) is not int
            or not 0 < max_bytes <= _MAX_BYTES
            or type(retain_data) is not bool
        ):
            raise FileRefusal("FILE_ARGUMENT_INVALID")
        self._path = path
        self._uid = uid
        self._gid = gid
        self._mode = mode
        self._max_bytes = max_bytes
        self._retain_data = retain_data
        self._descriptors: list[int] = []
        self._directories: list[tuple[int | None, str, int, os.stat_result]] = []
        self._leaf: tuple[int, str, os.stat_result] | None = None
        self.fd = -1
        self.sha256 = ""
        self.data = b""

    @property
    def identity(self) -> tuple[int, int]:
        """Device/inode of the retained object, for cross-role alias refusal."""
        if self._leaf is None or self.fd < 0:
            raise FileRefusal("FILE_ARGUMENT_INVALID")
        expected = self._leaf[2]
        return expected.st_dev, expected.st_ino

    def _check_file(self, info: os.stat_result) -> None:
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != self._uid
            or info.st_gid != self._gid
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) & 0o7022
            or (self._mode is not None and stat.S_IMODE(info.st_mode) != self._mode)
        ):
            raise FileRefusal("FILE_METADATA_UNSAFE")
        if not 0 < info.st_size <= self._max_bytes:
            raise FileRefusal("FILE_SIZE_INVALID")

    def __enter__(self) -> PinnedFile:
        if sys.platform != "linux":
            raise FileRefusal("NATIVE_LINUX_REQUIRED")
        if self._descriptors:
            raise FileRefusal("FILE_ARGUMENT_INVALID")
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
        try:
            parent: int | None = None
            parts = self._path[1:].split("/")
            for name in ("/", *parts[:-1]):
                before = os.lstat(name, dir_fd=parent)
                _check_directory(before)
                descriptor = os.open(name, flags | os.O_DIRECTORY, dir_fd=parent)
                self._descriptors.append(descriptor)
                if _metadata(before) != _metadata(os.fstat(descriptor)):
                    raise FileRefusal("FILE_CHANGED_DURING_CHECK")
                _no_acl(descriptor)
                self._directories.append((parent, name, descriptor, before))
                parent = descriptor
            if parent is None:
                raise FileRefusal("FILE_ARGUMENT_INVALID")
            before = os.lstat(parts[-1], dir_fd=parent)
            self._check_file(before)
            self.fd = os.open(parts[-1], flags, dir_fd=parent)
            self._descriptors.append(self.fd)
            if _metadata(before) != _metadata(os.fstat(self.fd)):
                raise FileRefusal("FILE_CHANGED_DURING_CHECK")
            _no_acl(self.fd)
            self._leaf = (parent, parts[-1], before)
            digest = hashlib.sha256()
            chunks = []
            total = 0
            deadline = time.monotonic() + 3
            while total <= self._max_bytes:
                if time.monotonic() >= deadline:
                    raise FileRefusal("FILE_READ_TIMEOUT")
                chunk = os.read(self.fd, min(65536, self._max_bytes + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > self._max_bytes:
                    raise FileRefusal("FILE_SIZE_INVALID")
                digest.update(chunk)
                if self._retain_data:
                    chunks.append(chunk)
            if total != before.st_size:
                raise FileRefusal("FILE_CHANGED_DURING_CHECK")
            self.recheck()
            self.sha256 = digest.hexdigest()
            self.data = b"".join(chunks)
            return self
        except OSError as exc:
            self.close()
            raise FileRefusal("FILE_UNAVAILABLE") from exc
        except BaseException:
            self.close()
            raise

    def recheck(self) -> None:
        """Refuse changed objects or parent/name bindings without reopening."""
        if self._leaf is None or self.fd < 0:
            raise FileRefusal("FILE_ARGUMENT_INVALID")
        try:
            for parent, name, descriptor, expected in self._directories:
                if _metadata(expected) != _metadata(os.fstat(descriptor)) or _metadata(
                    expected
                ) != _metadata(os.lstat(name, dir_fd=parent)):
                    raise FileRefusal("FILE_CHANGED_DURING_CHECK")
                _no_acl(descriptor)
            parent, name, expected = self._leaf
            if _metadata(expected) != _metadata(os.fstat(self.fd)) or _metadata(
                expected
            ) != _metadata(os.lstat(name, dir_fd=parent)):
                raise FileRefusal("FILE_CHANGED_DURING_CHECK")
            _no_acl(self.fd)
        except OSError as exc:
            raise FileRefusal("FILE_UNAVAILABLE") from exc

    def close(self) -> None:
        failed = False
        while self._descriptors:
            descriptor = self._descriptors.pop()
            try:
                os.close(descriptor)
            except OSError:
                failed = True
        self._directories.clear()
        self._leaf = None
        self.fd = -1
        if failed:
            raise FileRefusal("FILE_UNAVAILABLE")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
