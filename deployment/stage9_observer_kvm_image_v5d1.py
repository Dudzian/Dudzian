"""Read-only image preflight; local measurements never qualify an execution.

The admitted host profile is Linux with root-protected, kernel-immutable image
files. A stopped guest, an advisory lock or matching before/after timestamps is
insufficient. This module neither creates snapshots nor changes file flags.
QCOW2 backing names are checked before a metadata-only qemu-img invocation on
an inherited descriptor; qemu-img is never asked to traverse the backing chain.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import stat
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

_QEMU_IMG = "/usr/bin/qemu-img"
_MAX_IMAGE_BYTES = 128 * 1024 * 1024 * 1024
_MAX_CHAIN = 16
_MAX_OUTPUT = 1024 * 1024
_MAX_TOOL_BYTES = 128 * 1024 * 1024
_FS_IOC_GETFLAGS = 0x80086601
_FS_IMMUTABLE_FL = 0x00000010
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class ExpectedImageV5d1:
    path: str
    sha256: str
    format: Literal["raw", "qcow2"]


@dataclass(frozen=True, slots=True)
class ImageArtifactV5d1:
    path: str
    sha256: str
    format: str
    size: int
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int
    immutable_flag: bool
    backing_path: str | None
    internal_snapshot_count: int


@dataclass(frozen=True, slots=True)
class ImageMeasurementV5d1:
    status: Literal["IMAGE_MEASURED_LOCAL", "IMAGE_MEASUREMENT_BLOCKED"]
    reason_codes: tuple[str, ...]
    artifacts: tuple[ImageArtifactV5d1, ...] = ()
    profile: str = "stage9-observer-kvm-image-v5d1"
    qualification: Literal["BLOCKED"] = "BLOCKED"


class _ImageRefusal(Exception):
    pass


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise _ImageRefusal(reason)


def _path(value: str) -> tuple[str, ...]:
    _require(type(value) is str and 1 < len(value) <= 4096, "IMAGE_PATH_UNSAFE")
    _require(value.startswith("/") and str(PurePosixPath(value)) == value, "IMAGE_PATH_UNSAFE")
    parts = PurePosixPath(value).parts[1:]
    _require(
        bool(parts) and all(part not in {".", ".."} for part in parts) and "\x00" not in value,
        "IMAGE_PATH_UNSAFE",
    )
    return parts


def _protected(info: os.stat_result, *, directory: bool) -> None:
    _require(
        info.st_uid == 0
        and info.st_mode & 0o022 == 0
        and (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)),
        "IMAGE_PATH_UNPROTECTED",
    )


def _open_protected(path: str) -> int:
    """Walk held directory descriptors, refusing symlinks in every component."""
    parts = _path(path)
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        _protected(os.fstat(directory), directory=True)
        for part in parts[:-1]:
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory,
            )
            os.close(directory)
            directory = child
            _protected(os.fstat(directory), directory=True)
        _protected(os.stat(parts[-1], dir_fd=directory, follow_symlinks=False), directory=False)
        result = os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=directory
        )
        try:
            _protected(os.fstat(result), directory=False)
        except BaseException:
            os.close(result)
            raise
        return result
    finally:
        os.close(directory)


def _immutable(fd: int) -> bool:
    # fcntl is deliberately imported only on the admitted Linux host profile.
    if sys.platform == "linux":
        import fcntl

        flags = fcntl.ioctl(fd, _FS_IOC_GETFLAGS, bytes(8))
        return bool(struct.unpack("=Q", flags)[0] & _FS_IMMUTABLE_FL)
    raise _ImageRefusal("IMAGE_HOST_UNSUPPORTED")


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_uid,
        info.st_mode,
        info.st_nlink,
    )


def _pread(fd: int, size: int, offset: int) -> bytes:
    if sys.platform == "linux":
        return os.pread(fd, size, offset)
    raise _ImageRefusal("IMAGE_HOST_UNSUPPORTED")


def _digest(fd: int, size: int, deadline: float) -> str:
    result = hashlib.sha256()
    offset = 0
    while offset < size:
        _require(time.monotonic() < deadline, "IMAGE_MEASUREMENT_TIMEOUT")
        part = _pread(fd, min(1024 * 1024, size - offset), offset)
        _require(bool(part), "IMAGE_CHANGED_DURING_MEASUREMENT")
        result.update(part)
        offset += len(part)
    return result.hexdigest()


def _qcow_header(fd: int, size: int) -> tuple[str | None, int]:
    raw = _pread(fd, 104, 0)
    _require(len(raw) >= 72 and raw[:4] == b"QFI\xfb", "QCOW2_HEADER_INVALID")
    version = struct.unpack_from(">I", raw, 4)[0]
    _require(version in {2, 3}, "QCOW2_VERSION_UNSUPPORTED")
    backing_offset, backing_size, cluster_bits = struct.unpack_from(">QII", raw, 8)
    _require(9 <= cluster_bits <= 21, "QCOW2_HEADER_INVALID")
    crypt_method = struct.unpack_from(">I", raw, 32)[0]
    snapshots = struct.unpack_from(">I", raw, 60)[0]
    _require(crypt_method == 0, "QCOW2_ENCRYPTION_UNSUPPORTED")
    if version == 3:
        _require(len(raw) == 104, "QCOW2_HEADER_INVALID")
        incompatible = struct.unpack_from(">Q", raw, 72)[0]
        header_size = struct.unpack_from(">I", raw, 100)[0]
        # Includes external data files, dirty/corrupt state, compression and
        # extended-L2. Unknown features must not give qemu additional host paths.
        _require(incompatible == 0, "QCOW2_FEATURE_UNSUPPORTED")
        _require(header_size == 104, "QCOW2_HEADER_UNSUPPORTED")
    _require(snapshots == 0, "QCOW2_INTERNAL_SNAPSHOTS_UNSUPPORTED")
    _require((backing_offset == 0) == (backing_size == 0), "QCOW2_HEADER_INVALID")
    if backing_size == 0:
        return None, snapshots
    _require(
        backing_size <= 4096
        and backing_offset >= (104 if version == 3 else 72)
        and backing_offset + backing_size <= min(size, 1 << cluster_bits),
        "QCOW2_HEADER_INVALID",
    )
    backing = _pread(fd, backing_size, backing_offset)
    _require(len(backing) == backing_size, "QCOW2_HEADER_INVALID")
    try:
        name = backing.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise _ImageRefusal("QCOW2_BACKING_PATH_UNSAFE") from error
    _path(name)
    return name, snapshots


def _qemu_info(
    fd: int,
    image_format: str,
    deadline: float,
    *,
    approved_qemu_sha256: str | None = None,
) -> dict[str, object]:
    # Pin the executable inode as well as image inode. No shell, protocols,
    # caller-supplied options, --force-share, or backing-chain traversal.
    binary = _open_protected(_QEMU_IMG)
    process: subprocess.Popen[bytes] | None = None
    selector: selectors.BaseSelector | None = None
    try:
        selector = selectors.DefaultSelector()
        before = os.fstat(binary)
        _require(
            0 < before.st_size <= _MAX_TOOL_BYTES and before.st_nlink == 1,
            "IMAGE_TOOL_IDENTITY_MISMATCH",
        )
        if approved_qemu_sha256 is not None:
            _require(
                type(approved_qemu_sha256) is str
                and _SHA256.fullmatch(approved_qemu_sha256) is not None,
                "IMAGE_TOOL_IDENTITY_MISMATCH",
            )
            measured = _digest(binary, before.st_size, deadline)
            _require(
                measured == approved_qemu_sha256
                and _identity(os.fstat(binary)) == _identity(before),
                "IMAGE_TOOL_IDENTITY_MISMATCH",
            )
        process = subprocess.Popen(
            [
                f"/proc/self/fd/{binary}",
                "info",
                "--output=json",
                "-f",
                image_format,
                f"/proc/self/fd/{fd}",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            pass_fds=(fd, binary),
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
            close_fds=True,
        )
        _require(process.stdout is not None, "QEMU_IMG_OBSERVATION_INCOMPLETE")
        assert process.stdout is not None
        selector.register(process.stdout, selectors.EVENT_READ)
        output = bytearray()
        while selector.get_map():
            remaining = deadline - time.monotonic()
            _require(remaining > 0, "IMAGE_MEASUREMENT_TIMEOUT")
            for key, _ in selector.select(min(remaining, 0.1)):
                block = os.read(key.fd, 65536)
                if not block:
                    selector.unregister(key.fileobj)
                else:
                    output.extend(block)
                    _require(len(output) <= _MAX_OUTPUT, "QEMU_IMG_OUTPUT_LIMIT")
        _require(
            process.wait(timeout=max(0.001, deadline - time.monotonic())) == 0, "QEMU_IMG_FAILED"
        )
        parsed = json.loads(output)
        _require(type(parsed) is dict, "QEMU_IMG_OBSERVATION_INCOMPLETE")
        return dict(parsed)
    finally:
        try:
            if selector is not None:
                selector.close()
        finally:
            try:
                if process is not None:
                    try:
                        if process.poll() is None:
                            process.kill()
                    finally:
                        try:
                            process.wait()
                        finally:
                            if process.stdout is not None:
                                process.stdout.close()
            finally:
                os.close(binary)


def measure_image_v5d1(
    expected_chain: tuple[ExpectedImageV5d1, ...],
    *,
    approved_qemu_sha256: str | None = None,
) -> ImageMeasurementV5d1:
    """Measure an independently expected top-to-base chain, or refuse closed.

    Expected hashes bind local bytes to a manifest; their agreement does not
    authenticate that manifest or approve the controller/environment. Internal
    snapshots and mutable disks require a separately provisioned immutable view.
    No private path argument can turn this result into host qualification.
    The native aggregate supplies the independently approved qemu-tool hash;
    omission is only an unqualified local mechanism, never tool acceptance.
    Artifacts are provisional until the complete chain and final descriptor,
    path and immutable-flag checks succeed; refusals publish diagnostics only.
    """
    descriptors: list[int] = []
    artifacts: list[ImageArtifactV5d1] = []
    cleanup_reasons: list[str] = []
    try:
        _require(sys.platform == "linux", "IMAGE_HOST_UNSUPPORTED")
        _require(
            type(expected_chain) is tuple and 1 <= len(expected_chain) <= _MAX_CHAIN,
            "IMAGE_CHAIN_INVALID",
        )
        _require(
            all(
                type(item) is ExpectedImageV5d1
                and item.format in {"raw", "qcow2"}
                and type(item.sha256) is str
                and _SHA256.fullmatch(item.sha256) is not None
                for item in expected_chain
            ),
            "IMAGE_EXPECTATION_INVALID",
        )
        _require(
            len({item.path for item in expected_chain}) == len(expected_chain), "IMAGE_CHAIN_CYCLE"
        )
        before: list[os.stat_result] = []
        deadline = time.monotonic() + 120.0
        # Open and establish immutability of the complete independently expected
        # chain before allowing an image parser to inspect any file.
        for item in expected_chain:
            descriptor = _open_protected(item.path)
            descriptors.append(descriptor)
            info = os.fstat(descriptor)
            _require(
                0 < info.st_size <= _MAX_IMAGE_BYTES and info.st_nlink == 1,
                "IMAGE_FILE_UNSUPPORTED",
            )
            _require(_immutable(descriptor), "IMAGE_IMMUTABLE_VIEW_UNPROVEN")
            before.append(info)
        _require(
            len({(info.st_dev, info.st_ino) for info in before}) == len(before), "IMAGE_CHAIN_CYCLE"
        )
        for index, (item, descriptor, info) in enumerate(
            zip(expected_chain, descriptors, before, strict=True)
        ):
            backing, snapshots = (
                _qcow_header(descriptor, info.st_size) if item.format == "qcow2" else (None, 0)
            )
            expected_backing = (
                expected_chain[index + 1].path if index + 1 < len(expected_chain) else None
            )
            _require(backing == expected_backing, "IMAGE_BACKING_CHAIN_MISMATCH")
            measured = _digest(descriptor, info.st_size, deadline)
            _require(measured == item.sha256, "IMAGE_DIGEST_MISMATCH")
            observed = (
                _qemu_info(descriptor, item.format, deadline)
                if approved_qemu_sha256 is None
                else _qemu_info(
                    descriptor,
                    item.format,
                    deadline,
                    approved_qemu_sha256=approved_qemu_sha256,
                )
            )
            _require(observed.get("format") == item.format, "QEMU_IMG_FORMAT_MISMATCH")
            _require(observed.get("backing-filename") == backing, "QEMU_IMG_BACKING_MISMATCH")
            _require(not observed.get("snapshots"), "QCOW2_INTERNAL_SNAPSHOTS_UNSUPPORTED")
            artifacts.append(
                ImageArtifactV5d1(
                    item.path,
                    measured,
                    item.format,
                    info.st_size,
                    info.st_dev,
                    info.st_ino,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                    True,
                    backing,
                    snapshots,
                )
            )
        for item, descriptor, info in zip(expected_chain, descriptors, before, strict=True):
            # Also reopen the complete protected path to detect replacement or
            # directory-entry changes that a held descriptor alone would hide.
            reopened = _open_protected(item.path)
            try:
                _require(
                    _identity(os.fstat(descriptor)) == _identity(info)
                    and _identity(os.fstat(reopened)) == _identity(info)
                    and _immutable(descriptor)
                    and _immutable(reopened),
                    "IMAGE_CHANGED_DURING_MEASUREMENT",
                )
            finally:
                os.close(reopened)
        result = ImageMeasurementV5d1("IMAGE_MEASURED_LOCAL", (), tuple(artifacts))
    except _ImageRefusal as error:
        result = ImageMeasurementV5d1(
            "IMAGE_MEASUREMENT_BLOCKED", ("IMAGE_MEASUREMENT_BLOCKED", str(error))
        )
    except KeyboardInterrupt:
        result = ImageMeasurementV5d1(
            "IMAGE_MEASUREMENT_BLOCKED",
            ("IMAGE_MEASUREMENT_BLOCKED", "IMAGE_MEASUREMENT_INTERRUPTED"),
        )
    except (OSError, ValueError, TypeError, OverflowError, subprocess.SubprocessError):
        result = ImageMeasurementV5d1(
            "IMAGE_MEASUREMENT_BLOCKED",
            ("IMAGE_MEASUREMENT_BLOCKED", "IMAGE_OBSERVATION_INCOMPLETE"),
        )
    finally:
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except KeyboardInterrupt:
                cleanup_reasons.append("IMAGE_MEASUREMENT_INTERRUPTED")
            except OSError:
                cleanup_reasons.append("IMAGE_DESCRIPTOR_CLOSE_FAILED")
    if cleanup_reasons:
        return ImageMeasurementV5d1(
            "IMAGE_MEASUREMENT_BLOCKED",
            tuple(
                dict.fromkeys(("IMAGE_MEASUREMENT_BLOCKED", *result.reason_codes, *cleanup_reasons))
            ),
        )
    return result
