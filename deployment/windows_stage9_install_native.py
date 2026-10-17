"""Native protected-file lifetime for the separately authorized MSI executor.

No installation is performed here. The context pins the original inputs before
the caller reads policy or repeats the existing qualification. Its handles stay
open until the caller has waited for Windows Installer and written its report.
ACL decisions belong to windows_runner_acl, not a second ACL implementation.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import io
import json
import os
import re
import stat
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import IO, Any, Iterator, cast
from uuid import uuid4

from deployment.windows_runner_acl import (
    SYSTEM_SID,
    WINDOWS_ACL_TREE_SCRIPT,
    _native_system_directory,
    _path,
    native_powershell_path,
    validate_acl_tree,
)
from deployment.windows_stage10_runner_preflight import _read_json_command

_READ = 0x80000000
_WRITE = 0x40000000
_READ_ATTRIBUTES = 0x80
_SHARE_READ = 1
_SHARE_WRITE = 2
_OPEN_EXISTING = 3
_CREATE_NEW = 1
_OPEN_REPARSE_POINT = 0x00200000
_BACKUP_SEMANTICS = 0x02000000
_MAX_ENTRIES = 50000
_MAX_MSI_BYTES = 4 * 1024**3
_MAX_ARCHIVE_READ = 64 * 1024**2


class InstallationIntegrityError(ValueError):
    """Protected input, path, materialized bytes or ACL failed closed."""


class NativeInstallationUnavailable(RuntimeError):
    """A native x64 Windows boundary could not be established."""


class _FileInfo(ctypes.Structure):
    _fields_ = [
        ("attributes", ctypes.c_uint32),
        ("creation", ctypes.c_uint32 * 2),
        ("access", ctypes.c_uint32 * 2),
        ("write", ctypes.c_uint32 * 2),
        ("volume", ctypes.c_uint32),
        ("size_high", ctypes.c_uint32),
        ("size_low", ctypes.c_uint32),
        ("links", ctypes.c_uint32),
        ("index_high", ctypes.c_uint32),
        ("index_low", ctypes.c_uint32),
    ]


class _BasicInfo(ctypes.Structure):
    _fields_ = [
        ("creation", ctypes.c_int64),
        ("access", ctypes.c_int64),
        ("write", ctypes.c_int64),
        ("change", ctypes.c_int64),
        ("attributes", ctypes.c_uint32),
    ]


class _SecurityAttributes(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_uint32),
        ("descriptor", ctypes.c_void_p),
        ("inherit", ctypes.c_int),
    ]


@dataclass(frozen=True)
class _Identity:
    final_path: str
    attributes: int
    file_id: tuple[int, int, int]
    links: int
    size: int
    write: int
    change: int


def _checked_path(path: Path) -> Path:
    # Never resolve(): resolving first would erase evidence of a junction.
    candidate = _path(str(path))
    if candidate is None or candidate.parent == candidate:
        raise InstallationIntegrityError("a canonical absolute local Windows path is required")
    return Path(str(candidate))


def _check_identity(
    path: Path, identity: _Identity, *, directory: bool, native_os_binary: bool = False
) -> None:
    if (
        str(path).casefold() != identity.final_path.casefold()
        or identity.attributes & 0x400
        or bool(identity.attributes & 0x10) != directory
        or identity.links < 1
        or (identity.links != 1 and not native_os_binary)
    ):
        raise InstallationIntegrityError("reparse point, hardlink or changed final path")


def _collect_acl_tree(root: Path) -> Any:
    encoded_root = base64.b64encode(str(root).encode("utf-16-le")).decode("ascii")
    script = (
        WINDOWS_ACL_TREE_SCRIPT
        + "\n$ErrorActionPreference='Stop'\n"
        + "$root=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String('"
        + encoded_root
        + "'))\nCollect-RunnerAclTree -Root $root | ConvertTo-Json -Depth 14 -Compress"
    )
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return _read_json_command(
        [native_powershell_path(), "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded]
    )


def _require_safe_acl(root: Path) -> None:
    if validate_acl_tree(
        _collect_acl_tree(root),
        runner_root=str(root),
        service_sid=SYSTEM_SID,
        trusted_write_sids=frozenset(),
    ):
        raise InstallationIntegrityError("protected tree ACL qualification failed")


class _Native:
    """Small Win32 handle adapter; tests replace this boundary, never run MSI."""

    def __init__(self) -> None:
        if os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8:
            raise NativeInstallationUnavailable("native x64 Windows is required")
        loader = getattr(ctypes, "WinDLL", None)
        if loader is None:
            raise NativeInstallationUnavailable("native Windows DLL loading is unavailable")
        self.kernel = loader("kernel32.dll", use_last_error=True, winmode=0x00000800)
        self.security = loader("advapi32.dll", use_last_error=True, winmode=0x00000800)
        # SYSTEM_INFO begins with the processor architecture WORD. The buffer
        # is larger than the native structure and aligned to a pointer.
        system_info = (ctypes.c_uint64 * 8)()
        self.kernel.GetNativeSystemInfo.argtypes = [ctypes.c_void_p]
        self.kernel.GetNativeSystemInfo.restype = None
        self.kernel.GetNativeSystemInfo(ctypes.byref(system_info))
        if system_info[0] & 0xFFFF != 9:  # PROCESSOR_ARCHITECTURE_AMD64
            raise NativeInstallationUnavailable("native AMD64 Windows is required")
        signatures: dict[str, tuple[list[Any], Any]] = {
            "CreateFileW": (
                [
                    ctypes.c_wchar_p,
                    ctypes.c_uint32,
                    ctypes.c_uint32,
                    ctypes.c_void_p,
                    ctypes.c_uint32,
                    ctypes.c_uint32,
                    ctypes.c_void_p,
                ],
                ctypes.c_void_p,
            ),
            "GetFileInformationByHandle": (
                [ctypes.c_void_p, ctypes.POINTER(_FileInfo)],
                ctypes.c_int,
            ),
            "GetFileInformationByHandleEx": (
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32],
                ctypes.c_int,
            ),
            "GetFinalPathNameByHandleW": (
                [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32],
                ctypes.c_uint32,
            ),
            "CloseHandle": ([ctypes.c_void_p], ctypes.c_int),
            "ReadFile": (
                [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_void_p,
                ],
                ctypes.c_int,
            ),
            "WriteFile": (
                [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_void_p,
                ],
                ctypes.c_int,
            ),
            "SetFilePointerEx": (
                [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_uint32],
                ctypes.c_int,
            ),
            "SetEndOfFile": ([ctypes.c_void_p], ctypes.c_int),
            "FlushFileBuffers": ([ctypes.c_void_p], ctypes.c_int),
            "CreateDirectoryW": ([ctypes.c_wchar_p, ctypes.c_void_p], ctypes.c_int),
            "LocalFree": ([ctypes.c_void_p], ctypes.c_void_p),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.kernel, name)
            function.argtypes, function.restype = arguments, result
        convert = self.security.ConvertStringSecurityDescriptorToSecurityDescriptorW
        convert.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        convert.restype = ctypes.c_int

    @staticmethod
    def _failure() -> OSError:
        error = getattr(ctypes, "get_last_error")()
        return OSError(error, "native protected-file operation failed")

    @contextmanager
    def _protected_security(self) -> Iterator[_SecurityAttributes]:
        descriptor = ctypes.c_void_p()
        # Explicit protected DACL; the transfer identity receives no access.
        sddl = "O:BAG:BAD:P(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)"
        if not self.security.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl, 1, ctypes.byref(descriptor), None
        ):
            raise self._failure()
        try:
            yield _SecurityAttributes(ctypes.sizeof(_SecurityAttributes), descriptor, 0)
        finally:
            self.kernel.LocalFree(descriptor)

    def open(
        self, path: Path, *, create: bool = False, writable: bool = False, log: bool = False
    ) -> int:
        with self._protected_security() as security:
            handle = self.kernel.CreateFileW(
                str(path),
                _READ_ATTRIBUTES if log else _READ | (_WRITE if writable else 0),
                _SHARE_READ | (_SHARE_WRITE if log else 0),
                ctypes.byref(security) if create else None,
                _CREATE_NEW if create else _OPEN_EXISTING,
                _OPEN_REPARSE_POINT | _BACKUP_SEMANTICS,
                None,
            )
            # Capture CreateFileW's error before descriptor cleanup can change
            # the thread's last-error value.
            if handle in {None, ctypes.c_void_p(-1).value}:
                raise self._failure()
        return int(handle)

    def close(self, handle: int) -> None:
        if not self.kernel.CloseHandle(handle):
            raise self._failure()

    def identity(self, handle: int) -> _Identity:
        info, basic = _FileInfo(), _BasicInfo()
        if not self.kernel.GetFileInformationByHandle(handle, ctypes.byref(info)):
            raise self._failure()
        if not self.kernel.GetFileInformationByHandleEx(
            handle, 0, ctypes.byref(basic), ctypes.sizeof(basic)
        ):
            raise self._failure()
        buffer = ctypes.create_unicode_buffer(32768)
        length = self.kernel.GetFinalPathNameByHandleW(handle, buffer, len(buffer), 0)
        value = buffer.value
        if length == 0 or length >= len(buffer) or not value.startswith("\\\\?\\"):
            raise InstallationIntegrityError("native final path unavailable")
        return _Identity(
            value[4:],
            info.attributes,
            (info.volume, info.index_high, info.index_low),
            info.links,
            info.size_high * 2**32 + info.size_low,
            basic.write,
            basic.change,
        )

    def chunks(self, handle: int) -> Iterator[bytes]:
        if not self.kernel.SetFilePointerEx(handle, 0, None, 0):
            raise self._failure()
        buffer = ctypes.create_string_buffer(1024 * 1024)
        count = ctypes.c_uint32()
        while True:
            if not self.kernel.ReadFile(handle, buffer, len(buffer), ctypes.byref(count), None):
                raise self._failure()
            if count.value == 0:
                return
            yield buffer.raw[: count.value]

    def write(self, handle: int, chunks: Iterator[bytes]) -> None:
        if not self.kernel.SetFilePointerEx(handle, 0, None, 0):
            raise self._failure()
        for chunk in chunks:
            count = ctypes.c_uint32()
            if not self.kernel.WriteFile(handle, chunk, len(chunk), ctypes.byref(count), None):
                raise self._failure()
            if count.value != len(chunk):
                raise InstallationIntegrityError("incomplete protected-file write")
        if not self.kernel.SetEndOfFile(handle) or not self.kernel.FlushFileBuffers(handle):
            raise self._failure()

    def mkdir(self, path: Path) -> None:
        with self._protected_security() as security:
            if not self.kernel.CreateDirectoryW(str(path), ctypes.byref(security)):
                raise self._failure()


class _HandleReader(io.RawIOBase):
    """Seekable file adapter over the pinned handle, with no path reopening."""

    def __init__(self, native: _Native, handle: int) -> None:
        self.native, self.handle = native, handle
        self.position = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence not in {0, 1, 2}:
            raise ValueError("invalid seek origin")
        origin = (0, self.position, self.native.identity(self.handle).size)[whence]
        if origin + offset < 0:
            raise ValueError("negative seek")
        self.position = origin + offset
        return self.position

    def read(self, size: int = -1) -> bytes:
        if not self.native.kernel.SetFilePointerEx(self.handle, self.position, None, 0):
            raise self.native._failure()
        remaining = self.native.identity(self.handle).size - self.position
        length = min(remaining, size) if size >= 0 else remaining
        if length < 0 or length > _MAX_ARCHIVE_READ:
            raise InstallationIntegrityError("bounded archive read exceeded")
        buffer = ctypes.create_string_buffer(length)
        count = ctypes.c_uint32()
        if not self.native.kernel.ReadFile(self.handle, buffer, length, ctypes.byref(count), None):
            raise self.native._failure()
        self.position += count.value
        return buffer.raw[: count.value]


@dataclass
class _Pin:
    path: Path
    handle: int
    directory: bool
    identity: _Identity
    mutable: bool = False
    native_os_binary: bool = False


class LockedInputs:
    """Live capability owned by locked_installation_inputs, never serialized."""

    def __init__(
        self,
        native: _Native,
        *,
        artifact_root: Path,
        policy_path: Path,
        executor_root: Path,
        staging_root: Path,
    ) -> None:
        self._native = native
        self.artifact_root, self.policy_path = artifact_root, policy_path
        self.executor_root, self.staging_root = executor_root, staging_root
        self._pins: dict[str, _Pin] = {}
        self._file_ids: set[tuple[int, int, int]] = set()
        self._trees = (artifact_root, policy_path.parent, executor_root, staging_root)
        self._closed = False
        self._materialized = False
        self._native_paths: set[str] = set()
        self._attempt_report: _Pin | None = None
        self._attempt_directory: Path | None = None

    def _pin(
        self,
        path: Path,
        *,
        directory: bool,
        log: bool = False,
        writable: bool = False,
        create: bool = False,
        native_os_binary: bool = False,
    ) -> _Pin:
        key = str(path).casefold()
        if key in self._pins:
            pin = self._pins[key]
            if pin.directory != directory:
                raise InstallationIntegrityError("inconsistent protected path type")
            return pin
        if len(self._pins) >= _MAX_ENTRIES:
            raise InstallationIntegrityError("bounded protected inventory exceeded")
        handle = self._native.open(path, create=create, writable=writable, log=log)
        try:
            identity = self._native.identity(handle)
            _check_identity(path, identity, directory=directory, native_os_binary=native_os_binary)
            if identity.file_id in self._file_ids:
                raise InstallationIntegrityError("protected tree aliases an existing object")
            pin = _Pin(
                path, handle, directory, identity, mutable=log, native_os_binary=native_os_binary
            )
            self._pins[key] = pin
            self._file_ids.add(identity.file_id)
            return pin
        except BaseException:
            self._native.close(handle)
            raise

    def _walk(self, root: Path, depth: int = 0) -> set[str]:
        if depth > 256:
            raise InstallationIntegrityError("bounded protected directory depth exceeded")
        self._pin(root, directory=True)
        found = {str(root).casefold()}
        with os.scandir(root) as entries:
            for entry in entries:
                path = Path(entry.path)
                metadata = entry.stat(follow_symlinks=False)
                if (
                    stat.S_ISLNK(metadata.st_mode)
                    or getattr(metadata, "st_file_attributes", 0) & 0x400
                ):
                    raise InstallationIntegrityError("protected tree contains a reparse point")
                if stat.S_ISDIR(metadata.st_mode):
                    found.update(self._walk(path, depth + 1))
                elif stat.S_ISREG(metadata.st_mode):
                    self._pin(path, directory=False)
                    found.add(str(path).casefold())
                else:
                    raise InstallationIntegrityError("protected tree contains a special file")
        return found

    def _inventory(self) -> set[str]:
        found: set[str] = set()
        for root in self._trees:
            for parent in reversed(root.parents):
                self._pin(parent, directory=True)
                found.add(str(parent).casefold())
            found.update(self._walk(root))
        return found | self._native_paths

    def _establish(self) -> None:
        # Each path is pinned before any data/ACL observation is trusted.
        self._inventory()
        if str(self.policy_path).casefold() not in self._pins:
            raise InstallationIntegrityError("independent host policy is missing")
        if self._pins[str(self.policy_path).casefold()].directory:
            raise InstallationIntegrityError("independent host policy must be a regular file")
        for root in self._trees:
            _require_safe_acl(root)
        self.verify_pins()

    def verify_pins(self) -> None:
        if self._closed:
            raise InstallationIntegrityError("protected input context is closed")
        before = set(self._pins)
        if self._inventory() != before:
            raise InstallationIntegrityError("protected tree inventory changed")
        for pin in self._pins.values():
            identity = self._native.identity(pin.handle)
            _check_identity(
                pin.path, identity, directory=pin.directory, native_os_binary=pin.native_os_binary
            )
            if pin.mutable:
                stable = (
                    identity.file_id,
                    identity.final_path,
                    identity.attributes,
                    identity.links,
                )
                old = pin.identity
                if stable != (old.file_id, old.final_path, old.attributes, old.links):
                    raise InstallationIntegrityError("protected log identity changed")
            elif identity != pin.identity:
                raise InstallationIntegrityError("pinned protected object changed")

    def _refresh_owned_directory_metadata(self) -> None:
        # Only invoked after our own CREATE_NEW operations in staging.
        for pin in self._pins.values():
            if pin.directory and pin.path in {self.staging_root, self._attempt_directory}:
                fresh = self._native.identity(pin.handle)
                _check_identity(pin.path, fresh, directory=True)
                if fresh.file_id != pin.identity.file_id:
                    raise InstallationIntegrityError("protected staging directory changed")
                pin.identity = fresh

    def materialize_msi(
        self, *, archive: Path, msi_member: str, expected_msi_sha256: str
    ) -> LockedInstallation:
        if self._materialized:
            raise InstallationIntegrityError("one installation attempt per protected context")
        try:
            return self._materialize_msi(
                archive=archive,
                msi_member=msi_member,
                expected_msi_sha256=expected_msi_sha256,
            )
        except BaseException as exc:
            # No process capability escaped. Preserve a sanitized failure record
            # when materialization fails before returning to the executor.
            if self._attempt_report is not None:
                try:
                    self._write_pinned_report(
                        self._attempt_report,
                        {
                            "schema_version": 1,
                            "status": "MATERIALIZATION_FAILED",
                            "overall_status": "BLOCKED",
                            "installer_invoked": False,
                            "automatic_retry_authorized": False,
                            "run_stage10": False,
                        },
                    )
                except (OSError, RuntimeError, ValueError):
                    pass
            if isinstance(exc, (zipfile.BadZipFile, zipfile.LargeZipFile, EOFError)):
                raise InstallationIntegrityError("qualified ZIP could not be read") from exc
            raise

    def _write_pinned_report(self, pin: _Pin, report: dict[str, Any]) -> None:
        data = (
            json.dumps(report, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
        ).encode()
        self._native.write(pin.handle, iter((data,)))
        pin.identity = self._native.identity(pin.handle)

    def _materialize_msi(
        self, *, archive: Path, msi_member: str, expected_msi_sha256: str
    ) -> LockedInstallation:
        self.verify_pins()
        self._materialized = True
        archive = _checked_path(archive)
        if self.artifact_root not in archive.parents:
            raise InstallationIntegrityError("original archive is outside protected artifact tree")
        source = self._pins.get(str(archive).casefold())
        if source is None or source.directory:
            raise InstallationIntegrityError("original qualified archive is not pinned")
        if not isinstance(msi_member, str) or not isinstance(expected_msi_sha256, str):
            raise InstallationIntegrityError("exact qualified MSI member and digest are required")
        member = PurePosixPath(msi_member)
        if (
            not msi_member
            or "\\" in msi_member
            or member.is_absolute()
            or any(part in {"", ".", ".."} for part in msi_member.split("/"))
            or member.suffix.casefold() != ".msi"
            or re.fullmatch(r"[0-9a-f]{64}", expected_msi_sha256) is None
        ):
            raise InstallationIntegrityError("exact qualified MSI member and digest are required")
        directory = self.staging_root / ("installation-" + str(uuid4()))
        self._native.mkdir(directory)
        self._attempt_directory = directory
        self._pin(directory, directory=True)
        msi_path, log_path, report_path = (
            directory / name
            for name in ("qualified.msi", "windows-installer.log", "installation-report.json")
        )
        report = self._pin(report_path, directory=False, create=True, writable=True)
        self._attempt_report = report
        self._write_pinned_report(
            report,
            {
                "schema_version": 1,
                "status": "MATERIALIZATION_INCOMPLETE",
                "overall_status": "BLOCKED",
                "installer_invoked": False,
                "automatic_retry_authorized": False,
                "run_stage10": False,
            },
        )
        # The writable creation handle is closed before opening the immutable
        # read pin. Authorization is never evaluated in this gap: the final hash
        # is computed from the new read handle and retained through completion.
        write_handle = self._native.open(msi_path, create=True, writable=True)
        try:
            with zipfile.ZipFile(
                cast(IO[bytes], _HandleReader(self._native, source.handle))
            ) as package:
                matches = [item for item in package.infolist() if item.filename == msi_member]
                if len(matches) != 1:
                    raise InstallationIntegrityError(
                        "qualified MSI ZIP member is absent or duplicated"
                    )
                item = matches[0]
                if (
                    item.is_dir()
                    or item.flag_bits & 1
                    or not 0 < item.file_size <= _MAX_MSI_BYTES
                    or (item.external_attr >> 16) & 0o170000 not in {0, stat.S_IFREG}
                ):
                    raise InstallationIntegrityError("qualified MSI ZIP member type/size is unsafe")
                with package.open(item) as stream:
                    self._native.write(write_handle, _stream_chunks(stream, item.file_size))
        finally:
            self._native.close(write_handle)
        msi_pin = self._pin(msi_path, directory=False)
        if _handle_sha256(self._native, msi_pin.handle) != expected_msi_sha256:
            raise InstallationIntegrityError("materialized MSI SHA-256 differs from qualification")
        log = self._pin(log_path, directory=False, create=True, log=True)
        self._refresh_owned_directory_metadata()
        _require_safe_acl(directory)
        executable = _native_msiexec_path()
        # Native System32 is the same OS trust boundary used by the existing
        # native PowerShell collector. Windows services may maintain legitimate
        # WinSxS hardlinks to this OS executable; this exception never applies
        # to downloaded inputs, policy, checkout or materialized payload.
        for parent in reversed(executable.parents):
            self._pin(parent, directory=True)
            self._native_paths.add(str(parent).casefold())
        self._pin(executable, directory=False, native_os_binary=True)
        self._native_paths.add(str(executable).casefold())
        result = LockedInstallation(self, msi_pin, log, report, executable, expected_msi_sha256)
        result.verify_pins()
        return result

    def close(self) -> None:
        self._closed = True
        first: OSError | None = None
        for pin in reversed(list(self._pins.values())):
            try:
                self._native.close(pin.handle)
            except OSError as exc:
                first = first or exc
        self._pins.clear()
        self._file_ids.clear()
        if first is not None:
            raise first


def _stream_chunks(stream: IO[bytes], expected_size: int) -> Iterator[bytes]:
    total = 0
    while chunk := stream.read(1024 * 1024):
        total += len(chunk)
        if total > expected_size:
            raise InstallationIntegrityError("MSI decompression exceeded qualified size")
        yield chunk
    if total != expected_size:
        raise InstallationIntegrityError("MSI decompression was incomplete")


def _handle_sha256(native: _Native, handle: int) -> str:
    digest = hashlib.sha256()
    for chunk in native.chunks(handle):
        digest.update(chunk)
    return digest.hexdigest()


def _native_msiexec_path() -> Path:
    root = _path(_native_system_directory())
    if root is None or root.name.casefold() != "system32":
        raise NativeInstallationUnavailable("native Windows System32 is unavailable")
    return Path(str(root / "msiexec.exe"))


class LockedInstallation:
    def __init__(
        self,
        inputs: LockedInputs,
        msi: _Pin,
        log: _Pin,
        report: _Pin,
        executable: Path,
        digest: str,
    ) -> None:
        self._inputs, self._msi, self._report, self._digest = inputs, msi, report, digest
        self.msi_path, self.log_path = msi.path, log.path
        self.report_path, self.msiexec_path = report.path, executable
        self.journal_path = report.path.parent / "installation-intent.json"
        self._journal_recorded = False
        self.argv = [
            str(executable),
            "/i",
            str(msi.path),
            "/qn",
            "/norestart",
            "REBOOT=ReallySuppress",
            "ALLUSERS=1",
            "MSIINSTALLPERUSER=0",
            "/L*V",
            str(log.path),
        ]

    def verify_pins(self) -> None:
        self._inputs.verify_pins()
        if _handle_sha256(self._inputs._native, self._msi.handle) != self._digest:
            raise InstallationIntegrityError("pinned MSI content changed before installation")

    def write_report(self, report: dict[str, Any]) -> None:
        if self._inputs._closed:
            raise InstallationIntegrityError(
                "cannot write a report after releasing protected handles"
            )
        if not self._journal_recorded:
            data = (
                json.dumps(report, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
            ).encode()
            handle = self._inputs._native.open(self.journal_path, create=True, writable=True)
            try:
                self._inputs._native.write(handle, iter((data,)))
            finally:
                self._inputs._native.close(handle)
            self._inputs._pin(self.journal_path, directory=False)
            self._inputs._refresh_owned_directory_metadata()
            self._journal_recorded = True
        self._inputs._write_pinned_report(self._report, report)


@contextmanager
def locked_installation_inputs(
    *, artifact_root: Path, policy_path: Path, executor_root: Path, staging_root: Path
) -> Iterator[LockedInputs]:
    """Pin protected original data/code and retain it across process completion.

    There is no supplied ACL snapshot, policy trust list or platform override.
    The staging root must already exist under independent protected host ACLs.
    All staging outputs survive errors for diagnosis; this helper never retries.
    """
    native = _Native()
    roots = [
        _checked_path(path) for path in (artifact_root, policy_path, executor_root, staging_root)
    ]
    artifact, policy, executor, staging = roots
    # Keep code, policy, downloaded data and mutable outputs in separate trees.
    trees = (artifact, policy.parent, executor, staging)
    if any(
        left == right or left in right.parents or right in left.parents
        for index, left in enumerate(trees)
        for right in trees[index + 1 :]
    ):
        raise InstallationIntegrityError("protected trust domains must have separate roots")
    inputs = LockedInputs(
        native,
        artifact_root=artifact,
        policy_path=policy,
        executor_root=executor,
        staging_root=staging,
    )
    try:
        inputs._establish()
        yield inputs
    finally:
        inputs.close()
