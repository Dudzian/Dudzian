"""Read-only NTFS qualification of the actual opened custody objects.

The installer owns ACL provisioning. Runtime checks inherited protection and
pins the directory chain and file without FILE_SHARE_DELETE until use ends.
No chmod, ACL repair, path-following fallback, or caller-selected trust list.
"""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
import ctypes
from ctypes import wintypes
import importlib
import ntpath
import os
from pathlib import Path

from bot_core.local_signing_custody import LocalSigningCustodyError

_READ_CONTROL = 0x00020000
_REPARSE_POINT = 0x400
_DIRECTORY = 0x10
_INHERIT_ONLY = 0x08
_REPLACE_RIGHTS = 0x10000000 | 0x000D0040  # GENERIC_ALL, DELETE, WRITE_DAC/OWNER, DELETE_CHILD
_WRITE_RIGHTS = _REPLACE_RIGHTS | 0x40000000 | 0x00000116
_DRIVE_FIXED = 3
_FILE_PERSISTENT_ACLS = 0x8


def _native():
    if os.name != "nt":
        raise LocalSigningCustodyError("native Windows custody filesystem is unavailable")
    return (
        importlib.import_module("win32file"),
        importlib.import_module("win32security"),
        importlib.import_module("win32api"),
    )


def _trusted_sids(security, api) -> frozenset[str]:
    token = security.OpenProcessToken(api.GetCurrentProcess(), 0x0008)
    try:
        user = security.GetTokenInformation(token, security.TokenUser)
        user_sid = user[0] if isinstance(user, tuple) else user
        trusted = {security.ConvertSidToStringSid(user_sid), "S-1-5-18", "S-1-5-32-544"}
    finally:
        token.Close()
    for account in ("NT SERVICE\\CryptoHunterBackend", "NT SERVICE\\TrustedInstaller"):
        try:
            trusted.add(
                security.ConvertSidToStringSid(security.LookupAccountName(None, account)[0])
            )
        except api.error:
            # An absent optional service has no authority in this installation.
            pass
    return frozenset(trusted)


def _qualify_acl(handle, security, trusted: frozenset[str], *, custody: bool) -> None:
    descriptor = security.GetSecurityInfo(handle, security.SE_FILE_OBJECT, 0x1 | 0x4)
    owner = descriptor.GetSecurityDescriptorOwner()
    dacl = descriptor.GetSecurityDescriptorDacl()
    if owner is None or security.ConvertSidToStringSid(owner) not in trusted or dacl is None:
        raise LocalSigningCustodyError("custody NTFS owner or DACL is unsafe")
    # Ancestors may grant creation of subdirectories (FILE_APPEND_DATA), but
    # must forbid replacement, reparse mutation, and security changes.
    forbidden = _WRITE_RIGHTS if custody else _REPLACE_RIGHTS | 0x40000112
    for index in range(dacl.GetAceCount()):
        ace = dacl.GetAce(index)
        kind, flags = ace[0]
        if flags & _INHERIT_ONLY or kind in (1, 6):  # deny ACEs cannot grant mutation
            continue
        if kind not in (0, 5):
            raise LocalSigningCustodyError("custody NTFS DACL has an unsupported ACE")
        if ace[1] & forbidden and security.ConvertSidToStringSid(ace[-1]) not in trusted:
            raise LocalSigningCustodyError("custody NTFS DACL permits untrusted modification")


def _normalized(path: str) -> str:
    if path[:8].upper() == "\\\\?\\UNC\\":
        path = "\\\\" + path[8:]
    elif path.startswith("\\\\?\\"):
        path = path[4:]
    return ntpath.normcase(ntpath.normpath(path))


def _qualify_local_path(path: Path, files) -> None:
    drive, tail = ntpath.splitdrive(_normalized(str(path)))
    # Reject UNC/device namespaces before touching the remote path. A mapped
    # SMB drive has DOS syntax but reports DRIVE_REMOTE, not DRIVE_FIXED.
    if (
        len(drive) != 2
        or drive[1] != ":"
        or not tail.startswith("\\")
        or files.GetDriveType(drive + "\\") != _DRIVE_FIXED
    ):
        raise LocalSigningCustodyError("custody requires an absolute local fixed-drive path")


def _volume_information(handle) -> tuple[int, int, str]:
    """Query the opened object, rather than a mutable drive-letter mapping."""
    windows_ctypes = importlib.import_module("ctypes")
    query = windows_ctypes.WinDLL("kernel32", use_last_error=True).GetVolumeInformationByHandleW
    query.argtypes = (
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPWSTR,
        wintypes.DWORD,
    )
    query.restype = wintypes.BOOL
    serial, maximum_component, flags = wintypes.DWORD(), wintypes.DWORD(), wintypes.DWORD()
    filesystem = ctypes.create_unicode_buffer(32)
    if not query(
        int(handle),
        None,
        0,
        ctypes.byref(serial),
        ctypes.byref(maximum_component),
        ctypes.byref(flags),
        filesystem,
        len(filesystem),
    ):
        raise windows_ctypes.WinError(windows_ctypes.get_last_error())
    return serial.value, flags.value, filesystem.value


def _qualify_volume(handle, files, serial: int) -> None:
    # VOLUME_NAME_GUID does not resolve an SMB share. Both drive type and FS
    # identity are checked against the actual pinned handle's local volume.
    volume_path = files.GetFinalPathNameByHandle(handle, 0x1)
    root, _ = ntpath.splitdrive(volume_path)
    if (
        not root.lower().startswith("\\\\?\\volume{")
        or not root.endswith("}")
        or files.GetDriveType(root + "\\") != _DRIVE_FIXED
    ):
        raise LocalSigningCustodyError("custody opened object is not on a local fixed volume")
    actual_serial, flags, filesystem = _volume_information(handle)
    if (
        actual_serial != serial & 0xFFFFFFFF
        or filesystem != "NTFS"
        or not flags & _FILE_PERSISTENT_ACLS
    ):
        raise LocalSigningCustodyError("custody opened volume is not qualified local NTFS")


def _qualify(handle, path: Path, native, trusted, *, directory: bool, custody: bool):
    files, security, _ = native
    info = files.GetFileInformationByHandle(handle)
    if (
        info[0] & _REPARSE_POINT
        or bool(info[0] & _DIRECTORY) != directory
        or files.GetFileType(handle) != 1  # FILE_TYPE_DISK
        or (not directory and info[7] != 1)
        or _normalized(files.GetFinalPathNameByHandle(handle, 0)) != _normalized(str(path))
    ):
        raise LocalSigningCustodyError("custody opened object is redirected or unsafe")
    _qualify_volume(handle, files, info[4])
    _qualify_acl(handle, security, trusted, custody=custody)
    return info


@contextmanager
def pinned_directory(directory: Path, *, create: bool = False):
    """Pin each ancestor before opening/creating its child, then qualify NTFS."""
    with ExitStack() as stack:
        try:
            native = _native()
            files, security, api = native
            _qualify_local_path(directory, files)
            trusted = _trusted_sids(security, api)
            for path in (*reversed(directory.parents), directory):
                if create and not path.exists():
                    path.mkdir()  # inherit installer/user protection, never repair ACLs
                handle = files.CreateFile(
                    str(path),
                    _READ_CONTROL | 0x80,
                    0x1 | 0x2,
                    None,
                    3,
                    0x02000000 | 0x00200000,
                    None,  # BACKUP_SEMANTICS, OPEN_REPARSE_POINT
                )
                stack.callback(handle.Close)
                _qualify(handle, path, native, trusted, directory=True, custody=path == directory)
        except LocalSigningCustodyError:
            raise
        except Exception as exc:
            raise LocalSigningCustodyError("custody NTFS directory qualification failed") from exc
        yield native, trusted


@contextmanager
def open_custody_file(path: Path, *, write: bool = False, create: bool = False):
    """Yield a CRT descriptor owned by a verified, non-replaceable native handle."""
    with pinned_directory(path.parent) as (native, trusted):
        handle = None
        descriptor = None
        try:
            files, _, _ = native
            try:
                before = path.lstat()
            except FileNotFoundError:
                before = None
            handle = files.CreateFile(
                str(path),
                0x80000000 | _READ_CONTROL | (0x40000000 if write else 0),
                0x1 | 0x2,
                None,
                4 if create else 3,
                0x00200000,
                None,
            )
            info = _qualify(handle, path, native, trusted, directory=False, custody=True)
            file_id = (info[8] << 32) | info[9]
            if before is not None and before.st_ino != file_id:
                raise LocalSigningCustodyError("custody file changed before native open")
            msvcrt = importlib.import_module("msvcrt")
            descriptor = msvcrt.open_osfhandle(
                int(handle), (os.O_RDWR if write else os.O_RDONLY) | os.O_BINARY
            )
            handle.Detach()  # CRT now owns the same verified handle
        except LocalSigningCustodyError:
            if handle is not None and descriptor is None:
                handle.Close()
            raise
        except Exception as exc:
            if handle is not None and descriptor is None:
                handle.Close()
            raise LocalSigningCustodyError("custody NTFS file qualification failed") from exc
        try:
            yield descriptor
        finally:
            os.close(descriptor)
