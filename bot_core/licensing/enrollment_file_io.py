"""Bounded, immutable file delivery; interpretation and audit belong to the protocol.

These helpers do not select trust roots or confer enrollment authority. Callers
must validate the canonical artifact and journal its digest in their own state
transaction. Parent directories must already exist. POSIX operations use pinned
directory descriptors; Windows requires the installed local NTFS protections.
"""

from __future__ import annotations

import ctypes
import errno
import importlib
import os
import secrets
import stat
from contextlib import ExitStack, contextmanager
from pathlib import Path
from threading import local

MAXIMUM_ARTIFACT_BYTES = 2 * 1024 * 1024
_posix_sessions = local()


class EnrollmentFileError(ValueError):
    """An artifact cannot be delivered safely or exceeds its transport bounds."""


def _checked_path(path: Path) -> Path:
    try:
        path = Path(path)
        if "\x00" in str(path) or ".." in path.parts or path.name in ("", ".", ".."):
            raise EnrollmentFileError("artifact path is not a plain file path")
        if os.name == "nt" and (":" in path.name or path.name.endswith((" ", "."))):
            raise EnrollmentFileError("artifact filename is unsafe on Windows")
        return path.absolute()
    except (TypeError, OSError) as exc:
        raise EnrollmentFileError("artifact path is invalid") from exc


def _checked_maximum(maximum: int) -> int:
    if type(maximum) is not int or not 0 < maximum <= MAXIMUM_ARTIFACT_BYTES:
        raise EnrollmentFileError("artifact limit must be within the fixed transport ceiling")
    return maximum


def _posix_euid() -> int:
    """Resolve the Unix-only owner API without assuming a Windows stub exists."""
    get_euid = getattr(os, "geteuid", None)
    if get_euid is None:
        raise EnrollmentFileError("POSIX effective UID is unavailable")
    return int(get_euid())


def _posix_fchmod(descriptor: int) -> None:
    """Retain explicit POSIX mode hardening; never weaken it for mypy win32."""
    chmod_fd = getattr(os, "fchmod", None)
    if chmod_fd is None:
        raise EnrollmentFileError("POSIX descriptor chmod is unavailable")
    chmod_fd(descriptor, 0o600)


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _qualify_file(info: os.stat_result, maximum: int) -> None:
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise EnrollmentFileError("artifact must be a regular file with exactly one link")
    if info.st_size < 0 or info.st_size > maximum:
        raise EnrollmentFileError("artifact exceeds the transport size limit")


def _verify_session_directory(path: Path, descriptor: int) -> None:
    for held_path, held_identity in getattr(_posix_sessions, "entries", ()):
        if path == held_path:
            info = os.fstat(descriptor)
            if (info.st_dev, info.st_ino) != held_identity:
                raise EnrollmentFileError("artifact directory changed during the active session")


def _session_prefix(path: Path) -> bool:
    return any(
        held_path.is_relative_to(path)
        for held_path, _identity in getattr(_posix_sessions, "entries", ())
    )


def _read_descriptor(descriptor: int, maximum: int) -> bytes:
    before = os.fstat(descriptor)
    _qualify_file(before, maximum)
    chunks = []
    remaining = maximum + 1
    while remaining:
        chunk = os.read(descriptor, min(65536, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    raw = b"".join(chunks)
    after = os.fstat(descriptor)
    if len(raw) > maximum:
        raise EnrollmentFileError("artifact exceeds the transport size limit")
    if _identity(before) != _identity(after) or len(raw) != after.st_size:
        raise EnrollmentFileError("artifact changed while being read")
    return raw


@contextmanager
def _posix_directory(
    directory: Path, *, create: bool = False, protect: bool = False, lock: bool = True
):
    """Anchor every component without following a replaceable symlink.

    A cooperative directory lock serializes helper operations; object checks
    remain mandatory for untrusted producers. Linux publication uses native
    renameat2(RENAME_NOREPLACE), including crash-safe single-link identity.
    """
    fcntl = importlib.import_module("fcntl")
    descriptors = []
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        descriptor = os.open(directory.anchor, flags)
        descriptors.append(descriptor)
        walked = Path(directory.anchor)
        _verify_session_directory(walked, descriptor)
        for component in directory.parts[1:]:
            walked = walked / component
            session_prefix = _session_prefix(walked)
            if create and not session_prefix:
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                else:
                    os.fsync(descriptor)
            try:
                descriptor = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError as exc:
                if session_prefix:
                    raise EnrollmentFileError(
                        "artifact directory disappeared during the active session"
                    ) from exc
                raise
            descriptors.append(descriptor)
            _verify_session_directory(walked, descriptor)
        if protect:
            info = os.fstat(descriptor)
            if info.st_uid not in (0, _posix_euid()) or info.st_mode & 0o022:
                raise EnrollmentFileError("artifact directory permits untrusted modification")
        if lock:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield descriptor
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _posix_read(directory: int, name: str, maximum: int, *, output: bool = False) -> bytes:
    before = os.stat(name, dir_fd=directory, follow_symlinks=False)
    _qualify_file(before, maximum)
    if output and (before.st_uid != _posix_euid() or stat.S_IMODE(before.st_mode) != 0o600):
        raise EnrollmentFileError("existing artifact output owner or permissions are unsafe")
    try:
        descriptor = os.open(
            name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=directory
        )
    except FileNotFoundError as exc:
        raise EnrollmentFileError("artifact changed before being opened") from exc
    try:
        if _identity(before) != _identity(os.fstat(descriptor)):
            raise EnrollmentFileError("artifact changed before being opened")
        raw = _read_descriptor(descriptor, maximum)
        try:
            after = os.stat(name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise EnrollmentFileError("artifact path changed while being read") from exc
        if _identity(after) != _identity(os.fstat(descriptor)):
            raise EnrollmentFileError("artifact path changed while being read")
        return raw
    finally:
        os.close(descriptor)


def _write_descriptor(descriptor: int, raw: bytes) -> None:
    remaining = memoryview(raw)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise EnrollmentFileError("artifact write did not make progress")
        remaining = remaining[written:]
    os.fsync(descriptor)


def _rename_noreplace(directory: int, source: str, destination: str) -> None:
    """Linux native immutable rename; never emulate it with a crash-fragile link."""
    native = ctypes.CDLL(None, use_errno=True)
    rename = getattr(native, "renameat2", None)
    if rename is None:
        raise EnrollmentFileError("atomic immutable publication requires Linux renameat2")
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(directory, os.fsencode(source), directory, os.fsencode(destination), 0x1):
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), destination)
        raise OSError(error, os.strerror(error), destination)


def _posix_write(path: Path, raw: bytes, maximum: int) -> None:
    with _posix_directory(path.parent, protect=True) as directory:
        try:
            existing = _posix_read(directory, path.name, maximum, output=True)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if existing != raw:
                raise EnrollmentFileError("artifact destination already contains different bytes")
            os.fsync(directory)
            return
        temporary = ".pdsa-" + secrets.token_hex(16) + ".tmp"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=directory,
        )
        try:
            _posix_fchmod(descriptor)
            _write_descriptor(descriptor, raw)
            try:
                _rename_noreplace(directory, temporary, path.name)
            except FileExistsError:
                if _posix_read(directory, path.name, maximum, output=True) != raw:
                    raise EnrollmentFileError(
                        "artifact destination already contains different bytes"
                    ) from None
        finally:
            os.close(descriptor)
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass
            os.fsync(directory)


def _windows_read(path: Path, maximum: int) -> bytes:
    import bot_core.windows_custody_filesystem as windows_fs

    # Use the native information rather than CRT fstat: older supported Windows
    # Python versions do not expose native link/file identity in CRT stat fields.
    with windows_fs.pinned_directory(path.parent) as (native, trusted):
        files, _, _ = native
        handle = files.CreateFile(
            str(path),
            0x80000000 | 0x00020000,
            0x1,  # no write/delete sharing while checking and reading the object
            None,
            3,  # OPEN_EXISTING
            0x00200000,  # OPEN_REPARSE_POINT
            None,
        )
        try:
            before = windows_fs._qualify(
                handle, path, native, trusted, directory=False, custody=True
            )
            remaining = (before[5] << 32) | before[6]
            if remaining > maximum:
                raise EnrollmentFileError("artifact exceeds the transport size limit")
            chunks = []
            while remaining:
                _, chunk = files.ReadFile(handle, min(65536, remaining))
                if not chunk or len(chunk) > remaining:
                    raise EnrollmentFileError("artifact changed while being read")
                chunks.append(chunk)
                remaining -= len(chunk)
            after = windows_fs._qualify(
                handle, path, native, trusted, directory=False, custody=True
            )
            if before[:2] + before[3:] != after[:2] + after[3:]:
                raise EnrollmentFileError("artifact changed while being read")
            return b"".join(chunks)
        finally:
            handle.Close()


def _windows_write(path: Path, raw: bytes, maximum: int) -> None:
    import bot_core.windows_custody_filesystem as windows_fs

    with windows_fs.pinned_directory(path.parent) as (native, trusted):
        try:
            # lstat distinguishes a missing entry from a native qualification
            # failure. Existing reparse points must never become a fallback.
            path.lstat()
        except FileNotFoundError:
            existing = None
        else:
            existing = _windows_read(path, maximum)
        if existing is not None:
            if existing != raw:
                raise EnrollmentFileError("artifact destination already contains different bytes")
            return
        files, _, _ = native
        temporary = path.parent / (".pdsa-" + secrets.token_hex(16) + ".tmp")
        handle = files.CreateFile(
            str(temporary),
            0x80000000 | 0x40000000 | 0x00020000,  # GENERIC_READ/WRITE, READ_CONTROL
            0,
            None,
            1,  # CREATE_NEW, never OPEN_ALWAYS
            0x00200000 | 0x80000000,  # OPEN_REPARSE_POINT, WRITE_THROUGH
            None,
        )
        descriptor = None
        try:
            try:
                windows_fs._qualify(
                    handle, temporary, native, trusted, directory=False, custody=True
                )
                msvcrt = importlib.import_module("msvcrt")
                descriptor = msvcrt.open_osfhandle(int(handle), os.O_RDWR | os.O_BINARY)
                handle.Detach()
                _write_descriptor(descriptor, raw)
            finally:
                if descriptor is None:
                    handle.Close()
                else:
                    os.close(descriptor)
            try:
                # Same directory/volume; no REPLACE_EXISTING flag. WRITE_THROUGH
                # commits the move before returning, including its metadata.
                files.MoveFileEx(str(temporary), str(path), 0x8)
            except Exception as exc:
                if getattr(exc, "winerror", None) not in (80, 183):
                    raise
                if _windows_read(path, maximum) != raw:
                    raise EnrollmentFileError(
                        "artifact destination already contains different bytes"
                    ) from exc
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def ensure_artifact_directory(path: Path) -> None:
    """Create a protected directory chain without following symlinks/reparse points."""
    path = _checked_path(path)
    try:
        if os.name == "nt":
            from bot_core.windows_custody_filesystem import pinned_directory

            with pinned_directory(path, create=True):
                pass
        else:
            with _posix_directory(path, create=True, protect=True) as directory:
                os.fsync(directory)
    except EnrollmentFileError:
        raise
    except Exception as exc:
        raise EnrollmentFileError("artifact directory cannot be created safely") from exc


@contextmanager
def _windows_session_lock(path: Path):
    import bot_core.windows_custody_filesystem as windows_fs

    with windows_fs.pinned_directory(path) as (native, trusted):
        files, _, _ = native
        lock_path = path / ".pdsa-session.lock"
        handle = files.CreateFile(
            str(lock_path),
            0x80000000 | 0x40000000 | 0x00020000,
            0x1 | 0x2,  # sharing excludes DELETE, including during the lock
            None,
            4,  # OPEN_ALWAYS
            0x00200000,
            None,
        )
        try:
            info = windows_fs._qualify(
                handle, lock_path, native, trusted, directory=False, custody=True
            )
            if info[5] or info[6]:
                raise EnrollmentFileError("artifact session lock contains unexpected data")
            overlapped = importlib.import_module("pywintypes").OVERLAPPED()
            # pywin32 omits the native dwReserved argument: its API takes
            # (handle, flags, size_low, size_high, overlapped).
            files.LockFileEx(handle, 0x2, 1, 0, overlapped)
            try:
                yield
            finally:
                # pywin32 UnlockFileEx also omits native dwReserved.
                files.UnlockFileEx(handle, 1, 0, overlapped)
        finally:
            handle.Close()


@contextmanager
def _posix_session_lock(path: Path):
    fcntl = importlib.import_module("fcntl")
    with _posix_directory(path, protect=True, lock=False) as directory:
        previous = getattr(_posix_sessions, "entries", ())
        if any(held_path == path for held_path, _identity in previous):
            # This thread already owns the same qualified session lock. A new
            # open-file-description flock would deadlock instead of nesting.
            yield
            return
        descriptor = os.open(
            ".pdsa-session.lock",
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
            0o600,
            dir_fd=directory,
        )
        try:
            info = os.fstat(descriptor)
            _qualify_file(info, 1)
            if (
                info.st_uid not in (0, _posix_euid())
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_size
            ):
                raise EnrollmentFileError("artifact session lock permissions or data are unsafe")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            directory_info = os.fstat(directory)
            _posix_sessions.entries = previous + (
                (path, (directory_info.st_dev, directory_info.st_ino)),
            )
            try:
                # Recheck the name after a potentially blocking acquisition.
                # Nested I/O then enforces this same pinned directory identity.
                with _posix_directory(path, protect=True, lock=False):
                    pass
                yield
            finally:
                _posix_sessions.entries = previous
        finally:
            os.close(descriptor)


@contextmanager
def artifact_directory_lock(path: Path):
    """Serialize a session while pinning its protected directory and lock object.

    This is a cooperative process lock, never an authorization decision. Artifact
    helpers may be called inside the context. Directory creation is explicit.
    """
    path = _checked_path(path)
    with ExitStack() as stack:
        try:
            stack.enter_context(
                _windows_session_lock(path) if os.name == "nt" else _posix_session_lock(path)
            )
        except EnrollmentFileError:
            raise
        except Exception as exc:
            raise EnrollmentFileError("artifact session directory cannot be locked safely") from exc
        yield


def read_artifact_file(path: Path, maximum: int = MAXIMUM_ARTIFACT_BYTES) -> bytes:
    """Read one bounded regular file, rejecting aliases and concurrent changes."""
    maximum = _checked_maximum(maximum)
    path = _checked_path(path)
    try:
        if os.name == "nt":
            return _windows_read(path, maximum)
        with _posix_directory(path.parent) as directory:
            return _posix_read(directory, path.name, maximum)
    except EnrollmentFileError:
        raise
    except FileNotFoundError:
        raise
    except Exception as exc:
        cause: BaseException | None = exc
        while cause is not None:
            if isinstance(cause, FileNotFoundError) or getattr(cause, "winerror", None) in (2, 3):
                raise FileNotFoundError(str(path)) from exc
            cause = cause.__cause__
        raise EnrollmentFileError("artifact file cannot be opened safely") from exc


def write_artifact_file(path: Path, raw: bytes, maximum: int = MAXIMUM_ARTIFACT_BYTES) -> None:
    """Durably publish bytes once; identical retries succeed without replacement."""
    maximum = _checked_maximum(maximum)
    path = _checked_path(path)
    if not isinstance(raw, bytes) or len(raw) > maximum:
        raise EnrollmentFileError("artifact bytes are invalid or exceed the transport size limit")
    try:
        if os.name == "nt":
            _windows_write(path, raw, maximum)
        else:
            _posix_write(path, raw, maximum)
    except EnrollmentFileError:
        raise
    except Exception as exc:
        raise EnrollmentFileError("artifact file cannot be written safely") from exc
