"""Cross-platform physical publication fences for persistence artifacts."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

MOVEFILE_REPLACE_EXISTING = 0x00000001
MOVEFILE_WRITE_THROUGH = 0x00000008


class PhysicalDurabilityError(OSError):
    """A required physical durability fence could not be completed."""


def fsync_file(path: str | Path) -> None:
    """Flush one existing regular file through the operating-system boundary."""

    try:
        mode = "r+b" if os.name == "nt" else "rb"
        with Path(path).open(mode) as handle:
            os.fsync(handle.fileno())
    except OSError as exc:
        raise PhysicalDurabilityError(f"could not make file durable: {path}") from exc


def _flush_directory_posix(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def flush_created_directory_metadata(path: str | Path, *, _platform: str | None = None) -> None:
    """Fence directory creation where POSIX supports directory fsync.

    Windows deliberately has no synthetic directory-handle flush here.  Its
    critical publication fence is MoveFileExW with WRITE_THROUGH.
    """

    platform = os.name if _platform is None else _platform
    if platform == "nt":
        return
    if platform != "posix":
        raise PhysicalDurabilityError(f"unsupported filesystem platform: {platform}")
    try:
        _flush_directory_posix(Path(path).resolve())
    except OSError as exc:
        raise PhysicalDurabilityError(f"could not make directory metadata durable: {path}") from exc


def _move_file_ex_windows(
    source: Path,
    target: Path,
    *,
    _move: Callable[[str, str, int], bool] | None = None,
) -> None:
    """Use the documented Win32 durable rename primitive."""

    win_error: Callable[[int], OSError] | None = None
    get_last_error: Callable[[], int] | None = None
    move = _move
    if move is None:
        import ctypes
        from ctypes import wintypes

        win_dll = cast(Any, getattr(ctypes, "WinDLL"))
        get_last_error = cast(Callable[[], int], getattr(ctypes, "get_last_error"))
        win_error = cast(Callable[[int], OSError], getattr(ctypes, "WinError"))
        move = win_dll("kernel32", use_last_error=True).MoveFileExW
        move.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD)  # type: ignore[attr-defined]
        move.restype = wintypes.BOOL  # type: ignore[attr-defined]
    flags = MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH
    if not move(str(source), str(target), flags):
        if win_error is not None and get_last_error is not None:
            raise win_error(get_last_error())
        raise OSError("MoveFileExW returned FALSE")


def publish_file_atomically_durably(
    temporary: str | Path,
    target: str | Path,
    *,
    _platform: str | None = None,
) -> None:
    """Atomically publish an already-fsynced temp file with an OS-native fence."""

    source, destination = Path(temporary).resolve(), Path(target).resolve()
    platform = os.name if _platform is None else _platform
    try:
        if platform == "nt":
            _move_file_ex_windows(source, destination)
        elif platform == "posix":
            os.replace(source, destination)
        else:
            raise PhysicalDurabilityError(f"unsupported filesystem platform: {platform}")
        fsync_file(destination)
        if platform == "posix":
            _flush_directory_posix(destination.parent)
    except OSError as exc:
        if isinstance(exc, PhysicalDurabilityError):
            raise
        raise PhysicalDurabilityError(f"could not publish durable file: {destination}") from exc


def atomic_write_bytes_durably(path: str | Path, payload: bytes) -> None:
    """Write and durably publish exact bytes using a transient same-directory file."""

    target = Path(path).resolve()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        publish_file_atomically_durably(temporary, target)
    except OSError as exc:
        if isinstance(exc, PhysicalDurabilityError):
            raise
        raise PhysicalDurabilityError(f"could not publish durable file: {target}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def reinforce_published_file_durability(path: str | Path, payload: bytes) -> None:
    """Reassert regular-file and pathname durability without changing its bytes."""

    atomic_write_bytes_durably(path, payload)
