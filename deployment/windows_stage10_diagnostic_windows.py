"""Natywna publikacja nowej diagnostyki Stage 10 bez nadpisywania danych.

Uchwyty katalogów pozostają przypięte aż do końca operacji. Obie operacje
zmieniające katalog — utworzenie własnego pliku roboczego i publikacja — są
względne do zweryfikowanego uchwytu katalogu. To ochrona zapisu diagnostyki,
nie autorytet dowodów fizycznych ani nowy system Production Trust.
"""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
import ctypes
import importlib
import ntpath
import os
from pathlib import Path
import re
import secrets
from typing import Any, Iterator

from deployment.windows_stage10_diagnostic_output import (
    DiagnosticOutputError,
    MAX_DIAGNOSTIC_BYTES,
)

_READ_CONTROL = 0x00020000
_LIST_DIRECTORY = 0x00000001
_READ_ATTRIBUTES = 0x00000080
_SYNCHRONIZE = 0x00100000
_DELETE = 0x00010000
_GENERIC_READ = 0x80000000
_GENERIC_WRITE = 0x40000000
_SHARE_READ_WRITE = 0x1 | 0x2
_OPEN_REPARSE_POINT = 0x00200000
_BACKUP_SEMANTICS = 0x02000000
_SYNCHRONOUS_IO_NONALERT = 0x20
_NON_DIRECTORY_FILE = 0x40
_WRITE_THROUGH = 0x2
_OBJ_CASE_INSENSITIVE = 0x40
_OBJ_DONT_REPARSE = 0x1000
_FILE_OPEN = 1
_FILE_CREATE = 2
_FILE_RENAME_INFORMATION = 10
_NOT_FOUND = frozenset({0xC0000034, 0xC000003A, 0xC000000F})
_SYSTEM_SID = "S-1-5-18"
_ADMINISTRATORS_SID = "S-1-5-32-544"
# Stały SID TrustedInstaller jest dopuszczony wyłącznie dla przodków katalogu.
_TRUSTED_INSTALLER_SID = "S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464"


class _UnicodeString(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_uint16),
        ("maximum_length", ctypes.c_uint16),
        ("buffer", ctypes.c_void_p),
    ]


class _ObjectAttributes(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_uint32),
        ("root", ctypes.c_void_p),
        ("name", ctypes.POINTER(_UnicodeString)),
        ("attributes", ctypes.c_uint32),
        ("security_descriptor", ctypes.c_void_p),
        ("security_qos", ctypes.c_void_p),
    ]


class _IoStatus(ctypes.Structure):
    _fields_ = [("status_or_pointer", ctypes.c_void_p), ("information", ctypes.c_size_t)]


def _rename_record(directory: int, name: str) -> ctypes.Structure:
    encoded = name.encode("utf-16-le")

    class RenameInfo(ctypes.Structure):
        _fields_ = [
            ("replace_if_exists", ctypes.c_uint32),
            ("root_directory", ctypes.c_void_p),
            ("file_name_length", ctypes.c_uint32),
            ("file_name", ctypes.c_uint16 * (len(encoded) // 2)),
        ]

    record = RenameInfo()
    record.replace_if_exists = 0
    record.root_directory = directory
    record.file_name_length = len(encoded)
    ctypes.memmove(ctypes.addressof(record) + RenameInfo.file_name.offset, encoded, len(encoded))
    return record


def _checked_name(name: str) -> None:
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(10)),
        *(f"LPT{i}" for i in range(10)),
    }
    if (
        not isinstance(name, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name) is None
        or name.endswith(".")
        or name.split(".", 1)[0].upper() in reserved
    ):
        raise DiagnosticOutputError("Niedozwolona nazwa pliku diagnostyki Windows.")


class _Native:
    """Mała granica Win32/NT; modele testów zastępują całą tę granicę."""

    def __init__(self) -> None:
        if os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8:
            raise DiagnosticOutputError("Natywna publikacja wymaga 64-bit Windows.")
        self.fs = importlib.import_module("bot_core.windows_custody_filesystem")
        self.native = self.fs._native()
        self.files, self.security, self.api = self.native
        loader = getattr(ctypes, "WinDLL", None)
        if loader is None:
            raise DiagnosticOutputError("Brak natywnego loadera Windows.")
        self.kernel = loader("kernel32.dll", use_last_error=True, winmode=0x00000800)
        self.nt = loader("ntdll.dll", use_last_error=True, winmode=0x00000800)
        signatures = {
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
            "FlushFileBuffers": ([ctypes.c_void_p], ctypes.c_int),
            "CloseHandle": ([ctypes.c_void_p], ctypes.c_int),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.kernel, name)
            function.argtypes, function.restype = arguments, result
        self.nt.NtSetInformationFile.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_IoStatus),
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_int,
        ]
        self.nt.NtSetInformationFile.restype = ctypes.c_int32
        self.nt.NtCreateFile.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint32,
            ctypes.POINTER(_ObjectAttributes),
            ctypes.POINTER(_IoStatus),
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        self.nt.NtCreateFile.restype = ctypes.c_int32
        self.nt.RtlNtStatusToDosError.argtypes = [ctypes.c_int32]
        self.nt.RtlNtStatusToDosError.restype = ctypes.c_uint32

    @staticmethod
    def _failure() -> OSError:
        return OSError(getattr(ctypes, "get_last_error")(), "Odmowa natywnej operacji zapisu.")

    def _trusted(self) -> frozenset[str]:
        token = self.security.OpenProcessToken(self.api.GetCurrentProcess(), 0x0008)
        try:
            user = self.security.GetTokenInformation(token, self.security.TokenUser)
            sid = user[0] if isinstance(user, tuple) else user
            current_sid = self.security.ConvertSidToStringSid(sid)
        finally:
            token.Close()
        return frozenset({current_sid, _SYSTEM_SID, _ADMINISTRATORS_SID})

    @contextmanager
    def directory(self, root: Path) -> Iterator[dict[str, Any]]:
        # Nie rozwiązuj junctions przez resolve(), nawet dla walidacji wejścia.
        raw = str(root)
        drive, tail = ntpath.splitdrive(raw)
        if (
            len(drive) != 2
            or drive[1] != ":"
            or not tail.startswith("\\")
            or raw != ntpath.normpath(raw)
            or any(part in {".", ".."} for part in raw.replace("/", "\\").split("\\"))
        ):
            raise DiagnosticOutputError("Wymagany kanoniczny lokalny katalog Windows.")
        self.fs._qualify_local_path(root, self.files)
        trusted = self._trusted()
        ancestors = trusted | {_TRUSTED_INSTALLER_SID}
        with ExitStack() as stack:
            entries = []
            for path in (*reversed(root.parents), root):
                handle = self.files.CreateFile(
                    str(path),
                    _READ_CONTROL | _READ_ATTRIBUTES | _LIST_DIRECTORY,
                    _SHARE_READ_WRITE,
                    None,
                    3,
                    _OPEN_REPARSE_POINT | _BACKUP_SEMANTICS,
                    None,
                )
                stack.callback(handle.Close)
                allowed = trusted if path == root else ancestors
                self.fs._qualify(
                    handle, path, self.native, allowed, directory=True, custody=path == root
                )
                entries.append((handle, path, allowed))
            context = {
                "handle": int(entries[-1][0]),
                "path": root,
                "trusted": trusted,
                "entries": entries,
            }
            yield context

    def verify_directory(self, context: dict[str, Any]) -> None:
        for handle, path, trusted in context["entries"]:
            self.fs._qualify(
                handle, path, self.native, trusted, directory=True, custody=path == context["path"]
            )

    def _relative(self, context: dict[str, Any], name: str, *, create: bool) -> int | None:
        encoded = name.encode("utf-16-le")
        buffer = ctypes.create_string_buffer(encoded)
        string = _UnicodeString(len(encoded), len(encoded), ctypes.addressof(buffer))
        attributes = _ObjectAttributes(
            ctypes.sizeof(_ObjectAttributes),
            context["handle"],
            ctypes.pointer(string),
            _OBJ_CASE_INSENSITIVE | _OBJ_DONT_REPARSE,
            None,
            None,
        )
        handle, status = ctypes.c_void_p(), _IoStatus()
        access = _SYNCHRONIZE | _READ_ATTRIBUTES
        if create:
            access |= _GENERIC_READ | _GENERIC_WRITE | _READ_CONTROL | _DELETE
        result = self.nt.NtCreateFile(
            ctypes.byref(handle),
            access,
            ctypes.byref(attributes),
            ctypes.byref(status),
            None,
            0x80 if create else 0,
            0 if create else 0x7,
            _FILE_CREATE if create else _FILE_OPEN,
            _OPEN_REPARSE_POINT
            | _SYNCHRONOUS_IO_NONALERT
            | (_NON_DIRECTORY_FILE | _WRITE_THROUGH if create else 0),
            None,
            0,
        )
        if result < 0:
            if not create and result & 0xFFFFFFFF in _NOT_FOUND:
                return None
            error = self.nt.RtlNtStatusToDosError(result)
            raise OSError(error, "Odmowa względnej operacji pliku Windows.")
        if handle.value is None:
            raise DiagnosticOutputError("Brak uchwytu nowego pliku diagnostyki.")
        return int(handle.value)

    def exists(self, context: dict[str, Any], name: str) -> bool:
        handle = self._relative(context, name, create=False)
        if handle is None:
            return False
        self.close(handle)
        return True

    def create(self, context: dict[str, Any], name: str) -> int:
        handle = self._relative(context, name, create=True)
        if handle is None:
            raise DiagnosticOutputError("Nie utworzono nowego pliku diagnostyki.")
        return handle

    def verify_file(self, context: dict[str, Any], name: str, handle: int) -> None:
        self.fs._qualify(
            handle,
            context["path"] / name,
            self.native,
            context["trusted"],
            directory=False,
            custody=True,
        )

    def write(self, handle: int, raw: bytes) -> None:
        position = 0
        while position < len(raw):
            chunk = raw[position : position + 65536]
            buffer = ctypes.create_string_buffer(chunk)
            written = ctypes.c_uint32()
            if not self.kernel.WriteFile(handle, buffer, len(chunk), ctypes.byref(written), None):
                raise self._failure()
            if not 0 < written.value <= len(chunk):
                raise DiagnosticOutputError("Niepełny zapis diagnostyki Windows.")
            position += written.value
        if not self.kernel.FlushFileBuffers(handle):
            raise self._failure()

    def publish(self, context: dict[str, Any], handle: int, name: str) -> None:
        # Win32 FileRenameInfo rejected a non-null RootDirectory (error 87)
        # on the tested runner. Native FileRenameInformation retains the
        # verified directory-handle anchor and refuses destination replacement.
        record = _rename_record(context["handle"], name)
        io_status = _IoStatus()
        status = self.nt.NtSetInformationFile(
            handle,
            ctypes.byref(io_status),
            ctypes.byref(record),
            ctypes.sizeof(record),
            _FILE_RENAME_INFORMATION,
        )
        if status > 0:
            raise DiagnosticOutputError("Nieukończona natywna publikacja diagnostyki.")
        if status < 0:
            raise OSError(
                self.nt.RtlNtStatusToDosError(status),
                "Odmowa natywnej publikacji pliku.",
            )

    def close(self, handle: int) -> None:
        if not self.kernel.CloseHandle(handle):
            raise self._failure()


def publish_windows(root: Path, name: str, raw: bytes) -> None:
    """Publikuj wyłącznie nowy kompletny plik; nigdy nie usuwaj zastanych danych."""
    _checked_name(name)
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_DIAGNOSTIC_BYTES:
        raise DiagnosticOutputError("Niedozwolony rozmiar diagnostyki Windows.")
    try:
        native = _Native()
        with native.directory(root) as context:
            if native.exists(context, name):
                raise DiagnosticOutputError("Plik diagnostyki już istnieje.")
            temporary = ".stage10-" + secrets.token_hex(16) + ".partial"
            handle = native.create(context, temporary)
            try:
                native.verify_file(context, temporary, handle)
                native.write(handle, raw)
                native.verify_file(context, temporary, handle)
                native.verify_directory(context)
                native.publish(context, handle, name)
            finally:
                # Po błędzie pozostaje własny .partial; nie rozwiązuje się ścieżki
                # ponownie i nie usuwa żadnego pliku, nawet pliku tymczasowego.
                native.close(handle)
    except DiagnosticOutputError:
        raise
    except Exception as exc:
        raise DiagnosticOutputError(
            "Natywna publikacja diagnostyki Windows została odrzucona."
        ) from exc
