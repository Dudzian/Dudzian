"""Wyłączna publikacja diagnostyki w osobno zatwierdzonym katalogu.

Nie naprawiamy uprawnień i nie usuwamy żadnych istniejących nazw. Operacje
dotyczą przypiętych katalogów; pełny plik otrzymuje nazwę końcową wyłącznie
przez natywną, atomową publikację bez zastępowania. Pozostałości .partial
nie są raportami ani autorytatywnymi dowodami Stage 10.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from typing import Callable, Iterator

from bot_core.licensing.enrollment_file_io import (
    EnrollmentFileError,
    _rename_noreplace,
    _write_descriptor,
)

MAX_DIAGNOSTIC_BYTES = 2 * 1024 * 1024


class DiagnosticOutputError(ValueError):
    """Brak upoważnienia lub gwarancji bezpiecznej publikacji diagnostyki."""


def _checked_destination(
    output: Path, root: Path, protected_inputs: tuple[Path, ...]
) -> tuple[Path, str]:
    for path in (root, output):
        if not path.is_absolute() or ".." in path.parts or "\x00" in str(path):
            raise DiagnosticOutputError("wymagana bezwzględna ścieżka bez przekierowań")
    name = output.name
    stem = name.split(".", 1)[0].upper()
    if (
        output.parent != root
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,120}\.json", name) is None
        or stem in {"CON", "PRN", "AUX", "NUL"}
        or re.fullmatch(r"(?:COM|LPT)[1-9]", stem) is not None
        or any(output == path.absolute() for path in protected_inputs)
    ):
        raise DiagnosticOutputError("kolizja wejścia lub niedozwolona nazwa raportu")
    return root, name


def _posix_owner() -> int:
    getter = getattr(os, "geteuid", None)
    if getter is None:
        raise DiagnosticOutputError("brak natywnej obserwacji właściciela katalogu")
    return int(getter())


def _qualify_directory(info: os.stat_result, *, final: bool) -> None:
    owner = _posix_owner()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid not in ({owner} if final else {0, owner}):
        raise DiagnosticOutputError("niezatwierdzony właściciel lub typ katalogu")
    if info.st_mode & 0o022 and (final or not info.st_mode & stat.S_ISVTX):
        raise DiagnosticOutputError("katalog dopuszcza nieuprawnioną zmianę zawartości")


@contextmanager
def _pinned_posix_directory(root: Path) -> Iterator[tuple[int, Callable[[], None]]]:
    """Nie rozwiązuj ponownie ścieżki przy zapisie ani publikacji.

    Sticky ancestors (np. /tmp) chronią nazwę katalogu należącego do operatora.
    Katalog końcowy musi należeć do bieżącego UID i zabraniać obcego zapisu.
    Zmiana nazwy w łańcuchu powoduje odmowę; nie przekierowuje operacji dir_fd.
    """
    descriptors: list[int] = []
    entries: list[tuple[int, str, int, tuple[int, int]]] = []
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        descriptor = os.open(root.anchor, flags)
        descriptors.append(descriptor)
        _qualify_directory(os.fstat(descriptor), final=root == Path(root.anchor))
        for index, component in enumerate(root.parts[1:]):
            parent = descriptor
            descriptor = os.open(component, flags, dir_fd=parent)
            descriptors.append(descriptor)
            info = os.fstat(descriptor)
            _qualify_directory(info, final=index == len(root.parts) - 2)
            entries.append((parent, component, descriptor, (info.st_dev, info.st_ino)))

        def check() -> None:
            for index, (parent, name, child, identity) in enumerate(entries):
                observed = os.stat(name, dir_fd=parent, follow_symlinks=False)
                current = os.fstat(child)
                _qualify_directory(observed, final=index == len(entries) - 1)
                _qualify_directory(current, final=index == len(entries) - 1)
                if (observed.st_dev, observed.st_ino) != identity or (
                    current.st_dev,
                    current.st_ino,
                ) != identity:
                    raise DiagnosticOutputError("zmieniła się przypięta ścieżka katalogu")

        check()
        yield descriptor, check
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _check_collision(directory: int, name: str) -> None:
    try:
        os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise DiagnosticOutputError("ścieżka raportu już istnieje; wymagany nowy raport")


def _publish_posix(root: Path, name: str, raw: bytes) -> None:
    if sys.platform not in {"linux", "darwin"}:
        raise DiagnosticOutputError("brak natywnej publikacji bez zastępowania na tym systemie")
    with _pinned_posix_directory(root) as (directory, check):
        _check_collision(directory, name)
        temporary = ".stage10-" + secrets.token_hex(16) + ".partial"
        check()
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=directory,
        )
        try:
            initial = os.fstat(descriptor)
            if (
                not stat.S_ISREG(initial.st_mode)
                or initial.st_uid != _posix_owner()
                or initial.st_nlink != 1
                or initial.st_mode & 0o077
            ):
                raise DiagnosticOutputError("nieprawidłowa tożsamość nowego pliku diagnostyki")
            _write_descriptor(descriptor, raw)
            check()
            observed = os.stat(temporary, dir_fd=directory, follow_symlinks=False)
            if (
                (observed.st_dev, observed.st_ino) != (initial.st_dev, initial.st_ino)
                or not stat.S_ISREG(observed.st_mode)
                or observed.st_nlink != 1
                or observed.st_size != len(raw)
            ):
                raise DiagnosticOutputError("zmienił się plik roboczy diagnostyki")
            # Natywny renameat2(RENAME_NOREPLACE) / renameatx_np(RENAME_EXCL).
            # Ewentualny zwycięzca wyścigu pozostaje nietknięty.
            _rename_noreplace(directory, temporary, name)
            os.fsync(directory)
            check()
        finally:
            os.close(descriptor)
            # Nigdy nie usuwaj ponownie rozwiązywanej nazwy. Po błędzie .partial
            # zostaje do osobnego przeglądu; konsument Stage 10 go nie akceptuje.


def publish_diagnostic(
    output: Path,
    raw: bytes,
    *,
    approved_directory: Path,
    protected_inputs: tuple[Path, ...] = (),
) -> None:
    root, name = _checked_destination(output, approved_directory, protected_inputs)
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_DIAGNOSTIC_BYTES:
        raise DiagnosticOutputError("nieprawidłowy rozmiar diagnostyki")
    try:
        if os.name == "nt":
            from deployment.windows_stage10_diagnostic_windows import publish_windows

            publish_windows(root, name, raw)
        elif os.name == "posix":
            _publish_posix(root, name, raw)
        else:
            raise DiagnosticOutputError("brak bezpiecznej publikacji na tym systemie")
    except (OSError, EnrollmentFileError) as exc:
        raise DiagnosticOutputError("odmowa bezpiecznego zapisu diagnostyki") from exc
