"""Tworzenie paczek diagnostycznych dla wsparcia technicznego."""
from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

__all__ = [
    "DiagnosticsError",
    "DiagnosticsPackage",
    "create_diagnostics_package",
]


class DiagnosticsError(RuntimeError):
    """Wyjątek zgłaszany w przypadku problemów z paczką diagnostyczną."""


@dataclass(slots=True)
class DiagnosticsPackage:
    """Informacje o wygenerowanej paczce diagnostycznej."""

    archive_path: Path
    included_files: list[str]
    metadata: Mapping[str, object]


_DEFAULT_LOG_SOURCES: tuple[Path, ...] = (Path("logs"),)
_DEFAULT_CONFIG_SOURCES: tuple[Path, ...] = (Path("config"),)
_DEFAULT_REPORT_SOURCES: tuple[Path, ...] = (Path("reports"),)

_METADATA_FILENAME = "manifest/diagnostics.json"


def _ensure_directory(path: Path) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - środowisko plików
        raise DiagnosticsError(f"Nie można utworzyć katalogu docelowego {path}: {exc}") from exc
    return path


def _collect_sources(
    base_path: Path,
    sources: Sequence[Path],
) -> list[tuple[Path, str]]:
    collected: list[tuple[Path, str]] = []
    for source in sources:
        expanded = (source if source.is_absolute() else base_path / source).resolve()
        if not expanded.exists():
            continue
        if expanded.is_file():
            try:
                arcname = expanded.relative_to(base_path).as_posix()
            except ValueError:
                arcname = expanded.name
            collected.append((expanded, arcname))
            continue
        for file_path in expanded.rglob("*"):
            if file_path.is_file():
                try:
                    arcname = file_path.relative_to(base_path).as_posix()
                except ValueError:
                    arcname = f"extra/{file_path.name}"
                collected.append((file_path, arcname))
    return collected


def _resolve_sources(
    base_path: Path,
    logs: Sequence[Path] | None,
    config: Sequence[Path] | None,
    reports: Sequence[Path] | None,
    extra: Sequence[Path] | None,
) -> list[tuple[Path, str]]:
    sources: list[tuple[Path, str]] = []
    sources.extend(_collect_sources(base_path, logs or _DEFAULT_LOG_SOURCES))
    sources.extend(_collect_sources(base_path, config or _DEFAULT_CONFIG_SOURCES))
    sources.extend(_collect_sources(base_path, reports or _DEFAULT_REPORT_SOURCES))
    if extra:
        sources.extend(_collect_sources(base_path, extra))
    return sources


def _write_metadata(
    archive: zipfile.ZipFile,
    metadata: Mapping[str, object],
    included_files: Sequence[str],
) -> None:
    manifest: MutableMapping[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": list(sorted(included_files)),
        "environment": {
            "platform": os.name,
            "cwd": str(Path.cwd()),
        },
    }
    manifest.update(metadata)
    archive.writestr(_METADATA_FILENAME, json.dumps(manifest, indent=2, ensure_ascii=False))


def create_diagnostics_package(
    destination: Path,
    *,
    base_path: Path | None = None,
    logs: Sequence[Path] | None = None,
    config: Sequence[Path] | None = None,
    reports: Sequence[Path] | None = None,
    extra: Sequence[Path] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> DiagnosticsPackage:
    """Generuje paczkę ZIP z najważniejszymi artefaktami diagnostycznymi.

    :param destination: Katalog docelowy, w którym zostanie zapisane archiwum.
    :param base_path: Katalog bazowy projektu; domyślnie bieżący katalog roboczy.
    :param logs: Dodatkowe ścieżki logów do uwzględnienia.
    :param config: Dodatkowe ścieżki konfiguracji do uwzględnienia.
    :param reports: Dodatkowe ścieżki raportów do uwzględnienia.
    :param extra: Dowolne dodatkowe pliki lub katalogi.
    :param metadata: Informacje dodatkowe do umieszczenia w manifeście.
    :raises DiagnosticsError: gdy nie uda się utworzyć paczki.
    :return: Struktura z informacją o utworzonej paczce.
    """

    base_path = (base_path or Path.cwd()).resolve()
    if not base_path.exists():
        raise DiagnosticsError(f"Ścieżka bazowa {base_path} nie istnieje")

    destination = _ensure_directory(destination.expanduser().resolve())
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = destination / f"diagnostics_{timestamp}.zip"

    sources = _resolve_sources(base_path, logs, config, reports, extra)
    included = [arcname for _, arcname in sources]

    if not included:
        raise DiagnosticsError("Brak plików do zarchiwizowania – sprawdź konfigurację ścieżek")

    try:
        with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path, arcname in sources:
                archive.write(file_path, arcname)
            _write_metadata(archive, metadata or {}, included)
    except OSError as exc:  # pragma: no cover - zależności środowiskowe
        raise DiagnosticsError(f"Nie udało się utworzyć archiwum {archive_path}: {exc}") from exc

    return DiagnosticsPackage(archive_path=archive_path, included_files=included, metadata=metadata or {})
