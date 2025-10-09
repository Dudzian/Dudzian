"""Wspólne funkcje audytowe do analizy metadanych plików runtime."""

from __future__ import annotations

import hashlib
import logging
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

LOGGER = logging.getLogger(__name__)


def permissions_from_mode(mode: int) -> Mapping[str, Mapping[str, bool]]:
    """Zwraca mapę praw dostępu dla właściciela, grupy i innych."""

    return {
        "owner": {
            "read": bool(mode & stat.S_IRUSR),
            "write": bool(mode & stat.S_IWUSR),
            "execute": bool(mode & stat.S_IXUSR),
        },
        "group": {
            "read": bool(mode & stat.S_IRGRP),
            "write": bool(mode & stat.S_IWGRP),
            "execute": bool(mode & stat.S_IXGRP),
        },
        "others": {
            "read": bool(mode & stat.S_IROTH),
            "write": bool(mode & stat.S_IWOTH),
            "execute": bool(mode & stat.S_IXOTH),
        },
    }


def security_flags_from_mode(mode: int) -> Mapping[str, bool]:
    """Zwraca uproszczone flagi bezpieczeństwa wynikające z maski chmod."""

    return {
        "world_readable": bool(mode & stat.S_IROTH),
        "world_writable": bool(mode & stat.S_IWOTH),
        "world_executable": bool(mode & stat.S_IXOTH),
        "group_readable": bool(mode & stat.S_IRGRP),
        "group_writable": bool(mode & stat.S_IWGRP),
        "group_executable": bool(mode & stat.S_IXGRP),
        "owner_readable": bool(mode & stat.S_IRUSR),
        "owner_writable": bool(mode & stat.S_IWUSR),
        "owner_executable": bool(mode & stat.S_IXUSR),
    }


def directory_metadata(path: Path | str) -> dict[str, object]:
    """Zwraca metadane katalogu nadrzędnego wykorzystywane przy audycie."""

    candidate = Path(path).expanduser()
    info: dict[str, object] = {"path": str(candidate)}
    try:
        info["absolute_path"] = str(candidate.resolve(strict=False))
    except Exception:  # noqa: BLE001 - fallback dla nietypowych FS
        info["absolute_path"] = str(candidate.absolute())

    exists = candidate.exists()
    info["exists"] = exists
    info["is_dir"] = candidate.is_dir()
    info["writable"] = bool(exists and os.access(candidate, os.W_OK))

    if exists:
        try:
            stat_result = candidate.stat()
        except OSError as exc:  # pragma: no cover - rzadkie przypadki
            info["stat_error"] = str(exc)
        else:
            mode = stat.S_IMODE(stat_result.st_mode)
            info["mode_octal"] = format(mode, "04o")
            info["permissions"] = permissions_from_mode(mode)
            info["security_flags"] = security_flags_from_mode(mode)
            info["owner_uid"] = getattr(stat_result, "st_uid", None)
            info["owner_gid"] = getattr(stat_result, "st_gid", None)

    return info


def file_reference_metadata(path: Path | str, *, role: str | None = None) -> Mapping[str, object]:
    """Buduje metadane audytowe wskazanego pliku z dodatkowymi ostrzeżeniami."""

    candidate = Path(path).expanduser()
    metadata: dict[str, object] = {"path": str(candidate)}
    if role is not None:
        metadata["role"] = role

    parent = candidate.parent
    parent_meta = directory_metadata(parent)
    metadata["parent"] = parent_meta
    metadata["parent_directory"] = parent_meta["path"]
    metadata["parent_absolute_path"] = parent_meta.get("absolute_path", str(parent))
    metadata["parent_exists"] = bool(parent_meta.get("exists"))
    metadata["parent_is_dir"] = bool(parent_meta.get("is_dir"))
    metadata["parent_writable"] = bool(parent_meta.get("writable"))
    if "mode_octal" in parent_meta:
        metadata["parent_mode_octal"] = parent_meta["mode_octal"]
    if "security_flags" in parent_meta:
        metadata["parent_security_flags"] = parent_meta["security_flags"]

    try:
        metadata["absolute_path"] = str(candidate.resolve(strict=False))
    except Exception:  # noqa: BLE001 - fallback dla nietypowych FS
        metadata["absolute_path"] = str(candidate.absolute())

    warnings: list[str] = []
    parent_flags = parent_meta.get("security_flags")

    if parent_flags:
        if parent_flags.get("world_writable"):
            warnings.append(
                "Katalog nadrzędny jest zapisywalny dla wszystkich użytkowników – ogranicz prawa zapisu."
            )
        if role == "tls_key" and parent_flags.get("group_writable"):
            warnings.append(
                "Katalog z materiałem TLS ma uprawnienia zapisu dla grupy – rozważ zaostrzenie chmod."
            )

    if not metadata["parent_exists"]:
        warnings.append("Katalog nadrzędny nie istnieje – utwórz go przed startem procesu.")
    elif not metadata["parent_writable"]:
        warnings.append("Brak uprawnień do zapisu w katalogu nadrzędnym – zweryfikuj konfigurację.")

    try:
        stat_result = candidate.stat()
    except OSError as exc:
        metadata["exists"] = False
        metadata["stat_error"] = str(exc)
        if role in {"tls_cert", "tls_key", "tls_client_ca"}:
            warnings.append("Plik materiału TLS jest niedostępny – konfiguracja nie będzie kompletna.")
        metadata["security_warnings"] = warnings
        return metadata

    metadata["exists"] = True
    metadata["is_file"] = candidate.is_file()
    metadata["is_symlink"] = candidate.is_symlink()
    metadata["size_bytes"] = stat_result.st_size
    metadata["modified_time"] = datetime.fromtimestamp(
        stat_result.st_mtime, timezone.utc
    ).isoformat()

    mode = stat.S_IMODE(stat_result.st_mode)
    metadata["mode_octal"] = format(mode, "04o")
    metadata["permissions"] = permissions_from_mode(mode)
    metadata["security_flags"] = security_flags_from_mode(mode)
    metadata["owner_uid"] = getattr(stat_result, "st_uid", None)
    metadata["owner_gid"] = getattr(stat_result, "st_gid", None)

    try:
        hasher = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                if not chunk:
                    break
                hasher.update(chunk)
        metadata["sha256"] = hasher.hexdigest()
    except OSError:
        LOGGER.warning(
            "Nie udało się obliczyć sumy kontrolnej SHA-256 dla pliku %s", candidate,
            exc_info=True,
        )

    if metadata["security_flags"].get("world_readable"):
        warnings.append(
            "Plik jest dostępny do odczytu dla wszystkich użytkowników – sprawdź polityki RBAC."
        )
    if metadata["security_flags"].get("world_writable"):
        warnings.append(
            "Plik jest zapisywalny dla wszystkich użytkowników – potencjalne ryzyko bezpieczeństwa."
        )

    if role == "tls_key":
        if metadata["security_flags"].get("group_readable"):
            warnings.append(
                "Klucz prywatny TLS jest czytelny dla grupy – rozważ zaostrzenie chmod."
            )
        if metadata["security_flags"].get("world_readable"):
            warnings.append(
                "Klucz prywatny TLS jest czytelny dla innych użytkowników – wysokie ryzyko bezpieczeństwa."
            )
        if metadata["security_flags"].get("group_writable") or metadata["security_flags"].get(
            "world_writable"
        ):
            warnings.append(
                "Klucz prywatny TLS można nadpisać bez uprawnień administratora – ogranicz prawa zapisu."
            )
        if metadata.get("is_symlink"):
            warnings.append(
                "Klucz prywatny TLS jest dowiązaniem symbolicznym – upewnij się, że ścieżka jest zaufana."
            )

    if role in {"jsonl", "ui_alerts_jsonl"} and not metadata["exists"]:
        warnings.append(
            "Plik logu nie istnieje – zostanie utworzony przy pierwszym zapisie, upewnij się, że katalog jest poprawny."
        )

    metadata["security_warnings"] = warnings

    return metadata


def collect_security_warnings(payload: Any) -> list[dict[str, object]]:
    """Rekurencyjnie zbiera ostrzeżenia bezpieczeństwa z dowolnego payloadu."""

    results: list[dict[str, object]] = []

    def _walk(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, Mapping):
            warnings = node.get("security_warnings")
            if warnings:
                results.append(
                    {
                        "context_path": path,
                        "warnings": list(warnings),
                        "metadata": node,
                    }
                )
            for key, value in node.items():
                if key == "security_warnings":
                    continue
                _walk(value, path + (str(key),))
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for index, item in enumerate(node):
                _walk(item, path + (str(index),))

    _walk(payload, tuple())
    return results


def log_security_warnings(
    payload: Mapping[str, object],
    *,
    fail_on_warnings: bool,
    logger: logging.Logger,
    context: str,
) -> bool:
    """Loguje wszystkie ostrzeżenia bezpieczeństwa znalezione w payloadzie.

    Funkcja jest współdzielona przez skrypty CLI, dzięki czemu komunikaty
    audytowe mają jednolity format i poziom logowania. Zwraca informację,
    czy znaleziono jakiekolwiek ostrzeżenia.
    """

    entries = collect_security_warnings(payload)
    if not entries:
        return False

    level = logging.ERROR if fail_on_warnings else logging.WARNING
    for entry in entries:
        metadata = entry.get("metadata", {})
        warnings = entry.get("warnings", [])
        context_path = " -> ".join(entry.get("context_path", ())) or "<root>"
        resource = (
            metadata.get("absolute_path")
            or metadata.get("path")
            or metadata.get("parent_absolute_path")
            or metadata.get("parent_directory")
            or context
        )
        for warning in warnings:
            logger.log(
                level,
                "%s (%s): %s",
                context_path,
                context,
                f"{warning} [lokalizacja: {resource}]",
            )
    return True


__all__ = [
    "permissions_from_mode",
    "security_flags_from_mode",
    "directory_metadata",
    "file_reference_metadata",
    "collect_security_warnings",
    "log_security_warnings",
]
