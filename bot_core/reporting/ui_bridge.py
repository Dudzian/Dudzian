"""Bridge CLI exposing reporting artefacts to the Qt UI layer."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(slots=True)
class ReportFile:
    relative_path: str
    absolute_path: str
    size: int
    modified_at: datetime


@dataclass(slots=True)
class ReportEntry:
    identifier: str
    category: str
    summary_path: str | None
    summary: Mapping[str, object] | None
    exports: list[ReportFile]
    updated_at: datetime
    summary_error: str | None = None
    total_size: int = 0
    export_count: int = 0
    created_at: datetime | None = None


def _ensure_timezone(value: float) -> datetime:
    return datetime.fromtimestamp(value, tz=timezone.utc)


def _load_summary(path: Path) -> tuple[Mapping[str, object] | None, str | None]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, None
    except OSError as exc:
        return None, f"I/O error while reading summary: {exc}"
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return (
            None,
            f"Invalid JSON in summary: {exc.msg} (line {exc.lineno}, column {exc.colno})",
        )
    if isinstance(data, Mapping):
        return data, None
    return None, "Summary JSON must be an object"


def _scan_exports(directory: Path, base: Path) -> Iterable[ReportFile]:
    for candidate in sorted(directory.glob("*")):
        if not candidate.is_file():
            continue
        if candidate.name == "summary.json":
            continue
        stat = candidate.stat()
        yield ReportFile(
            relative_path=str(candidate.relative_to(base).as_posix()),
            absolute_path=str(candidate.resolve()),
            size=stat.st_size,
            modified_at=_ensure_timezone(stat.st_mtime),
        )


def _calculate_deletion_stats(target: Path) -> tuple[int, int, int]:
    files = 0
    directories = 0
    total_size = 0

    if target.is_symlink():
        try:
            stat = target.stat(follow_symlinks=False)
        except OSError:
            return 0, 0, 0
        return 1, 0, stat.st_size

    if target.is_dir():
        directories = 1
        for dirpath, dirnames, filenames in os.walk(target, followlinks=False):
            directories += len(dirnames)
            for filename in filenames:
                file_path = Path(dirpath) / filename
                try:
                    stat = file_path.stat(follow_symlinks=False)
                except OSError:
                    continue
                total_size += stat.st_size
                files += 1
        return files, directories, total_size

    if target.exists():
        try:
            stat = target.stat(follow_symlinks=False)
        except OSError:
            return 0, 0, 0
        return 1, 0, stat.st_size

    return 0, 0, 0


def _archive_destination_path(destination: Path, identifier_path: Path, archive_format: str) -> Path:
    if archive_format == "directory":
        return (destination / identifier_path).resolve(strict=False)

    name_candidate = identifier_path.name or identifier_path.stem or identifier_path.as_posix().replace("/", "_")
    if not name_candidate:
        name_candidate = "report"

    if archive_format == "zip":
        suffix = ".zip"
    elif archive_format == "tar":
        suffix = ".tar.gz"
    else:
        suffix = ""

    parent = destination / identifier_path.parent
    return (parent / f"{name_candidate}{suffix}").resolve(strict=False)


def _zip_write_directory(archive: zipfile.ZipFile, path: Path, added: set[str]) -> None:
    normalized = f"{path.as_posix().rstrip('/')}/"
    if normalized in added:
        return
    info = zipfile.ZipInfo(normalized)
    info.external_attr = 0o755 << 16
    archive.writestr(info, "")
    added.add(normalized)


def _write_zip_archive(source: Path, dest_path: Path, root_name: str) -> None:
    root_name = root_name or source.name or "report"
    directories: set[str] = set()
    with zipfile.ZipFile(dest_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if source.is_dir():
            _zip_write_directory(archive, Path(root_name), directories)
            for path in sorted(source.rglob("*")):
                rel = path.relative_to(source)
                arcname = Path(root_name) / rel
                if path.is_dir():
                    _zip_write_directory(archive, arcname, directories)
                else:
                    if arcname.parent != Path("."):
                        _zip_write_directory(archive, arcname.parent, directories)
                    archive.write(path, arcname=str(arcname.as_posix()))
        else:
            arcname = Path(root_name)
            if arcname.parent != Path("."):
                _zip_write_directory(archive, arcname.parent, directories)
            archive.write(source, arcname=str(arcname.as_posix()))


def _write_tar_archive(source: Path, dest_path: Path, root_name: str) -> None:
    root_name = root_name or source.name or "report"
    with tarfile.open(dest_path, "w:gz") as archive:
        arcname = Path(root_name)
        archive.add(source, arcname=str(arcname))


def _gather_reports(base: Path) -> list[ReportEntry]:
    if not base.exists():
        return []

    entries: dict[str, ReportEntry] = {}
    processed_directories: set[Path] = set()

    for summary_path in sorted(base.rglob("summary.json")):
        identifier = str(summary_path.parent.relative_to(base).as_posix())
        category = identifier.split("/", 1)[0] if "/" in identifier else identifier
        summary, summary_error = _load_summary(summary_path)
        exports = list(_scan_exports(summary_path.parent, base))
        stat = summary_path.stat()
        updated_at = _ensure_timezone(stat.st_mtime)
        total_size = sum(item.size for item in exports)
        created_at = updated_at
        if exports:
            earliest_export = min((item.modified_at for item in exports), default=updated_at)
            if earliest_export < created_at:
                created_at = earliest_export

        entries[identifier] = ReportEntry(
            identifier=identifier,
            category=category,
            summary_path=str(summary_path.resolve()),
            summary=summary,
            summary_error=summary_error,
            exports=exports,
            updated_at=updated_at,
            total_size=total_size,
            export_count=len(exports),
            created_at=created_at,
        )
        processed_directories.add(summary_path.parent)

    for archive in sorted(base.rglob("*.zip")):
        identifier = str(archive.parent.relative_to(base).as_posix())
        category = identifier.split("/", 1)[0] if "/" in identifier else identifier
        stat = archive.stat()
        updated_at = _ensure_timezone(stat.st_mtime)
        exports = list(_scan_exports(archive.parent, base))
        total_size = sum(item.size for item in exports)
        summary_entry = entries.get(identifier)
        processed_directories.add(archive.parent)
        created_at = min((item.modified_at for item in exports), default=updated_at)
        if summary_entry is None:
            entries[identifier] = ReportEntry(
                identifier=identifier,
                category=category,
                summary_path=None,
                summary=None,
                summary_error=None,
                exports=exports,
                updated_at=updated_at,
                total_size=total_size,
                export_count=len(exports),
                created_at=created_at,
            )
        else:
            summary_entry.exports = exports
            summary_entry.total_size = total_size
            summary_entry.export_count = len(exports)
            if updated_at > summary_entry.updated_at:
                summary_entry.updated_at = updated_at
            if summary_entry.created_at is None or created_at < summary_entry.created_at:
                summary_entry.created_at = created_at

    orphan_directories: set[Path] = set()
    root_level_files: list[Path] = []

    for candidate in sorted(base.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.name == "summary.json":
            continue
        if candidate.suffix == ".zip":
            continue
        parent = candidate.parent
        if parent in processed_directories:
            continue
        if parent == base:
            root_level_files.append(candidate)
        else:
            orphan_directories.add(parent)

    for directory in sorted(orphan_directories):
        exports = list(_scan_exports(directory, base))
        if not exports:
            continue
        identifier = str(directory.relative_to(base).as_posix())
        category = identifier.split("/", 1)[0] if "/" in identifier else identifier
        latest_modified = max(export.modified_at for export in exports)
        total_size = sum(item.size for item in exports)
        existing = entries.get(identifier)
        created_at = min((item.modified_at for item in exports), default=latest_modified)
        if existing is None:
            entries[identifier] = ReportEntry(
                identifier=identifier,
                category=category,
                summary_path=None,
                summary=None,
                summary_error=None,
                exports=exports,
                updated_at=latest_modified,
                total_size=total_size,
                export_count=len(exports),
                created_at=created_at,
            )
        else:
            if existing.summary_path is None:
                existing.exports = exports
                existing.total_size = total_size
                existing.export_count = len(exports)
                if latest_modified > existing.updated_at:
                    existing.updated_at = latest_modified
                if existing.created_at is None or created_at < existing.created_at:
                    existing.created_at = created_at

    for file_path in sorted(root_level_files):
        relative = file_path.relative_to(base).as_posix()
        category = relative.split("/", 1)[0] if "/" in relative else relative
        stat = file_path.stat()
        modified_at = _ensure_timezone(stat.st_mtime)
        export_entry = ReportFile(
            relative_path=relative,
            absolute_path=str(file_path.resolve()),
            size=stat.st_size,
            modified_at=modified_at,
        )
        existing = entries.get(relative)
        created_at = modified_at
        if existing is None:
            entries[relative] = ReportEntry(
                identifier=relative,
                category=category,
                summary_path=None,
                summary=None,
                summary_error=None,
                exports=[export_entry],
                updated_at=modified_at,
                total_size=export_entry.size,
                export_count=1,
                created_at=created_at,
            )
        else:
            if existing.summary_path is None:
                existing.exports = [export_entry]
                existing.total_size = export_entry.size
                existing.export_count = 1
                if modified_at > existing.updated_at:
                    existing.updated_at = modified_at
                if existing.created_at is None or created_at < existing.created_at:
                    existing.created_at = created_at

    return sorted(entries.values(), key=lambda entry: entry.updated_at, reverse=True)


def _entry_identifier_path(entry: ReportEntry) -> Path:
    identifier_path = Path(entry.identifier)
    if identifier_path.name == "exports" and identifier_path.parent != Path("."):
        identifier_path = identifier_path.parent
    return identifier_path


def _parse_iso_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid ISO date/time format: {value}"
        ) from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def _normalize_filters(
    args: argparse.Namespace,
) -> tuple[
    datetime | None,
    datetime | None,
    list[str],
    str,
    int | None,
    int | None,
    str,
    str,
    str | None,
    str,
]:
    since: datetime | None = getattr(args, "since", None)
    until: datetime | None = getattr(args, "until", None)
    categories_raw = getattr(args, "categories", None) or []
    summary_status_raw = getattr(args, "summary_status", "any")
    limit_raw = getattr(args, "limit", None)
    offset_raw = getattr(args, "offset", None)
    sort_key_raw = getattr(args, "sort_key", None)
    sort_direction_raw = getattr(args, "sort_direction", None)
    query_raw = getattr(args, "query", None)
    has_exports_raw = getattr(args, "has_exports", "any")

    if since is not None and until is not None and since > until:
        raise argparse.ArgumentTypeError("--since must be earlier than or equal to --until")

    categories: list[str] = []
    seen: set[str] = set()
    for value in categories_raw:
        if value is None:
            continue
        text_value = str(value).strip()
        if not text_value:
            continue
        if text_value in seen:
            continue
        categories.append(text_value)
        seen.add(text_value)

    summary_status = str(summary_status_raw or "any").strip().casefold()
    allowed_status = {"any", "valid", "missing", "invalid"}
    if summary_status not in allowed_status:
        raise argparse.ArgumentTypeError(
            f"Unsupported summary status filter: {summary_status_raw}"
        )

    limit: int | None = None
    if limit_raw is not None:
        limit = int(limit_raw)
        if limit <= 0:
            raise argparse.ArgumentTypeError("--limit must be greater than zero")

    offset: int | None = None
    if offset_raw is not None:
        offset = int(offset_raw)
        if offset < 0:
            raise argparse.ArgumentTypeError("--offset must be zero or positive")

    sort_key = str(sort_key_raw or "updated_at").strip().casefold()
    allowed_sort_keys = {"updated_at", "created_at", "name", "size"}
    if sort_key not in allowed_sort_keys:
        raise argparse.ArgumentTypeError(f"Unsupported sort key: {sort_key_raw}")

    sort_direction = str(sort_direction_raw or "desc").strip().casefold()
    allowed_directions = {"asc", "desc"}
    if sort_direction not in allowed_directions:
        raise argparse.ArgumentTypeError(
            f"Unsupported sort direction: {sort_direction_raw}"
        )

    query: str | None = None
    if query_raw is not None:
        text_query = str(query_raw).strip()
        if text_query:
            query = text_query

    has_exports = str(has_exports_raw or "any").strip().casefold()
    allowed_exports = {"any", "yes", "no"}
    if has_exports not in allowed_exports:
        raise argparse.ArgumentTypeError(
            f"Unsupported exports filter value: {has_exports_raw}"
        )

    return (
        since,
        until,
        categories,
        summary_status,
        limit,
        offset,
        sort_key,
        sort_direction,
        query,
        has_exports,
    )


def _apply_filters(
    entries: Iterable[ReportEntry],
    since: datetime | None,
    until: datetime | None,
    categories: Iterable[str],
    summary_status: str,
    has_exports: str,
    query: str | None,
) -> list[ReportEntry]:
    filtered_entries = list(entries)

    if since is not None or until is not None:
        temporal_filtered: list[ReportEntry] = []
        for entry in filtered_entries:
            if since is not None and entry.updated_at < since:
                continue
            if until is not None and entry.updated_at > until:
                continue
            temporal_filtered.append(entry)
        filtered_entries = temporal_filtered

    normalized_categories = [item.casefold() for item in categories]
    if normalized_categories:
        category_filtered: list[ReportEntry] = []
        for entry in filtered_entries:
            category = (entry.category or "").casefold()
            if category in normalized_categories:
                category_filtered.append(entry)
        filtered_entries = category_filtered

    if summary_status != "any":
        status_filtered: list[ReportEntry] = []
        for entry in filtered_entries:
            has_valid_summary = entry.summary_path is not None and entry.summary_error is None
            is_missing_summary = entry.summary_path is None
            is_invalid_summary = entry.summary_error is not None

            if summary_status == "valid" and has_valid_summary:
                status_filtered.append(entry)
            elif summary_status == "missing" and is_missing_summary:
                status_filtered.append(entry)
            elif summary_status == "invalid" and is_invalid_summary:
                status_filtered.append(entry)
        filtered_entries = status_filtered

    if has_exports != "any":
        exports_filtered: list[ReportEntry] = []
        for entry in filtered_entries:
            has_any_exports = bool(entry.exports)
            if has_exports == "yes" and has_any_exports:
                exports_filtered.append(entry)
            elif has_exports == "no" and not has_any_exports:
                exports_filtered.append(entry)
        filtered_entries = exports_filtered

    if query:
        normalized_query = query.casefold()
        query_filtered: list[ReportEntry] = []
        for entry in filtered_entries:
            haystack_parts = [
                entry.identifier,
                entry.category or "",
            ]
            if entry.summary and isinstance(entry.summary, Mapping):
                report_date = entry.summary.get("report_date")
                if report_date:
                    haystack_parts.append(str(report_date))
            haystack = " ".join(haystack_parts).casefold()
            if normalized_query in haystack:
                query_filtered.append(entry)
        filtered_entries = query_filtered

    return filtered_entries


def _sort_entries(
    entries: Iterable[ReportEntry], sort_key: str, sort_direction: str
) -> list[ReportEntry]:
    sorted_entries = list(entries)
    reverse = sort_direction != "asc"

    if sort_key == "created_at":
        sorted_entries.sort(
            key=lambda entry: entry.created_at or entry.updated_at,
            reverse=reverse,
        )
    elif sort_key == "name":
        sorted_entries.sort(
            key=lambda entry: (
                str(entry.summary.get("report_date", entry.identifier))
                if isinstance(entry.summary, Mapping)
                else entry.identifier
            ).casefold(),
            reverse=reverse,
        )
    elif sort_key == "size":
        sorted_entries.sort(key=lambda entry: entry.total_size, reverse=reverse)
    else:
        sorted_entries.sort(key=lambda entry: entry.updated_at, reverse=reverse)

    return sorted_entries


def _apply_pagination(
    entries: Iterable[ReportEntry], limit: int | None, offset: int | None
) -> list[ReportEntry]:
    paginated = list(entries)

    if offset:
        paginated = paginated[offset:]
    else:
        paginated = paginated[:]

    if limit:
        paginated = paginated[:limit]

    return paginated


def _aggregate_categories(entries: Iterable[ReportEntry]) -> list[Mapping[str, object]]:
    aggregates: dict[str, dict[str, object]] = {}
    for entry in entries:
        category = entry.category or ""
        bucket = aggregates.get(category)
        if bucket is None:
            label = category if category else "Bez kategorii"
            bucket = {
                "id": category,
                "label": label,
                "count": 0,
                "total_size": 0,
                "latest_updated_at": entry.updated_at,
                "earliest_updated_at": entry.created_at or entry.updated_at,
                "has_summary": False,
                "has_exports": False,
                "invalid_summary_count": 0,
                "export_count": 0,
                "missing_summary_count": 0,
            }
            aggregates[category] = bucket

        bucket["count"] += 1
        bucket["total_size"] += entry.total_size
        bucket["export_count"] += entry.export_count
        if entry.updated_at > bucket["latest_updated_at"]:
            bucket["latest_updated_at"] = entry.updated_at
        candidate_earliest = entry.created_at or entry.updated_at
        if candidate_earliest < bucket["earliest_updated_at"]:
            bucket["earliest_updated_at"] = candidate_earliest
        if entry.summary_path is not None and entry.summary_error is None:
            bucket["has_summary"] = True
        if entry.exports:
            bucket["has_exports"] = True
        if entry.summary_error:
            bucket["invalid_summary_count"] += 1
        if entry.summary_path is None:
            bucket["missing_summary_count"] += 1

    serialized: list[Mapping[str, object]] = []
    for bucket in aggregates.values():
        payload = dict(bucket)
        latest = payload["latest_updated_at"]
        if isinstance(latest, datetime):
            payload["latest_updated_at"] = latest.isoformat()
        earliest = payload["earliest_updated_at"]
        if isinstance(earliest, datetime):
            payload["earliest_updated_at"] = earliest.isoformat()
        serialized.append(payload)

    serialized.sort(key=lambda item: str(item.get("label", "")).casefold())
    return serialized


def _summarize_overview(entries: Iterable[ReportEntry]) -> Mapping[str, object]:
    entries = list(entries)
    summary: dict[str, object] = {
        "report_count": len(entries),
        "category_count": len({entry.category or "" for entry in entries}),
        "total_size": sum(entry.total_size for entry in entries),
        "export_count": sum(entry.export_count for entry in entries),
        "has_summary": any(
            entry.summary_path is not None and entry.summary_error is None for entry in entries
        ),
        "has_exports": any(entry.exports for entry in entries),
        "invalid_summary_count": sum(1 for entry in entries if entry.summary_error),
        "missing_summary_count": sum(1 for entry in entries if entry.summary_path is None),
        "latest_updated_at": None,
        "earliest_updated_at": None,
    }

    if entries:
        latest = max(entry.updated_at for entry in entries)
        earliest_candidates = [
            entry.created_at or entry.updated_at for entry in entries if entry.created_at or entry.updated_at
        ]
        earliest = min(earliest_candidates) if earliest_candidates else latest
        summary["latest_updated_at"] = latest.isoformat()
        summary["earliest_updated_at"] = earliest.isoformat()

    return summary


def _serialize(entry: ReportEntry) -> Mapping[str, object]:
    exports = [
        {
            "relative_path": item.relative_path,
            "absolute_path": item.absolute_path,
            "size": item.size,
            "modified_at": item.modified_at.isoformat(),
        }
        for item in entry.exports
    ]
    primary_export = exports[0] if exports else None
    if entry.summary is not None and isinstance(entry.summary, Mapping):
        display_name = str(entry.summary.get("report_date", entry.identifier))
    else:
        display_name = entry.identifier
    total_size = sum(item["size"] for item in exports)
    created_at = entry.created_at or entry.updated_at
    return {
        "relative_path": entry.identifier,
        "category": entry.category,
        "summary_path": entry.summary_path,
        "summary": entry.summary,
        "summary_error": entry.summary_error,
        "absolute_path": primary_export["absolute_path"] if primary_export else entry.summary_path,
        "exports": exports,
        "updated_at": entry.updated_at.isoformat(),
        "created_at": created_at.isoformat(),
        "display_name": display_name,
        "total_size": total_size,
        "export_count": len(exports),
        "has_summary": entry.summary_path is not None and entry.summary_error is None,
        "has_exports": bool(entry.exports),
    }


def cmd_overview(args: argparse.Namespace) -> int:
    base = Path(args.base_dir).expanduser() if args.base_dir else Path("var/reports")
    entries = _gather_reports(base)

    try:
        (
            since,
            until,
            categories,
            summary_status,
            limit,
            offset,
            sort_key,
            sort_direction,
            query,
            has_exports,
        ) = _normalize_filters(args)
    except argparse.ArgumentTypeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    filtered_entries = _apply_filters(
        entries,
        since,
        until,
        categories,
        summary_status,
        has_exports,
        query,
    )

    sorted_entries = _sort_entries(filtered_entries, sort_key, sort_direction)

    paginated_entries = _apply_pagination(sorted_entries, limit, offset)

    reports = [_serialize(entry) for entry in paginated_entries]
    category_stats = _aggregate_categories(filtered_entries)
    summary = _summarize_overview(filtered_entries)

    total_count = len(filtered_entries)
    returned_count = len(paginated_entries)
    normalized_offset = offset or 0
    has_more = (normalized_offset + returned_count) < total_count
    has_previous = normalized_offset > 0
    pagination = {
        "total_count": total_count,
        "returned_count": returned_count,
        "limit": limit,
        "offset": offset,
        "has_more": has_more,
        "has_previous": has_previous,
    }
    filters = {
        "since": since.isoformat() if since else None,
        "until": until.isoformat() if until else None,
        "categories": categories,
        "summary_status": summary_status,
        "limit": limit,
        "offset": offset,
        "sort_key": sort_key,
        "sort_direction": sort_direction,
        "query": query,
        "has_exports": has_exports,
    }
    payload = {
        "reports": reports,
        "base_directory": str(base),
        "categories": category_stats,
        "summary": summary,
        "filters": filters,
        "pagination": pagination,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    raw_path = (args.path or "").strip()
    if not raw_path:
        print("Ścieżka raportu do usunięcia jest wymagana", file=sys.stderr)
        return 2

    base = Path(args.base_dir).expanduser() if args.base_dir else Path("var/reports")
    base = base.resolve(strict=False)

    candidate = Path(raw_path)
    if candidate.is_absolute():
        target = candidate.resolve(strict=False)
    else:
        target = (base / candidate).resolve(strict=False)

    try:
        relative = target.relative_to(base)
    except ValueError:
        print("Ścieżka raportu musi znajdować się w katalogu bazowym", file=sys.stderr)
        return 2

    if relative == Path("."):
        print("Nie można usunąć katalogu bazowego raportów", file=sys.stderr)
        return 2

    files_to_remove = 0
    directories_to_remove = 0
    size_to_remove = 0

    payload: dict[str, object] = {
        "base_directory": str(base),
        "relative_path": relative.as_posix(),
        "absolute_path": str(target),
        "removed_files": 0,
        "removed_directories": 0,
        "removed_size": 0,
    }

    if getattr(args, "dry_run", False):
        payload["dry_run"] = True

    if not target.exists():
        payload["status"] = "not_found"
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    files_to_remove, directories_to_remove, size_to_remove = _calculate_deletion_stats(target)

    payload["removed_files"] = files_to_remove
    payload["removed_directories"] = directories_to_remove
    payload["removed_size"] = size_to_remove

    if getattr(args, "dry_run", False):
        payload["status"] = "preview"
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    try:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    except OSError as exc:
        payload["status"] = "error"
        payload["error"] = f"Nie udało się usunąć raportu: {exc}"
        print(json.dumps(payload, ensure_ascii=False))
        return 1

    payload["status"] = "deleted"
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def cmd_purge(args: argparse.Namespace) -> int:
    base = Path(args.base_dir).expanduser() if args.base_dir else Path("var/reports")
    base = base.resolve(strict=False)

    entries = _gather_reports(base)

    try:
        (
            since,
            until,
            categories,
            summary_status,
            limit,
            offset,
            sort_key,
            sort_direction,
            query,
            has_exports,
        ) = _normalize_filters(args)
    except argparse.ArgumentTypeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    filtered_entries = _apply_filters(
        entries,
        since,
        until,
        categories,
        summary_status,
        has_exports,
        query,
    )

    identifiers = {entry.identifier for entry in filtered_entries}

    def _has_parent_identifier(identifier: str) -> bool:
        candidate = Path(identifier)
        parent = candidate.parent
        while parent != Path("."):
            if parent.as_posix() in identifiers:
                return True
            parent = parent.parent
        return False

    filtered_entries = [
        entry for entry in filtered_entries if not _has_parent_identifier(entry.identifier)
    ]

    sorted_entries = _sort_entries(filtered_entries, sort_key, sort_direction)
    paginated_entries = _apply_pagination(sorted_entries, limit, offset)

    filters = {
        "since": since.isoformat() if since else None,
        "until": until.isoformat() if until else None,
        "categories": categories,
        "summary_status": summary_status,
        "limit": limit,
        "offset": offset,
        "sort_key": sort_key,
        "sort_direction": sort_direction,
        "query": query,
        "has_exports": has_exports,
    }

    dry_run = bool(getattr(args, "dry_run", False))

    payload: dict[str, object] = {
        "base_directory": str(base),
        "filters": filters,
        "matched_count": len(filtered_entries),
        "planned_count": len(paginated_entries),
        "removed_files": 0,
        "removed_directories": 0,
        "removed_size": 0,
        "targets": [],
    }

    if dry_run:
        payload["dry_run"] = True

    if not paginated_entries:
        payload["status"] = "empty"
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    total_files = 0
    total_directories = 0
    total_size = 0
    deleted_count = 0
    errors: list[str] = []
    targets_payload: list[dict[str, object]] = []

    for entry in paginated_entries:
        identifier_path = _entry_identifier_path(entry)
        identifier = identifier_path.as_posix()
        candidate = (base / identifier_path).resolve(strict=False)
        target_info: dict[str, object] = {
            "relative_path": identifier,
            "absolute_path": str(candidate),
            "removed_files": 0,
            "removed_directories": 0,
            "removed_size": 0,
        }

        try:
            relative = candidate.relative_to(base)
        except ValueError:
            message = "Ścieżka raportu wykracza poza katalog bazowy"
            target_info["status"] = "error"
            target_info["error"] = message
            errors.append(f"{identifier}: {message}")
            targets_payload.append(target_info)
            continue

        if relative == Path("."):
            message = "Pominięto katalog bazowy raportów"
            target_info["status"] = "error"
            target_info["error"] = message
            errors.append(f"{identifier}: {message}")
            targets_payload.append(target_info)
            continue

        exists = candidate.exists()
        files_to_remove, directories_to_remove, size_to_remove = _calculate_deletion_stats(candidate)

        target_info["removed_files"] = files_to_remove
        target_info["removed_directories"] = directories_to_remove
        target_info["removed_size"] = size_to_remove

        if dry_run:
            status = "preview"
        else:
            if not exists:
                status = "not_found"
            else:
                try:
                    if candidate.is_dir():
                        shutil.rmtree(candidate)
                    else:
                        candidate.unlink()
                    status = "deleted"
                except OSError as exc:
                    status = "error"
                    error_message = f"Nie udało się usunąć raportu: {exc}"
                    target_info["error"] = error_message
                    errors.append(f"{identifier}: {error_message}")

        target_info["status"] = status

        if status in {"deleted", "preview"}:
            total_files += files_to_remove
            total_directories += directories_to_remove
            total_size += size_to_remove
        if status == "deleted":
            deleted_count += 1

        targets_payload.append(target_info)

    payload["targets"] = targets_payload
    payload["removed_files"] = total_files
    payload["removed_directories"] = total_directories
    payload["removed_size"] = total_size
    payload["deleted_count"] = deleted_count

    if dry_run:
        payload["status"] = "preview"
    elif errors and deleted_count == 0:
        payload["status"] = "error"
        payload["errors"] = errors
    elif errors:
        payload["status"] = "partial_failure"
        payload["errors"] = errors
    else:
        payload["status"] = "completed"

    print(json.dumps(payload, ensure_ascii=False))
    return 0


def _resolve_archive_destination(base: Path, destination_raw: str | None) -> Path:
    if destination_raw:
        destination = Path(destination_raw).expanduser()
        if destination.is_absolute():
            return destination.resolve(strict=False)
        return (base / destination).resolve(strict=False)

    parent = base.parent if base.parent != base else base
    suggested = parent / f"{base.name}_archives"
    return suggested.resolve(strict=False)


def cmd_archive(args: argparse.Namespace) -> int:
    base = Path(args.base_dir).expanduser() if args.base_dir else Path("var/reports")
    base = base.resolve(strict=False)

    try:
        destination_raw = getattr(args, "destination", None)
    except AttributeError:
        destination_raw = None

    destination = _resolve_archive_destination(base, destination_raw)

    archive_format = getattr(args, "format", "directory") or "directory"

    try:
        destination.relative_to(base)
    except ValueError:
        pass
    else:
        print(
            "Katalog docelowy archiwum nie może znajdować się w katalogu raportów",
            file=sys.stderr,
        )
        return 2

    if destination == base:
        print("Katalog docelowy archiwum musi różnić się od katalogu raportów", file=sys.stderr)
        return 2

    entries = _gather_reports(base)

    try:
        (
            since,
            until,
            categories,
            summary_status,
            limit,
            offset,
            sort_key,
            sort_direction,
            query,
            has_exports,
        ) = _normalize_filters(args)
    except argparse.ArgumentTypeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    filtered_entries = _apply_filters(
        entries,
        since,
        until,
        categories,
        summary_status,
        has_exports,
        query,
    )

    identifiers = {entry.identifier for entry in filtered_entries}

    def _has_parent_identifier(identifier: str) -> bool:
        candidate = Path(identifier)
        parent = candidate.parent
        while parent != Path("."):
            if parent.as_posix() in identifiers:
                return True
            parent = parent.parent
        return False

    filtered_entries = [
        entry for entry in filtered_entries if not _has_parent_identifier(entry.identifier)
    ]

    sorted_entries = _sort_entries(filtered_entries, sort_key, sort_direction)
    paginated_entries = _apply_pagination(sorted_entries, limit, offset)

    filters = {
        "since": since.isoformat() if since else None,
        "until": until.isoformat() if until else None,
        "categories": categories,
        "summary_status": summary_status,
        "limit": limit,
        "offset": offset,
        "sort_key": sort_key,
        "sort_direction": sort_direction,
        "query": query,
        "has_exports": has_exports,
    }

    dry_run = bool(getattr(args, "dry_run", False))
    overwrite = bool(getattr(args, "overwrite", False))

    payload: dict[str, object] = {
        "base_directory": str(base),
        "destination_directory": str(destination),
        "format": archive_format,
        "filters": filters,
        "matched_count": len(filtered_entries),
        "planned_count": len(paginated_entries),
        "copied_files": 0,
        "copied_directories": 0,
        "copied_size": 0,
        "copied_count": 0,
        "targets": [],
    }

    if dry_run:
        payload["dry_run"] = True

    if not paginated_entries:
        payload["status"] = "empty"
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    total_files = 0
    total_directories = 0
    total_size = 0
    copied_count = 0
    errors: list[str] = []
    targets_payload: list[dict[str, object]] = []

    for entry in paginated_entries:
        identifier_path = _entry_identifier_path(entry)
        identifier = identifier_path.as_posix()
        candidate = (base / identifier_path).resolve(strict=False)

        dest_path = _archive_destination_path(destination, identifier_path, archive_format)

        target_info: dict[str, object] = {
            "relative_path": identifier,
            "absolute_path": str(candidate),
            "destination_path": str(dest_path),
            "copied_files": 0,
            "copied_directories": 0,
            "copied_size": 0,
        }

        try:
            candidate.relative_to(base)
        except ValueError:
            message = "Ścieżka raportu wykracza poza katalog bazowy"
            target_info["status"] = "error"
            target_info["error"] = message
            errors.append(f"{identifier}: {message}")
            targets_payload.append(target_info)
            continue

        if identifier_path == Path("."):
            message = "Pominięto katalog bazowy raportów"
            target_info["status"] = "error"
            target_info["error"] = message
            errors.append(f"{identifier}: {message}")
            targets_payload.append(target_info)
            continue

        if not candidate.exists():
            status = "not_found"
            target_info["status"] = status
            errors.append(f"{identifier}: Źródłowy raport nie istnieje")
            targets_payload.append(target_info)
            continue

        try:
            dest_path.relative_to(destination)
        except ValueError:
            message = "Ścieżka archiwum wykracza poza katalog docelowy"
            target_info["status"] = "error"
            target_info["error"] = message
            errors.append(f"{identifier}: {message}")
            targets_payload.append(target_info)
            continue

        if dest_path == candidate:
            message = "Katalog docelowy archiwum pokrywa się ze źródłem"
            target_info["status"] = "error"
            target_info["error"] = message
            errors.append(f"{identifier}: {message}")
            targets_payload.append(target_info)
            continue

        if dest_path.exists() and not overwrite:
            message = "Docelowy raport już istnieje"
            target_info["status"] = "skipped_existing"
            target_info["error"] = message
            errors.append(f"{identifier}: {message}")
            targets_payload.append(target_info)
            continue

        files_to_copy, directories_to_copy, size_to_copy = _calculate_deletion_stats(candidate)

        target_info["copied_files"] = files_to_copy
        target_info["copied_directories"] = directories_to_copy
        target_info["copied_size"] = size_to_copy

        if dry_run:
            status = "preview"
            target_info["status"] = status
            total_files += files_to_copy
            total_directories += directories_to_copy
            total_size += size_to_copy
            copied_count += 1
            targets_payload.append(target_info)
            continue

        try:
            dest_parent = dest_path.parent
            dest_parent.mkdir(parents=True, exist_ok=True)

            if dest_path.exists():
                if archive_format == "directory" and dest_path.is_dir():
                    shutil.rmtree(dest_path)
                else:
                    dest_path.unlink()

            archive_root_name = identifier_path.name or candidate.name or "report"

            if archive_format == "zip":
                _write_zip_archive(candidate, dest_path, archive_root_name)
            elif archive_format == "tar":
                _write_tar_archive(candidate, dest_path, archive_root_name)
            elif archive_format == "directory":
                if candidate.is_dir():
                    shutil.copytree(candidate, dest_path)
                else:
                    shutil.copy2(candidate, dest_path)
            else:
                if candidate.is_dir():
                    shutil.copytree(candidate, dest_path)
                else:
                    shutil.copy2(candidate, dest_path)
        except OSError as exc:
            status = "error"
            message = f"Nie udało się zarchiwizować raportu: {exc}"
            target_info["status"] = status
            target_info["error"] = message
            errors.append(f"{identifier}: {message}")
            targets_payload.append(target_info)
            continue

        status = "copied"
        target_info["status"] = status
        total_files += files_to_copy
        total_directories += directories_to_copy
        total_size += size_to_copy
        copied_count += 1
        targets_payload.append(target_info)

    payload["targets"] = targets_payload
    payload["copied_files"] = total_files
    payload["copied_directories"] = total_directories
    payload["copied_size"] = total_size
    payload["copied_count"] = copied_count

    if dry_run:
        payload["status"] = "preview"
    elif copied_count == 0 and errors:
        payload["status"] = "error"
        payload["errors"] = errors
    elif errors:
        payload["status"] = "partial_failure"
        payload["errors"] = errors
    else:
        payload["status"] = "completed"

    print(json.dumps(payload, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UI reporting bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    overview_parser = subparsers.add_parser("overview", help="Returns list of available reports")
    overview_parser.add_argument("--base-dir", dest="base_dir", default=None)
    overview_parser.add_argument("--since", type=_parse_iso_datetime, default=None)
    overview_parser.add_argument("--until", type=_parse_iso_datetime, default=None)
    overview_parser.add_argument("--category", dest="categories", action="append", default=None)
    overview_parser.add_argument(
        "--summary-status",
        dest="summary_status",
        choices=["any", "valid", "missing", "invalid"],
        default="any",
    )
    overview_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit liczby raportów zwracanych w odpowiedzi",
    )
    overview_parser.add_argument(
        "--offset",
        type=int,
        default=None,
        help="Pomiń określoną liczbę najnowszych raportów",
    )
    overview_parser.add_argument(
        "--sort",
        dest="sort_key",
        choices=["updated_at", "created_at", "name", "size"],
        default="updated_at",
        help="Pole sortowania raportów",
    )
    overview_parser.add_argument(
        "--sort-direction",
        dest="sort_direction",
        choices=["asc", "desc"],
        default="desc",
        help="Kierunek sortowania raportów",
    )
    overview_parser.add_argument(
        "--query",
        dest="query",
        default=None,
        help="Filtrowanie raportów po nazwie, kategorii lub dacie",
    )
    overview_parser.add_argument(
        "--has-exports",
        dest="has_exports",
        choices=["any", "yes", "no"],
        default="any",
        help="Filtruj raporty posiadające eksporty lub ich pozbawione",
    )

    delete_parser = subparsers.add_parser("delete", help="Usuń raport lub katalog eksportów")
    delete_parser.add_argument("path", help="Ścieżka (względna) raportu do usunięcia")
    delete_parser.add_argument("--base-dir", dest="base_dir", default=None)
    delete_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pokaż statystyki usuwania bez modyfikacji plików",
    )

    purge_parser = subparsers.add_parser(
        "purge",
        help="Usuń wiele raportów spełniających zadane filtry",
    )
    purge_parser.add_argument("--base-dir", dest="base_dir", default=None)
    purge_parser.add_argument("--since", type=_parse_iso_datetime, default=None)
    purge_parser.add_argument("--until", type=_parse_iso_datetime, default=None)
    purge_parser.add_argument("--category", dest="categories", action="append", default=None)
    purge_parser.add_argument(
        "--summary-status",
        dest="summary_status",
        choices=["any", "valid", "missing", "invalid"],
        default="any",
    )
    purge_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit liczby raportów usuwanych w jednym przebiegu",
    )
    purge_parser.add_argument(
        "--offset",
        type=int,
        default=None,
        help="Pomiń określoną liczbę raportów przed usuwaniem",
    )
    purge_parser.add_argument(
        "--sort",
        dest="sort_key",
        choices=["updated_at", "created_at", "name", "size"],
        default="updated_at",
        help="Pole sortowania raportów przed usunięciem",
    )
    purge_parser.add_argument(
        "--sort-direction",
        dest="sort_direction",
        choices=["asc", "desc"],
        default="desc",
        help="Kierunek sortowania raportów",
    )
    purge_parser.add_argument(
        "--query",
        dest="query",
        default=None,
        help="Tekstowe filtrowanie raportów przed usunięciem",
    )
    purge_parser.add_argument(
        "--has-exports",
        dest="has_exports",
        choices=["any", "yes", "no"],
        default="any",
        help="Filtruj raporty po obecności eksportów",
    )
    purge_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pokaż podgląd usuwania bez dotykania plików",
    )

    archive_parser = subparsers.add_parser(
        "archive",
        help="Skopiuj raporty spełniające filtry do katalogu archiwum",
    )
    archive_parser.add_argument("--base-dir", dest="base_dir", default=None)
    archive_parser.add_argument("--destination", dest="destination", default=None)
    archive_parser.add_argument("--since", type=_parse_iso_datetime, default=None)
    archive_parser.add_argument("--until", type=_parse_iso_datetime, default=None)
    archive_parser.add_argument("--category", dest="categories", action="append", default=None)
    archive_parser.add_argument(
        "--summary-status",
        dest="summary_status",
        choices=["any", "valid", "missing", "invalid"],
        default="any",
    )
    archive_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit liczby raportów archiwizowanych w jednym przebiegu",
    )
    archive_parser.add_argument(
        "--offset",
        type=int,
        default=None,
        help="Pomiń określoną liczbę raportów przed archiwizacją",
    )
    archive_parser.add_argument(
        "--sort",
        dest="sort_key",
        choices=["updated_at", "created_at", "name", "size"],
        default="updated_at",
        help="Pole sortowania raportów przed archiwizacją",
    )
    archive_parser.add_argument(
        "--sort-direction",
        dest="sort_direction",
        choices=["asc", "desc"],
        default="desc",
        help="Kierunek sortowania raportów",
    )
    archive_parser.add_argument(
        "--query",
        dest="query",
        default=None,
        help="Tekstowe filtrowanie raportów przed archiwizacją",
    )
    archive_parser.add_argument(
        "--has-exports",
        dest="has_exports",
        choices=["any", "yes", "no"],
        default="any",
        help="Filtruj raporty po obecności eksportów",
    )
    archive_parser.add_argument(
        "--format",
        dest="format",
        choices=["directory", "zip", "tar"],
        default="directory",
        help="Format archiwum tworzonych raportów",
    )
    archive_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pokaż podgląd archiwizacji bez kopiowania plików",
    )
    archive_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Nadpisz istniejące raporty w katalogu docelowym",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "overview":
        return cmd_overview(args)
    if args.command == "delete":
        return cmd_delete(args)
    if args.command == "purge":
        return cmd_purge(args)
    if args.command == "archive":
        return cmd_archive(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
