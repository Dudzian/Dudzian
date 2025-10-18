"""CLI bridge exposing reporting artefacts to the Qt UI layer."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ReportEntry:
    path: Path
    root: Path

    def to_dict(self) -> dict[str, Any]:
        stats = self._safe_stat(self.path)
        size = self._compute_size(self.path)
        modified = None
        if stats is not None:
            modified = datetime.fromtimestamp(stats.st_mtime, timezone.utc).isoformat()
        entry_type = "directory" if self.path.is_dir() else "file"
        relative = self._relative_path()
        result: dict[str, Any] = {
            "name": self.path.name,
            "path": str(self.path),
            "relative_path": str(relative) if relative is not None else str(self.path),
            "type": entry_type,
            "size_bytes": size,
        }
        if modified:
            result["modified"] = modified
        if entry_type == "directory":
            result["entries"] = self._count_entries(self.path)
        return result

    def _relative_path(self) -> Path | None:
        try:
            return self.path.resolve().relative_to(self.root.resolve())
        except Exception:  # pragma: no cover - ścieżki spoza magazynu
            return None

    @staticmethod
    def _safe_stat(path: Path):
        try:
            return path.stat()
        except OSError as exc:  # pragma: no cover - logujemy i kontynuujemy
            LOGGER.warning("Nie można pobrać statystyk raportu %s: %s", path, exc)
            return None

    @staticmethod
    def _compute_size(path: Path) -> int:
        if not path.exists():
            return 0
        if path.is_file():
            stats = ReportEntry._safe_stat(path)
            return 0 if stats is None else stats.st_size
        total = 0
        for child in path.rglob("*"):
            if child.is_file():
                stats = ReportEntry._safe_stat(child)
                if stats is None:
                    continue
                total += stats.st_size
        return total

    @staticmethod
    def _count_entries(path: Path) -> int:
        if not path.exists() or not path.is_dir():
            return 0
        return sum(1 for _ in path.iterdir())


def list_reports(*, root: str | None = None) -> dict[str, Any]:
    base = _resolve_root(root)
    if not base.exists():
        return {
            "status": "ok",
            "root": str(base),
            "reports": [],
        }

    entries = []
    for entry in sorted(base.iterdir(), key=lambda item: item.name):
        if entry.name.startswith("."):
            continue
        entries.append(ReportEntry(path=entry, root=base).to_dict())

    return {
        "status": "ok",
        "root": str(base),
        "reports": entries,
    }


def delete_report(report_path: str, *, root: str | None = None) -> dict[str, Any]:
    if not report_path:
        raise ValueError("Wymagana jest ścieżka raportu do usunięcia")

    base = _resolve_root(root) if root else None

    target = Path(report_path).expanduser()
    if base and not target.is_absolute():
        target = base / report_path

    target = target.expanduser()

    if base:
        root_resolved = base.expanduser().resolve()
        try:
            target_resolved = target.resolve(strict=False)
        except RuntimeError:  # pragma: no cover - według dokumentacji nie powinno wystąpić
            target_resolved = target
        if target_resolved == root_resolved:
            return {
                "status": "forbidden",
                "path": str(target),
                "root": str(root_resolved),
                "reason": "Nie można usunąć katalogu głównego raportów",
            }
        if root_resolved not in target_resolved.parents and target_resolved != root_resolved:
            return {
                "status": "forbidden",
                "path": str(target),
                "root": str(root_resolved),
                "reason": "Ścieżka spoza katalogu raportów",
            }

    if not target.exists():
        return {
            "status": "not_found",
            "path": str(target),
        }

    try:
        removed_entries = 1
        size = 0
        if target.is_dir():
            size = ReportEntry._compute_size(target)
            removed_entries = _remove_directory(target)
        else:
            stats = target.stat()
            size = stats.st_size
            target.unlink()
        return {
            "status": "ok",
            "path": str(target),
            "removed_entries": removed_entries,
            "size_bytes": size,
        }
    except OSError as exc:
        LOGGER.error("Błąd usuwania raportu %s: %s", target, exc)
        return {
            "status": "error",
            "path": str(target),
            "reason": str(exc),
        }


def _remove_directory(path: Path) -> int:
    count = 1  # katalog bazowy
    for _ in path.rglob("*"):
        count += 1
    shutil.rmtree(path)
    return count


def _resolve_root(root: str | None) -> Path:
    if root:
        return Path(root).expanduser()
    return Path("var/reports").expanduser()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UI reporting bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="Wypisuje dostępne raporty")
    list_parser.add_argument("--root", dest="root", default=None, help="Katalog z raportami")

    delete_parser = subparsers.add_parser("delete", help="Usuwa raport wraz z artefaktami")
    delete_parser.add_argument("path", help="Ścieżka raportu do usunięcia")
    delete_parser.add_argument("--root", dest="root", default=None, help="Katalog bazowy raportów")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "list":
        payload = list_reports(root=args.root)
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    if args.command == "delete":
        result = delete_report(args.path, root=args.root)
        print(json.dumps(result, ensure_ascii=False))
        if result.get("status") == "error":
            return 1
        return 0

    raise ValueError(f"Nieobsługiwane polecenie: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
