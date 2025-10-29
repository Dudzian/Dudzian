"""Validate bundler manifest entries against local artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class ManifestEntry:
    name: str
    size_bytes: int
    sha256: str

    @classmethod
    def from_payload(cls, payload: dict) -> "ManifestEntry":
        try:
            name = payload["name"]
            size = payload["size_bytes"]
            digest = payload["sha256"]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Manifest entry brakuje pola: {exc.args[0]}") from exc

        if not isinstance(name, str) or not name:
            raise ValueError("Manifest entry 'name' musi być niepustym stringiem")
        if not isinstance(size, int) or size < 0:
            raise ValueError("Manifest entry 'size_bytes' musi być nieujemną liczbą całkowitą")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Manifest entry 'sha256' musi być 64-znakowym hashem hex")
        return cls(name=name, size_bytes=size, sha256=digest.lower())


@dataclass
class ValidationIssue:
    path: Path
    reason: str


@dataclass
class ValidationResult:
    missing: list[ValidationIssue]
    mismatched: list[ValidationIssue]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.mismatched

    def report(self) -> str:
        lines: list[str] = []
        for label, issues in (("missing", self.missing), ("mismatched", self.mismatched)):
            if not issues:
                continue
            lines.append(f"Pliki {label}:")
            for issue in issues:
                lines.append(f"  - {issue.path}: {issue.reason}")
        return "\n".join(lines)


def _read_manifest(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Manifest nie istnieje: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Manifest nie jest poprawnym JSON-em ({exc})") from exc
    if not isinstance(payload, dict):
        raise SystemExit("Niepoprawny format manifestu: oczekiwano obiektu JSON")
    return payload


def _iter_entries(section: str, data: dict) -> Iterable[ManifestEntry]:
    entries = data.get(section, [])
    if not isinstance(entries, list):
        raise SystemExit(f"Sekcja '{section}' w manifeście musi być listą")
    for raw in entries:
        if not isinstance(raw, dict):
            raise SystemExit(f"Element sekcji '{section}' musi być obiektem JSON")
        yield ManifestEntry.from_payload(raw)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_section(section: str, root: Path, entries: Iterable[ManifestEntry]) -> ValidationResult:
    missing: list[ValidationIssue] = []
    mismatched: list[ValidationIssue] = []

    for entry in entries:
        target = root / entry.name
        if not target.exists():
            missing.append(ValidationIssue(path=target, reason="plik nie istnieje"))
            continue
        if not target.is_file():
            mismatched.append(ValidationIssue(path=target, reason="nie jest plikiem"))
            continue
        size = target.stat().st_size
        if size != entry.size_bytes:
            mismatched.append(ValidationIssue(path=target, reason=f"rozmiar {size}B != {entry.size_bytes}B"))
            continue
        digest = _sha256(target)
        if digest != entry.sha256:
            mismatched.append(ValidationIssue(path=target, reason=f"hash {digest} != {entry.sha256}"))
    return ValidationResult(missing=missing, mismatched=mismatched)


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Ścieżka do manifestu desktopowego")
    parser.add_argument(
        "--pyinstaller-root",
        type=Path,
        default=Path("var/dist/pyinstaller"),
        help="Katalog z artefaktami PyInstaller",
    )
    parser.add_argument(
        "--briefcase-root",
        type=Path,
        default=Path("var/dist/briefcase"),
        help="Katalog z artefaktami Briefcase",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    manifest = _read_manifest(manifest_path)

    results: dict[str, ValidationResult] = {}
    results["pyinstaller"] = _validate_section(
        "pyinstaller",
        args.pyinstaller_root.expanduser().resolve(),
        _iter_entries("pyinstaller", manifest),
    )
    results["briefcase"] = _validate_section(
        "briefcase",
        args.briefcase_root.expanduser().resolve(),
        _iter_entries("briefcase", manifest),
    )

    failures = [result for result in results.values() if not result.ok]
    if failures:
        details = [f"Sekcja '{section}':\n{result.report()}" for section, result in results.items() if not result.ok]
        raise SystemExit("\n".join(details))

    return 0


if __name__ == "__main__":  # pragma: no cover - wejście CLI
    raise SystemExit(main())
