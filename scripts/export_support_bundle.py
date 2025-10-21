#!/usr/bin/env python3
"""Build an offline support bundle with logs, reports and telemetry artifacts."""
from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import zipfile
from typing import Iterable, Mapping

DEFAULT_INCLUDES: dict[str, str] = {
    "logs": "logs",
    "reports": "var/reports",
    "licenses": "var/licenses",
    "metrics": "var/metrics",
    "audit": "var/audit",
}

SUPPORTED_FORMATS = {"tar.gz", "zip"}


class BundleError(RuntimeError):
    """Raised when the bundle cannot be created."""


def _parse_include(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise BundleError(f"Include spec must use label=path format: {spec!r}")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label:
        raise BundleError(f"Include spec is missing label: {spec!r}")
    if not raw_path:
        raise BundleError(f"Include spec is missing path: {spec!r}")
    expanded = Path(os.path.expanduser(raw_path)).resolve()
    return label, expanded


def _parse_metadata(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise BundleError(f"Metadata spec must use key=value format: {spec!r}")
    key, value = spec.split("=", 1)
    key = key.strip()
    if not key:
        raise BundleError(f"Metadata spec is missing key: {spec!r}")
    return key, value.strip()


def _iter_directory_stats(path: Path) -> tuple[int, int]:
    total_size = 0
    file_count = 0
    if path.is_file():
        try:
            total_size = path.stat().st_size
        except OSError:
            total_size = 0
        return total_size, 1
    if not path.exists():
        return 0, 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total_size += entry.stat().st_size
            except OSError:
                continue
            file_count += 1
    return total_size, file_count


def _default_timestamp(ts: str | None) -> str:
    if ts:
        return ts
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _resolve_output_path(
    root: Path,
    *,
    output: str | None,
    output_dir: str | None,
    basename: str,
    fmt: str,
    timestamp: str,
) -> Path:
    suffix = ".tar.gz" if fmt == "tar.gz" else ".zip"
    if output:
        output_path = Path(os.path.expanduser(output))
        if output_path.is_dir():
            filename = f"{basename}-{timestamp}{suffix}"
            return (output_path / filename).resolve()
        return output_path.resolve()

    base_dir = Path(os.path.expanduser(output_dir)) if output_dir else root / "var/support"
    if not base_dir.is_absolute():
        base_dir = (root / base_dir).resolve()
    filename = f"{basename}-{timestamp}{suffix}"
    return base_dir / filename


def _prepare_includes(
    root: Path, includes: Iterable[str], disabled: Iterable[str]
) -> list[tuple[str, Path]]:
    disabled_set = {label.strip().lower() for label in disabled}
    resolved: dict[str, Path] = {}

    for label, rel_path in DEFAULT_INCLUDES.items():
        if label.lower() in disabled_set:
            continue
        path = (root / rel_path).resolve()
        resolved[label] = path

    for spec in includes:
        label, path = _parse_include(spec)
        lower = label.lower()
        if lower in disabled_set:
            continue
        resolved[label] = path

    return list(resolved.items())


def _write_tar_bundle(bundle_path: Path, entries: Mapping[str, Path], manifest: Mapping[str, object]) -> None:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_path, mode="w:gz") as tar:
        for label, path in entries.items():
            if not path.exists():
                continue
            tar.add(path, arcname=f"{label}/", recursive=True)
        data = json.dumps(manifest, indent=2).encode("utf-8")
        info = tarfile.TarInfo(name="bundle_manifest.json")
        info.size = len(data)
        tar.addfile(info, fileobj=io.BytesIO(data))


def _write_zip_bundle(bundle_path: Path, entries: Mapping[str, Path], manifest: Mapping[str, object]) -> None:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for label, path in entries.items():
            if not path.exists():
                continue
            if path.is_file():
                zf.write(path, arcname=f"{label}/{path.name}")
            else:
                for entry in path.rglob("*"):
                    if not entry.is_file():
                        continue
                    relative = entry.relative_to(path)
                    zf.write(entry, arcname=f"{label}/{relative.as_posix()}")
        zf.writestr("bundle_manifest.json", json.dumps(manifest, indent=2))


def _build_manifest(
    entries: Iterable[tuple[str, Path]],
    *,
    fmt: str,
    bundle_path: Path,
    timestamp: str,
    metadata: Mapping[str, str],
) -> dict[str, object]:
    manifest_entries: list[dict[str, object]] = []
    for label, path in entries:
        size, files = _iter_directory_stats(path)
        manifest_entries.append(
            {
                "label": label,
                "source": str(path),
                "exists": path.exists(),
                "size_bytes": size,
                "file_count": files,
            }
        )
    return {
        "created_at": timestamp,
        "format": fmt,
        "bundle_path": str(bundle_path),
        "entries": manifest_entries,
        "metadata": dict(metadata),
    }


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Eksport pakietu wsparcia UI")
    parser.add_argument("--output", help="Docelowa ścieżka archiwum (plik lub katalog)")
    parser.add_argument("--output-dir", help="Katalog docelowy pakietów wsparcia")
    parser.add_argument("--basename", default="support-bundle", help="Bazowa nazwa pliku (bez rozszerzenia)")
    parser.add_argument(
        "--format",
        choices=sorted(SUPPORTED_FORMATS),
        default="tar.gz",
        help="Format archiwum",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="label=path",
        help="Dodatkowa ścieżka dołączona do pakietu (można podać wielokrotnie)",
    )
    parser.add_argument(
        "--disable",
        action="append",
        default=[],
        metavar="label",
        help="Wyłącz domyślny zasób (np. logs, reports)",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="key=value",
        help="Dodatkowa para klucz=wartość zapisana w manifeście",
    )
    parser.add_argument("--timestamp", help="Zewnętrznie narzucony znacznik czasu ISO-8601")
    parser.add_argument("--root", default=".", help="Katalog bazowy dla ścieżek względnych")
    parser.add_argument("--dry-run", action="store_true", help="Tylko generuj manifest, bez archiwizacji")

    args = parser.parse_args(argv)

    if args.format not in SUPPORTED_FORMATS:
        raise BundleError(f"Unsupported format: {args.format}")

    root = Path(os.path.expanduser(args.root)).resolve()
    timestamp = _default_timestamp(args.timestamp)
    includes = _prepare_includes(root, args.include, args.disable)
    metadata = dict(_parse_metadata(spec) for spec in args.metadata)
    metadata.setdefault("origin", "desktop_ui")

    safe_timestamp = timestamp.replace(":", "-") if ":" in timestamp else timestamp
    bundle_path = _resolve_output_path(
        root,
        output=args.output,
        output_dir=args.output_dir,
        basename=args.basename,
        fmt=args.format,
        timestamp=safe_timestamp,
    )

    manifest = _build_manifest(includes, fmt=args.format, bundle_path=bundle_path, timestamp=timestamp, metadata=metadata)

    if args.dry_run:
        print(json.dumps({"status": "preview", **manifest}, ensure_ascii=False))
        return 0

    entries_map = {label: path for label, path in includes if path.exists()}

    if args.format == "tar.gz":
        _write_tar_bundle(bundle_path, entries_map, manifest)
    else:
        _write_zip_bundle(bundle_path, entries_map, manifest)

    try:
        bundle_size = bundle_path.stat().st_size
    except OSError:
        bundle_size = 0

    output_payload = {
        "status": "ok",
        **manifest,
        "size_bytes": bundle_size,
    }
    print(json.dumps(output_payload, ensure_ascii=False))
    return 0


def main() -> int:  # pragma: no cover - entry point wrapper
    try:
        return run()
    except BundleError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stdout)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stdout)
        return 3


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
