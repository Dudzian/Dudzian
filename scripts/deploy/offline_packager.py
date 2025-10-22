"""Buduje pakiet instalacyjny do dystrybucji offline."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, Sequence

from scripts.build_desktop_installer import DEFAULT_REPORTS_DIR, build_bundle


DEFAULT_OFFLINE_OUTPUT = Path("var/dist/offline")
DEFAULT_DATASETS = (
    Path("data/trading_stub/datasets"),
    Path("data/license_samples"),
)
DEFAULT_DOCS = Path("docs/deployment")


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _copy_optional_file(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _ensure_clean_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _resolve_iterable(values: Sequence[str]) -> Iterable[Path]:
    for raw in values:
        candidate = Path(raw).expanduser()
        if candidate.exists():
            yield candidate


def _build_installer(args: argparse.Namespace, output_root: Path) -> Path:
    installer_args = argparse.Namespace(
        build_dir=str(Path(args.ui_build).expanduser().resolve()),
        output=str(output_root),
        reports=str(Path(args.reports).expanduser()),
        updater_script=args.updater_script,
        signing_key=args.signing_key,
        platform=args.platform,
    )
    archive = build_bundle(installer_args)
    return archive


def _list_relative(root: Path, bundle_root: Path) -> list[str]:
    if not root.exists():
        return []
    entries: list[str] = []
    for path in root.rglob("*"):
        if path.is_file():
            entries.append(str(path.relative_to(bundle_root)))
    return sorted(entries)


def build_offline_bundle(args: argparse.Namespace) -> Path:
    output_root = Path(args.output).expanduser().resolve()
    installer_output = output_root / "installer"
    offline_root = output_root / "bundle"

    installer_output.mkdir(parents=True, exist_ok=True)
    _ensure_clean_directory(offline_root)

    archive = _build_installer(args, installer_output)
    installer_archive = Path(archive)
    shutil.copy2(installer_archive, offline_root / installer_archive.name)

    offline_root.joinpath("config").mkdir(parents=True, exist_ok=True)
    _copy_tree(Path(args.config).expanduser(), offline_root / "config")

    datasets_root = offline_root / "datasets"
    for dataset in _resolve_iterable(args.datasets):
        destination = datasets_root / dataset.name
        _copy_tree(dataset, destination)

    docs_root = offline_root / "docs"
    if args.docs:
        docs_source = Path(args.docs).expanduser()
        if docs_source.is_file():
            _copy_optional_file(docs_source, docs_root / docs_source.name)
        else:
            _copy_tree(docs_source, docs_root)
    else:
        if DEFAULT_DOCS.exists():
            for default_doc in DEFAULT_DOCS.iterdir():
                _copy_optional_file(default_doc, docs_root / default_doc.name)

    for extra in _resolve_iterable(args.extra):
        target = offline_root / "extras" / extra.name
        if extra.is_dir():
            _copy_tree(extra, target)
        else:
            _copy_optional_file(extra, target)

    extras_root = offline_root / "extras"
    manifest = {
        "installer": installer_archive.name,
        "datasets": _list_relative(datasets_root, offline_root),
        "docs": _list_relative(docs_root, offline_root),
        "extras": _list_relative(extras_root, offline_root),
    }

    manifest_path = offline_root / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    archive_path = shutil.make_archive(str(output_root / "offline_bundle"), "gztar", root_dir=offline_root)
    return Path(archive_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build offline deployment bundle")
    parser.add_argument("--ui-build", required=True, help="Ścieżka do katalogu zbudowanej aplikacji Qt (Release)")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OFFLINE_OUTPUT),
        help="Katalog wyjściowy dla pakietu offline",
    )
    parser.add_argument(
        "--config",
        default="config",
        help="Katalog z konfiguracją dołączaną do pakietu",
    )
    parser.add_argument(
        "--reports",
        default=str(DEFAULT_REPORTS_DIR),
        help="Opcjonalny katalog raportów przekazywany do instalatora",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[str(path) for path in DEFAULT_DATASETS],
        help="Katalogi z danymi dodawane do pakietu offline",
    )
    parser.add_argument(
        "--docs",
        help="Katalog lub plik dokumentacji operatorskiej do dołączenia",
    )
    parser.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="Dodatkowe pliki lub katalogi (np. skrypty aktualizacji)",
    )
    parser.add_argument(
        "--updater-script",
        default="scripts/desktop_updater.py",
        help="Ścieżka do skryptu updatera używanego przez PyInstaller",
    )
    parser.add_argument(
        "--signing-key",
        help="Sekretny klucz HMAC do podpisania updatera",
    )
    parser.add_argument(
        "--platform",
        choices=["linux", "windows", "mac"],
        default="linux",
        help="Docelowa platforma binarki Qt",
    )

    args = parser.parse_args(argv)

    try:
        archive_path = build_offline_bundle(args)
    except Exception as exc:  # pragma: no cover - raportowanie CLI
        raise SystemExit(str(exc)) from exc

    print(f"Przygotowano pakiet offline: {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
