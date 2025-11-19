#!/usr/bin/env python3
"""Synchronizes marketplace catalog assets with installer bundles."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARKETPLACE_DIR = REPO_ROOT / "config" / "marketplace"
DEFAULT_INSTALLER_ROOT = REPO_ROOT / "deploy" / "packaging" / "samples"
_QA_REVIEW_TOKENS = ("qa", "quality")


class ReleaseBundleError(RuntimeError):
    """Domain error raised when bundle validation fails."""


def _load_catalog(path: Path) -> Mapping[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ReleaseBundleError("catalog.json musi zawierać obiekt JSON")
    return document


def _iter_packages(catalog: Mapping[str, object]) -> list[Mapping[str, object]]:
    packages = catalog.get("packages")
    if not isinstance(packages, list):
        raise ReleaseBundleError("catalog.json musi zawierać tablicę 'packages'")
    normalized: list[Mapping[str, object]] = []
    for entry in packages:
        if isinstance(entry, Mapping):
            normalized.append(entry)
    return normalized


def _is_qa_reviewer(entry: Mapping[str, object]) -> bool:
    for field in (entry.get("role"), entry.get("team"), entry.get("name")):
        if not isinstance(field, str):
            continue
        normalized = field.strip().lower()
        if not normalized:
            continue
        if any(token in normalized for token in _QA_REVIEW_TOKENS):
            return True
    return False


def packages_with_qa_reviews(packages: Iterable[Mapping[str, object]]) -> list[str]:
    """Return identifiers of packages approved by QA reviewers."""

    approved: list[str] = []
    for package in packages:
        package_id = str(package.get("package_id") or "").strip() or "<unknown>"
        release = package.get("release")
        if not isinstance(release, Mapping):
            continue
        status = str(release.get("review_status") or "").strip().lower()
        if status not in {"approved", "qa_approved"}:
            continue
        reviewers = release.get("reviewers") or []
        if not isinstance(reviewers, Sequence):
            continue
        if any(isinstance(reviewer, Mapping) and _is_qa_reviewer(reviewer) for reviewer in reviewers):
            approved.append(package_id)
    return approved


def ensure_minimum_qa_reviews(catalog: Mapping[str, object], *, minimum: int) -> list[str]:
    packages = _iter_packages(catalog)
    approved = packages_with_qa_reviews(packages)
    if len(approved) < minimum:
        raise ReleaseBundleError(
            "Katalog zawiera tylko {count} strategii z recenzjami QA (wymagane {minimum}).".format(
                count=len(approved),
                minimum=minimum,
            )
        )
    return approved


def _signature_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sig")


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_catalog_assets(
    *,
    catalog_path: Path,
    markdown_path: Path,
    packages_dir: Path,
    installer_root: Path,
) -> list[Path]:
    if not packages_dir.is_dir():
        raise ReleaseBundleError(f"Brak katalogu z paczkami marketplace: {packages_dir}")
    installer_config = installer_root / "config" / "marketplace"
    installer_config.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for source in (catalog_path, _signature_path(catalog_path), markdown_path, _signature_path(markdown_path)):
        if not source.exists():
            raise ReleaseBundleError(f"Brak wymaganego pliku katalogu: {source}")
        target = installer_config / source.name
        _copy_file(source, target)
        outputs.append(target)

    target_packages = installer_config / "packages"
    if target_packages.exists():
        shutil.rmtree(target_packages)
    shutil.copytree(packages_dir, target_packages)
    outputs.append(target_packages)
    return outputs


def ensure_git_clean(paths: Sequence[Path]) -> None:
    if not paths:
        return
    cmd = ["git", "diff", "--quiet", "--"] + [str(path) for path in paths]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise ReleaseBundleError(
            "Zmodyfikowany katalog Marketplace. Uruchom scripts/build_marketplace_catalog.py i zatwierdź zmiany."
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_MARKETPLACE_DIR / "catalog.json",
        help="Ścieżka do pliku catalog.json.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=DEFAULT_MARKETPLACE_DIR / "catalog.md",
        help="Ścieżka do pliku catalog.md.",
    )
    parser.add_argument(
        "--packages",
        type=Path,
        default=DEFAULT_MARKETPLACE_DIR / "packages",
        help="Katalog z podpisanymi paczkami marketplace.",
    )
    parser.add_argument(
        "--installer-root",
        action="append",
        type=Path,
        default=[DEFAULT_INSTALLER_ROOT],
        help="Katalog bazowy paczki instalatora (z podkatalogiem config/).",
    )
    parser.add_argument(
        "--minimum-qa-strategies",
        type=int,
        default=15,
        help="Minimalna liczba strategii zatwierdzonych przez QA.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Wykonaj wyłącznie weryfikację bez kopiowania plików do paczek.",
    )
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="Zakończ błędem jeśli catalog.md lub catalog.md.sig różnią się od wersji w repozytorium.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    catalog_path = args.catalog.expanduser().resolve()
    markdown_path = args.markdown.expanduser().resolve()
    packages_dir = args.packages.expanduser().resolve()

    catalog = _load_catalog(catalog_path)
    ensure_minimum_qa_reviews(catalog, minimum=args.minimum_qa_strategies)
    if args.require_clean:
        ensure_git_clean([markdown_path, _signature_path(markdown_path)])

    if not args.check_only:
        destinations = [Path(entry).expanduser().resolve() for entry in args.installer_root]
        for installer_root in destinations:
            copy_catalog_assets(
                catalog_path=catalog_path,
                markdown_path=markdown_path,
                packages_dir=packages_dir,
                installer_root=installer_root,
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
