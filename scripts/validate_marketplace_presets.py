#!/usr/bin/env python3
"""Walidator presetów Marketplace."""

import argparse
import json
import sys
from pathlib import Path

REQUIRED_RELEASE_FIELDS = ("review_status", "approved_at", "reviewers")


def _validate_file(path: Path) -> list[str]:
    problems: list[str] = []
    try:
        data = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        return [f"{path.name}: nie udało się wczytać JSON ({exc})"]

    catalog = data.get("catalog")
    if not isinstance(catalog, dict):
        return [f"{path.name}: brak sekcji catalog"]

    release = catalog.get("release")
    if not isinstance(release, dict):
        problems.append(f"{path.name}: brak sekcji catalog.release")
    else:
        for field in REQUIRED_RELEASE_FIELDS:
            value = release.get(field)
            if not value:
                problems.append(f"{path.name}: pole release.{field} jest puste")

    compatibility = catalog.get("exchange_compatibility")
    if not compatibility:
        problems.append(f"{path.name}: brak sekcji exchange_compatibility")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Sprawdza kompletność presetów Marketplace")
    parser.add_argument(
        "--presets",
        type=Path,
        default=Path("config/marketplace/presets"),
        help="Katalog ze specyfikacjami presetów",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=15,
        help="Minimalna liczba presetów wymagana do publikacji",
    )
    args = parser.parse_args()

    files = sorted(args.presets.rglob("*.json"))
    if len(files) < args.min_count:
        print(
            f"ERROR: Znaleziono {len(files)} presetów w {args.presets}, wymagane {args.min_count}",
            file=sys.stderr,
        )
        return 2

    failures: list[str] = []
    for path in files:
        failures.extend(_validate_file(path))

    if failures:
        print("ERROR: Niektóre presety są niekompletne:", file=sys.stderr)
        for problem in failures:
            print(f" - {problem}", file=sys.stderr)
        return 3

    print(f"OK: {len(files)} presetów spełnia wymagania katalogu.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
