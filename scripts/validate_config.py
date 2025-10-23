"""Walidacja spójności konfiguracji CoreConfig."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bot_core.config.loader import load_core_config
from bot_core.config.validation import validate_core_config


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Waliduje config/core.yaml i zależności środowisk.")
    parser.add_argument(
        "--config",
        default="config/core.yaml",
        help="Ścieżka do pliku konfiguracyjnego CoreConfig",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Zwróć wynik w formacie JSON (łatwiejszy do użycia w CI)",
    )
    parser.add_argument(
        "-p",
        "--profile",
        dest="profile",
        default=None,
        help=(
            "Opcjonalny profil środowiskowy (np. demo/paper/live) ograniczający zakres kontroli."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_core_config(args.config)
    result = validate_core_config(config, profile=args.profile)

    if args.json:
        payload = {
            "valid": result.is_valid(),
            "errors": result.errors,
            "warnings": result.warnings,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if result.errors:
            print("✖ Błędy konfiguracji:", file=sys.stderr)
            for entry in result.errors:
                print(f"  - {entry}", file=sys.stderr)
        else:
            print("✓ Konfiguracja jest spójna")

        if result.warnings:
            print("⚠ Ostrzeżenia:")
            for entry in result.warnings:
                print(f"  - {entry}")

    return 0 if result.is_valid() else 1


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    sys.exit(main())
