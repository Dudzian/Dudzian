"""Eksport domyślnych alokacji PortfolioGovernora do pliku YAML/JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.portfolio import (  # noqa: E402
    PortfolioAllocationExportError,
    export_allocations_from_core_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Eksportuje plik allocations_<governor>.yaml/json na podstawie core.yaml.",
    )
    parser.add_argument(
        "--core-config",
        default="config/core.yaml",
        help="Ścieżka do pliku core.yaml (domyślnie config/core.yaml)",
    )
    parser.add_argument(
        "--governor",
        required=True,
        help="Nazwa PortfolioGovernora z sekcji portfolio_governors w core.yaml",
    )
    parser.add_argument(
        "--environment",
        help="Opcjonalna nazwa środowiska do walidacji (environments.* w core.yaml)",
    )
    parser.add_argument(
        "--output",
        help="Ścieżka wynikowa (domyślnie var/audit/portfolio/allocations_<governor>.yaml)",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        output_path = Path(args.output).expanduser() if args.output else None
        document = export_allocations_from_core_config(
            Path(args.core_config).expanduser(),
            args.governor,
            output_path=output_path,
            environment=args.environment,
        )
        print(
            "Zapisano alokacje PortfolioGovernora",
            document.governor,
            "do",
            document.path,
        )
        print("Symbole:", ", ".join(document.allocations.keys()))
        return 0
    except PortfolioAllocationExportError as exc:
        print(f"Błąd eksportu alokacji: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())

