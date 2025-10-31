#!/usr/bin/env python3
"""Narzędzie CLI do generowania paczek diagnostycznych dla wsparcia technicznego."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

from core.support import create_diagnostics_package, DiagnosticsError


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generuje paczkę diagnostyczną bota")
    parser.add_argument(
        "--base-path",
        dest="base_path",
        type=Path,
        default=Path.cwd(),
        help="Katalog bazowy projektu (domyślnie bieżący katalog)",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=Path("logs/support/diagnostics"),
        help="Katalog docelowy dla archiwów diagnostycznych",
    )
    parser.add_argument(
        "--description",
        dest="description",
        type=str,
        default="",
        help="Opis zgłoszenia serwisowego dołączony do manifestu",
    )
    parser.add_argument(
        "--extra",
        dest="extra",
        type=Path,
        nargs="*",
        default=(),
        help="Dodatkowe pliki lub katalogi do umieszczenia w paczce",
    )
    parser.add_argument(
        "--metadata",
        dest="metadata",
        type=str,
        default=None,
        help="Dodatkowe dane JSON dodane do manifestu (np. informacje o użytkowniku)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    metadata = {
        "description": args.description.strip(),
        "generated_by": "cli.generate_diagnostics",
        "generated_at": datetime.utcnow().isoformat(),
    }
    if args.metadata:
        try:
            metadata.update(json.loads(args.metadata))
        except json.JSONDecodeError as exc:
            print(f"Niepoprawny JSON w --metadata: {exc}", file=sys.stderr)
            return 2

    try:
        package = create_diagnostics_package(
            args.output_dir,
            base_path=args.base_path,
            extra=args.extra,
            metadata=metadata,
        )
    except DiagnosticsError as exc:
        print(f"Błąd: {exc}", file=sys.stderr)
        return 1

    print("Utworzono paczkę diagnostyczną:", package.archive_path)
    print("Zawartość:")
    for entry in package.included_files:
        print(" -", entry)
    return 0


if __name__ == "__main__":  # pragma: no cover - uruchamiane jako skrypt
    sys.exit(main())
