#!/usr/bin/env python3
"""Sprawdza, czy bootstrapowy rejestr adapterów zawiera wymagane futuresy."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:  # pragma: no cover - konfiguracja ścieżki wykonywana raz
    sys.path.insert(0, str(ROOT))
if sys.path[1:2] and Path(sys.path[1]).resolve() == SCRIPT_DIR:  # pragma: no cover
    sys.path.pop(1)

_DEFAULT_REQUIRED = ("deribit_futures", "bitmex_futures")


def validate_required_adapter_factories(
    factories: Mapping[str, Any],
    *,
    required: tuple[str, ...],
) -> None:
    missing = [name for name in required if name not in factories]
    if missing:
        missing_list = ", ".join(missing)
        raise SystemExit(f"Brak wymaganych fabryk adapterów w bootstrapie: {missing_list}")


def load_registered_adapter_factories() -> Mapping[str, Any]:
    from bot_core.runtime.bootstrap import get_registered_adapter_factories

    return get_registered_adapter_factories()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require",
        dest="required",
        action="append",
        default=[],
        help="Klucz adaptera wymagany w rejestrze bootstrapowym (można podać wielokrotnie).",
    )
    args = parser.parse_args(argv)

    required = tuple(args.required) or _DEFAULT_REQUIRED
    factories = load_registered_adapter_factories()
    validate_required_adapter_factories(factories, required=required)
    print("Bootstrap adapter registry contains:", ", ".join(required))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
