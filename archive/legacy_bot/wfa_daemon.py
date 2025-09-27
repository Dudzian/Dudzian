# -*- coding: utf-8 -*-
"""Alias uruchamiający głównego daemona WFO z pakietu KryptoLowca."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    _current_file = Path(__file__).resolve()
    for _parent in _current_file.parents:
        candidate = _parent / "KryptoLowca" / "__init__.py"
        if candidate.exists():
            sys.path.insert(0, str(_parent))
            __package__ = "KryptoLowca"
            break
    else:  # pragma: no cover
        raise ModuleNotFoundError(
            "Nie można zlokalizować pakietu 'KryptoLowca'. Uruchom daemon z katalogu projektu lub"
            " zainstaluj pakiet w środowisku (pip install -e .)."
        )

from KryptoLowca.wfa_daemon import main as _main


def main() -> None:
    _main()


if __name__ == "__main__":  # pragma: no cover
    main()
