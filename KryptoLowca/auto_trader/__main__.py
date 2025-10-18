"""Pozwala uruchomić launcher papierowy poprzez ``python -m``."""

from __future__ import annotations

import sys

from .paper import main


def _entry() -> None:
    main(sys.argv[1:])


if __name__ == "__main__":  # pragma: no cover - uruchomienie modułu
    _entry()
