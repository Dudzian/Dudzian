"""Wejście modułu CLI uruchamianego poleceniem `python -m bot_core.cli`."""

from __future__ import annotations

import sys

from . import main


if __name__ == "__main__":  # pragma: no cover - wywołanie modułowe
    sys.exit(main())
