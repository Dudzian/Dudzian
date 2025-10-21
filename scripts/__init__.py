"""Pakiet CLI utrzymujący skrypty pomocnicze dla nowej architektury bota."""

from __future__ import annotations

from pathlib import Path

SCRIPT_ROOT: Path = Path(__file__).resolve().parent
__all__ = ["SCRIPT_ROOT"]
