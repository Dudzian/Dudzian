from __future__ import annotations

from pathlib import Path

__all__ = ["native_path"]


def native_path(value: Path | str) -> str:
    """Zwraca ścieżkę w natywnym formacie platformy."""

    return str(value.expanduser()) if isinstance(value, Path) else str(Path(value).expanduser())
