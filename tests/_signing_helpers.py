"""Pomocnicze funkcje do generowania kluczy HMAC w testach."""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["write_random_hmac_key"]


def write_random_hmac_key(path: Path, *, size: int = 48) -> bytes:
    """Zapisuje losowy klucz HMAC w zadanej ścieżce i zwraca jego wartość."""

    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = os.urandom(size)
    path.write_bytes(data)
    if os.name != "nt":
        path.chmod(0o600)
    return data
