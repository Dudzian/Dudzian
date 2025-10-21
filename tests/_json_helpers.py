"""Wspólne narzędzia do pracy z plikami JSON Lines w testach CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Wczytuje plik JSONL i zwraca listę obiektów."""

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


__all__ = ["read_jsonl"]

