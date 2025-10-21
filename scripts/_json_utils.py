"""Pomocnicze funkcje JSON wspólne dla skryptów audytowych."""

from __future__ import annotations

import json
from typing import Any


def dump_json(payload: Any, *, pretty: bool) -> str:
    """Zwraca reprezentację JSON ze znormalizowanym formatowaniem."""

    if pretty:
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


__all__ = ["dump_json"]
