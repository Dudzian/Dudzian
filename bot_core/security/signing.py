"""Pomocnicze funkcje podpisywania ładunków JSON (HMAC)."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from typing import Any, Mapping


_CANONICAL_SEPARATORS = (",", ":")


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Zwraca kanoniczną reprezentację JSON (UTF-8, sort_keys, brak spacji)."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=_CANONICAL_SEPARATORS,
    ).encode("utf-8")


def build_hmac_signature(
    payload: Mapping[str, Any],
    *,
    key: bytes,
    algorithm: str = "HMAC-SHA256",
    key_id: str | None = None,
) -> dict[str, str]:
    """Buduje podpis HMAC dla ładunku JSON."""

    digest = hmac.new(key, canonical_json_bytes(payload), hashlib.sha256).digest()
    signature = {
        "algorithm": algorithm,
        "value": base64.b64encode(digest).decode("ascii"),
    }
    if key_id:
        signature["key_id"] = str(key_id)
    return signature


__all__ = ["canonical_json_bytes", "build_hmac_signature"]
