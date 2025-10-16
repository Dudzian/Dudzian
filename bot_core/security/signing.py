"""Pomocnicze funkcje podpisywania ładunków JSON (HMAC)."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from typing import Any, Mapping, Sequence, TypeAlias


_CANONICAL_SEPARATORS = (",", ":")

# ``Mapping`` obejmuje dokumenty JSON, ``Sequence`` pozwala podpisywać listy kroków.
JsonPayload: TypeAlias = Mapping[str, Any] | Sequence[Any]


def canonical_json_bytes(payload: JsonPayload) -> bytes:
    """Zwraca kanoniczną reprezentację JSON (UTF-8, sort_keys, brak spacji)."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=_CANONICAL_SEPARATORS,
    ).encode("utf-8")


def build_hmac_signature(
    payload: JsonPayload,
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


def verify_hmac_signature(
    payload: JsonPayload,
    signature: Mapping[str, Any] | None,
    *,
    key: bytes | None,
    algorithm: str = "HMAC-SHA256",
) -> bool:
    """Weryfikuje podpis HMAC.

    Zwraca ``True`` gdy podpis jest poprawny. Jeśli brakuje klucza albo podpisu,
    funkcja zwraca ``False``.
    """

    if not key or not signature:
        return False

    if signature.get("algorithm") != algorithm:
        return False

    expected = build_hmac_signature(payload, key=key, algorithm=algorithm, key_id=signature.get("key_id"))
    actual_value = signature.get("value")
    expected_value = expected.get("value")
    if not isinstance(actual_value, str) or not isinstance(expected_value, str):
        return False
    return hmac.compare_digest(actual_value, expected_value)


__all__ = ["canonical_json_bytes", "build_hmac_signature", "verify_hmac_signature"]
