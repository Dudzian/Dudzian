"""The strict canonical JSON profile shared with Stage 9."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from deployment.canonical_json import canonical_json_bytes


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def parse_canonical(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid JSON") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError("document is not canonical JSON")
    return value


def exact(value: dict[str, Any], fields: set[str], name: str) -> None:
    if set(value) != fields:
        raise ValueError(f"{name}: missing or unknown fields")
