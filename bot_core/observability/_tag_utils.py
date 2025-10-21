"""Funkcje pomocnicze dotyczące tagów telemetrii UI."""

from __future__ import annotations

from typing import Mapping, Any


def extract_tag(payload: Mapping[str, Any]) -> str | None:
    tag = payload.get("tag")
    if isinstance(tag, str):
        normalized = tag.strip()
        if normalized:
            return normalized
    return None


__all__ = ["extract_tag"]
