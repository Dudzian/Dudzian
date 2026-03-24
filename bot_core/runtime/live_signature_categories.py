"""Wspólne klasyfikowanie kategorii dokumentów live readiness."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _extract_live_document_metadata(document: object) -> tuple[str, tuple[str, ...]]:
    if isinstance(document, Mapping):
        name = str(document.get("name", "")).strip().lower()
        signed_by_raw = document.get("signed_by") or ()
    else:
        name = str(getattr(document, "name", "")).strip().lower()
        signed_by_raw = getattr(document, "signed_by", None) or ()
    signed_by = tuple(str(entry).strip().lower() for entry in signed_by_raw if str(entry).strip())
    return name, signed_by


def categorize_live_document(document: object) -> tuple[str, ...]:
    categories: list[str] = []
    name_lower, signed_by = _extract_live_document_metadata(document)
    if "compliance" in signed_by or "kyc" in name_lower or "aml" in name_lower:
        categories.append("compliance")
    if "risk" in signed_by or "risk" in name_lower:
        categories.append("risk")
    if "penetration" in name_lower or "pentest" in name_lower:
        categories.append("penetration")
    return tuple(dict.fromkeys(categories))


def summarize_live_categories_from_documents(
    documents_by_name: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, bool], dict[str, bool]]:
    categories_ok = {"compliance": False, "risk": False, "penetration": False}
    categories_detected = {"compliance": False, "risk": False, "penetration": False}

    for entry in documents_by_name.values():
        if not isinstance(entry, Mapping):
            continue
        categories = categorize_live_document(entry)
        if not categories:
            continue

        for category in categories:
            categories_detected[category] = True
            categories_ok[category] = categories_ok[category] or bool(entry.get("status") == "ok")

    return categories_ok, categories_detected
