"""Read-only adapter budujący minimalny DecisionEnvelope-like view z mapowań runtime."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_CANONICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "decision_id": ("decision_id", "correlation_key"),
    "action": ("action", "intent"),
    "symbol": ("symbol",),
    "side": ("side",),
    "quantity": ("quantity",),
    "decision_source": ("decision_source", "source"),
    "effective_mode": ("effective_mode",),
    "model_version": ("model_version",),
    "inference_model": ("inference_model",),
    "inference_model_version": ("inference_model_version",),
    "confidence": ("confidence",),
    "score": ("score",),
    "rank": ("rank",),
    "opportunity_shadow_record_key": ("opportunity_shadow_record_key",),
    "opportunity_decision_timestamp": ("opportunity_decision_timestamp",),
    "performance_guard": ("performance_guard",),
    "risk_result": ("risk_result",),
    "risk_budget": ("risk_budget",),
    "blocking_reason": ("blocking_reason",),
    "blocking_reasons": ("blocking_reasons",),
    "environment_scope": ("environment_scope", "environment"),
    "portfolio_scope": ("portfolio_scope", "portfolio", "portfolio_id"),
    "provenance": ("provenance",),
}


def _pick_first(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def build_decision_envelope_view(
    metadata: Mapping[str, object], provenance: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Zwraca minimalny, read-only widok pól DecisionEnvelope z dostępnych mapowań."""
    merged: dict[str, object] = {}
    merged.update(metadata)
    if provenance:
        merged.update(provenance)

    autonomy_decision = metadata.get("opportunity_autonomy_decision")
    if isinstance(autonomy_decision, Mapping) and "effective_mode" in autonomy_decision:
        merged["effective_mode"] = autonomy_decision["effective_mode"]

    envelope: dict[str, object] = {}
    for field_name, aliases in _CANONICAL_ALIASES.items():
        value = _pick_first(merged, aliases)
        if value is not None:
            envelope[field_name] = value

    if provenance is not None and "provenance" not in envelope:
        envelope["provenance"] = dict(provenance)

    return envelope


__all__ = ["build_decision_envelope_view"]
