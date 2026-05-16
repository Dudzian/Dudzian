from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "docs/architecture/decision_envelope_feasibility_stage8c.md"

REQUIRED_CANONICAL_FIELDS = (
    "decision_id",
    "correlation_key",
    "action",
    "intent",
    "symbol",
    "side",
    "quantity",
    "decision_source",
    "effective_mode",
    "model_version",
    "inference_model",
    "inference_model_version",
    "confidence",
    "score",
    "rank",
    "opportunity_shadow_record_key",
    "opportunity_decision_timestamp",
    "performance_guard",
    "risk_result",
    "risk_budget",
    "blocking_reason",
    "blocking_reasons",
    "environment_scope",
    "portfolio_scope",
    "provenance",
)

REQUIRED_MIGRATION_SAFETY_TERMS = (
    "Non-goals",
    "Ryzyka przyszłej migracji",
    "Reason mapping drift",
    "Replay/final-label lineage regression",
    "Performance guard contract drift",
    "CI matrix drift",
)


def test_decision_envelope_feasibility_spec_exists() -> None:
    assert SPEC_PATH.exists(), f"Brak pliku spec: {SPEC_PATH}"


def test_decision_envelope_feasibility_spec_contains_required_contract_terms() -> None:
    content = SPEC_PATH.read_text(encoding="utf-8")
    lowered = content.lower()

    missing_fields = [field for field in REQUIRED_CANONICAL_FIELDS if field.lower() not in lowered]
    assert not missing_fields, f"Brak wymaganych pól canonical envelope: {missing_fields}"

    missing_terms = [
        term for term in REQUIRED_MIGRATION_SAFETY_TERMS if term.lower() not in lowered
    ]
    assert not missing_terms, f"Brak wymaganych terminów migracji/safety: {missing_terms}"
