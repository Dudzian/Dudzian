from __future__ import annotations

from bot_core.runtime.decision_envelope import (
    build_decision_envelope_view,
    normalize_decision_envelope_from_record,
)


def test_build_decision_envelope_view_maps_runtime_metadata() -> None:
    metadata = {
        "opportunity_shadow_record_key": "shadow-1",
        "opportunity_decision_timestamp": "2026-05-16T10:00:00Z",
        "opportunity_autonomy_decision": {"effective_mode": "shadow"},
        "decision_source": "opportunity_ai_shadow",
        "model_version": "mv-8d",
        "inference_model": "decision_model",
        "inference_model_version": "2026.05.16",
        "performance_guard": {"status": "ok"},
    }

    envelope = build_decision_envelope_view(metadata)

    assert envelope["opportunity_shadow_record_key"] == "shadow-1"
    assert envelope["opportunity_decision_timestamp"] == "2026-05-16T10:00:00Z"
    assert envelope["effective_mode"] == "shadow"
    assert envelope["decision_source"] == "opportunity_ai_shadow"
    assert envelope["model_version"] == "mv-8d"
    assert envelope["inference_model"] == "decision_model"
    assert envelope["inference_model_version"] == "2026.05.16"
    assert envelope["performance_guard"] == {"status": "ok"}


def test_build_decision_envelope_view_maps_provenance_aliases() -> None:
    metadata = {"decision_source": "runtime-source"}
    provenance = {
        "environment": "paper",
        "portfolio_id": "main",
        "source": "final-label",
        "model_version": "v1",
        "confidence": 0.73,
        "rank": 2,
    }

    envelope = build_decision_envelope_view(metadata, provenance)

    assert envelope["environment_scope"] == "paper"
    assert envelope["portfolio_scope"] == "main"
    assert envelope["decision_source"] == "runtime-source"
    assert envelope["model_version"] == "v1"
    assert envelope["confidence"] == 0.73
    assert envelope["rank"] == 2
    assert envelope["provenance"] == provenance


def test_build_decision_envelope_view_is_sparse_and_safe_for_empty_mapping() -> None:
    assert build_decision_envelope_view({}) == {}


def test_build_decision_envelope_view_prefers_metadata_over_provenance_on_conflicts() -> None:
    metadata = {
        "decision_source": "runtime-source",
        "model_version": "runtime-model",
        "confidence": 0.91,
        "rank": 1,
        "environment_scope": "paper",
        "portfolio_scope": "runtime-portfolio",
        "opportunity_autonomy_decision": {"effective_mode": "paper_autonomous"},
    }
    provenance = {
        "source": "final-label-source",
        "model_version": "provenance-model",
        "confidence": 0.42,
        "rank": 9,
        "environment": "live",
        "portfolio_id": "provenance-portfolio",
    }

    envelope = build_decision_envelope_view(metadata, provenance)

    assert envelope["decision_source"] == "runtime-source"
    assert envelope["model_version"] == "runtime-model"
    assert envelope["confidence"] == 0.91
    assert envelope["rank"] == 1
    assert envelope["environment_scope"] == "paper"
    assert envelope["portfolio_scope"] == "runtime-portfolio"
    assert envelope["effective_mode"] == "paper_autonomous"
    assert envelope["provenance"] == provenance


def test_build_decision_envelope_view_does_not_mutate_input_mappings() -> None:
    metadata = {
        "decision_source": "runtime-source",
        "model_version": "runtime-model",
        "opportunity_autonomy_decision": {
            "effective_mode": "paper_autonomous",
            "reason": "shadow_gate",
        },
        "performance_guard": {"status": "ok", "latency_ms": 42},
    }
    provenance = {
        "source": "provenance-source",
        "environment": "paper",
        "portfolio_id": "main",
        "confidence": 0.66,
    }
    metadata_before = {
        "decision_source": "runtime-source",
        "model_version": "runtime-model",
        "opportunity_autonomy_decision": {
            "effective_mode": "paper_autonomous",
            "reason": "shadow_gate",
        },
        "performance_guard": {"status": "ok", "latency_ms": 42},
    }
    provenance_before = {
        "source": "provenance-source",
        "environment": "paper",
        "portfolio_id": "main",
        "confidence": 0.66,
    }

    envelope = build_decision_envelope_view(metadata, provenance)

    assert metadata == metadata_before
    assert provenance == provenance_before
    assert envelope["provenance"] == provenance


def test_normalize_decision_envelope_from_record_supports_direct_mapping() -> None:
    source = {"decision_source": "runtime", "blocking_reason": "blocked"}
    record = {"decision_envelope": source}

    normalized = normalize_decision_envelope_from_record(record)

    assert normalized == {"decision_source": "runtime", "blocking_reason": "blocked"}
    assert normalized is not source


def test_normalize_decision_envelope_from_record_supports_meta_json_string() -> None:
    record = {
        "meta_decision_envelope": '{"decision_source":"runtime","effective_mode":"paper_autonomous"}'
    }

    normalized = normalize_decision_envelope_from_record(record)

    assert normalized == {"decision_source": "runtime", "effective_mode": "paper_autonomous"}


def test_normalize_decision_envelope_from_record_prefers_decision_over_meta() -> None:
    record = {
        "decision_envelope": {"decision_source": "primary"},
        "meta_decision_envelope": '{"decision_source":"secondary"}',
    }

    normalized = normalize_decision_envelope_from_record(record)

    assert normalized == {"decision_source": "primary"}


def test_normalize_decision_envelope_from_record_returns_none_for_missing_or_invalid_payload() -> (
    None
):
    assert normalize_decision_envelope_from_record({}) is None
    assert normalize_decision_envelope_from_record({"meta_decision_envelope": "not-json"}) is None
    assert normalize_decision_envelope_from_record({"meta_decision_envelope": "[]"}) is None
    assert normalize_decision_envelope_from_record({"meta_decision_envelope": '"plain"'}) is None
    assert normalize_decision_envelope_from_record({"meta_decision_envelope": "123"}) is None


def test_normalize_decision_envelope_from_record_is_read_only_for_input_record() -> None:
    nested = {"decision_source": "runtime", "provenance": {"source": "x"}}
    record = {"decision_envelope": nested}
    record_before = {
        "decision_envelope": {"decision_source": "runtime", "provenance": {"source": "x"}}
    }

    normalized = normalize_decision_envelope_from_record(record)

    assert normalized == nested
    assert record == record_before
    assert record["decision_envelope"] is nested
