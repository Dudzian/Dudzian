"""Pure-Python contract safety-net for decision payload normalization.

Intentionally duplicates key parser-contract scenarios from UI suite so they execute
without PySide/QML harness requirements.
"""

from ui.backend.decision_payload_normalizer import (
    _interpret_schema_version,
    parse_runtime_decision_entry,
    parse_runtime_decision_payload,
)


def test_contract_accepts_both_decision_object_aliases_and_first_wins() -> None:
    entry = parse_runtime_decision_entry(
        {
            "decision": {"state": "trade", "latency_ms": "6"},
            "Decision": {"state": "hold", "latencyMs": "9", "should_trade": "yes"},
        }
    ).to_payload()

    # _merge_decision_payload uses setdefault, so first alias encountered wins for duplicate keys.
    assert entry["decision"]["state"] == "trade"
    assert entry["decision"]["latencyMs"] == 6
    assert entry["decision"]["shouldTrade"] is True


def test_contract_parse_runtime_decision_payload_centralizes_alias_handling() -> None:
    decision = parse_runtime_decision_payload(
        {
            "confidence": "0.44",
            "latency_ms": "6",
            "Decision": {"signal": "from-Decision", "latencyMs": "9", "should_trade": "yes"},
            "decision": {"signal": "from-decision", "latency_ms": "11", "shouldTrade": False},
            "decision_confidence": "0.91",
            "decision_latency_ms": "11",
            "decision_should_trade": "0",
        }
    )

    assert decision["confidence"] == 0.91
    assert decision["latencyMs"] == 11
    assert decision["signal"] == "from-Decision"
    assert decision["shouldTrade"] is False


def test_contract_documents_prefix_heuristics_and_camelization() -> None:
    entry = parse_runtime_decision_entry(
        {
            "decision": {"source_model": "nested"},
            "decision_source_model": "prefixed",
            "decision__": "ignored",
            "ai_probability_score": "0.88",
            "ai__": "ignored",
            "market_regime_risk_score": "7",
            "market_regime_label": "trend",
            "market_regime__": "ignored",
        }
    ).to_payload()

    assert entry["decision"]["sourceModel"] == "prefixed"
    assert "" not in entry["decision"]
    assert entry["ai"] == {"probabilityScore": "0.88"}
    assert entry["marketRegime"]["riskScore"] == 7
    assert entry["marketRegime"]["label"] == "trend"


def test_contract_documents_incomplete_record_fallback_shape() -> None:
    entry = parse_runtime_decision_entry({"signals": "alpha; beta", "quantity": 1.5}).to_payload()

    # Base contract fallback when source payload is incomplete.
    assert entry["event"] == ""
    assert entry["timestamp"] == ""
    assert entry["environment"] == ""
    assert entry["portfolio"] == ""
    assert entry["riskProfile"] == ""
    assert entry["quantity"] == 1.5
    assert entry["signals"] == ["alpha", "beta"]


def test_contract_routes_unknown_fields_to_metadata() -> None:
    entry = parse_runtime_decision_entry(
        {
            "custom_field": "value",
            "metadata": {"custom_field": "metadata-value", "meta_only": 123},
        }
    ).to_payload()

    assert entry["metadata"]["custom_field"] == "value"
    assert entry["metadata"]["meta_only"] == 123



def test_contract_prefers_decision_prefixed_fields_over_nested_aliases() -> None:
    entry = parse_runtime_decision_entry(
        {
            "decision_confidence": "0.81",
            "decision_latency_ms": "19.5",
            "decision_state": "hold",
            "decision_signal": "short",
            "Decision": {
                "state": "trade",
                "signal": "long",
                "confidence": "0.22",
                "latency_ms": "7.0",
            },
        }
    ).to_payload()

    decision = entry["decision"]
    assert decision["state"] == "hold"
    assert decision["signal"] == "short"
    assert decision["confidence"] == 0.81
    assert decision["latencyMs"] == 19.5


def test_contract_prefers_prefixed_should_trade_over_nested_alias() -> None:
    entry = parse_runtime_decision_entry(
        {
            "decision_should_trade": "no",
            "decision": {"should_trade": "yes", "state": "trade"},
        }
    ).to_payload()

    assert entry["decision"]["shouldTrade"] is False
    assert entry["decision"]["state"] == "trade"


def test_contract_metadata_flattening_prefers_top_level_extras_on_conflict() -> None:
    entry = parse_runtime_decision_entry(
        {
            "source": "grpc",
            "metadata": {"source": "jsonl", "profile": "paper"},
        }
    ).to_payload()

    metadata = entry["metadata"]
    assert metadata["source"] == "grpc"
    assert metadata["profile"] == "paper"



def test_contract_precedence_mixed_decision_sources_single_payload() -> None:
    entry = parse_runtime_decision_entry(
        {
            "decision": {"state": "trade", "confidence": "0.11", "latency_ms": "5"},
            "Decision": {"state": "hold", "confidence": "0.22", "latency_ms": "6"},
            "decision_state": "monitor",
            "decision_confidence": "0.77",
            "decision_latency_ms": "17.5",
        }
    ).to_payload()

    decision = entry["decision"]
    # precedence in one payload: first nested alias seeds keys, second alias cannot override
    # due to setdefault, and decision_* prefixed fields override final values.
    assert decision["state"] == "monitor"
    assert decision["confidence"] == 0.77
    assert decision["latencyMs"] == 17.5



def test_contract_nested_decision_key_does_not_leak_to_metadata() -> None:
    entry = parse_runtime_decision_entry(
        {
            "event": "decision_made",
            "decision": {"state": "trade", "reason": "ok"},
            "source": "unit",
        }
    ).to_payload()

    assert "decision" not in entry["metadata"]
    assert entry["decision"]["state"] == "trade"
    assert entry["metadata"]["source"] == "unit"


def test_contract_non_mapping_decision_key_is_preserved_in_metadata() -> None:
    entry = parse_runtime_decision_entry(
        {
            "event": "decision_made",
            "decision": "malformed-decision-payload",
            "source": "unit",
        }
    ).to_payload()

    assert entry["metadata"]["decision"] == "malformed-decision-payload"
    assert entry["metadata"]["source"] == "unit"
    assert entry["decision"] == {}


def test_contract_nested_decision_alias_key_does_not_leak_to_metadata() -> None:
    entry = parse_runtime_decision_entry(
        {
            "event": "decision_made",
            "Decision": {"state": "hold", "reason": "risk"},
            "source": "unit",
        }
    ).to_payload()

    assert "Decision" not in entry["metadata"]
    assert entry["decision"]["state"] == "hold"
    assert entry["metadata"]["source"] == "unit"


def test_contract_non_mapping_decision_alias_key_is_preserved_in_metadata() -> None:
    entry = parse_runtime_decision_entry(
        {
            "event": "decision_made",
            "Decision": ["malformed", "decision", "payload"],
            "source": "unit",
        }
    ).to_payload()

    assert entry["metadata"]["Decision"] == ["malformed", "decision", "payload"]
    assert entry["metadata"]["source"] == "unit"
    assert entry["decision"] == {}


def test_contract_prefixed_decision_fields_do_not_leak_to_metadata() -> None:
    entry = parse_runtime_decision_entry(
        {
            "event": "decision_made",
            "decision_state": "monitor",
            "decision_should_trade": "yes",
            "source": "unit",
        }
    ).to_payload()

    assert "decision_state" not in entry["metadata"]
    assert "decision_should_trade" not in entry["metadata"]
    assert entry["decision"]["state"] == "monitor"
    assert entry["decision"]["shouldTrade"] is True
    assert entry["metadata"]["source"] == "unit"


def test_contract_mixed_decision_sources_keep_precedence_and_clean_metadata() -> None:
    entry = parse_runtime_decision_entry(
        {
            "decision": {"state": "trade", "confidence": "0.11", "latency_ms": "5"},
            "Decision": {"state": "hold", "confidence": "0.22", "latency_ms": "6"},
            "decision_state": "monitor",
            "decision_confidence": "0.77",
            "decision_latency_ms": "17.5",
            "source": "unit",
        }
    ).to_payload()

    decision = entry["decision"]
    assert decision["state"] == "monitor"
    assert decision["confidence"] == 0.77
    assert decision["latencyMs"] == 17.5
    assert "decision" not in entry["metadata"]
    assert "Decision" not in entry["metadata"]
    assert "decision_state" not in entry["metadata"]
    assert "decision_confidence" not in entry["metadata"]
    assert "decision_latency_ms" not in entry["metadata"]
    assert entry["metadata"]["source"] == "unit"


def test_contract_precedence_mixed_sources_reverse_alias_order() -> None:
    entry = parse_runtime_decision_entry(
        {
            "Decision": {
                "state": "hold",
                "confidence": "0.22",
                "latency_ms": "6",
                "signal": "from-Decision",
            },
            "decision": {
                "state": "trade",
                "confidence": "0.11",
                "latency_ms": "5",
                "signal": "from-decision",
            },
            "decision_state": "monitor",
            "decision_confidence": "0.66",
            "decision_latency_ms": "18.5",
        }
    ).to_payload()

    decision = entry["decision"]
    # reverse alias order contract: first alias (Decision) seeds keys, later alias (decision)
    # does not override duplicates, and decision_* prefixed fields still win at the end.
    assert decision["state"] == "monitor"
    assert decision["confidence"] == 0.66
    assert decision["latencyMs"] == 18.5
    assert decision["signal"] == "from-Decision"


def test_contract_schema_version_legacy_payload_defaults_and_does_not_leak_to_metadata() -> None:
    legacy_entry = parse_runtime_decision_entry({"event": "decision_made"}).to_payload()

    assert legacy_entry["schema_version"] == "1"
    assert "schema_version" not in legacy_entry["metadata"]


def test_contract_schema_version_explicit_payload_does_not_leak_to_metadata() -> None:
    versioned_entry = parse_runtime_decision_entry(
        {"event": "decision_made", "schema_version": 1}
    ).to_payload()

    assert versioned_entry["schema_version"] == "1"
    assert "schema_version" not in versioned_entry["metadata"]


def test_contract_schema_version_blank_payload_defaults_and_does_not_leak_to_metadata() -> None:
    blank_version_entry = parse_runtime_decision_entry(
        {"event": "decision_made", "schema_version": ""}
    ).to_payload()

    assert blank_version_entry["schema_version"] == "1"
    assert "schema_version" not in blank_version_entry["metadata"]


def test_contract_schema_version_metadata_only_does_not_leak_and_defaults_top_level() -> None:
    entry = parse_runtime_decision_entry(
        {"event": "decision_made", "metadata": {"schema_version": "9", "source": "jsonl"}}
    ).to_payload()

    assert entry["schema_version"] == "1"
    assert entry["metadata"]["source"] == "jsonl"
    assert "schema_version" not in entry["metadata"]


def test_contract_schema_version_top_level_conflict_with_metadata_keeps_top_level() -> None:
    entry = parse_runtime_decision_entry(
        {
            "event": "decision_made",
            "schema_version": 2,
            "metadata": {"schema_version": "9", "source": "jsonl"},
        }
    ).to_payload()

    assert entry["schema_version"] == "2"
    assert entry["metadata"]["source"] == "jsonl"
    assert "schema_version" not in entry["metadata"]


def test_contract_schema_version_camelcase_top_level_maps_to_schema_version() -> None:
    entry = parse_runtime_decision_entry({"event": "decision_made", "schemaVersion": 1}).to_payload()

    assert entry["schema_version"] == "1"
    assert "schema_version" not in entry["metadata"]
    assert "schemaVersion" not in entry["metadata"]


def test_contract_schema_version_camelcase_metadata_only_does_not_leak() -> None:
    entry = parse_runtime_decision_entry(
        {"event": "decision_made", "metadata": {"schemaVersion": "9", "source": "jsonl"}}
    ).to_payload()

    assert entry["schema_version"] == "1"
    assert entry["metadata"]["source"] == "jsonl"
    assert "schema_version" not in entry["metadata"]
    assert "schemaVersion" not in entry["metadata"]


def test_contract_schema_version_snake_case_wins_over_camel_case_top_level() -> None:
    entry = parse_runtime_decision_entry(
        {"event": "decision_made", "schema_version": "7", "schemaVersion": "9"}
    ).to_payload()

    assert entry["schema_version"] == "7"
    assert "schema_version" not in entry["metadata"]
    assert "schemaVersion" not in entry["metadata"]


def test_contract_schema_version_snake_case_wins_over_metadata_camel_case() -> None:
    entry = parse_runtime_decision_entry(
        {
            "event": "decision_made",
            "schema_version": "7",
            "metadata": {"schemaVersion": "9", "source": "jsonl"},
        }
    ).to_payload()

    assert entry["schema_version"] == "7"
    assert entry["metadata"]["source"] == "jsonl"
    assert "schema_version" not in entry["metadata"]
    assert "schemaVersion" not in entry["metadata"]


def test_contract_schema_version_blank_camel_case_defaults_and_does_not_leak() -> None:
    entry = parse_runtime_decision_entry({"event": "decision_made", "schemaVersion": ""}).to_payload()

    assert entry["schema_version"] == "1"
    assert "schema_version" not in entry["metadata"]
    assert "schemaVersion" not in entry["metadata"]


def test_contract_schema_version_blank_snake_case_wins_over_explicit_camel_case() -> None:
    entry = parse_runtime_decision_entry(
        {
            "event": "decision_made",
            "schema_version": "",
            "schemaVersion": "9",
            "metadata": {"source": "jsonl"},
        }
    ).to_payload()

    assert entry["schema_version"] == "1"
    assert entry["metadata"]["source"] == "jsonl"
    assert "schema_version" not in entry["metadata"]
    assert "schemaVersion" not in entry["metadata"]


def test_contract_interpret_schema_version_precedence_and_fallback_matrix() -> None:
    assert _interpret_schema_version({}) == "1"
    assert _interpret_schema_version({"schema_version": "7"}) == "7"
    assert _interpret_schema_version({"schemaVersion": "9"}) == "9"
    assert _interpret_schema_version({"schema_version": "", "schemaVersion": "9"}) == "1"
    assert _interpret_schema_version({"schema_version": "7", "schemaVersion": "9"}) == "7"
