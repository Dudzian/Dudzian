from ui.backend.decision_payload_normalizer import (
    parse_runtime_decision_entry,
    parse_runtime_decision_payload,
)


def test_parse_runtime_decision_entry_normalizes_market_regime_fields() -> None:
    entry = parse_runtime_decision_entry(
        {
            "event": "order_submitted",
            "market_regime": "bull",
            "market_regime_confidence": "0.83",
            "market_regime_risk_level": "elevated",
            "decision_should_trade": "true",
        }
    ).to_payload()

    regime = entry["marketRegime"]
    assert regime["regime"] == "bull"
    assert regime["confidence"] == 0.83
    assert regime["riskLevel"] == "elevated"
    assert entry["decision"]["shouldTrade"] is True


def test_parse_runtime_decision_entry_normalizes_decision_numeric_fields() -> None:
    entry = parse_runtime_decision_entry(
        {"decision_confidence": "0.71", "decision_latency_ms": "25.5"}
    ).to_payload()

    decision = entry["decision"]
    assert decision["confidence"] == 0.71
    assert decision["latencyMs"] == 25.5


def test_parse_runtime_decision_entry_normalizes_global_numeric_fields() -> None:
    entry = parse_runtime_decision_entry({"confidence": "0.9120", "latency_ms": "17"}).to_payload()

    decision = entry["decision"]
    assert decision["confidence"] == 0.912
    assert decision["latencyMs"] == 17


def test_parse_runtime_decision_entry_prefers_decision_specific_over_global_fields() -> None:
    entry = parse_runtime_decision_entry(
        {
            "confidence": "0.9120",
            "latency_ms": "17",
            "decision_confidence": "0.71",
            "decision_latency_ms": "25.5",
        }
    ).to_payload()

    decision = entry["decision"]
    assert decision["confidence"] == 0.71
    assert decision["latencyMs"] == 25.5


def test_parse_runtime_decision_entry_prefers_decision_prefixed_fields_over_nested_aliases() -> (
    None
):
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


def test_parse_runtime_decision_entry_prefers_prefixed_should_trade_over_nested_alias() -> None:
    entry = parse_runtime_decision_entry(
        {
            "decision_should_trade": "no",
            "decision": {"should_trade": "yes", "state": "trade"},
        }
    ).to_payload()

    assert entry["decision"]["shouldTrade"] is False
    assert entry["decision"]["state"] == "trade"


def test_parse_runtime_decision_entry_metadata_flattening_prefers_top_level_extras_on_conflict() -> (
    None
):
    entry = parse_runtime_decision_entry(
        {
            "source": "grpc",
            "metadata": {"source": "jsonl", "profile": "paper"},
        }
    ).to_payload()

    metadata = entry["metadata"]
    assert metadata["source"] == "grpc"
    assert metadata["profile"] == "paper"


def test_parse_runtime_decision_entry_accepts_both_decision_object_aliases_and_first_wins() -> None:
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


def test_parse_runtime_decision_payload_is_single_decision_entry_point_for_aliases() -> None:
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


def test_parse_runtime_decision_entry_documents_prefix_heuristics_and_camelization() -> None:
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


def test_parse_runtime_decision_entry_documents_incomplete_record_fallback_shape() -> None:
    entry = parse_runtime_decision_entry({"signals": "alpha; beta", "quantity": 1.5}).to_payload()

    # Base contract fallback when source payload is incomplete.
    assert entry["event"] == ""
    assert entry["timestamp"] == ""
    assert entry["environment"] == ""
    assert entry["portfolio"] == ""
    assert entry["riskProfile"] == ""
    assert entry["quantity"] == 1.5
    assert entry["signals"] == ["alpha", "beta"]


def test_parse_runtime_decision_entry_routes_unknown_fields_to_metadata() -> None:
    entry = parse_runtime_decision_entry(
        {
            "custom_field": "value",
            "metadata": {"custom_field": "metadata-value", "meta_only": 123},
        }
    ).to_payload()

    assert entry["metadata"]["custom_field"] == "value"
    assert entry["metadata"]["meta_only"] == 123


def test_parse_runtime_decision_entry_schema_version_backwards_compatible() -> None:
    legacy_entry = parse_runtime_decision_entry({"event": "decision_made"}).to_payload()
    versioned_entry = parse_runtime_decision_entry(
        {"event": "decision_made", "schema_version": "1"}
    ).to_payload()
    blank_version_entry = parse_runtime_decision_entry(
        {"event": "decision_made", "schema_version": ""}
    ).to_payload()

    assert legacy_entry["schema_version"] == "1"
    assert versioned_entry["schema_version"] == "1"
    assert blank_version_entry["schema_version"] == "1"
    assert "schema_version" not in legacy_entry["metadata"]
    assert "schema_version" not in versioned_entry["metadata"]
    assert "schema_version" not in blank_version_entry["metadata"]
