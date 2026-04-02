"""Pure-Python contract safety-net for decision payload normalization.

Intentionally duplicates key parser-contract scenarios from UI suite so they execute
without PySide/QML harness requirements.
"""

from ui.backend.decision_payload_normalizer import parse_runtime_decision_entry


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
