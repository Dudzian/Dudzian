from ui.backend.decision_payload_normalizer import parse_runtime_decision_entry


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
