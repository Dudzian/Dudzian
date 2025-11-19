from ui.backend import runtime_service as runtime_service_module
from ui.backend.runtime_service import RuntimeService


class _StubSink:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def emit_feed_health_event(
        self,
        *,
        severity: str,
        title: str,
        body: str,
        context: dict[str, object] | None = None,
        payload: dict[str, object] | None = None,
    ) -> None:
        self.events.append(
            {
                "severity": severity,
                "title": title,
                "body": body,
                "context": context or {},
                "payload": payload or {},
            }
        )


def test_normalize_ai_governor_record_adds_signals_and_decision_state() -> None:
    record = {
        "mode": "hedge",
        "timestamp": "2024-01-01T00:00:00Z",
        "reason": "vol spike",
        "confidence": 0.73,
        "decision": {
            "state": "executed",
            "signal": "volatility_breakout",
            "shouldTrade": True,
            "signals": [
                {"name": "atr", "value": 1.2, "weight": 0.6, "source": "risk"},
                {"name": "volume", "value": 2.3, "weight": 0.4},
            ],
        },
        "telemetry": {"cycleLatency": {"p95_ms": 42.0}},
    }

    normalized = runtime_service_module._normalize_ai_governor_record(record)
    assert normalized is not None
    assert normalized["decision"]["state"] == "executed"
    assert normalized["decision"]["shouldTrade"] is True
    assert normalized["signals"][0]["name"] == "atr"
    assert normalized["signals"][0]["weight"] == 0.6
    assert normalized["telemetry"]["cycleLatency"]["p95Ms"] == 42.0


def test_ai_schema_violation_emits_alert_and_skips_history(monkeypatch) -> None:
    sink = _StubSink()
    service = RuntimeService(
        decision_loader=lambda limit: [],
        feed_alert_sink=sink,
        ai_governor_loader=lambda: {},
    )
    service._set_ai_governor_snapshot({})

    invalid_record = {"mode": "hedge", "timestamp": 1234, "telemetry": {"cycleLatency": 5}}

    service._apply_ai_governor_records([invalid_record])

    assert service.aiGovernorSnapshot.get("history") == []
    assert any("schema" in event["body"] for event in sink.events)
