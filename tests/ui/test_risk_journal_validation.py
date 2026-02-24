import ui.backend.runtime_service as runtime_service_module
from ui.backend.runtime_service import RuntimeService


class _DummyAlertSink:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def emit_feed_health_event(self, **kwargs: object) -> None:  # pragma: no cover - interface shim
        self.events.append(kwargs)


def test_risk_journal_marks_incomplete_entries() -> None:
    entries = [
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "event": "missing_payload",
            "strategy": "alpha",
            "metadata": {},
            "decision": {},
        },
        {
            "timestamp": "2024-01-02T00:00:00Z",
            "event": "freeze_applied",
            "strategy": "alpha",
            "status": "freeze",
            "metadata": {
                "risk_action": "freeze",
                "risk_flags": ["drawdown_watch"],
                "stress_overrides": ["operator_ack"],
            },
        },
    ]

    metrics, timeline, diagnostics = runtime_service_module._build_risk_context(entries)

    assert metrics["blockCount"] == 0
    assert metrics["freezeCount"] == 1
    assert metrics["incompleteEntries"] == 1
    assert metrics["incompleteSamples"] == 1
    assert metrics["riskFlagCounts"] == {"drawdown_watch": 1}
    assert diagnostics["incompleteEntries"] == 1
    assert diagnostics["incomplete_entries"] == 1
    assert diagnostics["incomplete_samples"][0]["event"] == "missing_payload"
    assert diagnostics["incompleteSamples"][0]["event"] == "missing_payload"

    incomplete = next(item for item in timeline if item["event"] == "missing_payload")
    assert incomplete["isIncomplete"] is True
    assert "risk_action" in incomplete["missingFields"]
    assert "risk_flags|stress_overrides" in incomplete["missingFields"]

    complete = next(item for item in timeline if item["event"] == "freeze_applied")
    assert complete["isIncomplete"] is False
    assert complete["riskFlags"] == ["drawdown_watch"]


def test_risk_journal_emits_telemetry_warning() -> None:
    sink = _DummyAlertSink()
    runtime_service = RuntimeService(decision_loader=lambda limit: [], feed_alert_sink=sink)

    runtime_service._apply_risk_context(
        [
            {
                "timestamp": "2024-01-03T12:00:00Z",
                "event": "unvalidated",
                "metadata": {},
            }
        ]
    )

    assert sink.events
    assert sink.events[-1]["severity"] == "warning"
    assert sink.events[-1]["payload"]["incomplete_entries"] == 1

    runtime_service._apply_risk_context(
        [
            {
                "timestamp": "2024-01-04T12:00:00Z",
                "event": "validated",
                "metadata": {"risk_action": "unblock", "risk_flags": ["latency_spike"]},
            }
        ]
    )

    assert sink.events[-1]["severity"] == "info"
