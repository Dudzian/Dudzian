import logging

import pytest

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
            "event": "risk_missing_payload",
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
    assert diagnostics["incomplete_samples"][0]["event"] == "risk_missing_payload"
    assert diagnostics["incompleteSamples"][0]["event"] == "risk_missing_payload"

    incomplete = next(item for item in timeline if item["event"] == "risk_missing_payload")
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
                "event": "risk_unvalidated",
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


def test_risk_journal_ignores_non_risk_entries(caplog: pytest.LogCaptureFixture) -> None:
    service = RuntimeService(decision_loader=lambda limit: [], feed_alert_sink=_DummyAlertSink())

    with caplog.at_level(logging.WARNING, logger="ui.backend.runtime_service"):
        service._apply_risk_context(
            [
                {
                    "timestamp": "2024-01-05T10:00:00Z",
                    "event": "order_submitted",
                    "status": "submitted",
                    "metadata": {},
                    "decision": {},
                }
            ]
        )

    assert service.riskMetrics["totalEntries"] == 0
    assert service.riskMetrics["incompleteEntries"] == 0
    assert not any("Risk Journal" in message for message in caplog.messages)


@pytest.mark.parametrize(
    ("entry", "should_be_classified"),
    [
        pytest.param(
            {
                "event": "risk_update",
                "status": "ok",
                "metadata": {},
                "decision": {},
            },
            True,
            id="risk-prefix-event",
        ),
        pytest.param(
            {
                "event": "auto_risk_unfreeze",
                "status": "ok",
                "metadata": {},
                "decision": {},
            },
            True,
            id="event-with-risk-token",
        ),
        pytest.param(
            {
                "event": "position_update",
                "status": "manual_freeze",
                "metadata": {},
                "decision": {},
            },
            True,
            id="freeze-status",
        ),
        pytest.param(
            {
                "event": "position_update",
                "status": "blocked",
                "metadata": {},
                "decision": {},
            },
            True,
            id="blocked-status",
        ),
        pytest.param(
            {
                "event": "position_update",
                "status": "rejected",
                "metadata": {},
                "decision": {},
            },
            True,
            id="rejected-status",
        ),
        pytest.param(
            {
                "event": "blocked",
                "status": "ok",
                "metadata": {},
                "decision": {},
            },
            True,
            id="blocked-event",
        ),
        pytest.param(
            {
                "event": "rejected",
                "status": "ok",
                "metadata": {},
                "decision": {},
            },
            True,
            id="rejected-event",
        ),
        pytest.param(
            {
                "event": "position_update",
                "status": "ok",
                "metadata": {"risk_action": "rebalance"},
                "decision": {},
            },
            True,
            id="risk-action-field",
        ),
        pytest.param(
            {
                "event": "position_update",
                "status": "ok",
                "metadata": {"nested": {"stress_overrides": ["latency_spike"]}},
                "decision": {},
            },
            True,
            id="nested-legacy-risk-keys",
        ),
        pytest.param(
            {
                "event": "position_update",
                "status": "ok",
                "metadata": {"details": [{"stress_overrides": ["latency_spike"]}]},
                "decision": {},
            },
            True,
            id="nested-list-with-risk-keys",
        ),
        pytest.param(
            {
                "event": "order_submitted",
                "status": "submitted",
                "metadata": {},
                "decision": {},
            },
            False,
            id="plain-order-event",
        ),
        pytest.param(
            {
                "event": "order_filled",
                "status": "filled",
                "metadata": {"symbol": "BTC/USDT"},
                "decision": {"decision_state": "trade"},
            },
            False,
            id="trade-entry-without-risk-fields",
        ),
    ],
)
def test_risk_journal_entry_classification_matrix(
    entry: dict[str, object], should_be_classified: bool
) -> None:
    metrics, timeline, _diagnostics = runtime_service_module._build_risk_context([entry])

    if should_be_classified:
        assert metrics["totalEntries"] == 1
        assert len(timeline) == 1
    else:
        assert metrics["totalEntries"] == 0
        assert timeline == []
