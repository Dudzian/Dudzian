from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from bot_core.api import server as api_server
from bot_core.generated import trading_pb2
from google.protobuf import timestamp_pb2


class _DummyJournal:
    def __init__(self, records):
        self._records = list(records)

    def export(self):
        return list(self._records)

    def append(self, record):
        self._records.append(record)

    def replace(self, records):
        self._records = list(records)


class _DummyRuntimeContext:
    def __init__(self, journal, auto_trader=None):
        if auto_trader is None:
            auto_trader = SimpleNamespace(_decision_journal=journal)
        elif getattr(auto_trader, "_decision_journal", None) is None:
            auto_trader._decision_journal = journal
        self.auto_trader = auto_trader
        self.authorized = False

    def authorize(self, rpc_context):  # pragma: no cover - prosta flaga pomocnicza
        self.authorized = True


class _DummyRpcContext:
    def __init__(self):
        self._active = True

    def is_active(self):
        return self._active

    def invocation_metadata(self):
        return ()

    def abort(self, code, details):  # pragma: no cover - diagnostyka testowa
        raise RuntimeError(f"RPC aborted: {code}: {details}")

    def cancel(self):
        self._active = False


@pytest.fixture(autouse=True)
def _ensure_no_sleep(monkeypatch):
    monkeypatch.setattr(
        api_server._RuntimeServicer,
        "_sleep_with_context",
        staticmethod(lambda *_args, **_kwargs: None),
    )


def _record(event: str, timestamp: str = "2024-01-01T00:00:00+00:00"):
    return {
        "event": event,
        "timestamp": timestamp,
        "environment": "test",
        "portfolio": "alpha",
    }


def _ts(value: str) -> timestamp_pb2.Timestamp:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    ts = timestamp_pb2.Timestamp()
    ts.FromDatetime(dt)
    return ts


def test_stream_resends_snapshot_after_truncation():
    journal = _DummyJournal([
        _record("first"),
        _record("second", "2024-01-01T00:01:00+00:00"),
    ])
    context = _DummyRuntimeContext(journal)
    servicer = api_server._RuntimeServicer(context)
    request = trading_pb2.StreamDecisionsRequest(limit=1, poll_interval_seconds=0.01)
    rpc_context = _DummyRpcContext()

    stream = servicer.StreamDecisions(request, rpc_context)

    first = next(stream)
    assert first.HasField("snapshot")
    assert [entry.fields["event"] for entry in first.snapshot.records] == ["second"]
    assert context.authorized is True
    assert dict(first.cycle_metrics.values) == {}

    journal.append(_record("third", "2024-01-01T00:02:00+00:00"))
    second = next(stream)
    assert second.HasField("increment")
    assert second.increment.record.fields["event"] == "third"
    assert dict(second.cycle_metrics.values) == {}

    journal.replace([_record("third", "2024-01-01T00:02:00+00:00")])
    third = next(stream)
    assert third.HasField("snapshot")
    assert [entry.fields["event"] for entry in third.snapshot.records] == ["third"]
    assert dict(third.cycle_metrics.values) == {}

    rpc_context.cancel()


def test_list_decisions_filters_and_cursor():
    journal = _DummyJournal(
        [
            {
                "event": "order_submitted",
                "timestamp": "2024-01-01T00:01:00+00:00",
                "environment": "test",
                "portfolio": "alpha",
                "strategy": "trend_follow",
                "status": "submitted",
            },
            {
                "event": "order_filled",
                "timestamp": "2024-01-01T00:02:00+00:00",
                "environment": "test",
                "portfolio": "alpha",
                "strategy": "trend_follow",
                "status": "filled",
            },
            {
                "event": "order_filled",
                "timestamp": "2024-01-01T00:03:00+00:00",
                "environment": "test",
                "portfolio": "alpha",
                "strategy": "trend_follow",
                "status": "filled",
            },
        ]
    )
    context = _DummyRuntimeContext(journal)
    servicer = api_server._RuntimeServicer(context)

    filters = trading_pb2.DecisionJournalFilters(
        strategies=["trend_follow"],
        statuses=["filled"],
    )
    filters.since.CopyFrom(_ts("2024-01-01T00:02:00Z"))
    filters.until.CopyFrom(_ts("2024-01-01T00:04:00Z"))

    first = servicer.ListDecisions(
        trading_pb2.ListDecisionsRequest(limit=1, filters=filters),
        _DummyRpcContext(),
    )
    assert context.authorized is True
    assert first.total == 2
    assert first.cursor == 1
    assert first.has_more is True
    assert [entry.fields["event"] for entry in first.records] == ["order_filled"]

    second = servicer.ListDecisions(
        trading_pb2.ListDecisionsRequest(cursor=first.cursor, filters=filters),
        _DummyRpcContext(),
    )
    assert second.total == 2
    assert second.cursor == 2
    assert second.has_more is False
    assert [entry.fields["timestamp"] for entry in second.records] == ["2024-01-01T00:03:00+00:00"]
    assert dict(first.cycle_metrics.values) == {}
    assert dict(second.cycle_metrics.values) == {}


def test_list_decisions_includes_cycle_metrics():
    journal = _DummyJournal([_record("first")])

    class _MetricAutoTrader:
        def __init__(self) -> None:
            self._decision_journal = journal
            self._base_metric_labels = {"portfolio": "alpha"}

        def _snapshot_decision_metrics(self, labels):
            assert labels == self._base_metric_labels
            return {
                "cycles_total": 12.0,
                "strategy_switch_total": 2,
                "guardrail_blocks_total": 1,
            }

    context = _DummyRuntimeContext(journal, _MetricAutoTrader())
    servicer = api_server._RuntimeServicer(context)

    response = servicer.ListDecisions(
        trading_pb2.ListDecisionsRequest(limit=1),
        _DummyRpcContext(),
    )

    metrics = dict(response.cycle_metrics.values)
    assert metrics["cycles_total"] == pytest.approx(12.0)
    assert metrics["strategy_switch_total"] == pytest.approx(2.0)
    assert metrics["guardrail_blocks_total"] == pytest.approx(1.0)


def test_stream_includes_cycle_metrics_when_available():
    journal = _DummyJournal([_record("first")])

    class _MetricAutoTrader:
        def __init__(self) -> None:
            self._decision_journal = journal
            self._base_metric_labels = {"portfolio": "alpha"}

        def _snapshot_decision_metrics(self, labels):
            assert labels == self._base_metric_labels
            return {
                "cycles_total": 21.0,
                "strategy_switch_total": 5.0,
                "guardrail_blocks_total": 0.0,
            }

    context = _DummyRuntimeContext(journal, _MetricAutoTrader())
    servicer = api_server._RuntimeServicer(context)
    request = trading_pb2.StreamDecisionsRequest(limit=1, poll_interval_seconds=0.01)
    rpc_context = _DummyRpcContext()

    stream = servicer.StreamDecisions(request, rpc_context)
    first = next(stream)
    metrics = dict(first.cycle_metrics.values)
    assert metrics["cycles_total"] == pytest.approx(21.0)
    assert metrics["strategy_switch_total"] == pytest.approx(5.0)
    assert metrics["guardrail_blocks_total"] == pytest.approx(0.0)
