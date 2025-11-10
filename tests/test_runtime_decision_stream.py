from types import SimpleNamespace

import pytest

from bot_core.api import server as api_server
from bot_core.generated import trading_pb2


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
    def __init__(self, journal):
        self.auto_trader = SimpleNamespace(_decision_journal=journal)
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

    journal.append(_record("third", "2024-01-01T00:02:00+00:00"))
    second = next(stream)
    assert second.HasField("increment")
    assert second.increment.record.fields["event"] == "third"

    journal.replace([_record("third", "2024-01-01T00:02:00+00:00")])
    third = next(stream)
    assert third.HasField("snapshot")
    assert [entry.fields["event"] for entry in third.snapshot.records] == ["third"]

    rpc_context.cancel()
