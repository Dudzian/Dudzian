from __future__ import annotations

from bot_core.exchanges.streaming import LocalLongPollStream, StreamBatch
from bot_core.observability.metrics import MetricsRegistry
from bot_core.observability.ui_metrics import LongPollStreamMetricsCache


def test_long_poll_metrics_cache_collects_snapshots() -> None:
    registry = MetricsRegistry()
    stream = LocalLongPollStream(
        base_url="http://127.0.0.1:9999",
        path="/stream/demo",
        channels=["ticker"],
        adapter="binance",
        scope="spot",
        environment="paper",
        metrics_registry=registry,
    )

    stream._record_poll_latency(0.15)
    stream._record_reconnect_attempt(reason="network", attempt=1)
    stream._record_reconnect_result(status="success", duration=0.3, reason="network")
    stream._record_http_error(status=503, retryable=False, duration=0.25, reason="server_error")
    stream._record_delivery_lag(
        StreamBatch(channel="ticker", events=(), received_at=stream._clock() - 0.04)
    )

    cache = LongPollStreamMetricsCache(registry=registry, refresh_interval_seconds=0.0)
    snapshots = cache.snapshot(force_refresh=True)

    assert snapshots
    entry = snapshots[0]
    assert entry["labels"] == {"adapter": "binance", "scope": "spot", "environment": "paper"}
    assert entry["requestLatency"]["count"] == 1
    assert entry["httpErrors"]["total"] == 1
    assert entry["reconnects"]["success"] == 1
