from __future__ import annotations

import time

from bot_core.events import (
    DebounceRule,
    DummyMarketFeed,
    DummyMarketFeedConfig,
    EmitterAdapter,
    EventBus,
    EventType,
)


def test_event_bus_batch_delivery() -> None:
    bus = EventBus()
    batches: list[list] = []

    bus.subscribe(
        EventType.MARKET_TICK,
        lambda evts: batches.append(evts),
        rule=DebounceRule(window=1.0, max_batch=2),
    )

    bus.publish(EventType.MARKET_TICK, {"symbol": "BTC", "price": 1})
    bus.publish(EventType.MARKET_TICK, {"symbol": "BTC", "price": 2})

    assert len(batches) == 1
    assert [e.payload["price"] for e in batches[0]] == [1, 2]


def test_emitter_adapter_autotrade_status() -> None:
    adapter = EmitterAdapter()
    events: list = []
    adapter.subscribe(EventType.AUTOTRADE_STATUS, events.append)

    adapter.push_autotrade_status("enabled", detail={"symbol": "BTCUSDT"}, level="INFO")
    time.sleep(0.05)

    assert events
    payload = events[-1].payload
    assert payload["status"] == "enabled"
    assert payload["detail"]["symbol"] == "BTCUSDT"


def test_dummy_market_feed_generates_ticks() -> None:
    bus = EventBus()
    feed = DummyMarketFeed(bus, cfg=DummyMarketFeedConfig(max_ticks=3, interval_sec=0.01, seed=1))
    prices: list[float] = []

    bus.subscribe(EventType.MARKET_TICK, lambda ev: prices.append(ev.payload["price"]))

    feed.start().join(1.0)
    feed.stop()

    assert len(prices) == 3
    assert all(price > 0 for price in prices)
