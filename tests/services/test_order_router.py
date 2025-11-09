from __future__ import annotations

from typing import Any, Dict, Iterable, List

import pytest

from bot_core.events import Event, EventBus, EventType
from bot_core.services.order_router import PaperBroker, PaperBrokerConfig


@pytest.fixture()
def paper_broker() -> PaperBroker:
    bus = EventBus()
    return PaperBroker(bus, PaperBrokerConfig(symbol="BTCUSDT", initial_cash=1000.0))


def test_paper_broker_executes_and_tracks_state(paper_broker: PaperBroker) -> None:
    bus = paper_broker.bus
    fills: List[Dict[str, Any]] = []
    pnls: List[Dict[str, Any]] = []

    bus.subscribe(EventType.TRADE_EXECUTED, lambda event: _capture_payloads(event, fills))
    bus.subscribe(EventType.PNL_UPDATE, lambda event: _capture_payloads(event, pnls))

    bus.publish(EventType.MARKET_TICK, {"symbol": "BTCUSDT", "price": 100.0})
    paper_broker.place_order("BUY", 2.0, price=100.0)

    assert paper_broker.position_qty == pytest.approx(2.0)
    assert paper_broker.cash < 1000.0
    assert fills and fills[0]["status"] == "filled"
    assert pnls, "Powinien zostać opublikowany przynajmniej jeden update PnL"


def _capture_payloads(event: Event | Iterable[Event] | None, container: List[Dict[str, Any]]) -> None:
    if event is None:
        return
    events: Iterable[Event]
    if isinstance(event, Event):
        events = [event]
    elif isinstance(event, Iterable):
        events = event
    else:
        return
    for item in events:
        if isinstance(item, Event) and item.payload is not None:
            container.append(dict(item.payload))
