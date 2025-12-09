"""Warstwa I/O dla ExchangeManager: event bus, subskrypcje i publikacja zdarzeń."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from bot_core.exchanges.core import Event, EventBus


@dataclass(slots=True)
class ExchangeIOLayer:
    """Odpowiada za komunikację zdarzeniową warstwy giełdowej."""

    event_bus: EventBus = field(default_factory=EventBus)

    def publish_event(self, event_type: str, payload: Mapping[str, Any] | None = None) -> None:
        event_payload = dict(payload or {})
        self.event_bus.publish(Event(type=event_type, payload=event_payload))

    def subscribe(self, event_type: str, callback) -> None:
        self.event_bus.subscribe(event_type, callback)


__all__ = ["ExchangeIOLayer"]
