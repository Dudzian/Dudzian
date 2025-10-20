"""Warstwa zgodności delegująca do ``bot_core.events``."""
from __future__ import annotations

from bot_core.events.emitter import *  # noqa: F401,F403

__all__ = [
    "Event",
    "EventType",
    "DebounceRule",
    "EventBus",
    "EventEmitter",
    "EmitterConfig",
    "EmitterAdapter",
    "EventEmitterAdapter",
    "DummyMarketFeed",
    "DummyMarketFeedConfig",
    "wire_gui_logs_to_adapter",
    "unwire_gui_logs_from_adapter",
    "wire_logging_to_bus",
]
