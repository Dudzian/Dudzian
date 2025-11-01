"""Pakiet wspierający anonimową telemetrię."""

from .anonymous_collector import (
    AnonymousTelemetryCollector,
    DEFAULT_TELEMETRY_DIR,
    TelemetryError,
    TelemetryEvent,
    TelemetrySettings,
)

__all__ = [
    "AnonymousTelemetryCollector",
    "DEFAULT_TELEMETRY_DIR",
    "TelemetryError",
    "TelemetryEvent",
    "TelemetrySettings",
]
