"""Helpers for exporting UI telemetry snapshots to Prometheus and alerting."""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping

from bot_core.observability.metrics import (
    CounterMetric,
    GaugeMetric,
    HistogramMetric,
    MetricsRegistry,
    get_global_metrics_registry,
)

try:  # pragma: no cover - optional dependency during unit tests
    from bot_core.runtime.metrics_alerts import UiTelemetryAlertSink
except Exception:  # pragma: no cover
    UiTelemetryAlertSink = None  # type: ignore

LOGGER = logging.getLogger(__name__)


def _safe_json_loads(payload: str) -> Mapping[str, Any]:
    if not payload:
        return {}
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        LOGGER.debug("Niepoprawny JSON w polu notes telemetrii: %s", payload)
        return {}
    if isinstance(data, Mapping):
        return data
    LOGGER.debug("Pole notes powinno zawierać obiekt JSON, otrzymano: %r", data)
    return {}


def _screen_labels(screen_payload: Mapping[str, Any]) -> dict[str, str]:
    labels: dict[str, str] = {}
    index = screen_payload.get("index")
    if isinstance(index, (int, float)):
        index_int = int(index)
        if index_int >= 0:
            labels["screen_index"] = str(index_int)
    name = screen_payload.get("name")
    if isinstance(name, str) and name.strip():
        labels["screen_name"] = name.strip()
    return labels


class UiTelemetryPrometheusExporter:
    """Transforms UI telemetry snapshots into Prometheus metrics and alerts."""

    def __init__(
        self,
        *,
        registry: MetricsRegistry | None = None,
        alert_sink: UiTelemetryAlertSink | None = None,
    ) -> None:
        self._registry = registry or get_global_metrics_registry()
        self._fps_gauge: GaugeMetric = self._registry.gauge(
            "bot_ui_fps", "Ostatnia próbka FPS zgłoszona przez interfejs"
        )
        self._reduce_motion_state: GaugeMetric = self._registry.gauge(
            "bot_ui_reduce_motion_state", "Stan reduce-motion (1=aktywne,0=nieaktywne)"
        )
        self._reduce_motion_counter: CounterMetric = self._registry.counter(
            "bot_ui_reduce_motion_events_total", "Liczba przełączeń reduce-motion"
        )
        self._overlay_active: GaugeMetric = self._registry.gauge(
            "bot_ui_overlay_active", "Liczba aktywnych nakładek"
        )
        self._overlay_allowed: GaugeMetric = self._registry.gauge(
            "bot_ui_overlay_allowed", "Limit nakładek"
        )
        self._window_count: GaugeMetric = self._registry.gauge(
            "bot_ui_window_count", "Liczba aktywnych okien UI"
        )
        self._retry_backlog: GaugeMetric = self._registry.gauge(
            "bot_ui_retry_backlog",
            "Rozmiar bufora retry telemetrii UI (phase=before_flush/after_flush)",
        )
        self._screen_refresh_hz: GaugeMetric = self._registry.gauge(
            "bot_ui_screen_refresh_hz", "Częstotliwość odświeżania zgłaszana przez UI"
        )
        self._screen_device_pixel_ratio: GaugeMetric = self._registry.gauge(
            "bot_ui_screen_device_pixel_ratio",
            "Device Pixel Ratio aktywnego ekranu",
        )
        self._screen_resolution_px: GaugeMetric = self._registry.gauge(
            "bot_ui_screen_resolution_px",
            "Rozdzielczość ekranu w pikselach (etykieta dimension)",
        )
        self._jank_histogram: HistogramMetric = self._registry.histogram(
            "bot_ui_jank_frame_overrun_ms",
            "Budżet janku dla UI (wartości powyżej progu)",
            buckets=(1.0, 5.0, 10.0, 25.0, 50.0, 100.0),
        )
        self._alert_sink = alert_sink
        self._last_reduce_motion_state: bool | None = None

    def handle_snapshot(self, snapshot) -> None:
        """Updates metrics and forwards the snapshot to alert sinks."""

        payload = _safe_json_loads(getattr(snapshot, "notes", ""))
        event = str(payload.get("event") or "").strip()

        self._update_screen_metrics(payload)

        if getattr(snapshot, "HasField", None) and snapshot.HasField("fps"):
            self._fps_gauge.set(float(snapshot.fps))

        window_count = payload.get("window_count")
        if isinstance(window_count, (int, float)):
            self._window_count.set(float(window_count))

        backlog_before = payload.get("retry_backlog_before_send")
        if isinstance(backlog_before, (int, float)) and backlog_before >= 0:
            self._retry_backlog.set(float(backlog_before), labels={"phase": "before_flush"})

        backlog_after = payload.get("retry_backlog_after_flush")
        if isinstance(backlog_after, (int, float)) and backlog_after >= 0:
            self._retry_backlog.set(float(backlog_after), labels={"phase": "after_flush"})

        if event == "reduce_motion":
            active = bool(payload.get("active", False))
            self._reduce_motion_state.set(1.0 if active else 0.0)
            if self._last_reduce_motion_state is None or self._last_reduce_motion_state != active:
                label = {"state": "active" if active else "inactive"}
                self._reduce_motion_counter.inc(labels=label)
            self._last_reduce_motion_state = active

        elif event == "overlay_budget":
            active = payload.get("active_overlays")
            allowed = payload.get("allowed_overlays")
            if isinstance(active, (int, float)):
                self._overlay_active.set(float(active))
            if isinstance(allowed, (int, float)):
                self._overlay_allowed.set(float(allowed))

        elif event == "jank_spike":
            frame_ms = payload.get("frame_ms")
            threshold = payload.get("threshold_ms")
            if isinstance(frame_ms, (int, float)) and isinstance(threshold, (int, float)):
                overrun = float(frame_ms) - float(threshold)
                if overrun > 0:
                    self._jank_histogram.observe(overrun)

        if self._alert_sink is not None:
            try:
                self._alert_sink.handle_snapshot(snapshot)
            except Exception:  # pragma: no cover - alert sink should not break metrics
                LOGGER.exception("UiTelemetryAlertSink zgłosił wyjątek")

    def _update_screen_metrics(self, payload: Mapping[str, Any]) -> None:
        screen = payload.get("screen")
        if not isinstance(screen, Mapping):
            return

        labels = _screen_labels(screen)

        refresh_hz = screen.get("refresh_hz")
        if isinstance(refresh_hz, (int, float)) and refresh_hz > 0:
            self._screen_refresh_hz.set(float(refresh_hz), labels=labels)

        dpr = screen.get("device_pixel_ratio")
        if isinstance(dpr, (int, float)) and dpr > 0:
            self._screen_device_pixel_ratio.set(float(dpr), labels=labels)

        geometry = screen.get("geometry_px")
        if isinstance(geometry, Mapping):
            width = geometry.get("width")
            if isinstance(width, (int, float)) and width > 0:
                width_labels = dict(labels)
                width_labels["dimension"] = "width"
                self._screen_resolution_px.set(float(width), labels=width_labels)
            height = geometry.get("height")
            if isinstance(height, (int, float)) and height > 0:
                height_labels = dict(labels)
                height_labels["dimension"] = "height"
                self._screen_resolution_px.set(float(height), labels=height_labels)


__all__ = ["UiTelemetryPrometheusExporter"]

