"""Helpers for exporting UI telemetry snapshots to Prometheus and alerting."""

from __future__ import annotations

import json
import logging
import time
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


def _extract_tag(payload: Mapping[str, Any]) -> str | None:
    tag = payload.get("tag")
    if isinstance(tag, str):
        normalized = tag.strip()
        if normalized:
            return normalized
    return None


def _timestamp_to_epoch_seconds(snapshot) -> float | None:
    timestamp = getattr(snapshot, "generated_at", None)
    if timestamp is None:
        return None
    seconds = getattr(timestamp, "seconds", None)
    nanos = getattr(timestamp, "nanos", None)
    if seconds is None or nanos is None:
        return None
    try:
        total = float(seconds) + float(nanos) / 1_000_000_000.0
    except (TypeError, ValueError):
        return None
    if total <= 0:
        return None
    return total


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
        tag_activity_ttl_seconds: float = 300.0,
    ) -> None:
        self._registry = registry or get_global_metrics_registry()
        self._snapshot_total: CounterMetric = self._registry.counter(
            "bot_ui_snapshots_total", "Łączna liczba przetworzonych snapshotów UI"
        )
        self._snapshot_generated_at_seconds: GaugeMetric = self._registry.gauge(
            "bot_ui_snapshot_generated_at_seconds",
            "Znacznik czasu (epoch s) ostatniego snapshotu UI",
        )
        self._snapshot_delivery_latency: GaugeMetric = self._registry.gauge(
            "bot_ui_snapshot_delivery_latency_seconds",
            "Opóźnienie dostarczenia snapshotu względem znacznika generated_at",
        )
        self._snapshot_gap_seconds: GaugeMetric = self._registry.gauge(
            "bot_ui_snapshot_gap_seconds",
            "Odstęp czasowy pomiędzy kolejnymi snapshotami UI",
        )
        self._snapshot_gap_histogram: HistogramMetric = self._registry.histogram(
            "bot_ui_snapshot_gap_duration_seconds",
            "Rozkład odstępów czasowych pomiędzy snapshotami UI",
            buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
        )
        self._fps_gauge: GaugeMetric = self._registry.gauge(
            "bot_ui_fps", "Ostatnia próbka FPS zgłoszona przez interfejs"
        )
        self._reduce_motion_state: GaugeMetric = self._registry.gauge(
            "bot_ui_reduce_motion_state", "Stan reduce-motion (1=aktywne,0=nieaktywne)"
        )
        self._reduce_motion_counter: CounterMetric = self._registry.counter(
            "bot_ui_reduce_motion_events_total", "Liczba przełączeń reduce-motion"
        )
        self._reduce_motion_incident_active: GaugeMetric = self._registry.gauge(
            "bot_ui_reduce_motion_incident_active",
            "Czy incydent reduce-motion jest aktywny (0/1)",
        )
        self._reduce_motion_incident_age_seconds: GaugeMetric = self._registry.gauge(
            "bot_ui_reduce_motion_incident_age_seconds",
            "Czas trwania aktywnego incydentu reduce-motion",
        )
        self._reduce_motion_incident_started_at_seconds: GaugeMetric = self._registry.gauge(
            "bot_ui_reduce_motion_incident_started_at_seconds",
            "Znacznik czasu rozpoczęcia incydentu reduce-motion (epoch s)",
        )
        self._reduce_motion_incident_duration_histogram: HistogramMetric = (
            self._registry.histogram(
                "bot_ui_reduce_motion_incident_duration_seconds",
                "Czas trwania incydentów reduce-motion",
                buckets=(5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0),
            )
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
        self._retry_incident_active: GaugeMetric = self._registry.gauge(
            "bot_ui_retry_incident_active",
            "Czy incydent backlogu retry jest aktywny (0/1)",
        )
        self._retry_incident_age_seconds: GaugeMetric = self._registry.gauge(
            "bot_ui_retry_incident_age_seconds",
            "Czas trwania aktywnego incydentu backlogu retry",
        )
        self._retry_incident_started_at_seconds: GaugeMetric = self._registry.gauge(
            "bot_ui_retry_incident_started_at_seconds",
            "Znacznik czasu rozpoczęcia incydentu backlogu retry (epoch s)",
        )
        self._retry_incident_duration_histogram: HistogramMetric = self._registry.histogram(
            "bot_ui_retry_incident_duration_seconds",
            "Czas trwania incydentów backlogu retry",
            buckets=(5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0),
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
        self._tag_active: GaugeMetric = self._registry.gauge(
            "bot_ui_tag_active",
            "Czy tag UI jest aktywny w zadanym oknie czasowym (0=nieaktywny,1=aktywny)",
        )
        self._tag_active_count: GaugeMetric = self._registry.gauge(
            "bot_ui_tag_active_count",
            "Liczba tagów UI aktywnych w zadanym oknie czasowym",
        )
        self._tag_last_seen_seconds: GaugeMetric = self._registry.gauge(
            "bot_ui_tag_last_seen_seconds",
            "Znacznik czasu ostatniego snapshotu otrzymanego od danego tagu UI",
        )
        self._tag_inactive: GaugeMetric = self._registry.gauge(
            "bot_ui_tag_inactive",
            "Czy tag UI przekroczył okno aktywności (0=aktywny,1=nieaktywny)",
        )
        self._tag_inactive_count: GaugeMetric = self._registry.gauge(
            "bot_ui_tag_inactive_count",
            "Liczba tagów UI, które przekroczyły dozwolone okno aktywności",
        )
        self._tag_inactive_age_seconds: GaugeMetric = self._registry.gauge(
            "bot_ui_tag_inactive_age_seconds",
            "Czas w sekundach od ostatniego snapshotu tagu UI dla tagów nieaktywnych",
        )
        self._jank_histogram: HistogramMetric = self._registry.histogram(
            "bot_ui_jank_frame_overrun_ms",
            "Budżet janku dla UI (wartości powyżej progu)",
            buckets=(1.0, 5.0, 10.0, 25.0, 50.0, 100.0),
        )
        self._alert_sink = alert_sink
        self._last_reduce_motion_state: bool | None = None
        self._reduce_motion_state_by_tag: dict[str, bool] = {}
        self._reduce_motion_incident_started_epoch: float | None = None
        self._reduce_motion_incident_started_epoch_by_tag: dict[str, float] = {}
        self._retry_incident_started_epoch: float | None = None
        self._last_snapshot_generated_epoch: float | None = None
        self._last_snapshot_generated_epoch_by_tag: dict[str, float] = {}
        self._tag_last_seen_monotonic: dict[str, float] = {}
        self._tag_last_seen_epoch: dict[str, float] = {}
        self._tag_is_active: dict[str, bool] = {}
        self._tag_activity_ttl_seconds = max(0.0, float(tag_activity_ttl_seconds))

    def handle_snapshot(self, snapshot) -> None:
        """Updates metrics and forwards the snapshot to alert sinks."""

        monotonic_fn = getattr(time, "monotonic", None)
        if callable(monotonic_fn):  # pragma: no branch - prefer monotonic when dostępny
            try:
                now_monotonic = float(monotonic_fn())
            except Exception:  # pragma: no cover - awaryjny fallback dla nietypowych zegarów
                now_monotonic = float(time.time())
        else:  # pragma: no cover - środowiska bez monotonic
            now_monotonic = float(time.time())
        payload = _safe_json_loads(getattr(snapshot, "notes", ""))
        event = str(payload.get("event") or "").strip()
        tag_value = _extract_tag(payload)
        tag_labels = {"tag": tag_value} if tag_value else None

        self._snapshot_total.inc()
        if tag_labels:
            self._snapshot_total.inc(labels=tag_labels)

        timestamp_seconds = _timestamp_to_epoch_seconds(snapshot)
        effective_timestamp = timestamp_seconds if timestamp_seconds is not None else time.time()
        if timestamp_seconds is not None:
            self._snapshot_generated_at_seconds.set(timestamp_seconds)
            if tag_labels:
                self._snapshot_generated_at_seconds.set(timestamp_seconds, labels=tag_labels)
            delivery_latency = max(0.0, time.time() - timestamp_seconds)
            self._snapshot_delivery_latency.set(delivery_latency)
            if tag_labels:
                self._snapshot_delivery_latency.set(delivery_latency, labels=tag_labels)
            if self._last_snapshot_generated_epoch is not None:
                gap = max(0.0, timestamp_seconds - self._last_snapshot_generated_epoch)
                self._snapshot_gap_seconds.set(gap)
                self._snapshot_gap_histogram.observe(gap)
            if tag_labels:
                last_for_tag = self._last_snapshot_generated_epoch_by_tag.get(tag_value)  # type: ignore[arg-type]
                if last_for_tag is not None:
                    tag_gap = max(0.0, timestamp_seconds - last_for_tag)
                    self._snapshot_gap_seconds.set(tag_gap, labels=tag_labels)
                    self._snapshot_gap_histogram.observe(tag_gap, labels=tag_labels)
                self._last_snapshot_generated_epoch_by_tag[tag_value] = timestamp_seconds  # type: ignore[index]
            self._last_snapshot_generated_epoch = timestamp_seconds
        else:
            self._snapshot_delivery_latency.set(0.0)
            if tag_labels:
                self._snapshot_delivery_latency.set(0.0, labels=tag_labels)

        if tag_value:
            self._tag_last_seen_monotonic[tag_value] = now_monotonic
            self._tag_last_seen_epoch[tag_value] = effective_timestamp
            self._tag_is_active[tag_value] = True
            self._tag_last_seen_seconds.set(effective_timestamp, labels=tag_labels)
            self._tag_active.set(1.0, labels=tag_labels)

        self._update_screen_metrics(payload, tag_value)

        if getattr(snapshot, "HasField", None) and snapshot.HasField("fps"):
            fps_value = float(snapshot.fps)
            self._fps_gauge.set(fps_value)
            if tag_labels:
                self._fps_gauge.set(fps_value, labels=tag_labels)

        window_count = payload.get("window_count")
        if isinstance(window_count, (int, float)):
            window_value = float(window_count)
            self._window_count.set(window_value)
            if tag_labels:
                self._window_count.set(window_value, labels=tag_labels)

        backlog_before = payload.get("retry_backlog_before_send")
        if isinstance(backlog_before, (int, float)) and backlog_before >= 0:
            before_value = float(backlog_before)
            phase_labels = {"phase": "before_flush"}
            self._retry_backlog.set(before_value, labels=phase_labels)
            if tag_labels:
                tagged_phase = dict(phase_labels)
                tagged_phase.update(tag_labels)
                self._retry_backlog.set(before_value, labels=tagged_phase)

        backlog_after_value: float | None = None
        backlog_after = payload.get("retry_backlog_after_flush")
        if isinstance(backlog_after, (int, float)) and backlog_after >= 0:
            backlog_after_value = float(backlog_after)
            after_labels = {"phase": "after_flush"}
            self._retry_backlog.set(backlog_after_value, labels=after_labels)
            if tag_labels:
                tagged_after = dict(after_labels)
                tagged_after.update(tag_labels)
                self._retry_backlog.set(backlog_after_value, labels=tagged_after)

        self._update_retry_incident_metrics(snapshot, backlog_after_value)

        if event == "reduce_motion":
            active = bool(payload.get("active", False))
            state_value = 1.0 if active else 0.0
            self._reduce_motion_state.set(state_value)
            if tag_labels:
                self._reduce_motion_state.set(state_value, labels=tag_labels)
            if self._last_reduce_motion_state is None or self._last_reduce_motion_state != active:
                label = {"state": "active" if active else "inactive"}
                self._reduce_motion_counter.inc(labels=label)
            self._last_reduce_motion_state = active
            if tag_value:
                previous_tag_state = self._reduce_motion_state_by_tag.get(tag_value)
                if previous_tag_state is None or previous_tag_state != active:
                    tag_label = {"state": "active" if active else "inactive", "tag": tag_value}
                    self._reduce_motion_counter.inc(labels=tag_label)
                self._reduce_motion_state_by_tag[tag_value] = active

            now_epoch = timestamp_seconds if timestamp_seconds is not None else time.time()
            self._update_reduce_motion_incident_metrics(active, now_epoch)
            if tag_value:
                self._update_reduce_motion_incident_metrics_for_tag(tag_value, active, now_epoch)

        elif event == "overlay_budget":
            active = payload.get("active_overlays")
            allowed = payload.get("allowed_overlays")
            if isinstance(active, (int, float)):
                active_value = float(active)
                self._overlay_active.set(active_value)
                if tag_labels:
                    self._overlay_active.set(active_value, labels=tag_labels)
            if isinstance(allowed, (int, float)):
                allowed_value = float(allowed)
                self._overlay_allowed.set(allowed_value)
                if tag_labels:
                    self._overlay_allowed.set(allowed_value, labels=tag_labels)

        elif event == "jank_spike":
            frame_ms = payload.get("frame_ms")
            threshold = payload.get("threshold_ms")
            if isinstance(frame_ms, (int, float)) and isinstance(threshold, (int, float)):
                overrun = float(frame_ms) - float(threshold)
                if overrun > 0:
                    self._jank_histogram.observe(overrun)
                    if tag_labels:
                        self._jank_histogram.observe(overrun, labels=tag_labels)

        if self._alert_sink is not None:
            try:
                self._alert_sink.handle_snapshot(snapshot)
            except Exception:  # pragma: no cover - alert sink should not break metrics
                LOGGER.exception("UiTelemetryAlertSink zgłosił wyjątek")

        self._update_tag_activity(now_monotonic)

    def _update_screen_metrics(self, payload: Mapping[str, Any], tag: str | None) -> None:
        screen = payload.get("screen")
        if not isinstance(screen, Mapping):
            return

        labels = _screen_labels(screen)

        refresh_hz = screen.get("refresh_hz")
        if isinstance(refresh_hz, (int, float)) and refresh_hz > 0:
            refresh_value = float(refresh_hz)
            self._screen_refresh_hz.set(refresh_value, labels=labels)
            if tag:
                tagged_labels = dict(labels)
                tagged_labels["tag"] = tag
                self._screen_refresh_hz.set(refresh_value, labels=tagged_labels)

        dpr = screen.get("device_pixel_ratio")
        if isinstance(dpr, (int, float)) and dpr > 0:
            dpr_value = float(dpr)
            self._screen_device_pixel_ratio.set(dpr_value, labels=labels)
            if tag:
                tagged_labels = dict(labels)
                tagged_labels["tag"] = tag
                self._screen_device_pixel_ratio.set(dpr_value, labels=tagged_labels)

        geometry = screen.get("geometry_px")
        if isinstance(geometry, Mapping):
            width = geometry.get("width")
            if isinstance(width, (int, float)) and width > 0:
                width_labels = dict(labels)
                width_labels["dimension"] = "width"
                width_value = float(width)
                self._screen_resolution_px.set(width_value, labels=width_labels)
                if tag:
                    tagged_width = dict(width_labels)
                    tagged_width["tag"] = tag
                    self._screen_resolution_px.set(width_value, labels=tagged_width)
            height = geometry.get("height")
            if isinstance(height, (int, float)) and height > 0:
                height_labels = dict(labels)
                height_labels["dimension"] = "height"
                height_value = float(height)
                self._screen_resolution_px.set(height_value, labels=height_labels)
                if tag:
                    tagged_height = dict(height_labels)
                    tagged_height["tag"] = tag
                    self._screen_resolution_px.set(height_value, labels=tagged_height)

    def _update_tag_activity(self, now_monotonic: float) -> None:
        if not self._tag_last_seen_monotonic:
            self._tag_active_count.set(0.0)
            return

        ttl = self._tag_activity_ttl_seconds
        if ttl == 0:
            active_tags = len(self._tag_last_seen_monotonic)
            self._tag_active_count.set(float(active_tags))
            self._tag_inactive_count.set(0.0)
            for tag in self._tag_last_seen_monotonic:
                labels = {"tag": tag}
                self._tag_active.set(1.0, labels=labels)
                self._tag_inactive.set(0.0, labels=labels)
                self._tag_inactive_age_seconds.set(0.0, labels=labels)
                self._tag_is_active[tag] = True
            return

        active_tags = 0
        inactive_tags = 0
        stale_cutoff = ttl * 2 if ttl > 0 else float("inf")
        to_delete: list[str] = []
        for tag, last_seen in list(self._tag_last_seen_monotonic.items()):
            age = now_monotonic - last_seen
            is_active = age <= ttl
            if is_active:
                active_tags += 1
                labels = {"tag": tag}
                self._tag_active.set(1.0, labels=labels)
                self._tag_inactive.set(0.0, labels=labels)
                self._tag_inactive_age_seconds.set(0.0, labels=labels)
                self._tag_is_active[tag] = True
            else:
                labels = {"tag": tag}
                self._tag_active.set(0.0, labels=labels)
                if age > stale_cutoff:
                    self._tag_inactive.set(0.0, labels=labels)
                    self._tag_inactive_age_seconds.set(0.0, labels=labels)
                    self._tag_is_active.pop(tag, None)
                    self._finalize_reduce_motion_incident_for_tag(
                        tag, end_epoch=self._tag_last_seen_epoch.get(tag)
                    )
                    to_delete.append(tag)
                else:
                    self._tag_inactive.set(1.0, labels=labels)
                    self._tag_inactive_age_seconds.set(max(0.0, age), labels=labels)
                    inactive_tags += 1
                    self._tag_is_active[tag] = False
                    self._finalize_reduce_motion_incident_for_tag(
                        tag, end_epoch=self._tag_last_seen_epoch.get(tag)
                    )

        for tag in to_delete:
            labels = {"tag": tag}
            self._tag_active.set(0.0, labels=labels)
            self._tag_inactive.set(0.0, labels=labels)
            self._tag_inactive_age_seconds.set(0.0, labels=labels)
            self._tag_last_seen_seconds.set(0.0, labels=labels)
            self._tag_last_seen_monotonic.pop(tag, None)
            self._tag_last_seen_epoch.pop(tag, None)
            self._tag_is_active.pop(tag, None)
            self._finalize_reduce_motion_incident_for_tag(tag)

        self._tag_active_count.set(float(active_tags))
        self._tag_inactive_count.set(float(inactive_tags))

    def _update_reduce_motion_incident_metrics(self, active: bool, now_epoch: float) -> None:
        if active:
            if self._reduce_motion_incident_started_epoch is None:
                self._reduce_motion_incident_started_epoch = now_epoch
                self._reduce_motion_incident_started_at_seconds.set(now_epoch)
            elif self._reduce_motion_incident_started_epoch > now_epoch:
                self._reduce_motion_incident_started_epoch = now_epoch
                self._reduce_motion_incident_started_at_seconds.set(now_epoch)

            if self._reduce_motion_incident_started_epoch is not None:
                age = max(0.0, now_epoch - self._reduce_motion_incident_started_epoch)
            else:
                age = 0.0
            self._reduce_motion_incident_active.set(1.0)
            self._reduce_motion_incident_age_seconds.set(age)
            if self._reduce_motion_incident_started_epoch is not None:
                self._reduce_motion_incident_started_at_seconds.set(
                    self._reduce_motion_incident_started_epoch
                )
            return

        if self._reduce_motion_incident_started_epoch is not None:
            duration = max(0.0, now_epoch - self._reduce_motion_incident_started_epoch)
            self._reduce_motion_incident_duration_histogram.observe(duration)

        self._reduce_motion_incident_active.set(0.0)
        self._reduce_motion_incident_age_seconds.set(0.0)
        self._reduce_motion_incident_started_at_seconds.set(0.0)
        self._reduce_motion_incident_started_epoch = None

    def _update_reduce_motion_incident_metrics_for_tag(
        self, tag: str, active: bool, now_epoch: float
    ) -> None:
        labels = {"tag": tag}
        started = self._reduce_motion_incident_started_epoch_by_tag.get(tag)

        if active:
            if started is None or started > now_epoch:
                self._reduce_motion_incident_started_epoch_by_tag[tag] = now_epoch
                started = now_epoch
                self._reduce_motion_incident_started_at_seconds.set(now_epoch, labels=labels)
            age = max(0.0, now_epoch - started)
            self._reduce_motion_incident_active.set(1.0, labels=labels)
            self._reduce_motion_incident_age_seconds.set(age, labels=labels)
            self._reduce_motion_incident_started_at_seconds.set(started, labels=labels)
            return

        if started is not None:
            duration = max(0.0, now_epoch - started)
            self._reduce_motion_incident_duration_histogram.observe(
                duration, labels=labels
            )
        self._reduce_motion_incident_active.set(0.0, labels=labels)
        self._reduce_motion_incident_age_seconds.set(0.0, labels=labels)
        self._reduce_motion_incident_started_at_seconds.set(0.0, labels=labels)
        self._reduce_motion_incident_started_epoch_by_tag.pop(tag, None)

    def _finalize_reduce_motion_incident_for_tag(
        self, tag: str, *, end_epoch: float | None = None
    ) -> None:
        if tag not in self._reduce_motion_incident_started_epoch_by_tag:
            return

        labels = {"tag": tag}
        started = self._reduce_motion_incident_started_epoch_by_tag.pop(tag, None)
        if started is not None:
            end = end_epoch if end_epoch is not None else time.time()
            if end < started:
                end = started
            duration = max(0.0, end - started)
            self._reduce_motion_incident_duration_histogram.observe(duration, labels=labels)
        self._reduce_motion_incident_active.set(0.0, labels=labels)
        self._reduce_motion_incident_age_seconds.set(0.0, labels=labels)
        self._reduce_motion_incident_started_at_seconds.set(0.0, labels=labels)

    def _update_retry_incident_metrics(self, snapshot, backlog_after: float | None) -> None:
        if backlog_after is None:
            return

        timestamp_seconds = _timestamp_to_epoch_seconds(snapshot)
        now = timestamp_seconds if timestamp_seconds is not None else time.time()

        if backlog_after > 0:
            if self._retry_incident_started_epoch is None:
                self._retry_incident_started_epoch = now
                self._retry_incident_started_at_seconds.set(self._retry_incident_started_epoch)
            elif self._retry_incident_started_epoch > now:
                # Nie pozwalamy, by czas rozpoczęcia był w przyszłości względem bieżącej próbki
                self._retry_incident_started_epoch = now
                self._retry_incident_started_at_seconds.set(self._retry_incident_started_epoch)

            if self._retry_incident_started_epoch is not None:
                age = max(0.0, now - self._retry_incident_started_epoch)
            else:
                age = 0.0
            self._retry_incident_active.set(1.0)
            self._retry_incident_age_seconds.set(age)
            if self._retry_incident_started_epoch is not None:
                self._retry_incident_started_at_seconds.set(self._retry_incident_started_epoch)
            return

        # backlog_after == 0 -> incydent wygaszony
        if self._retry_incident_started_epoch is not None:
            duration = max(0.0, now - self._retry_incident_started_epoch)
            self._retry_incident_duration_histogram.observe(duration)

        self._retry_incident_active.set(0.0)
        self._retry_incident_age_seconds.set(0.0)
        self._retry_incident_started_at_seconds.set(0.0)
        self._retry_incident_started_epoch = None


__all__ = ["UiTelemetryPrometheusExporter"]

