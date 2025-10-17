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


def _normalize_threshold(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    return numeric


_PERFORMANCE_SEVERITY_LEVELS: dict[str, float] = {
    "warning": 1.0,
    "critical": 2.0,
}


class UiTelemetryPrometheusExporter:
    """Transforms UI telemetry snapshots into Prometheus metrics and alerts."""

    def __init__(
        self,
        *,
        registry: MetricsRegistry | None = None,
        alert_sink: UiTelemetryAlertSink | None = None,
        tag_activity_ttl_seconds: float = 300.0,
        overlay_critical_difference_threshold: float | int | None = 2,
        overlay_critical_duration_threshold_seconds: float | int | None = None,
        jank_incident_quiet_seconds: float = 15.0,
        jank_critical_over_ms: float | int | None = None,
        performance_event_to_frame_warning_ms: float | int | None = 45.0,
        performance_event_to_frame_critical_ms: float | int | None = 60.0,
        cpu_utilization_warning_percent: float | int | None = 85.0,
        cpu_utilization_critical_percent: float | int | None = 95.0,
        gpu_utilization_warning_percent: float | int | None = None,
        gpu_utilization_critical_percent: float | int | None = None,
        ram_usage_warning_megabytes: float | int | None = None,
        ram_usage_critical_megabytes: float | int | None = None,
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
        self._event_to_frame_p95_ms: GaugeMetric = self._registry.gauge(
            "bot_ui_event_to_frame_p95_ms",
            "Opóźnienie (p95) od zdarzenia do klatki w milisekundach",
        )
        self._event_to_frame_histogram: HistogramMetric = self._registry.histogram(
            "bot_ui_event_to_frame_p95_ms_distribution",
            "Rozkład opóźnienia (p95) od zdarzenia do klatki w milisekundach",
            buckets=(4.0, 8.0, 16.0, 24.0, 33.0, 50.0, 100.0, 250.0),
        )
        self._cpu_utilization_percent: GaugeMetric = self._registry.gauge(
            "bot_ui_cpu_utilization_percent",
            "Zużycie CPU procesu UI (w procentach)",
        )
        self._gpu_utilization_percent: GaugeMetric = self._registry.gauge(
            "bot_ui_gpu_utilization_percent",
            "Zużycie GPU procesu UI (w procentach)",
        )
        self._ram_usage_megabytes: GaugeMetric = self._registry.gauge(
            "bot_ui_ram_usage_megabytes",
            "Zużycie pamięci RAM przez UI (w MB)",
        )
        self._dropped_frames_total: GaugeMetric = self._registry.gauge(
            "bot_ui_dropped_frames_total",
            "Łączna liczba utraconych klatek zgłoszona przez UI",
        )
        self._processed_messages_per_second: GaugeMetric = self._registry.gauge(
            "bot_ui_processed_messages_per_second",
            "Przetworzone wiadomości per sekunda w kolejce UI",
        )
        self._performance_metric_state: GaugeMetric = self._registry.gauge(
            "bot_ui_performance_metric_state",
            "Aktualny poziom surowości metryk wydajności UI (0=OK,1=warning,2=critical)",
        )
        self._performance_incidents_total: CounterMetric = self._registry.counter(
            "bot_ui_performance_incidents_total",
            "Łączna liczba incydentów metryk wydajności UI",
        )
        self._performance_severity_transitions_total: CounterMetric = (
            self._registry.counter(
                "bot_ui_performance_severity_transitions_total",
                "Przejścia stanów surowości metryk wydajności UI",
            )
        )
        self._performance_incident_active: GaugeMetric = self._registry.gauge(
            "bot_ui_performance_incident_active",
            "Czy incydent metryki wydajności UI jest aktywny (0/1)",
        )
        self._performance_incident_age_seconds: GaugeMetric = self._registry.gauge(
            "bot_ui_performance_incident_age_seconds",
            "Czas trwania bieżącego incydentu metryki wydajności UI",
        )
        self._performance_incident_started_at_seconds: GaugeMetric = (
            self._registry.gauge(
                "bot_ui_performance_incident_started_at_seconds",
                "Znacznik czasu rozpoczęcia incydentu metryki wydajności UI (epoch s)",
            )
        )
        self._performance_incident_duration_histogram: HistogramMetric = (
            self._registry.histogram(
                "bot_ui_performance_incident_duration_seconds",
                "Czas trwania incydentów metryk wydajności UI",
                buckets=(5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0),
            )
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
        self._overlay_capacity_ratio: GaugeMetric = self._registry.gauge(
            "bot_ui_overlay_capacity_ratio",
            "Udział wykorzystania budżetu nakładek (active/allowed)",
        )
        self._overlay_excess: GaugeMetric = self._registry.gauge(
            "bot_ui_overlay_excess",
            "Liczba nakładek ponad dozwolony budżet (0 gdy brak naruszenia)",
        )
        self._overlay_violation_state: GaugeMetric = self._registry.gauge(
            "bot_ui_overlay_violation_state",
            "Czy budżet nakładek został przekroczony (0/1)",
        )
        self._overlay_incidents_total: CounterMetric = self._registry.counter(
            "bot_ui_overlay_incidents_total",
            "Liczba incydentów przekroczenia budżetu nakładek",
        )
        self._overlay_severity_transitions_total: CounterMetric = self._registry.counter(
            "bot_ui_overlay_severity_transitions_total",
            "Liczba zmian stanu surowości incydentu budżetu nakładek",
        )
        self._overlay_violation_severity_state: GaugeMetric = self._registry.gauge(
            "bot_ui_overlay_violation_severity_state",
            "Stan surowości aktywnego incydentu budżetu nakładek (0/1)",
        )
        self._overlay_incident_events_total: CounterMetric = self._registry.counter(
            "bot_ui_overlay_incident_events_total",
            "Liczba przejść incydentu budżetu nakładek z podziałem na surowość",
        )
        self._overlay_capacity_ratio_overrun: HistogramMetric = (
            self._registry.histogram(
                "bot_ui_overlay_capacity_ratio_overrun",
                "Rozkład przekroczeń budżetu nakładek względem limitu (ratio-1)",
                buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
            )
        )
        self._overlay_incident_active: GaugeMetric = self._registry.gauge(
            "bot_ui_overlay_incident_active",
            "Czy incydent przekroczenia budżetu nakładek jest aktywny (0/1)",
        )
        self._overlay_incident_age_seconds: GaugeMetric = self._registry.gauge(
            "bot_ui_overlay_incident_age_seconds",
            "Czas trwania aktywnego incydentu przekroczenia budżetu nakładek",
        )
        self._overlay_incident_started_at_seconds: GaugeMetric = (
            self._registry.gauge(
                "bot_ui_overlay_incident_started_at_seconds",
                "Znacznik czasu (epoch s) rozpoczęcia incydentu budżetu nakładek",
            )
        )
        self._overlay_incident_duration_histogram: HistogramMetric = (
            self._registry.histogram(
                "bot_ui_overlay_incident_duration_seconds",
                "Czas trwania incydentów przekroczenia budżetu nakładek",
                buckets=(5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0),
            )
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
        self._jank_incident_active: GaugeMetric = self._registry.gauge(
            "bot_ui_jank_incident_active",
            "Czy incydent janku jest aktywny (0/1)",
        )
        self._jank_incident_age_seconds: GaugeMetric = self._registry.gauge(
            "bot_ui_jank_incident_age_seconds",
            "Czas trwania aktualnego incydentu janku",
        )
        self._jank_incident_started_at_seconds: GaugeMetric = self._registry.gauge(
            "bot_ui_jank_incident_started_at_seconds",
            "Znacznik czasu rozpoczęcia aktywnego incydentu janku (epoch s)",
        )
        self._jank_incident_duration_histogram: HistogramMetric = (
            self._registry.histogram(
                "bot_ui_jank_incident_duration_seconds",
                "Czas trwania incydentów janku",
                buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
            )
        )
        self._jank_incidents_total: CounterMetric = self._registry.counter(
            "bot_ui_jank_incidents_total",
            "Liczba incydentów janku wykrytych w telemetrii UI",
        )
        self._jank_histogram: HistogramMetric = self._registry.histogram(
            "bot_ui_jank_frame_overrun_ms",
            "Budżet janku dla UI (wartości powyżej progu)",
            buckets=(1.0, 5.0, 10.0, 25.0, 50.0, 100.0),
        )
        self._jank_severity_state: GaugeMetric = self._registry.gauge(
            "bot_ui_jank_severity_state",
            "Stan surowości aktywnego incydentu janku (0/1)",
        )
        self._jank_severity_transitions_total: CounterMetric = (
            self._registry.counter(
                "bot_ui_jank_severity_transitions_total",
                "Liczba zmian stanu surowości incydentu janku",
            )
        )
        event_to_frame_warning = _normalize_threshold(
            performance_event_to_frame_warning_ms
        )
        event_to_frame_critical = _normalize_threshold(
            performance_event_to_frame_critical_ms
        )
        cpu_warning = _normalize_threshold(cpu_utilization_warning_percent)
        cpu_critical = _normalize_threshold(cpu_utilization_critical_percent)
        gpu_warning = _normalize_threshold(gpu_utilization_warning_percent)
        gpu_critical = _normalize_threshold(gpu_utilization_critical_percent)
        ram_warning = _normalize_threshold(ram_usage_warning_megabytes)
        ram_critical = _normalize_threshold(ram_usage_critical_megabytes)
        self._performance_configs: dict[str, dict[str, float | None | str]] = {}
        if event_to_frame_warning is not None or event_to_frame_critical is not None:
            self._performance_configs["event_to_frame_p95_ms"] = {
                "warning": event_to_frame_warning,
                "critical": event_to_frame_critical,
                "unit": "ms",
            }
        if cpu_warning is not None or cpu_critical is not None:
            self._performance_configs["cpu_utilization"] = {
                "warning": cpu_warning,
                "critical": cpu_critical,
                "unit": "%",
            }
        if gpu_warning is not None or gpu_critical is not None:
            self._performance_configs["gpu_utilization"] = {
                "warning": gpu_warning,
                "critical": gpu_critical,
                "unit": "%",
            }
        if ram_warning is not None or ram_critical is not None:
            self._performance_configs["ram_usage_megabytes"] = {
                "warning": ram_warning,
                "critical": ram_critical,
                "unit": "MB",
            }
        self._performance_last_severity: dict[str, str | None] = {}
        self._performance_last_severity_by_tag: dict[str, dict[str, str | None]] = {}
        self._performance_incident_started_epoch: dict[str, float | None] = {}
        self._performance_incident_started_epoch_by_tag: dict[str, dict[str, float]] = {}
        for metric_name in self._performance_configs:
            self._performance_last_severity[metric_name] = None
            self._performance_last_severity_by_tag[metric_name] = {}
            self._performance_incident_started_epoch[metric_name] = None
            self._performance_incident_started_epoch_by_tag[metric_name] = {}
        self._alert_sink = alert_sink
        self._last_reduce_motion_state: bool | None = None
        self._reduce_motion_state_by_tag: dict[str, bool] = {}
        self._reduce_motion_incident_started_epoch: float | None = None
        self._reduce_motion_incident_started_epoch_by_tag: dict[str, float] = {}
        self._overlay_incident_started_epoch: float | None = None
        self._overlay_incident_started_epoch_by_tag: dict[str, float] = {}
        self._last_overlay_violation_state: bool | None = None
        self._overlay_violation_state_by_tag: dict[str, bool] = {}
        if overlay_critical_difference_threshold is None:
            self._overlay_critical_difference_threshold: float | None = None
        else:
            threshold = float(overlay_critical_difference_threshold)
            self._overlay_critical_difference_threshold = (
                threshold if threshold > 0 else None
            )
        if overlay_critical_duration_threshold_seconds is None:
            self._overlay_critical_duration_threshold_seconds: float | None = None
        else:
            duration_threshold = float(overlay_critical_duration_threshold_seconds)
            self._overlay_critical_duration_threshold_seconds = (
                duration_threshold if duration_threshold > 0 else None
            )
        self._last_overlay_severity: str | None = None
        self._overlay_last_severity_by_tag: dict[str, str | None] = {}
        self._retry_incident_started_epoch: float | None = None
        self._last_snapshot_generated_epoch: float | None = None
        self._last_snapshot_generated_epoch_by_tag: dict[str, float] = {}
        self._tag_last_seen_monotonic: dict[str, float] = {}
        self._tag_last_seen_epoch: dict[str, float] = {}
        self._tag_is_active: dict[str, bool] = {}
        self._tag_activity_ttl_seconds = max(0.0, float(tag_activity_ttl_seconds))
        self._jank_incident_started_epoch: float | None = None
        self._jank_incident_last_seen_epoch: float | None = None
        self._jank_incident_started_epoch_by_tag: dict[str, float] = {}
        self._jank_incident_last_seen_epoch_by_tag: dict[str, float] = {}
        self._jank_incident_quiet_seconds = max(0.0, float(jank_incident_quiet_seconds))
        self._jank_critical_over_ms = (
            float(jank_critical_over_ms)
            if jank_critical_over_ms is not None
            else None
        )
        self._jank_last_severity: str | None = None
        self._jank_last_severity_by_tag: dict[str, str] = {}

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

        event_to_frame_value: float | None = None
        event_to_frame = getattr(snapshot, "event_to_frame_p95_ms", None)
        if isinstance(event_to_frame, (int, float)):
            event_to_frame_value = max(0.0, float(event_to_frame))
            self._event_to_frame_p95_ms.set(event_to_frame_value)
            self._event_to_frame_histogram.observe(event_to_frame_value)
            if tag_labels:
                self._event_to_frame_p95_ms.set(event_to_frame_value, labels=tag_labels)
                self._event_to_frame_histogram.observe(
                    event_to_frame_value, labels=tag_labels
                )

        cpu_value: float | None = None
        cpu_utilization = getattr(snapshot, "cpu_utilization", None)
        if isinstance(cpu_utilization, (int, float)):
            cpu_value = max(0.0, float(cpu_utilization))
            self._cpu_utilization_percent.set(cpu_value)
            if tag_labels:
                self._cpu_utilization_percent.set(cpu_value, labels=tag_labels)

        gpu_value: float | None = None
        gpu_utilization = getattr(snapshot, "gpu_utilization", None)
        if isinstance(gpu_utilization, (int, float)):
            gpu_value = max(0.0, float(gpu_utilization))
            self._gpu_utilization_percent.set(gpu_value)
            if tag_labels:
                self._gpu_utilization_percent.set(gpu_value, labels=tag_labels)

        ram_value: float | None = None
        ram_usage = getattr(snapshot, "ram_megabytes", None)
        if isinstance(ram_usage, (int, float)):
            ram_value = max(0.0, float(ram_usage))
            self._ram_usage_megabytes.set(ram_value)
            if tag_labels:
                self._ram_usage_megabytes.set(ram_value, labels=tag_labels)

        dropped_frames = getattr(snapshot, "dropped_frames", None)
        if isinstance(dropped_frames, (int, float)):
            dropped_value = max(0.0, float(dropped_frames))
            self._dropped_frames_total.set(dropped_value)
            if tag_labels:
                self._dropped_frames_total.set(dropped_value, labels=tag_labels)

        processed_messages = getattr(snapshot, "processed_messages_per_second", None)
        if isinstance(processed_messages, (int, float)):
            processed_value = max(0.0, float(processed_messages))
            self._processed_messages_per_second.set(processed_value)
            if tag_labels:
                self._processed_messages_per_second.set(
                    processed_value, labels=tag_labels
                )

        if event == "performance" and self._performance_configs:
            performance_values = {
                "event_to_frame_p95_ms": event_to_frame_value,
                "cpu_utilization": cpu_value,
                "gpu_utilization": gpu_value,
                "ram_usage_megabytes": ram_value,
            }
            now_epoch = (
                timestamp_seconds if timestamp_seconds is not None else time.time()
            )
            self._handle_performance_incident_states(performance_values, now_epoch)
            if tag_value:
                self._handle_performance_incident_states(
                    performance_values, now_epoch, tag=tag_value
                )

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
            active_raw = payload.get("active_overlays")
            allowed_raw = payload.get("allowed_overlays")
            active_value: float | None = None
            allowed_value: float | None = None
            if isinstance(active_raw, (int, float)):
                active_value = float(active_raw)
                self._overlay_active.set(active_value)
                if tag_labels:
                    self._overlay_active.set(active_value, labels=tag_labels)
            if isinstance(allowed_raw, (int, float)):
                allowed_value = float(allowed_raw)
                self._overlay_allowed.set(allowed_value)
                if tag_labels:
                    self._overlay_allowed.set(allowed_value, labels=tag_labels)
            if (
                active_value is not None
                and allowed_value is not None
                and allowed_value > 0
            ):
                ratio = active_value / allowed_value
                self._overlay_capacity_ratio.set(ratio)
                if tag_labels:
                    self._overlay_capacity_ratio.set(ratio, labels=tag_labels)
                if ratio > 1.0:
                    overrun_ratio = ratio - 1.0
                    self._overlay_capacity_ratio_overrun.observe(overrun_ratio)
                    if tag_labels:
                        self._overlay_capacity_ratio_overrun.observe(
                            overrun_ratio, labels=tag_labels
                        )
            excess_value = 0.0
            violation_active = False
            if active_value is not None and allowed_value is not None:
                excess_value = max(0.0, active_value - allowed_value)
                violation_active = excess_value > 0.0
            now_epoch = (
                timestamp_seconds if timestamp_seconds is not None else time.time()
            )
            severity: str | None = None
            severity_reason: str | None = None
            if violation_active:
                severity = "warning"
                severity_reason = "violation"
                difference = active_value - allowed_value if active_value is not None and allowed_value is not None else 0.0
                if (
                    self._overlay_critical_difference_threshold is not None
                    and difference >= self._overlay_critical_difference_threshold
                ):
                    severity = "critical"
                    severity_reason = "difference_threshold"
                elif self._overlay_critical_duration_threshold_seconds is not None:
                    started = self._overlay_incident_started_epoch
                    if started is not None:
                        age = max(0.0, now_epoch - started)
                    else:
                        age = 0.0
                    if age >= self._overlay_critical_duration_threshold_seconds:
                        severity = "critical"
                        severity_reason = "duration_threshold"
            self._overlay_excess.set(excess_value)
            if tag_labels:
                self._overlay_excess.set(excess_value, labels=tag_labels)

            violation_value = 1.0 if violation_active else 0.0
            self._overlay_violation_state.set(violation_value)
            if violation_active and not self._last_overlay_violation_state:
                self._overlay_incidents_total.inc()
            self._last_overlay_violation_state = violation_active

            if severity is not None:
                self._record_overlay_severity_transition(
                    severity, reason=severity_reason
                )
            else:
                self._record_overlay_severity_transition(
                    None, reason="recovered"
                )

            if tag_labels:
                self._overlay_violation_state.set(violation_value, labels=tag_labels)
                previous_tag_violation = self._overlay_violation_state_by_tag.get(tag_value)
                if violation_active and not previous_tag_violation:
                    self._overlay_incidents_total.inc(labels=tag_labels)
                self._overlay_violation_state_by_tag[tag_value] = violation_active
                if severity is not None:
                    self._record_overlay_severity_transition(
                        severity, tag=tag_value, reason=severity_reason
                    )
                else:
                    self._record_overlay_severity_transition(
                        None, tag=tag_value, reason="recovered"
                    )
            self._update_overlay_incident_metrics(active_value, allowed_value, now_epoch)
            if tag_value:
                self._update_overlay_incident_metrics_for_tag(
                    tag_value, active_value, allowed_value, now_epoch
                )

        elif event == "jank_spike":
            frame_ms = payload.get("frame_ms")
            threshold = payload.get("threshold_ms")
            overrun: float | None = None
            if isinstance(frame_ms, (int, float)) and isinstance(threshold, (int, float)):
                overrun = float(frame_ms) - float(threshold)
                if overrun > 0:
                    self._jank_histogram.observe(overrun)
                    if tag_labels:
                        self._jank_histogram.observe(overrun, labels=tag_labels)
            severity_payload = payload.get("severity")
            severity: str | None = None
            if isinstance(severity_payload, str):
                normalized = severity_payload.strip().lower()
                if normalized in {"warning", "critical"}:
                    severity = normalized
            if overrun is not None and overrun > 0:
                candidate = "warning"
                if (
                    self._jank_critical_over_ms is not None
                    and overrun >= self._jank_critical_over_ms
                ):
                    candidate = "critical"
                if severity is None:
                    severity = candidate
                elif severity != "critical" and candidate == "critical":
                    severity = "critical"
            if severity is not None:
                self._record_jank_severity_transition(severity, reason="spike")
                if tag_value:
                    self._record_jank_severity_transition(
                        severity, tag=tag_value, reason="spike"
                    )
            now_epoch = timestamp_seconds if timestamp_seconds is not None else time.time()
            self._update_jank_incident_metrics(now_epoch)
            if tag_value:
                self._update_jank_incident_metrics_for_tag(tag_value, now_epoch)

        if self._alert_sink is not None:
            try:
                self._alert_sink.handle_snapshot(snapshot)
            except Exception:  # pragma: no cover - alert sink should not break metrics
                LOGGER.exception("UiTelemetryAlertSink zgłosił wyjątek")

        self._update_tag_activity(now_monotonic)
        self._expire_stale_jank_incidents(
            timestamp_seconds if timestamp_seconds is not None else time.time()
        )

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

    def _get_performance_last_severity(
        self, metric: str, tag: str | None
    ) -> str | None:
        if tag is not None:
            return self._performance_last_severity_by_tag.get(metric, {}).get(tag)
        return self._performance_last_severity.get(metric)

    def _set_performance_last_severity(
        self, metric: str, tag: str | None, severity: str | None
    ) -> None:
        if tag is None:
            if severity is None:
                self._performance_last_severity.pop(metric, None)
            else:
                self._performance_last_severity[metric] = severity
            return

        tag_map = self._performance_last_severity_by_tag.setdefault(metric, {})
        if severity is None:
            tag_map.pop(tag, None)
            if not tag_map:
                self._performance_last_severity_by_tag.pop(metric, None)
        else:
            tag_map[tag] = severity

    def _record_performance_severity_transition(
        self,
        metric: str,
        severity: str | None,
        *,
        tag: str | None = None,
        reason: str | None = None,
    ) -> None:
        previous = self._get_performance_last_severity(metric, tag)
        labels = {"metric": metric}
        if tag is not None:
            labels["tag"] = tag

        if severity == previous:
            if severity is None:
                self._performance_metric_state.set(0.0, labels=labels)
            else:
                level = _PERFORMANCE_SEVERITY_LEVELS.get(severity, 0.0)
                self._performance_metric_state.set(level, labels=labels)
            return

        if previous is not None:
            self._performance_metric_state.set(0.0, labels=labels)

        if severity is None:
            self._performance_metric_state.set(0.0, labels=labels)
            self._set_performance_last_severity(metric, tag, None)
        else:
            level = _PERFORMANCE_SEVERITY_LEVELS.get(severity, 0.0)
            self._performance_metric_state.set(level, labels=labels)
            self._set_performance_last_severity(metric, tag, severity)

        transition_state = severity if severity is not None else "recovered"
        transition_reason = reason or ("recovered" if severity is None else "unspecified")
        transition_labels = {
            "metric": metric,
            "state": transition_state,
            "reason": transition_reason,
        }
        if tag is not None:
            transition_labels["tag"] = tag
        self._performance_severity_transitions_total.inc(labels=transition_labels)

    def _start_or_update_performance_incident(
        self,
        metric: str,
        now_epoch: float,
        *,
        tag: str | None = None,
    ) -> None:
        labels = {"metric": metric}
        if tag is not None:
            labels["tag"] = tag

        if tag is not None:
            tag_map = self._performance_incident_started_epoch_by_tag.setdefault(metric, {})
            started = tag_map.get(tag)
            if started is None or started > now_epoch:
                tag_map[tag] = now_epoch
                started_epoch = now_epoch
            else:
                started_epoch = float(started)
        else:
            started = self._performance_incident_started_epoch.get(metric)
            if started is None or started > now_epoch:
                self._performance_incident_started_epoch[metric] = now_epoch
                started_epoch = now_epoch
            else:
                started_epoch = float(started)

        age = max(0.0, now_epoch - started_epoch)
        self._performance_incident_active.set(1.0, labels=labels)
        self._performance_incident_age_seconds.set(age, labels=labels)
        self._performance_incident_started_at_seconds.set(started_epoch, labels=labels)

    def _finalize_performance_incident(
        self,
        metric: str,
        *,
        tag: str | None = None,
        end_epoch: float | None = None,
    ) -> None:
        labels = {"metric": metric}
        if tag is not None:
            labels["tag"] = tag

        now_epoch = end_epoch if end_epoch is not None else time.time()
        started_epoch: float | None
        if tag is not None:
            tag_map = self._performance_incident_started_epoch_by_tag.get(metric)
            if tag_map is None:
                started_epoch = None
            else:
                started_epoch = tag_map.pop(tag, None)
                if not tag_map:
                    self._performance_incident_started_epoch_by_tag.pop(metric, None)
        else:
            started_epoch = self._performance_incident_started_epoch.get(metric)
            self._performance_incident_started_epoch[metric] = None

        if started_epoch is not None:
            duration = max(0.0, now_epoch - started_epoch)
            self._performance_incident_duration_histogram.observe(duration, labels=labels)

        self._performance_incident_active.set(0.0, labels=labels)
        self._performance_incident_age_seconds.set(0.0, labels=labels)
        self._performance_incident_started_at_seconds.set(0.0, labels=labels)

    def _determine_performance_severity(
        self,
        value: float,
        config: Mapping[str, float | None | str],
    ) -> tuple[str | None, str | None]:
        critical_raw = config.get("critical")
        warning_raw = config.get("warning")
        critical = float(critical_raw) if isinstance(critical_raw, (int, float)) else None
        warning = float(warning_raw) if isinstance(warning_raw, (int, float)) else None

        if critical is not None and value >= critical:
            return "critical", "critical_threshold"
        if warning is not None and value >= warning:
            return "warning", "warning_threshold"
        return None, None

    def _handle_performance_incident_states(
        self,
        values: Mapping[str, float | None],
        now_epoch: float,
        *,
        tag: str | None = None,
    ) -> None:
        if not self._performance_configs:
            return

        for metric_name, config in self._performance_configs.items():
            value = values.get(metric_name)
            if value is None:
                continue

            severity, threshold_reason = self._determine_performance_severity(value, config)
            previous = self._get_performance_last_severity(metric_name, tag)

            if severity is None:
                if previous is None:
                    continue
                self._record_performance_severity_transition(
                    metric_name, None, tag=tag, reason="recovered"
                )
                self._finalize_performance_incident(
                    metric_name, tag=tag, end_epoch=now_epoch
                )
                continue

            if previous is None:
                reason = threshold_reason or "triggered"
            elif previous == severity:
                reason = None
            elif severity == "critical":
                reason = "promoted"
            else:
                reason = "demoted"

            self._record_performance_severity_transition(
                metric_name, severity, tag=tag, reason=reason
            )

            if previous is None:
                incident_labels = {"metric": metric_name, "severity": severity}
                if tag is not None:
                    incident_labels["tag"] = tag
                self._performance_incidents_total.inc(labels=incident_labels)

            self._start_or_update_performance_incident(
                metric_name, now_epoch, tag=tag
            )

    def _finalize_performance_incidents_for_tag(
        self,
        tag: str,
        *,
        reason: str,
        end_epoch: float | None = None,
    ) -> None:
        if not self._performance_configs:
            return

        for metric_name in list(self._performance_configs.keys()):
            previous = self._get_performance_last_severity(metric_name, tag)
            if previous is not None:
                self._record_performance_severity_transition(
                    metric_name, None, tag=tag, reason=reason
                )
            self._finalize_performance_incident(
                metric_name, tag=tag, end_epoch=end_epoch
            )

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
                    last_seen_epoch = self._tag_last_seen_epoch.get(tag)
                    self._finalize_reduce_motion_incident_for_tag(
                        tag, end_epoch=last_seen_epoch
                    )
                    self._finalize_overlay_incident_for_tag(
                        tag, end_epoch=last_seen_epoch, reason="expired"
                    )
                    self._finalize_jank_incident_for_tag(
                        tag, end_epoch=last_seen_epoch, reason="expired"
                    )
                    self._finalize_performance_incidents_for_tag(
                        tag, end_epoch=last_seen_epoch, reason="expired"
                    )
                    to_delete.append(tag)
                else:
                    self._tag_inactive.set(1.0, labels=labels)
                    self._tag_inactive_age_seconds.set(max(0.0, age), labels=labels)
                    inactive_tags += 1
                    self._tag_is_active[tag] = False
                    last_seen_epoch = self._tag_last_seen_epoch.get(tag)
                    self._finalize_reduce_motion_incident_for_tag(
                        tag, end_epoch=last_seen_epoch
                    )
                    self._finalize_overlay_incident_for_tag(
                        tag, end_epoch=last_seen_epoch, reason="inactive"
                    )
                    self._finalize_jank_incident_for_tag(
                        tag, end_epoch=last_seen_epoch, reason="inactive"
                    )
                    self._finalize_performance_incidents_for_tag(
                        tag, end_epoch=last_seen_epoch, reason="inactive"
                    )

        for tag in to_delete:
            labels = {"tag": tag}
            self._tag_active.set(0.0, labels=labels)
            self._tag_inactive.set(0.0, labels=labels)
            self._tag_inactive_age_seconds.set(0.0, labels=labels)
            self._tag_last_seen_seconds.set(0.0, labels=labels)
            self._tag_last_seen_monotonic.pop(tag, None)
            last_seen_epoch = self._tag_last_seen_epoch.pop(tag, None)
            self._tag_is_active.pop(tag, None)
            self._finalize_reduce_motion_incident_for_tag(tag)
            self._finalize_overlay_incident_for_tag(tag, reason="removed")
            self._finalize_jank_incident_for_tag(tag, reason="removed")
            self._finalize_performance_incidents_for_tag(
                tag, end_epoch=last_seen_epoch, reason="removed"
            )

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

    def _record_overlay_severity_transition(
        self,
        severity: str | None,
        *,
        tag: str | None = None,
        reason: str | None = None,
    ) -> None:
        if tag is not None:
            previous = self._overlay_last_severity_by_tag.get(tag)
        else:
            previous = self._last_overlay_severity

        if severity == previous:
            if severity is None:
                return
            labels = {"severity": severity}
            if tag is not None:
                labels["tag"] = tag
            self._overlay_violation_severity_state.set(1.0, labels=labels)
            return

        if previous is not None:
            labels = {"severity": previous}
            if tag is not None:
                labels["tag"] = tag
            self._overlay_violation_severity_state.set(0.0, labels=labels)

        if severity is None:
            if tag is not None:
                self._overlay_last_severity_by_tag.pop(tag, None)
            else:
                self._last_overlay_severity = None
        else:
            labels = {"severity": severity}
            if tag is not None:
                labels["tag"] = tag
            self._overlay_violation_severity_state.set(1.0, labels=labels)
            self._overlay_incident_events_total.inc(labels=labels)
            if tag is not None:
                self._overlay_last_severity_by_tag[tag] = severity
            else:
                self._last_overlay_severity = severity

        if severity is None:
            transition_state = "recovered"
            transition_reason = reason or "recovered"
        else:
            transition_state = severity
            transition_reason = reason or "unspecified"

        transition_labels = {"state": transition_state, "reason": transition_reason}
        if tag is not None:
            transition_labels["tag"] = tag
        self._overlay_severity_transitions_total.inc(labels=transition_labels)

    def _record_jank_severity_transition(
        self,
        severity: str | None,
        *,
        tag: str | None = None,
        reason: str | None = None,
    ) -> None:
        normalized = None
        if isinstance(severity, str):
            normalized_candidate = severity.strip().lower()
            if normalized_candidate:
                normalized = normalized_candidate
        severity = normalized

        if tag is not None:
            previous = self._jank_last_severity_by_tag.get(tag)
        else:
            previous = self._jank_last_severity

        if severity == previous:
            if severity is None:
                return
            labels = {"severity": severity}
            if tag is not None:
                labels["tag"] = tag
            self._jank_severity_state.set(1.0, labels=labels)
            return

        if previous is not None:
            labels = {"severity": previous}
            if tag is not None:
                labels["tag"] = tag
            self._jank_severity_state.set(0.0, labels=labels)

        if severity is None:
            if tag is not None:
                self._jank_last_severity_by_tag.pop(tag, None)
            else:
                self._jank_last_severity = None
        else:
            labels = {"severity": severity}
            if tag is not None:
                labels["tag"] = tag
            self._jank_severity_state.set(1.0, labels=labels)
            if tag is not None:
                self._jank_last_severity_by_tag[tag] = severity
            else:
                self._jank_last_severity = severity

        if severity is None:
            transition_state = "recovered"
            transition_reason = reason or "recovered"
        else:
            transition_state = severity
            transition_reason = reason or "unspecified"

        transition_labels = {"state": transition_state, "reason": transition_reason}
        if tag is not None:
            transition_labels["tag"] = tag
        self._jank_severity_transitions_total.inc(labels=transition_labels)

    def _update_overlay_incident_metrics(
        self,
        active: float | None,
        allowed: float | None,
        now_epoch: float,
    ) -> None:
        exceeded = (
            active is not None and allowed is not None and active > allowed
        )
        if exceeded:
            if (
                self._overlay_incident_started_epoch is None
                or self._overlay_incident_started_epoch > now_epoch
            ):
                self._overlay_incident_started_epoch = now_epoch
                self._overlay_incident_started_at_seconds.set(now_epoch)
            if self._overlay_incident_started_epoch is not None:
                age = max(0.0, now_epoch - self._overlay_incident_started_epoch)
            else:
                age = 0.0
            self._overlay_incident_active.set(1.0)
            self._overlay_incident_age_seconds.set(age)
            if self._overlay_incident_started_epoch is not None:
                self._overlay_incident_started_at_seconds.set(
                    self._overlay_incident_started_epoch
                )
            return

        if self._overlay_incident_started_epoch is not None:
            duration = max(0.0, now_epoch - self._overlay_incident_started_epoch)
            self._overlay_incident_duration_histogram.observe(duration)

        self._overlay_incident_active.set(0.0)
        self._overlay_incident_age_seconds.set(0.0)
        self._overlay_incident_started_at_seconds.set(0.0)
        self._overlay_incident_started_epoch = None
        self._overlay_violation_state.set(0.0)
        self._overlay_excess.set(0.0)
        self._last_overlay_violation_state = False
        self._record_overlay_severity_transition(None, reason="recovered")

    def _update_overlay_incident_metrics_for_tag(
        self,
        tag: str,
        active: float | None,
        allowed: float | None,
        now_epoch: float,
    ) -> None:
        labels = {"tag": tag}
        exceeded = (
            active is not None and allowed is not None and active > allowed
        )
        started = self._overlay_incident_started_epoch_by_tag.get(tag)

        if exceeded:
            if started is None or started > now_epoch:
                self._overlay_incident_started_epoch_by_tag[tag] = now_epoch
                started = now_epoch
                self._overlay_incident_started_at_seconds.set(now_epoch, labels=labels)
            age = max(0.0, now_epoch - started)
            self._overlay_incident_active.set(1.0, labels=labels)
            self._overlay_incident_age_seconds.set(age, labels=labels)
            self._overlay_incident_started_at_seconds.set(started, labels=labels)
            return

        if started is not None:
            duration = max(0.0, now_epoch - started)
            self._overlay_incident_duration_histogram.observe(
                duration, labels=labels
            )
        self._overlay_incident_active.set(0.0, labels=labels)
        self._overlay_incident_age_seconds.set(0.0, labels=labels)
        self._overlay_incident_started_at_seconds.set(0.0, labels=labels)
        self._overlay_incident_started_epoch_by_tag.pop(tag, None)
        self._overlay_violation_state.set(0.0, labels=labels)
        self._overlay_excess.set(0.0, labels=labels)
        self._overlay_violation_state_by_tag.pop(tag, None)
        self._record_overlay_severity_transition(
            None, tag=tag, reason="recovered"
        )

    def _finalize_overlay_incident_for_tag(
        self,
        tag: str,
        *,
        end_epoch: float | None = None,
        reason: str = "expired",
    ) -> None:
        if tag not in self._overlay_incident_started_epoch_by_tag:
            return

        labels = {"tag": tag}
        started = self._overlay_incident_started_epoch_by_tag.pop(tag, None)
        if started is not None:
            end = end_epoch if end_epoch is not None else time.time()
            if end < started:
                end = started
            duration = max(0.0, end - started)
            self._overlay_incident_duration_histogram.observe(duration, labels=labels)
        self._overlay_incident_active.set(0.0, labels=labels)
        self._overlay_incident_age_seconds.set(0.0, labels=labels)
        self._overlay_incident_started_at_seconds.set(0.0, labels=labels)
        self._overlay_violation_state.set(0.0, labels=labels)
        self._overlay_excess.set(0.0, labels=labels)
        self._overlay_violation_state_by_tag.pop(tag, None)
        self._record_overlay_severity_transition(
            None, tag=tag, reason=reason
        )

    def _update_jank_incident_metrics(self, now_epoch: float) -> None:
        if self._jank_incident_started_epoch is None or (
            self._jank_incident_started_epoch > now_epoch
        ):
            self._jank_incident_started_epoch = now_epoch
            self._jank_incident_started_at_seconds.set(now_epoch)
            self._jank_incidents_total.inc()
        self._jank_incident_last_seen_epoch = now_epoch
        if self._jank_incident_started_epoch is not None:
            age = max(0.0, now_epoch - self._jank_incident_started_epoch)
        else:
            age = 0.0
        self._jank_incident_active.set(1.0)
        self._jank_incident_age_seconds.set(age)
        if self._jank_incident_started_epoch is not None:
            self._jank_incident_started_at_seconds.set(
                self._jank_incident_started_epoch
            )

    def _update_jank_incident_metrics_for_tag(
        self, tag: str, now_epoch: float
    ) -> None:
        labels = {"tag": tag}
        started = self._jank_incident_started_epoch_by_tag.get(tag)
        if started is None or started > now_epoch:
            self._jank_incident_started_epoch_by_tag[tag] = now_epoch
            started = now_epoch
            self._jank_incident_started_at_seconds.set(now_epoch, labels=labels)
            self._jank_incidents_total.inc(labels=labels)
        self._jank_incident_last_seen_epoch_by_tag[tag] = now_epoch
        age = max(0.0, now_epoch - started)
        self._jank_incident_active.set(1.0, labels=labels)
        self._jank_incident_age_seconds.set(age, labels=labels)
        self._jank_incident_started_at_seconds.set(started, labels=labels)

    def _finalize_jank_incident(
        self,
        *,
        end_epoch: float | None = None,
        reason: str = "quiet",
    ) -> None:
        if self._jank_incident_started_epoch is None:
            self._record_jank_severity_transition(None, reason=reason)
            return
        end = end_epoch if end_epoch is not None else self._jank_incident_last_seen_epoch
        if end is None:
            end = time.time()
        started = self._jank_incident_started_epoch
        if started is None:
            self._record_jank_severity_transition(None, reason=reason)
            return
        if end < started:
            end = started
        duration = max(0.0, end - started)
        self._jank_incident_duration_histogram.observe(duration)
        self._jank_incident_active.set(0.0)
        self._jank_incident_age_seconds.set(0.0)
        self._jank_incident_started_at_seconds.set(0.0)
        self._jank_incident_started_epoch = None
        self._jank_incident_last_seen_epoch = None
        self._record_jank_severity_transition(None, reason=reason)

    def _finalize_jank_incident_for_tag(
        self,
        tag: str,
        *,
        end_epoch: float | None = None,
        reason: str = "quiet",
    ) -> None:
        if tag not in self._jank_incident_started_epoch_by_tag:
            self._jank_incident_active.set(0.0, labels={"tag": tag})
            self._jank_incident_age_seconds.set(0.0, labels={"tag": tag})
            self._jank_incident_started_at_seconds.set(0.0, labels={"tag": tag})
            self._jank_incident_last_seen_epoch_by_tag.pop(tag, None)
            self._record_jank_severity_transition(None, tag=tag, reason=reason)
            return

        labels = {"tag": tag}
        started = self._jank_incident_started_epoch_by_tag.pop(tag, None)
        last_seen = self._jank_incident_last_seen_epoch_by_tag.pop(tag, None)
        if started is not None:
            end = end_epoch if end_epoch is not None else last_seen
            if end is None:
                end = time.time()
            if end < started:
                end = started
            duration = max(0.0, end - started)
            self._jank_incident_duration_histogram.observe(duration, labels=labels)
        self._jank_incident_active.set(0.0, labels=labels)
        self._jank_incident_age_seconds.set(0.0, labels=labels)
        self._jank_incident_started_at_seconds.set(0.0, labels=labels)
        self._record_jank_severity_transition(None, tag=tag, reason=reason)

    def _expire_stale_jank_incidents(self, now_epoch: float) -> None:
        quiet = self._jank_incident_quiet_seconds
        if quiet == 0:
            return
        if (
            self._jank_incident_last_seen_epoch is not None
            and self._jank_incident_started_epoch is not None
            and now_epoch - self._jank_incident_last_seen_epoch >= quiet
        ):
            self._finalize_jank_incident(reason="quiet")

        for tag, last_seen in list(self._jank_incident_last_seen_epoch_by_tag.items()):
            if now_epoch - last_seen >= quiet:
                self._finalize_jank_incident_for_tag(
                    tag, end_epoch=last_seen, reason="quiet"
                )

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

