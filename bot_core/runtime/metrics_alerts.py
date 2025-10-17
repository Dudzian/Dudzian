"""Sinki telemetrii UI przekierowujące zdarzenia do routera alertów."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from bot_core.alerts import AlertMessage, DefaultAlertRouter

DEFAULT_UI_ALERTS_JSONL_PATH = Path("logs/ui_telemetry_alerts.jsonl")

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _MetricState:
    """Stan pojedynczej metryki wydajności do kontrolowania alertów."""

    severity: str | None = None
    started_wallclock: datetime | None = None
    started_iso: str | None = None
    last_threshold: float | None = None


def _screen_context(payload: Mapping[str, Any]) -> dict[str, str]:
    screen = payload.get("screen")
    if not isinstance(screen, Mapping):
        return {}

    context: dict[str, str] = {}

    name = screen.get("name")
    if isinstance(name, str) and name:
        context["screen_name"] = name

    index = screen.get("index")
    if isinstance(index, (int, float)):
        # JSON może zawierać indeks jako float – normalizujemy do int/str
        index_int = int(index)
        if index_int >= 0:
            context["screen_index"] = str(index_int)

    refresh = screen.get("refresh_hz")
    if isinstance(refresh, (int, float)) and refresh > 0:
        context["screen_refresh_hz"] = f"{float(refresh):.2f}"

    dpr = screen.get("device_pixel_ratio")
    if isinstance(dpr, (int, float)) and dpr > 0:
        context["screen_dpr"] = f"{float(dpr):.2f}"

    geometry = screen.get("geometry_px")
    if isinstance(geometry, Mapping):
        width = geometry.get("width")
        height = geometry.get("height")
        if isinstance(width, (int, float)) and isinstance(height, (int, float)):
            context["screen_resolution"] = f"{int(width)}x{int(height)}"

    return context


def _screen_summary(payload: Mapping[str, Any]) -> str:
    screen = payload.get("screen")
    if not isinstance(screen, Mapping):
        return ""

    parts: list[str] = []

    index = screen.get("index")
    name = screen.get("name")
    if isinstance(index, (int, float)) and int(index) >= 0:
        index_part = f"#{int(index)}"
        if isinstance(name, str) and name:
            parts.append(f"{index_part} ({name})")
        else:
            parts.append(index_part)
    elif isinstance(name, str) and name:
        parts.append(name)

    geometry = screen.get("geometry_px")
    if isinstance(geometry, Mapping):
        width = geometry.get("width")
        height = geometry.get("height")
        if isinstance(width, (int, float)) and isinstance(height, (int, float)):
            parts.append(f"{int(width)}x{int(height)} px")

    refresh = screen.get("refresh_hz")
    if isinstance(refresh, (int, float)) and refresh > 0:
        parts.append(f"{float(refresh):.0f} Hz")

    if not parts:
        return ""

    return ", ".join(parts)


def _extract_tag(payload: Mapping[str, Any]) -> str | None:
    tag = payload.get("tag")
    if isinstance(tag, str):
        normalized = tag.strip()
        if normalized:
            return normalized
    return None


def _timestamp_to_iso(snapshot: Any) -> str | None:
    ts = getattr(snapshot, "generated_at", None)
    if ts is None or not hasattr(ts, "seconds"):
        return None
    seconds = getattr(ts, "seconds", 0)
    nanos = getattr(ts, "nanos", 0)
    if seconds == 0 and nanos == 0:
        return None
    dt = datetime.fromtimestamp(seconds + nanos / 1_000_000_000, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _datetime_to_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_risk_profile(metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Ujednolica metadane profilu ryzyka do struktury serializowalnej JSON."""
    if metadata is None:
        return None

    def _normalize(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): _normalize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_normalize(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    return {str(key): _normalize(val) for key, val in metadata.items()}


class UiTelemetryAlertSink:
    """Analizuje snapshoty UI i wysyła alerty o spadku FPS / nakładkach / janku."""

    def __init__(
        self,
        router: DefaultAlertRouter,
        *,
        jsonl_path: str | Path | None = None,
        enable_reduce_motion_alerts: bool = True,
        enable_reduce_motion_incident_alerts: bool = True,
        enable_overlay_alerts: bool = True,
        enable_jank_alerts: bool = True,
        enable_retry_backlog_alerts: bool = True,
        enable_tag_inactivity_alerts: bool = True,
        enable_performance_alerts: bool = True,
        log_reduce_motion_events: bool = True,
        log_reduce_motion_incident_events: bool = True,
        log_overlay_events: bool = True,
        log_jank_events: bool = True,
        log_retry_backlog_events: bool = True,
        log_tag_inactivity_events: bool = True,
        log_performance_events: bool = True,
        reduce_motion_category: str = "ui.performance",
        reduce_motion_severity_active: str = "warning",
        reduce_motion_severity_recovered: str = "info",
        reduce_motion_incident_category: str = "ui.performance",
        reduce_motion_incident_severity_degraded: str = "warning",
        reduce_motion_incident_severity_recovered: str = "info",
        reduce_motion_incident_severity_critical: str = "critical",
        reduce_motion_incident_threshold_seconds: float | int = 30.0,
        reduce_motion_incident_realert_interval_seconds: float | int | None = 120.0,
        reduce_motion_incident_realert_cooldown_seconds: float | int | None = None,
        reduce_motion_incident_critical_threshold_seconds: float | int | None = None,
        overlay_category: str = "ui.performance",
        overlay_severity_exceeded: str = "warning",
        overlay_severity_recovered: str = "info",
        overlay_critical_threshold: int | None = 2,
        overlay_severity_critical: str = "critical",
        overlay_incident_realert_delta: int = 1,
        overlay_incident_realert_cooldown_seconds: float | int | None = None,
        overlay_incident_critical_after_seconds: float | int | None = None,
        jank_category: str = "ui.performance",
        jank_severity_spike: str = "warning",
        jank_severity_critical: str | None = None,
        jank_critical_over_ms: float | None = None,
        retry_backlog_category: str = "ui.performance",
        retry_backlog_severity_degraded: str = "warning",
        retry_backlog_severity_recovered: str = "info",
        retry_backlog_severity_critical: str = "critical",
        retry_backlog_threshold: int = 5,
        retry_backlog_realert_delta: int = 5,
        retry_backlog_critical_threshold: int | None = None,
        retry_backlog_realert_cooldown_seconds: float | int | None = None,
        retry_backlog_critical_after_seconds: float | int | None = None,
        tag_inactivity_category: str = "ui.availability",
        tag_inactivity_severity_inactive: str = "warning",
        tag_inactivity_severity_recovered: str = "info",
        tag_inactivity_threshold_seconds: float | int = 300.0,
        performance_category: str = "ui.performance",
        performance_severity_warning: str = "warning",
        performance_severity_critical: str = "critical",
        performance_severity_recovered: str = "info",
        performance_event_to_frame_warning_ms: float | int | None = 45.0,
        performance_event_to_frame_critical_ms: float | int | None = 60.0,
        cpu_utilization_warning_percent: float | int | None = 85.0,
        cpu_utilization_critical_percent: float | int | None = 95.0,
        gpu_utilization_warning_percent: float | int | None = None,
        gpu_utilization_critical_percent: float | int | None = None,
        ram_usage_warning_megabytes: float | int | None = None,
        ram_usage_critical_megabytes: float | int | None = None,
        risk_profile: Mapping[str, Any] | None = None,
    ) -> None:
        self._router = router
        self._jsonl_path = Path(jsonl_path) if jsonl_path else DEFAULT_UI_ALERTS_JSONL_PATH
        self._enable_reduce_motion_alerts = enable_reduce_motion_alerts
        self._enable_reduce_motion_incident_alerts = enable_reduce_motion_incident_alerts
        self._enable_overlay_alerts = enable_overlay_alerts
        self._enable_jank_alerts = enable_jank_alerts
        self._enable_retry_backlog_alerts = enable_retry_backlog_alerts
        self._enable_tag_inactivity_alerts = enable_tag_inactivity_alerts
        self._enable_performance_alerts = enable_performance_alerts
        self._log_reduce_motion_events = log_reduce_motion_events
        self._log_reduce_motion_incident_events = log_reduce_motion_incident_events
        self._log_overlay_events = log_overlay_events
        self._log_jank_events = log_jank_events
        self._log_retry_backlog_events = log_retry_backlog_events
        self._log_tag_inactivity_events = log_tag_inactivity_events
        self._log_performance_events = log_performance_events
        self._reduce_motion_category = self._category_with_suffix(
            reduce_motion_category, "reduce_motion"
        )
        self._reduce_motion_severity_active = reduce_motion_severity_active
        self._reduce_motion_severity_recovered = reduce_motion_severity_recovered
        self._reduce_motion_incident_category = self._category_with_suffix(
            reduce_motion_incident_category, "reduce_motion_incident"
        )
        self._reduce_motion_incident_severity_degraded = reduce_motion_incident_severity_degraded
        self._reduce_motion_incident_severity_recovered = reduce_motion_incident_severity_recovered
        self._reduce_motion_incident_severity_critical = reduce_motion_incident_severity_critical
        self._reduce_motion_incident_threshold_seconds = max(
            0.0, float(reduce_motion_incident_threshold_seconds)
        )
        if reduce_motion_incident_realert_interval_seconds is None:
            self._reduce_motion_incident_realert_interval_seconds: float | None = None
        else:
            interval = float(reduce_motion_incident_realert_interval_seconds)
            self._reduce_motion_incident_realert_interval_seconds = max(1.0, interval)
        if reduce_motion_incident_realert_cooldown_seconds is None:
            self._reduce_motion_incident_realert_cooldown_seconds: float | None = None
        else:
            cooldown = float(reduce_motion_incident_realert_cooldown_seconds)
            self._reduce_motion_incident_realert_cooldown_seconds = max(0.0, cooldown)
        if reduce_motion_incident_critical_threshold_seconds is None:
            self._reduce_motion_incident_critical_threshold_seconds: float | None = None
        else:
            configured = float(reduce_motion_incident_critical_threshold_seconds)
            critical_threshold = max(
                self._reduce_motion_incident_threshold_seconds, configured
            )
            self._reduce_motion_incident_critical_threshold_seconds = max(0.0, critical_threshold)
        self._overlay_category = self._category_with_suffix(overlay_category, "overlay_budget")
        self._overlay_severity_exceeded = overlay_severity_exceeded
        self._overlay_severity_recovered = overlay_severity_recovered
        self._overlay_critical_threshold = overlay_critical_threshold
        self._overlay_severity_critical = overlay_severity_critical
        self._overlay_incident_realert_delta = max(1, int(overlay_incident_realert_delta))
        if overlay_incident_realert_cooldown_seconds is None:
            self._overlay_incident_realert_cooldown_seconds: float | None = None
        else:
            cooldown = float(overlay_incident_realert_cooldown_seconds)
            self._overlay_incident_realert_cooldown_seconds = max(0.0, cooldown)
        if overlay_incident_critical_after_seconds is None:
            self._overlay_incident_critical_after_seconds: float | None = None
        else:
            duration_threshold = float(overlay_incident_critical_after_seconds)
            self._overlay_incident_critical_after_seconds = max(0.0, duration_threshold)
        self._jank_category = self._category_with_suffix(jank_category, "jank")
        self._jank_severity_spike = jank_severity_spike
        self._jank_severity_critical = jank_severity_critical
        self._jank_critical_over_ms = jank_critical_over_ms
        self._retry_backlog_category = self._category_with_suffix(
            retry_backlog_category, "retry_backlog"
        )
        self._retry_backlog_severity_degraded = retry_backlog_severity_degraded
        self._retry_backlog_severity_recovered = retry_backlog_severity_recovered
        self._retry_backlog_severity_critical = retry_backlog_severity_critical
        self._retry_backlog_threshold = max(0, int(retry_backlog_threshold))
        self._retry_backlog_realert_delta = max(1, int(retry_backlog_realert_delta))
        if retry_backlog_critical_threshold is None:
            self._retry_backlog_critical_threshold: int | None = None
        else:
            self._retry_backlog_critical_threshold = max(
                self._retry_backlog_threshold, int(retry_backlog_critical_threshold)
            )
        if retry_backlog_realert_cooldown_seconds is None:
            self._retry_backlog_realert_cooldown_seconds: float | None = None
        else:
            cooldown = float(retry_backlog_realert_cooldown_seconds)
            self._retry_backlog_realert_cooldown_seconds = max(0.0, cooldown)
        if retry_backlog_critical_after_seconds is None:
            self._retry_backlog_critical_after_seconds: float | None = None
        else:
            duration = float(retry_backlog_critical_after_seconds)
            self._retry_backlog_critical_after_seconds = max(0.0, duration)
        self._tag_inactivity_category = self._category_with_suffix(
            tag_inactivity_category, "tag_inactivity"
        )
        self._tag_inactivity_severity_inactive = tag_inactivity_severity_inactive
        self._tag_inactivity_severity_recovered = tag_inactivity_severity_recovered
        self._tag_inactivity_threshold_seconds = max(0.0, float(tag_inactivity_threshold_seconds))
        self._performance_category = self._category_with_suffix(
            performance_category, "performance_metric"
        )
        self._performance_severity_warning = performance_severity_warning
        self._performance_severity_critical = performance_severity_critical
        self._performance_severity_recovered = performance_severity_recovered

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

        self._performance_event_to_frame_warning_ms = _normalize_threshold(
            performance_event_to_frame_warning_ms
        )
        self._performance_event_to_frame_critical_ms = _normalize_threshold(
            performance_event_to_frame_critical_ms
        )
        self._cpu_utilization_warning_percent = _normalize_threshold(
            cpu_utilization_warning_percent
        )
        self._cpu_utilization_critical_percent = _normalize_threshold(
            cpu_utilization_critical_percent
        )
        self._gpu_utilization_warning_percent = _normalize_threshold(
            gpu_utilization_warning_percent
        )
        self._gpu_utilization_critical_percent = _normalize_threshold(
            gpu_utilization_critical_percent
        )
        self._ram_usage_warning_megabytes = _normalize_threshold(
            ram_usage_warning_megabytes
        )
        self._ram_usage_critical_megabytes = _normalize_threshold(
            ram_usage_critical_megabytes
        )
        self._lock = Lock()
        self._last_reduce_motion_state: bool | None = None
        self._reduce_motion_current_state: bool | None = None
        self._reduce_motion_incident_started_monotonic: float | None = None
        self._reduce_motion_incident_started_wallclock: datetime | None = None
        self._reduce_motion_incident_started_iso: str | None = None
        self._reduce_motion_incident_triggered: bool = False
        self._reduce_motion_incident_last_alert_monotonic: float | None = None
        self._reduce_motion_incident_last_alert_duration: float | None = None
        self._reduce_motion_incident_last_severity: str | None = None
        self._last_overlay_exceeded: bool | None = None
        self._overlay_incident_active: bool = False
        self._overlay_incident_started_monotonic: float | None = None
        self._overlay_incident_started_wallclock: datetime | None = None
        self._overlay_incident_started_iso: str | None = None
        self._overlay_incident_last_alert_monotonic: float | None = None
        self._overlay_incident_last_difference: int | None = None
        self._overlay_incident_last_severity: str | None = None
        self._last_jank_signature: tuple[int, int] | None = None
        self._retry_backlog_active: bool | None = None
        self._retry_backlog_last_alert_value: int | None = None
        self._retry_backlog_last_severity: str | None = None
        self._retry_backlog_last_alert_monotonic: float | None = None
        self._retry_backlog_first_trigger_monotonic: float | None = None
        self._retry_backlog_first_trigger_wallclock: datetime | None = None
        self._tag_inactivity_last_seen_monotonic: dict[str, float] = {}
        self._tag_inactivity_last_seen_wallclock: dict[str, datetime] = {}
        self._tag_inactivity_last_seen_iso: dict[str, str] = {}
        self._tag_inactivity_last_screen_context: dict[str, dict[str, str]] = {}
        self._tag_inactivity_last_screen_summary: dict[str, str] = {}
        self._tag_inactivity_active: dict[str, bool] = {}
        self._tag_inactivity_inactive_since_monotonic: dict[str, float] = {}
        self._tag_inactivity_inactive_since_wallclock: dict[str, datetime] = {}
        self._should_write_jsonl = bool(
            self._jsonl_path
            and (
                self._log_reduce_motion_events
                or self._log_overlay_events
                or self._log_jank_events
                or self._log_retry_backlog_events
                or self._log_tag_inactivity_events
                or self._log_performance_events
            )
        )
        if self._should_write_jsonl:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl_path.touch(exist_ok=True)

        # Metadane profilu ryzyka (opcjonalne)
        normalized_profile = _normalize_risk_profile(risk_profile)
        self._risk_profile_metadata = deepcopy(normalized_profile) if normalized_profile else None
        self._risk_profile_name: str | None = None
        self._risk_profile_origin: str | None = None
        if self._risk_profile_metadata is not None:
            name = self._risk_profile_metadata.get("name")
            if isinstance(name, str) and name.strip():
                self._risk_profile_name = name.strip()
            origin = self._risk_profile_metadata.get("origin")
            if isinstance(origin, str) and origin.strip():
                self._risk_profile_origin = origin.strip()

        self._performance_states: dict[str, dict[str, _MetricState]] = {}

    @property
    def jsonl_path(self) -> Path:
        return self._jsonl_path

    def handle_snapshot(self, snapshot) -> None:
        notes = getattr(snapshot, "notes", "")
        if not notes:
            return
        try:
            payload = json.loads(notes)
        except json.JSONDecodeError:
            _LOGGER.debug("Nieprawidłowy JSON w polu notes telemetrii: %s", notes)
            return

        event = payload.get("event")
        if event == "reduce_motion":
            self._update_reduce_motion_incident_state(
                snapshot, payload, bool(payload.get("active"))
            )
        else:
            reduce_motion_field = payload.get("reduce_motion")
            if isinstance(reduce_motion_field, bool):
                self._update_reduce_motion_incident_state(
                    snapshot, payload, reduce_motion_field
                )
        self._track_tag_inactivity(snapshot, payload)
        self._handle_retry_backlog(snapshot, payload)
        self._handle_performance_metrics(snapshot, payload)
        if event == "reduce_motion":
            self._handle_reduce_motion(snapshot, payload)
        elif event == "overlay_budget":
            self._handle_overlay_budget(snapshot, payload)
        elif event == "jank_spike":
            self._handle_jank_spike(snapshot, payload)

    def _handle_performance_metrics(self, snapshot, payload: Mapping[str, Any]) -> None:
        if not (self._enable_performance_alerts or self._log_performance_events):
            return

        metrics: list[dict[str, Any]] = []

        def _has_threshold(warning: float | None, critical: float | None) -> bool:
            return (warning is not None and warning > 0) or (
                critical is not None and critical > 0
            )

        event_to_frame_raw = getattr(snapshot, "event_to_frame_p95_ms", None)
        if (
            isinstance(event_to_frame_raw, (int, float))
            and _has_threshold(
                self._performance_event_to_frame_warning_ms,
                self._performance_event_to_frame_critical_ms,
            )
        ):
            event_to_frame_value = max(0.0, float(event_to_frame_raw))
            metrics.append(
                {
                    "metric": "event_to_frame_p95_ms",
                    "value": event_to_frame_value,
                    "warning": self._performance_event_to_frame_warning_ms,
                    "critical": self._performance_event_to_frame_critical_ms,
                    "unit": "ms",
                    "label": "Opóźnienie zdarzenie→klatka p95",
                    "warning_title": "Opóźnienie zdarzenie→klatka powyżej progu",
                    "critical_title": "Krytyczne opóźnienie zdarzenie→klatka",
                    "recovery_title": "Opóźnienie zdarzenie→klatka ustabilizowane",
                }
            )

        cpu_raw = getattr(snapshot, "cpu_utilization", None)
        if (
            isinstance(cpu_raw, (int, float))
            and _has_threshold(
                self._cpu_utilization_warning_percent,
                self._cpu_utilization_critical_percent,
            )
        ):
            cpu_value = max(0.0, float(cpu_raw))
            metrics.append(
                {
                    "metric": "cpu_utilization",
                    "value": cpu_value,
                    "warning": self._cpu_utilization_warning_percent,
                    "critical": self._cpu_utilization_critical_percent,
                    "unit": "%",
                    "label": "Zużycie CPU UI",
                    "warning_title": "Wysokie zużycie CPU UI",
                    "critical_title": "Krytyczne zużycie CPU UI",
                    "recovery_title": "Zużycie CPU UI w normie",
                }
            )

        gpu_raw = getattr(snapshot, "gpu_utilization", None)
        if (
            isinstance(gpu_raw, (int, float))
            and _has_threshold(
                self._gpu_utilization_warning_percent,
                self._gpu_utilization_critical_percent,
            )
        ):
            gpu_value = max(0.0, float(gpu_raw))
            metrics.append(
                {
                    "metric": "gpu_utilization",
                    "value": gpu_value,
                    "warning": self._gpu_utilization_warning_percent,
                    "critical": self._gpu_utilization_critical_percent,
                    "unit": "%",
                    "label": "Zużycie GPU UI",
                    "warning_title": "Wysokie zużycie GPU UI",
                    "critical_title": "Krytyczne zużycie GPU UI",
                    "recovery_title": "Zużycie GPU UI w normie",
                }
            )

        ram_raw = getattr(snapshot, "ram_megabytes", None)
        if (
            isinstance(ram_raw, (int, float))
            and _has_threshold(
                self._ram_usage_warning_megabytes,
                self._ram_usage_critical_megabytes,
            )
        ):
            ram_value = max(0.0, float(ram_raw))
            metrics.append(
                {
                    "metric": "ram_megabytes",
                    "value": ram_value,
                    "warning": self._ram_usage_warning_megabytes,
                    "critical": self._ram_usage_critical_megabytes,
                    "unit": "MB",
                    "label": "Zużycie RAM UI",
                    "warning_title": "Wysokie zużycie RAM UI",
                    "critical_title": "Krytyczne zużycie RAM UI",
                    "recovery_title": "Zużycie RAM UI w normie",
                }
            )

        if not metrics:
            return

        tag_value = _extract_tag(payload)
        state_key = tag_value or "__global__"

        for config in metrics:
            self._process_performance_metric(
                snapshot, payload, config, tag_value, state_key
            )

    def _track_tag_inactivity(self, snapshot, payload: Mapping[str, Any]) -> None:
        if not (self._enable_tag_inactivity_alerts or self._log_tag_inactivity_events):
            return

        threshold = self._tag_inactivity_threshold_seconds
        if threshold <= 0:
            return

        try:
            now_monotonic = float(time.monotonic())
        except Exception:  # pragma: no cover - awaryjny fallback przy nietypowym zegarze
            now_monotonic = float(time.time())
        now_wallclock = datetime.now(timezone.utc)

        tag_value = _extract_tag(payload)
        if tag_value:
            last_seen_iso = _timestamp_to_iso(snapshot)
            if last_seen_iso is None:
                last_seen_iso = _datetime_to_iso(now_wallclock)
            self._tag_inactivity_last_seen_monotonic[tag_value] = now_monotonic
            self._tag_inactivity_last_seen_wallclock[tag_value] = now_wallclock
            if last_seen_iso:
                self._tag_inactivity_last_seen_iso[tag_value] = last_seen_iso
            screen_context = _screen_context(payload)
            if screen_context:
                self._tag_inactivity_last_screen_context[tag_value] = dict(screen_context)
            else:
                self._tag_inactivity_last_screen_context.pop(tag_value, None)
            screen_summary = _screen_summary(payload)
            if screen_summary:
                self._tag_inactivity_last_screen_summary[tag_value] = screen_summary
            else:
                self._tag_inactivity_last_screen_summary.pop(tag_value, None)
            previous_state = self._tag_inactivity_active.get(tag_value)
            if previous_state is False:
                inactive_since_monotonic = self._tag_inactivity_inactive_since_monotonic.pop(
                    tag_value, None
                )
                inactive_since_wallclock = self._tag_inactivity_inactive_since_wallclock.pop(
                    tag_value, None
                )
                duration_seconds: float | None = None
                if inactive_since_monotonic is not None:
                    duration_seconds = max(0.0, now_monotonic - inactive_since_monotonic)
                self._emit_tag_inactivity_event(
                    tag=tag_value,
                    inactive=False,
                    severity=self._tag_inactivity_severity_recovered,
                    age_seconds=None,
                    duration_seconds=duration_seconds,
                    last_seen_iso=self._tag_inactivity_last_seen_iso.get(tag_value),
                    inactive_since_iso=_datetime_to_iso(inactive_since_wallclock),
                    screen_summary=self._tag_inactivity_last_screen_summary.get(tag_value),
                    screen_context=self._tag_inactivity_last_screen_context.get(tag_value),
                    snapshot=snapshot,
                )
            else:
                self._tag_inactivity_inactive_since_monotonic.pop(tag_value, None)
                self._tag_inactivity_inactive_since_wallclock.pop(tag_value, None)
            self._tag_inactivity_active[tag_value] = True

        self._evaluate_tag_inactivity(now_monotonic, now_wallclock)

    def _update_reduce_motion_incident_state(
        self, snapshot, payload: Mapping[str, Any], active: bool
    ) -> None:
        self._reduce_motion_current_state = active

        features_enabled = (
            self._enable_reduce_motion_incident_alerts
            or self._log_reduce_motion_incident_events
        )

        if not features_enabled:
            if not active:
                self._reduce_motion_incident_triggered = False
                self._reduce_motion_incident_started_monotonic = None
                self._reduce_motion_incident_started_wallclock = None
                self._reduce_motion_incident_started_iso = None
                self._reduce_motion_incident_last_alert_duration = None
                self._reduce_motion_incident_last_alert_monotonic = None
                self._reduce_motion_incident_last_severity = None
            elif self._reduce_motion_incident_started_monotonic is None:
                self._reduce_motion_incident_started_monotonic = time.monotonic()
                started_wallclock = datetime.now(timezone.utc)
                self._reduce_motion_incident_started_wallclock = started_wallclock
                started_iso = _timestamp_to_iso(snapshot)
                if started_iso is None:
                    started_iso = _datetime_to_iso(started_wallclock)
                self._reduce_motion_incident_started_iso = started_iso
                self._reduce_motion_incident_last_alert_duration = None
                self._reduce_motion_incident_last_alert_monotonic = None
                self._reduce_motion_incident_last_severity = None
            return

        monotonic_now: float | None = None

        def _monotonic() -> float:
            nonlocal monotonic_now
            if monotonic_now is None:
                monotonic_now = time.monotonic()
            return monotonic_now

        wallclock_now: datetime | None = None

        def _wallclock() -> datetime:
            nonlocal wallclock_now
            if wallclock_now is None:
                wallclock_now = datetime.now(timezone.utc)
            return wallclock_now

        if active and self._reduce_motion_incident_started_monotonic is None:
            self._reduce_motion_incident_started_monotonic = _monotonic()
            started_wallclock = _wallclock()
            self._reduce_motion_incident_started_wallclock = started_wallclock
            started_iso = _timestamp_to_iso(snapshot)
            if started_iso is None:
                started_iso = _datetime_to_iso(started_wallclock)
            self._reduce_motion_incident_started_iso = started_iso
            self._reduce_motion_incident_triggered = False
            self._reduce_motion_incident_last_alert_duration = None
            self._reduce_motion_incident_last_alert_monotonic = None
            self._reduce_motion_incident_last_severity = None

        threshold = self._reduce_motion_incident_threshold_seconds
        started_monotonic = self._reduce_motion_incident_started_monotonic
        duration_seconds: float | None = None
        if started_monotonic is not None:
            duration_seconds = max(0.0, _monotonic() - started_monotonic)

        if active:
            triggered = False
            if duration_seconds is not None:
                if threshold <= 0.0:
                    triggered = duration_seconds >= 0.0
                else:
                    triggered = duration_seconds >= threshold

            severity = self._reduce_motion_incident_severity_degraded
            escalation_reason: str | None = None
            if (
                triggered
                and self._reduce_motion_incident_critical_threshold_seconds is not None
                and duration_seconds is not None
                and duration_seconds >= self._reduce_motion_incident_critical_threshold_seconds
            ):
                severity = self._reduce_motion_incident_severity_critical
                escalation_reason = "critical_threshold"

            should_emit = False
            reason: str | None = None
            duration_delta: float | None = None
            if triggered:
                previous_triggered = self._reduce_motion_incident_triggered
                previous_severity = self._reduce_motion_incident_last_severity
                if not previous_triggered:
                    should_emit = True
                    reason = "threshold"
                elif previous_severity != severity:
                    should_emit = True
                    reason = "severity_change"
                else:
                    interval = self._reduce_motion_incident_realert_interval_seconds
                    if (
                        interval is not None
                        and self._reduce_motion_incident_last_alert_duration is not None
                        and duration_seconds is not None
                        and (
                            duration_seconds
                            - self._reduce_motion_incident_last_alert_duration
                        )
                        >= interval
                    ):
                        should_emit = True
                        reason = "realert_interval"
                        duration_delta = (
                            duration_seconds
                            - self._reduce_motion_incident_last_alert_duration
                        )
                    elif self._reduce_motion_incident_realert_cooldown_seconds is not None:
                        last_alert = self._reduce_motion_incident_last_alert_monotonic
                        if last_alert is None or (
                            _monotonic() - last_alert
                        ) >= self._reduce_motion_incident_realert_cooldown_seconds:
                            should_emit = True
                            reason = "realert_cooldown"
                            if (
                                duration_seconds is not None
                                and self._reduce_motion_incident_last_alert_duration
                                is not None
                            ):
                                duration_delta = (
                                    duration_seconds
                                    - self._reduce_motion_incident_last_alert_duration
                                )

            self._reduce_motion_incident_triggered = triggered
            if not triggered or not should_emit:
                return

            tag_value = _extract_tag(payload)
            fps_value = getattr(snapshot, "fps", None)
            context: dict[str, str] = {"active": "true"}
            if isinstance(fps_value, (int, float)):
                context["fps"] = f"{float(fps_value):.2f}"
            fps_target = payload.get("fps_target")
            if isinstance(fps_target, (int, float)):
                context["fps_target"] = str(fps_target)
            overlay_active = payload.get("overlay_active")
            if overlay_active is None:
                overlay_active = payload.get("active_overlays")
            if isinstance(overlay_active, (int, float)):
                context["overlay_active"] = str(int(overlay_active))
            overlay_allowed = payload.get("overlay_allowed")
            if overlay_allowed is None:
                overlay_allowed = payload.get("allowed_overlays")
            if isinstance(overlay_allowed, (int, float)):
                context["overlay_allowed"] = str(int(overlay_allowed))
            window_count = payload.get("window_count")
            if isinstance(window_count, (int, float)):
                context["window_count"] = str(int(window_count))
            context["reduce_motion_incident_threshold_seconds"] = f"{threshold:.3f}"
            if duration_seconds is not None:
                context["reduce_motion_incident_duration_seconds"] = (
                    f"{duration_seconds:.3f}"
                )
            if duration_delta is not None:
                context["reduce_motion_incident_duration_delta_seconds"] = (
                    f"{max(0.0, duration_delta):.3f}"
                )
            if tag_value:
                context["tag"] = tag_value
            if escalation_reason:
                context["reduce_motion_incident_escalation"] = escalation_reason
            if reason:
                context["reduce_motion_incident_reason"] = reason
            context["reduce_motion_incident_severity"] = severity
            started_iso = self._reduce_motion_incident_started_iso
            if started_iso:
                context["reduce_motion_incident_started_at"] = started_iso
            context.update(_screen_context(payload))
            context = self._context_with_risk_profile(context)
            screen_summary = _screen_summary(payload)

            log_payload = dict(payload)
            log_payload["event"] = "reduce_motion_incident"
            if duration_seconds is not None:
                log_payload["reduce_motion_incident_duration_seconds"] = duration_seconds
            log_payload["reduce_motion_incident_threshold_seconds"] = threshold
            if started_iso:
                log_payload["reduce_motion_incident_started_at"] = started_iso
            log_payload["reduce_motion_incident_severity"] = severity
            if reason:
                log_payload["reduce_motion_incident_reason"] = reason
            if escalation_reason:
                log_payload["reduce_motion_incident_escalation"] = escalation_reason
            if duration_delta is not None:
                log_payload["reduce_motion_incident_duration_delta_seconds"] = (
                    max(0.0, duration_delta)
                )

            if self._log_reduce_motion_incident_events:
                self._append_jsonl(
                    self._reduce_motion_incident_category,
                    severity,
                    log_payload,
                    snapshot,
                    context=context,
                )

            if self._enable_reduce_motion_incident_alerts:
                title = "Reduce motion utrzymuje się zbyt długo"
                if severity == self._reduce_motion_incident_severity_critical:
                    title = "Krytyczne wymuszenie reduce motion"
                body_parts = [
                    "Reduce motion aktywne od {:.1f} s (próg {:.1f} s).".format(
                        duration_seconds or 0.0, threshold
                    )
                ]
                if duration_delta is not None and duration_delta > 0:
                    body_parts.append(
                        "Dodatkowe {:.1f} s od poprzedniego alertu.".format(
                            duration_delta
                        )
                    )
                if escalation_reason == "critical_threshold":
                    body_parts.append(
                        "Eskalacja do poziomu krytycznego ze względu na długość incydentu."
                    )
                if screen_summary:
                    body_parts.append(f"Ekran: {screen_summary}.")

                message = AlertMessage(
                    category=self._reduce_motion_incident_category,
                    title=title,
                    body=" ".join(body_parts),
                    severity=severity,
                    context=context,
                )
                self._router.dispatch(message)

            self._reduce_motion_incident_last_alert_duration = duration_seconds
            self._reduce_motion_incident_last_alert_monotonic = _monotonic()
            self._reduce_motion_incident_last_severity = severity
            return

        started_iso = self._reduce_motion_incident_started_iso
        started_wallclock = self._reduce_motion_incident_started_wallclock
        duration_seconds = (
            max(0.0, _monotonic() - started_monotonic)
            if started_monotonic is not None
            else None
        )
        recovered_wallclock = _wallclock()
        recovered_iso = _timestamp_to_iso(snapshot)
        if recovered_iso is None:
            recovered_iso = _datetime_to_iso(recovered_wallclock)

        triggered_before = self._reduce_motion_incident_triggered
        self._reduce_motion_incident_triggered = False
        self._reduce_motion_incident_started_monotonic = None
        self._reduce_motion_incident_started_wallclock = None
        self._reduce_motion_incident_started_iso = None
        self._reduce_motion_incident_last_alert_duration = None
        self._reduce_motion_incident_last_alert_monotonic = None
        previous_severity = self._reduce_motion_incident_last_severity
        self._reduce_motion_incident_last_severity = None

        if not triggered_before or duration_seconds is None:
            return

        severity = self._reduce_motion_incident_severity_recovered
        tag_value = _extract_tag(payload)
        fps_value = getattr(snapshot, "fps", None)
        context = {"active": "false"}
        if isinstance(fps_value, (int, float)):
            context["fps"] = f"{float(fps_value):.2f}"
        fps_target = payload.get("fps_target")
        if isinstance(fps_target, (int, float)):
            context["fps_target"] = str(fps_target)
        overlay_active = payload.get("overlay_active")
        if overlay_active is None:
            overlay_active = payload.get("active_overlays")
        if isinstance(overlay_active, (int, float)):
            context["overlay_active"] = str(int(overlay_active))
        overlay_allowed = payload.get("overlay_allowed")
        if overlay_allowed is None:
            overlay_allowed = payload.get("allowed_overlays")
        if isinstance(overlay_allowed, (int, float)):
            context["overlay_allowed"] = str(int(overlay_allowed))
        window_count = payload.get("window_count")
        if isinstance(window_count, (int, float)):
            context["window_count"] = str(int(window_count))
        context["reduce_motion_incident_threshold_seconds"] = f"{threshold:.3f}"
        context["reduce_motion_incident_duration_seconds"] = (
            f"{duration_seconds:.3f}"
        )
        context["reduce_motion_incident_severity"] = severity
        if started_iso:
            context["reduce_motion_incident_started_at"] = started_iso
        if recovered_iso:
            context["reduce_motion_incident_recovered_at"] = recovered_iso
        if tag_value:
            context["tag"] = tag_value
        context.update(_screen_context(payload))
        context = self._context_with_risk_profile(context)
        screen_summary = _screen_summary(payload)

        log_payload = dict(payload)
        log_payload["event"] = "reduce_motion_incident_recovered"
        log_payload["reduce_motion_incident_duration_seconds"] = duration_seconds
        log_payload["reduce_motion_incident_threshold_seconds"] = threshold
        if started_iso:
            log_payload["reduce_motion_incident_started_at"] = started_iso
        if recovered_iso:
            log_payload["reduce_motion_incident_recovered_at"] = recovered_iso
        log_payload["reduce_motion_incident_previous_severity"] = previous_severity or ""

        if self._log_reduce_motion_incident_events:
            self._append_jsonl(
                self._reduce_motion_incident_category,
                severity,
                log_payload,
                snapshot,
                context=context,
            )

        if not self._enable_reduce_motion_incident_alerts:
            return

        body = (
            "Reduce motion wyłączony po {:.1f} s (próg {:.1f} s).".format(
                duration_seconds, threshold
            )
        )
        if screen_summary:
            body += f" Ekran: {screen_summary}."

        message = AlertMessage(
            category=self._reduce_motion_incident_category,
            title="Incydent reduce motion zakończony",
            body=body,
            severity=severity,
            context=context,
        )
        self._router.dispatch(message)

    def _handle_retry_backlog(self, snapshot, payload: Mapping[str, Any]) -> None:
        backlog_after_raw = payload.get("retry_backlog_after_flush")
        backlog_before_raw = payload.get("retry_backlog_before_send")
        try:
            backlog_after = int(backlog_after_raw)
        except (TypeError, ValueError):
            backlog_after = None
        try:
            backlog_before = int(backlog_before_raw)
        except (TypeError, ValueError):
            backlog_before = None

        if backlog_after is None and backlog_before is None:
            return

        backlog_after = max(0, backlog_after or 0)
        backlog_before = max(0, backlog_before or 0)

        threshold = self._retry_backlog_threshold
        triggered = backlog_after > 0 if threshold == 0 else backlog_after >= threshold

        monotonic_now: float | None = None

        def _monotonic() -> float:
            nonlocal monotonic_now
            if monotonic_now is None:
                monotonic_now = time.monotonic()
            return monotonic_now

        previous = self._retry_backlog_active
        if previous is None:
            self._retry_backlog_active = triggered
            if not triggered:
                self._retry_backlog_last_alert_value = None
                self._retry_backlog_last_severity = None
                self._retry_backlog_first_trigger_monotonic = None
                self._retry_backlog_first_trigger_wallclock = None
                return
            state_changed = True
        else:
            state_changed = previous != triggered
            if state_changed:
                self._retry_backlog_active = triggered
            elif not triggered:
                return

        should_emit = state_changed
        backlog_delta: int | None = None
        started_wallclock: datetime | None = self._retry_backlog_first_trigger_wallclock
        recovered_wallclock: datetime | None = None
        if triggered:
            last_alert_value = self._retry_backlog_last_alert_value
            if last_alert_value is not None:
                backlog_delta = backlog_after - last_alert_value
            elif state_changed:
                backlog_delta = backlog_after
        else:
            if not should_emit:
                self._retry_backlog_last_alert_value = None
                self._retry_backlog_last_severity = None
                self._retry_backlog_first_trigger_monotonic = None
                self._retry_backlog_first_trigger_wallclock = None
                return

        duration_seconds: float | None = None
        if triggered:
            if state_changed or self._retry_backlog_first_trigger_monotonic is None:
                self._retry_backlog_first_trigger_monotonic = _monotonic()
                self._retry_backlog_first_trigger_wallclock = datetime.now(timezone.utc)
                started_wallclock = self._retry_backlog_first_trigger_wallclock
                duration_seconds = 0.0
            else:
                duration_seconds = max(
                    0.0,
                    _monotonic() - self._retry_backlog_first_trigger_monotonic,
                )
                started_wallclock = self._retry_backlog_first_trigger_wallclock
        else:
            if self._retry_backlog_first_trigger_monotonic is not None:
                duration_seconds = max(
                    0.0,
                    _monotonic() - self._retry_backlog_first_trigger_monotonic,
                )
            started_wallclock = self._retry_backlog_first_trigger_wallclock
            if started_wallclock is not None:
                recovered_wallclock = datetime.now(timezone.utc)
            self._retry_backlog_first_trigger_monotonic = None
            self._retry_backlog_first_trigger_wallclock = None

        severity = (
            self._retry_backlog_severity_degraded
            if triggered
            else self._retry_backlog_severity_recovered
        )
        escalation_reason: str | None = None
        if triggered and self._retry_backlog_critical_threshold is not None:
            if backlog_after >= self._retry_backlog_critical_threshold:
                severity = self._retry_backlog_severity_critical
                escalation_reason = "threshold"

        if (
            triggered
            and severity != self._retry_backlog_severity_critical
            and self._retry_backlog_critical_after_seconds is not None
        ):
            current_duration = duration_seconds
            if current_duration is None and self._retry_backlog_first_trigger_monotonic is not None:
                current_duration = max(
                    0.0,
                    _monotonic() - self._retry_backlog_first_trigger_monotonic,
                )
            if current_duration is None:
                current_duration = 0.0
            if current_duration >= self._retry_backlog_critical_after_seconds:
                severity = self._retry_backlog_severity_critical
                escalation_reason = "duration"
            duration_seconds = current_duration

        previous_severity = self._retry_backlog_last_severity
        severity_change = (
            triggered
            and previous_severity is not None
            and previous_severity != severity
        )
        if severity_change:
            should_emit = True

        if triggered:
            self._retry_backlog_last_severity = severity
        else:
            self._retry_backlog_last_severity = None

        if triggered and not should_emit:
            if backlog_delta is None:
                backlog_delta = backlog_after
            if not severity_change and backlog_delta < self._retry_backlog_realert_delta:
                return
            if (
                not severity_change
                and self._retry_backlog_realert_cooldown_seconds
                and self._retry_backlog_realert_cooldown_seconds > 0
            ):
                now = _monotonic()
                last = self._retry_backlog_last_alert_monotonic
                if last is not None and (
                    now - last
                    < self._retry_backlog_realert_cooldown_seconds
                ):
                    return
            should_emit = True

        if triggered:
            self._retry_backlog_last_alert_value = backlog_after
        else:
            self._retry_backlog_last_alert_value = None
            self._retry_backlog_last_alert_monotonic = None

        if not (
            self._log_retry_backlog_events or self._enable_retry_backlog_alerts
        ):
            return

        window_count = payload.get("window_count")
        context: dict[str, str] = {
            "retry_backlog_after": str(backlog_after),
            "retry_backlog_before": str(backlog_before),
            "retry_backlog_threshold": str(threshold),
            "retry_backlog_severity": severity,
        }
        if backlog_delta is not None:
            context["retry_backlog_delta"] = str(backlog_delta)
        if duration_seconds is not None:
            context["retry_backlog_duration_seconds"] = f"{duration_seconds:.3f}"
        if escalation_reason is not None:
            context["retry_backlog_escalation"] = escalation_reason
        tag_value = _extract_tag(payload)
        if tag_value:
            context["tag"] = tag_value
        started_iso = _datetime_to_iso(started_wallclock)
        if started_iso is not None:
            context["retry_backlog_started_at"] = started_iso
        recovered_iso = _datetime_to_iso(recovered_wallclock)
        if recovered_iso is not None:
            context["retry_backlog_recovered_at"] = recovered_iso
        if window_count is not None:
            context["window_count"] = str(window_count)
        context.update(_screen_context(payload))
        context = self._context_with_risk_profile(context)
        screen_summary = _screen_summary(payload)

        if self._log_retry_backlog_events:
            self._append_jsonl(
                self._retry_backlog_category,
                severity,
                payload,
                snapshot,
                context=context,
            )

        if not self._enable_retry_backlog_alerts:
            return

        duration_fragment = ""
        if duration_seconds is not None and (not triggered or duration_seconds > 0.0):
            duration_fragment = f" Czas degradacji: {duration_seconds:.1f} s."

        if triggered:
            body = (
                "Bufor retry telemetrii wynosi {} (próg {}). "
                "Poprzedni backlog: {}."
            ).format(backlog_after, threshold or ">0", backlog_before)
            if backlog_delta is not None:
                sign = "+" if backlog_delta >= 0 else ""
                body += f" Zmiana od ostatniego alertu: {sign}{backlog_delta}."
            if escalation_reason == "duration":
                body += " Eskalacja do poziomu krytycznego po przekroczeniu limitu czasu trwania."
            body += duration_fragment
            title = "Bufor retry telemetrii narasta"
        else:
            body = (
                "Bufor retry telemetrii został opróżniony ({} -> {})."
            ).format(backlog_before, backlog_after)
            body += duration_fragment
            title = "Bufor retry telemetrii przywrócony"

        if screen_summary:
            body += f" Ekran: {screen_summary}."

        message = AlertMessage(
            category=self._retry_backlog_category,
            title=title,
            body=body,
            severity=severity,
            context=context,
        )
        self._router.dispatch(message)
        if triggered:
            self._retry_backlog_last_alert_monotonic = _monotonic()

    def _evaluate_tag_inactivity(
        self, now_monotonic: float, now_wallclock: datetime
    ) -> None:
        if not (self._enable_tag_inactivity_alerts or self._log_tag_inactivity_events):
            return

        threshold = self._tag_inactivity_threshold_seconds
        if threshold <= 0:
            return

        for tag, last_seen in list(self._tag_inactivity_last_seen_monotonic.items()):
            age = now_monotonic - last_seen
            if age < threshold:
                continue

            if self._tag_inactivity_active.get(tag, True) is False:
                continue

            last_seen_wallclock = self._tag_inactivity_last_seen_wallclock.get(tag)
            inactive_since_monotonic = last_seen + threshold
            inactive_since_wallclock = (
                last_seen_wallclock + timedelta(seconds=threshold)
                if last_seen_wallclock is not None
                else None
            )
            self._tag_inactivity_active[tag] = False
            self._tag_inactivity_inactive_since_monotonic[tag] = inactive_since_monotonic
            if inactive_since_wallclock is not None:
                self._tag_inactivity_inactive_since_wallclock[tag] = inactive_since_wallclock
            else:
                self._tag_inactivity_inactive_since_wallclock.pop(tag, None)
            self._emit_tag_inactivity_event(
                tag=tag,
                inactive=True,
                severity=self._tag_inactivity_severity_inactive,
                age_seconds=max(0.0, age),
                duration_seconds=None,
                last_seen_iso=self._tag_inactivity_last_seen_iso.get(tag),
                inactive_since_iso=_datetime_to_iso(inactive_since_wallclock),
                screen_summary=self._tag_inactivity_last_screen_summary.get(tag),
                screen_context=self._tag_inactivity_last_screen_context.get(tag),
                snapshot=None,
            )

    def _emit_tag_inactivity_event(
        self,
        *,
        tag: str,
        inactive: bool,
        severity: str,
        age_seconds: float | None,
        duration_seconds: float | None,
        last_seen_iso: str | None,
        inactive_since_iso: str | None,
        screen_summary: str | None,
        screen_context: Mapping[str, Any] | None,
        snapshot,
    ) -> None:
        if not (self._enable_tag_inactivity_alerts or self._log_tag_inactivity_events):
            return

        def _fmt(value: float) -> str:
            return f"{value:.3f}".rstrip("0").rstrip(".")

        context: dict[str, Any] = {
            "tag": tag,
            "tag_inactive": str(inactive).lower(),
            "tag_inactivity_threshold_seconds": _fmt(self._tag_inactivity_threshold_seconds),
        }
        if age_seconds is not None:
            context["tag_inactivity_age_seconds"] = _fmt(max(0.0, age_seconds))
        if duration_seconds is not None:
            context["tag_inactivity_duration_seconds"] = _fmt(max(0.0, duration_seconds))
        if last_seen_iso:
            context["tag_last_seen_at"] = last_seen_iso
        if inactive_since_iso:
            context["tag_inactive_since"] = inactive_since_iso
        if screen_context:
            context.update(screen_context)
        context = self._context_with_risk_profile(context)

        if inactive:
            title = f"Telemetria UI nieaktywna dla tagu {tag}"
            body_parts = [
                "Tag {} nie raportuje próbek telemetrii.".format(tag),
                "Przekroczono próg {} s.".format(
                    context["tag_inactivity_threshold_seconds"]
                ),
            ]
            if age_seconds is not None:
                body_parts.append(
                    "Brak danych od {} s.".format(context["tag_inactivity_age_seconds"])
                )
            if last_seen_iso:
                body_parts.append(f"Ostatni snapshot: {last_seen_iso}.")
            if screen_summary:
                body_parts.append(f"Ostatni znany ekran: {screen_summary}.")
        else:
            title = f"Telemetria UI przywrócona dla tagu {tag}"
            body_parts = ["Tag {} wznowił wysyłanie telemetrii.".format(tag)]
            if duration_seconds is not None:
                body_parts.append(
                    "Przerwa trwała {} s.".format(
                        context["tag_inactivity_duration_seconds"]
                    )
                )
            if last_seen_iso:
                body_parts.append(f"Ostatni snapshot: {last_seen_iso}.")
            if screen_summary:
                body_parts.append(f"Aktywny ekran: {screen_summary}.")

        if self._enable_tag_inactivity_alerts:
            message = AlertMessage(
                category=self._tag_inactivity_category,
                title=title,
                body=" ".join(body_parts),
                severity=severity,
                context=context,
            )
            self._router.dispatch(message)

        if self._log_tag_inactivity_events:
            payload: dict[str, Any] = {
                "event": "tag_inactivity" if inactive else "tag_inactivity_recovered",
                "tag": tag,
                "tag_inactivity_threshold_seconds": self._tag_inactivity_threshold_seconds,
            }
            if age_seconds is not None:
                payload["tag_inactivity_age_seconds"] = max(0.0, age_seconds)
            if duration_seconds is not None:
                payload["tag_inactivity_duration_seconds"] = max(0.0, duration_seconds)
            if last_seen_iso:
                payload["tag_last_seen_at"] = last_seen_iso
            if inactive_since_iso:
                payload["tag_inactive_since"] = inactive_since_iso
            self._append_jsonl(
                self._tag_inactivity_category,
                severity,
                payload,
                snapshot,
                context=context,
            )

    def _handle_reduce_motion(self, snapshot, payload: dict[str, Any]) -> None:
        active = bool(payload.get("active"))
        previous = self._last_reduce_motion_state
        if previous is not None and previous == active:
            return
        self._last_reduce_motion_state = active
        if not (self._log_reduce_motion_events or self._enable_reduce_motion_alerts):
            return
        fps_target = payload.get("fps_target", 0)
        overlay_active = payload.get("overlay_active", 0)
        overlay_allowed = payload.get("overlay_allowed", 0)
        fps_value = (
            float(snapshot.fps)
            if hasattr(snapshot, "fps") and getattr(snapshot, "fps") is not None
            else 0.0
        )
        severity = (
            self._reduce_motion_severity_active
            if active
            else self._reduce_motion_severity_recovered
        )
        title = "UI reduce motion aktywny" if active else "UI reduce motion wyłączony"
        body = (
            "Tryb reduce motion {} – FPS {:.1f} przy celu {}. Nakładki {} z {}."
        ).format("włączony" if active else "wyłączony", fps_value, fps_target, overlay_active, overlay_allowed)
        context = {
            "active": str(active).lower(),
            "fps": f"{fps_value:.2f}",
            "fps_target": str(fps_target),
            "overlay_active": str(overlay_active),
            "overlay_allowed": str(overlay_allowed),
            "window_count": str(payload.get("window_count", "")),
        }
        tag_value = _extract_tag(payload)
        if tag_value:
            context["tag"] = tag_value
        context.update(_screen_context(payload))
        context = self._context_with_risk_profile(context)
        screen_summary = _screen_summary(payload)
        if self._log_reduce_motion_events:
            self._append_jsonl(
                self._reduce_motion_category,
                severity,
                payload,
                snapshot,
                context=context,
            )
        if not self._enable_reduce_motion_alerts:
            return
        message = AlertMessage(
            category=self._reduce_motion_category,
            title=title,
            body=body + (f" Ekran: {screen_summary}." if screen_summary else ""),
            severity=severity,
            context=context,
        )
        self._router.dispatch(message)

    def _handle_overlay_budget(self, snapshot, payload: dict[str, Any]) -> None:
        active = int(payload.get("active_overlays", 0))
        allowed = int(payload.get("allowed_overlays", 0))
        difference = active - allowed
        exceeded = difference > 0
        self._last_overlay_exceeded = exceeded

        if not (self._log_overlay_events or self._enable_overlay_alerts):
            return

        now_monotonic = time.monotonic()
        now_wallclock = datetime.now(timezone.utc)
        reduce_motion = bool(payload.get("reduce_motion"))
        reason: str | None = None
        severity_cause: str | None = None
        duration = 0.0
        duration_delta: float | None = None

        if exceeded:
            if not self._overlay_incident_active:
                self._overlay_incident_active = True
                self._overlay_incident_started_monotonic = now_monotonic
                self._overlay_incident_started_wallclock = now_wallclock
                self._overlay_incident_started_iso = _datetime_to_iso(now_wallclock)
                reason = "initial"
            else:
                if self._overlay_incident_started_monotonic is not None:
                    duration = max(0.0, now_monotonic - self._overlay_incident_started_monotonic)
                if self._overlay_incident_last_alert_monotonic is not None:
                    duration_delta = max(
                        0.0, now_monotonic - self._overlay_incident_last_alert_monotonic
                    )
                else:
                    duration_delta = duration

            if self._overlay_incident_started_monotonic is not None:
                duration = max(0.0, now_monotonic - self._overlay_incident_started_monotonic)

            severity = self._overlay_severity_exceeded
            if (
                self._overlay_critical_threshold is not None
                and difference >= self._overlay_critical_threshold
                and self._overlay_severity_critical
            ):
                severity = self._overlay_severity_critical
                severity_cause = "difference_threshold"
            elif (
                self._overlay_severity_critical
                and self._overlay_incident_critical_after_seconds is not None
                and duration >= self._overlay_incident_critical_after_seconds
            ):
                severity = self._overlay_severity_critical
                severity_cause = "duration_threshold"

            should_alert = False
            if reason == "initial":
                should_alert = True
            else:
                severity_change = (
                    self._overlay_incident_last_severity is not None
                    and severity != self._overlay_incident_last_severity
                )
                if severity_change:
                    should_alert = True
                    reason = "severity_change"
                elif (
                    self._overlay_incident_last_difference is not None
                    and difference - self._overlay_incident_last_difference
                    >= self._overlay_incident_realert_delta
                ):
                    should_alert = True
                    reason = "realert_difference"
                elif (
                    self._overlay_incident_realert_cooldown_seconds is not None
                    and self._overlay_incident_last_alert_monotonic is not None
                    and now_monotonic - self._overlay_incident_last_alert_monotonic
                    >= self._overlay_incident_realert_cooldown_seconds
                ):
                    should_alert = True
                    reason = "realert_cooldown"

            if should_alert:
                self._overlay_incident_last_alert_monotonic = now_monotonic
                self._overlay_incident_last_difference = difference
                self._overlay_incident_last_severity = severity
                if duration_delta is None:
                    duration_delta = duration
            else:
                severity = self._overlay_incident_last_severity or severity

            context = {
                "active_overlays": str(active),
                "allowed_overlays": str(allowed),
                "reduce_motion": str(reduce_motion).lower(),
                "window_count": str(payload.get("window_count", "")),
                "difference": str(difference),
                "overlay_incident_active": "true",
                "overlay_incident_reason": reason or "noop",
                "overlay_incident_duration_seconds": f"{duration:.3f}",
            }
            if self._overlay_incident_started_iso:
                context["overlay_incident_started_at"] = self._overlay_incident_started_iso
            if duration_delta is not None:
                context["overlay_incident_duration_delta_seconds"] = f"{duration_delta:.3f}"
            if self._overlay_incident_realert_delta:
                context["overlay_incident_realert_delta"] = str(self._overlay_incident_realert_delta)
            if self._overlay_incident_realert_cooldown_seconds is not None:
                context["overlay_incident_realert_cooldown_seconds"] = (
                    f"{self._overlay_incident_realert_cooldown_seconds:.3f}"
                )
            if self._overlay_incident_critical_after_seconds is not None:
                context["overlay_incident_critical_after_seconds"] = (
                    f"{self._overlay_incident_critical_after_seconds:.3f}"
                )
            if self._overlay_critical_threshold is not None:
                context["overlay_incident_critical_difference"] = str(
                    self._overlay_critical_threshold
                )
            if severity_cause:
                context["overlay_incident_severity_cause"] = severity_cause

            tag_value = _extract_tag(payload)
            if tag_value:
                context["tag"] = tag_value
            context.update(_screen_context(payload))
            context = self._context_with_risk_profile(context)
            screen_summary = _screen_summary(payload)

            if should_alert and self._log_overlay_events:
                self._append_jsonl(
                    self._overlay_category,
                    self._overlay_incident_last_severity or severity,
                    payload,
                    snapshot,
                    context=context,
                )
            elif self._log_overlay_events and reason in {"initial", "severity_change", "realert_difference", "realert_cooldown"}:
                self._append_jsonl(
                    self._overlay_category,
                    severity,
                    payload,
                    snapshot,
                    context=context,
                )

            if not (should_alert and self._enable_overlay_alerts):
                return

            display_duration = ""
            if duration > 0.0:
                display_duration = f" Incydent trwa {duration:.1f} s."
            body = (
                "Przekroczono budżet nakładek: {} aktywnych przy limicie {}. Reduce motion: {}.{}"
            ).format(active, allowed, "tak" if reduce_motion else "nie", display_duration)
            title = "Budżet nakładek przekroczony"
            message = AlertMessage(
                category=self._overlay_category,
                title=title,
                body=body + (f" Ekran: {screen_summary}." if screen_summary else ""),
                severity=self._overlay_incident_last_severity or severity,
                context=context,
            )
            self._router.dispatch(message)
        else:
            if not self._overlay_incident_active:
                return

            duration = 0.0
            if self._overlay_incident_started_monotonic is not None:
                duration = max(0.0, now_monotonic - self._overlay_incident_started_monotonic)
            if duration_delta is None and self._overlay_incident_last_alert_monotonic is not None:
                duration_delta = max(
                    0.0, now_monotonic - self._overlay_incident_last_alert_monotonic
                )
            severity = self._overlay_severity_recovered
            context = {
                "active_overlays": str(active),
                "allowed_overlays": str(allowed),
                "reduce_motion": str(reduce_motion).lower(),
                "window_count": str(payload.get("window_count", "")),
                "difference": str(difference),
                "overlay_incident_active": "false",
                "overlay_incident_reason": "recovered",
                "overlay_incident_duration_seconds": f"{duration:.3f}",
            }
            if duration_delta is not None:
                context["overlay_incident_duration_delta_seconds"] = f"{duration_delta:.3f}"
            if self._overlay_incident_started_iso:
                context["overlay_incident_started_at"] = self._overlay_incident_started_iso
            context["overlay_incident_recovered_at"] = _datetime_to_iso(now_wallclock)
            if self._overlay_incident_realert_delta:
                context["overlay_incident_realert_delta"] = str(self._overlay_incident_realert_delta)
            if self._overlay_incident_realert_cooldown_seconds is not None:
                context["overlay_incident_realert_cooldown_seconds"] = (
                    f"{self._overlay_incident_realert_cooldown_seconds:.3f}"
                )
            if self._overlay_incident_critical_after_seconds is not None:
                context["overlay_incident_critical_after_seconds"] = (
                    f"{self._overlay_incident_critical_after_seconds:.3f}"
                )
            if self._overlay_critical_threshold is not None:
                context["overlay_incident_critical_difference"] = str(
                    self._overlay_critical_threshold
                )

            tag_value = _extract_tag(payload)
            if tag_value:
                context["tag"] = tag_value
            context.update(_screen_context(payload))
            context = self._context_with_risk_profile(context)
            screen_summary = _screen_summary(payload)

            if self._log_overlay_events:
                self._append_jsonl(
                    self._overlay_category,
                    severity,
                    payload,
                    snapshot,
                    context=context,
                )

            if self._enable_overlay_alerts:
                body = (
                    "Budżet nakładek wrócił do normy: {} aktywnych przy limicie {}. Reduce motion: {}. Incydent trwał {:.1f} s."
                ).format(active, allowed, "tak" if reduce_motion else "nie", duration)
                title = "Budżet nakładek przywrócony"
                message = AlertMessage(
                    category=self._overlay_category,
                    title=title,
                    body=body + (f" Ekran: {screen_summary}." if screen_summary else ""),
                    severity=severity,
                    context=context,
                )
                self._router.dispatch(message)

            self._overlay_incident_active = False
            self._overlay_incident_started_monotonic = None
            self._overlay_incident_started_wallclock = None
            self._overlay_incident_started_iso = None
            self._overlay_incident_last_alert_monotonic = None
            self._overlay_incident_last_difference = None
            self._overlay_incident_last_severity = None

    def _handle_jank_spike(self, snapshot, payload: dict[str, Any]) -> None:
        frame_ms_raw = payload.get("frame_ms")
        threshold_raw = payload.get("threshold_ms")
        try:
            frame_ms = float(frame_ms_raw)
        except (TypeError, ValueError):
            frame_ms = 0.0
        try:
            threshold_ms = float(threshold_raw) if threshold_raw is not None else 0.0
        except (TypeError, ValueError):
            threshold_ms = 0.0

        signature = (int(frame_ms * 100), int(threshold_ms * 100))
        if self._last_jank_signature is not None and self._last_jank_signature == signature:
            return
        self._last_jank_signature = signature

        over_budget_raw = payload.get("over_budget_ms")
        try:
            over_budget = float(over_budget_raw) if over_budget_raw is not None else None
        except (TypeError, ValueError):
            over_budget = None
        if over_budget is None and threshold_ms > 0.0 and frame_ms > threshold_ms:
            over_budget = frame_ms - threshold_ms

        severity = self._jank_severity_spike
        if (
            over_budget is not None
            and self._jank_critical_over_ms is not None
            and over_budget >= self._jank_critical_over_ms
            and self._jank_severity_critical
        ):
            severity = self._jank_severity_critical

        reduce_motion = bool(payload.get("reduce_motion"))
        overlay_active = payload.get("overlay_active")
        overlay_allowed = payload.get("overlay_allowed")
        ratio_raw = payload.get("ratio")
        try:
            ratio = float(ratio_raw) if ratio_raw is not None else None
        except (TypeError, ValueError):
            ratio = None

        context = {
            "frame_ms": f"{frame_ms:.3f}",
            "threshold_ms": f"{threshold_ms:.3f}",
            "reduce_motion": str(reduce_motion).lower(),
            "overlay_active": str(overlay_active) if overlay_active is not None else "",
            "overlay_allowed": str(overlay_allowed) if overlay_allowed is not None else "",
            "window_count": str(payload.get("window_count", "")),
        }
        if over_budget is not None:
            context["over_budget_ms"] = f"{over_budget:.3f}"
        if ratio is not None:
            context["ratio"] = f"{ratio:.3f}"
        if getattr(snapshot, "HasField", None) and snapshot.HasField("fps"):
            context["fps"] = f"{snapshot.fps:.4f}"
        tag_value = _extract_tag(payload)
        if tag_value:
            context["tag"] = tag_value
        context.update(_screen_context(payload))
        context = self._context_with_risk_profile(context)
        screen_summary = _screen_summary(payload)

        if self._log_jank_events:
            self._append_jsonl(
                self._jank_category,
                severity,
                payload,
                snapshot,
                context=context,
            )
        if not self._enable_jank_alerts:
            return

        body_parts = [
            f"Klatka trwała {frame_ms:.2f} ms (limit {threshold_ms:.2f} ms).",
            f"Reduce motion: {'tak' if reduce_motion else 'nie'}.",
        ]
        if overlay_active is not None and overlay_allowed is not None:
            body_parts.append(f"Nakładki: {overlay_active}/{overlay_allowed}.")
        if over_budget is not None:
            body_parts.append(f"Przekroczenie limitu o {over_budget:.2f} ms.")
        if ratio is not None:
            body_parts.append(f"Stosunek: {ratio:.2f}x.")
        if screen_summary:
            body_parts.append(f"Ekran: {screen_summary}.")
        message = AlertMessage(
            category=self._jank_category,
            title="Jank klatkowy UI",
            body=" ".join(body_parts),
            severity=severity,
            context=context,
        )
        self._router.dispatch(message)

    def _category_with_suffix(self, base: str, suffix: str) -> str:
        parts = [segment for segment in base.split(".") if segment]
        if suffix in parts:
            return ".".join(parts)
        return ".".join(parts + [suffix])

    def _append_jsonl(
        self,
        category: str,
        severity: str,
        payload: Mapping[str, Any],
        snapshot,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        if not (self._jsonl_path and self._should_write_jsonl):
            return
        generated_at = _timestamp_to_iso(snapshot) if snapshot is not None else None
        if generated_at is None:
            generated_at = _datetime_to_iso(datetime.now(timezone.utc))
        record = {
            "category": category,
            "severity": severity,
            "generated_at": generated_at,
            "fps": getattr(snapshot, "fps", None) if snapshot is not None else None,
            "payload": payload,
            "context": dict(context) if context is not None else None,
        }
        tag_value = _extract_tag(payload)
        if tag_value:
            record["tag"] = tag_value
        if self._risk_profile_metadata is not None:
            record["risk_profile"] = self._risk_profile_metadata
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self._jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def _process_performance_metric(
        self,
        snapshot,
        payload: Mapping[str, Any],
        config: Mapping[str, Any],
        tag_value: str | None,
        state_key: str,
    ) -> None:
        metric_name = str(config.get("metric"))
        metric_states = self._performance_states.setdefault(metric_name, {})
        state = metric_states.get(state_key)
        if state is None:
            state = _MetricState()
            metric_states[state_key] = state

        value = float(config.get("value", 0.0))
        warning_threshold = config.get("warning")
        critical_threshold = config.get("critical")

        severity: str | None = None
        threshold_value: float | None = None
        reason: str | None = None
        if critical_threshold is not None and value >= critical_threshold:
            severity = self._performance_severity_critical
            threshold_value = float(critical_threshold)
            reason = "critical_threshold"
        elif warning_threshold is not None and value >= warning_threshold:
            severity = self._performance_severity_warning
            threshold_value = float(warning_threshold)
            reason = "warning_threshold"

        previous_severity = state.severity
        if severity is not None and previous_severity == severity:
            state.last_threshold = threshold_value
            return

        if severity is not None:
            now_wallclock = datetime.now(timezone.utc)
            started_iso = _timestamp_to_iso(snapshot)
            if started_iso is None:
                started_iso = _datetime_to_iso(now_wallclock)
            state.severity = severity
            state.started_wallclock = now_wallclock
            state.started_iso = started_iso
            state.last_threshold = threshold_value

            context: dict[str, Any] = {
                "metric": metric_name,
                "metric_value": self._format_metric_value(value),
                "metric_unit": config.get("unit", ""),
                "metric_severity": severity,
            }
            if threshold_value is not None:
                context["metric_threshold"] = self._format_metric_value(threshold_value)
            if reason:
                context["metric_reason"] = reason
            if state.started_iso:
                context["metric_started_at"] = state.started_iso
            if tag_value:
                context["tag"] = tag_value
            screen_context = _screen_context(payload)
            if screen_context:
                context.update(screen_context)
            context = self._context_with_risk_profile(context)

            screen_summary = _screen_summary(payload)
            log_payload = dict(payload)
            log_payload["event"] = "performance_metric"
            log_payload["performance_metric"] = metric_name
            log_payload["performance_metric_value"] = value
            if threshold_value is not None:
                log_payload["performance_metric_threshold"] = threshold_value
            if reason:
                log_payload["performance_metric_reason"] = reason
            log_payload["performance_metric_severity"] = severity
            log_payload["performance_metric_unit"] = config.get("unit", "")
            if state.started_iso:
                log_payload["performance_metric_started_at"] = state.started_iso

            if self._log_performance_events:
                self._append_jsonl(
                    self._performance_category,
                    severity,
                    log_payload,
                    snapshot,
                    context=context,
                )

            if not self._enable_performance_alerts:
                return

            unit_suffix = f" {config.get('unit', '')}" if config.get("unit") else ""
            value_str = self._format_metric_value(value)
            threshold_str = (
                self._format_metric_value(threshold_value)
                if threshold_value is not None
                else ""
            )
            body_parts = [
                f"{config.get('label', metric_name)}: {value_str}{unit_suffix}.",
            ]
            if threshold_str:
                body_parts.append(f"Próg: {threshold_str}{unit_suffix}.")
            if (
                severity == self._performance_severity_critical
                and warning_threshold is not None
                and (threshold_value is None or threshold_value != warning_threshold)
            ):
                body_parts.append(
                    "Próg ostrzegawczy: "
                    f"{self._format_metric_value(float(warning_threshold))}{unit_suffix}."
                )
            if tag_value:
                body_parts.append(f"Tag: {tag_value}.")
            if screen_summary:
                body_parts.append(f"Ekran: {screen_summary}.")

            title = (
                config.get("critical_title")
                if severity == self._performance_severity_critical
                else config.get("warning_title")
            )
            message = AlertMessage(
                category=self._performance_category,
                title=title or "Alert wydajności UI",
                body=" ".join(body_parts),
                severity=severity,
                context=context,
            )
            self._router.dispatch(message)
            return

        if previous_severity is None:
            return

        previous_threshold = state.last_threshold
        started_iso = state.started_iso
        started_wallclock = state.started_wallclock
        state.severity = None
        state.started_iso = None
        state.started_wallclock = None
        state.last_threshold = None

        now_wallclock = datetime.now(timezone.utc)
        recovered_iso = _datetime_to_iso(now_wallclock)
        duration_seconds: float | None = None
        if started_wallclock is not None:
            duration_seconds = max(0.0, (now_wallclock - started_wallclock).total_seconds())

        context = {
            "metric": metric_name,
            "metric_value": self._format_metric_value(value),
            "metric_unit": config.get("unit", ""),
            "metric_previous_severity": previous_severity,
            "metric_reason": "recovered",
            "metric_severity": self._performance_severity_recovered,
        }
        if previous_threshold is not None:
            context["metric_threshold"] = self._format_metric_value(previous_threshold)
        if started_iso:
            context["metric_started_at"] = started_iso
        if recovered_iso:
            context["metric_recovered_at"] = recovered_iso
        if duration_seconds is not None:
            context["metric_duration_seconds"] = f"{duration_seconds:.3f}"
        if tag_value:
            context["tag"] = tag_value
        screen_context = _screen_context(payload)
        if screen_context:
            context.update(screen_context)
        context = self._context_with_risk_profile(context)

        screen_summary = _screen_summary(payload)
        log_payload = dict(payload)
        log_payload["event"] = "performance_metric_recovered"
        log_payload["performance_metric"] = metric_name
        log_payload["performance_metric_value"] = value
        log_payload["performance_metric_previous_severity"] = previous_severity
        if previous_threshold is not None:
            log_payload["performance_metric_threshold"] = previous_threshold
        if started_iso:
            log_payload["performance_metric_started_at"] = started_iso
        if recovered_iso:
            log_payload["performance_metric_recovered_at"] = recovered_iso
        if duration_seconds is not None:
            log_payload["performance_metric_duration_seconds"] = duration_seconds
        log_payload["performance_metric_unit"] = config.get("unit", "")

        if self._log_performance_events:
            self._append_jsonl(
                self._performance_category,
                self._performance_severity_recovered,
                log_payload,
                snapshot,
                context=context,
            )

        if not self._enable_performance_alerts:
            return

        unit_suffix = f" {config.get('unit', '')}" if config.get("unit") else ""
        value_str = self._format_metric_value(value)
        body_parts = [
            f"{config.get('label', metric_name)} spadło do {value_str}{unit_suffix}.",
        ]
        if previous_threshold is not None:
            body_parts.append(
                "Próg odniesienia: "
                f"{self._format_metric_value(previous_threshold)}{unit_suffix}."
            )
        if duration_seconds is not None:
            body_parts.append(f"Czas trwania: {duration_seconds:.1f} s.")
        if tag_value:
            body_parts.append(f"Tag: {tag_value}.")
        if screen_summary:
            body_parts.append(f"Ekran: {screen_summary}.")

        message = AlertMessage(
            category=self._performance_category,
            title=config.get("recovery_title") or "Wydajność UI w normie",
            body=" ".join(body_parts),
            severity=self._performance_severity_recovered,
            context=context,
        )
        self._router.dispatch(message)

    def _format_metric_value(self, value: float) -> str:
        absolute = abs(value)
        if absolute >= 1000:
            return f"{value:.0f}"
        if absolute >= 100:
            return f"{value:.1f}"
        if absolute >= 10:
            return f"{value:.1f}"
        if absolute >= 1:
            return f"{value:.2f}"
        return f"{value:.3f}"

    def _context_with_risk_profile(self, context: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(context)
        if self._risk_profile_name:
            result.setdefault("risk_profile", self._risk_profile_name)
        if self._risk_profile_origin:
            result.setdefault("risk_profile_origin", self._risk_profile_origin)
        return result


__all__ = ["UiTelemetryAlertSink", "DEFAULT_UI_ALERTS_JSONL_PATH"]
