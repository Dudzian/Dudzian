"""Sinki telemetrii UI przekierowujące zdarzenia do routera alertów."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from bot_core.alerts import AlertMessage, DefaultAlertRouter

DEFAULT_UI_ALERTS_JSONL_PATH = Path("logs/ui_telemetry_alerts.jsonl")

_LOGGER = logging.getLogger(__name__)


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
        enable_overlay_alerts: bool = True,
        enable_jank_alerts: bool = True,
        enable_retry_backlog_alerts: bool = True,
        enable_tag_inactivity_alerts: bool = True,
        log_reduce_motion_events: bool = True,
        log_overlay_events: bool = True,
        log_jank_events: bool = True,
        log_retry_backlog_events: bool = True,
        log_tag_inactivity_events: bool = True,
        reduce_motion_category: str = "ui.performance",
        reduce_motion_severity_active: str = "warning",
        reduce_motion_severity_recovered: str = "info",
        overlay_category: str = "ui.performance",
        overlay_severity_exceeded: str = "warning",
        overlay_severity_recovered: str = "info",
        overlay_critical_threshold: int | None = 2,
        overlay_severity_critical: str = "critical",
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
        risk_profile: Mapping[str, Any] | None = None,
    ) -> None:
        self._router = router
        self._jsonl_path = Path(jsonl_path) if jsonl_path else DEFAULT_UI_ALERTS_JSONL_PATH
        self._enable_reduce_motion_alerts = enable_reduce_motion_alerts
        self._enable_overlay_alerts = enable_overlay_alerts
        self._enable_jank_alerts = enable_jank_alerts
        self._enable_retry_backlog_alerts = enable_retry_backlog_alerts
        self._enable_tag_inactivity_alerts = enable_tag_inactivity_alerts
        self._log_reduce_motion_events = log_reduce_motion_events
        self._log_overlay_events = log_overlay_events
        self._log_jank_events = log_jank_events
        self._log_retry_backlog_events = log_retry_backlog_events
        self._log_tag_inactivity_events = log_tag_inactivity_events
        self._reduce_motion_category = self._category_with_suffix(
            reduce_motion_category, "reduce_motion"
        )
        self._reduce_motion_severity_active = reduce_motion_severity_active
        self._reduce_motion_severity_recovered = reduce_motion_severity_recovered
        self._overlay_category = self._category_with_suffix(overlay_category, "overlay_budget")
        self._overlay_severity_exceeded = overlay_severity_exceeded
        self._overlay_severity_recovered = overlay_severity_recovered
        self._overlay_critical_threshold = overlay_critical_threshold
        self._overlay_severity_critical = overlay_severity_critical
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
        self._lock = Lock()
        self._last_reduce_motion_state: bool | None = None
        self._last_overlay_exceeded: bool | None = None
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
        self._track_tag_inactivity(snapshot, payload)
        self._handle_retry_backlog(snapshot, payload)
        if event == "reduce_motion":
            self._handle_reduce_motion(snapshot, payload)
        elif event == "overlay_budget":
            self._handle_overlay_budget(snapshot, payload)
        elif event == "jank_spike":
            self._handle_jank_spike(snapshot, payload)

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
        exceeded = active > allowed
        previous = self._last_overlay_exceeded
        if previous is None and not exceeded:
            self._last_overlay_exceeded = False
            return
        if previous is not None and previous == exceeded:
            return
        self._last_overlay_exceeded = exceeded
        if not (self._log_overlay_events or self._enable_overlay_alerts):
            return
        reduce_motion = bool(payload.get("reduce_motion"))
        difference = active - allowed
        severity = (
            self._overlay_severity_exceeded
            if exceeded
            else self._overlay_severity_recovered
        )
        if (
            exceeded
            and self._overlay_critical_threshold is not None
            and difference >= self._overlay_critical_threshold
            and self._overlay_severity_critical
        ):
            severity = self._overlay_severity_critical
        if exceeded:
            body = (
                "Przekroczono budżet nakładek: {} aktywnych przy limicie {}. Reduce motion: {}."
            ).format(active, allowed, "tak" if reduce_motion else "nie")
        else:
            body = (
                "Budżet nakładek wrócił do normy: {} aktywnych przy limicie {}. Reduce motion: {}."
            ).format(active, allowed, "tak" if reduce_motion else "nie")
        title = "Budżet nakładek przekroczony" if exceeded else "Budżet nakładek przywrócony"
        context = {
            "active_overlays": str(active),
            "allowed_overlays": str(allowed),
            "reduce_motion": str(reduce_motion).lower(),
            "window_count": str(payload.get("window_count", "")),
            "difference": str(difference),
        }
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
        if not self._enable_overlay_alerts:
            return
        message = AlertMessage(
            category=self._overlay_category,
            title=title,
            body=body + (f" Ekran: {screen_summary}." if screen_summary else ""),
            severity=severity,
            context=context,
        )
        self._router.dispatch(message)

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

    def _context_with_risk_profile(self, context: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(context)
        if self._risk_profile_name:
            result.setdefault("risk_profile", self._risk_profile_name)
        if self._risk_profile_origin:
            result.setdefault("risk_profile_origin", self._risk_profile_origin)
        return result


__all__ = ["UiTelemetryAlertSink", "DEFAULT_UI_ALERTS_JSONL_PATH"]
