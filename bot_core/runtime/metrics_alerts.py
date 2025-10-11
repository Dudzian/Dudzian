"""Sinki telemetrii UI przekierowujące zdarzenia do routera alertów."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
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
        log_reduce_motion_events: bool = True,
        log_overlay_events: bool = True,
        log_jank_events: bool = True,
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
        risk_profile: Mapping[str, Any] | None = None,
    ) -> None:
        self._router = router
        self._jsonl_path = Path(jsonl_path) if jsonl_path else DEFAULT_UI_ALERTS_JSONL_PATH
        self._enable_reduce_motion_alerts = enable_reduce_motion_alerts
        self._enable_overlay_alerts = enable_overlay_alerts
        self._enable_jank_alerts = enable_jank_alerts
        self._log_reduce_motion_events = log_reduce_motion_events
        self._log_overlay_events = log_overlay_events
        self._log_jank_events = log_jank_events
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
        self._lock = Lock()
        self._last_reduce_motion_state: bool | None = None
        self._last_overlay_exceeded: bool | None = None
        self._last_jank_signature: tuple[int, int] | None = None
        self._should_write_jsonl = bool(
            self._jsonl_path
            and (
                self._log_reduce_motion_events
                or self._log_overlay_events
                or self._log_jank_events
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
        if event == "reduce_motion":
            self._handle_reduce_motion(snapshot, payload)
        elif event == "overlay_budget":
            self._handle_overlay_budget(snapshot, payload)
        elif event == "jank_spike":
            self._handle_jank_spike(snapshot, payload)

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
        record = {
            "category": category,
            "severity": severity,
            "generated_at": _timestamp_to_iso(snapshot),
            "fps": getattr(snapshot, "fps", None),
            "payload": payload,
            "context": dict(context) if context is not None else None,
        }
        if getattr(self, "_risk_profile_metadata", None) is not None:
            record["risk_profile"] = self._risk_profile_metadata
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self._jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def _context_with_risk_profile(self, context: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(context)
        if getattr(self, "_risk_profile_name", None):
            result.setdefault("risk_profile", self._risk_profile_name)
        if getattr(self, "_risk_profile_origin", None):
            result.setdefault("risk_profile_origin", self._risk_profile_origin)
        return result


__all__ = ["UiTelemetryAlertSink", "DEFAULT_UI_ALERTS_JSONL_PATH"]
