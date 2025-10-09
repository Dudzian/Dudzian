"""Sinki telemetrii UI przekierowujące zdarzenia do routera alertów."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from bot_core.alerts import AlertMessage, DefaultAlertRouter

DEFAULT_UI_ALERTS_JSONL_PATH = Path("logs/ui_telemetry_alerts.jsonl")

_LOGGER = logging.getLogger(__name__)


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


class UiTelemetryAlertSink:
    """Analizuje snapshoty UI i wysyła alerty o spadku FPS / nakładkach."""

    def __init__(self, router: DefaultAlertRouter, *, jsonl_path: str | Path | None = None) -> None:
        self._router = router
        self._jsonl_path = Path(jsonl_path) if jsonl_path else DEFAULT_UI_ALERTS_JSONL_PATH
        self._lock = Lock()
        if self._jsonl_path:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl_path.touch(exist_ok=True)

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

    def _handle_reduce_motion(self, snapshot, payload: dict[str, Any]) -> None:
        active = bool(payload.get("active"))
        fps_target = payload.get("fps_target", 0)
        overlay_active = payload.get("overlay_active", 0)
        overlay_allowed = payload.get("overlay_allowed", 0)
        fps_value = snapshot.fps if hasattr(snapshot, "fps") else 0.0
        severity = "warning" if active else "info"
        title = "UI reduce motion aktywny" if active else "UI reduce motion wyłączony"
        body = (
            "Tryb reduce motion {} – FPS {:.1f} przy celu {}. Nakładki {} z {}."
        ).format("włączony" if active else "wyłączony", fps_value, fps_target, overlay_active, overlay_allowed)
        message = AlertMessage(
            category="ui.performance.reduce_motion",
            title=title,
            body=body,
            severity=severity,
            context={
                "active": str(active).lower(),
                "fps": f"{fps_value:.2f}",
                "fps_target": str(fps_target),
                "overlay_active": str(overlay_active),
                "overlay_allowed": str(overlay_allowed),
                "window_count": str(payload.get("window_count", "")),
            },
        )
        self._router.dispatch(message)
        self._append_jsonl("ui.performance.reduce_motion", severity, payload, snapshot)

    def _handle_overlay_budget(self, snapshot, payload: dict[str, Any]) -> None:
        active = int(payload.get("active_overlays", 0))
        allowed = int(payload.get("allowed_overlays", 0))
        if active <= allowed:
            return
        reduce_motion = bool(payload.get("reduce_motion"))
        severity = "warning" if active - allowed <= 2 else "critical"
        body = (
            "Przekroczono budżet nakładek: {} aktywnych przy limicie {}. Reduce motion: {}."
        ).format(active, allowed, "tak" if reduce_motion else "nie")
        message = AlertMessage(
            category="ui.performance.overlay_budget",
            title="Budżet nakładek przekroczony",
            body=body,
            severity=severity,
            context={
                "active_overlays": str(active),
                "allowed_overlays": str(allowed),
                "reduce_motion": str(reduce_motion).lower(),
                "window_count": str(payload.get("window_count", "")),
            },
        )
        self._router.dispatch(message)
        self._append_jsonl("ui.performance.overlay_budget", severity, payload, snapshot)

    def _append_jsonl(self, category: str, severity: str, payload: dict[str, Any], snapshot) -> None:
        if not self._jsonl_path:
            return
        record = {
            "category": category,
            "severity": severity,
            "generated_at": _timestamp_to_iso(snapshot),
            "fps": getattr(snapshot, "fps", None),
            "payload": payload,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self._jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


__all__ = ["UiTelemetryAlertSink", "DEFAULT_UI_ALERTS_JSONL_PATH"]

