"""Helper utilities for AI-related alerts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .dispatcher import AlertSeverity, emit_alert


@dataclass(slots=True)
class DriftAlertPayload:
    """Structured payload describing a drift alert for AI models."""

    model_name: str
    drift_score: float
    threshold: float
    window: int
    backend: str = "decision_engine"
    extra: Mapping[str, float] | None = None


def emit_model_drift_alert(payload: DriftAlertPayload) -> None:
    """Emit a drift alert using the global dispatcher."""

    severity = AlertSeverity.ERROR if payload.drift_score > payload.threshold * 1.5 else AlertSeverity.WARNING
    context = {
        "model": payload.model_name,
        "backend": payload.backend,
        "drift_score": f"{payload.drift_score:.6f}",
        "threshold": f"{payload.threshold:.6f}",
        "window": str(payload.window),
    }
    if payload.extra:
        for key, value in payload.extra.items():
            context[key] = f"{float(value):.6f}"
    message = (
        f"Model drift detected for {payload.model_name}: score={payload.drift_score:.4f} "
        f"(threshold={payload.threshold:.4f})"
    )
    emit_alert(
        message,
        severity=severity,
        source=f"ai/{payload.backend}",
        context=context,
    )


__all__ = ["DriftAlertPayload", "emit_model_drift_alert"]
