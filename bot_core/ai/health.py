"""Monitorowanie kondycji backendu AI i wykrywanie degradacji."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Mapping, MutableMapping, Sequence


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_mapping(payload: Mapping[str, object] | None) -> Mapping[str, float]:
    if not payload:
        return {}
    normalized: MutableMapping[str, float] = {}
    for key, value in payload.items():
        try:
            normalized[str(key)] = float(value)  # type: ignore[assignment]
        except (TypeError, ValueError):
            continue
    return dict(normalized)


@dataclass(slots=True)
class HealthObservation:
    """Pojedyncza obserwacja jakości modelu."""

    model_name: str
    ok: bool
    metrics: Mapping[str, float] = field(default_factory=dict)
    thresholds: Mapping[str, float] = field(default_factory=dict)
    reason: str | None = None
    source: str = "quality"
    recorded_at: datetime = field(default_factory=_now)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model_name": self.model_name,
            "ok": self.ok,
            "source": self.source,
            "recorded_at": self.recorded_at.isoformat(),
        }
        if self.reason:
            payload["reason"] = self.reason
        if self.metrics:
            payload["metrics"] = dict(self.metrics)
        if self.thresholds:
            payload["thresholds"] = dict(self.thresholds)
        return payload


@dataclass(slots=True)
class ModelHealthStatus:
    """Zrzut bieżącego stanu kondycji backendu AI."""

    degraded: bool
    reason: str | None
    details: tuple[str, ...] = ()
    failing_models: tuple[str, ...] = ()
    backend_degraded: bool = False
    quality_failures: int = 0
    last_observation: HealthObservation | None = None
    updated_at: datetime = field(default_factory=_now)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "degraded": self.degraded,
            "updated_at": self.updated_at.isoformat(),
            "quality_failures": self.quality_failures,
            "backend_degraded": self.backend_degraded,
            "failing_models": list(self.failing_models),
        }
        if self.reason:
            payload["reason"] = self.reason
        if self.details:
            payload["details"] = list(self.details)
        if self.last_observation is not None:
            payload["last_observation"] = self.last_observation.as_dict()
        return payload


class ModelHealthMonitor:
    """Akumuluje obserwacje jakości modeli i wykrywa degradację."""

    def __init__(
        self,
        *,
        max_quality_observations: int = 50,
        consecutive_failure_threshold: int = 2,
    ) -> None:
        if consecutive_failure_threshold <= 0:
            raise ValueError("consecutive_failure_threshold musi być dodatni")
        if max_quality_observations <= 0:
            raise ValueError("max_quality_observations musi być dodatni")
        self._history: Deque[HealthObservation] = deque(maxlen=max_quality_observations)
        self._consecutive_failure_threshold = int(consecutive_failure_threshold)
        self._quality_failure_streak = 0
        self._quality_degraded = False
        self._backend_degraded = False
        self._degraded = False
        self._reason: str | None = None
        self._details: tuple[str, ...] = ()
        self._failing_models: set[str] = set()
        self._updated_at: datetime = _now()

    def record_quality(
        self,
        *,
        model_name: str,
        ok: bool,
        metrics: Mapping[str, object] | None = None,
        thresholds: Mapping[str, object] | None = None,
        reason: str | None = None,
        source: str = "quality",
    ) -> ModelHealthStatus:
        observation = HealthObservation(
            model_name=model_name,
            ok=bool(ok),
            metrics=_normalize_mapping(metrics),
            thresholds=_normalize_mapping(thresholds),
            reason=reason,
            source=source,
        )
        self._history.append(observation)
        if observation.ok:
            self._quality_failure_streak = 0
            self._failing_models.discard(observation.model_name)
            if self._quality_degraded and not self._failing_models:
                self._quality_degraded = False
                if not self._backend_degraded:
                    self._clear_degradation()
        else:
            self._quality_failure_streak += 1
            self._failing_models.add(observation.model_name)
            if self._quality_failure_streak >= self._consecutive_failure_threshold:
                self._quality_degraded = True
                derived_reason = reason or f"quality_thresholds_failed:{observation.model_name}"
                details = [f"model={observation.model_name}"]
                for key, value in observation.metrics.items():
                    details.append(f"{key}={value:.6f}")
                for key, value in observation.thresholds.items():
                    details.append(f"threshold_{key}={value:.6f}")
                self._set_degradation(derived_reason, tuple(details))
        self._updated_at = _now()
        return self.snapshot()

    def record_backend_failure(
        self,
        *,
        reason: str,
        details: Sequence[str] | None = None,
    ) -> ModelHealthStatus:
        self._backend_degraded = True
        self._set_degradation(reason, tuple(str(item) for item in details or ()))
        self._updated_at = _now()
        return self.snapshot()

    def resolve_backend_recovery(self) -> ModelHealthStatus:
        self._backend_degraded = False
        if not self._quality_degraded:
            self._clear_degradation()
        self._updated_at = _now()
        return self.snapshot()

    def snapshot(self) -> ModelHealthStatus:
        last_observation = self._history[-1] if self._history else None
        return ModelHealthStatus(
            degraded=self._degraded,
            reason=self._reason,
            details=self._details,
            failing_models=tuple(sorted(self._failing_models)),
            backend_degraded=self._backend_degraded,
            quality_failures=self._quality_failure_streak if self._quality_degraded else 0,
            last_observation=last_observation,
            updated_at=self._updated_at,
        )

    def is_degraded(self) -> bool:
        return self._degraded

    def _set_degradation(self, reason: str, details: tuple[str, ...]) -> None:
        self._degraded = True
        self._reason = reason
        self._details = details

    def _clear_degradation(self) -> None:
        self._degraded = False
        self._reason = None
        self._details = ()
        self._failing_models.clear()
        self._quality_failure_streak = 0


__all__ = ["HealthObservation", "ModelHealthMonitor", "ModelHealthStatus"]
