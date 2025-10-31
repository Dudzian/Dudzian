"""Definicje zdarzeń monitorujących runtime retrainingu i onboarding."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class MonitoringEvent:
    """Bazowe zdarzenie emitowane przez komponenty runtime."""

    timestamp: datetime = field(default_factory=_utc_now, init=False)


@dataclass(slots=True)
class MissingDataDetected(MonitoringEvent):
    """Informacja o brakujących porcjach danych treningowych."""

    source: str
    missing_batches: int


@dataclass(slots=True)
class DataDriftDetected(MonitoringEvent):
    """Wykrycie dryfu danych w trakcie przygotowania retrainingu."""

    source: str
    drift_score: float
    drift_threshold: float


@dataclass(slots=True)
class RetrainingDelayInjected(MonitoringEvent):
    """Zdarzenie informujące o celowym opóźnieniu startu retrainingu."""

    reason: str
    delay_seconds: float


@dataclass(slots=True)
class RetrainingCycleCompleted(MonitoringEvent):
    """Zdarzenie raportujące czas trwania i rezultat cyklu retrainingu."""

    source: str
    status: str
    duration_seconds: float
    drift_score: float | None = None
    metadata: dict[str, object] | None = None


@dataclass(slots=True)
class OnboardingCompleted(MonitoringEvent):
    """Informacja o pomyślnym zakończeniu kreatora onboardingowego."""

    duration_seconds: float
    license_id: str | None = None
    strategy: str | None = None
    exchange: str | None = None
    details: str | None = None
    onboarding_status_id: str | None = None


@dataclass(slots=True)
class OnboardingFailed(MonitoringEvent):
    """Informacja o błędzie podczas kreatora onboardingowego."""

    duration_seconds: float
    status_code: str
    status_message_id: str
    details: str | None = None
    strategy: str | None = None
    exchange: str | None = None
    onboarding_status_id: str | None = None


@dataclass(slots=True)
class ComplianceViolation(MonitoringEvent):
    """Alert naruszenia reguł zgodności wykryty przez audyt."""

    rule_id: str
    severity: str
    message: str
    metadata: dict[str, object] | None = None


@dataclass(slots=True)
class ComplianceAuditCompleted(MonitoringEvent):
    """Informacja o zakończeniu audytu zgodności."""

    passed: bool
    findings_total: int
    severity_breakdown: dict[str, int]
    config_path: str | None = None


class EventPublisher(Protocol):
    """Interfejs do publikacji zdarzeń monitorujących."""

    def __call__(self, event: MonitoringEvent) -> None:
        ...


__all__ = [
    "ComplianceAuditCompleted",
    "ComplianceViolation",
    "DataDriftDetected",
    "EventPublisher",
    "MissingDataDetected",
    "MonitoringEvent",
    "OnboardingCompleted",
    "OnboardingFailed",
    "RetrainingCycleCompleted",
    "RetrainingDelayInjected",
]
