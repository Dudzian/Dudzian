"""Wysokopoziomowe harmonogramy zadań runtime (auto retraining AI)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping, MutableMapping, Sequence

from bot_core.ai.pipeline import (
    AutoRetrainPolicy,
    ModelTrainingResult,
    TrainingManifest,
    load_training_manifest,
    register_profile_results,
    run_training_profile,
)
from bot_core.runtime.journal import TradingDecisionEvent, TradingDecisionJournal


_LOGGER = logging.getLogger(__name__)

Clock = Callable[[], datetime]


def _ensure_utc(timestamp: datetime | None) -> datetime:
    if timestamp is None:
        return datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _extract_summary_metrics(result: ModelTrainingResult) -> Mapping[str, float]:
    metrics = result.metrics or {}
    if not isinstance(metrics, Mapping):
        return {}
    summary = metrics.get("summary")
    if not isinstance(summary, Mapping) or not summary:
        summary = metrics.get("train") if isinstance(metrics.get("train"), Mapping) else {}
    extracted: dict[str, float] = {}
    for key in ("mae", "rmse", "directional_accuracy", "expected_pnl"):
        value = summary.get(key) if isinstance(summary, Mapping) else None
        if isinstance(value, (int, float)):
            extracted[key] = float(value)
    return extracted


@dataclass(slots=True)
class AutoRetrainResult:
    """Rezultat pojedynczego uruchomienia zadania auto-retrain."""

    profile_name: str
    timestamp: datetime
    success: bool
    failures: tuple[str, ...]
    metrics: Mapping[str, Mapping[str, float]]
    artifacts: Mapping[str, Path]


@dataclass(slots=True)
class _AutoRetrainJob:
    name: str
    profile_name: str
    policy: AutoRetrainPolicy
    manifest: TrainingManifest
    output_dir: Path
    ai_manager: "AIManager | None" = None
    journal: TradingDecisionJournal | None = None
    clock: Clock = field(default=lambda: datetime.now(timezone.utc))
    next_run: datetime | None = None
    last_result: AutoRetrainResult | None = None

    def is_due(self, now: datetime) -> bool:
        if self.next_run is None:
            return True
        return now >= self.next_run

    def run(self, now: datetime) -> AutoRetrainResult:
        summary = run_training_profile(
            self.manifest,
            self.profile_name,
            output_dir=self.output_dir,
        )
        metrics: MutableMapping[str, Mapping[str, float]] = {}
        failures: list[str] = []
        for result in summary.models:
            extracted = _extract_summary_metrics(result)
            metrics[result.spec.name] = extracted
            quality = self.policy.quality
            if quality:
                dir_accuracy = extracted.get("directional_accuracy")
                if (
                    quality.min_directional_accuracy is not None
                    and (dir_accuracy is None or dir_accuracy < quality.min_directional_accuracy)
                ):
                    failures.append(
                        f"{result.spec.name}: directional_accuracy {dir_accuracy!r} < {quality.min_directional_accuracy}"
                    )
                mae = extracted.get("mae")
                if quality.max_mae is not None and (mae is None or mae > quality.max_mae):
                    failures.append(
                        f"{result.spec.name}: mae {mae!r} > {quality.max_mae}"
                    )
                rmse = extracted.get("rmse")
                if quality.max_rmse is not None and (rmse is None or rmse > quality.max_rmse):
                    failures.append(
                        f"{result.spec.name}: rmse {rmse!r} > {quality.max_rmse}"
                    )
        success = not failures
        if success and self.ai_manager is not None:
            try:
                register_profile_results(
                    summary,
                    self.ai_manager,
                    repository_root=self.output_dir,
                    register_ensembles=True,
                    set_default_if_missing=True,
                )
            except Exception:  # pragma: no cover - logujemy i kontynuujemy
                _LOGGER.exception("Nie udało się zarejestrować modeli po retrainingu %s", self.name)
                success = False
                failures.append("registration_failed")
        event_metadata: dict[str, str] = {
            "profile": self.profile_name,
            "models": ",".join(result.spec.name for result in summary.models),
            "status": "success" if success else "failed",
        }
        if metrics:
            event_metadata["metrics"] = json.dumps(metrics, ensure_ascii=False)
        if failures:
            event_metadata["quality_failures"] = " | ".join(failures)
        for key, value in (self.policy.metadata or {}).items():
            event_metadata[f"policy_{key}"] = str(value)
        if self.journal is not None:
            try:
                self.journal.record(
                    TradingDecisionEvent(
                        event_type="ai_auto_retrain_succeeded" if success else "ai_auto_retrain_failed",
                        timestamp=now,
                        environment=self.policy.journal_environment,
                        portfolio=self.policy.journal_portfolio,
                        risk_profile=self.policy.journal_risk_profile,
                        schedule=f"auto_retrain:{self.profile_name}",
                        strategy=self.policy.journal_strategy,
                        metadata=event_metadata,
                    )
                )
            except Exception:  # pragma: no cover - journaling nie może blokować retrainingu
                _LOGGER.exception("Nie udało się zapisać zdarzenia auto-retrain %s", self.name)
        artifacts = {result.spec.name: result.artifact_path for result in summary.models}
        result = AutoRetrainResult(
            profile_name=self.profile_name,
            timestamp=now,
            success=success,
            failures=tuple(failures),
            metrics=dict(metrics),
            artifacts=artifacts,
        )
        self.last_result = result
        self.next_run = now + timedelta(seconds=float(self.policy.interval_seconds))
        return result


class AutoRetrainScheduler:
    """Prosty harmonogram uruchamiający zadania auto-retrain profili AI."""

    def __init__(
        self,
        manifest: TrainingManifest,
        *,
        output_dir: Path,
        ai_manager: "AIManager | None" = None,
        journal: TradingDecisionJournal | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._manifest = manifest
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._ai_manager = ai_manager
        self._journal = journal
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._jobs: MutableMapping[str, _AutoRetrainJob] = {}

    def register_profile(
        self,
        profile_name: str,
        *,
        policy: AutoRetrainPolicy | None = None,
        job_name: str | None = None,
    ) -> str:
        profile = self._manifest.profile(profile_name)
        selected_policy = policy or profile.auto_retrain
        if selected_policy is None:
            raise ValueError(f"Profil {profile_name!r} nie posiada konfiguracji auto_retrain")
        name = job_name or f"auto_retrain:{profile_name}"
        normalized = name.strip().lower()
        if normalized in self._jobs:
            raise ValueError(f"Zadanie {name!r} jest już zarejestrowane")
        job = _AutoRetrainJob(
            name=normalized,
            profile_name=profile.name,
            policy=selected_policy,
            manifest=self._manifest,
            output_dir=self._output_dir,
            ai_manager=self._ai_manager,
            journal=self._journal,
            clock=self._clock,
        )
        self._jobs[normalized] = job
        return normalized

    def run_pending(self, now: datetime | None = None) -> tuple[AutoRetrainResult, ...]:
        timestamp = _ensure_utc(now)
        results: list[AutoRetrainResult] = []
        for job in self._jobs.values():
            if job.is_due(timestamp):
                results.append(job.run(timestamp))
        return tuple(results)

    def jobs(self) -> Sequence[str]:
        return tuple(sorted(self._jobs))


__all__ = [
    "AutoRetrainScheduler",
    "AutoRetrainResult",
    "load_training_manifest",
]


if TYPE_CHECKING:  # pragma: no cover - tylko dla mypy
    from bot_core.ai.manager import AIManager
else:  # pragma: no cover - fallback w środowisku runtime
    AIManager = object  # type: ignore[misc,assignment]
