"""Local retraining scheduler coordinating AutoRetrainScheduler cycles."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

from bot_core.runtime.schedulers import AutoRetrainScheduler
from bot_core.ai.pipeline import TrainingManifest, load_training_manifest


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_field(expression: str, minimum: int, maximum: int) -> Sequence[int]:
    tokens = [token.strip() for token in expression.split(",") if token.strip()]
    if not tokens:
        return tuple(range(minimum, maximum + 1))
    values: set[int] = set()
    for token in tokens:
        if token == "*":
            values.update(range(minimum, maximum + 1))
            continue
        step = 1
        if "/" in token:
            base, step_str = token.split("/", 1)
            token = base or "*"
            try:
                step = max(1, int(step_str))
            except ValueError:
                step = 1
        if token == "*":
            values.update(range(minimum, maximum + 1, step))
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError:
                continue
            start = max(minimum, start)
            end = min(maximum, end)
            values.update(range(start, end + 1, step))
            continue
        try:
            number = int(token)
        except ValueError:
            continue
        if minimum <= number <= maximum:
            values.add(number)
    return tuple(sorted(values)) or tuple(range(minimum, maximum + 1))


@dataclass(slots=True)
class CronSchedule:
    """Minimal cron schedule evaluator supporting minute-level resolution."""

    expression: str
    minutes: Sequence[int] = field(init=False)
    hours: Sequence[int] = field(init=False)
    days: Sequence[int] = field(init=False)
    months: Sequence[int] = field(init=False)
    weekdays: Sequence[int] = field(init=False)

    def __post_init__(self) -> None:
        fields = (self.expression or "*").split()
        if len(fields) != 5:
            raise ValueError("Cron expression must contain 5 fields (m h dom mon dow)")
        self.minutes = _parse_field(fields[0], 0, 59)
        self.hours = _parse_field(fields[1], 0, 23)
        self.days = _parse_field(fields[2], 1, 31)
        self.months = _parse_field(fields[3], 1, 12)
        self.weekdays = _parse_field(fields[4], 0, 6)

    def next_after(self, reference: datetime) -> datetime:
        candidate = reference.replace(second=0, microsecond=0) + timedelta(minutes=1)
        for _ in range(525600):  # limit search to one year
            if (
                candidate.minute in self.minutes
                and candidate.hour in self.hours
                and candidate.day in self.days
                and candidate.month in self.months
                and candidate.weekday() in self.weekdays
            ):
                return candidate
            candidate += timedelta(minutes=1)
        return candidate


@dataclass(slots=True)
class LocalRetrainScheduler:
    """Wrapper around :class:`AutoRetrainScheduler` with cron-like cadence."""

    manifest: TrainingManifest
    cron: CronSchedule
    scheduler: AutoRetrainScheduler
    profiles: Sequence[str]
    _next_run: datetime = field(default_factory=_utc_now, init=False)

    def __post_init__(self) -> None:
        now = _utc_now()
        self._next_run = self.cron.next_after(now - timedelta(minutes=1))
        if not self.profiles:
            self.profiles = tuple(profile.name for profile in self.manifest.profiles)
        for profile_name in self.profiles:
            try:
                self.scheduler.register_profile(profile_name)
            except ValueError:
                continue

    @classmethod
    def build(
        cls,
        *,
        cron_expression: str,
        manifest_path: Path,
        output_dir: Path,
        ai_manager: object | None,
        journal: object | None,
    ) -> "LocalRetrainScheduler | None":
        try:
            manifest = load_training_manifest(manifest_path)
        except FileNotFoundError:
            return None
        cron = CronSchedule(cron_expression)
        scheduler = AutoRetrainScheduler(
            manifest,
            output_dir=output_dir,
            ai_manager=ai_manager,  # type: ignore[arg-type]
            journal=journal,  # type: ignore[arg-type]
        )
        profiles: Sequence[str] = ()
        return cls(manifest=manifest, cron=cron, scheduler=scheduler, profiles=profiles)

    def next_run(self) -> datetime:
        return self._next_run

    def maybe_run(self, *, now: datetime | None = None) -> Sequence[Mapping[str, object]]:
        reference = now or _utc_now()
        if reference < self._next_run:
            return ()
        results = self.scheduler.run_pending(reference)
        self._next_run = self.cron.next_after(reference)
        return results


__all__ = ["CronSchedule", "LocalRetrainScheduler"]
