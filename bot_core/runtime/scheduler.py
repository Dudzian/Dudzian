"""Ogólny harmonogram zadań dla operacji multi-portfelowych."""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from typing import Awaitable, Callable, Mapping, MutableMapping, Sequence

CallbackResult = Mapping[str, object] | None
TaskCallback = Callable[[datetime], Awaitable[CallbackResult] | CallbackResult]
Clock = Callable[[], datetime]


def _default_clock() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class ScheduleWindow:
    start: time | None = None
    end: time | None = None

    def contains(self, timestamp: datetime) -> bool:
        if self.start is None and self.end is None:
            return True
        current = timestamp.time()
        start = self.start.replace(tzinfo=None) if self.start is not None else None
        end = self.end.replace(tzinfo=None) if self.end is not None else None
        if self.start is not None and self.end is not None:
            if start <= end:
                return start <= current <= end
            return current >= start or current <= end
        if self.start is not None:
            return current >= start
        if self.end is not None:
            return current <= end
        return True


@dataclass(slots=True)
class ScheduledTask:
    name: str
    callback: TaskCallback
    priority: int = 0
    cooldown: timedelta = timedelta(seconds=0)
    window: ScheduleWindow = field(default_factory=ScheduleWindow)
    metadata: Mapping[str, object] = field(default_factory=dict)
    last_run: datetime | None = None
    _locked: bool = False

    def is_due(self, *, now: datetime) -> bool:
        if self._locked:
            return False
        if not self.window.contains(now):
            return False
        if self.last_run is None:
            return True
        return now - self.last_run >= self.cooldown


@dataclass(slots=True)
class TaskResult:
    name: str
    scheduled_at: datetime
    started_at: datetime
    finished_at: datetime
    success: bool
    payload: Mapping[str, object] | None = None
    error: BaseException | None = None

    def as_dict(self) -> Mapping[str, object]:
        data: dict[str, object] = {
            "name": self.name,
            "scheduled_at": self.scheduled_at.isoformat(),
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "success": self.success,
        }
        if self.payload:
            data["payload"] = dict(self.payload)
        if self.error:
            data["error"] = repr(self.error)
        return data


class CyclicTaskScheduler:
    """Prosty harmonogram obsługujący priorytety, cooldown i okna czasowe."""

    def __init__(self, *, clock: Clock | None = None) -> None:
        self._clock = clock or _default_clock
        self._tasks: MutableMapping[str, ScheduledTask] = {}

    def register(self, task: ScheduledTask) -> None:
        if task.name in self._tasks:
            raise ValueError(f"Task {task.name} already registered")
        self._tasks[task.name] = task

    def update(self, name: str, **changes: object) -> None:
        if name not in self._tasks:
            raise KeyError(name)
        task = self._tasks[name]
        for field_name, value in changes.items():
            if not hasattr(task, field_name):
                raise AttributeError(field_name)
            setattr(task, field_name, value)

    def tasks(self) -> Sequence[ScheduledTask]:
        return tuple(self._tasks.values())

    async def run_pending(self) -> tuple[TaskResult, ...]:
        now = self._clock()
        due = [task for task in self._tasks.values() if task.is_due(now=now)]
        due.sort(key=lambda item: (-item.priority, item.name))
        results: list[TaskResult] = []

        for task in due:
            task._locked = True
            started = self._clock()
            try:
                payload = await self._invoke(task.callback, started)
            except BaseException as exc:  # pragma: no cover - defensywne logowanie
                finished = self._clock()
                task.last_run = finished
                task._locked = False
                results.append(
                    TaskResult(
                        name=task.name,
                        scheduled_at=now,
                        started_at=started,
                        finished_at=finished,
                        success=False,
                        payload=None,
                        error=exc,
                    )
                )
            else:
                finished = self._clock()
                task.last_run = finished
                task._locked = False
                payload_mapping: Mapping[str, object] | None
                if payload is None:
                    payload_mapping = None
                else:
                    payload_mapping = dict(payload)
                results.append(
                    TaskResult(
                        name=task.name,
                        scheduled_at=now,
                        started_at=started,
                        finished_at=finished,
                        success=True,
                        payload=payload_mapping,
                        error=None,
                    )
                )
        return tuple(results)

    async def _invoke(
        self, callback: TaskCallback, started: datetime
    ) -> Mapping[str, object] | None:
        result = callback(started)
        if inspect.isawaitable(result):
            awaited = await result  # type: ignore[func-returns-value]
            if awaited is None:
                return None
            return dict(awaited)
        if result is None:
            return None
        return dict(result)


__all__ = [
    "ScheduleWindow",
    "ScheduledTask",
    "TaskResult",
    "CyclicTaskScheduler",
]
