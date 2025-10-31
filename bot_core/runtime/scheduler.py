"""Ogólny harmonogram zadań dla operacji multi-portfelowych."""
from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from typing import (
    Awaitable,
    Callable,
    Mapping,
    MutableMapping,
    Protocol,
    Sequence,
    TypeVar,
)

try:  # pragma: no cover - httpx może być opcjonalne w niektórych dystrybucjach
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore


_QueueResult = TypeVar("_QueueResult")

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


@dataclass(slots=True)
class _QueueLimits:
    max_concurrency: int
    burst: int


@dataclass(slots=True)
class _QueueState:
    limits: _QueueLimits
    semaphore: asyncio.Semaphore
    condition: asyncio.Condition
    pending: int = 0


class AsyncIOQueueListener(Protocol):
    """Interfejs powiadomień o zdarzeniach kolejki I/O."""

    def on_rate_limit_wait(self, *, key: str, waited: float, burst: int, pending: int) -> None:
        ...

    def on_timeout(self, *, key: str, duration: float, exception: BaseException) -> None:
        ...


class AsyncIOTaskQueue:
    """Dispatcher zarządzający limitami I/O dla operacji sieciowych."""

    def __init__(
        self,
        *,
        default_max_concurrency: int = 8,
        default_burst: int = 16,
        event_listener: AsyncIOQueueListener | None = None,
    ) -> None:
        if default_max_concurrency <= 0:
            raise ValueError("default_max_concurrency musi być dodatnie")
        if default_burst <= 0:
            raise ValueError("default_burst musi być dodatnie")
        self._default_limits = _QueueLimits(
            max_concurrency=int(default_max_concurrency),
            burst=int(default_burst),
        )
        self._queues: MutableMapping[str, _QueueState] = {}
        self._lock = asyncio.Lock()
        self._event_listener = event_listener

    def configure_exchange(
        self,
        name: str,
        *,
        max_concurrency: int | None = None,
        burst: int | None = None,
    ) -> None:
        """Aktualizuje limity dla wskazanego klucza (np. giełdy)."""

        normalized = self._normalize_key(name)
        if not normalized:
            raise ValueError("Nazwa kolejki nie może być pusta")
        limits = _QueueLimits(
            max_concurrency=self._default_limits.max_concurrency,
            burst=self._default_limits.burst,
        )
        if max_concurrency is not None:
            if max_concurrency <= 0:
                raise ValueError("max_concurrency musi być dodatnie")
            limits = _QueueLimits(
                max_concurrency=int(max_concurrency),
                burst=int(burst if burst is not None else limits.burst),
            )
        if burst is not None:
            if burst <= 0:
                raise ValueError("burst musi być dodatnie")
            limits = _QueueLimits(
                max_concurrency=int(max_concurrency if max_concurrency is not None else limits.max_concurrency),
                burst=int(burst),
            )
        self._queues[normalized] = _QueueState(
            limits,
            asyncio.Semaphore(limits.max_concurrency),
            asyncio.Condition(),
        )

    async def submit(
        self,
        key: str,
        coroutine_factory: Callable[[], Awaitable[_QueueResult]],
    ) -> _QueueResult:
        """Planuje wykonanie korutyny respektując limity dla klucza."""

        normalized = self._normalize_key(key)
        queue = await self._get_or_create_queue(normalized)
        async with queue.condition:
            wait_started: float | None = None
            while queue.pending >= queue.limits.burst:
                if wait_started is None and self._event_listener is not None:
                    wait_started = self._loop_time()
                await queue.condition.wait()
            queue.pending += 1
            pending_after = queue.pending
        if self._event_listener is not None and wait_started is not None:
            waited = self._loop_time() - wait_started
            self._event_listener.on_rate_limit_wait(
                key=normalized,
                waited=waited,
                burst=queue.limits.burst,
                pending=pending_after,
            )
        await queue.semaphore.acquire()
        started = self._loop_time()
        try:
            result = await coroutine_factory()
        except Exception as exc:
            if self._event_listener is not None and self._is_timeout_exception(exc):
                duration = self._loop_time() - started
                self._event_listener.on_timeout(
                    key=normalized,
                    duration=duration,
                    exception=exc,
                )
            raise
        finally:
            queue.semaphore.release()
            async with queue.condition:
                queue.pending -= 1
                queue.condition.notify(1)
        return result

    async def _get_or_create_queue(self, key: str) -> _QueueState:
        normalized = self._normalize_key(key)
        async with self._lock:
            existing = self._queues.get(normalized)
            if existing is not None:
                return existing
            limits = self._default_limits
            state = _QueueState(
                limits,
                asyncio.Semaphore(limits.max_concurrency),
                asyncio.Condition(),
            )
            self._queues[normalized] = state
            return state

    @staticmethod
    def _normalize_key(key: str) -> str:
        normalized = str(key).strip()
        return normalized or "default"

    @staticmethod
    def _loop_time() -> float:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return time.monotonic()
        return loop.time()

    @staticmethod
    def _is_timeout_exception(exc: BaseException) -> bool:
        if isinstance(exc, asyncio.TimeoutError):
            return True
        if httpx is not None and isinstance(exc, httpx.TimeoutException):  # type: ignore[attr-defined]
            return True
        return False


__all__ = [
    "ScheduleWindow",
    "ScheduledTask",
    "TaskResult",
    "CyclicTaskScheduler",
    "AsyncIOQueueListener",
    "AsyncIOTaskQueue",
]
