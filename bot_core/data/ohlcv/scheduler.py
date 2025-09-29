"""Harmonogram odświeżania danych OHLCV w tle."""
from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Sequence

from bot_core.data.ohlcv.backfill import BackfillSummary, OHLCVBackfillService

_LOGGER = logging.getLogger(__name__)


def _utc_now_ms() -> int:
    """Zwraca bieżący czas w milisekundach od epochy."""

    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


@dataclass(slots=True)
class _RefreshJob:
    """Pojedyncze zadanie odświeżania dla interwału i listy symboli."""

    name: str
    symbols: tuple[str, ...]
    interval: str
    lookback_ms: int
    frequency_seconds: int
    jitter_seconds: int = 0


def _compute_sleep_seconds(frequency_seconds: int, jitter_seconds: int) -> float:
    """Zwraca czas oczekiwania pomiędzy cyklami z losową wariacją."""

    base = max(1.0, float(frequency_seconds))
    if jitter_seconds <= 0:
        return base

    jitter = random.uniform(-jitter_seconds, jitter_seconds)
    candidate = base + jitter
    return max(1.0, candidate)


class OHLCVRefreshScheduler:
    """Prosty harmonogram oparty o asyncio uruchamiający inkrementalne aktualizacje."""

    def __init__(
        self,
        service: OHLCVBackfillService,
        *,
        now_provider: Callable[[], int] | None = None,
        on_job_complete: Callable[[str, Sequence[BackfillSummary], int], None] | None = None,
    ) -> None:
        self._service = service
        self._jobs: list[_RefreshJob] = []
        self._stop_event: asyncio.Event | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._now_provider = now_provider or _utc_now_ms
        self._on_job_complete = on_job_complete

    def add_job(
        self,
        *,
        symbols: Sequence[str],
        interval: str,
        lookback_ms: int,
        frequency_seconds: int,
        jitter_seconds: int = 0,
        name: str | None = None,
    ) -> None:
        """Rejestruje zadanie odświeżania danych w tle."""

        if not symbols:
            raise ValueError("Lista symboli nie może być pusta")
        if lookback_ms <= 0:
            raise ValueError("lookback_ms musi być dodatni")
        if frequency_seconds <= 0:
            raise ValueError("frequency_seconds musi być dodatni")
        if jitter_seconds < 0:
            raise ValueError("jitter_seconds nie może być ujemny")

        job_name = name or f"{interval}:{','.join(sorted(symbols))}"
        job = _RefreshJob(
            name=job_name,
            symbols=tuple(symbols),
            interval=interval,
            lookback_ms=lookback_ms,
            frequency_seconds=frequency_seconds,
            jitter_seconds=jitter_seconds,
        )
        self._jobs.append(job)
        _LOGGER.debug(
            "Dodano zadanie scheduler'a: name=%s, interval=%s, frequency=%ss, lookback_ms=%s",
            job.name,
            job.interval,
            job.frequency_seconds,
            job.lookback_ms,
        )

    async def run_forever(self) -> None:
        """Uruchamia harmonogram aż do przerwania (KeyboardInterrupt/stop)."""

        if self._tasks:
            raise RuntimeError("Scheduler jest już uruchomiony")

        self._stop_event = asyncio.Event()
        self._tasks = [
            asyncio.create_task(self._job_loop(job), name=f"ohlcv-refresh:{job.name}")
            for job in self._jobs
        ]

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            if self._stop_event:
                self._stop_event.set()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            raise
        finally:
            self._tasks.clear()
            self._stop_event = None

    def stop(self) -> None:
        """Wstrzymuje wszystkie zadania i pozwala zakończyć pętlę."""

        if self._stop_event and not self._stop_event.is_set():
            self._stop_event.set()

    async def _job_loop(self, job: _RefreshJob) -> None:
        assert self._stop_event is not None, "Stop event musi być ustawiony przed startem zadań"

        while not self._stop_event.is_set():
            end = self._now_provider()
            start = max(0, end - job.lookback_ms)
            try:
                summaries = self._service.synchronize(
                    symbols=job.symbols,
                    interval=job.interval,
                    start=start,
                    end=end,
                )
                if self._on_job_complete:
                    try:
                        self._on_job_complete(job.interval, summaries, end)
                    except Exception:  # pragma: no cover - alerty nie mogą zatrzymać schedulera
                        _LOGGER.exception(
                            "Błąd podczas obsługi hooka on_job_complete dla zadania %s", job.name
                        )
                fetched = sum(summary.fetched_candles for summary in summaries)
                if fetched:
                    _LOGGER.info(
                        "Scheduler %s pobrał %s nowych świec (interval=%s)",
                        job.name,
                        fetched,
                        job.interval,
                    )
                else:
                    _LOGGER.debug(
                        "Scheduler %s nie znalazł nowych świec (interval=%s)",
                        job.name,
                        job.interval,
                    )
            except Exception:  # pragma: no cover - zabezpieczenie na runtime
                _LOGGER.exception("Błąd podczas odświeżania danych dla zadania %s", job.name)

            try:
                sleep_seconds = _compute_sleep_seconds(job.frequency_seconds, job.jitter_seconds)
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=sleep_seconds,
                )
            except asyncio.TimeoutError:
                continue

