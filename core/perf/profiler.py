"""Lightweight CPU/GPU profilers with structured reporting."""
from __future__ import annotations

import cProfile
import contextlib
import io
import logging
import pstats
import time
from dataclasses import dataclass
from typing import Iterator

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - torch may be unavailable during CI
    torch = None  # type: ignore[assignment]


@dataclass(slots=True)
class ProfileReport:
    """Structured result of a profiling session."""

    name: str
    duration_s: float
    cpu_stats: str | None
    cpu_top: tuple[dict[str, object], ...] | None
    gpu_memory_delta: int | None
    gpu_peak_bytes: int | None
    timestamp: float

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable snapshot."""

        payload: dict[str, object] = {
            "name": self.name,
            "duration_s": round(self.duration_s, 6),
            "timestamp": self.timestamp,
        }
        if self.cpu_stats is not None:
            payload["cpu_stats"] = self.cpu_stats
        if self.cpu_top:
            payload["cpu_top"] = [
                {
                    "function": entry.get("function"),
                    "primitive_calls": int(entry.get("primitive_calls", 0)),
                    "total_calls": int(entry.get("total_calls", 0)),
                    "total_time": round(float(entry.get("total_time", 0.0)), 6),
                    "total_time_per_call": round(
                        float(entry.get("total_time_per_call", 0.0)), 6
                    ),
                    "cumulative_time": round(
                        float(entry.get("cumulative_time", 0.0)), 6
                    ),
                    "cumulative_time_per_call": round(
                        float(entry.get("cumulative_time_per_call", 0.0)), 6
                    ),
                }
                for entry in self.cpu_top
            ]
        if self.gpu_memory_delta is not None:
            payload["gpu_memory_delta"] = self.gpu_memory_delta
        if self.gpu_peak_bytes is not None:
            payload["gpu_peak_bytes"] = self.gpu_peak_bytes
        return payload


def _format_function_key(key: tuple[str, int, str]) -> str:
    filename, line, name = key
    return f"{name} ({filename}:{line})"


def _safe_divide(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / float(denominator)


def _extract_top_stats(
    stats_obj: pstats.Stats, limit: int
) -> tuple[dict[str, object], ...]:
    ordered = list(getattr(stats_obj, "fcn_list", ())) or list(stats_obj.stats.keys())
    entries: list[dict[str, object]] = []
    for key in ordered[:limit]:
        data = stats_obj.stats.get(key)
        if not data:
            continue
        primitive, total, total_time, cumulative_time, _ = data
        entries.append(
            {
                "function": _format_function_key(key),
                "primitive_calls": int(primitive),
                "total_calls": int(total),
                "total_time": float(total_time),
                "total_time_per_call": _safe_divide(float(total_time), int(total)),
                "cumulative_time": float(cumulative_time),
                "cumulative_time_per_call": _safe_divide(
                    float(cumulative_time), int(total)
                ),
            }
        )
    return tuple(entries)


class ProfilerSession:
    """Context manager collecting CPU and optional GPU performance data."""

    def __init__(
        self,
        name: str,
        *,
        sort: str = "tottime",
        limit: int = 25,
        enable_gpu: bool = True,
    ) -> None:
        self._name = name
        self._sort = sort
        self._limit = max(1, int(limit))
        self._enable_gpu = bool(enable_gpu)
        self._profile = cProfile.Profile()
        self._start: float | None = None
        self._gpu_start_bytes: int | None = None
        self._report: ProfileReport | None = None

    def __enter__(self) -> "ProfilerSession":
        self._start = time.perf_counter()
        if torch is not None and self._enable_gpu and torch.cuda.is_available():  # pragma: no cover - depends on CUDA
            try:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                self._gpu_start_bytes = int(torch.cuda.memory_allocated())
            except Exception:  # pragma: no cover - defensive, GPU ops are optional
                LOGGER.debug("Failed to initialise GPU profiling", exc_info=True)
                self._gpu_start_bytes = None
        self._profile.enable()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self._profile.disable()
        end = time.perf_counter()
        duration = max(end - (self._start or end), 0.0)
        stream = io.StringIO()
        stats_text: str | None = None
        top_entries: tuple[dict[str, object], ...] | None = None
        try:
            stats_obj = pstats.Stats(self._profile, stream=stream).sort_stats(self._sort)
            stats_obj.print_stats(self._limit)
            stats_text = stream.getvalue()
            top_entries = _extract_top_stats(stats_obj, self._limit)
        except Exception:  # pragma: no cover - profiling errors should not break runtime
            LOGGER.debug("Failed to collect CPU profiling stats", exc_info=True)
            stats_text = None
            top_entries = None
        gpu_delta: int | None = None
        gpu_peak: int | None = None
        if torch is not None and self._enable_gpu and torch.cuda.is_available():  # pragma: no cover - depends on CUDA
            try:
                torch.cuda.synchronize()
                peak = int(torch.cuda.max_memory_allocated())
                current = int(torch.cuda.memory_allocated())
                start_bytes = self._gpu_start_bytes or 0
                gpu_delta = current - start_bytes
                gpu_peak = peak
            except Exception:  # pragma: no cover - defensive guard
                LOGGER.debug("Failed to collect GPU profiling stats", exc_info=True)
                gpu_delta = None
                gpu_peak = None
        self._report = ProfileReport(
            name=self._name,
            duration_s=duration,
            cpu_stats=stats_text,
            cpu_top=top_entries,
            gpu_memory_delta=gpu_delta,
            gpu_peak_bytes=gpu_peak,
            timestamp=time.time(),
        )

    @property
    def report(self) -> ProfileReport | None:
        """Return the collected profile report."""

        return self._report


@contextlib.contextmanager
def profile_block(
    name: str,
    *,
    enabled: bool = True,
    sort: str = "tottime",
    limit: int = 25,
    enable_gpu: bool = True,
) -> Iterator[ProfilerSession | None]:
    """Context manager yielding an active :class:`ProfilerSession` when enabled."""

    if not enabled:
        yield None
        return
    session = ProfilerSession(name, sort=sort, limit=limit, enable_gpu=enable_gpu)
    with session:
        yield session
