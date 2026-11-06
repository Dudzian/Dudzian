"""Biblioteczny rejestrator TEST_ONLY; nie uruchamia VM ani nie nadaje zgód.

Wyniki zawsze mają klasyfikację MODEL_ONLY. Źródło opisuje pochodzenie
obserwacji, a nie kwalifikację platformy. CPU i RSS dotyczą wyłącznie procesu
rejestratora; RSS jest historycznym HWM procesu, nie maksimum żądania.
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import CodeType, FrameType
from typing import Any

CLIENT_DEADLINE_NS = 3_000_000_000
SOURCES = frozenset({"PROCESS", "POSTGRESQL", "MODEL"})
METRICS = frozenset(
    {
        "process_start",
        "listener_ready",
        "first_response",
        "client_exchange",
        "ipc",
        "cryptography",
        "postgresql",
        "identity",
        "cpu",
        "rss",
        "cgroup_cpu",
        "cgroup_memory",
        "profile_overhead",
        "semantic_verification",
        "model_dependency",
    }
)
UNITS = frozenset({"ns", "bytes", "count", "percent"})
TEMPERATURES = frozenset({"cold", "warm", "unspecified"})
OUTCOMES = frozenset({"complete", "error", "timeout", "observed"})


def _rss() -> int | None:
    if sys.platform != "linux":
        return None
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _label(value: str) -> bool:
    return (
        type(value) is str
        and 0 < len(value) <= 96
        and all(
            character.isascii() and (character.isalnum() or character in "_.-")
            for character in value
        )
    )


def _quantiles(values: list[int | float]) -> dict[str, Any]:
    ordered = sorted(values)
    size = len(ordered)
    return {
        "n": size,
        **{
            name: ordered[math.ceil(size * percentile) - 1] if size else None
            for name, percentile in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99))
        },
        "max": max(ordered) if ordered else None,
    }


class Recorder:
    """Zbiera ograniczone metadane i liczby, bez payloadu i treści wyjątków."""

    def __init__(self) -> None:
        self._samples: list[dict[str, Any]] = []

    @staticmethod
    def _validate(
        *,
        scenario: str,
        temperature: str,
        metric: str,
        source: str,
        unit: str,
        outcome: str,
        measured_for: str | None,
    ) -> None:
        if (
            not _label(scenario)
            or type(temperature) is not str
            or temperature not in TEMPERATURES
            or type(metric) is not str
            or metric not in METRICS
            or type(source) is not str
            or source not in SOURCES
            or type(unit) is not str
            or unit not in UNITS
            or type(outcome) is not str
            or outcome not in OUTCOMES
            or (
                measured_for is not None
                and (type(measured_for) is not str or measured_for not in METRICS)
            )
            or (metric == "client_exchange" and unit != "ns")
            or (metric in {"rss", "cgroup_memory"} and unit != "bytes")
            or (metric in {"cpu", "cgroup_cpu"} and unit != "ns")
        ):
            raise ValueError("PERFORMANCE_METADATA_INVALID")

    def record(
        self,
        value: int | float,
        *,
        scenario: str,
        temperature: str,
        metric: str,
        source: str,
        unit: str = "ns",
        outcome: str = "complete",
        measured_for: str | None = None,
    ) -> None:
        """Dodaj jawną obserwację; source jest deklaracją wywołującego."""
        self._validate(
            scenario=scenario,
            temperature=temperature,
            metric=metric,
            source=source,
            unit=unit,
            outcome=outcome,
            measured_for=measured_for,
        )
        if type(value) not in {int, float} or value < 0 or not math.isfinite(value):
            raise ValueError("PERFORMANCE_VALUE_INVALID")
        self._samples.append(
            {
                "scenario": scenario,
                "temperature": temperature,
                "metric": metric,
                "source": source,
                "unit": unit,
                "outcome": outcome,
                "value": value,
                "measured_for": measured_for,
            }
        )

    @contextmanager
    def sample(
        self,
        *,
        scenario: str,
        temperature: str,
        metric: str,
        source: str,
    ) -> Iterator[None]:
        """Mierz ścianę/CPU; wyjątek propaguje, timeout jest próbką przerwaną.

        Przerwany odcinek nie estymuje czasu potencjalnej późniejszej odpowiedzi.
        Rejestrator nie ustawia timeoutu, nie przycina czasu i nie ponawia próby.
        """
        self._validate(
            scenario=scenario,
            temperature=temperature,
            metric=metric,
            source=source,
            unit="ns",
            outcome="complete",
            measured_for=None,
        )
        started, cpu = time.perf_counter_ns(), time.process_time_ns()
        outcome = "complete"
        try:
            yield
        except BaseException as error:
            outcome = "timeout" if isinstance(error, TimeoutError) else "error"
            raise
        finally:
            elapsed, used_cpu = time.perf_counter_ns() - started, time.process_time_ns() - cpu
            self.record(
                elapsed,
                scenario=scenario,
                temperature=temperature,
                metric=metric,
                source=source,
                outcome=outcome,
            )
            self.record(
                used_cpu,
                scenario=scenario,
                temperature=temperature,
                metric="cpu",
                source="PROCESS",
                outcome=outcome,
                measured_for=metric,
            )
            rss = _rss()
            if rss is not None:
                self.record(
                    rss,
                    scenario=scenario,
                    temperature=temperature,
                    metric="rss",
                    source="PROCESS",
                    unit="bytes",
                    outcome=outcome,
                    measured_for=metric,
                )

    @contextmanager
    def profile(
        self,
        mapping: Mapping[CodeType, tuple[str, str]],
        *,
        scenario: str,
        temperature: str,
    ) -> Iterator[None]:
        """Profiluj wskazane obiekty kodu w bieżącym wątku, bez monkeypatch.

        Czasy są inkluzywne: zagnieżdżonych odcinków nie wolno sumować. Zdarzenie
        return występuje także przy wyjątku, dlatego outcome=observed nie orzeka
        o powodzeniu. Zewnętrzny sample rejestruje błąd całej operacji. Narzut
        hooka zawiera się w czasach; profile_overhead jest tylko jego estymacją.
        """
        if sys.getprofile() is not None:
            raise RuntimeError("ACTIVE_PROFILER_REFUSED")
        selected = dict(mapping)
        for code, pair in selected.items():
            if type(code) is not CodeType or type(pair) is not tuple or len(pair) != 2:
                raise ValueError("PERFORMANCE_PROFILE_MAPPING_INVALID")
            self._validate(
                scenario=scenario,
                temperature=temperature,
                metric=pair[0],
                source=pair[1],
                unit="ns",
                outcome="observed",
                measured_for=None,
            )
        # Walidacja działa również dla pustej mapy.
        self._validate(
            scenario=scenario,
            temperature=temperature,
            metric="profile_overhead",
            source="PROCESS",
            unit="ns",
            outcome="observed",
            measured_for=None,
        )
        active: dict[int, tuple[int, str, str]] = {}
        overhead = 0
        owner_pid = os.getpid()

        def hook(frame: FrameType, event: str, _argument: Any) -> None:
            nonlocal overhead
            if os.getpid() != owner_pid:
                return
            before = time.perf_counter_ns()
            if event == "call" and frame.f_code in selected:
                metric, source = selected[frame.f_code]
                active[id(frame)] = (before, metric, source)
            elif event == "return" and id(frame) in active:
                start, metric, source = active.pop(id(frame))
                self.record(
                    before - start,
                    scenario=scenario,
                    temperature=temperature,
                    metric=metric,
                    source=source,
                    outcome="observed",
                )
            overhead += time.perf_counter_ns() - before

        sys.setprofile(hook)
        try:
            yield
        finally:
            sys.setprofile(None)
            if os.getpid() == owner_pid:
                self.record(
                    overhead,
                    scenario=scenario,
                    temperature=temperature,
                    metric="profile_overhead",
                    source="PROCESS",
                    outcome="observed",
                )

    def summary(self) -> dict[str, Any]:
        """Zwróć kopię danych; brak promocji źródła do BASELINE_MEASURED."""
        keys = ("scenario", "temperature", "metric", "source", "unit", "measured_for")
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for sample in self._samples:
            grouped.setdefault(tuple(sample[key] for key in keys), []).append(sample)
        groups: list[dict[str, Any]] = []
        for identity, samples in grouped.items():
            complete = [sample["value"] for sample in samples if sample["outcome"] == "complete"]
            interrupted = [
                sample["value"] for sample in samples if sample["outcome"] in {"error", "timeout"}
            ]
            observed = [sample["value"] for sample in samples if sample["outcome"] == "observed"]
            groups.append(
                {
                    **dict(zip(keys, identity, strict=True)),
                    "n": len(samples),
                    "errors": sum(sample["outcome"] == "error" for sample in samples),
                    "timeouts": sum(sample["outcome"] == "timeout" for sample in samples),
                    "deadline_exceeded": sum(
                        sample["metric"] == "client_exchange"
                        and sample["value"] > CLIENT_DEADLINE_NS
                        for sample in samples
                    ),
                    "complete": _quantiles(complete),
                    "interrupted": _quantiles(interrupted),
                    "observed": _quantiles(observed),
                    "tail_1pct_expected": len(complete) / 100,
                    "uncertainty": "Brak przedziału ufności; zależność próbek i narzut obserwatora nieoszacowane.",
                }
            )
        return {
            "classification": "MODEL_ONLY",
            "client_deadline_ns": CLIENT_DEADLINE_NS,
            "samples": [dict(sample) for sample in self._samples],
            "groups": groups,
            "rss_scope": "Linux RUSAGE_SELF ru_maxrss: HWM całego życia procesu rejestratora; brak danych na innych OS.",
            "cpu_scope": "process_time_ns: cały proces rejestratora, bez dzieci i backendów PostgreSQL.",
        }
