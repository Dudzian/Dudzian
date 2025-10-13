"""Monitorowanie budżetów zasobów dla scheduler-a i strategii."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

__all__ = [
    "ResourceBudgets",
    "ResourceSample",
    "ResourceBudgetEvaluation",
    "evaluate_resource_sample",
]


@dataclass(slots=True)
class ResourceBudgets:
    """Docelowe limity zasobów dla procesu runtime."""

    cpu_percent: float
    memory_mb: float
    io_read_mb_s: float
    io_write_mb_s: float


@dataclass(slots=True)
class ResourceSample:
    """Zmierzone zużycie zasobów w zadanym oknie czasowym."""

    cpu_percent: float
    memory_mb: float
    io_read_mb_s: float
    io_write_mb_s: float


@dataclass(slots=True)
class ResourceBudgetEvaluation:
    """Wynik porównania próbek z budżetem."""

    status: str
    breaches: Mapping[str, float]
    warnings: Mapping[str, float]

    def as_dict(self) -> Mapping[str, object]:
        return {
            "status": self.status,
            "breaches": dict(self.breaches),
            "warnings": dict(self.warnings),
        }


def _ratio(actual: float, limit: float) -> float:
    if limit <= 0:
        return 0.0
    return actual / limit


def evaluate_resource_sample(
    budgets: ResourceBudgets,
    sample: ResourceSample,
    *,
    warning_threshold: float = 0.85,
) -> ResourceBudgetEvaluation:
    """Porównuje próbkę z budżetem i zwraca alerty."""

    metrics = {
        "cpu_percent": (sample.cpu_percent, budgets.cpu_percent),
        "memory_mb": (sample.memory_mb, budgets.memory_mb),
        "io_read_mb_s": (sample.io_read_mb_s, budgets.io_read_mb_s),
        "io_write_mb_s": (sample.io_write_mb_s, budgets.io_write_mb_s),
    }

    breaches: dict[str, float] = {}
    warnings: dict[str, float] = {}
    for name, (actual, limit) in metrics.items():
        ratio = _ratio(actual, limit)
        if ratio >= 1.0:
            breaches[name] = ratio
        elif ratio >= warning_threshold:
            warnings[name] = ratio

    if breaches:
        status = "error"
    elif warnings:
        status = "warning"
    else:
        status = "ok"

    return ResourceBudgetEvaluation(status=status, breaches=breaches, warnings=warnings)
