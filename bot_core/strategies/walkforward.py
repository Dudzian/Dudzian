"""Moduł walk-forward do podziału danych i wyboru parametrów strategii."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from bot_core.strategies.base import MarketSnapshot, WalkForwardOptimizer


class WalkForwardError(RuntimeError):
    """Wyjątek zgłaszany przy problemach z konfiguracją walk-forward."""


@dataclass(slots=True)
class WalkForwardWindow:
    """Opisuje długości okien in/out-of-sample dla analizy walk-forward."""

    training_size: int
    testing_size: int
    step_size: int | None = None

    def __post_init__(self) -> None:
        if self.training_size <= 0:
            raise WalkForwardError("Długość okna treningowego musi być dodatnia.")
        if self.testing_size <= 0:
            raise WalkForwardError("Długość okna testowego musi być dodatnia.")
        if self.step_size is not None and self.step_size <= 0:
            raise WalkForwardError("Krok przesuwania okna musi być dodatni.")

    @property
    def step(self) -> int:
        """Zwraca krok przesunięcia okna (domyślnie długość okresu testowego)."""

        return self.step_size or self.testing_size


class RollingWindowWalkForwardOptimizer(WalkForwardOptimizer):
    """Realizuje prostą analizę walk-forward na ruchomych oknach danych."""

    def __init__(
        self,
        window: WalkForwardWindow,
        parameter_grid: Sequence[Mapping[str, float]],
        scorer: Callable[[Sequence[MarketSnapshot], Mapping[str, float]], float],
        *,
        maximize: bool = True,
    ) -> None:
        if not parameter_grid:
            raise WalkForwardError("Lista kandydatów parametrów nie może być pusta.")
        self._window = window
        self._parameter_grid = tuple(parameter_grid)
        self._scorer = scorer
        self._maximize = maximize

    def split(
        self, data: Sequence[MarketSnapshot]
    ) -> Sequence[tuple[Sequence[MarketSnapshot], Sequence[MarketSnapshot]]]:
        """Dzieli dane na sekwencję par (in-sample, out-of-sample)."""

        total = len(data)
        train = self._window.training_size
        test = self._window.testing_size
        step = self._window.step

        if total < train + test:
            raise WalkForwardError(
                "Za mało danych do przeprowadzenia walk-forward. "
                f"Wymagane co najmniej {train + test} obserwacji, otrzymano {total}."
            )

        segments: list[tuple[Sequence[MarketSnapshot], Sequence[MarketSnapshot]]] = []
        start = 0
        while start + train + test <= total:
            in_sample = data[start : start + train]
            out_sample = data[start + train : start + train + test]
            segments.append((in_sample, out_sample))
            start += step

        if not segments:
            raise WalkForwardError("Nie udało się wygenerować żadnych segmentów walk-forward.")
        return tuple(segments)

    def select_parameters(self, in_sample: Sequence[MarketSnapshot]) -> Mapping[str, float]:
        """Wybiera najlepszy zestaw parametrów na podstawie funkcji ``scorer``."""

        best_score: float | None = None
        best_params: Mapping[str, float] | None = None

        for candidate in self._parameter_grid:
            score = self._scorer(in_sample, candidate)
            if best_score is None:
                best_score = score
                best_params = candidate
                continue

            if self._maximize and score > best_score:
                best_score = score
                best_params = candidate
            elif not self._maximize and score < best_score:
                best_score = score
                best_params = candidate

        if best_params is None:
            raise WalkForwardError("Nie udało się wybrać parametrów dla pustej próbki in-sample.")
        return best_params


__all__ = [
    "WalkForwardError",
    "WalkForwardWindow",
    "RollingWindowWalkForwardOptimizer",
]
