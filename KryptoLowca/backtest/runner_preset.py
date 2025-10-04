# backtest/runner_preset.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

from .preset_store import Preset, PresetStore  # lokalny, lekki import

logger = logging.getLogger(__name__)


@dataclass
class PresetRunResult:
    """
    Wynik uruchomienia backtestu dla jednego presetu.
    'metrics' powinny zawierać przynajmniej PF/Expectancy (ale tu nie narzucamy).
    """
    name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


class PresetBacktestRunner:
    """
    Minimalny runner, który utrzymuje API w ryzach (repo „zielone”).
    Prawdziwe uruchomienie strategii pozostaje świadomie niezaimplementowane, aby uniknąć
    błędów w środowiskach, gdzie brak jest zależności/strategii/danych.

    Jak używać docelowo:
      runner = PresetBacktestRunner(strategy_factory=my_factory)  # callable: params -> Strategy
      res = runner.run(data, "preset_name", store=my_store)
    """

    def __init__(self, strategy_factory: Optional[Callable[[Dict[str, Any]], Any]] = None) -> None:
        self.strategy_factory = strategy_factory

    # ----------------- public API -----------------

    def run(
        self,
        data: Any,
        preset: Union[Preset, Dict[str, Any], str],
        *,
        store: Optional[PresetStore] = None,
        evaluate: Optional[Callable[[Any], Dict[str, float]]] = None,
    ) -> PresetRunResult:
        """
        Uruchamia backtest dla pojedynczego presetu.
        - data: dowolny obiekt reprezentujący dane rynkowe
        - preset: Preset | dict parametrów | nazwa presetu (wtedy wymagany store)
        - evaluate: funkcja oceniająca metryki na podstawie „trades/eq curve” itp.
        """
        params, name = self._resolve_params_and_name(preset, store)

        if self.strategy_factory is None:
            # Repo ma być „zielone”: import przejdzie, ale świadomie nie wykonujemy backtestu.
            raise NotImplementedError(
                "PresetBacktestRunner: brak podłączonej strategy_factory. "
                "Przekaż strategy_factory=callable(params)->strategy aby uruchomić backtest."
            )

        strategy = self.strategy_factory(params)
        # Tu normalnie: trades = strategy.backtest(data) / engine.run(...)
        # Celowo zostawiamy brak implementacji, aby uniknąć ukrytych zależności:
        raise NotImplementedError(
            "PresetBacktestRunner.run: uruchomienie właściwego backtestu nie jest jeszcze zaimplementowane. "
            "Podłącz swoje środowisko/engine i oceń metryki w evaluate()."
        )

        # Przykład (docelowo):
        # metrics = evaluate(trades) if evaluate else self._default_evaluate(trades)
        # return PresetRunResult(name=name, metrics=metrics, extra={"params": params})

    def run_many(
        self,
        data: Any,
        presets: Sequence[Union[Preset, Dict[str, Any], str]],
        *,
        store: Optional[PresetStore] = None,
        evaluate: Optional[Callable[[Any], Dict[str, float]]] = None,
    ) -> List[PresetRunResult]:
        """
        Uruchamia sekwencyjnie wiele presetów i zwraca listę wyników.
        """
        results: List[PresetRunResult] = []
        for p in presets:
            results.append(self.run(data, p, store=store, evaluate=evaluate))
        return results

    def summarize(self, results: Iterable[PresetRunResult], key: str, higher_is_better: bool = True) -> Optional[PresetRunResult]:
        """
        Wybiera najlepszy wynik po zadanym kluczu metryki.
        """
        results = list(results)
        if not results:
            return None
        if higher_is_better:
            return max(results, key=lambda r: r.metrics.get(key, float("-inf")))
        return min(results, key=lambda r: r.metrics.get(key, float("inf")))

    # ----------------- helpers -----------------

    def _resolve_params_and_name(
        self,
        preset: Union[Preset, Dict[str, Any], str],
        store: Optional[PresetStore],
    ) -> "tuple[dict[str, Any], str]":
        if isinstance(preset, Preset):
            return preset.params, preset.name
        if isinstance(preset, dict):
            return preset, "<anonymous>"
        if isinstance(preset, str):
            if store is None:
                raise ValueError("Podano nazwę presetu, ale nie przekazano store=PresetStore.")
            obj = store.get_preset(preset)
            if obj is None:
                raise KeyError(f"Preset '{preset}' nie istnieje w store.")
            return obj.params, obj.name
        raise TypeError("preset musi być typu Preset | dict | str")

    def _default_evaluate(self, trades_or_equity: Any) -> Dict[str, float]:
        """
        Przykładowa domyślna ewaluacja (placeholder).
        """
        raise NotImplementedError("Nie zdefiniowano domyślnej ewaluacji metryk.")

