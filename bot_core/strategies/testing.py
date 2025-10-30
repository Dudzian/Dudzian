"""Tester parametrów strategii korzystający z walk-forward backtestu."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping, Sequence

import pandas as pd

from bot_core.backtest.walk_forward import (
    TransactionCostModel,
    WalkForwardBacktester,
    WalkForwardReport,
    WalkForwardSegment,
)

from .catalog import StrategyCatalog, StrategyDefinition


@dataclass(slots=True)
class ParameterTestResult:
    parameters: Mapping[str, Any]
    report: WalkForwardReport


@dataclass(slots=True)
class StrategyParameterTestReport:
    engine: str
    results: tuple[ParameterTestResult, ...]
    best: ParameterTestResult


class StrategyParameterTester:
    """Buduje raporty walk-forward dla zestawów parametrów."""

    def __init__(
        self,
        catalog: StrategyCatalog,
        *,
        cost_model: TransactionCostModel | None = None,
    ) -> None:
        self._catalog = catalog
        self._backtester = WalkForwardBacktester(catalog, cost_model=cost_model)

    def evaluate(
        self,
        engine: str,
        parameter_grid: Mapping[str, Sequence[Any]],
        dataset: Mapping[str, pd.DataFrame],
        segments: Sequence[WalkForwardSegment],
        *,
        base_parameters: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        initial_balance: float = 100_000.0,
    ) -> StrategyParameterTestReport:
        if not parameter_grid:
            raise ValueError("Siatka parametrów nie może być pusta")

        spec = self._catalog.get(engine)
        keys = sorted(parameter_grid.keys())
        combinations = list(product(*(parameter_grid[key] for key in keys)))
        if not combinations:
            raise ValueError("Brak kombinacji parametrów do przetestowania")

        results: list[ParameterTestResult] = []
        for values in combinations:
            candidate_parameters = dict(base_parameters or {})
            candidate_parameters.update({key: value for key, value in zip(keys, values)})
            definition = StrategyDefinition(
                name=f"{engine}_wf",
                engine=spec.key,
                license_tier=spec.license_tier,
                risk_classes=spec.risk_classes,
                required_data=spec.required_data,
                parameters=candidate_parameters,
                tags=spec.default_tags,
                metadata=metadata or {},
            )
            report = self._backtester.run(
                definition,
                dataset,
                segments,
                initial_balance=initial_balance,
            )
            results.append(ParameterTestResult(parameters=candidate_parameters, report=report))

        best = max(results, key=lambda item: item.report.total_return_pct)
        return StrategyParameterTestReport(engine=engine, results=tuple(results), best=best)


__all__ = [
    "StrategyParameterTester",
    "StrategyParameterTestReport",
    "ParameterTestResult",
]

