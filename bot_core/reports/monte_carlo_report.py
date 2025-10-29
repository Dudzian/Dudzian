"""Generator raportów dla symulacji Monte Carlo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from bot_core.simulation.monte_carlo.engine import MonteCarloResult


@dataclass
class MonteCarloReport:
    """Struktura raportu Monte Carlo."""

    summary: pd.DataFrame
    metadata: Dict[str, Any]
    price_paths: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """Reprezentuje raport w formacie słownikowym dogodnym do serializacji."""

        return {
            "summary": self.summary.reset_index().to_dict(orient="records"),
            "metadata": self.metadata,
        }


class MonteCarloReportBuilder:
    """Buduje raporty na bazie ``MonteCarloResult``."""

    def __init__(self, result: MonteCarloResult) -> None:
        self._result = result

    def build(self) -> MonteCarloReport:
        summary = self._build_summary_frame()
        metadata = self._build_metadata()
        return MonteCarloReport(summary=summary, metadata=metadata, price_paths=self._result.price_paths)

    def _build_summary_frame(self) -> pd.DataFrame:
        rows = []
        for name, strategy_result in self._result.strategy_results.items():
            metrics = dict(strategy_result.metrics)
            metrics["strategy"] = name
            metrics["mean_pnl"] = float(metrics.get("mean_pnl", np.nan))
            rows.append(metrics)
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.set_index("strategy").sort_index()
        return frame

    def _build_metadata(self) -> Dict[str, Any]:
        scenario = self._result.scenario
        risk = self._result.risk_parameters
        return {
            "model": scenario.model.value,
            "drift": scenario.drift,
            "volatility_mode": scenario.volatility.mode,
            "volatility_scaling": scenario.volatility.scaling,
            "confidence_level": risk.confidence_level,
            "num_paths": risk.num_paths,
            "horizon_days": risk.horizon_days,
            "time_step_days": risk.time_step_days,
            "probabilistic_drawdown": self._result.drawdown_probability,
        }


def build_monte_carlo_report(result: MonteCarloResult) -> MonteCarloReport:
    """Skrótowa funkcja pomocnicza do budowy raportu."""

    return MonteCarloReportBuilder(result).build()
