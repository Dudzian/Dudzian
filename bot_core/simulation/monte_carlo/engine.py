"""Silnik symulacji Monte Carlo dla strategii inwestycyjnych."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from .distributions import HistoricalPriceSeries, resample_returns
from .scenarios import ModelType, MonteCarloScenario
from bot_core.observability.pandas_warnings import capture_pandas_warnings


_LOGGER = logging.getLogger(__name__)


@runtime_checkable
class SimulationStrategy(Protocol):
    """Minimalny interfejs strategii wykorzystywanej w symulacji."""

    name: str

    def evaluate_path(self, prices: pd.Series) -> float:
        """Zwraca wynik PnL dla pojedynczej ścieżki cenowej."""


@dataclass
class RiskParameters:
    """Parametry ryzyka dla symulacji Monte Carlo."""

    horizon_days: int
    confidence_level: float
    num_paths: int = 1000
    time_step_days: float = 1.0
    drawdown_threshold: float = 0.1
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level powinno być w przedziale (0, 1)")
        if self.horizon_days <= 0:
            raise ValueError("horizon_days powinno być dodatnie")
        if self.time_step_days <= 0:
            raise ValueError("time_step_days powinno być dodatnie")
        if self.num_paths <= 0:
            raise ValueError("num_paths powinno być dodatnie")


@dataclass
class StrategyResult:
    """Wyniki symulacji dla pojedynczej strategii."""

    pnl_distribution: np.ndarray
    metrics: Mapping[str, float]


@dataclass
class MonteCarloResult:
    """Zestaw wyników symulacji Monte Carlo."""

    price_paths: np.ndarray
    drawdown_distribution: np.ndarray
    drawdown_probability: float
    strategy_results: Dict[str, StrategyResult]
    scenario: MonteCarloScenario
    risk_parameters: RiskParameters

    @property
    def path_horizon(self) -> int:
        return self.price_paths.shape[1] - 1


class MonteCarloEngine:
    """Silnik realizujący symulację Monte Carlo z różnymi modelami."""

    def __init__(self, scenario: MonteCarloScenario, risk_parameters: RiskParameters) -> None:
        self.scenario = scenario
        self.risk_parameters = risk_parameters
        self._rng = np.random.default_rng(seed=risk_parameters.seed)

    def run(
        self,
        strategies: Sequence[SimulationStrategy | Callable[[pd.Series], float]],
        historical_prices: pd.Series,
    ) -> MonteCarloResult:
        """Uruchamia symulację Monte Carlo dla przekazanych strategii."""

        historical = HistoricalPriceSeries.from_prices(historical_prices)
        calibrated = self.scenario.calibrate(historical)
        price_paths = self._generate_price_paths(historical, calibrated)
        drawdowns = self._compute_drawdowns(price_paths)
        drawdown_probability = float(
            np.mean(drawdowns >= self.risk_parameters.drawdown_threshold)
        )
        normalized = self._normalize_strategies(strategies)
        strategy_results = {
            strategy.name: self._evaluate_strategy(strategy, price_paths, drawdown_probability)
            for strategy in normalized
        }
        return MonteCarloResult(
            price_paths=price_paths,
            drawdown_distribution=drawdowns,
            drawdown_probability=drawdown_probability,
            strategy_results=strategy_results,
            scenario=calibrated,
            risk_parameters=self.risk_parameters,
        )

    # ------------------------------------------------------------------
    # Ścieżki cenowe
    def _generate_price_paths(
        self, historical: HistoricalPriceSeries, scenario: MonteCarloScenario
    ) -> np.ndarray:
        steps = int(np.ceil(self.risk_parameters.horizon_days / self.risk_parameters.time_step_days))
        dt = self.risk_parameters.time_step_days / 252.0
        num_paths = self.risk_parameters.num_paths
        price0 = float(historical.prices.iloc[-1])
        if steps <= 0:
            raise ValueError("Liczba kroków symulacji jest równa zero")

        volatility = scenario.volatility.resolve(historical)
        if scenario.model == ModelType.GBM:
            return self._generate_gbm_paths(price0, scenario.drift, volatility, num_paths, steps, dt)
        if scenario.model == ModelType.HESTON:
            return self._generate_heston_paths(price0, scenario, volatility, num_paths, steps, dt)
        if scenario.model == ModelType.BOOTSTRAP:
            return self._generate_bootstrap_paths(price0, historical, num_paths, steps)
        raise ValueError(f"Nieobsługiwany model: {scenario.model}")

    def _generate_gbm_paths(
        self,
        price0: float,
        drift: Optional[float],
        volatility: float,
        num_paths: int,
        steps: int,
        dt: float,
    ) -> np.ndarray:
        mu = drift if drift is not None else 0.0
        sigma = volatility
        increments = self._rng.normal(size=(num_paths, steps))
        shocks = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * increments)
        paths = np.empty((num_paths, steps + 1), dtype=float)
        paths[:, 0] = price0
        for t in range(1, steps + 1):
            paths[:, t] = paths[:, t - 1] * shocks[:, t - 1]
        return paths

    def _generate_heston_paths(
        self,
        price0: float,
        scenario: MonteCarloScenario,
        volatility: float,
        num_paths: int,
        steps: int,
        dt: float,
    ) -> np.ndarray:
        kappa = scenario.heston_kappa
        theta = scenario.heston_theta if scenario.heston_theta is not None else volatility**2
        sigma_v = scenario.heston_sigma
        rho = scenario.heston_rho
        v0 = scenario.heston_v0 if scenario.heston_v0 is not None else volatility**2
        mu = scenario.drift if scenario.drift is not None else 0.0

        z1 = self._rng.normal(size=(num_paths, steps))
        z2 = self._rng.normal(size=(num_paths, steps))
        dw1 = np.sqrt(dt) * z1
        dw2 = np.sqrt(dt) * (rho * z1 + np.sqrt(1 - rho**2) * z2)

        paths = np.empty((num_paths, steps + 1), dtype=float)
        paths[:, 0] = price0
        variances = np.full((num_paths,), v0, dtype=float)

        for t in range(steps):
            variances = np.clip(
                variances + kappa * (theta - variances) * dt + sigma_v * np.sqrt(np.maximum(variances, 1e-12)) * dw2[:, t],
                1e-12,
                None,
            )
            paths[:, t + 1] = paths[:, t] * np.exp(
                (mu - 0.5 * variances) * dt + np.sqrt(np.maximum(variances, 1e-12)) * dw1[:, t]
            )
        return paths

    def _generate_bootstrap_paths(
        self,
        price0: float,
        historical: HistoricalPriceSeries,
        num_paths: int,
        steps: int,
    ) -> np.ndarray:
        returns = historical.returns.to_numpy()
        paths = np.empty((num_paths, steps + 1), dtype=float)
        paths[:, 0] = price0
        for i in range(num_paths):
            sampled = resample_returns(returns, size=steps, random_state=self._rng)
            paths[i, 1:] = price0 * np.exp(np.cumsum(sampled))
        return paths

    # ------------------------------------------------------------------
    # Strategie i metryki
    def _normalize_strategies(
        self, strategies: Sequence[SimulationStrategy | Callable[[pd.Series], float]]
    ) -> List[SimulationStrategy]:
        normalized: List[SimulationStrategy] = []
        for idx, strategy in enumerate(strategies):
            if isinstance(strategy, SimulationStrategy):
                normalized.append(strategy)
            elif callable(strategy):
                normalized.append(_CallableStrategy(name=f"strategy_{idx}", func=strategy))
            else:
                raise TypeError("Strategia musi implementować SimulationStrategy lub być wywoływalna")
        return normalized

    def _evaluate_strategy(
        self,
        strategy: SimulationStrategy,
        price_paths: np.ndarray,
        drawdown_probability: float,
    ) -> StrategyResult:
        pnl = np.empty(price_paths.shape[0], dtype=float)
        with capture_pandas_warnings(
            _LOGGER, component="backtest.monte_carlo.strategy"
        ):
            for idx, path in enumerate(price_paths):
                price_series = pd.Series(path, name="price")
                pnl[idx] = float(strategy.evaluate_path(price_series))
        metrics = self._calculate_metrics(pnl, drawdown_probability)
        return StrategyResult(pnl_distribution=pnl, metrics=metrics)

    def _calculate_metrics(self, pnl: np.ndarray, drawdown_probability: float) -> Dict[str, float]:
        confidence = self.risk_parameters.confidence_level
        losses = -pnl
        var = float(np.quantile(losses, confidence))
        tail_losses = losses[losses >= var]
        if tail_losses.size == 0:
            cvar = 0.0
        else:
            cvar = float(tail_losses.mean())
        expected_shortfall = cvar
        return {
            "mean_pnl": float(pnl.mean()),
            "std_pnl": float(pnl.std(ddof=1) if pnl.size > 1 else 0.0),
            "VaR": var,
            "CVaR": cvar,
            "expected_shortfall": expected_shortfall,
            "probabilistic_drawdown": drawdown_probability,
        }

    def _compute_drawdowns(self, price_paths: np.ndarray) -> np.ndarray:
        drawdowns = np.empty(price_paths.shape[0], dtype=float)
        for idx, path in enumerate(price_paths):
            cumulative_max = np.maximum.accumulate(path)
            drawdown = 1.0 - np.divide(path, cumulative_max, out=np.zeros_like(path), where=cumulative_max != 0)
            drawdowns[idx] = float(drawdown.max(initial=0.0))
        return drawdowns


@dataclass
class _CallableStrategy:
    name: str
    func: Callable[[pd.Series], float]

    def evaluate_path(self, prices: pd.Series) -> float:
        return float(self.func(prices))
