"""Definicje scenariuszy symulacji Monte Carlo."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .distributions import HistoricalPriceSeries, annualize_volatility


class ModelType(str, Enum):
    """Obsługiwane modele generowania ścieżek cenowych."""

    GBM = "gbm"
    HESTON = "heston"
    BOOTSTRAP = "bootstrap"


@dataclass
class VolatilityConfig:
    """Konfiguracja zmienności wykorzystywana w symulacji."""

    mode: str = "historical"
    value: Optional[float] = None
    scaling: float = 1.0

    def resolve(self, historical: HistoricalPriceSeries, periods_per_year: int = 252) -> float:
        if self.mode == "historical":
            vol = annualize_volatility(historical.returns, periods_per_year)
            return float(vol * self.scaling)
        if self.mode == "constant":
            if self.value is None:
                raise ValueError("Dla trybu 'constant' należy podać wartość zmienności")
            return float(self.value * self.scaling)
        if self.mode == "scaled_hist":
            vol = annualize_volatility(historical.returns, periods_per_year)
            if self.value is None:
                raise ValueError("Dla trybu 'scaled_hist' należy podać współczynnik value")
            return float(vol * self.value * self.scaling)
        raise ValueError(f"Nieobsługiwany tryb zmienności: {self.mode}")


@dataclass
class MonteCarloScenario:
    """Parametry scenariusza Monte Carlo."""

    model: ModelType
    volatility: VolatilityConfig
    drift: Optional[float] = None
    heston_kappa: float = 1.5
    heston_theta: Optional[float] = None
    heston_sigma: float = 0.3
    heston_rho: float = -0.5
    heston_v0: Optional[float] = None

    def calibrate(self, historical: HistoricalPriceSeries, periods_per_year: int = 252) -> "MonteCarloScenario":
        """Zwraca nowy scenariusz ze skalibrowanymi parametrami."""

        drift = self.drift
        if drift is None:
            drift = float(historical.returns.mean() * periods_per_year)

        theta = self.heston_theta
        if theta is None:
            theta = float(np.clip(historical.returns.var() * periods_per_year, 1e-6, np.inf))

        v0 = self.heston_v0
        if v0 is None:
            v0 = float(np.clip(historical.returns.var() * periods_per_year, 1e-6, np.inf))

        return MonteCarloScenario(
            model=self.model,
            volatility=self.volatility,
            drift=drift,
            heston_kappa=self.heston_kappa,
            heston_theta=theta,
            heston_sigma=self.heston_sigma,
            heston_rho=self.heston_rho,
            heston_v0=v0,
        )
