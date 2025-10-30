"""Signal helpers for cross-exchange hedge strategies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import pandas as pd


@dataclass(slots=True)
class CrossExchangeHedgeConfig:
    """Konfiguracja sygnału zabezpieczającego ekspozycję spot/futures."""

    basis_scale: float = 0.01
    inventory_scale: float = 0.35
    latency_penalty: float = 0.2
    hedge_weight: float = 0.55
    inventory_weight: float = 0.25

    def clamp(self) -> "CrossExchangeHedgeConfig":
        self.basis_scale = max(1e-6, float(self.basis_scale))
        self.inventory_scale = max(1e-6, float(self.inventory_scale))
        self.latency_penalty = float(np.clip(self.latency_penalty, 0.0, 1.0))
        self.hedge_weight = float(np.clip(self.hedge_weight, 0.0, 1.0))
        self.inventory_weight = float(np.clip(self.inventory_weight, 0.0, 1.0))
        if self.hedge_weight + self.inventory_weight > 1.0:
            total = self.hedge_weight + self.inventory_weight
            self.hedge_weight /= total
            self.inventory_weight /= total
        return self


def _series(
    market_data: Optional[pd.DataFrame],
    column: str,
    index: pd.Index,
    default: float = 0.0,
) -> pd.Series:
    if market_data is None or market_data.empty or column not in market_data:
        return pd.Series(default, index=index)
    return market_data[column].reindex(index).interpolate(limit_direction="both").fillna(default)


def compute_cross_exchange_hedge_signal(
    *,
    spot_basis: pd.Series,
    inventory_skew: pd.Series,
    market_data: Optional[pd.DataFrame],
    config: CrossExchangeHedgeConfig | Mapping[str, float] | None = None,
) -> pd.Series:
    """Buduje sygnał sugerujący pozycję hedgingową cross-venue."""

    if isinstance(config, Mapping):
        cfg = CrossExchangeHedgeConfig(**config).clamp()
    elif isinstance(config, CrossExchangeHedgeConfig):
        cfg = config.clamp()
    else:
        cfg = CrossExchangeHedgeConfig().clamp()

    index = spot_basis.index
    latency = _series(market_data, "latency_ms", index, default=35.0)
    hedge_pressure = np.tanh(spot_basis / cfg.basis_scale)
    inventory_component = -np.tanh(inventory_skew / cfg.inventory_scale)

    latency_norm = np.clip(latency / 250.0, 0.0, 1.0)
    latency_penalty = latency_norm * cfg.latency_penalty

    combined = hedge_pressure * cfg.hedge_weight + inventory_component * cfg.inventory_weight
    adaptive_weight = 1.0 - cfg.hedge_weight - cfg.inventory_weight
    combined += hedge_pressure * adaptive_weight * (1.0 - latency_penalty)

    return pd.Series(np.clip(combined, -1.0, 1.0), index=index)


__all__ = [
    "CrossExchangeHedgeConfig",
    "compute_cross_exchange_hedge_signal",
]
