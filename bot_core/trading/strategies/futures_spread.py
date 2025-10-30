"""Utilities for futures spread mean-reversion signals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FuturesSpreadSignalConfig:
    """Parametry budowy sygnału bazującego na spreadach futures."""

    entry_z: float = 1.25
    exit_z: float = 0.4
    basis_scale: float = 0.015
    funding_scale: float = 0.0008
    carry_weight: float = 0.35
    funding_weight: float = 0.25

    def clamp(self) -> "FuturesSpreadSignalConfig":
        self.entry_z = max(1e-4, float(self.entry_z))
        self.exit_z = max(1e-4, float(self.exit_z))
        self.basis_scale = max(1e-6, float(self.basis_scale))
        self.funding_scale = max(1e-6, float(self.funding_scale))
        self.carry_weight = float(np.clip(self.carry_weight, 0.0, 1.0))
        self.funding_weight = float(np.clip(self.funding_weight, 0.0, 1.0))
        if self.carry_weight + self.funding_weight > 1.0:
            total = self.carry_weight + self.funding_weight
            self.carry_weight /= total
            self.funding_weight /= total
        return self


def _extract_series(market_data: Optional[pd.DataFrame], key: str, index: pd.Index) -> pd.Series:
    if market_data is None or market_data.empty:
        return pd.Series(0.0, index=index)
    if key not in market_data:
        return pd.Series(0.0, index=index)
    return market_data[key].reindex(index).interpolate(limit_direction="both").bfill().ffill()


def compute_futures_spread_signal(
    *,
    spread_zscore: pd.Series,
    market_data: Optional[pd.DataFrame],
    config: FuturesSpreadSignalConfig | Mapping[str, float] | None = None,
) -> pd.Series:
    """Buduje sygnał hedge'ujący rozjazd kontraktów futures."""

    if isinstance(config, Mapping):
        cfg = FuturesSpreadSignalConfig(**config).clamp()
    elif isinstance(config, FuturesSpreadSignalConfig):
        cfg = config.clamp()
    else:
        cfg = FuturesSpreadSignalConfig().clamp()

    index = spread_zscore.index
    basis = _extract_series(market_data, "basis", index)
    funding = _extract_series(market_data, "funding_rate", index)

    normalized_z = spread_zscore / cfg.entry_z
    normalized_z = normalized_z.clip(-3.0, 3.0)
    z_component = np.tanh(normalized_z)

    funding_component = np.tanh(funding / cfg.funding_scale)
    carry_component = -np.tanh(basis / cfg.basis_scale)

    combined = z_component * (1.0 - cfg.carry_weight - cfg.funding_weight)
    combined += funding_component * cfg.funding_weight
    combined += carry_component * cfg.carry_weight

    exit_mask = spread_zscore.abs() <= cfg.exit_z
    combined = pd.Series(np.clip(combined, -1.0, 1.0), index=index)
    combined[exit_mask] *= 0.2
    return combined


__all__ = [
    "FuturesSpreadSignalConfig",
    "compute_futures_spread_signal",
]
