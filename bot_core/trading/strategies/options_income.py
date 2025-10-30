"""Helper primitives for building options income signals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import pandas as pd


@dataclass(slots=True)
class OptionsIncomeSignalConfig:
    """Parametry normalizacji sygnału dla strategii dochodowej z opcji."""

    implied_realized_floor: float = 0.05
    spread_scale: float = 0.35
    delta_anchor_weight: float = 0.3
    theta_weight: float = 0.7
    anchor_window: int = 12

    def clamp(self) -> "OptionsIncomeSignalConfig":
        self.implied_realized_floor = max(1e-4, float(self.implied_realized_floor))
        self.spread_scale = max(1e-4, float(self.spread_scale))
        self.delta_anchor_weight = float(np.clip(self.delta_anchor_weight, 0.0, 1.0))
        self.theta_weight = float(np.clip(self.theta_weight, 0.0, 1.0))
        if self.theta_weight == 0.0 and self.delta_anchor_weight == 0.0:
            self.theta_weight = 0.7
            self.delta_anchor_weight = 0.3
        self.anchor_window = max(1, int(self.anchor_window))
        return self


def _infer_implied_vol_surface(
    market_data: Optional[pd.DataFrame], index: pd.Index
) -> pd.Series:
    if market_data is None or market_data.empty:
        return pd.Series(0.0, index=index)

    for column in ("implied_volatility", "iv", "atm_iv"):
        if column in market_data:
            series = market_data[column].reindex(index)
            return series.interpolate(limit_direction="both").bfill().ffill()

    return pd.Series(0.0, index=index)


def compute_options_income_signal(
    *,
    fast_price: pd.Series,
    atr: pd.Series,
    slow_anchor: pd.Series,
    market_data: Optional[pd.DataFrame],
    config: OptionsIncomeSignalConfig | Mapping[str, float] | None = None,
) -> pd.Series:
    """Buduje wygładzony sygnał theta-income na podstawie spreadu IV - RV."""

    if isinstance(config, Mapping):
        cfg = OptionsIncomeSignalConfig(**config).clamp()
    elif isinstance(config, OptionsIncomeSignalConfig):
        cfg = config.clamp()
    else:
        cfg = OptionsIncomeSignalConfig().clamp()

    atr = atr.replace(0.0, np.nan)
    realised = (atr / (fast_price.abs() + 1e-12)).rolling(window=cfg.anchor_window, min_periods=1).mean()
    realised = realised.fillna(cfg.implied_realized_floor)

    implied = _infer_implied_vol_surface(market_data, fast_price.index)
    implied = implied.fillna(realised)

    vol_spread = (implied - realised).fillna(0.0)
    spread_scaled = np.tanh(vol_spread / max(cfg.spread_scale, 1e-4))

    delta_anchor = (fast_price - slow_anchor) / (atr * 2.0)
    delta_anchor = delta_anchor.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-1.0, 1.0)

    signal = spread_scaled * cfg.theta_weight - delta_anchor * cfg.delta_anchor_weight
    return pd.Series(np.clip(signal, -1.0, 1.0), index=fast_price.index)


__all__ = [
    "OptionsIncomeSignalConfig",
    "compute_options_income_signal",
]
