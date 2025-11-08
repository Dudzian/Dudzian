"""Testowa implementacja pluginu strategii do testów loadera."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bot_core.trading.strategies.plugins import StrategyPlugin


class DummyAdaptiveMMPlugin(StrategyPlugin):
    """Prosta strategia generująca neutralne sygnały."""

    engine_key = None
    name = "acme_adaptive_mm"
    description = "Neutralny sygnał adaptacyjnego market makingu do testów."
    license_tier = "sandbox"
    capability = "adaptive_mm"
    tags = ("market-making", "test")

    def generate(self, indicators, params, *, market_data=None):
        index = indicators.ema_fast.index
        return pd.Series(np.zeros(len(index), dtype=float), index=index)


def build_dummy_plugin() -> StrategyPlugin:
    """Fabryka zwracająca instancję pluginu do testów."""

    return DummyAdaptiveMMPlugin()
