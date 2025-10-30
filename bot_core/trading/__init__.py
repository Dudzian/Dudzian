"""Pakiet strategii tradingowych w natywnym rdzeniu bota."""
from __future__ import annotations

import logging
from types import SimpleNamespace

_LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - moduł może być opcjonalny w środowiskach CI
    from . import auto_trade as _auto_trade
    from .auto_trade import AutoTradeConfig, AutoTradeEngine
except Exception as exc:  # pragma: no cover - degradacja środowiska
    _LOGGER.warning("Moduł auto_trade nie jest dostępny: %s", exc)
    _auto_trade = SimpleNamespace()

    class AutoTradeConfig:  # type: ignore[override]
        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError("Moduł auto_trade nie jest dostępny w tej dystrybucji")

    class AutoTradeEngine:  # type: ignore[override]
        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError("Moduł auto_trade nie jest dostępny w tej dystrybucji")

_ENGINE_AVAILABLE = True
try:  # pragma: no cover - moduł silnika może być pominięty w okrojonych buildach
    from . import engine as _engine
    from .engine import *  # noqa: F401,F403 - udostępnij publiczne API modułu silnika
except Exception as exc:  # pragma: no cover - degradacja do implementacji stub
    _LOGGER.warning("Moduł trading.engine nie został poprawnie załadowany: %s", exc)
    _engine = SimpleNamespace(__all__=[])
    _ENGINE_AVAILABLE = False

if _ENGINE_AVAILABLE:
    from .regime_workflow import RegimeSwitchDecision, RegimeSwitchWorkflow
    from .strategies import (
        ArbitrageStrategy,
        DayTradingStrategy,
        MeanReversionStrategy,
        StrategyCatalog,
        StrategyPlugin,
        TrendFollowingStrategy,
    )
else:  # pragma: no cover - udostępnij minimalne stuby
    RegimeSwitchDecision = SimpleNamespace  # type: ignore[assignment]
    RegimeSwitchWorkflow = SimpleNamespace  # type: ignore[assignment]

    class StrategyCatalog:  # type: ignore[override]
        @staticmethod
        def default() -> "StrategyCatalog":
            raise RuntimeError("StrategyCatalog niedostępny bez modułu trading.engine")

    StrategyPlugin = TrendFollowingStrategy = DayTradingStrategy = MeanReversionStrategy = ArbitrageStrategy = SimpleNamespace  # type: ignore[assignment]

__all__ = list(getattr(_engine, "__all__", ())) + [
    "AutoTradeConfig",
    "AutoTradeEngine",
    "RegimeSwitchDecision",
    "RegimeSwitchWorkflow",
    "StrategyCatalog",
    "StrategyPlugin",
    "TrendFollowingStrategy",
    "DayTradingStrategy",
    "MeanReversionStrategy",
    "ArbitrageStrategy",
]

if "_engine" in locals():
    del _engine
if "_auto_trade" in locals():
    del _auto_trade
