"""Pakiet strategii tradingowych w natywnym rdzeniu bota."""
from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any

_LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - tylko dla analiz statycznych
    from .auto_trade import AutoTradeConfig, AutoTradeEngine
    from .engine import (
        TechnicalIndicators,
        TechnicalIndicatorsService,
        TradingEngine,
        TradingEngineFactory,
        TradingParameters,
        TradingStrategies,
    )
    from .regime_workflow import RegimeSwitchDecision, RegimeSwitchWorkflow
    from .strategies import (
        ArbitrageStrategy,
        DayTradingStrategy,
        MeanReversionStrategy,
        StrategyCatalog,
        StrategyPlugin,
        TrendFollowingStrategy,
    )

_OPTIONAL_EXPORTS: dict[str, str] = {
    "AutoTradeConfig": ".auto_trade",
    "AutoTradeEngine": ".auto_trade",
    "RegimeSwitchDecision": ".regime_workflow",
    "RegimeSwitchWorkflow": ".regime_workflow",
    "StrategyCatalog": ".strategies",
    "StrategyPlugin": ".strategies",
    "TrendFollowingStrategy": ".strategies",
    "DayTradingStrategy": ".strategies",
    "MeanReversionStrategy": ".strategies",
    "ArbitrageStrategy": ".strategies",
    "TradingEngine": ".engine",
    "TradingEngineFactory": ".engine",
    "TradingParameters": ".engine",
    "TradingStrategies": ".engine",
    "TechnicalIndicators": ".engine",
    "TechnicalIndicatorsService": ".engine",
}

__all__ = sorted(_OPTIONAL_EXPORTS)


def __getattr__(name: str) -> Any:  # pragma: no cover - leniwy import
    module_name = _OPTIONAL_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'bot_core.trading' has no attribute {name!r}")
    try:
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, name)
    except Exception as exc:  # pragma: no cover - degradacja środowiska
        _LOGGER.warning("Moduł %s nie jest dostępny: %s", module_name, exc)
        raise RuntimeError(f"{name} is not available: {exc}") from exc
    globals()[name] = value
    return value
