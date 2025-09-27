"""Infrastruktura runtime nowej architektury bota."""

from bot_core.runtime.bootstrap import BootstrapContext, bootstrap_environment

# Kontrolery mogą się różnić między gałęziami – importujemy opcjonalnie.
try:
    from bot_core.runtime.controller import TradingController as _TradingController  # type: ignore
except Exception:
    _TradingController = None  # type: ignore

try:
    from bot_core.runtime.controller import DailyTrendController as _DailyTrendController  # type: ignore
except Exception:
    _DailyTrendController = None  # type: ignore

__all__ = ["BootstrapContext", "bootstrap_environment"]

# Eksportuj tylko te kontrolery, które są dostępne w danej gałęzi.
if _TradingController is not None:
    TradingController = _TradingController  # type: ignore
    __all__.append("TradingController")

if _DailyTrendController is not None:
    DailyTrendController = _DailyTrendController  # type: ignore
    __all__.append("DailyTrendController")
