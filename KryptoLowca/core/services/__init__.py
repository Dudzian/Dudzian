"""Szkielet usług core wykorzystywany przez trading engine."""

from .error_policy import exception_guard, guard_exceptions
from .execution_service import ExecutionService
from .data_provider import ExchangeDataProvider
from .risk_service import RiskAssessment, RiskService
from .signal_service import SignalService
try:
    from bot_core.runtime import PaperTradingAdapter
except Exception:  # pragma: no cover - kompatybilność gdy moduł runtime nie istnieje
    PaperTradingAdapter = None  # type: ignore

__all__ = [
    "exception_guard",
    "guard_exceptions",
    "ExecutionService",
    "ExchangeDataProvider",
    "RiskAssessment",
    "RiskService",
    "SignalService",
    "PaperTradingAdapter",
]
