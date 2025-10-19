# -*- coding: utf-8 -*-
"""Top-level package for the trading bot.

Ten plik sprawia, że katalog staje się pakietem Pythona i udostępnia najważniejsze klasy
(re-eksporty), aby importy były krótsze i kompatybilne wstecz.
"""
from __future__ import annotations

__all__ = [
    "DatabaseManager",
    "DBOptions",
    "AIManager",
    "ConfigManager",
    "ExchangeManager",
    "ReportManager",
    "RiskManagerAdapter",
    "SecurityManager",
    "KeyRotationManager",
    "SecretManager",
    "SecretBackend",
    "TradingEngine",
    "TradingError",
    "DashboardApp",
    "DashboardController",
    "WalkForwardOptimizer",
]

# --- Re-eksport menedżerów ---
try:
    from .managers.database_manager import DatabaseManager, DBOptions  # type: ignore
except Exception:  # pragma: no cover
    DatabaseManager = None  # type: ignore
    DBOptions = None  # type: ignore

try:
    from .ai_manager import AIManager  # type: ignore
except Exception:  # pragma: no cover
    AIManager = None  # type: ignore

try:
    from .managers.config_manager import ConfigManager  # type: ignore
except Exception:  # pragma: no cover
    ConfigManager = None  # type: ignore

try:
    from .managers.exchange_manager import ExchangeManager  # type: ignore
except Exception:  # pragma: no cover
    ExchangeManager = None  # type: ignore

try:
    from .managers.report_manager import ReportManager  # type: ignore
except Exception:  # pragma: no cover
    ReportManager = None  # type: ignore

try:
    from .managers.risk_manager_adapter import RiskManagerAdapter  # type: ignore
except Exception:  # pragma: no cover
    RiskManagerAdapter = None  # type: ignore

try:
    from .managers.security_manager import SecurityManager  # type: ignore
except Exception:  # pragma: no cover
    SecurityManager = None  # type: ignore

try:
    from .security import KeyRotationManager, SecretManager, SecretBackend  # type: ignore
except Exception:  # pragma: no cover
    KeyRotationManager = None  # type: ignore
    SecretManager = None  # type: ignore
    SecretBackend = None  # type: ignore

# --- Dashboard ---
try:
    from .dashboard import DashboardApp, DashboardController  # type: ignore
except Exception:  # pragma: no cover
    DashboardApp = None  # type: ignore
    DashboardController = None  # type: ignore

# --- Optymalizacja ---
try:
    from .services.wfo import WalkForwardOptimizer  # type: ignore
except Exception:  # pragma: no cover
    WalkForwardOptimizer = None  # type: ignore

# --- Re-eksport silnika ---
try:
    from .core.trading_engine import TradingEngine, TradingError  # type: ignore
except Exception:  # pragma: no cover
    class TradingError(Exception): ...
    TradingEngine = None  # type: ignore
