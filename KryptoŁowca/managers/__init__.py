# -*- coding: utf-8 -*-
"""Subpakiet z menedżerami (DB, giełda, AI, raporty, bezpieczeństwo)."""
from __future__ import annotations

__all__ = [
    "DatabaseManager", "DBOptions",
    "AIManager", "ConfigManager", "ExchangeManager",
    "ReportManager", "RiskManagerAdapter", "SecurityManager",
]

# Importy opcjonalne z ochroną na brak zależności środowiskowych
try:
    from .database_manager import DatabaseManager, DBOptions  # type: ignore
except Exception:  # pragma: no cover
    DatabaseManager = None  # type: ignore
    DBOptions = None  # type: ignore

try:
    from .ai_manager import AIManager  # type: ignore
except Exception:  # pragma: no cover
    AIManager = None  # type: ignore

try:
    from .config_manager import ConfigManager  # type: ignore
except Exception:  # pragma: no cover
    ConfigManager = None  # type: ignore

try:
    from .exchange_manager import ExchangeManager  # type: ignore
except Exception:  # pragma: no cover
    ExchangeManager = None  # type: ignore

try:
    from .report_manager import ReportManager  # type: ignore
except Exception:  # pragma: no cover
    ReportManager = None  # type: ignore

try:
    from .risk_manager_adapter import RiskManagerAdapter  # type: ignore
except Exception:  # pragma: no cover
    RiskManagerAdapter = None  # type: ignore

try:
    from .security_manager import SecurityManager  # type: ignore
except Exception:  # pragma: no cover
    SecurityManager = None  # type: ignore
