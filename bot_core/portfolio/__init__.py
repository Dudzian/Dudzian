"""Moduły zarządzania portfelem (Stage6) z wsteczną zgodnością.

Eksportuje:
- Pełne API Stage6 (governor/hypercare/io/decision_log).
- Dodatkowe typy z .models (gałąź main), jeśli są obecne w pakiecie.
"""

from __future__ import annotations

# --- Stage6 / rozszerzone API ---
from bot_core.portfolio.decision_log import PortfolioDecisionLog
from bot_core.portfolio.governor import (
    AssetPortfolioGovernorConfig,
    PortfolioAdjustment,
    PortfolioAdvisory,
    PortfolioAssetConfig,
    PortfolioDecision,
    PortfolioDriftTolerance,
    PortfolioGovernor,
    PortfolioRiskBudgetConfig,
    PortfolioSloOverrideConfig,
    StrategyPortfolioGovernor,
    StrategyPortfolioGovernorConfig,
)
from bot_core.portfolio.hypercare import (
    PortfolioCycleConfig,
    PortfolioCycleInputs,
    PortfolioCycleOutputConfig,
    PortfolioCycleResult,
    PortfolioHypercareCycle,
)
from bot_core.portfolio.io import (
    load_allocations_file,
    load_json_or_yaml,
    load_market_intel_report,
    parse_market_intel_payload,
    parse_slo_status_payload,
    parse_stress_overrides_payload,
    resolve_decision_log_config,
)
from bot_core.portfolio.scheduler import (
    CopyTradeInstruction,
    CopyTradingFollowerConfig,
    MultiPortfolioScheduler,
    PortfolioBinding,
    PortfolioScheduleResult,
    RebalanceInstruction,
    SchedulerEvent,
    StrategyHealthMonitor,
)

PortfolioGovernorConfig = AssetPortfolioGovernorConfig

# --- Prostszе typy z gałęzi main (opcjonalnie) ---
# Nie wszystkie repozytoria mają plik .models – import warunkowy zachowuje kompatybilność.
try:  # pragma: no cover - opcjonalne
    from .models import (
        PortfolioRebalanceDecision,
        StrategyAllocationDecision,
        StrategyMetricsSnapshot,
    )

    _LEGACY_EXPORTS = (
        "PortfolioRebalanceDecision",
        "StrategyAllocationDecision",
        "StrategyMetricsSnapshot",
    )
except Exception:  # pragma: no cover - brak .models w danej dystrybucji
    _LEGACY_EXPORTS = tuple()

__all__ = [
    # Stage6 – governor/hypercare/io/decision_log
    "PortfolioAdjustment",
    "PortfolioAdvisory",
    "PortfolioAssetConfig",
    "PortfolioCycleConfig",
    "PortfolioCycleInputs",
    "PortfolioCycleOutputConfig",
    "PortfolioCycleResult",
    "PortfolioDecision",
    "PortfolioDecisionLog",
    "PortfolioDriftTolerance",
    "PortfolioGovernor",
    "PortfolioGovernorConfig",
    "PortfolioHypercareCycle",
    "PortfolioRiskBudgetConfig",
    "PortfolioSloOverrideConfig",
    "StrategyPortfolioGovernor",
    "CopyTradeInstruction",
    "CopyTradingFollowerConfig",
    "MultiPortfolioScheduler",
    "PortfolioBinding",
    "PortfolioScheduleResult",
    "RebalanceInstruction",
    "SchedulerEvent",
    "StrategyHealthMonitor",
    "load_allocations_file",
    "load_json_or_yaml",
    "load_market_intel_report",
    "parse_market_intel_payload",
    "parse_slo_status_payload",
    "parse_stress_overrides_payload",
    "resolve_decision_log_config",
    # Opcjonalnie – typy z gałęzi main
    *_LEGACY_EXPORTS,
]
