"""Warstwa ryzyka — scalona wersja HEAD + main.

Łączy eksporty z obu gałęzi:
- bazowe interfejsy ryzyka, symulacje, profile
- pełny Stress Lab (konfiguracja/evaluator/kalibracja + narzędzia I/O) z HEAD – jeśli dostępny
- alternatywne klasy Stress Lab (MarketBaseline/StressLab/… ) z main – jeśli dostępne
"""

from __future__ import annotations

# --- Bazowe interfejsy / repo / silniki --------------------------------------
from bot_core.risk.base import RiskCheckResult, RiskEngine, RiskProfile, RiskRepository
from bot_core.risk.engine import InMemoryRiskRepository, ThresholdRiskEngine
from bot_core.risk.guardrails import (
    evaluate_backtest_guardrails,
    LossGuardrailConfig,
    RiskGuardrailMetricSet,
)
from bot_core.risk.portfolio_stress import (
    PortfolioStressBaseline,
    PortfolioStressPosition,
    PortfolioStressPositionResult,
    PortfolioStressScenarioResult,
    PortfolioStressReport,
    baseline_from_mapping as portfolio_stress_baseline_from_mapping,
    load_portfolio_stress_baseline,
    run_portfolio_stress,
)
from bot_core.risk.portfolio import (
    CorrelationAnalyzer,
    PositionSizing,
    RiskLevel,
    RiskManagement,
    RiskMetrics,
    VolatilityEstimator,
    backtest_risk_strategy,
    calculate_optimal_leverage,
    create_risk_manager,
)
from bot_core.risk.events import RiskDecisionEvent, RiskDecisionLog
from bot_core.risk.factory import build_risk_profile_from_config
from bot_core.risk.repository import FileRiskRepository
from bot_core.risk.simulation import (
    Candle,
    MarketDatasetLoader,
    ProfileSimulationResult,
    RiskSimulationReport,
    RiskSimulationRunner,
    SimulationSettings,
    StressTestResult,
    load_profiles_from_config,
    run_simulations_from_config,
)

# --- Stress Lab (HEAD) --------------------------------------------------------
_HAS_HEAD_STRESS = False
_HAS_HEAD_CALIB = False

try:
    # Pełne API Stress Lab z gałęzi HEAD
    from bot_core.risk.stress_lab import (
        StressLabConfig,
        StressLabEvaluator,
        StressLabReport,
        StressLabSeverityPolicy,
        StressOverrideRecommendation,
        StressScenarioInsight,
        write_overrides_csv,
        write_report_csv,
        write_report_json,
        write_report_signature,
    )

    _HAS_HEAD_STRESS = True
except Exception:  # pragma: no cover
    pass

try:
    # Moduł kalibracji z gałęzi HEAD
    from bot_core.risk.stress_lab_calibration import (
        StressLabCalibrator,
        StressLabCalibrationReport,
        StressLabCalibrationSegment,
        StressLabCalibrationSettings,
        StressLabSegmentThresholds,
        build_volume_segments,
        write_calibration_csv,
        write_calibration_json,
        write_calibration_signature,
    )

    _HAS_HEAD_CALIB = True
except Exception:  # pragma: no cover
    pass

# --- Stress Lab (main – alternatywne klasy) -----------------------------------
_HAS_MAIN_STRESS = False
try:
    from bot_core.risk.stress_lab import (  # type: ignore[no-redef]
        MarketBaseline,
        MarketStressMetrics,
        StressLab,              # alternatywny konstruktor systemu Stress Lab
        StressLabReport as _StressLabReportMain,  # alias lokalny, by uniknąć kolizji
        StressScenarioResult,
    )

    # Jeżeli wczytaliśmy już StressLabReport z HEAD, nie nadpisujemy go aliasem.
    if "StressLabReport" not in globals():
        StressLabReport = _StressLabReportMain  # type: ignore[assignment]

    _HAS_MAIN_STRESS = True
except Exception:  # pragma: no cover
    pass

# --- Profile ryzyka -----------------------------------------------------------
from bot_core.risk.profiles.aggressive import AggressiveProfile
from bot_core.risk.profiles.balanced import BalancedProfile
from bot_core.risk.profiles.conservative import ConservativeProfile
from bot_core.risk.profiles.manual import ManualProfile

# --- Publiczny interfejs (__all__) -------------------------------------------
__all__ = [
    # profile
    "AggressiveProfile",
    "BalancedProfile",
    "ConservativeProfile",
    "ManualProfile",
    # repo/silniki
    "FileRiskRepository",
    "InMemoryRiskRepository",
    "RiskCheckResult",
    "RiskEngine",
    "RiskProfile",
    "RiskRepository",
    "ThresholdRiskEngine",
    "evaluate_backtest_guardrails",
    "LossGuardrailConfig",
    "RiskGuardrailMetricSet",
    # zdarzenia/log
    "RiskDecisionEvent",
    "RiskDecisionLog",
    # fabryki/konfiguracja
    "build_risk_profile_from_config",
    # symulacje
    "Candle",
    "MarketDatasetLoader",
    "ProfileSimulationResult",
    "RiskSimulationReport",
    "RiskSimulationRunner",
    "SimulationSettings",
    "StressTestResult",
    "load_profiles_from_config",
    "run_simulations_from_config",
    # zarządzanie ryzykiem portfela
    "RiskLevel",
    "RiskMetrics",
    "PositionSizing",
    "VolatilityEstimator",
    "CorrelationAnalyzer",
    "RiskManagement",
    "create_risk_manager",
    "backtest_risk_strategy",
    "calculate_optimal_leverage",
    "PortfolioStressBaseline",
    "PortfolioStressPosition",
    "PortfolioStressPositionResult",
    "PortfolioStressScenarioResult",
    "PortfolioStressReport",
    "run_portfolio_stress",
    "load_portfolio_stress_baseline",
    "portfolio_stress_baseline_from_mapping",
]

# Eksporty Stress Lab (HEAD) – tylko jeśli moduły istnieją
if _HAS_HEAD_STRESS:
    __all__.extend(
        [
            "StressLabConfig",
            "StressLabEvaluator",
            "StressLabReport",
            "StressLabSeverityPolicy",
            "StressOverrideRecommendation",
            "StressScenarioInsight",
            "write_overrides_csv",
            "write_report_csv",
            "write_report_json",
            "write_report_signature",
        ]
    )

if _HAS_HEAD_CALIB:
    __all__.extend(
        [
            "StressLabCalibrator",
            "StressLabCalibrationReport",
            "StressLabCalibrationSegment",
            "StressLabCalibrationSettings",
            "StressLabSegmentThresholds",
            "build_volume_segments",
            "write_calibration_csv",
            "write_calibration_json",
            "write_calibration_signature",
        ]
    )

# Eksporty Stress Lab (main – alternatywne symbole) – tylko jeśli dostępne
if _HAS_MAIN_STRESS:
    __all__.extend(
        [
            "StressLab",
            "StressScenarioResult",
            "MarketStressMetrics",
            "MarketBaseline",
            # Uwaga: StressLabReport już dodany powyżej (HEAD) lub aliasowany tutaj.
            # Nie dodajemy duplikatu.
        ]
    )
