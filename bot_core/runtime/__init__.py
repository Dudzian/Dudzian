"""Infrastruktura runtime nowej architektury bota."""

try:  # pragma: no cover - środowiska testowe mogą nie mieć pełnego runtime
    from bot_core.runtime.bootstrap import BootstrapContext, bootstrap_environment
    from bot_core.runtime.resource_monitor import (
        ResourceBudgetEvaluation,
        ResourceBudgets,
        ResourceSample,
        evaluate_resource_sample,
    )
    from bot_core.runtime.scheduler_load_test import (
        LoadTestResult,
        LoadTestSettings,
        execute_scheduler_load_test,
    )
    from bot_core.runtime.stage5_hypercare import (
        Stage5ComplianceConfig,
        Stage5HypercareConfig,
        Stage5HypercareCycle,
        Stage5HypercareResult,
        Stage5HypercareVerificationResult,
        Stage5OemAcceptanceConfig,
        Stage5RotationConfig,
        Stage5SloConfig,
        Stage5TcoConfig,
        Stage5TrainingConfig,
        verify_stage5_hypercare_summary,
    )
    from bot_core.runtime.full_hypercare import (
        FullHypercareSummaryBuilder,
        FullHypercareSummaryConfig,
        FullHypercareSummaryResult,
        FullHypercareVerificationResult,
        verify_full_hypercare_summary,
    )
    from bot_core.runtime.stage6_hypercare import (
        Stage6HypercareConfig,
        Stage6HypercareCycle,
        Stage6HypercareResult,
        Stage6HypercareVerificationResult,
        verify_stage6_hypercare_summary,
    )
except Exception:  # pragma: no cover - fallback gdy zależności runtime są niekompletne
    BootstrapContext = None  # type: ignore
    bootstrap_environment = None  # type: ignore
    ResourceBudgetEvaluation = None  # type: ignore
    ResourceBudgets = None  # type: ignore
    ResourceSample = None  # type: ignore
    evaluate_resource_sample = None  # type: ignore
    LoadTestResult = None  # type: ignore
    LoadTestSettings = None  # type: ignore
    execute_scheduler_load_test = None  # type: ignore
    Stage5ComplianceConfig = None  # type: ignore
    Stage5HypercareConfig = None  # type: ignore
    Stage5HypercareCycle = None  # type: ignore
    Stage5HypercareResult = None  # type: ignore
    Stage5HypercareVerificationResult = None  # type: ignore
    Stage5OemAcceptanceConfig = None  # type: ignore
    Stage5RotationConfig = None  # type: ignore
    Stage5SloConfig = None  # type: ignore
    Stage5TcoConfig = None  # type: ignore
    Stage5TrainingConfig = None  # type: ignore
    verify_stage5_hypercare_summary = None  # type: ignore
    FullHypercareSummaryBuilder = None  # type: ignore
    FullHypercareSummaryConfig = None  # type: ignore
    FullHypercareSummaryResult = None  # type: ignore
    FullHypercareVerificationResult = None  # type: ignore
    verify_full_hypercare_summary = None  # type: ignore
    Stage6HypercareConfig = None  # type: ignore
    Stage6HypercareCycle = None  # type: ignore
    Stage6HypercareResult = None  # type: ignore
    Stage6HypercareVerificationResult = None  # type: ignore
    verify_stage6_hypercare_summary = None  # type: ignore

# --- Metrics service (opcjonalny – zależy od dostępności gRPC i wygenerowanych stubów) ---
try:
    from bot_core.runtime.metrics_service import (  # type: ignore
        MetricsServer,
        MetricsServiceServicer,
        MetricsSnapshotStore,
        MetricsSink,
        JsonlSink,
        ReduceMotionAlertSink,
        OverlayBudgetAlertSink,
        create_server as create_metrics_server,
        build_metrics_server_from_config,
    )
except Exception:  # pragma: no cover - brak wygenerowanych stubów lub grpcio
    MetricsServer = None  # type: ignore
    MetricsServiceServicer = None  # type: ignore
    MetricsSnapshotStore = None  # type: ignore
    MetricsSink = None  # type: ignore
    JsonlSink = None  # type: ignore
    ReduceMotionAlertSink = None  # type: ignore
    OverlayBudgetAlertSink = None  # type: ignore
    create_metrics_server = None  # type: ignore
    build_metrics_server_from_config = None  # type: ignore

# --- Risk service (opcjonalny – zależy od wygenerowanych stubów) ---
try:
    from bot_core.runtime.risk_service import (  # type: ignore
        RiskExposure,
        RiskServer,
        RiskServiceServicer,
        RiskSnapshot,
        RiskSnapshotBuilder,
        RiskSnapshotPublisher,
        RiskSnapshotStore,
        build_risk_server_from_config,
    )
except Exception:  # pragma: no cover - brak stubów risk service
    RiskExposure = None  # type: ignore
    RiskServer = None  # type: ignore
    RiskServiceServicer = None  # type: ignore
    RiskSnapshot = None  # type: ignore
    RiskSnapshotBuilder = None  # type: ignore
    RiskSnapshotPublisher = None  # type: ignore
    RiskSnapshotStore = None  # type: ignore
    build_risk_server_from_config = None  # type: ignore

try:  # pragma: no cover - eksporter metryk jest opcjonalny
    from bot_core.runtime.risk_metrics import RiskMetricsExporter  # type: ignore
except Exception:  # pragma: no cover - brak zależności opcjonalnych
    RiskMetricsExporter = None  # type: ignore

# --- Kontrolery / pipeline (opcjonalne – różnice między gałęziami) ---
try:
    from bot_core.runtime.controller import TradingController as _TradingController  # type: ignore
except Exception:
    _TradingController = None  # type: ignore

try:
    from bot_core.runtime.controller import DailyTrendController as _DailyTrendController  # type: ignore
except Exception:
    _DailyTrendController = None  # type: ignore

try:
    from bot_core.runtime.realtime import (  # type: ignore
        DailyTrendRealtimeRunner as _DailyTrendRealtimeRunner
    )
except Exception:  # pragma: no cover - starsze gałęzie mogą nie mieć modułu realtime
    _DailyTrendRealtimeRunner = None  # type: ignore

try:
    from bot_core.runtime.pipeline import (  # type: ignore
        DailyTrendPipeline,
        build_daily_trend_pipeline,
        create_trading_controller,
    )
except Exception:  # pragma: no cover - starsze gałęzie mogą nie mieć modułu pipeline
    DailyTrendPipeline = None  # type: ignore
    build_daily_trend_pipeline = None  # type: ignore
    create_trading_controller = None  # type: ignore

# --- Publiczny interfejs modułu ---
__all__ = [
    "BootstrapContext",
    "bootstrap_environment",
    "ResourceBudgets",
    "ResourceSample",
    "ResourceBudgetEvaluation",
    "evaluate_resource_sample",
    "LoadTestSettings",
    "LoadTestResult",
    "execute_scheduler_load_test",
    "Stage5HypercareCycle",
    "Stage5HypercareConfig",
    "Stage5HypercareResult",
    "Stage5HypercareVerificationResult",
    "Stage5TcoConfig",
    "Stage5RotationConfig",
    "Stage5ComplianceConfig",
    "Stage5TrainingConfig",
    "Stage5SloConfig",
    "Stage5OemAcceptanceConfig",
    "verify_stage5_hypercare_summary",
    "FullHypercareSummaryBuilder",
    "FullHypercareSummaryConfig",
    "FullHypercareSummaryResult",
    "FullHypercareVerificationResult",
    "verify_full_hypercare_summary",
    "Stage6HypercareCycle",
    "Stage6HypercareConfig",
    "Stage6HypercareResult",
    "Stage6HypercareVerificationResult",
    "verify_stage6_hypercare_summary",
]

# Eksport elementów metrics service tylko jeśli są dostępne
if MetricsServer is not None:
    __all__.extend(
        [
            "MetricsServer",
            "MetricsServiceServicer",
            "MetricsSnapshotStore",
            "MetricsSink",
            "JsonlSink",
            "ReduceMotionAlertSink",
            "OverlayBudgetAlertSink",
            "create_metrics_server",
            "build_metrics_server_from_config",
        ]
    )

# Eksport elementów risk service tylko jeśli są dostępne
if RiskServer is not None:
    __all__.extend(
        [
            "RiskExposure",
            "RiskServer",
            "RiskServiceServicer",
            "RiskSnapshot",
            "RiskSnapshotBuilder",
            "RiskSnapshotPublisher",
            "RiskSnapshotStore",
            "build_risk_server_from_config",
        ]
    )

if RiskMetricsExporter is not None:
    __all__.append("RiskMetricsExporter")

# Eksportuj tylko te kontrolery, które są dostępne w danej gałęzi.
if _TradingController is None:
    # Defensywny fallback, gdy bezpośredni import się nie powiódł
    try:  # pragma: no cover
        from bot_core.runtime import controller as _controller_module  # type: ignore

        _TradingController = getattr(_controller_module, "TradingController", None)
    except Exception:  # pragma: no cover
        _TradingController = None  # type: ignore

if _TradingController is not None:
    TradingController = _TradingController  # type: ignore
    __all__.append("TradingController")

if _DailyTrendController is not None:
    DailyTrendController = _DailyTrendController  # type: ignore
    __all__.append("DailyTrendController")

if _DailyTrendRealtimeRunner is not None:
    DailyTrendRealtimeRunner = _DailyTrendRealtimeRunner  # type: ignore
    __all__.append("DailyTrendRealtimeRunner")

if DailyTrendPipeline is not None and build_daily_trend_pipeline is not None:
    __all__.extend(["DailyTrendPipeline", "build_daily_trend_pipeline"])
    if create_trading_controller is not None:
        __all__.append("create_trading_controller")
