"""Infrastruktura runtime nowej architektury bota."""

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
