"""Infrastruktura runtime nowej architektury bota."""

from bot_core.runtime.bootstrap import BootstrapContext, bootstrap_environment

try:
    from bot_core.runtime.metrics_service import (  # type: ignore
        MetricsServer,
        MetricsServiceServicer,
        MetricsSnapshotStore,
        MetricsSink,
        JsonlSink,
        create_server as create_metrics_server,
        build_metrics_server_from_config,
    )
except Exception:  # pragma: no cover - brak wygenerowanych stubów lub grpcio
    MetricsServer = None  # type: ignore
    MetricsServiceServicer = None  # type: ignore
    MetricsSnapshotStore = None  # type: ignore
    MetricsSink = None  # type: ignore
    JsonlSink = None  # type: ignore
    create_metrics_server = None  # type: ignore
    build_metrics_server_from_config = None  # type: ignore

# Kontrolery mogą się różnić między gałęziami – importujemy opcjonalnie.
try:
    from bot_core.runtime.controller import TradingController as _TradingController  # type: ignore
except Exception:
    _TradingController = None  # type: ignore

try:
    from bot_core.runtime.controller import DailyTrendController as _DailyTrendController  # type: ignore
except Exception:
    _DailyTrendController = None  # type: ignore

try:
    from bot_core.runtime.realtime import DailyTrendRealtimeRunner as _DailyTrendRealtimeRunner  # type: ignore
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

__all__ = ["BootstrapContext", "bootstrap_environment"]

if MetricsServer is not None:
    __all__.extend(
        [
            "MetricsServer",
            "MetricsServiceServicer",
            "MetricsSnapshotStore",
            "MetricsSink",
            "JsonlSink",
            "create_metrics_server",
            "build_metrics_server_from_config",
        ]
    )

# Eksportuj tylko te kontrolery, które są dostępne w danej gałęzi.
if _TradingController is None:
    try:  # pragma: no cover - defensywny fallback, gdy pierwszy import się nie powiedzie
        from bot_core.runtime import controller as _controller_module  # type: ignore

        _TradingController = getattr(_controller_module, "TradingController", None)
    except Exception:  # pragma: no cover - brak dostępnego kontrolera w starszej gałęzi
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
