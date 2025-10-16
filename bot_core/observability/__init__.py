"""Pakiet narzędzi obserwowalności (metryki, eksport)."""
from importlib import import_module
from typing import TYPE_CHECKING, Any

from bot_core.observability.alert_overrides import (
    AlertOverride,
    AlertOverrideBuilder,
    AlertOverrideManager,
    load_overrides_document,
)
from bot_core.observability.dashboard_sync import (
    DashboardDefinition,
    build_dashboard_annotations_payload,
    load_dashboard_definition,
    load_overrides_from_document,
    save_dashboard_annotations,
)
from bot_core.observability.io import load_slo_definitions, load_slo_measurements
from bot_core.observability.metrics import (
    CounterMetric,
    GaugeMetric,
    HistogramMetric,
    MetricsRegistry,
    get_global_metrics_registry,
)
from bot_core.observability.server import MetricsHTTPServer, start_http_server
from bot_core.observability.slo import (
    SLOCompositeDefinition,
    SLOCompositeStatus,
    SLODefinition,
    SLOMeasurement,
    SLOMonitor,
    SLOReport,
    SLOStatus,
    evaluate_slo,
    write_slo_results_csv,
)

if TYPE_CHECKING:  # pragma: no cover - tylko dla type-checkera
    from bot_core.observability.hypercare import (
        BundleConfig,
        DashboardSyncConfig,
        ObservabilityCycleConfig,
        ObservabilityCycleResult,
        ObservabilityHypercareCycle,
        OverridesOutputConfig,
        SLOOutputConfig,
    )

_HYPERCARE_EXPORTS = {
    "BundleConfig",
    "DashboardSyncConfig",
    "ObservabilityCycleConfig",
    "ObservabilityCycleResult",
    "ObservabilityHypercareCycle",
    "OverridesOutputConfig",
    "SLOOutputConfig",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - mechanizm leniwy
    if name in _HYPERCARE_EXPORTS:
        module = import_module("bot_core.observability.hypercare")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(name)

__all__ = [
    "CounterMetric",
    "GaugeMetric",
    "HistogramMetric",
    "MetricsRegistry",
    "get_global_metrics_registry",
    "MetricsHTTPServer",
    "start_http_server",
    "AlertOverride",
    "AlertOverrideBuilder",
    "AlertOverrideManager",
    "load_overrides_document",
    "DashboardDefinition",
    "build_dashboard_annotations_payload",
    "load_dashboard_definition",
    "load_overrides_from_document",
    "save_dashboard_annotations",
    "BundleConfig",
    "DashboardSyncConfig",
    "ObservabilityCycleConfig",
    "ObservabilityCycleResult",
    "ObservabilityHypercareCycle",
    "OverridesOutputConfig",
    "SLOOutputConfig",
    "load_slo_definitions",
    "load_slo_measurements",
    "SLOCompositeDefinition",
    "SLOCompositeStatus",
    "SLODefinition",
    "SLOMeasurement",
    "SLOMonitor",
    "SLOReport",
    "SLOStatus",
    "evaluate_slo",
    "write_slo_results_csv",
]
