"""Pakiet narzędzi obserwowalności (metryki, eksport)."""
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - tylko dla type-checkera
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
        DataFeedMetricSet,
        GaugeMetric,
        HistogramMetric,
        MetricsRegistry,
        get_data_feed_metrics,
        get_global_metrics_registry,
    )
    from bot_core.observability.exporters import (
        LocalPrometheusExporter,
        PrometheusExporterConfig,
    )
    from bot_core.observability.server import MetricsHTTPServer, start_http_server
    from bot_core.observability.ui_metrics import UiTelemetryPrometheusExporter
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
    from bot_core.observability.hypercare import (
        BundleConfig,
        DashboardSyncConfig,
        ObservabilityCycleConfig,
        ObservabilityCycleResult,
        ObservabilityHypercareCycle,
        OverridesOutputConfig,
        SLOOutputConfig,
    )

_LAZY_IMPORTS = {
    "AlertOverride": "bot_core.observability.alert_overrides",
    "AlertOverrideBuilder": "bot_core.observability.alert_overrides",
    "AlertOverrideManager": "bot_core.observability.alert_overrides",
    "load_overrides_document": "bot_core.observability.alert_overrides",
    "DashboardDefinition": "bot_core.observability.dashboard_sync",
    "build_dashboard_annotations_payload": "bot_core.observability.dashboard_sync",
    "load_dashboard_definition": "bot_core.observability.dashboard_sync",
    "load_overrides_from_document": "bot_core.observability.dashboard_sync",
    "save_dashboard_annotations": "bot_core.observability.dashboard_sync",
    "load_slo_definitions": "bot_core.observability.io",
    "load_slo_measurements": "bot_core.observability.io",
    "CounterMetric": "bot_core.observability.metrics",
    "DataFeedMetricSet": "bot_core.observability.metrics",
    "GaugeMetric": "bot_core.observability.metrics",
    "HistogramMetric": "bot_core.observability.metrics",
    "MetricsRegistry": "bot_core.observability.metrics",
    "get_data_feed_metrics": "bot_core.observability.metrics",
    "get_global_metrics_registry": "bot_core.observability.metrics",
    "LocalPrometheusExporter": "bot_core.observability.exporters",
    "PrometheusExporterConfig": "bot_core.observability.exporters",
    "MetricsHTTPServer": "bot_core.observability.server",
    "start_http_server": "bot_core.observability.server",
    "UiTelemetryPrometheusExporter": "bot_core.observability.ui_metrics",
    "SLOCompositeDefinition": "bot_core.observability.slo",
    "SLOCompositeStatus": "bot_core.observability.slo",
    "SLODefinition": "bot_core.observability.slo",
    "SLOMeasurement": "bot_core.observability.slo",
    "SLOMonitor": "bot_core.observability.slo",
    "SLOReport": "bot_core.observability.slo",
    "SLOStatus": "bot_core.observability.slo",
    "evaluate_slo": "bot_core.observability.slo",
    "write_slo_results_csv": "bot_core.observability.slo",
    "BundleConfig": "bot_core.observability.hypercare",
    "DashboardSyncConfig": "bot_core.observability.hypercare",
    "ObservabilityCycleConfig": "bot_core.observability.hypercare",
    "ObservabilityCycleResult": "bot_core.observability.hypercare",
    "ObservabilityHypercareCycle": "bot_core.observability.hypercare",
    "OverridesOutputConfig": "bot_core.observability.hypercare",
    "SLOOutputConfig": "bot_core.observability.hypercare",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - mechanizm leniwy
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = list(_LAZY_IMPORTS.keys())
