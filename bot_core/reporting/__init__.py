"""Moduły raportowania i archiwizacji danych operacyjnych."""

from bot_core.reporting.model_quality import (
    DEFAULT_QUALITY_DIR as MODEL_QUALITY_DIR,
    load_latest_quality_payload,
    persist_quality_report,
)
from bot_core.reporting.paper import generate_daily_paper_report
from bot_core.reporting.optimization import (
    export_report as export_optimization_report,
    render_html_report as render_optimization_html,
    render_json_report as render_optimization_json,
)
from bot_core.reporting.tco import (
    TcoCategoryBreakdown,
    TcoCostItem,
    TcoSummary,
    TcoUsageMetrics,
    aggregate_costs,
    load_cost_items,
    write_summary_csv,
    write_summary_json,
    write_summary_signature,
)
from . import ui_bridge

__all__ = [
    "MODEL_QUALITY_DIR",
    "load_latest_quality_payload",
    "persist_quality_report",
    "generate_daily_paper_report",
    "render_optimization_html",
    "render_optimization_json",
    "export_optimization_report",
    "TcoCategoryBreakdown",
    "TcoCostItem",
    "TcoSummary",
    "TcoUsageMetrics",
    "aggregate_costs",
    "load_cost_items",
    "write_summary_csv",
    "write_summary_json",
    "write_summary_signature",
    "ui_bridge",
]
