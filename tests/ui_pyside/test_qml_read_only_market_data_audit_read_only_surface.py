"""FUNCTIONAL-PREVIEW-10.4 read-only market-data audit UI surface guards."""

from __future__ import annotations

import json
from dataclasses import is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ui.pyside_app.preview_action_dispatch_bridge_snapshot import (
    READ_ONLY_MARKET_DATA_UI_SURFACE_STATUS,
    build_paper_runtime_action_dispatch_bridge_snapshot,
)
from ui.pyside_app.preview_read_only_market_data_audit_envelope import (
    READ_ONLY_MARKET_DATA_AUDIT_ENVELOPE_STATUS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OPERATOR_DASHBOARD = REPO_ROOT / "ui" / "pyside_app" / "qml" / "views" / "OperatorDashboard.qml"
QML_ROOT = REPO_ROOT / "ui" / "pyside_app" / "qml"
ALLOWED_PREVIEW_SELECT_ACTION_CALL = 'paperRuntimeActionDispatchBridge.previewSelectAction("paper_runtime_snapshot_refresh_requested")'
EXPECTED_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"]
SIMPLE_TYPES = (dict, list, str, bool, int, float, type(None))


def _source(path: Path = OPERATOR_DASHBOARD) -> str:
    return path.read_text(encoding="utf-8")


def _qml_files() -> tuple[Path, ...]:
    return tuple(sorted(QML_ROOT.rglob("*.qml")))


def _assert_simple_types_only(value: object) -> None:
    assert isinstance(value, SIMPLE_TYPES)
    assert not is_dataclass(value)
    assert not isinstance(value, MappingProxyType)
    if isinstance(value, dict):
        for key, nested in value.items():
            assert isinstance(key, str)
            _assert_simple_types_only(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_simple_types_only(nested)


def _market_data_card_source() -> str:
    source = _source()
    start = source.index("operatorDashboardReadOnlyMarketDataAuditReadOnlyCard")
    end = source.index("operatorDashboardDecisionEngineDryRunAuditCard", start)
    return source[start:end]


def test_snapshot_contains_qml_safe_read_only_market_data_audit_fields() -> None:
    snapshot = build_paper_runtime_action_dispatch_bridge_snapshot()

    _assert_simple_types_only(snapshot)
    json.dumps(snapshot, sort_keys=True)
    assert snapshot["read_only_market_data_audit_envelope"]["step"] == "10.3"
    assert (
        snapshot["read_only_market_data_audit_status"]
        == READ_ONLY_MARKET_DATA_AUDIT_ENVELOPE_STATUS
    )
    assert snapshot["read_only_market_data_audit_next_step"] == "FUNCTIONAL-PREVIEW-10.4"
    assert snapshot["read_only_market_data_audit_ready_for_ui_surface"] is True
    assert snapshot["read_only_market_data_audit_ready_for_block_h_4"] is True
    assert snapshot["read_only_market_data_audit_event_count"] == 4
    assert snapshot["read_only_market_data_audited_symbols"] == EXPECTED_SYMBOLS
    assert snapshot["read_only_market_data_normal_preview_symbols"] == ["BTC/USDT", "ETH/USDT"]
    assert snapshot["read_only_market_data_low_liquidity_preview_symbols"] == ["SOL/USDT"]
    assert snapshot["read_only_market_data_stale_preview_symbols"] == ["ADA/USDT"]


def test_snapshot_market_data_no_network_no_fetch_no_export_summaries_are_true() -> None:
    snapshot = build_paper_runtime_action_dispatch_bridge_snapshot()

    assert snapshot["read_only_market_data_no_network_summary"] == {
        "network_io_allowed_now": False,
        "network_io_performed": False,
        "exchange_connection_opened": False,
        "no_network_io": True,
    }
    assert snapshot["read_only_market_data_no_fetch_summary"] == {
        "market_data_fetch_allowed_now": False,
        "market_data_fetch_performed": False,
        "no_market_fetch": True,
    }
    assert snapshot["read_only_market_data_no_export_summary"] == {
        "audit_export_allowed_now": False,
        "audit_export_performed": False,
        "export_performed": False,
        "no_audit_export": True,
    }
    assert snapshot["read_only_market_data_quality_summary"] == {
        "normal_preview_count": 2,
        "low_liquidity_preview_count": 1,
        "stale_preview_count": 1,
    }
    assert (
        snapshot["read_only_market_data_ui_read_only_summary"]["ui_surface_status"]
        == READ_ONLY_MARKET_DATA_UI_SURFACE_STATUS
    )
    assert snapshot["read_only_market_data_ui_read_only_summary"]["qml_method_calls_added"] is False


def test_qml_read_only_market_data_audit_card_and_labels_exist() -> None:
    source = _source()

    for token in (
        "operatorDashboardReadOnlyMarketDataAuditReadOnlyCard",
        "operatorDashboardReadOnlyMarketDataAuditStatusLabel",
        "operatorDashboardReadOnlyMarketDataAuditEventCountLabel",
        "operatorDashboardReadOnlyMarketDataAuditSymbolsLabel",
        "operatorDashboardReadOnlyMarketDataAuditQualityLabel",
        "operatorDashboardReadOnlyMarketDataAuditNoNetworkLabel",
        "operatorDashboardReadOnlyMarketDataAuditNoFetchLabel",
        "operatorDashboardReadOnlyMarketDataAuditNoExportLabel",
        "operatorDashboardReadOnlyMarketDataAuditNextStepLabel",
        "Read-only market data audit",
        "Audit events: %1",
        "Symbols: %1",
        "Quality: normal: %1, low-liquidity: %2, stale: %3",
        "No network I/O: %1",
        "No market fetch: %1",
        "No audit export: %1",
        "Next step: %1",
    ):
        assert token in source


def test_qml_read_only_market_data_card_has_no_interactive_controls_or_bridge_calls() -> None:
    card = _market_data_card_source()

    for forbidden in (
        "Button",
        "IconButton",
        "MouseArea",
        "TapHandler",
        "onClicked",
        "paperRuntimeActionDispatchBridge.",
        "previewSelectAction(",
        "previewSelectSourceControl",
        "resetPreviewSelection",
        "start_runtime",
        "start_loop",
        "stop_runtime",
        "pause_runtime",
        "resume_runtime",
        "submit_order",
        "place_order",
        "create_order",
        "send_order",
        "fill_order",
        "fetch_market_data",
        "fetch_balance",
        "fetch_account",
        "runtime_loop",
        "scheduler",
        "TradingController",
        "DecisionEnvelope",
    ):
        assert forbidden not in card


def test_qml_keeps_single_allowed_preview_select_action_call() -> None:
    all_qml = "\n".join(_source(path) for path in _qml_files())

    assert all_qml.count("paperRuntimeActionDispatchBridge.previewSelectAction(") == 1
    assert all_qml.count(ALLOWED_PREVIEW_SELECT_ACTION_CALL) == 1
    assert "paperRuntimeActionDispatchBridge.previewSelectSourceControl" not in all_qml
    assert "paperRuntimeActionDispatchBridge.resetPreviewSelection" not in all_qml
