"""Tests for FUNCTIONAL-PREVIEW-10.7 read-only market data bridge snapshot fields."""

from __future__ import annotations

import json
from dataclasses import is_dataclass
from types import MappingProxyType
from typing import Any

from ui.pyside_app.preview_action_dispatch_bridge_snapshot import (
    READ_ONLY_MARKET_DATA_BRIDGE_SNAPSHOT_DECISION,
    READ_ONLY_MARKET_DATA_BRIDGE_SNAPSHOT_NEXT_STEP,
    READ_ONLY_MARKET_DATA_BRIDGE_SNAPSHOT_STATUS,
    build_paper_runtime_action_dispatch_bridge_snapshot,
)

SIMPLE_TYPES = (dict, list, str, bool, int, float, type(None))
EXPECTED_FIELDS = {
    "read_only_market_data_bridge_snapshot_status",
    "read_only_market_data_bridge_snapshot_decision",
    "read_only_market_data_bridge_snapshot_next_step",
    "read_only_market_data_bridge_snapshot_next_step_title",
    "read_only_market_data_bridge_snapshot_ready_for_block_h_8",
    "read_only_market_data_controlled_refresh_preview",
    "read_only_market_data_controlled_refresh_status",
    "read_only_market_data_controlled_refresh_next_step",
    "read_only_market_data_controlled_refresh_ready_for_block_h_7",
    "read_only_market_data_allowed_refresh_preview_count",
    "read_only_market_data_default_refresh_selection_id",
    "read_only_market_data_allowed_refresh_symbols",
    "read_only_market_data_normal_refresh_preview_symbols",
    "read_only_market_data_low_liquidity_refresh_preview_symbols",
    "read_only_market_data_stale_refresh_preview_symbols",
    "read_only_market_data_refresh_preview_boundary_summary",
    "read_only_market_data_no_refresh_summary",
    "read_only_market_data_no_fetch_summary",
    "read_only_market_data_no_network_summary",
    "read_only_market_data_no_bridge_api_change_summary",
    "read_only_market_data_bridge_snapshot_summary",
}


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


def test_bridge_snapshot_contains_all_10_7_read_only_market_data_fields() -> None:
    snapshot = build_paper_runtime_action_dispatch_bridge_snapshot()

    assert EXPECTED_FIELDS <= set(snapshot)
    assert (
        snapshot["read_only_market_data_bridge_snapshot_status"]
        == READ_ONLY_MARKET_DATA_BRIDGE_SNAPSHOT_STATUS
    )
    assert (
        snapshot["read_only_market_data_bridge_snapshot_decision"]
        == READ_ONLY_MARKET_DATA_BRIDGE_SNAPSHOT_DECISION
    )
    assert (
        snapshot["read_only_market_data_bridge_snapshot_next_step"]
        == READ_ONLY_MARKET_DATA_BRIDGE_SNAPSHOT_NEXT_STEP
        == "FUNCTIONAL-PREVIEW-10.8"
    )
    assert snapshot["read_only_market_data_bridge_snapshot_ready_for_block_h_8"] is True


def test_bridge_snapshot_read_only_market_data_payload_is_qml_safe_json() -> None:
    snapshot = build_paper_runtime_action_dispatch_bridge_snapshot()
    market_data_payload = {
        key: value for key, value in snapshot.items() if key.startswith("read_only_market_data_")
    }

    _assert_simple_types_only(market_data_payload)
    encoded = json.dumps(market_data_payload, sort_keys=True)
    assert "read_only_market_data_bridge_snapshot_ready_no_qml_actions_no_refresh" in encoded


def test_bridge_snapshot_references_10_6_controlled_refresh_preview_without_refresh() -> None:
    snapshot = build_paper_runtime_action_dispatch_bridge_snapshot()
    preview = snapshot["read_only_market_data_controlled_refresh_preview"]

    assert preview["step"] == "10.6"
    assert preview["ready_for_block_h_7"] is True
    assert preview["next_step"] == "FUNCTIONAL-PREVIEW-10.7"
    assert snapshot["read_only_market_data_controlled_refresh_ready_for_block_h_7"] is True
    assert (
        snapshot["read_only_market_data_controlled_refresh_status"]
        == "read_only_market_data_controlled_refresh_preview_ready_no_refresh_performed"
    )
    assert (
        snapshot["read_only_market_data_controlled_refresh_next_step"] == "FUNCTIONAL-PREVIEW-10.7"
    )


def test_bridge_snapshot_refresh_preview_symbol_summaries_are_static_allowlist() -> None:
    snapshot = build_paper_runtime_action_dispatch_bridge_snapshot()

    assert snapshot["read_only_market_data_allowed_refresh_preview_count"] == 4
    assert (
        snapshot["read_only_market_data_default_refresh_selection_id"] == "btc_usdt_static_fixture"
    )
    assert snapshot["read_only_market_data_allowed_refresh_symbols"] == [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "ADA/USDT",
    ]
    assert snapshot["read_only_market_data_normal_refresh_preview_symbols"] == [
        "BTC/USDT",
        "ETH/USDT",
    ]
    assert snapshot["read_only_market_data_low_liquidity_refresh_preview_symbols"] == ["SOL/USDT"]
    assert snapshot["read_only_market_data_stale_refresh_preview_symbols"] == ["ADA/USDT"]


def test_bridge_snapshot_no_refresh_no_fetch_no_network_no_bridge_api_flags() -> None:
    snapshot = build_paper_runtime_action_dispatch_bridge_snapshot()

    assert snapshot["read_only_market_data_no_refresh_summary"] == {
        "refresh_execution_allowed_now": False,
        "refresh_performed_now": False,
        "controlled_refresh_performed": False,
        "refresh_performed": False,
        "no_real_refresh": True,
    }
    assert snapshot["read_only_market_data_no_fetch_summary"] == {
        "market_data_fetch_allowed_now": False,
        "market_data_fetch_performed": False,
        "no_market_fetch": True,
    }
    assert snapshot["read_only_market_data_no_network_summary"] == {
        "network_io_allowed_now": False,
        "network_io_performed": False,
        "exchange_connection_opened": False,
        "no_network_io": True,
    }
    assert snapshot["read_only_market_data_no_bridge_api_change_summary"] == {
        "bridge_api_changes_allowed": False,
        "bridge_api_changes_performed": False,
        "new_qml_method_calls_allowed": False,
        "qml_changes_performed": False,
        "no_bridge_api_changes": True,
        "no_new_qml_method_calls": True,
    }


def test_bridge_snapshot_summary_is_data_only_qml_safe_and_ready_for_10_8() -> None:
    snapshot = build_paper_runtime_action_dispatch_bridge_snapshot()
    summary = snapshot["read_only_market_data_bridge_snapshot_summary"]

    assert summary == {
        "snapshot_status": READ_ONLY_MARKET_DATA_BRIDGE_SNAPSHOT_STATUS,
        "snapshot_decision": READ_ONLY_MARKET_DATA_BRIDGE_SNAPSHOT_DECISION,
        "snapshot_data_only": True,
        "controlled_refresh_preview_read": True,
        "qml_safe": True,
        "refresh_performed": False,
        "market_data_fetch_performed": False,
        "network_io_performed": False,
        "bridge_api_changes_performed": False,
        "qml_changes_performed": False,
        "next_step": "FUNCTIONAL-PREVIEW-10.8",
    }
