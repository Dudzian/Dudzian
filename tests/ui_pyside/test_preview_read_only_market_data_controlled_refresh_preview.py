from __future__ import annotations

import ast
import json
from pathlib import Path

from ui.pyside_app.preview_read_only_market_data_controlled_refresh_preview import (
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_READ_ONLY_MARKET_DATA_CONTROLLED_REFRESH_PREVIEW_SCHEMA_VERSION,
    READ_ONLY_MARKET_DATA_CONTROLLED_REFRESH_PREVIEW_DECISION,
    READ_ONLY_MARKET_DATA_CONTROLLED_REFRESH_PREVIEW_STATUS,
    build_preview_read_only_market_data_controlled_refresh_preview,
    build_preview_read_only_market_data_controlled_refresh_result,
)

EXPECTED_ALLOWED = [
    ("btc_usdt_static_fixture", "BTC/USDT", "static_fixture", "ok_preview_only"),
    ("eth_usdt_static_fixture", "ETH/USDT", "static_fixture", "ok_preview_only"),
    (
        "sol_usdt_low_liquidity_preview",
        "SOL/USDT",
        "static_fixture_low_liquidity_preview",
        "low_liquidity_preview_only",
    ),
    ("ada_usdt_stale_preview", "ADA/USDT", "static_fixture_stale_preview", "stale_preview_only"),
]
EXPECTED_PREVIEW_FIELDS = {
    "selection_id",
    "symbol",
    "source",
    "quality_status",
    "refresh_preview_status",
    "refresh_preview_label",
    "refresh_allowed",
    "refresh_performed",
    "network_io_allowed",
    "network_io_performed",
    "market_data_fetch_allowed",
    "market_data_fetch_performed",
    "execution_allowed",
    "execution_performed",
    "account_data_allowed",
    "order_or_fill_data_allowed",
    "credentials_or_secrets_allowed",
}
FALSE_PREVIEW_FLAGS = {
    "refresh_allowed",
    "refresh_performed",
    "network_io_allowed",
    "network_io_performed",
    "market_data_fetch_allowed",
    "market_data_fetch_performed",
    "execution_allowed",
    "execution_performed",
    "account_data_allowed",
    "order_or_fill_data_allowed",
    "credentials_or_secrets_allowed",
}
DENIED_FIELD_NAMES = {
    "account_id",
    "balance",
    "position",
    "order_id",
    "fill_id",
    "api_key",
    "secret",
    "private_endpoint",
}
EXPECTED_REJECTED_IDS = [
    "",
    "unknown_pair",
    "live_btc_usdt",
    "fetch_market_data",
    "refresh_market_data",
    "account_balance",
]
EXPECTED_BLOCKED_CAPABILITIES = [
    "market data adapter implementation now",
    "network I/O",
    "live market data fetch now",
    "controlled refresh execution now",
    "exchange API connection",
    "private account endpoint access",
    "account balance fetch",
    "positions fetch",
    "orders fetch",
    "fills fetch",
    "order generation",
    "order submission",
    "fill simulation",
    "lifecycle mutation",
    "runtime loop",
    "scheduler",
    "audit export",
    "bridge API changes",
    "TradingController / DecisionEnvelope",
    "live/testnet/account/secrets/export/cloud",
    "QML changes / new QML calls",
    "EXE packaging",
]
EXPECTED_SOURCE_BOUNDARIES = [
    "no PySide import",
    "no QML import",
    "no runtime loop import",
    "no scheduler import",
    "no TradingController import",
    "no DecisionEnvelope import",
    "no strategy/AI/scoring/recommendation import",
    "no order module import",
    "no live adapter import",
    "no testnet adapter import",
    "no market data runtime adapter import",
    "no account module import",
    "no secrets module import",
    "no filesystem I/O",
    "no network I/O",
    "no QML changes",
    "no bridge API changes",
    "no .bat changes",
    "no app.py changes",
    "no dependency declarations changes",
    "no workflow changes",
]


def test_controlled_refresh_preview_is_plain_json_serializable_dict() -> None:
    preview = build_preview_read_only_market_data_controlled_refresh_preview()

    assert isinstance(preview, dict)
    assert preview["schema_version"] == (
        PREVIEW_READ_ONLY_MARKET_DATA_CONTROLLED_REFRESH_PREVIEW_SCHEMA_VERSION
    )
    assert json.loads(json.dumps(preview)) == preview


def test_status_decision_next_step_and_selection_gate_reference_are_correct() -> None:
    preview = build_preview_read_only_market_data_controlled_refresh_preview()

    assert preview["block"] == "H"
    assert preview["step"] == "10.6"
    assert (
        preview["market_data_controlled_refresh_preview_status"]
        == READ_ONLY_MARKET_DATA_CONTROLLED_REFRESH_PREVIEW_STATUS
    )
    assert (
        preview["market_data_controlled_refresh_preview_decision"]
        == READ_ONLY_MARKET_DATA_CONTROLLED_REFRESH_PREVIEW_DECISION
    )
    assert preview["ready_for_block_h_7"] is True
    assert preview["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-10.7"
    assert preview["next_step_title"] == NEXT_STEP_TITLE
    assert preview["selection_gate_reference"] == {
        "schema_version": "preview_read_only_market_data_selection_gate.v1",
        "market_data_selection_gate_kind": "functional_preview_block_h_read_only_market_data_selection_gate",
        "market_data_selection_gate_status": "read_only_market_data_selection_gate_ready_no_refresh_no_network_io",
        "market_data_selection_gate_decision": "BUILD_SELECTION_GATE_ONLY_NO_REFRESH_NO_NETWORK_IO_NO_QML_ACTIONS",
        "ready_for_block_h_6": True,
        "next_step": "FUNCTIONAL-PREVIEW-10.6",
        "next_step_title": "READ-ONLY MARKET DATA CONTROLLED REFRESH PREVIEW",
        "status": "ready_for_functional_preview_10_6_read_only_market_data_controlled_refresh_preview",
    }


def test_scope_and_metadata_are_preview_only_static_non_executable() -> None:
    preview = build_preview_read_only_market_data_controlled_refresh_preview()
    scope = preview["controlled_refresh_preview_scope"]
    metadata = preview["controlled_refresh_preview"]

    assert scope["scope_name"] == "read_only_market_data_controlled_refresh_preview"
    for key, value in scope.items():
        if key == "scope_name":
            continue
        assert value is (
            key
            in {
                "controlled_refresh_preview_only",
                "selection_gate_reference_required",
                "allowed_refresh_previews_static_only",
                "refresh_result_preview_only",
            }
        )
    assert metadata["controlled_refresh_preview_available"] is True
    assert metadata["controlled_refresh_preview_static_only"] is True
    assert metadata["controlled_refresh_preview_executable"] is False
    assert metadata["allowed_refresh_preview_count"] == 4
    assert metadata["default_selection_id"] == "btc_usdt_static_fixture"
    assert metadata["refresh_source_mode"] == "local_static_selection_preview_no_network_io"


def test_allowed_refresh_previews_are_exact_and_all_flags_false() -> None:
    previews = build_preview_read_only_market_data_controlled_refresh_preview()[
        "allowed_refresh_previews"
    ]

    assert len(previews) == 4
    for preview, (selection_id, symbol, source, quality) in zip(
        previews, EXPECTED_ALLOWED, strict=True
    ):
        assert set(preview) == EXPECTED_PREVIEW_FIELDS
        assert DENIED_FIELD_NAMES.isdisjoint(preview)
        assert preview["selection_id"] == selection_id
        assert preview["symbol"] == symbol
        assert preview["source"] == source
        assert preview["quality_status"] == quality
        assert preview["refresh_preview_status"] == "accepted_preview_only_no_refresh_performed"
        for flag in FALSE_PREVIEW_FLAGS:
            assert preview[flag] is False


def test_rejected_examples_boundary_summary_evidence_and_checks_are_complete() -> None:
    preview = build_preview_read_only_market_data_controlled_refresh_preview()
    rejected = preview["rejected_refresh_preview_examples"]

    assert [item["selection_id"] for item in rejected] == EXPECTED_REJECTED_IDS
    for item in rejected:
        assert item["refresh_preview_status"] == "rejected_fail_closed_no_refresh_performed"
        for flag in FALSE_PREVIEW_FLAGS:
            assert item[flag] is False

    summary = preview["refresh_preview_boundary_summary"]
    assert summary == {
        "allowed_refresh_preview_count": 4,
        "rejected_example_count": 6,
        "allowed_symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"],
        "normal_preview_symbols": ["BTC/USDT", "ETH/USDT"],
        "low_liquidity_preview_symbols": ["SOL/USDT"],
        "stale_preview_symbols": ["ADA/USDT"],
        "all_allowed_previews_preview_only": True,
        "all_allowed_previews_no_refresh_performed": True,
        "all_allowed_previews_no_network": True,
        "all_allowed_previews_no_fetch": True,
        "all_allowed_previews_no_execution": True,
        "all_allowed_previews_no_account_data": True,
        "all_allowed_previews_no_order_or_fill_data": True,
        "all_allowed_previews_no_credentials_or_secrets": True,
        "unknown_previews_rejected_fail_closed": True,
        "live_or_private_previews_rejected_fail_closed": True,
        "fetch_or_refresh_requests_rejected_fail_closed": True,
    }
    evidence = preview["no_refresh_no_fetch_no_execution_evidence"]
    for key, value in evidence.items():
        assert value is (
            key
            in {
                "controlled_refresh_preview_evaluated",
                "selection_gate_read",
                "allowed_refresh_previews_built_from_static_selection_gate",
            }
        )
    checks = preview["boundary_checks"]
    true_keys = {
        "local_only",
        "block_h_controlled_refresh_preview_only",
        "selection_gate_reference_valid",
        "read_only_market_data_scope_defined",
        "allowed_refresh_previews_static_only",
        "public_market_data_allowed_future_only",
        "recorded_replay_allowed_future_only",
        "exe_direction_preserved",
        "ready_for_block_h_7",
    }
    for key, value in checks.items():
        assert value is (key in true_keys)


def test_blocked_capabilities_source_boundaries_future_steps_and_status_are_complete() -> None:
    preview = build_preview_read_only_market_data_controlled_refresh_preview()

    assert preview["blocked_capabilities"] == EXPECTED_BLOCKED_CAPABILITIES
    assert preview["source_boundaries"] == EXPECTED_SOURCE_BOUNDARIES
    assert preview["future_steps"] == [
        "functional_preview_10_7_read_only_market_data_bridge_snapshot",
        "functional_preview_10_8_read_only_market_data_closure_audit",
        "functional_preview_10_9_block_h_closure",
    ]
    assert preview["status"] == (
        "ready_for_functional_preview_10_7_read_only_market_data_bridge_snapshot"
    )


def test_result_helper_accepts_only_allowlist_and_rejects_everything_else() -> None:
    for selection_id, symbol, source, quality in EXPECTED_ALLOWED:
        result = build_preview_read_only_market_data_controlled_refresh_result(selection_id)
        assert result["result_status"] == "accepted_preview_only_no_refresh_performed"
        assert result["selection_allowed"] is True
        assert result["selection_id"] == selection_id
        assert result["symbol"] == symbol
        assert result["source"] == source
        assert result["quality_status"] == quality
        for key, value in result.items():
            if key in {"result_status", "selection_id", "symbol", "source", "quality_status"}:
                continue
            assert value is (key == "selection_allowed")

    for selection_id in [None, *EXPECTED_REJECTED_IDS, "private_endpoint", "order_id"]:
        result = build_preview_read_only_market_data_controlled_refresh_result(selection_id)
        assert result["result_status"] == "rejected_fail_closed_no_refresh_performed"
        assert result["selection_allowed"] is False
        for key, value in result.items():
            if key in {"result_status", "selection_id", "reason"}:
                continue
            assert value is False


def test_helper_imports_only_selection_gate_and_has_no_runtime_or_io_calls() -> None:
    source_path = Path("ui/pyside_app/preview_read_only_market_data_controlled_refresh_preview.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports: list[str] = []
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                calls.append(func.id)
            elif isinstance(func, ast.Attribute):
                calls.append(func.attr)

    assert imports == [
        "__future__",
        "typing",
        "ui.pyside_app.preview_read_only_market_data_selection_gate",
    ]
    forbidden_import_tokens = [
        "PySide",
        "QML",
        "runtime",
        "scheduler",
        "TradingController",
        "DecisionEnvelope",
        "order",
        "live",
        "testnet",
        "adapter",
        "account",
        "secrets",
        "requests",
        "urllib",
        "httpx",
        "aiohttp",
        "socket",
        "websocket",
    ]
    assert not any(token in imported for token in forbidden_import_tokens for imported in imports)
    forbidden_calls = {
        "open",
        "read_text",
        "write_text",
        "requests",
        "subprocess",
        "urllib",
        "httpx",
        "aiohttp",
        "socket",
        "websocket",
        "QQmlApplicationEngine",
        "TradingController",
        "DecisionEnvelope",
        "start_runtime",
        "start_loop",
        "submit_order",
        "place_order",
        "create_order",
        "send_order",
        "fill_order",
        "fetch_market_data",
        "fetch_balance",
        "fetch_account",
        "refresh_market_data",
        "export",
    }
    assert forbidden_calls.isdisjoint(calls)
