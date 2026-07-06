"""Tests for FUNCTIONAL-PREVIEW-12.4 Block J risk governor gate matrix."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_risk_governor_gate_matrix import (
    BLOCK_ID,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_RISK_GOVERNOR_GATE_MATRIX_KIND,
    PREVIEW_RISK_GOVERNOR_GATE_MATRIX_SCHEMA_VERSION,
    READY_FOR_BLOCK_J_5,
    RISK_GOVERNOR_GATE_MATRIX_DECISION,
    RISK_GOVERNOR_GATE_MATRIX_STATUS,
    STATUS,
    STEP_ID,
    build_preview_risk_governor_gate_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "ui" / "pyside_app" / "preview_risk_governor_gate_matrix.py"

TOP_LEVEL_FIELDS = [
    "schema_version",
    "risk_governor_gate_matrix_kind",
    "block",
    "step",
    "risk_governor_gate_matrix_status",
    "risk_governor_gate_matrix_decision",
    "ready_for_block_j_5",
    "next_step",
    "next_step_title",
    "risk_contract_reference",
    "risk_limits_read_model_reference",
    "risk_limits_static_fixture_reference",
    "kill_switch_read_model_reference",
    "risk_governor_gate_matrix_scope",
    "risk_governor_gate_matrix_entries",
    "default_risk_governor_gate_matrix_selection",
    "risk_governor_gate_matrix_summary",
    "risk_governor_gate_matrix_cross_reference",
    "gate_matrix_contract",
    "blocked_risk_governor_gate_matrix_capabilities",
    "risk_governor_gate_matrix_boundaries",
    "non_activation_evidence",
    "source_boundaries",
    "future_steps",
    "status",
]
GATE_IDS = [
    "contract_present_gate",
    "limits_read_model_present_gate",
    "static_fixture_present_gate",
    "kill_switch_read_model_present_gate",
    "order_flow_blocked_gate",
    "private_endpoint_blocked_gate",
    "network_blocked_gate",
    "live_trading_blocked_gate",
]
ENTRY_IDS = [f"risk_governor_gate_matrix_{gate_id}" for gate_id in GATE_IDS]
ENTRY_FIELDS = [
    "gate_matrix_entry_id",
    "source_gate_id",
    "display_name",
    "gate_classification",
    "required_prior_step",
    "required_prior_artifact",
    "gate_condition",
    "gate_state_now",
    "gate_result_now",
    "blocks_runtime_until_future_gate",
    "blocks_order_flow_now",
    "blocks_private_endpoint_now",
    "blocks_network_now",
    "blocks_live_trading_now",
    "eligible_for_12_5_closure_audit",
    "operator_visibility",
    "safe_for_offline_tests",
    "notes",
]
EXPECTED_GATE_VALUES = {
    "contract_present_gate": (
        "Risk contract present gate",
        "FUNCTIONAL-PREVIEW-12.0",
        "preview_risk_governor_limits_kill_switch_contract",
        "risk contract exists and remains contract-only",
    ),
    "limits_read_model_present_gate": (
        "Risk limits read model present gate",
        "FUNCTIONAL-PREVIEW-12.1",
        "preview_risk_governor_limits_read_model",
        "limits read model exists and remains read-only",
    ),
    "static_fixture_present_gate": (
        "Risk limits static fixture present gate",
        "FUNCTIONAL-PREVIEW-12.2",
        "preview_risk_limits_static_fixture",
        "static fixture exists and remains example-only",
    ),
    "kill_switch_read_model_present_gate": (
        "Kill switch read model present gate",
        "FUNCTIONAL-PREVIEW-12.3",
        "preview_kill_switch_read_model",
        "kill switch read model exists and cannot trigger runtime",
    ),
    "order_flow_blocked_gate": (
        "Order flow blocked gate",
        "FUNCTIONAL-PREVIEW-12.4",
        "risk_governor_gate_matrix",
        "order generation and submission remain blocked",
    ),
    "private_endpoint_blocked_gate": (
        "Private endpoint blocked gate",
        "FUNCTIONAL-PREVIEW-12.4",
        "risk_governor_gate_matrix",
        "private endpoint access remains blocked",
    ),
    "network_blocked_gate": (
        "Network blocked gate",
        "FUNCTIONAL-PREVIEW-12.4",
        "risk_governor_gate_matrix",
        "network, DNS, HTTP and WebSocket access remain blocked",
    ),
    "live_trading_blocked_gate": (
        "Live trading blocked gate",
        "FUNCTIONAL-PREVIEW-12.4",
        "risk_governor_gate_matrix",
        "live production trading remains blocked",
    ),
}
BLOCKED_CAPABILITIES = [
    "risk runtime enforcement",
    "limit runtime enforcement",
    "kill switch runtime trigger",
    "manual kill switch trigger",
    "automatic kill switch trigger",
    "kill switch state mutation",
    "order generation",
    "order submission",
    "order cancel",
    "order replace",
    "position mutation",
    "private endpoint access",
    "account read",
    "balance read",
    "positions read",
    "orders read",
    "fills read",
    "market data read",
    "adapter instantiation",
    "adapter runtime wiring",
    "runtime loop",
    "scheduler",
    "network I/O",
    "DNS lookup",
    "HTTP request",
    "WebSocket connection",
    "credential read",
    "secret read",
    "secure store read",
    "secure store write",
    "config file read",
    "config discovery",
    "YAML parse",
    "JSON parse",
    "environment variable read",
    "TradingController change",
    "DecisionEnvelope change",
    "QML action dispatch",
    "bridge API changes",
    "PyInstaller/EXE packaging",
]
SOURCE_BOUNDARIES = [
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
    "no sandbox adapter import",
    "no exchange adapter runtime import",
    "no account module import",
    "no secrets module import",
    "no security store import",
    "no filesystem I/O",
    "no config file read",
    "no config discovery",
    "no YAML parse",
    "no JSON parse",
    "no environment variable read",
    "no credential read",
    "no credential validation",
    "no secret material handling",
    "no secure store read",
    "no secure store write",
    "no real market data read",
    "no private endpoint access",
    "no account read",
    "no balance read",
    "no positions read",
    "no orders read",
    "no fills read",
    "no order generation",
    "no order submission",
    "no order cancel",
    "no order replace",
    "no position mutation",
    "no network I/O",
    "no DNS lookup",
    "no HTTP request",
    "no WebSocket connection",
    "no QML changes",
    "no bridge API changes",
    "no .bat changes",
    "no app.py changes",
    "no dependency declarations changes",
    "no workflow changes",
]


def _assert_plain(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            assert isinstance(key, str)
            _assert_plain(nested)
        return
    if isinstance(value, list):
        for nested in value:
            _assert_plain(nested)
        return
    assert value is None or isinstance(value, str | bool | int | float)


def test_gate_matrix_is_plain_serializable_and_has_exact_top_level_fields() -> None:
    matrix = build_preview_risk_governor_gate_matrix()
    assert list(matrix) == TOP_LEVEL_FIELDS
    _assert_plain(matrix)
    assert json.loads(json.dumps(matrix)) == matrix


def test_identity_status_decision_and_references() -> None:
    matrix = build_preview_risk_governor_gate_matrix()
    assert matrix["schema_version"] == PREVIEW_RISK_GOVERNOR_GATE_MATRIX_SCHEMA_VERSION
    assert matrix["risk_governor_gate_matrix_kind"] == PREVIEW_RISK_GOVERNOR_GATE_MATRIX_KIND
    assert matrix["block"] == BLOCK_ID
    assert matrix["step"] == STEP_ID
    assert matrix["risk_governor_gate_matrix_status"] == RISK_GOVERNOR_GATE_MATRIX_STATUS
    assert matrix["risk_governor_gate_matrix_decision"] == RISK_GOVERNOR_GATE_MATRIX_DECISION
    assert matrix["ready_for_block_j_5"] is READY_FOR_BLOCK_J_5
    assert matrix["next_step"] == NEXT_STEP
    assert matrix["next_step_title"] == NEXT_STEP_TITLE
    assert matrix["status"] == STATUS
    assert matrix["risk_contract_reference"]["ready_for_block_j_1"] is True
    assert matrix["risk_contract_reference"]["next_step"] == "FUNCTIONAL-PREVIEW-12.1"
    assert matrix["risk_limits_read_model_reference"]["ready_for_block_j_2"] is True
    assert matrix["risk_limits_read_model_reference"]["next_step"] == "FUNCTIONAL-PREVIEW-12.2"
    assert matrix["risk_limits_static_fixture_reference"]["ready_for_block_j_3"] is True
    assert matrix["risk_limits_static_fixture_reference"]["next_step"] == "FUNCTIONAL-PREVIEW-12.3"
    assert matrix["kill_switch_read_model_reference"]["ready_for_block_j_4"] is True
    assert matrix["kill_switch_read_model_reference"]["next_step"] == "FUNCTIONAL-PREVIEW-12.4"


def test_scope_entries_selection_summary_cross_reference_and_contract() -> None:
    matrix = build_preview_risk_governor_gate_matrix()
    scope = matrix["risk_governor_gate_matrix_scope"]
    assert scope["scope_name"] == "risk_governor_gate_matrix"
    for key in [
        "gate_matrix_only",
        "derived_from_risk_contract_12_0",
        "derived_from_limits_read_model_12_1",
        "derived_from_static_fixture_12_2",
        "derived_from_kill_switch_read_model_12_3",
        "exe_direction_preserved",
    ]:
        assert scope[key] is True
    for key, value in scope.items():
        if key.endswith("allowed_now") or key.endswith("allowed") or key.endswith("in_scope"):
            assert value is False
    entries = matrix["risk_governor_gate_matrix_entries"]
    assert [entry["source_gate_id"] for entry in entries] == GATE_IDS
    assert [entry["gate_matrix_entry_id"] for entry in entries] == ENTRY_IDS
    for entry in entries:
        assert list(entry) == ENTRY_FIELDS
        expected = EXPECTED_GATE_VALUES[entry["source_gate_id"]]
        assert (
            entry["display_name"],
            entry["required_prior_step"],
            entry["required_prior_artifact"],
            entry["gate_condition"],
        ) == expected
        assert entry["gate_classification"] == "static_gate_matrix_entry_only"
        assert entry["gate_state_now"] == "static_contract_verified"
        assert entry["gate_result_now"] == "blocked_until_future_gate"
        assert entry["blocks_runtime_until_future_gate"] is True
        assert entry["blocks_order_flow_now"] is True
        assert entry["blocks_private_endpoint_now"] is True
        assert entry["blocks_network_now"] is True
        assert entry["blocks_live_trading_now"] is True
        assert entry["eligible_for_12_5_closure_audit"] is True
        assert entry["operator_visibility"] == "future_read_only_gate_matrix"
        assert entry["safe_for_offline_tests"] is True
        assert entry["notes"]
    assert matrix["default_risk_governor_gate_matrix_selection"] == {
        "gate_matrix_entry_id": "risk_governor_gate_matrix_contract_present_gate",
        "source_gate_id": "contract_present_gate",
        "reason": "first gate verifies Block J contract chain; static only, no runtime, no order flow",
        "gate_result_now": "blocked_until_future_gate",
        "order_submission_allowed_now": False,
    }
    assert matrix["risk_governor_gate_matrix_summary"] == {
        "entry_count": 8,
        "default_selection_id": "risk_governor_gate_matrix_contract_present_gate",
        "runtime_enabled_entry_count": 0,
        "order_flow_enabled_entry_count": 0,
        "private_endpoint_enabled_entry_count": 0,
        "network_enabled_entry_count": 0,
        "live_trading_enabled_entry_count": 0,
        "entries_blocking_runtime_until_future_gate": 8,
        "entries_blocking_order_flow_now": 8,
        "entries_blocking_private_endpoint_now": 8,
        "entries_blocking_network_now": 8,
        "entries_blocking_live_trading_now": 8,
        "entries_eligible_for_12_5_closure_audit": 8,
        "offline_safe_entry_count": 8,
        "safe_to_render_in_future_ui_as_read_only": True,
        "safe_for_runtime_execution_now": False,
        "safe_for_order_execution_now": False,
        "ready_for_12_5_closure_audit": True,
    }
    assert matrix["risk_governor_gate_matrix_cross_reference"] == {
        "referenced_steps": [
            "FUNCTIONAL-PREVIEW-12.0",
            "FUNCTIONAL-PREVIEW-12.1",
            "FUNCTIONAL-PREVIEW-12.2",
            "FUNCTIONAL-PREVIEW-12.3",
        ],
        "referenced_artifacts": [
            "preview_risk_governor_limits_kill_switch_contract",
            "preview_risk_governor_limits_read_model",
            "preview_risk_limits_static_fixture",
            "preview_kill_switch_read_model",
        ],
        "gate_matrix_entry_ids": ENTRY_IDS,
        "runtime_blocking_gate_ids": ENTRY_IDS,
        "order_flow_blocking_gate_ids": ENTRY_IDS,
        "private_endpoint_blocking_gate_ids": ENTRY_IDS,
        "network_blocking_gate_ids": ENTRY_IDS,
        "live_trading_blocking_gate_ids": ENTRY_IDS,
        "entries_requiring_12_5_closure_audit": ENTRY_IDS,
    }
    assert all(matrix["gate_matrix_contract"].values())
    assert (
        matrix["gate_matrix_contract"]["gate_matrix_contract_id"]
        == "block_j_risk_governor_gate_matrix_contract"
    )


def test_capabilities_boundaries_evidence_boundaries_and_future_steps() -> None:
    matrix = build_preview_risk_governor_gate_matrix()
    assert matrix["blocked_risk_governor_gate_matrix_capabilities"] == BLOCKED_CAPABILITIES
    boundaries = matrix["risk_governor_gate_matrix_boundaries"]
    assert "risk_governor_gate_matrix_balance_read_blocked" in boundaries
    assert all(boundaries.values())
    evidence = matrix["non_activation_evidence"]
    for key in [
        "risk_contract_12_0_read",
        "risk_limits_read_model_12_1_read",
        "risk_limits_static_fixture_12_2_read",
        "kill_switch_read_model_12_3_read",
        "risk_governor_gate_matrix_built",
    ]:
        assert evidence[key] is True
    for key, value in evidence.items():
        if key not in {
            "risk_contract_12_0_read",
            "risk_limits_read_model_12_1_read",
            "risk_limits_static_fixture_12_2_read",
            "kill_switch_read_model_12_3_read",
            "risk_governor_gate_matrix_built",
        }:
            assert value is False
    assert matrix["source_boundaries"] == SOURCE_BOUNDARIES
    assert matrix["future_steps"] == ["functional_preview_12_5_block_j_closure_audit"]


def test_source_imports_and_forbidden_calls_stay_inside_static_boundary() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: list[tuple[str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imports.append((node.module or "", None))
        elif isinstance(node, ast.Import):
            imports.extend((alias.name, None) for alias in node.names)
    assert imports == [
        ("__future__", None),
        ("typing", None),
        ("ui.pyside_app.preview_kill_switch_read_model", None),
        ("ui.pyside_app.preview_risk_governor_limits_kill_switch_contract", None),
        ("ui.pyside_app.preview_risk_governor_limits_read_model", None),
        ("ui.pyside_app.preview_risk_limits_static_fixture", None),
    ]
    forbidden_fragments = [
        "open(",
        "read_text",
        "write_text",
        "getenv(",
        "environ[",
        "requests.",
        "subprocess.",
        "urllib.",
        "httpx.",
        "aiohttp.",
        "socket.",
        "websocket.",
        "getaddrinfo(",
        "create_connection(",
        "QQmlApplicationEngine(",
        "start_runtime(",
        "start_loop(",
        "submit_order(",
        "place_order(",
        "create_order(",
        "send_order(",
        "fill_order(",
        "cancel_order(",
        "replace_order(",
        "withdraw(",
        "transfer(",
        "fetch_market_data(",
        "fetch_" + "balance(",
        "fetch_account(",
        "fetch_positions(",
        "fetch_orders(",
        "fetch_fills(",
        "refresh_market_data(",
        "export(",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in source
