"""Tests for FUNCTIONAL-PREVIEW-11.2 Block I adapter read model."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from ui.pyside_app.preview_testnet_sandbox_adapter_read_model import (
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_TESTNET_SANDBOX_ADAPTER_READ_MODEL_KIND,
    PREVIEW_TESTNET_SANDBOX_ADAPTER_READ_MODEL_SCHEMA_VERSION,
    READY_FOR_BLOCK_I_3,
    STATUS,
    TESTNET_SANDBOX_ADAPTER_READ_MODEL_DECISION,
    TESTNET_SANDBOX_ADAPTER_READ_MODEL_STATUS,
    build_preview_testnet_sandbox_adapter_read_model,
)

SIMPLE_TYPES = (dict, list, str, bool, int, float, type(None))
SOURCE = Path("ui/pyside_app/preview_testnet_sandbox_adapter_read_model.py")
EXPECTED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "testnet_sandbox_adapter_read_model_kind",
    "block",
    "step",
    "testnet_sandbox_adapter_read_model_status",
    "testnet_sandbox_adapter_read_model_decision",
    "ready_for_block_i_3",
    "next_step",
    "next_step_title",
    "handoff_reference",
    "read_model_scope",
    "adapter_read_model_entries",
    "default_adapter_read_model_selection",
    "adapter_read_model_summary",
    "blocked_adapter_runtime_capabilities",
    "read_model_boundaries",
    "non_activation_evidence",
    "source_boundaries",
    "future_steps",
    "status",
}
EXPECTED_ENTRY_FIELDS = {
    "adapter_read_model_id",
    "source_capability",
    "display_name",
    "read_model_classification",
    "adapter_surface_type",
    "evidence_paths",
    "main_symbols",
    "eligible_for_11_3_static_connectivity_fixture",
    "eligible_for_11_4_config_gate",
    "eligible_for_11_5_credentials_gate",
    "eligible_for_11_6_public_market_data_probe_preview",
    "eligible_for_11_7_private_endpoint_gate",
    "runtime_allowed_now",
    "network_io_allowed_now",
    "credentials_allowed_now",
    "private_endpoint_allowed_now",
    "order_submission_allowed_now",
    "requires_risk_governor_before_execution",
    "requires_observability_before_soak",
    "requires_live_gate_before_live",
    "operator_visibility",
    "notes",
}
EXPECTED_ORDER = [
    "read_only_market_data_provider",
    "exchange_adapter_layer",
    "exchange_network_guard",
    "paper_execution_oracle",
]
EXPECTED_BLOCKED = [
    "adapter instantiation",
    "adapter config read",
    "testnet connection",
    "sandbox connection",
    "live connection",
    "network I/O",
    "credentials read",
    "secrets read",
    "private endpoint access",
    "public market data fetch",
    "account fetch",
    "balance fetch",
    "positions fetch",
    "orders fetch",
    "fills fetch",
    "order submission",
    "fill simulation",
    "runtime loop",
    "scheduler",
    "QML action dispatch",
    "bridge API changes",
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
    "no sandbox adapter import",
    "no exchange adapter runtime import",
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
EXPECTED_FUTURE_STEPS = [
    "functional_preview_11_3_testnet_sandbox_static_connectivity_fixture",
    "functional_preview_11_4_testnet_sandbox_adapter_config_gate",
    "functional_preview_11_5_testnet_sandbox_credentials_gate_contract",
    "functional_preview_11_6_testnet_sandbox_public_market_data_probe_preview",
    "functional_preview_11_7_testnet_sandbox_private_endpoint_gate",
    "functional_preview_11_8_block_i_closure_audit",
]
FORBIDDEN_IMPORT_ROOTS = {
    "PySide6",
    "qml",
    "runtime",
    "scheduler",
    "order",
    "live",
    "exchange",
    "account",
    "secrets",
    "network",
    "requests",
    "urllib",
    "httpx",
    "aiohttp",
    "socket",
    "websocket",
    "pathlib",
    "subprocess",
}
FORBIDDEN_CALLS = {
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
ALLOWED_IMPORTS = [
    "__future__",
    "typing",
    "ui.pyside_app.preview_testnet_sandbox_backend_capability_handoff",
]


def _assert_simple_types_only(value: object) -> None:
    assert isinstance(value, SIMPLE_TYPES)
    if isinstance(value, dict):
        for key, nested in value.items():
            assert isinstance(key, str)
            _assert_simple_types_only(nested)
    if isinstance(value, list):
        for nested in value:
            _assert_simple_types_only(nested)


def _model() -> dict[str, object]:
    return build_preview_testnet_sandbox_adapter_read_model()


def test_model_is_plain_json_serializable_dict_with_exact_top_level_fields() -> None:
    model = _model()

    assert isinstance(model, dict)
    _assert_simple_types_only(model)
    assert json.loads(json.dumps(model, sort_keys=True)) == model
    assert set(model) == EXPECTED_TOP_LEVEL_FIELDS


def test_identity_status_decision_and_next_step() -> None:
    model = _model()

    assert model["schema_version"] == PREVIEW_TESTNET_SANDBOX_ADAPTER_READ_MODEL_SCHEMA_VERSION
    assert (
        model["testnet_sandbox_adapter_read_model_kind"]
        == PREVIEW_TESTNET_SANDBOX_ADAPTER_READ_MODEL_KIND
    )
    assert model["block"] == "I"
    assert model["step"] == "11.2"
    assert (
        model["testnet_sandbox_adapter_read_model_status"]
        == TESTNET_SANDBOX_ADAPTER_READ_MODEL_STATUS
    )
    assert (
        model["testnet_sandbox_adapter_read_model_decision"]
        == TESTNET_SANDBOX_ADAPTER_READ_MODEL_DECISION
    )
    assert model["ready_for_block_i_3"] is READY_FOR_BLOCK_I_3 is True
    assert model["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-11.3"
    assert (
        model["next_step_title"] == NEXT_STEP_TITLE == "TESTNET/SANDBOX STATIC CONNECTIVITY FIXTURE"
    )
    assert model["status"] == STATUS


def test_handoff_reference_points_to_11_1() -> None:
    reference = _model()["handoff_reference"]

    assert set(reference) == {
        "schema_version",
        "testnet_sandbox_backend_capability_handoff_kind",
        "testnet_sandbox_backend_capability_handoff_status",
        "testnet_sandbox_backend_capability_handoff_decision",
        "ready_for_block_i_2",
        "next_step",
        "next_step_title",
        "status",
    }
    assert reference["ready_for_block_i_2"] is True
    assert reference["next_step"] == "FUNCTIONAL-PREVIEW-11.2"
    assert reference["next_step_title"] == "TESTNET/SANDBOX ADAPTER READ MODEL"


def test_scope_is_static_read_model_only_with_blocked_activation_flags() -> None:
    scope = _model()["read_model_scope"]

    assert scope["scope_name"] == "testnet_sandbox_adapter_read_model"
    assert scope["read_model_only"] is True
    assert scope["static_model_only"] is True
    assert scope["derived_from_handoff_11_1"] is True
    assert scope["exe_direction_preserved"] is True
    for key, value in scope.items():
        if key not in {
            "scope_name",
            "read_model_only",
            "static_model_only",
            "derived_from_handoff_11_1",
            "exe_direction_preserved",
        }:
            assert value is False


def test_entries_are_exact_ordered_static_adapter_read_models() -> None:
    entries = _model()["adapter_read_model_entries"]

    assert len(entries) == 4
    assert [entry["source_capability"] for entry in entries] == EXPECTED_ORDER
    for entry in entries:
        assert set(entry) == EXPECTED_ENTRY_FIELDS
        assert entry["runtime_allowed_now"] is False
        assert entry["network_io_allowed_now"] is False
        assert entry["credentials_allowed_now"] is False
        assert entry["private_endpoint_allowed_now"] is False
        assert entry["order_submission_allowed_now"] is False

    assert (
        entries[0]
        | {
            "evidence_paths": entries[0]["evidence_paths"],
            "main_symbols": entries[0]["main_symbols"],
            "notes": entries[0]["notes"],
        }
        == entries[0]
    )
    assert (
        entries[0]["adapter_read_model_id"] == "adapter_read_model_read_only_market_data_provider"
    )
    assert entries[0]["display_name"] == "Read-only market data provider"
    assert entries[0]["read_model_classification"] == "ready_for_contract_gate"
    assert entries[0]["adapter_surface_type"] == "public_market_data_read_model"
    assert entries[0]["operator_visibility"] == "read_only_future"
    assert entries[1]["operator_visibility"] == "blocked_until_gated"
    assert entries[2]["operator_visibility"] == "safety_guard_future"
    assert entries[3]["operator_visibility"] == "comparison_only_future"


def test_summary_counts_match_entries_and_default_selection() -> None:
    model = _model()
    entries = model["adapter_read_model_entries"]
    summary = model["adapter_read_model_summary"]

    assert model["default_adapter_read_model_selection"] == {
        "adapter_read_model_id": "adapter_read_model_read_only_market_data_provider",
        "source_capability": "read_only_market_data_provider",
        "reason": "lowest-risk read-model candidate; no credentials, private endpoint, order submission, or runtime activation",
        "runtime_allowed_now": False,
        "network_io_allowed_now": False,
    }
    assert summary == {
        "entry_count": 4,
        "default_selection_id": "adapter_read_model_read_only_market_data_provider",
        "runtime_enabled_entry_count": sum(entry["runtime_allowed_now"] for entry in entries),
        "network_enabled_entry_count": sum(entry["network_io_allowed_now"] for entry in entries),
        "credentials_enabled_entry_count": sum(
            entry["credentials_allowed_now"] for entry in entries
        ),
        "private_endpoint_enabled_entry_count": sum(
            entry["private_endpoint_allowed_now"] for entry in entries
        ),
        "order_submission_enabled_entry_count": sum(
            entry["order_submission_allowed_now"] for entry in entries
        ),
        "entries_eligible_for_11_3_static_connectivity_fixture": 4,
        "entries_eligible_for_11_4_config_gate": 3,
        "entries_eligible_for_11_5_credentials_gate": 1,
        "entries_eligible_for_11_6_public_market_data_probe_preview": 2,
        "entries_eligible_for_11_7_private_endpoint_gate": 2,
        "safe_to_render_in_future_ui_as_read_only": True,
        "safe_for_runtime_execution_now": False,
    }


def test_boundaries_blocked_capabilities_evidence_sources_and_future_steps() -> None:
    model = _model()

    assert model["blocked_adapter_runtime_capabilities"] == EXPECTED_BLOCKED
    assert all(model["read_model_boundaries"].values())
    evidence = model["non_activation_evidence"]
    assert evidence["handoff_11_1_read"] is True
    assert evidence["adapter_read_model_built"] is True
    for key, value in evidence.items():
        if key not in {"handoff_11_1_read", "adapter_read_model_built"}:
            assert value is False
    assert model["source_boundaries"] == EXPECTED_SOURCE_BOUNDARIES
    assert model["future_steps"] == EXPECTED_FUTURE_STEPS


def test_source_imports_only_safe_typing_and_handoff_helper() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")

    assert imported_modules == ALLOWED_IMPORTS
    for module in imported_modules:
        root = module.split(".", maxsplit=1)[0]
        assert root not in FORBIDDEN_IMPORT_ROOTS


def test_source_has_no_forbidden_runtime_or_io_calls() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            if isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)

    assert not (called & FORBIDDEN_CALLS)
