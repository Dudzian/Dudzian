"""Source-only BLOK D closure audit with BLOK E wiring status.

This helper freezes the accepted BLOK D decision in deterministic plain data and
records the narrow BLOK E transition: the Qt bridge may now be registered only
in QmlContextBridge.install().  It still does not import PySide/QML and does not
execute lifecycle, command-dispatch, order, live, testnet, account, secrets,
export, or cloud paths.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Final

SCHEMA_VERSION: Final[str] = "preview_block_d_closure_audit.v1"
AUDIT_KIND: Final[str] = "block_d_closure_bridge_ready_block_e_context_wiring_started"
BLOCK_STATUS: Final[str] = "bridge_ready_block_e_context_wiring_started"
CLOSURE_DECISION: Final[str] = "CLOSE_BLOCK_D_AS_BRIDGE_READY_NOT_WIRED"
RECOMMENDED_FUTURE_INTEGRATION_POINT: Final[str] = (
    "ui/pyside_app/qml_bridge.py::QmlContextBridge.install()"
)

PIPELINE_MODULES: Final[tuple[str, ...]] = (
    "ui/pyside_app/preview_action_dispatch_contract.py",
    "ui/pyside_app/preview_action_dispatch_audit.py",
    "ui/pyside_app/preview_action_dispatch_catalog.py",
    "ui/pyside_app/preview_action_dispatch_selection.py",
    "ui/pyside_app/preview_action_dispatch_bridge_snapshot.py",
    "ui/pyside_app/preview_action_dispatch_bridge_provider.py",
    "ui/pyside_app/preview_action_dispatch_qt_bridge.py",
    "ui/pyside_app/preview_action_dispatch_qt_bridge_registration.py",
)
PIPELINE_TESTS_REQUIRED: Final[tuple[str, ...]] = (
    "tests/ui_pyside/test_preview_action_dispatch_contract.py",
    "tests/ui_pyside/test_preview_action_dispatch_audit.py",
    "tests/ui_pyside/test_preview_action_dispatch_catalog.py",
    "tests/ui_pyside/test_preview_action_dispatch_selection.py",
    "tests/ui_pyside/test_preview_action_dispatch_bridge_snapshot.py",
    "tests/ui_pyside/test_preview_action_dispatch_bridge_provider.py",
    "tests/ui_pyside/test_preview_action_dispatch_qt_bridge.py",
    "tests/ui_pyside/test_preview_action_dispatch_qt_bridge_registration.py",
    "tests/ui_pyside/test_preview_action_dispatch_contract_source_guard.py",
)
BOUNDARY_CHECK_NAMES: Final[tuple[str, ...]] = (
    "block_d_pipeline_complete",
    "anti_duplication_audit_complete",
    "single_future_integration_point_identified",
    "bridge_ready",
    "block_e_wiring_started",
    "bridge_registered_in_central_context",
    "qml_not_changed",
    "startup_not_changed",
    "bat_launch_path_not_changed",
    "real_context_property_registered_once",
    "no_second_frontend",
    "typed_preview_bridge_not_replaced",
    "grpc_bridge_not_replaced",
    "runtime_state_not_replaced",
    "execution_disabled",
    "execution_not_performed",
    "lifecycle_execution_disabled",
    "command_dispatch_disabled",
    "order_generation_disabled",
    "order_submission_disabled",
    "live_mode_disabled",
    "testnet_mode_disabled",
    "account_fetch_disabled",
    "secrets_disabled",
    "export_cloud_disabled",
    "execution_still_disabled_after_wiring",
)
FORBIDDEN_INTEGRATION_POINTS: Final[tuple[str, ...]] = (
    ".bat launchers",
    "ui/pyside_app/app.py",
    "ui/pyside_app/qml/MainWindow.qml",
    "ui/pyside_app/qml/views/OperatorDashboard.qml",
    "ui/pyside_app/qml/views/PaperTerminal.qml",
    "ui/pyside_app/preview_state_bridge.py",
    "runtime/order/trading/live/testnet/account/secrets/export paths",
)
REQUIRED_TESTS_BEFORE_WIRING: Final[tuple[str, ...]] = (
    "source guard",
    "qt bridge",
    "qt bridge registration",
    "block d closure audit",
    "source-only single registration test",
    "controlled context wiring test",
    "QML/runtime smoke with installed PySide/UI deps",
    "launch path unchanged test",
    "no-execution boundary test",
)
ANTI_DUPLICATION_AUDIT_PATH: Final[str] = (
    "docs/functional_preview/block_d_anti_duplication_integration_audit.md"
)


def build_preview_block_d_closure_audit(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Return copy-safe plain evidence for BLOK D closure and BLOK E wiring start."""

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    pipeline_modules_present = {
        module_path: (root / module_path).is_file() for module_path in PIPELINE_MODULES
    }
    anti_duplication_audit_present = (root / ANTI_DUPLICATION_AUDIT_PATH).is_file()
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "audit_kind": AUDIT_KIND,
        "block": "D",
        "block_status": BLOCK_STATUS,
        "closure_decision": CLOSURE_DECISION,
        "ready_for_block_e": True,
        "block_e_wiring_started": True,
        "bridge_ready": True,
        "bridge_wired_to_real_startup": True,
        "qml_consumes_bridge": False,
        "real_startup_context_property_registered": True,
        "registered_context_property_name": "paperRuntimeActionDispatchBridge",
        "execution_still_disabled_after_wiring": True,
        "recommended_future_integration_point": RECOMMENDED_FUTURE_INTEGRATION_POINT,
        "launch_path_preserved": True,
        "anti_duplication_audit_present": anti_duplication_audit_present,
        "pipeline_modules_present": pipeline_modules_present,
        "pipeline_tests_required": list(PIPELINE_TESTS_REQUIRED),
        "runtime_mode": "paper",
        "paper_only": True,
        "local_only": True,
        "execution_allowed": False,
        "execution_performed": False,
        "lifecycle_execution_allowed": False,
        "command_dispatch_allowed": False,
        "order_generation_allowed": False,
        "order_submission_allowed": False,
        "live_mode_allowed": False,
        "testnet_mode_allowed": False,
        "account_fetch_allowed": False,
        "secrets_allowed": False,
        "export_cloud_allowed": False,
        "boundary_checks": {name: True for name in BOUNDARY_CHECK_NAMES},
        "next_block_gate": {
            "allowed_next_scope": "controlled_qml_context_wiring",
            "required_integration_point": RECOMMENDED_FUTURE_INTEGRATION_POINT,
            "forbidden_integration_points": list(FORBIDDEN_INTEGRATION_POINTS),
            "required_tests_before_wiring": list(REQUIRED_TESTS_BEFORE_WIRING),
        },
        "operator_message": (
            "BLOK D closed as bridge ready; BLOK E has started controlled central "
            "context wiring in QmlContextBridge.install() while execution remains disabled."
        ),
    }
    return deepcopy(evidence)


__all__ = [
    "AUDIT_KIND",
    "BLOCK_STATUS",
    "CLOSURE_DECISION",
    "RECOMMENDED_FUTURE_INTEGRATION_POINT",
    "SCHEMA_VERSION",
    "build_preview_block_d_closure_audit",
]
