"""Source-only guards for BLOK E read-only QML consumption of action dispatch snapshot."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
QML_ROOT = REPO_ROOT / "ui" / "pyside_app" / "qml"
OPERATOR_DASHBOARD = QML_ROOT / "views" / "OperatorDashboard.qml"
APP = REPO_ROOT / "ui" / "pyside_app" / "app.py"
BAT_LAUNCHERS = (
    REPO_ROOT / "run_ui_preview_visible_doubleclick.bat",
    REPO_ROOT / "scripts" / "windows" / "run_ui_preview_visible.bat",
)
ALLOWED_PREVIEW_SELECT_ACTION_CALL = 'paperRuntimeActionDispatchBridge.previewSelectAction("paper_runtime_snapshot_refresh_requested")'
FORBIDDEN_QML_BRIDGE_METHODS = (
    "previewSelectSourceControl",
    "resetPreviewSelection",
)
FORBIDDEN_NEW_QML_HANDLER_TOKENS = (
    "Button.onClicked",
    "onClicked:",
    "MouseArea",
    "Connections",
    "TapHandler",
    "Keys.onPressed",
    "Keys.onReleased",
    "Shortcut",
)
FORBIDDEN_EXECUTION_TOKENS = (
    "dispatch_command",
    "execute_command",
    "start_runtime",
    "start_loop",
    "submit_order",
    "create_order",
    "place_order",
    "send_order",
    "fill_order",
)
FORBIDDEN_PATH_TERMS = (
    "TradingController",
    "DecisionEnvelope",
    "live_adapter",
    "testnet_adapter",
    "account_balance_fetch",
    "load_dotenv",
    "keyring",
    "export_to_cloud",
)


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _qml_files() -> tuple[Path, ...]:
    return tuple(sorted(QML_ROOT.rglob("*.qml")))


def _source_files_under(root: Path) -> tuple[Path, ...]:
    source_suffixes = {".py", ".qml"}
    return tuple(
        sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in source_suffixes
        )
    )


def _source_lines_containing(pattern: str, root: Path) -> list[str]:
    return [
        line.strip()
        for path in _source_files_under(root)
        for line in _source(path).splitlines()
        if pattern in line
    ]


def test_operator_dashboard_reads_action_dispatch_snapshot_read_only() -> None:
    source = _source(OPERATOR_DASHBOARD)

    assert source.count("paperRuntimeActionDispatchBridge.snapshot") == 1
    assert "readonly property var actionDispatchSnapshot" in source
    assert "readonly property string actionDispatchStatus" in source
    assert "readonly property bool actionDispatchExecutionDisabled" in source
    assert "readonly property var actionDispatchActions" in source
    assert 'snapshotValue(actionDispatchSnapshot, "actions", [])' in source
    assert "readonly property int actionDispatchActionCount" in source
    assert "operatorDashboardActionDispatchReadOnlySnapshot" in source
    assert "previewActionDispatchReadOnlySnapshotLabel" in source


def test_operator_dashboard_exposes_no_execution_snapshot_evidence() -> None:
    source = _source(OPERATOR_DASHBOARD)

    for token in (
        "status",
        "snapshot_kind",
        "provider_status",
        "qt_bridge_kind",
        "execution_allowed",
        "execution_performed",
        "selected_result",
        "result_status",
        "catalog_action_found",
        "execution disabled",
        "actionDispatchActions",
        "actionDispatchActionCount",
        "action_count",
        "allowed paper action names/source controls",
        "source_control",
        "audit_status",
        "safe_to_bind_from_ui",
        "no order submission",
        "no lifecycle execution",
        "preview read-only disabled not executed",
    ):
        assert token in source


def test_operator_dashboard_disabled_intent_selection_preflight_surface_is_read_only_and_non_clickable() -> (
    None
):
    source = _source(OPERATOR_DASHBOARD)

    assert "readonly property bool actionDispatchSelectionPreflightLocked" in source
    assert "readonly property string actionDispatchSelectionPreflightStatus" in source
    assert "function actionDispatchDisabledIntentSummary(actions)" in source
    assert "operatorDashboardActionDispatchDisabledIntentSelectionPreflight" in source
    assert "previewActionDispatchDisabledIntentSelectionPreflightLabel" in source
    assert "actionDispatchDisabledIntentSummary(actionDispatchActions)" in source
    assert "actionDispatchActions" in source

    assert "readonly property var actionDispatchSelectionPreviewGate" in source
    assert "readonly property string actionDispatchSelectionPreviewGateStatus" in source
    assert "operatorDashboardActionDispatchSelectionPreviewGate" in source
    assert "previewActionDispatchSelectionPreviewGateLabel" in source
    assert "selection preview gate: controlled preview-only" in source
    assert "only snapshot refresh previewSelectAction literal is enabled" in source
    assert "method calls allowed now: one controlled preview call" in source
    assert "still blocked: previewSelectSourceControl, resetPreviewSelection" in source
    assert "paper/local only" in source

    preflight_start = source.index(
        "operatorDashboardActionDispatchDisabledIntentSelectionPreflight"
    )
    preflight_end = source.index("operatorDashboardActionDispatchSelectionPreviewGate")
    preflight_source = source[preflight_start:preflight_end]

    for token in (
        "selection locked",
        "disabled_preflight_only",
        "disabled intent candidates",
        "future interaction gate required",
        "method calls disabled",
        "bridge selection APIs not called",
        "execution disabled",
        "no runtime execution",
        "no order submission",
        "no lifecycle execution",
        "read-only preflight only not executed",
        "disabled read-only preflight only not executed",
        "source_control",
        "audit_status",
        "safe_to_bind_from_ui",
        "execution_allowed",
        "execution_performed",
    ):
        assert token in source

    for token in ("Button", "IconButton", *FORBIDDEN_NEW_QML_HANDLER_TOKENS):
        assert token not in preflight_source
    assert "previewSelectAction(" not in preflight_source
    for method in FORBIDDEN_QML_BRIDGE_METHODS:
        assert method not in preflight_source


def test_operator_dashboard_action_catalog_surface_is_read_only_and_non_clickable() -> None:
    source = _source(OPERATOR_DASHBOARD)

    assert "operatorDashboardActionDispatchReadOnlyActionCatalog" in source
    assert "previewActionDispatchReadOnlyActionCatalogLabel" in source
    catalog_start = source.index("operatorDashboardActionDispatchReadOnlyActionCatalog")
    catalog_end = source.index("operatorDashboardActionDispatchDisabledIntentSelectionPreflight")
    catalog_source = source[catalog_start:catalog_end]

    for token in ("Button", "IconButton", *FORBIDDEN_NEW_QML_HANDLER_TOKENS):
        assert token not in catalog_source
    for method in FORBIDDEN_QML_BRIDGE_METHODS:
        assert method not in catalog_source


def test_qml_bridge_consumption_is_limited_to_operator_dashboard_snapshot_binding() -> None:
    consumers = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in _qml_files()
        if "paperRuntimeActionDispatchBridge" in _source(path)
    ]

    assert consumers == ["ui/pyside_app/qml/views/OperatorDashboard.qml"]


def test_qml_calls_only_allowed_snapshot_refresh_preview_select_action() -> None:
    offenders: dict[str, list[str]] = {}
    for path in _qml_files():
        source = _source(path)
        preview_call_count = source.count("paperRuntimeActionDispatchBridge.previewSelectAction(")
        hits = []
        expected_call_count = 1 if path == OPERATOR_DASHBOARD else 0
        if preview_call_count != expected_call_count:
            hits.append("previewSelectAction")
        if path == OPERATOR_DASHBOARD and ALLOWED_PREVIEW_SELECT_ACTION_CALL not in source:
            hits.append("previewSelectAction")
        hits.extend(
            token
            for token in FORBIDDEN_QML_BRIDGE_METHODS
            if f".{token}(" in source or f"paperRuntimeActionDispatchBridge.{token}" in source
        )
        if hits:
            offenders[path.relative_to(REPO_ROOT).as_posix()] = hits

    assert offenders == {}


def test_operator_dashboard_preview_selection_result_is_visible_and_no_execution() -> None:
    source = _source(OPERATOR_DASHBOARD)

    assert "property var actionDispatchLastPreviewSelectionResult" in source
    assert "operatorDashboardActionDispatchPreviewSelectionResult" in source
    for token in (
        "result_status",
        "requested_action",
        "normalized_action",
        "execution_allowed",
        "execution_performed",
        "order_submission_allowed",
        "lifecycle_execution_allowed",
        "accepted intent not executed",
        "no runtime/order/lifecycle execution",
    ):
        assert token in source


def test_operator_dashboard_preview_select_handler_is_fail_closed_and_literal_only() -> None:
    source = _source(OPERATOR_DASHBOARD)

    assert source.count("onClicked: root.previewSelectSnapshotRefreshOnly()") == 1
    assert source.count(ALLOWED_PREVIEW_SELECT_ACTION_CALL) == 1
    assert "bridge_unavailable_fail_closed" in source
    assert 'typeof paperRuntimeActionDispatchBridge.previewSelectAction !== "function"' in source
    assert "previewSelectAction(action" not in source
    assert "previewSelectAction(actionDispatchActions" not in source
    assert "previewSelectAction(model" not in source


def test_qml_read_only_change_does_not_add_handlers_in_diff() -> None:
    diff = subprocess.run(
        ["git", "diff", "--", "ui/pyside_app/qml"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    added_lines = [
        line for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")
    ]

    allowed_handler = '+            Components.PreviewCard { objectName: "operatorDashboardActionDispatchSelectionPreviewGate"'
    offenders = []
    for token in FORBIDDEN_NEW_QML_HANDLER_TOKENS:
        for line in added_lines:
            if token in line and not (token == "onClicked:" and allowed_handler in line):
                offenders.append(token)
    assert offenders == []


def test_bat_launchers_remain_dev_preview_only_without_bridge_logic() -> None:
    for path in BAT_LAUNCHERS:
        source = _source(path)
        assert "paperRuntimeActionDispatchBridge" not in source
        for method in FORBIDDEN_QML_BRIDGE_METHODS:
            assert method not in source


def test_no_second_engine_or_app_py_ad_hoc_bridge_registration() -> None:
    engine_construction_lines = _source_lines_containing(
        "QQmlApplicationEngine(",
        REPO_ROOT / "ui" / "pyside_app",
    )
    assert engine_construction_lines == ["engine = QQmlApplicationEngine()"]

    app_source = _source(APP)
    assert "paperRuntimeActionDispatchBridge" not in app_source
    assert "register_paper_runtime_action_dispatch_qt_bridge" not in app_source


def test_read_only_qml_consumption_does_not_add_execution_or_sensitive_paths() -> None:
    source = _source(OPERATOR_DASHBOARD)

    for token in (*FORBIDDEN_EXECUTION_TOKENS, *FORBIDDEN_PATH_TERMS):
        assert token not in source


def test_no_new_bat_files_added_for_exe_direction() -> None:
    result = subprocess.run(
        ["git", "status", "--short", "--", "*.bat", "scripts/**/*.bat"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert not any(line.startswith("??") for line in result.stdout.splitlines())
