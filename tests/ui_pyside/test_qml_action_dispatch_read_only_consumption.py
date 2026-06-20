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
FORBIDDEN_QML_BRIDGE_METHODS = (
    "previewSelectAction",
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


def test_operator_dashboard_action_catalog_surface_is_read_only_and_non_clickable() -> None:
    source = _source(OPERATOR_DASHBOARD)

    assert "operatorDashboardActionDispatchReadOnlyActionCatalog" in source
    assert "previewActionDispatchReadOnlyActionCatalogLabel" in source
    catalog_start = source.index("operatorDashboardActionDispatchReadOnlyActionCatalog")
    catalog_end = source.index("operatorDashboardBlockCReadOnlyBindingSummary")
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


def test_qml_does_not_call_action_dispatch_bridge_methods() -> None:
    offenders: dict[str, list[str]] = {}
    for path in _qml_files():
        source = _source(path)
        hits = [token for token in FORBIDDEN_QML_BRIDGE_METHODS if token in source]
        if hits:
            offenders[path.relative_to(REPO_ROOT).as_posix()] = hits

    assert offenders == {}


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

    offenders = [
        token
        for token in FORBIDDEN_NEW_QML_HANDLER_TOKENS
        if any(token in line for line in added_lines)
    ]
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
