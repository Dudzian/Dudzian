import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytestmark = pytest.mark.qml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:  # pragma: no cover - zależne od środowiska CI
    from PySide6.QtCore import QObject, QMetaObject, QUrl
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtWidgets import QApplication
except Exception:  # pragma: no cover - brak Qt
    QObject = QMetaObject = QUrl = QQmlApplicationEngine = QApplication = None  # type: ignore[assignment]

from core.reporting.guardrails_reporter import (
    GuardrailLogRecord,
    GuardrailReport,
    GuardrailReportEndpoint,
)
from ui.backend.runbook_controller import RunbookController
from tests.ui._qml_tree import find_by_object_name, walk_qml_items
from tests.ui._qt_utils import qt_wait


def _safe_name(value: object) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"[^A-Za-z0-9_]", "_", text)


def _obj_name(obj: object) -> str | None:
    try:
        return obj.objectName()  # type: ignore[attr-defined]
    except Exception:
        return None


def _find_button_via_repeater(
    repeater: object,
    repeater_count: int,
    target_name: str,
) -> tuple[object | None, bool]:
    item_at = getattr(repeater, "itemAt", None)
    if not callable(item_at):
        return None, False
    delegate_capped_any = False
    for i in range(min(repeater_count, 10)):
        try:
            delegate = item_at(i)
        except Exception:
            delegate = None
        if delegate is None:
            continue
        items, capped = walk_qml_items(delegate)
        delegate_capped_any = delegate_capped_any or capped
        found = next((obj for obj in items if _obj_name(obj) == target_name), None)
        if found is not None:
            return found, delegate_capped_any
    return None, delegate_capped_any


def _build_sample_report() -> GuardrailReport:
    generated_at = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    logs = (
        GuardrailLogRecord(
            timestamp=generated_at,
            level="ERROR",
            message="TIMEOUT queue=binance_spot waited=5.000000s",
            event="TIMEOUT",
            metadata={"queue": "binance_spot", "environment": "paper"},
        ),
    )
    return GuardrailReport(
        generated_at=generated_at,
        summaries=(),
        logs=logs,
        recommendations=(),
    )


class _StaticEndpoint(GuardrailReportEndpoint):
    def __init__(self, report: GuardrailReport) -> None:
        super().__init__(report_factory=lambda: report)


@pytest.mark.skipif(QObject is None, reason="Wymagany PySide6 do testów UI")
def test_runbook_panel_action_button_executes_script(tmp_path: Path) -> None:
    runbook_dir = tmp_path / "runbooks"
    metadata_dir = runbook_dir / "metadata"
    actions_dir = tmp_path / "actions"
    runbook_dir.mkdir()
    metadata_dir.mkdir()
    actions_dir.mkdir()

    (runbook_dir / "strategy_incident_playbook.md").write_text("# Strategia L1/L2\n", encoding="utf-8")
    metadata_dir.joinpath("strategy_incident_playbook.yml").write_text(
        """
        id: strategy_incident_playbook
        automatic_actions:
          - id: restart_queue
            label: Restartuj kolejkę
            script: restart_queue.py
        manual_steps:
          - Sprawdź limit zapytań
        """,
        encoding="utf-8",
    )

    script_path = actions_dir / "restart_queue.py"
    script_path.write_text(
        """
from __future__ import annotations
from pathlib import Path

(Path(__file__).resolve().parent / "ui_action_invoked.txt").write_text("ok", encoding="utf-8")
        """,
        encoding="utf-8",
    )

    report = _build_sample_report()
    controller = RunbookController(
        report_endpoint=_StaticEndpoint(report),
        runbook_directory=runbook_dir,
        actions_directory=actions_dir,
    )
    assert controller.refreshAlerts()

    app = QApplication.instance() or QApplication([])
    engine = QQmlApplicationEngine()
    collected_warnings: list[str] = []

    def _format_warning(warning: object) -> str:
        try:
            return warning.toString()
        except Exception:
            return str(warning)

    try:
        def _collect(warnings: list[object]) -> None:
            collected_warnings.extend(_format_warning(warning) for warning in warnings)

        engine.warnings.connect(_collect)  # type: ignore[attr-defined]
    except Exception:
        pass
    engine.rootContext().setContextProperty("runbookController", controller)
    qml_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "dashboard" / "RunbookPanel.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    try:
        warn_attr = getattr(engine, "warnings", None)
        if callable(warn_attr):
            collected_warnings.extend(_format_warning(warning) for warning in warn_attr())
    except Exception:
        pass
    assert engine.rootObjects(), "Nie udało się załadować RunbookPanel.qml"
    app.processEvents()

    root = engine.rootObjects()[0]
    timeout = 10.0 if sys.platform.startswith("win") else 5.0
    deadline = time.monotonic() + timeout
    repeater = None
    while time.monotonic() < deadline:
        app.processEvents()
        repeater = find_by_object_name(root, "runbookPanelRepeater")
        count = repeater.property("count") if repeater is not None else None
        if isinstance(count, int) and count >= 1:
            break
        qt_wait(10)
    count = repeater.property("count") if repeater is not None else None
    if not isinstance(count, int) or count < 1:
        alerts = getattr(controller, "alerts", None)
        pytest.fail(
            "Brak alertów w repeaterze (runbookPanelRepeater). "
            f"repeater={repeater!r} "
            f"count={count!r} "
            f"alerts_type={type(alerts).__name__} "
            f"alerts_len={(len(alerts) if isinstance(alerts, list) else 'n/a')} "
            f"lastUpdated={getattr(controller, 'lastUpdated', None)!r} "
            f"errorMessage={getattr(controller, 'errorMessage', None)!r}"
        )
    runbook_id = ""
    alerts_data = getattr(controller, "alerts", None)
    if isinstance(alerts_data, list) and alerts_data:
        first_alert = alerts_data[0]
        if isinstance(first_alert, dict):
            runbook_id = first_alert.get("runbookId", "") or ""
        else:
            runbook_id = getattr(first_alert, "runbookId", "") or ""
    runbook_prefix = f"runbookAlertFrame_{_safe_name(runbook_id)}" if runbook_id else ""

    target_name = "runbookActionButton_restart_queue"
    deadline = time.monotonic() + timeout
    button = None
    last_items_root: list[object] | None = None
    last_capped_root = False
    delegate_capped_any = False
    while time.monotonic() < deadline:
        app.processEvents()
        rep_count = None
        try:
            rep_count = repeater.property("count") if repeater is not None else None
        except Exception:
            rep_count = None
        if repeater is not None and isinstance(rep_count, int) and rep_count > 0:
            found, capped = _find_button_via_repeater(repeater, rep_count, target_name)
            delegate_capped_any = delegate_capped_any or capped
            button = found
            if button is not None:
                break
        items_root, capped_root = walk_qml_items(root)
        last_items_root = items_root
        last_capped_root = capped_root
        button = next((obj for obj in items_root if _obj_name(obj) == target_name), None)
        if button is not None:
            break
        qt_wait(10)
    if button is None:
        alerts = getattr(controller, "alerts", None)
        first = alerts[0] if isinstance(alerts, list) and alerts else None
        if isinstance(first, dict):
            auto_actions = first.get("automaticActions")
        else:
            auto_actions = getattr(first, "automaticActions", None)
        if last_items_root is None:
            items_root, capped_root = walk_qml_items(root)
        else:
            items_root, capped_root = last_items_root, last_capped_root
        created_names = []
        try:
            for obj in items_root:
                try:
                    name = obj.objectName()  # type: ignore[attr-defined]
                except Exception:
                    continue
                if (name or "").startswith("runbookActionButton_"):
                    created_names.append(name)
        except Exception:
            created_names = ["(failed to enumerate)"]
        actions_repeaters = []
        try:
            for obj in items_root:
                try:
                    name = obj.objectName() or ""  # type: ignore[attr-defined]
                except Exception:
                    name = ""
                if name.startswith("runbookActionsRepeater_"):
                    model = None
                    item_at = None
                    try:
                        model = obj.property("model")
                    except Exception:
                        model = "(model property unavailable)"
                    try:
                        item_at = obj.itemAt(0)
                    except Exception:
                        item_at = "(itemAt unavailable)"
                    actions_repeaters.append(
                        {
                            "objectName": name,
                            "count": obj.property("count"),
                            "model": model,
                            "itemAt0": item_at,
                        }
                    )
        except Exception:
            actions_repeaters = ["(failed to enumerate)"]
        panel_repeater_count = None
        try:
            panel_repeater_count = repeater.property("count") if repeater is not None else None
        except Exception:
            panel_repeater_count = "(count unavailable)"
        runbook_names = []
        try:
            for obj in items_root:
                try:
                    name = obj.objectName()  # type: ignore[attr-defined]
                except Exception:
                    continue
                if (name or "").startswith("runbook"):
                    runbook_names.append(name)
        except Exception:
            runbook_names = ["(failed to enumerate)"]
        if isinstance(runbook_names, list) and runbook_names and runbook_names != ["(failed to enumerate)"]:
            runbook_names = sorted(set(runbook_names))[:200]
        alert_frame_names = []
        try:
            for obj in items_root:
                try:
                    name = obj.objectName()  # type: ignore[attr-defined]
                except Exception:
                    continue
                if (name or "").startswith("runbookAlertFrame_"):
                    alert_frame_names.append(name)
        except Exception:
            alert_frame_names = ["(failed to enumerate)"]
        if isinstance(alert_frame_names, list) and alert_frame_names and alert_frame_names != ["(failed to enumerate)"]:
            alert_frame_names = sorted(set(alert_frame_names))[:200]
        alert_item_name = None
        alert_item_type = None
        fallback_alert = None
        if runbook_prefix:
            # Prefix is used only for diagnostics, not for button lookup.
            fallback_alert = next(
                (
                    obj
                    for obj in items_root
                    if (_obj_name(obj) or "").startswith(runbook_prefix)
                ),
                None,
            )
        if fallback_alert is None:
            fallback_alert = next(
                (obj for obj in items_root if _obj_name(obj) == "runbookAlertFrame_first"),
                None,
            )
        if fallback_alert is None:
            fallback_alert = next(
                (obj for obj in items_root if _obj_name(obj) == "runbookAlertFrame_0"),
                None,
            )
        if fallback_alert is not None:
            alert_item_type = type(fallback_alert).__name__
            try:
                alert_item_name = fallback_alert.objectName()  # type: ignore[attr-defined]
            except Exception:
                alert_item_name = None
        rep_item0_type = None
        rep_item0_name = None
        rep_item0_child_count = None
        rep_item0_tree_size = None
        rep_item0_tree_capped = None
        try:
            if repeater is not None and callable(getattr(repeater, "itemAt", None)):
                rep_item0 = repeater.itemAt(0)
                rep_item0_type = type(rep_item0).__name__ if rep_item0 is not None else None
                rep_item0_name = _obj_name(rep_item0) if rep_item0 is not None else None
                if rep_item0 is not None:
                    try:
                        rep_item0_child_count = sum(1 for _ in rep_item0.childItems())  # type: ignore[attr-defined]
                    except Exception:
                        rep_item0_child_count = "(childItems unavailable)"
                    rep_items, rep_capped = walk_qml_items(rep_item0)
                    rep_item0_tree_size = len(rep_items)
                    rep_item0_tree_capped = rep_capped
        except Exception:
            rep_item0_type = "(itemAt failed)"
        pytest.fail(
            "Przycisk akcji nie został wyrenderowany. "
            f"alerts_type={type(alerts).__name__} "
            f"alerts_len={(len(alerts) if isinstance(alerts, list) else 'n/a')} "
            f"first_alert_type={type(first).__name__} "
            f"automaticActions={auto_actions!r} "
            f"runbook_id={runbook_id!r} "
            f"runbook_prefix={runbook_prefix!r} "
            f"alert_item_found={bool(fallback_alert)} "
            f"alert_item_name={alert_item_name!r} "
            f"alert_item_type={alert_item_type!r} "
            f"runbook_panel_count={count!r} "
            f"runbook_panel_repeater_count={panel_repeater_count!r} "
            f"created_action_buttons={created_names!r} "
            f"runbook_action_repeaters={actions_repeaters!r} "
            f"runbook_named_items={runbook_names!r} "
            f"runbook_alert_frames={alert_frame_names!r} "
            f"alert_item_tree_size={len(items_root)} "
            f"alert_item_tree_capped={capped_root} "
            f"delegate_tree_capped_any={delegate_capped_any} "
            f"repeater_has_itemAt={callable(getattr(repeater, 'itemAt', None))} "
            f"repeater_item0_type={rep_item0_type!r} "
            f"repeater_item0_name={rep_item0_name!r} "
            f"repeater_item0_child_count={rep_item0_child_count!r} "
            f"repeater_item0_tree_size={rep_item0_tree_size!r} "
            f"repeater_item0_tree_capped={rep_item0_tree_capped!r} "
            f"qml_warnings={collected_warnings!r}"
        )

    print(f"Using runbook action button: {button.objectName()}")
    QMetaObject.invokeMethod(button, "click")
    app.processEvents()

    assert (actions_dir / "ui_action_invoked.txt").exists(), "Skrypt nie został wykonany przez przycisk"
