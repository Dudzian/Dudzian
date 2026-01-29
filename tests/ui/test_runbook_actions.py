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
from tests.ui._qt_utils import qt_wait


def _walk_qml_items(obj: object, limit: int = 8000) -> tuple[list[object], bool]:
    out: list[object] = []
    stack: list[object] = [obj]
    seen: set[int] = set()
    capped = False
    while stack:
        if len(out) >= limit:
            capped = True
            break
        cur = stack.pop()
        if cur is None:
            continue
        ident = id(cur)
        if ident in seen:
            continue
        seen.add(ident)
        out.append(cur)
        # QtQuickControls often hide the "real" content under QObject children even if childItems() is non-empty.
        # Walk both graphs; `seen` + `limit` keep it safe.
        try:
            child_items = cur.childItems()  # type: ignore[attr-defined]
        except Exception:
            child_items = None
        if child_items is not None:
            try:
                for item in child_items:
                    if item is not None:
                        stack.append(item)
            except Exception:
                pass
        try:
            stack.extend(cur.children())  # type: ignore[attr-defined]
        except Exception:
            pass
    return out, capped


def _find_by_object_name(root_obj: object, name: str) -> object | None:
    items, _ = _walk_qml_items(root_obj)
    for obj in items:
        try:
            object_name = obj.objectName()  # type: ignore[attr-defined]
        except Exception:
            object_name = None
        if object_name == name:
            return obj
    return None


def _find_by_object_name_prefix(root_obj: object, prefix: str) -> object | None:
    items, _ = _walk_qml_items(root_obj)
    for obj in items:
        try:
            object_name = obj.objectName()  # type: ignore[attr-defined]
        except Exception:
            object_name = None
        if object_name and object_name.startswith(prefix):
            return obj
    return None


def _safe_name(value: object) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"[^A-Za-z0-9_]", "_", text)


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
        repeater = root.findChild(QObject, "runbookPanelRepeater")
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

    deadline = time.monotonic() + timeout
    button = None
    alert_item = None
    while time.monotonic() < deadline:
        app.processEvents()
        alert_item = None
        if runbook_prefix:
            alert_item = _find_by_object_name_prefix(root, runbook_prefix)
        if alert_item is None:
            alert_item = root.findChild(QObject, "runbookAlertFrame_first")
        if alert_item is None:
            alert_item = root.findChild(QObject, "runbookAlertFrame_0")
        if alert_item is None:
            qt_wait(10)
            continue
        try:
            button = alert_item.findChild(QObject, "runbookActionButton_restart_queue")
        except Exception:
            button = None
        if button is None:
            button = _find_by_object_name(alert_item, "runbookActionButton_restart_queue")
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
        fallback_alert = alert_item
        if fallback_alert is None:
            if runbook_prefix:
                fallback_alert = _find_by_object_name_prefix(root, runbook_prefix)
        if fallback_alert is None:
            fallback_alert = root.findChild(QObject, "runbookAlertFrame_first")
        if fallback_alert is None:
            fallback_alert = root.findChild(QObject, "runbookAlertFrame_0")
        if fallback_alert is None:
            pytest.fail(
                "Nie udało się pobrać delegata alertu do diagnostyki. "
                f"alerts_type={type(alerts).__name__} "
                f"alerts_len={(len(alerts) if isinstance(alerts, list) else 'n/a')} "
                f"first_alert_type={type(first).__name__} "
                f"automaticActions={auto_actions!r} "
                f"qml_warnings={collected_warnings!r}"
            )
        items, capped = _walk_qml_items(fallback_alert)
        created_names = []
        try:
            for obj in items:
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
            for obj in items:
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
            for obj in items:
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
            for obj in root.findChildren(QObject):
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
        alert_item_type = type(alert_item).__name__ if alert_item is not None else None
        if alert_item is not None:
            try:
                alert_item_name = alert_item.objectName()  # type: ignore[attr-defined]
            except Exception:
                alert_item_name = None
        pytest.fail(
            "Przycisk akcji nie został wyrenderowany. "
            f"alerts_type={type(alerts).__name__} "
            f"alerts_len={(len(alerts) if isinstance(alerts, list) else 'n/a')} "
            f"first_alert_type={type(first).__name__} "
            f"automaticActions={auto_actions!r} "
            f"runbook_id={runbook_id!r} "
            f"runbook_prefix={runbook_prefix!r} "
            f"alert_item_found={bool(alert_item)} "
            f"alert_item_name={alert_item_name!r} "
            f"alert_item_type={alert_item_type!r} "
            f"runbook_panel_count={count!r} "
            f"runbook_panel_repeater_count={panel_repeater_count!r} "
            f"created_action_buttons={created_names!r} "
            f"runbook_action_repeaters={actions_repeaters!r} "
            f"runbook_named_items={runbook_names!r} "
            f"runbook_alert_frames={alert_frame_names!r} "
            f"alert_item_tree_size={len(items)} "
            f"alert_item_tree_capped={capped} "
            f"qml_warnings={collected_warnings!r}"
        )

    print(f"Using runbook action button: {button.objectName()}")
    QMetaObject.invokeMethod(button, "click")
    app.processEvents()

    assert (actions_dir / "ui_action_invoked.txt").exists(), "Skrypt nie został wykonany przez przycisk"
