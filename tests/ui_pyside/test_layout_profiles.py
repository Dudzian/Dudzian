from __future__ import annotations

from pathlib import Path

from ui.pyside_app.controllers import LayoutProfileController


def _panel_specs() -> list[dict[str, object]]:
    return [
        {"panelId": "sidePanel", "title": "Licencja", "defaultColumn": 0, "defaultOrder": 0},
        {"panelId": "chartView", "title": "Chart", "defaultColumn": 1, "defaultOrder": 0},
        {"panelId": "strategyWorkbench", "title": "Strategie", "defaultColumn": 1, "defaultOrder": 1},
        {"panelId": "modeWizardPanel", "title": "Tryby", "defaultColumn": 1, "defaultOrder": 2},
        {"panelId": "strategyManagerPanel", "title": "Marketplace", "defaultColumn": 1, "defaultOrder": 3},
        {"panelId": "telemetryPanel", "title": "Telemetria", "defaultColumn": 0, "defaultOrder": 1},
        {"panelId": "diagnosticsPanel", "title": "Diagnostyka", "defaultColumn": 0, "defaultOrder": 2},
    ]


def test_layout_profiles_roundtrip(tmp_path: Path) -> None:
    storage = tmp_path / "ui_layouts.json"
    controller = LayoutProfileController(storage_path=storage)
    controller.registerPanels(_panel_specs())

    assert any(panel["panelId"] == "sidePanel" for panel in controller.layout)
    controller.updatePanelPosition("chartView", 0, 0)
    controller.setPanelVisibility("strategyWorkbench", False)
    controller.setColumnCount(3)

    catalog = controller.availablePanels
    assert any(entry["panelId"] == "strategyWorkbench" and entry["visible"] is False for entry in catalog)

    controller2 = LayoutProfileController(storage_path=storage)
    controller2.registerPanels(_panel_specs())

    assert controller2.columnCount == 3
    restored_layout = controller2.layout
    chart_entry = next(panel for panel in restored_layout if panel["panelId"] == "chartView")
    assert chart_entry["column"] == 0
    assert chart_entry["order"] == 1
    assert controller2.isPanelVisible("strategyWorkbench") is False
