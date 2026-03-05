"""Kontroler odpowiadający za profile i układ paneli w PySide6."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping

from PySide6.QtCore import QObject, Property, Signal, Slot

_LOGGER = logging.getLogger(__name__)

_DEFAULT_LAYOUT = [
    {"panelId": "sidePanel", "column": 0, "order": 0, "visible": True},
    {"panelId": "chartView", "column": 1, "order": 0, "visible": True},
    {"panelId": "strategyWorkbench", "column": 1, "order": 1, "visible": True},
    {"panelId": "telemetryPanel", "column": 0, "order": 1, "visible": True},
    {"panelId": "diagnosticsPanel", "column": 0, "order": 2, "visible": True},
]


@dataclass(slots=True)
class _ProfileState:
    column_count: int
    panels: list[dict[str, Any]]


class LayoutProfileController(QObject):
    """Zarządza profilami układu i zapisuje ustawienia w pliku JSON."""

    layoutChanged = Signal()
    profilesChanged = Signal()
    activeProfileChanged = Signal()
    columnCountChanged = Signal()
    availablePanelsChanged = Signal()

    def __init__(self, storage_path: Path | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        repo_root = Path(__file__).resolve().parents[3]
        default_path = repo_root / "var" / "ui_layouts.json"
        self._storage_path = (storage_path or default_path).expanduser()
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage: MutableMapping[str, Any] = self._load_storage()
        profiles = self._storage.setdefault("profiles", {})
        if not profiles:
            profiles["default"] = self._default_profile_payload()
        self._active_profile = str(self._storage.get("active_profile") or next(iter(profiles)))
        if self._active_profile not in profiles:
            self._active_profile = next(iter(profiles))
        profile_state = self._profile_state(self._active_profile)
        self._column_count = max(1, int(profile_state.column_count))
        self._layout_state: list[dict[str, Any]] = [dict(panel) for panel in profile_state.panels]
        self._panel_catalog: dict[str, dict[str, Any]] = {}
        self._normalize_layout()

    @Property("QStringList", notify=profilesChanged)
    def profiles(self) -> list[str]:  # pragma: no cover - getter wywoływany w QML
        return list(self._storage.get("profiles", {}).keys())

    @Property(str, notify=activeProfileChanged)
    def activeProfile(self) -> str:  # pragma: no cover - getter wykorzystywany w QML
        return self._active_profile

    @Property(int, notify=columnCountChanged)
    def columnCount(self) -> int:  # pragma: no cover - getter wykorzystywany w QML
        return self._column_count

    @Property("QVariantList", notify=layoutChanged)
    def layout(self) -> list[dict[str, Any]]:  # pragma: no cover - getter wykorzystywany w QML
        return [dict(entry) for entry in self._layout_state]

    @Property("QVariantList", notify=availablePanelsChanged)
    def availablePanels(self) -> list[dict[str, Any]]:  # pragma: no cover - getter w QML
        catalog = []
        for panel_id, payload in self._panel_catalog.items():
            entry = dict(payload)
            entry["panelId"] = panel_id
            entry["visible"] = self._is_panel_visible(panel_id)
            catalog.append(entry)
        return catalog

    @Slot(str)
    def activateProfile(self, profile: str) -> None:
        profile = profile.strip()
        if not profile:
            return
        profiles = self._storage.setdefault("profiles", {})
        if profile not in profiles:
            profiles[profile] = self._default_profile_payload()
        if profile == self._active_profile:
            return
        self._active_profile = profile
        profile_state = self._profile_state(profile)
        self._column_count = max(1, int(profile_state.column_count))
        self._layout_state = [dict(panel) for panel in profile_state.panels]
        self._normalize_layout()
        self._storage["active_profile"] = profile
        self._persist_storage()
        self.activeProfileChanged.emit()
        self.columnCountChanged.emit()
        self.layoutChanged.emit()

    @Slot(int)
    def setColumnCount(self, columns: int) -> None:
        columns = max(1, min(int(columns), 4))
        if columns == self._column_count:
            return
        self._column_count = columns
        self._normalize_layout()
        self._persist_profile()
        self.columnCountChanged.emit()
        self.layoutChanged.emit()

    @Slot("QVariantList")
    def registerPanels(self, panel_specs: Iterable[Mapping[str, Any]]) -> None:
        updated_catalog = False
        layout_updated = False
        specs = list(panel_specs)
        for spec in specs:
            panel_id = str(spec.get("panelId") or spec.get("id") or "").strip()
            if not panel_id:
                continue
            title = str(spec.get("title") or panel_id)
            icon = str(spec.get("icon") or "")
            default_column = int(spec.get("defaultColumn", 0))
            default_order = int(spec.get("defaultOrder", len(self._layout_state)))
            payload = {
                "title": title,
                "icon": icon,
                "defaultColumn": default_column,
                "defaultOrder": default_order,
            }
            if self._panel_catalog.get(panel_id) != payload:
                self._panel_catalog[panel_id] = payload
                updated_catalog = True
            if not self._find_panel(panel_id):
                self._layout_state.append(
                    {
                        "panelId": panel_id,
                        "column": max(0, default_column),
                        "order": max(0, default_order),
                        "visible": bool(spec.get("visible", True)),
                    }
                )
                layout_updated = True
        if layout_updated:
            self._normalize_layout()
            self._persist_profile()
            self.layoutChanged.emit()
        if updated_catalog:
            self.availablePanelsChanged.emit()

    @Slot(str, int, int)
    def updatePanelPosition(self, panel_id: str, column: int, order: int) -> None:
        panel = self._find_panel(panel_id)
        if panel is None:
            return
        column = max(0, min(int(column), max(0, self._column_count - 1)))
        order = max(0, int(order))
        if panel["column"] == column and panel["order"] == order:
            return
        panel["column"] = column
        panel["order"] = order
        self._normalize_layout()
        self._persist_profile()
        self.layoutChanged.emit()

    @Slot(str, bool)
    def setPanelVisibility(self, panel_id: str, visible: bool) -> None:
        panel = self._find_panel(panel_id)
        if panel is None:
            if visible:
                spec = self._panel_catalog.get(panel_id)
                if spec is None:
                    return
                self._layout_state.append(
                    {
                        "panelId": panel_id,
                        "column": int(spec.get("defaultColumn", 0)),
                        "order": int(spec.get("defaultOrder", len(self._layout_state))),
                        "visible": True,
                    }
                )
            else:
                return
        else:
            if bool(panel.get("visible", True)) == bool(visible):
                return
            panel["visible"] = bool(visible)
        self._normalize_layout()
        self._persist_profile()
        self.availablePanelsChanged.emit()
        self.layoutChanged.emit()

    @Slot(str, result=bool)
    def isPanelVisible(self, panel_id: str) -> bool:
        return self._is_panel_visible(panel_id)

    def _is_panel_visible(self, panel_id: str) -> bool:
        panel = self._find_panel(panel_id)
        if panel is None:
            return False
        return bool(panel.get("visible", True))

    def _find_panel(self, panel_id: str) -> MutableMapping[str, Any] | None:
        for panel in self._layout_state:
            if str(panel.get("panelId")) == panel_id:
                return panel
        return None

    def _normalize_layout(self) -> None:
        max_columns = max(1, self._column_count)
        for panel in self._layout_state:
            panel["column"] = max(0, min(int(panel.get("column", 0)), max_columns - 1))
            panel["order"] = max(0, int(panel.get("order", 0)))
        for column in range(max_columns):
            column_panels = [p for p in self._layout_state if p.get("column") == column]
            column_panels.sort(key=lambda item: item.get("order", 0))
            for idx, panel in enumerate(column_panels):
                panel["order"] = idx
        self._layout_state.sort(key=lambda item: (item.get("column", 0), item.get("order", 0)))

    def _profile_state(self, profile: str) -> _ProfileState:
        payload = self._storage.setdefault("profiles", {}).get(profile) or {}
        column_count = int(payload.get("column_count", 2))
        panels_payload = payload.get("panels")
        if not isinstance(panels_payload, list) or not panels_payload:
            panels = [dict(entry) for entry in _DEFAULT_LAYOUT]
        else:
            panels = [dict(entry) for entry in panels_payload if isinstance(entry, Mapping)]
        return _ProfileState(column_count=max(1, column_count), panels=panels)

    def _persist_profile(self) -> None:
        profiles = self._storage.setdefault("profiles", {})
        profiles[self._active_profile] = {
            "column_count": self._column_count,
            "panels": [dict(entry) for entry in self._layout_state],
        }
        self._storage["active_profile"] = self._active_profile
        self._persist_storage()

    def _persist_storage(self) -> None:
        payload = json.dumps(self._storage, indent=2, ensure_ascii=False)
        self._storage_path.write_text(payload + "\n", encoding="utf-8")

    def _load_storage(self) -> MutableMapping[str, Any]:
        if not self._storage_path.exists():
            return {}
        try:
            data = json.loads(self._storage_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - diagnostyka rzadkich przypadków
            _LOGGER.warning("Nie udało się wczytać %s: %s", self._storage_path, exc)
            return {}
        if not isinstance(data, MutableMapping):
            return {}
        return data

    def _default_profile_payload(self) -> dict[str, Any]:
        return {
            "column_count": 2,
            "panels": [dict(entry) for entry in _DEFAULT_LAYOUT],
        }
