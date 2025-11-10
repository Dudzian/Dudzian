from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.ui._qt import require_pyside6

pytestmark = pytest.mark.qml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

require_pyside6()

from PySide6.QtCore import QObject, QMetaObject, Qt, QUrl, Slot, Q_ARG
from PySide6.QtQml import QQmlApplicationEngine

try:  # pragma: no cover - zależy od środowiska wykonawczego
    from PySide6.QtWidgets import QApplication
except ImportError as exc:  # brak bibliotek (np. libGL)
    pytest.skip(f"Brak zależności QtWidgets: {exc}", allow_module_level=True)


class StubMarketplaceController(QObject):
    def __init__(self) -> None:
        super().__init__()
        self._presets: list[dict[str, object]] = [
            {
                "presetId": "grid_classic",
                "name": "Grid Classic",
                "version": "1.0",
                "profile": "grid",
                "tags": ["grid", "spot"],
                "signatureVerified": True,
                "issues": [],
            }
        ]
        self.list_calls = 0
        self.import_calls: list[str] = []
        self.activate_calls: list[str] = []
        self.remove_calls: list[str] = []
        self.export_calls: list[tuple[str, str, str]] = []

    @Slot(result="QVariantList")
    def marketplaceListPresets(self) -> list[dict[str, object]]:  # type: ignore[override]
        self.list_calls += 1
        return list(self._presets)

    @Slot(QUrl, result="QVariantMap")
    def marketplaceImportPreset(self, url: QUrl) -> dict[str, object]:  # type: ignore[override]
        self.import_calls.append(url.toString())
        self._presets.append(
            {
                "presetId": "grid_pro",
                "name": "Grid Pro",
                "version": "2.0",
                "profile": "grid",
                "tags": ["grid", "pro"],
                "signatureVerified": True,
                "issues": [],
            }
        )
        return {
            "success": True,
            "preset": self._presets[-1],
            "sourcePath": url.toString(),
        }

    @Slot(str, result="QVariantMap")
    def marketplaceActivatePreset(self, preset_id: str) -> dict[str, object]:  # type: ignore[override]
        self.activate_calls.append(preset_id)
        preset = next((p for p in self._presets if p["presetId"] == preset_id), None)
        return {
            "success": True,
            "preset": preset or {"presetId": preset_id, "name": preset_id},
        }

    @Slot(str, result="QVariantMap")
    def marketplaceRemovePreset(self, preset_id: str) -> dict[str, object]:  # type: ignore[override]
        self.remove_calls.append(preset_id)
        self._presets = [p for p in self._presets if p["presetId"] != preset_id]
        return {"success": True, "presetId": preset_id}

    @Slot(str, str, QUrl, result="QVariantMap")
    def marketplaceExportPreset(  # type: ignore[override]
        self, preset_id: str, fmt: str, destination: QUrl
    ) -> dict[str, object]:
        self.export_calls.append((preset_id, fmt, destination.toString()))
        preset = next((p for p in self._presets if p["presetId"] == preset_id), None)
        return {
            "success": True,
            "preset": preset or {"presetId": preset_id, "name": preset_id},
            "path": destination.toString(),
        }


@pytest.mark.timeout(20)
def test_marketplace_view_refresh_and_actions(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])

    controller = StubMarketplaceController()
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("appController", controller)

    view_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "views" / "Marketplace.qml"
    engine.load(QUrl.fromLocalFile(str(view_path)))
    assert engine.rootObjects(), "Nie udało się załadować widoku Marketplace"

    root = engine.rootObjects()[0]
    assert isinstance(root, QObject)

    QMetaObject.invokeMethod(root, "refreshPresets", Qt.DirectConnection)
    app.processEvents()

    presets_variant = root.property("presets")
    assert isinstance(presets_variant, list)
    assert len(presets_variant) == 1
    assert controller.list_calls >= 1

    import_url = QUrl.fromLocalFile(str(tmp_path / "preset.yaml"))
    QMetaObject.invokeMethod(
        root,
        "importPresetFromUrl",
        Qt.DirectConnection,
        Q_ARG(QUrl, import_url),
    )
    app.processEvents()
    assert controller.import_calls[-1] == import_url.toString()
    presets_variant = root.property("presets")
    assert len(presets_variant) == 2

    first_preset = presets_variant[0]
    export_url = QUrl.fromLocalFile(str(tmp_path / "out.yaml"))
    QMetaObject.invokeMethod(
        root,
        "exportPreset",
        Qt.DirectConnection,
        Q_ARG("QVariant", first_preset),
        Q_ARG(QUrl, export_url),
    )
    app.processEvents()
    assert controller.export_calls[-1] == (
        first_preset.get("presetId"),
        root.property("exportFormat"),
        export_url.toString(),
    )

    QMetaObject.invokeMethod(
        root,
        "activatePreset",
        Qt.DirectConnection,
        Q_ARG("QVariant", first_preset),
    )
    app.processEvents()
    assert controller.activate_calls[-1] == first_preset.get("presetId")

    QMetaObject.invokeMethod(
        root,
        "removePreset",
        Qt.DirectConnection,
        Q_ARG("QVariant", first_preset),
    )
    app.processEvents()
    assert controller.remove_calls[-1] == first_preset.get("presetId")

    for obj in engine.rootObjects():
        obj.deleteLater()
    engine.deleteLater()
    app.processEvents()
