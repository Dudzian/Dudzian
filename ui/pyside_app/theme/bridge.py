"""Bridge udostępniający kolory i ikony do QML."""

from __future__ import annotations

from typing import List

from PySide6.QtCore import QObject, Property, Signal, Slot

from .registry import ThemeRegistry, load_default_theme


class ThemeBridge(QObject):
    """Ekspozycja ThemeRegistry jako obiektu QML."""

    paletteChanged = Signal()

    def __init__(self, registry: ThemeRegistry | None = None, palette: str = "dark") -> None:
        super().__init__()
        self._registry = registry or load_default_theme()
        self._palette = palette if palette in self._registry.available_palettes() else "dark"

    @Property(str, notify=paletteChanged)
    def palette(self) -> str:
        return self._palette

    @Slot(str)
    def setPalette(self, palette: str) -> None:  # noqa: N802 (Qt naming)
        if palette == self._palette:
            return
        if palette not in self._registry.available_palettes():
            return
        self._palette = palette
        self.paletteChanged.emit()

    @Slot(str, result=str)
    def color(self, token: str) -> str:
        return self._registry.color(self._palette, token)

    @Slot(str, result=str)
    def iconUrl(self, token: str) -> str:  # noqa: N802
        return self._registry.icon_url(token)

    @Slot(str, result=list)
    def gradient(self, token: str) -> List[str]:  # noqa: N802
        return self._registry.gradient(self._palette, token)

    @Slot(str, result=str)
    def iconGlyph(self, token: str) -> str:  # noqa: N802
        return self._registry.icon_glyph(token)
