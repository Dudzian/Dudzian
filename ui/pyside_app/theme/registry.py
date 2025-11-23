"""Ładowanie palet kolorów i ikon dla PySide6 UI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, MutableMapping, Optional, cast


THEME_DIR = Path(__file__).resolve().parent
DEFAULT_PALETTE_PATH = THEME_DIR / "palette.json"
DEFAULT_ICONS_PATH = THEME_DIR / "icons.json"


@dataclass
class ThemeRegistry:
    """Przechowuje dane motywu i udostępnia helpery do QML."""

    palettes: Mapping[str, Mapping[str, str]]
    gradients: Mapping[str, Mapping[str, List[str]]]
    icons: Mapping[str, Mapping[str, str]]
    icons_dir: Path

    def available_palettes(self) -> Iterable[str]:
        return self.palettes.keys()

    def _palette_table(self, palette: str) -> Mapping[str, str]:
        if palette not in self.palettes:
            raise KeyError(f"Nieobsługiwany motyw: {palette}")
        return self.palettes[palette]

    def color(self, palette: str, token: str) -> str:
        table = self._palette_table(palette)
        if token not in table:
            raise KeyError(f"Brak koloru '{token}' w motywie {palette}")
        return table[token]

    def gradient(self, palette: str, token: str) -> List[str]:
        pal_gradients = self.gradients.get(palette, {})
        colors = pal_gradients.get(token, [])
        return list(colors)

    def icon_url(self, token: str) -> str:
        icon = self.icons.get(token)
        if not icon:
            raise KeyError(f"Brak ikony '{token}' w rejestrze")
        icon_type = icon.get("type", "svg")
        if icon_type != "svg":
            raise ValueError(f"Nieobsługiwany typ ikony: {icon_type}")
        path = self.icons_dir / icon["file"]
        if not path.exists():
            raise FileNotFoundError(path)
        return path.as_uri()

    def icon_glyph(self, token: str) -> str:
        icon = self.icons.get(token)
        if not icon:
            return ""
        glyph = icon.get("glyph")
        if isinstance(glyph, str) and glyph:
            return glyph
        return ""


def _load_json(path: Path) -> MutableMapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return cast(MutableMapping[str, object], json.load(handle))


def load_default_theme(
    palette_path: Optional[Path] = None,
    icons_path: Optional[Path] = None,
) -> ThemeRegistry:
    palette_data = _load_json(palette_path or DEFAULT_PALETTE_PATH)
    icons_data = _load_json(icons_path or DEFAULT_ICONS_PATH)
    palettes = cast(Mapping[str, Mapping[str, str]], palette_data.get("palettes", {}))
    gradients = cast(Mapping[str, Mapping[str, List[str]]], palette_data.get("gradients", {}))
    icons = cast(Mapping[str, Mapping[str, str]], icons_data.get("icons", {}))
    return ThemeRegistry(
        palettes=palettes, gradients=gradients, icons=icons, icons_dir=THEME_DIR / "icons"
    )
