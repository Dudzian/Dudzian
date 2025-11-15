"""Testy weryfikujące motyw PySide6 i dostępność ikon."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from ui.pyside_app.theme import load_default_theme


REQUIRED_COLOR_TOKENS = [
    "background",
    "surface",
    "surfaceElevated",
    "accent",
    "textPrimary",
]

REQUIRED_ICONS = [
    "refresh",
    "fingerprint",
    "shield",
    "cloud",
    "package",
    "diagnostics",
    "mode_wizard",
    "scalping",
    "swing",
    "hedge",
    "futures",
    "copy",
    "strategy_manager",
]


def _path_from_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    return Path(parsed.path)


def test_theme_contains_expected_colors() -> None:
    registry = load_default_theme()
    for token in REQUIRED_COLOR_TOKENS:
        assert registry.color("dark", token), f"Brak koloru {token}"
        assert registry.color("light", token), f"Brak koloru {token} w jasnym motywie"


def test_theme_icons_exist_on_disk() -> None:
    registry = load_default_theme()
    for icon in REQUIRED_ICONS:
        uri = registry.icon_url(icon)
        icon_path = _path_from_uri(uri)
        assert icon_path.exists(), f"Ikona {icon} nie istnieje: {icon_path}"
        assert icon_path.suffix == ".svg"
