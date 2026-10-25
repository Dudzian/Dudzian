from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

from tests.ui import _qt


@pytest.fixture
def windows_offscreen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    for name in ("QT_QPA_FONTDIR", "WINDIR", "SystemRoot"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("root_variable", ["WINDIR", "SystemRoot"])
def test_windows_offscreen_uses_existing_system_fonts(
    windows_offscreen, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, root_variable: str
) -> None:
    fonts = tmp_path / "Windows" / "Fonts"
    fonts.mkdir(parents=True)
    # The setup helper checks paths/extensions; Qt validates the font bytes.
    (fonts / "system.OTF").write_bytes(b"font fixture")
    monkeypatch.setenv(root_variable, str(fonts.parent))
    if root_variable == "SystemRoot":
        monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen:configfile=custom.json")

    assert _qt.prepare_windows_offscreen_fonts() == fonts
    assert os.environ["QT_QPA_FONTDIR"] == str(fonts)


def test_windows_offscreen_preserves_explicit_font_directory(
    windows_offscreen, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fonts = tmp_path / "Custom Fonts"
    fonts.mkdir()
    (fonts / "custom.ttf").write_bytes(b"font fixture")
    monkeypatch.setenv("QT_QPA_FONTDIR", str(fonts))
    monkeypatch.setenv("WINDIR", str(tmp_path / "invalid-system-root"))

    assert _qt.prepare_windows_offscreen_fonts() == fonts
    assert os.environ["QT_QPA_FONTDIR"] == str(fonts)


def test_windows_offscreen_rejects_missing_system_root(windows_offscreen) -> None:
    with pytest.raises(AssertionError, match="WINDIR or SystemRoot"):
        _qt.prepare_windows_offscreen_fonts()
    assert "QT_QPA_FONTDIR" not in os.environ


@pytest.mark.parametrize("case", ["missing", "empty", "explicit-blank", "explicit-unsupported"])
def test_windows_offscreen_rejects_unusable_font_directory(
    windows_offscreen, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str
) -> None:
    fonts = tmp_path / "Windows" / "Fonts"
    monkeypatch.setenv("WINDIR", str(fonts.parent))
    if case != "missing":
        fonts.mkdir(parents=True)
    if case == "explicit-blank":
        (fonts / "system.ttf").write_bytes(b"font fixture")
        monkeypatch.setenv("QT_QPA_FONTDIR", "")
    elif case == "explicit-unsupported":
        (fonts / "system.ttf").write_bytes(b"font fixture")
        custom = tmp_path / "custom"
        custom.mkdir()
        (custom / "not-a-font.txt").write_text("not a font", encoding="utf-8")
        (custom / "collection.ttc").write_bytes(b"not scanned by Qt 6.7.2 FreeType")
        monkeypatch.setenv("QT_QPA_FONTDIR", str(custom))
    previous = os.environ.get("QT_QPA_FONTDIR")

    with pytest.raises(AssertionError, match="Windows offscreen Qt"):
        _qt.prepare_windows_offscreen_fonts()
    assert os.environ.get("QT_QPA_FONTDIR") == previous


@pytest.mark.parametrize(
    ("platform", "qpa"),
    [("linux", "offscreen"), ("darwin", "offscreen"), ("win32", "windows"), ("win32", "minimal")],
)
def test_other_platforms_do_not_configure_font_environment(
    monkeypatch: pytest.MonkeyPatch, platform: str, qpa: str
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setenv("QT_QPA_PLATFORM", qpa)
    monkeypatch.setenv("QT_QPA_FONTDIR", "keep-explicit-setting")

    assert _qt.prepare_windows_offscreen_fonts() is None
    assert os.environ["QT_QPA_FONTDIR"] == "keep-explicit-setting"
