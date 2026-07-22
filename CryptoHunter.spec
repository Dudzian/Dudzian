# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onedir build spec for the CryptoHunter Windows desktop app."""

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_delvewheel_libs_directory,
    collect_submodules,
)

ROOT = Path(SPECPATH)  # noqa: F821

app_datas = [
    (str(ROOT / "ui" / "pyside_app" / "qml"), "ui/pyside_app/qml"),
    (str(ROOT / "ui" / "qml"), "ui/qml"),
    (str(ROOT / "ui" / "config" / "preview_local.yaml"), "ui/config"),
    (
        str(ROOT / "ui" / "pyside_app" / "theme" / "palette.json"),
        "ui/pyside_app/theme",
    ),
    (
        str(ROOT / "ui" / "pyside_app" / "theme" / "icons.json"),
        "ui/pyside_app/theme",
    ),
    (
        str(ROOT / "ui" / "pyside_app" / "theme" / "icons"),
        "ui/pyside_app/theme/icons",
    ),
]

app_binaries = []

app_datas, app_binaries = collect_delvewheel_libs_directory(
    "numpy",
    datas=app_datas,
    binaries=app_binaries,
)

hiddenimports = [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtQuickControls2",
    "ui.pyside_app",
    "ui.pyside_app.__main__",
    "ui.pyside_app.app",
    "ui.pyside_app.runtime_paths",
    "ui.pyside_app.smoke",
]
hiddenimports += collect_submodules("ui.backend")

block_cipher = None

a = Analysis(  # noqa: F821
    [str(ROOT / "ui" / "pyside_app" / "__main__.py")],
    pathex=[str(ROOT)],
    binaries=app_binaries,
    datas=app_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)  # noqa: F821
exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CryptoHunter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="CryptoHunter",
)
