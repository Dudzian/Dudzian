# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onedir build spec for the GymOS Windows desktop app."""

from pathlib import Path

ROOT = Path(SPECPATH)

app_datas = [
    (str(ROOT / "ui" / "pyside_app" / "qml"), "ui/pyside_app/qml"),
    (str(ROOT / "ui" / "qml"), "ui/qml"),
    (str(ROOT / "ui" / "config" / "preview_local.yaml"), "ui/config"),
]
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

block_cipher = None

a = Analysis(
    [str(ROOT / "ui" / "pyside_app" / "__main__.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=app_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="GymOS",
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
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="GymOS",
)
