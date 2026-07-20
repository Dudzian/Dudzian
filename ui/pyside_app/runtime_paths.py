"""Runtime resource path resolution for source and PyInstaller builds."""

from __future__ import annotations

import sys
from pathlib import Path

BUNDLED_CONFIG = Path("ui/config/preview_local.yaml")
BUNDLED_QML = Path("ui/pyside_app/qml/MainWindow.qml")
SHARED_QML_ROOT = Path("ui/qml")
PYSIDE_QML_ROOT = Path("ui/pyside_app/qml")


def is_frozen_app() -> bool:
    """Return true when running from a PyInstaller-frozen executable."""

    return bool(getattr(sys, "frozen", False))


def bundle_root() -> Path:
    """Resolve the immutable app bundle root without using the current directory."""

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(str(meipass)).resolve()
    if is_frozen_app():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def resolve_resource_path(explicit: str | Path | None, bundled_relative: str | Path) -> Path:
    """Resolve an explicit path or a path relative to the source/frozen bundle root."""

    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    return (bundle_root() / Path(bundled_relative)).resolve()


def default_config_path() -> Path:
    """Return bundled safe preview configuration path."""

    return resolve_resource_path(None, BUNDLED_CONFIG)


def default_qml_path() -> Path:
    """Return bundled main QML entrypoint path."""

    return resolve_resource_path(None, BUNDLED_QML)


def qml_import_roots(qml_file: Path | None = None) -> list[Path]:
    """Return QML import roots for source and frozen execution."""

    root = bundle_root()
    candidates = [
        root / PYSIDE_QML_ROOT,
        root / SHARED_QML_ROOT,
    ]
    if qml_file is not None:
        candidates.append(qml_file.parent)
    resolved: list[Path] = []
    for candidate in candidates:
        path = candidate.resolve()
        if path.exists() and path not in resolved:
            resolved.append(path)
    return resolved
