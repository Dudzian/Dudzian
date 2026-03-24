from __future__ import annotations

from typing import Callable, Iterable

from tests.ui._qt_utils import teardown_hosted_qml_engine_core
from tests.ui._qt import require_pyside6
from tests.ui._qml_tree import walk_qml_items

require_pyside6()

from PySide6.QtCore import QObject

try:
    from PySide6.QtQuick import QQuickItem, QQuickWindow
except ImportError as exc:  # pragma: no cover - zależne od środowiska CI
    import pytest

    pytest.skip(
        f"Brak zależności systemowych Qt Quick (np. libEGL.so.1): {exc}",
        allow_module_level=True,
    )


def safe_qml_property(obj: QObject, name: str) -> object:
    try:
        return obj.property(name)
    except RuntimeError as exc:
        return f"<unavailable:{name}:{exc}>"


def collect_object_names(root: QObject, prefix: str) -> list[str]:
    names: list[str] = []
    items, _ = walk_qml_items(root)
    for child in items:
        if not isinstance(child, QObject):
            continue
        try:
            child_name = child.objectName()
        except RuntimeError:
            continue
        if child_name.startswith(prefix):
            names.append(child_name)
    return sorted(set(names))


def ensure_item_has_host_window(
    root: QObject,
    *,
    default_width: int = 960,
    default_height: int = 600,
) -> QQuickWindow | None:
    if not isinstance(root, QQuickItem):
        return None

    existing_window = safe_qml_property(root, "window")
    if isinstance(existing_window, QObject):
        return None

    host_window = QQuickWindow()
    host_content = host_window.contentItem()
    if isinstance(host_content, QQuickItem):
        root.setParentItem(host_content)

    width = safe_qml_property(root, "implicitWidth")
    height = safe_qml_property(root, "implicitHeight")
    host_window.setWidth(
        int(width) if isinstance(width, (int, float)) and width > 0 else default_width
    )
    host_window.setHeight(
        int(height) if isinstance(height, (int, float)) and height > 0 else default_height
    )
    root.setWidth(host_window.width())
    root.setHeight(host_window.height())
    host_window.show()
    return host_window


def is_item_hosted_in_window(root: QObject, host_window: QQuickWindow | None) -> bool:
    if host_window is None or not isinstance(root, QQuickItem):
        return False

    host_content = host_window.contentItem()
    if not isinstance(host_content, QQuickItem):
        return False

    item: QQuickItem | None = root
    while isinstance(item, QQuickItem):
        if item is host_content:
            return True
        item = item.parentItem()
    return False


def teardown_hosted_item_window(root: QObject, host_window: QQuickWindow | None) -> None:
    if host_window is None:
        return

    try:
        host_content = host_window.contentItem()
    except RuntimeError:
        host_content = None

    if isinstance(root, QQuickItem) and isinstance(host_content, QQuickItem):
        try:
            if root.parentItem() is host_content:
                root.setParentItem(None)
        except RuntimeError:
            pass

    try:
        host_window.close()
    except RuntimeError:
        pass

    try:
        host_window.deleteLater()
    except RuntimeError:
        pass


def teardown_hosted_qml_engine(
    root: QObject,
    host_window: QQuickWindow | None,
    engine: object,
    *,
    process_events: Callable[[], None] | None = None,
    context_properties_to_clear: Iterable[str] = (),
) -> None:
    """Spójny teardown dla rootów hostowanych przez ensure_item_has_host_window()."""

    teardown_hosted_item_window(root, host_window)
    if process_events is not None:
        process_events()

    teardown_hosted_qml_engine_core(
        engine,
        process_events=process_events,
        context_properties_to_clear=context_properties_to_clear,
    )
