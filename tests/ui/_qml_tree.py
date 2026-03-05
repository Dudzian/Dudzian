from __future__ import annotations


def walk_qml_items(obj: object, limit: int = 8000) -> tuple[list[object], bool]:
    """Traverse both childItems() and QObject children to catch QtQuickControls content.

    Some QtQuickControls expose only wrapper items via childItems(), while the real content
    lives under QObject.children(), so we walk both graphs to avoid missing nodes.
    The seen set and traversal limit are deliberate guardrails against cycles and graph blowups.
    """
    out: list[object] = []
    stack: list[object] = [obj]
    seen: set[int] = set()
    capped = False
    while stack:
        if len(out) >= limit:
            capped = True
            break
        cur = stack.pop()
        if cur is None:
            continue
        ident = id(cur)
        if ident in seen:
            continue
        seen.add(ident)
        out.append(cur)
        try:
            child_items = cur.childItems()  # type: ignore[attr-defined]
        except Exception:
            child_items = None
        if child_items is not None:
            try:
                for item in child_items:
                    if item is not None:
                        stack.append(item)
            except Exception:
                pass
        try:
            stack.extend(cur.children())  # type: ignore[attr-defined]
        except Exception:
            pass
    return out, capped


def find_by_object_name(root_obj: object, name: str) -> object | None:
    items, _ = walk_qml_items(root_obj)
    for obj in items:
        try:
            object_name = obj.objectName()  # type: ignore[attr-defined]
        except Exception:
            object_name = None
        if object_name == name:
            return obj
    return None
