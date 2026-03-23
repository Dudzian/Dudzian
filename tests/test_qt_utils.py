from __future__ import annotations

import ast
from pathlib import Path

from tests import _qt_utils
from tests.ui import _qt_utils as _ui_qt_utils


class _FakeRoot:
    def __init__(self) -> None:
        self.delete_later_calls = 0

    def deleteLater(self) -> None:
        self.delete_later_calls += 1


class _FakeContext:
    def __init__(self) -> None:
        self.properties: dict[str, object] = {}

    def setContextProperty(self, name: str, value: object) -> None:
        self.properties[name] = value


class _FakeEngine:
    def __init__(self) -> None:
        self.roots = [_FakeRoot(), _FakeRoot()]
        self.context = _FakeContext()
        self.clear_component_cache_calls = 0
        self.collect_garbage_calls = 0
        self.delete_later_calls = 0

    def rootContext(self) -> _FakeContext:
        return self.context

    def rootObjects(self) -> list[_FakeRoot]:
        return self.roots

    def clearComponentCache(self) -> None:
        self.clear_component_cache_calls += 1

    def collectGarbage(self) -> None:
        self.collect_garbage_calls += 1

    def deleteLater(self) -> None:
        self.delete_later_calls += 1


def test_teardown_qml_engine_can_skip_root_deletion(monkeypatch) -> None:
    engine = _FakeEngine()
    cleanup_calls: list[str] = []
    process_events_calls: list[str] = []

    monkeypatch.setattr(
        _qt_utils,
        "force_qt_cleanup",
        lambda *, process_events=None, process_rounds=10: cleanup_calls.append(
            f"cleanup:{process_rounds}"
        ) or (process_events() if process_events is not None else None),
    )

    _qt_utils.teardown_qml_engine(
        engine,
        process_events=lambda: process_events_calls.append("process"),
        context_properties_to_clear=("alpha", "beta"),
        delete_root_objects=False,
    )

    assert [root.delete_later_calls for root in engine.roots] == [0, 0]
    assert engine.context.properties == {"alpha": None, "beta": None}
    assert engine.clear_component_cache_calls == 1
    assert engine.collect_garbage_calls == 1
    assert engine.delete_later_calls == 1
    assert cleanup_calls == ["cleanup:10", "cleanup:10"]
    assert process_events_calls == ["process", "process"]


def test_ui_qt_utils_shim_is_module_alias() -> None:
    assert _ui_qt_utils is _qt_utils


def test_ui_qt_utils_shim_follows_alias_only_policy() -> None:
    """Policy test: shim ma pozostać minimalnym aliasem bez lokalnych dodatków."""
    module_path = Path(__file__).resolve().parent / "ui" / "_qt_utils.py"
    module_ast = ast.parse(module_path.read_text(encoding="utf-8"))

    assert len(module_ast.body) == 3
    assert isinstance(module_ast.body[0], ast.Import)
    assert [alias.name for alias in module_ast.body[0].names] == ["sys"]

    assert isinstance(module_ast.body[1], ast.ImportFrom)
    assert module_ast.body[1].module == "tests"
    assert [alias.name for alias in module_ast.body[1].names] == ["_qt_utils"]
    assert [alias.asname for alias in module_ast.body[1].names] == ["_shared_qt_utils"]

    assign = module_ast.body[2]
    assert isinstance(assign, ast.Assign)
    assert len(assign.targets) == 1
    target = assign.targets[0]
    assert isinstance(target, ast.Subscript)
    assert isinstance(target.value, ast.Attribute)
    assert isinstance(target.value.value, ast.Name)
    assert target.value.value.id == "sys"
    assert target.value.attr == "modules"
    assert isinstance(target.slice, ast.Name)
    assert target.slice.id == "__name__"
    assert isinstance(assign.value, ast.Name)
    assert assign.value.id == "_shared_qt_utils"
