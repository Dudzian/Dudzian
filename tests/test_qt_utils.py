from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest

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
        "_flush_deferred_qt_objects",
        lambda receivers, *, process_events=None: (
            cleanup_calls.append("owned_cleanup")
            or (process_events() if process_events is not None else None)
        ),
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
    assert cleanup_calls == ["owned_cleanup"]
    assert process_events_calls == ["process"]


def test_scoped_deferred_cleanup_rejects_global_none_receiver(monkeypatch) -> None:
    calls = []

    class Application:
        @staticmethod
        def sendPostedEvents(receiver, event):
            calls.append((receiver, event))

    monkeypatch.setattr(_qt_utils, "QCoreApplication", Application)
    with pytest.raises(ValueError, match="explicit QObject receivers"):
        _qt_utils._flush_deferred_qt_objects((None,))
    assert calls == []


def test_teardown_hosted_qml_engine_core_is_non_aggressive(monkeypatch) -> None:
    engine = _FakeEngine()
    cleanup_calls: list[str] = []
    process_events_calls: list[str] = []

    monkeypatch.setattr(
        _qt_utils,
        "force_qt_cleanup",
        lambda *, process_events=None, process_rounds=10: cleanup_calls.append(
            f"cleanup:{process_rounds}"
        ),
    )

    _qt_utils.teardown_hosted_qml_engine_core(
        engine,
        process_events=lambda: process_events_calls.append("process"),
        context_properties_to_clear=("alpha", "beta"),
    )

    assert [root.delete_later_calls for root in engine.roots] == [0, 0]
    assert engine.context.properties == {"alpha": None, "beta": None}
    assert engine.clear_component_cache_calls == 0
    assert engine.collect_garbage_calls == 0
    assert engine.delete_later_calls == 1
    assert cleanup_calls == []
    assert process_events_calls == ["process", "process"]


def test_teardown_hosted_qml_engine_alias_forwards_to_core(monkeypatch) -> None:
    call_args: list[tuple[object, object, tuple[str, ...]]] = []

    monkeypatch.setattr(
        _qt_utils,
        "teardown_hosted_qml_engine_core",
        lambda engine, *, process_events=None, context_properties_to_clear=(): call_args.append(
            (engine, process_events, tuple(context_properties_to_clear))
        ),
    )
    engine = _FakeEngine()
    process_events = lambda: None

    _qt_utils.teardown_hosted_qml_engine(
        engine,
        process_events=process_events,
        context_properties_to_clear=("alpha", "beta"),
    )

    assert call_args == [(engine, process_events, ("alpha", "beta"))]


def test_teardown_qml_engine_flushes_roots_before_cache_clear(monkeypatch) -> None:
    engine = _FakeEngine()
    call_order: list[str] = []

    def fake_cleanup(receivers, *, process_events=None) -> None:
        call_order.append("roots" if list(receivers) == engine.roots else "engine")
        if process_events is not None:
            process_events()

    monkeypatch.setattr(_qt_utils, "_flush_deferred_qt_objects", fake_cleanup)

    original_clear = engine.clearComponentCache

    def clear_component_cache() -> None:
        call_order.append("clear_component_cache")
        original_clear()

    engine.clearComponentCache = clear_component_cache

    _qt_utils.teardown_qml_engine(
        engine,
        process_events=lambda: call_order.append("process_events"),
        delete_root_objects=True,
    )

    assert [root.delete_later_calls for root in engine.roots] == [1, 1]
    assert call_order == [
        "roots",
        "process_events",
        "clear_component_cache",
        "engine",
        "process_events",
    ]


@pytest.mark.qml
def test_native_engine_cleanup_preserves_another_owners_pending_qobject() -> None:
    environment = os.environ.copy()
    environment.update(QT_QPA_PLATFORM="offscreen", PYTEST_REQUIRE_QML="1")
    code = """
from tests.ui._qt import require_pyside6
require_pyside6(modules=('QtCore', 'QtQml', 'QtWidgets'))
from PySide6.QtCore import QCoreApplication, QEvent, QObject
from PySide6.QtQml import QQmlApplicationEngine
from tests._qt_utils import teardown_qml_engine
import shiboken6
app = QCoreApplication([])
unrelated = QObject()
unrelated_destroyed = []
unrelated.destroyed.connect(lambda: unrelated_destroyed.append(True))
unrelated.deleteLater()
engine = QQmlApplicationEngine()
engine.loadData(b'import QtQml\\nQtObject { objectName: "owned-root" }')
assert engine.rootObjects()
root = engine.rootObjects()[0]
teardown_qml_engine(engine, process_events=app.processEvents)
assert not shiboken6.isValid(root), 'owned root was not destroyed'
assert not shiboken6.isValid(engine), 'owned engine was not destroyed'
assert shiboken6.isValid(unrelated), "another owner's pending object was destroyed"
assert unrelated_destroyed == []
QCoreApplication.sendPostedEvents(unrelated, QEvent.DeferredDelete)
assert not shiboken6.isValid(unrelated)
assert unrelated_destroyed == [True]
"""
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


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


def test_teardown_hosted_qml_engine_contract_ast() -> None:
    """Policy test: helper ma utrzymać ścisły, celowo rygorystyczny kontrakt teardownu."""
    module_path = Path(__file__).resolve().parent / "ui" / "_qml_hosting.py"
    module_ast = ast.parse(module_path.read_text(encoding="utf-8"))

    helper = next(
        node
        for node in module_ast.body
        if isinstance(node, ast.FunctionDef) and node.name == "teardown_hosted_qml_engine"
    )

    # Indeksy body[x] są intencjonalnie "sztywne": chcemy wykrywać nawet
    # pozornie niewinne refaktory, które mogłyby zmienić semantykę helpera.
    # 1) teardown_hosted_item_window(root, host_window)
    first_stmt = helper.body[1]
    assert isinstance(first_stmt, ast.Expr)
    assert isinstance(first_stmt.value, ast.Call)
    assert isinstance(first_stmt.value.func, ast.Name)
    assert first_stmt.value.func.id == "teardown_hosted_item_window"

    # 2) if process_events is not None: process_events()
    second_stmt = helper.body[2]
    assert isinstance(second_stmt, ast.If)
    assert isinstance(second_stmt.test, ast.Compare)
    assert isinstance(second_stmt.test.left, ast.Name)
    assert second_stmt.test.left.id == "process_events"
    assert len(second_stmt.test.ops) == 1
    assert isinstance(second_stmt.test.ops[0], ast.IsNot)
    assert len(second_stmt.test.comparators) == 1
    assert isinstance(second_stmt.test.comparators[0], ast.Constant)
    assert second_stmt.test.comparators[0].value is None
    assert len(second_stmt.body) == 1
    assert isinstance(second_stmt.body[0], ast.Expr)
    process_call = second_stmt.body[0].value
    assert isinstance(process_call, ast.Call)
    assert isinstance(process_call.func, ast.Name)
    assert process_call.func.id == "process_events"
    assert process_call.args == []

    # 3) teardown_hosted_qml_engine_core(..., context_properties_to_clear=...)
    third_stmt = helper.body[3]
    assert isinstance(third_stmt, ast.Expr)
    assert isinstance(third_stmt.value, ast.Call)
    assert isinstance(third_stmt.value.func, ast.Name)
    assert third_stmt.value.func.id == "teardown_hosted_qml_engine_core"

    keyword_map = {keyword.arg: keyword.value for keyword in third_stmt.value.keywords}
    assert "process_events" in keyword_map
    assert "context_properties_to_clear" in keyword_map
    assert isinstance(keyword_map["process_events"], ast.Name)
    assert keyword_map["process_events"].id == "process_events"
