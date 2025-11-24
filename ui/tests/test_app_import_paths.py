"""Testy logiki wejścia aplikacji PySide6 (bez uruchamiania Qt)."""

from __future__ import annotations

from pathlib import Path
import sys
import types


class DummySignal:
    def __init__(self, *_args, **_kwargs):
        self._callbacks: list[types.FunctionType] = []

    def connect(self, callback):  # type: ignore[no-untyped-def]
        self._callbacks.append(callback)

    def emit(self, *args, **kwargs):  # noqa: ANN001, D401 - API Qt
        for callback in list(self._callbacks):
            try:
                callback(*args, **kwargs)
            except Exception:
                pass


def DummySlot(*_args, **_kwargs):  # noqa: D401 - imitacja dekoratora Qt
    def decorator(func):
        return func

    return decorator


def DummyProperty(*_args, **_kwargs):  # noqa: D401 - imitacja dekoratora Qt
    def decorator(func):
        class _Property:
            def __init__(self, fget):
                self.fget = fget

            def __call__(self, *args, **kwargs):
                return self.fget(*args, **kwargs)

            def setter(self, _setter_func):
                return self

        return _Property(func)

    return decorator


class DummyQObject:
    def __init__(self, *_args, **_kwargs):
        super().__init__()


class DummyQTimer:
    def __init__(self, *_args, **_kwargs):
        self.timeout = DummySignal()
        self.interval: int | None = None
        self.single_shot = False

    def setInterval(self, ms: int):
        self.interval = ms

    def setSingleShot(self, flag: bool):
        self.single_shot = flag

    def start(self, *_args, **_kwargs):
        return None

    def stop(self):
        return None

    def isActive(self) -> bool:
        return False


class DummyContext:
    def setContextProperty(self, *_args, **_kwargs):  # noqa: N802 - zgodnie z API Qt
        return None


class DummyQGuiApplication:
    _instance: "DummyQGuiApplication | None" = None

    def __init__(self, _argv: list[str]):
        DummyQGuiApplication._instance = self

    @classmethod
    def instance(cls) -> "DummyQGuiApplication | None":
        return cls._instance

    def exec(self) -> int:  # pragma: no cover - nie używamy w testach import path
        return 0


class DummyQUrl:
    @staticmethod
    def fromLocalFile(path: str) -> str:
        return path


class DummyEngine:
    def __init__(self):
        self.import_paths: list[str] = []
        self.loaded: str | None = None

    def addImportPath(self, path: str) -> None:  # noqa: N802 - zgodnie z API Qt
        self.import_paths.append(path)

    def load(self, url: str) -> None:  # noqa: N802 - zgodnie z API Qt
        self.loaded = url

    def rootObjects(self) -> list[object]:  # noqa: N802 - zgodnie z API Qt
        return [object()] if self.loaded else []

    def rootContext(self) -> DummyContext:  # noqa: N802 - zgodnie z API Qt
        return DummyContext()


class DummyBridge:
    def __init__(self, engine: DummyEngine, config: object, enable_cloud_runtime: bool):
        self.engine = engine
        self.config = config
        self.enable_cloud_runtime = enable_cloud_runtime
        self.installed = False

    def install(self) -> None:
        self.installed = True


def _install_pyside6_dummies() -> None:
    """Rejestruje minimalne moduły PySide6 wymagane do importu app.py."""

    qtgui = types.ModuleType("PySide6.QtGui")
    qtgui.QGuiApplication = DummyQGuiApplication

    qtqml = types.ModuleType("PySide6.QtQml")
    qtqml.QQmlApplicationEngine = DummyEngine

    qtcore = types.ModuleType("PySide6.QtCore")
    qtcore.QUrl = DummyQUrl
    qtcore.QObject = DummyQObject
    qtcore.Signal = DummySignal
    qtcore.Slot = DummySlot
    qtcore.Property = DummyProperty
    qtcore.QTimer = DummyQTimer

    pyside6 = types.ModuleType("PySide6")
    pyside6.QtGui = qtgui
    pyside6.QtQml = qtqml
    pyside6.QtCore = qtcore

    sys.modules.setdefault("PySide6", pyside6)
    sys.modules.setdefault("PySide6.QtGui", qtgui)
    sys.modules.setdefault("PySide6.QtQml", qtqml)
    sys.modules.setdefault("PySide6.QtCore", qtcore)


_install_pyside6_dummies()

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.pyside_app import app


def test_custom_qml_directory_is_added_to_import_paths(monkeypatch, tmp_path):
    config_path = tmp_path / "ui.yaml"
    config_path.write_text("history_limit: 5\n", encoding="utf-8")

    qml_dir = tmp_path / "custom"
    qml_dir.mkdir()
    qml_file = qml_dir / "Main.qml"
    qml_file.write_text("import QtQuick 2.15\nItem {}\n", encoding="utf-8")

    monkeypatch.setattr(app, "QGuiApplication", DummyQGuiApplication)
    monkeypatch.setattr(app, "QQmlApplicationEngine", DummyEngine)
    monkeypatch.setattr(app, "QUrl", DummyQUrl)
    monkeypatch.setattr(app, "QmlContextBridge", DummyBridge)

    options = app.AppOptions(config_path=config_path, qml_path=Path(qml_file))
    pyside_app = app.BotPysideApplication(options)
    engine = pyside_app.load()

    assert qml_file.parent.as_posix() in engine.import_paths
