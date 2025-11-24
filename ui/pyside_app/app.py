"""Uruchomienie aplikacji PySide6 oraz rejestracja kontekstu QML."""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QUrl

from .config import UiAppConfig, load_ui_app_config
from .qml_bridge import QmlContextBridge

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AppOptions:
    """Parametry startowe przekazywane z CLI lub testów."""

    config_path: Path
    profile: str | None = None
    enable_cloud_runtime: bool = False
    qml_path: Path | None = None


class BotPysideApplication:
    """Inicjuje QGuiApplication, ładuje QML i kontrolery backendu."""

    def __init__(self, options: AppOptions) -> None:
        self._options = options
        self._qt_app: QGuiApplication | None = None
        self._engine: QQmlApplicationEngine | None = None
        self._config: UiAppConfig | None = None

    @property
    def engine(self) -> QQmlApplicationEngine | None:
        return self._engine

    def _ensure_qt_app(self) -> QGuiApplication:
        instance = QGuiApplication.instance()
        if instance is not None:
            return instance  # pragma: no cover - wykorzystywane w testach, gdy istnieje globalna instancja
        self._qt_app = QGuiApplication(sys.argv)
        return self._qt_app

    def load(self) -> QQmlApplicationEngine:
        """Buduje silnik QML wraz z kontekstem."""

        app = self._ensure_qt_app()
        _LOGGER.debug("QGuiApplication instance: %s", app)
        self._config = load_ui_app_config(
            self._options.config_path,
            profile=self._options.profile,
            default_qml=self._options.qml_path,
        )
        qml_file = (self._options.qml_path or self._config.qml_entrypoint).resolve()
        engine = QQmlApplicationEngine()
        qml_paths = {
            Path(__file__).resolve().parent / "qml",
            Path(__file__).resolve().parent.parent / "qml",
            qml_file.parent,
        }
        for import_path in qml_paths:
            engine.addImportPath(import_path.as_posix())
        bridge = QmlContextBridge(
            engine,
            self._config,
            enable_cloud_runtime=self._options.enable_cloud_runtime,
        )
        bridge.install()
        engine.load(QUrl.fromLocalFile(qml_file.as_posix()))
        if not engine.rootObjects():  # pragma: no cover - informacyjne
            raise RuntimeError(f"Nie udało się załadować QML z {qml_file}")
        self._engine = engine
        return engine

    def run(self) -> int:
        """Ładuje QML i uruchamia główną pętlę zdarzeń."""

        engine = self.load()
        if not engine.rootObjects():  # pragma: no cover - zabezpieczenie
            return 1
        qt_app = QGuiApplication.instance()
        if qt_app is None:
            raise RuntimeError("Brak instancji QGuiApplication po zainicjowaniu UI")
        _LOGGER.info("PySide6 UI gotowe – załadowano %d obiektów root", len(engine.rootObjects()))
        return qt_app.exec()


def parse_cli_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Uruchamia PySide6 UI dla Stage6")
    parser.add_argument(
        "--config",
        default="ui/config/example.yaml",
        help="Ścieżka do profilu UI (domyślnie ui/config/example.yaml)",
    )
    parser.add_argument(
        "--profile",
        help="Opcjonalny profil z sekcji profiles w pliku YAML",
    )
    parser.add_argument(
        "--enable-cloud-runtime",
        action="store_true",
        help="Sygnalizuje, że UI ma komunikować się z backendem cloudowym",
    )
    parser.add_argument(
        "--qml",
        help="Własny plik QML (domyślnie ui/pyside_app/qml/MainWindow.qml)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Poziom logowania PySide6 UI",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_cli_arguments(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    options = AppOptions(
        config_path=Path(args.config).expanduser(),
        profile=args.profile,
        enable_cloud_runtime=args.enable_cloud_runtime,
        qml_path=Path(args.qml).expanduser() if args.qml else None,
    )
    app = BotPysideApplication(options)
    return app.run()


if __name__ == "__main__":  # pragma: no cover - uruchomienie modułu
    raise SystemExit(main())
