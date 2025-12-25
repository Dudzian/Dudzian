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
    """Centralna definicja opcji startowych oraz ich walidacji."""

    config_path: Path
    profile: str | None = None
    enable_cloud_runtime: bool = False
    qml_path: Path | None = None
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        object.__setattr__(self, "config_path", Path(self.config_path).expanduser().resolve())
        qml_path = Path(self.qml_path).expanduser().resolve() if self.qml_path else None
        object.__setattr__(self, "qml_path", qml_path)
        object.__setattr__(self, "log_level", str(self.log_level).upper())
        self.validate()

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
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
        return parser

    @classmethod
    def parse(cls, argv: list[str] | None = None) -> "AppOptions":
        """Buduje instancję na podstawie argumentów CLI."""

        args = cls.build_parser().parse_args(argv)
        return cls.from_namespace(args)

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "AppOptions":
        """Tworzy opcje na bazie przestrzeni nazw argparse."""

        return cls(
            config_path=Path(args.config),
            profile=args.profile,
            enable_cloud_runtime=args.enable_cloud_runtime,
            qml_path=Path(args.qml) if args.qml else None,
            log_level=args.log_level,
        )

    def validate(self) -> None:
        """Waliduje obecność kluczowych ścieżek i poziom logowania."""

        if not self.config_path.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku konfiguracji UI: {self.config_path}")
        if self.qml_path is not None and not self.qml_path.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku QML: {self.qml_path}")
        if not hasattr(logging, self.log_level):
            raise ValueError(f"Nieprawidłowy poziom logowania: {self.log_level}")

    @property
    def logging_level(self) -> int:
        """Przekłada nazwę poziomu na wartość liczbową używaną w logging."""

        return getattr(logging, self.log_level, logging.INFO)


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
        qml_paths = [
            Path(__file__).resolve().parent / "qml",
            Path(__file__).resolve().parent.parent / "qml",
            qml_file.parent,
        ]
        seen_paths: set[str] = set()
        for import_path in qml_paths:
            import_path_str = import_path.resolve().as_posix()
            if import_path_str in seen_paths:
                continue
            engine.addImportPath(import_path_str)
            seen_paths.add(import_path_str)
        collected_warnings: list[str] = []

        def _on_warnings(warnings: list) -> None:
            for warning in warnings:
                try:
                    location = warning.url().toString() if warning.url().isValid() else qml_file.as_uri()
                    collected_warnings.append(
                        f"{location}:{warning.line()}:{warning.column()}: {warning.description()}"
                    )
                except Exception:  # pragma: no cover - zabezpieczenie na wypadek zmian API
                    collected_warnings.append(str(warning))

        try:
            engine.warnings.connect(_on_warnings)
        except Exception:  # pragma: no cover - defensywnie w razie nietypowych środowisk
            pass
        bridge = QmlContextBridge(
            engine,
            self._config,
            enable_cloud_runtime=self._options.enable_cloud_runtime,
        )
        bridge.install()
        engine.load(QUrl.fromLocalFile(qml_file.as_posix()))
        for message in collected_warnings:
            _LOGGER.error("QML warning: %s", message)
        if not engine.rootObjects():  # pragma: no cover - informacyjne
            details = "; ".join(collected_warnings) if collected_warnings else "brak szczegółów"
            raise RuntimeError(f"Nie udało się załadować QML z {qml_file}: {details}")
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


def main(argv: list[str] | None = None) -> int:
    options = AppOptions.parse(argv)
    logging.basicConfig(level=options.logging_level)
    app = BotPysideApplication(options)
    return app.run()


if __name__ == "__main__":  # pragma: no cover - uruchomienie modułu
    raise SystemExit(main())
