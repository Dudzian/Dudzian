"""Rejestracja kontrolerów PySide6 w kontekście QML."""
from __future__ import annotations

from PySide6.QtQml import QQmlApplicationEngine

from ui.backend import DiagnosticsController, LicensingController, RuntimeService

from .config import UiAppConfig
from .controllers import (
    LayoutProfileController,
    ModeWizardController,
    StrategyManagementController,
)
from .theme import ThemeBridge, load_default_theme


class QmlContextBridge:
    """Wstrzykuje kontrolery i konfigurację do QQmlApplicationEngine."""

    def __init__(
        self,
        engine: QQmlApplicationEngine,
        config: UiAppConfig,
        *,
        enable_cloud_runtime: bool,
    ) -> None:
        self._engine = engine
        self._config = config
        self._enable_cloud_runtime = enable_cloud_runtime
        self.runtime_service = RuntimeService(
            default_limit=max(5, config.decision_limit),
            cloud_runtime_enabled=enable_cloud_runtime,
        )
        self.licensing_controller = LicensingController()
        self.diagnostics_controller = DiagnosticsController()
        self.layout_controller = LayoutProfileController()
        self.mode_wizard_controller = ModeWizardController(self.runtime_service, config)
        self.strategy_management_controller = StrategyManagementController()
        registry = load_default_theme()
        self.theme_bridge = ThemeBridge(registry, palette=config.theme_palette)

    def install(self) -> None:
        context = self._engine.rootContext()
        context.setContextProperty("uiConfig", self._config.as_variant())
        context.setContextProperty("cloudRuntimeEnabled", self._enable_cloud_runtime)
        context.setContextProperty("runtimeService", self.runtime_service)
        context.setContextProperty("licensingController", self.licensing_controller)
        context.setContextProperty("diagnosticsController", self.diagnostics_controller)
        context.setContextProperty("layoutController", self.layout_controller)
        context.setContextProperty("modeWizardController", self.mode_wizard_controller)
        context.setContextProperty(
            "strategyManagementController", self.strategy_management_controller
        )
        context.setContextProperty("theme", self.theme_bridge)
