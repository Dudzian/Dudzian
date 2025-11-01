import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6", reason="Wymagany PySide6 do testów UI")

from PySide6.QtCore import QObject, Qt, QMetaObject, QUrl  # type: ignore[attr-defined]
from PySide6.QtQml import QQmlApplicationEngine  # type: ignore[attr-defined]

try:  # pragma: no cover - zależne od środowiska CI
    from PySide6.QtWidgets import QApplication  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - brak bibliotek systemowych
    pytest.skip(f"Brak zależności QtWidgets: {exc}", allow_module_level=True)

from bot_core.strategies.public import StrategyDescriptor
from ui.backend.onboarding_service import OnboardingService


class _StubSecretStore:
    def __init__(self) -> None:
        self.saved: list[tuple[str, str, str, str | None]] = []
        self._token = "onboarding.strategy.credentials.secured"

    def save_exchange_credentials(self, credentials) -> None:  # pragma: no cover - prosty stub
        self.saved.append(
            (
                credentials.normalized_exchange(),
                credentials.api_key,
                credentials.api_secret,
                credentials.api_passphrase,
            )
        )

    def security_details_token(self) -> str:
        return self._token


def _strategy_loader() -> tuple[StrategyDescriptor, ...]:
    return (
        StrategyDescriptor(
            name="grid",
            engine="grid",
            title="Grid Trading",
            license_tier="standard",
            risk_classes=("grid", "neutral"),
            required_data=("ohlcv",),
            tags=("demo",),
            metadata={},
        ),
    )


def _load_strategy_step(service: OnboardingService):
    app = QApplication.instance() or QApplication([])
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("onboardingService", service)
    qml_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "onboarding" / "StrategySetupStep.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    assert engine.rootObjects(), "Nie udało się załadować StrategySetupStep.qml"
    root = engine.rootObjects()[0]
    return root, engine, app  # type: ignore[return-value]


@pytest.mark.timeout(30)
def test_onboarding_service_handles_selection_and_credentials() -> None:
    store = _StubSecretStore()
    service = OnboardingService(
        strategy_loader=_strategy_loader,
        secret_store=store,
        available_exchanges=("binance",),
    )
    assert service.refreshStrategies() is True
    assert service.configurationReady is False

    assert service.selectStrategy("grid") is True
    assert service.selectedStrategy == "grid"
    assert service.configurationReady is False

    assert service.importApiKey("binance", "APIKEY", "SECRET", "") is True
    assert service.configurationReady is True
    assert store.saved == [("binance", "APIKEY", "SECRET", None)]


@pytest.mark.timeout(30)
def test_strategy_setup_step_updates_service() -> None:
    store = _StubSecretStore()
    service = OnboardingService(
        strategy_loader=_strategy_loader,
        secret_store=store,
        available_exchanges=("binance",),
    )
    service.refreshStrategies()

    root, engine, app = _load_strategy_step(service)

    list_view = root.findChild(QObject, "strategySetupList")
    exchange_combo = root.findChild(QObject, "strategySetupExchangeCombo")
    api_key = root.findChild(QObject, "strategySetupApiKeyField")
    api_secret = root.findChild(QObject, "strategySetupApiSecretField")
    save_button = root.findChild(QObject, "strategySetupSaveButton")

    assert list_view is not None
    assert exchange_combo is not None
    assert api_key is not None
    assert api_secret is not None
    assert save_button is not None

    list_view.setProperty("currentIndex", 0)
    app.processEvents()
    assert service.selectedStrategy == "grid"

    exchange_combo.setProperty("currentIndex", 0)
    api_key.setProperty("text", "APIKEY")
    api_secret.setProperty("text", "SECRET")

    QMetaObject.invokeMethod(save_button, "click", Qt.DirectConnection)
    app.processEvents()

    assert service.configurationReady is True
    assert store.saved == [("binance", "APIKEY", "SECRET", None)]

    engine.deleteLater()
    app.quit()
