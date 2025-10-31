"""Logika onboardingowa obsługująca wybór strategii i import kluczy API."""
from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Iterable, Sequence

from PySide6.QtCore import QObject, Property, Signal, Slot

from core.runtime.strategy_catalog import StrategyDescriptor, list_available_strategies
from core.security.secret_store import ExchangeCredentials, SecretStore, SecretStoreError


_DEFAULT_EXCHANGES: tuple[str, ...] = (
    "binance",
    "binanceus",
    "kraken",
    "coinbase",
    "bitfinex",
)


class OnboardingService(QObject):
    """Zapewnia dane dla kroków onboardingu (strategie + klucze API)."""

    strategiesChanged = Signal()
    selectedStrategyChanged = Signal()
    configurationReadyChanged = Signal()
    statusMessageIdChanged = Signal()
    statusDetailsChanged = Signal()
    lastSavedExchangeChanged = Signal()
    availableExchangesChanged = Signal()

    def __init__(
        self,
        *,
        strategy_loader: Callable[[], Sequence[StrategyDescriptor]] | None = None,
        secret_store: SecretStore | None = None,
        available_exchanges: Iterable[str] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._strategy_loader = strategy_loader or list_available_strategies
        self._secret_store = secret_store or SecretStore()
        exchanges = tuple(dict.fromkeys((available_exchanges or _DEFAULT_EXCHANGES)))
        self._available_exchanges = exchanges or _DEFAULT_EXCHANGES
        self._strategies: tuple[StrategyDescriptor, ...] = ()
        self._strategies_cache: list[dict] = []
        self._selected_strategy = ""
        self._selected_strategy_title = ""
        self._has_credentials = False
        self._status_message_id = ""
        self._status_details = ""
        self._last_saved_exchange = ""
        self._cached_ready_state = False

    # ------------------------------------------------------------------
    @Property("QVariantList", notify=strategiesChanged)
    def strategies(self) -> list[dict]:  # type: ignore[override]
        return list(self._strategies_cache)

    @Property(str, notify=selectedStrategyChanged)
    def selectedStrategy(self) -> str:  # type: ignore[override]
        return self._selected_strategy

    @Property(str, notify=selectedStrategyChanged)
    def selectedStrategyTitle(self) -> str:  # type: ignore[override]
        return self._selected_strategy_title

    @Property(bool, notify=configurationReadyChanged)
    def configurationReady(self) -> bool:  # type: ignore[override]
        return bool(self._selected_strategy and self._has_credentials)

    @Property(str, notify=statusMessageIdChanged)
    def statusMessageId(self) -> str:  # type: ignore[override]
        return self._status_message_id

    @Property(str, notify=statusDetailsChanged)
    def statusDetails(self) -> str:  # type: ignore[override]
        return self._status_details

    @Property(str, notify=lastSavedExchangeChanged)
    def lastSavedExchange(self) -> str:  # type: ignore[override]
        return self._last_saved_exchange

    @Property("QStringList", notify=availableExchangesChanged)
    def availableExchanges(self) -> list[str]:  # type: ignore[override]
        return list(self._available_exchanges)

    # ------------------------------------------------------------------
    @Slot(result=bool)
    def refreshStrategies(self) -> bool:
        """Odświeża listę dostępnych strategii."""

        try:
            descriptors = tuple(self._strategy_loader())
        except Exception as exc:  # pragma: no cover - awaryjne logowanie
            self._strategies = ()
            self._strategies_cache = []
            self._set_status("onboarding.strategy.error.load", str(exc))
            self._emit_strategies_updated()
            return False

        self._strategies = descriptors
        self._strategies_cache = [self._descriptor_to_dict(item) for item in descriptors]
        self._emit_strategies_updated()
        if self._selected_strategy:
            self._sync_selection(self._selected_strategy)
        return True

    @Slot(str, result=bool)
    def selectStrategy(self, name: str) -> bool:
        """Aktualizuje wybraną strategię w kreatorze."""

        normalized = str(name or "").strip()
        if not normalized:
            self._set_status("onboarding.strategy.error.selection", "")
            return False
        if not self._sync_selection(normalized):
            self._set_status("onboarding.strategy.error.unknown", normalized)
            return False
        descriptor = self._find_descriptor(normalized)
        title = descriptor.title if descriptor else normalized
        self._set_status("onboarding.strategy.selection.ok", title)
        return True

    @Slot(str, str, str, str, result=bool)
    def importApiKey(self, exchange: str, api_key: str, api_secret: str, passphrase: str = "") -> bool:
        """Zapisuje dane API w magazynie sekretów."""

        exchange_id = str(exchange or "").strip()
        if not exchange_id:
            self._set_status("onboarding.strategy.error.credentials", "Brak identyfikatora giełdy")
            return False
        credentials = ExchangeCredentials(
            exchange=exchange_id,
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=passphrase or None,
        )
        try:
            self._secret_store.save_exchange_credentials(credentials)
        except SecretStoreError as exc:
            self._set_status("onboarding.strategy.error.secretStore", str(exc))
            return False
        self._has_credentials = True
        self._last_saved_exchange = credentials.normalized_exchange()
        self.lastSavedExchangeChanged.emit()
        self._set_status("onboarding.strategy.credentials.saved", self._last_saved_exchange)
        self._update_configuration_state()
        return True

    # ------------------------------------------------------------------
    def _descriptor_to_dict(self, descriptor: StrategyDescriptor) -> dict:
        payload = asdict(descriptor)
        payload.setdefault("title", descriptor.title)
        payload.setdefault("licenseTier", descriptor.license_tier)
        payload.setdefault("riskClasses", list(descriptor.risk_classes))
        payload.setdefault("requiredData", list(descriptor.required_data))
        payload.setdefault("tags", list(descriptor.tags))
        payload.setdefault("metadata", dict(descriptor.metadata))
        return payload

    def _emit_strategies_updated(self) -> None:
        self.strategiesChanged.emit()
        self._update_configuration_state()

    def _find_descriptor(self, name: str) -> StrategyDescriptor | None:
        normalized = name.strip().lower()
        for descriptor in self._strategies:
            if descriptor.name.lower() == normalized or descriptor.engine.lower() == normalized:
                return descriptor
        return None

    def _sync_selection(self, name: str) -> bool:
        descriptor = self._find_descriptor(name)
        if descriptor is None:
            return False
        if descriptor.name != self._selected_strategy or descriptor.title != self._selected_strategy_title:
            self._selected_strategy = descriptor.name
            self._selected_strategy_title = descriptor.title
            self.selectedStrategyChanged.emit()
        self._update_configuration_state()
        return True

    def _update_configuration_state(self) -> None:
        ready = self.configurationReady
        if ready != self._cached_ready_state:
            self._cached_ready_state = ready
            self.configurationReadyChanged.emit()

    def _set_status(self, message_id: str, details: str) -> None:
        if message_id != self._status_message_id:
            self._status_message_id = message_id
            self.statusMessageIdChanged.emit()
        if details != self._status_details:
            self._status_details = details
            self.statusDetailsChanged.emit()


__all__ = ["OnboardingService"]
