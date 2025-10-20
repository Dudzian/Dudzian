"""Logika kontrolera sesji tradingowej."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from tkinter import messagebox

from KryptoLowca.database_manager import DatabaseManager
from KryptoLowca.managers.security_manager import SecurityManager
from KryptoLowca.managers.config_manager import ConfigManager
from KryptoLowca.managers.exchange_manager import ExchangeManager
from KryptoLowca.managers.report_manager import ReportManager
from KryptoLowca.managers.risk_manager_adapter import RiskManager
from bot_core.runtime.metadata import RiskManagerSettings
from KryptoLowca.managers.ai_manager import AIManager
from KryptoLowca.core.trading_engine import TradingEngine
from bot_core.security.guards import (
    CapabilityGuard,
    LicenseCapabilityError,
    get_capability_guard,
)

from .state import AppState


logger = logging.getLogger(__name__)


class TradingSessionController:
    """Skapsułkowana logika zarządzania sesją tradingową."""

    def __init__(
        self,
        state: AppState,
        db: DatabaseManager,
        security: SecurityManager,
        config: ConfigManager,
        reporter: ReportManager,
        risk: RiskManager,
        ai_manager: AIManager,
    ) -> None:
        self.state = state
        self.db = db
        self.security = security
        self.config = config
        self.reporter = reporter
        self.risk = risk
        self.ai_manager = ai_manager
        self.exchange: Optional[ExchangeManager] = None
        self.engine = TradingEngine()
        self._reserved_slot: str | None = None
        self._attach_engine_callbacks()

    # ------------------------------------------------------------------
    def _attach_engine_callbacks(self) -> None:
        if hasattr(self.engine, "set_report_manager"):
            try:
                self.engine.set_report_manager(self.reporter)
            except Exception:  # pragma: no cover - defensywne
                logger.exception("Nie udało się przypisać reportera do silnika")
        if hasattr(self.engine, "on_event"):
            try:
                self.engine.on_event(self._handle_engine_event)
            except Exception:  # pragma: no cover - defensywne
                logger.exception("Nie udało się podpiąć handlera zdarzeń")

    # ------------------------------------------------------------------
    def _handle_engine_event(self, event: Dict[str, Any]) -> None:
        """Reaguj na zdarzenia TradingEngine."""

        if not event:
            return
        event_type = event.get("type", "unknown")
        logger.info("Zdarzenie TradingEngine: %s", event)
        if event_type == "position_opened":
            symbol = event.get("symbol", "?")
            self.state.open_positions[symbol] = event
        elif event_type == "position_closed":
            symbol = event.get("symbol")
            if symbol and symbol in self.state.open_positions:
                self.state.open_positions.pop(symbol, None)
        self.state.status.set(f"Odebrano zdarzenie: {event_type}")

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.state.running:
            return
        guard = self._resolve_guard()
        slot_kind: str | None = None
        reserved = False
        if guard is not None:
            try:
                slot_kind = self._ensure_license_allows_start(guard)
                guard.reserve_slot(slot_kind)
                reserved = True
            except LicenseCapabilityError as exc:
                logger.warning("Blokada licencyjna przy starcie sesji: %s", exc)
                self.state.status.set(str(exc))
                notice_var = getattr(self.state, "license_notice", None)
                if hasattr(notice_var, "set"):
                    try:
                        notice_var.set(str(exc))
                    except Exception:  # pragma: no cover - defensywne logowanie
                        logger.debug("Nie udało się zaktualizować komunikatu licencji", exc_info=True)
                messagebox.showerror("Licencja", str(exc))
                return
        try:
            self.ensure_exchange()
            self.state.running = True
            self.state.status.set("Sesja handlowa uruchomiona")
            logger.info("Trading session started")
        except Exception as exc:  # pragma: no cover - reakcja UI
            logger.exception("Nie udało się uruchomić sesji")
            self.state.status.set("Błąd przy uruchamianiu sesji")
            if guard is not None and reserved and slot_kind:
                try:
                    guard.release_slot(slot_kind)
                except Exception:  # pragma: no cover - defensywne logowanie
                    logger.debug("Nie udało się zwolnić slotu licencyjnego", exc_info=True)
                finally:
                    reserved = False
            self._reserved_slot = None
            messagebox.showerror("Trading GUI", str(exc))
            return
        if guard is not None and reserved and slot_kind:
            self._reserved_slot = slot_kind
        else:
            self._reserved_slot = None

    # ------------------------------------------------------------------
    def stop(self) -> None:
        if not self.state.running:
            return
        guard = self._resolve_guard()
        if guard is not None and self._reserved_slot:
            try:
                guard.release_slot(self._reserved_slot)
            except Exception:  # pragma: no cover - defensywne logowanie
                logger.debug("Nie udało się zwolnić slotu przy zatrzymywaniu", exc_info=True)
            finally:
                self._reserved_slot = None
        self.state.running = False
        self.state.status.set("Sesja zatrzymana")
        logger.info("Trading session stopped")

    # ------------------------------------------------------------------
    def ensure_exchange(self) -> ExchangeManager:
        """Zapewnia instancję ``ExchangeManager`` zgodną z aktualnym stanem GUI."""

        if self.exchange is not None:
            return self.exchange
        network = self.state.network.get().lower()
        use_testnet = network != "live"
        logger.info("Inicjalizuję ExchangeManager dla sieci: %s", network)
        self.exchange = ExchangeManager(testnet=use_testnet)
        return self.exchange

    # ------------------------------------------------------------------
    def get_exchange(self) -> Optional[ExchangeManager]:
        """Zwraca bieżący ``ExchangeManager`` jeśli został już utworzony."""

        return self.exchange

    # ------------------------------------------------------------------
    def update_risk_settings(self, settings: RiskManagerSettings) -> None:
        """Aktualizuje konfigurację adaptera ryzyka dla bieżącej sesji."""

        if not isinstance(settings, RiskManagerSettings):
            raise TypeError("Oczekiwano instancji RiskManagerSettings")

        config = settings.to_dict()
        self.state.risk_manager_settings = settings
        self.state.risk_manager_config = config
        mode = self.state.mode.get() if hasattr(self.state.mode, "get") else "paper"
        self.risk = RiskManager(config=config, db_manager=self.db, mode=str(mode or "paper"))

    # ------------------------------------------------------------------
    def _resolve_guard(self) -> CapabilityGuard | None:
        guard = getattr(self.state, "capability_guard", None)
        if isinstance(guard, CapabilityGuard):
            return guard
        return get_capability_guard()

    # ------------------------------------------------------------------
    def _ensure_license_allows_start(self, guard: CapabilityGuard) -> str:
        network_var = getattr(self.state, "network", None)
        try:
            network_value = network_var.get() if hasattr(network_var, "get") else network_var
        except Exception:
            network_value = None
        network_text = str(network_value or "").strip().lower()

        if network_text == "live":
            message = "Tryb live wymaga edycji Pro. Skontaktuj się z opiekunem licencji."
            if not guard.capabilities.is_environment_allowed("live"):
                raise LicenseCapabilityError(message, capability="environment")
            guard.require_edition("pro", message=message)
            slot_kind = "live_controller"
        else:
            aliases = ("demo", "paper", "testnet")
            if not any(guard.capabilities.is_environment_allowed(alias) for alias in aliases):
                raise LicenseCapabilityError(
                    "Licencja nie obejmuje środowiska demo/testnet. Skontaktuj się z opiekunem licencji.",
                    capability="environment",
                )
            slot_kind = "paper_controller"

        mode_var = getattr(self.state, "mode", None)
        try:
            mode_value = mode_var.get() if hasattr(mode_var, "get") else mode_var
        except Exception:
            mode_value = None
        mode_text = str(mode_value or "").strip().lower()
        if mode_text == "futures":
            guard.require_module(
                "futures",
                message="Dodaj moduł Futures, aby aktywować handel kontraktami.",
            )

        guard.require_runtime(
            "auto_trader",
            message="Licencja nie obejmuje modułu AutoTrader. Skontaktuj się z opiekunem licencji.",
        )

        return slot_kind


__all__ = ["TradingSessionController"]
