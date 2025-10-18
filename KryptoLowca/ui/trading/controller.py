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
        try:
            self.ensure_exchange()
            self.state.running = True
            self.state.status.set("Sesja handlowa uruchomiona")
            logger.info("Trading session started")
        except Exception as exc:  # pragma: no cover - reakcja UI
            logger.exception("Nie udało się uruchomić sesji")
            self.state.status.set("Błąd przy uruchamianiu sesji")
            messagebox.showerror("Trading GUI", str(exc))

    # ------------------------------------------------------------------
    def stop(self) -> None:
        if not self.state.running:
            return
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


__all__ = ["TradingSessionController"]
