from __future__ import annotations

import logging
from typing import Any, Optional

from tkinter import messagebox

from bot_core.security.guards import (
    CapabilityGuard,
    LicenseCapabilityError,
    get_capability_guard,
)

from .state import AppState

__all__ = ["TradingSessionController"]

logger = logging.getLogger(__name__)


def _tk_value(candidate: Any, default: str = "") -> str:
    """Bezpiecznie odczytuje wartość z obiektu Tkintera lub zwraca domyślną."""

    try:
        if hasattr(candidate, "get"):
            return str(candidate.get())
    except Exception:  # pragma: no cover - defensywne logowanie
        logger.debug("Nie udało się odczytać wartości z kontrolki Tkinter", exc_info=True)
        return default
    return str(candidate or default)


class TradingSessionController:
    """Minimalna implementacja kontrolera GUI dla testów licencyjnych Stage6."""

    def __init__(
        self,
        state: AppState,
        db: Any,
        secret_storage: Any,
        config_service: Any,
        reporter: Any,
        risk: Any,
        ai_manager: Any,
        *,
        exchange_manager: Optional[Any] = None,
        market_intel: Optional[Any] = None,
    ) -> None:
        self.state = state
        self.db = db
        self.secret_storage = secret_storage
        self.config_service = config_service
        self.reporter = reporter
        self.risk = risk
        self.ai_manager = ai_manager
        self.exchange = exchange_manager
        self.market_intel = market_intel
        self._reserved_slot: str | None = None

    # ------------------------------------------------------------------
    def _resolve_guard(self) -> CapabilityGuard | None:
        guard = getattr(self.state, "capability_guard", None)
        if isinstance(guard, CapabilityGuard):
            return guard
        try:
            return get_capability_guard()
        except Exception:  # pragma: no cover - defensywne logowanie
            logger.debug("Brak globalnego strażnika licencyjnego", exc_info=True)
            return None

    # ------------------------------------------------------------------
    def _ensure_license_allows_start(self, guard: CapabilityGuard) -> str:
        network_text = _tk_value(getattr(self.state, "network", None), default="paper").strip().lower()
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

        mode_text = _tk_value(getattr(self.state, "mode", None), default="spot").strip().lower()
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
        except Exception as exc:  # pragma: no cover - reakcja UI
            logger.exception("Nie udało się uruchomić sesji")
            self.state.status.set("Błąd przy uruchamianiu sesji")
            if guard is not None and reserved and slot_kind:
                try:
                    guard.release_slot(slot_kind)
                except Exception:  # pragma: no cover - defensywne logowanie
                    logger.debug("Nie udało się zwolnić slotu licencyjnego", exc_info=True)
            self._reserved_slot = None
            messagebox.showerror("Trading GUI", str(exc))
            return

        self.state.running = True
        self.state.status.set("Sesja handlowa uruchomiona")
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

    # ------------------------------------------------------------------
    def ensure_exchange(self) -> Any:
        """Zwraca wstrzykniętego menedżera giełdy lub zgłasza błąd."""

        if self.exchange is None:
            raise RuntimeError("ExchangeManager nie jest dostępny w tej konfiguracji")
        return self.exchange

    # ------------------------------------------------------------------
    def get_exchange(self) -> Any | None:
        return self.exchange
