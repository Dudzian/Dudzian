# services/execution_service.py
# -*- coding: utf-8 -*-
"""Centralna warstwa egzekucji zleceń.

Moduł kapsułkuje współpracę z :class:`ExchangeManager`, dzięki czemu GUI,
AutoTrader oraz inne serwisy nie muszą powielać logiki wyceny, kwantyzacji
oraz aktualizacji stanu pozycji i sald. W Fazie 0 skupiamy się na rynku
spot/paper — futures zostaną rozszerzone w kolejnych iteracjach.

Najważniejsze zalety użycia serwisu:

* Spójne wyliczanie ceny egzekucji (VWAP + fallback w bps),
* Jedno miejsce odpowiedzialne za kwantyzację ilości i obsługę wyjątków,
* Proste pobieranie aktualnych sald i listy pozycji do odświeżania GUI,
* Gotowy punkt startowy do dalszej rozbudowy (np. trailing, OCO, SL/TP).

Serwis nie przechowuje stanu — operuje na bieżąco na fasadzie
``ExchangeManager``. Dzięki temu zachowujemy kompatybilność z trybem paper
oraz z przyszłymi backendami CCXT.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, List

from managers.exchange_manager import ExchangeManager
from managers.exchange_core import Mode, OrderDTO, PositionDTO

log = logging.getLogger(__name__)


class ExecutionService:
    """Wspólne API do przygotowania i egzekucji zleceń."""

    def __init__(self, exchange_manager: ExchangeManager) -> None:
        self._exchange_manager = exchange_manager

    # ------------------------------------------------------------------
    # Quote & sizing helpers
    # ------------------------------------------------------------------
    def quote_market(
        self,
        symbol: str,
        side: str,
        *,
        amount: Optional[float] = None,
        fallback_bps: float = 5.0,
        limit: int = 50,
    ) -> Tuple[Optional[float], float]:
        """Zwraca parę ``(cena, slip_bps)`` wykorzystując order book lub ticker."""

        side_norm = "buy" if str(side).lower() == "buy" else "sell"
        return self._exchange_manager.simulate_vwap_price(
            symbol,
            side_norm,
            amount=amount,
            fallback_bps=fallback_bps,
            limit=limit,
        )

    def calculate_quantity(self, symbol: str, notional: float, price: float) -> float:
        """Kwantyzuje ilość na podstawie wartości nominalnej i ceny."""

        if price <= 0:
            return 0.0
        raw_qty = float(notional) / float(price)
        return self._exchange_manager.quantize_amount(symbol, raw_qty)

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------
    def execute_market(
        self,
        symbol: str,
        side: str,
        quantity: float,
        *,
        client_order_id: Optional[str] = None,
    ) -> OrderDTO:
        """Składa zlecenie MARKET w aktualnie ustawionym trybie giełdy."""

        side_upper = "BUY" if str(side).upper() == "BUY" else "SELL"
        log.debug(
            "Executing market order: symbol=%s side=%s qty=%s mode=%s",
            symbol,
            side_upper,
            quantity,
            getattr(self._exchange_manager, "mode", Mode.PAPER),
        )
        return self._exchange_manager.create_order(
            symbol,
            side_upper,
            "MARKET",
            float(quantity),
            None,
            client_order_id,
        )

    # ------------------------------------------------------------------
    # Portfolio state
    # ------------------------------------------------------------------
    def fetch_balance(self) -> Dict[str, Any]:
        """Zwraca słownik z saldem z fasady giełdowej."""

        return self._exchange_manager.fetch_balance()

    def list_positions(self, symbol: Optional[str] = None) -> List[PositionDTO]:
        """Pobiera listę pozycji (paper/spot/futures)."""

        return self._exchange_manager.fetch_positions(symbol)

    def set_paper_balance(self, amount: float, asset: str = "USDT") -> None:
        """Aktualizuje balans papierowy poprzez fasadę."""

        self._exchange_manager.set_paper_balance(amount, asset=asset)

    def quantize_amount(self, symbol: str, amount: float) -> float:
        """Kwantyzuje ilość zgodnie z regułami rynku."""

        return self._exchange_manager.quantize_amount(symbol, amount)

    def quantize_price(self, symbol: str, price: float) -> float:
        """Kwantyzuje cenę zgodnie z regułami rynku."""

        return self._exchange_manager.quantize_price(symbol, price)

    # ------------------------------------------------------------------
    # Dostęp pomocniczy
    # ------------------------------------------------------------------
    @property
    def mode(self) -> Mode:
        """Zwraca aktualny tryb działania giełdy."""

        mode = getattr(self._exchange_manager, "mode", Mode.PAPER)
        if isinstance(mode, Mode):
            return mode
        try:
            return Mode(str(mode))
        except Exception:
            return Mode.PAPER

    @property
    def exchange_manager(self) -> ExchangeManager:
        """Udostępnia wewnętrzny ``ExchangeManager`` (np. dla AutoTradera)."""

        return self._exchange_manager

