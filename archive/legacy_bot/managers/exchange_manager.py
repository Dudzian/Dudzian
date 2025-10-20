# managers/exchange_manager.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field

from .exchange_core import (
    Mode,
    OrderSide,
    OrderType,
    OrderStatus,
    MarketRules,
    OrderDTO,
    TradeDTO,
    PositionDTO,
    Event,
    EventBus,
    BaseBackend,
    PaperBackend,
)

# CCXT jest używane jako publiczny feed danych i backend live/testnet
try:
    import ccxt  # type: ignore
except Exception:  # pragma: no cover
    ccxt = None  # GUI poradzi sobie – paper będzie działał, ale bez live

log = logging.getLogger(__name__)


class _CCXTPublicFeed(BaseBackend):
    """
    Prosty backend do pobierania rynków/tickerów/OHLCV z CCXT (public).
    Używany przez:
    - PaperBackend jako źródło cen
    - ExchangeManager w trybie SPOT/FUTURES (public/podgląd)
    """

    def __init__(self, exchange_id: str = "binance", testnet: bool = False) -> None:
        super().__init__(event_bus=EventBus())
        if ccxt is None:
            raise RuntimeError("CCXT nie jest zainstalowane.")
        self.exchange_id = exchange_id
        self.testnet = bool(testnet)
        self.client = getattr(ccxt, exchange_id)({
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot",  # dla public feed bez znaczenia
            }
        })
        if exchange_id == "binance" and self.testnet:
            # public markets też dostępne z produkcji; testnet endpoint głównie do private
            # zostawiamy standard, bo OHLCV/ticker są z rynku głównego
            pass
        self._markets: Dict[str, Any] = {}
        self._rules: Dict[str, MarketRules] = {}

    def load_markets(self) -> Dict[str, MarketRules]:
        self._markets = self.client.load_markets()
        rules: Dict[str, MarketRules] = {}
        for s, m in self._markets.items():
            lim = (m.get("limits") or {})
            a = lim.get("amount") or {}
            p = lim.get("price") or {}
            c = m.get("precision") or {}
            # wykrywanie stepów (amount/price)
            amt_step = a.get("step", 0.0) or (10 ** -float(c.get("amount", 8))) if c.get("amount") is not None else 0.0
            price_step = p.get("step", 0.0) or (10 ** -float(c.get("price", 8))) if c.get("price") is not None else 0.0
            mn = (lim.get("cost") or {}).get("min", 0.0) or 0.0
            rules[s] = MarketRules(
                symbol=s,
                price_step=float(price_step or 0.0),
                amount_step=float(amt_step or 0.0),
                min_notional=float(mn or 0.0),
                min_amount=float(a.get("min") or 0.0),
                max_amount=float(a.get("max") or 0.0) if a.get("max") is not None else None,
                min_price=float(p.get("min") or 0.0) if p.get("min") is not None else None,
                max_price=float(p.get("max") or 0.0) if p.get("max") is not None else None,
            )
        self._rules = rules
        return rules

    def get_market_rules(self, symbol: str) -> Optional[MarketRules]:
        return self._rules.get(symbol)

    def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            return self.client.fetch_ticker(symbol)
        except Exception as e:
            log.warning("fetch_ticker failed: %s", e)
            return None

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> Optional[List[List[float]]]:
        try:
            return self.client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as e:
            log.warning("fetch_ohlcv failed: %s", e)
            return None

    # Backend publiczny nie tworzy zamówień/pozycji
    def create_order(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    def cancel_order(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    def fetch_open_orders(self, *a, **k) -> List[OrderDTO]:  # pragma: no cover
        return []

    def fetch_positions(self, *a, **k) -> List[PositionDTO]:  # pragma: no cover
        return []


class _CCXTPrivateBackend(_CCXTPublicFeed):
    """
    Prywatny backend (SPOT/FUTURES) do realnych/testnetowych zleceń przez CCXT.
    W Fazie 0 wspieramy MARKET i LIMIT (bez OCO/SL/TP – dojdą w Fazie 3).
    """

    def __init__(self, exchange_id: str = "binance", testnet: bool = False, futures: bool = False,
                 api_key: Optional[str] = None, secret: Optional[str] = None) -> None:
        super().__init__(exchange_id=exchange_id, testnet=testnet)
        if ccxt is None:
            raise RuntimeError("CCXT nie jest zainstalowane.")
        opts: Dict[str, Any] = {
            "enableRateLimit": True,
            "apiKey": api_key or "",
            "secret": secret or "",
            "options": {"defaultType": "future" if futures else "spot"}
        }
        self.client = getattr(ccxt, exchange_id)(opts)
        if exchange_id == "binance" and testnet:
            if futures:
                self.client.set_sandbox_mode(True)  # futures testnet
            else:
                # spot testnet używa endpointu testnet.binance.vision
                self.client.urls["api"] = self.client.urls["test"]
        self.futures = futures
        self.mode = Mode.FUTURES if futures else Mode.SPOT

    def create_order(self, symbol: str, side: OrderSide, type: OrderType,
                     quantity: float, price: Optional[float] = None,
                     client_order_id: Optional[str] = None) -> OrderDTO:
        self.load_markets() if not self._rules else None  # upewnij się, że są reguły
        mr = self.get_market_rules(symbol)
        if not mr:
            raise RuntimeError(f"Brak reguł rynku dla {symbol}. Najpierw 'Load Markets'.")

        qty = mr.quantize_amount(float(quantity))
        if qty <= 0:
            raise ValueError("Ilość po kwantyzacji = 0.")
        px = None
        params: Dict[str, Any] = {}
        if type == OrderType.LIMIT:
            if price is None:
                raise ValueError("Cena wymagana dla LIMIT.")
            px = mr.quantize_price(float(price))
        if client_order_id:
            # Binance: 'newClientOrderId' dla spot/futures
            key = "newClientOrderId"
            params[key] = client_order_id

        # kontrola minNotional (dla LIMIT użyjemy ceny limit, dla MARKET użyjemy last)
        if type == OrderType.MARKET:
            t = self.fetch_ticker(symbol) or {}
            last = t.get("last") or t.get("close") or t.get("bid") or t.get("ask")
            if not last:
                raise RuntimeError(f"Brak ceny MARKET dla {symbol}.")
            notional = qty * float(last)
        else:
            notional = qty * float(px)
        mn = mr.min_notional or 0.0
        if mn and notional < mn:
            raise ValueError(f"Notional {notional:.8f} < minNotional {mn:.8f} dla {symbol}")

        ccxt_type = "market" if type == OrderType.MARKET else "limit"
        ccxt_side = side.value.lower()
        resp = self.client.create_order(symbol, ccxt_type, ccxt_side, qty, px, params)
        oid = resp.get("id") or resp.get("orderId")
        status = resp.get("status", "open").upper()
        if status == "CLOSED":
            status = "FILLED"
        odto = OrderDTO(
            id=int(oid) if isinstance(oid, str) and oid.isdigit() else oid,
            client_order_id=client_order_id,
            symbol=symbol, side=side, type=type, quantity=qty,
            price=px, status=OrderStatus(status),
            mode=self.mode
        )
        return odto

    def cancel_order(self, order_id: Any, symbol: str) -> bool:
        try:
            self.client.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            log.error("cancel_order failed: %s", e)
            return False

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[OrderDTO]:
        try:
            orders = self.client.fetch_open_orders(symbol) if symbol else self.client.fetch_open_orders()
        except Exception as e:
            log.error("fetch_open_orders failed: %s", e)
            return []
        out: List[OrderDTO] = []
        for o in orders:
            st = (o.get("status") or "open").upper()
            if st == "CLOSED":
                st = "FILLED"
            out.append(OrderDTO(
                id=o.get("id") or o.get("orderId"),
                client_order_id=o.get("clientOrderId") or o.get("info", {}).get("clientOrderId"),
                symbol=o.get("symbol"),
                side=OrderSide.BUY if (o.get("side","").lower()=="buy") else OrderSide.SELL,
                type=OrderType.MARKET if (o.get("type","").lower()=="market") else OrderType.LIMIT,
                quantity=float(o.get("amount") or o.get("filled") or 0.0),
                price=float(o.get("price") or 0.0) if o.get("price") else None,
                status=OrderStatus(st),
                mode=self.mode
            ))
        return out

    # Pozycje – dostępne głównie na futures; w Fazie 0 zwracamy pustą listę dla spota
    def fetch_positions(self, symbol: Optional[str] = None) -> List[PositionDTO]:
        if not self.futures:
            return []
        try:
            positions = self.client.fetch_positions([symbol] if symbol else None)
        except Exception as e:
            log.error("fetch_positions failed: %s", e)
            return []
        out: List[PositionDTO] = []
        for p in positions or []:
            amt = float(p.get("contracts") or p.get("amount") or 0.0)
            if abs(amt) < 1e-12:
                continue
            side = "LONG" if amt > 0 else "SHORT"
            out.append(PositionDTO(
                symbol=p.get("symbol"), side=side,
                quantity=abs(amt),
                avg_price=float(p.get("entryPrice") or 0.0),
                unrealized_pnl=float(p.get("unrealizedPnl") or 0.0),
                mode=Mode.FUTURES
            ))
        return out


# =========================================
#          Exchange Manager (Facade)
# =========================================

class ExchangeManager:
    """
    Jedna fasada dla GUI/strategii:
      - set_mode(Mode.PAPER/SPOT/FUTURES)
      - set_credentials()
      - load_markets()
      - fetch_ticker(), fetch_ohlcv()
      - quantize_amount/price(), min_notional()
      - create_order()/cancel_order()/fetch_open_orders()
      - fetch_positions()

    W Fazie 0:
      - PAPER: MARKET z natychmiastowym fill’em do DB (limit/SL/TP – panel).
      - SPOT/FUTURES: MARKET/LIMIT przez CCXT (bez OCO/SL/TP).
    """

    def __init__(self, exchange_id: str = "binance") -> None:
        self.exchange_id = exchange_id
        self.mode: Mode = Mode.PAPER
        self._testnet: bool = False
        self._futures: bool = False
        self._api_key: Optional[str] = None
        self._secret: Optional[str] = None

        self._event_bus = EventBus()
        # public feed CCXT (do cen/rynku); inicjalizowany przy load_markets()
        self._public: Optional[_CCXTPublicFeed] = None
        # prywatny backend dla SPOT/FUTURES
        self._private: Optional[_CCXTPrivateBackend] = None
        # paper backend korzysta z _public dla cen/rynku
        self._paper: Optional[PaperBackend] = None

        log.info("ExchangeManager initialized (Core 2.0)")

    # --- tryb/prywatne dane ---

    def set_mode(self, *, paper: bool = False, spot: bool = False, futures: bool = False, testnet: bool = False) -> None:
        if paper:
            self.mode = Mode.PAPER
            self._futures = False
            self._testnet = False
        elif futures:
            self.mode = Mode.FUTURES
            self._futures = True
            self._testnet = bool(testnet)
        else:
            self.mode = Mode.SPOT
            self._futures = False
            self._testnet = bool(testnet)
        log.info("Mode set to %s (futures=%s, testnet=%s)", self.mode.value, self._futures, self._testnet)
        # reset backendów – wymaga ponownego load_markets()
        self._private = None
        self._paper = None
        # public zostawiamy – ale jeśli zmieniło się testnet, zrobimy reload przy load_markets()

    def set_credentials(self, api_key: Optional[str], secret: Optional[str]) -> None:
        self._api_key = (api_key or "").strip() or None
        self._secret = (secret or "").strip() or None
        log.info("Credentials set (lengths): api_key=%s, secret=%s",
                 len(self._api_key or 0), len(self._secret or 0))

    # --- inicjalizacja rynków ---

    def _ensure_public(self) -> _CCXTPublicFeed:
        if self._public is None:
            self._public = _CCXTPublicFeed(exchange_id=self.exchange_id, testnet=self._testnet)
        return self._public

    def _ensure_private(self) -> _CCXTPrivateBackend:
        if not self._api_key or not self._secret:
            raise RuntimeError("Brak API Key/Secret – ustaw w GUI i spróbuj ponownie.")
        if self._private is None:
            self._private = _CCXTPrivateBackend(
                exchange_id=self.exchange_id,
                testnet=self._testnet,
                futures=self._futures,
                api_key=self._api_key,
                secret=self._secret
            )
            self._private.load_markets()
        return self._private

    def _ensure_paper(self) -> PaperBackend:
        pub = self._ensure_public()
        if self._paper is None:
            self._paper = PaperBackend(price_feed_backend=pub, event_bus=self._event_bus)
            self._paper.load_markets()
        return self._paper

    def load_markets(self) -> Dict[str, MarketRules]:
        pub = self._ensure_public()
        rules = pub.load_markets()
        log.info("Loaded %s markets (public)", len(rules))
        # odśwież papier jeśli aktywny
        if self.mode == Mode.PAPER and self._paper:
            self._paper.load_markets()
        return rules

    # --- dane ---

    def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self._ensure_public().fetch_ticker(symbol)

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> Optional[List[List[float]]]:
        return self._ensure_public().fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    # --- reguły/kwantyzacja ---

    def get_market_rules(self, symbol: str) -> Optional[MarketRules]:
        pub = self._ensure_public()
        if not pub.get_market_rules(symbol):
            pub.load_markets()
        return pub.get_market_rules(symbol)

    def quantize_amount(self, symbol: str, amount: float) -> float:
        mr = self.get_market_rules(symbol)
        return mr.quantize_amount(amount) if mr else float(f"{amount:.8f}")

    def quantize_price(self, symbol: str, price: float) -> float:
        mr = self.get_market_rules(symbol)
        return mr.quantize_price(price) if mr else float(f"{price:.8f}")

    def min_notional(self, symbol: str) -> float:
        mr = self.get_market_rules(symbol)
        return float(mr.min_notional) if mr else 0.0

    # --- zlecenia ---

    def create_order(self, symbol: str, side: str, type: str,
                     quantity: float, price: Optional[float] = None,
                     client_order_id: Optional[str] = None) -> OrderDTO:
        side_e = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
        type_e = OrderType.MARKET if type.upper() == "MARKET" else OrderType.LIMIT

        if self.mode == Mode.PAPER:
            return self._ensure_paper().create_order(
                symbol, side_e, type_e, quantity, price, client_order_id
            )
        # SPOT/FUTURES (testnet/live)
        backend = self._ensure_private()
        return backend.create_order(symbol, side_e, type_e, quantity, price, client_order_id)

    def cancel_order(self, order_id: Any, symbol: str) -> bool:
        if self.mode == Mode.PAPER:
            # w Fazie 0 papier nie trzyma otwartych zleceń – panel obsługuje LIMIT
            return False
        return self._ensure_private().cancel_order(order_id, symbol)

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[OrderDTO]:
        if self.mode == Mode.PAPER:
            return []
        return self._ensure_private().fetch_open_orders(symbol)

    def fetch_positions(self, symbol: Optional[str] = None) -> List[PositionDTO]:
        if self.mode == Mode.PAPER:
            return self._ensure_paper().fetch_positions(symbol)
        return self._ensure_private().fetch_positions(symbol)

    # --- events ---

    def on(self, event_type: str, callback) -> None:
        self._event_bus.subscribe(event_type, callback)
