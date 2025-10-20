# managers/exchange_manager.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import ccxt
except Exception as e:
    raise SystemExit("Brak biblioteki 'ccxt'. Zainstaluj: python -m pip install ccxt") from e

__all__ = ["ExchangeManager", "ExchangeAdapter"]

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s'))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


@dataclass
class _MarketInfo:
    symbol: str
    amount_step: Optional[float] = None
    price_step: Optional[float] = None
    min_qty: Optional[float] = None
    min_notional: Optional[float] = None


class ExchangeManager:
    """
    Prosty, synchroniczny wrapper CCXT pod potrzeby trading_gui.py.
    Obsługuje: testnet/live, spot (futures pominięte na teraz), pobieranie rynków, OHLCV, ticker, orderbook,
    oraz pomocnicze funkcje do kwantyzacji ilości i symulacji VWAP.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        testnet: bool = False,
        enable_rate_limit: bool = True,
        default_type: str = "spot",
    ) -> None:
        self.exchange_id = exchange_id
        self.testnet = bool(testnet)
        self._client = self._build_ccxt(exchange_id, api_key, secret, self.testnet, enable_rate_limit, default_type)
        self._markets: Dict[str, _MarketInfo] = {}
        self._load_markets_safely()

    # ------------- CCXT init -------------
    @staticmethod
    def _build_ccxt(exchange_id: str, api_key: Optional[str], secret: Optional[str], testnet: bool,
                    enable_rate_limit: bool, default_type: str):
        ex_cls = getattr(ccxt, exchange_id)
        opts: Dict[str, Any] = {"enableRateLimit": enable_rate_limit, "options": {}}
        if api_key:
            opts["apiKey"] = api_key
        if secret:
            opts["secret"] = secret
        client = ex_cls(opts)
        # testnet jeśli dostępny
        if hasattr(client, "set_sandbox_mode"):
            try:
                client.set_sandbox_mode(testnet)
            except Exception:
                pass
        # wymuś SPOT na Binance (i ew. innych)
        try:
            client.options = client.options or {}
            client.options["defaultType"] = default_type
        except Exception:
            pass
        return client

    # ------------- Markets / symbol meta -------------
    def _load_markets_safely(self) -> None:
        try:
            markets = self._client.load_markets()
            self._markets.clear()
            for sym, m in markets.items():
                info = self._extract_market_info(sym, m)
                self._markets[sym] = info
            logger.info("Załadowano rynki: %d symboli", len(self._markets))
        except Exception as e:
            logger.warning("Nie udało się wczytać rynków: %s", e)

    @staticmethod
    def _to_float_safe(v) -> Optional[float]:
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    def _extract_market_info(self, symbol: str, market: Dict[str, Any]) -> _MarketInfo:
        prec = (market.get("precision") or {})
        limits = (market.get("limits") or {})
        amount_min = (limits.get("amount") or {}).get("min", None)
        cost_min = (limits.get("cost") or {}).get("min", None)

        info = market.get("info") or {}
        filters = info.get("filters") or []

        step_size = None
        tick_size = None
        min_notional_f = None

        for f in filters:
            t = f.get("filterType")
            if t == "LOT_SIZE":
                step_size = self._to_float_safe(f.get("stepSize"))
                if amount_min is None:
                    amount_min = self._to_float_safe(f.get("minQty"))
            elif t == "PRICE_FILTER":
                tick_size = self._to_float_safe(f.get("tickSize"))
            elif t in ("MIN_NOTIONAL", "NOTIONAL"):
                v = self._to_float_safe(f.get("minNotional") or f.get("notional"))
                if v is not None:
                    min_notional_f = v

        if cost_min is None and min_notional_f is not None:
            cost_min = min_notional_f

        # wyznacz krok z precyzji jeśli brak
        if step_size is None and prec.get("amount") is not None:
            step_size = 10 ** (-int(prec["amount"]))
        if tick_size is None and prec.get("price") is not None:
            tick_size = 10 ** (-int(prec["price"]))

        # fallback dla Binance testnet: minNotional bywa puste → 10 USDT
        if self._client.id == "binance" and (cost_min is None or cost_min <= 0):
            cost_min = 10.0

        return _MarketInfo(
            symbol=symbol,
            amount_step=step_size,
            price_step=tick_size,
            min_qty=self._to_float_safe(amount_min),
            min_notional=self._to_float_safe(cost_min),
        )

    # publiczne API dla GUI
    def fetch_markets(self) -> List[str]:
        """
        Zwraca listę symboli, np. ['BTC/USDT', 'ETH/USDT', ...]
        """
        if not self._markets:
            self._load_markets_safely()
        return list(self._markets.keys())

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 500) -> Optional[List[List[float]]]:
        try:
            return self._client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as e:
            logger.warning("fetch_ohlcv(%s) error: %s", symbol, e)
            return None

    def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            return self._client.fetch_ticker(symbol)
        except Exception as e:
            logger.warning("fetch_ticker(%s) error: %s", symbol, e)
            return None

    def fetch_order_book(self, symbol: str, limit: int = 50) -> Optional[Dict[str, Any]]:
        try:
            return self._client.fetch_order_book(symbol, limit=limit)
        except Exception as e:
            logger.warning("fetch_order_book(%s) error: %s", symbol, e)
            return None

    # batch pobrań do GUI (tkinter preferuje sync)
    def fetch_batch(
        self,
        symbols: Iterable[str],
        timeframe: str = "1m",
        use_orderbook: bool = False,
        limit_ohlcv: int = 500,
    ) -> List[Tuple[str, Optional[List[List[float]]], Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]]:
        out: List[Tuple[str, Optional[List[List[float]]], Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]] = []
        for sym in symbols:
            ohlcv = None
            ticker = None
            orderbook = None
            err: Optional[str] = None
            try:
                ohlcv = self.fetch_ohlcv(sym, timeframe=timeframe, limit=limit_ohlcv)
            except Exception as e:
                err = f"ohlcv: {e}"
            try:
                ticker = self.fetch_ticker(sym)
            except Exception as e:
                err = (err + f"; ticker: {e}") if err else f"ticker: {e}"
            if use_orderbook:
                try:
                    orderbook = self.fetch_order_book(sym, limit=50)
                except Exception as e:
                    err = (err + f"; orderbook: {e}") if err else f"orderbook: {e}"
            out.append((sym, ohlcv, ticker, orderbook, err))
        return out

    # --------- Kwantyzacja / kroki ---------
    def _get_market_info(self, symbol: str) -> _MarketInfo:
        if symbol not in self._markets:
            self._load_markets_safely()
        return self._markets.get(symbol, _MarketInfo(symbol))

    def quantize_amount(self, symbol: str, amount: float) -> float:
        """
        Dopasowuje ilość do kroku (stepSize) i minQty. Używa floor do kroku.
        """
        if amount is None:
            return 0.0
        info = self._get_market_info(symbol)
        step = info.amount_step or 0.0
        if step and step > 0:
            amount = math.floor(amount / step + 1e-12) * step
        if info.min_qty is not None and amount < info.min_qty:
            # zbyt mało – zwróć 0 żeby GUI nie próbowało składać
            return 0.0
        return float(amount)

    # --------- VWAP (symulacja z orderbooka) ---------
    def simulate_vwap_price(self, symbol: str, side: str, amount: Optional[float], fallback_bps: float = 5.0
                            ) -> Tuple[Optional[float], float]:
        """
        Liczy VWAP po dobieraniu zleceń z książki (bids/asks). Jeśli brak orderbooka albo amount None,
        zwraca cenę z tickera i slippage w bps = fallback_bps.
        """
        try:
            ob = self.fetch_order_book(symbol, limit=50)
            tk = self.fetch_ticker(symbol) or {}
            mid = None
            if tk:
                last = tk.get("last") or tk.get("close") or tk.get("bid") or tk.get("ask")
                mid = float(last) if last else None
            if not ob or not amount or amount <= 0:
                return (mid, float(fallback_bps))
            side = side.lower().strip()
            levels = ob["asks"] if side == "buy" else ob["bids"]
            need = float(amount)
            got = 0.0
            cost = 0.0
            for price, qty in levels:
                take = min(need - got, float(qty))
                if take <= 0:
                    break
                cost += take * float(price)
                got += take
            if got <= 0:
                return (mid, float(fallback_bps))
            vwap = cost / got
            if mid:
                bps = (abs(vwap - mid) / mid) * 10000.0
            else:
                bps = fallback_bps
            return (float(vwap), float(bps))
        except Exception as e:
            logger.warning("simulate_vwap_price(%s) failed: %s", symbol, e)
            tk = self.fetch_ticker(symbol) or {}
            last = tk.get("last") or tk.get("close") or tk.get("bid") or tk.get("ask")
            mid = float(last) if last else None
            return (mid, float(fallback_bps))

    # --------- PROSTE ZLECENIA (opcjonalnie wykorzystywane przez GUI w przyszłości) ---------
    def create_order(self, symbol: str, side: str, type: str, amount: float, price: Optional[float] = None,
                     client_order_id: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if self._client.id == "binance" and client_order_id:
            params["newClientOrderId"] = client_order_id
        if type.upper() == "MARKET":
            return self._client.create_order(symbol, "market", side.lower(), amount, None, params)
        elif type.upper() == "LIMIT":
            if price is None:
                raise ValueError("price wymagane dla LIMIT")
            return self._client.create_order(symbol, "limit", side.lower(), amount, float(price), params)
        else:
            raise NotImplementedError("Obsługujemy MARKET i LIMIT. STOP/TP dodamy później.")

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        try:
            self._client.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.warning("cancel_order(%s) error: %s", order_id, e)
            return False

    def fetch_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        try:
            return self._client.fetch_open_orders(symbol)
        except Exception as e:
            logger.warning("fetch_open_orders(%s) error: %s", symbol, e)
            return []

    # --------- Zamknięcie ---------
    def close(self) -> None:
        try:
            if hasattr(self._client, "close"):
                self._client.close()
        except Exception:
            pass


ExchangeAdapter = ExchangeManager
