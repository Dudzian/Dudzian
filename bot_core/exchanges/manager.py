"""Natywna implementacja fasady ExchangeManager."""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, Field

from bot_core.database.manager import DatabaseManager
from bot_core.exchanges.core import (
    BaseBackend,
    EventBus,
    MarketRules,
    Mode,
    OrderDTO,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionDTO,
    PaperBackend,
)

try:  # pragma: no cover
    import ccxt  # type: ignore
except Exception:  # pragma: no cover
    ccxt = None


log = logging.getLogger(__name__)


class _CCXTPublicFeed(BaseBackend):
    """Backend publiczny CCXT używany do cen oraz reguł rynku."""

    def __init__(self, exchange_id: str = "binance", testnet: bool = False) -> None:
        super().__init__(event_bus=EventBus())
        if ccxt is None:
            raise RuntimeError("CCXT nie jest zainstalowane.")
        self.exchange_id = exchange_id
        self.testnet = bool(testnet)
        self.client = getattr(ccxt, exchange_id)(
            {
                "enableRateLimit": True,
                "options": {
                    "defaultType": "spot",
                },
            }
        )
        if exchange_id == "binance" and self.testnet:
            pass
        self._markets: Dict[str, Any] = {}
        self._rules: Dict[str, MarketRules] = {}

    def load_markets(self) -> Dict[str, MarketRules]:
        self._markets = self.client.load_markets()
        rules: Dict[str, MarketRules] = {}
        for symbol, meta in self._markets.items():
            limits = (meta.get("limits") or {})
            amount_limits = limits.get("amount") or {}
            price_limits = limits.get("price") or {}
            precision = meta.get("precision") or {}
            amount_step = amount_limits.get("step", 0.0) or (
                (10 ** -float(precision.get("amount", 8))) if precision.get("amount") is not None else 0.0
            )
            price_step = price_limits.get("step", 0.0) or (
                (10 ** -float(precision.get("price", 8))) if precision.get("price") is not None else 0.0
            )
            min_notional = (limits.get("cost") or {}).get("min", 0.0) or 0.0
            rules[symbol] = MarketRules(
                symbol=symbol,
                price_step=float(price_step or 0.0),
                amount_step=float(amount_step or 0.0),
                min_notional=float(min_notional or 0.0),
                min_amount=float(amount_limits.get("min") or 0.0),
                max_amount=float(amount_limits.get("max") or 0.0)
                if amount_limits.get("max") is not None
                else None,
                min_price=float(price_limits.get("min") or 0.0)
                if price_limits.get("min") is not None
                else None,
                max_price=float(price_limits.get("max") or 0.0)
                if price_limits.get("max") is not None
                else None,
            )
        self._rules = rules
        return rules

    def get_market_rules(self, symbol: str) -> Optional[MarketRules]:
        return self._rules.get(symbol)

    def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            return self.client.fetch_ticker(symbol)
        except Exception as exc:
            log.warning("fetch_ticker failed: %s", exc)
            return None

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 500
    ) -> Optional[List[List[float]]]:
        try:
            return self.client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as exc:
            log.warning("fetch_ohlcv failed: %s", exc)
            return None

    def fetch_order_book(self, symbol: str, limit: int = 50) -> Optional[Dict[str, Any]]:
        try:
            return self.client.fetch_order_book(symbol, limit=limit)
        except Exception as exc:
            log.warning("fetch_order_book failed: %s", exc)
            return None

    def create_order(self, *args, **kwargs):  # pragma: no cover - interfejs
        raise NotImplementedError

    def cancel_order(self, *args, **kwargs):  # pragma: no cover - interfejs
        raise NotImplementedError

    def fetch_open_orders(self, *args, **kwargs) -> List[OrderDTO]:  # pragma: no cover
        return []

    def fetch_positions(self, *args, **kwargs) -> List[PositionDTO]:  # pragma: no cover
        return []


class _CCXTPrivateBackend(_CCXTPublicFeed):
    """Prywatny backend CCXT dla trybu SPOT/FUTURES."""

    def __init__(
        self,
        exchange_id: str = "binance",
        testnet: bool = False,
        futures: bool = False,
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
    ) -> None:
        super().__init__(exchange_id=exchange_id, testnet=testnet)
        if ccxt is None:
            raise RuntimeError("CCXT nie jest zainstalowane.")
        options: Dict[str, Any] = {
            "enableRateLimit": True,
            "apiKey": api_key or "",
            "secret": secret or "",
            "options": {"defaultType": "future" if futures else "spot"},
        }
        self.client = getattr(ccxt, exchange_id)(options)
        if exchange_id == "binance" and testnet:
            if futures:
                self.client.set_sandbox_mode(True)
            else:
                self.client.urls["api"] = self.client.urls["test"]
        self.futures = futures
        self.mode = Mode.FUTURES if futures else Mode.SPOT

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> OrderDTO:
        if not self._rules:
            self.load_markets()
        rules = self.get_market_rules(symbol)
        if not rules:
            raise RuntimeError(f"Brak reguł rynku dla {symbol}. Najpierw 'Load Markets'.")

        qty = rules.quantize_amount(float(quantity))
        if qty <= 0:
            raise ValueError("Ilość po kwantyzacji = 0.")

        px = None
        params: Dict[str, Any] = {}
        if type == OrderType.LIMIT:
            if price is None:
                raise ValueError("Cena wymagana dla LIMIT.")
            px = rules.quantize_price(float(price))

        if client_order_id:
            params["newClientOrderId"] = client_order_id

        if type == OrderType.MARKET:
            ticker = self.fetch_ticker(symbol) or {}
            last = ticker.get("last") or ticker.get("close") or ticker.get("bid") or ticker.get("ask")
            if not last:
                raise RuntimeError(f"Brak ceny MARKET dla {symbol}.")
            notional = qty * float(last)
        else:
            notional = qty * float(px)

        min_notional = rules.min_notional or 0.0
        if min_notional and notional < min_notional:
            raise ValueError(
                f"Notional {notional:.8f} < minNotional {min_notional:.8f} dla {symbol}"
            )

        ccxt_type = "market" if type == OrderType.MARKET else "limit"
        ccxt_side = side.value.lower()
        response = self.client.create_order(symbol, ccxt_type, ccxt_side, qty, px, params)
        order_id = response.get("id") or response.get("orderId")
        status = response.get("status", "open").upper()
        if status == "CLOSED":
            status = "FILLED"

        return OrderDTO(
            id=int(order_id) if isinstance(order_id, str) and order_id.isdigit() else order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            type=type,
            quantity=qty,
            price=px,
            status=OrderStatus(status),
            mode=self.mode,
        )

    def cancel_order(self, order_id: Any, symbol: str) -> bool:
        try:
            self.client.cancel_order(order_id, symbol)
            return True
        except Exception as exc:
            log.error("cancel_order failed: %s", exc)
            return False

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[OrderDTO]:
        try:
            orders = (
                self.client.fetch_open_orders(symbol)
                if symbol
                else self.client.fetch_open_orders()
            )
        except Exception as exc:
            log.error("fetch_open_orders failed: %s", exc)
            return []

        out: List[OrderDTO] = []
        for entry in orders:
            status = (entry.get("status") or "open").upper()
            if status == "CLOSED":
                status = "FILLED"
            out.append(
                OrderDTO(
                    id=entry.get("id") or entry.get("orderId"),
                    client_order_id=entry.get("clientOrderId")
                    or entry.get("info", {}).get("clientOrderId"),
                    symbol=entry.get("symbol"),
                    side=OrderSide.BUY
                    if (entry.get("side", "").lower() == "buy")
                    else OrderSide.SELL,
                    type=OrderType.MARKET
                    if (entry.get("type", "").lower() == "market")
                    else OrderType.LIMIT,
                    quantity=float(entry.get("amount") or entry.get("filled") or 0.0),
                    price=float(entry.get("price") or 0.0) if entry.get("price") else None,
                    status=OrderStatus(status),
                    mode=self.mode,
                )
            )
        return out

    def fetch_positions(self, symbol: Optional[str] = None) -> List[PositionDTO]:
        if not self.futures:
            return []
        try:
            positions = self.client.fetch_positions([symbol] if symbol else None)
        except Exception as exc:
            log.error("fetch_positions failed: %s", exc)
            return []

        out: List[PositionDTO] = []
        for position in positions or []:
            amount = float(position.get("contracts") or position.get("amount") or 0.0)
            if abs(amount) < 1e-12:
                continue
            side = "LONG" if amount > 0 else "SHORT"
            out.append(
                PositionDTO(
                    symbol=position.get("symbol"),
                    side=side,
                    quantity=abs(amount),
                    avg_price=float(position.get("entryPrice") or 0.0),
                    unrealized_pnl=float(position.get("unrealizedPnl") or 0.0),
                    mode=Mode.FUTURES,
                )
            )
        return out

    def fetch_balance(self) -> Dict[str, Any]:
        try:
            return self.client.fetch_balance()
        except Exception as exc:
            log.error("fetch_balance failed: %s", exc)
            return {}


class ExchangeManager:
    """Fasada do obsługi wymiany w trybach paper/spot/futures."""

    def __init__(
        self,
        exchange_id: str = "binance",
        *,
        paper_initial_cash: float = 10_000.0,
        paper_cash_asset: str = "USDT",
        db_url: Optional[str] = None,
    ) -> None:
        self.exchange_id = exchange_id
        self.mode: Mode = Mode.PAPER
        self._testnet: bool = False
        self._futures: bool = False
        self._api_key: Optional[str] = None
        self._secret: Optional[str] = None

        self._event_bus = EventBus()
        self._public: Optional[_CCXTPublicFeed] = None
        self._private: Optional[_CCXTPrivateBackend] = None
        self._paper: Optional[PaperBackend] = None
        self._paper_initial_cash = float(paper_initial_cash)
        self._paper_cash_asset = paper_cash_asset.upper()
        self._paper_fee_rate = getattr(PaperBackend, "FEE_RATE", 0.001)
        self._db_url = db_url or "sqlite+aiosqlite:///trading.db"
        self._db: Optional[DatabaseManager] = None
        self._db_failed: bool = False

        log.info("ExchangeManager initialized (bot_core)")

    def set_mode(
        self,
        *,
        paper: bool = False,
        spot: bool = False,
        futures: bool = False,
        testnet: bool = False,
    ) -> None:
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
        self._private = None
        self._paper = None

    def set_credentials(self, api_key: Optional[str], secret: Optional[str]) -> None:
        self._api_key = (api_key or "").strip() or None
        self._secret = (secret or "").strip() or None
        log.info(
            "Credentials set (lengths): api_key=%s, secret=%s",
            len(self._api_key or 0),
            len(self._secret or 0),
        )

    def _ensure_public(self) -> _CCXTPublicFeed:
        if self._public is None:
            self._public = _CCXTPublicFeed(exchange_id=self.exchange_id, testnet=self._testnet)
        return self._public

    def _ensure_private(self) -> _CCXTPrivateBackend:
        if not self._api_key or not self._secret:
            raise RuntimeError("Brak API Key/Secret – ustaw je przed użyciem trybu live/testnet.")
        if self._private is None:
            self._private = _CCXTPrivateBackend(
                exchange_id=self.exchange_id,
                testnet=self._testnet,
                futures=self._futures,
                api_key=self._api_key,
                secret=self._secret,
            )
            self._private.load_markets()
        return self._private

    def _ensure_paper(self) -> PaperBackend:
        public = self._ensure_public()
        if self._paper is None:
            self._paper = PaperBackend(
                price_feed_backend=public,
                event_bus=self._event_bus,
                initial_cash=self._paper_initial_cash,
                cash_asset=self._paper_cash_asset,
                fee_rate=self._paper_fee_rate,
                database=self._ensure_db(),
            )
            self._paper.load_markets()
        return self._paper

    def _ensure_db(self) -> Optional[DatabaseManager]:
        if self._db_failed:
            return None
        if self._db is None:
            try:
                self._db = DatabaseManager(self._db_url)
                self._db.sync.init_db()
            except Exception as exc:  # pragma: no cover
                log.warning("DatabaseManager init failed (%s): %s", self._db_url, exc)
                self._db = None
                self._db_failed = True
        return self._db

    def set_paper_balance(self, amount: float, asset: Optional[str] = None) -> None:
        self._paper_initial_cash = float(amount)
        if asset:
            self._paper_cash_asset = asset.upper()
        if self._paper is not None:
            self._paper._cash_balance = max(0.0, float(amount))  # type: ignore[attr-defined]
            if asset:
                self._paper._cash_asset = self._paper_cash_asset  # type: ignore[attr-defined]

    def set_paper_fee_rate(self, fee_rate: float) -> None:
        self._paper_fee_rate = max(0.0, float(fee_rate))
        if self._paper is not None:
            self._paper.set_fee_rate(self._paper_fee_rate)

    def get_paper_fee_rate(self) -> float:
        if self._paper is not None:
            return self._paper.get_fee_rate()
        return self._paper_fee_rate

    def load_markets(self) -> Dict[str, MarketRules]:
        public = self._ensure_public()
        rules = public.load_markets()
        log.info("Loaded %s markets (public)", len(rules))
        if self.mode == Mode.PAPER and self._paper:
            self._paper.load_markets()
        return rules

    def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self._ensure_public().fetch_ticker(symbol)

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> Optional[List[List[float]]]:
        return self._ensure_public().fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def fetch_order_book(self, symbol: str, limit: int = 50) -> Optional[Dict[str, Any]]:
        return self._ensure_public().fetch_order_book(symbol, limit=limit)

    def fetch_batch(
        self,
        symbols: Iterable[str],
        *,
        timeframe: str = "1m",
        use_orderbook: bool = False,
        limit_ohlcv: int = 500,
    ) -> List[Tuple[str, Optional[List[List[float]]], Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]]:
        results: List[
            Tuple[
                str,
                Optional[List[List[float]]],
                Optional[Dict[str, Any]],
                Optional[Dict[str, Any]],
                Optional[str],
            ]
        ] = []
        for symbol in symbols:
            ohlcv: Optional[List[List[float]]] = None
            ticker: Optional[Dict[str, Any]] = None
            orderbook: Optional[Dict[str, Any]] = None
            errors: List[str] = []

            try:
                ohlcv = self.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit_ohlcv)
            except Exception as exc:  # pragma: no cover - defensywnie
                errors.append(f"ohlcv: {exc}")

            try:
                ticker = self.fetch_ticker(symbol)
            except Exception as exc:  # pragma: no cover - defensywnie
                errors.append(f"ticker: {exc}")

            if use_orderbook:
                try:
                    orderbook = self.fetch_order_book(symbol, limit=50)
                except Exception as exc:  # pragma: no cover - defensywnie
                    errors.append(f"orderbook: {exc}")

            error_msg = "; ".join(errors) if errors else None
            results.append((symbol, ohlcv, ticker, orderbook, error_msg))

        return results

    def get_market_rules(self, symbol: str) -> Optional[MarketRules]:
        public = self._ensure_public()
        if not public.get_market_rules(symbol):
            public.load_markets()
        return public.get_market_rules(symbol)

    def quantize_amount(self, symbol: str, amount: float) -> float:
        rules = self.get_market_rules(symbol)
        return rules.quantize_amount(amount) if rules else float(f"{amount:.8f}")

    def quantize_price(self, symbol: str, price: float) -> float:
        rules = self.get_market_rules(symbol)
        return rules.quantize_price(price) if rules else float(f"{price:.8f}")

    def min_notional(self, symbol: str) -> float:
        rules = self.get_market_rules(symbol)
        return float(rules.min_notional) if rules else 0.0

    def simulate_vwap_price(
        self,
        symbol: str,
        side: str,
        amount: Optional[float],
        fallback_bps: float = 5.0,
        limit: int = 50,
    ) -> Tuple[Optional[float], float]:
        try:
            ticker = self.fetch_ticker(symbol) or {}
            last = ticker.get("last") or ticker.get("close") or ticker.get("bid") or ticker.get("ask")
            mid = float(last) if last else None
            if amount is None or amount <= 0:
                return (mid, float(fallback_bps))

            order_book = self.fetch_order_book(symbol, limit=limit) or {}
            side_lower = side.lower().strip()
            levels = order_book.get("asks") if side_lower == "buy" else order_book.get("bids")
            if not levels:
                return (mid, float(fallback_bps))

            remaining = float(amount)
            taken = 0.0
            cost = 0.0
            for price, qty in levels:
                take_qty = min(remaining - taken, float(qty))
                if take_qty <= 0:
                    break
                cost += take_qty * float(price)
                taken += take_qty
                if taken >= remaining - 1e-12:
                    break

            if taken <= 0:
                return (mid, float(fallback_bps))

            vwap = cost / taken
            if mid:
                slip_bps = abs(vwap - mid) / mid * 10_000.0
            else:
                slip_bps = float(fallback_bps)
            return (float(vwap), float(slip_bps))
        except Exception as exc:
            log.warning("simulate_vwap_price failed for %s: %s", symbol, exc)
            try:
                fallback = self.fetch_ticker(symbol) or {}
                last = fallback.get("last") or fallback.get("close")
                return (float(last) if last else None, float(fallback_bps))
            except Exception:
                return (None, float(fallback_bps))

    def fetch_balance(self) -> Dict[str, Any]:
        if self.mode == Mode.PAPER:
            return self._ensure_paper().fetch_balance()
        backend = self._ensure_private()
        raw = backend.fetch_balance()
        return self._normalize_balance(raw)

    @staticmethod
    def _normalize_balance(balance: Any) -> Dict[str, Any]:
        if not isinstance(balance, dict):
            return {}
        result: Dict[str, Any] = dict(balance)
        for key in ("free", "total", "used"):
            section = balance.get(key)
            if not isinstance(section, dict):
                continue
            normalized: Dict[str, float] = {}
            for asset, amount in section.items():
                try:
                    normalized[asset] = float(amount)
                except Exception:
                    continue
                result.setdefault(asset, normalized[asset])
            result[key] = normalized
        return result

    def create_order(
        self,
        symbol: str,
        side: str,
        type: str,
        quantity: float,
        price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> OrderDTO:
        side_enum = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
        type_enum = OrderType.MARKET if type.upper() == "MARKET" else OrderType.LIMIT

        if self.mode == Mode.PAPER:
            return self._ensure_paper().create_order(symbol, side_enum, type_enum, quantity, price, client_order_id)
        backend = self._ensure_private()
        return backend.create_order(symbol, side_enum, type_enum, quantity, price, client_order_id)

    def cancel_order(self, order_id: Any, symbol: str) -> bool:
        if self.mode == Mode.PAPER:
            return self._ensure_paper().cancel_order(order_id, symbol)
        return self._ensure_private().cancel_order(order_id, symbol)

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[OrderDTO]:
        if self.mode == Mode.PAPER:
            return self._ensure_paper().fetch_open_orders(symbol)
        return self._ensure_private().fetch_open_orders(symbol)

    def fetch_positions(self, symbol: Optional[str] = None) -> List[PositionDTO]:
        if self.mode == Mode.PAPER:
            return self._ensure_paper().fetch_positions(symbol)

        if self.mode == Mode.SPOT:
            try:
                db = self._ensure_db()
                if db:
                    positions = db.sync.get_open_positions(mode=Mode.SPOT.value)
                else:
                    positions = []
            except Exception as exc:
                log.debug("DB fallback failed: %s", exc)
                positions = []

            if positions:
                out: List[PositionDTO] = []
                for entry in positions:
                    try:
                        qty = float(entry.get("quantity") or 0.0)
                    except Exception:
                        continue
                    if qty <= 0:
                        continue
                    side_val = entry.get("side") or "LONG"
                    avg_price = float(entry.get("avg_price") or 0.0)
                    unreal = float(entry.get("unrealized_pnl") or 0.0)
                    sym = entry.get("symbol") or ""
                    out.append(
                        PositionDTO(
                            symbol=sym,
                            side=side_val,
                            quantity=qty,
                            avg_price=avg_price,
                            unrealized_pnl=unreal,
                            mode=Mode.SPOT,
                        )
                    )
                if out:
                    return out

            try:
                backend = self._ensure_private()
                balance = backend.fetch_balance()
            except Exception as exc:
                log.warning("Spot balance fallback failed: %s", exc)
                return []
            normalized = self._normalize_balance(balance)
            return self._positions_from_balance(normalized, symbol)

        return self._ensure_private().fetch_positions(symbol)

    def _positions_from_balance(
        self,
        balance: Dict[str, Any],
        symbol: Optional[str],
    ) -> List[PositionDTO]:
        if isinstance(balance.get("total"), dict):
            totals = dict(balance.get("total") or {})
        elif isinstance(balance.get("free"), dict):
            totals = dict(balance.get("free") or {})
        else:
            totals = {
                key: value
                for key, value in balance.items()
                if key not in {"free", "used", "total", "info"}
            }

        preferred_quotes = {
            self._paper_cash_asset.upper(),
            "USDT",
            "USD",
            "USDC",
            "BUSD",
            "EUR",
        }
        fallback_quote = self._paper_cash_asset.upper()
        markets: Dict[str, Any] = {}
        try:
            public = self._ensure_public()
            markets = public._markets or public.load_markets()
        except Exception as exc:  # pragma: no cover - informacyjne
            log.debug("Market load failed for balance conversion: %s", exc)

        symbol_filter = symbol
        base_filter: Optional[str] = None
        if symbol_filter:
            try:
                base_filter = symbol_filter.split("/")[0].upper()
            except Exception:
                base_filter = symbol_filter.upper()

        out: List[PositionDTO] = []
        for asset, amount in totals.items():
            try:
                qty = float(amount)
            except Exception:
                continue
            if qty <= 0:
                continue
            base = asset.upper()
            if base_filter and base != base_filter:
                continue
            if base in preferred_quotes:
                continue

            resolved_symbol = None
            if symbol_filter and base_filter == base:
                resolved_symbol = symbol_filter
            else:
                resolved_symbol = self._resolve_symbol_from_markets(
                    base, markets, preferred_quotes, fallback_quote
                )
            if resolved_symbol is None:
                resolved_symbol = base

            price = 0.0
            if resolved_symbol and "/" in resolved_symbol:
                try:
                    ticker = self.fetch_ticker(resolved_symbol) or {}
                    price = float(
                        ticker.get("last")
                        or ticker.get("close")
                        or ticker.get("bid")
                        or ticker.get("ask")
                        or 0.0
                    )
                except Exception:
                    price = 0.0

            out.append(
                PositionDTO(
                    symbol=resolved_symbol,
                    side="LONG",
                    quantity=qty,
                    avg_price=price,
                    unrealized_pnl=0.0,
                    mode=Mode.SPOT,
                )
            )
        return out

    def process_paper_tick(
        self,
        symbol: str,
        price: float,
        *,
        timestamp: Optional[dt.datetime] = None,
    ) -> None:
        if self.mode != Mode.PAPER:
            raise RuntimeError("process_paper_tick dostępne tylko w trybie paper")
        backend = self._ensure_paper()
        processor = getattr(backend, "process_tick", None)
        if not callable(processor):
            raise RuntimeError("Paper backend nie obsługuje process_tick")
        processor(symbol, price, timestamp=timestamp)

    def _resolve_symbol_from_markets(
        self,
        base: str,
        markets: Dict[str, Any],
        preferred_quotes: set,
        fallback_quote: str,
    ) -> Optional[str]:
        candidates: List[Tuple[str, str]] = []
        for symbol in markets.keys():
            if not isinstance(symbol, str) or "/" not in symbol:
                continue
            base_part, quote_part = symbol.split("/", 1)
            if base_part.upper() != base.upper():
                continue
            candidates.append((quote_part.upper(), symbol))

        for quote, candidate in candidates:
            if quote in preferred_quotes:
                return candidate
        if candidates:
            return candidates[0][1]
        if fallback_quote:
            return f"{base}/{fallback_quote}"
        return None

    def on(self, event_type: str, callback) -> None:
        self._event_bus.subscribe(event_type, callback)


__all__ = ["ExchangeManager", "Mode", "OrderDTO", "OrderSide", "OrderType", "OrderStatus", "PositionDTO"]

