"""Natywna implementacja fasady ExchangeManager."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)
from bot_core.exchanges.binance.futures import BinanceFuturesAdapter
from bot_core.exchanges.binance.margin import BinanceMarginAdapter
from bot_core.exchanges.kraken.futures import KrakenFuturesAdapter
from bot_core.exchanges.kraken.margin import KrakenMarginAdapter
from bot_core.exchanges.health import (
    CircuitBreaker,
    HealthCheck,
    HealthMonitor,
    RetryPolicy,
    Watchdog,
)
from bot_core.exchanges.zonda.margin import ZondaMarginAdapter

try:  # pragma: no cover
    import ccxt  # type: ignore
except Exception:  # pragma: no cover
    ccxt = None


log = logging.getLogger(__name__)


_NATIVE_MARGIN_ADAPTERS = {
    "binance": BinanceMarginAdapter,
    "kraken": KrakenMarginAdapter,
    "zonda": ZondaMarginAdapter,
}

_NATIVE_FUTURES_ADAPTERS = {
    "binance": BinanceFuturesAdapter,
    "kraken": KrakenFuturesAdapter,
}

_STATUS_MAPPING = {
    "NEW": OrderStatus.OPEN,
    "OPEN": OrderStatus.OPEN,
    "PENDING_NEW": OrderStatus.OPEN,
    "PENDING": OrderStatus.OPEN,
    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
    "PARTIALLY": OrderStatus.PARTIALLY_FILLED,
    "FILLED": OrderStatus.FILLED,
    "CANCELED": OrderStatus.CANCELED,
    "CANCELLED": OrderStatus.CANCELED,
    "PENDING_CANCEL": OrderStatus.CANCELED,
    "EXPIRED": OrderStatus.CANCELED,
    "REJECTED": OrderStatus.REJECTED,
}


def _map_order_status(raw: object) -> OrderStatus:
    if isinstance(raw, OrderStatus):
        return raw
    value = str(raw or "").upper()
    return _STATUS_MAPPING.get(value, OrderStatus.OPEN)


def _map_order_side(raw: object) -> OrderSide:
    if isinstance(raw, OrderSide):
        return raw
    return OrderSide.BUY if str(raw or "").upper() == "BUY" else OrderSide.SELL


def _map_order_type(raw: object) -> OrderType:
    if isinstance(raw, OrderType):
        return raw
    value = str(raw or "").upper()
    if value == "LIMIT":
        return OrderType.LIMIT
    if value == "MARKET":
        return OrderType.MARKET
    return OrderType.MARKET if "MARKET" in value else OrderType.LIMIT


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
        self._db_url = db_url or "sqlite+aiosqlite:///trading.db"
        self._db: Optional[DatabaseManager] = None
        self._db_failed: bool = False
        self._native_adapter = None
        self._native_adapter_settings: Dict[tuple[Mode, str], Dict[str, object]] = {}
        self._watchdog: Watchdog | None = None
        default_margin_type = os.getenv("BINANCE_MARGIN_TYPE")
        if self.exchange_id == "binance" and default_margin_type:
            self._native_adapter_settings[(Mode.MARGIN, self.exchange_id)] = {
                "margin_type": default_margin_type,
            }

        log.info("ExchangeManager initialized (bot_core)")

    def set_mode(
        self,
        *,
        paper: bool = False,
        spot: bool = False,
        margin: bool = False,
        futures: bool = False,
        testnet: bool = False,
    ) -> None:
        selected = [paper, spot, margin, futures]
        if sum(1 for flag in selected if flag) > 1:
            raise ValueError("Można wybrać tylko jeden tryb: paper, spot, margin lub futures.")

        if paper:
            self.mode = Mode.PAPER
            self._futures = False
            self._testnet = False
        elif futures:
            self.mode = Mode.FUTURES
            self._futures = True
            self._testnet = bool(testnet)
        elif margin:
            self.mode = Mode.MARGIN
            self._futures = False
            self._testnet = bool(testnet)
        else:
            self.mode = Mode.SPOT
            self._futures = False
            self._testnet = bool(testnet)

        log.info("Mode set to %s (futures=%s, testnet=%s)", self.mode.value, self._futures, self._testnet)
        self._private = None
        self._paper = None
        self._native_adapter = None

    def set_credentials(self, api_key: Optional[str], secret: Optional[str]) -> None:
        self._api_key = (api_key or "").strip() or None
        self._secret = (secret or "").strip() or None
        log.info(
            "Credentials set (lengths): api_key=%s, secret=%s",
            len(self._api_key or 0),
            len(self._secret or 0),
        )
        self._native_adapter = None

    def configure_native_adapter(
        self,
        *,
        settings: Mapping[str, object],
        mode: Mode | None = None,
    ) -> None:
        if not isinstance(settings, Mapping):
            raise TypeError("Konfiguracja adaptera musi być mapowaniem.")
        target_mode = mode or self.mode
        if target_mode not in {Mode.MARGIN, Mode.FUTURES}:
            raise ValueError("Konfiguracja natywnego adaptera jest dostępna tylko dla trybów margin/futures.")
        self._native_adapter_settings[(target_mode, self.exchange_id)] = dict(settings)
        self._native_adapter = None

    def set_watchdog(self, watchdog: Watchdog | None) -> None:
        """Ustawia współdzielony watchdog dla natywnych adapterów margin/futures."""

        if watchdog is not None and not isinstance(watchdog, Watchdog):
            raise TypeError("Watchdog musi być instancją klasy bot_core.exchanges.health.Watchdog")
        self._watchdog = watchdog
        self._native_adapter = None

    def configure_watchdog(
        self,
        *,
        retry_policy: Mapping[str, object] | None = None,
        circuit_breaker: Mapping[str, object] | None = None,
        retry_exceptions: Sequence[type[Exception]] | None = None,
    ) -> None:
        """Buduje i ustawia watchdog na podstawie przekazanych parametrów."""

        kwargs: Dict[str, object] = {}
        if retry_policy is not None:
            if not isinstance(retry_policy, Mapping):
                raise TypeError("retry_policy musi być mapowaniem z parametrami RetryPolicy")
            kwargs["retry_policy"] = RetryPolicy(**dict(retry_policy))
        if circuit_breaker is not None:
            if not isinstance(circuit_breaker, Mapping):
                raise TypeError("circuit_breaker musi być mapowaniem z parametrami CircuitBreaker")
            kwargs["circuit_breaker"] = CircuitBreaker(**dict(circuit_breaker))
        if retry_exceptions is not None:
            if not isinstance(retry_exceptions, Sequence):
                raise TypeError("retry_exceptions musi być sekwencją klas wyjątków")
            normalized: list[type[Exception]] = []
            for exc in retry_exceptions:
                if not isinstance(exc, type) or not issubclass(exc, Exception):
                    raise TypeError("retry_exceptions musi zawierać klasy wyjątków")
                normalized.append(exc)
            kwargs["retry_exceptions"] = tuple(normalized)
        self._watchdog = Watchdog(**kwargs)
        self._native_adapter = None

    def create_health_monitor(self, checks: Iterable[HealthCheck]) -> HealthMonitor:
        """Buduje `HealthMonitor` współdzielący strażnika z adapterami."""

        if not isinstance(checks, Iterable):
            raise TypeError("checks musi być iterowalną sekwencją HealthCheck")

        normalized: list[HealthCheck] = []
        for check in checks:
            if not isinstance(check, HealthCheck):
                raise TypeError("checks musi zawierać instancje HealthCheck")
            normalized.append(check)

        return HealthMonitor(normalized, watchdog=self._ensure_watchdog())

    def _ensure_public(self) -> _CCXTPublicFeed:
        if self._public is None:
            self._public = _CCXTPublicFeed(exchange_id=self.exchange_id, testnet=self._testnet)
        return self._public

    def _resolve_environment(self) -> Environment:
        if self.mode is Mode.PAPER:
            return Environment.PAPER

        candidates = []
        if self.exchange_id.startswith("binance"):
            candidates.append(os.getenv("BINANCE_ENVIRONMENT"))
        if self.exchange_id.startswith("kraken"):
            candidates.append(os.getenv("KRAKEN_ENVIRONMENT"))
        if self.exchange_id.startswith("zonda"):
            candidates.append(os.getenv("ZONDA_ENVIRONMENT"))
        candidates.append(os.getenv("EXCHANGE_ENVIRONMENT"))

        for candidate in candidates:
            if not candidate:
                continue
            try:
                environment = Environment(candidate.strip().lower())
            except (ValueError, AttributeError):
                continue
            if environment is Environment.PAPER:
                return Environment.PAPER
            return environment

        return Environment.TESTNET if self._testnet else Environment.LIVE

    def _get_adapter_settings(self) -> Dict[str, object]:
        return dict(self._native_adapter_settings.get((self.mode, self.exchange_id), {}))

    def _ensure_watchdog(self) -> Watchdog:
        if self._watchdog is None:
            self._watchdog = Watchdog()
        return self._watchdog

    def _ensure_native_adapter(self):
        if self.mode not in {Mode.MARGIN, Mode.FUTURES}:
            raise RuntimeError("Natywny adapter dostępny jest wyłącznie w trybach margin/futures.")
        if not self._api_key or not self._secret:
            raise RuntimeError("Brak API Key/Secret – ustaw je przed użyciem trybu live/testnet.")

        if self.mode == Mode.MARGIN:
            factory = _NATIVE_MARGIN_ADAPTERS.get(self.exchange_id)
        else:
            factory = _NATIVE_FUTURES_ADAPTERS.get(self.exchange_id)

        if factory is None:
            raise RuntimeError(
                f"Brak natywnego adaptera dla giełdy {self.exchange_id} w trybie {self.mode.value}."
            )

        if self._native_adapter is None:
            environment = self._resolve_environment()
            credentials = ExchangeCredentials(
                key_id=self._api_key,
                secret=self._secret,
                environment=environment,
                permissions=("read", "trade"),
            )
            settings = self._get_adapter_settings()
            kwargs: Dict[str, object] = {"environment": environment}
            if settings:
                kwargs["settings"] = settings
            kwargs["watchdog"] = self._ensure_watchdog()
            self._native_adapter = factory(credentials, **kwargs)

        return self._native_adapter

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

        if self.mode in {Mode.MARGIN, Mode.FUTURES}:
            adapter = self._ensure_native_adapter()
            snapshot = adapter.fetch_account_snapshot()
            balances = {
                asset: float(amount)
                for asset, amount in dict(snapshot.balances).items()
                if isinstance(asset, str)
            }
            free = dict(balances)
            total = dict(balances)
            used = {asset: 0.0 for asset in balances}
            # Zachowujemy historyczną strukturę (total/free/klucze walut), aby
            # istniejący kod (np. silnik handlowy) nadal znajdował dostępny
            # kapitał w snapshotach dla trybu margin/futures.
            return {
                **balances,
                "balances": balances,
                "total_equity": float(snapshot.total_equity),
                "available_margin": float(snapshot.available_margin),
                "maintenance_margin": float(snapshot.maintenance_margin),
                "total": total,
                "free": free,
                "used": used,
            }

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

        if self.mode in {Mode.MARGIN, Mode.FUTURES}:
            rules = self.get_market_rules(symbol)
            if not rules:
                self.load_markets()
                rules = self.get_market_rules(symbol)
            if not rules:
                raise RuntimeError(f"Brak reguł rynku dla {symbol}. Najpierw załaduj rynek.")

            qty = rules.quantize_amount(float(quantity))
            if qty <= 0:
                raise ValueError("Ilość po kwantyzacji = 0.")

            price_value: Optional[float] = None
            if type_enum is OrderType.LIMIT:
                if price is None:
                    raise ValueError("Cena wymagana dla LIMIT.")
                price_value = rules.quantize_price(float(price))

            if type_enum is OrderType.MARKET:
                ticker = self.fetch_ticker(symbol) or {}
                last = ticker.get("last") or ticker.get("close") or ticker.get("bid") or ticker.get("ask")
                if not last:
                    raise RuntimeError(f"Brak ceny MARKET dla {symbol}.")
                notional = qty * float(last)
            else:
                notional = qty * float(price_value or 0.0)

            min_notional = rules.min_notional or 0.0
            if min_notional and notional < min_notional:
                raise ValueError(
                    f"Notional {notional:.8f} < minNotional {min_notional:.8f} dla {symbol}"
                )

            adapter = self._ensure_native_adapter()
            request = OrderRequest(
                symbol=symbol,
                side=side_enum.value,
                quantity=qty,
                order_type=type_enum.value,
                price=price_value,
                client_order_id=client_order_id,
            )
            result = adapter.place_order(request)
            raw_payload = result.raw_response if isinstance(result.raw_response, Mapping) else {}
            resolved_client_id = client_order_id
            if not resolved_client_id and isinstance(raw_payload, Mapping):
                candidate = (
                    raw_payload.get("clientOrderId")
                    or raw_payload.get("client_order_id")
                    or raw_payload.get("userref")
                )
                if isinstance(candidate, str) and candidate:
                    resolved_client_id = candidate
            order_identifier = result.order_id
            try:
                parsed_id = int(order_identifier) if order_identifier is not None else None
            except (TypeError, ValueError):
                parsed_id = None

            return OrderDTO(
                id=parsed_id,
                client_order_id=resolved_client_id,
                symbol=symbol,
                side=side_enum,
                type=type_enum,
                quantity=qty,
                price=price_value,
                status=_map_order_status(result.status),
                mode=self.mode,
                extra={
                    "order_id": order_identifier,
                    "filled_quantity": result.filled_quantity,
                    "avg_price": result.avg_price,
                    "raw_response": raw_payload,
                },
            )

        backend = self._ensure_private()
        return backend.create_order(symbol, side_enum, type_enum, quantity, price, client_order_id)

    def cancel_order(self, order_id: Any, symbol: str) -> bool:
        if self.mode == Mode.PAPER:
            return False
        if self.mode in {Mode.MARGIN, Mode.FUTURES}:
            try:
                adapter = self._ensure_native_adapter()
                adapter.cancel_order(str(order_id), symbol=symbol)
                return True
            except Exception as exc:
                log.error("cancel_order failed (native): %s", exc)
                return False
        return self._ensure_private().cancel_order(order_id, symbol)

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[OrderDTO]:
        if self.mode == Mode.PAPER:
            return []
        if self.mode in {Mode.MARGIN, Mode.FUTURES}:
            try:
                adapter = self._ensure_native_adapter()
                native_orders = adapter.fetch_open_orders()
            except Exception as exc:
                log.error("fetch_open_orders failed (native): %s", exc)
                return []

            result: List[OrderDTO] = []
            for entry in native_orders or []:
                raw_symbol = getattr(entry, "symbol", symbol or "")
                order_symbol = raw_symbol if isinstance(raw_symbol, str) else symbol or ""
                price_value = getattr(entry, "price", None)
                if price_value in (None, ""):
                    resolved_price = None
                else:
                    try:
                        resolved_price = float(price_value)
                    except Exception:
                        resolved_price = None
                quantity_value = getattr(entry, "orig_quantity", getattr(entry, "quantity", 0.0))
                try:
                    resolved_quantity = float(quantity_value)
                except Exception:
                    resolved_quantity = 0.0
                order_identifier = getattr(entry, "order_id", None)
                try:
                    parsed_id = int(order_identifier) if order_identifier is not None else None
                except (TypeError, ValueError):
                    parsed_id = None
                result.append(
                    OrderDTO(
                        id=parsed_id,
                        client_order_id=getattr(entry, "client_order_id", None),
                        symbol=order_symbol,
                        side=_map_order_side(getattr(entry, "side", "BUY")),
                        type=_map_order_type(getattr(entry, "order_type", "LIMIT")),
                        quantity=resolved_quantity,
                        price=resolved_price,
                        status=_map_order_status(getattr(entry, "status", "OPEN")),
                        mode=self.mode,
                        extra={"order_id": order_identifier},
                    )
                )
            return result
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

        if self.mode == Mode.FUTURES:
            try:
                adapter = self._ensure_native_adapter()
                native_positions = adapter.fetch_positions()
            except RuntimeError as exc:
                log.warning("Fallback to CCXT futures backend: %s", exc)
            except Exception as exc:
                log.error("fetch_positions failed (native): %s", exc)
                return []
            else:
                result: List[PositionDTO] = []
                for entry in native_positions or []:
                    try:
                        quantity = float(getattr(entry, "quantity", 0.0) or 0.0)
                    except Exception:
                        quantity = 0.0
                    if abs(quantity) < 1e-12:
                        continue
                    avg_price = getattr(entry, "entry_price", getattr(entry, "avg_price", 0.0))
                    try:
                        resolved_avg = float(avg_price)
                    except Exception:
                        resolved_avg = 0.0
                    try:
                        pnl = float(getattr(entry, "unrealized_pnl", 0.0) or 0.0)
                    except Exception:
                        pnl = 0.0
                    result.append(
                        PositionDTO(
                            symbol=str(getattr(entry, "symbol", "")),
                            side=str(getattr(entry, "side", "LONG")),
                            quantity=abs(quantity),
                            avg_price=resolved_avg,
                            unrealized_pnl=pnl,
                            mode=Mode.FUTURES,
                        )
                    )
                return result

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

