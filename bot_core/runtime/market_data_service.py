"""Serwis gRPC udostępniający dane o instrumentach giełdowych.

Moduł udostępnia również tryb pracy bez gRPC wykorzystujący lokalny poller
REST oparty na :class:`ExchangeManager`, co pozwala na korzystanie z tych samych
danych rynku w środowiskach, gdzie stuby gRPC nie są dostępne.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent import futures
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping

try:  # pragma: no cover - zależne od opcjonalnych pakietów
    import grpc
except Exception:  # pragma: no cover - brak biblioteki grpcio
    grpc = None  # type: ignore

try:  # pragma: no cover - zależne od wygenerowanych stubów
    from bot_core.generated import trading_pb2, trading_pb2_grpc  # type: ignore
except Exception:  # pragma: no cover - brak stubów trading
    trading_pb2 = None  # type: ignore
    trading_pb2_grpc = None  # type: ignore

from bot_core.exchanges.manager import ExchangeManager
from bot_core.runtime._grpc_support import GrpcServerLifecycleMixin

_LOG = logging.getLogger(__name__)

ManagerLookup = Callable[[str], ExchangeManager | None] | Mapping[str, ExchangeManager]


class _DefaultManagerProvider:
    """Buduje i cache'uje instancje ExchangeManager dla wskazanych giełd."""

    def __init__(
        self,
        *,
        profile: str | None = None,
        config_dir: str | os.PathLike[str] | None = None,
        profile_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        self._cache: MutableMapping[str, ExchangeManager] = {}
        self._lock = threading.Lock()
        normalized_profile = (profile or "").strip().lower()
        self._profile_name: str | None = normalized_profile or None
        self._config_dir = config_dir
        self._profile_overrides = dict(profile_overrides or {})

    def __call__(self, exchange: str) -> ExchangeManager:
        normalized = (exchange or "").strip().lower()
        if not normalized:
            raise ValueError("exchange identifier cannot be empty")
        with self._lock:
            manager = self._cache.get(normalized)
            if manager is None:
                manager = ExchangeManager(exchange_id=normalized)
                if self._profile_name:
                    try:
                        overrides = dict(self._profile_overrides)
                        manager.apply_environment_profile(
                            self._profile_name,
                            config_dir=self._config_dir,
                            overrides=overrides or None,
                        )
                    except Exception as exc:  # pragma: no cover - diagnostyka profili
                        _LOG.exception(
                            "Nie udało się zastosować profilu środowiska %s dla %s: %s",
                            self._profile_name,
                            normalized,
                            exc,
                        )
                        raise
                self._cache[normalized] = manager
        return manager


def _build_instrument_payload(
    exchange_upper: str,
    symbol: str,
    rules: Any,
    market_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    base, quote = _split_symbol_text(symbol)
    if market_meta:
        base = str(market_meta.get("base") or market_meta.get("baseId") or base or "").upper()
        quote = str(market_meta.get("quote") or market_meta.get("quoteId") or quote or "").upper()
    venue = str(
        market_meta.get("id")
        or market_meta.get("symbol")
        or symbol.replace("/", "").replace("-", "").replace(":", "")
    )

    payload = {
        "instrument": {
            "exchange": exchange_upper,
            "symbol": symbol,
            "venue_symbol": venue,
            "quote_currency": quote,
            "base_currency": base,
        },
        "price_step": float(getattr(rules, "price_step", 0.0) or 0.0),
        "amount_step": float(getattr(rules, "amount_step", 0.0) or 0.0),
        "min_notional": float(getattr(rules, "min_notional", 0.0) or 0.0),
        "min_amount": float(getattr(rules, "min_amount", 0.0) or 0.0),
    }

    max_amount = getattr(rules, "max_amount", None)
    if max_amount is not None:
        payload["max_amount"] = float(max_amount)
    min_price = getattr(rules, "min_price", None)
    if min_price is not None:
        payload["min_price"] = float(min_price)
    max_price = getattr(rules, "max_price", None)
    if max_price is not None:
        payload["max_price"] = float(max_price)
    return payload


def _split_symbol_text(symbol: str) -> tuple[str, str]:
    if not symbol:
        return "", ""
    for sep in ("/", "-", ":", "_"):
        if sep in symbol:
            base, quote = symbol.split(sep, 1)
            return base.strip().upper(), quote.strip().upper()
    return symbol.strip().upper(), ""


class MarketDataServiceServicer(
    trading_pb2_grpc.MarketDataServiceServicer if trading_pb2_grpc else object  # type: ignore[misc]
):
    """Implementacja usług gRPC rynku dla desktopowego frontendu."""

    def __init__(
        self,
        manager_lookup: ManagerLookup | None = None,
        *,
        cache_ttl: float = 300.0,
        logger: logging.Logger | None = None,
    ) -> None:
        if trading_pb2 is None or trading_pb2_grpc is None:
            raise RuntimeError("Uruchomienie MarketDataService wymaga wygenerowanych stubów trading_pb2*")
        if grpc is None:
            raise RuntimeError("Uruchomienie MarketDataService wymaga biblioteki grpcio")
        try:  # pragma: no cover - zależne od wersji gRPC
            super().__init__()  # type: ignore[misc]
        except Exception:
            pass

        self._default_provider: _DefaultManagerProvider | None = None
        if manager_lookup is None:
            self._default_provider = _DefaultManagerProvider()
            self._manager_provider: Callable[[str], ExchangeManager | None] = self._default_provider
        elif callable(manager_lookup):
            self._manager_provider = manager_lookup
        else:
            mapping = {key.strip().upper(): value for key, value in manager_lookup.items()}

            def _mapping_provider(exchange_upper: str) -> ExchangeManager | None:
                return mapping.get(exchange_upper.strip().upper())

            self._manager_provider = _mapping_provider

        self._cache_ttl = max(0.0, float(cache_ttl))
        self._cache: Dict[str, tuple[float, List[trading_pb2.TradableInstrumentMetadata]]] = {}
        self._cache_lock = threading.Lock()
        self._logger = logger or _LOG

    # ------------------------------------------------------------------
    # gRPC API
    # ------------------------------------------------------------------
    def ListTradableInstruments(self, request, context):  # noqa: N802 - konwencja gRPC
        if trading_pb2 is None:
            raise RuntimeError("Brak modułu trading_pb2")
        exchange = (request.exchange or "").strip()
        if not exchange:
            return self._abort(context, grpc.StatusCode.INVALID_ARGUMENT if grpc else None, "exchange jest wymagany")

        exchange_upper = exchange.upper()
        now = time.monotonic()
        if self._cache_ttl > 0:
            with self._cache_lock:
                cached = self._cache.get(exchange_upper)
                if cached and (now - cached[0]) <= self._cache_ttl:
                    response = trading_pb2.ListTradableInstrumentsResponse()
                    response.instruments.extend(self._clone_metadata(entry) for entry in cached[1])
                    return response

        manager = self._manager_provider(exchange_upper)
        if manager is None:
            if self._default_provider is not None:
                try:
                    manager = self._default_provider(exchange_upper)
                except Exception as exc:  # pragma: no cover - diagnostyka
                    self._logger.debug("Nie udało się utworzyć ExchangeManager dla %s: %s", exchange_upper, exc)
                    manager = None
        if manager is None:
            return self._abort(
                context,
                grpc.StatusCode.NOT_FOUND if grpc else None,
                f"Exchange '{exchange_upper}' nie jest skonfigurowana",
            )

        try:
            entries = self._build_instrument_list(manager, exchange_upper)
        except Exception as exc:
            self._logger.exception("ListTradableInstruments nie powiodło się dla %s", exchange_upper)
            return self._abort(context, grpc.StatusCode.UNAVAILABLE if grpc else None, str(exc))

        if self._cache_ttl > 0:
            snapshot = [self._clone_metadata(item) for item in entries]
            with self._cache_lock:
                self._cache[exchange_upper] = (time.monotonic(), snapshot)

        response = trading_pb2.ListTradableInstrumentsResponse()
        response.instruments.extend(self._clone_metadata(item) for item in entries)
        return response

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_instrument_list(
        self,
        manager: ExchangeManager,
        exchange_upper: str,
    ) -> List[trading_pb2.TradableInstrumentMetadata]:
        if trading_pb2 is None:
            raise RuntimeError("Brak modułu trading_pb2")

        rules_map = manager.load_markets()
        public = getattr(manager, "_public", None)
        if public is None:
            ensure_public = getattr(manager, "_ensure_public", None)
            if callable(ensure_public):  # pragma: no cover - zależne od implementacji
                try:
                    public = ensure_public()
                except Exception:
                    public = None
        markets: Mapping[str, Any]
        markets = getattr(public, "_markets", {}) if public is not None else {}

        instruments: List[trading_pb2.TradableInstrumentMetadata] = []
        for symbol, rules in rules_map.items():
            market_meta = markets.get(symbol, {}) if isinstance(markets, Mapping) else {}
            instruments.append(self._build_metadata(exchange_upper, symbol, rules, market_meta))
        instruments.sort(key=lambda item: item.instrument.symbol or "")
        return instruments

    def _build_metadata(
        self,
        exchange_upper: str,
        symbol: str,
        rules: Any,
        market_meta: Mapping[str, Any],
    ) -> trading_pb2.TradableInstrumentMetadata:
        if trading_pb2 is None:
            raise RuntimeError("Brak modułu trading_pb2")

        base, quote = self._split_symbol(symbol)
        if market_meta:
            base = str(market_meta.get("base") or market_meta.get("baseId") or base or "").upper()
            quote = str(market_meta.get("quote") or market_meta.get("quoteId") or quote or "").upper()
        venue = str(
            market_meta.get("id")
            or market_meta.get("symbol")
            or symbol.replace("/", "").replace("-", "").replace(":", "")
        )

        instrument = trading_pb2.Instrument(
            exchange=exchange_upper,
            symbol=symbol,
            venue_symbol=venue,
            quote_currency=quote,
            base_currency=base,
        )
        metadata = trading_pb2.TradableInstrumentMetadata(
            instrument=instrument,
            price_step=float(getattr(rules, "price_step", 0.0) or 0.0),
            amount_step=float(getattr(rules, "amount_step", 0.0) or 0.0),
            min_notional=float(getattr(rules, "min_notional", 0.0) or 0.0),
            min_amount=float(getattr(rules, "min_amount", 0.0) or 0.0),
        )

        max_amount = getattr(rules, "max_amount", None)
        if max_amount is not None:
            metadata.max_amount = float(max_amount)
        min_price = getattr(rules, "min_price", None)
        if min_price is not None:
            metadata.min_price = float(min_price)
        max_price = getattr(rules, "max_price", None)
        if max_price is not None:
            metadata.max_price = float(max_price)
        return metadata

    @staticmethod
    def _split_symbol(symbol: str) -> tuple[str, str]:
        return _split_symbol_text(symbol)

    @staticmethod
    def _clone_metadata(metadata: trading_pb2.TradableInstrumentMetadata) -> trading_pb2.TradableInstrumentMetadata:
        clone = trading_pb2.TradableInstrumentMetadata()
        clone.CopyFrom(metadata)
        return clone

    @staticmethod
    def _abort(context, status_code, message):
        if context is not None and grpc is not None and status_code is not None:
            context.abort(status_code, message)
        raise RuntimeError(message)


class MarketDataServer(GrpcServerLifecycleMixin):
    """Pełny serwer gRPC wystawiający usługi rynku danych."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        max_workers: int = 8,
        manager_lookup: ManagerLookup | None = None,
        cache_ttl: float = 300.0,
        server_credentials: Any | None = None,
    ) -> None:
        if grpc is None or trading_pb2_grpc is None:
            raise RuntimeError("Uruchomienie MarketDataServer wymaga pakietów grpcio oraz trading_pb2*")
        self._servicer = MarketDataServiceServicer(manager_lookup, cache_ttl=cache_ttl)
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        trading_pb2_grpc.add_MarketDataServiceServicer_to_server(self._servicer, self._server)
        self._address = f"{host}:{port}"
        if server_credentials is not None:
            bound_port = self._server.add_secure_port(self._address, server_credentials)
            self._tls_enabled = True
        else:
            bound_port = self._server.add_insecure_port(self._address)
            self._tls_enabled = False
        if bound_port == 0:
            raise RuntimeError("Nie udało się zbindować portu dla MarketDataServer")
        if port == 0:
            self._address = f"{host}:{bound_port}"

    @property
    def address(self) -> str:
        return self._address

    @property
    def servicer(self) -> MarketDataServiceServicer:
        return self._servicer

    @property
    def tls_enabled(self) -> bool:
        return getattr(self, "_tls_enabled", False)

    def start(self) -> None:
        self._server.start()


class RestMarketDataPoller:
    """Lekki poller REST korzystający z :class:`ExchangeManager` bez gRPC."""

    def __init__(
        self,
        exchanges: Iterable[str],
        *,
        manager_lookup: ManagerLookup | None = None,
        interval: float = 120.0,
        logger: logging.Logger | None = None,
        profile: str | None = None,
        config_dir: str | os.PathLike[str] | None = None,
        profile_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        exchanges = [str(exchange or "").strip() for exchange in exchanges if exchange]
        if not exchanges:
            raise ValueError("RestMarketDataPoller wymaga listy giełd do odpytywania")
        if interval <= 0:
            raise ValueError("Interwał odpytywania musi być dodatni")

        self._exchanges = sorted({exchange.upper() for exchange in exchanges})
        self._interval = float(interval)
        self._logger = logger or _LOG
        self._default_provider: _DefaultManagerProvider | None = None
        if manager_lookup is None:
            self._default_provider = _DefaultManagerProvider(
                profile=profile,
                config_dir=config_dir,
                profile_overrides=profile_overrides,
            )
            self._manager_provider: Callable[[str], ExchangeManager | None] = self._default_provider
        elif callable(manager_lookup):
            self._manager_provider = manager_lookup
        else:
            mapping = {key.strip().upper(): value for key, value in manager_lookup.items()}

            def _mapping_provider(exchange_upper: str) -> ExchangeManager | None:
                return mapping.get(exchange_upper.strip().upper())

            self._manager_provider = _mapping_provider

        self._snapshots: Dict[str, list[Dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="rest-market-data-poller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 1.0)
            self._thread = None

    def refresh_now(self) -> None:
        for exchange in self._exchanges:
            self._update_exchange(exchange)

    def snapshot(self, exchange: str) -> list[Dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._snapshots.get(exchange.upper(), [])]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop_event.is_set():
            start = time.monotonic()
            try:
                self.refresh_now()
            except Exception as exc:  # pragma: no cover - logowanie diagnostyczne
                self._logger.exception("Błąd podczas odświeżania danych REST: %s", exc)
            elapsed = time.monotonic() - start
            timeout = max(0.0, self._interval - elapsed)
            self._stop_event.wait(timeout)

    def _update_exchange(self, exchange_upper: str) -> None:
        manager = self._manager_provider(exchange_upper)
        if manager is None and self._default_provider is not None:
            try:
                manager = self._default_provider(exchange_upper)
            except Exception as exc:  # pragma: no cover - logowanie diagnostyczne
                self._logger.debug("Nie udało się utworzyć ExchangeManager dla %s: %s", exchange_upper, exc)
                manager = None
        if manager is None:
            self._logger.warning("Pominięto odświeżenie danych dla nieznanej giełdy: %s", exchange_upper)
            return

        try:
            rules_map = manager.load_markets()
            public = getattr(manager, "_public", None)
            if public is None:
                ensure_public = getattr(manager, "_ensure_public", None)
                if callable(ensure_public):
                    public = ensure_public()
            markets: Mapping[str, Any]
            markets = getattr(public, "_markets", {}) if public is not None else {}
        except Exception as exc:  # pragma: no cover - zależne od CCXT
            self._logger.exception("Nie udało się pobrać listy instrumentów dla %s", exchange_upper)
            return

        entries: list[Dict[str, Any]] = []
        for symbol, rules in rules_map.items():
            market_meta = markets.get(symbol, {}) if isinstance(markets, Mapping) else {}
            entries.append(_build_instrument_payload(exchange_upper, symbol, rules, market_meta))
        entries.sort(key=lambda item: item["instrument"]["symbol"] or "")
        with self._lock:
            self._snapshots[exchange_upper] = entries


__all__ = ["MarketDataServiceServicer", "MarketDataServer", "RestMarketDataPoller"]
