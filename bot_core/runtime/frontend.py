"""Bootstrap helpers used by desktop frontends and AutoTrader."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from bot_core.logging import get_logger

try:  # pragma: no cover - market intel optional in stripped distributions
    from bot_core.market_intel import MarketIntelAggregator
except Exception:  # pragma: no cover - keep defensive fallback in tests
    MarketIntelAggregator = None  # type: ignore[assignment]

try:  # pragma: no cover - ExchangeManager may be excluded from minimal bundles
    from bot_core.exchanges.manager import ExchangeManager
except Exception:  # pragma: no cover
    ExchangeManager = None  # type: ignore[assignment]

try:  # pragma: no cover - optional runtime utilities
    from bot_core.runtime.paths import DesktopAppPaths, resolve_core_config_path
except Exception:  # pragma: no cover
    DesktopAppPaths = None  # type: ignore[assignment]
    resolve_core_config_path = None  # type: ignore[assignment]

try:  # pragma: no cover - configuration loader may be missing in tests
    from bot_core.config.loader import load_core_config
except Exception:  # pragma: no cover
    load_core_config = None  # type: ignore[assignment]

try:  # pragma: no cover - execution/runtime modules may be optional
    from bot_core.execution.live_router import LiveExecutionRouter
    from bot_core.exchanges.base import (
        AccountSnapshot,
        Environment as ExecutionEnvironment,
        ExchangeAdapter,
        ExchangeCredentials,
        OrderRequest as CoreOrderRequest,
        OrderResult as CoreOrderResult,
    )
except Exception:  # pragma: no cover
    LiveExecutionRouter = None  # type: ignore[assignment]
    ExchangeAdapter = None  # type: ignore[assignment]
    ExecutionEnvironment = None  # type: ignore[assignment]
    ExchangeCredentials = None  # type: ignore[assignment]
    CoreOrderRequest = None  # type: ignore[assignment]
    CoreOrderResult = None  # type: ignore[assignment]


logger = get_logger(__name__)


@dataclass(frozen=True)
class FrontendBootstrap:
    """Collection of services shared by GUI, dashboard and AutoTrader."""

    exchange_manager: Any | None
    market_intel: Any | None
    execution_service: Any | None = None
    router: Any | None = None


def _normalise_path(candidate: Path) -> Path:
    expanded = candidate.expanduser()
    try:
        return expanded.resolve(strict=False)
    except Exception:  # pragma: no cover - platform specific
        return expanded


def _resolve_frontend_config_path(
    paths: DesktopAppPaths | None,
    explicit: str | Path | None,
) -> Path | None:
    candidates: list[Path] = []
    if explicit is not None:
        try:
            candidates.append(Path(explicit))
        except Exception:  # pragma: no cover - invalid path
            logger.debug("Niepoprawna jawna ścieżka konfiguracji frontendu", exc_info=True)
    if paths is not None:
        app_root = getattr(paths, "app_root", None)
        if isinstance(app_root, Path):
            candidates.append(app_root / "config" / "core.yaml")
            candidates.append(app_root.parent / "config" / "core.yaml")
    if resolve_core_config_path is not None:
        try:
            candidates.append(resolve_core_config_path())
        except Exception:  # pragma: no cover - resolver unavailable
            logger.debug("Nie udało się ustalić ścieżki konfiguracji z resolvera", exc_info=True)

    normalised: list[Path] = []
    for candidate in candidates:
        try:
            normalised.append(_normalise_path(Path(candidate)))
        except Exception:  # pragma: no cover - invalid candidate
            logger.debug("Pominięto niepoprawną kandydacką ścieżkę konfiguracji", exc_info=True)
    for candidate in normalised:
        if candidate.exists():
            return candidate
    return normalised[0] if normalised else None


def _load_market_intel_from_config(config_path: str | Path | None) -> Any | None:
    if MarketIntelAggregator is None or load_core_config is None:
        return None

    candidate_path: Path | None = None
    if config_path is not None:
        try:
            candidate_path = Path(config_path)
        except Exception:  # pragma: no cover
            logger.debug("Nie udało się sparsować ścieżki konfiguracji market intel", exc_info=True)
            candidate_path = None
    elif resolve_core_config_path is not None:
        try:
            candidate_path = resolve_core_config_path()
        except Exception:  # pragma: no cover
            logger.debug("Nie udało się uzyskać domyślnej ścieżki konfiguracji core", exc_info=True)

    if candidate_path is None:
        return None

    resolved_path = _normalise_path(candidate_path)
    try:
        core_config = load_core_config(resolved_path)
    except FileNotFoundError:
        logger.debug("Brak pliku konfiguracji core dla market intel: %s", resolved_path)
        return None
    except Exception:  # pragma: no cover
        logger.debug("Nie udało się wczytać konfiguracji core dla market intel", exc_info=True)
        return None

    market_intel_config = getattr(core_config, "market_intel", None)
    if market_intel_config is None:
        return None

    try:
        return MarketIntelAggregator(market_intel_config)  # type: ignore[call-arg]
    except FileNotFoundError:
        sqlite_cfg = getattr(market_intel_config, "sqlite", None)
        logger.debug(
            "Brak źródła market intel wskazanego w konfiguracji: %s",
            getattr(sqlite_cfg, "path", None),
        )
        return None
    except ValueError:
        logger.debug("Sekcja market_intel w konfiguracji jest wyłączona lub niekompletna")
        return None
    except Exception:  # pragma: no cover - optional dependency missing
        logger.debug(
            "Nie udało się zainicjować MarketIntelAggregator z konfiguracji",
            exc_info=True,
        )
        return None


def _build_in_memory_market_intel() -> Any | None:
    if MarketIntelAggregator is None:
        return None

    class _InMemoryCacheStorage:
        def __init__(self) -> None:
            self._payloads: dict[str, dict[str, Any]] = {}

        def read(self, key: str) -> dict[str, Any]:
            default_rows = (
                (0.0, 26_500.0, 120.0),
                (60.0, 26_750.0, 118.0),
                (120.0, 26_900.0, 130.0),
            )
            return self._payloads.get(
                key,
                {
                    "columns": ("open_time", "close", "volume"),
                    "rows": default_rows,
                },
            )

        def write(self, key: str, payload: dict[str, Any]) -> None:
            self._payloads[key] = payload

        def metadata(self) -> dict[str, str]:
            return {}

        def latest_timestamp(self, key: str) -> float | None:  # noqa: ARG002
            return None

    storage = _InMemoryCacheStorage()
    try:
        return MarketIntelAggregator(storage)  # type: ignore[call-arg]
    except Exception:  # pragma: no cover - defensive
        logger.debug("Nie udało się zainicjować MarketIntelAggregator", exc_info=True)
        return None


@lru_cache(maxsize=8)
def bootstrap_market_intel(
    config_path: str | Path | None = None,
    *,
    fallback: str = "memory",
) -> Any | None:
    """Return shared market intel aggregator for desktop applications."""

    aggregator = _load_market_intel_from_config(config_path)
    if aggregator is not None:
        return aggregator

    if fallback == "memory":
        return _build_in_memory_market_intel()
    if fallback == "none":
        return None
    raise ValueError(f"Nieobsługiwany fallback dla market intel: {fallback}")


def _build_exchange_db_url(paths: DesktopAppPaths | None) -> str | None:
    if paths is None:
        return None
    db_file = getattr(paths, "db_file", None)
    if not isinstance(db_file, Path):
        return None
    return f"sqlite+aiosqlite:///{db_file}"


def bootstrap_exchange_manager(
    *,
    paths: DesktopAppPaths | None = None,
    exchange_id: str | None = None,
) -> Any | None:
    """Instantiate ``ExchangeManager`` adjusted to desktop paths."""

    if ExchangeManager is None:
        logger.debug("ExchangeManager nie jest dostępny w tej dystrybucji")
        return None

    kwargs: dict[str, Any] = {}
    if exchange_id:
        kwargs["exchange_id"] = exchange_id
    db_url = _build_exchange_db_url(paths)
    if db_url:
        kwargs["db_url"] = db_url
    try:
        return ExchangeManager(**kwargs)
    except Exception:
        logger.debug("Nie udało się zainicjować ExchangeManager", exc_info=True)
        return None


@dataclass(frozen=True)
class _ExecutionBootstrap:
    execution_service: Any | None
    router: Any | None


class _ExchangeManagerAdapter(ExchangeAdapter if ExchangeAdapter is not None else object):
    """Adapter bridging ``ExchangeManager`` to execution router APIs."""

    def __init__(
        self,
        manager: Any,
        *,
        environment: str | None = None,
        name: str = "primary",
        sandbox: bool | None = None,
        key_id: str | None = None,
    ) -> None:
        if ExchangeAdapter is None or ExchangeCredentials is None:
            raise RuntimeError("ExchangeAdapter nie jest dostępny w tej dystrybucji")
        if sandbox is True and environment is None:
            environment = "testnet"
        env = self._resolve_environment(environment)
        credentials = ExchangeCredentials(
            key_id=key_id or f"{getattr(manager, 'exchange_id', 'exchange')}-{env.value}",
            environment=env,
        )
        super().__init__(credentials)  # type: ignore[misc]
        self._manager = manager
        self.name = name
        self._env = env
        self._order_symbols: dict[str, str] = {}
        self._client_symbols: dict[str, str] = {}

    @staticmethod
    def _resolve_environment(candidate: str | None) -> "ExecutionEnvironment":
        if ExecutionEnvironment is None:  # pragma: no cover
            raise RuntimeError("ExecutionEnvironment nie jest dostępne")
        if not candidate:
            return ExecutionEnvironment.PAPER
        normalised = candidate.lower().strip()
        if normalised in {"live", "prod", "production"}:
            return ExecutionEnvironment.LIVE
        if normalised in {"testnet", "demo"}:
            return ExecutionEnvironment.TESTNET
        return ExecutionEnvironment.PAPER

    def configure_network(
        self,
        *,
        ip_allowlist: Sequence[str] | None = None,  # noqa: ARG002 - compatibility
    ) -> None:
        if ip_allowlist:  # pragma: no cover - diagnostic only
            logger.debug("Zignorowano allowlist dla adaptera ExchangeManager: %s", ip_allowlist)
        return None

    def fetch_account_snapshot(self) -> "AccountSnapshot":
        if AccountSnapshot is None:
            raise RuntimeError("AccountSnapshot nie jest dostępny")
        balance = getattr(self._manager, "fetch_balance", lambda: {})()
        balances: dict[str, float] = {}
        total_equity = 0.0
        available_margin = 0.0
        if isinstance(balance, Mapping):
            total_section = balance.get("total")
            if isinstance(total_section, Mapping):
                for asset, amount in total_section.items():
                    try:
                        balances[str(asset)] = float(amount)
                    except Exception:
                        continue
                total_equity = sum(balances.values())
            free_section = balance.get("free")
            if isinstance(free_section, Mapping):
                try:
                    available_margin = sum(
                        float(value) for value in free_section.values() if isinstance(value, (int, float))
                    )
                except Exception:
                    available_margin = 0.0
        return AccountSnapshot(
            balances=balances,
            total_equity=float(total_equity),
            available_margin=float(available_margin),
            maintenance_margin=0.0,
        )

    def fetch_symbols(self) -> Sequence[str]:
        loader = getattr(self._manager, "load_markets", None)
        if callable(loader):
            try:
                markets = loader() or {}
                return tuple(str(symbol) for symbol in markets.keys())
            except Exception:
                logger.debug("Nie udało się pobrać listy rynków w adapterze ExchangeManager", exc_info=True)
        return ()

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: int | None = None,  # noqa: ARG002 - compatibility
        end: int | None = None,  # noqa: ARG002
        limit: int | None = None,
    ) -> Sequence[Sequence[float]]:
        fetcher = getattr(self._manager, "fetch_ohlcv", None)
        if callable(fetcher):
            candles = fetcher(symbol, timeframe=interval, limit=limit or 500)
            if candles:
                return tuple(tuple(map(float, row)) for row in candles)
        return ()

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): self._json_safe(raw) for key, raw in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def place_order(self, request: "CoreOrderRequest") -> "CoreOrderResult":
        creator = getattr(self._manager, "create_order", None)
        if not callable(creator):
            raise RuntimeError("ExchangeManager nie obsługuje tworzenia zleceń")
        dto = creator(
            request.symbol,
            request.side,
            request.order_type,
            request.quantity,
            request.price,
            request.client_order_id,
        )
        order_id = str(
            getattr(dto, "id", None)
            or getattr(dto, "client_order_id", None)
            or f"{request.symbol}:{getattr(dto, 'ts', 0.0)}"
        )
        self._order_symbols[order_id] = request.symbol
        cid = str(request.client_order_id).strip() if request.client_order_id is not None else ""
        if cid:
            self._client_symbols[cid] = request.symbol
        if CoreOrderResult is None:
            raise RuntimeError("OrderResult nie jest dostępny")
        status = getattr(dto, "status", "unknown")
        status_value = status.value if hasattr(status, "value") else str(status)
        price_value = getattr(dto, "price", None)
        try:
            avg_price = float(price_value) if price_value is not None else None
        except Exception:
            avg_price = None
        try:
            raw_payload = self._json_safe(getattr(dto, "__dict__", {"repr": repr(dto)}))
        except Exception:
            raw_payload = {"repr": repr(dto)}
        return CoreOrderResult(
            order_id=order_id,
            status=status_value,
            filled_quantity=float(getattr(dto, "quantity", request.quantity)),
            avg_price=avg_price,
            raw_response=dict(raw_payload) if isinstance(raw_payload, Mapping) else {"payload": raw_payload},
        )

    def _call_with_optional_symbol(self, fetcher: Any, symbol: str | None, *, include_none: bool = False) -> Any:
        if symbol:
            call_patterns = (
                ((symbol,), {}),
                ((), {"symbol": symbol}),
                ((), {}),
            )
        else:
            call_patterns = (
                ((), {}),
                ((None,), {}),
                ((), {"symbol": None}),
            ) if include_none else (((), {}),)

        last_error: TypeError | None = None
        for args, kwargs in call_patterns:
            try:
                return fetcher(*args, **kwargs)
            except TypeError as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        raise TypeError("No compatible call pattern")

    def _build_reconciled_result(
        self,
        order: Any,
        *,
        cid: str,
        resolved_symbol: str | None,
    ) -> "CoreOrderResult":
        status = getattr(order, "status", "unknown")
        status_value = status.value if hasattr(status, "value") else str(status)
        extra = getattr(order, "extra", {})
        filled_candidate = extra.get("filled_quantity") if isinstance(extra, Mapping) else None
        if filled_candidate is None:
            filled_candidate = getattr(order, "quantity", 0.0)
        avg_candidate = extra.get("avg_price") if isinstance(extra, Mapping) else None
        if avg_candidate is None:
            avg_candidate = getattr(order, "price", 0.0)
        order_symbol = getattr(order, "symbol", None) or resolved_symbol
        timestamp = getattr(order, "ts", 0.0)
        extra_order_id = extra.get("order_id") if isinstance(extra, Mapping) else None
        order_id = str(
            getattr(order, "id", None)
            or getattr(order, "order_id", None)
            or extra_order_id
            or cid
            or f"{order_symbol}:{timestamp}"
        )
        if order_symbol:
            self._order_symbols[order_id] = str(order_symbol)
        if hasattr(order, "model_dump"):
            dumped_order = self._json_safe(order.model_dump())
        elif hasattr(order, "dict"):
            dumped_order = self._json_safe(order.dict())
        else:
            dumped_order = self._json_safe(getattr(order, "__dict__", {"repr": repr(order)}))
        try:
            filled_quantity = float(filled_candidate)
        except (TypeError, ValueError):
            filled_quantity = 0.0
        try:
            avg_price = float(avg_candidate) if avg_candidate is not None else None
        except (TypeError, ValueError):
            avg_price = None
        return CoreOrderResult(
            order_id=order_id,
            status=status_value,
            filled_quantity=filled_quantity,
            avg_price=avg_price,
            raw_response={
                "source": "exchange_manager",
                "reconcile": "client_order_id",
                "order": dumped_order,
            },
        )

    def _find_order_by_client_id(self, orders: Any, cid: str) -> Any | None:
        for order in orders or ():
            order_cid_raw = getattr(order, "client_order_id", None)
            order_cid = str(order_cid_raw).strip() if order_cid_raw is not None else ""
            if order_cid == cid:
                return order
        return None

    def fetch_order_by_client_id(
        self,
        client_order_id: str,
        *,
        symbol: str | None = None,
    ) -> "CoreOrderResult | None":
        cid = str(client_order_id).strip() if client_order_id is not None else ""
        if not cid:
            return None
        if CoreOrderResult is None:
            raise RuntimeError("OrderResult nie jest dostępny")

        resolved_symbol = symbol or self._client_symbols.get(cid)

        direct_fetcher = getattr(self._manager, "fetch_order_by_client_id", None)
        if callable(direct_fetcher):
            order = None
            call_patterns: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
            if resolved_symbol is not None:
                call_patterns.append(((cid, resolved_symbol), {}))
            call_patterns.extend(
                [
                    ((cid,), {"symbol": resolved_symbol}),
                    ((cid,), {}),
                    ((), {"client_order_id": cid, "symbol": resolved_symbol}),
                    ((), {"client_order_id": cid}),
                ]
            )
            for args, kwargs in call_patterns:
                if kwargs.get("symbol") is None:
                    kwargs = {key: value for key, value in kwargs.items() if key != "symbol"}
                try:
                    order = direct_fetcher(*args, **kwargs)
                except TypeError:
                    continue
                break
            if order is not None:
                order_cid_raw = getattr(order, "client_order_id", None)
                order_cid = str(order_cid_raw).strip() if order_cid_raw is not None else ""
                if order_cid == cid:
                    return self._build_reconciled_result(order, cid=cid, resolved_symbol=resolved_symbol)

        for source_name in ("fetch_orders", "fetch_closed_orders", "fetch_open_orders"):
            source = getattr(self._manager, source_name, None)
            if not callable(source):
                continue
            try:
                orders = self._call_with_optional_symbol(source, resolved_symbol, include_none=True)
            except TypeError:
                continue
            order = self._find_order_by_client_id(orders, cid)
            if order is not None:
                return self._build_reconciled_result(order, cid=cid, resolved_symbol=resolved_symbol)
        return None

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:
        resolver = getattr(self._manager, "cancel_order", None)
        if not callable(resolver):
            raise RuntimeError("ExchangeManager nie obsługuje anulowania zleceń")
        resolved_symbol = symbol or self._order_symbols.get(order_id)
        if resolved_symbol is None:
            logger.debug("Brak powiązanego symbolu dla anulacji %s", order_id)
            return
        try:
            resolver(order_id, resolved_symbol)
        except Exception:
            logger.debug("Nie udało się anulować zlecenia %s", order_id, exc_info=True)

    def stream_public_data(self, *, channels: Sequence[str]) -> object:  # noqa: ARG002
        return SimpleNamespace(close=lambda: None)

    def stream_private_data(self, *, channels: Sequence[str]) -> object:  # noqa: ARG002
        return SimpleNamespace(close=lambda: None)


def _bootstrap_execution_runtime(
    manager: Any | None,
    *,
    environment: str | None,
) -> _ExecutionBootstrap:
    if os.getenv("DUDZIAN_TEST_MODE", "").strip().lower() in {"1", "true", "yes", "on"}:
        logger.info("Pomijam LiveExecutionRouter w trybie testowym.")
        return _ExecutionBootstrap(None, None)
    if manager is None or LiveExecutionRouter is None or ExchangeAdapter is None:
        return _ExecutionBootstrap(None, None)

    try:
        adapter = _ExchangeManagerAdapter(manager, environment=environment)
    except Exception:
        logger.debug("Nie udało się zbudować adaptera ExchangeManager", exc_info=True)
        return _ExecutionBootstrap(None, None)

    try:
        router = LiveExecutionRouter(adapters={"primary": adapter}, default_route=("primary",))
    except Exception:
        logger.debug("Nie udało się zainicjować LiveExecutionRouter", exc_info=True)
        return _ExecutionBootstrap(None, None)

    return _ExecutionBootstrap(router, router)


def bootstrap_frontend_services(
    *,
    paths: DesktopAppPaths | None = None,
    config_path: str | Path | None = None,
    exchange_id: str | None = None,
    environment: str | None = None,
    portfolio_id: str = "default",  # noqa: ARG002 - reserved for future use
    risk_profile: str = "baseline",  # noqa: ARG002 - reserved for future use
) -> FrontendBootstrap:
    """Build a shared set of services for GUI/dashboard/AutoTrader."""

    exchange_manager = bootstrap_exchange_manager(paths=paths, exchange_id=exchange_id)
    resolved_config = _resolve_frontend_config_path(paths, config_path)
    market_intel = bootstrap_market_intel(config_path=resolved_config)
    execution = _bootstrap_execution_runtime(
        exchange_manager,
        environment=environment,
    )
    return FrontendBootstrap(
        exchange_manager=exchange_manager,
        market_intel=market_intel,
        execution_service=execution.execution_service,
        router=execution.router,
    )


__all__ = [
    "FrontendBootstrap",
    "bootstrap_exchange_manager",
    "bootstrap_frontend_services",
    "bootstrap_market_intel",
]
