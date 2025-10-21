"""Wspólne funkcje bootstrapu dla aplikacji korzystających z ExchangeManagera."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from KryptoLowca.logging_utils import get_logger

try:  # pragma: no cover - zależność opcjonalna
    from bot_core.market_intel import MarketIntelAggregator  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - fallback gdy moduł nie jest dostępny
    MarketIntelAggregator = None  # type: ignore[assignment]

try:  # pragma: no cover - podstawowy ExchangeManager może nie być obecny w każdej dystrybucji
    from KryptoLowca.managers.exchange_manager import ExchangeManager
except Exception:  # pragma: no cover - środowiska testowe bez modułu bot_core
    ExchangeManager = None  # type: ignore[assignment]

try:  # pragma: no cover - helper path może być niedostępny
    from bot_core.runtime.paths import DesktopAppPaths, resolve_core_config_path
except Exception:  # pragma: no cover - fallback gdy moduł paths nie istnieje
    DesktopAppPaths = None  # type: ignore[assignment]
    resolve_core_config_path = None  # type: ignore[assignment]

try:  # pragma: no cover - ładowanie konfiguracji może być opcjonalne
    from bot_core.config.loader import load_core_config
except Exception:  # pragma: no cover - fallback gdy loader nie jest dostępny
    load_core_config = None  # type: ignore[assignment]


try:  # pragma: no cover - opcjonalne moduły egzekucji
    from bot_core.execution.live_router import LiveExecutionRouter
    from bot_core.execution.base import ExecutionContext as CoreExecutionContext
    from bot_core.exchanges.base import (
        AccountSnapshot,
        Environment as ExecutionEnvironment,
        ExchangeAdapter,
        ExchangeCredentials,
        OrderRequest as CoreOrderRequest,
        OrderResult as CoreOrderResult,
    )
except Exception:  # pragma: no cover - fallback gdy moduły nie są dostępne
    LiveExecutionRouter = None  # type: ignore[assignment]
    CoreExecutionContext = None  # type: ignore[assignment]
    ExchangeAdapter = None  # type: ignore[assignment]
    ExecutionEnvironment = None  # type: ignore[assignment]
    ExchangeCredentials = None  # type: ignore[assignment]
    CoreOrderRequest = None  # type: ignore[assignment]
    CoreOrderResult = None  # type: ignore[assignment]

try:  # pragma: no cover - menedżer wielu kont jest opcjonalny
    from KryptoLowca.managers.multi_account_manager import MultiExchangeAccountManager
except Exception:  # pragma: no cover - brak modułu w środowiskach uproszczonych
    MultiExchangeAccountManager = None  # type: ignore[assignment]


logger = get_logger(__name__)


@dataclass(frozen=True)
class FrontendBootstrap:
    """Zbiorczy rezultat bootstrapu usług frontowych."""

    exchange_manager: Any | None
    market_intel: Any | None
    execution_service: Any | None = None
    account_manager: Any | None = None
    router: Any | None = None


def _normalise_config_path(candidate: Path) -> Path:
    expanded = candidate.expanduser()
    try:
        return expanded.resolve(strict=False)
    except Exception:  # pragma: no cover - środowisko bez pełnego FS
        return expanded


def _resolve_frontend_config_path(
    paths: DesktopAppPaths | None,
    explicit: str | Path | None,
) -> Path | None:
    candidates: list[Path] = []
    if explicit is not None:
        try:
            candidates.append(Path(explicit))
        except Exception:  # pragma: no cover - niepoprawna ścieżka
            logger.debug("Niepoprawna jawna ścieżka konfiguracji frontendu", exc_info=True)
    if paths is not None:
        app_root = getattr(paths, "app_root", None)
        if isinstance(app_root, Path):
            candidates.append(app_root / "config" / "core.yaml")
            candidates.append(app_root.parent / "config" / "core.yaml")
    if resolve_core_config_path is not None:
        try:
            candidates.append(resolve_core_config_path())
        except Exception:  # pragma: no cover - środowisko bez resolvera
            logger.debug("Nie udało się ustalić ścieżki konfiguracji z resolvera", exc_info=True)

    normalised: list[Path] = []
    for candidate in candidates:
        try:
            normalised.append(_normalise_config_path(Path(candidate)))
        except Exception:  # pragma: no cover - niepoprawna ścieżka
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
        except Exception:  # pragma: no cover - niepoprawna ścieżka
            logger.debug("Nie udało się sparsować ścieżki konfiguracji market intel", exc_info=True)
            candidate_path = None
    elif resolve_core_config_path is not None:
        try:
            candidate_path = resolve_core_config_path()
        except Exception:  # pragma: no cover - środowisko bez resolvera
            logger.debug("Nie udało się uzyskać domyślnej ścieżki konfiguracji core", exc_info=True)

    if candidate_path is None:
        return None

    resolved_path = _normalise_config_path(candidate_path)
    try:
        core_config = load_core_config(resolved_path)
    except FileNotFoundError:
        logger.debug("Brak pliku konfiguracji core dla market intel: %s", resolved_path)
        return None
    except Exception:  # pragma: no cover - błędy loadera
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
    except Exception:  # pragma: no cover - zależność opcjonalna
        logger.debug(
            "Nie udało się zainicjować MarketIntelAggregator z konfiguracji",
            exc_info=True,
        )
        return None


def _build_in_memory_market_intel() -> Any | None:
    """Tworzy lekki agregator market-intel oparty na pamięci."""

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
            return 120.0

    storage = _InMemoryCacheStorage()
    try:
        return MarketIntelAggregator(storage)  # type: ignore[call-arg]
    except Exception:  # pragma: no cover - defensywne logowanie
        logger.debug("Nie udało się zainicjować MarketIntelAggregator", exc_info=True)
        return None


@lru_cache(maxsize=8)
def bootstrap_market_intel(
    config_path: str | Path | None = None,
    *,
    fallback: str = "memory",
) -> Any | None:
    """Zwraca współdzielony agregator market intel używany w aplikacjach desktopowych."""

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
    """Tworzy instancję ``ExchangeManager`` dopasowaną do ścieżek aplikacji."""

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
    account_manager: Any | None
    router: Any | None


class _ExchangeManagerAdapter(ExchangeAdapter if ExchangeAdapter is not None else object):
    """Adapter LiveExecutionRouter oparty o :class:`ExchangeManager`."""

    def __init__(
        self,
        manager: Any,
        *,
        environment: str | None = None,
        name: str = "primary",
    ) -> None:
        if ExchangeAdapter is None or ExchangeCredentials is None:
            raise RuntimeError("ExchangeAdapter nie jest dostępny w tej dystrybucji")
        env = self._resolve_environment(environment)
        credentials = ExchangeCredentials(
            key_id=f"{getattr(manager, 'exchange_id', 'exchange')}-{env.value}",
            environment=env,
        )
        super().__init__(credentials)  # type: ignore[misc]
        self._manager = manager
        self.name = name
        self._env = env
        self._order_symbols: dict[str, str] = {}

    @staticmethod
    def _resolve_environment(candidate: str | None) -> "ExecutionEnvironment":
        if ExecutionEnvironment is None:  # pragma: no cover - brak modułu egzekucji
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
        ip_allowlist: Sequence[str] | None = None,  # noqa: ARG002 - zgodność interfejsu
    ) -> None:
        # ExchangeManager nie wymaga dodatkowej konfiguracji sieciowej z perspektywy adaptera.
        if ip_allowlist:  # pragma: no cover - wyłącznie diagnostyka środowisk spec.
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
        start: int | None = None,  # noqa: ARG002 - zgodność podpisu
        end: int | None = None,  # noqa: ARG002
        limit: int | None = None,
    ) -> Sequence[Sequence[float]]:
        fetcher = getattr(self._manager, "fetch_ohlcv", None)
        if callable(fetcher):
            candles = fetcher(symbol, timeframe=interval, limit=limit or 500)
            if candles:
                return tuple(tuple(map(float, row)) for row in candles)
        return ()

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
        if CoreOrderResult is None:
            raise RuntimeError("OrderResult nie jest dostępny")
        status = getattr(dto, "status", "unknown")
        status_value = status.value if hasattr(status, "value") else str(status)
        price_value = getattr(dto, "price", None)
        try:
            avg_price = float(price_value) if price_value is not None else None
        except Exception:
            avg_price = None
        raw_payload: Mapping[str, Any] | dict[str, Any]
        try:
            raw_payload = dto.model_dump()  # type: ignore[attr-defined]
        except Exception:
            raw_payload = getattr(dto, "__dict__", {}) or {"repr": repr(dto)}
        return CoreOrderResult(
            order_id=order_id,
            status=status_value,
            filled_quantity=float(getattr(dto, "quantity", request.quantity)),
            avg_price=avg_price,
            raw_response=dict(raw_payload),
        )

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
    market_intel: Any | None,
    environment: str | None,
    portfolio_id: str,
    risk_profile: str,
) -> _ExecutionBootstrap:
    if (
        manager is None
        or LiveExecutionRouter is None
        or CoreExecutionContext is None
        or MultiExchangeAccountManager is None
        or ExchangeAdapter is None
        or CoreOrderRequest is None
    ):
        return _ExecutionBootstrap(None, None, None)

    try:
        adapter = _ExchangeManagerAdapter(manager, environment=environment)
    except Exception:
        logger.debug("Nie udało się zbudować adaptera ExchangeManager", exc_info=True)
        return _ExecutionBootstrap(None, None, None)

    try:
        router = LiveExecutionRouter(adapters={"primary": adapter}, default_route=("primary",))
    except Exception:
        logger.debug("Nie udało się zainicjować LiveExecutionRouter", exc_info=True)
        return _ExecutionBootstrap(None, None, None)

    base_context = CoreExecutionContext(
        portfolio_id=str(portfolio_id or "default"),
        risk_profile=str(risk_profile or "baseline"),
        environment=(environment or "paper"),
        metadata={"exchange": getattr(manager, "exchange_id", "exchange")},
    )

    try:
        account_manager = MultiExchangeAccountManager(
            router,
            base_context=base_context,
            market_intel=market_intel,
        )
        account_manager.register_account(
            exchange=str(getattr(manager, "exchange_id", "exchange")),
            account="primary",
            execution_route="primary",
            environment=environment,
            metadata={"source": "frontend-bootstrap"},
        )
    except Exception:
        logger.debug("Nie udało się zainicjować MultiExchangeAccountManager", exc_info=True)
        return _ExecutionBootstrap(None, None, router)

    try:
        from KryptoLowca.core.services.execution_service import ExecutionService
    except Exception:  # pragma: no cover - brak modułu w środowisku testowym
        logger.debug("ExecutionService nie jest dostępny – pomijam bootstrap egzekucji", exc_info=True)
        return _ExecutionBootstrap(None, account_manager, router)

    service = ExecutionService(
        None,
        router=router,
        account_manager=account_manager,
    )
    return _ExecutionBootstrap(service, account_manager, router)


def bootstrap_frontend_services(
    *,
    paths: DesktopAppPaths | None = None,
    config_path: str | Path | None = None,
    exchange_id: str | None = None,
    environment: str | None = None,
    portfolio_id: str = "default",
    risk_profile: str = "baseline",
) -> FrontendBootstrap:
    """Buduje wspólny zestaw usług wymaganych przez GUI/dashboard/AutoTradera."""

    exchange_manager = bootstrap_exchange_manager(paths=paths, exchange_id=exchange_id)
    resolved_config = _resolve_frontend_config_path(paths, config_path)
    market_intel = bootstrap_market_intel(config_path=resolved_config)
    execution = _bootstrap_execution_runtime(
        exchange_manager,
        market_intel=market_intel,
        environment=environment,
        portfolio_id=portfolio_id,
        risk_profile=risk_profile,
    )
    return FrontendBootstrap(
        exchange_manager=exchange_manager,
        market_intel=market_intel,
        execution_service=execution.execution_service,
        account_manager=execution.account_manager,
        router=execution.router,
    )


__all__ = [
    "FrontendBootstrap",
    "bootstrap_exchange_manager",
    "bootstrap_frontend_services",
    "bootstrap_market_intel",
]

