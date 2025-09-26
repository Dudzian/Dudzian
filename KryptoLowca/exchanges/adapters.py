"""Adaptery integrujące ccxt z ExchangeManagerem."""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

try:  # pragma: no cover - importowany tylko jeżeli ccxt istnieje
    import ccxt.async_support as ccxt_async  # type: ignore
except Exception:  # pragma: no cover - fallback dla starszych instalacji
    try:
        import ccxt.asyncio as ccxt_async  # type: ignore
    except Exception:  # pragma: no cover - brak ccxt
        ccxt_async = None  # type: ignore


logger = logging.getLogger(__name__)


class AdapterError(RuntimeError):
    """Wyjątek zgłaszany, gdy nie można utworzyć adaptera giełdy."""


@dataclass(slots=True)
class BaseExchangeAdapter:
    """Minimalny kontrakt adaptera giełdowego."""

    exchange_id: str
    description: str
    sandbox: bool = True
    params: Dict[str, Any] = field(default_factory=dict)

    async def connect(self) -> Any:  # pragma: no cover - interfejs
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover - interfejs
        raise NotImplementedError


@dataclass(slots=True)
class CCXTExchangeAdapter(BaseExchangeAdapter):
    """Adapter budujący klienta ccxt w wariancie asynchronicznym."""

    _client: Optional[Any] = field(default=None, init=False, repr=False)

    async def connect(self) -> Any:
        if ccxt_async is None:  # pragma: no cover - brak biblioteki
            raise AdapterError("Biblioteka ccxt (async_support) nie jest dostępna")
        try:
            exchange_cls = getattr(ccxt_async, self.exchange_id)
        except AttributeError as exc:  # pragma: no cover - nieznana giełda
            raise AdapterError(
                f"Giełda '{self.exchange_id}' nie jest obsługiwana przez ccxt"
            ) from exc

        options = dict(self.params)
        options.setdefault("enableRateLimit", True)
        try:
            client = exchange_cls(options)
        except Exception as exc:
            raise AdapterError(f"Nie udało się utworzyć klienta ccxt: {exc}") from exc

        if self.sandbox and hasattr(client, "set_sandbox_mode"):
            try:
                maybe = client.set_sandbox_mode(True)
                if inspect.isawaitable(maybe):
                    await maybe
            except Exception as exc:  # pragma: no cover - ostrzegamy, ale kontynuujemy
                logger.warning("Nie udało się włączyć trybu sandbox dla %s: %s", self.exchange_id, exc)
        self._client = client
        return client

    async def close(self) -> None:
        if not self._client:
            return
        close_cb = getattr(self._client, "close", None)
        if close_cb is not None:
            try:
                result = close_cb()
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:  # pragma: no cover - logujemy, lecz nie przerywamy
                logger.debug("Błąd podczas zamykania klienta %s: %s", self.exchange_id, exc)
        self._client = None


class ExchangeAdapterFactory:
    """Rejestr umożliwiający dynamiczne tworzenie adapterów giełdowych."""

    _custom_factories: Dict[str, Callable[[Dict[str, Any]], BaseExchangeAdapter]] = {}
    _ccxt_aliases: Dict[str, str] = {
        "binance": "binance",
        "binanceusdm": "binanceusdm",
        "coinbase": "coinbase",
        "coinbasepro": "coinbasepro",
        "okx": "okx",
        "okex": "okx",
        "zonda": "zonda",
        "bitbay": "zonda",
    }

    @classmethod
    def register(cls, name: str, factory: Callable[[Dict[str, Any]], BaseExchangeAdapter]) -> None:
        cls._custom_factories[name.lower()] = factory

    @classmethod
    def create(cls, name: str, **options: Any) -> BaseExchangeAdapter:
        key = name.lower()
        if key in cls._custom_factories:
            return cls._custom_factories[key](options)

        ccxt_id = cls._ccxt_aliases.get(key)
        if ccxt_id:
            params = dict(options)
            sandbox = bool(params.pop("sandbox", params.pop("testnet", True)))
            params.setdefault("apiKey", params.pop("api_key", ""))
            params.setdefault("secret", params.pop("api_secret", ""))
            return CCXTExchangeAdapter(
                exchange_id=ccxt_id,
                description=f"ccxt::{ccxt_id}",
                sandbox=sandbox,
                params=params,
            )
        raise AdapterError(f"Nieobsługiwany adapter giełdy: {name}")


create_exchange_adapter = ExchangeAdapterFactory.create

__all__ = [
    "AdapterError",
    "BaseExchangeAdapter",
    "CCXTExchangeAdapter",
    "ExchangeAdapterFactory",
    "create_exchange_adapter",
]
