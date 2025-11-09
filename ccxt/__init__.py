"""Minimalny stub biblioteki ccxt na potrzeby testów jednostkowych."""
from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Optional


_real_module: ModuleType | None = None
entry: str | None = None
current_origin = Path(getattr(__spec__, "origin", "")) if "__spec__" in globals() else None

for entry in list(sys.path)[1:]:
    try:
        spec = importlib.util.find_spec("ccxt", [entry])
    except (ImportError, AttributeError, ValueError):
        continue
    if (
        spec
        and spec.loader
        and getattr(spec.loader, "exec_module", None)
        and spec.origin
        and (
            current_origin is None
            or Path(spec.origin).resolve(strict=False)
            != current_origin.resolve(strict=False)
        )
    ):
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _real_module = module
        break

if _real_module is not None:
    sys.modules[__name__] = _real_module
    builtins.ccxt = _real_module
    globals().update(_real_module.__dict__)
    __all__ = getattr(_real_module, "__all__", tuple(name for name in globals() if not name.startswith("_")))
else:
    class AuthenticationError(Exception):
        """Zastępczy wyjątek ccxt AuthenticationError."""


    base = SimpleNamespace(errors=SimpleNamespace(AuthenticationError=AuthenticationError))


    class _BaseExchange:
        """Minimalny kontener udostępniający parse_ticker dla testów kontraktowych."""

        def parse_ticker(self, data: Dict[str, Any], symbol: Optional[str] = None) -> Dict[str, Any]:
            bid = self._extract_bid(data)
            ask = self._extract_ask(data)
            last = self._extract_last(data)
            return {
                "symbol": symbol or data.get("symbol"),
                "bid": bid,
                "ask": ask,
                "last": last,
                "info": data,
            }

        def _extract_bid(self, data: Dict[str, Any]) -> float:
            for key in ("bid", "bidPrice", "bidPx", "bid1Price"):
                value = data.get(key)
                if value is not None:
                    return float(value)
            return 0.0

        def _extract_ask(self, data: Dict[str, Any]) -> float:
            for key in ("ask", "askPrice", "askPx", "ask1Price"):
                value = data.get(key)
                if value is not None:
                    return float(value)
            return 0.0

        def _extract_last(self, data: Dict[str, Any]) -> float:
            for key in ("last", "lastPrice", "close", "c"):
                value = data.get(key)
                if isinstance(value, (list, tuple)):
                    value = value[0] if value else None
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        continue
            return 0.0


    class binance(_BaseExchange):
        pass


    class bitstamp(_BaseExchange):
        pass


    class bybit(_BaseExchange):
        pass


    class okx(_BaseExchange):
        pass


    class kraken(_BaseExchange):
        pass


    class zonda(_BaseExchange):
        pass

    __all__ = ["AuthenticationError", "base", "binance", "bitstamp", "bybit", "okx", "kraken", "zonda"]

    # Ułatwienie dla testów odwołujących się do globalnego `ccxt` bez importu.
    builtins.ccxt = sys.modules[__name__]

if entry is not None:
    del entry
del _real_module
