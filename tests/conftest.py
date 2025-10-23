"""Globalna konfiguracja testów."""
from __future__ import annotations

import os
import sys
from types import ModuleType
from typing import Iterable

import pytest


if "nacl" not in sys.modules:
    nacl_module = ModuleType("nacl")
    nacl_exceptions = ModuleType("nacl.exceptions")
    nacl_exceptions.BadSignatureError = Exception

    class _VerifyKey:
        def verify(self, *_args: object, **_kwargs: object) -> None:
            return None

    class _SigningKey:
        def __init__(self) -> None:
            self.verify_key = self

        @classmethod
        def generate(cls) -> "_SigningKey":
            return cls()

        def sign(self, payload: bytes) -> "_SignatureWrapper":
            return _SignatureWrapper(b"stub-signature" + payload[:1])

        def encode(self) -> bytes:
            return b"stub-signing-key"

    class _SignatureWrapper:
        def __init__(self, signature: bytes) -> None:
            self.signature = signature

    nacl_signing = ModuleType("nacl.signing")
    nacl_signing.VerifyKey = lambda *_args, **_kwargs: _VerifyKey()
    nacl_signing.SigningKey = _SigningKey
    nacl_module.exceptions = nacl_exceptions
    nacl_module.signing = nacl_signing
    sys.modules["nacl"] = nacl_module
    sys.modules["nacl.exceptions"] = nacl_exceptions
    sys.modules["nacl.signing"] = nacl_signing

# Import modułu zapewniającego, że katalog repozytorium znajduje się na sys.path.
# Dzięki temu wszystkie testy mogą importować kod projektu niezależnie od miejsca uruchomienia.
import tests._pathbootstrap  # noqa: F401  # pylint: disable=unused-import


_FAST_ENV_VAR = "PYTEST_FAST"
_FAST_MODE_ENABLED = False


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Włącza tryb fast – testy integracyjne zewnętrznych usług są mockowane lub pomijane.",
    )


def _interval_to_milliseconds(interval: str | None) -> float:
    if not interval:
        return 60_000.0
    units = {"m": 60_000.0, "h": 3_600_000.0, "d": 86_400_000.0}
    try:
        value = int(interval[:-1])
        unit = interval[-1].lower()
    except (TypeError, ValueError, AttributeError):
        return 60_000.0
    base = units.get(unit, 60_000.0)
    return max(float(value) * base, 60_000.0)


def _enable_fast_mode() -> None:
    from bot_core.data.base import OHLCVResponse
    from bot_core.data.ohlcv.cache import PublicAPIDataSource

    columns = ("open_time", "open", "high", "low", "close", "volume")

    def _mock_fetch(
        self: PublicAPIDataSource,
        request,
    ) -> OHLCVResponse:  # pragma: no cover - deterministyczny stub do testów
        start = float(getattr(request, "start", 0.0) or 1_700_000_000_000)
        end = getattr(request, "end", None)
        limit = getattr(request, "limit", None)
        interval = getattr(request, "interval", "1m")
        step = _interval_to_milliseconds(interval)

        if isinstance(limit, int) and limit > 0:
            size = limit
        else:
            if end is not None:
                try:
                    size = int(max((float(end) - start) / step, 0.0)) + 1
                except (TypeError, ValueError):
                    size = 120
            else:
                size = 120
        size = max(size, 1)

        rows = []
        for index in range(size):
            timestamp = start + index * step
            base_price = 100.0 + (index % 20) * 0.5
            high = base_price * 1.01
            low = base_price * 0.99
            close = base_price * 1.005
            volume = 10.0 + index
            rows.append(
                (
                    float(timestamp),
                    float(base_price),
                    float(high),
                    float(low),
                    float(close),
                    float(volume),
                )
            )

        return OHLCVResponse(columns=columns, rows=tuple(rows))

    def _mock_warm_cache(
        self: PublicAPIDataSource, symbols: Iterable[str], intervals: Iterable[str]
    ) -> None:  # pragma: no cover - stub
        del self, symbols, intervals

    PublicAPIDataSource.fetch_ohlcv = _mock_fetch  # type: ignore[assignment]
    PublicAPIDataSource.warm_cache = _mock_warm_cache  # type: ignore[assignment]


def pytest_configure(config: pytest.Config) -> None:
    fast_from_cli = bool(config.getoption("--fast"))
    fast_from_env = os.getenv(_FAST_ENV_VAR, "").lower() in {"1", "true", "yes", "on"}
    fast_mode = fast_from_cli or fast_from_env
    config.fast_mode = fast_mode  # type: ignore[attr-defined]

    if fast_mode:
        global _FAST_MODE_ENABLED
        if not _FAST_MODE_ENABLED:
            _enable_fast_mode()
            _FAST_MODE_ENABLED = True
        os.environ.setdefault(_FAST_ENV_VAR, "1")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not getattr(config, "fast_mode", False):  # type: ignore[attr-defined]
        return
    skip_marker = pytest.mark.skip(reason="pomijam test integracyjny w trybie fast")
    for item in items:
        if "integration" in item.keywords or "external" in item.keywords:
            item.add_marker(skip_marker)

