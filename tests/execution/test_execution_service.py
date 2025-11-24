import types

import pytest

from bot_core.config.models import RuntimeExecutionLiveSettings, RuntimeExecutionSettings
from bot_core.execution.execution_service import build_live_execution_service
from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)


class _DummyAdapter(ExchangeAdapter):
    name = "dummy"

    def __init__(self) -> None:
        super().__init__(ExchangeCredentials(key_id="dummy"))

    def configure_network(self, *, ip_allowlist=None) -> None:  # pragma: no cover - not used in test
        return None

    def fetch_account_snapshot(self) -> AccountSnapshot:  # pragma: no cover - not used in test
        return AccountSnapshot(
            balances={},
            total_equity=0.0,
            available_margin=0.0,
            maintenance_margin=0.0,
        )

    def fetch_symbols(self):  # pragma: no cover - not used in test
        return []

    def fetch_ohlcv(self, symbol, interval, start=None, end=None, limit=None):  # pragma: no cover - not used in test
        return []

    def place_order(self, request: OrderRequest) -> OrderResult:
        return OrderResult(
            order_id="1",
            status="filled",
            filled_quantity=request.quantity,
            avg_price=request.price,
            raw_response={},
        )

    def cancel_order(self, order_id: str, *, symbol=None) -> None:  # pragma: no cover - not used in test
        return None

    def stream_public_data(self, *, channels):  # pragma: no cover - not used in test
        return types.SimpleNamespace()

    def stream_private_data(self, *, channels):  # pragma: no cover - not used in test
        return types.SimpleNamespace()


class _Bootstrap:
    def __init__(self) -> None:
        self.adapter = _DummyAdapter()


class _Environment:
    exchange = "dummy"
    environment = Environment.LIVE


@pytest.mark.parametrize("default_route", [(), ("dummy",)])
def test_build_live_execution_service_imports_and_defaults(default_route):
    settings = RuntimeExecutionSettings(
        live=RuntimeExecutionLiveSettings(enabled=True, default_route=default_route)
    )

    router = build_live_execution_service(
        bootstrap_ctx=_Bootstrap(),
        environment=_Environment(),
        runtime_settings=settings,
    )

    assert router.list_adapters() == ("dummy",)
