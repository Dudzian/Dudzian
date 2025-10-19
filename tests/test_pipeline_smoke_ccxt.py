"""Smoke testy pipeline'u danych dla nowych giełd CCXT."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from bot_core.config.loader import load_core_config
from bot_core.data import (
    ExchangeDataToolkit,
    OHLCVRequest,
    prepare_exchange_data_toolkit,
)
from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)


@dataclass
class _StaticOrder:
    order_id: str
    status: str
    filled: float


class _StaticAdapter(ExchangeAdapter):
    """Minimalny adapter spot wykorzystany w smoke testach pipeline'u."""

    name = "static_ccxt_adapter"

    def __init__(self, credentials: ExchangeCredentials) -> None:
        super().__init__(credentials)
        self._orders: dict[str, _StaticOrder] = {}

    def configure_network(self, *, ip_allowlist: Sequence[str] | None = None) -> None:  # noqa: D401 - interfejs
        self._ip_allowlist = tuple(ip_allowlist or ())  # type: ignore[attr-defined]

    def fetch_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            balances={"USDT": 10_000.0},
            total_equity=10_000.0,
            available_margin=9_500.0,
            maintenance_margin=500.0,
        )

    def fetch_symbols(self) -> Iterable[str]:
        return ("BTC_USDT", "ETH_USDT")

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: int | None = None,
        end: int | None = None,
        limit: int | None = None,
    ) -> Sequence[Sequence[float]]:
        base = float(start or 0)
        step = 60_000.0
        rows = [
            [base, 100.0, 110.0, 95.0, 105.0, 1_000.0],
            [base + step, 105.0, 115.0, 100.0, 110.0, 1_200.0],
        ]
        if limit is not None:
            return rows[:limit]
        return rows

    def place_order(self, request: OrderRequest) -> OrderResult:
        order = _StaticOrder(order_id=f"{request.symbol}-1", status="open", filled=request.quantity / 2)
        self._orders[order.order_id] = order
        return OrderResult(
            order_id=order.order_id,
            status=order.status,
            filled_quantity=order.filled,
            avg_price=request.price,
            raw_response={"symbol": request.symbol},
        )

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:
        self._orders.pop(order_id, None)

    def stream_public_data(self, *, channels: Sequence[str]):  # noqa: D401 - interfejs
        raise NotImplementedError

    def stream_private_data(self, *, channels: Sequence[str]):  # noqa: D401 - interfejs
        raise NotImplementedError


def test_prepare_exchange_data_toolkit_for_new_ccxt_exchanges(tmp_path: Path) -> None:
    config = load_core_config("config/core.yaml")
    credentials = ExchangeCredentials(key_id="demo", secret="demo", environment=Environment.PAPER)
    adapter = _StaticAdapter(credentials)

    environments = ("coinbase_paper", "okx_paper", "kucoin_paper", "bybit_paper")

    for name in environments:
        environment_cfg = config.environments[name]
        toolkit = prepare_exchange_data_toolkit(
            environment_cfg,
            adapter,
            base_directory=tmp_path / name,
            enable_snapshots=True,
            allow_network_upstream=True,
        )
        assert isinstance(toolkit, ExchangeDataToolkit)
        assert toolkit.cache_directory.exists()
        assert toolkit.manifest_path.exists()
        assert toolkit.namespace

        request = OHLCVRequest(symbol="BTC_USDT", interval="1m", start=0, end=120_000, limit=1)
        response = toolkit.data_source.fetch_ohlcv(request)
        assert response.rows
        assert response.columns
