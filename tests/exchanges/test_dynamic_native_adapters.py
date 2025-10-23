from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pytest
import yaml

from bot_core.config.loader import load_core_config
from bot_core.exchanges import manager as manager_module
from bot_core.exchanges.base import AccountSnapshot, ExchangeAdapter, OrderRequest, OrderResult
from bot_core.exchanges.core import MarketRules, Mode
from bot_core.exchanges.manager import ExchangeManager


@pytest.fixture
def reset_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manager_module, "_NATIVE_ADAPTER_REGISTRY", {})
    monkeypatch.setattr(manager_module, "_DYNAMIC_ADAPTERS_INITIALIZED", False)
    monkeypatch.setattr(manager_module, "_DYNAMIC_ADAPTERS_SOURCE", None)
    monkeypatch.setattr(manager_module, "_DYNAMIC_ADAPTER_KEYS", set())


@dataclass
class FakeNativeOrder:
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: float | None
    status: str = "OPEN"
    client_order_id: str | None = None
    orig_quantity: float = field(init=False)

    def __post_init__(self) -> None:
        self.orig_quantity = self.quantity


@dataclass
class FakeNativePosition:
    symbol: str
    quantity: float
    entry_price: float
    side: str = "LONG"
    unrealized_pnl: float = 0.0


class FakeLongPoll:
    def __init__(self, kind: str, channels: Sequence[str]):
        self.kind = kind
        self.channels = tuple(channels)
        self._cursor = 0

    def poll(self) -> Mapping[str, Any]:
        if not self.channels:
            return {"kind": self.kind, "channel": None, "sequence": self._cursor}
        channel = self.channels[self._cursor % len(self.channels)]
        self._cursor += 1
        return {"kind": self.kind, "channel": channel, "sequence": self._cursor}


class FakeNativeAdapter(ExchangeAdapter):
    name = "fake-native"

    def __init__(
        self,
        factory_path: str,
        credentials,
        *,
        environment,
        settings: Mapping[str, Any] | None = None,
        watchdog: Any | None = None,
    ) -> None:
        super().__init__(credentials)
        self.factory_path = factory_path
        self.environment = environment
        self.settings = dict(settings or {})
        self.watchdog = watchdog
        self.calls: list[tuple[str, Any]] = []
        self._orders: list[FakeNativeOrder] = []
        self._positions: list[FakeNativePosition] = [
            FakeNativePosition(symbol="BTC_USDT", quantity=0.75, entry_price=20_050.0)
        ]

    def configure_network(self, *, ip_allowlist: Sequence[str] | None = None) -> None:
        self.calls.append(("configure_network", tuple(ip_allowlist or ())))

    def fetch_account_snapshot(self) -> AccountSnapshot:
        self.calls.append(("fetch_account_snapshot",))
        return AccountSnapshot(
            balances={"USDT": 100_000.0, "BTC": 1.0},
            total_equity=100_000.0,
            available_margin=80_000.0,
            maintenance_margin=5_000.0,
        )

    def fetch_symbols(self) -> Iterable[str]:
        self.calls.append(("fetch_symbols",))
        return ["BTC_USDT", "ETH_USDT"]

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: float | None = None,
        end: float | None = None,
        limit: int | None = None,
    ) -> Sequence[Sequence[float]]:
        self.calls.append(("fetch_ohlcv", symbol, interval, limit))
        return [[0.0, 20_000.0, 20_200.0, 19_950.0, 20_050.0, 12.0]]

    def place_order(self, request: OrderRequest) -> OrderResult:
        self.calls.append(("place_order", request.symbol, request.order_type))
        client_id = request.client_order_id or f"{self.credentials.key_id}-{len(self._orders) + 1}"
        order_id = f"{self.factory_path.split(':')[-1]}-{len(self._orders) + 1}"
        order = FakeNativeOrder(
            order_id=order_id,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            price=request.price,
            client_order_id=client_id,
        )
        self._orders.append(order)
        return OrderResult(
            order_id=order_id,
            status="OPEN",
            filled_quantity=0.0,
            avg_price=None,
            raw_response={"clientOrderId": client_id},
        )

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:
        self.calls.append(("cancel_order", order_id, symbol))
        for order in self._orders:
            if order.order_id == order_id or order.client_order_id == order_id:
                order.status = "CANCELED"

    def stream_public_data(self, *, channels: Sequence[str]):
        self.calls.append(("stream_public", tuple(channels)))
        return FakeLongPoll("public", channels)

    def stream_private_data(self, *, channels: Sequence[str]):
        self.calls.append(("stream_private", tuple(channels)))
        return FakeLongPoll("private", channels)

    def fetch_open_orders(self) -> Sequence[FakeNativeOrder]:
        self.calls.append(("fetch_open_orders",))
        return [order for order in self._orders if order.status == "OPEN"]

    def fetch_positions(self) -> Sequence[FakeNativePosition]:
        self.calls.append(("fetch_positions",))
        return list(self._positions)


class FakePublicFeed:
    def __init__(
        self,
        exchange_id: str = "binance",
        testnet: bool = False,
        futures: bool = False,
        *,
        market_type: str | None = None,
        error_handler: Any | None = None,
    ) -> None:
        self.exchange_id = exchange_id
        self.testnet = testnet
        self.futures = futures
        self.market_type = market_type or ("future" if futures else "spot")
        self._error_handler = error_handler
        self._markets = {
            "BTC_USDT": MarketRules(
                symbol="BTC_USDT",
                price_step=0.1,
                amount_step=0.001,
                min_notional=10.0,
                min_amount=0.001,
                max_amount=10.0,
                min_price=100.0,
                max_price=100_000.0,
            ),
            "ETH_USDT": MarketRules(
                symbol="ETH_USDT",
                price_step=0.05,
                amount_step=0.01,
                min_notional=5.0,
                min_amount=0.01,
                max_amount=500.0,
                min_price=10.0,
                max_price=20_000.0,
            ),
        }

    def load_markets(self) -> Mapping[str, MarketRules]:
        return dict(self._markets)

    def fetch_ticker(self, symbol: str) -> Mapping[str, float]:
        return {"symbol": symbol, "last": 20_050.0, "close": 20_040.0, "bid": 20_030.0, "ask": 20_060.0}

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> Sequence[Sequence[float]]:
        return [[0.0, 20_000.0, 20_200.0, 19_950.0, 20_050.0, 12.0]]

    def fetch_order_book(self, symbol: str, limit: int = 50) -> Mapping[str, Sequence[Sequence[float]]]:
        return {
            "bids": [[20_030.0, 1.5], [20_020.0, 1.0]],
            "asks": [[20_060.0, 1.2], [20_070.0, 0.9]],
        }

    def get_market_rules(self, symbol: str) -> MarketRules | None:
        return self._markets.get(symbol)


@pytest.fixture
def fake_native_factories(monkeypatch: pytest.MonkeyPatch) -> list[FakeNativeAdapter]:
    created: list[FakeNativeAdapter] = []

    def fake_import(path: str):
        def factory(credentials, **kwargs):
            adapter = FakeNativeAdapter(path, credentials, **kwargs)
            created.append(adapter)
            return adapter

        return factory

    monkeypatch.setattr(manager_module, "_import_adapter_factory", fake_import)
    return created


@pytest.fixture
def fake_public_feed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manager_module, "_CCXTPublicFeed", FakePublicFeed)


def test_dynamic_adapters_loaded_from_config(reset_registry: None) -> None:
    manager_module._load_dynamic_native_adapters()

    expected_pairs = {
        (Mode.MARGIN, "binance"),
        (Mode.FUTURES, "binance"),
        (Mode.MARGIN, "kraken"),
        (Mode.FUTURES, "kraken"),
        (Mode.MARGIN, "zonda"),
        (Mode.MARGIN, "coinbase"),
        (Mode.FUTURES, "coinbase"),
        (Mode.MARGIN, "okx"),
        (Mode.FUTURES, "okx"),
        (Mode.MARGIN, "bybit"),
        (Mode.FUTURES, "bybit"),
    }

    registry = manager_module._NATIVE_ADAPTER_REGISTRY
    assert manager_module._DYNAMIC_ADAPTERS_SOURCE is not None
    assert expected_pairs.issubset(registry)

    coinbase_margin = registry[(Mode.MARGIN, "coinbase")]
    assert coinbase_margin.supports_testnet is False
    assert coinbase_margin.default_settings["product_type"] == "margin"

    okx_margin = registry[(Mode.MARGIN, "okx")]
    assert okx_margin.default_settings["marginMode"] == "cross"

    bybit_futures = registry[(Mode.FUTURES, "bybit")]
    assert bybit_futures.default_settings["hedgeMode"] is True

    zonda_margin = registry[(Mode.MARGIN, "zonda")]
    assert zonda_margin.supports_testnet is False


def test_iter_registered_native_adapters_exposes_metadata(reset_registry: None) -> None:
    manager_module._load_dynamic_native_adapters()

    infos = list(manager_module.iter_registered_native_adapters())
    assert len(infos) >= 10

    coinbase_margin = next(
        info
        for info in infos
        if info.exchange_id == "coinbase" and info.mode is Mode.MARGIN
    )
    assert coinbase_margin.supports_testnet is False
    assert coinbase_margin.dynamic is True
    assert coinbase_margin.source == manager_module._DYNAMIC_ADAPTERS_SOURCE
    assert coinbase_margin.default_settings["product_type"] == "margin"

    futures_only = list(manager_module.iter_registered_native_adapters(mode=Mode.FUTURES))
    assert futures_only
    assert all(info.mode is Mode.FUTURES for info in futures_only)


def test_iter_registered_native_adapters_accepts_string_filter(reset_registry: None) -> None:
    manager_module._load_dynamic_native_adapters()

    infos = list(manager_module.iter_registered_native_adapters(mode="margin"))
    assert infos
    assert all(info.mode is Mode.MARGIN for info in infos)


def test_get_native_adapter_info_returns_metadata(reset_registry: None) -> None:
    manager_module._load_dynamic_native_adapters()

    info = manager_module.get_native_adapter_info(exchange_id="coinbase", mode=Mode.MARGIN)
    assert info is not None
    assert info.exchange_id == "coinbase"
    assert info.supports_testnet is False
    assert info.dynamic is True

    missing = manager_module.get_native_adapter_info(exchange_id="unknown", mode=Mode.MARGIN)
    assert missing is None

    with pytest.raises(ValueError):
        manager_module.get_native_adapter_info(exchange_id="", mode=Mode.MARGIN)


def test_register_native_adapter_accepts_string_mode(reset_registry: None) -> None:
    sentinel_factory = object()

    manager_module.register_native_adapter(
        exchange_id="custom", mode="margin", factory=sentinel_factory
    )

    info = manager_module.get_native_adapter_info(exchange_id="custom", mode=Mode.MARGIN)
    assert info is not None
    assert info.factory is sentinel_factory


def test_register_native_adapter_rejects_invalid_mode(reset_registry: None) -> None:
    with pytest.raises(ValueError):
        manager_module.register_native_adapter(exchange_id="custom", mode="spot", factory=object())


def test_unregister_native_adapter_protects_dynamic_by_default(reset_registry: None) -> None:
    manager_module._load_dynamic_native_adapters()

    removed = manager_module.unregister_native_adapter(exchange_id="binance", mode=Mode.MARGIN)
    assert removed is False

    info = manager_module.get_native_adapter_info(exchange_id="binance", mode=Mode.MARGIN)
    assert info is not None and info.dynamic is True


def test_unregister_native_adapter_allows_manual_cleanup(reset_registry: None) -> None:
    manager_module._load_dynamic_native_adapters()

    def manual_factory(credentials, **kwargs):  # noqa: D401 - lokalna atrapa
        return FakeNativeAdapter("manual", credentials, **kwargs)

    manager_module.register_native_adapter(
        exchange_id="coinbase",
        mode=Mode.MARGIN,
        factory=manual_factory,
    )

    removed = manager_module.unregister_native_adapter(exchange_id="coinbase", mode=Mode.MARGIN)
    assert removed is True

    assert manager_module.get_native_adapter_info(exchange_id="coinbase", mode=Mode.MARGIN) is None

    manager_module.reload_native_adapters()

    info = manager_module.get_native_adapter_info(exchange_id="coinbase", mode=Mode.MARGIN)
    assert info is not None and info.dynamic is True


def test_unregister_native_adapter_can_force_dynamic(reset_registry: None) -> None:
    manager_module._load_dynamic_native_adapters()

    removed = manager_module.unregister_native_adapter(
        exchange_id="okx",
        mode=Mode.FUTURES,
        allow_dynamic=True,
    )
    assert removed is True
    assert manager_module.get_native_adapter_info(exchange_id="okx", mode=Mode.FUTURES) is None

    manager_module.reload_native_adapters()
    restored = manager_module.get_native_adapter_info(exchange_id="okx", mode=Mode.FUTURES)
    assert restored is not None and restored.dynamic is True


@pytest.mark.parametrize(
    "exchange_id,mode",
    [
        ("binance", Mode.MARGIN),
        ("binance", Mode.FUTURES),
        ("kraken", Mode.MARGIN),
        ("kraken", Mode.FUTURES),
        ("zonda", Mode.MARGIN),
        ("coinbase", Mode.MARGIN),
        ("coinbase", Mode.FUTURES),
        ("okx", Mode.MARGIN),
        ("okx", Mode.FUTURES),
        ("bybit", Mode.MARGIN),
        ("bybit", Mode.FUTURES),
    ],
)
def test_exchange_manager_integration_for_configured_adapters(
    exchange_id: str,
    mode: Mode,
    reset_registry: None,
    fake_native_factories: list[FakeNativeAdapter],
    fake_public_feed: None,
) -> None:
    manager = ExchangeManager(exchange_id=exchange_id)
    manager.set_mode(
        paper=False,
        spot=False,
        margin=mode is Mode.MARGIN,
        futures=mode is Mode.FUTURES,
    )
    manager.set_credentials("api-key", "secret-key")

    markets = manager.load_markets()
    assert "BTC_USDT" in markets

    batch = manager.fetch_batch(["BTC_USDT"], timeframe="1m", use_orderbook=True)
    assert batch and batch[0][0] == "BTC_USDT"

    balance = manager.fetch_balance()
    assert balance["total_equity"] == pytest.approx(100_000.0)

    order = manager.create_order("BTC_USDT", "BUY", "LIMIT", 1.0, price=20_050.0, client_order_id="client-1")
    assert order.client_order_id == "client-1"

    open_orders = manager.fetch_open_orders()
    assert len(open_orders) == 1
    assert open_orders[0].symbol == "BTC_USDT"

    manager.cancel_order(order.extra["order_id"], symbol="BTC_USDT")
    assert manager.fetch_open_orders() == []

    adapter = manager._ensure_native_adapter()
    assert isinstance(adapter, FakeNativeAdapter)
    private_stream = adapter.stream_private_data(channels=("orders", "fills"))
    event = private_stream.poll()
    assert event["kind"] == "private"
    assert event["channel"] in {"orders", "fills"}

    if mode is Mode.FUTURES:
        positions = manager.fetch_positions()
        assert positions and positions[0].mode is Mode.FUTURES

    assert any(call[0] == "place_order" for call in adapter.calls)
    assert any(call[0] == "fetch_account_snapshot" for call in adapter.calls)


def test_exchange_accounts_cover_multi_account_portfolio() -> None:
    config = load_core_config(Path("config/core.yaml"))

    accounts = config.exchange_accounts
    assert {"binance", "kraken", "zonda", "coinbase", "okx", "bybit"} <= set(accounts)

    profile_to_accounts: dict[str, set[tuple[str, str]]] = {}
    for exchange_id, entries in accounts.items():
        for env_name, entry in entries.items():
            profile_to_accounts.setdefault(entry.risk_profile, set()).add((exchange_id, env_name))

    assert len(profile_to_accounts.get("balanced", set())) >= 3
    assert len(profile_to_accounts.get("conservative", set())) >= 2

    governor = config.portfolio_governors["stage6_core"]
    assignments: dict[str, tuple[str, str]] = {}
    used_accounts: set[tuple[str, str]] = set()

    for asset in governor.assets:
        candidates = list(profile_to_accounts.get(asset.risk_budget, ()))
        if not candidates:
            candidates = list(profile_to_accounts.get("balanced", ()))
        assert candidates, f"Brak skonfigurowanych kont dla profilu {asset.risk_budget}"

        for candidate in sorted(candidates):
            if candidate not in used_accounts:
                assignments[asset.symbol] = candidate
                used_accounts.add(candidate)
                break
        else:  # pragma: no cover - defensywne zabezpieczenie testu
            pytest.fail(f"Nie udało się przypisać unikalnego konta dla {asset.symbol}")

    assert len(assignments) == len(governor.assets)
    assert len({exchange for exchange, _ in assignments.values()}) >= 2


def test_reload_native_adapters_replaces_registry(
    tmp_path: Path,
    reset_registry: None,
    fake_native_factories: list[FakeNativeAdapter],
) -> None:
    manager_module._load_dynamic_native_adapters()
    assert (Mode.MARGIN, "binance") in manager_module._NATIVE_ADAPTER_REGISTRY

    raw_config = yaml.safe_load(Path("config/core.yaml").read_text(encoding="utf-8"))
    raw_config["exchange_adapters"] = {
        "kraken": {
            "margin": {
                "class_path": "tests.fake:KrakenMarginAdapter",
                "supports_testnet": True,
                "default_settings": {"product_type": "margin"},
            }
        }
    }

    custom_config = tmp_path / "core_alt.yaml"
    custom_config.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    manager_module.reload_native_adapters(custom_config)

    registry = manager_module._NATIVE_ADAPTER_REGISTRY
    assert set(registry) == {(Mode.MARGIN, "kraken")}
    assert manager_module._DYNAMIC_ADAPTERS_SOURCE == custom_config

    infos = list(manager_module.iter_registered_native_adapters())
    assert infos and infos[0].exchange_id == "kraken"
    assert infos[0].dynamic is True
    assert infos[0].source == custom_config


def test_reload_preserves_manual_override(reset_registry: None) -> None:
    manager_module._load_dynamic_native_adapters()

    def manual_factory(credentials, **kwargs):  # noqa: D401 - lokalna atrapa
        return FakeNativeAdapter("manual", credentials, **kwargs)

    manager_module.register_native_adapter(
        exchange_id="coinbase",
        mode=Mode.MARGIN,
        factory=manual_factory,
    )

    manager_module.reload_native_adapters()

    registration = manager_module._NATIVE_ADAPTER_REGISTRY[(Mode.MARGIN, "coinbase")]
    assert registration.factory is manual_factory
    assert registration.dynamic is False


def test_temporary_native_adapter_restores_previous_entry(reset_registry: None) -> None:
    def base_factory(credentials, **kwargs):
        return FakeNativeAdapter("base", credentials, **kwargs)

    def override_factory(credentials, **kwargs):
        return FakeNativeAdapter("override", credentials, **kwargs)

    manager_module.register_native_adapter(
        exchange_id="binance",
        mode=Mode.MARGIN,
        factory=base_factory,
        default_settings={"leverage_mode": "isolated"},
        supports_testnet=False,
    )

    with manager_module.temporary_native_adapter(
        exchange_id="binance",
        mode=Mode.MARGIN,
        factory=override_factory,
        default_settings={"leverage_mode": "cross"},
        supports_testnet=True,
    ) as info:
        assert info.exchange_id == "binance"
        assert info.factory is override_factory
        assert info.default_settings["leverage_mode"] == "cross"
        assert info.supports_testnet is True
        assert (Mode.MARGIN, "binance") in manager_module._NATIVE_ADAPTER_REGISTRY

    restored = manager_module.get_native_adapter_info(exchange_id="binance", mode=Mode.MARGIN)
    assert restored is not None
    assert restored.factory is base_factory
    assert restored.default_settings["leverage_mode"] == "isolated"
    assert restored.supports_testnet is False
    assert (Mode.MARGIN, "binance") not in manager_module._DYNAMIC_ADAPTER_KEYS


def test_temporary_native_adapter_removes_new_registration(reset_registry: None) -> None:
    def temp_factory(credentials, **kwargs):
        return FakeNativeAdapter("temp", credentials, **kwargs)

    key = (Mode.FUTURES, "okx")
    assert key not in manager_module._NATIVE_ADAPTER_REGISTRY

    with manager_module.temporary_native_adapter(
        exchange_id="okx",
        mode=Mode.FUTURES,
        factory=temp_factory,
        dynamic=True,
    ) as info:
        assert info.exchange_id == "okx"
        assert key in manager_module._NATIVE_ADAPTER_REGISTRY
        assert key in manager_module._DYNAMIC_ADAPTER_KEYS

    assert key not in manager_module._NATIVE_ADAPTER_REGISTRY
    assert key not in manager_module._DYNAMIC_ADAPTER_KEYS


def test_temporary_native_adapter_accepts_string_mode(reset_registry: None) -> None:
    def factory(credentials, **kwargs):
        return FakeNativeAdapter("string-mode", credentials, **kwargs)

    key = (Mode.MARGIN, "kraken")
    assert key not in manager_module._NATIVE_ADAPTER_REGISTRY

    with manager_module.temporary_native_adapter(
        exchange_id="kraken", mode="margin", factory=factory
    ) as info:
        assert info.mode is Mode.MARGIN
        assert key in manager_module._NATIVE_ADAPTER_REGISTRY

    assert key not in manager_module._NATIVE_ADAPTER_REGISTRY


def test_temporary_native_adapters_handles_multiple_entries(reset_registry: None) -> None:
    def first_factory(credentials, **kwargs):
        return FakeNativeAdapter("first", credentials, **kwargs)

    def second_factory(credentials, **kwargs):
        return FakeNativeAdapter("second", credentials, **kwargs)

    manager_module.register_native_adapter(
        exchange_id="okx",
        mode=Mode.FUTURES,
        factory=first_factory,
        default_settings={"leverage": 50},
    )

    specs = [
        {
            "exchange_id": "bybit",
            "mode": Mode.MARGIN,
            "factory": first_factory,
            "supports_testnet": False,
            "dynamic": True,
        },
        manager_module.NativeAdapterInfo(
            exchange_id="okx",
            mode=Mode.FUTURES,
            factory=second_factory,
            default_settings={"leverage": 75},
            supports_testnet=True,
            source=None,
            dynamic=False,
        ),
    ]

    with manager_module.temporary_native_adapters(*specs) as infos:
        assert len(infos) == 2
        assert infos[0].exchange_id == "bybit"
        assert infos[1].factory is second_factory
        assert (Mode.MARGIN, "bybit") in manager_module._NATIVE_ADAPTER_REGISTRY
        registration = manager_module._NATIVE_ADAPTER_REGISTRY[(Mode.FUTURES, "okx")]
        assert registration.factory is second_factory
        assert registration.default_settings["leverage"] == 75

    assert (Mode.MARGIN, "bybit") not in manager_module._NATIVE_ADAPTER_REGISTRY
    restored = manager_module.get_native_adapter_info(exchange_id="okx", mode=Mode.FUTURES)
    assert restored is not None
    assert restored.factory is first_factory
    assert restored.default_settings["leverage"] == 50


def test_temporary_native_adapters_entries_keyword(reset_registry: None) -> None:
    def factory(credentials, **kwargs):
        return FakeNativeAdapter("kw", credentials, **kwargs)

    with manager_module.temporary_native_adapters(
        entries=[
            {
                "exchange_id": "kraken",
                "mode": Mode.MARGIN,
                "factory": factory,
                "dynamic": True,
            }
        ]
    ) as infos:
        assert len(infos) == 1
        assert infos[0].exchange_id == "kraken"
        assert (Mode.MARGIN, "kraken") in manager_module._NATIVE_ADAPTER_REGISTRY


def test_temporary_native_adapters_support_string_modes(reset_registry: None) -> None:
    def margin_factory(credentials, **kwargs):
        return FakeNativeAdapter("margin", credentials, **kwargs)

    def futures_factory(credentials, **kwargs):
        return FakeNativeAdapter("futures", credentials, **kwargs)

    specs = [
        {"exchange_id": "bybit", "mode": "margin", "factory": margin_factory},
        {"exchange_id": "bybit", "mode": "futures", "factory": futures_factory},
    ]

    with manager_module.temporary_native_adapters(*specs) as infos:
        assert {info.mode for info in infos} == {Mode.MARGIN, Mode.FUTURES}
        assert (Mode.MARGIN, "bybit") in manager_module._NATIVE_ADAPTER_REGISTRY
        assert (Mode.FUTURES, "bybit") in manager_module._NATIVE_ADAPTER_REGISTRY


def test_temporary_native_adapters_validates_input(reset_registry: None) -> None:
    with pytest.raises(ValueError):
        with manager_module.temporary_native_adapters():
            pass

    with pytest.raises(TypeError):
        with manager_module.temporary_native_adapters(object()):
            pass

    with pytest.raises(KeyError):
        with manager_module.temporary_native_adapters({"exchange_id": "binance", "mode": Mode.MARGIN}):
            pass

    with pytest.raises(ValueError):
        with manager_module.temporary_native_adapters(
            {"exchange_id": "binance", "mode": "spot", "factory": object()}
        ):
            pass
