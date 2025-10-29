from __future__ import annotations

from dataclasses import dataclass

import pytest

from bot_core.exchanges.base import AccountSnapshot, Environment
from bot_core.exchanges.core import MarketRules
from bot_core.exchanges import manager as manager_module
from bot_core.exchanges.manager import ExchangeManager, Mode, register_native_adapter
from bot_core.exchanges.paper_simulator import PaperMarginSimulator


@dataclass
class _AdapterInit:
    environment: Environment
    settings: dict[str, object]
    watchdog: object | None
    passphrase: str | None


class _FakeMarginAdapter:
    last_init: _AdapterInit | None = None

    def __init__(self, credentials, *, environment, settings=None, watchdog=None):
        self.credentials = credentials
        self.environment = environment
        self.settings = dict(settings or {})
        self.watchdog = watchdog
        _FakeMarginAdapter.last_init = _AdapterInit(
            environment,
            self.settings,
            watchdog,
            credentials.passphrase,
        )

    def fetch_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            balances={"USDT": 1_000.0},
            total_equity=1_000.0,
            available_margin=800.0,
            maintenance_margin=120.0,
        )


class _DummyFeed:
    def __init__(self, price: float) -> None:
        self.price = float(price)
        self.rules = {
            "BTC/USDT": MarketRules(
                symbol="BTC/USDT",
                price_step=0.1,
                amount_step=0.001,
                min_notional=10.0,
            )
        }

    def load_markets(self):
        return self.rules

    def get_market_rules(self, symbol: str):
        return self.rules.get(symbol)

    def fetch_ticker(self, symbol: str):
        return {"last": self.price}


register_native_adapter(
    exchange_id="testex",
    mode=Mode.MARGIN,
    factory=_FakeMarginAdapter,
    default_settings={"native": True},
)


def test_exchange_manager_builds_registered_adapter(monkeypatch):
    manager = ExchangeManager("testex")
    manager.set_mode(margin=True, testnet=True)
    manager.set_credentials("key", "secret", passphrase="phrase")
    manager.configure_watchdog(retry_policy={"max_attempts": 2, "base_delay": 0.1})
    manager.configure_native_adapter(settings={"custom": "value"})

    snapshot = manager.fetch_balance()

    assert snapshot["total_equity"] == pytest.approx(1_000.0)
    init = _FakeMarginAdapter.last_init
    assert init is not None
    assert init.environment in {Environment.TESTNET, Environment.LIVE}
    assert init.watchdog is not None
    assert init.settings["native"] is True
    assert init.settings["custom"] == "value"
    assert init.passphrase == "phrase"


def test_fetch_account_snapshot_emits_mark_event(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = ExchangeManager()
    monkeypatch.setattr(manager, "_ensure_public", lambda: _DummyFeed(100.0))
    manager.set_mode(paper=True)
    manager.set_paper_variant("margin")
    captured: list[dict[str, object]] = []
    manager.on("ACCOUNT_MARK", lambda event: captured.append(dict(event.payload)))

    snapshot = manager.fetch_account_snapshot()

    assert isinstance(snapshot, AccountSnapshot)
    assert captured
    payload = captured[-1]
    assert payload["snapshot"]["total_equity"] == pytest.approx(snapshot.total_equity)


def test_paper_variant_switches_to_margin_simulator(monkeypatch):
    manager = ExchangeManager("testex")
    fake_public = _DummyFeed(100.0)
    monkeypatch.setattr(manager, "_ensure_public", lambda: fake_public)
    manager.set_mode(paper=True)
    manager.set_paper_variant("margin")
    manager.configure_paper_simulator(leverage_limit=5.0)
    paper_backend = manager._ensure_paper()
    assert isinstance(paper_backend, PaperMarginSimulator)
    balance = manager.fetch_balance()
    assert "USDT" in balance


def test_public_feed_uses_futures_default_type(monkeypatch):
    created: list[dict[str, object]] = []

    class _Client:
        def __init__(self, options: dict[str, object]) -> None:
            self.options = options

        def load_markets(self):
            return {}

    class _Module:
        def binance(self, options: dict[str, object]) -> _Client:  # type: ignore[override]
            created.append(options)
            return _Client(options)

    monkeypatch.setattr(manager_module, "ccxt", _Module())

    mgr = ExchangeManager("binance")
    mgr.set_mode(futures=True)

    feed = mgr._ensure_public()

    assert created
    assert created[0]["options"]["defaultType"] == "future"
    assert feed.futures is True


def test_public_feed_uses_margin_default_type(monkeypatch):
    created: list[dict[str, object]] = []

    class _Client:
        def __init__(self, options: dict[str, object]) -> None:
            self.options = options

    class _Module:
        def binance(self, options: dict[str, object]) -> _Client:  # type: ignore[override]
            created.append(options)
            return _Client(options)

    monkeypatch.setattr(manager_module, "ccxt", _Module())

    mgr = ExchangeManager("binance")
    mgr.set_mode(margin=True)

    feed = mgr._ensure_public()

    assert created
    assert created[0]["options"]["defaultType"] == "margin"
    assert feed.futures is False
    assert feed.market_type == "margin"


def test_private_backend_receives_passphrase(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[dict[str, object]] = []

    class _Client:
        def __init__(self, options: dict[str, object]) -> None:
            self.options = options

        def load_markets(self):
            return {}

    class _Module:
        def binance(self, options: dict[str, object]) -> _Client:  # type: ignore[override]
            created.append(options)
            return _Client(options)

    monkeypatch.setattr(manager_module, "ccxt", _Module())

    mgr = ExchangeManager("binance")
    mgr.set_mode(margin=True)
    mgr.set_credentials("key", "secret", passphrase="phrase")

    backend = mgr._ensure_private()

    assert created
    assert created[-1]["password"] == "phrase"
    assert backend.client.options["options"]["defaultType"] == "margin"


def test_exchange_manager_reports_paper_configuration() -> None:
    manager = ExchangeManager("binance")
    manager.set_mode(paper=True)

    assert manager.get_paper_variant() == "spot"
    assert manager.get_paper_cash_asset() == "USDT"
    assert manager.get_paper_initial_cash() == pytest.approx(10_000.0)
    assert manager.get_paper_fee_rate() == pytest.approx(0.001)
    assert manager.get_paper_simulator_settings() == {}

    manager.set_paper_variant("futures")
    manager.set_paper_balance(55_000.0, asset="eur")
    manager.set_paper_fee_rate(0.0005)

    assert manager.get_paper_variant() == "futures"
    assert manager.get_paper_cash_asset() == "EUR"
    assert manager.get_paper_initial_cash() == pytest.approx(55_000.0)
    assert manager.get_paper_fee_rate() == pytest.approx(0.0005)
    assert manager.get_paper_simulator_settings() == {
        "leverage_limit": pytest.approx(10.0),
        "maintenance_margin_ratio": pytest.approx(0.05),
        "funding_rate": pytest.approx(0.0001),
        "funding_interval_seconds": pytest.approx(0.0),
    }


def test_configure_paper_simulator_merges_overrides() -> None:
    manager = ExchangeManager("binance")
    manager.set_mode(paper=True)
    manager.set_paper_variant("margin")

    manager.configure_paper_simulator(leverage_limit=4.5)
    manager.configure_paper_simulator(funding_rate=0.0002)
    manager.configure_paper_simulator(funding_interval_seconds=7_200)

    settings = manager.get_paper_simulator_settings()
    assert settings["leverage_limit"] == pytest.approx(4.5)
    assert settings["maintenance_margin_ratio"] == pytest.approx(0.15)
    assert settings["funding_rate"] == pytest.approx(0.0002)
    assert settings["funding_interval_seconds"] == pytest.approx(7_200.0)


def test_configure_paper_simulator_accepts_custom_keys() -> None:
    manager = ExchangeManager("binance")
    manager.set_mode(paper=True)
    manager.set_paper_variant("margin")

    manager.configure_paper_simulator(liquidation_buffer=0.05)
    manager.configure_paper_simulator(funding_rate=0.0003)

    settings = manager.get_paper_simulator_settings()
    assert settings["liquidation_buffer"] == pytest.approx(0.05)
    assert settings["funding_rate"] == pytest.approx(0.0003)


def test_configure_paper_simulator_requires_numeric_values() -> None:
    manager = ExchangeManager("binance")
    manager.set_mode(paper=True)
    manager.set_paper_variant("margin")

    with pytest.raises(ValueError, match="liczbowej"):
        manager.configure_paper_simulator(liquidation_buffer="abc")


def test_configure_paper_simulator_rejects_non_positive_interval() -> None:
    manager = ExchangeManager("binance")
    manager.set_mode(paper=True)
    manager.set_paper_variant("margin")

    with pytest.raises(ValueError, match="dodatniej"):
        manager.configure_paper_simulator(funding_interval_seconds=0)


def test_apply_environment_profile_uses_yaml_configuration() -> None:
    manager = ExchangeManager("binance")

    manager.apply_environment_profile("paper")

    assert manager.mode is Mode.PAPER
    assert manager.get_paper_cash_asset() == "USDT"

    profile = manager.describe_environment_profile()
    assert profile is not None
    assert profile.get("name") == "paper"


def test_apply_environment_profile_expands_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BINANCE_LIVE_KEY", "abc")
    monkeypatch.setenv("BINANCE_LIVE_SECRET", "xyz")
    manager = ExchangeManager("binance")

    manager.apply_environment_profile("live")

    assert manager.mode is Mode.MARGIN
    assert manager._api_key == "abc"
    assert manager._secret == "xyz"


def test_updating_credentials_rebuilds_private_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[dict[str, object]] = []

    class _Client:
        def __init__(self, options: dict[str, object]) -> None:
            self.options = options

        def load_markets(self):
            return {}

    class _Module:
        def binance(self, options: dict[str, object]) -> _Client:  # type: ignore[override]
            created.append(options)
            return _Client(options)

    monkeypatch.setattr(manager_module, "ccxt", _Module())

    mgr = ExchangeManager("binance")
    mgr.set_mode(margin=True)
    mgr.set_credentials("key", "secret")
    first_backend = mgr._ensure_private()

    assert created and created[-1]["apiKey"] == "key"

    mgr.set_credentials("key2", "secret2")
    second_backend = mgr._ensure_private()

    assert created[-1]["apiKey"] == "key2"
    assert first_backend is not second_backend
    assert second_backend.client.options["options"]["defaultType"] == "margin"
