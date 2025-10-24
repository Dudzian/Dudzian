import grpc
import pytest

from bot_core.exchanges.core import MarketRules
from bot_core.runtime.market_data_service import MarketDataServiceServicer
from bot_core.generated import trading_pb2


class FakeManager:
    def __init__(self, markets: dict[str, dict[str, object]], rules: dict[str, MarketRules]) -> None:
        self._public = type("_Public", (), {"_markets": markets})()
        self._rules = rules
        self.load_calls = 0

    def load_markets(self) -> dict[str, MarketRules]:
        self.load_calls += 1
        return self._rules


class DummyContext:
    def __init__(self) -> None:
        self.code: grpc.StatusCode | None = None
        self.details: str | None = None

    def abort(self, code: grpc.StatusCode, details: str) -> None:
        self.code = code
        self.details = details
        raise RuntimeError(details)

    def invocation_metadata(self):  # pragma: no cover - konwencja gRPC
        return ()


@pytest.fixture(name="manager")
def manager_fixture() -> FakeManager:
    markets = {
        "BTC/USDT": {
            "id": "BTCUSDT",
            "symbol": "BTC/USDT",
            "base": "BTC",
            "quote": "USDT",
        }
    }
    rules = {
        "BTC/USDT": MarketRules(
            symbol="BTC/USDT",
            price_step=0.01,
            amount_step=0.001,
            min_notional=10.0,
            min_amount=0.001,
            max_amount=25.0,
            min_price=0.1,
            max_price=125000.0,
        )
    }
    return FakeManager(markets, rules)


def _make_request(exchange: str) -> trading_pb2.ListTradableInstrumentsRequest:
    return trading_pb2.ListTradableInstrumentsRequest(exchange=exchange)


def test_list_tradable_instruments_uses_manager_data(manager: FakeManager) -> None:
    servicer = MarketDataServiceServicer(manager_lookup={"BINANCE": manager})
    context = DummyContext()

    response = servicer.ListTradableInstruments(_make_request("BINANCE"), context)

    assert [item.instrument.symbol for item in response.instruments] == ["BTC/USDT"]
    entry = response.instruments[0]
    assert entry.instrument.exchange == "BINANCE"
    assert entry.instrument.venue_symbol == "BTCUSDT"
    assert entry.instrument.quote_currency == "USDT"
    assert entry.instrument.base_currency == "BTC"
    assert entry.price_step == pytest.approx(0.01)
    assert entry.amount_step == pytest.approx(0.001)
    assert entry.min_notional == pytest.approx(10.0)
    assert entry.min_amount == pytest.approx(0.001)
    assert entry.max_amount == pytest.approx(25.0)
    assert entry.min_price == pytest.approx(0.1)
    assert entry.max_price == pytest.approx(125000.0)
    assert manager.load_calls == 1

    # Kolejne wywołanie korzysta z cache'u
    servicer.ListTradableInstruments(_make_request("BINANCE"), context)
    assert manager.load_calls == 1


def test_list_tradable_instruments_without_cache(manager: FakeManager) -> None:
    servicer = MarketDataServiceServicer(manager_lookup={"BINANCE": manager}, cache_ttl=0.0)
    context = DummyContext()

    servicer.ListTradableInstruments(_make_request("binance"), context)
    servicer.ListTradableInstruments(_make_request("BINANCE"), context)
    assert manager.load_calls == 2


def test_list_tradable_instruments_handles_missing_markets() -> None:
    rules = {"FOO/BAR": MarketRules(symbol="FOO/BAR")}
    manager = FakeManager({}, rules)
    servicer = MarketDataServiceServicer(manager_lookup={"FOOEX": manager})
    context = DummyContext()

    response = servicer.ListTradableInstruments(_make_request("FOOEX"), context)
    entry = response.instruments[0]
    assert entry.instrument.venue_symbol == "FOOBAR"
    assert entry.instrument.base_currency == "FOO"
    assert entry.instrument.quote_currency == "BAR"


def test_list_tradable_instruments_requires_exchange() -> None:
    servicer = MarketDataServiceServicer(manager_lookup={})
    context = DummyContext()

    with pytest.raises(RuntimeError):
        servicer.ListTradableInstruments(_make_request(""), context)
    assert context.code == grpc.StatusCode.INVALID_ARGUMENT


def test_list_tradable_instruments_unknown_exchange() -> None:
    servicer = MarketDataServiceServicer(manager_lookup={})
    context = DummyContext()

    with pytest.raises(RuntimeError):
        servicer.ListTradableInstruments(_make_request("UNKNOWN"), context)
    assert context.code == grpc.StatusCode.NOT_FOUND
