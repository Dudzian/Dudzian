from __future__ import annotations

import subprocess
import sys
from itertools import islice
from pathlib import Path

import pytest
grpc = pytest.importorskip("grpc")
pytest.importorskip("grpc_tools")
from google.protobuf import empty_pb2
import yaml

from bot_core.testing import (
    TradingStubServer,
    build_default_dataset,
    load_dataset_from_yaml,
    merge_datasets,
)


pytestmark = pytest.mark.requires_trading_stubs


@pytest.fixture(scope="module", autouse=True)
def ensure_trading_stubs() -> None:
    subprocess.run(
        [sys.executable, "scripts/generate_trading_stubs.py", "--skip-cpp"],
        check=True,
    )


@pytest.fixture(scope="module")
def trading_modules():
    generated_dir = Path("bot_core/generated").resolve()
    if str(generated_dir) not in sys.path:
        sys.path.insert(0, str(generated_dir))
    from bot_core.generated import trading_pb2, trading_pb2_grpc

    return trading_pb2, trading_pb2_grpc


def test_history_and_stream_responses(trading_modules) -> None:
    trading_pb2, trading_pb2_grpc = trading_modules
    dataset = build_default_dataset()
    first_candles = next(iter(dataset.history.values()))
    instrument = first_candles[0].instrument
    granularity = first_candles[0].granularity

    with TradingStubServer(dataset, port=0) as server:
        channel = grpc.insecure_channel(server.address)
        grpc.channel_ready_future(channel).result(timeout=5)
        market_stub = trading_pb2_grpc.MarketDataServiceStub(channel)

        history = market_stub.GetOhlcvHistory(
            trading_pb2.GetOhlcvHistoryRequest(
                instrument=instrument,
                granularity=granularity,
                limit=2,
            )
        )
        assert len(history.candles) == 2
        assert history.has_more is True
        assert history.next_start_time.seconds != 0

        stream = market_stub.StreamOhlcv(
            trading_pb2.StreamOhlcvRequest(
                instrument=instrument,
                granularity=granularity,
                deliver_snapshots=True,
            )
        )
        updates = list(stream)
        assert updates[0].HasField("snapshot")
        assert len(updates[0].snapshot.candles) == 2
        assert updates[1].HasField("increment")
        assert updates[1].increment.candle.sequence == 3
        channel.close()


def test_stream_repeat_mode(trading_modules) -> None:
    trading_pb2, trading_pb2_grpc = trading_modules
    dataset = build_default_dataset()
    instrument = next(iter(dataset.history.values()))[0].instrument
    granularity = next(iter(dataset.history.values()))[0].granularity

    with TradingStubServer(dataset, port=0, stream_repeat=True, stream_interval=0.0) as server:
        channel = grpc.insecure_channel(server.address)
        grpc.channel_ready_future(channel).result(timeout=5)
        market_stub = trading_pb2_grpc.MarketDataServiceStub(channel)

        stream = market_stub.StreamOhlcv(
            trading_pb2.StreamOhlcvRequest(
                instrument=instrument,
                granularity=granularity,
                deliver_snapshots=False,
            )
        )
        updates = list(islice(stream, 4))
        increment_count = sum(1 for update in updates if update.HasField("increment"))
        assert increment_count == 4  # wszystkie aktualizacje to incrementy
        channel.close()


def test_orders_risk_metrics_and_health(trading_modules) -> None:
    trading_pb2, trading_pb2_grpc = trading_modules
    dataset = build_default_dataset()
    instrument = next(iter(dataset.history.values()))[0].instrument
    granularity = next(iter(dataset.history.values()))[0].granularity

    with TradingStubServer(dataset, port=0) as server:
        channel = grpc.insecure_channel(server.address)
        grpc.channel_ready_future(channel).result(timeout=5)

        order_stub = trading_pb2_grpc.OrderServiceStub(channel)
        risk_stub = trading_pb2_grpc.RiskServiceStub(channel)
        metrics_stub = trading_pb2_grpc.MetricsServiceStub(channel)
        health_stub = trading_pb2_grpc.HealthServiceStub(channel)

        submit = order_stub.SubmitOrder(
            trading_pb2.SubmitOrderRequest(
                instrument=instrument,
                side=trading_pb2.OrderSide.ORDER_SIDE_BUY,
                type=trading_pb2.OrderType.ORDER_TYPE_MARKET,
                quantity=0.1,
                price=0.0,
                time_in_force=trading_pb2.TimeInForce.TIME_IN_FORCE_GTC,
            )
        )
        assert submit.status == trading_pb2.OrderStatus.ORDER_STATUS_ACCEPTED

        cancel = order_stub.CancelOrder(
            trading_pb2.CancelOrderRequest(instrument=instrument, order_id=submit.order_id)
        )
        assert cancel.status == trading_pb2.OrderStatus.ORDER_STATUS_ACCEPTED

        risk = risk_stub.GetRiskState(trading_pb2.RiskStateRequest())
        assert risk.profile == trading_pb2.RiskProfile.RISK_PROFILE_BALANCED

        stream_states = list(risk_stub.StreamRiskState(trading_pb2.RiskStateRequest()))
        assert stream_states
        assert stream_states[0].generated_at == risk.generated_at

        ack = metrics_stub.PushMetrics(
            trading_pb2.MetricsSnapshot(
                generated_at=risk.generated_at,
                event_to_frame_p95_ms=90.0,
                fps=60.0,
            )
        )
        assert ack.accepted is True

        metrics = list(metrics_stub.StreamMetrics(trading_pb2.MetricsRequest(include_ui_metrics=True)))
        assert metrics

        health = health_stub.Check(empty_pb2.Empty())
        assert health.version.startswith("stub")
        channel.close()


def test_tradable_instruments_rpc(trading_modules) -> None:
    trading_pb2, trading_pb2_grpc = trading_modules
    dataset = build_default_dataset()

    fallback = trading_pb2.TradableInstrumentMetadata(
        instrument=trading_pb2.Instrument(
            exchange="GENERIC",
            symbol="BCH/USDT",
            venue_symbol="BCHUSDT",
            quote_currency="USDT",
            base_currency="BCH",
        ),
        price_step=0.01,
        amount_step=0.1,
        min_notional=25.0,
        min_amount=0.01,
    )
    dataset.set_tradable_instruments("*", [fallback])

    specific = trading_pb2.TradableInstrumentMetadata(
        instrument=trading_pb2.Instrument(
            exchange="COINBASE",
            symbol="SOL/USD",
            venue_symbol="SOLUSD",
            quote_currency="USD",
            base_currency="SOL",
        ),
        price_step=0.01,
        amount_step=0.001,
        min_notional=1.0,
        min_amount=0.01,
    )
    dataset.set_tradable_instruments("COINBASE", [specific])

    with TradingStubServer(dataset, port=0) as server:
        channel = grpc.insecure_channel(server.address)
        grpc.channel_ready_future(channel).result(timeout=5)
        market_stub = trading_pb2_grpc.MarketDataServiceStub(channel)

        binance_response = market_stub.ListTradableInstruments(
            trading_pb2.ListTradableInstrumentsRequest(exchange="BINANCE")
        )
        assert len(binance_response.instruments) >= 2
        assert {
            item.instrument.symbol for item in binance_response.instruments
        } >= {"BTC/USDT", "ETH/USDT"}

        coinbase_response = market_stub.ListTradableInstruments(
            trading_pb2.ListTradableInstrumentsRequest(exchange="coinbase")
        )
        assert [item.instrument.symbol for item in coinbase_response.instruments] == [
            "SOL/USD"
        ]

        fallback_response = market_stub.ListTradableInstruments(
            trading_pb2.ListTradableInstrumentsRequest(exchange="UNKNOWN")
        )
        assert [item.instrument.symbol for item in fallback_response.instruments] == [
            "BCH/USDT"
        ]
        channel.close()


def test_yaml_loader_builds_dataset(tmp_path: Path, trading_modules) -> None:
    trading_pb2, _ = trading_modules
    cfg_path = tmp_path / "stub.yaml"
    cfg = {
        "market_data": [
            {
                "instrument": {
                    "exchange": "KRAKEN",
                    "symbol": "ETH/EUR",
                    "venue_symbol": "ETHEUR",
                    "quote_currency": "EUR",
                    "base_currency": "ETH",
                },
                "granularity": "PT5M",
                "candles": [
                    {
                        "open_time": "2024-03-01T00:00:00Z",
                        "open": 2500.0,
                        "high": 2520.0,
                        "low": 2490.0,
                        "close": 2510.0,
                        "volume": 12.0,
                    }
                ],
                "stream": {
                    "snapshot": [
                        {
                            "open_time": "2024-03-01T00:00:00Z",
                            "open": 2500.0,
                            "high": 2520.0,
                            "low": 2490.0,
                            "close": 2510.0,
                            "volume": 12.0,
                        }
                    ],
                    "increments": [
                        {
                            "open_time": "2024-03-01T00:05:00Z",
                            "open": 2510.0,
                            "high": 2530.0,
                            "low": 2505.0,
                            "close": 2525.0,
                            "volume": 8.0,
                            "closed": False,
                        }
                    ],
                },
            }
        ],
        "risk_states": [
            {
                "states": [
                    {
                        "profile": "RISK_PROFILE_CONSERVATIVE",
                        "portfolio_value": 500000.0,
                        "current_drawdown": 0.01,
                        "generated_at": "2024-03-01T00:10:00Z",
                    }
                ]
            }
        ],
        "metrics": [
            {
                "generated_at": "2024-03-01T00:15:00Z",
                "event_to_frame_p95_ms": 95.0,
                "fps": 120.0,
            }
        ],
        "health": {
            "version": "stub-ci",
            "git_commit": "abcdef1",
            "started_at": "2024-03-01T00:00:00Z",
        },
        "tradable_instruments": [
            {
                "exchange": "KRAKEN",
                "instrument": {
                    "exchange": "KRAKEN",
                    "symbol": "ADA/EUR",
                    "venue_symbol": "ADAEUR",
                    "quote_currency": "EUR",
                    "base_currency": "ADA",
                },
                "price_step": 0.0001,
                "amount_step": 1.0,
                "min_notional": 10.0,
                "min_amount": 5.0,
            }
        ],
    }
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    dataset = load_dataset_from_yaml(cfg_path)
    key = next(iter(dataset.history.keys()))
    candles = dataset.history[key]
    assert len(candles) == 1
    assert dataset.stream_increments[key]
    assert dataset.metrics
    assert dataset.health.version == "stub-ci"
    listings = dataset.list_tradable_instruments("KRAKEN")
    assert len(listings) == 1
    assert listings[0].instrument.venue_symbol == "ADAEUR"


def test_multi_asset_dataset_and_performance_guard(trading_modules) -> None:
    trading_pb2, trading_pb2_grpc = trading_modules
    overlay_path = Path("data/trading_stub/datasets/multi_asset_performance.yaml").resolve()
    overlay_dataset = load_dataset_from_yaml(overlay_path)
    merged_dataset = merge_datasets(build_default_dataset(), overlay_dataset)

    guard = merged_dataset.performance_guard
    assert guard["fps_target"] == 120
    # wartości domyślne nadal dostępne po nałożeniu datasetu
    assert guard["max_overlay_count"] == 3
    assert guard["overlays"]["disable_secondary_when_fps_below"] == 55

    target_key = next(key for key in merged_dataset.history if key[1] == "ETH/EUR")
    candles = merged_dataset.history[target_key]
    instrument = candles[0].instrument
    granularity = candles[0].granularity

    with TradingStubServer(
        merged_dataset,
        port=0,
        stream_repeat=True,
        stream_interval=0.0,
    ) as server:
        assert server.performance_guard["fps_target"] == 120
        channel = grpc.insecure_channel(server.address)
        grpc.channel_ready_future(channel).result(timeout=5)
        market_stub = trading_pb2_grpc.MarketDataServiceStub(channel)

        stream = market_stub.StreamOhlcv(
            trading_pb2.StreamOhlcvRequest(
                instrument=instrument,
                granularity=granularity,
                deliver_snapshots=True,
            )
        )
        updates = list(islice(stream, 4))
        assert updates and updates[0].HasField("snapshot")
        assert len(updates[0].snapshot.candles) == 1
        increment_updates = [u for u in updates if u.HasField("increment")]
        assert increment_updates
        assert increment_updates[0].increment.candle.instrument.symbol == "ETH/EUR"
        channel.close()
