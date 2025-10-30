import time

import time

import grpc
import pytest
from google.protobuf import empty_pb2

from bot_core.api.server import build_local_runtime_context, LocalRuntimeServer
from bot_core.generated import trading_pb2, trading_pb2_grpc


_INTERVAL_TO_ISO = {
    "1m": "PT1M",
    "3m": "PT3M",
    "5m": "PT5M",
    "15m": "PT15M",
    "30m": "PT30M",
    "1h": "PT1H",
    "2h": "PT2H",
    "4h": "PT4H",
    "6h": "PT6H",
    "8h": "PT8H",
    "12h": "PT12H",
    "1d": "P1D",
    "3d": "P3D",
    "1w": "P1W",
    "1M": "P1M",
}


@pytest.mark.integration
@pytest.mark.usefixtures("tmp_path")
def test_local_runtime_gRPC_paper_pipeline():
    context = build_local_runtime_context(config_path="config/runtime.yaml")
    context.start()
    server = LocalRuntimeServer(context, host="127.0.0.1", port=0)
    server.start()
    channel = grpc.insecure_channel(server.address)
    try:
        grpc.channel_ready_future(channel).result(timeout=5)

        health_stub = trading_pb2_grpc.HealthServiceStub(channel)
        health_response = health_stub.Check(empty_pb2.Empty())
        assert health_response.version

        markets = getattr(context.pipeline.execution_service, "_markets")
        symbol = context.primary_symbol
        metadata = markets[symbol]
        iso_duration = _INTERVAL_TO_ISO.get(getattr(context.pipeline.controller, "interval", "1h"), "PT1H")
        instrument = trading_pb2.Instrument(
            exchange=(context.exchange_name or "PAPER").upper(),
            symbol=symbol,
            venue_symbol=symbol.replace("/", "").replace("-", ""),
            quote_currency=metadata.quote_asset.upper(),
            base_currency=metadata.base_asset.upper(),
        )

        market_stub = trading_pb2_grpc.MarketDataServiceStub(channel)
        history_request = trading_pb2.GetOhlcvHistoryRequest(
            instrument=instrument,
            granularity=trading_pb2.CandleGranularity(iso8601_duration=iso_duration),
            limit=50,
        )
        history_response = market_stub.GetOhlcvHistory(history_request)
        assert len(history_response.candles) > 0

        order_stub = trading_pb2_grpc.OrderServiceStub(channel)
        submit_request = trading_pb2.SubmitOrderRequest(
            instrument=instrument,
            side=trading_pb2.ORDER_SIDE_BUY,
            type=trading_pb2.ORDER_TYPE_MARKET,
            quantity=metadata.min_quantity or 0.001,
        )
        order_response = order_stub.SubmitOrder(submit_request)
        assert order_response.status == trading_pb2.ORDER_STATUS_ACCEPTED

        time.sleep(0.5)
        risk_stub = trading_pb2_grpc.RiskServiceStub(channel)
        risk_state = risk_stub.GetRiskState(trading_pb2.RiskStateRequest())
        assert risk_state.profile != trading_pb2.RISK_PROFILE_UNSPECIFIED
    finally:
        channel.close()
        server.stop(0.5)
        context.stop()
