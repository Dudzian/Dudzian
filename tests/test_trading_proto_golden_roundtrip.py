from __future__ import annotations

import tempfile
from importlib import resources
from pathlib import Path

import pytest
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

pytest.importorskip("grpc_tools")
from grpc_tools import protoc


PROTO_DIR = Path("proto")
GOOGLE_PROTO_INCLUDE = resources.files("grpc_tools") / "_proto"

# Deterministyczne reprezentacje wybranych wiadomości tradingowych.
# Wartości wygenerowane poprzez serializację wiadomości opisanych w testach.
EXPECTED_HISTORY_HEX = (
    "0a6a0a270a0742494e414e434512084254432f555344541a07425443555344542204555344542a03425443"
    "1206088081c8ac0619000000000082e4402100000000808ee4402900000000c07be4403100000000008ce440"
    "39ae47e17a14ae284040014a060a045054314d50010a6a0a270a0742494e414e434512084254432f55534454"
    "1a07425443555344542204555344542a03425443120608bc81c8ac061900000000008ce4402100000000009b"
    "e44029000000000082e4403100000000c094e44039cdcccccccccc214040014a060a045054314d50021a0608"
    "f881c8ac06"
)

EXPECTED_STREAM_SNAPSHOT_HEX = (
    "0ace010a660a230a064b52414b454e12074254432f4555521a0658425445555222034555522a034254431206"
    "08c08feead0619000000000017e1402100000000c029e1402900000000800ae14031000000000021e1403900"
    "0000000080394040014a060a045054354d50640a640a230a064b52414b454e12074254432f4555521a065842"
    "5445555222034555522a03425443120608ec91eead0619000000000021e14021000000000030e14029000000"
    "00401de1403100000000402ce140390000000000c032404a060a045054354d5065"
)

EXPECTED_STREAM_INCREMENT_HEX = (
    "12660a640a230a064b52414b454e12074254432f4555521a0658425445555222034555522a03425443120608"
    "ec91eead0619000000000021e14021000000000030e1402900000000401de1403100000000402ce140390000000000c032404a060a045054354d5065"
)


def _load_descriptor_pool() -> descriptor_pool.DescriptorPool:
    with tempfile.TemporaryDirectory() as tmp:
        descriptor_path = Path(tmp) / "trading.desc"
        args = [
            "protoc",
            f"--proto_path={PROTO_DIR}",
            f"--proto_path={GOOGLE_PROTO_INCLUDE}",
            f"--descriptor_set_out={descriptor_path}",
            "--include_imports",
            "trading.proto",
        ]
        result = protoc.main(args)
        if result != 0:
            pytest.skip("protoc not available in test environment")
        descriptor_set = descriptor_pb2.FileDescriptorSet()
        descriptor_set.ParseFromString(descriptor_path.read_bytes())

    pool = descriptor_pool.DescriptorPool()
    for file_proto in descriptor_set.file:
        pool.Add(file_proto)
    return pool


def _make_pool() -> descriptor_pool.DescriptorPool:
    return _load_descriptor_pool()


def _get_cls(pool: descriptor_pool.DescriptorPool, fq_name: str):
    return message_factory.GetMessageClass(pool.FindMessageTypeByName(fq_name))


def _make_timestamp(pool: descriptor_pool.DescriptorPool, iso: str):
    ts_cls = _get_cls(pool, "google.protobuf.Timestamp")
    ts = ts_cls()
    ts.FromJsonString(iso)
    return ts


def _make_instrument(pool: descriptor_pool.DescriptorPool, **kwargs):
    cls = _get_cls(pool, "botcore.trading.v1.Instrument")
    return cls(**kwargs)


def _make_granularity(pool: descriptor_pool.DescriptorPool, iso: str):
    cls = _get_cls(pool, "botcore.trading.v1.CandleGranularity")
    return cls(iso8601_duration=iso)


def _make_candle(pool: descriptor_pool.DescriptorPool, **kwargs):
    cls = _get_cls(pool, "botcore.trading.v1.OhlcvCandle")
    return cls(**kwargs)


def test_history_response_serialization_matches_golden() -> None:
    pool = _make_pool()

    instrument = _make_instrument(
        pool,
        exchange="BINANCE",
        symbol="BTC/USDT",
        venue_symbol="BTCUSDT",
        quote_currency="USDT",
        base_currency="BTC",
    )
    granularity = _make_granularity(pool, "PT1M")

    response_cls = _get_cls(pool, "botcore.trading.v1.GetOhlcvHistoryResponse")

    response = response_cls(
        candles=[
            _make_candle(
                pool,
                instrument=instrument,
                open_time=_make_timestamp(pool, "2024-01-01T00:00:00Z"),
                open=42000.0,
                high=42100.0,
                low=41950.0,
                close=42080.0,
                volume=12.34,
                closed=True,
                granularity=granularity,
                sequence=1,
            ),
            _make_candle(
                pool,
                instrument=instrument,
                open_time=_make_timestamp(pool, "2024-01-01T00:01:00Z"),
                open=42080.0,
                high=42200.0,
                low=42000.0,
                close=42150.0,
                volume=8.9,
                closed=True,
                granularity=granularity,
                sequence=2,
            ),
        ],
        has_more=False,
        next_start_time=_make_timestamp(pool, "2024-01-01T00:02:00Z"),
    )

    assert response.SerializeToString().hex() == EXPECTED_HISTORY_HEX


def test_stream_updates_match_golden_frames() -> None:
    pool = _make_pool()

    instrument = _make_instrument(
        pool,
        exchange="KRAKEN",
        symbol="BTC/EUR",
        venue_symbol="XBTEUR",
        quote_currency="EUR",
        base_currency="BTC",
    )
    granularity = _make_granularity(pool, "PT5M")

    candle_closed = _make_candle(
        pool,
        instrument=instrument,
        open_time=_make_timestamp(pool, "2024-02-01T12:00:00Z"),
        open=35000.0,
        high=35150.0,
        low=34900.0,
        close=35080.0,
        volume=25.5,
        closed=True,
        granularity=granularity,
        sequence=100,
    )
    candle_live = _make_candle(
        pool,
        instrument=instrument,
        open_time=_make_timestamp(pool, "2024-02-01T12:05:00Z"),
        open=35080.0,
        high=35200.0,
        low=35050.0,
        close=35170.0,
        volume=18.75,
        closed=False,
        granularity=granularity,
        sequence=101,
    )

    snapshot_cls = _get_cls(pool, "botcore.trading.v1.StreamOhlcvSnapshot")
    increment_cls = _get_cls(pool, "botcore.trading.v1.StreamOhlcvIncrement")
    update_cls = _get_cls(pool, "botcore.trading.v1.StreamOhlcvUpdate")

    snapshot_update = update_cls(snapshot=snapshot_cls(candles=[candle_closed, candle_live]))
    increment_update = update_cls(increment=increment_cls(candle=candle_live))

    assert snapshot_update.SerializeToString().hex() == EXPECTED_STREAM_SNAPSHOT_HEX
    assert increment_update.SerializeToString().hex() == EXPECTED_STREAM_INCREMENT_HEX
