from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from bot_core.data import (
    CachedOHLCVSource,
    OHLCVRequest,
    PublicAPIDataSource,
    create_cached_ohlcv_source,
)
from bot_core.data.ohlcv.cache import OfflineOnlyDataSource
from bot_core.data.ohlcv.parquet_storage import ParquetCacheStorage
from bot_core.data.ohlcv.sqlite_storage import SQLiteCacheStorage
from bot_core.data.ohlcv.storage import DualCacheStorage
from bot_core.exchanges.base import ExchangeAdapter, ExchangeCredentials
from bot_core.exchanges.errors import ExchangeNetworkError


@dataclass
class _FakeAdapter(ExchangeAdapter):
    name: str = "fake_exchange"

    def __init__(self) -> None:
        super().__init__(ExchangeCredentials(key_id="k"))
        self.calls: list[tuple[str, tuple]] = []

    def configure_network(self, *, ip_allowlist=None) -> None:  # pragma: no cover - niewykorzystywane
        self.calls.append(("configure_network", tuple(ip_allowlist or ())))

    def fetch_account_snapshot(self):  # pragma: no cover - niepotrzebne
        raise NotImplementedError

    def fetch_symbols(self):  # pragma: no cover - niepotrzebne
        return []

    def fetch_ohlcv(self, symbol, interval, start=None, end=None, limit=None):
        self.calls.append(("fetch_ohlcv", (symbol, interval, start, end, limit)))
        base = float(start or 0)
        return [
            [base, 1.0, 2.0, 0.5, 1.5, 10.0],
            [base + 60_000, 1.5, 2.5, 1.0, 2.0, 12.0],
        ]

    def place_order(self, request):  # pragma: no cover - niepotrzebne
        raise NotImplementedError

    def cancel_order(self, order_id, *, symbol=None):  # pragma: no cover - niepotrzebne
        raise NotImplementedError

    def stream_public_data(self, *, channels):  # pragma: no cover - niepotrzebne
        raise NotImplementedError

    def stream_private_data(self, *, channels):  # pragma: no cover - niepotrzebne
        raise NotImplementedError


class _FailingUpstream(PublicAPIDataSource):
    def fetch_ohlcv(self, request: OHLCVRequest):
        raise ExchangeNetworkError("offline")


def _sample_rows(start: float = 0.0):
    return [
        [start, 10.0, 12.0, 9.5, 11.0, 42.0],
        [start + 60_000, 11.0, 13.0, 10.0, 12.0, 40.0],
    ]


def test_create_cached_source_uses_parquet_and_manifest(tmp_path: Path):
    adapter = _FakeAdapter()
    source = create_cached_ohlcv_source(
        adapter,
        cache_directory=tmp_path / "cache",
        manifest_path=tmp_path / "manifest.sqlite",
        enable_snapshots=True,
    )

    request = OHLCVRequest(symbol="BTC/USDT", interval="1m", start=0, end=180_000, limit=2)
    response = source.fetch_ohlcv(request)

    assert len(response.rows) == 2
    assert len(adapter.calls) == 2
    assert adapter.calls[0][0] == "fetch_ohlcv" and adapter.calls[0][1][2] == 0
    assert adapter.calls[1][0] == "fetch_ohlcv" and adapter.calls[1][1][2] >= 60_000
    # Snapshot powinien odpytać adapter o świeżą świecę w ograniczonym zakresie.
    adapter.calls.clear()
    cached = source.fetch_ohlcv(request)
    assert cached.rows == response.rows
    assert len(adapter.calls) == 1
    first_call = adapter.calls[0]
    assert first_call[0] == "fetch_ohlcv" and first_call[1][2] >= 60_000

    storage = ParquetCacheStorage(tmp_path / "cache", namespace=adapter.name)
    payload = storage.read("BTC/USDT::1m")
    assert payload["rows"]

    manifest = SQLiteCacheStorage(tmp_path / "manifest.sqlite", store_rows=False)
    metadata = manifest.metadata()
    assert metadata[f"last_timestamp::BTC/USDT::1m"]


def test_cached_source_fallbacks_to_cache_on_network_error(tmp_path: Path):
    parquet = ParquetCacheStorage(tmp_path / "cache", namespace="offline")
    manifest = SQLiteCacheStorage(tmp_path / "manifest.sqlite", store_rows=False)
    storage = DualCacheStorage(parquet, manifest)

    cached_payload = {
        "columns": ["open_time", "open", "high", "low", "close", "volume"],
        "rows": [
            [120_000.0, 10.0, 11.0, 9.5, 10.5, 31.0],
            [180_000.0, 10.5, 11.5, 10.0, 11.0, 29.0],
        ],
    }
    storage.write("BTC/USDT::1m", cached_payload)

    adapter = _FakeAdapter()
    upstream = _FailingUpstream(exchange_adapter=adapter)
    source = CachedOHLCVSource(
        storage=storage,
        upstream=upstream,
        snapshot_fetcher=None,
    )

    request = OHLCVRequest(symbol="BTC/USDT", interval="1m", start=0, end=180_000, limit=2)
    response = source.fetch_ohlcv(request)

    assert response.rows == cached_payload["rows"]


def test_snapshot_fetcher_merges_latest_rows(tmp_path: Path):
    adapter = _FakeAdapter()

    def _snapshot(req: OHLCVRequest):
        return [[req.end, 12.0, 13.5, 11.5, 12.8, 50.0]]

    parquet = ParquetCacheStorage(tmp_path / "cache", namespace="snap")
    manifest = SQLiteCacheStorage(tmp_path / "manifest.sqlite", store_rows=False)
    storage = DualCacheStorage(parquet, manifest)

    upstream = PublicAPIDataSource(exchange_adapter=adapter)
    source = CachedOHLCVSource(
        storage=storage,
        upstream=upstream,
        snapshot_fetcher=_snapshot,
        snapshots_enabled=True,
    )

    request = OHLCVRequest(symbol="BTC/USDT", interval="1m", start=0, end=180_000, limit=2)
    rows = source.fetch_ohlcv(request).rows

    assert rows[-1][0] == pytest.approx(request.end)


def test_cached_source_rehydrates_snapshot_when_missing(tmp_path: Path):
    adapter = _FakeAdapter()

    parquet = ParquetCacheStorage(tmp_path / "cache", namespace="fallback")
    manifest = SQLiteCacheStorage(tmp_path / "manifest.sqlite", store_rows=False)
    storage = DualCacheStorage(parquet, manifest)

    upstream = PublicAPIDataSource(exchange_adapter=adapter)
    source = CachedOHLCVSource(
        storage=storage,
        upstream=upstream,
        snapshot_fetcher=None,
        snapshots_enabled=True,
    )

    request = OHLCVRequest(symbol="BTC/USDT", interval="1m", start=0, end=180_000, limit=2)

    source.fetch_ohlcv(request)
    adapter.calls.clear()

    cached = source.fetch_ohlcv(request)

    assert cached.rows
    assert len(adapter.calls) == 1
    (snapshot_call,) = adapter.calls
    assert snapshot_call[0] == "fetch_ohlcv" and snapshot_call[1][2] >= 60_000
    assert source.snapshot_fetcher is not None


def test_cached_source_skips_upstream_when_cache_suffices(tmp_path: Path):
    adapter = _FakeAdapter()

    parquet = ParquetCacheStorage(tmp_path / "cache", namespace="fallback_ready")
    manifest = SQLiteCacheStorage(tmp_path / "manifest.sqlite", store_rows=False)
    storage = DualCacheStorage(parquet, manifest)

    storage.write(
        "BTC/USDT::1m",
        {
            "columns": ["open_time", "open", "high", "low", "close", "volume"],
            "rows": [
                [120_000.0, 10.0, 11.0, 9.5, 10.5, 31.0],
                [180_000.0, 10.5, 11.5, 10.0, 11.0, 29.0],
            ],
        },
    )

    upstream = PublicAPIDataSource(exchange_adapter=adapter)
    source = CachedOHLCVSource(
        storage=storage,
        upstream=upstream,
        snapshot_fetcher=None,
        snapshots_enabled=True,
    )

    request = OHLCVRequest(symbol="BTC/USDT", interval="1m", start=0, end=180_000, limit=2)
    response = source.fetch_ohlcv(request)

    assert response.rows
    assert len(adapter.calls) == 1
    (snapshot_call,) = adapter.calls
    assert snapshot_call[0] == "fetch_ohlcv" and snapshot_call[1][2] >= 60_000
    assert source.snapshot_fetcher is not None


def test_cached_source_hits_upstream_when_limit_not_met(tmp_path: Path):
    adapter = _FakeAdapter()

    parquet = ParquetCacheStorage(tmp_path / "cache", namespace="limit_gap")
    manifest = SQLiteCacheStorage(tmp_path / "manifest.sqlite", store_rows=False)
    storage = DualCacheStorage(parquet, manifest)

    storage.write(
        "BTC/USDT::1m",
        {
            "columns": ["open_time", "open", "high", "low", "close", "volume"],
            "rows": _sample_rows(),
        },
    )

    upstream = PublicAPIDataSource(exchange_adapter=adapter)
    source = CachedOHLCVSource(
        storage=storage,
        upstream=upstream,
        snapshot_fetcher=None,
        snapshots_enabled=True,
    )

    request = OHLCVRequest(symbol="BTC/USDT", interval="1m", start=0, end=180_000, limit=3)
    response = source.fetch_ohlcv(request)

    assert response.rows  # upstream powinien spróbować uzupełnić brakującą świecę
    assert len(adapter.calls) >= 1
    first_call = adapter.calls[0]
    assert first_call[0] == "fetch_ohlcv" and first_call[1][2] == 0


def test_cached_source_refreshes_stale_limit_window(tmp_path: Path):
    adapter = _FakeAdapter()

    parquet = ParquetCacheStorage(tmp_path / "cache", namespace="stale_limit")
    manifest = SQLiteCacheStorage(tmp_path / "manifest.sqlite", store_rows=False)
    storage = DualCacheStorage(parquet, manifest)

    # Cache zawiera zbyt stare świece – powinien nastąpić refresh upstream.
    storage.write(
        "BTC/USDT::1m",
        {
            "columns": ["open_time", "open", "high", "low", "close", "volume"],
            "rows": [
                [0.0, 10.0, 11.0, 9.5, 10.5, 30.0],
                [60_000.0, 10.5, 11.5, 10.0, 11.0, 28.0],
            ],
        },
    )

    upstream = PublicAPIDataSource(exchange_adapter=adapter)
    source = CachedOHLCVSource(
        storage=storage,
        upstream=upstream,
        snapshot_fetcher=None,
        snapshots_enabled=True,
    )

    request = OHLCVRequest(symbol="BTC/USDT", interval="1m", start=0, end=240_000, limit=2)

    response = source.fetch_ohlcv(request)

    assert len(response.rows) == 2
    assert response.rows[-1][0] == pytest.approx(request.end)

    assert len(adapter.calls) == 2
    first_call, second_call = adapter.calls
    assert first_call[0] == "fetch_ohlcv" and first_call[1][2] == 0
    assert second_call[0] == "fetch_ohlcv" and second_call[1][2] >= 120_000


def test_cached_source_refreshes_when_limit_window_has_gaps(tmp_path: Path):
    adapter = _FakeAdapter()

    parquet = ParquetCacheStorage(tmp_path / "cache", namespace="limit_gap_refresh")
    manifest = SQLiteCacheStorage(tmp_path / "manifest.sqlite", store_rows=False)
    storage = DualCacheStorage(parquet, manifest)

    storage.write(
        "BTC/USDT::1m",
        {
            "columns": ["open_time", "open", "high", "low", "close", "volume"],
            "rows": [
                [120_000.0, 10.0, 11.0, 9.5, 10.5, 31.0],
                [240_000.0, 10.5, 11.5, 10.0, 11.0, 29.0],
                [360_000.0, 11.0, 12.0, 10.5, 11.5, 27.0],
            ],
        },
    )

    upstream = PublicAPIDataSource(exchange_adapter=adapter)
    source = CachedOHLCVSource(
        storage=storage,
        upstream=upstream,
        snapshot_fetcher=None,
        snapshots_enabled=True,
    )

    request = OHLCVRequest(symbol="BTC/USDT", interval="1m", start=0, end=360_000, limit=3)

    response = source.fetch_ohlcv(request)

    assert len(response.rows) == 3
    assert response.rows[-1][0] == pytest.approx(request.end)

    assert len(adapter.calls) == 2
    first_call, second_call = adapter.calls
    assert first_call[0] == "fetch_ohlcv" and first_call[1][2] == 0
    assert second_call[0] == "fetch_ohlcv" and second_call[1][2] >= 240_000


def test_cached_source_skips_upstream_for_contiguous_range(tmp_path: Path):
    adapter = _FakeAdapter()

    parquet = ParquetCacheStorage(tmp_path / "cache", namespace="range_ready")
    manifest = SQLiteCacheStorage(tmp_path / "manifest.sqlite", store_rows=False)
    storage = DualCacheStorage(parquet, manifest)

    storage.write(
        "BTC/USDT::1m",
        {
            "columns": ["open_time", "open", "high", "low", "close", "volume"],
            "rows": [
                [0.0, 10.0, 11.0, 9.5, 10.5, 31.0],
                [60_000.0, 10.5, 11.5, 10.0, 11.0, 29.0],
                [120_000.0, 11.0, 12.0, 10.5, 11.5, 27.0],
            ],
        },
    )

    upstream = PublicAPIDataSource(exchange_adapter=adapter)

    snapshot_hits: list[OHLCVRequest] = []

    def _snapshot(req: OHLCVRequest):
        snapshot_hits.append(req)
        return [[req.end, 11.5, 12.5, 10.8, 11.9, 25.0]]

    source = CachedOHLCVSource(
        storage=storage,
        upstream=upstream,
        snapshot_fetcher=_snapshot,
        snapshots_enabled=True,
    )

    request = OHLCVRequest(symbol="BTC/USDT", interval="1m", start=0, end=120_000, limit=None)
    response = source.fetch_ohlcv(request)

    assert response.rows[-1][0] == pytest.approx(request.end)
    assert adapter.calls == []
    assert len(snapshot_hits) == 1


def test_cached_source_refreshes_range_request_with_gaps(tmp_path: Path):
    adapter = _FakeAdapter()

    parquet = ParquetCacheStorage(tmp_path / "cache", namespace="range_gap")
    manifest = SQLiteCacheStorage(tmp_path / "manifest.sqlite", store_rows=False)
    storage = DualCacheStorage(parquet, manifest)

    storage.write(
        "BTC/USDT::1m",
        {
            "columns": ["open_time", "open", "high", "low", "close", "volume"],
            "rows": [
                [0.0, 10.0, 11.0, 9.5, 10.5, 31.0],
                [120_000.0, 10.5, 11.5, 10.0, 11.0, 29.0],
                [240_000.0, 11.0, 12.0, 10.5, 11.5, 27.0],
            ],
        },
    )

    upstream = PublicAPIDataSource(exchange_adapter=adapter)
    source = CachedOHLCVSource(
        storage=storage,
        upstream=upstream,
        snapshot_fetcher=None,
        snapshots_enabled=False,
    )

    request = OHLCVRequest(symbol="BTC/USDT", interval="1m", start=0, end=240_000, limit=None)
    response = source.fetch_ohlcv(request)

    assert len(response.rows) >= 3
    assert adapter.calls == [
        ("fetch_ohlcv", ("BTC/USDT", "1m", 0, 240_000, None)),
    ]


def test_cached_source_range_respects_tolerance_without_refresh(tmp_path: Path):
    adapter = _FakeAdapter()

    parquet = ParquetCacheStorage(tmp_path / "cache", namespace="range_tolerance")
    manifest = SQLiteCacheStorage(tmp_path / "manifest.sqlite", store_rows=False)
    storage = DualCacheStorage(parquet, manifest)

    storage.write(
        "BTC/USDT::1m",
        {
            "columns": ["open_time", "open", "high", "low", "close", "volume"],
            "rows": [
                [2_000.0, 10.0, 11.0, 9.5, 10.5, 31.0],
                [62_000.0, 10.5, 11.5, 10.0, 11.0, 29.0],
                [122_000.0, 11.0, 12.0, 10.5, 11.5, 27.0],
                [182_000.0, 11.2, 12.2, 10.8, 11.7, 26.0],
            ],
        },
    )

    upstream = PublicAPIDataSource(exchange_adapter=adapter)
    source = CachedOHLCVSource(
        storage=storage,
        upstream=upstream,
        snapshot_fetcher=None,
        snapshots_enabled=True,
    )

    request = OHLCVRequest(symbol="BTC/USDT", interval="1m", start=0, end=182_000, limit=None)
    response = source.fetch_ohlcv(request)

    assert response.rows[-1][0] == pytest.approx(request.end)
    assert len(adapter.calls) == 1
    (snapshot_call,) = adapter.calls
    assert snapshot_call[0] == "fetch_ohlcv"
    assert snapshot_call[1][2] >= 60_000
    assert snapshot_call[1][4] == 1


def test_cached_source_range_aligns_cached_tail_with_tolerance(tmp_path: Path):
    adapter = _FakeAdapter()

    parquet = ParquetCacheStorage(tmp_path / "cache", namespace="range_tail")
    manifest = SQLiteCacheStorage(tmp_path / "manifest.sqlite", store_rows=False)
    storage = DualCacheStorage(parquet, manifest)

    storage.write(
        "BTC/USDT::1m",
        {
            "columns": ["open_time", "open", "high", "low", "close", "volume"],
            "rows": [
                [0.0, 10.0, 11.0, 9.5, 10.5, 31.0],
                [60_000.0, 10.5, 11.5, 10.0, 11.0, 29.0],
                [120_000.0, 11.0, 12.0, 10.5, 11.5, 27.0],
                [182_000.0, 11.2, 12.3, 10.9, 11.8, 26.5],
            ],
        },
    )

    upstream = PublicAPIDataSource(exchange_adapter=adapter)
    source = CachedOHLCVSource(
        storage=storage,
        upstream=upstream,
        snapshot_fetcher=None,
        snapshots_enabled=False,
    )

    request = OHLCVRequest(symbol="BTC/USDT", interval="1m", start=0, end=180_000, limit=None)
    response = source.fetch_ohlcv(request)

    assert response.rows[-1][0] == pytest.approx(request.end)
    assert adapter.calls == []


def test_cached_source_range_aligns_cached_head_with_tolerance(tmp_path: Path):
    adapter = _FakeAdapter()

    parquet = ParquetCacheStorage(tmp_path / "cache", namespace="range_head")
    manifest = SQLiteCacheStorage(tmp_path / "manifest.sqlite", store_rows=False)
    storage = DualCacheStorage(parquet, manifest)

    storage.write(
        "BTC/USDT::1m",
        {
            "columns": ["open_time", "open", "high", "low", "close", "volume"],
            "rows": [
                [-2_000.0, 10.0, 11.0, 9.5, 10.5, 31.0],
                [58_000.0, 10.5, 11.5, 10.0, 11.0, 29.0],
                [120_000.0, 11.0, 12.0, 10.5, 11.5, 27.0],
            ],
        },
    )

    upstream = PublicAPIDataSource(exchange_adapter=adapter)
    source = CachedOHLCVSource(
        storage=storage,
        upstream=upstream,
        snapshot_fetcher=None,
        snapshots_enabled=False,
    )

    request = OHLCVRequest(symbol="BTC/USDT", interval="1m", start=0, end=120_000, limit=None)
    response = source.fetch_ohlcv(request)

    assert response.rows[0][0] == pytest.approx(request.start)
    assert adapter.calls == []

def test_cached_source_range_refreshes_when_start_outside_tolerance(tmp_path: Path):
    adapter = _FakeAdapter()

    parquet = ParquetCacheStorage(tmp_path / "cache", namespace="range_misaligned")
    manifest = SQLiteCacheStorage(tmp_path / "manifest.sqlite", store_rows=False)
    storage = DualCacheStorage(parquet, manifest)

    storage.write(
        "BTC/USDT::1m",
        {
            "columns": ["open_time", "open", "high", "low", "close", "volume"],
            "rows": [
                [5_000.0, 10.0, 11.0, 9.5, 10.5, 31.0],
                [65_000.0, 10.5, 11.5, 10.0, 11.0, 29.0],
                [125_000.0, 11.0, 12.0, 10.5, 11.5, 27.0],
            ],
        },
    )

    upstream = PublicAPIDataSource(exchange_adapter=adapter)
    source = CachedOHLCVSource(
        storage=storage,
        upstream=upstream,
        snapshot_fetcher=None,
        snapshots_enabled=False,
    )

    request = OHLCVRequest(symbol="BTC/USDT", interval="1m", start=0, end=125_000, limit=None)
    response = source.fetch_ohlcv(request)

    assert len(response.rows) >= 3
    assert adapter.calls == [
        ("fetch_ohlcv", ("BTC/USDT", "1m", 0, 125_000, None)),
    ]


def test_cached_source_handles_unknown_interval_with_basic_bounds(tmp_path: Path):
    adapter = _FakeAdapter()

    parquet = ParquetCacheStorage(tmp_path / "cache", namespace="range_unknown")
    manifest = SQLiteCacheStorage(tmp_path / "manifest.sqlite", store_rows=False)
    storage = DualCacheStorage(parquet, manifest)

    storage.write(
        "BTC/USDT::weird",
        {
            "columns": ["open_time", "open", "high", "low", "close", "volume"],
            "rows": [
                [0.0, 10.0, 11.0, 9.5, 10.5, 31.0],
                [60_000.0, 10.5, 11.5, 10.0, 11.0, 29.0],
            ],
        },
    )

    upstream = PublicAPIDataSource(exchange_adapter=adapter)
    source = CachedOHLCVSource(
        storage=storage,
        upstream=upstream,
        snapshot_fetcher=None,
        snapshots_enabled=False,
    )

    request = OHLCVRequest(symbol="BTC/USDT", interval="weird", start=0, end=60_000, limit=None)
    response = source.fetch_ohlcv(request)

    assert len(response.rows) == 2
    assert adapter.calls == []


def test_snapshot_rows_align_with_request_end_and_manifest(tmp_path: Path):
    adapter = _FakeAdapter()

    def _snapshot(req: OHLCVRequest):
        return [[req.end - 60_000, 2.0, 3.0, 1.5, 2.5, 15.0]]

    parquet = ParquetCacheStorage(tmp_path / "cache", namespace="manifest")
    manifest = SQLiteCacheStorage(tmp_path / "manifest.sqlite", store_rows=False)
    storage = DualCacheStorage(parquet, manifest)

    upstream = PublicAPIDataSource(exchange_adapter=adapter)
    source = CachedOHLCVSource(
        storage=storage,
        upstream=upstream,
        snapshot_fetcher=_snapshot,
        snapshots_enabled=True,
    )

    request = OHLCVRequest(symbol="BTC/USDT", interval="1m", start=0, end=180_000, limit=3)
    response = source.fetch_ohlcv(request)

    assert response.rows[-1][0] == pytest.approx(request.end)

    metadata = manifest.metadata()
    last_key = "last_timestamp::BTC/USDT::1m"
    count_key = "row_count::BTC/USDT::1m"
    assert metadata[last_key] == str(int(request.end))
    assert int(metadata[count_key]) >= 3


def test_snapshot_rows_preserve_limit_window_order(tmp_path: Path):
    adapter = _FakeAdapter()

    def _snapshot(req: OHLCVRequest):
        return [[req.end - 60_000, 3.0, 4.0, 2.5, 3.5, 18.0]]

    parquet = ParquetCacheStorage(tmp_path / "cache", namespace="snapshot_order")
    manifest = SQLiteCacheStorage(tmp_path / "manifest.sqlite", store_rows=False)
    storage = DualCacheStorage(parquet, manifest)

    upstream = PublicAPIDataSource(exchange_adapter=adapter)
    source = CachedOHLCVSource(
        storage=storage,
        upstream=upstream,
        snapshot_fetcher=_snapshot,
        snapshots_enabled=True,
    )

    request = OHLCVRequest(symbol="BTC/USDT", interval="1m", start=0, end=180_000, limit=3)
    response = source.fetch_ohlcv(request)

    assert len(response.rows) == 3
    assert response.rows[-1][0] == pytest.approx(request.end)
    assert response.rows[-2][0] == pytest.approx(request.end - 60_000)


def test_snapshot_rows_skip_alignment_when_gap_too_large(tmp_path: Path):
    adapter = _FakeAdapter()

    def _snapshot(req: OHLCVRequest):
        return [[req.end - 600_000, 4.0, 5.0, 3.5, 4.5, 20.0]]

    parquet = ParquetCacheStorage(tmp_path / "cache", namespace="snapshot_gap")
    manifest = SQLiteCacheStorage(tmp_path / "manifest.sqlite", store_rows=False)
    storage = DualCacheStorage(parquet, manifest)

    upstream = PublicAPIDataSource(exchange_adapter=adapter)
    source = CachedOHLCVSource(
        storage=storage,
        upstream=upstream,
        snapshot_fetcher=_snapshot,
        snapshots_enabled=True,
    )

    request = OHLCVRequest(symbol="BTC/USDT", interval="1m", start=300_000, end=960_000, limit=3)
    response = source.fetch_ohlcv(request)

    assert len(response.rows) == 2
    assert response.rows[-1][0] == pytest.approx(360_000.0)
    assert response.rows[-1][0] != pytest.approx(request.end)


def test_snapshot_fetcher_handles_network_errors(tmp_path: Path):
    adapter = _FakeAdapter()

    def _snapshot(_: OHLCVRequest):
        raise ExchangeNetworkError("snapshot-down")

    parquet = ParquetCacheStorage(tmp_path / "cache", namespace="snapshot_error")
    manifest = SQLiteCacheStorage(tmp_path / "manifest.sqlite", store_rows=False)
    storage = DualCacheStorage(parquet, manifest)

    upstream = PublicAPIDataSource(exchange_adapter=adapter)
    source = CachedOHLCVSource(
        storage=storage,
        upstream=upstream,
        snapshot_fetcher=_snapshot,
        snapshots_enabled=True,
    )

    request = OHLCVRequest(symbol="BTC/USDT", interval="1m", start=0, end=120_000, limit=2)
    response = source.fetch_ohlcv(request)

    assert len(response.rows) == 2
    assert any(call[0] == "fetch_ohlcv" for call in adapter.calls)


def test_public_api_data_source_uses_exchange_adapter():
    adapter = _FakeAdapter()
    source = PublicAPIDataSource(exchange_adapter=adapter)

    request = OHLCVRequest(symbol="BTC/USDT", interval="1m", start=0, end=120_000, limit=2)
    response = source.fetch_ohlcv(request)

    assert response.columns == (
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    )
    assert len(response.rows) == 2
    assert adapter.calls == [
        ("fetch_ohlcv", ("BTC/USDT", "1m", 0, 120_000, 2)),
    ]


def test_offline_source_avoids_network_and_snapshots(tmp_path: Path):
    adapter = _FakeAdapter()
    source = create_cached_ohlcv_source(
        adapter,
        cache_directory=tmp_path / "cache",
        manifest_path=tmp_path / "manifest.sqlite",
        enable_snapshots=True,
        allow_network_upstream=False,
    )

    payload = {
        "columns": ["open_time", "open", "high", "low", "close", "volume"],
        "rows": _sample_rows(),
    }
    source.storage.write("BTC/USDT::1m", payload)

    request = OHLCVRequest(symbol="BTC/USDT", interval="1m", start=0, end=180_000, limit=2)
    response = source.fetch_ohlcv(request)

    assert response.rows == payload["rows"]
    assert adapter.calls == []
    assert isinstance(source.upstream, OfflineOnlyDataSource)
    assert source.snapshot_fetcher is None
