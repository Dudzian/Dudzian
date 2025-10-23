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

    storage.write(
        "BTC/USDT::1m",
        {
            "columns": ["open_time", "open", "high", "low", "close", "volume"],
            "rows": _sample_rows(),
        },
    )

    adapter = _FakeAdapter()
    upstream = _FailingUpstream(exchange_adapter=adapter)
    source = CachedOHLCVSource(
        storage=storage,
        upstream=upstream,
        snapshot_fetcher=None,
    )

    request = OHLCVRequest(symbol="BTC/USDT", interval="1m", start=0, end=180_000, limit=2)
    response = source.fetch_ohlcv(request)

    assert response.rows == _sample_rows()


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

    request = OHLCVRequest(symbol="BTC/USDT", interval="1m", start=0, end=180_000, limit=3)
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
    assert len(adapter.calls) == 2
    first_call, second_call = adapter.calls
    assert first_call[0] == "fetch_ohlcv" and first_call[1][2] == 0
    assert second_call[0] == "fetch_ohlcv" and second_call[1][2] >= 60_000


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
