from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import pyarrow.parquet as pq

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.data.ohlcv import ParquetCacheStorage, DualCacheStorage, SQLiteCacheStorage


def _timestamp(year: int, month: int, day: int) -> float:
    return datetime(year, month, day, tzinfo=timezone.utc).timestamp() * 1000


def _payload(*timestamps: float) -> dict[str, object]:
    rows = []
    for index, ts in enumerate(timestamps, start=1):
        rows.append([ts, float(index), float(index + 1), float(index + 2), float(index + 3), float(index + 4)])
    return {
        "columns": ("open_time", "open", "high", "low", "close", "volume"),
        "rows": rows,
    }


def test_parquet_storage_writes_partitioned_files(tmp_path) -> None:
    storage = ParquetCacheStorage(tmp_path, namespace="binance_spot")
    key = "BTCUSDT::1d"
    storage.write(key, _payload(_timestamp(2024, 1, 1), _timestamp(2024, 2, 1)))

    jan_file = tmp_path / "binance_spot" / "BTCUSDT" / "1d" / "year=2024" / "month=01" / "data.parquet"
    feb_file = tmp_path / "binance_spot" / "BTCUSDT" / "1d" / "year=2024" / "month=02" / "data.parquet"

    assert jan_file.exists()
    assert feb_file.exists()

    read_payload = storage.read(key)
    assert len(read_payload["rows"]) == 2
    assert read_payload["rows"][0][0] == _timestamp(2024, 1, 1)
    assert read_payload["rows"][1][0] == _timestamp(2024, 2, 1)


def test_parquet_storage_deduplicates_rows(tmp_path) -> None:
    storage = ParquetCacheStorage(tmp_path, namespace="binance_spot")
    key = "ETHUSDT::1d"
    ts = _timestamp(2024, 3, 1)
    storage.write(key, _payload(ts))

    updated_payload = _payload(ts)
    updated_payload["rows"][0][4] = 99.0
    storage.write(key, updated_payload)

    read_payload = storage.read(key)
    assert len(read_payload["rows"]) == 1
    assert read_payload["rows"][0][4] == 99.0


def test_parquet_latest_timestamp_reads_last_partition(tmp_path) -> None:
    storage = ParquetCacheStorage(tmp_path, namespace="kraken_spot")
    key = "SOLUSDT::1h"
    ts1 = _timestamp(2023, 12, 1)
    ts2 = _timestamp(2024, 1, 15)
    storage.write(key, _payload(ts1, ts2))

    assert storage.latest_timestamp(key) == ts2


def test_dual_cache_storage_updates_manifest(tmp_path) -> None:
    parquet_storage = ParquetCacheStorage(tmp_path / "parquet", namespace="zonda_spot")
    manifest_storage = SQLiteCacheStorage(tmp_path / "manifest.sqlite3", store_rows=False)
    storage = DualCacheStorage(primary=parquet_storage, manifest=manifest_storage)
    key = "BTCPLN::1d"
    ts = _timestamp(2024, 5, 1)

    storage.write(key, _payload(ts))

    # Dane powinny być możliwe do odczytu z warstwy Parquet.
    read_payload = storage.read(key)
    assert read_payload["rows"][0][0] == ts

    # Manifest przechowuje ostatni znacznik czasu oraz liczbę świec.
    manifest_metadata = manifest_storage.metadata()
    assert manifest_metadata.get(f"row_count::BTCPLN::1d") == "1"
    assert storage.latest_timestamp(key) == ts

    # Plik Parquet istnieje i zawiera pojedynczy rekord.
    parquet_file = tmp_path / "parquet" / "zonda_spot" / "BTCPLN" / "1d" / "year=2024" / "month=05" / "data.parquet"
    table = pq.read_table(parquet_file)
    assert table.num_rows == 1
