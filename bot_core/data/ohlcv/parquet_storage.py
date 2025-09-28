"""Magazyn świec OHLCV zapisujący dane w formacie Parquet."""
from __future__ import annotations

import json
import threading
from collections.abc import MutableMapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from bot_core.data.base import CacheStorage

_COLUMNS: tuple[str, ...] = ("open_time", "open", "high", "low", "close", "volume")
_PARTITION_FILENAME = "data.parquet"
_METADATA_FILENAME = "metadata.json"


class _ParquetMetadata(MutableMapping[str, str]):
    """Lekka mapa klucz→wartość przechowywana w pliku JSON."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._data: dict[str, str] = self._load()

    def _load(self) -> dict[str, str]:
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}
        return {str(key): str(value) for key, value in raw.items()}

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(self._data, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        tmp_path.replace(self._path)

    def __getitem__(self, key: str) -> str:
        with self._lock:
            if key not in self._data:
                raise KeyError(key)
            return self._data[key]

    def __setitem__(self, key: str, value: str) -> None:
        with self._lock:
            self._data[str(key)] = str(value)
            self._flush()

    def __delitem__(self, key: str) -> None:
        with self._lock:
            if key not in self._data:
                raise KeyError(key)
            del self._data[key]
            self._flush()

    def __iter__(self):
        with self._lock:
            return iter(dict(self._data))

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class ParquetCacheStorage(CacheStorage):
    """CacheStorage zapisujący świeczki w strukturze exchange/symbol/interval/year/month."""

    def __init__(self, base_path: str | Path, *, namespace: str) -> None:
        self._base_path = Path(base_path)
        self._namespace = namespace
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _root(self) -> Path:
        return self._base_path / self._namespace

    def _symbol_dir(self, symbol: str, interval: str) -> Path:
        return self._root() / symbol / interval

    def _partition_path(self, symbol: str, interval: str, timestamp_ms: float) -> Path:
        dt = datetime.fromtimestamp(float(timestamp_ms) / 1000.0, tz=timezone.utc)
        base = self._symbol_dir(symbol, interval)
        year_dir = base / f"year={dt.year:04d}"
        month_dir = year_dir / f"month={dt.month:02d}"
        return month_dir / _PARTITION_FILENAME

    def _columns_mapping(self, columns: Sequence[str]) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for index, name in enumerate(columns):
            mapping[name] = index
        missing = [column for column in _COLUMNS if column not in mapping]
        if missing:
            raise ValueError(f"Brak kolumn wymaganych przez ParquetCacheStorage: {missing}")
        return mapping

    def _normalize_row(self, row: Sequence[float], mapping: Mapping[str, int]) -> list[float]:
        return [float(row[mapping[column]]) for column in _COLUMNS]

    def read(self, key: str) -> Mapping[str, Sequence[Sequence[float]]]:
        symbol, interval = key.split("::", maxsplit=1)
        base = self._symbol_dir(symbol, interval)
        if not base.exists():
            raise KeyError(key)

        rows: list[list[float]] = []
        for year_dir in sorted(base.glob("year=*")):
            for month_dir in sorted(year_dir.glob("month=*")):
                file_path = month_dir / _PARTITION_FILENAME
                if not file_path.exists():
                    continue
                table = pq.read_table(file_path)
                data = table.to_pydict()
                if not data:
                    continue
                open_times = [float(value) for value in data.get("open_time", [])]
                if not open_times:
                    continue
                length = len(open_times)
                columns_data = {column: [float(value) for value in data.get(column, [])] for column in _COLUMNS}
                for idx in range(length):
                    rows.append([columns_data[column][idx] for column in _COLUMNS])

        if not rows:
            raise KeyError(key)

        rows.sort(key=lambda item: item[0])
        return {"columns": _COLUMNS, "rows": rows}

    def write(self, key: str, payload: Mapping[str, Sequence[Sequence[float]]]) -> None:
        symbol, interval = key.split("::", maxsplit=1)
        rows = payload.get("rows", [])
        if not rows:
            return

        columns = payload.get("columns", _COLUMNS)
        mapping = self._columns_mapping(columns)
        partitions: dict[tuple[int, int], dict[float, list[float]]] = {}

        for row in rows:
            if not row:
                continue
            normalized = self._normalize_row(row, mapping)
            timestamp = normalized[0]
            dt = datetime.fromtimestamp(timestamp / 1000.0, tz=timezone.utc)
            bucket = (dt.year, dt.month)
            partition = partitions.setdefault(bucket, {})
            partition[timestamp] = normalized

        for _, partition_rows in sorted(partitions.items(), key=lambda item: item[0]):
            sample_timestamp = next(iter(partition_rows))
            partition_path = self._partition_path(symbol, interval, sample_timestamp)
            with self._lock:
                existing: dict[float, list[float]] = {}
                if partition_path.exists():
                    table = pq.read_table(partition_path)
                    data = table.to_pydict()
                    open_times = [float(value) for value in data.get("open_time", [])]
                    for idx, open_time in enumerate(open_times):
                        existing[open_time] = [float(data[column][idx]) for column in _COLUMNS]

                existing.update(partition_rows)
                sorted_keys = sorted(existing)
                payload_dict = {
                    column: [existing[key][index] for key in sorted_keys]
                    for index, column in enumerate(_COLUMNS)
                }
                table = pa.table(payload_dict)
                partition_path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, partition_path)

    def metadata(self) -> MutableMapping[str, str]:
        return _ParquetMetadata(self._root() / _METADATA_FILENAME)

    def latest_timestamp(self, key: str) -> float | None:
        symbol, interval = key.split("::", maxsplit=1)
        base = self._symbol_dir(symbol, interval)
        if not base.exists():
            return None

        for year_dir in sorted(base.glob("year=*"), reverse=True):
            for month_dir in sorted(year_dir.glob("month=*"), reverse=True):
                file_path = month_dir / _PARTITION_FILENAME
                if not file_path.exists():
                    continue
                table = pq.read_table(file_path, columns=["open_time"])
                if table.num_rows == 0:
                    continue
                values = [float(value.as_py()) for value in table.column("open_time")]
                if values:
                    return max(values)
        return None


__all__ = ["ParquetCacheStorage"]
