from __future__ import annotations

import importlib.util

import pytest


def test_parquet_cache_storage_optional_dependency(tmp_path):
    from bot_core.data.sources import ParquetCacheStorage

    if importlib.util.find_spec("pyarrow") is None:
        with pytest.raises(RuntimeError, match="ParquetCacheStorage wymaga opcjonalnej zależności"):
            ParquetCacheStorage(tmp_path, namespace="test")
        return

    storage = ParquetCacheStorage(tmp_path, namespace="test")
    assert storage is not None
