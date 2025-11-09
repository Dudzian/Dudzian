from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts.build_market_intel_metrics import main as build_metrics


def test_build_market_intel_metrics_ohlcv_respects_cache_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import build_market_intel_metrics as cli

    captured: dict[str, object] = {}

    class _StubStorage:
        def __init__(self, base, *, namespace):
            captured["base"] = Path(base)
            captured["namespace"] = namespace

    class _StubSnapshot:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def to_dict(self) -> dict[str, object]:
            return {"symbol": self.symbol}

    class _StubAggregator:
        def __init__(self, storage) -> None:
            captured["storage"] = storage

        def build_many(self, queries):
            captured["queries"] = tuple(queries)
            return {query.symbol: _StubSnapshot(query.symbol) for query in queries}

    class _StubQuery:
        def __init__(self, symbol: str, interval: str, lookback_bars: int) -> None:
            self.symbol = symbol
            self.interval = interval
            self.lookback_bars = lookback_bars

    class _StubAsset:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

    class _StubGovernor:
        def __init__(self) -> None:
            self.assets = (_StubAsset("BTC/USDT"),)

    class _StubEnv:
        def __init__(self) -> None:
            self.exchange = "coinbase_spot"
            self.data_cache_path = str(tmp_path / "cache")
            self.data_source = type("DS", (), {"cache_namespace": "custom_offline"})()

    class _StubConfig:
        def __init__(self) -> None:
            self.environments = {"coinbase_offline": _StubEnv()}
            self.portfolio_governors = {"offline_gov": _StubGovernor()}

    monkeypatch.setattr(cli, "load_core_config", lambda path: _StubConfig())
    monkeypatch.setattr(cli, "_HAS_OHLCV", True)
    monkeypatch.setattr(cli, "OHLCVAggregator", _StubAggregator)
    monkeypatch.setattr(cli, "MarketIntelQuery", _StubQuery)
    monkeypatch.setattr(
        "bot_core.data.ohlcv.parquet_storage.ParquetCacheStorage", _StubStorage
    )

    output_path = tmp_path / "metrics.json"
    exit_code = build_metrics(
        [
            "--mode",
            "ohlcv",
            "--config",
            "config/core.yaml",
            "--environment",
            "coinbase_offline",
            "--governor",
            "offline_gov",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert captured["namespace"] == "custom_offline"
    assert captured["base"] == tmp_path / "cache"
    assert output_path.exists()


@pytest.fixture()
def sqlite_dataset(tmp_path: Path) -> Path:
    db_path = tmp_path / "market_metrics.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE market_metrics (
                symbol TEXT PRIMARY KEY,
                mid_price REAL,
                avg_depth_usd REAL,
                avg_spread_bps REAL,
                funding_rate_bps REAL,
                sentiment_score REAL,
                realized_volatility REAL,
                weight REAL
            )
            """
        )
        conn.executemany(
            "INSERT INTO market_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("AAAUSD", 101.0, 125000.0, 5.0, 4.0, 0.25, 0.35, 1.0),
                ("BBBUSD", 55.0, 95000.0, 6.0, 3.5, 0.45, 0.4, 0.8),
            ],
        )
        conn.commit()
    return db_path


def test_build_market_intel_metrics_cli(tmp_path: Path, sqlite_dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_dir = tmp_path / "metrics"
    manifest_path = tmp_path / "manifest.json"

    argv = [
        "--config",
        "config/core.yaml",
        "--output-dir",
        str(output_dir),
        "--manifest",
        str(manifest_path),
        "--sqlite-path",
        str(sqlite_dataset),
        "--required-symbol",
        "AAAUSD",
        "--required-symbol",
        "BBBUSD",
    ]

    exit_code = build_metrics(argv)
    assert exit_code == 0

    payload = json.loads((output_dir / "aaausd.json").read_text())
    assert payload["baseline"]["mid_price"] == pytest.approx(101.0)
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["count"] == 2


@pytest.mark.smoke
def test_build_market_intel_metrics_populate_sqlite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sqlite_path = tmp_path / "market_metrics.sqlite"
    output_dir = tmp_path / "metrics"
    manifest_path = tmp_path / "manifest.json"

    argv = [
        "--mode",
        "sqlite",
        "--config",
        "config/core.yaml",
        "--sqlite-path",
        str(sqlite_path),
        "--output-dir",
        str(output_dir),
        "--manifest",
        str(manifest_path),
        "--required-symbol",
        "BTCUSDT",
        "--required-symbol",
        "ETHUSDT",
        "--required-symbol",
        "SOLUSDT",
        "--populate-sqlite",
        "--sqlite-provider",
        "stage6_samples.market_intel:build_provider",
    ]

    monkeypatch.setenv("MARKET_INTEL_SQLITE_PROVIDER", "stage6_samples.market_intel:build_provider")
    exit_code = build_metrics(argv)

    assert exit_code == 0
    assert sqlite_path.exists()
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        path = output_dir / f"{symbol.lower()}.json"
        assert path.exists(), f"Brak pliku metryk dla {symbol}"
        payload = json.loads(path.read_text())
        assert "baseline" in payload

    manifest = json.loads(manifest_path.read_text())
    assert manifest["count"] == 3
