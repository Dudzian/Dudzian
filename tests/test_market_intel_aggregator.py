from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from bot_core.config.models import MarketIntelConfig, MarketIntelSqliteConfig
from bot_core.market_intel import MarketIntelAggregator


def _create_sqlite_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
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
                ("AAAUSD", 101.5, 125000.0, 4.5, 2.5, 0.2, 0.33, 1.1),
                ("BBBUSD", 55.1, 95000.0, 5.2, 3.0, 0.4, 0.45, 0.9),
            ],
        )
        conn.commit()


def test_market_intel_aggregator_writes_metrics(tmp_path: Path) -> None:
    db_path = tmp_path / "metrics.sqlite"
    _create_sqlite_db(db_path)

    config = MarketIntelConfig(
        enabled=True,
        output_directory=str(tmp_path / "out"),
        manifest_path=str(tmp_path / "manifest.json"),
        sqlite=MarketIntelSqliteConfig(path=str(db_path)),
        required_symbols=("AAAUSD", "BBBUSD"),
        default_weight=1.0,
    )

    aggregator = MarketIntelAggregator(config)
    written = aggregator.write_outputs()

    output_dir = tmp_path / "out"
    assert (output_dir / "aaausd.json").exists()
    payload = json.loads((output_dir / "aaausd.json").read_text())
    assert payload["baseline"]["mid_price"] == pytest.approx(101.5)
    assert payload["baseline"]["weight"] == pytest.approx(1.1)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["count"] == 2
    symbols = {entry["symbol"] for entry in manifest["entries"]}
    assert symbols == {"AAAUSD", "BBBUSD"}


def test_market_intel_missing_symbol_raises(tmp_path: Path) -> None:
    db_path = tmp_path / "metrics.sqlite"
    _create_sqlite_db(db_path)

    config = MarketIntelConfig(
        enabled=True,
        output_directory=str(tmp_path / "out"),
        manifest_path=None,
        sqlite=MarketIntelSqliteConfig(path=str(db_path)),
        required_symbols=("AAAUSD", "MISSING"),
        default_weight=1.0,
    )

    aggregator = MarketIntelAggregator(config)
    with pytest.raises(ValueError) as exc:
        aggregator.build()
    assert "Brakuje wymaganych symboli" in str(exc.value)
