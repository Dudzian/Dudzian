from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts.build_market_intel_metrics import main as build_metrics


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
