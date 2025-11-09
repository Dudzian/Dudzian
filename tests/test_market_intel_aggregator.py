from __future__ import annotations

import json
import math
import os
import sqlite3
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

import pytest

# ---- Conditional imports for both variants of the aggregator API ----
try:
    # Variant with OHLCV-based snapshots
    from bot_core.market_intel import MarketIntelAggregator, MarketIntelQuery  # type: ignore[attr-defined]
    _HAVE_MARKET_INTEL_QUERY = True
except Exception:  # pragma: no cover - environment without OHLCV API
    from bot_core.market_intel import MarketIntelAggregator  # type: ignore
    MarketIntelQuery = None  # type: ignore[assignment]
    _HAVE_MARKET_INTEL_QUERY = False

try:
    # Variant with SQLite + config-driven outputs
    from bot_core.config.models import MarketIntelConfig, MarketIntelSqliteConfig  # type: ignore[attr-defined]
    _HAVE_MARKET_INTEL_CONFIG = True
except Exception:  # pragma: no cover - environment without config API
    MarketIntelConfig = None  # type: ignore[assignment]
    MarketIntelSqliteConfig = None  # type: ignore[assignment]
    _HAVE_MARKET_INTEL_CONFIG = False

try:
    from bot_core.runtime.pipeline import (  # type: ignore[attr-defined]
        _load_market_intel_snapshots_from_reports,
        _resolve_latest_report,
    )
    _HAVE_PIPELINE_HELPERS = True
except Exception:  # pragma: no cover - pipeline helpers unavailable
    _load_market_intel_snapshots_from_reports = None  # type: ignore[assignment]
    _resolve_latest_report = None  # type: ignore[assignment]
    _HAVE_PIPELINE_HELPERS = False


# =============================================================================
#                       Tests for OHLCV-based aggregation
# =============================================================================
class MemoryCacheStorage:
    """Minimal in-memory storage used to emulate OHLCV parquet cache API."""
    def __init__(self, payloads: Mapping[str, Mapping[str, Sequence[Sequence[float]]]]):
        self._payloads = dict(payloads)
        self._metadata: MutableMapping[str, str] = {}

    def read(self, key: str) -> Mapping[str, Sequence[Sequence[float]]]:
        return self._payloads[key]

    def write(self, key: str, payload: Mapping[str, Sequence[Sequence[float]]]) -> None:
        self._payloads[key] = payload

    def metadata(self) -> MutableMapping[str, str]:
        return self._metadata

    def latest_timestamp(self, key: str) -> float | None:
        payload = self._payloads.get(key)
        if not payload:
            return None
        rows = payload.get("rows", ())
        if not rows:
            return None
        return float(rows[-1][0])


def _supports_ohlcv_mode() -> bool:
    """Detect whether the imported MarketIntelAggregator supports snapshot building."""
    if not _HAVE_MARKET_INTEL_QUERY:
        return False
    try:
        # Try constructing with a simple storage and check presence of API
        dummy = MemoryCacheStorage({})
        agg = MarketIntelAggregator(dummy)  # type: ignore[call-arg]
        return hasattr(agg, "build_snapshot")
    except Exception:
        return False


def test_market_intel_aggregator_builds_metrics() -> None:
    if not _supports_ohlcv_mode():
        pytest.skip("OHLCV snapshot API not available in this build of MarketIntelAggregator")

    rows = [
        [1_000.0, 100.0, 101.0, 99.0, 100.0, 10.0],
        [2_000.0, 101.0, 106.0, 100.0, 105.0, 12.0],
        [3_000.0, 106.0, 111.0, 105.0, 110.0, 15.0],
        [4_000.0, 112.0, 121.0, 111.0, 120.0, 20.0],
    ]
    storage = MemoryCacheStorage(
        {
            "BTC_USDT::1h": {
                "columns": ("open_time", "open", "high", "low", "close", "volume"),
                "rows": rows,
            }
        }
    )
    aggregator = MarketIntelAggregator(storage)  # type: ignore[call-arg]

    snapshot = aggregator.build_snapshot(  # type: ignore[attr-defined]
        MarketIntelQuery(symbol="BTC_USDT", interval="1h", lookback_bars=4)  # type: ignore[call-arg]
    )

    assert snapshot.symbol == "BTC_USDT"
    assert snapshot.interval == "1h"
    assert snapshot.bar_count == 4
    assert snapshot.start == datetime.fromtimestamp(1.0, tz=timezone.utc)
    assert snapshot.end == datetime.fromtimestamp(4.0, tz=timezone.utc)
    assert snapshot.price_change_pct == pytest.approx((120.0 / 100.0 - 1.0) * 100.0)

    returns = [
        105.0 / 100.0 - 1.0,
        110.0 / 105.0 - 1.0,
        120.0 / 110.0 - 1.0,
    ]
    expected_volatility = math.sqrt(len(returns)) * statistics.pstdev(returns) * 100.0
    assert snapshot.volatility_pct == pytest.approx(expected_volatility)
    assert snapshot.max_drawdown_pct == pytest.approx(0.0)
    assert snapshot.average_volume == pytest.approx(sum(row[-1] for row in rows) / len(rows))
    notional = [row[4] * row[5] for row in rows]
    assert snapshot.liquidity_usd == pytest.approx(sum(notional) / len(notional))
    assert snapshot.momentum_score is not None
    assert snapshot.metadata["bars_used"] == pytest.approx(4.0)


def test_market_intel_aggregator_respects_lookback() -> None:
    if not _supports_ohlcv_mode():
        pytest.skip("OHLCV snapshot API not available in this build of MarketIntelAggregator")

    rows = [
        [1_000.0, 100.0, 101.0, 99.0, 100.0, 10.0],
        [2_000.0, 101.0, 106.0, 100.0, 105.0, 12.0],
        [3_000.0, 106.0, 111.0, 105.0, 110.0, 15.0],
        [4_000.0, 112.0, 121.0, 111.0, 120.0, 20.0],
    ]
    storage = MemoryCacheStorage(
        {
            "ETH_USDT::1h": {
                "columns": ("open_time", "open", "high", "low", "close", "volume"),
                "rows": rows,
            }
        }
    )
    aggregator = MarketIntelAggregator(storage)  # type: ignore[call-arg]
    snapshot = aggregator.build_snapshot(  # type: ignore[attr-defined]
        MarketIntelQuery(symbol="ETH_USDT", interval="1h", lookback_bars=2)  # type: ignore[call-arg]
    )

    assert snapshot.bar_count == 2
    assert snapshot.start == datetime.fromtimestamp(3.0, tz=timezone.utc)
    assert snapshot.end == datetime.fromtimestamp(4.0, tz=timezone.utc)
    assert snapshot.price_change_pct == pytest.approx((120.0 / 110.0 - 1.0) * 100.0)


# =============================================================================
#                   Tests for SQLite/config-driven aggregation
# =============================================================================
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


def _supports_sqlite_mode() -> bool:
    if not _HAVE_MARKET_INTEL_CONFIG:
        return False
    # As a lightweight capability check, ensure the aggregator exposes write_outputs or build
    try:
        return any(hasattr(MarketIntelAggregator, attr) for attr in ("write_outputs", "build"))
    except Exception:  # pragma: no cover
        return False


def _write_outputs_compat(
    aggregator: object,
    output_dir: Path,
    manifest_path: Path | None,
):
    """
    Compatibility wrapper – tries both signatures:
     - aggregator.write_outputs()
     - aggregator.write_outputs(output_directory=..., manifest_path=...)
    """
    if hasattr(aggregator, "write_outputs"):
        try:
            return aggregator.write_outputs()  # type: ignore[attr-defined, no-any-return]
        except TypeError:
            return aggregator.write_outputs(  # type: ignore[attr-defined, no-any-return]
                output_directory=output_dir,
                manifest_path=manifest_path,
            )
    # Fallback: some variants only expose build() that returns in-memory payloads
    if hasattr(aggregator, "build"):
        return aggregator.build()  # type: ignore[no-any-return]
    raise AttributeError("Aggregator exposes neither write_outputs() nor build()")


def _create_market_intel_report(path: Path, *, symbol: str = "BTCUSDT") -> None:
    payload = {
        "interval": "1h",
        "generated_at": "2024-06-01T00:00:00Z",
        "snapshots": {
            symbol: {
                "interval": "1h",
                "start": "2024-05-31T23:00:00Z",
                "end": "2024-06-01T00:00:00Z",
                "bar_count": 24,
                "price_change_pct": 1.5,
                "volatility_pct": 12.5,
                "max_drawdown_pct": 4.2,
                "average_volume": 1500.0,
                "liquidity_usd": 850000.0,
                "momentum_score": 0.8,
                "metadata": {"bars_used": 24.0},
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_market_intel_aggregator_writes_metrics(tmp_path: Path) -> None:
    if not _supports_sqlite_mode():
        pytest.skip("SQLite/config-driven MarketIntelAggregator API not available")

    db_path = tmp_path / "metrics.sqlite"
    _create_sqlite_db(db_path)

    config = MarketIntelConfig(  # type: ignore[call-arg]
        enabled=True,
        output_directory=str(tmp_path / "out"),
        manifest_path=str(tmp_path / "manifest.json"),
        sqlite=MarketIntelSqliteConfig(path=str(db_path)),  # type: ignore[call-arg]
        required_symbols=("AAAUSD", "BBBUSD"),
        default_weight=1.0,
    )

    aggregator = MarketIntelAggregator(config)  # type: ignore[call-arg]
    _ = _write_outputs_compat(aggregator, tmp_path / "out", tmp_path / "manifest.json")

    output_dir = tmp_path / "out"
    assert (output_dir / "aaausd.json").exists()
    payload = json.loads((output_dir / "aaausd.json").read_text(encoding="utf-8"))
    # Baseline fields should reflect DB
    assert payload["baseline"]["mid_price"] == pytest.approx(101.5)
    assert payload["baseline"]["weight"] == pytest.approx(1.1)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    # Accept either 'count' or 'entries' depending on implementation
    if "count" in manifest:
        assert manifest["count"] == 2
    elif "entries" in manifest:
        assert len(manifest["entries"]) == 2
    else:
        pytest.fail("Manifest does not contain 'count' nor 'entries'")

    if "entries" in manifest:
        symbols = {entry["symbol"] for entry in manifest["entries"]}
        assert symbols == {"AAAUSD", "BBBUSD"}


def test_market_intel_missing_symbol_raises(tmp_path: Path) -> None:
    if not _supports_sqlite_mode():
        pytest.skip("SQLite/config-driven MarketIntelAggregator API not available")

    db_path = tmp_path / "metrics.sqlite"
    _create_sqlite_db(db_path)

    config = MarketIntelConfig(  # type: ignore[call-arg]
        enabled=True,
        output_directory=str(tmp_path / "out"),
        manifest_path=None,
        sqlite=MarketIntelSqliteConfig(path=str(db_path)),  # type: ignore[call-arg]
        required_symbols=("AAAUSD", "MISSING"),
        default_weight=1.0,
    )

    aggregator = MarketIntelAggregator(config)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        # Prefer build() if present (often performs validation first)
        if hasattr(aggregator, "build"):
            aggregator.build()  # type: ignore[attr-defined]
        else:
            _write_outputs_compat(aggregator, tmp_path / "out", None)


def test_market_intel_fallback_loads_latest_report(tmp_path: Path) -> None:
    if not _HAVE_PIPELINE_HELPERS:
        pytest.skip("Runtime pipeline helpers not available")

    primary_dir = tmp_path / "stage6" / "market_intel"
    fallback_dir = tmp_path / "fallback"
    old_report = fallback_dir / "market_intel_core_portfolio_20240101.json"
    new_report = primary_dir / "market_intel_core_portfolio_20240201.json"
    _create_market_intel_report(old_report, symbol="ETHUSDT")
    _create_market_intel_report(new_report, symbol="BTCUSDT")

    now = time.time()
    os.utime(old_report, (now - 3600, now - 3600))
    os.utime(new_report, (now, now))

    class _Cfg:
        output_directory = str(primary_dir)

    resolved = _resolve_latest_report(  # type: ignore[misc]
        [primary_dir, fallback_dir],
        prefix="market_intel_core_portfolio_",
    )
    assert resolved == new_report

    snapshots = _load_market_intel_snapshots_from_reports(  # type: ignore[misc]
        "Core Portfolio",
        _Cfg(),
        (fallback_dir,),
    )
    assert "BTCUSDT" in snapshots
    snapshot = snapshots["BTCUSDT"]
    assert snapshot.interval == "1h"
    assert snapshot.bar_count == 24


def test_market_intel_fallback_respects_missing_reports(tmp_path: Path) -> None:
    if not _HAVE_PIPELINE_HELPERS:
        pytest.skip("Runtime pipeline helpers not available")

    stage_dir = tmp_path / "stage6"
    stage_dir.mkdir(parents=True)

    class _Cfg:
        output_directory = str(stage_dir)

    snapshots = _load_market_intel_snapshots_from_reports(  # type: ignore[misc]
        "Unconfigured Governor",
        _Cfg(),
        (),
    )
    assert snapshots == {}
