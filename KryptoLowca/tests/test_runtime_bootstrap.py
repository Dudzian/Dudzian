from __future__ import annotations

import sqlite3
from pathlib import Path

from KryptoLowca.runtime.bootstrap import (
    bootstrap_frontend_services,
    bootstrap_market_intel,
)


def test_frontend_bootstrap_provides_execution_stack() -> None:
    services = bootstrap_frontend_services()

    assert services.exchange_manager is not None
    assert services.router is not None
    assert services.account_manager is not None
    assert services.execution_service is not None

    adapters = getattr(services.router, "list_adapters", lambda: ())()
    assert adapters, "Router powinien udostępniać co najmniej jeden adapter"

    exchanges = getattr(services.account_manager, "supported_exchanges", ())
    assert exchanges, "MultiExchangeAccountManager powinien znać obsługiwane giełdy"


def _create_market_intel_sqlite(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE market_metrics (
                symbol TEXT,
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
                ("BTC_USDT", 26750.0, 8_500_000.0, 12.0, 1.5, 0.35, 0.62, 1.0),
                ("ETH_USDT", 1700.0, 3_200_000.0, 18.0, -0.5, 0.22, 0.48, 0.8),
            ],
        )


def test_bootstrap_market_intel_uses_core_config(tmp_path: Path) -> None:
    db_path = tmp_path / "market.sqlite"
    _create_market_intel_sqlite(db_path)

    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        "\n".join(
            [
                "market_intel:",
                "  enabled: true",
                "  sqlite:",
                f"    path: {db_path}",
                "    table: market_metrics",
                "    symbol_column: symbol",
                "    mid_price_column: mid_price",
                "    depth_column: avg_depth_usd",
                "    spread_column: avg_spread_bps",
                "    funding_column: funding_rate_bps",
                "    sentiment_column: sentiment_score",
                "    volatility_column: realized_volatility",
                "    weight_column: weight",
            ]
        ),
        encoding="utf-8",
    )

    bootstrap_market_intel.cache_clear()
    try:
        aggregator = bootstrap_market_intel(config_path=config_path)
        assert aggregator is not None
        assert getattr(aggregator, "_mode", None) == "sqlite"

        baselines = aggregator.build()
        assert baselines
        assert getattr(baselines[0], "symbol", None) == "BTC_USDT"
    finally:
        bootstrap_market_intel.cache_clear()
