import asyncio
from pathlib import Path

import pytest

from bot_core.database import CURRENT_SCHEMA_VERSION, DatabaseManager


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def db(tmp_path: Path):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/native.db"
    manager = DatabaseManager(db_url=db_url)
    _run(manager.init_db(create=True))
    try:
        yield manager
    finally:
        state = getattr(manager, "_state", None)
        if state and state.engine is not None:
            _run(state.engine.dispose())


def test_initialization(db):
    version = _run(db.get_schema_version())
    assert version == CURRENT_SCHEMA_VERSION


def test_order_and_trade_roundtrip(db):
    order_id = _run(
        db.record_order(
            {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "MARKET",
                "quantity": 0.1,
            }
        )
    )
    assert order_id > 0

    trade_id = _run(
        db.record_trade(
            {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.1,
                "price": 30000,
                "order_id": order_id,
            }
        )
    )
    assert trade_id > 0

    trades = _run(db.fetch_trades(symbol="BTCUSDT", limit=10))
    assert trades and trades[0]["order_id"] == order_id


def test_logging_and_metrics(db):
    user_id = _run(db.ensure_user("native@example.com"))
    _run(db.log(user_id, "INFO", "hello", category="system", context={"foo": "bar"}))
    logs = _run(db.fetch_logs(level="INFO", source="system", limit=5))
    assert logs and logs[0]["message"] == "hello"

    metric_id = _run(
        db.log_performance_metric(
            {
                "metric": "expectancy",
                "value": 1.2,
                "symbol": "BTCUSDT",
                "mode": "paper",
            }
        )
    )
    assert metric_id > 0

    metrics = _run(db.fetch_performance_metrics(symbol="BTCUSDT", limit=5))
    assert metrics and metrics[0]["metric"] == "expectancy"


def test_export_helpers(db, tmp_path: Path):
    _run(
        db.record_order(
            {
                "symbol": "ETHUSDT",
                "side": "SELL",
                "type": "LIMIT",
                "quantity": 1.5,
                "price": 2000,
            }
        )
    )

    csv_path = _run(db.export_trades_csv(path=tmp_path / "trades.csv"))
    assert csv_path.exists()

    json_path = _run(db.export_table_json(table="orders", path=tmp_path / "orders.json"))
    assert json_path.exists()
