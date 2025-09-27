# test_database_manager.py
# -*- coding: utf-8 -*-
"""
Unit tests for database_manager.py.
"""
import asyncio
import json
import pytest
from datetime import datetime, timezone
from pathlib import Path

from KryptoLowca.database_manager import DatabaseManager, DBOptions, DatabaseConnectionError, MigrationError

@pytest.fixture
async def db(tmp_path):
    options = DBOptions(db_url=f"sqlite+aiosqlite:///{tmp_path}/test.db", timeout_s=10.0)
    db = await DatabaseManager.create(options)
    user_id = await db.ensure_user("tester@example.com")
    return db, user_id

@pytest.mark.asyncio
async def test_user_and_config(db):
    dbm, uid = db
    assert uid > 0
    preset = {"risk_per_trade": 0.01, "ai_threshold_bps": 7}
    await dbm.save_user_config(uid, "default", preset)
    loaded = await dbm.load_user_config(uid, "default")
    assert loaded == preset

@pytest.mark.asyncio
async def test_logging_and_export(db):
    dbm, uid = db
    context = {"k": 1}
    await dbm.log(uid, "INFO", "hello", category="app", context=context)
    rows = await dbm.get_logs(user_id=uid, level="INFO", category="app", limit=10)
    assert len(rows) == 1
    assert rows[0]["message"] == "hello"
    assert rows[0]["context"] == context
    csv_data = await dbm.export_logs(rows, fmt="csv")
    assert "message,context" in csv_data
    json_data = await dbm.export_logs(rows, fmt="json")
    parsed = json.loads(json_data)
    assert parsed[0]["context"]["k"] == 1

@pytest.mark.asyncio
async def test_trades_and_pnl(db):
    dbm, uid = db
    now = datetime.now(timezone.utc).isoformat()
    batch = [
        {"ts": now, "symbol": "BTCUSDT", "side": "BUY", "qty": 0.5, "entry": 30000, "exit": 31000, "pnl": 500},
        {"ts": now, "symbol": "BTCUSDT", "side": "SELL", "qty": 0.5, "entry": 32000, "exit": 31500, "pnl": -250},
        {"ts": now, "symbol": "ETHUSDT", "side": "BUY", "qty": 1.0, "entry": 2000, "exit": 2100, "pnl": 100},
    ]
    await dbm.batch_insert_trades(uid, batch)
    pnl = await dbm.get_pnl_by_symbol(uid)
    assert pytest.approx(pnl["BTCUSDT"], 1e-8) == 250.0
    assert pytest.approx(pnl["ETHUSDT"], 1e-8) == 100.0
    pnl_daily = await dbm.get_pnl_by_symbol(uid, group_by="day")
    assert len(pnl_daily) > 0

@pytest.mark.asyncio
async def test_positions(db):
    dbm, uid = db
    await dbm.upsert_position(uid, "BTCUSDT", 1.25, 30500.0)
    await dbm.upsert_position(uid, "BTCUSDT", 0.75, 31000.0)
    pos = await dbm.get_positions(uid)
    assert pos[0]["symbol"] == "BTCUSDT"
    assert pytest.approx(pos[0]["qty"], 1e-8) == 0.75
    assert pytest.approx(pos[0]["avg_entry"], 1e-8) == 31000.0

@pytest.mark.asyncio
async def test_reporting_feed(db):
    dbm, uid = db
    now = datetime.now(timezone.utc).isoformat()
    await dbm.insert_trade(uid, {"ts": now, "symbol": "ADAUSDT", "side": "BUY", "qty": 100, "entry": 0.5, "pnl": 5})
    feed = await dbm.feed_reporting_trades(uid, "ADAUSDT")
    assert feed and "commission" in feed[0] and "slippage" in feed[0]

@pytest.mark.asyncio
async def test_backup_restore(db, tmp_path):
    dbm, uid = db
    backup_path = tmp_path / "backup.db"
    await dbm.log(uid, "INFO", "test")
    await dbm.backup_database(backup_path)
    assert backup_path.exists()
    await dbm.restore_database(backup_path)
    logs = await dbm.get_logs(uid)
    assert len(logs) == 1

@pytest.mark.asyncio
async def test_invalid_input(db):
    dbm, uid = db
    with pytest.raises(ValueError):
        await dbm.ensure_user("")
    with pytest.raises(ValueError):
        await dbm.insert_trade(uid, {"ts": "2025-08-21", "qty": 1.0})
    with pytest.raises(ValueError):
        await dbm.upsert_position(uid, "BTCUSDT", -1.0, 30000.0)