# tests/test_database_manager.py
# -*- coding: utf-8 -*-
"""
Unit tests for database_manager.py.
"""
import asyncio
import json
import time
import pytest
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from KryptoLowca.database_manager import DatabaseManager, DBOptions
from bot_core.auto_trader import RiskDecision
from bot_core.database import CURRENT_SCHEMA_VERSION, RiskAuditLog


@pytest.fixture
async def db(tmp_path: Path):
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
    await dbm.insert_trade(
        uid,
        {"ts": now, "symbol": "ADAUSDT", "side": "BUY", "qty": 100, "entry": 0.5, "pnl": 5},
    )
    feed = await dbm.feed_reporting_trades(uid, "ADAUSDT")
    assert feed and "commission" in feed[0] and "slippage" in feed[0]


@pytest.mark.asyncio
async def test_backup_restore(db, tmp_path: Path):
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


@pytest.mark.asyncio
async def test_schema_version(db):
    dbm, _ = db
    version = await dbm.get_schema_version()
    assert version == CURRENT_SCHEMA_VERSION


@pytest.mark.asyncio
async def test_performance_and_risk_logging(db):
    dbm, _ = db
    metric_id = await dbm.log_performance_metric(
        {
            "metric": "auto_trader_expectancy",
            "value": 1.5,
            "symbol": "BTC/USDT",
            "mode": "paper",
            "window": 10,
            "context": {"source": "test"},
        }
    )
    assert metric_id > 0
    metrics = await dbm.fetch_performance_metrics(metric="auto_trader_expectancy")
    assert metrics and metrics[0]["metric"] == "auto_trader_expectancy"

    risk_id = await dbm.log_risk_limit(
        {
            "symbol": "BTC/USDT",
            "max_fraction": 0.2,
            "recommended_size": 0.1,
            "mode": "paper",
            "details": {"kelly": 0.05},
        }
    )
    assert risk_id > 0
    limits = await dbm.fetch_risk_limits(symbol="BTC/USDT")
    assert limits and limits[0]["symbol"] == "BTC/USDT"

    audit_entry_id = await dbm.log_risk_audit(
        {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "state": "warn",
            "reason": "trade_risk_pct",
            "fraction": 0.05,
            "price": 101.0,
            "mode": "paper",
            "schema_version": 1,
            "details": {"limit_events": [{"type": "trade_risk_pct", "value": 0.05, "threshold": 0.02}]},
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "ts": time.time(),
        }
    )
    assert audit_entry_id > 0
    audit_rows = await dbm.fetch_risk_audits(symbol="BTC/USDT")
    assert audit_rows and audit_rows[0]["symbol"] == "BTC/USDT"
    assert audit_rows[0]["details"]["limit_events"][0]["type"] == "trade_risk_pct"

    # --- Extended telemetry/audit logging ---
    rate_id = await dbm.log_rate_limit_snapshot(
        {
            "bucket_name": "global",
            "window_seconds": 60.0,
            "capacity": 1200,
            "count": 600,
            "usage": 0.5,
            "max_usage": 0.75,
            "reset_in_seconds": 10.0,
            "mode": "paper",
            "context": {"limit_triggered": False},
        }
    )
    assert rate_id > 0
    rate_snapshots = await dbm.fetch_rate_limit_snapshots(bucket="global")
    assert rate_snapshots and rate_snapshots[0]["bucket_name"] == "global"

    audit_id = await dbm.log_security_audit(
        {
            "action": "decrypt_keys",
            "status": "ok",
            "detail": "unit-test",
            "metadata": {"source": "tests"},
        }
    )
    assert audit_id > 0
    audits = await dbm.fetch_security_audit(action="decrypt_keys")
    assert audits and audits[0]["detail"] == "unit-test"


@pytest.mark.asyncio
async def test_risk_audit_logged(db):
    dbm, _ = db

    decision = RiskDecision(
        should_trade=True,
        fraction=0.25,
        state="warn",
        reason="risk_clamped",
        details={"limit_events": ["max_fraction", "portfolio_cap"]},
        mode="paper",
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
    )

    await dbm.log_risk_audit(
        {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "state": decision.state,
            "reason": decision.reason,
            "fraction": decision.fraction,
            "price": 101.0,
            "mode": decision.mode,
            "schema_version": 1,
            "limit_events": decision.details.get("limit_events"),
            "details": decision.details,
            "stop_loss_pct": decision.stop_loss_pct,
            "take_profit_pct": decision.take_profit_pct,
            "should_trade": decision.should_trade,
            "ts": time.time(),
        }
    )

    async with dbm.session() as session:
        rows = (await session.execute(select(RiskAuditLog))).scalars().all()

    assert rows and rows[0].symbol == "BTC/USDT"
    assert rows[0].state == "warn"
    assert pytest.approx(rows[0].fraction, 1e-8) == 0.25
    assert rows[0].should_trade is True
    assert pytest.approx(rows[0].take_profit_pct, 1e-8) == 0.04
    assert pytest.approx(rows[0].stop_loss_pct, 1e-8) == 0.02
    assert json.loads(rows[0].limit_events) == ["max_fraction", "portfolio_cap"]
