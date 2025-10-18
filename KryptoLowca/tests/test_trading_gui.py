# test_trading_gui.py
# -*- coding: utf-8 -*-
"""
Unit tests for trading_gui.py.
"""
import os
import sys
import pytest
import tkinter as tk
import pandas as pd
import numpy as np
import asyncio
from pathlib import Path
from textwrap import dedent

sys.path.append(os.getcwd())

from KryptoLowca.trading_gui import TradingGUI
from KryptoLowca.trading_strategies import TradingParameters, EngineConfig
from KryptoLowca.managers.exchange_manager import ExchangeManager
from KryptoLowca.managers.exchange_core import Mode
from KryptoLowca.managers.config_manager import ConfigManager
from KryptoLowca.database_manager import DatabaseManager

class DummyExchange:
    def __init__(self):
        self._inited = False

    def init(self):
        self._inited = True

    def load_markets(self):
        return {"BTC/USDT": {}, "ETH/USDT": {}, "XRP/USDT": {}}

    def fetch_ohlcv(self, symbol, timeframe="1m", limit=100):
        ts0 = 1_700_000_000_000
        data = []
        price = 100.0
        for i in range(limit):
            ts = ts0 + i * 60_000
            o = price * (1.0 + 0.0001 * i)
            h = o * 1.01
            l = o * 0.99
            c = o * (1.0 + 0.00005)
            v = 10.0 + i
            data.append([ts, o, h, l, c, v])
            price = c
        return data

    def place_market_order(self, symbol, side, qty):
        return {"id": "X", "symbol": symbol, "side": side, "qty": qty}

class DummyAI:
    async def predict_series(self, symbol: str, df: pd.DataFrame):
        return pd.Series(np.full(len(df), 0.0002), index=df.index)

    async def train_all_models(self, symbol: str, data: pd.DataFrame, model_types: list, seq_len: int, epochs: int, batch: int,
                               progress_callback=None) -> dict:
        return {m: {"hit_rate": 0.55} for m in model_types}

class DummyRisk:
    def __init__(self, cfg=None):
        pass

class DummyReporter:
    def __init__(self, *a, **k):
        pass

    def export_pdf(self, fn):
        with open(fn, "wb") as f:
            f.write(b"%PDF-1.4 test")

class DummyDB:
    def __init__(self, *a, **k):
        pass

    async def log(self, user_id: int, level: str, msg: str, category: str = "general", context=None):
        pass

    async def ensure_user(self, email: str) -> int:
        return 1

    async def insert_trade(self, user_id: int, trade: dict):
        pass

    async def upsert_position(self, user_id: int, symbol: str, qty: float, avg_entry: float):
        pass

    async def get_positions(self, user_id: int) -> list:
        return []

    async def get_pnl_by_symbol(self, user_id: int, symbol=None, since_ts=None, until_ts=None, group_by="symbol") -> dict:
        return {"BTC/USDT": 100.0}

class DummyCfg:
    def __init__(self, *a, **k):
        self._store = {}

    async def save_user_config(self, user_id: int, preset_name: str, data: dict):
        self._store[preset_name] = data

    async def load_config(self, preset_name=None, user_id=None) -> dict:
        return self._store.get(preset_name, {
            "ai": {"threshold_bps": 5.0, "seq_len": 40, "epochs": 30, "batch_size": 64, "model_dir": "models"},
            "trade": {"risk_per_trade": 0.01, "max_leverage": 1.0, "stop_loss_pct": 0.02, "take_profit_pct": 0.05, "max_open_positions": 5},
            "exchange": {"exchange_name": "binance", "testnet": True}
        })

    def load_ai_config(self, preset_name=None, user_id=None):
        return {"threshold_bps": 5.0, "seq_len": 40, "epochs": 30, "batch_size": 64, "model_dir": "models"}

    def load_trade_config(self, preset_name=None, user_id=None):
        return TradingParameters()

    def list_presets(self):
        return list(self._store.keys())

class DummySec:
    def __init__(self, *a, **k):
        self._d = {}

    def save_encrypted_keys(self, d, pwd):
        self._d["pwd"] = pwd
        self._d["data"] = d

    def load_encrypted_keys(self, pwd):
        assert pwd == self._d.get("pwd")
        return self._d.get("data", {})

class DummyEngine:
    def __init__(self):
        self._cb = None
        self._ai_threshold_bps = 0.0
        self.tp = TradingParameters()
        self.ec = EngineConfig()
        self.last_live_tick = None

    def on_event(self, cb):
        self._cb = cb

    def set_parameters(self, tp, ec):
        self.tp, self.ec = tp, ec

    def configure(self, ex, ai, risk):
        self.ex, self.ai, self.risk = ex, ai, risk

    def execute_live_tick(self, symbol, df, preds):
        self.last_live_tick = (symbol, len(df))
        return {"symbol": symbol, "side": "long", "strength": 0.6, "qty_hint": 0.25, "price_ref": df["close"].iloc[-1]}

@pytest.fixture
async def app(monkeypatch, tmp_path):
    monkeypatch.setattr("trading_gui.ExchangeManager", DummyExchange)
    monkeypatch.setattr("trading_gui.AIManager", DummyAI)
    monkeypatch.setattr("trading_gui.RiskManager", DummyRisk)
    monkeypatch.setattr("trading_gui.ReportManager", DummyReporter)
    monkeypatch.setattr("trading_gui.DatabaseManager", DummyDB)
    monkeypatch.setattr("trading_gui.ConfigManager", DummyCfg)
    monkeypatch.setattr("trading_gui.SecurityManager", DummySec)
    monkeypatch.setattr("trading_gui.TradingEngine", DummyEngine)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tkinter display not available")
    root.withdraw()
    app = TradingGUI(root, enable_web_api=False)
    await asyncio.sleep(0.1)  # Wait for initialization
    yield app
    try:
        root.destroy()
    except Exception:
        pass

@pytest.mark.asyncio
async def test_load_markets_and_select(app):
    await app._load_markets()
    assert len(app.symbol_vars) >= 3
    keys = list(app.symbol_vars.keys())[:2]
    for k in keys:
        app.symbol_vars[k].set(True)
    app._apply_symbol_selection()
    assert set(app.selected_symbols) == set(keys)

@pytest.mark.asyncio
async def test_worker_one_iteration(app):
    await app._load_markets()
    k = list(app.symbol_vars.keys())[0]
    app.symbol_vars[k].set(True)
    app._apply_symbol_selection()
    await app._process_symbol(k)
    assert k in app.paper_positions
    assert app.engine.last_live_tick[0] == k

@pytest.mark.asyncio
async def test_backtest_and_report_export(app, tmp_path):
    await app._load_markets()
    k = list(app.symbol_vars.keys())[0]
    app.symbol_vars[k].set(True)
    app._apply_symbol_selection()
    await app._run_backtest()
    out = tmp_path / "report.pdf"
    app.reporter.export_pdf = lambda fn: out.write_bytes(b"%PDF-1.4 test")
    await app._export_pdf_report()
    assert out.exists()

@pytest.mark.asyncio
async def test_presets_roundtrip(app):
    data = app._gather_settings()
    await app.config_mgr.save_user_config(1, "unit", data)
    got = await app.config_mgr.load_config(preset_name="unit", user_id=1)
    assert got and got["ai"]["enable"] == data["ai"]["enable"]

@pytest.mark.asyncio
async def test_keys_roundtrip(app):
    app.password_var.set("s3cret")
    app.testnet_key.set("A" * 16)
    app.testnet_secret.set("B" * 16)
    app.live_key.set("C" * 16)
    app.live_secret.set("D" * 16)
    await app._save_keys()
    app.testnet_key.set("")
    app.testnet_secret.set("")
    app.live_key.set("")
    app.live_secret.set("")
    await app._load_keys()
    assert app.testnet_key.get().startswith("A") and app.live_secret.get().startswith("D")

@pytest.mark.asyncio
async def test_dashboard_update(app):
    await app._update_dashboard()
    assert "PnL: 100.0" in app.pnl_var.get()
    assert "Positions: 0" in app.positions_var.get()

@pytest.mark.asyncio
async def test_risk_profile_section_updates(app):
    app.set_risk_profile_context("balanced")
    app.risk_manager_settings = {
        "risk_per_trade": 0.015,
        "portfolio_risk": 0.25,
    }
    await asyncio.sleep(0)

    assert app.risk_profile_display_var.get() == "balanced"
    limits = app.risk_limits_display_var.get()
    assert "1.5% per trade" in limits
    assert "25.0% exposure cap" in limits


@pytest.mark.asyncio
async def test_invalid_symbol_selection(app):
    await app._load_markets()
    app._apply_symbol_selection()
    assert not app.selected_symbols
    # Should not crash
    await app._run_backtest()


def _write_core_config(
    path: Path,
    *,
    max_daily_loss: float,
    max_position: float,
    hard_drawdown: float,
    stop_loss: float = 1.5,
) -> None:
    path.write_text(
        dedent(
            f"""
            risk_profiles:
              balanced:
                max_daily_loss_pct: {max_daily_loss}
                max_position_pct: {max_position}
                target_volatility: 0.1
                max_leverage: 3.0
                stop_loss_atr_multiple: {stop_loss}
                max_open_positions: 5
                hard_drawdown_pct: {hard_drawdown}
            environments:
              paper_env:
                name: paper_env
                exchange: binance_spot
                environment: paper
                keychain_key: dummy
                data_cache_path: ./cache
                risk_profile: balanced
                alert_channels: []
                required_permissions: []
                forbidden_permissions: []
            """
        ).strip()
    )


@pytest.mark.asyncio
async def test_reload_risk_manager_settings_updates_gui(app, tmp_path):
    core_path = tmp_path / "core.yaml"
    _write_core_config(core_path, max_daily_loss=0.02, max_position=0.05, hard_drawdown=0.08, stop_loss=1.5)

    app.core_config_path = core_path
    app.core_environment = "paper_env"

    old_mgr = app.risk_mgr
    profile_name, settings, _ = app.reload_risk_manager_settings()

    assert profile_name == "balanced"
    assert settings["max_daily_loss_pct"] == pytest.approx(0.02)
    assert app.max_daily_loss_pct == pytest.approx(0.02)
    assert app.risk_per_trade.get() == pytest.approx(0.05)
    assert app.portfolio_risk.get() == pytest.approx(0.08)
    assert app.risk_profile_var.get() == "balanced"
    assert app.trail_atr_mult_var.get() == pytest.approx(1.5)
    assert app.risk_mgr is not old_mgr

    _write_core_config(core_path, max_daily_loss=0.03, max_position=0.07, hard_drawdown=0.1, stop_loss=2.25)

    profile_name2, settings2, _ = app.reload_risk_manager_settings()
    assert profile_name2 == "balanced"
    assert settings2["max_daily_loss_pct"] == pytest.approx(0.03)
    assert app.max_daily_loss_pct == pytest.approx(0.03)
    assert app.trail_atr_mult_var.get() == pytest.approx(2.25)


def test_sync_positions_from_service_spot(tmp_path, monkeypatch):
    db_path = tmp_path / "spot_positions.db"
    db_url = f"sqlite+aiosqlite:///{db_path.as_posix()}"
    ex_mgr = ExchangeManager(exchange_id="binance", db_url=db_url)
    ex_mgr.set_mode(spot=True)
    db = ex_mgr._ensure_db()
    assert db is not None
    db.sync.upsert_position({
        "symbol": "BTC/USDT",
        "side": "LONG",
        "quantity": 0.5,
        "avg_price": 25_000.0,
        "unrealized_pnl": 123.45,
        "mode": "live",
    })

    import KryptoLowca.trading_gui as tg

    monkeypatch.setattr(tg, "DatabaseManager", DummyDB)
    monkeypatch.setattr(tg, "SecurityManager", DummySec)
    monkeypatch.setattr(tg, "ConfigManager", DummyCfg)
    monkeypatch.setattr(tg, "ReportManager", DummyReporter)
    monkeypatch.setattr(tg, "RiskManager", DummyRisk)
    monkeypatch.setattr(tg, "AIManager", DummyAI)
    monkeypatch.setattr(tg, "TradingEngine", DummyEngine)

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tkinter display not available")
    root.withdraw()
    try:
        app = tg.TradingGUI(root, enable_web_api=False)
        app.ex_mgr = ex_mgr
        app.mode_var.set("Spot")
        app.network_var.set("Live")

        positions = app._sync_positions_from_service()
        assert any(p.symbol == "BTC/USDT" for p in positions)

        synced = app._open_positions.get("BTC/USDT")
        assert synced is not None
        assert synced["side"] == "buy"
        assert synced["qty"] == pytest.approx(0.5)
        assert synced["entry"] == pytest.approx(25_000.0)
    finally:
        try:
            root.destroy()
        except Exception:
            pass