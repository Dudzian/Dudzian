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

sys.path.append(os.getcwd())

from trading_gui import TradingGUI, TradingParameters, EngineConfig
from managers.config_manager import ConfigManager
from managers.database_manager import DatabaseManager
from managers.ai_manager import AIManager

class DummyExchange:
    def __init__(self):
        self._inited = False

    def init(self):
        self._inited = True

    def set_mode(self, *a, **k):
        self.mode_args = (a, k)

    def set_paper_balance(self, amount, asset="USDT"):
        self.paper_balance = float(amount)
        self.paper_asset = asset

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

    def simulate_vwap_price(self, symbol, side, amount=None, fallback_bps=5.0):
        return (100.0, fallback_bps)

    def quantize_amount(self, symbol, amount):
        return float(amount)

    def place_market_order(self, symbol, side, qty):
        return {"id": "X", "symbol": symbol, "side": side, "qty": qty}

    def create_order(self, symbol, side, type_, quantity, price=None, client_order_id=None):
        return {
            "symbol": symbol,
            "side": side,
            "type": type_,
            "quantity": quantity,
            "price": price,
            "client_order_id": client_order_id,
        }

class DummyAI:
    async def predict_series(self, symbol: str, df: pd.DataFrame):
        return pd.Series(np.full(len(df), 0.0002), index=df.index)

    async def train_all_models(self, symbol: str, data: pd.DataFrame, model_types: list, seq_len: int, epochs: int, batch: int,
                               progress_callback=None) -> dict:
        return {m: {"hit_rate": 0.55} for m in model_types}

class DummyRisk:
    def __init__(self, cfg=None, **kwargs):
        self.cfg = cfg or {}

    def calculate_position_size(self, *a, **k):
        return 0.05

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
    root = tk.Tk()
    root.withdraw()
    app = TradingGUI(root)
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
async def test_invalid_symbol_selection(app):
    await app._load_markets()
    app._apply_symbol_selection()
    assert not app.selected_symbols
    # Should not crash
    await app._run_backtest()


@pytest.mark.asyncio
async def test_bridge_execute_trade_sell_closes_position(app):
    symbol = "BTC/USDT"
    initial_qty = 0.75

    class StubExchange:
        def __init__(self):
            self.quantized = None

        def simulate_vwap_price(self, symbol, side, amount=None, fallback_bps=5.0):
            return (105.0, fallback_bps)

        def quantize_amount(self, symbol, amount):
            self.quantized = amount
            return float(amount)

    stub_ex = StubExchange()
    app.ex_mgr = stub_ex
    app._ensure_exchange = lambda: None
    app._open_positions[symbol] = {"side": "buy", "qty": initial_qty, "entry": 95.0, "ts": "2024-01-01 00:00:00"}
    app.paper_balance = 500.0
    app.paper_balance_var.set(f"{app.paper_balance:,.2f}")

    captured = {}

    def fake_execute_market(self, symbol_arg, side_arg, quantity_arg, *, price=None):
        captured["symbol"] = symbol_arg
        captured["side"] = side_arg
        captured["qty"] = quantity_arg
        captured["price"] = price
        return {"status": "ok"}

    app.execute_market = fake_execute_market.__get__(app, type(app))

    app._bridge_execute_trade(symbol, "sell", 104.0)

    assert captured["symbol"] == symbol
    assert captured["side"] == "sell"
    assert captured["qty"] == pytest.approx(initial_qty)
    assert stub_ex.quantized == pytest.approx(initial_qty)
    assert symbol not in app._open_positions
    expected_balance = 500.0 + initial_qty * 105.0
    assert app.paper_balance == pytest.approx(expected_balance)