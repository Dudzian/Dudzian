# test_trading_gui.py
# -*- coding: utf-8 -*-
"""Minimalne testy TradingGUI z nową warstwą ExecutionService."""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import os

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if not os.environ.get("DISPLAY"):
    pytest.skip("Brak DISPLAY – testy GUI pominięte w trybie headless", allow_module_level=True)

from managers.exchange_core import Mode, OrderDTO, OrderSide, OrderStatus, OrderType, PositionDTO


class DummyExchange:
    def __init__(self, exchange_id: str = "binance", paper_initial_cash: float = 10_000.0) -> None:
        self.exchange_id = exchange_id
        self.mode = Mode.PAPER
        self.balance = float(paper_initial_cash)
        self.positions: dict[str, PositionDTO] = {}
        self.markets = {"BTC/USDT": {}, "ETH/USDT": {}}

    def set_mode(self, *, paper: bool = False, spot: bool = False, futures: bool = False, testnet: bool = False) -> None:
        if paper:
            self.mode = Mode.PAPER
        elif futures:
            self.mode = Mode.FUTURES
        else:
            self.mode = Mode.SPOT

    def set_paper_balance(self, amount: float, asset: str = "USDT") -> None:
        self.balance = float(amount)

    def set_credentials(self, api_key: str | None, secret: str | None) -> None:  # pragma: no cover - noop
        self.api_key = api_key
        self.secret = secret

    def load_markets(self):
        return self.markets

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 100):
        base = 100.0 if symbol == "BTC/USDT" else 50.0
        data = []
        for i in range(limit):
            ts = 1_700_000_000_000 + i * 60_000
            o = base + i * 0.1
            h = o * 1.01
            l = o * 0.99
            c = o * 1.001
            v = 10.0 + i
            data.append([ts, o, h, l, c, v])
        return data

    def fetch_ticker(self, symbol: str):
        return {"last": 100.0 if symbol == "BTC/USDT" else 50.0, "close": 100.0}

    def simulate_vwap_price(self, symbol: str, side: str, amount=None, fallback_bps: float = 5.0, limit: int = 50):
        price = 100.0 if symbol == "BTC/USDT" else 50.0
        return price, fallback_bps

    def quantize_amount(self, symbol: str, amount: float) -> float:
        return float(f"{amount:.3f}")

    def fetch_balance(self):
        return {"free": {"USDT": self.balance}, "total": {"USDT": self.balance}}

    def fetch_positions(self, symbol: str | None = None):
        if symbol:
            pos = self.positions.get(symbol)
            return [pos] if pos else []
        return list(self.positions.values())

    def create_order(self, symbol: str, side: str, type_: str, quantity: float, price=None, client_order_id=None) -> OrderDTO:
        side_up = side.upper()
        avg_price = price or (100.0 if symbol == "BTC/USDT" else 50.0)
        order = OrderDTO(
            id=1,
            client_order_id=client_order_id,
            symbol=symbol,
            side=OrderSide.BUY if side_up == "BUY" else OrderSide.SELL,
            type=OrderType.MARKET if type_.upper() == "MARKET" else OrderType.LIMIT,
            quantity=float(quantity),
            price=avg_price,
            status=OrderStatus.FILLED,
            mode=self.mode,
        )
        if side_up == "BUY":
            self.balance = max(0.0, self.balance - avg_price * quantity)
            self.positions[symbol] = PositionDTO(
                symbol=symbol,
                side="LONG",
                quantity=quantity,
                avg_price=avg_price,
                unrealized_pnl=0.0,
                mode=self.mode,
            )
        else:
            self.balance += avg_price * quantity
            self.positions.pop(symbol, None)
        return order


class DummyAI:
    def predict_series(self, df: pd.DataFrame, feature_cols: list[str]):
        return pd.Series(np.full(len(df), 0.01), index=df.index)


class DummyRisk:
    def calculate_position_size(self, symbol, signal, market_data, portfolio):
        return 0.1


class DummyReporter:
    def __init__(self, *a, **k):
        pass

    def export_pdf(self, fn: Path):
        fn.write_bytes(b"%PDF-1.4 test")


class DummyDB:
    async def log(self, *a, **k):  # pragma: no cover - GUI używa sync wrappera
        return None

    async def ensure_user(self, *a, **k):
        return 1

    async def get_positions(self, *a, **k):
        return []


class DummyConfig:
    def list_presets(self):
        return []

    async def load_config(self, *a, **k):
        return {}


class DummySecurity:
    def save_encrypted_keys(self, *a, **k):
        pass

    def load_encrypted_keys(self, *a, **k):
        return {}


class DummyEngine:
    def __init__(self):
        self.tp = None
        self.ec = None
        self.last_plan = None

    def on_event(self, cb):
        self._cb = cb

    async def configure(self, *a, **k):  # pragma: no cover - brak async calli
        return None

    def set_parameters(self, tp, ec):
        self.tp = tp
        self.ec = ec

    async def execute_live_tick(self, symbol, df, preds):
        self.last_plan = symbol
        return {"symbol": symbol, "side": "buy", "qty_hint": 0.1, "price_ref": df["close"].iloc[-1]}


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr("trading_gui.ExchangeManager", DummyExchange)
    monkeypatch.setattr("trading_gui.AIManager", DummyAI)
    monkeypatch.setattr("trading_gui.RiskManager", DummyRisk)
    monkeypatch.setattr("trading_gui.ReportManager", DummyReporter)
    monkeypatch.setattr("trading_gui.DatabaseManager", DummyDB)
    monkeypatch.setattr("trading_gui.ConfigManager", DummyConfig)
    monkeypatch.setattr("trading_gui.SecurityManager", DummySecurity)
    monkeypatch.setattr("trading_gui.TradingEngine", DummyEngine)

    root = tk.Tk()
    root.withdraw()
    gui_module = __import__("trading_gui").trading_gui
    gui = gui_module.TradingGUI(root)
    root.update()
    yield gui
    try:
        root.destroy()
    except Exception:
        pass


def test_load_markets_and_apply_settings(app):
    app._load_markets()
    assert len(app.symbol_vars) == 2
    app.symbol_vars["BTC/USDT"].set(True)
    app._apply_symbol_selection()
    assert "BTC/USDT" in app.selected_symbols


def test_bridge_execute_trade_updates_positions(app):
    app._load_markets()
    app.symbol_vars["BTC/USDT"].set(True)
    app._apply_symbol_selection()
    app._market_data["BTC/USDT"] = pd.DataFrame(
        [[1, 100, 101, 99, 100.5, 10]], columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    app._bridge_execute_trade("BTC/USDT", "buy", 100.0)
    assert "BTC/USDT" in app._open_positions
    app._bridge_execute_trade("BTC/USDT", "sell", 100.0)
    assert "BTC/USDT" not in app._open_positions


def test_export_report(tmp_path, app):
    app._load_markets()
    out = tmp_path / "report.pdf"
    app.reporter.export_pdf(out)
    assert out.exists()
