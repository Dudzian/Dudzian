"""Regression tests for exit reason parsing in backtest reports."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bot_core.backtest import trade_loader


def _write_trades(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    frame = pd.DataFrame(rows)
    csv_path = tmp_path / "trades.csv"
    frame.to_csv(csv_path, index=False)
    return csv_path


def _fills(*tags: str) -> str:
    fills = [{"tag": "ENTRY"}]
    fills.extend({"tag": tag} for tag in tags)
    return json.dumps(fills)


def test_load_trades_normalizes_supported_exit_reasons(tmp_path: Path) -> None:
    rows = [
        {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "entry_ts": 1_600_000_000,
            "exit_ts": 1_600_000_300,
            "entry_price": 100.0,
            "exit_price_wap": 101.5,
            "pnl_usdt": 1.5,
            "pnl_pct": 0.015,
            "r_multiple": 1.5,
            "fills_json": _fills("STOP_LOSS"),
        },
        {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "entry_ts": 1_600_000_600,
            "exit_ts": 1_600_000_960,
            "entry_price": 100.0,
            "exit_price_wap": 99.0,
            "pnl_usdt": -1.0,
            "pnl_pct": -0.01,
            "r_multiple": -1.0,
            "fills_json": _fills("TAKEPROFIT"),
        },
    ]
    _write_trades(tmp_path, rows)

    frame, _ = trade_loader.load_trades(tmp_path)

    assert frame.loc[0, "exit_reason"] == "stop_loss"
    assert frame.loc[1, "exit_reason"] == "take_profit"
    assert str(frame["exit_reason"].dtype) == "string"


def test_load_trades_preserves_unknown_tags(tmp_path: Path) -> None:
    rows = [
        {
            "symbol": "ETHUSDT",
            "timeframe": "4h",
            "entry_ts": 1_600_001_200,
            "exit_ts": 1_600_001_800,
            "entry_price": 200.0,
            "exit_price_wap": 205.0,
            "pnl_usdt": 5.0,
            "pnl_pct": 0.025,
            "r_multiple": 2.5,
            "fills_json": _fills("Manual Exit"),
        }
    ]
    _write_trades(tmp_path, rows)

    frame, _ = trade_loader.load_trades(tmp_path)

    assert frame.loc[0, "exit_reason"] == "manual_exit"
