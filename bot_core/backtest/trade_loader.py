"""Helpers for loading backtest trade CSV files."""

from __future__ import annotations

import json
import pathlib
from typing import Tuple

import pandas as pd

from bot_core.trading.exit_reasons import ExitReason


def _ts_to_dt(series: pd.Series) -> pd.Series:
    """Convert a numeric timestamp column to a UTC datetime series."""

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.max() > 1e12:
        return pd.to_datetime(numeric, unit="ms", utc=True)
    return pd.to_datetime(numeric, unit="s", utc=True)


def _infer_exit_reason(row: pd.Series) -> str | None:
    """Determine the exit reason tag from a serialized fills payload."""

    try:
        fills_json = row.get("fills_json")
        if not fills_json or pd.isna(fills_json):
            return None
        fills = json.loads(fills_json)
        for fill in reversed(fills):
            tag = fill.get("tag")
            if tag and tag != "ENTRY":
                normalized = ExitReason.normalize(tag, allow_unknown=True)
                if normalized:
                    return normalized
    except Exception:
        pass
    return None


def load_trades(input_dir: pathlib.Path) -> Tuple[pd.DataFrame, pathlib.Path]:
    """Load the trades CSV produced by a backtest run.

    The dataframe mirrors the legacy loader behaviour, enriching the data with
    derived datetime columns, holding duration and normalized exit reasons.
    """

    csv_path = input_dir / "trades.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Brak pliku: {csv_path}")

    frame = pd.read_csv(csv_path)

    if "entry_ts" in frame:
        frame["entry_time_utc"] = _ts_to_dt(frame["entry_ts"])
    if "exit_ts" in frame:
        frame["exit_time_utc"] = _ts_to_dt(frame["exit_ts"])

    if "entry_time_utc" in frame and "exit_time_utc" in frame:
        frame["hold_minutes"] = (
            (frame["exit_time_utc"] - frame["entry_time_utc"]).dt.total_seconds() / 60
        ).round(1)
    else:
        frame["hold_minutes"] = None

    if "fills_json" in frame:
        frame["exit_reason"] = frame.apply(_infer_exit_reason, axis=1)
    else:
        frame["exit_reason"] = None

    if "exit_reason" in frame:
        frame["exit_reason"] = frame["exit_reason"].astype("string")

    return frame, csv_path


__all__ = ["load_trades"]

