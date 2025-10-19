"""Warstwa zgodności dla metryk backtestowych."""
from __future__ import annotations

from bot_core.backtest.metrics import MetricsResult, compute_metrics, to_dict

__all__ = ["MetricsResult", "compute_metrics", "to_dict"]
