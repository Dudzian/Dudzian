"""Modularny pakiet AutoTradera z warstwą zgodności."""

from __future__ import annotations

from bot_core.alerts import AlertSeverity, emit_alert

from .app import AutoTrader, RiskDecision
from .paper import (
    DEFAULT_PAPER_BALANCE,
    DEFAULT_SYMBOL,
    HeadlessTradingStub,
    PaperAutoTradeApp,
    PaperAutoTradeOptions,
    parse_cli_args,
)

__all__ = [
    "AlertSeverity",
    "AutoTrader",
    "DEFAULT_PAPER_BALANCE",
    "DEFAULT_SYMBOL",
    "HeadlessTradingStub",
    "PaperAutoTradeApp",
    "PaperAutoTradeOptions",
    "RiskDecision",
    "emit_alert",
    "parse_cli_args",
]
