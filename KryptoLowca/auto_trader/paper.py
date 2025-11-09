"""Shim delegujący do nowej implementacji ``bot_core.auto_trader.paper_app``."""

from __future__ import annotations

from bot_core.auto_trader.paper_app import (
    DEFAULT_PAPER_BALANCE,
    DEFAULT_SYMBOL,
    HeadlessTradingStub,
    PaperAutoTradeApp,
    PaperAutoTradeOptions,
    main,
    parse_cli_args,
)

__all__ = [
    "DEFAULT_SYMBOL",
    "DEFAULT_PAPER_BALANCE",
    "PaperAutoTradeOptions",
    "HeadlessTradingStub",
    "PaperAutoTradeApp",
    "parse_cli_args",
    "main",
]
