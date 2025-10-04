# managers/report_manager.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any
from KryptoLowca.reporting import EnhancedReporting, TradeInfo  # istniejący moduł

class ReportManager:
    def __init__(self, db_path: str):
        self.reporter = EnhancedReporting(db_path=db_path)

    def log_trade_event(self, trade_info: dict | TradeInfo, symbol: str, order: dict | None, mode: str):
        if isinstance(trade_info, dict):
            self.reporter.log_trade(trade_info, symbol=symbol, order=order, mode=mode)
        else:
            self.reporter.log_trade(trade_info, symbol=symbol, order=order, mode=mode)

    def export_pdf(self, filename: str, mode: str | None = None, symbol: str | None = None,
                   start_date: str | None = None, end_date: str | None = None):
        self.reporter.export_to_pdf(filename, mode=mode, symbol=symbol, start_date=start_date, end_date=end_date)
