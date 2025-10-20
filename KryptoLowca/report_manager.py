# -*- coding: utf-8 -*-
"""Lekki wrapper raportów na potrzeby GUI i szybkich skryptów."""
from __future__ import annotations

from reporting import EnhancedReporting, TradeInfo

__all__ = ["ReportManager"]


class ReportManager:
    """Adapter delegujący do modułu :mod:`reporting`.

    Został przeniesiony z ``KryptoLowca.managers.report_manager`` w ramach
    porządkowania publicznego API projektu.
    """

    def __init__(self, db_path: str) -> None:
        self.reporter = EnhancedReporting(db_path=db_path)

    def log_trade_event(
        self,
        trade_info: dict | TradeInfo,
        symbol: str,
        order: dict | None,
        mode: str,
    ) -> None:
        if isinstance(trade_info, dict):
            self.reporter.log_trade(
                trade_info,
                symbol=symbol,
                order=order,
                mode=mode,
            )
        else:
            self.reporter.log_trade(
                trade_info,
                symbol=symbol,
                order=order,
                mode=mode,
            )

    def export_pdf(
        self,
        filename: str,
        mode: str | None = None,
        symbol: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> None:
        self.reporter.export_to_pdf(
            filename,
            mode=mode,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
