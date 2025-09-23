# managers/risk_manager_adapter.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from risk_management import create_risk_manager  # istniejący moduł


class RiskManager:
    """Adapter zapewniający jednolity kontrakt dla GUI/TradingEngine."""

    def __init__(self, config: Dict[str, Any]):
        self.risk_mgr = create_risk_manager(config)

    def calculate_position_size(
        self,
        symbol: str,
        signal: Any,
        market_data: Any,
        portfolio: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Zwraca rekomendowaną frakcję kapitału (0..1).

        Adapter normalizuje sygnał/market_data do formatu oczekiwanego przez
        `risk_management.RiskManagement.calculate_position_size`.
        """

        portfolio_ctx = portfolio or {}

        if isinstance(signal, dict):
            signal_payload = dict(signal)
        else:
            try:
                strength = float(signal)
            except Exception:
                strength = 0.0
            signal_payload = {
                "symbol": symbol,
                "strength": abs(strength),
                "confidence": min(1.0, abs(strength) / 100.0 if strength else 0.5),
                "direction": "LONG" if strength >= 0 else "SHORT",
                "prediction": strength,
            }

        if isinstance(market_data, pd.DataFrame):
            market_df = market_data
        elif isinstance(market_data, dict):
            if "df" in market_data and isinstance(market_data["df"], pd.DataFrame):
                market_df = market_data["df"]
            else:
                price = market_data.get("price")
                market_df = pd.DataFrame({"close": [float(price or 0.0)]})
        else:
            market_df = pd.DataFrame({"close": [0.0]})

        sizing = self.risk_mgr.calculate_position_size(symbol, signal_payload, market_df, portfolio_ctx)

        if hasattr(sizing, "recommended_size"):
            return float(getattr(sizing, "recommended_size", 0.0))

        try:
            return float(sizing)
        except Exception:
            return 0.0
